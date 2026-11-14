"""Unified AI Provider Incident and Cooldown Management.

Enforces the canonical lifecycle for AI provider incidents across Psychobot instances:
- Database-backed transactional state in ai_provider_incidents and ai_incident_notifications.
- Zero flat state files; zero cross-process race conditions.
- Strict fail-closed instance identity via notification_routing.detect_current_instance().
- In shared DB (Telegram + MAX): MAX registers incidents; companion Telegram bot delivers alerts.
- Dedicated per-recipient delivery tracking with CAS leases, backoff retries, and lease sweepers.
- Granular operational recovery matching exact provider, key fingerprint, operation, and model scope.
- Key rotation supersedes old incidents without generating false recovery alerts.
- Safe static templating with zero secrets, user prompts, or raw tracebacks in notification text.
"""

from __future__ import annotations

import hashlib
import html
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Sequence

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
from sqlalchemy import and_, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from database import (
    AIIncidentNotificationModel,
    AIProviderIncidentModel,
    async_session_maker,
)
from notification_outbox import BACKOFF_DELAYS
from notification_routing import (
    detect_current_instance,
    get_instance_by_pm2_name,
    is_valid_telegram_user_id,
    resolve_companion_telegram_instance,
    resolve_target_admin_ids,
)
from notification_transport import is_bot_blocked_reason
from provider_failures import ProviderFailure, normalize_exception

logger = logging.getLogger("ai_incident_manager")

DEFAULT_REPEAT_ALERT_SECONDS = 6 * 3600  # 6 hours
OUTBOX_LEASE_SECONDS = 60
MAX_OUTBOX_ATTEMPTS = 5


def get_provider_account_fingerprint(api_key: str | None, base_url: str | None = None) -> str:
    """Return an opaque 12-char cryptographic fingerprint for an API key + base URL."""
    if not api_key or not str(api_key).strip():
        return "unconfigured"
    raw = str(api_key).strip()
    if base_url and str(base_url).strip():
        raw = f"{raw}@{str(base_url).strip().rstrip('/')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def get_provider_api_key(config: Any, provider: str) -> str | None:
    """Safely extract configured API key for a provider from AIConfig."""
    if not config:
        return None
    p_norm = provider.strip().lower()
    if p_norm in ("claude", "anthropic"):
        return getattr(config, "claude_api_key", None)
    return getattr(config, f"{p_norm}_api_key", None)


def is_provider_active_for_bot(config: Any, provider: str) -> bool:
    """Check if the given provider is actively used in any role in AIConfig."""
    if not config:
        return False
    p_norm = provider.strip().lower()
    active_roles = [
        getattr(config, "provider", None),
        getattr(config, "fallback_provider", None),
        getattr(config, "vision_provider", None),
        getattr(config, "vision_fallback_provider", None),
        getattr(config, "transcription_provider", None),
        getattr(config, "image_generation_provider", None),
        getattr(config, "image_edit_provider", None),
    ]
    aliases = {
        "claude": {"claude", "anthropic"},
        "anthropic": {"claude", "anthropic"},
    }
    match_set = aliases.get(p_norm, {p_norm})
    return any(str(role).strip().lower() in match_set for role in active_roles if role)


def _format_incident_message_text(
    provider: str,
    operation: str,
    model_scope: str,
    incident_type: str,
    detail_code: str,
    evidence_tier: str,
    event_type: str,
    fallback_provider: str | None = None,
    fallback_model: str | None = None,
) -> str:
    """Render a strictly structured HTML alert message without user data or raw exception text."""
    p_esc = html.escape(str(provider).upper())
    op_esc = html.escape(str(operation))
    mod_esc = html.escape(str(model_scope))
    code_esc = html.escape(str(detail_code))
    time_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    if event_type == "resolved":
        return (
            f"✅ <b>AI-провайдер восстановил работу</b>\n\n"
            f"Провайдер: <code>{p_esc}</code>\n"
            f"Канал: <code>{op_esc}</code>\n"
            f"Модель: <code>{mod_esc}</code>\n"
            f"Время: <code>{time_str}</code>\n"
            f"Статус: Работоспособность подтверждена успешной операцией."
        )

    if event_type == "fallback_used" and fallback_provider:
        fb_p_esc = html.escape(str(fallback_provider).upper())
        fb_m_esc = html.escape(str(fallback_model or "—"))
        return (
            f"⚠️ <b>Основной AI-провайдер дал сбой — использован резервный</b>\n\n"
            f"Основной: <code>{p_esc}</code> (<code>{mod_esc}</code>)\n"
            f"Сбой: <code>{code_esc}</code> ({html.escape(incident_type)})\n"
            f"Резервный: <code>{fb_p_esc}</code> (<code>{fb_m_esc}</code>) — ✅ Ответ получен\n"
            f"Канал: <code>{op_esc}</code>\n"
            f"Время: <code>{time_str}</code>"
        )

    tier_label = (
        "Подтверждено API провайдера"
        if evidence_tier == "CONFIRMED_STRUCTURED"
        else "На основе ответа провайдера"
    )

    prefix = "🚨 <b>Сбой AI-провайдера</b>" if event_type == "opened" else "⏰ <b>Напоминание: Сбой AI-провайдера сохраняется</b>"
    return (
        f"{prefix}\n\n"
        f"Провайдер: <code>{p_esc}</code>\n"
        f"Канал: <code>{op_esc}</code>\n"
        f"Модель: <code>{mod_esc}</code>\n"
        f"Тип сбоя: <code>{html.escape(incident_type)}</code>\n"
        f"Код: <code>{code_esc}</code>\n"
        f"Уровень: {tier_label}\n"
        f"Время: <code>{time_str}</code>\n"
        f"Требуется вмешательство оператора для восстановления канала."
    )


async def record_ai_provider_failure(
    origin_instance_id: str | None,
    provider: str,
    error: Exception,
    operation: str,
    model: str | None = None,
    fallback_provider: str | None = None,
    fallback_model: str | None = None,
    fallback_succeeded: bool = False,
    credential_scope: str | None = None,
    session_maker: Any = None,
) -> AIProviderIncidentModel | None:
    """Record an AI provider failure, opening or updating an active incident and enqueuing outbox events."""
    current_inst = detect_current_instance()
    if not origin_instance_id:
        if not current_inst:
            logger.critical("[AI Incident] Fail-closed: cannot determine bot instance identity.")
            return None
        origin_instance_id = current_inst["pm2_name"]

    source_def = get_instance_by_pm2_name(origin_instance_id) or current_inst
    if not source_def:
        logger.critical("[AI Incident] Fail-closed: origin_instance_id %r not in registry.", origin_instance_id)
        return None

    companion_def = resolve_companion_telegram_instance(source_def)
    if not companion_def:
        logger.error("[AI Incident] No companion Telegram instance found for %s. Dropping incident.", origin_instance_id)
        return None

    companion_instance_id = companion_def["pm2_name"]
    authorized_instance_id = companion_instance_id  # Delivery worker belongs strictly to the companion Telegram bot

    # Normalize error to canonical ProviderFailure
    existing_failure = getattr(error, "failure", None)
    failure = existing_failure if isinstance(existing_failure, ProviderFailure) else normalize_exception(error, provider=provider, operation=operation)

    # Transient or non-operator failures do NOT create long-lived operator incidents
    if not failure.operator_action_required:
        logger.info(
            "[AI Incident] Non-operator failure for %s (%s, %s): not creating operator incident.",
            provider, operation, failure.classification,
        )
        return None

    prov_norm = (provider or failure.provider or "unknown").strip().upper()
    op_norm = (operation or failure.operation or "chat").strip().lower()
    mod_norm = str(model or "*").strip()
    inc_type = failure.classification
    det_code = str(failure.detail_code or failure.provider_error_code or "unspecified")[:128]
    evidence_tier = (
        "CONFIRMED_STRUCTURED"
        if failure.classification_source in ("provider_structured", "http_status_fallback", "transport", "sdk_typed_exception")
        else "INFERRED_MESSAGE"
    )

    sm = session_maker or async_session_maker
    now = datetime.utcnow()

    async with sm() as session:
        # If credential_scope not passed, compute from AIConfig
        if not credential_scope:
            from database import AIConfig
            ai_cfg = await session.get(AIConfig, 1)
            api_k = get_provider_api_key(ai_cfg, prov_norm)
            b_url = getattr(ai_cfg, "kie_base_url", None) if prov_norm == "KIE" else None
            credential_scope = get_provider_account_fingerprint(api_k, b_url)

        # 1. Atomic query / UPSERT with partial unique index
        stmt = select(AIProviderIncidentModel).where(
            AIProviderIncidentModel.origin_instance_id == origin_instance_id,
            AIProviderIncidentModel.provider == prov_norm,
            AIProviderIncidentModel.credential_scope == credential_scope,
            AIProviderIncidentModel.operation == op_norm,
            AIProviderIncidentModel.model_scope == mod_norm,
            AIProviderIncidentModel.incident_type == inc_type,
            AIProviderIncidentModel.status == "active",
        )
        existing = await session.scalar(stmt)

        incident: AIProviderIncidentModel
        is_new = False

        if existing:
            incident = existing
            await session.execute(
                update(AIProviderIncidentModel)
                .where(AIProviderIncidentModel.id == existing.id)
                .values(
                    occurrence_count=AIProviderIncidentModel.occurrence_count + 1,
                    last_seen_at=now,
                    detail_code=det_code,
                    evidence_tier=evidence_tier,
                )
            )
        else:
            new_inc = AIProviderIncidentModel(
                origin_instance_id=origin_instance_id,
                companion_instance_id=companion_instance_id,
                provider=prov_norm,
                credential_scope=credential_scope,
                operation=op_norm,
                model_scope=mod_norm,
                incident_type=inc_type,
                evidence_tier=evidence_tier,
                detail_code=det_code,
                http_status=failure.http_status,
                classification_source=failure.classification_source,
                status="active",
                occurrence_count=1,
                reminder_cycle=0,
                first_seen_at=now,
                last_seen_at=now,
                last_alerted_at=now,
            )
            try:
                async with session.begin_nested():
                    session.add(new_inc)
                    await session.flush()
                incident = new_inc
                is_new = True
            except IntegrityError:
                # Concurrent worker won the race
                existing = await session.scalar(stmt)
                if not existing:
                    raise
                incident = existing
                await session.execute(
                    update(AIProviderIncidentModel)
                    .where(AIProviderIncidentModel.id == existing.id)
                    .values(
                        occurrence_count=AIProviderIncidentModel.occurrence_count + 1,
                        last_seen_at=now,
                        detail_code=det_code,
                    )
                )

        # 2. Enqueue notification event if newly opened or fallback_used
        event_type = "fallback_used" if fallback_succeeded else "opened"
        if is_new or fallback_succeeded:
            admin_ids = await resolve_target_admin_ids(companion_def, current_instance=source_def, session=session)
            if admin_ids:
                msg_text = _format_incident_message_text(
                    provider=prov_norm,
                    operation=op_norm,
                    model_scope=mod_norm,
                    incident_type=inc_type,
                    detail_code=det_code,
                    evidence_tier=evidence_tier,
                    event_type=event_type,
                    fallback_provider=fallback_provider,
                    fallback_model=fallback_model,
                )
                for admin_id in admin_ids:
                    if not is_valid_telegram_user_id(admin_id):
                        continue
                    notif = AIIncidentNotificationModel(
                        incident_id=incident.id,
                        event_type=event_type,
                        cycle_index=0,
                        recipient_id=admin_id,
                        authorized_instance_id=authorized_instance_id,
                        message_text=msg_text,
                        status="pending",
                        attempts=0,
                        max_attempts=MAX_OUTBOX_ATTEMPTS,
                        next_retry_at=now,
                    )
                    try:
                        async with session.begin_nested():
                            session.add(notif)
                            await session.flush()
                    except IntegrityError:
                        pass  # Deduplicated by UNIQUE(incident_id, event_type, cycle_index, recipient_id)

        await session.commit()
        return incident


async def record_ai_operation_success(
    origin_instance_id: str | None,
    provider: str,
    credential_scope: str,
    operation: str,
    model: str | None = None,
    session_maker: Any = None,
) -> bool:
    """Record an operational success, resolving any active matching incident strictly within this scope."""
    current_inst = detect_current_instance()
    if not origin_instance_id:
        if not current_inst:
            return False
        origin_instance_id = current_inst["pm2_name"]

    source_def = get_instance_by_pm2_name(origin_instance_id) or current_inst
    companion_def = resolve_companion_telegram_instance(source_def) if source_def else None
    authorized_instance_id = companion_def["pm2_name"] if companion_def else origin_instance_id

    prov_norm = provider.strip().upper()
    op_norm = operation.strip().lower()
    mod_norm = str(model or "*").strip()

    sm = session_maker or async_session_maker
    now = datetime.utcnow()
    resolved_any = False

    async with sm() as session:
        # Match active incidents for this exact resource scope
        stmt = select(AIProviderIncidentModel).where(
            AIProviderIncidentModel.origin_instance_id == origin_instance_id,
            AIProviderIncidentModel.provider == prov_norm,
            AIProviderIncidentModel.credential_scope == credential_scope,
            AIProviderIncidentModel.operation == op_norm,
            or_(
                AIProviderIncidentModel.model_scope == "*",
                AIProviderIncidentModel.model_scope == mod_norm,
            ),
            AIProviderIncidentModel.status == "active",
        )
        matching_incidents = (await session.execute(stmt)).scalars().all()

        for inc in matching_incidents:
            resolved_any = True
            inc.status = "resolved"
            inc.resolved_at = now

            # 1. Cancel in-flight and pending reminder/opened notifications for this incident
            await session.execute(
                update(AIIncidentNotificationModel)
                .where(
                    AIIncidentNotificationModel.incident_id == inc.id,
                    AIIncidentNotificationModel.event_type.in_(["opened", "repeat_reminder", "fallback_used"]),
                    AIIncidentNotificationModel.status.in_(["pending", "processing"]),
                )
                .values(
                    status="cancelled",
                    last_error="superseded_by_recovery",
                    updated_at=now,
                )
            )

            # 2. Enqueue 'resolved' notification event
            if companion_def:
                admin_ids = await resolve_target_admin_ids(companion_def, current_instance=source_def, session=session)
                msg_text = _format_incident_message_text(
                    provider=inc.provider,
                    operation=inc.operation,
                    model_scope=inc.model_scope,
                    incident_type=inc.incident_type,
                    detail_code=inc.detail_code,
                    evidence_tier=inc.evidence_tier,
                    event_type="resolved",
                )
                for admin_id in admin_ids:
                    if not is_valid_telegram_user_id(admin_id):
                        continue
                    notif = AIIncidentNotificationModel(
                        incident_id=inc.id,
                        event_type="resolved",
                        cycle_index=0,
                        recipient_id=admin_id,
                        authorized_instance_id=authorized_instance_id,
                        message_text=msg_text,
                        status="pending",
                        attempts=0,
                        max_attempts=MAX_OUTBOX_ATTEMPTS,
                        next_retry_at=now,
                    )
                    try:
                        async with session.begin_nested():
                            session.add(notif)
                            await session.flush()
                    except IntegrityError:
                        pass

        if resolved_any:
            await session.commit()

    return resolved_any


async def handle_key_rotation(
    origin_instance_id: str | None,
    provider: str,
    new_key: str | None,
    base_url: str | None = None,
    session_maker: Any = None,
) -> int:
    """Supersede active incidents when an API key is rotated in AIConfig, without false replenishment alerts."""
    current_inst = detect_current_instance()
    if not origin_instance_id:
        if not current_inst:
            return 0
        origin_instance_id = current_inst["pm2_name"]

    prov_norm = provider.strip().upper()
    new_fp = get_provider_account_fingerprint(new_key, base_url)
    sm = session_maker or async_session_maker
    now = datetime.utcnow()
    superseded_count = 0

    async with sm() as session:
        stmt = select(AIProviderIncidentModel).where(
            AIProviderIncidentModel.origin_instance_id == origin_instance_id,
            AIProviderIncidentModel.provider == prov_norm,
            AIProviderIncidentModel.credential_scope != new_fp,
            AIProviderIncidentModel.status == "active",
        )
        active_old = (await session.execute(stmt)).scalars().all()
        for inc in active_old:
            inc.status = "superseded"
            inc.resolved_at = now
            superseded_count += 1

            # Cancel pending notifications
            await session.execute(
                update(AIIncidentNotificationModel)
                .where(
                    AIIncidentNotificationModel.incident_id == inc.id,
                    AIIncidentNotificationModel.status.in_(["pending", "processing"]),
                )
                .values(
                    status="cancelled",
                    last_error="superseded_by_key_rotation",
                    updated_at=now,
                )
            )

        if superseded_count > 0:
            await session.commit()
            logger.info(
                "[AI Incident] Superseded %d incidents for %s due to key rotation (new_fp=%s).",
                superseded_count, prov_norm, new_fp,
            )

    return superseded_count


async def check_incident_repeat_reminders(
    bot: Bot | None = None,
    repeat_seconds: int = DEFAULT_REPEAT_ALERT_SECONDS,
    session_maker: Any = None,
) -> int:
    """Atomic reminder cycle advance: CAS check + notification enqueue in a single transaction."""
    current_inst = detect_current_instance()
    if not current_inst:
        return 0
    current_bot_instance_id = current_inst["pm2_name"]

    sm = session_maker or async_session_maker
    now = datetime.utcnow()
    reminders_created = 0

    async with sm() as session:
        cutoff = now - timedelta(seconds=repeat_seconds)
        stmt = select(AIProviderIncidentModel).where(
            AIProviderIncidentModel.status == "active",
            or_(
                AIProviderIncidentModel.last_alerted_at.is_(None),
                AIProviderIncidentModel.last_alerted_at <= cutoff,
            ),
        )
        due_incidents = (await session.execute(stmt)).scalars().all()

        for inc in due_incidents:
            target_cycle = inc.reminder_cycle + 1
            cas_stmt = (
                update(AIProviderIncidentModel)
                .where(
                    AIProviderIncidentModel.id == inc.id,
                    AIProviderIncidentModel.status == "active",
                    AIProviderIncidentModel.reminder_cycle == inc.reminder_cycle,
                    or_(
                        AIProviderIncidentModel.last_alerted_at.is_(None),
                        AIProviderIncidentModel.last_alerted_at <= cutoff,
                    ),
                )
                .values(
                    reminder_cycle=target_cycle,
                    last_alerted_at=now,
                )
            )
            res = await session.execute(cas_stmt)
            if getattr(res, "rowcount", 0) != 1:
                continue

            source_def = get_instance_by_pm2_name(inc.origin_instance_id) or current_inst
            companion_def = resolve_companion_telegram_instance(source_def) if source_def else None
            if not companion_def:
                continue

            admin_ids = await resolve_target_admin_ids(companion_def, current_instance=source_def, session=session)
            msg_text = _format_incident_message_text(
                provider=inc.provider,
                operation=inc.operation,
                model_scope=inc.model_scope,
                incident_type=inc.incident_type,
                detail_code=inc.detail_code,
                evidence_tier=inc.evidence_tier,
                event_type="repeat_reminder",
            )
            for admin_id in admin_ids:
                if not is_valid_telegram_user_id(admin_id):
                    continue
                notif = AIIncidentNotificationModel(
                    incident_id=inc.id,
                    event_type="repeat_reminder",
                    cycle_index=target_cycle,
                    recipient_id=admin_id,
                    authorized_instance_id=companion_def["pm2_name"],
                    message_text=msg_text,
                    status="pending",
                    attempts=0,
                    max_attempts=MAX_OUTBOX_ATTEMPTS,
                    next_retry_at=now,
                )
                try:
                    async with session.begin_nested():
                        session.add(notif)
                        await session.flush()
                    reminders_created += 1
                except IntegrityError:
                    pass

        await session.commit()

    return reminders_created


async def process_ai_incident_notifications(
    bot: Bot | None = None,
    batch_size: int = 20,
    session_maker: Any = None,
) -> int:
    """Outbox worker: claims tasks strictly for authorized_instance_id, validates permissions, and dispatches."""
    current_inst = detect_current_instance()
    if not current_inst:
        logger.error("[AI Incident Worker] Fail-closed: cannot determine instance identity.")
        return 0
    current_bot_instance_id = current_inst["pm2_name"]

    sm = session_maker or async_session_maker
    now = datetime.utcnow()
    delivered_count = 0

    # 1. Scan candidates strictly matching authorized_instance_id with correct AND/OR precedence
    candidates: Sequence[int] = []
    async with sm() as session:
        scan_stmt = (
            select(AIIncidentNotificationModel.id)
            .where(
                and_(
                    AIIncidentNotificationModel.authorized_instance_id == current_bot_instance_id,
                    or_(
                        and_(
                            AIIncidentNotificationModel.status == "pending",
                            AIIncidentNotificationModel.next_retry_at <= now,
                        ),
                        and_(
                            AIIncidentNotificationModel.status == "processing",
                            AIIncidentNotificationModel.lease_until < now,
                        ),
                    ),
                )
            )
            .order_by(AIIncidentNotificationModel.next_retry_at.asc())
            .limit(batch_size)
        )
        candidates = (await session.execute(scan_stmt)).scalars().all()

    if not candidates:
        return 0

    # Dispatch via provided Bot or create verified Bot from current_inst token
    async def _do_send(row_id: int) -> bool:
        token = str(uuid.uuid4())
        lease_duration = timedelta(seconds=OUTBOX_LEASE_SECONDS)
        ts_now = datetime.utcnow()

        async with sm() as session:
            claim_stmt = (
                update(AIIncidentNotificationModel)
                .where(
                    AIIncidentNotificationModel.id == row_id,
                    AIIncidentNotificationModel.authorized_instance_id == current_bot_instance_id,
                    or_(
                        and_(
                            AIIncidentNotificationModel.status == "pending",
                            AIIncidentNotificationModel.next_retry_at <= ts_now,
                        ),
                        and_(
                            AIIncidentNotificationModel.status == "processing",
                            AIIncidentNotificationModel.lease_until < ts_now,
                        ),
                    ),
                )
                .values(
                    status="processing",
                    claim_token=token,
                    lease_until=ts_now + lease_duration,
                    attempts=AIIncidentNotificationModel.attempts + 1,
                    updated_at=ts_now,
                )
            )
            res = await session.execute(claim_stmt)
            await session.commit()
            if getattr(res, "rowcount", 0) != 1:
                return False

            row = await session.get(AIIncidentNotificationModel, row_id)
            if not row:
                return False

            # Pre-Send Check 1: Parent incident freshness
            parent_inc = await session.get(AIProviderIncidentModel, row.incident_id)
            if not parent_inc or (parent_inc.status != "active" and row.event_type in ("opened", "repeat_reminder", "fallback_used")):
                row.status = "cancelled"
                row.last_error = f"incident_{parent_inc.status if parent_inc else 'missing'}_before_send"
                await session.commit()
                return False

            # Pre-Send Check 2: Dynamic recipient permissions verification
            target_inst = get_instance_by_pm2_name(row.authorized_instance_id) or current_inst
            allowed_admins = await resolve_target_admin_ids(target_inst, current_instance=current_inst, session=session)
            if row.recipient_id not in allowed_admins or not is_valid_telegram_user_id(row.recipient_id):
                row.status = "failed_terminal"
                row.last_error = "recipient_permission_revoked_or_invalid"
                await session.commit()
                logger.warning(
                    "[AI Incident Worker] Dropped delivery for revoked/invalid recipient_id=%s via %s",
                    row.recipient_id, row.authorized_instance_id,
                )
                return False

            msg_text = row.message_text
            recipient_id = row.recipient_id
            attempts = row.attempts

        # Send HTTP RPC outside DB transaction
        send_success = False
        error_reason: str | None = None
        is_terminal = False
        retry_delay = BACKOFF_DELAYS[min(attempts - 1, len(BACKOFF_DELAYS) - 1)]

        try:
            if bot is not None:
                await bot.send_message(recipient_id, msg_text, parse_mode="HTML")
            else:
                from telegram_client import create_telegram_bot
                bot_token = os.environ.get("BOT_TOKEN")
                if not bot_token:
                    logger.error("[AI Incident Worker] Missing BOT_TOKEN for dispatch.")
                    return False
                async with create_telegram_bot(bot_token) as t_bot:
                    await t_bot.send_message(recipient_id, msg_text, parse_mode="HTML")
            send_success = True
        except TelegramForbiddenError as exc:
            send_success = False
            error_reason = str(exc)
            is_terminal = True
        except TelegramBadRequest as exc:
            send_success = False
            error_reason = str(exc)
            if is_bot_blocked_reason(str(exc)):
                is_terminal = True
            elif "chat not found" in str(exc).lower():
                is_terminal = True
        except TelegramRetryAfter as exc:
            send_success = False
            error_reason = str(exc)
            retry_delay = timedelta(seconds=getattr(exc, "retry_after", 30))
        except Exception as exc:
            send_success = False
            error_reason = str(exc)
            if is_bot_blocked_reason(str(exc)):
                is_terminal = True

        # Post-Send CAS Update
        ts_post = datetime.utcnow()
        async with sm() as session:
            if send_success:
                stmt = (
                    update(AIIncidentNotificationModel)
                    .where(
                        AIIncidentNotificationModel.id == row_id,
                        AIIncidentNotificationModel.claim_token == token,
                        AIIncidentNotificationModel.status == "processing",
                    )
                    .values(
                        status="delivered",
                        delivered_at=ts_post,
                        updated_at=ts_post,
                    )
                )
                await session.execute(stmt)
                await session.commit()
                return True
            else:
                new_status = "failed_terminal" if is_terminal or attempts >= MAX_OUTBOX_ATTEMPTS else "pending"
                stmt = (
                    update(AIIncidentNotificationModel)
                    .where(
                        AIIncidentNotificationModel.id == row_id,
                        AIIncidentNotificationModel.claim_token == token,
                        AIIncidentNotificationModel.status == "processing",
                    )
                    .values(
                        status=new_status,
                        last_error=error_reason[:500] if error_reason else "unknown_error",
                        next_retry_at=ts_post + retry_delay,
                        updated_at=ts_post,
                    )
                )
                await session.execute(stmt)
                await session.commit()
                return False

    for c_id in candidates:
        if await _do_send(c_id):
            delivered_count += 1

    return delivered_count
