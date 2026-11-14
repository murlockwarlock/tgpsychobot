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
- Safe typed error reasons (no raw exception strings or payload leaks in DB).
- Absolute fail-safety: incident manager exceptions never break operational AI calls.
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
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramUnauthorizedError,
)
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
    extract_telegram_bot_id,
    get_instance_by_id,
    get_instance_by_pm2_name,
    is_valid_telegram_user_id,
    resolve_companion_telegram_instance,
    resolve_target_admin_ids,
    resolve_target_bot_token,
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

    if incident_type == "low_credit_warning":
        if event_type == "resolved":
            return (
                f"✅ <b>Баланс KIE пополнен</b>\n\n"
                f"Провайдер: <code>{p_esc}</code>\n"
                f"Время: <code>{time_str}</code>\n"
                f"Статус: Баланс превысил порог предупреждения."
            )
        return (
            f"⚠️ <b>Предупреждение KIE: низкий баланс</b>\n\n"
            f"Провайдер: <code>{p_esc}</code>\n"
            f"Код: <code>{code_esc}</code>\n"
            f"Время: <code>{time_str}</code>"
        )

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

    prefix = (
        "🚨 <b>Сбой AI-провайдера</b>"
        if event_type == "opened"
        else "⏰ <b>Напоминание: Сбой AI-провайдера сохраняется</b>"
    )
    return (
        f"{prefix}\n\n"
        f"Провайдер: <code>{p_esc}</code>\n"
        f"Канал: <code>{op_esc}</code>\n"
        f"Модель: <code>{mod_esc}</code>\n"
        f"Тип сбоя: <code>{html.escape(incident_type)}</code>\n"
        f"Код: <code>{code_esc}</code>\n"
        f"Уровень: {tier_label}\n"
        f"Время: <code>{time_str}</code>\n\n"
        f"Требуется вмешательство оператора."
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
    """
    Record an AI provider failure, opening or updating an active incident and enqueuing outbox events.
    Fail-safe: catches all internal exceptions and logs warning, never raising to caller.
    """
    try:
        return await _record_ai_provider_failure_impl(
            origin_instance_id=origin_instance_id,
            provider=provider,
            error=error,
            operation=operation,
            model=model,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
            fallback_succeeded=fallback_succeeded,
            credential_scope=credential_scope,
            session_maker=session_maker,
        )
    except Exception as exc:
        logger.warning("[AI Incident] Unexpected error recording failure for %s: %s", provider, exc)
        return None


async def _record_ai_provider_failure_impl(
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

    # Verify origin bot token consistency
    source_platform = source_def.get("platform")
    if source_platform == "telegram":
        env_token = os.environ.get("BOT_TOKEN")
        expected_pid = source_def.get("platform_id")
        if env_token and expected_pid:
            token_pid = extract_telegram_bot_id(env_token)
            if token_pid != expected_pid:
                logger.critical(
                    "[AI Incident] Fail-closed: BOT_TOKEN mismatch (token ID %s != expected %s).",
                    token_pid,
                    expected_pid,
                )
                return None

    companion_def = resolve_companion_telegram_instance(source_def)
    if not companion_def:
        logger.error("[AI Incident] No companion Telegram instance found for %s. Dropping incident.", origin_instance_id)
        return None

    if source_platform == "max":
        comp_env_token = os.environ.get("COMPANION_TELEGRAM_BOT_TOKEN")
        comp_pid = companion_def.get("platform_id")
        if comp_env_token and comp_pid:
            token_pid = extract_telegram_bot_id(comp_env_token)
            if token_pid != comp_pid:
                logger.critical(
                    "[AI Incident] Fail-closed: COMPANION_TELEGRAM_BOT_TOKEN mismatch (%s != %s).",
                    token_pid,
                    comp_pid,
                )
                return None

    companion_instance_id = companion_def["pm2_name"]
    authorized_instance_id = companion_instance_id  # Delivery worker belongs strictly to companion Telegram bot

    # Normalize error to canonical ProviderFailure
    existing_failure = getattr(error, "failure", None)
    failure = (
        existing_failure
        if isinstance(existing_failure, ProviderFailure)
        else normalize_exception(error, provider=provider, operation=operation)
    )

    # Transient or non-operator failures do NOT create long-lived operator incidents
    if not failure.operator_action_required:
        logger.info(
            "[AI Incident] Non-operator failure for %s (%s, %s): not creating operator incident.",
            provider,
            operation,
            failure.classification,
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
                detail_code=det_code,
                evidence_tier=evidence_tier,
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
    """
    Record an operational success, resolving any active matching incident strictly within this scope.
    Fail-safe: catches all internal exceptions and logs warning, never raising to caller.
    """
    try:
        return await _record_ai_operation_success_impl(
            origin_instance_id=origin_instance_id,
            provider=provider,
            credential_scope=credential_scope,
            operation=operation,
            model=model,
            session_maker=session_maker,
        )
    except Exception as exc:
        logger.warning("[AI Incident] Unexpected error recording operation success for %s: %s", provider, exc)
        return False


async def _record_ai_operation_success_impl(
    origin_instance_id: str | None,
    provider: str,
    credential_scope: str,
    operation: str,
    model: str | None = None,
    session_maker: Any = None,
) -> bool:
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
        active_incidents = (await session.execute(stmt)).scalars().all()
        if not active_incidents:
            return False

        admin_ids = (
            await resolve_target_admin_ids(companion_def, current_instance=source_def, session=session)
            if companion_def
            else ()
        )

        for inc in active_incidents:
            inc.status = "resolved"
            inc.resolved_at = now
            resolved_any = True

            # Cancel any pending/processing 'opened', 'repeat_reminder' or 'fallback_used' notifications
            await session.execute(
                update(AIIncidentNotificationModel)
                .where(
                    AIIncidentNotificationModel.incident_id == inc.id,
                    AIIncidentNotificationModel.status.in_(["pending", "processing"]),
                )
                .values(
                    status="cancelled",
                    last_error="superseded_by_recovery",
                    updated_at=now,
                )
            )

            # Enqueue single recovery notification per admin
            if admin_ids:
                msg_text = _format_incident_message_text(
                    provider=prov_norm,
                    operation=op_norm,
                    model_scope=inc.model_scope,
                    incident_type=inc.incident_type,
                    detail_code=inc.detail_code or "recovered",
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

        await session.commit()

    if resolved_any:
        logger.info(
            "[AI Incident] Operational recovery confirmed for %s (%s, %s, %s). Resolved %d active incidents.",
            prov_norm,
            credential_scope,
            op_norm,
            mod_norm,
            len(active_incidents),
        )

    return resolved_any


async def handle_key_rotation(
    origin_instance_id: str | None,
    provider: str,
    new_key: str,
    base_url: str | None = None,
    session_maker: Any = None,
) -> int:
    """
    Supersede active incidents when an API key is rotated in AIConfig, without false replenishment alerts.
    Supersedes across the database, ensuring Telegram and companion MAX instances in shared DB are both transitioned.
    """
    try:
        return await _handle_key_rotation_impl(
            origin_instance_id=origin_instance_id,
            provider=provider,
            new_key=new_key,
            base_url=base_url,
            session_maker=session_maker,
        )
    except Exception as exc:
        logger.warning("[AI Incident] Unexpected error in handle_key_rotation: %s", exc)
        return 0


async def _handle_key_rotation_impl(
    origin_instance_id: str | None,
    provider: str,
    new_key: str,
    base_url: str | None = None,
    session_maker: Any = None,
) -> int:
    prov_norm = provider.strip().upper()
    new_fp = get_provider_account_fingerprint(new_key, base_url)
    sm = session_maker or async_session_maker
    now = datetime.utcnow()
    superseded_count = 0

    async with sm() as session:
        # Supersede ALL active incidents in this DB for provider whose credential_scope != new_fp
        stmt = select(AIProviderIncidentModel).where(
            AIProviderIncidentModel.provider == prov_norm,
            AIProviderIncidentModel.credential_scope != new_fp,
            AIProviderIncidentModel.status == "active",
        )
        active_old = (await session.execute(stmt)).scalars().all()
        for inc in active_old:
            inc.status = "superseded"
            inc.resolved_at = now
            superseded_count += 1

            # Cancel pending/processing notifications without sending alerts
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

        await session.commit()

    logger.info(
        "[AI Incident] Key rotation for %s: superseded %d active incidents (new fp=%s).",
        prov_norm,
        superseded_count,
        new_fp,
    )
    return superseded_count


async def check_incident_repeat_reminders(
    bot: Bot | None = None,
    repeat_interval_seconds: int = DEFAULT_REPEAT_ALERT_SECONDS,
    session_maker: Any = None,
    repeat_seconds: int | None = None,
    **kwargs: Any,
) -> int:
    """Check active incidents and enqueue repeat reminder outbox rows if cooldown elapsed."""
    if repeat_seconds is not None:
        repeat_interval_seconds = repeat_seconds
    current_inst = detect_current_instance()
    if not current_inst:
        return 0
    current_bot_instance_id = current_inst["pm2_name"]

    sm = session_maker or async_session_maker
    now = datetime.utcnow()
    threshold = now - timedelta(seconds=repeat_interval_seconds)
    reminders_created = 0

    async with sm() as session:
        stmt = select(AIProviderIncidentModel).where(
            AIProviderIncidentModel.origin_instance_id == current_bot_instance_id,
            AIProviderIncidentModel.status == "active",
            AIProviderIncidentModel.last_alerted_at <= threshold,
        )
        stale_incidents = (await session.execute(stmt)).scalars().all()
        if not stale_incidents:
            return 0

        companion_def = resolve_companion_telegram_instance(current_inst)
        if not companion_def:
            return 0
        admin_ids = await resolve_target_admin_ids(companion_def, current_instance=current_inst, session=session)
        if not admin_ids:
            return 0

        authorized_instance_id = companion_def["pm2_name"]

        for inc in stale_incidents:
            new_cycle = inc.reminder_cycle + 1
            inc.reminder_cycle = new_cycle
            inc.last_alerted_at = now

            msg_text = _format_incident_message_text(
                provider=inc.provider,
                operation=inc.operation,
                model_scope=inc.model_scope,
                incident_type=inc.incident_type,
                detail_code=inc.detail_code or "persisting",
                evidence_tier=inc.evidence_tier,
                event_type="repeat_reminder",
            )

            for admin_id in admin_ids:
                if not is_valid_telegram_user_id(admin_id):
                    continue
                notif = AIIncidentNotificationModel(
                    incident_id=inc.id,
                    event_type="repeat_reminder",
                    cycle_index=new_cycle,
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
    """
    Outbox worker: claims tasks strictly for authorized_instance_id, validates bot identity and permissions,
    and dispatches via Telegram.
    """
    current_inst = detect_current_instance()
    if not current_inst:
        logger.error("[AI Incident Worker] Fail-closed: cannot determine instance identity.")
        return 0
    if current_inst.get("platform") != "telegram":
        logger.error(
            "[AI Incident Worker] Worker running on non-Telegram instance %s. Refusing delivery.",
            current_inst.get("pm2_name"),
        )
        return 0

    current_bot_instance_id = current_inst["pm2_name"]
    expected_pid = current_inst.get("platform_id")

    # Strict Bot Identity Verification
    actual_bot = bot
    if actual_bot is not None:
        cand_token = getattr(actual_bot, "token", None)
        token_pid = extract_telegram_bot_id(cand_token) if cand_token else None
        if token_pid != expected_pid:
            logger.critical(
                "[AI Incident Worker] Foreign bot token supplied: token ID %s != expected %s for %s. Refusing delivery.",
                token_pid,
                expected_pid,
                current_bot_instance_id,
            )
            return 0
    else:
        resolved_tok = resolve_target_bot_token(current_inst, is_companion=False)
        if not resolved_tok or extract_telegram_bot_id(resolved_tok) != expected_pid:
            logger.critical(
                "[AI Incident Worker] Valid target bot token missing for %s. Fail-closed.",
                current_bot_instance_id,
            )
            return 0
        actual_bot = Bot(token=resolved_tok)

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

    # Dispatch via verified Bot
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
            if res.rowcount == 0:
                return False  # Lost race or lease taken

            # Fetch claimed row
            row = await session.get(AIIncidentNotificationModel, row_id)
            if not row or row.claim_token != token:
                return False

            # Check parent incident freshness: if resolved/superseded before sending 'opened' or 'repeat_reminder', skip!
            parent_inc = await session.get(AIProviderIncidentModel, row.incident_id)
            if parent_inc and parent_inc.status != "active" and row.event_type in ("opened", "repeat_reminder", "fallback_used"):
                row.status = "skipped_stale"
                row.last_error = f"incident_{parent_inc.status}_before_send"
                row.updated_at = ts_now
                await session.commit()
                return False

            # Dynamic Admin Authorization Verification
            allowed_admins = await resolve_target_admin_ids(current_inst, current_instance=current_inst, session=session)
            if row.recipient_id not in allowed_admins:
                row.status = "skipped_unauthorized"
                row.last_error = "recipient_permission_revoked_or_invalid"
                row.updated_at = ts_now
                await session.commit()
                return False

            attempts = row.attempts

        # Dispatch message via actual_bot
        send_success = False
        is_terminal = False
        retry_delay = BACKOFF_DELAYS[min(attempts, len(BACKOFF_DELAYS) - 1)]
        error_code = "delivery_success"

        try:
            await actual_bot.send_message(
                chat_id=row.recipient_id,
                text=row.message_text,
                parse_mode="HTML",
            )
            send_success = True
        except TelegramForbiddenError:
            send_success = False
            is_terminal = True
            error_code = "telegram_bot_blocked"
        except TelegramBadRequest as exc:
            send_success = False
            exc_str = str(exc).lower()
            if is_bot_blocked_reason(str(exc)):
                is_terminal = True
                error_code = "telegram_bot_blocked"
            elif "chat not found" in exc_str:
                is_terminal = True
                error_code = "telegram_chat_not_found"
            else:
                error_code = "telegram_bad_request"
        except TelegramRetryAfter as exc:
            send_success = False
            error_code = "telegram_rate_limited"
            retry_delay = timedelta(seconds=getattr(exc, "retry_after", 30))
        except TelegramNetworkError:
            send_success = False
            error_code = "telegram_network_error"
        except TelegramUnauthorizedError:
            send_success = False
            is_terminal = True
            error_code = "telegram_unauthorized_token"
        except Exception as exc:
            send_success = False
            if is_bot_blocked_reason(str(exc)):
                is_terminal = True
                error_code = "telegram_bot_blocked"
            else:
                error_code = "unexpected_delivery_error"

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
                        last_error="delivered_ok",
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
                        last_error=error_code,
                        next_retry_at=ts_post + retry_delay,
                        updated_at=ts_post,
                    )
                )
                await session.execute(stmt)
                await session.commit()
                return False

    for cand_id in candidates:
        if await _do_send(cand_id):
            delivered_count += 1

    return delivered_count
