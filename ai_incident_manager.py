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

import asyncio
import hashlib
import html
import logging
import math
import os
import re
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
from sqlalchemy import and_, case, delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from database import (
    AIIncidentNotificationModel,
    AIProviderIncidentModel,
    async_session_maker,
)
from notification_routing import (
    detect_current_instance,
    extract_telegram_bot_id,
    get_instance_by_id,
    get_instance_by_pm2_name,
    is_valid_telegram_user_id,
    resolve_companion_telegram_instance,
    resolve_target_admin_ids,
    resolve_target_bot_token,
    verify_bot_identity_live,
)
from notification_transport import is_bot_blocked_reason
from provider_failures import (
    CANONICAL_CLASSES,
    KNOWN_EXCEPTION_NAMES,
    KNOWN_TECHNICAL_CODES,
    ProviderFailure,
    canonical_classification,
    normalize_exception,
)

logger = logging.getLogger("ai_incident_manager")


class IncidentSecurityError(RuntimeError):
    """Raised when instance identity cannot be determined or origin instance is unauthorized."""
    pass

DEFAULT_REPEAT_ALERT_SECONDS = 6 * 3600  # 6 hours
OUTBOX_LEASE_SECONDS = 60
MAX_OUTBOX_RETRIES = 5
MAX_OUTBOX_ATTEMPTS = 6  # 1 initial attempt + 5 retries = 6 total attempts

AI_INCIDENT_BACKOFF_DELAYS: tuple[timedelta, ...] = (
    timedelta(seconds=15),
    timedelta(minutes=1),
    timedelta(minutes=5),
    timedelta(minutes=15),
    timedelta(hours=1),
)


def get_ai_incident_outbox_backoff(attempts: int) -> timedelta:
    """
    Return the canonical AI incident backoff delay for the given attempt count.
    Claimed attempts is 1-based (after claim, attempts is 1 on first failure):
      attempt 1 (initial attempt failed) -> 15 seconds (before retry 1)
      attempt 2 (retry 1 failed) -> 1 minute (before retry 2)
      attempt 3 (retry 2 failed) -> 5 minutes (before retry 3)
      attempt 4 (retry 3 failed) -> 15 minutes (before retry 4)
      attempt 5 (retry 4 failed) -> 1 hour (before retry 5 / attempt 6)
      attempt >= 6 -> terminal failure
    """
    idx = min(max(0, attempts - 1), len(AI_INCIDENT_BACKOFF_DELAYS) - 1)
    return AI_INCIDENT_BACKOFF_DELAYS[idx]


def sanitize_technical_code(
    code: Any,
    classification: str | None = None,
    default: str = "unspecified",
) -> str:
    """
    Sanitize and validate an error/detail code before persisting or formatting.
    Guarantees no arbitrary strings, user data, or PII can ever be stored or rendered.
    """
    if code is None:
        return default
    s = str(code).strip()
    if not s:
        return default
    s_lower = s.lower()
    if s in KNOWN_EXCEPTION_NAMES:
        return s
    if s_lower in KNOWN_TECHNICAL_CODES or s_lower in CANONICAL_CLASSES:
        return s[:64]
    if s.isdigit() and len(s) == 3 and (400 <= int(s) <= 599):  # 3-digit HTTP status code
        return s
    # Check if code has low-credit pattern "balance_X.XX_below_Y.YY"
    if re.match(r"^balance_\d+(\.\d+)?_below_\d+(\.\d+)?$", s):
        return s
    if re.match(r"^inferred_[a-z0-9_]{1,50}$", s_lower):
        return s[:64]
    if s in ("superseded_by_key_rotation", "recovered", "persisting"):
        return s
    # Fallback to valid canonical classification if available
    if classification:
        canon = canonical_classification(classification)
        if canon and canon in CANONICAL_CLASSES and canon != "unknown":
            return canon
    return default


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
    safe_code = sanitize_technical_code(detail_code, classification=incident_type, default="unspecified")
    code_esc = html.escape(safe_code)
    time_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    safe_inc_type = canonical_classification(incident_type)
    if not safe_inc_type or safe_inc_type == "unknown":
        if incident_type in ("low_credit_warning",):
            safe_inc_type = incident_type
        else:
            safe_inc_type = "provider_failure"
    inc_type_esc = html.escape(safe_inc_type)

    if incident_type == "low_credit_warning":
        if event_type == "resolved":
            return (
                f"✅ <b>Баланс {p_esc} пополнен</b>\n\n"
                f"Провайдер: <code>{p_esc}</code>\n"
                f"Время: <code>{time_str}</code>\n"
                f"Статус: Баланс превысил порог предупреждения."
            )
        if event_type == "repeat_reminder":
            return (
                f"⏰ <b>Напоминание: низкий баланс {p_esc}</b>\n\n"
                f"Провайдер: <code>{p_esc}</code>\n"
                f"Код: <code>{code_esc}</code>\n"
                f"Время: <code>{time_str}</code>\n"
                f"Статус: Баланс остается ниже порога предупреждения."
            )
        return (
            f"⚠️ <b>Предупреждение {p_esc}: низкий баланс</b>\n\n"
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
            f"Сбой: <code>{code_esc}</code> ({inc_type_esc})\n"
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
        f"Тип сбоя: <code>{inc_type_esc}</code>\n"
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
    enqueue_notification: bool = True,
    attempt_started_at: datetime | None = None,
) -> AIProviderIncidentModel | None:
    """
    Record an AI provider failure, opening or updating an active incident and enqueuing outbox events.
    Fail-safe: catches all internal exceptions and logs error, never raising to caller.
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
            enqueue_notification=enqueue_notification,
            attempt_started_at=attempt_started_at,
        )
    except Exception as exc:
        logger.error("[AI Incident] Unexpected error recording failure for %s: %s", provider, exc, exc_info=True)
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
    enqueue_notification: bool = True,
    attempt_started_at: datetime | None = None,
) -> AIProviderIncidentModel | None:
    current_inst = detect_current_instance()
    if not current_inst:
        logger.critical("[AI Incident] Fail-closed: cannot determine bot instance identity.")
        return None

    if not origin_instance_id:
        origin_instance_id = current_inst["pm2_name"]
        source_def = current_inst
    else:
        source_def = get_instance_by_pm2_name(origin_instance_id)
        if not source_def:
            logger.critical("[AI Incident] Fail-closed: origin_instance_id %r not in registry.", origin_instance_id)
            return None

        # Verify origin belongs to calling instance or valid companion relation
        is_same = (
            source_def.get("pm2_name") == current_inst.get("pm2_name")
            or source_def.get("bot_instance_id") == current_inst.get("bot_instance_id")
        )
        is_companion = (
            source_def.get("companion_instance") in (current_inst.get("pm2_name"), current_inst.get("bot_instance_id"))
            or current_inst.get("companion_instance") in (source_def.get("pm2_name"), source_def.get("bot_instance_id"))
        )
        if not (is_same or is_companion):
            logger.critical(
                "[AI Incident] Fail-closed: origin_instance_id %r does not belong to current instance %r and is not a valid companion.",
                origin_instance_id,
                current_inst.get("pm2_name"),
            )
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

    prov_norm = (provider or failure.provider or "unknown").strip().upper()
    op_norm = (operation or failure.operation or "chat").strip().lower()
    mod_norm = str(model or "*").strip()
    evidence_tier = (
        "CONFIRMED_STRUCTURED"
        if (
            failure.classification_source in ("provider_structured", "http_status_fallback", "transport", "sdk_typed_exception")
            or failure.http_status is not None
        )
        else "INFERRED_MESSAGE"
    )
    raw_cls = getattr(failure, "classification", None)
    canon_cls = canonical_classification(raw_cls)
    if evidence_tier == "CONFIRMED_STRUCTURED":
        if canon_cls and canon_cls in CANONICAL_CLASSES and canon_cls != "unknown":
            inc_type = canon_cls
        elif str(raw_cls) in ("low_credit_warning",):
            inc_type = str(raw_cls)
        else:
            inc_type = "unknown"
        det_code = sanitize_technical_code(
            failure.detail_code or failure.provider_error_code,
            classification=inc_type,
            default="unspecified",
        )
    else:
        # Inferred diagnostic text without structured proof:
        # Prevent unconfirmed provider errors (e.g. unverified provider_5xx or auth)
        # from falsely polluting database incident metrics in incident_type.
        if canon_cls in ("application_internal", "unknown"):
            inc_type = canon_cls
            det_code = sanitize_technical_code(
                failure.detail_code or failure.provider_error_code,
                classification=inc_type,
                default="unspecified",
            )
        else:
            inc_type = "unknown"
            inferred_code = f"inferred_{canon_cls}" if canon_cls else "inferred_unspecified"
            det_code = sanitize_technical_code(
                inferred_code,
                classification="unknown",
                default="inferred_unspecified",
            )

    sm = session_maker or async_session_maker
    now = datetime.utcnow()

    async with sm() as session:
        from database import AIConfig
        ai_cfg = await session.get(AIConfig, 1)
        current_api_k = get_provider_api_key(ai_cfg, prov_norm) if ai_cfg else None
        current_b_url = getattr(ai_cfg, "kie_base_url", None) if (ai_cfg and prov_norm == "KIE") else None
        current_active_scope = (
            get_provider_account_fingerprint(current_api_k, current_b_url)
            if current_api_k
            else None
        )

        if not credential_scope:
            logger.warning("[AI Incident] Credential scope not passed for %s (%s); reading current AIConfig.", prov_norm, op_norm)
            credential_scope = current_active_scope or "unconfigured"

        # Check if this credential scope was already retired/superseded by a key rotation
        stmt_superseded = (
            select(AIProviderIncidentModel)
            .where(
                AIProviderIncidentModel.provider == prov_norm,
                AIProviderIncidentModel.credential_scope == credential_scope,
                AIProviderIncidentModel.status == "superseded",
                or_(
                    AIProviderIncidentModel.origin_instance_id == origin_instance_id,
                    AIProviderIncidentModel.companion_instance_id == origin_instance_id,
                ),
            )
            .order_by(AIProviderIncidentModel.resolved_at.desc())
            .limit(1)
        )
        existing_superseded = await session.scalar(stmt_superseded)

        is_retired = False
        if existing_superseded is not None:
            # Check verified configuration lifecycle: if this credential_scope matches currently configured key,
            # it was reactivated (A -> B -> A) and may be active again.
            is_active_now = False
            if current_api_k:
                if credential_scope in (
                    get_provider_account_fingerprint(current_api_k, current_b_url),
                    get_provider_account_fingerprint(current_api_k, None),
                ):
                    is_active_now = True

            if not is_active_now:
                is_retired = True
                logger.info(
                    "[AI Incident] Finished AI attempt for %s belongs to retired credential %s (current active is %s). Preserving record as superseded.",
                    prov_norm,
                    credential_scope,
                    current_active_scope,
                )
            elif attempt_started_at and existing_superseded.resolved_at:
                # If this specific request was started BEFORE the key was rotated out at resolved_at,
                # it belongs to the previous superseded generation and must NOT open a new incident.
                att_dt = attempt_started_at.replace(tzinfo=None) if attempt_started_at.tzinfo else attempt_started_at
                res_dt = existing_superseded.resolved_at.replace(tzinfo=None) if existing_superseded.resolved_at.tzinfo else existing_superseded.resolved_at
                if att_dt <= res_dt:
                    is_retired = True
                    logger.info(
                        "[AI Incident] Finished AI attempt for %s started at %s before rotation resolved_at %s. Preserving record as superseded.",
                        prov_norm,
                        attempt_started_at,
                        existing_superseded.resolved_at,
                    )

        is_operator_incident = bool(failure.operator_action_required) and evidence_tier == "CONFIRMED_STRUCTURED"
        if is_retired:
            target_status = "superseded"
            enqueue_notification = False
        elif is_operator_incident:
            target_status = "active"
        else:
            target_status = "transient"
            enqueue_notification = False

        # 1. Atomic query / UPSERT for this target status
        stmt = select(AIProviderIncidentModel).where(
            AIProviderIncidentModel.origin_instance_id == origin_instance_id,
            AIProviderIncidentModel.provider == prov_norm,
            AIProviderIncidentModel.credential_scope == credential_scope,
            AIProviderIncidentModel.operation == op_norm,
            AIProviderIncidentModel.model_scope == mod_norm,
            AIProviderIncidentModel.incident_type == inc_type,
            AIProviderIncidentModel.status == target_status,
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
                http_status=getattr(failure, "http_status", None),
                classification_source=getattr(failure, "classification_source", "provider_structured"),
                status=target_status,
                occurrence_count=1,
                reminder_cycle=0,
                first_seen_at=now,
                last_seen_at=now,
                last_alerted_at=now,
                resolved_at=now if target_status == "superseded" else None,
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
                        http_status=getattr(failure, "http_status", None),
                    )
                )

        # 2. Enqueue notification event if newly opened active operator incident or fallback_used
        event_type = "fallback_used" if fallback_succeeded else "opened"
        if target_status == "active" and enqueue_notification and (is_new or fallback_succeeded):
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
    attempt_started_at: datetime | None = None,
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
            attempt_started_at=attempt_started_at,
        )
    except Exception as exc:
        logger.error("[AI Incident] Unexpected error recording operation success for %s: %s", provider, exc, exc_info=True)
        return False


async def _record_ai_operation_success_impl(
    origin_instance_id: str | None,
    provider: str,
    credential_scope: str,
    operation: str,
    model: str | None = None,
    session_maker: Any = None,
    attempt_started_at: datetime | None = None,
) -> bool:
    current_inst = detect_current_instance()
    if not current_inst:
        logger.critical("[AI Incident] Fail-closed: cannot determine bot instance identity for success recording.")
        return False

    if not origin_instance_id:
        origin_instance_id = current_inst["pm2_name"]
        source_def = current_inst
    else:
        source_def = get_instance_by_pm2_name(origin_instance_id)
        if not source_def:
            logger.critical("[AI Incident] Fail-closed: origin_instance_id %r not in registry for success recording.", origin_instance_id)
            return False

        is_same = (
            source_def.get("pm2_name") == current_inst.get("pm2_name")
            or source_def.get("bot_instance_id") == current_inst.get("bot_instance_id")
        )
        is_companion = (
            source_def.get("companion_instance") in (current_inst.get("pm2_name"), current_inst.get("bot_instance_id"))
            or current_inst.get("companion_instance") in (source_def.get("pm2_name"), source_def.get("bot_instance_id"))
        )
        if not (is_same or is_companion):
            logger.critical(
                "[AI Incident] Fail-closed: origin_instance_id %r does not belong to current instance %r and is not a valid companion.",
                origin_instance_id,
                current_inst.get("pm2_name"),
            )
            return False

    companion_def = resolve_companion_telegram_instance(source_def) if source_def else None
    authorized_instance_id = companion_def["pm2_name"] if companion_def else origin_instance_id

    prov_norm = provider.strip().upper()
    op_norm = operation.strip().lower()
    mod_norm = str(model or "*").strip()

    sm = session_maker or async_session_maker
    now = datetime.utcnow()
    resolved_any = False

    async with sm() as session:
        # Check verified lifecycle: if this attempt started before this credential was rotated out,
        # it belongs to a superseded credential generation and must NOT resolve new incidents.
        if attempt_started_at:
            att_dt = attempt_started_at.replace(tzinfo=None) if attempt_started_at.tzinfo else attempt_started_at
            stmt_superseded = (
                select(AIProviderIncidentModel)
                .where(
                    AIProviderIncidentModel.provider == prov_norm,
                    AIProviderIncidentModel.credential_scope == credential_scope,
                    AIProviderIncidentModel.status == "superseded",
                    or_(
                        AIProviderIncidentModel.origin_instance_id == origin_instance_id,
                        AIProviderIncidentModel.companion_instance_id == origin_instance_id,
                    ),
                )
                .order_by(AIProviderIncidentModel.resolved_at.desc())
                .limit(1)
            )
            existing_superseded = await session.scalar(stmt_superseded)
            if existing_superseded and existing_superseded.resolved_at:
                res_dt = existing_superseded.resolved_at.replace(tzinfo=None) if existing_superseded.resolved_at.tzinfo else existing_superseded.resolved_at
                if att_dt <= res_dt:
                    logger.info(
                        "[AI Incident] Finished AI operation success for %s started at %s before rotation resolved_at %s. Ignoring stale success.",
                        prov_norm,
                        attempt_started_at,
                        existing_superseded.resolved_at,
                    )
                    return False

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

        # An active incident cannot be resolved by an attempt that was started before the incident occurred
        filtered_incidents = []
        for inc in active_incidents:
            if attempt_started_at and inc.first_seen_at:
                att_dt = attempt_started_at.replace(tzinfo=None) if attempt_started_at.tzinfo else attempt_started_at
                inc_seen_dt = inc.first_seen_at.replace(tzinfo=None) if inc.first_seen_at.tzinfo else inc.first_seen_at
                if att_dt < inc_seen_dt:
                    logger.info(
                        "[AI Incident] Active incident %s (first seen %s) cannot be resolved by earlier attempt started at %s.",
                        inc.id,
                        inc.first_seen_at,
                        attempt_started_at,
                    )
                    continue
            filtered_incidents.append(inc)

        if not filtered_incidents:
            return False

        admin_ids = (
            await resolve_target_admin_ids(companion_def, current_instance=source_def, session=session)
            if companion_def
            else ()
        )

        for inc in filtered_incidents:
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
    old_key: str | None = None,
    old_credential_scope: str | None = None,
    session_maker: Any = None,
    session: AsyncSession | None = None,
) -> int:
    """
    Supersede active incidents when an API key is rotated in AIConfig, without false replenishment alerts.
    Supersedes across the database, ensuring Telegram and companion MAX instances in shared DB are both transitioned.
    Supports running within an existing database transaction (pass session) or managing its own.
    Fails closed with IncidentSecurityError if bot instance identity or authorization cannot be established.
    """
    try:
        return await _handle_key_rotation_impl(
            origin_instance_id=origin_instance_id,
            provider=provider,
            new_key=new_key,
            base_url=base_url,
            old_key=old_key,
            old_credential_scope=old_credential_scope,
            session_maker=session_maker,
            session=session,
        )
    except IncidentSecurityError:
        # Never mask security / authorization fail-closed errors
        raise
    except Exception as exc:
        logger.error("[AI Incident] Unexpected error in handle_key_rotation: %s", exc, exc_info=True)
        if session is not None:
            raise
        return 0


async def _handle_key_rotation_impl(
    origin_instance_id: str | None,
    provider: str,
    new_key: str,
    base_url: str | None = None,
    old_key: str | None = None,
    old_credential_scope: str | None = None,
    session_maker: Any = None,
    session: AsyncSession | None = None,
) -> int:
    current_inst = detect_current_instance()
    if not origin_instance_id:
        if not current_inst:
            logger.critical("[AI Incident] Fail-closed: cannot determine instance identity for key rotation.")
            raise IncidentSecurityError("Cannot determine current bot instance identity for key rotation.")
        source_def = current_inst
        origin_instance_id = current_inst.get("pm2_name")
    else:
        source_def = get_instance_by_pm2_name(origin_instance_id)
        if not source_def:
            logger.critical("[AI Incident] Fail-closed: origin_instance_id %r not in registry for key rotation.", origin_instance_id)
            raise IncidentSecurityError(f"Origin instance {origin_instance_id!r} not found in registry.")
        if current_inst:
            is_same = (
                source_def.get("pm2_name") == current_inst.get("pm2_name")
                or source_def.get("bot_instance_id") == current_inst.get("bot_instance_id")
            )
            is_companion = (
                source_def.get("companion_instance") in (current_inst.get("pm2_name"), current_inst.get("bot_instance_id"))
                or current_inst.get("companion_instance") in (source_def.get("pm2_name"), source_def.get("bot_instance_id"))
            )
            if not (is_same or is_companion):
                logger.critical(
                    "[AI Incident] Fail-closed: origin_instance_id %r not allowed from %r during key rotation.",
                    origin_instance_id,
                    current_inst.get("pm2_name"),
                )
                raise IncidentSecurityError(
                    f"Origin instance {origin_instance_id!r} not authorized from current instance {current_inst.get('pm2_name')!r}."
                )
        else:
            logger.critical("[AI Incident] Fail-closed: cannot determine instance identity for key rotation with origin %r.", origin_instance_id)
            raise IncidentSecurityError("Cannot determine current bot instance identity for key rotation.")

    prov_norm = provider.strip().upper()
    new_fp = get_provider_account_fingerprint(new_key, base_url)
    sm = session_maker or async_session_maker
    now = datetime.utcnow()

    # Determine origin and companion instance names to avoid touching foreign bot incidents in shared DB
    companion_def = resolve_companion_telegram_instance(source_def) if source_def else None
    origin_names = {source_def.get("pm2_name"), source_def.get("bot_instance_id")}
    if companion_def:
        origin_names.update({companion_def.get("pm2_name"), companion_def.get("bot_instance_id")})
    origin_names.discard(None)

    # Safe no-op check: if identical key is being submitted, do not touch any incidents
    if old_key is not None and old_key == new_key:
        logger.info("[AI Incident] No-op key rotation for %s: old_key matches new_key.", prov_norm)
        return 0
    if old_credential_scope is not None and old_credential_scope == new_fp:
        logger.info("[AI Incident] No-op key rotation for %s: old_credential_scope matches new_fp.", prov_norm)
        return 0

    old_fps: set[str] = set()
    old_fp_passed = False
    if old_credential_scope:
        old_fps.add(old_credential_scope)
        old_fp_passed = True
    elif old_key:
        old_b_url = base_url if prov_norm == "KIE" else None
        old_fps.add(get_provider_account_fingerprint(old_key, old_b_url))
        old_fps.add(get_provider_account_fingerprint(old_key, None))
        old_fp_passed = True
    old_fps.discard("unconfigured")
    old_fps.discard(new_fp)

    if old_fp_passed and not old_fps:
        logger.info("[AI Incident] No-op key rotation for %s: old fingerprint matches new_fp.", prov_norm)
        return 0

    async def _execute_rotation(sess: AsyncSession) -> int:
        nonlocal old_fps, old_fp_passed
        superseded_count = 0

        # If caller did not provide old key/scope, inspect AIConfig for this provider
        if not old_fp_passed:
            from database import AIConfig
            ai_cfg = await sess.get(AIConfig, 1)
            old_k = get_provider_api_key(ai_cfg, prov_norm) if ai_cfg else None
            old_b_url = getattr(ai_cfg, "kie_base_url", None) if (ai_cfg and prov_norm == "KIE") else None
            if old_k:
                if old_k == new_key:
                    logger.info("[AI Incident] No-op key rotation for %s: configured key matches new_key.", prov_norm)
                    return 0
                old_fps.add(get_provider_account_fingerprint(old_k, old_b_url))
                old_fps.add(get_provider_account_fingerprint(old_k, None))
                old_fps.discard("unconfigured")
                old_fps.discard(new_fp)

        # If no distinct prior key exists, do NOT perform a broad fallback (no credential_scope != new_fp)
        if not old_fps:
            logger.info("[AI Incident] No distinct prior key found for %s rotation; no incidents superseded.", prov_norm)
            return 0

        target_tombstones = set(old_fps)

        # Supersede active and transient incidents ONLY for the replaced credential scope(s)
        stmt = select(AIProviderIncidentModel).where(
            AIProviderIncidentModel.provider == prov_norm,
            AIProviderIncidentModel.status.in_(["active", "transient"]),
            AIProviderIncidentModel.credential_scope.in_(old_fps),
            or_(
                AIProviderIncidentModel.origin_instance_id.in_(origin_names),
                AIProviderIncidentModel.companion_instance_id.in_(origin_names),
            ),
        )

        active_old = (await sess.execute(stmt)).scalars().all()
        for inc in active_old:
            inc.status = "superseded"
            inc.resolved_at = now
            superseded_count += 1
            if inc.credential_scope and inc.credential_scope != new_fp:
                target_tombstones.add(inc.credential_scope)

            # Cancel pending/processing notifications without sending alerts
            await sess.execute(
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

        # Ensure a tombstone record exists for old fingerprints so concurrent in-flight failures
        # are recognized as superseded and do not open a new active incident.
        # Historical superseded records are NEVER deleted (audit preserved).
        for tomb_fp in target_tombstones:
            if not tomb_fp or tomb_fp == "unconfigured" or tomb_fp == new_fp:
                continue
            tomb_stmt = select(AIProviderIncidentModel).where(
                AIProviderIncidentModel.provider == prov_norm,
                AIProviderIncidentModel.credential_scope == tomb_fp,
                AIProviderIncidentModel.status == "superseded",
                AIProviderIncidentModel.origin_instance_id.in_(origin_names),
            ).limit(1)
            tomb = await sess.scalar(tomb_stmt)
            if not tomb:
                tomb = AIProviderIncidentModel(
                    origin_instance_id=source_def.get("pm2_name"),
                    companion_instance_id=companion_def.get("pm2_name") if companion_def else None,
                    provider=prov_norm,
                    credential_scope=tomb_fp,
                    operation="chat",
                    model_scope="*",
                    incident_type="key_rotated",
                    status="superseded",
                    occurrence_count=1,
                    reminder_cycle=0,
                    first_seen_at=now,
                    last_seen_at=now,
                    last_alerted_at=now,
                    resolved_at=now,
                )
                sess.add(tomb)
            else:
                tomb.resolved_at = now

        if prov_norm == "KIE":
            # Reconcile kie_credit_alert_sent with whether new_fp has an active low_credit_warning for this bot
            stmt_active_warn = select(AIProviderIncidentModel).where(
                AIProviderIncidentModel.provider == "KIE",
                AIProviderIncidentModel.credential_scope == new_fp,
                AIProviderIncidentModel.incident_type == "low_credit_warning",
                AIProviderIncidentModel.status == "active",
                or_(
                    AIProviderIncidentModel.origin_instance_id.in_(origin_names),
                    AIProviderIncidentModel.companion_instance_id.in_(origin_names),
                ),
            )
            has_active_warn = (await sess.scalar(stmt_active_warn)) is not None
            from database import AIConfig
            ai_cfg = await sess.get(AIConfig, 1)
            if ai_cfg and ai_cfg.kie_credit_alert_sent != has_active_warn:
                ai_cfg.kie_credit_alert_sent = has_active_warn

        logger.info(
            "[AI Incident] Key rotation for %s: superseded %d incidents (target tombstones=%s, new fp=%s).",
            prov_norm,
            superseded_count,
            target_tombstones,
            new_fp,
        )
        return superseded_count

    if session is not None:
        return await _execute_rotation(session)
    else:
        async with sm() as sess:
            async with sess.begin():
                return await _execute_rotation(sess)


async def record_low_credit_warning(
    provider: str,
    remaining_credits: float,
    threshold: float,
    api_key: str,
    base_url: str | None = None,
    session_maker: Any = None,
) -> bool:
    """
    Record a soft warning for low credit balance via durable outbox.
    Creates an active incident with incident_type='low_credit_warning' and outbox rows for each admin.
    """
    try:
        current_inst = detect_current_instance()
        if not current_inst:
            return False
        companion_def = resolve_companion_telegram_instance(current_inst)
        if not companion_def:
            return False

        sm = session_maker or async_session_maker
        now = datetime.utcnow()
        prov_norm = (provider or "KIE").strip().upper()
        fp = get_provider_account_fingerprint(api_key, base_url)
        origin_id = current_inst["pm2_name"]
        companion_id = companion_def.get("pm2_name")
        auth_instance_id = companion_def["pm2_name"]

        async with sm() as session:
            stmt = select(AIProviderIncidentModel).where(
                AIProviderIncidentModel.origin_instance_id == origin_id,
                AIProviderIncidentModel.provider == prov_norm,
                AIProviderIncidentModel.credential_scope == fp,
                AIProviderIncidentModel.operation == "account",
                AIProviderIncidentModel.model_scope == "*",
                AIProviderIncidentModel.incident_type == "low_credit_warning",
                AIProviderIncidentModel.status == "active",
            )
            existing = await session.scalar(stmt)
            if existing:
                existing.last_seen_at = now
                existing.detail_code = f"balance_{remaining_credits:.2f}_below_{threshold:.2f}"
                await session.commit()
                return True

            inc = AIProviderIncidentModel(
                origin_instance_id=origin_id,
                companion_instance_id=companion_id,
                provider=prov_norm,
                credential_scope=fp,
                operation="account",
                model_scope="*",
                incident_type="low_credit_warning",
                evidence_tier="CONFIRMED_STRUCTURED",
                detail_code=f"balance_{remaining_credits:.2f}_below_{threshold:.2f}",
                status="active",
                occurrence_count=1,
                reminder_cycle=0,
                first_seen_at=now,
                last_seen_at=now,
                last_alerted_at=now,
            )
            try:
                async with session.begin_nested():
                    session.add(inc)
                    await session.flush()
            except IntegrityError:
                existing = await session.scalar(stmt)
                if not existing:
                    return False
                inc = existing

            admin_ids = await resolve_target_admin_ids(companion_def, current_instance=current_inst, session=session)
            msg_text = (
                f"⚠️ <b>Предупреждение {prov_norm}: низкий баланс</b>\n\n"
                f"Баланс {prov_norm} снизился до <code>{remaining_credits:.2f}</code> кредитов "
                f"(порог предупреждения: <code>{threshold:.2f}</code>)."
            )
            for admin_id in admin_ids:
                if not is_valid_telegram_user_id(admin_id):
                    continue
                notif = AIIncidentNotificationModel(
                    incident_id=inc.id,
                    event_type="opened",
                    cycle_index=0,
                    recipient_id=admin_id,
                    authorized_instance_id=auth_instance_id,
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
            return True
    except Exception as exc:
        logger.error("[AI Incident] Failed to record low credit warning: %s", exc, exc_info=True)
        return False


async def resolve_low_credit_warning(
    provider: str,
    remaining_credits: float,
    threshold: float,
    api_key: str,
    base_url: str | None = None,
    session_maker: Any = None,
) -> bool:
    """
    Resolve active low credit warning incident and enqueue resolved notifications for all admins.
    """
    try:
        current_inst = detect_current_instance()
        if not current_inst:
            return False
        companion_def = resolve_companion_telegram_instance(current_inst)
        if not companion_def:
            return False

        sm = session_maker or async_session_maker
        now = datetime.utcnow()
        prov_norm = (provider or "KIE").strip().upper()
        fp = get_provider_account_fingerprint(api_key, base_url)
        auth_instance_id = companion_def["pm2_name"]

        async with sm() as session:
            stmt = select(AIProviderIncidentModel).where(
                or_(
                    AIProviderIncidentModel.origin_instance_id == current_inst["pm2_name"],
                    AIProviderIncidentModel.companion_instance_id == current_inst["pm2_name"],
                ),
                AIProviderIncidentModel.provider == prov_norm,
                AIProviderIncidentModel.credential_scope == fp,
                AIProviderIncidentModel.operation == "account",
                AIProviderIncidentModel.model_scope == "*",
                AIProviderIncidentModel.incident_type == "low_credit_warning",
                AIProviderIncidentModel.status == "active",
            )
            active_inc = await session.scalar(stmt)
            if not active_inc:
                return False

            active_inc.status = "resolved"
            active_inc.resolved_at = now

            admin_ids = await resolve_target_admin_ids(companion_def, current_instance=current_inst, session=session)
            msg_text = (
                f"✅ <b>Баланс {prov_norm} пополнен</b>\n\n"
                f"Текущий остаток: <code>{remaining_credits:.2f}</code> кредитов "
                f"(порог: <code>{threshold:.2f}</code>)."
            )
            for admin_id in admin_ids:
                if not is_valid_telegram_user_id(admin_id):
                    continue
                notif = AIIncidentNotificationModel(
                    incident_id=active_inc.id,
                    event_type="resolved",
                    cycle_index=0,
                    recipient_id=admin_id,
                    authorized_instance_id=auth_instance_id,
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
            return True
    except Exception as exc:
        logger.error("[AI Incident] Failed to resolve low credit warning: %s", exc, exc_info=True)
        return False


_enqueue_locks: dict[int, asyncio.Lock] = {}
_enqueue_lock_guard = asyncio.Lock()


async def _get_incident_enqueue_lock(incident_id: int) -> asyncio.Lock:
    async with _enqueue_lock_guard:
        if len(_enqueue_locks) > 2000:
            _enqueue_locks.clear()
        lock = _enqueue_locks.get(incident_id)
        if lock is None:
            lock = asyncio.Lock()
            _enqueue_locks[incident_id] = lock
        return lock


async def enqueue_ai_notification_event(
    incident_id: int,
    origin_instance_id: str | None,
    event_type: str,
    text: str,
    cooldown_seconds: int = 1800,
    session_maker: Any = None,
    max_attempts: int | None = None,
) -> bool:
    """
    Enqueue a durable AI notification event (terminal_failure, output_budget_exhausted, fallback_used)
    for all valid Telegram admins using canonical ownership, fail-closed isolation, and companion routing.
    - Validates incident existence, current instance identity, caller ownership / companion linkage.
    - Suppresses stale notifications when parent incident is superseded.
    - Dedupes against recent notifications of the same event_type for this incident within cooldown_seconds in DB.
    - Creates separate outbox row for EACH valid Telegram admin.
    """
    try:
        current_inst = detect_current_instance()
        if not current_inst:
            logger.critical("[AI Incident] Fail-closed: cannot determine bot instance identity for %s alert.", event_type)
            return False

        if not origin_instance_id:
            origin_instance_id = current_inst["pm2_name"]
            source_def = current_inst
        else:
            source_def = get_instance_by_pm2_name(origin_instance_id)
            if not source_def:
                logger.critical("[AI Incident] Fail-closed: origin_instance_id %r not in registry.", origin_instance_id)
                return False

            is_same = (
                source_def.get("pm2_name") == current_inst.get("pm2_name")
                or source_def.get("bot_instance_id") == current_inst.get("bot_instance_id")
            )
            is_companion = (
                source_def.get("companion_instance") in (current_inst.get("pm2_name"), current_inst.get("bot_instance_id"))
                or current_inst.get("companion_instance") in (source_def.get("pm2_name"), source_def.get("bot_instance_id"))
            )
            if not (is_same or is_companion):
                logger.critical(
                    "[AI Incident] Fail-closed: origin_instance_id %r not allowed from %r for %s alert.",
                    origin_instance_id,
                    current_inst.get("pm2_name"),
                    event_type,
                )
                return False

        sm = session_maker or async_session_maker
        inc_lock = await _get_incident_enqueue_lock(incident_id)

        async with inc_lock:
            now = datetime.utcnow()
            async with sm() as session:
                async with session.begin():
                    # Lock parent incident row in DB to serialize concurrent callers across processes
                    stmt_lock = (
                        select(AIProviderIncidentModel)
                        .where(AIProviderIncidentModel.id == incident_id)
                        .with_for_update()
                    )
                    parent_inc = await session.scalar(stmt_lock)
                    if not parent_inc:
                        logger.error("[AI Incident] Parent incident %s not found for %s alert.", incident_id, event_type)
                        return False

                    if parent_inc.status == "superseded":
                        logger.info("[AI Incident] Parent incident %s is superseded. Suppressing %s alert.", incident_id, event_type)
                        return False

                    # Verification of incident owner / caller authority (fail-closed isolation)
                    inc_origin = parent_inc.origin_instance_id
                    inc_origin_def = get_instance_by_pm2_name(inc_origin)
                    if not inc_origin_def:
                        logger.critical("[AI Incident] Fail-closed: incident %s origin %r not in registry.", incident_id, inc_origin)
                        return False

                    caller_is_origin = (
                        inc_origin in (current_inst.get("pm2_name"), current_inst.get("bot_instance_id"))
                    )
                    caller_is_companion = (
                        current_inst.get("companion_instance") in (inc_origin, parent_inc.companion_instance_id)
                        or inc_origin_def.get("companion_instance") in (current_inst.get("pm2_name"), current_inst.get("bot_instance_id"))
                    )
                    if not (caller_is_origin or caller_is_companion):
                        logger.critical(
                            "[AI Incident] Fail-closed: caller %r has no authority over incident %s owned by %r.",
                            current_inst.get("pm2_name"),
                            incident_id,
                            inc_origin,
                        )
                        return False

                    if origin_instance_id:
                        orig_is_match = (
                            origin_instance_id in (inc_origin, inc_origin_def.get("bot_instance_id"))
                            or inc_origin in (source_def.get("pm2_name"), source_def.get("bot_instance_id"))
                        )
                        if not orig_is_match:
                            logger.critical(
                                "[AI Incident] Fail-closed: provided origin %r does not match incident %s origin %r.",
                                origin_instance_id,
                                incident_id,
                                inc_origin,
                            )
                            return False

                    # Canonical companion and target admin IDs are strictly determined from parent incident's origin
                    canonical_companion_def = resolve_companion_telegram_instance(inc_origin_def)
                    if not canonical_companion_def:
                        logger.error("[AI Incident] No companion Telegram instance found for incident %s (%s).", incident_id, inc_origin)
                        return False

                    authorized_instance_id = canonical_companion_def["pm2_name"]

                    # Check true elapsed time since last notification of this event_type on this incident
                    recent_stmt = (
                        select(func.max(AIIncidentNotificationModel.created_at))
                        .where(
                            AIIncidentNotificationModel.incident_id == incident_id,
                            AIIncidentNotificationModel.event_type == event_type,
                        )
                    )
                    last_created_at = await session.scalar(recent_stmt)
                    if last_created_at is not None:
                        elapsed = (now - last_created_at).total_seconds()
                        if elapsed < cooldown_seconds:
                            logger.info(
                                "[AI Incident] %s alert for incident %s suppressed by DB cooldown (elapsed %.2fs < %ss).",
                                event_type,
                                incident_id,
                                elapsed,
                                cooldown_seconds,
                            )
                            return False

                    admin_ids = await resolve_target_admin_ids(canonical_companion_def, current_instance=inc_origin_def, session=session)
                    if not admin_ids:
                        return False

                    max_cycle_stmt = select(func.max(AIIncidentNotificationModel.cycle_index)).where(
                        AIIncidentNotificationModel.incident_id == incident_id,
                        AIIncidentNotificationModel.event_type == event_type,
                    )
                    max_cycle = await session.scalar(max_cycle_stmt)
                    next_cycle = 0 if max_cycle is None else max_cycle + 1

                    enqueued_count = 0
                    eff_max_attempts = (
                        max_attempts
                        if (isinstance(max_attempts, int) and max_attempts > 0)
                        else MAX_OUTBOX_ATTEMPTS
                    )
                    for adm_id in admin_ids:
                        if not is_valid_telegram_user_id(adm_id):
                            continue
                        notif = AIIncidentNotificationModel(
                            incident_id=incident_id,
                            event_type=event_type,
                            cycle_index=next_cycle,
                            recipient_id=adm_id,
                            authorized_instance_id=authorized_instance_id,
                            message_text=text,
                            status="pending",
                            attempts=0,
                            max_attempts=eff_max_attempts,
                            next_retry_at=now,
                            created_at=now,
                            updated_at=now,
                        )
                        try:
                            async with session.begin_nested():
                                session.add(notif)
                                await session.flush()
                            enqueued_count += 1
                        except IntegrityError:
                            pass

                    parent_inc.last_alerted_at = now
                    return enqueued_count > 0
    except Exception as exc:
        logger.error("[AI Incident] Failed to enqueue %s alert for incident %s: %s", event_type, incident_id, exc, exc_info=True)
        return False


async def enqueue_terminal_ai_failure_notification(
    incident_id: int,
    origin_instance_id: str | None,
    text: str,
    cooldown_seconds: int = 1800,
    session_maker: Any = None,
    max_attempts: int | None = None,
) -> bool:
    """
    Enqueue a durable terminal AI failure notification for all Telegram admins.
    - Dedupes against recent terminal_failure notifications for this incident within cooldown_seconds in DB.
    - Creates separate outbox row for EACH valid Telegram admin.
    - Ensures fail-closed instance identity and companion routing.
    """
    return await enqueue_ai_notification_event(
        incident_id=incident_id,
        origin_instance_id=origin_instance_id,
        event_type="terminal_failure",
        text=text,
        cooldown_seconds=cooldown_seconds,
        session_maker=session_maker,
        max_attempts=max_attempts,
    )


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
    try:
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
                or_(
                    AIProviderIncidentModel.origin_instance_id == current_bot_instance_id,
                    AIProviderIncidentModel.companion_instance_id == current_bot_instance_id,
                ),
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
                old_cycle = inc.reminder_cycle
                new_cycle = old_cycle + 1

                # Explicit CAS atomic update: only bump if cycle has not changed and last_alerted_at <= threshold
                stmt_cas = (
                    update(AIProviderIncidentModel)
                    .where(
                        AIProviderIncidentModel.id == inc.id,
                        AIProviderIncidentModel.status == "active",
                        AIProviderIncidentModel.reminder_cycle == old_cycle,
                        AIProviderIncidentModel.last_alerted_at <= threshold,
                    )
                    .values(
                        reminder_cycle=new_cycle,
                        last_alerted_at=now,
                    )
                )
                cas_res = await session.execute(stmt_cas)
                if cas_res.rowcount != 1:
                    # Concurrent worker already claimed and advanced this cycle
                    continue

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
    except Exception as exc:
        logger.error("[AI Incident] Unexpected error in check_incident_repeat_reminders: %s", exc, exc_info=True)
        return 0


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
    cand_token = getattr(actual_bot, "token", None) if actual_bot is not None else None
    if cand_token is None:
        cand_token = resolve_target_bot_token(current_inst, is_companion=False)
        if not cand_token:
            logger.critical(
                "[AI Incident Worker] Valid target bot token missing for %s. Fail-closed.",
                current_bot_instance_id,
            )
            return 0
        actual_bot = Bot(token=cand_token)

    from notification_routing import verify_bot_identity_live
    try:
        await verify_bot_identity_live(
            actual_bot or cand_token,
            expected_username=current_inst.get("username"),
            expected_id=expected_pid,
            use_cache=True,
        )
    except Exception as id_exc:
        logger.critical(
            "[AI Incident Worker] Live bot identity verification failed for %s (expected_id=%s): %s. Refusing delivery.",
            current_bot_instance_id,
            expected_pid,
            id_exc,
        )
        return 0

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
                            or_(
                                AIIncidentNotificationModel.next_retry_at <= now,
                                AIIncidentNotificationModel.next_retry_at.is_(None),
                            ),
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
            # Canonical SQL expression for effective max attempts (falls back to MAX_OUTBOX_ATTEMPTS if NULL or <= 0)
            effective_max_attempts_sql = case(
                (
                    and_(
                        AIIncidentNotificationModel.max_attempts.isnot(None),
                        AIIncidentNotificationModel.max_attempts > 0,
                    ),
                    AIIncidentNotificationModel.max_attempts,
                ),
                else_=MAX_OUTBOX_ATTEMPTS,
            )

            # 1. Finalize exhausted rows: whether in 'processing' (with expired lease) or 'pending' (due for retry),
            # if attempts >= effective_max_attempts, immediately transition to 'failed_terminal' WITHOUT calling Telegram.
            finalize_exhausted_stmt = (
                update(AIIncidentNotificationModel)
                .where(
                    AIIncidentNotificationModel.id == row_id,
                    AIIncidentNotificationModel.authorized_instance_id == current_bot_instance_id,
                    AIIncidentNotificationModel.attempts >= effective_max_attempts_sql,
                    or_(
                        and_(
                            AIIncidentNotificationModel.status == "processing",
                            AIIncidentNotificationModel.lease_until < ts_now,
                        ),
                        and_(
                            AIIncidentNotificationModel.status == "pending",
                            or_(
                                AIIncidentNotificationModel.next_retry_at <= ts_now,
                                AIIncidentNotificationModel.next_retry_at.is_(None),
                            ),
                        ),
                    ),
                )
                .values(
                    status="failed_terminal",
                    next_retry_at=None,
                    lease_until=None,
                    last_error=case(
                        (
                            AIIncidentNotificationModel.status == "processing",
                            "lease_expired_max_attempts_exhausted",
                        ),
                        else_="max_attempts_exhausted",
                    ),
                    updated_at=ts_now,
                )
            )
            fin_res = await session.execute(finalize_exhausted_stmt)
            if fin_res.rowcount > 0:
                await session.commit()
                return False

            # 2. Atomic claim statement:
            # - For 'pending' rows: allowed ONLY if attempts < effective_max_attempts
            # - For 'processing' rows with expired lease: allowed ONLY if attempts < effective_max_attempts
            claim_stmt = (
                update(AIIncidentNotificationModel)
                .where(
                    AIIncidentNotificationModel.id == row_id,
                    AIIncidentNotificationModel.authorized_instance_id == current_bot_instance_id,
                    AIIncidentNotificationModel.attempts < effective_max_attempts_sql,
                    or_(
                        and_(
                            AIIncidentNotificationModel.status == "pending",
                            or_(
                                AIIncidentNotificationModel.next_retry_at <= ts_now,
                                AIIncidentNotificationModel.next_retry_at.is_(None),
                            ),
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
                return False  # Lost race, lease taken, or already exhausted

            # Fetch claimed row
            row = await session.get(AIIncidentNotificationModel, row_id)
            if not row or row.claim_token != token:
                return False

            # Check parent incident freshness: if resolved/superseded before sending, skip alerts (opened, repeat_reminder, terminal_failure, fallback_used).
            # For resolved event notifications, ensure parent incident is indeed resolved (and not superseded).
            parent_inc = await session.get(AIProviderIncidentModel, row.incident_id)
            if not parent_inc:
                row.status = "skipped_stale"
                row.last_error = "parent_incident_deleted"
                row.updated_at = ts_now
                await session.commit()
                return False

            if row.event_type == "resolved":
                if parent_inc.status != "resolved":
                    row.status = "skipped_stale"
                    row.last_error = f"incident_not_resolved_{parent_inc.status}"
                    row.updated_at = ts_now
                    await session.commit()
                    return False
            else:
                if parent_inc.status in ("resolved", "superseded"):
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
            raw_max_attempts = getattr(row, "max_attempts", None)
            effective_max_attempts = (
                raw_max_attempts
                if (isinstance(raw_max_attempts, int) and raw_max_attempts > 0)
                else MAX_OUTBOX_ATTEMPTS
            )

        # Dispatch message via actual_bot
        send_success = False
        is_terminal = False
        retry_delay = get_ai_incident_outbox_backoff(attempts)
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
            retry_after_val = getattr(exc, "retry_after", 30)
            try:
                retry_after_sec = float(retry_after_val)
                if not math.isfinite(retry_after_sec) or retry_after_sec <= 0:
                    retry_after_sec = 30.0
            except (TypeError, ValueError):
                retry_after_sec = 30.0
            retry_delay = timedelta(seconds=max(1.0, retry_after_sec))
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
                res = await session.execute(stmt)
                await session.commit()
                if (res.rowcount or 0) > 0:
                    return True
                logger.warning(
                    "[AI Outbox] Post-send delivery CAS missed row %s with token %s (lease expired or stolen). Not marking as delivered.",
                    row_id,
                    token,
                )
                return False
            else:
                new_status = "failed_terminal" if is_terminal or attempts >= effective_max_attempts else "pending"
                next_retry = None if new_status == "failed_terminal" else (ts_post + retry_delay)
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
                        next_retry_at=next_retry,
                        updated_at=ts_post,
                    )
                )
                res = await session.execute(stmt)
                await session.commit()
                if (res.rowcount or 0) == 0:
                    logger.warning(
                        "[AI Outbox] Post-send failure CAS missed row %s with token %s (claim superseded).",
                        row_id,
                        token,
                    )
                return False

    for cand_id in candidates:
        if await _do_send(cand_id):
            delivered_count += 1

    return delivered_count
