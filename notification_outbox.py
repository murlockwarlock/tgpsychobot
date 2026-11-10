from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Awaitable, Callable

from aiogram import Bot
from sqlalchemy import and_, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from config import OWNER_IDS
from database import (
    BotGeneralConfig,
    PaymentNotificationOutbox,
    SubscriptionConfig,
    SubscriptionPlan,
    User,
    UserSubscription,
    YookassaRecurringAttempt,
    async_session_maker,
    get_all_admin_ids,
)
from notification_renderer import render_outbox_message
from notification_transport import (
    NotificationDeliveryResult,
    build_robokassa_pending_keyboard,
    build_subscribe_keyboard,
    is_bot_blocked_reason,
    send_notification_transport_result,
)
from translation_service import (
    normalize_locale,
    refresh_translation_cache,
    resolve_effective_locale,
    runtime_enabled_languages,
    runtime_language_selection_enabled,
    translation_cache,
    translate,
)

log = logging.getLogger(__name__)


class NotificationPolicy(str, Enum):
    NONE = "none"
    TERMINAL_ONLY = "terminal_only"
    ALL_USER_EVENTS = "all_user_events"


BACKOFF_DELAYS = [
    timedelta(minutes=2),
    timedelta(minutes=15),
    timedelta(hours=1),
    timedelta(hours=4),
    timedelta(hours=16),
    timedelta(hours=28),
]

OUTBOX_DELIVERY_TIMEOUT_SECONDS = 150
OUTBOX_LEASE_SECONDS = 300
TERMINAL_ALERT_BACKOFF_SECONDS = [30, 60, 120, 300, 600]


async def _prepare_outbox_payload(
    session,
    recipient_id: int,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], str]:
    user = None if recipient_id >= 100_000_000_000 else await session.get(User, recipient_id)
    config = await session.get(BotGeneralConfig, 1)
    admin_recipient = bool(recipient_id in OWNER_IDS or getattr(user, "is_admin", False))
    locale = resolve_effective_locale(
        None if recipient_id >= 100_000_000_000 else getattr(user, "telegram_language_code", None),
        getattr(config, "telegram_default_language", "ru") if config else "ru",
        runtime_language_selection_enabled(config),
        runtime_enabled_languages(config),
        platform="max" if recipient_id >= 100_000_000_000 else "telegram",
        admin=admin_recipient,
    )
    localized = dict(payload)
    payload_locale = normalize_locale(localized.get("keyboard_locale"))
    if payload_locale:
        locale = payload_locale
    plan_fields = (
        ("plan_id", "plan_name"),
        ("paid_plan_id", "paid_plan_name"),
        ("current_plan_id", "current_plan_name"),
    )
    for id_field, name_field in plan_fields:
        plan_id = localized.get(id_field)
        plan = await session.get(SubscriptionPlan, plan_id) if plan_id is not None else None
        if plan is None and localized.get(name_field):
            plan = await session.scalar(
                select(SubscriptionPlan).where(SubscriptionPlan.name == localized[name_field]).limit(1)
            )
        if plan is not None:
            localized[name_field] = translate(
                f"plan.{plan.id}.name",
                locale,
                fallback=plan.name,
                source=plan.name,
            )
    overrides = {}
    if locale != "ru":
        for key in translation_cache.snapshot.sources:
            if key.startswith("notification."):
                translated = translate(key, locale)
                source = translation_cache.snapshot.sources.get(key)
                if translated and source and translated != source:
                    overrides[key] = translated
    return localized, overrides, locale


def _normalize_datetime_for_compare(value: datetime) -> datetime:
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _reminder_bucket_for_subscription(end_date: datetime, now: datetime) -> str | None:
    time_left = _normalize_datetime_for_compare(end_date) - _normalize_datetime_for_compare(now)
    if timedelta(days=2) <= time_left < timedelta(days=2, hours=1):
        return "2d"
    if timedelta(days=1) <= time_left < timedelta(days=1, hours=1):
        return "1d"
    if timedelta(hours=2) <= time_left < timedelta(hours=3):
        return "2h"
    return None


async def _subscription_reminder_is_current(
    session,
    recipient_id: int,
    payload: dict[str, Any],
    now: datetime,
) -> tuple[bool, str | None]:
    snapshot = payload.get("subscription_snapshot")
    if not isinstance(snapshot, dict):
        return True, None

    snapshot_id = snapshot.get("id")
    snapshot_end_date = snapshot.get("end_date")
    if snapshot_id is None or not isinstance(snapshot_end_date, str):
        return False, "invalid_snapshot"
    try:
        subscription_id = int(snapshot_id)
        expected_end_date = _normalize_datetime_for_compare(datetime.fromisoformat(snapshot_end_date))
    except (TypeError, ValueError):
        return False, "invalid_snapshot"

    subscription = await session.get(UserSubscription, subscription_id)
    if subscription is None:
        return False, "subscription_missing"
    if int(subscription.user_id) != int(recipient_id):
        return False, "recipient_changed"
    if _normalize_datetime_for_compare(subscription.end_date) != expected_end_date:
        return False, "end_date_changed"

    if "plan_id" in snapshot and subscription.plan_id != snapshot.get("plan_id"):
        return False, "plan_changed"
    if "payment_provider" in snapshot and subscription.payment_provider != snapshot.get("payment_provider"):
        return False, "provider_changed"
    if "pending_robokassa_invoice_id" in snapshot:
        expected_invoice = snapshot.get("pending_robokassa_invoice_id")
        actual_invoice = subscription.pending_robokassa_invoice_id
        if str(actual_invoice) != str(expected_invoice):
            return False, "pending_invoice_changed"

    bucket = snapshot.get("reminder_bucket")
    if bucket in {"2d", "1d", "2h"}:
        if _reminder_bucket_for_subscription(subscription.end_date, now) != bucket:
            return False, "reminder_window_changed"
    elif bucket in {"expired", "trial_finished"}:
        current_end_date = _normalize_datetime_for_compare(subscription.end_date)
        current_now = _normalize_datetime_for_compare(now)
        if current_end_date > current_now:
            return False, "subscription_renewed"

    return True, None


async def _payment_attempt_notification_is_current(
    session,
    recipient_id: int,
    payload: dict[str, Any],
) -> tuple[bool, str | None]:
    snapshot = payload.get("payment_attempt_snapshot")
    if not isinstance(snapshot, dict):
        return True, None

    snapshot_id = snapshot.get("id")
    if snapshot_id is None:
        return False, "invalid_snapshot"
    try:
        attempt_id = int(snapshot_id)
    except (TypeError, ValueError):
        return False, "invalid_snapshot"

    attempt = await session.get(YookassaRecurringAttempt, attempt_id)
    if attempt is None:
        return False, "attempt_missing"
    try:
        recipient_key = int(recipient_id)
        attempt_user_id = int(attempt.user_id)
    except (TypeError, ValueError):
        return False, "invalid_snapshot"
    if attempt_user_id != recipient_key:
        return False, "recipient_changed"

    expected_subscription_id = snapshot.get("subscription_id")
    if expected_subscription_id is not None:
        try:
            if int(attempt.subscription_id) != int(expected_subscription_id):
                return False, "subscription_changed"
        except (TypeError, ValueError):
            return False, "invalid_snapshot"
    expected_user_id = snapshot.get("user_id")
    if expected_user_id is not None:
        try:
            if attempt_user_id != int(expected_user_id):
                return False, "user_changed"
        except (TypeError, ValueError):
            return False, "invalid_snapshot"
    if "idempotency_key" in snapshot and attempt.idempotency_key != snapshot.get("idempotency_key"):
        return False, "idempotency_key_changed"
    if "status" in snapshot and attempt.status != snapshot.get("status"):
        return False, "status_changed"
    if "payment_id" in snapshot and attempt.payment_id != snapshot.get("payment_id"):
        return False, "payment_id_changed"

    return True, None


def _mark_subscription_reminder_cancelled(reason: str, now: datetime) -> dict[str, Any]:
    return {
        "status": "cancelled",
        "claim_token": None,
        "lease_until": None,
        "last_error": f"subscription_reminder_stale:{reason}",
        "updated_at": now,
    }


def get_outbox_backoff(attempt_num: int) -> timedelta:
    if 1 <= attempt_num <= len(BACKOFF_DELAYS):
        return BACKOFF_DELAYS[attempt_num - 1]
    return timedelta(hours=28)


def build_canonical_outbox_key(
    provider: str,
    entity_type: str,
    entity_id: str | int,
    recipient_id: int,
    event_kind: str,
) -> str:
    return f"{provider.lower()}:{entity_type.lower()}:{entity_id}:{recipient_id}:{event_kind.lower()}"


def get_canonical_key_for_yookassa_cancellation(
    payment_id: str,
    user_id: int,
    action_taken: str,
    attempt_count: int,
) -> str:
    if action_taken == "deactivate":
        event_kind = "deactivate"
    elif action_taken == "unknown_cancellation":
        event_kind = "unknown_cancellation"
    elif action_taken == "provider_error":
        event_kind = "provider_error"
    elif action_taken == "limit_exceeded":
        event_kind = "final_decline" if attempt_count >= 3 else f"retry_limit_{attempt_count}"
    elif action_taken == "declined":
        event_kind = "final_decline" if attempt_count >= 3 else f"retry_failed_{attempt_count}"
    else:
        event_kind = action_taken
    return build_canonical_outbox_key("yookassa", "payment", payment_id, user_id, event_kind)


def get_canonical_key_for_yookassa_success(
    payment_id: str,
    user_id: int,
    action: str,
    is_recurring: bool = True,
    reconciliation_details: dict[str, Any] | None = None,
) -> str:
    if action == "success":
        event_kind = "renewal_success" if is_recurring else "purchase_success"
    elif action == "manual_reconciliation_required":
        reason = (reconciliation_details or {}).get("reason", "unknown")
        event_kind = f"manual_review_{reason}"
    else:
        event_kind = action
    return build_canonical_outbox_key("yookassa", "payment", payment_id, user_id, event_kind)


async def enqueue_outbox_event(
    session: AsyncSession,
    unique_key: str,
    provider: str,
    recipient_id: int,
    event_type: str,
    payload: dict[str, Any],
    payment_id: str | None = None,
    attempt_id: int | None = None,
) -> PaymentNotificationOutbox | None:
    """
    Enqueues a durable notification outbox row inside an existing business transaction
    using a nested savepoint. If unique_key already exists, the savepoint rolls back
    harmlessly without invalidating the parent transaction.
    """
    row = PaymentNotificationOutbox(
        unique_key=unique_key,
        provider=provider,
        payment_id=str(payment_id) if payment_id else None,
        attempt_id=attempt_id,
        recipient_id=recipient_id,
        event_type=event_type,
        event_payload_json=json.dumps(payload, ensure_ascii=False),
        status="pending",
        attempts=0,
        max_attempts=7,
        next_retry_at=datetime.utcnow(),
    )
    try:
        async with session.begin_nested():
            session.add(row)
            await session.flush()
        return row
    except IntegrityError:
        return None


async def dispatch_outbox_by_key(
    bot: Bot,
    unique_key: str,
    deliver_func: Callable[..., Awaitable[bool | NotificationDeliveryResult]] = send_notification_transport_result,
    session_maker: Any | None = None,
) -> bool:
    """
    Post-commit immediate dispatch helper.
    1. Claims the target row by unique_key via CAS lease (status='processing', lease_until=now+60s).
    2. If claim fails (worker already owns or delivered it), exits cleanly.
    3. If claim succeeds, renders message from immutable JSON snapshot and calls deliver_func.
    4. On success: marks status='delivered', delivered_at=now.
    5. On failure: marks status='pending' (or 'failed_terminal'), sets next_retry_at with backoff.
    Never throws exceptions to caller; never alters caller payment state.
    """
    sm = session_maker or async_session_maker
    try:
        try:
            await refresh_translation_cache(sm)
        except Exception:
            log.warning("Could not refresh translation cache before outbox dispatch", exc_info=True)
        async with sm() as session:
            token = str(uuid.uuid4())
            lease_duration = timedelta(seconds=OUTBOX_LEASE_SECONDS)
            now = datetime.utcnow()

            stmt = (
                update(PaymentNotificationOutbox)
                .where(
                    PaymentNotificationOutbox.unique_key == unique_key,
                    or_(
                        and_(
                            PaymentNotificationOutbox.status == "pending",
                            PaymentNotificationOutbox.next_retry_at <= now,
                        ),
                        and_(
                            PaymentNotificationOutbox.status == "processing",
                            PaymentNotificationOutbox.lease_until < now,
                        ),
                    ),
                )
                .values(
                    status="processing",
                    claim_token=token,
                    lease_until=now + lease_duration,
                    attempts=PaymentNotificationOutbox.attempts + 1,
                    updated_at=now,
                )
            )
            res = await session.execute(stmt)
            await session.commit()
            if getattr(res, "rowcount", 0) != 1:
                return False

            row = await session.scalar(
                select(PaymentNotificationOutbox).where(
                    PaymentNotificationOutbox.unique_key == unique_key,
                    PaymentNotificationOutbox.claim_token == token,
                )
            )
            if not row:
                return False

            payload = json.loads(row.event_payload_json)
            if row.event_type == "subscription_reminder":
                is_current, stale_reason = await _subscription_reminder_is_current(
                    session,
                    row.recipient_id,
                    payload,
                    datetime.utcnow(),
                )
                if is_current:
                    is_current, stale_reason = await _payment_attempt_notification_is_current(
                        session,
                        row.recipient_id,
                        payload,
                    )
                if not is_current:
                    now_cancelled = datetime.utcnow()
                    await session.execute(
                        update(PaymentNotificationOutbox)
                        .where(
                            PaymentNotificationOutbox.id == row.id,
                            PaymentNotificationOutbox.claim_token == token,
                        )
                        .values(**_mark_subscription_reminder_cancelled(stale_reason or "stale", now_cancelled))
                    )
                    await session.commit()
                    log.info(
                        "OUTBOX_SUBSCRIPTION_REMINDER_CANCELLED | key=%s | recipient=%s | reason=%s",
                        row.unique_key,
                        row.recipient_id,
                        stale_reason or "stale",
                    )
                    return False

            payload, text_overrides, locale = await _prepare_outbox_payload(
                session,
                row.recipient_id,
                payload,
            )
            text, parse_mode, keyboard_type = render_outbox_message(
                row.event_type,
                payload,
                row.recipient_id,
                text_overrides=text_overrides,
            )
            reply_markup = (
                build_subscribe_keyboard(locale)
                if keyboard_type == "subscribe"
                else build_robokassa_pending_keyboard(locale)
                if keyboard_type == "robokassa_pending"
                else None
            )

            is_timeout = False
            try:
                delivery_result = await asyncio.wait_for(
                    deliver_func(
                        bot,
                        row.recipient_id,
                        text,
                        keyboard_type=keyboard_type,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                    ),
                    timeout=OUTBOX_DELIVERY_TIMEOUT_SECONDS,
                )
                if isinstance(delivery_result, NotificationDeliveryResult):
                    delivered = delivery_result.delivered
                    retryable = delivery_result.retryable
                    failure_reason = delivery_result.reason
                else:
                    delivered = bool(delivery_result)
                    retryable = True
                    failure_reason = None
            except asyncio.TimeoutError:
                delivered = False
                is_timeout = True
                retryable = True
                failure_reason = "transport_delivery_timeout"
                log.warning(
                    "OUTBOX_DELIVERY_TIMEOUT | key=%s | recipient=%s | timeout=%ss",
                    row.unique_key,
                    row.recipient_id,
                    OUTBOX_DELIVERY_TIMEOUT_SECONDS,
                )
            except Exception as e:
                delivered = False
                retryable = True
                failure_reason = f"{type(e).__name__}:{e}"
                log.warning(
                    "OUTBOX_DELIVERY_EXCEPTION | key=%s | recipient=%s | err=%s",
                    row.unique_key,
                    row.recipient_id,
                    e,
                )
            now_done = datetime.utcnow()

            if delivered:
                await session.execute(
                    update(PaymentNotificationOutbox)
                    .where(PaymentNotificationOutbox.id == row.id, PaymentNotificationOutbox.claim_token == token)
                    .values(
                        status="delivered",
                        delivered_at=now_done,
                        claim_token=None,
                        lease_until=None,
                        updated_at=now_done,
                    )
                )
                await session.commit()
                log.info("OUTBOX_DELIVERED | key=%s | recipient=%s | event=%s", row.unique_key, row.recipient_id, row.event_type)
                return True
            else:
                backoff = get_outbox_backoff(row.attempts)
                new_status = (
                    "failed_terminal"
                    if not retryable or row.attempts >= row.max_attempts
                    else "pending"
                )
                if not retryable:
                    last_err = failure_reason or "transport_delivery_terminal"
                else:
                    last_err = "transport_delivery_timeout" if is_timeout else "transport_delivery_failed"

                try:
                    payload_data = json.loads(row.event_payload_json) if row.event_payload_json else {}
                except Exception:
                    payload_data = {}

                is_blocked = is_bot_blocked_reason(failure_reason) or is_bot_blocked_reason(last_err)
                already_alerted = (
                    bool(payload_data.get("terminal_alert_sent"))
                    or payload_data.get("terminal_alert_status") == "sent"
                )
                should_alert = (
                    new_status == "failed_terminal"
                    and row.attempts >= row.max_attempts
                    and not is_blocked
                    and not already_alerted
                )
                if should_alert:
                    payload_data["terminal_alert_status"] = "pending"
                    payload_data["terminal_alert_sent"] = False
                elif is_blocked:
                    payload_data["terminal_alert_status"] = "suppressed_blocked"
                    payload_data["terminal_alert_sent"] = False

                await session.execute(
                    update(PaymentNotificationOutbox)
                    .where(PaymentNotificationOutbox.id == row.id, PaymentNotificationOutbox.claim_token == token)
                    .values(
                        status=new_status,
                        claim_token=None,
                        lease_until=None,
                        next_retry_at=now_done if should_alert else (now_done + backoff),
                        last_error=last_err,
                        event_payload_json=json.dumps(payload_data, ensure_ascii=False),
                        updated_at=now_done,
                    )
                )
                await session.commit()
                if new_status == "failed_terminal":
                    if is_blocked:
                        log.info(
                            "OUTBOX_BLOCKED_TERMINAL | key=%s | recipient=%s | event=%s | attempts=%s | reason=%s",
                            row.unique_key,
                            row.recipient_id,
                            row.event_type,
                            row.attempts,
                            last_err,
                        )
                    else:
                        log.error(
                            "OUTBOX_FAILED_TERMINAL | key=%s | recipient=%s | event=%s | attempts=%s | reason=%s",
                            row.unique_key,
                            row.recipient_id,
                            row.event_type,
                            row.attempts,
                            last_err,
                        )
                    if should_alert:
                        row.last_error = last_err
                        row.event_payload_json = json.dumps(payload_data, ensure_ascii=False)
                        await _send_admin_terminal_failure_alert(bot, row, session_maker=sm)
                else:
                    log.warning(
                        "OUTBOX_DELIVERY_RETRY_SCHEDULED | key=%s | attempts=%s | next_retry=%s | reason=%s",
                        row.unique_key,
                        row.attempts,
                        now_done + backoff,
                        failure_reason or last_err,
                    )
                return False
    except Exception as e:
        log.error("Error during dispatch_outbox_by_key for %s: %s", unique_key, e, exc_info=e)
        return False


async def process_pending_terminal_failure_alerts(
    bot: Bot | None,
    batch_size: int = 50,
    session_maker: Any | None = None,
) -> int:
    """
    Scans for outbox records in failed_terminal that have a pending terminal failure alert,
    and safely dispatches them via Autobus with atomic CAS claim.
    Never resends messages to recipient users.
    """
    sm = session_maker or async_session_maker
    now = datetime.utcnow()
    delivered_alerts = 0

    try:
        async with sm() as session:
            candidates = (
                await session.execute(
                    select(PaymentNotificationOutbox)
                    .where(
                        PaymentNotificationOutbox.status == "failed_terminal",
                        PaymentNotificationOutbox.attempts >= PaymentNotificationOutbox.max_attempts,
                        PaymentNotificationOutbox.next_retry_at <= now,
                        or_(
                            PaymentNotificationOutbox.claim_token.is_(None),
                            PaymentNotificationOutbox.lease_until < now,
                        ),
                        or_(
                            PaymentNotificationOutbox.event_payload_json.like('%"terminal_alert_status": "pending"%'),
                            PaymentNotificationOutbox.event_payload_json.like('%"terminal_alert_status":"pending"%'),
                        ),
                    )
                    .order_by(PaymentNotificationOutbox.next_retry_at.asc())
                    .limit(batch_size)
                )
            ).scalars().all()

        for row in candidates:
            success = await _send_admin_terminal_failure_alert(bot, row, session_maker=sm)
            if success:
                delivered_alerts += 1
    except Exception as e:
        log.error("Error in process_pending_terminal_failure_alerts: %s", e, exc_info=e)

    return delivered_alerts


async def process_payment_notification_outbox(
    bot: Bot,
    deliver_func: Callable[..., Awaitable[bool | NotificationDeliveryResult]] = send_notification_transport_result,
    batch_size: int = 50,
    session_maker: Any | None = None,
) -> int:
    """
    Periodic background outbox worker.
    Scans for pending or expired-lease items, claims each using atomic CAS lease,
    delivers via deliver_func, and updates state.
    Also retries pending terminal failure alerts without re-sending to user.
    Returns number of successfully delivered items.
    """
    delivered_count = 0
    now = datetime.utcnow()
    sm = session_maker or async_session_maker

    try:
        async with sm() as session:
            candidates = (
                await session.execute(
                    select(PaymentNotificationOutbox.unique_key)
                    .where(
                        or_(
                            and_(
                                PaymentNotificationOutbox.status == "pending",
                                PaymentNotificationOutbox.next_retry_at <= now,
                            ),
                            and_(
                                PaymentNotificationOutbox.status == "processing",
                                PaymentNotificationOutbox.lease_until < now,
                            ),
                        )
                    )
                    .order_by(PaymentNotificationOutbox.next_retry_at.asc())
                    .limit(batch_size)
                )
            ).scalars().all()

        for key in candidates:
            success = await dispatch_outbox_by_key(bot, key, deliver_func=deliver_func, session_maker=sm)
            if success:
                delivered_count += 1

        # Safely retry any pending terminal failure alerts for failed_terminal records
        await process_pending_terminal_failure_alerts(bot, batch_size=batch_size, session_maker=sm)

    except Exception as e:
        log.error("Error in process_payment_notification_outbox: %s", e, exc_info=e)

    return delivered_count


async def _send_admin_terminal_failure_alert(
    bot: Bot | None,
    row: PaymentNotificationOutbox,
    session_maker: Any | None = None,
) -> bool:
    """Sends a bounded admin notification on terminal notification delivery failure via Autobus.

    Guarantees:
    - Zero alerts if recipient user blocked the bot or reason is bot blocked.
    - Zero alerts if attempts < max_attempts.
    - Atomic CAS claim ensures no duplicate alerts across concurrent workers.
    - Updates event_payload_json with terminal_alert_status='sent' and terminal_alert_sent=True
      ONLY upon confirmed successful delivery via Autobus.
    - If delivery fails or raises, terminal_alert_status remains 'pending' with retry backoff.
    - Returns True if delivered, False otherwise.
    """
    row_id = getattr(row, "id", None)
    unique_key = getattr(row, "unique_key", None)
    claim_token = str(uuid.uuid4())
    claimed = False
    sm = session_maker or async_session_maker

    try:
        last_err = getattr(row, "last_error", None)
        if is_bot_blocked_reason(last_err):
            return False
        attempts = getattr(row, "attempts", 0)
        max_attempts = getattr(row, "max_attempts", 7)
        if attempts < max_attempts:
            return False

        now = datetime.utcnow()
        lease_duration = timedelta(seconds=OUTBOX_LEASE_SECONDS)

        # Check SubscriptionConfig first
        async with sm() as session:
            config = await session.get(SubscriptionConfig, 1)
            if not config or not config.notifications_enabled:
                log.info("Admin notifications disabled via config, suppressing terminal alert for %s", unique_key or row_id)
                return False

        # Attempt atomic CAS claim on the row if row_id or unique_key is available
        if row_id is not None or unique_key is not None:
            async with sm() as session:
                cond = [
                    PaymentNotificationOutbox.status == "failed_terminal",
                    PaymentNotificationOutbox.attempts >= PaymentNotificationOutbox.max_attempts,
                    or_(
                        PaymentNotificationOutbox.claim_token.is_(None),
                        PaymentNotificationOutbox.lease_until < now,
                    ),
                    or_(
                        PaymentNotificationOutbox.event_payload_json.like('%"terminal_alert_status": "pending"%'),
                        PaymentNotificationOutbox.event_payload_json.like('%"terminal_alert_status":"pending"%'),
                    ),
                ]
                if row_id is not None:
                    cond.append(PaymentNotificationOutbox.id == row_id)
                else:
                    cond.append(PaymentNotificationOutbox.unique_key == unique_key)

                stmt = (
                    update(PaymentNotificationOutbox)
                    .where(*cond)
                    .values(
                        claim_token=claim_token,
                        lease_until=now + lease_duration,
                        updated_at=now,
                    )
                )
                res = await session.execute(stmt)
                await session.commit()
                if getattr(res, "rowcount", 0) == 1:
                    claimed = True
                else:
                    log.info("Could not acquire terminal alert claim for %s (rowcount != 1)", unique_key or row_id)
                    return False

        # Load fresh row and payload under claim
        target = row
        payload_data: dict[str, Any] = {}
        if claimed and (row_id is not None or unique_key is not None):
            async with sm() as session:
                fresh_row = await session.scalar(
                    select(PaymentNotificationOutbox).where(
                        (PaymentNotificationOutbox.id == row_id)
                        if row_id is not None
                        else (PaymentNotificationOutbox.unique_key == unique_key)
                    )
                )
                if fresh_row:
                    target = fresh_row
                    if is_bot_blocked_reason(fresh_row.last_error):
                        await session.execute(
                            update(PaymentNotificationOutbox)
                            .where(PaymentNotificationOutbox.id == fresh_row.id, PaymentNotificationOutbox.claim_token == claim_token)
                            .values(claim_token=None, lease_until=None)
                        )
                        await session.commit()
                        return False
                    try:
                        payload_data = json.loads(fresh_row.event_payload_json) if fresh_row.event_payload_json else {}
                    except Exception:
                        payload_data = {}
                    if payload_data.get("terminal_alert_sent") or payload_data.get("terminal_alert_status") == "sent":
                        await session.execute(
                            update(PaymentNotificationOutbox)
                            .where(PaymentNotificationOutbox.id == fresh_row.id, PaymentNotificationOutbox.claim_token == claim_token)
                            .values(claim_token=None, lease_until=None)
                        )
                        await session.commit()
                        return True

        alert_text = (
            f"🚨 СБОЙ ДОСТАВКИ УВЕДОМЛЕНИЯ (Outbox Terminal Failure)\n\n"
            f"Событие: {getattr(target, 'event_type', 'unknown')}\n"
            f"Получатель: [id={getattr(target, 'recipient_id', 'unknown')}]\n"
            f"Провайдер: {getattr(target, 'provider', 'unknown')}\n"
            f"PayId: {getattr(target, 'payment_id', None) or 'none'}\n"
            f"Ключ: {getattr(target, 'unique_key', unique_key or 'unknown')}\n"
            f"Число попыток: {getattr(target, 'attempts', attempts)}/{getattr(target, 'max_attempts', max_attempts)}\n"
            f"Статус: failed_terminal (подписка и платеж не затронуты)"
        )
        from notification_routing import send_admin_notification, EventCategory
        delivered = await send_admin_notification(
            alert_text,
            category=EventCategory.SYSTEM_DIAGNOSTIC,
        )

        now_done = datetime.utcnow()
        if claimed and (row_id is not None or unique_key is not None):
            async with sm() as session:
                cond_update = [PaymentNotificationOutbox.claim_token == claim_token]
                if row_id is not None:
                    cond_update.append(PaymentNotificationOutbox.id == row_id)
                else:
                    cond_update.append(PaymentNotificationOutbox.unique_key == unique_key)

                if delivered:
                    payload_data["terminal_alert_status"] = "sent"
                    payload_data["terminal_alert_sent"] = True
                    payload_data["terminal_alert_delivered_at"] = now_done.isoformat()
                    await session.execute(
                        update(PaymentNotificationOutbox)
                        .where(*cond_update)
                        .values(
                            claim_token=None,
                            lease_until=None,
                            event_payload_json=json.dumps(payload_data, ensure_ascii=False),
                            updated_at=now_done,
                        )
                    )
                    await session.commit()
                    log.info("OUTBOX_TERMINAL_ALERT_DELIVERED | key=%s | recipient=%s", unique_key, getattr(target, "recipient_id", None))
                else:
                    alert_attempts = int(payload_data.get("terminal_alert_attempts", 0)) + 1
                    payload_data["terminal_alert_attempts"] = alert_attempts
                    payload_data["terminal_alert_status"] = "pending"
                    payload_data["terminal_alert_sent"] = False
                    alert_backoff = TERMINAL_ALERT_BACKOFF_SECONDS[min(alert_attempts - 1, len(TERMINAL_ALERT_BACKOFF_SECONDS) - 1)]
                    next_retry = now_done + timedelta(seconds=alert_backoff)
                    await session.execute(
                        update(PaymentNotificationOutbox)
                        .where(*cond_update)
                        .values(
                            claim_token=None,
                            lease_until=None,
                            next_retry_at=next_retry,
                            event_payload_json=json.dumps(payload_data, ensure_ascii=False),
                            updated_at=now_done,
                        )
                    )
                    await session.commit()
                    log.warning(
                        "OUTBOX_TERMINAL_ALERT_RETRY_SCHEDULED | key=%s | alert_attempts=%s | next_retry=%s",
                        unique_key,
                        alert_attempts,
                        next_retry,
                    )

        return bool(delivered)
    except Exception as exc:
        log.warning("Could not dispatch outbox terminal failure alert via Autobus: %s", exc)
        if claimed and (row_id is not None or unique_key is not None):
            try:
                async with sm() as session:
                    cond_rel = [PaymentNotificationOutbox.claim_token == claim_token]
                    if row_id is not None:
                        cond_rel.append(PaymentNotificationOutbox.id == row_id)
                    else:
                        cond_rel.append(PaymentNotificationOutbox.unique_key == unique_key)
                    await session.execute(
                        update(PaymentNotificationOutbox)
                        .where(*cond_rel)
                        .values(claim_token=None, lease_until=None)
                    )
                    await session.commit()
            except Exception:
                pass
        return False
