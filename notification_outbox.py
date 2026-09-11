from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Awaitable, Callable

from aiogram import Bot
from sqlalchemy import and_, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from database import PaymentNotificationOutbox, SubscriptionConfig, async_session_maker, get_all_admin_ids
from notification_renderer import render_outbox_message
from notification_transport import send_notification_transport

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
    deliver_func: Callable[..., Awaitable[bool]] = send_notification_transport,
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
            text, parse_mode, keyboard_type = render_outbox_message(row.event_type, payload, row.recipient_id)

            is_timeout = False
            try:
                delivered = await asyncio.wait_for(
                    deliver_func(bot, row.recipient_id, text, keyboard_type=keyboard_type, parse_mode=parse_mode),
                    timeout=OUTBOX_DELIVERY_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                delivered = False
                is_timeout = True
                log.warning(
                    "OUTBOX_DELIVERY_TIMEOUT | key=%s | recipient=%s | timeout=%ss",
                    row.unique_key,
                    row.recipient_id,
                    OUTBOX_DELIVERY_TIMEOUT_SECONDS,
                )
            except Exception as e:
                delivered = False
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
                new_status = "failed_terminal" if row.attempts >= row.max_attempts else "pending"
                last_err = "transport_delivery_timeout" if is_timeout else "transport_delivery_failed"
                await session.execute(
                    update(PaymentNotificationOutbox)
                    .where(PaymentNotificationOutbox.id == row.id, PaymentNotificationOutbox.claim_token == token)
                    .values(
                        status=new_status,
                        claim_token=None,
                        lease_until=None,
                        next_retry_at=now_done + backoff,
                        last_error=last_err,
                        updated_at=now_done,
                    )
                )
                await session.commit()
                if new_status == "failed_terminal":
                    log.error(
                        "OUTBOX_FAILED_TERMINAL | key=%s | recipient=%s | event=%s | attempts=%s",
                        row.unique_key,
                        row.recipient_id,
                        row.event_type,
                        row.attempts,
                    )
                    await _send_admin_terminal_failure_alert(bot, row)
                else:
                    log.warning(
                        "OUTBOX_DELIVERY_RETRY_SCHEDULED | key=%s | attempts=%s | next_retry=%s",
                        row.unique_key,
                        row.attempts,
                        now_done + backoff,
                    )
                return False
    except Exception as e:
        log.error("Error during dispatch_outbox_by_key for %s: %s", unique_key, e, exc_info=e)
        return False


async def process_payment_notification_outbox(
    bot: Bot,
    deliver_func: Callable[..., Awaitable[bool]] = send_notification_transport,
    batch_size: int = 50,
    session_maker: Any | None = None,
) -> int:
    """
    Periodic background outbox worker.
    Scans for pending or expired-lease items, claims each using atomic CAS lease,
    delivers via deliver_func, and updates state.
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

    except Exception as e:
        log.error("Error in process_payment_notification_outbox: %s", e, exc_info=e)

    return delivered_count


async def _send_admin_terminal_failure_alert(bot: Bot, row: PaymentNotificationOutbox) -> None:
    """Sends a bounded admin notification on terminal notification delivery failure."""
    try:
        async with async_session_maker() as session:
            config = await session.get(SubscriptionConfig, 1)
            if not config or not config.notifications_enabled:
                return
            admin_ids = await get_all_admin_ids()

        alert_text = (
            f"🚨 СБОЙ ДОСТАВКИ УВЕДОМЛЕНИЯ (Outbox Terminal Failure)\n\n"
            f"Событие: {row.event_type}\n"
            f"Получатель: [id={row.recipient_id}]\n"
            f"Провайдер: {row.provider}\n"
            f"PayId: {row.payment_id or 'none'}\n"
            f"Ключ: {row.unique_key}\n"
            f"Число попыток: {row.attempts}/{row.max_attempts}\n"
            f"Статус: failed_terminal (подписка и платеж не затронуты)"
        )
        for admin_id in admin_ids:
            try:
                await bot.send_message(admin_id, alert_text)
            except Exception:
                pass
    except Exception:
        pass
