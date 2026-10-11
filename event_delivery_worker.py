"""
Durable Delivery Worker for Event Reminder Chains.

Provides:
- Polling and claim of pending deliveries with micro-batches and leases.
- Pre-send validation adhering to strict lock ordering (3 -> 4 -> 5 -> 6).
- Point of No Return commit before any HTTP send.
- Cross-platform dispatch for Telegram and MAX outside DB transactions.
- Fine-grained transport error classification (sent, pending/retry, blocked, failed, uncertain).
- Lease recovery sweeper protecting against lost or stuck deliveries.
- Rate limiting / cooldown management.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from datetime import datetime, timedelta
from typing import Any

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from database import (
    EventDelivery,
    EventRegistration,
    EventScenario,
    EventStep,
    User,
)
from event_service import (
    DEFAULT_ADAPTIVE_CONFIRMATION_TEMPLATE,
    DEFAULT_CONFIRMATION_TEMPLATE,
    DEFAULT_LIVE_TEMPLATE,
    DEFAULT_RESCHEDULE_TEMPLATE,
    prepare_and_lock_delivery_for_sending,
    render_event_template,
)
from max_messenger_bot.api import MaxApiClient, MaxApiError
from max_messenger_bot.identity import is_max_user_id, raw_max_user_id
from max_messenger_bot.keyboards import callback_button, inline_keyboard
from max_messenger_bot.models import extract_sent_message_id

class PreTransmissionError(Exception):
    """Raised when an error occurs strictly before HTTP transport transmission begins."""
    pass


log = logging.getLogger(__name__)


# Global platform cooldown trackers (MSK / UTC naive)
_telegram_rate_limited_until: datetime | None = None
_max_rate_limited_until: datetime | None = None


def get_telegram_cooldown() -> datetime | None:
    return _telegram_rate_limited_until


def set_telegram_cooldown(until: datetime | None) -> None:
    global _telegram_rate_limited_until
    _telegram_rate_limited_until = until


def get_max_cooldown() -> datetime | None:
    return _max_rate_limited_until


def set_max_cooldown(until: datetime | None) -> None:
    global _max_rate_limited_until
    _max_rate_limited_until = until


def reset_cooldowns() -> None:
    global _telegram_rate_limited_until, _max_rate_limited_until
    _telegram_rate_limited_until = None
    _max_rate_limited_until = None



async def claim_pending_deliveries(
    session_maker: async_sessionmaker[AsyncSession],
    batch_size: int = 50,
    lease_duration_seconds: int = 60,
    *,
    now: datetime | None = None,
) -> list[tuple[int, str]]:
    """
    Claims a micro-batch of pending deliveries whose due_at has arrived.

    Returns a list of (delivery_id, claim_token) tuples that were successfully claimed.
    """
    effective_now = now or datetime.utcnow()
    lease_until = effective_now + timedelta(seconds=lease_duration_seconds)

    async with session_maker() as session:
        # Find candidate deliveries belonging to active, non-archived scenarios
        candidate_stmt = (
            select(EventDelivery.id, EventDelivery.platform)
            .join(EventScenario, EventDelivery.event_id == EventScenario.id)
            .where(
                EventDelivery.status == "pending",
                EventDelivery.due_at <= effective_now,
                EventDelivery.next_attempt_at <= effective_now,
                or_(EventDelivery.expires_at.is_(None), EventDelivery.expires_at > effective_now),
                EventScenario.is_active.is_(True),
                EventScenario.is_archived.is_(False),
            )
            .order_by(EventDelivery.due_at.asc(), EventDelivery.id.asc())
            .limit(batch_size * 2)
        )
        candidate_rows = (await session.execute(candidate_stmt)).all()

        if not candidate_rows:
            return []

        # Filter by platform cooldowns
        valid_ids: list[int] = []
        tg_cooldown = get_telegram_cooldown()
        max_cooldown = get_max_cooldown()

        for d_id, platform in candidate_rows:
            if platform == "telegram" and tg_cooldown and tg_cooldown > effective_now:
                continue
            if platform == "max" and max_cooldown and max_cooldown > effective_now:
                continue
            valid_ids.append(d_id)
            if len(valid_ids) >= batch_size:
                break

        if not valid_ids:
            return []

        claimed_results: list[tuple[int, str]] = []
        for delivery_id in valid_ids:
            token = secrets.token_hex(16)
            claim_stmt = (
                update(EventDelivery)
                .where(
                    EventDelivery.id == delivery_id,
                    EventDelivery.status == "pending",
                )
                .values(
                    status="claimed",
                    claim_token=token,
                    claimed_at=effective_now,
                    lease_until=lease_until,
                    updated_at=effective_now,
                )
            )
            res = await session.execute(claim_stmt)
            if res.rowcount == 1:
                claimed_results.append((delivery_id, token))

        await session.commit()
        return claimed_results


async def recover_expired_leases(
    session_maker: async_sessionmaker[AsyncSession],
    *,
    now: datetime | None = None,
) -> tuple[int, int]:
    """
    Recovers deliveries whose lease has expired:
    1. 'claimed' deliveries with expired lease where send never started -> return to 'pending'.
    2. 'sending' deliveries with expired lease (>180s) -> transition to 'uncertain'.

    Returns (recovered_claimed, recovered_sending).
    """
    effective_now = now or datetime.utcnow()

    async with session_maker() as session:
        # 1. Recover un-started claimed deliveries back to pending
        claim_recover_stmt = (
            update(EventDelivery)
            .where(
                EventDelivery.status == "claimed",
                EventDelivery.lease_until.is_not(None),
                EventDelivery.lease_until < effective_now,
                EventDelivery.send_started_at.is_(None),
            )
            .values(
                status="pending",
                claim_token=None,
                lease_until=None,
                updated_at=effective_now,
            )
        )
        res_claimed = await session.execute(claim_recover_stmt)
        recovered_claimed = res_claimed.rowcount or 0

        # 2. Mark stuck sending deliveries (>180s past lease) as uncertain
        sending_recover_stmt = (
            update(EventDelivery)
            .where(
                EventDelivery.status == "sending",
                EventDelivery.lease_until.is_not(None),
                EventDelivery.lease_until < effective_now,
            )
            .values(
                status="uncertain",
                finished_at=effective_now,
                last_error="Lease expired in sending state (>180s)",
                claim_token=None,
                lease_until=None,
                updated_at=effective_now,
            )
        )
        res_sending = await session.execute(sending_recover_stmt)
        recovered_sending = res_sending.rowcount or 0

        await session.commit()
        return recovered_claimed, recovered_sending


async def finalize_delivery_outcome(
    session_maker: async_sessionmaker[AsyncSession],
    delivery_id: int,
    claim_token: str,
    status: str,
    *,
    sent_at: datetime | None = None,
    finished_at: datetime | None = None,
    external_message_id: str | None = None,
    last_error: str | None = None,
    next_attempt_at: datetime | None = None,
    now: datetime | None = None,
) -> bool:
    """
    Finalizes the status of a delivery after HTTP transport call completes.
    Only updates if status is still 'sending' and claim_token matches.
    """
    effective_now = now or datetime.utcnow()

    async with session_maker() as session:
        values: dict[str, Any] = {
            "status": status,
            "claim_token": None,
            "lease_until": None,
            "updated_at": effective_now,
        }
        if status == "pending":
            values["send_started_at"] = None
            values["claimed_at"] = None
        if sent_at is not None:

            values["sent_at"] = sent_at
        if finished_at is not None:
            values["finished_at"] = finished_at
        if external_message_id is not None:
            values["external_message_id"] = external_message_id
        if last_error is not None:
            values["last_error"] = last_error
        if next_attempt_at is not None:
            values["next_attempt_at"] = next_attempt_at

        stmt = (
            update(EventDelivery)
            .where(
                EventDelivery.id == delivery_id,
                EventDelivery.status == "sending",
                EventDelivery.claim_token == claim_token,
            )
            .values(**values)
        )
        res = await session.execute(stmt)
        await session.commit()

        if res.rowcount == 0:
            log.warning(
                "Finalizer could not update delivery_id=%s (status may have been moved by sweeper)",
                delivery_id,
            )
            return False
        return True


async def process_single_delivery(
    session_maker: async_sessionmaker[AsyncSession],
    delivery_id: int,
    claim_token: str,
    *,
    bot: Bot | None = None,
    max_client: MaxApiClient | None = None,
    now: datetime | None = None,
) -> str:
    """
    Executes the pre-send protocol and dispatches network send for a single claimed delivery.

    Returns the resulting status string.
    """
    effective_now = now or datetime.utcnow()

    # Step 1: Read delivery and relation context without locks
    async with session_maker() as session:
        delivery = await session.get(EventDelivery, delivery_id)
        if not delivery or delivery.status != "claimed" or delivery.claim_token != claim_token:
            return "stale_claim"

        scenario = await session.get(EventScenario, delivery.event_id)
        if not scenario:
            return "scenario_not_found"

        user = await session.get(User, delivery.user_id)
        step = await session.get(EventStep, delivery.step_id) if delivery.step_id else None

        # Determine message template
        if delivery.message_kind == "scheduled_step":
            template = step.message_text if step else ""
        elif delivery.message_kind == "immediate_confirmation":
            cutoff = scenario.starts_at - timedelta(minutes=15)
            if effective_now >= cutoff:
                template = DEFAULT_ADAPTIVE_CONFIRMATION_TEMPLATE
            else:
                template = DEFAULT_CONFIRMATION_TEMPLATE
        elif delivery.message_kind == "immediate_live":
            template = DEFAULT_LIVE_TEMPLATE
        elif delivery.message_kind == "reschedule_notice":
            template = DEFAULT_RESCHEDULE_TEMPLATE
        else:
            template = ""

        rendered_text = render_event_template(
            template, scenario, user, platform=delivery.platform, now=effective_now
        )
        expected_content_version = scenario.content_version
        expected_step_version = step.step_version if step else None
        event_id = delivery.event_id
        user_id = delivery.user_id
        platform = delivery.platform

    # Step 2: Pre-send lock and CAS validation (Point of No Return inside)
    ok, locked_delivery, skip_reason = await prepare_and_lock_delivery_for_sending(
        session_maker,
        delivery_id=delivery_id,
        claim_token=claim_token,
        expected_content_version=expected_content_version,
        expected_step_version=expected_step_version,
        rendered_text=rendered_text,
        template_text=template,
        now=effective_now,
    )
    if not ok:
        log.info("Pre-send rejected delivery %s: reason=%s", delivery_id, skip_reason)
        return f"pre_send_rejected:{skip_reason}"

    # Step 3: Network send OUTSIDE database locks
    send_started = effective_now
    try:
        if platform == "telegram":
            if bot is None:
                raise RuntimeError("Telegram bot transport is not configured")

            markup = InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="❌ Не смогу прийти", callback_data=f"ev_cancel:{event_id}")]
                ]
            )
            msg = await bot.send_message(
                chat_id=user_id,
                text=rendered_text,
                parse_mode="HTML",
                reply_markup=markup,
            )
            external_id = str(msg.message_id)

        elif platform == "max":
            if max_client is None:
                raise RuntimeError("MAX client transport is not configured")

            attachments = inline_keyboard([
                [callback_button("❌ Не смогу прийти", f"ev_cancel:{event_id}")]
            ])
            sent_resp = await max_client.send_message(
                user_id=raw_max_user_id(user_id),
                text=rendered_text,
                attachments=attachments,
            )
            external_id = extract_sent_message_id(sent_resp) or (
                str(sent_resp.get("message_id") or sent_resp.get("mid") or sent_resp.get("id"))
                if isinstance(sent_resp, dict) and (sent_resp.get("message_id") or sent_resp.get("mid") or sent_resp.get("id"))
                else None
            )
        else:
            raise ValueError(f"Unknown platform '{platform}'")

        finished_at = effective_now
        await finalize_delivery_outcome(
            session_maker,
            delivery_id,
            claim_token,
            status="sent",
            sent_at=finished_at,
            finished_at=finished_at,
            external_message_id=external_id,
            now=effective_now,
        )
        return "sent"

    except TelegramForbiddenError as exc:
        log.warning("Telegram delivery %s blocked by user %s: %s", delivery_id, user_id, exc)
        await finalize_delivery_outcome(
            session_maker,
            delivery_id,
            claim_token,
            status="blocked",
            finished_at=effective_now,
            last_error=str(exc),
            now=effective_now,
        )
        return "blocked"

    except TelegramRetryAfter as exc:
        retry_delay = getattr(exc, "retry_after", 30)
        log.warning("Telegram delivery %s rate limited, retry after %ss: %s", delivery_id, retry_delay, exc)
        set_telegram_cooldown(effective_now + timedelta(seconds=retry_delay))
        await finalize_delivery_outcome(
            session_maker,
            delivery_id,
            claim_token,
            status="pending",
            next_attempt_at=effective_now + timedelta(seconds=retry_delay),
            last_error=str(exc),
            now=effective_now,
        )
        return "rate_limited"

    except TelegramBadRequest as exc:
        log.error("Telegram delivery %s failed with bad request: %s", delivery_id, exc)
        await finalize_delivery_outcome(
            session_maker,
            delivery_id,
            claim_token,
            status="failed",
            finished_at=effective_now,
            last_error=str(exc),
            now=effective_now,
        )
        return "failed"

    except MaxApiError as exc:
        error_code = getattr(exc, "error_code", "") or ""
        status_code = getattr(exc, "status", None)
        err_msg = str(exc)

        if error_code in ("user.blocked", "blocked") or "blocked" in err_msg.lower():
            log.warning("MAX delivery %s blocked by user %s: %s", delivery_id, user_id, exc)
            await finalize_delivery_outcome(
                session_maker,
                delivery_id,
                claim_token,
                status="blocked",
                finished_at=effective_now,
                last_error=err_msg,
                now=effective_now,
            )
            return "blocked"

        if status_code == 429:
            log.warning("MAX delivery %s rate limited: %s", delivery_id, exc)
            set_max_cooldown(effective_now + timedelta(seconds=30))
            await finalize_delivery_outcome(
                session_maker,
                delivery_id,
                claim_token,
                status="pending",
                next_attempt_at=effective_now + timedelta(seconds=30),
                last_error=err_msg,
                now=effective_now,
            )
            return "rate_limited"

        if status_code == 403:
            log.error("MAX delivery %s failed with 403 configuration error: %s", delivery_id, exc)
            await finalize_delivery_outcome(
                session_maker,
                delivery_id,
                claim_token,
                status="failed",
                finished_at=effective_now,
                last_error=err_msg,
                now=effective_now,
            )
            return "failed"

        if "timeout" in err_msg.lower():
            log.error("MAX delivery %s timed out, marking uncertain: %s", delivery_id, exc)
            await finalize_delivery_outcome(
                session_maker,
                delivery_id,
                claim_token,
                status="uncertain",
                finished_at=effective_now,
                last_error=err_msg,
                now=effective_now,
            )
            return "uncertain"

        log.error("MAX delivery %s failed: %s", delivery_id, exc)
        await finalize_delivery_outcome(
            session_maker,
            delivery_id,
            claim_token,
            status="failed",
            finished_at=effective_now,
            last_error=err_msg,
            now=effective_now,
        )
        return "failed"

    except (asyncio.TimeoutError, TimeoutError) as exc:
        log.error("Delivery %s timed out post-request, marking uncertain: %s", delivery_id, exc)
        await finalize_delivery_outcome(
            session_maker,
            delivery_id,
            claim_token,
            status="uncertain",
            finished_at=effective_now,
            last_error=f"Timeout after send initiated: {exc}",
            now=effective_now,
        )
        return "uncertain"

    except PreTransmissionError as exc:
        log.warning("Delivery %s provable pre-transmission error (safe retry): %s", delivery_id, exc)
        await finalize_delivery_outcome(
            session_maker,
            delivery_id,
            claim_token,
            status="pending",
            next_attempt_at=effective_now + timedelta(seconds=60),
            last_error=f"Pre-transmission error: {exc}",
            now=effective_now,
        )
        return "pre_transmission_retry"

    except Exception as exc:
        log.exception("Delivery %s encountered ambiguous error after send began, marking uncertain: %s", delivery_id, exc)
        await finalize_delivery_outcome(
            session_maker,
            delivery_id,
            claim_token,
            status="uncertain",
            finished_at=effective_now,
            last_error=f"Ambiguous network exception: {exc}",
            now=effective_now,
        )
        return "uncertain"



async def run_event_delivery_worker_cycle(
    session_maker: async_sessionmaker[AsyncSession],
    *,
    bot: Bot | None = None,
    max_client: MaxApiClient | None = None,
    batch_size: int = 25,
    now: datetime | None = None,
) -> dict[str, int]:
    """
    Runs one cycle of the delivery worker:
    1. Lease recovery sweeper.
    2. Claims due pending deliveries.
    3. Processes claimed deliveries concurrently.
    """
    effective_now = now or datetime.utcnow()
    stats: dict[str, int] = {
        "recovered_claimed": 0,
        "recovered_sending": 0,
        "claimed": 0,
        "sent": 0,
        "blocked": 0,
        "failed": 0,
        "uncertain": 0,
        "retried": 0,
        "rejected": 0,
    }

    # Step 1: Sweeper
    rec_c, rec_s = await recover_expired_leases(session_maker, now=effective_now)
    stats["recovered_claimed"] = rec_c
    stats["recovered_sending"] = rec_s

    # Step 2: Claim
    claimed = await claim_pending_deliveries(
        session_maker, batch_size=batch_size, now=effective_now
    )
    stats["claimed"] = len(claimed)

    # Step 3: Process
    for delivery_id, claim_token in claimed:
        outcome = await process_single_delivery(
            session_maker,
            delivery_id,
            claim_token,
            bot=bot,
            max_client=max_client,
            now=effective_now,
        )
        if outcome == "sent":
            stats["sent"] += 1
        elif outcome == "blocked":
            stats["blocked"] += 1
        elif outcome == "failed":
            stats["failed"] += 1
        elif outcome == "uncertain":
            stats["uncertain"] += 1
        elif "rate_limited" in outcome or "retry" in outcome:
            stats["retried"] += 1
        elif "pre_send_rejected" in outcome:
            stats["rejected"] += 1

    return stats


async def dispatch_immediate_delivery(
    session_maker: async_sessionmaker[AsyncSession],
    delivery_id: int,
    *,
    bot: Bot | None = None,
    max_client: MaxApiClient | None = None,
    now: datetime | None = None,
) -> str | None:
    """
    Attempts to atomically claim and dispatch a specific pending delivery immediately
    using the durable delivery worker protocol.
    """
    effective_now = now or datetime.utcnow()
    token = secrets.token_hex(16)
    lease_until = effective_now + timedelta(seconds=60)

    async with session_maker() as session:
        claim_stmt = (
            update(EventDelivery)
            .where(
                EventDelivery.id == delivery_id,
                EventDelivery.status == "pending",
            )
            .values(
                status="claimed",
                claim_token=token,
                claimed_at=effective_now,
                lease_until=lease_until,
                updated_at=effective_now,
            )
        )
        res = await session.execute(claim_stmt)
        await session.commit()
        if (res.rowcount or 0) == 0:
            return None

    return await process_single_delivery(
        session_maker,
        delivery_id=delivery_id,
        claim_token=token,
        bot=bot,
        max_client=max_client,
        now=effective_now,
    )

