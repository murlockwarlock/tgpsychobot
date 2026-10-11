"""
Service layer for Event Reminder Chains.

Provides:
- User event registration, duplicate handling, cancellation, rejoin, reschedule.
- User metadata synchronization with multiset merge (protecting against lost updates).
- User transaction runner with process-local locks and SQLite WAL retry support.
- Pre-send CAS and lock hierarchy validation (Scenario -> Registration -> Step -> Delivery).
- Template rendering with Moscow timezone (MSK).
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, TypeVar

from sqlalchemy import and_, or_, select, text, update
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from database import (
    DeepLinkNamespace,
    EventDelivery,
    EventRegistration,
    EventScenario,
    EventStep,
    User,
)
from max_messenger_bot.identity import is_max_user_id
from time_helpers import MSK, format_msk, to_msk, utc_now
from user_metadata import load_metadata_records

log = logging.getLogger(__name__)

T = TypeVar("T")

class AsyncReentrantLock:
    """An asyncio-compatible re-entrant lock for task-level nesting."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task | None = None
        self._count: int = 0

    async def acquire(self) -> bool:
        current_task = asyncio.current_task()
        if self._owner is not None and self._owner == current_task:
            self._count += 1
            return True
        await self._lock.acquire()
        self._owner = current_task
        self._count = 1
        return True

    def release(self) -> None:
        current_task = asyncio.current_task()
        if self._owner != current_task:
            raise RuntimeError(f"Cannot release lock owned by {self._owner} from {current_task}")
        self._count -= 1
        if self._count == 0:
            self._owner = None
            self._lock.release()

    async def __aenter__(self) -> AsyncReentrantLock:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


# In-memory per-user reentrant locks for local transaction serialization
_user_tx_locks: dict[int, AsyncReentrantLock] = {}
_locks_guard = asyncio.Lock()


def get_user_tx_lock(user_id: int) -> AsyncReentrantLock:
    """Returns an AsyncReentrantLock for the given user_id."""
    lock = _user_tx_locks.get(user_id)
    if lock is None:
        lock = AsyncReentrantLock()
        _user_tx_locks[user_id] = lock
    return lock


def is_sqlite_busy_exception(exc: Exception) -> bool:
    """Checks whether an exception is an SQLite lock conflict."""
    msg = str(exc).lower()
    return any(
        err in msg
        for err in (
            "database is locked",
            "sqlite_busy",
            "busy_snapshot",
            "locking protocol",
            "cannot start a transaction within a transaction",
        )
    )


def _record_sig(r: dict[str, Any]) -> str:
    """Normalized signature of a metadata record for stable multiset matching."""
    data = r.get("data")
    data_json = json.dumps(data, sort_keys=True, ensure_ascii=False) if isinstance(data, dict) else ""
    return f"{r.get('saved_at')}|{data_json}"


async def sync_and_append_user_metadata(
    session: AsyncSession,
    user: User,
    data_blocks: list[dict[str, Any]],
    *,
    saved_at: str | None = None,
) -> str:
    """
    Safely reads fresh metadata_json from DB (FOR UPDATE on PG) and merges with new blocks.

    Consistency guarantees:
    1. Executes strictly under session.no_autoflush to prevent flushing dirty User state prematurely.
    2. On PostgreSQL: SELECT ... FOR UPDATE locks the user row until commit().
    3. On SQLite: Serializes with get_user_tx_lock(user.id) unless caller already holds it.
    4. Multiset merge preserves uncommitted local records if called multiple times before commit.
    5. Eliminates duplicate blocks on transaction retries.
    """
    bind = session.get_bind()
    dialect_name = getattr(getattr(bind, "dialect", None), "name", "")
    is_postgres = dialect_name == "postgresql"

    async def _do_sync() -> str:
        with session.no_autoflush:
            stmt = select(User.metadata_json).where(User.id == user.id)
            if is_postgres:
                stmt = stmt.with_for_update()
            fresh_db_raw = await session.scalar(stmt)

        db_records = load_metadata_records(fresh_db_raw)
        orm_records = load_metadata_records(user.metadata_json)

        # Count frequencies of committed DB records
        db_counts: dict[str, int] = {}
        for r in db_records:
            sig = _record_sig(r)
            db_counts[sig] = db_counts.get(sig, 0) + 1

        # Extract local uncommitted records already in ORM
        uncommitted_local: list[dict[str, Any]] = []
        for r in orm_records:
            sig = _record_sig(r)
            if db_counts.get(sig, 0) > 0:
                db_counts[sig] -= 1
            else:
                uncommitted_local.append(r)

        # Base records = fresh committed DB snapshot + local uncommitted ORM records
        combined_records = list(db_records) + uncommitted_local

        # Add new incoming blocks
        timestamp = saved_at or datetime.now(timezone.utc).isoformat()
        for block in data_blocks:
            data = block.get("data")
            if not isinstance(data, dict):
                continue
            combined_records.append({
                "saved_at": timestamp,
                "data": data,
                "raw_json": block.get("raw_json") or json.dumps(data, ensure_ascii=False, indent=2),
            })

        new_raw = json.dumps(
            {"_format": "records_v1", "records": combined_records},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        user.metadata_json = new_raw
        return new_raw

    if not is_postgres:
        async with get_user_tx_lock(user.id):
            return await _do_sync()
    else:
        return await _do_sync()


async def run_user_transaction(
    session_maker: async_sessionmaker[AsyncSession],
    user_id: int,
    operation_fn: Callable[[AsyncSession, User], Awaitable[T]],
    *,
    max_retries: int = 5,
    base_backoff_sec: float = 0.05,
) -> T:
    """
    Executes a user transaction with atomicity guarantees across SQLite and PostgreSQL.

    - Holds asyncio.Lock(user_id) for the full duration of the transaction.
    - Uses SELECT ... FOR UPDATE on PostgreSQL.
    - Automatically retries on SQLite WAL conflicts (SQLITE_BUSY_SNAPSHOT).
    - Guarantees transactional persistence of User.metadata_json and event tables.
    """
    async with get_user_tx_lock(user_id):
        for attempt in range(max_retries):
            session = session_maker()
            try:
                bind = session.get_bind()
                dialect_name = getattr(getattr(bind, "dialect", None), "name", "")
                is_postgres = dialect_name == "postgresql"

                if not is_postgres:
                    await session.execute(text("BEGIN IMMEDIATE"))
                else:
                    await session.begin()

                stmt = select(User).where(User.id == user_id)
                if is_postgres:
                    stmt = stmt.with_for_update()

                user = await session.scalar(stmt)
                if user is None:
                    raise ValueError(f"Пользователь {user_id} не найден")

                result = await operation_fn(session, user)
                await session.commit()
                return result

            except (OperationalError, DBAPIError) as exc:
                if is_sqlite_busy_exception(exc) and attempt < max_retries - 1:
                    log.warning(
                        "SQLite busy conflict for user_id=%s (attempt %s/%s), retrying: %s",
                        user_id, attempt + 1, max_retries, exc
                    )
                    await session.rollback()
                    await asyncio.sleep(base_backoff_sec * (2 ** attempt))
                    continue
                await session.rollback()
                raise
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

        raise RuntimeError(f"Exceeded max retries for user_id={user_id}")


# Template formatting

DEFAULT_CONFIRMATION_TEMPLATE = (
    "Вы успешно зарегистрированы на мероприятие «{event_title}»!\n\n"
    "📅 Дата: {event_date}\n"
    "⏰ Время: {event_time} (МСК)\n\n"
    "Мы пришлем вам напоминания перед началом."
)

DEFAULT_ADAPTIVE_CONFIRMATION_TEMPLATE = (
    "Вы успешно зарегистрированы на мероприятие «{event_title}»!\n\n"
    "Мероприятие скоро начнётся или уже в эфире!\n"
    "🔗 Ссылка для подключения: {link}"
)

DEFAULT_LIVE_TEMPLATE = (
    "Мероприятие «{event_title}» скоро начнётся / уже в эфире!\n\n"
    "🔗 Ваша ссылка для входа: {link}"
)

DEFAULT_RESCHEDULE_TEMPLATE = (
    "Дата и время мероприятия «{event_title}» изменены!\n\n"
    "Новое время: {event_date} в {event_time} (МСК).\n"
    "{link}"
)


def render_event_template(
    template: str,
    scenario: EventScenario,
    user: User | None = None,
    platform: str = "telegram",
    *,
    now: datetime | None = None,
) -> str:
    """
    Renders an event message template with user and scenario substitutions.
    Substitutions supported:
    - {name}: user's display name or 'друг'
    - {event_title}, {title}: scenario title
    - {event_date}, {new_date_msk}: MSK date (dd.mm.yyyy)
    - {event_time}, {new_time_msk}: MSK time (HH:MM)
    - {link}, {link_or_place}: webinar link or meeting place
    - {place}: meeting place or webinar link
    """
    user_name = "друг"
    if user:
        raw_name = (user.first_name or user.name or "").strip()
        if raw_name:
            user_name = raw_name

    link_val = (scenario.link or scenario.place or "").strip()
    place_val = (scenario.place or scenario.link or "").strip()
    date_val = format_msk(scenario.starts_at, "%d.%m.%Y")
    time_val = format_msk(scenario.starts_at, "%H:%M")

    subs = {
        "name": user_name,
        "event_title": scenario.title,
        "title": scenario.title,
        "event_date": date_val,
        "new_date_msk": date_val,
        "event_time": time_val,
        "new_time_msk": time_val,
        "link": link_val,
        "link_or_place": link_val,
        "place": place_val,
    }

    result = template
    for key, val in subs.items():
        result = result.replace(f"{{{key}}}", val)
    return result


@dataclass(frozen=True)
class RegistrationResult:
    status: str  # 'registered', 'already_registered', 'rejoined', 'event_ended', 'scenario_not_found', 'scenario_inactive'
    registration: EventRegistration | None
    scenario: EventScenario | None
    mode: str
    immediate_delivery: EventDelivery | None


@dataclass(frozen=True)
class CancelResult:
    cancelled: bool
    already_cancelled: bool
    registration_id: int


@dataclass(frozen=True)
class RejoinResult:
    registration_id: int
    generation: int
    mode: str
    immediate_delivery: EventDelivery


@dataclass(frozen=True)
class RescheduleResult:
    event_id: int
    new_starts_at: datetime
    schedule_revision: int
    registrations_count: int


async def register_user_for_event(
    session_maker: async_sessionmaker[AsyncSession],
    user_id: int,
    event_code: str,
    platform: str = "telegram",
    *,
    now: datetime | None = None,
) -> RegistrationResult:

    """
    Registers user for an event scenario.

    Semantics:
    - Re-entry by already registered active user -> status='already_registered' (no duplicates).
    - Cancelled user re-entering -> seamlessly rejoins with new generation and mode.
    - Standard vs Urgent Late determined based on now relative to starts_at - 15m.
    - Rejects ended events (now >= starts_at + duration).
    """
    effective_now = now or datetime.utcnow()
    norm_code = event_code.strip().lower()

    async def _op(session: AsyncSession, user: User) -> RegistrationResult:
        # 1. Level 3: EventScenario FOR SHARE
        is_pg = session.bind and session.bind.dialect.name == "postgresql"
        scen_stmt = select(EventScenario).where(EventScenario.code == norm_code)
        if is_pg:
            scen_stmt = scen_stmt.with_for_update(read=True)
        scenario = await session.scalar(scen_stmt)

        if scenario is None or scenario.is_archived:
            return RegistrationResult(
                status="scenario_not_found",
                registration=None,
                scenario=None,
                mode="",
                immediate_delivery=None,
            )

        if not scenario.is_active:
            return RegistrationResult(
                status="scenario_inactive",
                registration=None,
                scenario=scenario,
                mode="",
                immediate_delivery=None,
            )

        duration = timedelta(minutes=scenario.duration_minutes)
        end_time = scenario.starts_at + duration
        if effective_now >= end_time:
            return RegistrationResult(
                status="event_ended",
                registration=None,
                scenario=scenario,
                mode="",
                immediate_delivery=None,
            )

        # 2. Level 4: Check existing registration FOR UPDATE
        reg_stmt = select(EventRegistration).where(
            EventRegistration.event_id == scenario.id,
            EventRegistration.user_id == user.id,
        )
        if is_pg:
            reg_stmt = reg_stmt.with_for_update()
        reg = await session.scalar(reg_stmt)

        if reg is not None and reg.status == "started":
            # Already actively registered: idempotent return, no duplicate delivery or gen increment
            return RegistrationResult(
                status="already_registered",
                registration=reg,
                scenario=scenario,
                mode=reg.registration_mode,
                immediate_delivery=None,
            )

        cutoff = scenario.starts_at - timedelta(minutes=15)
        is_urgent = effective_now >= cutoff
        target_mode = "urgent_late" if is_urgent else "standard"

        if reg is not None and reg.status == "cancelled":
            # Rejoin flow for previously cancelled registration
            reg.generation += 1
            new_gen = reg.generation
            reg.status = "started"
            reg.registration_mode = target_mode
            reg.rejoined_at = effective_now
            reg.cancelled_at = None

            # Level 5: Steps FOR SHARE in stable id order
            steps_stmt = (
                select(EventStep)
                .where(EventStep.event_id == scenario.id)
                .order_by(EventStep.id.asc())
            )
            if is_pg:
                steps_stmt = steps_stmt.with_for_update(read=True)
            steps = (await session.execute(steps_stmt)).scalars().all()
            steps_by_id = {s.id: s for s in steps}

            # Level 6: Deliveries FOR UPDATE in stable id order
            deliv_stmt = (
                select(EventDelivery)
                .where(
                    EventDelivery.registration_id == reg.id,
                    EventDelivery.status == "skipped",
                    EventDelivery.skip_reason == "cancelled",
                )
                .order_by(EventDelivery.id.asc())
            )
            if is_pg:
                deliv_stmt = deliv_stmt.with_for_update()
            cancelled_deliveries = (await session.execute(deliv_stmt)).scalars().all()

            if target_mode == "standard":
                for d in cancelled_deliveries:
                    if d.message_kind == "scheduled_step":
                        if d.step_id:
                            step = steps_by_id.get(d.step_id)
                            if step is None or step.is_archived:
                                d.status = "skipped"
                                d.skip_reason = "step_archived"
                                continue
                            new_due = scenario.starts_at + timedelta(minutes=step.offset_minutes)
                            if new_due <= effective_now:
                                d.status = "skipped"
                                d.skip_reason = "late_registration"
                                continue
                            d.due_at = new_due
                            d.next_attempt_at = new_due
                            d.schedule_revision = scenario.schedule_revision
                            d.registration_generation = new_gen
                            d.status = "pending"
                            d.skip_reason = None
                            d.claim_token = None
                            d.lease_until = None
                            d.send_started_at = None
                            d.claimed_at = None
                            d.updated_at = effective_now

                confirm_key = f"ev:{scenario.id}:reg:{reg.id}:gen:{new_gen}:kind:immediate_confirmation:r0"
                imm = EventDelivery(
                    event_id=scenario.id,
                    user_id=user.id,
                    registration_id=reg.id,
                    step_id=None,
                    retry_index=0,
                    message_kind="immediate_confirmation",
                    delivery_key=confirm_key,
                    due_at=effective_now,
                    next_attempt_at=effective_now,
                    expires_at=end_time,
                    schedule_revision=scenario.schedule_revision,
                    registration_generation=new_gen,
                    platform=platform,
                    status="pending",
                )
                session.add(imm)
            else:
                for d in cancelled_deliveries:
                    if d.message_kind == "scheduled_step":
                        d.status = "skipped"
                        d.skip_reason = "late_mode_suppressed"

                live_key = f"ev:{scenario.id}:reg:{reg.id}:gen:{new_gen}:kind:immediate_live:r0"
                imm = EventDelivery(
                    event_id=scenario.id,
                    user_id=user.id,
                    registration_id=reg.id,
                    step_id=None,
                    retry_index=0,
                    message_kind="immediate_live",
                    delivery_key=live_key,
                    due_at=effective_now,
                    next_attempt_at=effective_now,
                    expires_at=end_time,
                    schedule_revision=scenario.schedule_revision,
                    registration_generation=new_gen,
                    platform=platform,
                    status="pending",
                )
                session.add(imm)

            rejoin_payload = {
                "type": "event_scenario",
                "scenario_id": scenario.id,
                "code": scenario.code,
                "status": "started",
                "action": "rejoined",
                "generation": new_gen,
                "timestamp": effective_now.isoformat(),
            }
            await sync_and_append_user_metadata(
                session,
                user,
                [{"data": rejoin_payload, "raw_json": json.dumps(rejoin_payload, ensure_ascii=False)}],
            )
            return RegistrationResult(
                status="rejoined",
                registration=reg,
                scenario=scenario,
                mode=target_mode,
                immediate_delivery=imm,
            )

        # Brand new registration
        reg = EventRegistration(
            event_id=scenario.id,
            user_id=user.id,
            status="started",
            generation=1,
            registration_mode=target_mode,
            registered_at=effective_now,
        )
        session.add(reg)
        await session.flush()

        steps_stmt = (
            select(EventStep)
            .where(EventStep.event_id == scenario.id, EventStep.is_archived.is_(False))
            .order_by(EventStep.offset_minutes.asc(), EventStep.sort_order.asc(), EventStep.id.asc())
        )
        if is_pg:
            steps_stmt = steps_stmt.with_for_update(read=True)
        steps = (await session.execute(steps_stmt)).scalars().all()


        if target_mode == "standard":
            confirm_key = f"ev:{scenario.id}:reg:{reg.id}:gen:1:kind:immediate_confirmation:r0"
            imm = EventDelivery(
                event_id=scenario.id,
                user_id=user.id,
                registration_id=reg.id,
                step_id=None,
                retry_index=0,
                message_kind="immediate_confirmation",
                delivery_key=confirm_key,
                due_at=effective_now,
                next_attempt_at=effective_now,
                expires_at=end_time,
                schedule_revision=scenario.schedule_revision,
                registration_generation=1,
                platform=platform,
                status="pending",
            )
            session.add(imm)

            for step in steps:
                step_due = scenario.starts_at + timedelta(minutes=step.offset_minutes)
                step_key = f"ev:{scenario.id}:reg:{reg.id}:step:{step.id}:r0"
                if step_due <= effective_now:
                    step_status = "skipped"
                    step_skip_reason = "late_registration"
                else:
                    step_status = "pending"
                    step_skip_reason = None

                session.add(
                    EventDelivery(
                        event_id=scenario.id,
                        user_id=user.id,
                        registration_id=reg.id,
                        step_id=step.id,
                        retry_index=0,
                        message_kind="scheduled_step",
                        delivery_key=step_key,
                        due_at=step_due,
                        next_attempt_at=max(effective_now, step_due),
                        schedule_revision=scenario.schedule_revision,
                        registration_generation=1,
                        platform=platform,
                        status=step_status,
                        skip_reason=step_skip_reason,
                    )
                )
        else:
            live_key = f"ev:{scenario.id}:reg:{reg.id}:gen:1:kind:immediate_live:r0"
            imm = EventDelivery(
                event_id=scenario.id,
                user_id=user.id,
                registration_id=reg.id,
                step_id=None,
                retry_index=0,
                message_kind="immediate_live",
                delivery_key=live_key,
                due_at=effective_now,
                next_attempt_at=effective_now,
                expires_at=end_time,
                schedule_revision=scenario.schedule_revision,
                registration_generation=1,
                platform=platform,
                status="pending",
            )
            session.add(imm)

            for step in steps:
                step_due = scenario.starts_at + timedelta(minutes=step.offset_minutes)
                step_key = f"ev:{scenario.id}:reg:{reg.id}:step:{step.id}:r0"
                session.add(
                    EventDelivery(
                        event_id=scenario.id,
                        user_id=user.id,
                        registration_id=reg.id,
                        step_id=step.id,
                        retry_index=0,
                        message_kind="scheduled_step",
                        delivery_key=step_key,
                        due_at=step_due,
                        next_attempt_at=effective_now,
                        schedule_revision=scenario.schedule_revision,
                        registration_generation=1,
                        platform=platform,
                        status="skipped",
                        skip_reason="late_mode_suppressed",
                    )
                )

        start_payload = {
            "type": "event_scenario",
            "scenario_id": scenario.id,
            "code": scenario.code,
            "status": "started",
            "generation": 1,
            "timestamp": effective_now.isoformat(),
        }
        await sync_and_append_user_metadata(
            session,
            user,
            [{"data": start_payload, "raw_json": json.dumps(start_payload, ensure_ascii=False)}],
        )

        return RegistrationResult(
            status="registered",
            registration=reg,
            scenario=scenario,
            mode=target_mode,
            immediate_delivery=imm,
        )

    return await run_user_transaction(session_maker, user_id, _op)


async def cancel_event_registration(
    session_maker: async_sessionmaker[AsyncSession],
    user_id: int,
    event_id: int,
    *,
    now: datetime | None = None,
) -> CancelResult:
    """
    Cancels an active event registration.

    Semantics:
    - Idempotent: repeated cancel returns already_cancelled=True.
    - Registration status -> 'cancelled', cancelled_at set.
    - generation is NOT decremented.
    - Unsent deliveries of current generation -> skipped('cancelled').
    - History (sent, uncertain, terminal) is preserved.
    """
    effective_now = now or datetime.utcnow()

    async def _op(session: AsyncSession, user: User) -> CancelResult:
        is_pg = session.bind and session.bind.dialect.name == "postgresql"
        scen_stmt = select(EventScenario).where(EventScenario.id == event_id)
        if is_pg:
            scen_stmt = scen_stmt.with_for_update(read=True)
        scenario = await session.scalar(scen_stmt)
        if scenario is None:
            raise ValueError(f"Мероприятие {event_id} не найдено")

        reg_stmt = select(EventRegistration).where(
            EventRegistration.event_id == event_id,
            EventRegistration.user_id == user.id,
        )
        if is_pg:
            reg_stmt = reg_stmt.with_for_update()
        reg = await session.scalar(reg_stmt)
        if reg is None:
            raise ValueError(f"Регистрация на мероприятие {event_id} не найдена")

        if reg.status == "cancelled":
            return CancelResult(cancelled=False, already_cancelled=True, registration_id=reg.id)

        reg.status = "cancelled"
        reg.cancelled_at = effective_now

        deliv_stmt = select(EventDelivery).where(
            EventDelivery.registration_id == reg.id,
            EventDelivery.registration_generation == reg.generation,
            EventDelivery.status.in_(["pending", "claimed"]),
        )
        if is_pg:
            deliv_stmt = deliv_stmt.with_for_update()
        cancellable_deliveries = (await session.execute(deliv_stmt)).scalars().all()

        for d in cancellable_deliveries:
            if d.status in ("pending", "claimed"):
                d.status = "skipped"
                d.skip_reason = "cancelled"
                d.claim_token = None
                d.lease_until = None
                d.send_started_at = None
                d.claimed_at = None
                d.updated_at = effective_now


        cancel_payload = {
            "type": "event_scenario",
            "scenario_id": event_id,
            "code": scenario.code,
            "status": "cancelled",
            "generation": reg.generation,
            "timestamp": effective_now.isoformat(),
        }
        await sync_and_append_user_metadata(
            session,
            user,
            [{"data": cancel_payload, "raw_json": json.dumps(cancel_payload, ensure_ascii=False)}],
        )

        return CancelResult(cancelled=True, already_cancelled=False, registration_id=reg.id)

    return await run_user_transaction(session_maker, user_id, _op)


async def rejoin_event_registration(
    session_maker: async_sessionmaker[AsyncSession],
    user_id: int,
    event_id: int,
    platform: str | None = None,
    *,
    now: datetime | None = None,
) -> RejoinResult:
    """
    Rejoins a previously cancelled event registration.

    Semantics:
    - Only allowed from 'cancelled' status.
    - Increments generation.
    - Updates registration_mode to active generation mode (standard vs urgent_late).
    - NEVER restores old immediate deliveries (confirmation, live, reschedule_notice).
    - Only restores eligible future scheduled_step deliveries (if due_at > now and mode == standard).
    - Creates a single new immediate delivery for the new generation.
    """
    effective_now = now or datetime.utcnow()
    target_platform = platform or ("max" if is_max_user_id(user_id) else "telegram")

    async def _op(session: AsyncSession, user: User) -> RejoinResult:
        is_pg = session.bind and session.bind.dialect.name == "postgresql"
        # 1. Level 3: EventScenario FOR SHARE
        scen_stmt = select(EventScenario).where(EventScenario.id == event_id)
        if is_pg:
            scen_stmt = scen_stmt.with_for_update(read=True)
        scenario = await session.scalar(scen_stmt)
        if scenario is None or scenario.is_archived:
            raise ValueError("Сценарий не найден либо находится в архиве")

        cutoff = scenario.starts_at - timedelta(minutes=15)
        end_time = scenario.starts_at + timedelta(minutes=scenario.duration_minutes)
        if effective_now >= end_time:
            raise ValueError("Мероприятие уже завершено")

        # 2. Level 4: EventRegistration FOR UPDATE
        reg_stmt = select(EventRegistration).where(
            EventRegistration.event_id == event_id,
            EventRegistration.user_id == user.id,
        )
        if is_pg:
            reg_stmt = reg_stmt.with_for_update()
        reg = await session.scalar(reg_stmt)
        if reg is None or reg.status != "cancelled":
            raise ValueError("Регистрация не найдена либо уже активна")

        reg.generation += 1
        new_gen = reg.generation
        reg.status = "started"
        reg.rejoined_at = effective_now
        reg.cancelled_at = None

        is_urgent = effective_now >= cutoff
        new_mode = "urgent_late" if is_urgent else "standard"
        reg.registration_mode = new_mode

        # 3. Level 5: Steps FOR SHARE in stable id order
        steps_stmt = (
            select(EventStep)
            .where(EventStep.event_id == scenario.id)
            .order_by(EventStep.id.asc())
        )
        if is_pg:
            steps_stmt = steps_stmt.with_for_update(read=True)
        steps = (await session.execute(steps_stmt)).scalars().all()
        steps_by_id = {s.id: s for s in steps}

        # 4. Level 6: EventDelivery FOR UPDATE in stable id order
        deliv_stmt = (
            select(EventDelivery)
            .where(
                EventDelivery.registration_id == reg.id,
                EventDelivery.status == "skipped",
                EventDelivery.skip_reason == "cancelled",
            )
            .order_by(EventDelivery.id.asc())
        )
        if is_pg:
            deliv_stmt = deliv_stmt.with_for_update()
        cancelled_deliveries = (await session.execute(deliv_stmt)).scalars().all()

        if new_mode == "standard":
            for d in cancelled_deliveries:
                if d.message_kind == "scheduled_step":
                    if d.step_id:
                        step = steps_by_id.get(d.step_id)
                        if step is None or step.is_archived:
                            d.status = "skipped"
                            d.skip_reason = "step_archived"
                            continue
                        new_due = scenario.starts_at + timedelta(minutes=step.offset_minutes)
                        if new_due <= effective_now:
                            d.status = "skipped"
                            d.skip_reason = "late_registration"
                            continue
                        d.due_at = new_due
                        d.next_attempt_at = new_due
                        d.schedule_revision = scenario.schedule_revision
                        d.registration_generation = new_gen
                        d.status = "pending"
                        d.skip_reason = None
                        d.claim_token = None
                        d.lease_until = None
                        d.send_started_at = None
                        d.claimed_at = None
                        d.updated_at = effective_now

            confirm_key = f"ev:{event_id}:reg:{reg.id}:gen:{new_gen}:kind:immediate_confirmation:r0"
            imm = EventDelivery(
                event_id=event_id,
                user_id=user.id,
                registration_id=reg.id,
                step_id=None,
                retry_index=0,
                message_kind="immediate_confirmation",
                delivery_key=confirm_key,
                due_at=effective_now,
                next_attempt_at=effective_now,
                expires_at=end_time,
                schedule_revision=scenario.schedule_revision,
                registration_generation=new_gen,
                platform=target_platform,
                status="pending",
            )
            session.add(imm)
        else:
            for d in cancelled_deliveries:
                if d.message_kind == "scheduled_step":
                    d.status = "skipped"
                    d.skip_reason = "late_mode_suppressed"

            live_key = f"ev:{event_id}:reg:{reg.id}:gen:{new_gen}:kind:immediate_live:r0"
            imm = EventDelivery(
                event_id=event_id,
                user_id=user.id,
                registration_id=reg.id,
                step_id=None,
                retry_index=0,
                message_kind="immediate_live",
                delivery_key=live_key,
                due_at=effective_now,
                next_attempt_at=effective_now,
                expires_at=end_time,
                schedule_revision=scenario.schedule_revision,
                registration_generation=new_gen,
                platform=target_platform,
                status="pending",
            )
            session.add(imm)

        rejoin_payload = {
            "type": "event_scenario",
            "scenario_id": event_id,
            "code": scenario.code,
            "status": "started",
            "action": "rejoined",
            "generation": new_gen,
            "timestamp": effective_now.isoformat(),
        }
        await sync_and_append_user_metadata(
            session,
            user,
            [{"data": rejoin_payload, "raw_json": json.dumps(rejoin_payload, ensure_ascii=False)}],
        )


        return RejoinResult(
            registration_id=reg.id,
            generation=new_gen,
            mode=new_mode,
            immediate_delivery=imm,
        )

    return await run_user_transaction(session_maker, user_id, _op)


async def reschedule_event(
    session_maker: async_sessionmaker[AsyncSession],
    event_id: int,
    new_starts_at: datetime,
    *,
    now: datetime | None = None,
) -> RescheduleResult:
    """
    Reschedules event scenario start time.

    Semantics:
    - Increments schedule_revision and content_version.
    - Recalculates only future/unsent scheduled deliveries for active registrations in standard mode.
    - Stale unsent reschedule_notices and immediate_lives are invalidated with skip_reason='rescheduled_past'.
    - Old delivery revisions are NOT rewritten for historical records.
    - Exactly one new reschedule_notice is queued per active registration.
    - For urgent_late active registrations, queues new immediate_live at max(now, new_starts_at - 15m).
    - Rejects reschedule into completely past window (new_starts_at + duration <= now).
    """
    effective_now = now or datetime.utcnow()
    max_retries = 5

    for attempt in range(max_retries):
        session = session_maker()
        try:
            bind = session.get_bind()
            dialect_name = getattr(getattr(bind, "dialect", None), "name", "")
            is_postgres = dialect_name == "postgresql"

            if not is_postgres:
                await session.execute(text("BEGIN IMMEDIATE"))
            else:
                await session.begin()

            # 1. Level 3: EventScenario FOR UPDATE
            scen_stmt = select(EventScenario).where(EventScenario.id == event_id)
            if is_postgres:
                scen_stmt = scen_stmt.with_for_update()
            scenario = await session.scalar(scen_stmt)
            if scenario is None or scenario.is_archived:
                raise ValueError(f"Сценарий {event_id} не найден либо архивирован")

            duration = timedelta(minutes=scenario.duration_minutes)
            end_time = new_starts_at + duration
            if end_time <= effective_now:
                raise ValueError("Нельзя перенести мероприятие на время, которое уже полностью завершилось")

            scenario.starts_at = new_starts_at
            scenario.schedule_revision += 1
            scenario.content_version += 1
            new_rev = scenario.schedule_revision

            # 2. Level 4: Active registrations FOR UPDATE
            reg_stmt = select(EventRegistration).where(
                EventRegistration.event_id == event_id,
                EventRegistration.status == "started",
            )
            if is_postgres:
                reg_stmt = reg_stmt.with_for_update()
            registrations = (await session.execute(reg_stmt)).scalars().all()
            reg_map = {reg.id: reg for reg in registrations}
            reg_ids = list(reg_map.keys())

            # 3. Level 5: Steps FOR SHARE
            steps_stmt = select(EventStep).where(EventStep.event_id == event_id)
            if is_postgres:
                steps_stmt = steps_stmt.with_for_update(read=True)
            steps = (await session.execute(steps_stmt)).scalars().all()
            step_offsets = {step.id: step.offset_minutes for step in steps}

            if reg_ids:
                # 4. Level 6: EventDeliveries for active registrations FOR UPDATE
                deliv_stmt = (
                    select(EventDelivery)
                    .where(
                        EventDelivery.event_id == event_id,
                        EventDelivery.registration_id.in_(reg_ids),
                        or_(
                            EventDelivery.status.in_(["pending", "claimed"]),
                            and_(
                                EventDelivery.status == "skipped",
                                EventDelivery.skip_reason.in_(
                                    ["late_registration", "rescheduled_past"]
                                ),
                            ),
                            and_(
                                EventDelivery.status == "skipped",
                                EventDelivery.skip_reason == "expired",
                                EventDelivery.message_kind == "immediate_confirmation",
                            ),
                        ),
                    )
                )
                if is_postgres:
                    deliv_stmt = deliv_stmt.with_for_update()
                deliveries = (await session.execute(deliv_stmt)).scalars().all()

                for d in deliveries:
                    if d.status == "claimed":
                        d.status = "pending"
                        d.claim_token = None
                        d.lease_until = None
                        d.send_started_at = None
                        d.claimed_at = None

                    if d.message_kind in ("reschedule_notice", "immediate_live"):
                        # Stale notices and live links are invalidated, preserving their original revision
                        d.status = "skipped"
                        d.skip_reason = "rescheduled_past"
                        d.claim_token = None
                        d.lease_until = None
                        d.send_started_at = None
                        d.claimed_at = None

                    elif d.message_kind == "immediate_confirmation":
                        d.expires_at = end_time
                        d.schedule_revision = new_rev
                        if d.status == "skipped" and d.skip_reason == "expired":
                            if end_time > effective_now:
                                d.status = "pending"
                                d.skip_reason = None
                                d.claim_token = None
                                d.lease_until = None
                                d.send_started_at = None
                                d.claimed_at = None

                    elif d.step_id is not None:
                        reg = reg_map.get(d.registration_id)
                        if reg and reg.registration_mode == "urgent_late":
                            # Urgent late registrations keep steps suppressed
                            continue

                        offset = step_offsets.get(d.step_id, 0)
                        new_due_at = new_starts_at + timedelta(minutes=offset)
                        d.due_at = new_due_at
                        d.next_attempt_at = new_due_at
                        d.schedule_revision = new_rev

                        if new_due_at <= effective_now:
                            d.status = "skipped"
                            d.skip_reason = "rescheduled_past"
                        else:
                            d.status = "pending"
                            d.skip_reason = None
                            d.claim_token = None
                            d.lease_until = None
                            d.send_started_at = None
                            d.claimed_at = None

            # 5. Create up-to-date notices for each active registration
            for reg in registrations:
                reg_platform = "max" if is_max_user_id(reg.user_id) else "telegram"

                notice_key = f"ev:{event_id}:reg:{reg.id}:gen:{reg.generation}:rev:{new_rev}:kind:reschedule_notice:r0"
                session.add(
                    EventDelivery(
                        event_id=event_id,
                        user_id=reg.user_id,
                        registration_id=reg.id,
                        step_id=None,
                        retry_index=0,
                        message_kind="reschedule_notice",
                        delivery_key=notice_key,
                        due_at=effective_now,
                        next_attempt_at=effective_now,
                        expires_at=end_time,
                        schedule_revision=new_rev,
                        registration_generation=reg.generation,
                        platform=reg_platform,
                        status="pending",
                    )
                )

                if reg.registration_mode == "urgent_late":
                    live_key = f"ev:{event_id}:reg:{reg.id}:gen:{reg.generation}:rev:{new_rev}:kind:immediate_live:r0"
                    live_due_at = max(effective_now, new_starts_at - timedelta(minutes=15))
                    session.add(
                        EventDelivery(
                            event_id=event_id,
                            user_id=reg.user_id,
                            registration_id=reg.id,
                            step_id=None,
                            retry_index=0,
                            message_kind="immediate_live",
                            delivery_key=live_key,
                            due_at=live_due_at,
                            next_attempt_at=live_due_at,
                            expires_at=end_time,
                            schedule_revision=new_rev,
                            registration_generation=reg.generation,
                            platform=reg_platform,
                            status="pending",
                        )
                    )

            await session.commit()
            return RescheduleResult(
                event_id=event_id,
                new_starts_at=new_starts_at,
                schedule_revision=new_rev,
                registrations_count=len(registrations),
            )

        except (OperationalError, DBAPIError) as exc:
            if is_sqlite_busy_exception(exc) and attempt < max_retries - 1:
                log.warning("SQLite busy during reschedule_event (attempt %s), retrying: %s", attempt + 1, exc)
                await session.rollback()
                await asyncio.sleep(0.05 * (2 ** attempt))
                continue
            await session.rollback()
            raise
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    raise RuntimeError(f"Exceeded max retries for reschedule_event(event_id={event_id})")



async def prepare_and_lock_delivery_for_sending(
    session_maker: async_sessionmaker[AsyncSession],
    delivery_id: int,
    claim_token: str,
    expected_content_version: int,
    expected_step_version: int | None,
    rendered_text: str,
    template_text: str,
    *,
    now: datetime | None = None,
    max_retries: int = 5,
    base_backoff_sec: float = 0.05,
) -> tuple[bool, EventDelivery | None, str | None]:
    """
    Serializes pre-send checks strictly following Lock Ordering:
    Scenario (Level 3 FOR SHARE) -> Registration (Level 4 FOR SHARE) ->
    Step (Level 5 FOR SHARE) -> Delivery (Level 6 FOR UPDATE).

    On SQLite WAL: BEGIN IMMEDIATE prior to reading.
    Point of No Return: commit of CAS update to status='sending' releases all DB locks.
    Network HTTP send must happen ONLY AFTER this function returns success=True.
    """
    for attempt in range(max_retries):
        session = session_maker()
        try:
            bind = session.get_bind()
            dialect_name = getattr(getattr(bind, "dialect", None), "name", "")
            is_postgres = dialect_name == "postgresql"
            current_now = now if now is not None else datetime.utcnow()
            new_lease = current_now + timedelta(seconds=180)

            if not is_postgres:
                await session.execute(text("BEGIN IMMEDIATE"))
            else:
                await session.begin()

            # Step 0: Preliminary read of references without locks
            prelim_stmt = select(
                EventDelivery.event_id,
                EventDelivery.registration_id,
                EventDelivery.step_id,
                EventDelivery.status,
                EventDelivery.claim_token,
                EventDelivery.lease_until,
                EventDelivery.due_at,
                EventDelivery.next_attempt_at,
            ).where(EventDelivery.id == delivery_id)
            prelim = (await session.execute(prelim_stmt)).one_or_none()
            if not prelim:
                await session.rollback()
                return False, None, "not_found"

            event_id, reg_id, step_id, prelim_status, prelim_token, prelim_lease, due_at, next_attempt_at = prelim

            if prelim_status != "claimed" or prelim_token != claim_token:
                await session.rollback()
                return False, None, "stale_or_invalid_claim"
            if not prelim_lease or prelim_lease <= current_now:
                await session.rollback()
                return False, None, "stale_or_expired_lease"
            if due_at > current_now or next_attempt_at > current_now:
                await session.rollback()
                return False, None, "not_due_yet"

            # Step 1: Level 3 - EventScenario FOR SHARE
            scen_stmt = select(EventScenario).where(EventScenario.id == event_id)
            if is_postgres:
                scen_stmt = scen_stmt.with_for_update(read=True)
            scenario = await session.scalar(scen_stmt)
            if not scenario or scenario.is_archived:
                await session.rollback()
                return False, None, "scenario_archived"
            if scenario.content_version != expected_content_version:
                await session.rollback()
                return False, None, "content_version_mismatch"

            # Step 2: Level 4 - EventRegistration FOR SHARE
            reg_stmt = select(EventRegistration).where(EventRegistration.id == reg_id)
            if is_postgres:
                reg_stmt = reg_stmt.with_for_update(read=True)
            reg = await session.scalar(reg_stmt)
            if not reg:
                await session.rollback()
                return False, None, "registration_not_found"

            # Step 3: Level 5 - EventStep FOR SHARE
            step = None
            if step_id is not None:
                step_stmt = select(EventStep).where(EventStep.id == step_id)
                if is_postgres:
                    step_stmt = step_stmt.with_for_update(read=True)
                step = await session.scalar(step_stmt)
                if expected_step_version is not None and step and step.step_version != expected_step_version:
                    await session.rollback()
                    return False, None, "step_version_mismatch"

            # Step 4: Level 6 - EventDelivery FOR UPDATE
            deliv_stmt = select(EventDelivery).where(EventDelivery.id == delivery_id)
            if is_postgres:
                deliv_stmt = deliv_stmt.with_for_update()
            delivery = await session.scalar(deliv_stmt)
            if not delivery:
                await session.rollback()
                return False, None, "not_found"

            # Step 5: Post-lock re-verification
            if delivery.event_id != event_id or delivery.registration_id != reg_id or delivery.step_id != step_id:
                await session.rollback()
                return False, None, "links_changed"

            if delivery.status != "claimed" or delivery.claim_token != claim_token:
                await session.rollback()
                return False, None, "stale_or_invalid_claim"

            if not delivery.lease_until or delivery.lease_until <= current_now:
                await session.rollback()
                return False, None, "stale_or_expired_lease"

            if delivery.due_at > current_now or delivery.next_attempt_at > current_now:
                await session.rollback()
                return False, None, "not_due_yet"

            if not scenario.is_active:
                if delivery.due_at > current_now:
                    delivery.status = "pending"
                    delivery.claim_token = None
                    delivery.lease_until = None
                else:
                    delivery.status = "skipped"
                    delivery.skip_reason = "disabled_past"
                    delivery.claim_token = None
                    delivery.lease_until = None
                await session.commit()
                return False, delivery, "scenario_inactive"

            if scenario.schedule_revision != delivery.schedule_revision:
                await session.rollback()
                return False, None, "schedule_rescheduled"

            if reg.status != "started" or reg.generation != delivery.registration_generation:
                delivery.status = "skipped"
                delivery.skip_reason = "cancelled"
                delivery.claim_token = None
                delivery.lease_until = None
                await session.commit()
                return False, delivery, "registration_cancelled"

            if delivery.step_id is not None and (not step or step.is_archived):
                delivery.status = "skipped"
                delivery.skip_reason = "step_archived"
                delivery.claim_token = None
                delivery.lease_until = None
                await session.commit()
                return False, delivery, "step_archived"

            if delivery.expires_at and delivery.expires_at <= current_now:
                delivery.status = "skipped"
                delivery.skip_reason = "expired"
                delivery.finished_at = current_now
                delivery.claim_token = None
                delivery.lease_until = None
                await session.commit()
                return False, delivery, "expired"

            # Step 6: Atomic CAS UPDATE
            current_now = now if now is not None else datetime.utcnow()
            new_lease = current_now + timedelta(seconds=180)
            cas_stmt = (
                update(EventDelivery)
                .where(
                    EventDelivery.id == delivery_id,
                    EventDelivery.status == "claimed",
                    EventDelivery.claim_token == claim_token,
                    EventDelivery.schedule_revision == scenario.schedule_revision,
                    EventDelivery.registration_generation == reg.generation,
                    EventDelivery.lease_until.is_not(None),
                    EventDelivery.lease_until > current_now,
                    or_(EventDelivery.expires_at.is_(None), EventDelivery.expires_at > current_now),
                )
                .values(
                    status="sending",
                    send_started_at=current_now,
                    lease_until=new_lease,
                    attempt_count=EventDelivery.attempt_count + 1,
                    rendered_text_snapshot=rendered_text,
                    template_text_snapshot=template_text,
                    updated_at=current_now,
                )
            )
            res = await session.execute(cas_stmt)
            if res.rowcount != 1:
                await session.rollback()
                return False, None, "cas_failed"

            # Point of No Return commit
            await session.commit()
            return True, delivery, None

        except (OperationalError, DBAPIError) as exc:
            if is_sqlite_busy_exception(exc) and attempt < max_retries - 1:
                log.warning("SQLite busy during pre-send (%s/%s), retrying: %s", attempt + 1, max_retries, exc)
                await session.rollback()
                await asyncio.sleep(base_backoff_sec * (2 ** attempt))
                continue
            await session.rollback()
            raise
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    raise RuntimeError(f"Exceeded max attempts for pre-send on delivery_id={delivery_id}")


# ==============================================================================
# Telegram and MAX Platform Adapters
# ==============================================================================

async def handle_event_start_telegram(
    bot: Any,
    chat_id: int,
    user_id: int,
    start_arg: str | None,
    *,
    session_maker: async_sessionmaker[AsyncSession] | None = None,
    now: datetime | None = None,
) -> bool:
    """
    Handles Telegram /start <event_code>. Returns True if handled as an event, False otherwise.
    """
    if not start_arg:
        return False
    from database import async_session_maker as default_session_maker
    sm = session_maker or default_session_maker
    norm_code = start_arg.strip().lower()
    effective_now = now or datetime.utcnow()

    async with sm() as session:
        scenario = await session.scalar(
            select(EventScenario).where(EventScenario.code == norm_code)
        )
        if scenario is None or scenario.is_archived:
            return False

    reg_res = await register_user_for_event(sm, user_id, norm_code, platform="telegram", now=effective_now)

    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    if reg_res.status == "scenario_not_found":
        return False

    if reg_res.status == "scenario_inactive":
        await bot.send_message(
            chat_id=chat_id,
            text="Регистрация на данное мероприятие временно недоступна.",
            parse_mode="HTML",
        )
        return True

    if reg_res.status == "event_ended":
        title = reg_res.scenario.title if reg_res.scenario else norm_code
        await bot.send_message(
            chat_id=chat_id,
            text=f"Регистрация на мероприятие «{title}» завершена, так как мероприятие уже прошло.",
            parse_mode="HTML",
        )
        return True

    scen = reg_res.scenario
    assert scen is not None
    markup = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="❌ Не смогу прийти", callback_data=f"ev_cancel:{scen.id}")]]
    )

    if reg_res.status == "already_registered":
        duration = timedelta(minutes=scen.duration_minutes)
        is_live = scen.starts_at <= effective_now < scen.starts_at + duration
        if is_live:
            text_body = (
                f"Вы уже зарегистрированы на мероприятие «{scen.title}»!\n\n"
                f"Оно сейчас в эфире!\n"
                f"🔗 Ваша ссылка для входа: {scen.link or scen.place}"
            )
        else:
            text_body = (
                f"Вы уже зарегистрированы на мероприятие «{scen.title}»!\n\n"
                f"📅 Дата: {format_msk(scen.starts_at, '%d.%m.%Y')}\n"
                f"⏰ Время: {format_msk(scen.starts_at, '%H:%M')} (МСК)"
            )
        await bot.send_message(chat_id=chat_id, text=text_body, parse_mode="HTML", reply_markup=markup)
        return True

    if reg_res.status in ("registered", "rejoined"):
        if reg_res.immediate_delivery:
            from event_delivery_worker import dispatch_immediate_delivery
            await dispatch_immediate_delivery(
                sm,
                reg_res.immediate_delivery.id,
                bot=bot,
                now=effective_now,
            )
        return True

    return False



async def handle_event_callback_telegram(
    callback: Any,
    *,
    session_maker: async_sessionmaker[AsyncSession] | None = None,
    now: datetime | None = None,
) -> bool:
    """
    Handles Telegram callbacks: ev_cancel:<event_id> and ev_rejoin:<event_id>.
    """
    data = str(callback.data or "")
    if not (data.startswith("ev_cancel:") or data.startswith("ev_rejoin:")):
        return False

    from database import async_session_maker as default_session_maker
    sm = session_maker or default_session_maker
    user_id = callback.from_user.id
    effective_now = now or datetime.utcnow()
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    if data.startswith("ev_cancel:"):
        try:
            event_id = int(data.split(":", 1)[1])
        except ValueError:
            await callback.answer("Ошибка формата", show_alert=True)
            return True

        res = await cancel_event_registration(sm, user_id, event_id, now=effective_now)
        await callback.answer("Регистрация отменена")
        markup = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="✅ Снова смогу прийти", callback_data=f"ev_rejoin:{event_id}")]]
        )
        cancel_text = "Регистрация отменена. Если ваши планы изменятся, вы можете вернуться в любое время."
        try:
            await callback.message.edit_text(cancel_text, reply_markup=markup, parse_mode="HTML")
        except Exception:
            await callback.message.answer(cancel_text, reply_markup=markup, parse_mode="HTML")
        return True

    if data.startswith("ev_rejoin:"):
        try:
            event_id = int(data.split(":", 1)[1])
        except ValueError:
            await callback.answer("Ошибка формата", show_alert=True)
            return True

        try:
            res = await rejoin_event_registration(sm, user_id, event_id, platform="telegram", now=effective_now)
            await callback.answer("Участие возобновлено!")
            markup = InlineKeyboardMarkup(
                inline_keyboard=[[InlineKeyboardButton(text="❌ Не смогу прийти", callback_data=f"ev_cancel:{event_id}")]]
            )
            rejoin_text = "Вы снова зарегистрированы на мероприятие! Мы пришлем вам актуальные ссылки и напоминания."
            try:
                await callback.message.edit_text(rejoin_text, reply_markup=markup, parse_mode="HTML")
            except Exception:
                await callback.message.answer(rejoin_text, reply_markup=markup, parse_mode="HTML")
        except Exception as exc:
            await callback.answer(f"Не удалось возобновить: {exc}", show_alert=True)
        return True

    return False


async def handle_event_start_max(
    client: Any,
    chat_id: int,
    user_id: int,
    start_payload: str | None,
    *,
    session_maker: async_sessionmaker[AsyncSession] | None = None,
    now: datetime | None = None,
) -> bool:
    """
    Handles MAX start/deep-link with event code. Returns True if handled as an event, False otherwise.
    """
    if not start_payload:
        return False
    from database import async_session_maker as default_session_maker
    sm = session_maker or default_session_maker
    norm_code = start_payload.strip().lower()
    effective_now = now or datetime.utcnow()

    async with sm() as session:
        scenario = await session.scalar(
            select(EventScenario).where(EventScenario.code == norm_code)
        )
        if scenario is None or scenario.is_archived:
            return False

    reg_res = await register_user_for_event(sm, user_id, norm_code, platform="max", now=effective_now)

    from max_messenger_bot.keyboards import callback_button, inline_keyboard

    if reg_res.status == "scenario_not_found":
        return False

    if reg_res.status == "scenario_inactive":
        await client.send_message(
            chat_id=chat_id,
            text="Регистрация на данное мероприятие временно недоступна.",
        )
        return True

    if reg_res.status == "event_ended":
        title = reg_res.scenario.title if reg_res.scenario else norm_code
        await client.send_message(
            chat_id=chat_id,
            text=f"Регистрация на мероприятие «{title}» завершена, так как мероприятие уже прошло.",
        )
        return True

    scen = reg_res.scenario
    assert scen is not None
    attachments = inline_keyboard([[callback_button("❌ Не смогу прийти", f"ev_cancel:{scen.id}")]])

    if reg_res.status == "already_registered":
        duration = timedelta(minutes=scen.duration_minutes)
        is_live = scen.starts_at <= effective_now < scen.starts_at + duration
        if is_live:
            text_body = (
                f"Вы уже зарегистрированы на мероприятие «{scen.title}»!\n\n"
                f"Оно сейчас в эфире!\n"
                f"🔗 Ваша ссылка для входа: {scen.link or scen.place}"
            )
        else:
            text_body = (
                f"Вы уже зарегистрированы на мероприятие «{scen.title}»!\n\n"
                f"📅 Дата: {format_msk(scen.starts_at, '%d.%m.%Y')}\n"
                f"⏰ Время: {format_msk(scen.starts_at, '%H:%M')} (МСК)"
            )
        await client.send_message(chat_id=chat_id, text=text_body, attachments=attachments)
        return True

    if reg_res.status in ("registered", "rejoined"):
        if reg_res.immediate_delivery:
            from event_delivery_worker import dispatch_immediate_delivery
            await dispatch_immediate_delivery(
                sm,
                reg_res.immediate_delivery.id,
                max_client=client,
                now=effective_now,
            )
        return True

    return False



async def handle_event_callback_max(
    client: Any,
    callback: Any,
    *,
    session_maker: async_sessionmaker[AsyncSession] | None = None,
    now: datetime | None = None,
) -> bool:
    """
    Handles MAX callbacks: ev_cancel:<event_id> and ev_rejoin:<event_id>.
    """
    data = str(getattr(callback, "payload", "") or "")
    if not (data.startswith("ev_cancel:") or data.startswith("ev_rejoin:")):
        return False

    from database import async_session_maker as default_session_maker
    sm = session_maker or default_session_maker
    user_id = callback.sender.user_id
    chat_id = callback.chat_id
    effective_now = now or datetime.utcnow()
    from max_messenger_bot.keyboards import callback_button, inline_keyboard

    if data.startswith("ev_cancel:"):
        try:
            event_id = int(data.split(":", 1)[1])
        except ValueError:
            await client.answer_callback(callback.callback_id)
            return True

        res = await cancel_event_registration(sm, user_id, event_id, now=effective_now)
        await client.answer_callback(callback.callback_id)
        attachments = inline_keyboard([[callback_button("✅ Снова смогу прийти", f"ev_rejoin:{event_id}")]])
        cancel_text = "Регистрация отменена. Если ваши планы изменятся, вы можете вернуться в любое время."
        await client.send_message(chat_id=chat_id, text=cancel_text, attachments=attachments)
        return True

    if data.startswith("ev_rejoin:"):
        try:
            event_id = int(data.split(":", 1)[1])
        except ValueError:
            await client.answer_callback(callback.callback_id)
            return True

        try:
            res = await rejoin_event_registration(sm, user_id, event_id, platform="max", now=effective_now)
            await client.answer_callback(callback.callback_id)
            attachments = inline_keyboard([[callback_button("❌ Не смогу прийти", f"ev_cancel:{event_id}")]])
            rejoin_text = "Вы снова зарегистрированы на мероприятие! Мы пришлем вам актуальные ссылки и напоминания."
            await client.send_message(chat_id=chat_id, text=rejoin_text, attachments=attachments)
        except Exception as exc:
            await client.answer_callback(callback.callback_id)
            await client.send_message(chat_id=chat_id, text=f"Не удалось возобновить: {exc}")
        return True

    return False

