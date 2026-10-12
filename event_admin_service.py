"""
Service layer and shared business helpers for Event Reminder Chains Admin UI.

Provides platform-neutral logic for:
- Datetime & offset parsing/formatting (MSK to UTC and UTC to MSK).
- Input validation (codes, types, links, places, durations, offsets).
- Scenario management (list, get, create, edit, copy, toggle active, archive, reschedule).
- Reminder step management (list, add, edit, archive, reorder).
- Delivery status monitoring & problem delivery reporting.
- Manual resolution of uncertain deliveries (force retry, mark sent, mark skipped).
- Test message rendering and dispatch without chain registration.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from database import (
    DeepLinkNamespace,
    EventDelivery,
    EventDeliveryResolution,
    EventRegistration,
    EventScenario,
    EventStep,
    User,
    create_event_delivery_force_retry,
    reserve_start_code,
)
from event_service import (
    NON_RECOVERABLE_STATUSES,
    RescheduleResult,
    render_event_template,
    reschedule_event,
)
from time_helpers import MSK, format_msk, to_msk, utc_now

log = logging.getLogger(__name__)

# Preset offsets for admin quick-buttons
PRESET_OFFSETS: list[tuple[int, str]] = [
    (-1440, "24 ч. до начала"),
    (-120, "2 ч. до начала"),
    (0, "В момент начала"),
    (120, "2 ч. после начала"),
]

EVENT_DELIVERY_STATUSES = [
    "pending",
    "claimed",
    "sending",
    "sent",
    "skipped",
    "blocked",
    "failed",
    "uncertain",
    "resolved_sent",
    "resolved_skipped",
]

PROBLEM_DELIVERY_STATUSES = ["uncertain", "failed", "blocked"]


@dataclass(frozen=True)
class ScenarioSummary:
    id: int
    code: str
    event_type: str
    title: str
    starts_at: datetime  # UTC
    starts_at_msk: str
    duration_minutes: int
    place: str | None
    link: str | None
    is_active: bool
    is_archived: bool
    schedule_revision: int
    content_version: int
    registrations_count: int


@dataclass(frozen=True)
class StepItem:
    id: int
    event_id: int
    offset_minutes: int
    offset_display: str
    message_text: str
    step_version: int
    sort_order: int
    is_archived: bool


@dataclass(frozen=True)
class DeliveryStats:
    pending: int = 0
    claimed: int = 0
    sending: int = 0
    sent: int = 0
    skipped: int = 0
    blocked: int = 0
    failed: int = 0
    uncertain: int = 0
    resolved_sent: int = 0
    resolved_skipped: int = 0

    @property
    def total(self) -> int:
        return (
            self.pending
            + self.claimed
            + self.sending
            + self.sent
            + self.skipped
            + self.blocked
            + self.failed
            + self.uncertain
            + self.resolved_sent
            + self.resolved_skipped
        )

    @property
    def problematic_total(self) -> int:
        return self.uncertain + self.failed + self.blocked


@dataclass(frozen=True)
class DeliveryItem:
    id: int
    event_id: int
    user_id: int
    user_display: str
    message_kind: str
    step_id: int | None
    status: str
    due_at_msk: str
    attempt_count: int
    last_error: str | None
    sent_or_finished_msk: str | None
    platform: str
    retry_index: int
    parent_delivery_id: int | None


# ==============================================================================
# DATETIME & OFFSET HELPERS
# ==============================================================================

def parse_msk_datetime(raw: str) -> datetime:
    """
    Parses a date and time string entered in Moscow Time (MSK, UTC+3)
    and converts it to a naive UTC datetime for DB storage.

    Accepts formats:
    - 'DD.MM.YYYY HH:MM'
    - 'DD.MM.YY HH:MM'
    - 'YYYY-MM-DD HH:MM'
    """
    cleaned = raw.strip()
    patterns = [
        ("%d.%m.%Y %H:%M", True),
        ("%d.%m.%y %H:%M", True),
        ("%Y-%m-%d %H:%M", False),
    ]

    dt_obj: datetime | None = None
    for pattern, _ in patterns:
        try:
            dt_obj = datetime.strptime(cleaned, pattern)
            break
        except ValueError:
            continue

    if dt_obj is None:
        raise ValueError(
            "Неверный формат даты и времени. Используйте формат: ДД.ММ.ГГГГ ЧЧ:ММ "
            "(например: 25.10.2026 19:00)"
        )

    # Attach MSK timezone (+03:00)
    aware_msk = dt_obj.replace(tzinfo=MSK)
    # Convert to UTC and drop tzinfo for DB naive storage
    utc_dt = aware_msk.astimezone(timezone.utc).replace(tzinfo=None)
    return utc_dt


def format_msk_datetime(dt: datetime, fmt: str = "%d.%m.%Y %H:%M (МСК)") -> str:
    """Formats a UTC datetime as MSK string."""
    return format_msk(dt, fmt)


def parse_offset_minutes(raw: str) -> int:
    """
    Parses arbitrary offset string or presets into signed offset_minutes.

    Examples:
    - '0', '0m', 'в момент начала' -> 0
    - '-1440', '-24h', '-24 ч', '-24ч', '-1d', '-1 д' -> -1440
    - '-120', '-2h', '-2 ч', '-2ч' -> -120
    - '120', '+120', '+2h', '+2 ч', '2h', '2 ч' -> 120
    - '-30', '-30m', '-30 мин' -> -30
    - '30', '+30', '+30m', '30 мин' -> 30
    """
    s = raw.strip().lower()
    if s in {"0", "0m", "0 мин", "0м", "в момент", "в момент начала", "сейчас"}:
        return 0

    # Direct integer
    try:
        return int(s)
    except ValueError:
        pass

    # Regex for signed/unsigned value with unit
    match = re.match(r"^([+-]?\s*\d+)\s*(д|d|день|дня|дней|ч|h|час|часа|часов|м|m|мин|минут|минуты)?$", s)
    if not match:
        raise ValueError(
            "Не удалось распознать смещение. Примеры: '-24h', '-2h', '0', '+2h', '-30m', или число минут: '-120'."
        )

    val_str, unit = match.groups()
    val = int(val_str.replace(" ", ""))

    if not unit or unit in ("м", "m", "мин", "минут", "минуты"):
        return val
    elif unit in ("ч", "h", "час", "часа", "часов"):
        return val * 60
    elif unit in ("д", "d", "день", "дня", "дней"):
        return val * 1440
    else:
        raise ValueError(f"Неизвестная единица измерения времени '{unit}'")


def format_offset_minutes(offset: int) -> str:
    """Formats offset_minutes into human-friendly Russian text."""
    if offset == 0:
        return "В момент начала"
    if offset < 0:
        abs_val = abs(offset)
        if abs_val % 1440 == 0:
            days = abs_val // 1440
            d_word = "д." if days != 1 else "день"
            return f"За {days} {d_word} до начала"
        elif abs_val % 60 == 0:
            hours = abs_val // 60
            h_word = "ч." if hours != 1 else "час"
            return f"За {hours} {h_word} до начала"
        else:
            return f"За {abs_val} мин. до начала"
    else:
        if offset % 1440 == 0:
            days = offset // 1440
            d_word = "д." if days != 1 else "день"
            return f"Через {days} {d_word} после начала"
        elif offset % 60 == 0:
            hours = offset // 60
            h_word = "ч." if hours != 1 else "час"
            return f"Через {hours} {h_word} после начала"
        else:
            return f"Через {offset} мин. после начала"


# ==============================================================================
# INPUT VALIDATION
# ==============================================================================

def validate_event_code(code: str) -> str:
    """Validates unique scenario code according to system rules."""
    norm = code.strip().lower()
    if not norm:
        raise ValueError("Код сценария не может быть пустым.")
    if not re.match(r"^[a-z0-9_-]{3,64}$", norm):
        raise ValueError(
            "Код должен содержать от 3 до 64 символов (латинские буквы, цифры, дефис, подчеркивание)."
        )
    if norm in {"sub", "ref", "test", "menu", "start_message", "admin", "promo"}:
        raise ValueError(f"Код '{norm}' зарезервирован системой.")
    if any(norm.startswith(p) for p in ("topic_", "ref_", "ev_", "admin_")):
        raise ValueError(f"Код не может начинаться с системного префикса ('topic_', 'ref_', 'ev_', 'admin_').")
    return norm


def validate_scenario_input(
    *,
    code: str,
    event_type: str,
    title: str,
    starts_at_utc: datetime,
    duration_minutes: int,
    place: str | None,
    link: str | None,
    now_utc: datetime | None = None,
) -> None:
    """Pre-commit validation of scenario fields."""
    validate_event_code(code)

    if event_type not in ("webinar", "meeting"):
        raise ValueError("Тип сценария должен быть 'webinar' или 'meeting'.")

    title_clean = (title or "").strip()
    if not title_clean or len(title_clean) > 255:
        raise ValueError("Название сценария обязательно и должно содержать от 1 до 255 символов.")

    if not (15 <= duration_minutes <= 1440):
        raise ValueError("Продолжительность мероприятия должна быть от 15 до 1440 минут (24 ч).")

    link_clean = (link or "").strip() if link else None
    place_clean = (place or "").strip() if place else None

    if event_type == "webinar":
        if not link_clean:
            raise ValueError("Для вебинара обязательно укажите ссылку (link).")
        if place_clean:
            raise ValueError("Для вебинара поле места проведения (place) должно быть пустым.")
    elif event_type == "meeting":
        if not place_clean:
            raise ValueError("Для очной встречи обязательно укажите место проведения (place).")

    eff_now = now_utc or datetime.utcnow()
    end_time = starts_at_utc + timedelta(minutes=duration_minutes)
    if end_time <= eff_now:
        raise ValueError("Нельзя создать или перенести мероприятие на уже завершившееся время.")


def validate_step_input(offset_minutes: int, message_text: str) -> None:
    """Pre-commit validation of reminder step."""
    if not isinstance(offset_minutes, int):
        raise ValueError("Смещение должно быть целым числом минут.")
    text_clean = (message_text or "").strip()
    if not text_clean:
        raise ValueError("Текст напоминания не может быть пустым.")


# ==============================================================================
# SCENARIO CRUD OPERATIONS
# ==============================================================================

async def list_scenarios(
    session_maker: async_sessionmaker[AsyncSession],
    *,
    include_archived: bool = True,
) -> list[ScenarioSummary]:
    """Lists scenarios with registrations count, ordered by starts_at desc."""
    async with session_maker() as session:
        # Subquery for registration counts
        reg_subq = (
            select(
                EventRegistration.event_id,
                func.count(EventRegistration.id).label("reg_count"),
            )
            .group_by(EventRegistration.event_id)
            .subquery()
        )

        stmt = (
            select(EventScenario, func.coalesce(reg_subq.c.reg_count, 0))
            .outerjoin(reg_subq, EventScenario.id == reg_subq.c.event_id)
        )
        if not include_archived:
            stmt = stmt.where(EventScenario.is_archived == False)

        stmt = stmt.order_by(EventScenario.starts_at.desc(), EventScenario.id.desc())

        rows = (await session.execute(stmt)).all()
        result: list[ScenarioSummary] = []
        for scen, reg_count in rows:
            result.append(
                ScenarioSummary(
                    id=scen.id,
                    code=scen.code,
                    event_type=scen.event_type,
                    title=scen.title,
                    starts_at=scen.starts_at,
                    starts_at_msk=format_msk_datetime(scen.starts_at),
                    duration_minutes=scen.duration_minutes,
                    place=scen.place,
                    link=scen.link,
                    is_active=scen.is_active,
                    is_archived=scen.is_archived,
                    schedule_revision=scen.schedule_revision,
                    content_version=scen.content_version,
                    registrations_count=int(reg_count or 0),
                )
            )
        return result


async def get_scenario_summary(
    session_maker: async_sessionmaker[AsyncSession],
    event_id: int,
) -> ScenarioSummary | None:
    """Retrieves scenario summary by ID."""
    async with session_maker() as session:
        reg_count = await session.scalar(
            select(func.count(EventRegistration.id)).where(EventRegistration.event_id == event_id)
        )
        scen = await session.get(EventScenario, event_id)
        if not scen:
            return None
        return ScenarioSummary(
            id=scen.id,
            code=scen.code,
            event_type=scen.event_type,
            title=scen.title,
            starts_at=scen.starts_at,
            starts_at_msk=format_msk_datetime(scen.starts_at),
            duration_minutes=scen.duration_minutes,
            place=scen.place,
            link=scen.link,
            is_active=scen.is_active,
            is_archived=scen.is_archived,
            schedule_revision=scen.schedule_revision,
            content_version=scen.content_version,
            registrations_count=int(reg_count or 0),
        )


async def create_scenario(
    session_maker: async_sessionmaker[AsyncSession],
    *,
    code: str,
    event_type: str,
    title: str,
    starts_at_utc: datetime,
    duration_minutes: int = 120,
    place: str | None = None,
    link: str | None = None,
    is_active: bool = False,
) -> ScenarioSummary:
    """
    Creates a new scenario with deep-link reservation according to lock order:
    DeepLinkNamespace -> EventScenario.
    """
    validate_scenario_input(
        code=code,
        event_type=event_type,
        title=title,
        starts_at_utc=starts_at_utc,
        duration_minutes=duration_minutes,
        place=place,
        link=link,
    )
    norm_code = validate_event_code(code)

    async with session_maker() as session:
        bind = session.get_bind()
        is_postgres = getattr(getattr(bind, "dialect", None), "name", "") == "postgresql"

        if not is_postgres:
            await session.execute(text("BEGIN IMMEDIATE"))
        else:
            await session.begin()

        # 1. Level 1: Reserve deep link namespace first
        await reserve_start_code(session, norm_code, kind="event")

        # 2. Level 3: Insert EventScenario
        scen = EventScenario(
            code=norm_code,
            event_type=event_type,
            title=title.strip(),
            starts_at=starts_at_utc,
            duration_minutes=duration_minutes,
            place=place.strip() if place and place.strip() else None,
            link=link.strip() if link and link.strip() else None,
            is_active=is_active,
            is_archived=False,
            schedule_revision=1,
            content_version=1,
        )
        session.add(scen)
        await session.flush()

        # Update reference_id on DeepLinkNamespace
        ns_entry = await session.get(DeepLinkNamespace, norm_code)
        if ns_entry:
            ns_entry.reference_id = str(scen.id)

        await session.commit()

        return ScenarioSummary(
            id=scen.id,
            code=scen.code,
            event_type=scen.event_type,
            title=scen.title,
            starts_at=scen.starts_at,
            starts_at_msk=format_msk_datetime(scen.starts_at),
            duration_minutes=scen.duration_minutes,
            place=scen.place,
            link=scen.link,
            is_active=scen.is_active,
            is_archived=scen.is_archived,
            schedule_revision=scen.schedule_revision,
            content_version=scen.content_version,
            registrations_count=0,
        )


async def update_scenario_fields(
    session_maker: async_sessionmaker[AsyncSession],
    event_id: int,
    *,
    title: str | None = None,
    duration_minutes: int | None = None,
    place: str | None = None,
    link: str | None = None,
) -> ScenarioSummary:
    """Updates content fields of a scenario in a short locked transaction."""
    async with session_maker() as session:
        bind = session.get_bind()
        is_postgres = getattr(getattr(bind, "dialect", None), "name", "") == "postgresql"

        if not is_postgres:
            await session.execute(text("BEGIN IMMEDIATE"))
        else:
            await session.begin()

        stmt = select(EventScenario).where(EventScenario.id == event_id)
        if is_postgres:
            stmt = stmt.with_for_update()

        scen = await session.scalar(stmt)
        if not scen:
            raise ValueError(f"Сценарий {event_id} не найден.")
        if scen.is_archived:
            raise ValueError("Нельзя редактировать архивный сценарий.")

        new_title = title.strip() if title is not None else scen.title
        new_dur = duration_minutes if duration_minutes is not None else scen.duration_minutes
        new_place = place.strip() if place is not None else scen.place
        new_link = link.strip() if link is not None else scen.link

        validate_scenario_input(
            code=scen.code,
            event_type=scen.event_type,
            title=new_title,
            starts_at_utc=scen.starts_at,
            duration_minutes=new_dur,
            place=new_place,
            link=new_link,
        )

        scen.title = new_title
        scen.duration_minutes = new_dur
        scen.place = new_place if new_place else None
        scen.link = new_link if new_link else None
        scen.content_version += 1
        scen.updated_at = utc_now()

        await session.commit()
        return await get_scenario_summary(session_maker, event_id)  # type: ignore[return-value]


async def toggle_scenario_active(
    session_maker: async_sessionmaker[AsyncSession],
    event_id: int,
) -> bool:
    """Toggles active state of a scenario. Rejects activating archived scenario."""
    async with session_maker() as session:
        bind = session.get_bind()
        is_postgres = getattr(getattr(bind, "dialect", None), "name", "") == "postgresql"

        if not is_postgres:
            await session.execute(text("BEGIN IMMEDIATE"))
        else:
            await session.begin()

        stmt = select(EventScenario).where(EventScenario.id == event_id)
        if is_postgres:
            stmt = stmt.with_for_update()

        scen = await session.scalar(stmt)
        if not scen:
            raise ValueError(f"Сценарий {event_id} не найден.")
        if scen.is_archived:
            raise ValueError("Нельзя активировать архивный сценарий.")

        scen.is_active = not scen.is_active
        scen.updated_at = utc_now()
        new_val = scen.is_active

        await session.commit()
        return new_val


async def archive_scenario(
    session_maker: async_sessionmaker[AsyncSession],
    event_id: int,
) -> None:
    """Archives a scenario while preserving registrations, deliveries and audit."""
    async with session_maker() as session:
        bind = session.get_bind()
        is_postgres = getattr(getattr(bind, "dialect", None), "name", "") == "postgresql"

        if not is_postgres:
            await session.execute(text("BEGIN IMMEDIATE"))
        else:
            await session.begin()

        stmt = select(EventScenario).where(EventScenario.id == event_id)
        if is_postgres:
            stmt = stmt.with_for_update()

        scen = await session.scalar(stmt)
        if not scen:
            raise ValueError(f"Сценарий {event_id} не найден.")

        scen.is_archived = True
        scen.is_active = False
        scen.archived_at = utc_now()
        scen.updated_at = utc_now()

        await session.commit()


async def reschedule_scenario(
    session_maker: async_sessionmaker[AsyncSession],
    event_id: int,
    new_starts_at_utc: datetime,
) -> RescheduleResult:
    """
    Reschedules scenario using existing backend service contract:
    reschedule_event().
    """
    return await reschedule_event(session_maker, event_id, new_starts_at_utc)


async def copy_scenario(
    session_maker: async_sessionmaker[AsyncSession],
    source_event_id: int,
    new_code: str,
    new_starts_at_utc: datetime,
    new_title: str | None = None,
) -> ScenarioSummary:
    """
    Copies an existing scenario to a new upcoming event.

    Semantics:
    - Creates a new scenario with unique code and deep link reservation.
    - Copies title/type/duration/link/place.
    - Copies active reminder steps with their order, offset_minutes, message_text.
    - Sets new starts_at.
    - Leaves new scenario inactive (is_active=False).
    - STRICTLY DOES NOT clone registrations, deliveries, or resolutions.
    """
    norm_code = validate_event_code(new_code)

    async with session_maker() as session:
        bind = session.get_bind()
        is_postgres = getattr(getattr(bind, "dialect", None), "name", "") == "postgresql"

        if not is_postgres:
            await session.execute(text("BEGIN IMMEDIATE"))
        else:
            await session.begin()

        # Load source scenario and active steps
        source_stmt = (
            select(EventScenario)
            .options(selectinload(EventScenario.steps))
            .where(EventScenario.id == source_event_id)
        )
        source = await session.scalar(source_stmt)
        if not source:
            raise ValueError(f"Исходный сценарий {source_event_id} не найден.")

        target_title = new_title.strip() if new_title and new_title.strip() else source.title
        validate_scenario_input(
            code=norm_code,
            event_type=source.event_type,
            title=target_title,
            starts_at_utc=new_starts_at_utc,
            duration_minutes=source.duration_minutes,
            place=source.place,
            link=source.link,
        )

        # 1. Level 1: Reserve deep link namespace first
        await reserve_start_code(session, norm_code, kind="event")

        # 2. Level 3: Insert new EventScenario
        new_scen = EventScenario(
            code=norm_code,
            event_type=source.event_type,
            title=target_title,
            starts_at=new_starts_at_utc,
            duration_minutes=source.duration_minutes,
            place=source.place,
            link=source.link,
            is_active=False,
            is_archived=False,
            schedule_revision=1,
            content_version=1,
        )
        session.add(new_scen)
        await session.flush()

        # Update reference_id on DeepLinkNamespace
        ns_entry = await session.get(DeepLinkNamespace, norm_code)
        if ns_entry:
            ns_entry.reference_id = str(new_scen.id)

        # 3. Level 5: Clone active steps
        active_steps = sorted(
            [s for s in source.steps if not s.is_archived],
            key=lambda x: (x.sort_order, x.offset_minutes, x.id),
        )
        for s in active_steps:
            cloned_step = EventStep(
                event_id=new_scen.id,
                offset_minutes=s.offset_minutes,
                message_text=s.message_text,
                sort_order=s.sort_order,
                step_version=1,
                is_archived=False,
            )
            session.add(cloned_step)

        await session.commit()

        return ScenarioSummary(
            id=new_scen.id,
            code=new_scen.code,
            event_type=new_scen.event_type,
            title=new_scen.title,
            starts_at=new_scen.starts_at,
            starts_at_msk=format_msk_datetime(new_scen.starts_at),
            duration_minutes=new_scen.duration_minutes,
            place=new_scen.place,
            link=new_scen.link,
            is_active=new_scen.is_active,
            is_archived=new_scen.is_archived,
            schedule_revision=new_scen.schedule_revision,
            content_version=new_scen.content_version,
            registrations_count=0,
        )


# ==============================================================================
# REMINDER STEP OPERATIONS
# ==============================================================================

async def list_steps(
    session_maker: async_sessionmaker[AsyncSession],
    event_id: int,
    *,
    include_archived: bool = False,
) -> list[StepItem]:
    """Lists reminder steps for a scenario ordered by sort_order, offset_minutes, id."""
    async with session_maker() as session:
        stmt = (
            select(EventStep)
            .where(EventStep.event_id == event_id)
        )
        if not include_archived:
            stmt = stmt.where(EventStep.is_archived == False)

        stmt = stmt.order_by(EventStep.sort_order.asc(), EventStep.offset_minutes.asc(), EventStep.id.asc())
        steps = (await session.execute(stmt)).scalars().all()

        return [
            StepItem(
                id=s.id,
                event_id=s.event_id,
                offset_minutes=s.offset_minutes,
                offset_display=format_offset_minutes(s.offset_minutes),
                message_text=s.message_text,
                step_version=s.step_version,
                sort_order=s.sort_order,
                is_archived=s.is_archived,
            )
            for s in steps
        ]


async def get_step_item(
    session_maker: async_sessionmaker[AsyncSession],
    step_id: int,
) -> StepItem | None:
    """Retrieves single step item by ID."""
    async with session_maker() as session:
        s = await session.get(EventStep, step_id)
        if not s:
            return None
        return StepItem(
            id=s.id,
            event_id=s.event_id,
            offset_minutes=s.offset_minutes,
            offset_display=format_offset_minutes(s.offset_minutes),
            message_text=s.message_text,
            step_version=s.step_version,
            sort_order=s.sort_order,
            is_archived=s.is_archived,
        )


async def add_step(
    session_maker: async_sessionmaker[AsyncSession],
    event_id: int,
    offset_minutes: int,
    message_text: str,
) -> StepItem:
    """Adds a new reminder step to scenario."""
    validate_step_input(offset_minutes, message_text)

    async with session_maker() as session:
        bind = session.get_bind()
        is_postgres = getattr(getattr(bind, "dialect", None), "name", "") == "postgresql"

        if not is_postgres:
            await session.execute(text("BEGIN IMMEDIATE"))
        else:
            await session.begin()

        scen_stmt = select(EventScenario).where(EventScenario.id == event_id)
        if is_postgres:
            scen_stmt = scen_stmt.with_for_update()
        scen = await session.scalar(scen_stmt)
        if not scen:
            raise ValueError(f"Сценарий {event_id} не найден.")
        if scen.is_archived:
            raise ValueError("Нельзя добавлять шаги в архивный сценарий.")

        max_order = await session.scalar(
            select(func.coalesce(func.max(EventStep.sort_order), -1)).where(EventStep.event_id == event_id)
        )
        sort_order = int(max_order if max_order is not None else -1) + 1

        step = EventStep(
            event_id=event_id,
            offset_minutes=offset_minutes,
            message_text=message_text.strip(),
            sort_order=sort_order,
            step_version=1,
            is_archived=False,
        )
        session.add(step)
        scen.content_version += 1
        await session.commit()

        return StepItem(
            id=step.id,
            event_id=step.event_id,
            offset_minutes=step.offset_minutes,
            offset_display=format_offset_minutes(step.offset_minutes),
            message_text=step.message_text,
            step_version=step.step_version,
            sort_order=step.sort_order,
            is_archived=step.is_archived,
        )


async def update_step(
    session_maker: async_sessionmaker[AsyncSession],
    step_id: int,
    *,
    offset_minutes: int | None = None,
    message_text: str | None = None,
) -> StepItem:
    """
    Updates an existing reminder step.

    Semantics:
    - Increments step_version.
    - Increments scenario content_version.
    - If offset_minutes changed, recalculates due_at / next_attempt_at for future unsent
      pending deliveries.
    - Does NOT mutate Point of No Return deliveries (sent, sending, claimed, uncertain, terminal).
    """
    async with session_maker() as session:
        bind = session.get_bind()
        is_postgres = getattr(getattr(bind, "dialect", None), "name", "") == "postgresql"

        if not is_postgres:
            await session.execute(text("BEGIN IMMEDIATE"))
        else:
            await session.begin()

        step_stmt = select(EventStep).where(EventStep.id == step_id)
        if is_postgres:
            step_stmt = step_stmt.with_for_update()
        step = await session.scalar(step_stmt)
        if not step:
            raise ValueError(f"Шаг {step_id} не найден.")
        if step.is_archived:
            raise ValueError("Нельзя редактировать архивный шаг.")

        scen_stmt = select(EventScenario).where(EventScenario.id == step.event_id)
        if is_postgres:
            scen_stmt = scen_stmt.with_for_update()
        scen = await session.scalar(scen_stmt)
        if not scen:
            raise ValueError("Сценарий шага не найден.")
        if scen.is_archived:
            raise ValueError("Нельзя редактировать шаг архивного сценария.")

        new_offset = offset_minutes if offset_minutes is not None else step.offset_minutes
        new_text = message_text.strip() if message_text is not None else step.message_text
        validate_step_input(new_offset, new_text)

        offset_changed = (new_offset != step.offset_minutes)

        step.offset_minutes = new_offset
        step.message_text = new_text
        step.step_version += 1
        step.updated_at = utc_now()
        scen.content_version += 1
        scen.updated_at = utc_now()

        # If offset changed, recalculate due_at for pending deliveries of this step
        if offset_changed:
            new_due_at = scen.starts_at + timedelta(minutes=new_offset)
            pending_deliveries_stmt = (
                select(EventDelivery)
                .where(
                    EventDelivery.step_id == step.id,
                    EventDelivery.status == "pending",
                )
            )
            if is_postgres:
                pending_deliveries_stmt = pending_deliveries_stmt.with_for_update()
            pending_deliveries = (await session.execute(pending_deliveries_stmt)).scalars().all()
            for d in pending_deliveries:
                d.due_at = new_due_at
                d.next_attempt_at = new_due_at
                d.updated_at = utc_now()

        await session.commit()

        return StepItem(
            id=step.id,
            event_id=step.event_id,
            offset_minutes=step.offset_minutes,
            offset_display=format_offset_minutes(step.offset_minutes),
            message_text=step.message_text,
            step_version=step.step_version,
            sort_order=step.sort_order,
            is_archived=step.is_archived,
        )


async def archive_step(
    session_maker: async_sessionmaker[AsyncSession],
    step_id: int,
) -> None:
    """
    Archives a reminder step.
    Unsent pending deliveries for this step are safely skipped.
    """
    async with session_maker() as session:
        bind = session.get_bind()
        is_postgres = getattr(getattr(bind, "dialect", None), "name", "") == "postgresql"

        if not is_postgres:
            await session.execute(text("BEGIN IMMEDIATE"))
        else:
            await session.begin()

        step_stmt = select(EventStep).where(EventStep.id == step_id)
        if is_postgres:
            step_stmt = step_stmt.with_for_update()
        step = await session.scalar(step_stmt)
        if not step:
            raise ValueError(f"Шаг {step_id} не найден.")

        step.is_archived = True
        step.updated_at = utc_now()

        # Safely skip pending unsent deliveries
        pending_deliveries_stmt = (
            select(EventDelivery)
            .where(
                EventDelivery.step_id == step.id,
                EventDelivery.status == "pending",
            )
        )
        if is_postgres:
            pending_deliveries_stmt = pending_deliveries_stmt.with_for_update()
        pending_deliveries = (await session.execute(pending_deliveries_stmt)).scalars().all()
        now = utc_now()
        for d in pending_deliveries:
            d.status = "skipped"
            d.skip_reason = "step_archived"
            d.finished_at = now
            d.updated_at = now

        await session.commit()


async def reorder_steps(
    session_maker: async_sessionmaker[AsyncSession],
    event_id: int,
    ordered_step_ids: list[int],
) -> None:
    """Updates sort_order of steps according to ordered_step_ids."""
    async with session_maker() as session:
        bind = session.get_bind()
        is_postgres = getattr(getattr(bind, "dialect", None), "name", "") == "postgresql"

        if not is_postgres:
            await session.execute(text("BEGIN IMMEDIATE"))
        else:
            await session.begin()

        for idx, sid in enumerate(ordered_step_ids):
            step = await session.get(EventStep, sid)
            if step and step.event_id == event_id:
                step.sort_order = idx
                step.updated_at = utc_now()

        await session.commit()


# ==============================================================================
# MONITORING & DELIVERY STATUS
# ==============================================================================

async def get_scenario_delivery_stats(
    session_maker: async_sessionmaker[AsyncSession],
    event_id: int,
) -> DeliveryStats:
    """Aggregates delivery counts across all 10 possible statuses."""
    async with session_maker() as session:
        stmt = (
            select(EventDelivery.status, func.count(EventDelivery.id))
            .where(EventDelivery.event_id == event_id)
            .group_by(EventDelivery.status)
        )
        rows = (await session.execute(stmt)).all()
        counts: dict[str, int] = {status: count for status, count in rows}

        return DeliveryStats(
            pending=counts.get("pending", 0),
            claimed=counts.get("claimed", 0),
            sending=counts.get("sending", 0),
            sent=counts.get("sent", 0),
            skipped=counts.get("skipped", 0),
            blocked=counts.get("blocked", 0),
            failed=counts.get("failed", 0),
            uncertain=counts.get("uncertain", 0),
            resolved_sent=counts.get("resolved_sent", 0),
            resolved_skipped=counts.get("resolved_skipped", 0),
        )


async def list_problem_deliveries(
    session_maker: async_sessionmaker[AsyncSession],
    event_id: int,
    *,
    status_filter: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[DeliveryItem], int]:
    """
    Returns problem deliveries (uncertain, failed, blocked) for monitoring.
    Never returns internal claim_token to user-facing display.
    """
    async with session_maker() as session:
        base_stmt = select(EventDelivery).where(EventDelivery.event_id == event_id)
        if status_filter and status_filter in PROBLEM_DELIVERY_STATUSES:
            base_stmt = base_stmt.where(EventDelivery.status == status_filter)
        else:
            base_stmt = base_stmt.where(EventDelivery.status.in_(PROBLEM_DELIVERY_STATUSES))

        # Total count
        count_stmt = select(func.count()).select_from(base_stmt.subquery())
        total = int((await session.scalar(count_stmt)) or 0)

        # Query items
        stmt = (
            base_stmt
            .options(selectinload(EventDelivery.user))
            .order_by(EventDelivery.due_at.desc(), EventDelivery.id.desc())
            .limit(limit)
            .offset(offset)
        )
        deliveries = (await session.execute(stmt)).scalars().all()

        items: list[DeliveryItem] = []
        for d in deliveries:
            u = d.user
            user_display = (
                f"{u.first_name or u.name or ''} (id: {d.user_id})".strip()
                if u
                else f"Пользователь {d.user_id}"
            )
            finished_msk = (
                format_msk_datetime(d.finished_at)
                if d.finished_at
                else (format_msk_datetime(d.sent_at) if d.sent_at else None)
            )

            items.append(
                DeliveryItem(
                    id=d.id,
                    event_id=d.event_id,
                    user_id=d.user_id,
                    user_display=user_display,
                    message_kind=d.message_kind,
                    step_id=d.step_id,
                    status=d.status,
                    due_at_msk=format_msk_datetime(d.due_at),
                    attempt_count=d.attempt_count,
                    last_error=d.last_error,
                    sent_or_finished_msk=finished_msk,
                    platform=d.platform,
                    retry_index=d.retry_index,
                    parent_delivery_id=d.parent_delivery_id,
                )
            )

        return items, total


async def get_delivery_item(
    session_maker: async_sessionmaker[AsyncSession],
    delivery_id: int,
) -> DeliveryItem | None:
    """Loads a single delivery item for inspection and manual resolution."""
    async with session_maker() as session:
        stmt = (
            select(EventDelivery)
            .options(selectinload(EventDelivery.user))
            .where(EventDelivery.id == delivery_id)
        )
        d = await session.scalar(stmt)
        if not d:
            return None
        u = d.user
        user_display = (
            f"{u.first_name or u.name or ''} (id: {d.user_id})".strip()
            if u
            else f"Пользователь {d.user_id}"
        )
        finished_msk = (
            format_msk_datetime(d.finished_at)
            if d.finished_at
            else (format_msk_datetime(d.sent_at) if d.sent_at else None)
        )
        return DeliveryItem(
            id=d.id,
            event_id=d.event_id,
            user_id=d.user_id,
            user_display=user_display,
            message_kind=d.message_kind,
            step_id=d.step_id,
            status=d.status,
            due_at_msk=format_msk_datetime(d.due_at),
            attempt_count=d.attempt_count,
            last_error=d.last_error,
            sent_or_finished_msk=finished_msk,
            platform=d.platform,
            retry_index=d.retry_index,
            parent_delivery_id=d.parent_delivery_id,
        )


# ==============================================================================
# MANUAL RESOLUTION
# ==============================================================================

async def resolve_uncertain_delivery(
    session_maker: async_sessionmaker[AsyncSession],
    delivery_id: int,
    admin_id: int,
    resolution_type: str,
    resolution_reason: str,
) -> tuple[EventDelivery, bool]:
    """
    Manually resolves an uncertain event delivery.

    Resolution types:
    - 'force_retry': Uses existing create_event_delivery_force_retry backend. Idempotent.
    - 'mark_sent': Marks delivery as 'resolved_sent' and records audit resolution.
    - 'mark_skipped': Marks delivery as 'resolved_skipped' and records audit resolution.

    Invariants:
    - Strictly rejects resolving deliveries that are NOT in status 'uncertain'.
    - Rejects 'failed', 'sent', 'blocked', 'resolved_sent', 'resolved_skipped'.
    - Uses row locks and nested transaction.
    """
    if resolution_type not in ("force_retry", "mark_sent", "mark_skipped"):
        raise ValueError(f"Недопустимый тип резолюции: '{resolution_type}'")

    reason_clean = (resolution_reason or "").strip()
    if not reason_clean:
        raise ValueError("Причина резолюции обязательна.")

    async with session_maker() as session:
        bind = session.get_bind()
        is_postgres = getattr(getattr(bind, "dialect", None), "name", "") == "postgresql"

        if not is_postgres:
            await session.execute(text("BEGIN IMMEDIATE"))
        else:
            await session.begin()

        if resolution_type == "force_retry":
            # Delegate to existing backend helper
            res, created = await create_event_delivery_force_retry(
                session=session,
                delivery_id=delivery_id,
                admin_id=admin_id,
                resolution_reason=reason_clean,
                raise_on_duplicate=False,
            )
            await session.commit()
            return res, created

        # Handling 'mark_sent' and 'mark_skipped'
        stmt = select(EventDelivery).where(EventDelivery.id == delivery_id)
        if is_postgres:
            stmt = stmt.with_for_update()

        delivery = await session.scalar(stmt)
        if not delivery:
            raise ValueError(f"Доставка {delivery_id} не найдена.")

        if delivery.status != "uncertain":
            raise ValueError(
                f"Доставка {delivery_id} в статусе '{delivery.status}' не может быть разрешена вручную "
                "(ожидается статус 'uncertain')."
            )

        now = utc_now()
        target_status = "resolved_sent" if resolution_type == "mark_sent" else "resolved_skipped"
        delivery.status = target_status
        if resolution_type == "mark_skipped":
            delivery.skip_reason = "manual_resolution"
        delivery.finished_at = now
        delivery.updated_at = now

        resolution = EventDeliveryResolution(
            delivery_id=delivery.id,
            admin_id=admin_id,
            resolution_type=resolution_type,
            resolution_reason=reason_clean,
            resolved_at=now,
        )
        session.add(resolution)
        await session.commit()

        return delivery, True


# ==============================================================================
# TEST SEND TO SELF
# ==============================================================================

async def render_test_step(
    session_maker: async_sessionmaker[AsyncSession],
    event_id: int,
    step_id: int | None = None,
    admin_user: User | None = None,
) -> str:
    """
    Renders template for a scenario step using existing backend renderer.
    If step_id is None, renders a preview of the event confirmation message.
    """
    async with session_maker() as session:
        scen = await session.get(EventScenario, event_id)
        if not scen:
            raise ValueError(f"Сценарий {event_id} не найден.")

        if step_id is not None:
            step = await session.get(EventStep, step_id)
            if not step:
                raise ValueError(f"Шаг {step_id} не найден.")
            template = step.message_text
        else:
            # Default preview confirmation template
            template = (
                "✅ Вы зарегистрированы на «{event_title}»!\n\n"
                "📅 Дата: {event_date} в {event_time}\n"
                "📍 Место / Ссылка: {link_or_place}"
            )

        return render_event_template(scenario=scen, template=template, user=admin_user)
