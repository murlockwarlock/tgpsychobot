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
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from database import (
    DeepLinkNamespace,
    EventDelivery,
    EventDeliveryPart,
    EventDeliveryResolution,
    EventRegistration,
    EventScenario,
    EventScenarioMedia,
    EventStep,
    User,
    create_event_delivery_force_retry,
    reserve_start_code,
)
from event_service import (
    NON_RECOVERABLE_STATUSES,
    RescheduleResult,
    render_event_message,
    render_event_template,
    reschedule_event,
)
from max_messenger_bot.identity import is_max_user_id
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
_UNSET = object()


async def _stop_content_changed_deliveries(
    session: AsyncSession,
    event_id: int,
    now_utc: datetime,
    *,
    step_id: int | None = None,
    is_postgres: bool = False,
    preserve_duration_immediates: bool = False,
) -> None:
    conditions = [
        EventDelivery.event_id == event_id,
        EventDelivery.status.in_(("pending", "claimed", "sending")),
    ]
    if step_id is not None:
        conditions.append(EventDelivery.step_id == step_id)
    stale_stmt = select(EventDelivery).where(*conditions)
    if is_postgres:
        stale_stmt = stale_stmt.with_for_update()
    for delivery in (await session.execute(stale_stmt)).scalars().all():
        if (
            preserve_duration_immediates
            and delivery.message_kind in ("immediate_confirmation", "immediate_live", "reschedule_notice")
            and delivery.status in ("pending", "claimed")
        ):
            continue
        if delivery.status == "sending":
            delivery.status = "uncertain"
            delivery.skip_reason = "content_changed_partial"
            delivery.last_error = "Delivery interrupted after event content changed; an in-flight part may have been accepted."
        else:
            delivery.status = "skipped"
            delivery.skip_reason = "content_changed"
        delivery.finished_at = now_utc
        delivery.claim_token = None
        delivery.lease_until = None
        delivery.send_started_at = None
        delivery.updated_at = now_utc
        parts_stmt = select(EventDeliveryPart).where(
            EventDeliveryPart.delivery_id == delivery.id,
            EventDeliveryPart.status.in_(("pending", "sending")),
        )
        for part in (await session.execute(parts_stmt)).scalars().all():
            if delivery.status == "uncertain" and part.status == "sending":
                part.status = "uncertain"
                part.last_error = "In-flight part may have been accepted before event content changed."
            else:
                part.status = "skipped"
                part.last_error = "Stopped because event content changed."
            part.finished_at = now_utc
            part.claim_token = None
            part.lease_until = None
            part.updated_at = now_utc


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
    description: str | None = None
    media_order: str = "media_top"
    media_count: int = 0


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
    include_event_details: bool = False


@dataclass(frozen=True)
class ScenarioMediaItem:
    id: int
    event_id: int
    file_type: str
    file_id: str
    max_token: str | None
    source_platform: str | None
    sort_order: int


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


@dataclass(frozen=True)
class DeliveryPartItem:
    id: int
    part_index: int
    part_kind: str
    status: str
    attempt_count: int
    last_error: str | None
    external_message_id: str | None


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
    allow_ended: bool = False,
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
    if not allow_ended and end_time <= eff_now:
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
        media_subq = (
            select(
                EventScenarioMedia.event_id,
                func.count(EventScenarioMedia.id).label("media_count"),
            )
            .group_by(EventScenarioMedia.event_id)
            .subquery()
        )

        stmt = (
            select(EventScenario, func.coalesce(reg_subq.c.reg_count, 0), func.coalesce(media_subq.c.media_count, 0))
            .outerjoin(reg_subq, EventScenario.id == reg_subq.c.event_id)
            .outerjoin(media_subq, EventScenario.id == media_subq.c.event_id)
        )
        if not include_archived:
            stmt = stmt.where(EventScenario.is_archived == False)

        stmt = stmt.order_by(EventScenario.starts_at.desc(), EventScenario.id.desc())

        rows = (await session.execute(stmt)).all()
        result: list[ScenarioSummary] = []
        for scen, reg_count, media_count in rows:
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
                    description=scen.description,
                    media_order=scen.media_order,
                    media_count=int(media_count or 0),
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
        media_count = await session.scalar(
            select(func.count(EventScenarioMedia.id)).where(EventScenarioMedia.event_id == event_id)
        )
        scen = await session.scalar(
            select(EventScenario)
            .options(selectinload(EventScenario.media))
            .where(EventScenario.id == event_id)
        )
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
            description=scen.description,
            media_order=scen.media_order,
            media_count=int(media_count or 0),
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
    description: str | None = None,
    media_order: str = "media_top",
    is_active: bool = False,
    now_utc: datetime | None = None,
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
        now_utc=now_utc,
    )
    norm_code = validate_event_code(code)
    if media_order not in {"media_top", "text_top"}:
        raise ValueError("Порядок должен быть media_top или text_top.")
    eff_now = now_utc or utc_now()

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
            description=description.strip() if description and description.strip() else None,
            media_order=media_order,
            created_at=eff_now,
            updated_at=eff_now,
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
            description=scen.description,
            media_order=scen.media_order,
            media_count=0,
        )


async def list_scenario_media(
    session_maker: async_sessionmaker[AsyncSession],
    event_id: int,
) -> list[ScenarioMediaItem]:
    async with session_maker() as session:
        rows = (
            await session.execute(
                select(EventScenarioMedia)
                .where(EventScenarioMedia.event_id == event_id)
                .order_by(EventScenarioMedia.sort_order.asc(), EventScenarioMedia.id.asc())
            )
        ).scalars().all()
        return [
            ScenarioMediaItem(
                id=row.id,
                event_id=row.event_id,
                file_type=row.file_type,
                file_id=row.file_id,
                max_token=row.max_token,
                source_platform=row.source_platform,
                sort_order=row.sort_order,
            )
            for row in rows
        ]


async def replace_scenario_media(
    session_maker: async_sessionmaker[AsyncSession],
    event_id: int,
    attachments: list[dict[str, Any]],
    *,
    now_utc: datetime | None = None,
) -> list[ScenarioMediaItem]:
    if len(attachments) > 100:
        raise ValueError("Можно добавить не более 100 медиафайлов.")
    normalized: list[dict[str, Any]] = []
    for index, attachment in enumerate(attachments):
        file_type = str(attachment.get("file_type") or attachment.get("type") or "").lower()
        file_id = str(attachment.get("file_id") or "").strip()
        if file_type not in {"photo", "video"}:
            raise ValueError("Поддерживаются только фото и видео.")
        if not file_id:
            raise ValueError("У каждого медиафайла должен быть идентификатор исходного файла.")
        normalized.append({
            "file_type": file_type,
            "file_id": file_id,
            "max_token": str(attachment.get("max_token") or "").strip() or None,
            "source_platform": str(attachment.get("source_platform") or "").strip() or None,
            "source_file_id": str(attachment.get("source_file_id") or file_id).strip() or file_id,
            "sort_order": index,
        })

    eff_now = now_utc or utc_now()
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
        scenario = await session.scalar(stmt)
        if not scenario:
            raise ValueError(f"Сценарий {event_id} не найден.")
        if scenario.is_archived:
            raise ValueError("Нельзя редактировать архивный сценарий.")
        current = (
            await session.execute(
                select(EventScenarioMedia).where(EventScenarioMedia.event_id == event_id)
            )
        ).scalars().all()
        changed = len(current) != len(normalized) or any(
            row.file_type != item["file_type"]
            or row.file_id != item["file_id"]
            or row.max_token != item["max_token"]
            or row.sort_order != item["sort_order"]
            for row, item in zip(sorted(current, key=lambda value: (value.sort_order, value.id)), normalized)
        )
        if changed:
            for row in current:
                await session.delete(row)
            await session.flush()
            session.add_all([
                EventScenarioMedia(event_id=event_id, created_at=eff_now, updated_at=eff_now, **item)
                for item in normalized
            ])
            scenario.content_version += 1
            scenario.updated_at = eff_now
            await _stop_content_changed_deliveries(
                session,
                event_id,
                eff_now,
                is_postgres=is_postgres,
            )
        await session.commit()
    return await list_scenario_media(session_maker, event_id)


async def set_step_event_details(
    session_maker: async_sessionmaker[AsyncSession],
    step_id: int,
    enabled: bool,
    *,
    now_utc: datetime | None = None,
) -> StepItem:
    eff_now = now_utc or utc_now()
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
        scenario = await session.get(EventScenario, step.event_id)
        if not scenario or scenario.is_archived or step.is_archived:
            raise ValueError("Нельзя редактировать архивный сценарий или шаг.")
        changed = bool(enabled) != bool(step.include_event_details)
        if changed:
            step.include_event_details = bool(enabled)
            step.step_version += 1
            scenario.content_version += 1
            step.updated_at = eff_now
            scenario.updated_at = eff_now
            await _stop_content_changed_deliveries(
                session,
                scenario.id,
                eff_now,
                step_id=step.id,
                is_postgres=is_postgres,
            )
        await session.commit()
    return await get_step_item(session_maker, step_id)  # type: ignore[return-value]


async def update_scenario_fields(
    session_maker: async_sessionmaker[AsyncSession],
    event_id: int,
    *,
    title: str | None = None,
    duration_minutes: int | None = None,
    place: str | None = None,
    link: str | None = None,
    description: str | None | object = _UNSET,
    media_order: str | object = _UNSET,
    now_utc: datetime | None = None,
) -> ScenarioSummary:
    """Updates content fields of a scenario in a short locked transaction, adjusting expiry for unsent immediate deliveries."""
    eff_now = now_utc or utc_now()
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
        new_description = (
            description.strip() if isinstance(description, str) and description.strip() else None
            if description is not _UNSET
            else scen.description
        )
        new_media_order = media_order if media_order is not _UNSET else scen.media_order
        if new_media_order not in {"media_top", "text_top"}:
            raise ValueError("Порядок должен быть media_top или text_top.")

        validate_scenario_input(
            code=scen.code,
            event_type=scen.event_type,
            title=new_title,
            starts_at_utc=scen.starts_at,
            duration_minutes=new_dur,
            place=new_place,
            link=new_link,
            now_utc=eff_now,
            allow_ended=True,
        )

        duration_changed = (new_dur != scen.duration_minutes)
        content_changed = (
            new_title != scen.title
            or duration_changed
            or new_place != scen.place
            or new_link != scen.link
            or new_description != scen.description
            or new_media_order != scen.media_order
        )

        scen.title = new_title
        scen.duration_minutes = new_dur
        scen.place = new_place if new_place else None
        scen.link = new_link if new_link else None
        scen.description = new_description
        scen.media_order = new_media_order
        if content_changed:
            scen.content_version += 1
        scen.updated_at = eff_now

        if content_changed:
            await _stop_content_changed_deliveries(
                session,
                event_id,
                eff_now,
                is_postgres=is_postgres,
                preserve_duration_immediates=duration_changed,
            )

        if duration_changed:
            new_end_time = scen.starts_at + timedelta(minutes=new_dur)
            imm_delivs_stmt = (
                select(EventDelivery)
                .join(EventRegistration, EventDelivery.registration_id == EventRegistration.id)
                .where(
                    EventDelivery.event_id == event_id,
                    EventDelivery.message_kind.in_(("immediate_confirmation", "immediate_live", "reschedule_notice")),
                    EventDelivery.status.in_(("pending", "claimed")),
                    EventDelivery.schedule_revision == scen.schedule_revision,
                    EventDelivery.registration_generation == EventRegistration.generation,
                    EventRegistration.status == "started",
                )
            )
            if is_postgres:
                imm_delivs_stmt = imm_delivs_stmt.with_for_update()
            imm_delivs = (await session.execute(imm_delivs_stmt)).scalars().all()
            for d in imm_delivs:
                d.content_version_snapshot = scen.content_version
                if new_end_time <= eff_now:
                    d.status = "skipped"
                    d.skip_reason = "expired"
                    d.finished_at = eff_now
                    d.claim_token = None
                    d.lease_until = None
                else:
                    d.expires_at = new_end_time
                d.updated_at = eff_now

        await session.commit()
        return await get_scenario_summary(session_maker, event_id)  # type: ignore[return-value]


async def toggle_scenario_active(
    session_maker: async_sessionmaker[AsyncSession],
    event_id: int,
    *,
    now_utc: datetime | None = None,
) -> bool:
    """Toggles active state of a scenario. Rejects activating archived scenario. Sweeps overdue deliveries."""
    eff_now = now_utc or utc_now()
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
        scen.updated_at = eff_now
        new_val = scen.is_active

        # Reconcile pending deliveries that were missed / overdue during pause:
        # Missed scheduled deliveries whose due_at has already passed must become terminal skipped,
        # so that when the scenario is reactivated, worker never blasts old reminders.
        past_delivs_stmt = (
            select(EventDelivery)
            .where(
                EventDelivery.event_id == event_id,
                EventDelivery.status == "pending",
                EventDelivery.due_at <= eff_now,
            )
        )
        if is_postgres:
            past_delivs_stmt = past_delivs_stmt.with_for_update()
        past_delivs = (await session.execute(past_delivs_stmt)).scalars().all()
        for d in past_delivs:
            d.status = "skipped"
            d.skip_reason = "disabled_past"
            d.finished_at = eff_now
            d.updated_at = eff_now

        await session.commit()
        return new_val


async def archive_scenario(
    session_maker: async_sessionmaker[AsyncSession],
    event_id: int,
    *,
    now_utc: datetime | None = None,
) -> None:
    """Archives a scenario while safely skipping unsent deliveries and preserving sent history and audit."""
    eff_now = now_utc or utc_now()
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
        scen.archived_at = eff_now
        scen.updated_at = eff_now

        # Safe unsent pending deliveries become terminal skipped
        pending_stmt = (
            select(EventDelivery)
            .where(
                EventDelivery.event_id == event_id,
                EventDelivery.status == "pending",
            )
        )
        if is_postgres:
            pending_stmt = pending_stmt.with_for_update()
        pending_delivs = (await session.execute(pending_stmt)).scalars().all()
        for d in pending_delivs:
            d.status = "skipped"
            d.skip_reason = "scenario_archived"
            d.finished_at = eff_now
            d.updated_at = eff_now

        # Safe un-started claimed deliveries with expired lease also become skipped
        claimed_expired_stmt = (
            select(EventDelivery)
            .where(
                EventDelivery.event_id == event_id,
                EventDelivery.status == "claimed",
                EventDelivery.lease_until <= eff_now,
            )
        )
        if is_postgres:
            claimed_expired_stmt = claimed_expired_stmt.with_for_update()
        claimed_delivs = (await session.execute(claimed_expired_stmt)).scalars().all()
        for d in claimed_delivs:
            d.status = "skipped"
            d.skip_reason = "scenario_archived"
            d.finished_at = eff_now
            d.claim_token = None
            d.lease_until = None
            d.updated_at = eff_now

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
            .options(selectinload(EventScenario.steps), selectinload(EventScenario.media))
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
            description=source.description,
            media_order=source.media_order,
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
                include_event_details=s.include_event_details,
            )
            session.add(cloned_step)

        source_media = sorted(source.media, key=lambda media: (media.sort_order, media.id))
        for media in source_media:
            session.add(EventScenarioMedia(
                event_id=new_scen.id,
                file_type=media.file_type,
                file_id=media.file_id,
                max_token=media.max_token,
                source_platform=media.source_platform,
                source_file_id=media.source_file_id,
                sort_order=media.sort_order,
            ))

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
            description=new_scen.description,
            media_order=new_scen.media_order,
            media_count=len(source_media),
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
                include_event_details=s.include_event_details,
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
            include_event_details=s.include_event_details,
        )


async def add_step(
    session_maker: async_sessionmaker[AsyncSession],
    event_id: int,
    offset_minutes: int,
    message_text: str,
    *,
    include_event_details: bool = False,
    now_utc: datetime | None = None,
) -> StepItem:
    """Adds a new reminder step to scenario and schedules deliveries for current active registrations."""
    validate_step_input(offset_minutes, message_text)
    eff_now = now_utc or utc_now()

    async with session_maker() as session:
        bind = session.get_bind()
        is_postgres = getattr(getattr(bind, "dialect", None), "name", "") == "postgresql"

        if not is_postgres:
            await session.execute(text("BEGIN IMMEDIATE"))
        else:
            await session.begin()

        # Level 3: Scenario FOR UPDATE
        scen_stmt = select(EventScenario).where(EventScenario.id == event_id)
        if is_postgres:
            scen_stmt = scen_stmt.with_for_update()
        scen = await session.scalar(scen_stmt)
        if not scen:
            raise ValueError(f"Сценарий {event_id} не найден.")
        if scen.is_archived:
            raise ValueError("Нельзя добавлять шаги в архивный сценарий.")

        # Level 4: Active Registrations in stable order
        active_regs_stmt = (
            select(EventRegistration)
            .where(
                EventRegistration.event_id == event_id,
                EventRegistration.status == "started",
            )
            .order_by(EventRegistration.id.asc())
        )
        if is_postgres:
            active_regs_stmt = active_regs_stmt.with_for_update(read=True)
        active_regs = (await session.execute(active_regs_stmt)).scalars().all()

        max_order = await session.scalar(
            select(func.coalesce(func.max(EventStep.sort_order), -1)).where(EventStep.event_id == event_id)
        )
        sort_order = int(max_order if max_order is not None else -1) + 1

        # Level 5: Step
        step = EventStep(
            event_id=event_id,
            offset_minutes=offset_minutes,
            message_text=message_text.strip(),
            sort_order=sort_order,
            step_version=1,
            is_archived=False,
            include_event_details=include_event_details,
        )
        session.add(step)
        await session.flush()  # populate step.id

        # Level 6: Deliveries for each active registration
        step_due = scen.starts_at + timedelta(minutes=offset_minutes)
        new_content_version = scen.content_version + 1
        for reg in active_regs:
            step_key = f"ev:{scen.id}:reg:{reg.id}:step:{step.id}:r0"
            if reg.registration_mode == "urgent_late":
                step_status = "skipped"
                step_skip_reason = "late_mode_suppressed"
            else:
                if step_due > eff_now:
                    step_status = "pending"
                    step_skip_reason = None
                else:
                    step_status = "skipped"
                    step_skip_reason = "late_registration"

            reg_platform = "max" if is_max_user_id(reg.user_id) else "telegram"
            session.add(
                EventDelivery(
                    event_id=scen.id,
                    user_id=reg.user_id,
                    registration_id=reg.id,
                    step_id=step.id,
                    retry_index=0,
                    message_kind="scheduled_step",
                    delivery_key=step_key,
                    due_at=step_due,
                    next_attempt_at=max(eff_now, step_due),
                    schedule_revision=scen.schedule_revision,
                    registration_generation=reg.generation,
                    platform=reg_platform,
                    content_version_snapshot=new_content_version,
                    status=step_status,
                    skip_reason=step_skip_reason,
                )
            )

        scen.content_version = new_content_version
        scen.updated_at = eff_now
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
            include_event_details=step.include_event_details,
        )


async def update_step(
    session_maker: async_sessionmaker[AsyncSession],
    step_id: int,
    *,
    offset_minutes: int | None = None,
    message_text: str | None = None,
    include_event_details: bool | None = None,
    now_utc: datetime | None = None,
) -> StepItem:
    """
    Updates an existing reminder step.

    Semantics:
    - Lock order: Scenario -> Step -> Deliveries.
    - Increments step_version and scenario content_version.
    - If offset_minutes changed:
      - recalculated due in the future: safe pending/claimed deliveries remain/revert to pending with new due and cleared claim.
      - recalculated due in the past: safe pending/claimed deliveries become terminal skipped with skip_reason='step_rescheduled_past'.
    - Does NOT mutate Point of No Return deliveries (sent, sending, uncertain, blocked, failed, terminal).
    """
    eff_now = now_utc or utc_now()
    async with session_maker() as session:
        bind = session.get_bind()
        is_postgres = getattr(getattr(bind, "dialect", None), "name", "") == "postgresql"

        if not is_postgres:
            await session.execute(text("BEGIN IMMEDIATE"))
        else:
            await session.begin()

        # Step 0: Non-locking preliminary read of event_id
        target_event_id = await session.scalar(
            select(EventStep.event_id).where(EventStep.id == step_id)
        )
        if target_event_id is None:
            raise ValueError(f"Шаг {step_id} не найден.")

        # Level 3: Scenario FOR UPDATE
        scen_stmt = select(EventScenario).where(EventScenario.id == target_event_id)
        if is_postgres:
            scen_stmt = scen_stmt.with_for_update()
        scen = await session.scalar(scen_stmt)
        if not scen:
            raise ValueError("Сценарий шага не найден.")
        if scen.is_archived:
            raise ValueError("Нельзя редактировать шаг архивного сценария.")

        # Level 5: Step FOR UPDATE
        step_stmt = select(EventStep).where(EventStep.id == step_id)
        if is_postgres:
            step_stmt = step_stmt.with_for_update()
        step = await session.scalar(step_stmt)
        if not step:
            raise ValueError(f"Шаг {step_id} не найден.")
        if step.is_archived:
            raise ValueError("Нельзя редактировать архивный шаг.")

        new_offset = offset_minutes if offset_minutes is not None else step.offset_minutes
        new_text = message_text.strip() if message_text is not None else step.message_text
        new_include_event_details = (
            include_event_details if include_event_details is not None else step.include_event_details
        )
        validate_step_input(new_offset, new_text)

        offset_changed = (new_offset != step.offset_minutes)
        text_changed = new_text != step.message_text or bool(new_include_event_details) != bool(step.include_event_details)
        content_changed = offset_changed or text_changed
        new_content_version = scen.content_version + (1 if content_changed else 0)

        step.offset_minutes = new_offset
        step.message_text = new_text
        step.include_event_details = bool(new_include_event_details)
        if content_changed:
            step.step_version += 1
            step.updated_at = eff_now
            scen.content_version = new_content_version
            scen.updated_at = eff_now

        # Level 6: If offset changed, update/skip safe pending and claimed deliveries of this step
        if offset_changed:
            new_due_at = scen.starts_at + timedelta(minutes=new_offset)
            safe_deliveries_stmt = (
                select(EventDelivery)
                .where(
                    EventDelivery.step_id == step.id,
                    EventDelivery.status.in_(("pending", "claimed")),
                )
                .order_by(EventDelivery.id.asc())
            )
            if is_postgres:
                safe_deliveries_stmt = safe_deliveries_stmt.with_for_update()
            safe_deliveries = (await session.execute(safe_deliveries_stmt)).scalars().all()
            for d in safe_deliveries:
                d.content_version_snapshot = new_content_version
                if new_due_at > eff_now:
                    d.status = "pending"
                    d.due_at = new_due_at
                    d.next_attempt_at = new_due_at
                    d.claim_token = None
                    d.claimed_at = None
                    d.lease_until = None
                    d.send_started_at = None
                    d.skip_reason = None
                    d.finished_at = None
                else:
                    d.status = "skipped"
                    d.skip_reason = "step_rescheduled_past"
                    d.finished_at = eff_now
                    d.claim_token = None
                    d.claimed_at = None
                    d.lease_until = None
                    d.send_started_at = None
                d.updated_at = eff_now

        if text_changed:
            await _stop_content_changed_deliveries(
                session,
                scen.id,
                eff_now,
                step_id=step.id,
                is_postgres=is_postgres,
            )

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
            include_event_details=step.include_event_details,
        )


async def archive_step(
    session_maker: async_sessionmaker[AsyncSession],
    step_id: int,
    *,
    now_utc: datetime | None = None,
) -> None:
    """
    Archives a reminder step.
    Unsent pending deliveries for this step are safely skipped.
    """
    eff_now = now_utc or utc_now()
    async with session_maker() as session:
        bind = session.get_bind()
        is_postgres = getattr(getattr(bind, "dialect", None), "name", "") == "postgresql"

        if not is_postgres:
            await session.execute(text("BEGIN IMMEDIATE"))
        else:
            await session.begin()

        # Preliminary non-locking read
        target_event_id = await session.scalar(
            select(EventStep.event_id).where(EventStep.id == step_id)
        )
        if target_event_id is None:
            raise ValueError(f"Шаг {step_id} не найден.")

        # Level 3: Scenario lock
        scen_stmt = select(EventScenario).where(EventScenario.id == target_event_id)
        if is_postgres:
            scen_stmt = scen_stmt.with_for_update()
        scen = await session.scalar(scen_stmt)
        if not scen:
            raise ValueError("Сценарий шага не найден.")
        if scen.is_archived:
            raise ValueError("Нельзя архивировать шаг архивного сценария.")

        # Level 5: Step lock
        step_stmt = select(EventStep).where(EventStep.id == step_id)
        if is_postgres:
            step_stmt = step_stmt.with_for_update()
        step = await session.scalar(step_stmt)
        if not step:
            raise ValueError(f"Шаг {step_id} не найден.")

        step.is_archived = True
        step.updated_at = eff_now

        # Level 6: Deliveries lock
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
            d.status = "skipped"
            d.skip_reason = "step_archived"
            d.finished_at = eff_now
            d.updated_at = eff_now

        scen.content_version += 1
        scen.updated_at = eff_now
        await session.commit()


async def reorder_steps(
    session_maker: async_sessionmaker[AsyncSession],
    event_id: int,
    ordered_step_ids: list[int],
) -> None:
    """Updates sort_order of steps according to ordered_step_ids safely under lock."""
    if len(ordered_step_ids) != len(set(ordered_step_ids)):
        raise ValueError("Список ID шагов содержит дубликаты.")

    async with session_maker() as session:
        bind = session.get_bind()
        is_postgres = getattr(getattr(bind, "dialect", None), "name", "") == "postgresql"

        if not is_postgres:
            await session.execute(text("BEGIN IMMEDIATE"))
        else:
            await session.begin()

        # Level 3: Scenario lock
        scen_stmt = select(EventScenario).where(EventScenario.id == event_id)
        if is_postgres:
            scen_stmt = scen_stmt.with_for_update()
        scen = await session.scalar(scen_stmt)
        if not scen:
            raise ValueError(f"Сценарий {event_id} не найден.")
        if scen.is_archived:
            raise ValueError("Нельзя менять порядок шагов в архивном сценарии.")

        # Level 5: Target active Steps locked in stable order
        steps_stmt = (
            select(EventStep)
            .where(
                EventStep.event_id == event_id,
                EventStep.is_archived.is_(False),
            )
            .order_by(EventStep.id.asc())
        )
        if is_postgres:
            steps_stmt = steps_stmt.with_for_update()
        active_steps = (await session.execute(steps_stmt)).scalars().all()
        step_map = {s.id: s for s in active_steps}

        if set(ordered_step_ids) != set(step_map.keys()):
            raise ValueError("Передан неполный или некорректный список шагов для изменения порядка.")

        now = utc_now()
        for idx, sid in enumerate(ordered_step_ids):
            step_map[sid].sort_order = idx
            step_map[sid].updated_at = now

        scen.updated_at = now
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


async def list_delivery_parts(
    session_maker: async_sessionmaker[AsyncSession],
    delivery_id: int,
) -> list[DeliveryPartItem]:
    async with session_maker() as session:
        parts = (
            await session.execute(
                select(EventDeliveryPart)
                .where(EventDeliveryPart.delivery_id == delivery_id)
                .order_by(EventDeliveryPart.part_index.asc())
            )
        ).scalars().all()
        return [
            DeliveryPartItem(
                id=part.id,
                part_index=part.part_index,
                part_kind=part.part_kind,
                status=part.status,
                attempt_count=part.attempt_count,
                last_error=part.last_error,
                external_message_id=part.external_message_id,
            )
            for part in parts
        ]


# ==============================================================================
# MANUAL RESOLUTION
# ==============================================================================

async def resolve_uncertain_delivery(
    session_maker: async_sessionmaker[AsyncSession],
    delivery_id: int,
    admin_id: int,
    resolution_type: str,
    resolution_reason: str,
    part_id: int | None = None,
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
            try:
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
            except IntegrityError:
                await session.rollback()
                async with session_maker() as s2:
                    child_stmt = select(EventDelivery).where(EventDelivery.parent_delivery_id == delivery_id)
                    existing_child = (await s2.execute(child_stmt)).scalars().first()
                    if existing_child:
                        return existing_child, False
                raise

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
        part_status = target_status
        part_stmt = (
            select(EventDeliveryPart)
            .where(EventDeliveryPart.delivery_id == delivery.id)
            .order_by(EventDeliveryPart.part_index.asc())
        )
        if is_postgres:
            part_stmt = part_stmt.with_for_update()
        parts = (await session.execute(part_stmt)).scalars().all()
        uncertain_parts = [part for part in parts if part.status == "uncertain"]
        if uncertain_parts:
            target_part = next((part for part in uncertain_parts if part.id == part_id), None) if part_id else uncertain_parts[0]
            if target_part is None:
                raise ValueError(f"Часть доставки {part_id} не находится в статусе uncertain.")
            target_part.status = part_status
            target_part.finished_at = now
            target_part.sent_at = now if resolution_type == "mark_sent" else target_part.sent_at
            target_part.claim_token = None
            target_part.lease_until = None
            target_part.send_started_at = None
            target_part.updated_at = now

            remaining_uncertain = any(part.status == "uncertain" for part in parts)
            remaining_sending = any(part.status == "sending" for part in parts)
            remaining_active = any(part.status in {"pending", "sending", "uncertain"} for part in parts)
            delivery.claim_token = None
            delivery.lease_until = None
            delivery.claimed_at = None
            delivery.send_started_at = None
            delivery.updated_at = now
            if remaining_uncertain or remaining_sending:
                delivery.status = "uncertain"
                delivery.finished_at = None
                delivery.skip_reason = None
                delivery.last_error = "Часть доставки разрешена вручную; ожидаются остальные uncertain-части."
            elif remaining_active:
                delivery.status = "pending"
                delivery.finished_at = None
                delivery.skip_reason = None
                delivery.next_attempt_at = now
                delivery.last_error = "Часть доставки разрешена вручную; оставшиеся части возобновлены."
            else:
                delivery.status = target_status
                delivery.finished_at = now
                delivery.skip_reason = "manual_resolution" if resolution_type == "mark_skipped" else None
        else:
            if parts or part_id is not None:
                raise ValueError("У uncertain-доставки нет разрешаемой uncertain-части.")
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

        include_details = step_id is None or bool(step and step.include_event_details)
        return render_event_message(
            scenario=scen,
            template=template,
            user=admin_user,
            include_event_details=include_details,
        )
