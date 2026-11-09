"""
MAX Messenger Bot Admin Service for Event Reminder Chains.

Provides:
- Scenario management (Webinar / Meeting list, card, CRUD, active toggle, archive).
- Rescheduling with affected registration counts.
- Scenario copy with unique code reservation.
- Reminder step CRUD, presets and arbitrary offsets, reordering.
- Delivery monitoring and problematic deliveries review.
- Manual resolution of uncertain deliveries (force retry, mark sent, mark skipped).
- Test message send to self via MAX client.
"""

from __future__ import annotations

import html
import logging
import asyncio
from functools import wraps
from datetime import datetime
from typing import Any
from urllib.parse import quote

from event_admin_service import (
    EVENT_TITLE_MAX_LENGTH,
    EVENT_BUTTON_TITLE_PREVIEW_CHARS,
    compact_event_title,
    PRESET_OFFSETS,
    PROBLEM_DELIVERY_STATUSES,
    DeliveryItem,
    DeliveryStats,
    ScenarioSummary,
    StepItem,
    add_step,
    archive_scenario,
    archive_step,
    copy_scenario,
    create_scenario,
    format_msk_datetime,
    format_offset_minutes,
    get_delivery_item,
    get_scenario_delivery_stats,
    get_scenario_summary,
    get_step_item,
    list_problem_deliveries,
    list_delivery_parts,
    list_scenario_media,
    list_scenarios,
    list_steps,
    parse_msk_datetime,
    parse_offset_minutes,
    reorder_steps,
    reschedule_scenario,
    resolve_uncertain_delivery,
    set_step_event_details,
    toggle_scenario_active,
    update_scenario_fields,
    replace_scenario_media,
    update_step,
    validate_event_code,
    validate_scenario_input,
    validate_step_input,
)
from admin_html_preview import truncate_html_preview
from event_content_validator import (
    ALLOWED_EVENT_HTML_TAGS,
    MEDIA_FORMAT_CAPTION,
    MEDIA_FORMAT_LEGACY_AUTO,
    MEDIA_FORMAT_SEPARATE_MEDIA_TOP,
    MEDIA_FORMAT_SEPARATE_TEXT_TOP,
    EventContentValidationError,
    calculate_available_description_chars,
    format_event_html_title,
    parse_visible_telegram_text,
    render_plain_text_for_max,
    strip_telegram_custom_emoji,
)
from event_delivery_worker import build_test_delivery_plan
from ..api import MaxApiClient
from ..keyboards import callback_button, inline_keyboard
from ..legacy import User, async_session_maker
from ..storage import StateStore

log = logging.getLogger(__name__)


def _format_display_name(fmt: str) -> str:
    if fmt == MEDIA_FORMAT_CAPTION:
        return "💬 Фото/видео с текстом (подпись)"
    if fmt == MEDIA_FORMAT_SEPARATE_TEXT_TOP:
        return "📝 Текст отдельно, медиа ниже"
    if fmt == MEDIA_FORMAT_SEPARATE_MEDIA_TOP:
        return "🖼️ Текст отдельно, медиа выше"
    return "Стандартный (авто)"

MAX_EVENT_INPUT_STATES = frozenset({
    "max_event_create_code",
    "max_event_create_title",
    "max_event_create_starts_at",
    "max_event_create_duration",
    "max_event_create_link",
    "max_event_create_place",
    "max_event_create_meeting_link",
    "max_event_edit_title",
    "max_event_edit_duration",
    "max_event_edit_link",
    "max_event_edit_place",
    "max_event_edit_description",
    "max_event_edit_media",
    "max_event_edit_media_add",
    "max_event_reschedule_starts_at",
    "max_event_copy_code",
    "max_event_copy_starts_at",
    "max_event_step_offset_custom",
    "max_event_step_text",
    "max_event_step_edit_offset_custom",
    "max_event_step_edit_text",
    "max_event_resolve_reason",
})

_EVENT_DRAFT_KEYS = frozenset({
    "description_draft",
    "draft_title",
    "draft_format",
})

_EVENT_MUTATION_LOCKS: dict[int, asyncio.Lock] = {}


def _event_user_lock(func):
    @wraps(func)
    async def wrapped(*args, **kwargs):
        user_id = kwargs.get("user_id")
        if user_id is None and len(args) > 3:
            user_id = args[3]
        if user_id is None:
            return await func(*args, **kwargs)
        lock = _EVENT_MUTATION_LOCKS.setdefault(int(user_id), asyncio.Lock())
        async with lock:
            return await func(*args, **kwargs)

    return wrapped


def has_active_event_draft(snapshot: Any | None) -> bool:
    if snapshot is None or not isinstance(getattr(snapshot, "data", None), dict):
        return False
    data = snapshot.data
    if data.get("event_id") is None:
        return False
    if data.get("draft_title") is not None or data.get("draft_format") is not None:
        return True
    if data.get("description_draft") is not None and data.get("description_input_started", True):
        return True
    if bool(data.get("media_draft_dirty")):
        return True
    return False


def has_foreign_event_draft(snapshot: Any | None, event_id: int) -> bool:
    return bool(
        has_active_event_draft(snapshot)
        and snapshot.data.get("event_id") != event_id
    )


async def reject_foreign_event_draft(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    event_id: int,
) -> bool:
    snapshot = await states.get(user_id)
    if has_foreign_event_draft(snapshot, event_id):
        await client.send_message(
            chat_id=chat_id,
            text="Это старое меню другого мероприятия. Сначала завершите или отмените текущее редактирование.",
        )
        return True
    return False

PAGE_SIZE = 6


def _event_open_callback(event_id: int, back: str) -> str:
    if back.startswith("admin_ev_page_"):
        parts = back.split("_")
        if len(parts) == 5:
            return f"admin_ev_open_{event_id}_{parts[3]}_{parts[4]}"
    return f"admin_ev_open_{event_id}"


def _event_edit_menu_callback(event_id: int, back: str) -> str:
    if back.startswith("admin_ev_page_"):
        parts = back.split("_")
        if len(parts) == 5:
            return f"admin_ev_edit_menu_{event_id}_{parts[3]}_{parts[4]}"
    return f"admin_ev_edit_menu_{event_id}"


def _event_steps_callback(event_id: int, back: str) -> str:
    if back.startswith("admin_ev_page_"):
        parts = back.split("_")
        if len(parts) == 5:
            return f"admin_ev_steps_{event_id}_{parts[3]}_{parts[4]}"
    return f"admin_ev_steps_{event_id}"


def _event_step_callback(step_id: int, back: str) -> str:
    if back.startswith("admin_ev_page_"):
        parts = back.split("_")
        if len(parts) == 5:
            return f"admin_ev_step_{step_id}_{parts[3]}_{parts[4]}"
    return f"admin_ev_step_{step_id}"


def _event_step_action_callback(action: str, step_id: int, back: str) -> str:
    prefix = "admin_ev_test_step" if action == "test" else f"admin_ev_step_{action}"
    if back.startswith("admin_ev_page_"):
        parts = back.split("_")
        if len(parts) == 5:
            return f"{prefix}_{step_id}_{parts[3]}_{parts[4]}"
    return f"{prefix}_{step_id}"


def _event_step_value_callback(action: str, step_id: int, value: int, back: str) -> str:
    prefix = f"admin_ev_step_{action}"
    if back.startswith("admin_ev_page_"):
        parts = back.split("_")
        if len(parts) == 5:
            return f"{prefix}_{step_id}_{value}_{parts[3]}_{parts[4]}"
    return f"{prefix}_{step_id}_{value}"


PROBLEMS_PAGE_SIZE = 5


def _back(payload: str) -> list[dict]:
    return [callback_button("⬅️ Назад", payload)]


async def validate_event_input_state(snapshot) -> bool:
    if snapshot is None or snapshot.state not in MAX_EVENT_INPUT_STATES:
        return True
    return True


# ==============================================================================
# SCENARIOS LIST & CARD
# ==============================================================================

async def show_scenarios(
    client: MaxApiClient,
    chat_id: int,
    page: int = 0,
    include_archived: bool = False,
    *,
    back: str = "admin_panel",
) -> None:
    scenarios = await list_scenarios(async_session_maker, include_archived=include_archived)
    total_count = len(scenarios)
    total_pages = max(1, (total_count + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))

    page_scenarios = scenarios[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]

    rows = []
    for s in page_scenarios:
        status_icon = "📦" if s.is_archived else ("🟢" if s.is_active else "⚪️")
        type_icon = "💻" if s.event_type == "webinar" else "📍"
        label = f"{status_icon} {type_icon} {compact_event_title(s.title, limit=EVENT_BUTTON_TITLE_PREVIEW_CHARS)} ({s.starts_at_msk[:10]})"
        rows.append([callback_button(label, f"admin_ev_open_{s.id}_{page}_{1 if include_archived else 0}")])

    # Pagination
    nav_row = []
    archive_flag = 1 if include_archived else 0
    if page > 0:
        nav_row.append(callback_button("⬅️ Назад", f"admin_ev_page_{page - 1}_{archive_flag}"))
    if page < total_pages - 1:
        nav_row.append(callback_button("Далее ➡️", f"admin_ev_page_{page + 1}_{archive_flag}"))
    if nav_row:
        rows.append(nav_row)

    # Actions
    rows.append([
        callback_button("➕ Вебинар", "admin_ev_create_webinar"),
        callback_button("➕ Встреча", "admin_ev_create_meeting"),
    ])

    archive_toggle_text = "📁 Скрыть архив" if include_archived else "📁 Показать архив"
    archive_toggle_cb = "admin_ev_toggle_archive_view_0" if include_archived else "admin_ev_toggle_archive_view_1"
    rows.append([callback_button(archive_toggle_text, archive_toggle_cb)])
    rows.append(_back(back))

    text_lines = [
        f"📅 <b>Мероприятия и цепочки напоминаний ({total_count})</b>\n",
        f"Страница {page + 1} из {total_pages}\n\n",
    ]
    if not page_scenarios:
        text_lines.append("Мероприятий пока нет. Создайте вебинар или очную встречу кнопками ниже.")
    else:
        for idx, s in enumerate(page_scenarios):
            status = "📦 В архиве" if s.is_archived else ("🟢 Активно" if s.is_active else "⚪️ Выключено")
            kind = "Вебинар" if s.event_type == "webinar" else "Очная встреча"
            text_lines.append(
                f"<b>{idx + 1 + page * PAGE_SIZE}. {format_event_html_title(compact_event_title(s.title), platform='max')}</b>\n"
                f"• Код: <code>{s.code}</code> | Тип: {kind}\n"
                f"• Начало: {s.starts_at_msk}\n"
                f"• Статус: {status} | Регистраций: <b>{s.registrations_count}</b>\n\n"
            )

    await client.send_message(
        chat_id=chat_id,
        text="".join(text_lines),
        attachments=inline_keyboard(rows),
    )


async def show_scenario(
    client: MaxApiClient,
    chat_id: int,
    event_id: int,
    *,
    back: str = "admin_events",
    notice: str | None = None,
) -> None:
    s = await get_scenario_summary(async_session_maker, event_id)
    if not s:
        await client.send_message(chat_id=chat_id, text="Мероприятие не найдено.")
        return

    status_str = "📦 В архиве" if s.is_archived else ("🟢 Активно (запись открыта)" if s.is_active else "⚪️ Выключено (запись на паузе)")
    kind_str = "Вебинар" if s.event_type == "webinar" else "Очная встреча"

    loc_lines = []
    if s.event_type == "webinar":
        loc_lines.append(f"• <b>Ссылка:</b> {html.escape(s.link or '—')}")
    else:
        loc_lines.append(f"• <b>Место:</b> {html.escape(s.place or '—')}")
        if s.link:
            loc_lines.append(f"• <b>Ссылка:</b> {html.escape(s.link)}")

    loc_block = "\n".join(loc_lines)

    bot_name = getattr(client, "bot_name", None)
    if isinstance(bot_name, str) and bot_name:
        reg_link = f"https://max.ru/{quote(bot_name)}?start={s.code}"
        link_block = (
            f"• <b>Ссылка для регистрации:</b> <code>{reg_link}</code>\n"
            f"• <b>Код события:</b> <code>{s.code}</code>\n"
        )
    else:
        link_block = f"• <b>Код события:</b> <code>{s.code}</code>\n"

    warning_block = (
        f"⚠️ <b>Внимание:</b> {html.escape(s.content_warning)}\n"
        if not s.is_content_valid and s.content_warning
        else ""
    )

    text = (
        f"📅 <b>Карточка мероприятия: {format_event_html_title(s.title, platform='max')}</b>\n\n"
        f"{link_block}"
        f"• <b>Тип:</b> {kind_str}\n"
        f"• <b>Дата и время начала:</b> {s.starts_at_msk}\n"
        f"• <b>Длительность:</b> {s.duration_minutes} мин.\n"
        f"{loc_block}\n"
        f"• <b>Статус:</b> {status_str}\n"
        f"• <b>Регистраций:</b> <b>{s.registrations_count}</b>\n"
        f"• <b>Ревизия:</b> {s.schedule_revision} (контент v{s.content_version})\n"
        f"• <b>Описание:</b> {html.escape((s.description or '—')[:160])}{'…' if s.description and len(s.description) > 160 else ''}\n"
        f"• <b>Медиа:</b> {s.media_count} | формат: {_format_display_name(s.media_format)}\n"
        f"{warning_block}"
    )
    if notice:
        text = f"{text}\n{notice}"

    rows = []
    if not s.is_archived:
        toggle_label = "⚪️ Выключить запись" if s.is_active else "🟢 Включить запись"
        rows.append([callback_button(toggle_label, f"admin_ev_toggle_{s.id}")])

    rows.append([callback_button("📝 Цепочка сообщений (шаги)", _event_steps_callback(s.id, back))])
    rows.append([callback_button("📊 Статус отправок", f"admin_ev_stats_{s.id}")])

    if not s.is_archived:
        rows.append([callback_button("⏰ Перенести время", f"admin_ev_reschedule_{s.id}")])
        if back.startswith("admin_ev_page_"):
            back_parts = back.split("_")
            edit_callback = f"admin_ev_edit_menu_{s.id}_{back_parts[3]}_{back_parts[4]}" if len(back_parts) == 5 else f"admin_ev_edit_menu_{s.id}"
        else:
            edit_callback = f"admin_ev_edit_menu_{s.id}"
        rows.append([callback_button("✏️ Редактировать", edit_callback)])
        rows.append([callback_button("📋 Скопировать", f"admin_ev_copy_{s.id}")])
        rows.append([callback_button("🚀 Тестовая отправка себе", f"admin_ev_test_send_menu_{s.id}")])
        rows.append([callback_button("📦 В архив", f"admin_ev_archive_ask_{s.id}")])
    else:
        rows.append([callback_button("📋 Скопировать в новое", f"admin_ev_copy_{s.id}")])
        rows.append([callback_button("🚀 Тестовая отправка себе", f"admin_ev_test_send_menu_{s.id}")])

    rows.append(_back(back))

    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=inline_keyboard(rows),
    )


async def toggle_scenario(
    client: MaxApiClient,
    chat_id: int,
    event_id: int,
    *,
    back: str = "admin_events",
) -> None:
    try:
        new_active = await toggle_scenario_active(async_session_maker, event_id)
        msg_text = "🟢 Регистрация включена!" if new_active else "⚪️ Регистрация приостановлена."
        await client.send_message(chat_id=chat_id, text=msg_text)
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")
        return

    await show_scenario(client, chat_id, event_id, back=back)


async def archive_scenario_ask(
    client: MaxApiClient,
    chat_id: int,
    event_id: int,
    *,
    back: str = "admin_events",
) -> None:
    s = await get_scenario_summary(async_session_maker, event_id)
    if not s:
        return
    rows = [
        [callback_button("✅ Да, архивировать", f"admin_ev_archive_yes_{event_id}")],
        _back(_event_open_callback(event_id, back)),
    ]
    await client.send_message(
        chat_id=chat_id,
        text=f"📦 Архивировать мероприятие «<b>{format_event_html_title(s.title, platform='max')}</b>»?\n\nВсе регистрации и история доставок сохранятся.",
        attachments=inline_keyboard(rows),
    )


async def archive_scenario_confirm(
    client: MaxApiClient,
    chat_id: int,
    event_id: int,
    *,
    back: str = "admin_events",
) -> None:
    try:
        await archive_scenario(async_session_maker, event_id)
        await client.send_message(chat_id=chat_id, text="✅ Мероприятие архивировано.")
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")
        return

    await show_scenario(client, chat_id, event_id, back=back)


# ==============================================================================
# CREATE SCENARIO FLOW
# ==============================================================================

async def start_scenario_create(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    event_type: str,
) -> None:
    await states.set(
        user_id,
        chat_id,
        "max_event_create_code",
        {"create_event_type": event_type},
    )
    type_label = "вебинара" if event_type == "webinar" else "очной встречи"
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"➕ <b>Создание {type_label}</b>\n\n"
            f"<b>Шаг 1 из 5:</b> Введите уникальный код события для /start ссылки.\n\n"
            f"• Только латиница, цифры, дефис и подчеркивание (например: <code>webinar_ai</code>)\n"
            f"• Длина: от 3 до 64 символов\n"
            f"• Не используйте системные префиксы (<code>topic_</code>, <code>ref_</code>, <code>ev_</code>)"
        ),
        attachments=inline_keyboard([_back("admin_events")]),
    )


async def receive_create_code(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    raw_code: str,
) -> None:
    try:
        norm_code = validate_event_code(raw_code)
    except ValueError as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка в коде: {exc}\nПопробуйте ещё раз:")
        return

    await states.update(user_id, create_code=norm_code)
    await states.set(user_id, chat_id, "max_event_create_title", (await states.get(user_id)).data)
    await client.send_message(
        chat_id=chat_id,
        text=f"✅ Код принят: <code>{norm_code}</code>\n\n<b>Шаг 2 из 5:</b> Введите название мероприятия (до {EVENT_TITLE_MAX_LENGTH} символов):",
    )


async def receive_create_title(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    raw_title: str,
) -> None:
    title = raw_title.strip()
    vis_text, visible_len, _ = parse_visible_telegram_text(title)
    if not title or not vis_text.strip():
        await client.send_message(chat_id=chat_id, text="❌ Название мероприятия не может быть пустым.")
        return
    eff_visible_len = max(visible_len, len(vis_text))
    if eff_visible_len > EVENT_TITLE_MAX_LENGTH:
        await client.send_message(
            chat_id=chat_id,
            text=f"❌ Длина названия ({eff_visible_len} симв.) превышает максимально допустимую ({EVENT_TITLE_MAX_LENGTH} симв.).",
        )
        return
    if len(title) > EVENT_TITLE_MAX_LENGTH:
        await client.send_message(
            chat_id=chat_id,
            text=f"❌ Длина названия с учётом разметки и эмодзи ({len(title)} симв.) превышает лимит хранилища ({EVENT_TITLE_MAX_LENGTH} симв.). Сократите текст или форматирование.",
        )
        return

    state_obj = await states.get(user_id)
    data = state_obj.data if state_obj else {}
    data["create_title"] = title
    await states.set(user_id, chat_id, "max_event_create_starts_at", data)

    await client.send_message(
        chat_id=chat_id,
        text=(
            f"✅ Название: <b>{format_event_html_title(title, platform='max')}</b>\n\n"
            f"<b>Шаг 3 из 5:</b> Введите дату и время начала мероприятия по <b>Московскому времени (МСК)</b>.\n\n"
            f"Формат: <code>ДД.ММ.ГГГГ ЧЧ:ММ</code>\n"
            f"Пример: <code>25.10.2026 19:00</code>"
        ),
    )


async def receive_create_starts_at(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    raw_dt: str,
) -> None:
    try:
        utc_dt = parse_msk_datetime(raw_dt)
        if utc_dt <= datetime.utcnow():
            await client.send_message(chat_id=chat_id, text="❌ Мероприятие не может начинаться в прошлом. Введите будущую дату и время:")
            return
    except ValueError as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ {exc}\nПопробуйте ещё раз:")
        return

    state_obj = await states.get(user_id)
    data = state_obj.data if state_obj else {}
    data["create_starts_at"] = utc_dt.isoformat()
    data["create_starts_at_msk"] = raw_dt
    await states.set(user_id, chat_id, "max_event_create_duration", data)

    await client.send_message(
        chat_id=chat_id,
        text=(
            f"✅ Время начала: {raw_dt} (МСК)\n\n"
            f"<b>Шаг 4 из 5:</b> Введите продолжительность мероприятия в минутах (по умолчанию 120 мин, от 15 до 1440):"
        ),
    )


async def receive_create_duration(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    raw_dur: str,
) -> None:
    try:
        dur = int(raw_dur.strip())
        if not (15 <= dur <= 1440):
            raise ValueError()
    except Exception:
        await client.send_message(chat_id=chat_id, text="❌ Введите целое число минут от 15 до 1440 (например, 120):")
        return

    state_obj = await states.get(user_id)
    data = state_obj.data if state_obj else {}
    data["create_duration"] = dur
    event_type = data.get("create_event_type")

    if event_type == "webinar":
        await states.set(user_id, chat_id, "max_event_create_link", data)
        await client.send_message(
            chat_id=chat_id,
            text="<b>Шаг 5 из 5:</b> Введите ссылку на вебинар (Zoom, YouTube, Google Meet и т.д.):",
        )
    else:
        await states.set(user_id, chat_id, "max_event_create_place", data)
        await client.send_message(
            chat_id=chat_id,
            text="<b>Шаг 5 из 5:</b> Введите адрес / место проведения очной встречи:",
        )


async def receive_create_link(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    raw_link: str,
) -> None:
    link = raw_link.strip()
    if not link or len(link) < 3:
        await client.send_message(chat_id=chat_id, text="❌ Ссылка на вебинар обязательна. Введите корректную ссылку:")
        return

    state_obj = await states.get(user_id)
    data = state_obj.data if state_obj else {}
    await _finish_scenario_create(client, states, chat_id, user_id, data, place=None, link=link)


async def receive_create_place(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    raw_place: str,
) -> None:
    place = raw_place.strip()
    if not place or len(place) < 2:
        await client.send_message(chat_id=chat_id, text="❌ Место встречи обязательно. Введите адрес или название площадки:")
        return

    state_obj = await states.get(user_id)
    data = state_obj.data if state_obj else {}
    data["create_place"] = place
    await states.set(user_id, chat_id, "max_event_create_meeting_link", data)

    rows = [
        [callback_button("Пропустить ссылку", "admin_ev_skip_meeting_link")],
    ]
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"✅ Место встречи: <b>{html.escape(place)}</b>\n\n"
            f"Хотите добавить опциональную ссылку (на карту или сайт площадки)? "
            f"Отправьте ссылку сообщением или нажмите «Пропустить ссылку»:"
        ),
        attachments=inline_keyboard(rows),
    )


async def skip_meeting_link(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
) -> None:
    state_obj = await states.get(user_id)
    data = state_obj.data if state_obj else {}
    await _finish_scenario_create(client, states, chat_id, user_id, data, place=data.get("create_place"), link=None)


async def receive_create_meeting_link(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    raw_link: str,
) -> None:
    link = raw_link.strip()
    state_obj = await states.get(user_id)
    data = state_obj.data if state_obj else {}
    await _finish_scenario_create(client, states, chat_id, user_id, data, place=data.get("create_place"), link=link)


async def _finish_scenario_create(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    data: dict[str, Any],
    place: str | None,
    link: str | None,
) -> None:
    code = data["create_code"]
    event_type = data["create_event_type"]
    title = data["create_title"]
    starts_at_utc = datetime.fromisoformat(data["create_starts_at"])
    duration = data["create_duration"]

    try:
        scen = await create_scenario(
            async_session_maker,
            code=code,
            event_type=event_type,
            title=title,
            starts_at_utc=starts_at_utc,
            duration_minutes=duration,
            place=place,
            link=link,
            media_format=MEDIA_FORMAT_SEPARATE_MEDIA_TOP,
            is_active=False,
        )
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text="🎉 <b>Мероприятие успешно создано!</b>")
        await show_scenario(client, chat_id, scen.id)
    except Exception as exc:
        log.exception("MAX create scenario error: %s", exc)
        await client.send_message(
            chat_id=chat_id,
            text=f"❌ Ошибка создания: {exc}",
            attachments=inline_keyboard([_back("admin_events")]),
        )


# ==============================================================================
# EDIT SCENARIO FIELDS
# ==============================================================================

async def show_edit_menu(
    client: MaxApiClient,
    chat_id: int,
    event_id: int,
    *,
    back: str = "admin_events",
    states: StateStore | None = None,
    user_id: int | None = None,
) -> None:
    s = await get_scenario_summary(async_session_maker, event_id)
    if not s:
        return
    snapshot = await states.get(user_id) if (states and user_id is not None) else None
    if has_foreign_event_draft(snapshot, event_id):
        await client.send_message(
            chat_id=chat_id,
            text="Это старое меню другого мероприятия. Сначала завершите или отмените текущее редактирование.",
        )
        return
    if snapshot and snapshot.data.get("event_id") == event_id:
        back = snapshot.data.get("event_list_back", back)
    has_draft = bool(
        snapshot
        and snapshot.data.get("event_id") == event_id
        and (
            snapshot.data.get("draft_title") is not None
            or snapshot.data.get("draft_format") is not None
            or (snapshot.data.get("description_draft") is not None and snapshot.data.get("description_input_started", True))
            or bool(snapshot.data.get("media_draft_dirty"))
        )
    )
    cur_title = (snapshot.data.get("draft_title") or s.title) if (snapshot and snapshot.data.get("event_id") == event_id and snapshot.data.get("draft_title")) else s.title
    cur_fmt = (snapshot.data.get("draft_format") or s.media_format) if (snapshot and snapshot.data.get("event_id") == event_id and snapshot.data.get("draft_format")) else s.media_format

    rows = []
    if has_draft:
        rows.append([callback_button("💾 Сохранить изменения черновика", f"admin_ev_description_save_{s.id}")])
        rows.append([callback_button("❌ Отмена (сбросить черновик)", f"admin_ev_cancel_draft_{s.id}")])

    rows.extend([
        [callback_button("✏️ Изменить название", f"admin_ev_edit_title_{s.id}")],
        [callback_button("⏱ Изменить длительность", f"admin_ev_edit_duration_{s.id}")],
        [callback_button("📝 Описание", f"admin_ev_edit_description_{s.id}")],
        [callback_button("🖼 Медиа", f"admin_ev_edit_media_{s.id}")],
        [callback_button(
            f"⚙️ Формат: {_format_display_name(cur_fmt)}",
            f"admin_ev_format_menu_{s.id}",
        )],
        [callback_button(
            f"↕️ Порядок: {'медиа → текст' if s.media_order == 'media_top' else 'текст → медиа'}",
            f"admin_ev_toggle_media_order_{s.id}",
        )],
        [callback_button("👁 Предпросмотр Telegram", f"admin_ev_preview_tg_{s.id}")],
        [callback_button("👁 Предпросмотр MAX", f"admin_ev_preview_max_{s.id}")],
    ])
    if s.event_type == "webinar":
        rows.append([callback_button("🔗 Изменить ссылку на вебинар", f"admin_ev_edit_link_{s.id}")])
    else:
        rows.append([callback_button("📍 Изменить место встречи", f"admin_ev_edit_place_{s.id}")])
        link_label = "🔗 Изменить ссылку" if s.link else "🔗 Добавить ссылку"
        rows.append([callback_button(link_label, f"admin_ev_edit_link_{s.id}")])

    rows.append(_back(_event_open_callback(s.id, back)))
    draft_notice = "\n\n⚠️ <b>Есть несохранённый черновик.</b> Нажмите «Сохранить изменения черновика» для валидации и записи." if has_draft else ""
    await client.send_message(
        chat_id=chat_id,
        text=f"✏️ <b>Редактирование мероприятия: {format_event_html_title(cur_title, platform='max')}</b>{draft_notice}\n\nВыберите поле для изменения:",
        attachments=inline_keyboard(rows),
    )


async def show_format_menu(
    client: MaxApiClient,
    chat_id: int,
    event_id: int,
    *,
    back: str = "admin_events",
    states: StateStore | None = None,
    user_id: int | None = None,
) -> None:
    s = await get_scenario_summary(async_session_maker, event_id)
    if not s:
        return
    snapshot = await states.get(user_id) if (states and user_id) else None
    if has_foreign_event_draft(snapshot, event_id):
        await client.send_message(
            chat_id=chat_id,
            text="Это старое меню другого мероприятия. Сначала завершите или отмените текущее редактирование.",
        )
        return
    if snapshot and snapshot.data.get("event_id") == event_id:
        back = snapshot.data.get("event_list_back", back)
    has_same_event = bool(snapshot and snapshot.data.get("event_id") == event_id)
    current_fmt = (snapshot.data.get("draft_format") or s.media_format) if has_same_event else s.media_format
    back_target = _event_edit_menu_callback(event_id, back)
    rows = [
        [callback_button(
            f"{'🔘 ' if current_fmt == MEDIA_FORMAT_SEPARATE_MEDIA_TOP else '⚪️ '}🖼️ Текст отдельно, медиа выше",
            f"admin_ev_set_fmt_{event_id}_{MEDIA_FORMAT_SEPARATE_MEDIA_TOP}",
        )],
        [callback_button(
            f"{'🔘 ' if current_fmt == MEDIA_FORMAT_SEPARATE_TEXT_TOP else '⚪️ '}📝 Текст отдельно, медиа ниже",
            f"admin_ev_set_fmt_{event_id}_{MEDIA_FORMAT_SEPARATE_TEXT_TOP}",
        )],
        [callback_button(
            f"{'🔘 ' if current_fmt == MEDIA_FORMAT_CAPTION else '⚪️ '}💬 Фото/видео с текстом (подпись)",
            f"admin_ev_set_fmt_{event_id}_{MEDIA_FORMAT_CAPTION}",
        )],
        _back(back_target),
    ]
    title_text = format_event_html_title(snapshot.data.get("draft_title") or s.title, platform="max") if (has_same_event and snapshot.data.get("draft_title")) else format_event_html_title(s.title, platform="max")
    text = (
        f"⚙️ <b>Формат доставки для: {title_text}</b>\n\n"
        f"Текущий формат: <b>{_format_display_name(current_fmt)}</b>\n\n"
        f"<b>Доступные форматы:</b>\n"
        f"• <b>Текст отдельно, медиа выше:</b> отправка медиа, затем текст отдельным сообщением (до 3900 симв.).\n"
        f"• <b>Текст отдельно, медиа ниже:</b> сначала текст сообщения (до 3900 симв.), затем медиа.\n"
        f"• <b>Фото/видео с текстом (подпись):</b> одно медиа с подписью (до 900 симв. в TG, до 3900 в MAX).\n\n"
        f"Выберите формат:"
    )
    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=inline_keyboard(rows),
    )


async def set_format_handler(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    event_id: int,
    new_fmt: str,
) -> None:
    snapshot = await states.get(user_id)
    if has_foreign_event_draft(snapshot, event_id):
        await client.send_message(
            chat_id=chat_id,
            text="Это старое меню другого мероприятия. Сначала завершите или отмените текущее редактирование.",
        )
        return

    has_draft = bool(
        snapshot
        and snapshot.data.get("event_id") == event_id
        and (
            snapshot.data.get("draft_title") is not None
            or snapshot.data.get("draft_format") is not None
            or (snapshot.data.get("description_draft") is not None and snapshot.data.get("description_input_started", True))
            or bool(snapshot.data.get("media_draft_dirty"))
        )
    )
    list_back = snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events"

    if has_draft:
        snapshot.data["draft_format"] = new_fmt
        await states.set(user_id, chat_id, snapshot.state, snapshot.data)
        rows = [
            [callback_button("💾 Сохранить изменения черновика", f"admin_ev_description_save_{event_id}")],
            [callback_button("✏️ Редактировать текст", f"admin_ev_edit_description_{event_id}")],
            [callback_button("✏️ Изменить название", f"admin_ev_edit_title_{event_id}")],
            [callback_button("🖼 Управление медиа", f"admin_ev_edit_media_{event_id}")],
            [callback_button("⚙️ Выбрать другой формат", f"admin_ev_format_menu_{event_id}")],
            [callback_button("❌ Отмена (сбросить черновик)", f"admin_ev_cancel_draft_{event_id}")],
            _back(_event_edit_menu_callback(event_id, list_back)),
        ]
        desc_info = f"\n📝 Описание в черновике: {len(snapshot.data['description_draft'])} симв." if snapshot.data.get("description_draft") is not None else ""
        await client.send_message(
            chat_id=chat_id,
            text=(
                f"⚙️ Выбран формат: <b>{_format_display_name(new_fmt)}</b>{desc_info}\n\n"
                f"<i>Черновик ещё не сохранён в базу данных. Нажмите «Сохранить изменения черновика» для валидации и записи.</i>"
            ),
            attachments=inline_keyboard(rows),
        )
        return

    try:
        s = await update_scenario_fields(
            async_session_maker,
            event_id,
            media_format=new_fmt,
        )
        await client.send_message(chat_id=chat_id, text=f"✅ Формат изменён: {_format_display_name(s.media_format)}")
        await show_edit_menu(client, chat_id, event_id, back=list_back)
    except EventContentValidationError as err:
        rec = f"\n\n💡 Рекомендация: {err.recommendation}" if err.recommendation else ""
        await client.send_message(chat_id=chat_id, text=f"❌ Нельзя выбрать этот формат: {err.message}{rec}")
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")


async def show_telegram_preview(client: MaxApiClient, chat_id: int, event_id: int, user: Any = None, *, back: str = "admin_events") -> None:
    s = await get_scenario_summary(async_session_maker, event_id)
    if not s:
        return
    admin_user = user
    if isinstance(user, int):
        async with async_session_maker() as session:
            admin_user = await session.get(User, user)
    elif admin_user is None:
        async with async_session_maker() as session:
            admin_user = await session.get(User, 1)

    plan = await build_test_delivery_plan(
        async_session_maker,
        event_id=event_id,
        step_id=None,
        admin_user=admin_user,
    )
    msg_count = len(plan)
    order_lines = []
    first_sample = ""
    for idx, part in enumerate(plan, 1):
        kind = part.get("kind")
        if kind == "media_caption":
            cap = part.get("caption", "")
            first_sample = cap
            order_lines.append(f"{idx}. Сообщение с медиа и подписью ({len(cap)} симв.)")
        elif kind == "media":
            items = part.get("items") or []
            if len(items) == 1:
                order_lines.append(f"{idx}. Одно медиа-сообщение")
            else:
                order_lines.append(f"{idx}. Альбом из {len(items)} медиа")
        elif kind == "album":
            m_cnt = len(part.get("media", []))
            order_lines.append(f"{idx}. Альбом из {m_cnt} медиа")
        elif kind == "single_media":
            order_lines.append(f"{idx}. Одно медиа-сообщение")
        elif kind == "text":
            txt = part.get("text", "")
            if not first_sample:
                first_sample = txt
            order_lines.append(f"{idx}. Текстовое сообщение ({len(txt)} симв.)")
        elif kind == "control":
            order_lines.append(f"{idx}. Кнопка управления регистрацией")

    steps = await list_steps(async_session_maker, event_id, include_archived=False)
    available_budget = calculate_available_description_chars(
        s,
        media_format=s.media_format,
        steps=steps,
    )["conservative"]
    desc_len = len(s.description or "")

    preview_body = (
        f"👁 <b>Предпросмотр доставки (Telegram)</b>\n\n"
        f"• <b>Формат:</b> {_format_display_name(s.media_format)}\n"
        f"• <b>Сообщений к отправке:</b> {msg_count}\n"
        f"• <b>Порядок доставки:</b>\n" + "\n".join(f"  {od}" for od in order_lines) + "\n\n"
        f"• <b>Бюджет описания:</b> {desc_len} / {available_budget} симв. (остаток: {available_budget - desc_len})\n"
        f"• <b>Статус валидации:</b> {'✅ Пройдена' if s.is_content_valid else f'❌ Превышение ({s.content_warning})'}\n\n"
        f"<b>Пример текста первого сообщения:</b>\n"
        f"<blockquote>{truncate_html_preview(first_sample, 1200, allowed_tags=ALLOWED_EVENT_HTML_TAGS) or '—'}</blockquote>"
    )
    rows = [_back(_event_edit_menu_callback(event_id, back))]
    await client.send_message(
        chat_id=chat_id,
        text=preview_body,
        attachments=inline_keyboard(rows),
    )


async def show_max_preview(client: MaxApiClient, chat_id: int, event_id: int, user: Any = None, *, back: str = "admin_events") -> None:
    s = await get_scenario_summary(async_session_maker, event_id)
    if not s:
        return
    admin_user = user
    if isinstance(user, int):
        async with async_session_maker() as session:
            admin_user = await session.get(User, user)
    elif admin_user is None:
        async with async_session_maker() as session:
            admin_user = await session.get(User, 1)

    plan = await build_test_delivery_plan(
        async_session_maker,
        event_id=event_id,
        step_id=None,
        admin_user=admin_user,
    )
    msg_count = len(plan)
    order_lines = []
    first_sample = ""
    max_attach_per_msg = 0
    for idx, part in enumerate(plan, 1):
        kind = part.get("kind")
        if kind == "media_caption":
            cap = render_plain_text_for_max(part.get("caption", ""))
            first_sample = cap
            items_cnt = len(part.get("items") or [])
            attach_cnt = items_cnt + 1  # 1 media + 1 control keyboard
            max_attach_per_msg = max(max_attach_per_msg, attach_cnt)
            order_lines.append(f"{idx}. Сообщение с медиа и текстом ({len(cap)} симв., {attach_cnt} вложения)")
        elif kind == "media":
            items = part.get("items") or []
            attach_cnt = len(items)
            max_attach_per_msg = max(max_attach_per_msg, attach_cnt)
            if len(items) == 1:
                order_lines.append(f"{idx}. Одно медиа-сообщение (1 вложение)")
            else:
                order_lines.append(f"{idx}. Альбом из {len(items)} медиа ({len(items)} вложений)")
        elif kind == "album":
            m_cnt = len(part.get("media", []))
            max_attach_per_msg = max(max_attach_per_msg, m_cnt)
            order_lines.append(f"{idx}. Сообщение с {m_cnt} вложениями")
        elif kind == "single_media":
            max_attach_per_msg = max(max_attach_per_msg, 1)
            order_lines.append(f"{idx}. Сообщение с 1 медиа-вложением")
        elif kind == "text":
            txt = render_plain_text_for_max(part.get("text", ""))
            if not first_sample:
                first_sample = txt
            attach_cnt = 1  # control keyboard
            max_attach_per_msg = max(max_attach_per_msg, attach_cnt)
            order_lines.append(f"{idx}. Текст ({len(txt)} симв., {attach_cnt} вложение)")
        elif kind == "control":
            attach_cnt = 1
            max_attach_per_msg = max(max_attach_per_msg, attach_cnt)
            order_lines.append(f"{idx}. Кнопка управления (1 вложение)")

    plain_sample = html.escape(first_sample[:1200])

    preview_body = (
        f"👁 <b>Предпросмотр доставки (MAX)</b>\n\n"
        f"• <b>Формат:</b> {_format_display_name(s.media_format)}\n"
        f"• <b>Сообщений к отправке:</b> {msg_count}\n"
        f"• <b>Порядок доставки:</b>\n" + "\n".join(f"  {od}" for od in order_lines) + "\n\n"
        f"• <b>Ограничения MAX:</b>\n"
        f"  - Макс. вложений в сообщении: {max_attach_per_msg} / 12 ({'✅ в норме' if max_attach_per_msg <= 12 else '❌ превышение!'})\n"
        f"  - Лимит текста: до 3900 симв. ({'✅ в норме' if len(first_sample) <= 3900 else '❌ превышение!'})\n"
        f"  - Rate limit: 2 сообщ./сек в один диалог (очередь с интервалом 0.5с)\n\n"
        f"<b>Пример текста для MAX (plain text / без HTML):</b>\n"
        f"<blockquote>{plain_sample or '—'}</blockquote>"
    )
    rows = [_back(_event_edit_menu_callback(event_id, back))]
    await client.send_message(
        chat_id=chat_id,
        text=preview_body,
        attachments=inline_keyboard(rows),
    )


def _description_editor_keyboard(event_id: int, back: str = "admin_events") -> list[list[dict]]:
    return [
        [callback_button("✅ Сохранить и выйти", f"admin_ev_description_save_{event_id}")],
        [callback_button("🗑 Удалить описание", f"admin_ev_description_delete_{event_id}")],
        _back(_event_edit_menu_callback(event_id, back)),
    ]


async def start_edit_description(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, event_id: int) -> None:
    snapshot = await states.get(user_id)
    if has_foreign_event_draft(snapshot, event_id):
        await client.send_message(
            chat_id=chat_id,
            text="Это старое меню другого мероприятия. Сначала завершите или отмените текущее редактирование.",
        )
        return

    s = await get_scenario_summary(async_session_maker, event_id)
    if not s:
        return
    list_back = snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events"
    existing_draft = (
        snapshot.data.get("description_draft")
        if (snapshot and snapshot.data.get("event_id") == event_id and snapshot.data.get("description_draft") is not None)
        else (s.description or "")
    )
    draft_fmt = snapshot.data.get("draft_format") if snapshot and snapshot.data.get("event_id") == event_id else None
    draft_title = snapshot.data.get("draft_title") if snapshot and snapshot.data.get("event_id") == event_id else None
    new_data: dict[str, Any] = dict(snapshot.data) if snapshot and snapshot.data.get("event_id") == event_id else {}
    new_data.update({
        "event_id": event_id,
        "description_draft": existing_draft,
        "description_input_started": False,
        "event_list_back": list_back,
    })
    if draft_fmt:
        new_data["draft_format"] = draft_fmt
    if draft_title:
        new_data["draft_title"] = draft_title
    await states.set(user_id, chat_id, "max_event_edit_description", new_data)

    eff_fmt = draft_fmt or s.media_format
    steps = await list_steps(async_session_maker, event_id, include_archived=False)
    available_chars = calculate_available_description_chars(
        s,
        media_format=eff_fmt,
        steps=steps,
    )["conservative"]
    _, used_chars, _ = parse_visible_telegram_text(existing_draft)
    remaining = available_chars - used_chars
    status_line = (
        f"• Доступный остаток: <b>{remaining}</b> симв. (использовано {used_chars} из {available_chars})\n"
        if remaining >= 0
        else f"⚠️ <b>Внимание:</b> Лимит превышен на {abs(remaining)} симв.!\n"
    )

    display_title = format_event_html_title(draft_title or s.title, platform="max")
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"📝 <b>Описание мероприятия: {display_title}</b>\n\n"
            f"• Формат доставки: <b>{_format_display_name(eff_fmt)}</b>\n"
            f"{status_line}\n"
            "Отправьте текст сообщения. Каждое новое сообщение добавляется к черновику.\n\n"
            f"Текущий черновик:\n<blockquote>{truncate_html_preview(existing_draft, 2800, allowed_tags=ALLOWED_EVENT_HTML_TAGS) or '—'}</blockquote>"
        ),
        attachments=inline_keyboard(_description_editor_keyboard(event_id, back=list_back)),
    )


async def receive_edit_description(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, raw_text: str) -> None:
    snapshot = await states.get(user_id)
    if not snapshot:
        return
    event_id = snapshot.data["event_id"]
    previous = snapshot.data.get("description_draft", "") if snapshot.data.get("description_input_started") else ""
    combined = f"{previous}\n{raw_text}" if previous and raw_text else (previous or raw_text)
    snapshot.data["description_draft"] = combined
    snapshot.data["description_input_started"] = True
    await states.set(user_id, chat_id, snapshot.state, snapshot.data)

    s = await get_scenario_summary(async_session_maker, event_id)
    eff_fmt = snapshot.data.get("draft_format") or (s.media_format if s else MEDIA_FORMAT_SEPARATE_MEDIA_TOP)
    steps = await list_steps(async_session_maker, event_id, include_archived=False)
    available_chars = calculate_available_description_chars(
        s,
        media_format=eff_fmt,
        steps=steps,
    )["conservative"]
    _, used_chars, _ = parse_visible_telegram_text(combined)
    remaining = available_chars - used_chars
    status_line = (
        f"• Доступный остаток: <b>{remaining}</b> симв. (использовано {used_chars} из {available_chars})\n"
        if remaining >= 0
        else f"⚠️ <b>Внимание:</b> Превышение лимита на {abs(remaining)} симв.!\n"
    )

    list_back = snapshot.data.get("event_list_back", "admin_events")
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"📝 <b>Черновик обновлён</b> ({used_chars} симв.)\n"
            f"{status_line}\n"
            f"<blockquote>{truncate_html_preview(combined, 2800, allowed_tags=ALLOWED_EVENT_HTML_TAGS) or '—'}</blockquote>"
        ),
        attachments=inline_keyboard(_description_editor_keyboard(snapshot.data["event_id"], back=list_back)),
    )


@_event_user_lock
async def save_description(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    event_id: int | None = None,
) -> None:
    snapshot = await states.get(user_id)
    if not snapshot:
        await client.send_message(chat_id=chat_id, text="Черновик устарел или не найден.")
        return
    draft_event_id = snapshot.data.get("event_id")
    if event_id is not None and draft_event_id != event_id:
        await client.send_message(chat_id=chat_id, text="Черновик принадлежит другому мероприятию или устарел.")
        return
    event_id = draft_event_id
    if event_id is None:
        return
    draft_desc = snapshot.data.get("description_draft")
    draft_title = snapshot.data.get("draft_title")
    draft_fmt = snapshot.data.get("draft_format")
    list_back = snapshot.data.get("event_list_back", "admin_events")

    kwargs: dict[str, Any] = {}
    if draft_desc is not None and snapshot.data.get("description_input_started", True):
        kwargs["description"] = draft_desc
    if draft_title is not None:
        kwargs["title"] = draft_title
    if draft_fmt is not None:
        kwargs["media_format"] = draft_fmt
    if snapshot.data.get("media_draft_dirty") and isinstance(snapshot.data.get("media_draft"), list):
        kwargs["media_items"] = snapshot.data.get("media_draft")

    try:
        s = await update_scenario_fields(async_session_maker, event_id, **kwargs)
        await states.clear(user_id)
        await client.send_message(
            chat_id=chat_id,
            text=f"✅ Изменения мероприятия <b>{format_event_html_title(s.title, platform='max')}</b> сохранены.",
        )
        await show_scenario(
            client,
            chat_id,
            event_id,
            back=list_back,
            notice="✅ Изменения мероприятия сохранены.",
        )
    except EventContentValidationError as err:
        rec = f"\n\n💡 Рекомендация: {err.recommendation}" if err.recommendation else ""
        error_msg = (
            f"❌ <b>Невозможно сохранить изменения!</b>\n\n"
            f"• Платформа: <b>{err.platform.upper()}</b>\n"
            f"• Фактическая длина: <b>{err.actual_length}</b> симв. (лимит: {err.limit_length})\n"
            f"• Превышение: <b>{err.excess}</b> симв.\n"
            f"• Причина: {html.escape(err.message)}{rec}\n\n"
            f"<i>Черновик сохранён. Вы можете изменить текст или формат доставки.</i>"
        )
        rows = [
            [callback_button("✏️ Редактировать текст", f"admin_ev_edit_description_{event_id}")],
            [callback_button("✏️ Изменить название", f"admin_ev_edit_title_{event_id}")],
            [callback_button("⚙️ Изменить формат доставки", f"admin_ev_format_menu_{event_id}")],
            [callback_button("🖼 Управление медиа", f"admin_ev_edit_media_{event_id}")],
            [callback_button("💾 Повторить сохранение", f"admin_ev_description_save_{event_id}")],
            [callback_button("❌ Отмена (сбросить черновик)", f"admin_ev_cancel_draft_{event_id}")],
        ]
        await client.send_message(
            chat_id=chat_id,
            text=error_msg,
            attachments=inline_keyboard(rows),
        )
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")


@_event_user_lock
async def cancel_draft(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, event_id: int) -> None:
    snapshot = await states.get(user_id)
    if has_foreign_event_draft(snapshot, event_id):
        await client.send_message(chat_id=chat_id, text="Черновик принадлежит другому мероприятию или устарел.")
        return
    list_back = snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events"
    await states.clear(user_id)
    await client.send_message(
        chat_id=chat_id,
        text="<i>Черновик сброшен. Все несохранённые изменения отменены.</i>",
    )
    await show_scenario(
        client,
        chat_id,
        event_id,
        back=list_back,
        notice="<i>Черновик сброшен. Все несохранённые изменения отменены.</i>",
    )


async def delete_description(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, event_id: int) -> None:
    snapshot = await states.get(user_id)
    if has_foreign_event_draft(snapshot, event_id):
        await client.send_message(chat_id=chat_id, text="Черновик принадлежит другому мероприятию или устарел.")
        return
    list_back = snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events"
    new_data = dict(snapshot.data) if snapshot and snapshot.data.get("event_id") == event_id else {}
    new_data.update({
        "event_id": event_id,
        "description_draft": "",
        "description_input_started": True,
        "event_list_back": list_back,
    })
    await states.set(user_id, chat_id, "max_event_edit_description", new_data)
    await client.send_message(
        chat_id=chat_id,
        text="🗑 Описание очищено в черновике. Нажмите «Сохранить и выйти» для применения.",
        attachments=inline_keyboard(_description_editor_keyboard(event_id, back=list_back)),
    )


def _media_editor_keyboard(event_id: int, draft: list[dict[str, Any]], back: str = "admin_events") -> list[list[dict]]:
    rows: list[list[dict]] = []
    for index, item in enumerate(draft):
        row = [callback_button(f"🗑 {index + 1}. {'Фото' if item['file_type'] == 'photo' else 'Видео'}", f"admin_ev_media_remove_{event_id}_{index}")]
        if index > 0:
            row.append(callback_button("⬆️", f"admin_ev_media_up_{event_id}_{index}"))
        if index < len(draft) - 1:
            row.append(callback_button("⬇️", f"admin_ev_media_down_{event_id}_{index}"))
        rows.append(row)
    rows.extend([
        [callback_button("➕ Добавить фото/видео", f"admin_ev_media_add_{event_id}")],
        [callback_button("✅ Сохранить и выйти", f"admin_ev_media_save_{event_id}")],
        _back(_event_edit_menu_callback(event_id, back)),
    ])
    return rows


async def show_media_editor(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, event_id: int) -> None:
    snapshot = await states.get(user_id)
    if has_foreign_event_draft(snapshot, event_id):
        await client.send_message(chat_id=chat_id, text="Это старое меню другого мероприятия. Сначала завершите или отмените текущее редактирование.")
        return
    list_back = snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events"
    preserve_draft = snapshot and snapshot.state in {"max_event_edit_media", "max_event_edit_media_add"} and snapshot.data.get("event_id") == event_id and snapshot.data.get("media_draft_dirty") and isinstance(snapshot.data.get("media_draft"), list)
    if preserve_draft:
        draft = list(snapshot.data["media_draft"])
    else:
        media = await list_scenario_media(async_session_maker, event_id)
        draft = [
            {
                "file_type": item.file_type,
                "file_id": item.file_id,
                "max_token": item.max_token,
                "source_platform": item.source_platform,
                "source_file_id": item.source_file_id or item.file_id,
            }
            for item in media
        ]
    new_data = dict(snapshot.data) if snapshot and snapshot.data.get("event_id") == event_id else {}
    new_data.update({"event_id": event_id, "media_draft": draft, "media_draft_dirty": bool(preserve_draft), "event_list_back": list_back})
    await states.set(user_id, chat_id, "max_event_edit_media", new_data)
    await client.send_message(
        chat_id=chat_id,
        text=f"🖼 <b>Медиа мероприятия</b>\n\nВ черновике: <b>{len(draft)}</b>. Отправьте медиа после кнопки добавления и сохраните изменения.",
        attachments=inline_keyboard(_media_editor_keyboard(event_id, draft, back=list_back)),
    )


async def start_add_media(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, event_id: int) -> None:
    snapshot = await states.get(user_id)
    if has_foreign_event_draft(snapshot, event_id):
        await client.send_message(chat_id=chat_id, text="Это старое меню другого мероприятия. Сначала завершите или отмените текущее редактирование.")
        return
    draft = snapshot.data.get("media_draft", []) if snapshot else []
    new_data = dict(snapshot.data) if snapshot and snapshot.data.get("event_id") == event_id else {}
    new_data.update({"event_id": event_id, "media_draft": draft, "media_draft_dirty": snapshot.data.get("media_draft_dirty", False) if snapshot else False, "event_list_back": snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events"})
    await states.set(user_id, chat_id, "max_event_edit_media_add", new_data)
    await client.send_message(chat_id=chat_id, text="🖼 Отправьте фото или видео одним сообщением.", attachments=inline_keyboard([_back(f"admin_ev_edit_media_{event_id}")]))


async def receive_media(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, *, media_type: str | None, media_token: str | None) -> None:
    snapshot = await states.get(user_id)
    if not snapshot or not media_token or media_type not in {"photo", "image", "video"}:
        await client.send_message(chat_id=chat_id, text="❌ Отправьте фото или видео.")
        return
    draft = list(snapshot.data.get("media_draft", []))
    draft.append({
        "file_type": "photo" if media_type == "image" else media_type,
        "file_id": media_token,
        "max_token": media_token,
        "source_platform": "max",
        "source_file_id": media_token,
    })
    new_data = dict(snapshot.data)
    new_data.update({"event_id": snapshot.data["event_id"], "media_draft": draft, "media_draft_dirty": True, "event_list_back": snapshot.data.get("event_list_back", "admin_events")})
    await states.set(user_id, chat_id, "max_event_edit_media", new_data)
    list_back = snapshot.data.get("event_list_back", "admin_events")
    await client.send_message(chat_id=chat_id, text=f"✅ Медиа добавлено. В черновике: {len(draft)}.", attachments=inline_keyboard(_media_editor_keyboard(snapshot.data["event_id"], draft, back=list_back)))


async def mutate_media_draft(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    event_id: int,
    index: int,
    direction: int | None = None,
) -> None:
    snapshot = await states.get(user_id)
    if not snapshot:
        return
    if snapshot.data.get("event_id") != event_id:
        await client.send_message(chat_id=chat_id, text="Черновик принадлежит другому мероприятию или устарел.")
        return
    draft = list(snapshot.data.get("media_draft", []))
    if 0 <= index < len(draft):
        if direction is None:
            draft.pop(index)
        else:
            target = index + direction
            if 0 <= target < len(draft):
                draft[index], draft[target] = draft[target], draft[index]
    await states.update(user_id, media_draft=draft, media_draft_dirty=True)
    list_back = snapshot.data.get("event_list_back", "admin_events")
    await client.send_message(chat_id=chat_id, text=f"🖼 В черновике: {len(draft)}.", attachments=inline_keyboard(_media_editor_keyboard(event_id, draft, back=list_back)))


async def save_media(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, event_id: int) -> None:
    snapshot = await states.get(user_id)
    if not snapshot or snapshot.data.get("event_id") != event_id:
        await client.send_message(chat_id=chat_id, text="Черновик принадлежит другому мероприятию или устарел.")
        return
    draft = snapshot.data.get("media_draft", []) if snapshot else []
    list_back = snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events"
    draft_desc = snapshot.data.get("description_draft")
    draft_title = snapshot.data.get("draft_title")
    draft_fmt = snapshot.data.get("draft_format")

    kwargs: dict[str, Any] = {"media_items": draft}
    if draft_desc is not None and snapshot.data.get("description_input_started", True):
        kwargs["description"] = draft_desc
    if draft_title is not None:
        kwargs["title"] = draft_title
    if draft_fmt is not None:
        kwargs["media_format"] = draft_fmt

    try:
        s = await update_scenario_fields(async_session_maker, event_id, **kwargs)
        await states.clear(user_id)
        await client.send_message(
            chat_id=chat_id,
            text=f"✅ Медиа и изменения мероприятия <b>{format_event_html_title(s.title, platform='max')}</b> сохранены.",
            attachments=inline_keyboard([_back(_event_open_callback(event_id, list_back))]),
        )
    except EventContentValidationError as err:
        rec = f"\n\n💡 Рекомендация: {err.recommendation}" if err.recommendation else ""
        error_msg = (
            f"❌ <b>Невозможно сохранить изменения медиа!</b>\n\n"
            f"• Платформа: <b>{err.platform.upper()}</b>\n"
            f"• Фактическая длина: <b>{err.actual_length}</b> симв. (лимит: {err.limit_length})\n"
            f"• Превышение: <b>{err.excess}</b> симв.\n"
            f"• Причина: {html.escape(err.message)}{rec}\n\n"
            f"<i>Черновик сохранён. Вы можете изменить формат доставки, текст или количество медиа.</i>"
        )
        rows = [
            [callback_button("✏️ Редактировать текст", f"admin_ev_edit_description_{event_id}")],
            [callback_button("✏️ Изменить название", f"admin_ev_edit_title_{event_id}")],
            [callback_button("⚙️ Изменить формат доставки", f"admin_ev_format_menu_{event_id}")],
            [callback_button("🖼 Управление медиа", f"admin_ev_edit_media_{event_id}")],
            [callback_button("💾 Повторить сохранение медиа", f"admin_ev_media_save_{event_id}")],
            [callback_button("❌ Отмена (сбросить черновик)", f"admin_ev_cancel_draft_{event_id}")],
        ]
        await client.send_message(
            chat_id=chat_id,
            text=error_msg,
            attachments=inline_keyboard(rows),
        )
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")


async def toggle_media_order(client: MaxApiClient, chat_id: int, event_id: int, *, back: str = "admin_events") -> None:
    s = await get_scenario_summary(async_session_maker, event_id)
    if not s:
        return
    try:
        updated = await update_scenario_fields(async_session_maker, event_id, media_order="text_top" if s.media_order == "media_top" else "media_top")
        await show_edit_menu(client, chat_id, updated.id, back=back)
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")


async def start_edit_title(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, event_id: int) -> None:
    snapshot = await states.get(user_id)
    if has_foreign_event_draft(snapshot, event_id):
        await client.send_message(chat_id=chat_id, text="Это старое меню другого мероприятия. Сначала завершите или отмените текущее редактирование.")
        return
    list_back = snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events"
    draft_data = snapshot.data if (snapshot and snapshot.data.get("event_id") == event_id) else {}
    has_draft = bool(
        draft_data.get("description_draft") is not None
        or draft_data.get("draft_format") is not None
        or bool(draft_data.get("media_draft_dirty"))
    )
    new_data = dict(draft_data)
    new_data.update({"event_id": event_id, "event_list_back": list_back})
    await states.set(user_id, chat_id, "max_event_edit_title", new_data)
    back_cb = _event_edit_menu_callback(event_id, list_back) if has_draft else _event_open_callback(event_id, list_back)
    await client.send_message(
        chat_id=chat_id,
        text=f"✏️ Введите новое название мероприятия (до {EVENT_TITLE_MAX_LENGTH} символов):",
        attachments=inline_keyboard([_back(back_cb)]),
    )


async def receive_edit_title(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, raw_title: str) -> None:
    state_obj = await states.get(user_id)
    if not state_obj:
        return
    event_id = state_obj.data["event_id"]
    list_back = state_obj.data.get("event_list_back", "admin_events")
    title = raw_title.strip()
    vis_text, visible_len, _ = parse_visible_telegram_text(title)

    if not title or not vis_text.strip():
        await client.send_message(chat_id=chat_id, text="❌ Название мероприятия не может быть пустым.")
        return

    eff_visible_len = max(visible_len, len(vis_text))
    if eff_visible_len > EVENT_TITLE_MAX_LENGTH:
        await client.send_message(
            chat_id=chat_id,
            text=f"❌ Длина названия ({eff_visible_len} симв.) превышает максимально допустимую ({EVENT_TITLE_MAX_LENGTH} симв.).",
        )
        return

    if len(title) > EVENT_TITLE_MAX_LENGTH:
        await client.send_message(
            chat_id=chat_id,
            text=f"❌ Длина названия с учётом разметки и эмодзи ({len(title)} симв.) превышает лимит хранилища ({EVENT_TITLE_MAX_LENGTH} симв.). Сократите текст или форматирование.",
        )
        return

    has_active_draft = (
        state_obj.data.get("description_draft") is not None
        or state_obj.data.get("draft_format") is not None
        or bool(state_obj.data.get("media_draft_dirty"))
    )

    if has_active_draft:
        state_obj.data["draft_title"] = title
        await states.set(user_id, chat_id, state_obj.state, state_obj.data)
        rows = [
            [callback_button("💾 Сохранить изменения черновика", f"admin_ev_description_save_{event_id}")],
            [callback_button("✏️ Редактировать текст", f"admin_ev_edit_description_{event_id}")],
            [callback_button("⚙️ Изменить формат доставки", f"admin_ev_format_menu_{event_id}")],
            [callback_button("🖼 Управление медиа", f"admin_ev_edit_media_{event_id}")],
            [callback_button("❌ Отмена (сбросить черновик)", f"admin_ev_cancel_draft_{event_id}")],
            _back(_event_edit_menu_callback(event_id, list_back)),
        ]
        await client.send_message(
            chat_id=chat_id,
            text=(
                f"✏️ <b>Название в черновике обновлено!</b>\n\n"
                f"Новое название: <b>{format_event_html_title(title, platform='max')}</b>\n\n"
                f"<i>Черновик ещё не сохранён в базу данных. Нажмите «Сохранить изменения черновика» для валидации и записи.</i>"
            ),
            attachments=inline_keyboard(rows),
        )
        return

    try:
        scen = await update_scenario_fields(async_session_maker, event_id, title=title)
        await states.clear(user_id)
        await client.send_message(
            chat_id=chat_id,
            text=f"✅ Название обновлено: <b>{format_event_html_title(scen.title, platform='max')}</b>",
        )
        await show_scenario(client, chat_id, event_id, back=list_back)
    except EventContentValidationError as err:
        state_obj.data["draft_title"] = title
        await states.set(user_id, chat_id, state_obj.state, state_obj.data)
        rec = f"\n\n💡 Рекомендация: {err.recommendation}" if err.recommendation else ""
        error_msg = (
            f"❌ <b>Невозможно сохранить название!</b>\n\n"
            f"• Платформа: <b>{err.platform.upper()}</b>\n"
            f"• Фактическая длина: <b>{err.actual_length}</b> симв. (лимит: {err.limit_length})\n"
            f"• Превышение: <b>{err.excess}</b> симв.\n"
            f"• Причина: {html.escape(err.message)}{rec}\n\n"
            f"<i>Черновик сохранён. Вы можете изменить текст или формат доставки.</i>"
        )
        rows = [
            [callback_button("✏️ Редактировать текст", f"admin_ev_edit_description_{event_id}")],
            [callback_button("✏️ Изменить название", f"admin_ev_edit_title_{event_id}")],
            [callback_button("⚙️ Изменить формат доставки", f"admin_ev_format_menu_{event_id}")],
            [callback_button("🖼 Управление медиа", f"admin_ev_edit_media_{event_id}")],
            [callback_button("💾 Повторить сохранение", f"admin_ev_description_save_{event_id}")],
            [callback_button("❌ Отмена (сбросить черновик)", f"admin_ev_cancel_draft_{event_id}")],
        ]
        await client.send_message(
            chat_id=chat_id,
            text=error_msg,
            attachments=inline_keyboard(rows),
        )
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")


async def start_edit_duration(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, event_id: int) -> None:
    snapshot = await states.get(user_id)
    if has_foreign_event_draft(snapshot, event_id):
        await client.send_message(chat_id=chat_id, text="Это старое меню другого мероприятия. Сначала завершите или отмените текущее редактирование.")
        return
    list_back = snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events"
    await states.set(user_id, chat_id, "max_event_edit_duration", {"event_id": event_id, "event_list_back": list_back})
    await client.send_message(
        chat_id=chat_id,
        text="⏱ Введите новую длительность в минутах (от 15 до 1440):",
        attachments=inline_keyboard([_back(_event_open_callback(event_id, list_back))]),
    )


async def receive_edit_duration(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, raw_dur: str) -> None:
    state_obj = await states.get(user_id)
    event_id = state_obj.data["event_id"]
    list_back = state_obj.data.get("event_list_back", "admin_events")
    try:
        dur = int(raw_dur.strip())
        scen = await update_scenario_fields(async_session_maker, event_id, duration_minutes=dur)
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text=f"✅ Длительность обновлена: {scen.duration_minutes} мин.")
        await show_scenario(client, chat_id, event_id, back=list_back)
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")


async def start_edit_link(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, event_id: int) -> None:
    snapshot = await states.get(user_id)
    if has_foreign_event_draft(snapshot, event_id):
        await client.send_message(chat_id=chat_id, text="Это старое меню другого мероприятия. Сначала завершите или отмените текущее редактирование.")
        return
    list_back = snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events"
    await states.set(user_id, chat_id, "max_event_edit_link", {"event_id": event_id, "event_list_back": list_back})
    await client.send_message(
        chat_id=chat_id,
        text="🔗 Введите новую ссылку:",
        attachments=inline_keyboard([_back(_event_open_callback(event_id, list_back))]),
    )


async def receive_edit_link(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, raw_link: str) -> None:
    state_obj = await states.get(user_id)
    event_id = state_obj.data["event_id"]
    list_back = state_obj.data.get("event_list_back", "admin_events")
    try:
        scen = await update_scenario_fields(async_session_maker, event_id, link=raw_link)
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text="✅ Ссылка обновлена.")
        await show_scenario(client, chat_id, event_id, back=list_back)
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")


async def start_edit_place(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, event_id: int) -> None:
    snapshot = await states.get(user_id)
    if has_foreign_event_draft(snapshot, event_id):
        await client.send_message(chat_id=chat_id, text="Это старое меню другого мероприятия. Сначала завершите или отмените текущее редактирование.")
        return
    list_back = snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events"
    await states.set(user_id, chat_id, "max_event_edit_place", {"event_id": event_id, "event_list_back": list_back})
    await client.send_message(
        chat_id=chat_id,
        text="📍 Введите новое место проведения встречи:",
        attachments=inline_keyboard([_back(_event_open_callback(event_id, list_back))]),
    )


async def receive_edit_place(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, raw_place: str) -> None:
    state_obj = await states.get(user_id)
    event_id = state_obj.data["event_id"]
    list_back = state_obj.data.get("event_list_back", "admin_events")
    try:
        scen = await update_scenario_fields(async_session_maker, event_id, place=raw_place)
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text="✅ Место проведения обновлено.")
        await show_scenario(client, chat_id, event_id, back=list_back)
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")


# ==============================================================================
# RESCHEDULE EVENT
# ==============================================================================

async def start_reschedule(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, event_id: int) -> None:
    snapshot = await states.get(user_id)
    if has_foreign_event_draft(snapshot, event_id):
        await client.send_message(chat_id=chat_id, text="Это старое меню другого мероприятия. Сначала завершите или отмените текущее редактирование.")
        return
    s = await get_scenario_summary(async_session_maker, event_id)
    if not s:
        return
    list_back = snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events"
    await states.set(user_id, chat_id, "max_event_reschedule_starts_at", {"event_id": event_id, "event_list_back": list_back})
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"⏰ <b>Перенос времени мероприятия: {format_event_html_title(s.title, platform='max')}</b>\n\n"
            f"• Текущее время начала: <b>{s.starts_at_msk}</b>\n"
            f"• Затронуто регистраций: <b>{s.registrations_count}</b>\n\n"
            f"Введите новое время начала по <b>Московскому времени (МСК)</b>:\n"
            f"Формат: <code>ДД.ММ.ГГГГ ЧЧ:ММ</code>\n"
            f"Пример: <code>28.10.2026 20:00</code>"
        ),
        attachments=inline_keyboard([_back(_event_open_callback(event_id, list_back))]),
    )


async def receive_reschedule(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, raw_dt: str) -> None:
    state_obj = await states.get(user_id)
    event_id = state_obj.data["event_id"]
    list_back = state_obj.data.get("event_list_back", "admin_events")
    try:
        utc_dt = parse_msk_datetime(raw_dt)
        res = await reschedule_scenario(async_session_maker, event_id, utc_dt)
        await states.clear(user_id)
        await client.send_message(
            chat_id=chat_id,
            text=(
                f"🎉 <b>Мероприятие перенесено!</b>\n\n"
                f"• Новое время: <b>{format_msk_datetime(res.new_starts_at)}</b>\n"
                f"• Ревизия расписания: <b>{res.schedule_revision}</b>\n"
                f"• Затронуто регистраций: <b>{res.registrations_count}</b>"
            ),
        )
        await show_scenario(client, chat_id, event_id, back=list_back)
    except Exception as exc:
        log.exception("MAX reschedule error: %s", exc)
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка переноса: {exc}\nПопробуйте ещё раз:")


# ==============================================================================
# COPY SCENARIO
# ==============================================================================

async def start_copy(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, source_id: int) -> None:
    snapshot = await states.get(user_id)
    if has_foreign_event_draft(snapshot, source_id):
        await client.send_message(chat_id=chat_id, text="Это старое меню другого мероприятия. Сначала завершите или отмените текущее редактирование.")
        return
    s = await get_scenario_summary(async_session_maker, source_id)
    if not s:
        return
    list_back = snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events"
    await states.set(user_id, chat_id, "max_event_copy_code", {"source_id": source_id, "event_list_back": list_back})
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"📋 <b>Копирование сценария: {format_event_html_title(s.title, platform='max')}</b>\n\n"
            f"Все параметры и шаги скопируются. Регистрации и отправки НЕ копируются.\n\n"
            f"<b>Шаг 1 из 2:</b> Введите уникальный код нового сценария (например: <code>{s.code}_next</code>):"
        ),
        attachments=inline_keyboard([_back(_event_open_callback(source_id, list_back))]),
    )


async def receive_copy_code(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, raw_code: str) -> None:
    try:
        norm_code = validate_event_code(raw_code)
    except ValueError as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка в коде: {exc}\nПопробуйте снова:")
        return

    state_obj = await states.get(user_id)
    data = state_obj.data if state_obj else {}
    data["copy_code"] = norm_code
    await states.set(user_id, chat_id, "max_event_copy_starts_at", data)

    await client.send_message(
        chat_id=chat_id,
        text=(
            f"✅ Код: <code>{norm_code}</code>\n\n"
            f"<b>Шаг 2 из 2:</b> Введите дату и время начала нового мероприятия по <b>Московскому времени (МСК)</b>:\n"
            f"Формат: <code>ДД.ММ.ГГГГ ЧЧ:ММ</code>\n"
            f"Пример: <code>01.11.2026 19:00</code>"
        ),
    )


async def receive_copy_starts_at(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, raw_dt: str) -> None:
    state_obj = await states.get(user_id)
    data = state_obj.data if state_obj else {}
    source_id = data["source_id"]
    new_code = data["copy_code"]

    try:
        utc_dt = parse_msk_datetime(raw_dt)
        new_scen = await copy_scenario(
            async_session_maker,
            source_event_id=source_id,
            new_code=new_code,
            new_starts_at_utc=utc_dt,
        )
        list_back = data.get("event_list_back", "admin_events")
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text="🎉 <b>Сценарий успешно скопирован!</b>")
        await show_scenario(client, chat_id, new_scen.id, back=list_back)
    except Exception as exc:
        log.exception("MAX copy error: %s", exc)
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка копирования: {exc}\nПопробуйте снова:")


# ==============================================================================
# REMINDER STEPS CRUD
# ==============================================================================

async def show_steps(
    client: MaxApiClient,
    chat_id: int,
    event_id: int,
    *,
    back: str = "admin_events",
) -> None:
    s = await get_scenario_summary(async_session_maker, event_id)
    if not s:
        return
    steps = await list_steps(async_session_maker, event_id, include_archived=False)

    rows = []
    for idx, st in enumerate(steps):
        short = compact_event_title(st.message_text, limit=25)
        label = f"{idx + 1}. {st.offset_display} — {short}"
        rows.append([callback_button(label, _event_step_callback(st.id, back))])

    if not s.is_archived:
        rows.append([callback_button("➕ Добавить шаг", _event_step_action_callback("add", event_id, back))])

    rows.append(_back(_event_open_callback(event_id, back)))

    text_lines = [
        f"📝 <b>Цепочка напоминаний: {format_event_html_title(s.title, platform='max')}</b>\n\n",
    ]
    if not steps:
        text_lines.append("Шагов пока нет. Добавьте шаги с нужным временем отправки.")
    else:
        text_lines.append(f"Всего шагов: {len(steps)}:\n\n")
        for idx, st in enumerate(steps):
            preview = truncate_html_preview(st.message_text, 100, allowed_tags=ALLOWED_EVENT_HTML_TAGS)
            text_lines.append(
                f"<b>{idx + 1}. {st.offset_display}</b> (v{st.step_version})\n"
                f"<blockquote>{preview}</blockquote>\n\n"
            )

    await client.send_message(
        chat_id=chat_id,
        text="".join(text_lines),
        attachments=inline_keyboard(rows),
    )


async def show_step(
    client: MaxApiClient,
    chat_id: int,
    step_id: int,
    *,
    back: str = "admin_events",
) -> None:
    st = await get_step_item(async_session_maker, step_id)
    if not st:
        await client.send_message(chat_id=chat_id, text="Шаг не найден.")
        return
    s = await get_scenario_summary(async_session_maker, st.event_id)
    if not s:
        return

    all_steps = await list_steps(async_session_maker, st.event_id, include_archived=False)
    idx = next((i for i, step in enumerate(all_steps) if step.id == st.id), 0)
    is_first = (idx == 0)
    is_last = (idx == len(all_steps) - 1)

    rows = []
    if not s.is_archived:
        rows.append([callback_button("✏️ Изменить текст", _event_step_action_callback("edit_text", st.id, back))])
        rows.append([callback_button("⏱ Изменить время отправки", _event_step_action_callback("edit_offset", st.id, back))])
        rows.append([callback_button(
            f"📎 Детали события: {'вкл.' if st.include_event_details else 'выкл.'}",
            _event_step_action_callback("toggle_details", st.id, back),
        )])

        order_row = []
        if not is_first:
            order_row.append(callback_button("⬆️ Выше", _event_step_action_callback("up", st.id, back)))
        if not is_last:
            order_row.append(callback_button("⬇️ Ниже", _event_step_action_callback("down", st.id, back)))
        if order_row:
            rows.append(order_row)

        rows.append([callback_button("🗑 Удалить шаг", _event_step_action_callback("del_ask", st.id, back))])

    rows.append([callback_button("🚀 Тестовая отправка шага", _event_step_action_callback("test", st.id, back))])
    rows.append(_back(_event_steps_callback(st.event_id, back)))

    text = (
        f"📝 <b>Шаг напоминания {idx + 1} из {len(all_steps)}</b>\n\n"
        f"• Мероприятие: {format_event_html_title(s.title, platform='max')}\n"
        f"• Время отправки: <b>{st.offset_display}</b> ({st.offset_minutes} мин.)\n"
        f"• Версия: v{st.step_version}\n\n"
        f"• Детали события: {'включены' if st.include_event_details else 'выключены'}\n\n"
        f"<b>Текст сообщения:</b>\n"
        f"<blockquote>{truncate_html_preview(st.message_text, 2800, allowed_tags=ALLOWED_EVENT_HTML_TAGS)}</blockquote>"
    )

    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=inline_keyboard(rows),
    )


async def toggle_step_event_details(
    client: MaxApiClient,
    chat_id: int,
    step_id: int,
    *,
    back: str = "admin_events",
) -> None:
    step = await get_step_item(async_session_maker, step_id)
    if not step:
        return
    try:
        await set_step_event_details(async_session_maker, step_id, not step.include_event_details)
        await show_step(client, chat_id, step_id, back=back)
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")


async def start_step_add(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    event_id: int,
    *,
    back: str | None = None,
) -> None:
    snapshot = await states.get(user_id)
    list_back = back or (snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events")
    await states.set(
        user_id,
        chat_id,
        "max_event_step_offset",
        {"event_id": event_id, "event_list_back": list_back},
    )
    rows = []
    for offset_val, label in PRESET_OFFSETS:
        rows.append([callback_button(label, f"admin_ev_offset_preset_{event_id}_{offset_val}")])
    rows.append([callback_button("✏️ Ввести другое смещение", f"admin_ev_offset_custom_{event_id}")])
    rows.append(_back(_event_steps_callback(event_id, list_back)))

    await client.send_message(
        chat_id=chat_id,
        text="➕ <b>Добавление шага</b>\n\n<b>Шаг 1 из 2:</b> Выберите время отправки относительно начала мероприятия:",
        attachments=inline_keyboard(rows),
    )


async def select_step_offset_preset(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    event_id: int,
    offset_val: int,
) -> None:
    snapshot = await states.get(user_id)
    list_back = snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events"
    await states.set(
        user_id,
        chat_id,
        "max_event_step_text",
        {"event_id": event_id, "offset_minutes": offset_val, "event_list_back": list_back},
    )
    disp = format_offset_minutes(offset_val)
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"✅ Время отправки: <b>{disp}</b>\n\n"
            f"<b>Шаг 2 из 2:</b> Введите текст напоминания.\n"
            f"Доступные плейсхолдеры:\n"
            f"• <code>{{name}}</code> — имя\n"
            f"• <code>{{event_title}}</code> — название\n"
            f"• <code>{{event_date}}</code> — дата\n"
            f"• <code>{{event_time}}</code> — время МСК\n"
            f"• <code>{{link}}</code> — ссылка"
        ),
        attachments=inline_keyboard([_back(_event_steps_callback(event_id, list_back))]),
    )


async def start_step_offset_custom(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    event_id: int,
) -> None:
    snapshot = await states.get(user_id)
    list_back = snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events"
    await states.set(
        user_id,
        chat_id,
        "max_event_step_offset_custom",
        {"event_id": event_id, "event_list_back": list_back},
    )
    await client.send_message(
        chat_id=chat_id,
        text="⏱ Введите смещение (например: <code>-24h</code>, <code>-2h</code>, <code>0</code>, <code>+2h</code>, <code>-30m</code>):",
        attachments=inline_keyboard([_back(_event_steps_callback(event_id, list_back))]),
    )


async def receive_step_offset_custom(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    raw_offset: str,
) -> None:
    state_obj = await states.get(user_id)
    event_id = state_obj.data["event_id"]
    try:
        offset = parse_offset_minutes(raw_offset)
    except ValueError as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ {exc}\nПопробуйте снова:")
        return

    await select_step_offset_preset(client, states, chat_id, user_id, event_id, offset)


async def receive_step_text(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    raw_text: str,
) -> None:
    state_obj = await states.get(user_id)
    data = state_obj.data if state_obj else {}
    event_id = data["event_id"]
    offset = data["offset_minutes"]
    list_back = data.get("event_list_back", "admin_events")

    try:
        validate_step_input(offset, raw_text)
        st = await add_step(async_session_maker, event_id, offset, raw_text)
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text="🎉 <b>Шаг успешно добавлен!</b>")
        await show_steps(client, chat_id, event_id, back=list_back)
    except Exception as exc:
        log.exception("MAX add step error: %s", exc)
        await client.send_message(
            chat_id=chat_id,
            text=f"❌ Ошибка: {exc}\n\nПожалуйста, исправьте разметку и отправьте текст снова или нажмите «Назад»:",
            attachments=inline_keyboard([_back(_event_steps_callback(event_id, list_back))]),
        )


async def start_edit_step_text(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    step_id: int,
    *,
    back: str | None = None,
) -> None:
    st = await get_step_item(async_session_maker, step_id)
    if not st:
        return
    snapshot = await states.get(user_id)
    list_back = back or (snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events")
    await states.set(
        user_id,
        chat_id,
        "max_event_step_edit_text",
        {"step_id": step_id, "event_list_back": list_back},
    )
    await client.send_message(
        chat_id=chat_id,
        text=f"✏️ <b>Редактирование текста</b>\n\nТекущий:\n<blockquote>{truncate_html_preview(st.message_text, 2800, allowed_tags=ALLOWED_EVENT_HTML_TAGS)}</blockquote>\n\nВведите новый текст:",
        attachments=inline_keyboard([_back(_event_step_callback(step_id, list_back))]),
    )


async def receive_edit_step_text(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    raw_text: str,
) -> None:
    state_obj = await states.get(user_id)
    data = state_obj.data
    step_id = data["step_id"]
    list_back = data.get("event_list_back", "admin_events")
    try:
        updated = await update_step(async_session_maker, step_id, message_text=raw_text)
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text=f"✅ Текст обновлен (v{updated.step_version}).")
        await show_step(client, chat_id, step_id, back=list_back)
    except Exception as exc:
        await client.send_message(
            chat_id=chat_id,
            text=f"❌ Ошибка: {exc}\n\nПожалуйста, исправьте разметку и отправьте текст снова или нажмите «Назад»:",
            attachments=inline_keyboard([_back(_event_step_callback(step_id, list_back))]),
        )


async def start_edit_step_offset(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    step_id: int,
    *,
    back: str | None = None,
) -> None:
    st = await get_step_item(async_session_maker, step_id)
    if not st:
        return
    snapshot = await states.get(user_id)
    list_back = back or (snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events")
    await states.set(
        user_id,
        chat_id,
        "max_event_step_edit_offset",
        {"step_id": step_id, "event_list_back": list_back},
    )
    rows = []
    for offset_val, label in PRESET_OFFSETS:
        callback = _event_step_value_callback("edit_preset", step_id, offset_val, list_back)
        rows.append([callback_button(label, callback)])
    rows.append([callback_button("✏️ Ввести другое смещение", _event_step_action_callback("edit_custom", step_id, list_back))])
    rows.append(_back(_event_step_callback(step_id, list_back)))

    await client.send_message(
        chat_id=chat_id,
        text=f"⏱ Текущее время отправки: <b>{st.offset_display}</b>\n\nВыберите новое смещение:",
        attachments=inline_keyboard(rows),
    )


async def select_edit_step_offset_preset(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    step_id: int,
    offset_val: int,
) -> None:
    snapshot = await states.get(user_id)
    list_back = snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events"
    try:
        updated = await update_step(async_session_maker, step_id, offset_minutes=offset_val)
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text=f"✅ Время отправки обновлено: <b>{updated.offset_display}</b>")
        await show_step(client, chat_id, step_id, back=list_back)
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")


async def start_edit_step_offset_custom(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    step_id: int,
    *,
    back: str | None = None,
) -> None:
    snapshot = await states.get(user_id)
    list_back = back or (snapshot.data.get("event_list_back", "admin_events") if snapshot else "admin_events")
    await states.set(
        user_id,
        chat_id,
        "max_event_step_edit_offset_custom",
        {"step_id": step_id, "event_list_back": list_back},
    )
    await client.send_message(
        chat_id=chat_id,
        text="⏱ Введите произвольное смещение (например: <code>-24h</code>, <code>-2h</code>, <code>0</code>, <code>+2h</code>):",
        attachments=inline_keyboard([_back(_event_step_callback(step_id, list_back))]),
    )


async def receive_edit_step_offset_custom(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    raw_offset: str,
) -> None:
    state_obj = await states.get(user_id)
    data = state_obj.data
    step_id = data["step_id"]
    list_back = data.get("event_list_back", "admin_events")
    try:
        offset = parse_offset_minutes(raw_offset)
        updated = await update_step(async_session_maker, step_id, offset_minutes=offset)
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text=f"✅ Время отправки обновлено: <b>{updated.offset_display}</b>")
        await show_step(client, chat_id, step_id, back=list_back)
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ {exc}\nПопробуйте ещё раз:")


async def step_move_up(
    client: MaxApiClient,
    chat_id: int,
    step_id: int,
    *,
    back: str = "admin_events",
) -> None:
    st = await get_step_item(async_session_maker, step_id)
    if not st:
        return
    all_steps = await list_steps(async_session_maker, st.event_id, include_archived=False)
    ids = [s.id for s in all_steps]
    idx = ids.index(step_id)
    if idx > 0:
        ids[idx - 1], ids[idx] = ids[idx], ids[idx - 1]
        await reorder_steps(async_session_maker, st.event_id, ids)
    await show_step(client, chat_id, step_id, back=back)


async def step_move_down(
    client: MaxApiClient,
    chat_id: int,
    step_id: int,
    *,
    back: str = "admin_events",
) -> None:
    st = await get_step_item(async_session_maker, step_id)
    if not st:
        return
    all_steps = await list_steps(async_session_maker, st.event_id, include_archived=False)
    ids = [s.id for s in all_steps]
    idx = ids.index(step_id)
    if idx < len(ids) - 1:
        ids[idx + 1], ids[idx] = ids[idx], ids[idx + 1]
        await reorder_steps(async_session_maker, st.event_id, ids)
    await show_step(client, chat_id, step_id, back=back)


async def ask_delete_step(
    client: MaxApiClient,
    chat_id: int,
    step_id: int,
    *,
    back: str = "admin_events",
) -> None:
    st = await get_step_item(async_session_maker, step_id)
    if not st:
        return
    rows = [
        [callback_button("✅ Да, архивировать", _event_step_action_callback("del_yes", step_id, back))],
        _back(_event_step_callback(step_id, back)),
    ]
    await client.send_message(
        chat_id=chat_id,
        text=f"🗑 Удалить/архивировать шаг «<b>{st.offset_display}</b>»?\n\nОжидающие отправки этого шага будут пропущены.",
        attachments=inline_keyboard(rows),
    )


async def confirm_delete_step(
    client: MaxApiClient,
    chat_id: int,
    step_id: int,
    *,
    back: str = "admin_events",
) -> None:
    st = await get_step_item(async_session_maker, step_id)
    if not st:
        return
    await archive_step(async_session_maker, step_id)
    await client.send_message(chat_id=chat_id, text="✅ Шаг архивирован.")
    await show_steps(client, chat_id, st.event_id, back=back)


# ==============================================================================
# MONITORING & MANUAL RESOLUTION
# ==============================================================================

async def show_stats(
    client: MaxApiClient,
    chat_id: int,
    event_id: int,
    *,
    back: str = "admin_events",
) -> None:
    s = await get_scenario_summary(async_session_maker, event_id)
    if not s:
        return
    stats = await get_scenario_delivery_stats(async_session_maker, event_id)

    rows = []
    if stats.problematic_total > 0:
        rows.append([callback_button(f"⚠️ Проблемные доставки ({stats.problematic_total})", f"admin_ev_problems_{event_id}_0")])
    rows.append([callback_button("🔄 Обновить", f"admin_ev_stats_{event_id}")])
    rows.append(_back(_event_open_callback(event_id, back)))

    lines = [
        f"📊 <b>Мониторинг доставок: {format_event_html_title(s.title, platform='max')}</b>\n\n",
        f"• Регистраций: <b>{s.registrations_count}</b>\n",
        f"• Всего сообщений в очереди/отправлено: <b>{stats.total}</b>\n\n",
        f"<b>Статусы:</b>\n",
        f"• pending (ожидают): {stats.pending}\n",
        f"• claimed (в обработке): {stats.claimed}\n",
        f"• sending (отправляются): {stats.sending}\n",
        f"• sent (доставлено): {stats.sent}\n",
        f"• skipped (пропущено): {stats.skipped}\n",
        f"• blocked (бот заблокирован): {stats.blocked}\n",
        f"• failed (ошибки): {stats.failed}\n",
        f"• uncertain (неопределенно): {stats.uncertain}\n",
        f"• resolved_sent (вручную доставлено): {stats.resolved_sent}\n",
        f"• resolved_skipped (вручную пропущено): {stats.resolved_skipped}\n",
    ]
    if stats.problematic_total > 0:
        lines.append(f"\n⚠️ <b>Требуют внимания: {stats.problematic_total}</b>")

    await client.send_message(
        chat_id=chat_id,
        text="".join(lines),
        attachments=inline_keyboard(rows),
    )


async def show_problems(
    client: MaxApiClient,
    chat_id: int,
    event_id: int,
    page: int = 0,
) -> None:
    page = max(0, page)
    items, total = await list_problem_deliveries(
        async_session_maker,
        event_id,
        limit=PROBLEMS_PAGE_SIZE,
        offset=page * PROBLEMS_PAGE_SIZE,
    )
    total_pages = max(1, (total + PROBLEMS_PAGE_SIZE - 1) // PROBLEMS_PAGE_SIZE)
    safe_page = min(page, total_pages - 1)
    if safe_page != page:
        page = safe_page
        items, total = await list_problem_deliveries(
            async_session_maker,
            event_id,
            limit=PROBLEMS_PAGE_SIZE,
            offset=page * PROBLEMS_PAGE_SIZE,
        )

    rows = []
    for it in items:
        status_icon = "❓" if it.status == "uncertain" else ("🛑" if it.status == "blocked" else "❌")
        item_callback = f"admin_ev_problem_item_{it.id}"
        if page > 0:
            item_callback += f"_{page}"
        rows.append([callback_button(f"{status_icon} ID {it.id}: {it.user_display[:20]} ({it.status})", item_callback)])

    nav_row = []
    if page > 0:
        nav_row.append(callback_button("⬅️ Назад", f"admin_ev_problems_{event_id}_{page - 1}"))
    if page < total_pages - 1:
        nav_row.append(callback_button("Далее ➡️", f"admin_ev_problems_{event_id}_{page + 1}"))
    if nav_row:
        rows.append(nav_row)

    rows.append(_back(f"admin_ev_stats_{event_id}"))

    lines = [
        f"⚠️ <b>Проблемные доставки ({total})</b>\n",
        f"Страница {page + 1} из {total_pages}\n\n",
    ]
    for it in items:
        lines.append(
            f"• <b>Доставка ID {it.id}</b> | {html.escape(it.user_display)}\n"
            f"  Статус: <b>{it.status}</b> | Тип: {it.message_kind}\n"
            f"  Время: {it.due_at_msk} | Попыток: {it.attempt_count}\n"
            f"  Ошибка: <code>{html.escape((it.last_error or '—')[:80])}</code>\n\n"
        )

    await client.send_message(
        chat_id=chat_id,
        text="".join(lines),
        attachments=inline_keyboard(rows),
    )


async def show_problem_detail(
    client: MaxApiClient,
    chat_id: int,
    delivery_id: int,
    problem_page: int = 0,
) -> None:
    it = await get_delivery_item(async_session_maker, delivery_id)
    if not it:
        await client.send_message(chat_id=chat_id, text="Доставка не найдена.")
        return

    parts = await list_delivery_parts(async_session_maker, delivery_id)
    rows = []
    if it.status == "uncertain":
        uncertain_parts = [part for part in parts if part.status == "uncertain"]
        if uncertain_parts:
            for part in uncertain_parts:
                suffix = f"_{part.id}"
                rows.append([callback_button(f"🔄 Часть {part.part_index + 1}: повторить", f"admin_ev_resolve_force_{it.id}{suffix}")])
                rows.append([callback_button(f"✅ Часть {part.part_index + 1}: доставлено", f"admin_ev_resolve_sent_{it.id}{suffix}")])
                rows.append([callback_button(f"⏭ Часть {part.part_index + 1}: пропустить", f"admin_ev_resolve_skipped_{it.id}{suffix}")])
        else:
            rows.append([callback_button("🔄 Повторить отправку (Force Retry)", f"admin_ev_resolve_force_{it.id}")])
            rows.append([callback_button("✅ Отметить отправленным (Mark Sent)", f"admin_ev_resolve_sent_{it.id}")])
            rows.append([callback_button("⏭ Отметить пропущенным (Mark Skipped)", f"admin_ev_resolve_skipped_{it.id}")])

    part_lines = ""
    if parts:
        part_lines = "\n" + "".join(
            f"• Часть {part.part_index + 1}: <b>{part.part_kind}</b> — <b>{part.status}</b>\n"
            for part in parts
        )

    rows.append(_back(f"admin_ev_problems_{it.event_id}_{problem_page}"))

    text = (
        f"🔍 <b>Детали доставки ID {it.id}</b>\n\n"
        f"• Получатель: {html.escape(it.user_display)}\n"
        f"• Платформа: {it.platform}\n"
        f"• Тип сообщения: {it.message_kind}\n"
        f"• Статус: <b>{it.status}</b>\n"
        f"• Время (МСК): {it.due_at_msk}\n"
        f"• Попыток: {it.attempt_count}\n"
        f"• Индекс retry: {it.retry_index}\n"
        f"• Ошибка:\n<code>{html.escape(it.last_error or '—')}</code>"
        f"{part_lines}"
    )

    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=inline_keyboard(rows),
    )


async def start_resolve(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    delivery_id: int,
    action: str,
    part_id: int | None = None,
    problem_page: int = 0,
    list_back: str = "admin_events",
) -> None:
    type_map = {
        "force": ("force_retry", "Повторная отправка"),
        "sent": ("mark_sent", "Отметить отправленным"),
        "skipped": ("mark_skipped", "Отметить пропущенным"),
    }
    res_type, label = type_map[action]
    await states.set(
        user_id,
        chat_id,
        "max_event_resolve_reason",
        {
            "delivery_id": delivery_id,
            "res_type": res_type,
            "part_id": part_id,
            "problem_page": problem_page,
            "event_list_back": list_back,
        },
    )

    await client.send_message(
        chat_id=chat_id,
        text=(
            f"📝 <b>Разрешение доставки ID {delivery_id}: {label}</b>\n\n"
            + ("⚠️ Повторная отправка может привести к дублю части медиа, если провайдер уже принял запрос.\n\n" if res_type == "force_retry" else "")
            + "Введите причину разрешения (для аудита):"
        ),
        attachments=inline_keyboard([
            _back(
                f"admin_ev_problem_item_{delivery_id}_{problem_page}"
                if problem_page > 0
                else f"admin_ev_problem_item_{delivery_id}"
            )
        ]),
    )


async def receive_resolve_reason(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    raw_reason: str,
) -> None:
    reason = raw_reason.strip()
    if not reason:
        await client.send_message(chat_id=chat_id, text="❌ Причина обязательна:")
        return

    state_obj = await states.get(user_id)
    delivery_id = state_obj.data["delivery_id"]
    res_type = state_obj.data["res_type"]
    part_id = state_obj.data.get("part_id")
    problem_page = int(state_obj.data.get("problem_page", 0))
    list_back = state_obj.data.get("event_list_back", "admin_events")

    try:
        deliv, ok = await resolve_uncertain_delivery(
            async_session_maker,
            delivery_id=delivery_id,
            admin_id=user_id,
            resolution_type=res_type,
            resolution_reason=reason,
            part_id=part_id,
        )
        await states.clear(user_id)
        if list_back != "admin_events":
            await states.set(
                user_id,
                chat_id,
                "max_event_card",
                {"event_list_back": list_back, "problem_page": problem_page},
            )
        await client.send_message(chat_id=chat_id, text=f"✅ Доставка ID {delivery_id} успешно разрешена ({deliv.status})!")
        await show_problem_detail(client, chat_id, delivery_id, problem_page=problem_page)
    except Exception as exc:
        log.exception("MAX resolution error: %s", exc)
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")


# ==============================================================================
# TEST SEND TO SELF
# ==============================================================================

async def show_test_send_menu(
    client: MaxApiClient,
    chat_id: int,
    event_id: int,
    *,
    back: str = "admin_events",
) -> None:
    steps = await list_steps(async_session_maker, event_id, include_archived=False)
    rows = [
        [callback_button("✉️ Подтверждение регистрации", f"admin_ev_do_test_send_{event_id}_0")],
    ]
    for idx, s in enumerate(steps):
        short = compact_event_title(s.message_text, limit=20)
        rows.append([callback_button(f"✉️ Шаг {idx + 1} ({s.offset_display}): {short}", f"admin_ev_do_test_send_{event_id}_{s.id}")])

    rows.append(_back(_event_open_callback(event_id, back)))

    await client.send_message(
        chat_id=chat_id,
        text="🚀 <b>Тестовая отправка сообщения себе</b>\n\nВыберите сообщение для проверки шаблона и отображения:",
        attachments=inline_keyboard(rows),
    )


async def do_test_send(
    client: MaxApiClient,
    chat_id: int,
    user_id: int,
    event_id: int,
    step_id: int | None,
) -> None:
    # Get user object if available
    admin_user = None
    async with async_session_maker() as session:
        admin_user = await session.get(User, user_id)

    from event_delivery_worker import (
        build_test_delivery_plan,
        ensure_max_test_media_materialized,
        send_test_delivery_plan,
    )
    from event_media import EventMediaMaterializationError

    try:
        await ensure_max_test_media_materialized(
            async_session_maker,
            event_id,
            step_id,
            client,
        )
    except EventMediaMaterializationError as exc:
        log.exception("MAX test-send media materialization failed for event=%s", event_id)
        await client.send_message(
            chat_id=chat_id,
            text=f"❌ Не удалось подготовить медиа для тестовой отправки в MAX: {exc}",
        )
        return

    plan = await build_test_delivery_plan(
        async_session_maker,
        event_id,
        step_id,
        admin_user,
        header="🧪 <b>[ТЕСТОВОЕ СООБЩЕНИЕ MAX]</b>",
    )
    await send_test_delivery_plan(
        plan,
        platform="max",
        user_id=user_id,
        event_id=event_id,
        max_client=client,
    )
