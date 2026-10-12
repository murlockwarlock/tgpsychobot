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
from datetime import datetime
from typing import Any
from urllib.parse import quote

from event_admin_service import (
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
    list_scenarios,
    list_steps,
    parse_msk_datetime,
    parse_offset_minutes,
    render_test_step,
    reorder_steps,
    reschedule_scenario,
    resolve_uncertain_delivery,
    toggle_scenario_active,
    update_scenario_fields,
    update_step,
    validate_event_code,
    validate_scenario_input,
    validate_step_input,
)
from ..api import MaxApiClient
from ..keyboards import callback_button, inline_keyboard
from ..legacy import User, async_session_maker
from ..storage import StateStore

log = logging.getLogger(__name__)

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
    "max_event_reschedule_starts_at",
    "max_event_copy_code",
    "max_event_copy_starts_at",
    "max_event_step_offset_custom",
    "max_event_step_text",
    "max_event_step_edit_offset_custom",
    "max_event_step_edit_text",
    "max_event_resolve_reason",
})

PAGE_SIZE = 6
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
        label = f"{status_icon} {type_icon} {s.title} ({s.starts_at_msk[:10]})"
        rows.append([callback_button(label, f"admin_ev_open_{s.id}")])

    # Pagination
    nav_row = []
    if page > 0:
        nav_row.append(callback_button("◀️ Назад", f"admin_ev_page_{page - 1}"))
    if page < total_pages - 1:
        nav_row.append(callback_button("Вперёд ▶️", f"admin_ev_page_{page + 1}"))
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
                f"<b>{idx + 1 + page * PAGE_SIZE}. {html.escape(s.title)}</b>\n"
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

    text = (
        f"📅 <b>Карточка мероприятия: {html.escape(s.title)}</b>\n\n"
        f"{link_block}"
        f"• <b>Тип:</b> {kind_str}\n"
        f"• <b>Дата и время начала:</b> {s.starts_at_msk}\n"
        f"• <b>Длительность:</b> {s.duration_minutes} мин.\n"
        f"{loc_block}\n"
        f"• <b>Статус:</b> {status_str}\n"
        f"• <b>Регистраций:</b> <b>{s.registrations_count}</b>\n"
        f"• <b>Ревизия:</b> {s.schedule_revision} (контент v{s.content_version})\n"
    )

    rows = []
    if not s.is_archived:
        toggle_label = "⚪️ Выключить запись" if s.is_active else "🟢 Включить запись"
        rows.append([callback_button(toggle_label, f"admin_ev_toggle_{s.id}")])

    rows.append([callback_button("📝 Цепочка сообщений (шаги)", f"admin_ev_steps_{s.id}")])
    rows.append([callback_button("📊 Статус отправок", f"admin_ev_stats_{s.id}")])

    if not s.is_archived:
        rows.append([callback_button("⏰ Перенести время", f"admin_ev_reschedule_{s.id}")])
        rows.append([callback_button("✏️ Редактировать", f"admin_ev_edit_menu_{s.id}")])
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


async def toggle_scenario(client: MaxApiClient, chat_id: int, event_id: int) -> None:
    try:
        new_active = await toggle_scenario_active(async_session_maker, event_id)
        msg_text = "🟢 Регистрация включена!" if new_active else "⚪️ Регистрация приостановлена."
        await client.send_message(chat_id=chat_id, text=msg_text)
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")
        return

    await show_scenario(client, chat_id, event_id)


async def archive_scenario_ask(client: MaxApiClient, chat_id: int, event_id: int) -> None:
    s = await get_scenario_summary(async_session_maker, event_id)
    if not s:
        return
    rows = [
        [callback_button("✅ Да, архивировать", f"admin_ev_archive_yes_{event_id}")],
        _back(f"admin_ev_open_{event_id}"),
    ]
    await client.send_message(
        chat_id=chat_id,
        text=f"📦 Архивировать мероприятие «<b>{html.escape(s.title)}</b>»?\n\nВсе регистрации и история доставок сохранятся.",
        attachments=inline_keyboard(rows),
    )


async def archive_scenario_confirm(client: MaxApiClient, chat_id: int, event_id: int) -> None:
    try:
        await archive_scenario(async_session_maker, event_id)
        await client.send_message(chat_id=chat_id, text="✅ Мероприятие архивировано.")
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")
        return

    await show_scenario(client, chat_id, event_id)


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
        text=f"✅ Код принят: <code>{norm_code}</code>\n\n<b>Шаг 2 из 5:</b> Введите название мероприятия (до 255 символов):",
    )


async def receive_create_title(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    raw_title: str,
) -> None:
    title = raw_title.strip()
    if not title or len(title) > 255:
        await client.send_message(chat_id=chat_id, text="❌ Название не должно быть пустым и не может превышать 255 символов:")
        return

    state_obj = await states.get(user_id)
    data = state_obj.data if state_obj else {}
    data["create_title"] = title
    await states.set(user_id, chat_id, "max_event_create_starts_at", data)

    await client.send_message(
        chat_id=chat_id,
        text=(
            f"✅ Название: <b>{html.escape(title)}</b>\n\n"
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

async def show_edit_menu(client: MaxApiClient, chat_id: int, event_id: int) -> None:
    s = await get_scenario_summary(async_session_maker, event_id)
    if not s:
        return
    rows = [
        [callback_button("✏️ Изменить название", f"admin_ev_edit_title_{s.id}")],
        [callback_button("⏱ Изменить длительность", f"admin_ev_edit_duration_{s.id}")],
    ]
    if s.event_type == "webinar":
        rows.append([callback_button("🔗 Изменить ссылку на вебинар", f"admin_ev_edit_link_{s.id}")])
    else:
        rows.append([callback_button("📍 Изменить место встречи", f"admin_ev_edit_place_{s.id}")])
        link_label = "🔗 Изменить ссылку" if s.link else "🔗 Добавить ссылку"
        rows.append([callback_button(link_label, f"admin_ev_edit_link_{s.id}")])

    rows.append(_back(f"admin_ev_open_{s.id}"))
    await client.send_message(
        chat_id=chat_id,
        text=f"✏️ <b>Редактирование мероприятия: {html.escape(s.title)}</b>\n\nВыберите поле для изменения:",
        attachments=inline_keyboard(rows),
    )


async def start_edit_title(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, event_id: int) -> None:
    await states.set(user_id, chat_id, "max_event_edit_title", {"event_id": event_id})
    await client.send_message(
        chat_id=chat_id,
        text="✏️ Введите новое название мероприятия (до 255 символов):",
        attachments=inline_keyboard([_back(f"admin_ev_open_{event_id}")]),
    )


async def receive_edit_title(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, raw_title: str) -> None:
    state_obj = await states.get(user_id)
    event_id = state_obj.data["event_id"]
    try:
        scen = await update_scenario_fields(async_session_maker, event_id, title=raw_title)
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text=f"✅ Название обновлено: <b>{html.escape(scen.title)}</b>")
        await show_scenario(client, chat_id, event_id)
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")


async def start_edit_duration(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, event_id: int) -> None:
    await states.set(user_id, chat_id, "max_event_edit_duration", {"event_id": event_id})
    await client.send_message(
        chat_id=chat_id,
        text="⏱ Введите новую длительность в минутах (от 15 до 1440):",
        attachments=inline_keyboard([_back(f"admin_ev_open_{event_id}")]),
    )


async def receive_edit_duration(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, raw_dur: str) -> None:
    state_obj = await states.get(user_id)
    event_id = state_obj.data["event_id"]
    try:
        dur = int(raw_dur.strip())
        scen = await update_scenario_fields(async_session_maker, event_id, duration_minutes=dur)
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text=f"✅ Длительность обновлена: {scen.duration_minutes} мин.")
        await show_scenario(client, chat_id, event_id)
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")


async def start_edit_link(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, event_id: int) -> None:
    await states.set(user_id, chat_id, "max_event_edit_link", {"event_id": event_id})
    await client.send_message(
        chat_id=chat_id,
        text="🔗 Введите новую ссылку:",
        attachments=inline_keyboard([_back(f"admin_ev_open_{event_id}")]),
    )


async def receive_edit_link(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, raw_link: str) -> None:
    state_obj = await states.get(user_id)
    event_id = state_obj.data["event_id"]
    try:
        scen = await update_scenario_fields(async_session_maker, event_id, link=raw_link)
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text="✅ Ссылка обновлена.")
        await show_scenario(client, chat_id, event_id)
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")


async def start_edit_place(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, event_id: int) -> None:
    await states.set(user_id, chat_id, "max_event_edit_place", {"event_id": event_id})
    await client.send_message(
        chat_id=chat_id,
        text="📍 Введите новое место проведения встречи:",
        attachments=inline_keyboard([_back(f"admin_ev_open_{event_id}")]),
    )


async def receive_edit_place(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, raw_place: str) -> None:
    state_obj = await states.get(user_id)
    event_id = state_obj.data["event_id"]
    try:
        scen = await update_scenario_fields(async_session_maker, event_id, place=raw_place)
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text="✅ Место проведения обновлено.")
        await show_scenario(client, chat_id, event_id)
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")


# ==============================================================================
# RESCHEDULE EVENT
# ==============================================================================

async def start_reschedule(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, event_id: int) -> None:
    s = await get_scenario_summary(async_session_maker, event_id)
    if not s:
        return
    await states.set(user_id, chat_id, "max_event_reschedule_starts_at", {"event_id": event_id})
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"⏰ <b>Перенос времени мероприятия: {html.escape(s.title)}</b>\n\n"
            f"• Текущее время начала: <b>{s.starts_at_msk}</b>\n"
            f"• Затронуто регистраций: <b>{s.registrations_count}</b>\n\n"
            f"Введите новое время начала по <b>Московскому времени (МСК)</b>:\n"
            f"Формат: <code>ДД.ММ.ГГГГ ЧЧ:ММ</code>\n"
            f"Пример: <code>28.10.2026 20:00</code>"
        ),
        attachments=inline_keyboard([_back(f"admin_ev_open_{event_id}")]),
    )


async def receive_reschedule(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, raw_dt: str) -> None:
    state_obj = await states.get(user_id)
    event_id = state_obj.data["event_id"]
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
        await show_scenario(client, chat_id, event_id)
    except Exception as exc:
        log.exception("MAX reschedule error: %s", exc)
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка переноса: {exc}\nПопробуйте ещё раз:")


# ==============================================================================
# COPY SCENARIO
# ==============================================================================

async def start_copy(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, source_id: int) -> None:
    s = await get_scenario_summary(async_session_maker, source_id)
    if not s:
        return
    await states.set(user_id, chat_id, "max_event_copy_code", {"source_id": source_id})
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"📋 <b>Копирование сценария: {html.escape(s.title)}</b>\n\n"
            f"Все параметры и шаги скопируются. Регистрации и отправки НЕ копируются.\n\n"
            f"<b>Шаг 1 из 2:</b> Введите уникальный код нового сценария (например: <code>{s.code}_next</code>):"
        ),
        attachments=inline_keyboard([_back(f"admin_ev_open_{source_id}")]),
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
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text="🎉 <b>Сценарий успешно скопирован!</b>")
        await show_scenario(client, chat_id, new_scen.id)
    except Exception as exc:
        log.exception("MAX copy error: %s", exc)
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка копирования: {exc}\nПопробуйте снова:")


# ==============================================================================
# REMINDER STEPS CRUD
# ==============================================================================

async def show_steps(client: MaxApiClient, chat_id: int, event_id: int) -> None:
    s = await get_scenario_summary(async_session_maker, event_id)
    if not s:
        return
    steps = await list_steps(async_session_maker, event_id, include_archived=False)

    rows = []
    for idx, st in enumerate(steps):
        short = (st.message_text[:25] + "...") if len(st.message_text) > 25 else st.message_text
        label = f"{idx + 1}. {st.offset_display} — {short.replace('\n', ' ')}"
        rows.append([callback_button(label, f"admin_ev_step_{st.id}")])

    if not s.is_archived:
        rows.append([callback_button("➕ Добавить шаг", f"admin_ev_step_add_{event_id}")])

    rows.append(_back(f"admin_ev_open_{event_id}"))

    text_lines = [
        f"📝 <b>Цепочка напоминаний: {html.escape(s.title)}</b>\n\n",
    ]
    if not steps:
        text_lines.append("Шагов пока нет. Добавьте шаги с нужным временем отправки.")
    else:
        text_lines.append(f"Всего шагов: {len(steps)}:\n\n")
        for idx, st in enumerate(steps):
            text_lines.append(
                f"<b>{idx + 1}. {st.offset_display}</b> (v{st.step_version})\n"
                f"<blockquote>{html.escape(st.message_text[:80])}</blockquote>\n\n"
            )

    await client.send_message(
        chat_id=chat_id,
        text="".join(text_lines),
        attachments=inline_keyboard(rows),
    )


async def show_step(client: MaxApiClient, chat_id: int, step_id: int) -> None:
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
        rows.append([callback_button("✏️ Изменить текст", f"admin_ev_step_edit_text_{st.id}")])
        rows.append([callback_button("⏱ Изменить время отправки", f"admin_ev_step_edit_offset_{st.id}")])

        order_row = []
        if not is_first:
            order_row.append(callback_button("⬆️ Выше", f"admin_ev_step_up_{st.id}"))
        if not is_last:
            order_row.append(callback_button("⬇️ Ниже", f"admin_ev_step_down_{st.id}"))
        if order_row:
            rows.append(order_row)

        rows.append([callback_button("🗑 Удалить шаг", f"admin_ev_step_del_ask_{st.id}")])

    rows.append([callback_button("🚀 Тестовая отправка шага", f"admin_ev_test_step_{st.id}")])
    rows.append(_back(f"admin_ev_steps_{st.event_id}"))

    text = (
        f"📝 <b>Шаг напоминания {idx + 1} из {len(all_steps)}</b>\n\n"
        f"• Мероприятие: {html.escape(s.title)}\n"
        f"• Время отправки: <b>{st.offset_display}</b> ({st.offset_minutes} мин.)\n"
        f"• Версия: v{st.step_version}\n\n"
        f"<b>Текст сообщения:</b>\n"
        f"<blockquote>{html.escape(st.message_text)}</blockquote>"
    )

    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=inline_keyboard(rows),
    )


async def start_step_add(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, event_id: int) -> None:
    await states.set(user_id, chat_id, "max_event_step_offset", {"event_id": event_id})
    rows = []
    for offset_val, label in PRESET_OFFSETS:
        rows.append([callback_button(label, f"admin_ev_offset_preset_{event_id}_{offset_val}")])
    rows.append([callback_button("✏️ Ввести другое смещение", f"admin_ev_offset_custom_{event_id}")])
    rows.append(_back(f"admin_ev_steps_{event_id}"))

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
    await states.set(
        user_id,
        chat_id,
        "max_event_step_text",
        {"event_id": event_id, "offset_minutes": offset_val},
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
        attachments=inline_keyboard([_back(f"admin_ev_steps_{event_id}")]),
    )


async def start_step_offset_custom(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    event_id: int,
) -> None:
    await states.set(user_id, chat_id, "max_event_step_offset_custom", {"event_id": event_id})
    await client.send_message(
        chat_id=chat_id,
        text="⏱ Введите смещение (например: <code>-24h</code>, <code>-2h</code>, <code>0</code>, <code>+2h</code>, <code>-30m</code>):",
        attachments=inline_keyboard([_back(f"admin_ev_steps_{event_id}")]),
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

    try:
        validate_step_input(offset, raw_text)
        st = await add_step(async_session_maker, event_id, offset, raw_text)
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text="🎉 <b>Шаг успешно добавлен!</b>")
        await show_steps(client, chat_id, event_id)
    except Exception as exc:
        log.exception("MAX add step error: %s", exc)
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")


async def start_edit_step_text(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    step_id: int,
) -> None:
    st = await get_step_item(async_session_maker, step_id)
    if not st:
        return
    await states.set(user_id, chat_id, "max_event_step_edit_text", {"step_id": step_id})
    await client.send_message(
        chat_id=chat_id,
        text=f"✏️ <b>Редактирование текста</b>\n\nТекущий:\n<blockquote>{html.escape(st.message_text)}</blockquote>\n\nВведите новый текст:",
        attachments=inline_keyboard([_back(f"admin_ev_step_{step_id}")]),
    )


async def receive_edit_step_text(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    raw_text: str,
) -> None:
    state_obj = await states.get(user_id)
    step_id = state_obj.data["step_id"]
    try:
        updated = await update_step(async_session_maker, step_id, message_text=raw_text)
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text=f"✅ Текст обновлен (v{updated.step_version}).")
        await show_step(client, chat_id, step_id)
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")


async def start_edit_step_offset(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    step_id: int,
) -> None:
    st = await get_step_item(async_session_maker, step_id)
    if not st:
        return
    rows = []
    for offset_val, label in PRESET_OFFSETS:
        rows.append([callback_button(label, f"admin_ev_step_edit_preset_{step_id}_{offset_val}")])
    rows.append([callback_button("✏️ Ввести другое смещение", f"admin_ev_step_edit_custom_{step_id}")])
    rows.append(_back(f"admin_ev_step_{step_id}"))

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
    try:
        updated = await update_step(async_session_maker, step_id, offset_minutes=offset_val)
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text=f"✅ Время отправки обновлено: <b>{updated.offset_display}</b>")
        await show_step(client, chat_id, step_id)
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")


async def start_edit_step_offset_custom(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    step_id: int,
) -> None:
    await states.set(user_id, chat_id, "max_event_step_edit_offset_custom", {"step_id": step_id})
    await client.send_message(
        chat_id=chat_id,
        text="⏱ Введите произвольное смещение (например: <code>-24h</code>, <code>-2h</code>, <code>0</code>, <code>+2h</code>):",
        attachments=inline_keyboard([_back(f"admin_ev_step_{step_id}")]),
    )


async def receive_edit_step_offset_custom(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    raw_offset: str,
) -> None:
    state_obj = await states.get(user_id)
    step_id = state_obj.data["step_id"]
    try:
        offset = parse_offset_minutes(raw_offset)
        updated = await update_step(async_session_maker, step_id, offset_minutes=offset)
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text=f"✅ Время отправки обновлено: <b>{updated.offset_display}</b>")
        await show_step(client, chat_id, step_id)
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"❌ {exc}\nПопробуйте ещё раз:")


async def step_move_up(client: MaxApiClient, chat_id: int, step_id: int) -> None:
    st = await get_step_item(async_session_maker, step_id)
    if not st:
        return
    all_steps = await list_steps(async_session_maker, st.event_id, include_archived=False)
    ids = [s.id for s in all_steps]
    idx = ids.index(step_id)
    if idx > 0:
        ids[idx - 1], ids[idx] = ids[idx], ids[idx - 1]
        await reorder_steps(async_session_maker, st.event_id, ids)
    await show_step(client, chat_id, step_id)


async def step_move_down(client: MaxApiClient, chat_id: int, step_id: int) -> None:
    st = await get_step_item(async_session_maker, step_id)
    if not st:
        return
    all_steps = await list_steps(async_session_maker, st.event_id, include_archived=False)
    ids = [s.id for s in all_steps]
    idx = ids.index(step_id)
    if idx < len(ids) - 1:
        ids[idx + 1], ids[idx] = ids[idx], ids[idx + 1]
        await reorder_steps(async_session_maker, st.event_id, ids)
    await show_step(client, chat_id, step_id)


async def ask_delete_step(client: MaxApiClient, chat_id: int, step_id: int) -> None:
    st = await get_step_item(async_session_maker, step_id)
    if not st:
        return
    rows = [
        [callback_button("✅ Да, архивировать", f"admin_ev_step_del_yes_{step_id}")],
        _back(f"admin_ev_step_{step_id}"),
    ]
    await client.send_message(
        chat_id=chat_id,
        text=f"🗑 Удалить/архивировать шаг «<b>{st.offset_display}</b>»?\n\nОжидающие отправки этого шага будут пропущены.",
        attachments=inline_keyboard(rows),
    )


async def confirm_delete_step(client: MaxApiClient, chat_id: int, step_id: int) -> None:
    st = await get_step_item(async_session_maker, step_id)
    if not st:
        return
    await archive_step(async_session_maker, step_id)
    await client.send_message(chat_id=chat_id, text="✅ Шаг архивирован.")
    await show_steps(client, chat_id, st.event_id)


# ==============================================================================
# MONITORING & MANUAL RESOLUTION
# ==============================================================================

async def show_stats(client: MaxApiClient, chat_id: int, event_id: int) -> None:
    s = await get_scenario_summary(async_session_maker, event_id)
    if not s:
        return
    stats = await get_scenario_delivery_stats(async_session_maker, event_id)

    rows = []
    if stats.problematic_total > 0:
        rows.append([callback_button(f"⚠️ Проблемные доставки ({stats.problematic_total})", f"admin_ev_problems_{event_id}_0")])
    rows.append([callback_button("🔄 Обновить", f"admin_ev_stats_{event_id}")])
    rows.append(_back(f"admin_ev_open_{event_id}"))

    lines = [
        f"📊 <b>Мониторинг доставок: {html.escape(s.title)}</b>\n\n",
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


async def show_problems(client: MaxApiClient, chat_id: int, event_id: int, page: int = 0) -> None:
    items, total = await list_problem_deliveries(
        async_session_maker,
        event_id,
        limit=PROBLEMS_PAGE_SIZE,
        offset=page * PROBLEMS_PAGE_SIZE,
    )
    total_pages = max(1, (total + PROBLEMS_PAGE_SIZE - 1) // PROBLEMS_PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))

    rows = []
    for it in items:
        status_icon = "❓" if it.status == "uncertain" else ("🛑" if it.status == "blocked" else "❌")
        rows.append([callback_button(f"{status_icon} ID {it.id}: {it.user_display[:20]} ({it.status})", f"admin_ev_problem_item_{it.id}")])

    nav_row = []
    if page > 0:
        nav_row.append(callback_button("◀️ Назад", f"admin_ev_problems_{event_id}_{page - 1}"))
    if page < total_pages - 1:
        nav_row.append(callback_button("Вперёд ▶️", f"admin_ev_problems_{event_id}_{page + 1}"))
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


async def show_problem_detail(client: MaxApiClient, chat_id: int, delivery_id: int) -> None:
    it = await get_delivery_item(async_session_maker, delivery_id)
    if not it:
        await client.send_message(chat_id=chat_id, text="Доставка не найдена.")
        return

    rows = []
    if it.status == "uncertain":
        rows.append([callback_button("🔄 Повторить отправку (Force Retry)", f"admin_ev_resolve_force_{it.id}")])
        rows.append([callback_button("✅ Отметить отправленным (Mark Sent)", f"admin_ev_resolve_sent_{it.id}")])
        rows.append([callback_button("⏭ Отметить пропущенным (Mark Skipped)", f"admin_ev_resolve_skipped_{it.id}")])

    rows.append(_back(f"admin_ev_problems_{it.event_id}_0"))

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
        {"delivery_id": delivery_id, "res_type": res_type},
    )

    await client.send_message(
        chat_id=chat_id,
        text=f"📝 <b>Разрешение доставки ID {delivery_id}: {label}</b>\n\nВведите причину разрешения (для аудита):",
        attachments=inline_keyboard([_back(f"admin_ev_problem_item_{delivery_id}")]),
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

    try:
        deliv, ok = await resolve_uncertain_delivery(
            async_session_maker,
            delivery_id=delivery_id,
            admin_id=user_id,
            resolution_type=res_type,
            resolution_reason=reason,
        )
        await states.clear(user_id)
        await client.send_message(chat_id=chat_id, text=f"✅ Доставка ID {delivery_id} успешно разрешена ({deliv.status})!")
        await show_problem_detail(client, chat_id, delivery_id)
    except Exception as exc:
        log.exception("MAX resolution error: %s", exc)
        await client.send_message(chat_id=chat_id, text=f"❌ Ошибка: {exc}")


# ==============================================================================
# TEST SEND TO SELF
# ==============================================================================

async def show_test_send_menu(client: MaxApiClient, chat_id: int, event_id: int) -> None:
    steps = await list_steps(async_session_maker, event_id, include_archived=False)
    rows = [
        [callback_button("✉️ Подтверждение регистрации", f"admin_ev_do_test_send_{event_id}_0")],
    ]
    for idx, s in enumerate(steps):
        short = (s.message_text[:20] + "...") if len(s.message_text) > 20 else s.message_text
        rows.append([callback_button(f"✉️ Шаг {idx + 1} ({s.offset_display}): {short}", f"admin_ev_do_test_send_{event_id}_{s.id}")])

    rows.append(_back(f"admin_ev_open_{event_id}"))

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

    rendered = await render_test_step(
        async_session_maker,
        event_id=event_id,
        step_id=step_id,
        admin_user=admin_user,
    )

    await client.send_message(
        chat_id=chat_id,
        text=f"🧪 <b>[ТЕСТОВОЕ СООБЩЕНИЕ MAX]</b>\n\n{rendered}",
    )
