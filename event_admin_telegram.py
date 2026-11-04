"""
Telegram Admin UI for Event Reminder Chains.

Provides:
- Scenario management (Webinar / Live meeting creation, editing, active toggle, archive).
- Safe rescheduling using reschedule_event() backend.
- Scenario copy with start code reservation.
- Reminder step CRUD, presets & arbitrary offsets, versioning.
- Delivery monitoring with aggregate stats and problematic deliveries inspect/resolve.
- Manual resolution for uncertain deliveries (force retry, mark sent, mark skipped).
- Test send to self without registration/delivery pollution.
"""

from __future__ import annotations

import html
import logging
from datetime import datetime
from typing import Any

from aiogram import BaseMiddleware, Bot, F, Router
from aiogram.filters import Filter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from database import async_session_maker, get_all_admin_ids
from admin_html_preview import truncate_html_preview
from event_admin_service import (
    PRESET_OFFSETS,
    PROBLEM_DELIVERY_STATUSES,
    DeliveryItem,
    DeliveryPartItem,
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
from event_delivery_worker import build_test_delivery_plan, send_test_delivery_plan

log = logging.getLogger(__name__)

router = Router(name="event_admin")
_answered_callbacks: set[str] = set()


class DatabaseAdminFilter(Filter):
    async def __call__(self, event: CallbackQuery | Message) -> bool:
        user = event.from_user
        if not user:
            return False
        admin_ids = await get_all_admin_ids()
        return user.id in admin_ids


class EnsureCallbackAnsweredMiddleware(BaseMiddleware):
    async def __call__(self, handler, event: CallbackQuery, data):
        try:
            return await handler(event, data)
        finally:
            if event.id in _answered_callbacks:
                _answered_callbacks.discard(event.id)
            else:
                try:
                    await event.answer()
                except Exception:
                    pass


router.callback_query.filter(DatabaseAdminFilter())
router.message.filter(DatabaseAdminFilter())
router.callback_query.middleware(EnsureCallbackAnsweredMiddleware())


class EventAdminStates(StatesGroup):
    # Create scenario
    create_code = State()
    create_title = State()
    create_starts_at = State()
    create_duration = State()
    create_link = State()
    create_place = State()
    create_meeting_link = State()

    # Edit scenario
    edit_title = State()
    edit_duration = State()
    edit_link = State()
    edit_place = State()
    edit_description = State()
    edit_media = State()
    edit_media_add = State()

    # Reschedule
    reschedule_starts_at = State()

    # Copy
    copy_code = State()
    copy_starts_at = State()
    copy_title = State()

    # Step add/edit
    step_offset_custom = State()
    step_text = State()
    step_edit_offset_custom = State()
    step_edit_text = State()

    # Manual resolve
    resolve_reason = State()


# ==============================================================================
# KEYBOARD BUILDERS
# ==============================================================================

SCENARIOS_PAGE_SIZE = 6
PROBLEMS_PAGE_SIZE = 5


def _scenarios_list_keyboard(
    scenarios: list[ScenarioSummary],
    page: int,
    total_pages: int,
    include_archived: bool,
) -> Any:
    builder = InlineKeyboardBuilder()

    for s in scenarios:
        status_icon = "📦" if s.is_archived else ("🟢" if s.is_active else "⚪️")
        type_icon = "💻" if s.event_type == "webinar" else "📍"
        btn_text = f"{status_icon} {type_icon} {s.title} ({s.starts_at_msk[:10]})"
        builder.button(text=btn_text, callback_data=f"admin_ev_open_{s.id}_{page}_{1 if include_archived else 0}")

    builder.adjust(1)

    # Pagination row
    nav_row: list[InlineKeyboardButton] = []
    if page > 0:
        archive_flag = 1 if include_archived else 0
        nav_row.append(InlineKeyboardButton(text="⬅️ Назад", callback_data=f"admin_ev_page_{page - 1}_{archive_flag}"))
    if page < total_pages - 1:
        archive_flag = 1 if include_archived else 0
        nav_row.append(InlineKeyboardButton(text="Далее ➡️", callback_data=f"admin_ev_page_{page + 1}_{archive_flag}"))
    if nav_row:
        builder.row(*nav_row)

    # Actions row
    builder.row(
        InlineKeyboardButton(text="➕ Вебинар", callback_data="admin_ev_create_webinar"),
        InlineKeyboardButton(text="➕ Встреча", callback_data="admin_ev_create_meeting"),
    )

    archive_toggle_text = "📁 Скрыть архив" if include_archived else "📁 Показать архив"
    archive_toggle_cb = "admin_ev_toggle_archive_view_0" if include_archived else "admin_ev_toggle_archive_view_1"
    builder.row(InlineKeyboardButton(text=archive_toggle_text, callback_data=archive_toggle_cb))
    builder.row(InlineKeyboardButton(text="⬅️ В админ-панель", callback_data="admin_panel"))

    return builder.as_markup()


def _event_open_callback(event_id: int, back: str) -> str:
    if back.startswith("admin_ev_page_"):
        parts = back.split("_")
        if len(parts) == 5:
            return f"admin_ev_open_{event_id}_{parts[3]}_{parts[4]}"
    return f"admin_ev_open_{event_id}"


def _event_steps_callback(event_id: int, back: str) -> str:
    if back.startswith("admin_ev_page_"):
        parts = back.split("_")
        if len(parts) == 5:
            return f"admin_ev_steps_{event_id}_{parts[3]}_{parts[4]}"
    return f"admin_ev_steps_{event_id}"


def _event_edit_menu_callback(event_id: int, back: str) -> str:
    if back.startswith("admin_ev_page_"):
        parts = back.split("_")
        if len(parts) == 5:
            return f"admin_ev_edit_menu_{event_id}_{parts[3]}_{parts[4]}"
    return f"admin_ev_edit_menu_{event_id}"


def _scenario_card_keyboard(s: ScenarioSummary, back: str = "admin_events") -> Any:
    builder = InlineKeyboardBuilder()

    # Active / Inactive toggle
    if not s.is_archived:
        toggle_label = "⚪️ Выключить запись" if s.is_active else "🟢 Включить запись"
        builder.button(text=toggle_label, callback_data=f"admin_ev_toggle_{s.id}")

    builder.button(text="📝 Цепочка сообщений (шаги)", callback_data=_event_steps_callback(s.id, back))
    builder.button(text="📊 Статус отправок", callback_data=f"admin_ev_stats_{s.id}")

    if not s.is_archived:
        builder.button(text="⏰ Перенести время", callback_data=f"admin_ev_reschedule_{s.id}")
        builder.button(text="✏️ Редактировать", callback_data=_event_edit_menu_callback(s.id, back))
        builder.button(text="📋 Скопировать", callback_data=f"admin_ev_copy_{s.id}")
        builder.button(text="🚀 Тестовая отправка себе", callback_data=f"admin_ev_test_send_menu_{s.id}")
        builder.button(text="📦 В архив", callback_data=f"admin_ev_archive_ask_{s.id}")
    else:
        builder.button(text="📋 Скопировать в новое", callback_data=f"admin_ev_copy_{s.id}")
        builder.button(text="🚀 Тестовая отправка себе", callback_data=f"admin_ev_test_send_menu_{s.id}")

    builder.button(text="⬅️ К списку мероприятий", callback_data=back)
    builder.adjust(1)
    return builder.as_markup()


def _scenario_edit_keyboard(s: ScenarioSummary, *, list_back: str = "admin_events") -> Any:
    builder = InlineKeyboardBuilder()
    builder.button(text="✏️ Изменить название", callback_data=f"admin_ev_edit_title_{s.id}")
    builder.button(text="⏱ Изменить длительность", callback_data=f"admin_ev_edit_duration_{s.id}")
    if s.event_type == "webinar":
        builder.button(text="🔗 Изменить ссылку на вебинар", callback_data=f"admin_ev_edit_link_{s.id}")
    else:
        builder.button(text="📍 Изменить место встречи", callback_data=f"admin_ev_edit_place_{s.id}")
        link_btn = "🔗 Изменить ссылку" if s.link else "🔗 Добавить ссылку"
        builder.button(text=link_btn, callback_data=f"admin_ev_edit_link_{s.id}")

    builder.button(text="📝 Описание", callback_data=f"admin_ev_edit_description_{s.id}")
    builder.button(text="🖼 Медиа", callback_data=f"admin_ev_edit_media_{s.id}")
    builder.button(
        text=f"↕️ Порядок: {'медиа → текст' if s.media_order == 'media_top' else 'текст → медиа'}",
        callback_data=f"admin_ev_toggle_media_order_{s.id}",
    )

    builder.button(text="⬅️ Назад в карточку", callback_data=_event_open_callback(s.id, list_back))
    builder.adjust(1)
    return builder.as_markup()


def _steps_list_keyboard(
    event_id: int,
    steps: list[StepItem],
    is_archived: bool,
    *,
    back: str = "admin_events",
) -> Any:
    builder = InlineKeyboardBuilder()

    for idx, st in enumerate(steps):
        prefix = f"{idx + 1}. {st.offset_display}"
        short_text = (st.message_text[:30] + "...") if len(st.message_text) > 30 else st.message_text
        clean_text = short_text.replace("\n", " ")
        builder.button(text=f"{prefix} — {clean_text}", callback_data=f"admin_ev_step_{st.id}")

    builder.adjust(1)

    if not is_archived:
        builder.button(text="➕ Добавить шаг", callback_data=f"admin_ev_step_add_{event_id}")

    builder.button(text="⬅️ Назад в мероприятие", callback_data=_event_open_callback(event_id, back))
    builder.adjust(1)
    return builder.as_markup()


def _step_card_keyboard(
    event_id: int,
    step: StepItem,
    is_first: bool,
    is_last: bool,
    is_archived: bool,
    *,
    back: str = "admin_events",
) -> Any:
    builder = InlineKeyboardBuilder()

    if not is_archived:
        builder.button(text="✏️ Изменить текст", callback_data=f"admin_ev_step_edit_text_{step.id}")
        builder.button(text="⏱ Изменить время отправки", callback_data=f"admin_ev_step_edit_offset_{step.id}")
        builder.button(
            text=f"📎 Детали события: {'вкл.' if step.include_event_details else 'выкл.'}",
            callback_data=f"admin_ev_step_toggle_details_{step.id}",
        )

        # Reorder row
        order_row: list[InlineKeyboardButton] = []
        if not is_first:
            order_row.append(InlineKeyboardButton(text="⬆️ Выше", callback_data=f"admin_ev_step_up_{step.id}"))
        if not is_last:
            order_row.append(InlineKeyboardButton(text="⬇️ Ниже", callback_data=f"admin_ev_step_down_{step.id}"))
        if order_row:
            builder.row(*order_row)

        builder.button(text="🗑 Удалить/Архивировать шаг", callback_data=f"admin_ev_step_del_ask_{step.id}")

    builder.button(text="🚀 Тестовая отправка шага", callback_data=f"admin_ev_test_step_{step.id}")
    builder.button(text="⬅️ К списку шагов", callback_data=_event_steps_callback(event_id, back))
    builder.adjust(1)
    return builder.as_markup()


def _step_offset_picker_keyboard(event_id: int, back_cb: str) -> Any:
    builder = InlineKeyboardBuilder()
    for offset_val, label in PRESET_OFFSETS:
        builder.button(text=label, callback_data=f"admin_ev_offset_preset_{event_id}_{offset_val}")
    builder.button(text="✏️ Ввести другое смещение", callback_data=f"admin_ev_offset_custom_{event_id}")
    builder.button(text="⬅️ Отмена", callback_data=back_cb)
    builder.adjust(1)
    return builder.as_markup()


def _step_edit_offset_picker_keyboard(step_id: int) -> Any:
    builder = InlineKeyboardBuilder()
    for offset_val, label in PRESET_OFFSETS:
        builder.button(text=label, callback_data=f"admin_ev_step_edit_preset_{step_id}_{offset_val}")
    builder.button(text="✏️ Ввести другое смещение", callback_data=f"admin_ev_step_edit_custom_{step_id}")
    builder.button(text="⬅️ Отмена", callback_data=f"admin_ev_step_{step_id}")
    builder.adjust(1)
    return builder.as_markup()


def _delivery_stats_keyboard(event_id: int, stats: DeliveryStats, back: str = "admin_events") -> Any:
    builder = InlineKeyboardBuilder()
    if stats.problematic_total > 0:
        builder.button(
            text=f"⚠️ Проблемные доставки ({stats.problematic_total})",
            callback_data=f"admin_ev_problems_{event_id}_0",
        )
    builder.button(text="🔄 Обновить", callback_data=f"admin_ev_stats_{event_id}")
    builder.button(text="⬅️ Назад в мероприятие", callback_data=_event_open_callback(event_id, back))
    builder.adjust(1)
    return builder.as_markup()


def _problem_deliveries_keyboard(
    event_id: int,
    items: list[DeliveryItem],
    page: int,
    total_pages: int,
    status_filter: str | None = None,
) -> Any:
    builder = InlineKeyboardBuilder()

    for item in items:
        status_icon = "❓" if item.status == "uncertain" else ("🛑" if item.status == "blocked" else "❌")
        btn_text = f"{status_icon} ID {item.id}: {item.user_display[:20]} ({item.status})"
        item_callback = f"admin_ev_problem_item_{item.id}"
        if page > 0:
            item_callback += f"_{page}"
        builder.button(text=btn_text, callback_data=item_callback)

    builder.adjust(1)

    nav_row: list[InlineKeyboardButton] = []
    if page > 0:
        nav_row.append(InlineKeyboardButton(text="⬅️ Назад", callback_data=f"admin_ev_problems_{event_id}_{page - 1}"))
    if page < total_pages - 1:
        nav_row.append(InlineKeyboardButton(text="Далее ➡️", callback_data=f"admin_ev_problems_{event_id}_{page + 1}"))
    if nav_row:
        builder.row(*nav_row)

    builder.row(InlineKeyboardButton(text="⬅️ К статистике", callback_data=f"admin_ev_stats_{event_id}"))
    return builder.as_markup()


def _delivery_resolution_keyboard(
    item: DeliveryItem,
    parts: list[DeliveryPartItem] | None = None,
    problem_page: int = 0,
) -> Any:
    builder = InlineKeyboardBuilder()

    if item.status == "uncertain":
        uncertain_parts = [part for part in (parts or []) if part.status == "uncertain"]
        if uncertain_parts:
            for part in uncertain_parts:
                suffix = f"_{part.id}"
                builder.button(text=f"🔄 Часть {part.part_index + 1}: повторить", callback_data=f"admin_ev_resolve_force_{item.id}{suffix}")
                builder.button(text=f"✅ Часть {part.part_index + 1}: доставлено", callback_data=f"admin_ev_resolve_sent_{item.id}{suffix}")
                builder.button(text=f"⏭ Часть {part.part_index + 1}: пропустить", callback_data=f"admin_ev_resolve_skipped_{item.id}{suffix}")
        else:
            builder.button(text="🔄 Повторить отправку (Force Retry)", callback_data=f"admin_ev_resolve_force_{item.id}")
            builder.button(text="✅ Отметить отправленным (Mark Sent)", callback_data=f"admin_ev_resolve_sent_{item.id}")
            builder.button(text="⏭ Отметить пропущенным (Mark Skipped)", callback_data=f"admin_ev_resolve_skipped_{item.id}")

    builder.button(text="⬅️ К списку проблемных", callback_data=f"admin_ev_problems_{item.event_id}_{problem_page}")
    builder.adjust(1)
    return builder.as_markup()


def _test_send_picker_keyboard(
    event_id: int,
    steps: list[StepItem],
    *,
    back: str = "admin_events",
) -> Any:
    builder = InlineKeyboardBuilder()
    builder.button(text="✉️ Подтверждение регистрации", callback_data=f"admin_ev_do_test_send_{event_id}_0")
    for idx, s in enumerate(steps):
        short = (s.message_text[:25] + "...") if len(s.message_text) > 25 else s.message_text
        clean = short.replace("\n", " ")
        builder.button(
            text=f"✉️ Шаг {idx + 1} ({s.offset_display}): {clean}",
            callback_data=f"admin_ev_do_test_send_{event_id}_{s.id}",
        )
    builder.button(text="⬅️ Отмена", callback_data=_event_open_callback(event_id, back))
    builder.adjust(1)
    return builder.as_markup()


def _confirm_keyboard(yes_cb: str, no_cb: str) -> Any:
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="✅ Да, подтверждаю", callback_data=yes_cb),
        InlineKeyboardButton(text="❌ Отмена", callback_data=no_cb),
    )
    return builder.as_markup()


# ==============================================================================
# ENTRY POINT & SCENARIOS LIST
# ==============================================================================

@router.callback_query(F.data == "admin_events")
async def show_events_root(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await _show_scenarios_page(callback, page=0, include_archived=False)


@router.callback_query(F.data.startswith("admin_ev_page_"))
async def show_events_page(callback: CallbackQuery, state: FSMContext) -> None:
    data = await state.get_data()
    parts = callback.data.split("_")
    page = int(parts[3])
    inc_arch = bool(int(parts[4])) if len(parts) > 4 else bool(data.get("include_archived", False))
    await state.clear()
    await state.update_data(include_archived=inc_arch)
    await _show_scenarios_page(callback, page=page, include_archived=inc_arch)


@router.callback_query(F.data.startswith("admin_ev_toggle_archive_view_"))
async def toggle_archive_view(callback: CallbackQuery, state: FSMContext) -> None:
    inc_arch = callback.data.endswith("_1")
    await state.update_data(include_archived=inc_arch)
    await _show_scenarios_page(callback, page=0, include_archived=inc_arch)


async def _show_scenarios_page(callback: CallbackQuery, page: int, include_archived: bool) -> None:
    scenarios = await list_scenarios(async_session_maker, include_archived=include_archived)
    total_count = len(scenarios)
    total_pages = max(1, (total_count + SCENARIOS_PAGE_SIZE - 1) // SCENARIOS_PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))

    page_scenarios = scenarios[page * SCENARIOS_PAGE_SIZE : (page + 1) * SCENARIOS_PAGE_SIZE]

    text_lines = [
        "📅 <b>Сценарии мероприятий и цепочки напоминаний</b>\n",
        f"Всего мероприятий: <b>{total_count}</b> (страница {page + 1} из {total_pages})\n",
    ]

    if not page_scenarios:
        text_lines.append("<i>Мероприятия пока не созданы. Нажмите кнопку ниже, чтобы создать вебинар или очную встречу.</i>")
    else:
        for idx, s in enumerate(page_scenarios):
            status = "📦 В архиве" if s.is_archived else ("🟢 Активно" if s.is_active else "⚪️ Выключено")
            kind = "Вебинар" if s.event_type == "webinar" else "Очная встреча"
            text_lines.append(
                f"<b>{idx + 1 + page * SCENARIOS_PAGE_SIZE}. {html.escape(s.title)}</b>\n"
                f"• Код: <code>{s.code}</code> | Тип: {kind}\n"
                f"• Начало: {s.starts_at_msk}\n"
                f"• Статус: {status} | Регистраций: <b>{s.registrations_count}</b>\n"
            )

    msg = callback.message
    await msg.edit_text(
        "".join(text_lines),
        parse_mode="HTML",
        reply_markup=_scenarios_list_keyboard(page_scenarios, page, total_pages, include_archived),
    )


# ==============================================================================
# SCENARIO CARD & LIFECYCLE
# ==============================================================================

async def _get_bot_username(bot: Bot | None) -> str | None:
    if not bot:
        return None
    try:
        me = await bot.get_me()
        if hasattr(me, "username") and isinstance(me.username, str) and me.username:
            return me.username
    except Exception:
        pass
    return None


@router.callback_query(F.data.startswith("admin_ev_open_"))
async def open_scenario_card(callback: CallbackQuery, state: FSMContext) -> None:
    parts = callback.data.split("_")
    event_id = int(parts[3])
    list_back = f"admin_ev_page_{parts[4]}_{parts[5]}" if len(parts) >= 6 else "admin_events"
    await state.clear()
    await state.update_data(event_list_back=list_back)
    scen = await get_scenario_summary(async_session_maker, event_id)
    if not scen:
        await callback.answer("Мероприятие не найдено.", show_alert=True)
        return

    bot_username = await _get_bot_username(callback.bot)
    text = _format_scenario_card_text(scen, bot_username=bot_username)
    await callback.message.edit_text(
        text,
        parse_mode="HTML",
        reply_markup=_scenario_card_keyboard(scen, list_back),
    )


def _format_scenario_card_text(s: ScenarioSummary, bot_username: str | None = None) -> str:
    status_str = "📦 В архиве" if s.is_archived else ("🟢 Активно (регистрация включена)" if s.is_active else "⚪️ Выключено (регистрация приостановлена)")
    kind_str = "Вебинар" if s.event_type == "webinar" else "Очная встреча"

    loc_lines = []
    if s.event_type == "webinar":
        loc_lines.append(f"• <b>Ссылка:</b> {html.escape(s.link or '—')}")
    else:
        loc_lines.append(f"• <b>Место:</b> {html.escape(s.place or '—')}")
        if s.link:
            loc_lines.append(f"• <b>Ссылка:</b> {html.escape(s.link)}")

    loc_block = "\n".join(loc_lines)
    details_block = (
        f"• <b>Описание:</b> {html.escape(s.description[:160])}{'…' if len(s.description) > 160 else ''}\n"
        if s.description else "• <b>Описание:</b> —\n"
    )

    if bot_username:
        reg_link = f"https://t.me/{html.escape(bot_username)}?start={html.escape(s.code)}"
        code_and_link = (
            f"• <b>Ссылка для регистрации:</b> <code>{reg_link}</code>\n"
            f"• <b>Код события (/start):</b> <code>{s.code}</code>\n"
        )
    else:
        code_and_link = f"• <b>Код события (/start):</b> <code>{s.code}</code>\n"

    return (
        f"📅 <b>Карточка мероприятия: {html.escape(s.title)}</b>\n\n"
        f"{code_and_link}"
        f"• <b>Тип:</b> {kind_str}\n"
        f"• <b>Дата и время начала:</b> {s.starts_at_msk}\n"
        f"• <b>Длительность:</b> {s.duration_minutes} мин.\n"
        f"{loc_block}\n"
        f"{details_block}"
        f"• <b>Медиа:</b> {s.media_count} | порядок: {'медиа → текст' if s.media_order == 'media_top' else 'текст → медиа'}\n"
        f"• <b>Статус:</b> {status_str}\n"
        f"• <b>Всего регистраций:</b> <b>{s.registrations_count}</b>\n"
        f"• <b>Ревизия расписания:</b> {s.schedule_revision} (версия контента: {s.content_version})\n"
    )


@router.callback_query(F.data.startswith("admin_ev_toggle_"))
async def toggle_active_handler(callback: CallbackQuery, state: FSMContext | None = None) -> None:
    if callback.data.startswith("admin_ev_toggle_media_order_"):
        await toggle_media_order(callback, state)
        return
    state_data = await state.get_data() if state is not None else {}
    list_back = state_data.get("event_list_back", "admin_events")
    event_id = int(callback.data.split("_")[-1])
    try:
        new_active = await toggle_scenario_active(async_session_maker, event_id)
        msg_text = "Регистрация включена!" if new_active else "Регистрация приостановлена."
        await callback.answer(msg_text)
    except Exception as exc:
        await callback.answer(f"Ошибка: {exc}", show_alert=True)
        return

    scen = await get_scenario_summary(async_session_maker, event_id)
    if scen:
        bot_username = await _get_bot_username(callback.bot)
        await callback.message.edit_text(
            _format_scenario_card_text(scen, bot_username=bot_username),
            parse_mode="HTML",
            reply_markup=_scenario_card_keyboard(scen, list_back),
        )


@router.callback_query(F.data.startswith("admin_ev_archive_ask_"))
async def ask_archive_handler(callback: CallbackQuery, state: FSMContext | None = None) -> None:
    event_id = int(callback.data.split("_")[-1])
    state_data = await state.get_data() if state is not None else {}
    list_back = state_data.get("event_list_back", "admin_events")
    scen = await get_scenario_summary(async_session_maker, event_id)
    if not scen:
        await callback.answer("Мероприятие не найдено.", show_alert=True)
        return

    await callback.message.edit_text(
        f"📦 Вы уверены, что хотите архивировать мероприятие «<b>{html.escape(scen.title)}</b>»?\n\n"
        f"История регистраций, отправок и отчётов будет полностью сохранена. "
        f"Новые регистрации станут недоступны.",
        parse_mode="HTML",
        reply_markup=_confirm_keyboard(
            yes_cb=f"admin_ev_archive_yes_{event_id}",
            no_cb=_event_open_callback(event_id, list_back),
        ),
    )


@router.callback_query(F.data.startswith("admin_ev_archive_yes_"))
async def archive_yes_handler(callback: CallbackQuery, state: FSMContext | None = None) -> None:
    event_id = int(callback.data.split("_")[-1])
    state_data = await state.get_data() if state is not None else {}
    list_back = state_data.get("event_list_back", "admin_events")
    try:
        await archive_scenario(async_session_maker, event_id)
        await callback.answer("Мероприятие архивировано.")
    except Exception as exc:
        await callback.answer(f"Ошибка: {exc}", show_alert=True)
        return

    scen = await get_scenario_summary(async_session_maker, event_id)
    if scen:
        bot_username = await _get_bot_username(callback.bot)
        await callback.message.edit_text(
            _format_scenario_card_text(scen, bot_username=bot_username),
            parse_mode="HTML",
            reply_markup=_scenario_card_keyboard(scen, list_back),
        )


# ==============================================================================
# CREATE SCENARIO FLOW
# ==============================================================================

@router.callback_query(F.data.startswith("admin_ev_create_"))
async def start_scenario_creation(callback: CallbackQuery, state: FSMContext) -> None:
    event_type = callback.data.split("_")[-1]  # 'webinar' | 'meeting'
    await state.clear()
    await state.update_data(create_event_type=event_type)
    await state.set_state(EventAdminStates.create_code)

    type_label = "вебинара" if event_type == "webinar" else "очной встречи"
    await callback.message.edit_text(
        f"➕ <b>Создание {type_label}</b>\n\n"
        f"<b>Шаг 1 из 5:</b> Введите уникальный код события для /start ссылки.\n\n"
        f"• Только латиница, цифры, дефис и подчеркивание (например: <code>webinar_ai</code>)\n"
        f"• Длина: от 3 до 64 символов\n"
        f"• Не используйте системные префиксы (<code>topic_</code>, <code>ref_</code>, <code>ev_</code>)",
        parse_mode="HTML",
        reply_markup=InlineKeyboardBuilder().button(text="❌ Отмена", callback_data="admin_events").as_markup(),
    )


@router.message(EventAdminStates.create_code)
async def process_create_code(message: Message, state: FSMContext) -> None:
    raw_code = (message.text or "").strip()
    try:
        norm_code = validate_event_code(raw_code)
    except ValueError as exc:
        await message.answer(f"❌ Ошибка в коде: {exc}\nПопробуйте ещё раз:")
        return

    await state.update_data(create_code=norm_code)
    await state.set_state(EventAdminStates.create_title)
    await message.answer(
        f"✅ Код принят: <code>{norm_code}</code>\n\n"
        f"<b>Шаг 2 из 5:</b> Введите название мероприятия (до 255 символов):",
        parse_mode="HTML",
    )


@router.message(EventAdminStates.create_title)
async def process_create_title(message: Message, state: FSMContext) -> None:
    title = (message.text or "").strip()
    if not title or len(title) > 255:
        await message.answer("❌ Название не должно быть пустым и не может превышать 255 символов. Попробуйте ещё раз:")
        return

    await state.update_data(create_title=title)
    await state.set_state(EventAdminStates.create_starts_at)
    await message.answer(
        f"✅ Название: <b>{html.escape(title)}</b>\n\n"
        f"<b>Шаг 3 из 5:</b> Введите дату и время начала мероприятия по <b>Московскому времени (МСК)</b>.\n\n"
        f"Формат: <code>ДД.ММ.ГГГГ ЧЧ:ММ</code>\n"
        f"Пример: <code>25.10.2026 19:00</code>",
        parse_mode="HTML",
    )


@router.message(EventAdminStates.create_starts_at)
async def process_create_starts_at(message: Message, state: FSMContext) -> None:
    raw = (message.text or "").strip()
    try:
        utc_dt = parse_msk_datetime(raw)
        if utc_dt <= datetime.utcnow():
            await message.answer("❌ Мероприятие не может начинаться в прошлом. Введите будущую дату и время:")
            return
    except ValueError as exc:
        await message.answer(f"❌ {exc}\nПопробуйте ещё раз:")
        return

    await state.update_data(create_starts_at=utc_dt.isoformat(), create_starts_at_msk=raw)
    await state.set_state(EventAdminStates.create_duration)
    await message.answer(
        f"✅ Время начала: {raw} (МСК)\n\n"
        f"<b>Шаг 4 из 5:</b> Введите продолжительность мероприятия в минутах (по умолчанию 120 мин, от 15 до 1440):",
    )


@router.message(EventAdminStates.create_duration)
async def process_create_duration(message: Message, state: FSMContext) -> None:
    raw = (message.text or "").strip()
    try:
        duration = int(raw)
        if not (15 <= duration <= 1440):
            raise ValueError()
    except Exception:
        await message.answer("❌ Введите целое число минут от 15 до 1440 (например, 120):")
        return

    await state.update_data(create_duration=duration)
    data = await state.get_data()
    event_type = data["create_event_type"]

    if event_type == "webinar":
        await state.set_state(EventAdminStates.create_link)
        await message.answer(
            f"<b>Шаг 5 из 5:</b> Введите ссылку на вебинар (Zoom, YouTube, Google Meet и т.д.):",
        )
    else:
        await state.set_state(EventAdminStates.create_place)
        await message.answer(
            f"<b>Шаг 5 из 5:</b> Введите адрес / место проведения очной встречи:",
        )


@router.message(EventAdminStates.create_link)
async def process_create_link(message: Message, state: FSMContext) -> None:
    link = (message.text or "").strip()
    if not link or len(link) < 3:
        await message.answer("❌ Ссылка на вебинар обязательна. Введите корректную ссылку:")
        return

    data = await state.get_data()
    await _finalize_scenario_creation(message, state, data, place=None, link=link)


@router.message(EventAdminStates.create_place)
async def process_create_place(message: Message, state: FSMContext) -> None:
    place = (message.text or "").strip()
    if not place or len(place) < 2:
        await message.answer("❌ Место встречи обязательно. Введите адрес или название зала:")
        return

    await state.update_data(create_place=place)
    await state.set_state(EventAdminStates.create_meeting_link)

    kb = InlineKeyboardBuilder()
    kb.button(text="Пропустить ссылку", callback_data="admin_ev_skip_meeting_link")
    await message.answer(
        f"✅ Место встречи: <b>{html.escape(place)}</b>\n\n"
        f"Хотите добавить опциональную ссылку (например, на карту или сайт площадки)? "
        f"Отправьте ссылку сообщением или нажмите «Пропустить»:",
        parse_mode="HTML",
        reply_markup=kb.as_markup(),
    )


@router.callback_query(F.data == "admin_ev_skip_meeting_link")
async def skip_meeting_link(callback: CallbackQuery, state: FSMContext) -> None:
    data = await state.get_data()
    await _finalize_scenario_creation(callback.message, state, data, place=data["create_place"], link=None)


@router.message(EventAdminStates.create_meeting_link)
async def process_meeting_link(message: Message, state: FSMContext) -> None:
    link = (message.text or "").strip()
    data = await state.get_data()
    await _finalize_scenario_creation(message, state, data, place=data["create_place"], link=link)


async def _finalize_scenario_creation(
    target_msg: Message,
    state: FSMContext,
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
        await state.clear()
        bot_username = await _get_bot_username(target_msg.bot)
        card_text = _format_scenario_card_text(scen, bot_username=bot_username)
        await target_msg.answer(
            f"🎉 <b>Мероприятие успешно создано!</b>\n\n{card_text}",
            parse_mode="HTML",
            reply_markup=_scenario_card_keyboard(scen),
        )
    except Exception as exc:
        log.exception("Failed to create scenario: %s", exc)
        await target_msg.answer(
            f"❌ Ошибка при создании мероприятия: {exc}\nПопробуйте снова.",
            reply_markup=InlineKeyboardBuilder().button(text="В меню", callback_data="admin_events").as_markup(),
        )


# ==============================================================================
# EDIT SCENARIO FIELDS
# ==============================================================================

@router.callback_query(F.data.startswith("admin_ev_edit_menu_"))
async def show_edit_menu(callback: CallbackQuery, state: FSMContext) -> None:
    state_data = await state.get_data()
    parts = callback.data.split("_")
    list_back = (
        f"admin_ev_page_{parts[4]}_{parts[5]}"
        if len(parts) >= 7
        else state_data.get("event_list_back", "admin_events")
    )
    await state.clear()
    await state.update_data(event_list_back=list_back)
    event_id = int(parts[4])
    scen = await get_scenario_summary(async_session_maker, event_id)
    if not scen:
        await callback.answer("Мероприятие не найдено.", show_alert=True)
        return

    await callback.message.edit_text(
        f"✏️ <b>Редактирование мероприятия: {html.escape(scen.title)}</b>\n\n"
        f"Выберите поле, которое хотите изменить:",
        parse_mode="HTML",
        reply_markup=_scenario_edit_keyboard(scen, list_back=list_back),
    )


@router.callback_query(F.data.startswith("admin_ev_edit_title_"))
async def start_edit_title(callback: CallbackQuery, state: FSMContext) -> None:
    event_id = int(callback.data.split("_")[-1])
    state_data = await state.get_data()
    list_back = state_data.get("event_list_back", "admin_events")
    await state.clear()
    await state.update_data(edit_event_id=event_id, event_list_back=list_back)
    await state.set_state(EventAdminStates.edit_title)
    await callback.message.edit_text(
        "✏️ Введите новое название мероприятия (до 255 символов):",
        reply_markup=InlineKeyboardBuilder().button(text="⬅️ Отмена", callback_data=_event_open_callback(event_id, list_back)).as_markup(),
    )


@router.message(EventAdminStates.edit_title)
async def process_edit_title(message: Message, state: FSMContext) -> None:
    title = (message.text or "").strip()
    data = await state.get_data()
    event_id = data["edit_event_id"]
    list_back = data.get("event_list_back", "admin_events")
    try:
        scen = await update_scenario_fields(async_session_maker, event_id, title=title)
        await state.clear()
        await message.answer(
            f"✅ Название обновлено на: <b>{html.escape(scen.title)}</b>",
            parse_mode="HTML",
            reply_markup=_scenario_card_keyboard(scen, list_back),
        )
    except Exception as exc:
        await message.answer(f"❌ Ошибка: {exc}")


@router.callback_query(F.data.startswith("admin_ev_edit_duration_"))
async def start_edit_duration(callback: CallbackQuery, state: FSMContext) -> None:
    event_id = int(callback.data.split("_")[-1])
    state_data = await state.get_data()
    list_back = state_data.get("event_list_back", "admin_events")
    await state.clear()
    await state.update_data(edit_event_id=event_id, event_list_back=list_back)
    await state.set_state(EventAdminStates.edit_duration)
    await callback.message.edit_text(
        "⏱ Введите новую продолжительность мероприятия в минутах (от 15 до 1440):",
        reply_markup=InlineKeyboardBuilder().button(text="⬅️ Отмена", callback_data=_event_open_callback(event_id, list_back)).as_markup(),
    )


@router.message(EventAdminStates.edit_duration)
async def process_edit_duration(message: Message, state: FSMContext) -> None:
    raw = (message.text or "").strip()
    data = await state.get_data()
    event_id = data["edit_event_id"]
    list_back = data.get("event_list_back", "admin_events")
    try:
        dur = int(raw)
        scen = await update_scenario_fields(async_session_maker, event_id, duration_minutes=dur)
        await state.clear()
        await message.answer(
            f"✅ Длительность обновлена: {scen.duration_minutes} мин.",
            reply_markup=_scenario_card_keyboard(scen, list_back),
        )
    except Exception as exc:
        await message.answer(f"❌ Ошибка: {exc}")


@router.callback_query(F.data.startswith("admin_ev_edit_link_"))
async def start_edit_link(callback: CallbackQuery, state: FSMContext) -> None:
    event_id = int(callback.data.split("_")[-1])
    state_data = await state.get_data()
    list_back = state_data.get("event_list_back", "admin_events")
    await state.clear()
    await state.update_data(edit_event_id=event_id, event_list_back=list_back)
    await state.set_state(EventAdminStates.edit_link)
    await callback.message.edit_text(
        "🔗 Введите новую ссылку на мероприятие:",
        reply_markup=InlineKeyboardBuilder().button(text="⬅️ Отмена", callback_data=_event_open_callback(event_id, list_back)).as_markup(),
    )


@router.message(EventAdminStates.edit_link)
async def process_edit_link(message: Message, state: FSMContext) -> None:
    link = (message.text or "").strip()
    data = await state.get_data()
    event_id = data["edit_event_id"]
    list_back = data.get("event_list_back", "admin_events")
    try:
        scen = await update_scenario_fields(async_session_maker, event_id, link=link)
        await state.clear()
        await message.answer(
            f"✅ Ссылка обновлена: {html.escape(scen.link or '—')}",
            parse_mode="HTML",
            reply_markup=_scenario_card_keyboard(scen, list_back),
        )
    except Exception as exc:
        await message.answer(f"❌ Ошибка: {exc}")


@router.callback_query(F.data.startswith("admin_ev_edit_place_"))
async def start_edit_place(callback: CallbackQuery, state: FSMContext) -> None:
    event_id = int(callback.data.split("_")[-1])
    state_data = await state.get_data()
    list_back = state_data.get("event_list_back", "admin_events")
    await state.clear()
    await state.update_data(edit_event_id=event_id, event_list_back=list_back)
    await state.set_state(EventAdminStates.edit_place)
    await callback.message.edit_text(
        "📍 Введите новое место проведения встречи:",
        reply_markup=InlineKeyboardBuilder().button(text="⬅️ Отмена", callback_data=_event_open_callback(event_id, list_back)).as_markup(),
    )


@router.message(EventAdminStates.edit_place)
async def process_edit_place(message: Message, state: FSMContext) -> None:
    place = (message.text or "").strip()
    data = await state.get_data()
    event_id = data["edit_event_id"]
    list_back = data.get("event_list_back", "admin_events")
    try:
        scen = await update_scenario_fields(async_session_maker, event_id, place=place)
        await state.clear()
        await message.answer(
            f"✅ Место проведения обновлено: <b>{html.escape(scen.place or '—')}</b>",
            parse_mode="HTML",
            reply_markup=_scenario_card_keyboard(scen, list_back),
        )
    except Exception as exc:
        await message.answer(f"❌ Ошибка: {exc}")


def _description_editor_keyboard(event_id: int) -> Any:
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Сохранить и выйти", callback_data=f"admin_ev_description_save_{event_id}")
    builder.button(text="🗑 Удалить описание", callback_data=f"admin_ev_description_delete_{event_id}")
    builder.button(text="❌ Отмена", callback_data=f"admin_ev_edit_menu_{event_id}")
    builder.adjust(1)
    return builder.as_markup()


@router.callback_query(F.data.startswith("admin_ev_edit_description_"))
async def start_edit_description(callback: CallbackQuery, state: FSMContext) -> None:
    event_id = int(callback.data.split("_")[-1])
    scen = await get_scenario_summary(async_session_maker, event_id)
    if not scen:
        await callback.answer("Мероприятие не найдено.", show_alert=True)
        return
    state_data = await state.get_data()
    list_back = state_data.get("event_list_back", "admin_events")
    await state.clear()
    await state.update_data(
        edit_event_id=event_id,
        description_draft=scen.description or "",
        description_input_started=False,
        event_list_back=list_back,
    )
    await state.set_state(EventAdminStates.edit_description)
    await callback.message.edit_text(
        "📝 <b>Описание мероприятия</b>\n\n"
        "Отправьте новый текст. Следующие сообщения добавляются к черновику, чтобы сохранить описание длиннее 4096 символов. Поддерживается длинное описание без лимита caption; при доставке оно будет разделено безопасно.\n\n"
        f"Текущий черновик:\n<blockquote>{truncate_html_preview(scen.description, 2800, allowed_tags={'b', 'i', 'u', 's', 'code', 'pre', 'blockquote', 'a', 'tg-spoiler', 'strike'}) or '—'}</blockquote>",
        parse_mode="HTML",
        reply_markup=_description_editor_keyboard(event_id),
    )


@router.message(EventAdminStates.edit_description)
async def process_edit_description(message: Message, state: FSMContext) -> None:
    text = message.html_text or message.text or message.caption or ""
    data = await state.get_data()
    event_id = data["edit_event_id"]
    previous = data.get("description_draft", "") if data.get("description_input_started") else ""
    combined = f"{previous}\n{text}" if previous and text else (previous or text)
    await state.update_data(description_draft=combined, description_input_started=True)
    await message.answer(
        f"📝 Черновик обновлён ({len(combined)} символов).\n\n<blockquote>{truncate_html_preview(combined, 2800, allowed_tags={'b', 'i', 'u', 's', 'code', 'pre', 'blockquote', 'a', 'tg-spoiler', 'strike'}) or '—'}</blockquote>",
        parse_mode="HTML",
        reply_markup=_description_editor_keyboard(event_id),
    )


@router.callback_query(F.data.startswith("admin_ev_description_save_"))
async def save_description(callback: CallbackQuery, state: FSMContext) -> None:
    event_id = int(callback.data.split("_")[-1])
    data = await state.get_data()
    try:
        scen = await update_scenario_fields(
            async_session_maker,
            event_id,
            description=data.get("description_draft", ""),
        )
        list_back = data.get("event_list_back", "admin_events")
        await state.clear()
        await callback.message.edit_text(
            "✅ Описание сохранено.",
            reply_markup=_scenario_card_keyboard(scen, list_back),
        )
    except Exception as exc:
        await callback.answer(f"❌ Ошибка: {exc}", show_alert=True)


@router.callback_query(F.data.startswith("admin_ev_description_delete_"))
async def delete_description(callback: CallbackQuery, state: FSMContext) -> None:
    event_id = int(callback.data.split("_")[-1])
    await state.update_data(edit_event_id=event_id, description_draft="", description_input_started=True)
    await state.set_state(EventAdminStates.edit_description)
    await callback.message.edit_text(
        "🗑 Описание будет удалено после нажатия «Сохранить и выйти».",
        reply_markup=_description_editor_keyboard(event_id),
    )


def _media_editor_keyboard(event_id: int, draft: list[dict[str, Any]]) -> Any:
    builder = InlineKeyboardBuilder()
    for index, item in enumerate(draft):
        label = f"{index + 1}. {'Фото' if item['file_type'] == 'photo' else 'Видео'}"
        builder.button(text=f"🗑 {label}", callback_data=f"admin_ev_media_remove_{event_id}_{index}")
        if index > 0:
            builder.button(text=f"⬆️ {index + 1}", callback_data=f"admin_ev_media_up_{event_id}_{index}")
        if index < len(draft) - 1:
            builder.button(text=f"⬇️ {index + 1}", callback_data=f"admin_ev_media_down_{event_id}_{index}")
    builder.button(text="➕ Добавить фото/видео", callback_data=f"admin_ev_media_add_{event_id}")
    builder.button(text="✅ Сохранить и выйти", callback_data=f"admin_ev_media_save_{event_id}")
    builder.button(text="❌ Отмена", callback_data=f"admin_ev_edit_menu_{event_id}")
    builder.adjust(3)
    return builder.as_markup()


async def _render_media_editor(message: Message, event_id: int, state: FSMContext, *, edit: bool = True) -> None:
    data = await state.get_data()
    draft = data.get("media_draft", [])
    text = (
        "🖼 <b>Медиа мероприятия</b>\n\n"
        f"В черновике: <b>{len(draft)}</b>. Фото и видео сохраняются в заданном порядке.\n"
        "Добавьте вложения сообщениями, затем сохраните изменения."
    )
    if edit:
        await message.edit_text(text, parse_mode="HTML", reply_markup=_media_editor_keyboard(event_id, draft))
    else:
        await message.answer(text, parse_mode="HTML", reply_markup=_media_editor_keyboard(event_id, draft))


@router.callback_query(F.data.startswith("admin_ev_edit_media_"))
async def start_edit_media(callback: CallbackQuery, state: FSMContext) -> None:
    event_id = int(callback.data.split("_")[-1])
    current_state = await state.get_state()
    state_data = await state.get_data()
    list_back = state_data.get("event_list_back", "admin_events")
    preserve_draft = current_state in {EventAdminStates.edit_media.state, EventAdminStates.edit_media_add.state} and state_data.get("edit_event_id") == event_id and state_data.get("media_draft_dirty") and isinstance(state_data.get("media_draft"), list)
    if preserve_draft:
        draft = list(state_data["media_draft"])
    else:
        media = await list_scenario_media(async_session_maker, event_id)
        draft = [{"file_type": item.file_type, "file_id": item.file_id, "max_token": item.max_token, "source_platform": item.source_platform} for item in media]
    await state.clear()
    await state.update_data(edit_event_id=event_id, media_draft=draft, media_draft_dirty=preserve_draft, event_list_back=list_back)
    await state.set_state(EventAdminStates.edit_media)
    await _render_media_editor(callback.message, event_id, state)


@router.callback_query(F.data.startswith("admin_ev_media_add_"))
async def start_add_media(callback: CallbackQuery, state: FSMContext) -> None:
    event_id = int(callback.data.split("_")[-1])
    await state.update_data(edit_event_id=event_id)
    await state.set_state(EventAdminStates.edit_media_add)
    await callback.message.edit_text(
        "🖼 Отправьте фото или видео одним сообщением. Для отмены нажмите кнопку ниже.",
        reply_markup=InlineKeyboardBuilder().button(text="⬅️ К медиа", callback_data=f"admin_ev_edit_media_{event_id}").as_markup(),
    )


@router.message(EventAdminStates.edit_media_add)
async def receive_media(message: Message, state: FSMContext) -> None:
    file_type = None
    file_id = None
    if getattr(message, "photo", None):
        file_type = "photo"
        file_id = message.photo[-1].file_id
    elif getattr(message, "video", None):
        file_type = "video"
        file_id = message.video.file_id
    if not file_type or not file_id:
        await message.answer("❌ Отправьте фото или видео.")
        return
    data = await state.get_data()
    draft = list(data.get("media_draft", []))
    draft.append({"file_type": file_type, "file_id": file_id, "max_token": None, "source_platform": "telegram"})
    await state.update_data(media_draft=draft, media_draft_dirty=True)
    await state.set_state(EventAdminStates.edit_media)
    await _render_media_editor(message, data["edit_event_id"], state, edit=False)


async def _mutate_media_draft(callback: CallbackQuery, state: FSMContext, direction: int | None = None) -> None:
    parts = callback.data.split("_")
    event_id = int(parts[4])
    index = int(parts[5]) if len(parts) > 5 else None
    data = await state.get_data()
    draft = list(data.get("media_draft", []))
    if index is not None and 0 <= index < len(draft):
        if direction is None:
            draft.pop(index)
        else:
            target = index + direction
            if 0 <= target < len(draft):
                draft[index], draft[target] = draft[target], draft[index]
    await state.update_data(media_draft=draft, media_draft_dirty=True)
    await _render_media_editor(callback.message, event_id, state)


@router.callback_query(F.data.startswith("admin_ev_media_remove_"))
async def remove_media(callback: CallbackQuery, state: FSMContext) -> None:
    await _mutate_media_draft(callback, state)


@router.callback_query(F.data.startswith("admin_ev_media_up_"))
async def move_media_up(callback: CallbackQuery, state: FSMContext) -> None:
    await _mutate_media_draft(callback, state, -1)


@router.callback_query(F.data.startswith("admin_ev_media_down_"))
async def move_media_down(callback: CallbackQuery, state: FSMContext) -> None:
    await _mutate_media_draft(callback, state, 1)


@router.callback_query(F.data.startswith("admin_ev_media_save_"))
async def save_media(callback: CallbackQuery, state: FSMContext) -> None:
    event_id = int(callback.data.split("_")[-1])
    data = await state.get_data()
    try:
        await replace_scenario_media(async_session_maker, event_id, data.get("media_draft", []))
        scen = await get_scenario_summary(async_session_maker, event_id)
        await state.clear()
        await callback.message.edit_text("✅ Медиа сохранены.", reply_markup=_scenario_card_keyboard(scen, data.get("event_list_back", "admin_events")))
    except Exception as exc:
        await callback.answer(f"❌ Ошибка: {exc}", show_alert=True)


@router.callback_query(F.data.startswith("admin_ev_toggle_media_order_"))
async def toggle_media_order(callback: CallbackQuery, state: FSMContext | None = None) -> None:
    event_id = int(callback.data.split("_")[-1])
    state_data = await state.get_data() if state is not None else {}
    list_back = state_data.get("event_list_back", "admin_events")
    scen = await get_scenario_summary(async_session_maker, event_id)
    if not scen:
        await callback.answer("Мероприятие не найдено.", show_alert=True)
        return
    updated = await update_scenario_fields(
        async_session_maker,
        event_id,
        media_order="text_top" if scen.media_order == "media_top" else "media_top",
    )
    await callback.message.edit_text(
        f"✅ Порядок изменён: {'текст → медиа' if updated.media_order == 'text_top' else 'медиа → текст'}",
        reply_markup=_scenario_edit_keyboard(updated, list_back=list_back),
    )


# ==============================================================================
# RESCHEDULE EVENT TIME
# ==============================================================================

@router.callback_query(F.data.startswith("admin_ev_reschedule_"))
async def start_reschedule(callback: CallbackQuery, state: FSMContext) -> None:
    event_id = int(callback.data.split("_")[-1])
    scen = await get_scenario_summary(async_session_maker, event_id)
    if not scen:
        await callback.answer("Мероприятие не найдено.", show_alert=True)
        return

    await state.clear()
    await state.update_data(reschedule_event_id=event_id)
    await state.set_state(EventAdminStates.reschedule_starts_at)

    await callback.message.edit_text(
        f"⏰ <b>Перенос времени мероприятия: {html.escape(scen.title)}</b>\n\n"
        f"• Текущее время начала: <b>{scen.starts_at_msk}</b>\n"
        f"• Активных регистраций: <b>{scen.registrations_count}</b>\n\n"
        f"Введите новое время начала по <b>Московскому времени (МСК)</b>:\n"
        f"Формат: <code>ДД.ММ.ГГГГ ЧЧ:ММ</code>\n"
        f"Пример: <code>28.10.2026 20:00</code>",
        parse_mode="HTML",
        reply_markup=InlineKeyboardBuilder().button(text="⬅️ Отмена", callback_data=f"admin_ev_open_{event_id}").as_markup(),
    )


@router.message(EventAdminStates.reschedule_starts_at)
async def process_reschedule(message: Message, state: FSMContext) -> None:
    raw = (message.text or "").strip()
    data = await state.get_data()
    event_id = data["reschedule_event_id"]

    try:
        utc_dt = parse_msk_datetime(raw)
        res = await reschedule_scenario(async_session_maker, event_id, utc_dt)
        await state.clear()

        scen = await get_scenario_summary(async_session_maker, event_id)
        assert scen is not None
        await message.answer(
            f"🎉 <b>Мероприятие успешно перенесено!</b>\n\n"
            f"• Новое время: <b>{format_msk_datetime(res.new_starts_at)}</b>\n"
            f"• Ревизия расписания: <b>{res.schedule_revision}</b>\n"
            f"• Затронуто регистраций: <b>{res.registrations_count}</b>\n"
            f"Пользователям запланированы актуальные уведомления о переносе.",
            parse_mode="HTML",
            reply_markup=_scenario_card_keyboard(scen),
        )
    except Exception as exc:
        log.exception("Reschedule failed: %s", exc)
        await message.answer(
            f"❌ Ошибка переноса времени: {exc}\nПопробуйте ещё раз (или нажмите отмену в карточке):",
            reply_markup=InlineKeyboardBuilder().button(text="⬅️ В карточку", callback_data=f"admin_ev_open_{event_id}").as_markup(),
        )


# ==============================================================================
# COPY SCENARIO
# ==============================================================================

@router.callback_query(F.data.startswith("admin_ev_copy_"))
async def start_copy(callback: CallbackQuery, state: FSMContext) -> None:
    source_id = int(callback.data.split("_")[-1])
    scen = await get_scenario_summary(async_session_maker, source_id)
    if not scen:
        await callback.answer("Мероприятие не найдено.", show_alert=True)
        return

    await state.clear()
    await state.update_data(copy_source_id=source_id)
    await state.set_state(EventAdminStates.copy_code)

    await callback.message.edit_text(
        f"📋 <b>Копирование сценария: {html.escape(scen.title)}</b>\n\n"
        f"Будут скопированы все параметры и шаги напоминаний. Старые регистрации и отправки НЕ копируются.\n\n"
        f"<b>Шаг 1 из 2:</b> Введите уникальный код нового сценария (например: <code>{scen.code}_next</code>):",
        parse_mode="HTML",
        reply_markup=InlineKeyboardBuilder().button(text="⬅️ Отмена", callback_data=f"admin_ev_open_{source_id}").as_markup(),
    )


@router.message(EventAdminStates.copy_code)
async def process_copy_code(message: Message, state: FSMContext) -> None:
    code = (message.text or "").strip()
    try:
        norm_code = validate_event_code(code)
    except ValueError as exc:
        await message.answer(f"❌ Ошибка в коде: {exc}\nПопробуйте снова:")
        return

    await state.update_data(copy_code=norm_code)
    await state.set_state(EventAdminStates.copy_starts_at)
    await message.answer(
        f"✅ Код: <code>{norm_code}</code>\n\n"
        f"<b>Шаг 2 из 2:</b> Введите дату и время начала нового мероприятия по <b>Московскому времени (МСК)</b>:\n"
        f"Формат: <code>ДД.ММ.ГГГГ ЧЧ:ММ</code>\n"
        f"Пример: <code>01.11.2026 19:00</code>",
        parse_mode="HTML",
    )


@router.message(EventAdminStates.copy_starts_at)
async def process_copy_starts_at(message: Message, state: FSMContext) -> None:
    raw = (message.text or "").strip()
    data = await state.get_data()
    source_id = data["copy_source_id"]
    new_code = data["copy_code"]

    try:
        utc_dt = parse_msk_datetime(raw)
        new_scen = await copy_scenario(
            async_session_maker,
            source_event_id=source_id,
            new_code=new_code,
            new_starts_at_utc=utc_dt,
        )
        await state.clear()
        bot_username = await _get_bot_username(message.bot)
        card_text = _format_scenario_card_text(new_scen, bot_username=bot_username)
        await message.answer(
            f"🎉 <b>Сценарий успешно скопирован!</b>\n\n{card_text}",
            parse_mode="HTML",
            reply_markup=_scenario_card_keyboard(new_scen),
        )
    except Exception as exc:
        log.exception("Copy failed: %s", exc)
        await message.answer(f"❌ Ошибка копирования: {exc}\nПопробуйте снова:")


# ==============================================================================
# REMINDER STEPS CRUD
# ==============================================================================

@router.callback_query(F.data.startswith("admin_ev_steps_"))
async def show_steps_list(callback: CallbackQuery, state: FSMContext) -> None:
    state_data = await state.get_data()
    parts = callback.data.split("_")
    event_id = int(parts[3])
    list_back = (
        f"admin_ev_page_{parts[4]}_{parts[5]}"
        if len(parts) >= 6
        else state_data.get("event_list_back", "admin_events")
    )
    await state.clear()
    await state.update_data(event_list_back=list_back)
    scen = await get_scenario_summary(async_session_maker, event_id)
    if not scen:
        await callback.answer("Мероприятие не найдено.", show_alert=True)
        return

    steps = await list_steps(async_session_maker, event_id, include_archived=False)

    text_lines = [
        f"📝 <b>Цепочка сообщений-напоминаний</b>\n",
        f"Мероприятие: <b>{html.escape(scen.title)}</b>\n\n",
    ]

    if not steps:
        text_lines.append(
            "<i>Шагов напоминаний пока нет. Добавьте шаги (например, за 24 ч. до начала, "
            "за 2 ч., в момент начала и т.д.).</i>"
        )
    else:
        text_lines.append(f"Всего шагов: <b>{len(steps)}</b>. Нажмите на шаг для просмотра и редактирования:\n\n")
        for idx, st in enumerate(steps):
            text_lines.append(
                f"<b>{idx + 1}. {st.offset_display}</b> (v{st.step_version})\n"
                f"<blockquote>{html.escape(st.message_text[:100])}</blockquote>\n"
            )

    await callback.message.edit_text(
        "".join(text_lines),
        parse_mode="HTML",
        reply_markup=_steps_list_keyboard(event_id, steps, scen.is_archived, back=list_back),
    )


@router.callback_query(F.data.startswith("admin_ev_step_add_"))
async def start_add_step(callback: CallbackQuery, state: FSMContext) -> None:
    state_data = await state.get_data()
    list_back = state_data.get("event_list_back", "admin_events")
    event_id = int(callback.data.split("_")[-1])
    await state.clear()
    await state.update_data(step_event_id=event_id, event_list_back=list_back)

    await callback.message.edit_text(
        "➕ <b>Добавление шага напоминания</b>\n\n"
        "<b>Шаг 1 из 2:</b> Выберите время отправки относительно начала мероприятия "
        "или выберите ручной ввод смещения:",
        parse_mode="HTML",
        reply_markup=_step_offset_picker_keyboard(event_id, back_cb=_event_steps_callback(event_id, list_back)),
    )


@router.callback_query(F.data.startswith("admin_ev_offset_preset_"))
async def process_offset_preset(callback: CallbackQuery, state: FSMContext) -> None:
    parts = callback.data.split("_")
    event_id = int(parts[4])
    offset_val = int(parts[5])
    state_data = await state.get_data()
    list_back = state_data.get("event_list_back", "admin_events")

    await state.update_data(step_event_id=event_id, step_offset=offset_val)
    await state.set_state(EventAdminStates.step_text)

    disp = format_offset_minutes(offset_val)
    await callback.message.edit_text(
        f"✅ Время отправки: <b>{disp}</b>\n\n"
        f"<b>Шаг 2 из 2:</b> Введите текст напоминания.\n\n"
        f"Поддерживаемые плейсхолдеры:\n"
        f"• <code>{{name}}</code> — имя пользователя (или «друг»)\n"
        f"• <code>{{event_title}}</code> — название мероприятия\n"
        f"• <code>{{event_date}}</code> — дата начала (ДД.ММ.ГГГГ)\n"
        f"• <code>{{event_time}}</code> — время начала (ЧЧ:ММ по МСК)\n"
        f"• <code>{{link}}</code> — ссылка на мероприятие\n"
        f"• <code>{{place}}</code> — место встречи\n",
        parse_mode="HTML",
        reply_markup=InlineKeyboardBuilder().button(
            text="⬅️ Отмена", callback_data=_event_steps_callback(event_id, list_back)
        ).as_markup(),
    )


@router.callback_query(F.data.startswith("admin_ev_offset_custom_"))
async def start_offset_custom(callback: CallbackQuery, state: FSMContext) -> None:
    event_id = int(callback.data.split("_")[-1])
    state_data = await state.get_data()
    await state.update_data(
        step_event_id=event_id,
        event_list_back=state_data.get("event_list_back", "admin_events"),
    )
    await state.set_state(EventAdminStates.step_offset_custom)

    await callback.message.edit_text(
        "⏱ <b>Ввод произвольного смещения</b>\n\n"
        "Введите время отправки относительно начала мероприятия:\n"
        "• <code>-24h</code> или <code>-1440</code> — за 24 ч. до начала\n"
        "• <code>-2h</code> или <code>-120</code> — за 2 ч. до начала\n"
        "• <code>0</code> — в момент начала\n"
        "• <code>+2h</code> или <code>120</code> — через 2 ч. после начала\n"
        "• <code>-30m</code> — за 30 мин. до начала",
        parse_mode="HTML",
        reply_markup=InlineKeyboardBuilder().button(
            text="⬅️ Отмена",
            callback_data=_event_steps_callback(event_id, state_data.get("event_list_back", "admin_events")),
        ).as_markup(),
    )


@router.message(EventAdminStates.step_offset_custom)
async def process_offset_custom(message: Message, state: FSMContext) -> None:
    raw = (message.text or "").strip()
    data = await state.get_data()
    event_id = data["step_event_id"]
    try:
        offset = parse_offset_minutes(raw)
    except ValueError as exc:
        await message.answer(f"❌ {exc}\nПопробуйте ещё раз:")
        return

    await state.update_data(step_offset=offset)
    await state.set_state(EventAdminStates.step_text)
    disp = format_offset_minutes(offset)
    await message.answer(
        f"✅ Время отправки: <b>{disp}</b> ({offset} мин.)\n\n"
        f"<b>Шаг 2 из 2:</b> Введите текст напоминания:\n"
        f"(доступны <code>{{name}}</code>, <code>{{event_title}}</code>, <code>{{event_date}}</code>, <code>{{event_time}}</code>, <code>{{link}}</code>)",
        parse_mode="HTML",
    )


@router.message(EventAdminStates.step_text)
async def process_step_text(message: Message, state: FSMContext) -> None:
    text = (message.text or "").strip()
    data = await state.get_data()
    event_id = data["step_event_id"]
    offset = data["step_offset"]
    list_back = data.get("event_list_back", "admin_events")

    try:
        validate_step_input(offset, text)
        st = await add_step(async_session_maker, event_id, offset, text)
        await state.clear()
        await state.update_data(event_list_back=list_back)
        scen = await get_scenario_summary(async_session_maker, event_id)
        assert scen is not None
        steps = await list_steps(async_session_maker, event_id, include_archived=False)
        await message.answer(
            f"🎉 <b>Шаг напоминания успешно добавлен!</b>\n\n"
            f"• Время: <b>{st.offset_display}</b>\n"
            f"• Версия: v{st.step_version}\n"
            f"• Текст:\n<blockquote>{html.escape(st.message_text)}</blockquote>",
            parse_mode="HTML",
            reply_markup=_steps_list_keyboard(event_id, steps, scen.is_archived, back=list_back),
        )
    except Exception as exc:
        log.exception("Add step failed: %s", exc)
        await message.answer(f"❌ Ошибка: {exc}")


async def _render_step_card_view(
    callback: CallbackQuery,
    step_id: int,
    *,
    back: str = "admin_events",
) -> None:
    step = await get_step_item(async_session_maker, step_id)
    if not step:
        await callback.answer("Шаг не найден.", show_alert=True)
        return

    scen = await get_scenario_summary(async_session_maker, step.event_id)
    if not scen:
        await callback.answer("Мероприятие не найдено.", show_alert=True)
        return

    all_steps = await list_steps(async_session_maker, step.event_id, include_archived=False)
    idx = next((i for i, s in enumerate(all_steps) if s.id == step.id), 0)
    is_first = (idx == 0)
    is_last = (idx == len(all_steps) - 1)

    text = (
        f"📝 <b>Шаг напоминания {idx + 1} из {len(all_steps)}</b>\n\n"
        f"• <b>Мероприятие:</b> {html.escape(scen.title)}\n"
        f"• <b>Время отправки:</b> <b>{step.offset_display}</b> ({step.offset_minutes} мин.)\n"
        f"• <b>Версия шага:</b> v{step.step_version}\n\n"
        f"• <b>Детали события:</b> {'включены' if step.include_event_details else 'выключены'}\n\n"
        f"<b>Текст сообщения:</b>\n"
        f"<blockquote>{html.escape(step.message_text)}</blockquote>"
    )

    if callback.message:
        await callback.message.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=_step_card_keyboard(
                step.event_id, step, is_first, is_last, scen.is_archived, back=back
            ),
        )


@router.callback_query(F.data.regexp(r"^admin_ev_step_(\d+)$"))
async def open_step_card(callback: CallbackQuery, state: FSMContext) -> None:
    state_data = await state.get_data()
    list_back = state_data.get("event_list_back", "admin_events")
    await state.clear()
    await state.update_data(event_list_back=list_back)
    step_id = int(callback.data.split("_")[-1])
    await _render_step_card_view(callback, step_id, back=list_back)


@router.callback_query(F.data.startswith("admin_ev_step_toggle_details_"))
async def toggle_step_event_details(callback: CallbackQuery, state: FSMContext | None = None) -> None:
    state_data = await state.get_data() if state is not None else {}
    list_back = state_data.get("event_list_back", "admin_events")
    step_id = int(callback.data.split("_")[-1])
    step = await get_step_item(async_session_maker, step_id)
    if not step:
        await callback.answer("Шаг не найден.", show_alert=True)
        return
    try:
        await set_step_event_details(async_session_maker, step_id, not step.include_event_details)
        await _render_step_card_view(callback, step_id, back=list_back)
    except Exception as exc:
        await callback.answer(f"❌ Ошибка: {exc}", show_alert=True)


@router.callback_query(F.data.startswith("admin_ev_step_edit_text_"))
async def start_edit_step_text(callback: CallbackQuery, state: FSMContext) -> None:
    step_id = int(callback.data.split("_")[-1])
    step = await get_step_item(async_session_maker, step_id)
    if not step:
        await callback.answer("Шаг не найден.", show_alert=True)
        return

    state_data = await state.get_data()
    list_back = state_data.get("event_list_back", "admin_events")
    await state.clear()
    await state.update_data(
        edit_step_id=step_id,
        edit_step_event_id=step.event_id,
        event_list_back=list_back,
    )
    await state.set_state(EventAdminStates.step_edit_text)

    await callback.message.edit_text(
        f"✏️ <b>Редактирование текста шага ({step.offset_display})</b>\n\n"
        f"Текущий текст:\n<blockquote>{html.escape(step.message_text)}</blockquote>\n\n"
        f"Введите новый текст сообщения:",
        parse_mode="HTML",
        reply_markup=InlineKeyboardBuilder().button(text="⬅️ Отмена", callback_data=f"admin_ev_step_{step_id}").as_markup(),
    )


@router.message(EventAdminStates.step_edit_text)
async def process_edit_step_text(message: Message, state: FSMContext) -> None:
    text = (message.text or "").strip()
    data = await state.get_data()
    step_id = data["edit_step_id"]
    list_back = data.get("event_list_back", "admin_events")

    try:
        updated = await update_step(async_session_maker, step_id, message_text=text)
        await state.clear()
        await state.update_data(event_list_back=list_back)
        scen = await get_scenario_summary(async_session_maker, updated.event_id)
        assert scen is not None
        all_steps = await list_steps(async_session_maker, updated.event_id, include_archived=False)
        idx = next((i for i, s in enumerate(all_steps) if s.id == updated.id), 0)

        await message.answer(
            f"✅ Текст шага обновлен (новая версия: v{updated.step_version})!\n\n"
            f"<blockquote>{html.escape(updated.message_text)}</blockquote>",
            parse_mode="HTML",
            reply_markup=_step_card_keyboard(
                updated.event_id,
                updated,
                idx == 0,
                idx == len(all_steps) - 1,
                scen.is_archived,
                back=list_back,
            ),
        )
    except Exception as exc:
        await message.answer(f"❌ Ошибка: {exc}")


@router.callback_query(F.data.startswith("admin_ev_step_edit_offset_"))
async def start_edit_step_offset(callback: CallbackQuery, state: FSMContext) -> None:
    step_id = int(callback.data.split("_")[-1])
    step = await get_step_item(async_session_maker, step_id)
    if not step:
        await callback.answer("Шаг не найден.", show_alert=True)
        return

    state_data = await state.get_data()
    list_back = state_data.get("event_list_back", "admin_events")
    await state.clear()
    await state.update_data(
        edit_step_id=step_id,
        edit_step_event_id=step.event_id,
        event_list_back=list_back,
    )

    await callback.message.edit_text(
        f"⏱ <b>Изменение времени отправки шага</b>\n\n"
        f"Текущее смещение: <b>{step.offset_display}</b>\n\n"
        f"Выберите новое время отправки или введите вручную:",
        parse_mode="HTML",
        reply_markup=_step_edit_offset_picker_keyboard(step_id),
    )


@router.callback_query(F.data.startswith("admin_ev_step_edit_preset_"))
async def process_edit_offset_preset(callback: CallbackQuery, state: FSMContext) -> None:
    parts = callback.data.split("_")
    step_id = int(parts[5])
    offset_val = int(parts[6])
    state_data = await state.get_data()
    list_back = state_data.get("event_list_back", "admin_events")

    try:
        updated = await update_step(async_session_maker, step_id, offset_minutes=offset_val)
        await state.clear()
        await state.update_data(event_list_back=list_back)
        scen = await get_scenario_summary(async_session_maker, updated.event_id)
        assert scen is not None
        all_steps = await list_steps(async_session_maker, updated.event_id, include_archived=False)
        idx = next((i for i, s in enumerate(all_steps) if s.id == updated.id), 0)

        await callback.message.edit_text(
            f"✅ Время отправки обновлено: <b>{updated.offset_display}</b> (v{updated.step_version})",
            parse_mode="HTML",
            reply_markup=_step_card_keyboard(
                updated.event_id,
                updated,
                idx == 0,
                idx == len(all_steps) - 1,
                scen.is_archived,
                back=list_back,
            ),
        )
    except Exception as exc:
        await callback.answer(f"❌ Ошибка: {exc}", show_alert=True)


@router.callback_query(F.data.startswith("admin_ev_step_edit_custom_"))
async def start_edit_offset_custom(callback: CallbackQuery, state: FSMContext) -> None:
    step_id = int(callback.data.split("_")[-1])
    state_data = await state.get_data()
    await state.update_data(
        edit_step_id=step_id,
        event_list_back=state_data.get("event_list_back", "admin_events"),
    )
    await state.set_state(EventAdminStates.step_edit_offset_custom)

    await callback.message.edit_text(
        "⏱ Введите произвольное смещение (например: <code>-24h</code>, <code>-2h</code>, <code>0</code>, <code>+2h</code>, <code>-30m</code>):",
        parse_mode="HTML",
        reply_markup=InlineKeyboardBuilder().button(
            text="⬅️ Отмена", callback_data=f"admin_ev_step_{step_id}"
        ).as_markup(),
    )


@router.message(EventAdminStates.step_edit_offset_custom)
async def process_edit_offset_custom(message: Message, state: FSMContext) -> None:
    raw = (message.text or "").strip()
    data = await state.get_data()
    step_id = data["edit_step_id"]
    list_back = data.get("event_list_back", "admin_events")

    try:
        offset = parse_offset_minutes(raw)
        updated = await update_step(async_session_maker, step_id, offset_minutes=offset)
        await state.clear()
        await state.update_data(event_list_back=list_back)
        scen = await get_scenario_summary(async_session_maker, updated.event_id)
        assert scen is not None
        all_steps = await list_steps(async_session_maker, updated.event_id, include_archived=False)
        idx = next((i for i, s in enumerate(all_steps) if s.id == updated.id), 0)

        await message.answer(
            f"✅ Время отправки обновлено: <b>{updated.offset_display}</b> (v{updated.step_version})",
            parse_mode="HTML",
            reply_markup=_step_card_keyboard(
                updated.event_id,
                updated,
                idx == 0,
                idx == len(all_steps) - 1,
                scen.is_archived,
                back=list_back,
            ),
        )
    except Exception as exc:
        await message.answer(f"❌ Ошибка: {exc}")


@router.callback_query(F.data.startswith("admin_ev_step_up_"))
async def step_move_up(callback: CallbackQuery, state: FSMContext | None = None) -> None:
    state_data = await state.get_data() if state is not None else {}
    list_back = state_data.get("event_list_back", "admin_events")
    step_id = int(callback.data.split("_")[-1])
    step = await get_step_item(async_session_maker, step_id)
    if not step:
        await callback.answer("Шаг не найден.", show_alert=True)
        return
    all_steps = await list_steps(async_session_maker, step.event_id, include_archived=False)
    ids = [s.id for s in all_steps]
    if step_id in ids:
        idx = ids.index(step_id)
        if idx > 0:
            ids[idx - 1], ids[idx] = ids[idx], ids[idx - 1]
            await reorder_steps(async_session_maker, step.event_id, ids)
            await callback.answer("Перемещено выше")
    await _render_step_card_view(callback, step_id, back=list_back)


@router.callback_query(F.data.startswith("admin_ev_step_down_"))
async def step_move_down(callback: CallbackQuery, state: FSMContext | None = None) -> None:
    state_data = await state.get_data() if state is not None else {}
    list_back = state_data.get("event_list_back", "admin_events")
    step_id = int(callback.data.split("_")[-1])
    step = await get_step_item(async_session_maker, step_id)
    if not step:
        await callback.answer("Шаг не найден.", show_alert=True)
        return
    all_steps = await list_steps(async_session_maker, step.event_id, include_archived=False)
    ids = [s.id for s in all_steps]
    if step_id in ids:
        idx = ids.index(step_id)
        if idx < len(ids) - 1:
            ids[idx + 1], ids[idx] = ids[idx], ids[idx + 1]
            await reorder_steps(async_session_maker, step.event_id, ids)
            await callback.answer("Перемещено ниже")
    await _render_step_card_view(callback, step_id, back=list_back)


@router.callback_query(F.data.startswith("admin_ev_step_del_ask_"))
async def ask_delete_step(callback: CallbackQuery, state: FSMContext | None = None) -> None:
    step_id = int(callback.data.split("_")[-1])
    step = await get_step_item(async_session_maker, step_id)
    if not step:
        return
    await callback.message.edit_text(
        f"🗑 Удалить/архивировать шаг «<b>{step.offset_display}</b>»?\n\n"
        f"Ещё не отправленные запланированные доставки для этого шага будут пропущены.",
        parse_mode="HTML",
        reply_markup=_confirm_keyboard(
            yes_cb=f"admin_ev_step_del_yes_{step_id}",
            no_cb=f"admin_ev_step_{step_id}",
        ),
    )


@router.callback_query(F.data.startswith("admin_ev_step_del_yes_"))
async def delete_step_yes(callback: CallbackQuery, state: FSMContext | None = None) -> None:
    state_data = await state.get_data() if state is not None else {}
    list_back = state_data.get("event_list_back", "admin_events")
    step_id = int(callback.data.split("_")[-1])
    step = await get_step_item(async_session_maker, step_id)
    if not step:
        return
    await archive_step(async_session_maker, step_id)
    await callback.answer("Шаг архивирован.")

    scen = await get_scenario_summary(async_session_maker, step.event_id)
    assert scen is not None
    steps = await list_steps(async_session_maker, step.event_id, include_archived=False)
    await callback.message.edit_text(
        f"Шаг успешно удалён.\n\nВсего активных шагов: {len(steps)}",
        reply_markup=_steps_list_keyboard(step.event_id, steps, scen.is_archived, back=list_back),
    )


# ==============================================================================
# MONITORING & MANUAL RESOLUTION
# ==============================================================================

@router.callback_query(F.data.startswith("admin_ev_stats_"))
async def show_delivery_stats(callback: CallbackQuery, state: FSMContext | None = None) -> None:
    event_id = int(callback.data.split("_")[-1])
    state_data = await state.get_data() if state is not None else {}
    list_back = state_data.get("event_list_back", "admin_events")
    scen = await get_scenario_summary(async_session_maker, event_id)
    if not scen:
        await callback.answer("Мероприятие не найдено.", show_alert=True)
        return

    stats = await get_scenario_delivery_stats(async_session_maker, event_id)

    status_lines = [
        f"📊 <b>Мониторинг отправок: {html.escape(scen.title)}</b>\n",
        f"• Всего регистраций: <b>{scen.registrations_count}</b>\n",
        f"• Всего запланированных/созданных доставок: <b>{stats.total}</b>\n\n",
        f"<b>Агрегаты по статусам:</b>\n",
        f"• Ожидают отправки (pending): <b>{stats.pending}</b>\n",
        f"• Захвачены воркером (claimed): <b>{stats.claimed}</b>\n",
        f"• Отправляются (sending): <b>{stats.sending}</b>\n",
        f"• Успешно отправлены (sent): <b>{stats.sent}</b>\n",
        f"• Пропущены (skipped): <b>{stats.skipped}</b>\n",
        f"• Заблокированы пользователем (blocked): <b>{stats.blocked}</b>\n",
        f"• Ошибки отправки (failed): <b>{stats.failed}</b>\n",
        f"• Неопределённый статус (uncertain): <b>{stats.uncertain}</b>\n",
        f"• Разрешено вручную: доставлено (resolved_sent): <b>{stats.resolved_sent}</b>\n",
        f"• Разрешено вручную: пропущено (resolved_skipped): <b>{stats.resolved_skipped}</b>\n",
    ]

    if stats.problematic_total > 0:
        status_lines.append(
            f"\n⚠️ <b>Требуют внимания ({stats.problematic_total}):</b>\n"
            f"Неопределённых (uncertain): {stats.uncertain}, Ошибок: {stats.failed}, Блокировок: {stats.blocked}"
        )

    await callback.message.edit_text(
        "".join(status_lines),
        parse_mode="HTML",
        reply_markup=_delivery_stats_keyboard(event_id, stats, list_back),
    )


@router.callback_query(F.data.startswith("admin_ev_problems_"))
async def show_problem_deliveries(callback: CallbackQuery) -> None:
    parts = callback.data.split("_")
    event_id = int(parts[3])
    page = max(0, int(parts[4]))

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

    text_lines = [
        f"⚠️ <b>Проблемные доставки ({total})</b>\n",
        f"Страница {page + 1} из {total_pages}\n\n",
    ]

    if not items:
        text_lines.append("<i>Проблемных доставок нет.</i>")
    else:
        for it in items:
            text_lines.append(
                f"• <b>Доставка ID {it.id}</b> | {html.escape(it.user_display)}\n"
                f"  Статус: <b>{it.status}</b> | Тип: {it.message_kind}\n"
                f"  Время: {it.due_at_msk} | Попыток: {it.attempt_count}\n"
                f"  Ошибка: <code>{html.escape((it.last_error or '—')[:80])}</code>\n\n"
            )

    await callback.message.edit_text(
        "".join(text_lines),
        parse_mode="HTML",
        reply_markup=_problem_deliveries_keyboard(event_id, items, page, total_pages),
    )


@router.callback_query(F.data.startswith("admin_ev_problem_item_"))
async def open_problem_delivery(callback: CallbackQuery, state: FSMContext | None = None) -> None:
    callback_parts = callback.data.split("_")
    delivery_id = int(callback_parts[4])
    problem_page = int(callback_parts[5]) if len(callback_parts) > 5 else 0
    if state is not None:
        state_data = await state.get_data()
        if await state.get_state() == EventAdminStates.resolve_reason.state:
            await state.clear()
            restore_back = state_data.get("event_list_back") or state_data.get("res_event_list_back")
            if restore_back:
                await state.update_data(event_list_back=restore_back)
        await state.update_data(problem_page=problem_page)
    item = await get_delivery_item(async_session_maker, delivery_id)
    if not item:
        await callback.answer("Доставка не найдена.", show_alert=True)
        return
    parts = await list_delivery_parts(async_session_maker, delivery_id)
    part_lines = ""
    if parts:
        part_lines = "• <b>Части:</b>\n" + "".join(
            f"  {part.part_index + 1}. {part.part_kind} — <b>{part.status}</b>" + (f" — {html.escape(part.last_error[:80])}" if part.last_error else "") + "\n"
            for part in parts
        )

    text = (
        f"🔍 <b>Детали доставки ID {item.id}</b>\n\n"
        f"• <b>Получатель:</b> {html.escape(item.user_display)}\n"
        f"• <b>Платформа:</b> {item.platform}\n"
        f"• <b>Тип сообщения:</b> {item.message_kind}\n"
        f"• <b>Статус:</b> <b>{item.status}</b>\n"
        f"• <b>Запланированное время (МСК):</b> {item.due_at_msk}\n"
        f"• <b>Время завершения (МСК):</b> {item.sent_or_finished_msk or '—'}\n"
        f"• <b>Число попыток:</b> {item.attempt_count}\n"
        f"• <b>Индекс retry:</b> {item.retry_index}\n"
        f"• <b>Родительская доставка:</b> {item.parent_delivery_id or '—'}\n"
        f"• <b>Последняя ошибка:</b>\n"
        f"<code>{html.escape(item.last_error or '—')}</code>\n"
        f"{part_lines}"
    )

    await callback.message.edit_text(
        text,
        parse_mode="HTML",
        reply_markup=_delivery_resolution_keyboard(item, parts, problem_page),
    )


# Resolving uncertain delivery: prompt reason for force_retry, mark_sent, mark_skipped
@router.callback_query(F.data.startswith("admin_ev_resolve_"))
async def prompt_resolution_reason(callback: CallbackQuery, state: FSMContext) -> None:
    parts = callback.data.split("_")
    action_type = parts[3]  # 'force', 'sent', 'skipped'
    delivery_id = int(parts[4])
    part_id = int(parts[5]) if len(parts) > 5 else None
    state_data = await state.get_data()
    problem_page = int(state_data.get("problem_page", 0))
    list_back = state_data.get("event_list_back", "admin_events")

    type_map = {
        "force": ("force_retry", "Повторная отправка (Force Retry)"),
        "sent": ("mark_sent", "Отметить успешно отправленным"),
        "skipped": ("mark_skipped", "Отметить пропущенным"),
    }
    res_type, title = type_map[action_type]

    await state.clear()
    await state.update_data(
        res_delivery_id=delivery_id,
        res_type=res_type,
        res_part_id=part_id,
        res_problem_page=problem_page,
        res_event_list_back=list_back,
    )
    await state.set_state(EventAdminStates.resolve_reason)

    warning = "⚠️ Повторная отправка может привести к дублю части медиа, если провайдер уже принял запрос.\n\n" if res_type == "force_retry" else ""
    await callback.message.edit_text(
        f"📝 <b>Ручное разрешение доставки ID {delivery_id}</b>\n\n"
        f"Действие: <b>{title}</b>\n\n"
        f"{warning}"
        "Введите причину разрешения (обязательно для аудита):",
        parse_mode="HTML",
        reply_markup=InlineKeyboardBuilder().button(
            text="⬅️ Отмена",
            callback_data=(
                f"admin_ev_problem_item_{delivery_id}_{problem_page}"
                if problem_page > 0
                else f"admin_ev_problem_item_{delivery_id}"
            ),
        ).as_markup(),
    )


@router.message(EventAdminStates.resolve_reason)
async def process_resolution_reason(message: Message, state: FSMContext) -> None:
    reason = (message.text or "").strip()
    if not reason:
        await message.answer("❌ Причина обязательна. Введите причину разрешения:")
        return

    data = await state.get_data()
    delivery_id = data["res_delivery_id"]
    res_type = data["res_type"]
    part_id = data.get("res_part_id")
    problem_page = int(data.get("res_problem_page", 0))
    list_back = data.get("res_event_list_back", "admin_events")
    admin_id = message.from_user.id if message.from_user else 0

    try:
        deliv, ok = await resolve_uncertain_delivery(
            async_session_maker,
            delivery_id=delivery_id,
            admin_id=admin_id,
            resolution_type=res_type,
            resolution_reason=reason,
            part_id=part_id,
        )
        await state.clear()
        await state.update_data(event_list_back=list_back, problem_page=problem_page)
        updated_item = await get_delivery_item(async_session_maker, delivery_id)
        assert updated_item is not None

        updated_parts = await list_delivery_parts(async_session_maker, delivery_id)
        await message.answer(
            f"✅ Доставка ID {delivery_id} успешно разрешена ({res_type})!\n"
            f"Текущий статус: <b>{updated_item.status}</b>",
            parse_mode="HTML",
            reply_markup=_delivery_resolution_keyboard(
                updated_item,
                updated_parts,
                problem_page,
            ),
        )
    except Exception as exc:
        log.exception("Resolution failed: %s", exc)
        await message.answer(
            f"❌ Ошибка разрешения: {exc}",
            reply_markup=InlineKeyboardBuilder().button(
                text="⬅️ Назад",
                callback_data=(
                    f"admin_ev_problem_item_{delivery_id}_{int(data.get('res_problem_page', 0))}"
                    if int(data.get("res_problem_page", 0)) > 0
                    else f"admin_ev_problem_item_{delivery_id}"
                ),
            ).as_markup(),
        )


# ==============================================================================
# TEST SEND TO SELF
# ==============================================================================

@router.callback_query(F.data.startswith("admin_ev_test_send_menu_"))
async def show_test_send_picker(callback: CallbackQuery, state: FSMContext | None = None) -> None:
    event_id = int(callback.data.split("_")[-1])
    state_data = await state.get_data() if state is not None else {}
    list_back = state_data.get("event_list_back", "admin_events")
    steps = await list_steps(async_session_maker, event_id, include_archived=False)
    await callback.message.edit_text(
        "🚀 <b>Тестовая отправка сообщения себе</b>\n\n"
        "Выберите сообщение для отправки. Бот отрендерит его с вашим именем "
        "и отправит в этот чат Telegram. Реальные пользователи и статус доставок не затрагиваются:",
        parse_mode="HTML",
        reply_markup=_test_send_picker_keyboard(event_id, steps, back=list_back),
    )


@router.callback_query(F.data.startswith("admin_ev_test_step_"))
async def test_send_single_step(callback: CallbackQuery, bot: Bot) -> None:
    step_id = int(callback.data.split("_")[-1])
    step = await get_step_item(async_session_maker, step_id)
    if not step:
        await callback.answer("Шаг не найден.", show_alert=True)
        return

    admin_user = callback.from_user
    plan = await build_test_delivery_plan(
        async_session_maker,
        step.event_id,
        step.id,
        admin_user,
        header=f"🧪 <b>[ТЕСТОВОЕ НАПОМИНАНИЕ — {step.offset_display}]</b>",
    )
    await send_test_delivery_plan(
        plan,
        platform="telegram",
        user_id=callback.message.chat.id,
        event_id=step.event_id,
        bot=bot,
    )
    await callback.answer("Тестовое сообщение отправлено в чат!")


@router.callback_query(F.data.startswith("admin_ev_do_test_send_"))
async def do_test_send(callback: CallbackQuery, bot: Bot) -> None:
    parts = callback.data.split("_")
    event_id = int(parts[5])
    step_id = int(parts[6])
    target_step_id = None if step_id == 0 else step_id

    admin_user = callback.from_user
    header = "🧪 <b>[ТЕСТОВОЕ СООБЩЕНИЕ]</b>"
    plan = await build_test_delivery_plan(
        async_session_maker,
        event_id,
        target_step_id,
        admin_user,
        header=header,
    )
    await send_test_delivery_plan(
        plan,
        platform="telegram",
        user_id=callback.message.chat.id,
        event_id=event_id,
        bot=bot,
    )
    await callback.answer("Тестовое сообщение отправлено в чат!")
