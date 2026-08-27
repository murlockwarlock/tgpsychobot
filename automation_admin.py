"""Admin UI for event handlers, state statistics and inactivity follow-ups."""

from __future__ import annotations

import html
import json
import re
from datetime import datetime

from aiogram import BaseMiddleware, Bot, F, Router
from aiogram.filters import Filter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder
from sqlalchemy import delete, distinct, func, select
from sqlalchemy.orm import selectinload

from database import (
    AutomationAction,
    AutomationCondition,
    AutomationConversationState,
    AutomationHandler,
    AutomationStepTransition,
    FollowupCampaign,
    FollowupDelivery,
    FollowupDeliveryAttempt,
    FollowupRun,
    FollowupStep,
    Topic,
    User,
    async_session_maker,
    event_handler_topic_association,
    followup_campaign_topic_association,
    get_all_admin_ids,
)
from followups import (
    FOLLOWUP_METADATA_OPERATOR_LABELS,
    FOLLOWUP_ATTEMPT_CLAIMED,
    FOLLOWUP_ATTEMPT_RETRYABLE,
    FOLLOWUP_ATTEMPT_RETRY_EXHAUSTED,
    FOLLOWUP_ATTEMPT_DELIVERED,
    FOLLOWUP_ATTEMPT_UNCERTAIN,
    FOLLOWUP_STAGE_MODE_LABELS,
    FOLLOWUP_STAGE_MODES,
    _campaign_matches_scope,
    check_campaign_eligibility,
    parse_followup_csv,
    send_followup_step,
)
from time_helpers import format_msk


import logging

router = Router(name="automation_admin")
_answered_callback_ids: set[str] = set()
_manual_followup_tests_inflight: set[tuple[int, int]] = set()


async def _answer_callback(callback: CallbackQuery, *args, **kwargs) -> None:
    await callback.answer(*args, **kwargs)
    callback_id = getattr(callback, "id", None)
    if callback_id:
        _answered_callback_ids.add(callback_id)


async def _safe_edit_text_or_markup(target, text: str, reply_markup=None, disable_web_page_preview: bool = False):
    msg = target.message if hasattr(target, "message") and getattr(target, "message", None) is not None else target
    try:
        await msg.edit_text(
            text,
            parse_mode="HTML",
            reply_markup=reply_markup,
            disable_web_page_preview=disable_web_page_preview,
        )
    except Exception as exc:
        err_str = str(exc)
        if "message is not modified" in err_str or "is not modified" in err_str:
            try:
                await msg.edit_reply_markup(reply_markup=reply_markup)
            except Exception:
                pass
        else:
            logging.exception("Failed to edit admin menu message: %s", exc)


class EnsureCallbackAnsweredMiddleware(BaseMiddleware):
    async def __call__(self, handler, event: CallbackQuery, data):
        try:
            return await handler(event, data)
        finally:
            if event.id in _answered_callback_ids:
                _answered_callback_ids.discard(event.id)
            else:
                try:
                    await event.answer()
                except Exception:
                    # A late/expired callback must not turn a successful DB action
                    # into an application error.
                    pass


class DatabaseAdminFilter(Filter):
    async def __call__(self, event) -> bool:
        return bool(event.from_user and event.from_user.id in await get_all_admin_ids())


router.callback_query.filter(DatabaseAdminFilter())
router.message.filter(DatabaseAdminFilter())
router.callback_query.middleware(EnsureCallbackAnsweredMiddleware())

STAGE_STATS_PAGE_SIZE = 8
STAGE_USERS_PAGE_SIZE = 10


class AutomationAdminStates(StatesGroup):
    handler_name = State()
    condition_value = State()
    condition_edit_value = State()
    action_value = State()
    action_edit_value = State()
    campaign_name = State()
    followup_campaign_rename = State()
    followup_step = State()
    followup_step_edit = State()
    followup_stage_values = State()
    followup_metadata_field = State()
    followup_metadata_operator = State()
    followup_metadata_value = State()
    followup_stop_events = State()
    quiet_hours = State()
    jitter = State()


def _back(callback_data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text="⬅️ Назад", callback_data=callback_data)


async def _navigation_topic_id(state: FSMContext | None, key: str) -> int | None:
    if state is None:
        return None
    data = await state.get_data()
    value = data.get(key)
    return int(value) if value is not None else None


async def _reset_navigation_context(
    state: FSMContext | None,
    key: str,
    topic_id: int | None,
) -> None:
    if state is None:
        return
    await state.clear()
    await state.update_data(**{key: topic_id})


def _automation_menu_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="⚡ Обработчики событий", callback_data="automation_handlers")
    builder.button(text="💬 Догоняющие сообщения", callback_data="followup_campaigns")
    builder.button(text="📊 Статистика этапов", callback_data="automation_stage_stats")
    builder.button(text="📋 Формат DATA", callback_data="automation_data_help")
    builder.button(text="⬅️ В админ-панель", callback_data="admin_panel")
    builder.adjust(1)
    return builder.as_markup()


@router.callback_query(F.data == "automation_menu")
async def automation_menu(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text(
        "⚙️ <b>Автоматизации</b>\n\n"
        "Здесь можно настроить действия, которые бот выполняет сам:\n\n"
        "<b>Обработчики событий</b> — например, отправить администратору заявку, "
        "когда пользователь оставил контакт.\n\n"
        "<b>Догоняющие сообщения</b> — напомнить пользователю о диалоге после паузы.\n\n"
        "<b>Статистика этапов</b> — посмотреть, до какого места доходят пользователи.\n\n"
        "<b>Формат DATA</b> — краткая справка для настройки сценария.",
        reply_markup=_automation_menu_keyboard(),
    )


@router.callback_query(F.data == "automation_data_help")
async def automation_data_help(callback: CallbackQuery):
    example = html.escape(
        'Ответ бота:\n'
        'Спасибо! Мы получили ваши контакты.\n\n'
        '<DATA>\n'
        '{\n'
        '  "current_state": {"current_step": "completed"},\n'
        '  "events": ["CONTACTS_RECEIVED"],\n'
        '  "metadata": {"contact": "+79000000000"}\n'
        '}\n'
        '</DATA>'
    )
    await callback.message.edit_text(
        "📋 <b>Единый формат DATA</b>\n\n"
        "Обычно администратору не нужно менять этот блок. Попросите промпт-инженера передать точные названия событий, этапов и полей.\n\n"
        f"<pre>{example}</pre>\n\n"
        "Блок ставится один раз в конце ответа и пользователю не показывается.\n"
        "• <code>current_step</code> — этап, на котором продолжится сценарий.\n"
        "• <code>events</code> — что произошло сейчас; событий может быть несколько.\n"
        "• <code>metadata</code> — сведения о пользователе; они сохраняются автоматически.\n"
        "• <code>save_mode</code> можно не указывать: по умолчанию новые данные объединяются со старыми.\n"
        "• <code>snapshot</code> — отдельная запись в истории, а не новый диалог.\n\n"
        "Названия чувствительны к регистру: <code>completed</code> и <code>COMPLETED</code> — разные значения.",
        reply_markup=InlineKeyboardBuilder().row(_back("automation_menu")).as_markup(),
    )


@router.callback_query(F.data == "automation_stage_stats")
async def automation_stage_stats(callback: CallbackQuery):
    await _show_automation_stage_stats(callback)


@router.callback_query(F.data.regexp(r"^automation_stage_stats_page_(\d+)$"))
async def automation_stage_stats_page(callback: CallbackQuery):
    await _show_automation_stage_stats(callback, page=int(callback.data.rsplit("_", 1)[1]))


@router.callback_query(F.data == "admin_automation_stage_stats")
async def admin_automation_stage_stats(callback: CallbackQuery):
    await _show_automation_stage_stats(callback, origin="a")


@router.callback_query(F.data.regexp(r"^admin_automation_stats_page_(\d+)$"))
async def admin_automation_stage_stats_page(callback: CallbackQuery):
    await _show_automation_stage_stats(
        callback,
        page=int(callback.data.rsplit("_", 1)[1]),
        origin="a",
    )


@router.callback_query(F.data.regexp(r"^topic_automation_stats_(\d+)$"))
async def topic_automation_stage_stats(callback: CallbackQuery):
    await _show_automation_stage_stats(callback, topic_id=int(callback.data.rsplit("_", 1)[1]))


@router.callback_query(F.data.regexp(r"^topic_automation_stats_page_(\d+)_(\d+)$"))
async def topic_automation_stage_stats_page(callback: CallbackQuery):
    topic_id, page = map(int, callback.data.rsplit("_", 2)[-2:])
    await _show_automation_stage_stats(callback, topic_id=topic_id, page=page)


@router.callback_query(F.data.regexp(r"^automation_stage_users_(\d+)_(\d+)_(g|t|a)$"))
async def automation_stage_users(callback: CallbackQuery):
    _, _, _, anchor_id, page, origin = callback.data.split("_")
    await _show_automation_stage_users(
        callback,
        anchor_id=int(anchor_id),
        page=int(page),
        origin=origin,
    )


async def _show_automation_stage_stats(
    callback: CallbackQuery,
    topic_id: int | None = None,
    page: int = 0,
    origin: str = "g",
):
    selected_topic_id = topic_id
    async with async_session_maker() as session:
        transition_stmt = select(
            AutomationStepTransition.topic_id,
            AutomationStepTransition.current_step,
            func.count(AutomationStepTransition.id),
            func.count(distinct(AutomationStepTransition.user_id)),
            func.min(AutomationStepTransition.id),
        )
        current_stmt = select(
            AutomationConversationState.topic_id,
            AutomationConversationState.current_step,
            func.count(distinct(AutomationConversationState.user_id)),
        ).where(AutomationConversationState.current_step.is_not(None))
        if selected_topic_id is not None:
            transition_stmt = transition_stmt.where(AutomationStepTransition.topic_id == selected_topic_id)
            current_stmt = current_stmt.where(AutomationConversationState.topic_id == selected_topic_id)
        transition_rows = (
            await session.execute(
                transition_stmt
                .group_by(AutomationStepTransition.topic_id, AutomationStepTransition.current_step)
                .order_by(AutomationStepTransition.topic_id, func.min(AutomationStepTransition.id))
            )
        ).all()
        current_rows = (
            await session.execute(
                current_stmt
                .group_by(AutomationConversationState.topic_id, AutomationConversationState.current_step)
            )
        ).all()
        topic_ids = {row[0] for row in transition_rows if row[0]}
        topic_names = dict((await session.execute(select(Topic.id, Topic.name).where(Topic.id.in_(topic_ids)))).all()) if topic_ids else {}
        selected_topic = await session.get(Topic, selected_topic_id) if selected_topic_id is not None else None

    current_map = {(topic_id, step): count for topic_id, step, count in current_rows}
    total_pages = max(1, (len(transition_rows) + STAGE_STATS_PAGE_SIZE - 1) // STAGE_STATS_PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    page_rows = transition_rows[page * STAGE_STATS_PAGE_SIZE:(page + 1) * STAGE_STATS_PAGE_SIZE]
    builder = InlineKeyboardBuilder()
    if not transition_rows:
        text = (
            f"📊 <b>Статистика этапов{f' — {html.escape(selected_topic.name)}' if selected_topic else ''}</b>\n\n"
            "Переходов пока нет. Они появятся, когда модель впервые вернёт "
            "<code>current_state.current_step</code> в новом DATA-блоке."
        )
    else:
        heading = f"Статистика этапов — {html.escape(selected_topic.name)}" if selected_topic else "Статистика этапов"
        lines = [
            f"📊 <b>{heading}</b>",
            "",
            "Считаются уникальные пользователи и реальные смены <code>current_step</code>.",
        ]
        if total_pages > 1:
            lines.append(f"Страница <b>{page + 1}/{total_pages}</b>.")
        active_topic = object()
        for row_topic_id, step, entries, users, anchor_id in page_rows:
            if row_topic_id != active_topic:
                active_topic = row_topic_id
                title = (
                    "Основной диалог"
                    if row_topic_id == 0
                    else topic_names.get(row_topic_id, f"Тема #{row_topic_id}")
                )
                lines.extend(["", f"<b>{html.escape(title)}</b>"])
            current = current_map.get((row_topic_id, step), 0)
            lines.extend([
                "",
                f"<code>{html.escape(step)}</code>",
                f"👥 Вошли: <b>{users}</b>   ·   🔁 Переходы: <b>{entries}</b>   ·   📍 Сейчас: <b>{current}</b>",
            ])
            button_step = step if len(step) <= 30 else f"{step[:27]}…"
            builder.row(InlineKeyboardButton(
                text=f"👥 {button_step} · {users}",
                callback_data=f"automation_stage_users_{anchor_id}_0_{'t' if selected_topic_id is not None else origin}",
            ))
        text = "\n".join(lines)

    pagination = []
    if page > 0:
        previous_callback = (
            f"topic_automation_stats_page_{selected_topic_id}_{page - 1}"
            if selected_topic_id is not None
            else (
                f"admin_automation_stats_page_{page - 1}"
                if origin == "a"
                else f"automation_stage_stats_page_{page - 1}"
            )
        )
        pagination.append(InlineKeyboardButton(text="⬅️", callback_data=previous_callback))
    if page + 1 < total_pages:
        next_callback = (
            f"topic_automation_stats_page_{selected_topic_id}_{page + 1}"
            if selected_topic_id is not None
            else (
                f"admin_automation_stats_page_{page + 1}"
                if origin == "a"
                else f"automation_stage_stats_page_{page + 1}"
            )
        )
        pagination.append(InlineKeyboardButton(text="➡️", callback_data=next_callback))
    if pagination:
        builder.row(*pagination)
    back_callback = (
        f"edit_topic_{selected_topic_id}"
        if selected_topic_id is not None
        else ("admin_stats" if origin == "a" else "automation_menu")
    )
    builder.row(_back(back_callback))
    await callback.message.edit_text(
        text,
        reply_markup=builder.as_markup(),
    )


async def _show_automation_stage_users(
    callback: CallbackQuery,
    *,
    anchor_id: int,
    page: int,
    origin: str,
):
    async with async_session_maker() as session:
        anchor = await session.get(AutomationStepTransition, anchor_id)
        if anchor is None:
            await _answer_callback(callback, "Этап больше не найден.", show_alert=True)
            return

        topic_id = anchor.topic_id
        current_step = anchor.current_step
        total_users = await session.scalar(
            select(func.count(distinct(AutomationStepTransition.user_id))).where(
                AutomationStepTransition.topic_id == topic_id,
                AutomationStepTransition.current_step == current_step,
            )
        ) or 0
        total_pages = max(1, (total_users + STAGE_USERS_PAGE_SIZE - 1) // STAGE_USERS_PAGE_SIZE)
        page = max(0, min(page, total_pages - 1))
        rows = (
            await session.execute(
                select(
                    User.id,
                    User.name,
                    User.first_name,
                    User.username,
                    func.count(AutomationStepTransition.id),
                    func.max(AutomationStepTransition.created_at),
                )
                .join(User, User.id == AutomationStepTransition.user_id)
                .where(
                    AutomationStepTransition.topic_id == topic_id,
                    AutomationStepTransition.current_step == current_step,
                )
                .group_by(User.id, User.name, User.first_name, User.username)
                .order_by(func.max(AutomationStepTransition.created_at).desc(), User.id)
                .offset(page * STAGE_USERS_PAGE_SIZE)
                .limit(STAGE_USERS_PAGE_SIZE)
            )
        ).all()
        current_user_ids = set((await session.scalars(
            select(distinct(AutomationConversationState.user_id)).where(
                AutomationConversationState.topic_id == topic_id,
                AutomationConversationState.current_step == current_step,
            )
        )).all())
        topic = await session.get(Topic, topic_id) if topic_id else None

    topic_name = topic.name if topic else "Основной диалог"
    lines = [
        "👥 <b>Пользователи этапа</b>",
        f"Тема: <b>{html.escape(topic_name)}</b>",
        f"Этап: <code>{html.escape(current_step)}</code>",
        f"Уникальных пользователей: <b>{total_users}</b>",
        f"Страница: <b>{page + 1}/{total_pages}</b>",
    ]
    if not rows:
        lines.extend(["", "Пользователей пока нет."])
    for index, (user_id, name, first_name, username, entries, last_entry) in enumerate(
        rows,
        start=page * STAGE_USERS_PAGE_SIZE + 1,
    ):
        display_name = name or first_name or (f"@{username}" if username else f"ID {user_id}")
        display_name = display_name[:60]
        safe_username = username[:32] if username else None
        username_text = f" · @{html.escape(safe_username)}" if safe_username else ""
        current_text = " · 📍 сейчас" if user_id in current_user_ids else ""
        lines.extend([
            "",
            f"{index}. <a href=\"tg://user?id={user_id}\">{html.escape(display_name)}</a>{username_text}",
            f"<code>{user_id}</code> · входов: <b>{entries}</b>{current_text}",
            f"Последний вход: {format_msk(last_entry)}",
        ])

    builder = InlineKeyboardBuilder()
    pagination = []
    if page > 0:
        pagination.append(InlineKeyboardButton(
            text="⬅️",
            callback_data=f"automation_stage_users_{anchor_id}_{page - 1}_{origin}",
        ))
    if page + 1 < total_pages:
        pagination.append(InlineKeyboardButton(
            text="➡️",
            callback_data=f"automation_stage_users_{anchor_id}_{page + 1}_{origin}",
        ))
    if pagination:
        builder.row(*pagination)
    if origin == "t":
        stats_callback = f"topic_automation_stats_{topic_id}"
    elif origin == "a":
        stats_callback = "admin_automation_stage_stats"
    else:
        stats_callback = "automation_stage_stats"
    builder.row(_back(stats_callback))
    await callback.message.edit_text(
        "\n".join(lines),
        reply_markup=builder.as_markup(),
        disable_web_page_preview=True,
    )


async def _handler_with_relations(session, handler_id: int):
    return await session.scalar(
        select(AutomationHandler)
        .where(AutomationHandler.id == handler_id)
        .options(
            selectinload(AutomationHandler.topics),
            selectinload(AutomationHandler.conditions),
            selectinload(AutomationHandler.actions),
        )
    )


def _condition_label(condition: AutomationCondition) -> str:
    if condition.condition_type == "event":
        source = "Событие"
    elif condition.condition_type == "current_step":
        source = "Этап"
    else:
        source = f"Данные пользователя: {condition.field_path}"
    return f"{source} = {condition.expected_value}"[:55]


def _action_label(action: AutomationAction) -> str:
    if action.action_type == "save_metadata":
        return "Сохранить метаданные"
    recipient = "всем админам" if action.recipient_type == "all_admins" else f"ID {action.recipient_user_id}"
    return f"Сообщение → {recipient}"


@router.callback_query(F.data == "automation_handlers")
async def automation_handlers(callback: CallbackQuery, state: FSMContext | None = None):
    await _show_automation_handlers(callback, state=state)


@router.callback_query(F.data.regexp(r"^topic_automation_handlers_(\d+)$"))
async def topic_automation_handlers(callback: CallbackQuery, state: FSMContext | None = None):
    await _show_automation_handlers(
        callback,
        state=state,
        topic_id=int(callback.data.rsplit("_", 1)[1]),
    )


def _handler_applies_to_topic(handler: AutomationHandler, topic_id: int) -> bool:
    return bool(handler.all_topics or topic_id in {topic.id for topic in handler.topics})


async def _show_automation_handlers(
    callback: CallbackQuery,
    *,
    state: FSMContext | None = None,
    topic_id: int | None = None,
):
    await _reset_navigation_context(state, "automation_return_topic_id", topic_id)
    async with async_session_maker() as session:
        handlers = (
            await session.execute(
                select(AutomationHandler)
                .options(selectinload(AutomationHandler.topics))
                .order_by(AutomationHandler.id)
            )
        ).scalars().all()
        topic = await session.get(Topic, topic_id) if topic_id is not None else None
    if topic_id is not None:
        handlers = [item for item in handlers if _handler_applies_to_topic(item, topic_id)]
    builder = InlineKeyboardBuilder()
    for item in handlers:
        status = "✅" if item.is_active else "⏸"
        view_button = InlineKeyboardButton(
            text=f"{'🌐' if item.all_topics else status} {item.name}",
            callback_data=f"automation_handler_{item.id}",
        )
        if topic_id is not None and not item.all_topics:
            builder.row(
                view_button,
                InlineKeyboardButton(
                    text="✖️ Отвязать",
                    callback_data=f"topic_automation_handler_unlink_{topic_id}_{item.id}",
                ),
            )
        else:
            builder.row(view_button)
    add_callback = f"topic_automation_handler_add_{topic_id}" if topic_id is not None else "automation_handler_add"
    back_callback = f"edit_topic_{topic_id}" if topic_id is not None else "automation_menu"
    builder.row(InlineKeyboardButton(text="➕ Новый обработчик", callback_data=add_callback))
    builder.row(_back(back_callback))
    await _safe_edit_text_or_markup(
        callback,
        f"⚡ <b>Обработчики событий ({len(handlers)}){f' — {html.escape(topic.name)}' if topic else ''}</b>\n\n"
        "Обработчик — это правило «если произошло событие, сделать действие». "
        "Он сработает, только если подходит выбранная область и выполнены все условия. "
        "После создания добавьте условие, действие и включите обработчик."
        + ("\n\nЗдесь показаны обработчики этой темы. Созданный здесь обработчик привяжется к ней автоматически." if topic else ""),
        reply_markup=builder.as_markup(),
    )


@router.callback_query(F.data.regexp(r"^topic_automation_handler_unlink_(\d+)_(\d+)$"))
async def topic_automation_handler_unlink(callback: CallbackQuery):
    topic_id, handler_id = map(int, callback.data.rsplit("_", 2)[-2:])
    async with async_session_maker() as session:
        item = await session.get(AutomationHandler, handler_id)
        if item is not None and not item.all_topics:
            await session.execute(
                delete(event_handler_topic_association).where(
                    event_handler_topic_association.c.handler_id == handler_id,
                    event_handler_topic_association.c.topic_id == topic_id,
                )
            )
            remaining_topics = await session.scalar(
                select(func.count()).select_from(event_handler_topic_association).where(
                    event_handler_topic_association.c.handler_id == handler_id
                )
            )
            if not remaining_topics and not item.include_main_dialogue:
                item.is_active = False
            await session.commit()
    await _answer_callback(callback, "Обработчик отвязан от темы.")
    callback.data = f"topic_automation_handlers_{topic_id}"
    await topic_automation_handlers(callback)


@router.callback_query(F.data == "automation_handler_add")
async def automation_handler_add(callback: CallbackQuery, state: FSMContext):
    await _start_automation_handler_add(callback, state)


@router.callback_query(F.data.regexp(r"^topic_automation_handler_add_(\d+)$"))
async def topic_automation_handler_add(callback: CallbackQuery, state: FSMContext):
    await _start_automation_handler_add(
        callback,
        state,
        topic_id=int(callback.data.rsplit("_", 1)[1]),
    )


async def _start_automation_handler_add(
    callback: CallbackQuery,
    state: FSMContext,
    topic_id: int | None = None,
):
    await state.set_state(AutomationAdminStates.handler_name)
    await state.update_data(preset_topic_id=topic_id)
    back_callback = f"topic_automation_handlers_{topic_id}" if topic_id is not None else "automation_handlers"
    await callback.message.edit_text(
        "<b>Название обработчика</b>\n\n"
        "Введите понятное внутреннее название, например: «Лид готов к консультации». "
        "После создания обработчик будет выключен, пока вы не добавите условия и действия.",
        reply_markup=InlineKeyboardBuilder().row(_back(back_callback)).as_markup(),
    )


@router.message(AutomationAdminStates.handler_name)
async def automation_handler_name_received(message: Message, state: FSMContext):
    name = (message.text or "").strip()
    if not 2 <= len(name) <= 100:
        await message.answer("Название должно содержать от 2 до 100 символов.")
        return
    data = await state.get_data()
    preset_topic_id = data.get("preset_topic_id")
    async with async_session_maker() as session:
        item = AutomationHandler(
            name=name,
            is_active=False,
            include_main_dialogue=preset_topic_id is None,
        )
        if preset_topic_id is not None:
            topic = await session.get(Topic, int(preset_topic_id))
            if topic is None:
                await message.answer("Тема больше не существует. Обработчик не создан.")
                await state.clear()
                return
            item.topics.append(topic)
        session.add(item)
        await session.commit()
        await session.refresh(item)
        handler_id = item.id
    return_topic_id = int(preset_topic_id) if preset_topic_id is not None else None
    await _reset_navigation_context(state, "automation_return_topic_id", return_topic_id)
    await message.answer("✅ Обработчик создан выключенным.")
    await _show_handler(message, handler_id, edit=False, state=state)


async def _show_handler(
    target,
    handler_id: int,
    *,
    edit: bool = True,
    state: FSMContext | None = None,
):
    return_topic_id = await _navigation_topic_id(state, "automation_return_topic_id")
    back_callback = (
        f"topic_automation_handlers_{return_topic_id}"
        if return_topic_id is not None
        else "automation_handlers"
    )
    async with async_session_maker() as session:
        item = await _handler_with_relations(session, handler_id)
    if item is None:
        text = "Обработчик не найден."
        markup = InlineKeyboardBuilder().row(_back(back_callback)).as_markup()
    else:
        topic_names = [topic.name for topic in item.topics]
        if item.all_topics:
            topic_text = "все темы и основной диалог"
        else:
            scopes = (["основной диалог"] if item.include_main_dialogue else []) + topic_names
            topic_text = ", ".join(scopes) or "не выбраны"
        valid = bool(item.conditions and item.actions and (item.all_topics or item.include_main_dialogue or item.topics))
        status = "✅ включён" if item.is_active else "⏸ выключен"
        builder = InlineKeyboardBuilder()
        builder.button(text=f"Статус: {status}", callback_data=f"automation_handler_toggle_{item.id}")
        builder.button(text="💬 Темы", callback_data=f"automation_handler_topics_{item.id}")
        builder.button(text=f"🔎 Условия ({len(item.conditions)})", callback_data=f"automation_conditions_{item.id}")
        builder.button(text=f"⚙️ Действия ({len(item.actions)})", callback_data=f"automation_actions_{item.id}")
        builder.button(text="🗑 Удалить", callback_data=f"automation_handler_delete_ask_{item.id}")
        builder.row(_back(back_callback))
        builder.adjust(1)
        markup = builder.as_markup()
        warning = "" if valid else "\n\n⚠️ Для включения выберите область, добавьте хотя бы одно условие и одно действие."
        text = (
            f"⚡ <b>{html.escape(item.name)}</b>\n\n"
            f"Статус: {status}\n"
            f"Область: {html.escape(topic_text)}\n"
            f"Условий: {len(item.conditions)} (все должны совпасть)\n"
            f"Действий: {len(item.actions)}"
            f"{warning}"
        )
    if edit:
        await target.edit_text(text, reply_markup=markup)
    else:
        await target.answer(text, reply_markup=markup)


@router.callback_query(F.data.regexp(r"^automation_handler_(\d+)$"))
async def automation_handler_view(callback: CallbackQuery, state: FSMContext | None = None):
    await _show_handler(callback.message, int(callback.data.rsplit("_", 1)[1]), state=state)


@router.callback_query(F.data.regexp(r"^automation_handler_toggle_(\d+)$"))
async def automation_handler_toggle(callback: CallbackQuery, state: FSMContext | None = None):
    handler_id = int(callback.data.rsplit("_", 1)[1])
    async with async_session_maker() as session:
        item = await _handler_with_relations(session, handler_id)
        if item is None:
            await _answer_callback(callback, "Обработчик не найден", show_alert=True)
            return
        if not item.is_active and not (
            item.conditions and item.actions and (item.all_topics or item.include_main_dialogue or item.topics)
        ):
            await _answer_callback(callback, "Сначала выберите область, добавьте условие и действие.", show_alert=True)
            return
        item.is_active = not item.is_active
        await session.commit()
    await _show_handler(callback.message, handler_id, state=state)


@router.callback_query(F.data.regexp(r"^automation_handler_topics_(\d+)$"))
async def automation_handler_topics(callback: CallbackQuery):
    handler_id = int(callback.data.rsplit("_", 1)[1])
    async with async_session_maker() as session:
        item = await _handler_with_relations(session, handler_id)
        topics = (await session.execute(select(Topic).order_by(Topic.name))).scalars().all()
    if item is None:
        await _answer_callback(callback, "Обработчик не найден", show_alert=True)
        return
    selected = {topic.id for topic in item.topics}
    builder = InlineKeyboardBuilder()
    builder.button(
        text=f"{'✅' if item.all_topics else '❌'} Все темы",
        callback_data=f"automation_htopic_all_{handler_id}",
    )
    builder.button(
        text=f"{'✅' if item.include_main_dialogue else '❌'} Основной диалог",
        callback_data=f"automation_htopic_main_{handler_id}",
    )
    for topic in topics:
        builder.button(
            text=f"{'✅' if topic.id in selected else '❌'} {topic.name}",
            callback_data=f"automation_htopic_{handler_id}_{topic.id}",
        )
    builder.row(_back(f"automation_handler_{handler_id}"))
    builder.adjust(1)
    await callback.message.edit_text(
        "💬 <b>Темы обработчика</b>\n\n"
        "Отметьте, где должно работать правило.\n"
        "«Все темы» включает основной диалог и будущие темы. "
        "Если правило нужно только в одном месте, отметьте основной диалог или конкретную тему.",
        reply_markup=builder.as_markup(),
    )


@router.callback_query(F.data.regexp(r"^automation_htopic_(all|main)_(\d+)$"))
async def automation_handler_scope_toggle(callback: CallbackQuery):
    parts = callback.data.split("_")
    scope, handler_id = parts[-2], int(parts[-1])
    await _answer_callback(callback)
    async with async_session_maker() as session:
        item = await session.get(AutomationHandler, handler_id)
        if item:
            if scope == "all":
                item.all_topics = not item.all_topics
            else:
                item.include_main_dialogue = not item.include_main_dialogue
            await session.commit()
    callback.data = f"automation_handler_topics_{handler_id}"
    await automation_handler_topics(callback)


@router.callback_query(F.data.regexp(r"^automation_htopic_(\d+)_(\d+)$"))
async def automation_handler_topic_toggle(callback: CallbackQuery):
    _, _, handler_id_raw, topic_id_raw = callback.data.split("_")
    handler_id, topic_id = int(handler_id_raw), int(topic_id_raw)
    await _answer_callback(callback)
    async with async_session_maker() as session:
        exists = await session.scalar(
            select(event_handler_topic_association.c.handler_id).where(
                event_handler_topic_association.c.handler_id == handler_id,
                event_handler_topic_association.c.topic_id == topic_id,
            )
        )
        if exists:
            await session.execute(delete(event_handler_topic_association).where(
                event_handler_topic_association.c.handler_id == handler_id,
                event_handler_topic_association.c.topic_id == topic_id,
            ))
        else:
            await session.execute(event_handler_topic_association.insert().values(handler_id=handler_id, topic_id=topic_id))
        await session.commit()
    callback.data = f"automation_handler_topics_{handler_id}"
    await automation_handler_topics(callback)


@router.callback_query(F.data.regexp(r"^automation_conditions_(\d+)$"))
async def automation_conditions(callback: CallbackQuery):
    handler_id = int(callback.data.rsplit("_", 1)[1])
    async with async_session_maker() as session:
        item = await _handler_with_relations(session, handler_id)
    if item is None:
        return
    builder = InlineKeyboardBuilder()
    for condition in item.conditions:
        builder.button(
            text=f"🔍 {_condition_label(condition)}",
            callback_data=f"automation_condition_view_{handler_id}_{condition.id}",
        )
    builder.button(text="➕ Событие", callback_data=f"automation_condition_add_{handler_id}_event")
    builder.button(text="➕ Текущий этап", callback_data=f"automation_condition_add_{handler_id}_current_step")
    builder.button(text="➕ Метаданные", callback_data=f"automation_condition_add_{handler_id}_metadata")
    builder.row(_back(f"automation_handler_{handler_id}"))
    builder.adjust(1)
    text = (
        f"🔎 <b>Условия ({len(item.conditions)})</b>\n\n"
        "Все добавленные условия должны совпасть одновременно.\n"
        "Обычно достаточно одного условия «Событие» — точного названия события из сценария.\n"
        "Для проверки данных используйте «Метаданные»: например, <code>profile.city = Москва</code>.\n"
        "Проверка всегда означает точное совпадение.\n\n"
        "• Нажмите на условие 🔍, чтобы просмотреть подробности, отредактировать или удалить его."
    )
    await _safe_edit_text_or_markup(callback, text, reply_markup=builder.as_markup())


@router.callback_query(F.data.regexp(r"^automation_condition_view_(\d+)_(\d+)$"))
async def automation_condition_view(callback: CallbackQuery):
    parts = callback.data.split("_")
    handler_id = int(parts[3])
    condition_id = int(parts[4])

    async with async_session_maker() as session:
        condition = await session.get(AutomationCondition, condition_id)
        if not condition or condition.handler_id != handler_id:
            await callback.answer("Условие не найдено.", show_alert=True)
            return

        cond_type = condition.condition_type
        field_path = condition.field_path
        exp_val = condition.expected_value
    type_label = "Событие" if cond_type == "event" else ("Текущий этап" if cond_type == "current_step" else "Метаданные")
    field_info = f"▫️ <b>Поле:</b> <code>{html.escape(field_path or '')}</code>\n" if field_path else ""
    text = (
        f"🔎 <b>Условие #{condition_id}</b>\n\n"
        f"▫️ <b>Тип:</b> {type_label}\n"
        f"{field_info}"
        "▫️ <b>Проверка:</b> точное совпадение\n"
        f"▫️ <b>Ожидаемое значение:</b> <code>{html.escape(exp_val or '')}</code>"
    )

    builder = InlineKeyboardBuilder()
    builder.button(text="✏️ Изменить значение", callback_data=f"automation_condition_edit_{handler_id}_{condition_id}")
    builder.button(text="🗑 Удалить условие", callback_data=f"automation_condition_delete_{handler_id}_{condition_id}")
    builder.row(_back(f"automation_conditions_{handler_id}"))
    builder.adjust(1)

    await callback.message.edit_text(text, parse_mode="HTML", reply_markup=builder.as_markup())


@router.callback_query(F.data.regexp(r"^automation_condition_edit_(\d+)_(\d+)$"))
async def automation_condition_edit(callback: CallbackQuery, state: FSMContext):
    parts = callback.data.split("_")
    handler_id = int(parts[3])
    condition_id = int(parts[4])

    async with async_session_maker() as session:
        condition = await session.get(AutomationCondition, condition_id)
        if not condition or condition.handler_id != handler_id:
            await callback.answer("Условие не найдено.", show_alert=True)
            return
        cond_type = condition.condition_type
        field_path = condition.field_path
        exp_val = condition.expected_value

    await state.set_state(AutomationAdminStates.condition_edit_value)
    await state.update_data(handler_id=handler_id, condition_id=condition_id, condition_type=cond_type)

    if cond_type == "metadata":
        hint = (
            f"Текущее значение: <code>{html.escape(field_path or '')} = {html.escape(exp_val or '')}</code>.\n\n"
            "Введите новое значение в формате: <code>поле = значение</code>. "
            "Проверка означает точное совпадение."
        )
    else:
        hint = f"Текущее значение: <code>{html.escape(exp_val or '')}</code>.\n\nВведите новое значение:"

    await callback.message.edit_text(
        f"✏️ <b>Редактирование условия #{condition_id}</b>\n\n{hint}",
        reply_markup=InlineKeyboardBuilder().row(_back(f"automation_condition_view_{handler_id}_{condition_id}")).as_markup(),
    )


@router.message(AutomationAdminStates.condition_edit_value)
async def automation_condition_edit_received(message: Message, state: FSMContext):
    data = await state.get_data()
    value = (message.text or "").strip()
    field_path = None
    if data["condition_type"] == "metadata":
        if "=" not in value:
            await message.answer("Нужен формат: <code>путь = значение</code>.")
            return
        field_path, value = (part.strip() for part in value.split("=", 1))
        if not field_path or not value:
            await message.answer("Путь и значение не должны быть пустыми.")
            return
    elif not value:
        await message.answer("Значение не должно быть пустым.")
        return

    async with async_session_maker() as session:
        condition = await session.get(AutomationCondition, data["condition_id"])
        if condition:
            condition.expected_value = value
            if field_path is not None:
                condition.field_path = field_path
            await session.commit()

    await state.clear()
    await message.answer("✅ Условие успешно обновлено!")
    await _show_handler(message, data["handler_id"], edit=False, state=state)


@router.callback_query(F.data.regexp(r"^automation_condition_add_(\d+)_(event|current_step|metadata)$"))
async def automation_condition_add(callback: CallbackQuery, state: FSMContext):
    match = re.match(r"^automation_condition_add_(\d+)_(event|current_step|metadata)$", callback.data)
    handler_id, condition_type = int(match.group(1)), match.group(2)
    await state.set_state(AutomationAdminStates.condition_value)
    await state.update_data(handler_id=handler_id, condition_type=condition_type)
    if condition_type == "metadata":
        hint = (
            "Введите поле и значение: <code>profile.city = Москва</code>.\n"
            "Вложенные поля разделяются точкой. Проверка означает точное совпадение.\n"
            "Если значение хранится списком, например <code>[\"portfolio\"]</code>, "
            "попросите промпт-инженера сохранить отдельное поле-строку."
        )
    elif condition_type == "current_step":
        hint = "Введите точное название этапа из сценария, например <code>completed</code>."
    else:
        hint = "Введите точное название события из сценария, например <code>CONTACTS_RECEIVED</code>."
    await callback.message.edit_text(
        f"<b>Новое условие</b>\n\n{hint}",
        reply_markup=InlineKeyboardBuilder().row(_back(f"automation_conditions_{handler_id}")).as_markup(),
    )


@router.message(AutomationAdminStates.condition_value)
async def automation_condition_received(message: Message, state: FSMContext):
    data = await state.get_data()
    value = (message.text or "").strip()
    field_path = None
    if data["condition_type"] == "metadata":
        if "=" not in value:
            await message.answer("Нужен формат: <code>путь = значение</code>.")
            return
        field_path, value = (part.strip() for part in value.split("=", 1))
        if not field_path or not value:
            await message.answer("Путь и значение не должны быть пустыми.")
            return
    elif not value:
        await message.answer("Значение не должно быть пустым.")
        return
    async with async_session_maker() as session:
        order = await session.scalar(
            select(func.count(AutomationCondition.id)).where(AutomationCondition.handler_id == data["handler_id"])
        ) or 0
        session.add(AutomationCondition(
            handler_id=data["handler_id"],
            condition_type=data["condition_type"],
            field_path=field_path,
            operator="equals",
            expected_value=value,
            sort_order=order,
        ))
        await session.commit()
    return_topic_id = data.get("automation_return_topic_id")
    await _reset_navigation_context(state, "automation_return_topic_id", return_topic_id)
    await message.answer("✅ Условие добавлено. Оно будет проверяться вместе с остальными.")
    await _show_handler(message, data["handler_id"], edit=False, state=state)


@router.callback_query(F.data.regexp(r"^automation_condition_delete_(\d+)_(\d+)$"))
async def automation_condition_delete(callback: CallbackQuery):
    parts = callback.data.split("_")
    handler_id_raw, condition_id_raw = parts[3], parts[4]
    await _answer_callback(callback)
    async with async_session_maker() as session:
        condition = await session.get(AutomationCondition, int(condition_id_raw))
        if condition and condition.handler_id == int(handler_id_raw):
            await session.delete(condition)
            await session.commit()
    callback.data = f"automation_conditions_{handler_id_raw}"
    await automation_conditions(callback)


@router.callback_query(F.data.regexp(r"^automation_actions_(\d+)$"))
async def automation_actions(callback: CallbackQuery):
    handler_id = int(callback.data.rsplit("_", 1)[1])
    async with async_session_maker() as session:
        item = await _handler_with_relations(session, handler_id)
    if item is None:
        return
    builder = InlineKeyboardBuilder()
    for action in item.actions:
        builder.button(
            text=f"⚙️ {_action_label(action)}",
            callback_data=f"automation_action_view_{handler_id}_{action.id}",
        )
    builder.button(text="➕ Сообщение всем админам", callback_data=f"automation_action_add_{handler_id}_admins")
    builder.button(text="➕ Сообщение выбранному ID", callback_data=f"automation_action_add_{handler_id}_user")
    builder.button(text="➕ Сохранить метаданные", callback_data=f"automation_action_add_{handler_id}_metadata")
    builder.row(_back(f"automation_handler_{handler_id}"))
    builder.adjust(1)
    text = (
        f"⚙️ <b>Действия ({len(item.actions)})</b>\n\n"
        "Действия выполняются сверху вниз. Например: сначала отправить уведомление, затем сохранить метку.\n"
        "Для одного события каждое действие выполняется только один раз.\n\n"
        "• Нажмите на действие ⚙️, чтобы просмотреть подробности, отредактировать или удалить его."
    )
    await _safe_edit_text_or_markup(callback, text, reply_markup=builder.as_markup())


@router.callback_query(F.data.regexp(r"^automation_action_view_(\d+)_(\d+)$"))
async def automation_action_view(callback: CallbackQuery):
    parts = callback.data.split("_")
    handler_id = int(parts[3])
    action_id = int(parts[4])

    async with async_session_maker() as session:
        action = await session.get(AutomationAction, action_id)
        if not action or action.handler_id != handler_id:
            await callback.answer("Действие не найдено.", show_alert=True)
            return

        act_type = action.action_type
        rec_type = action.recipient_type
        rec_id = action.recipient_user_id
        tpl = action.message_template
        meta = action.metadata_json

    if act_type == "save_metadata":
        type_str = "Сохранение метаданных"
        content_str = f"<code>{html.escape(meta or '{}')}</code>"
    else:
        rec_str = "Всем администраторам" if rec_type == "all_admins" else f"ID: {rec_id}"
        type_str = f"Сообщение ({rec_str})"
        content_str = f"<code>{html.escape(tpl or '')}</code>"

    text = (
        f"⚙️ <b>Действие #{action_id}</b>\n\n"
        f"▫️ <b>Тип:</b> {type_str}\n"
        f"▫️ <b>Содержимое:</b>\n{content_str}"
    )

    builder = InlineKeyboardBuilder()
    builder.button(text="✏️ Изменить значение/шаблон", callback_data=f"automation_action_edit_{handler_id}_{action_id}")
    builder.button(text="🗑 Удалить действие", callback_data=f"automation_action_delete_{handler_id}_{action_id}")
    builder.row(_back(f"automation_actions_{handler_id}"))
    builder.adjust(1)

    await callback.message.edit_text(text, parse_mode="HTML", reply_markup=builder.as_markup())


@router.callback_query(F.data.regexp(r"^automation_action_edit_(\d+)_(\d+)$"))
async def automation_action_edit(callback: CallbackQuery, state: FSMContext):
    parts = callback.data.split("_")
    handler_id = int(parts[3])
    action_id = int(parts[4])

    async with async_session_maker() as session:
        action = await session.get(AutomationAction, action_id)
        if not action or action.handler_id != handler_id:
            await callback.answer("Действие не найдено.", show_alert=True)
            return
        act_type = action.action_type
        rec_type = action.recipient_type
        rec_id = action.recipient_user_id
        tpl = action.message_template
        meta = action.metadata_json

    await state.set_state(AutomationAdminStates.action_edit_value)
    await state.update_data(
        handler_id=handler_id,
        action_id=action_id,
        action_type=act_type,
        recipient_type=rec_type,
        recipient_user_id=rec_id,
    )

    if act_type == "save_metadata":
        hint = (
            f"Текущие данные: <code>{html.escape(meta or '{}')}</code>\n\n"
            "Введите JSON-объект с данными, которые нужно добавить или изменить. "
            "Старые данные не стираются; массивы заменяются целиком."
        )
    elif rec_type == "selected_user":
        hint = (
            f"Текущее значение:\n<code>{rec_id}\n{html.escape(tpl or '')}</code>\n\n"
            "Первая строка — Telegram ID получателя, со второй строки — новый шаблон сообщения."
        )
    else:
        hint = f"Текущий шаблон:\n<code>{html.escape(tpl or '')}</code>\n\nВведите новый шаблон сообщения для всех администраторов:"

    await callback.message.edit_text(
        f"✏️ <b>Редактирование действия #{action_id}</b>\n\n{hint}",
        reply_markup=InlineKeyboardBuilder().row(_back(f"automation_action_view_{handler_id}_{action_id}")).as_markup(),
    )


@router.message(AutomationAdminStates.action_edit_value)
async def automation_action_edit_received(message: Message, state: FSMContext):
    data = await state.get_data()
    raw = (message.text or "").strip()
    act_type = data["action_type"]
    rec_type = data.get("recipient_type")

    updates = {}
    if act_type == "save_metadata":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            await message.answer(f"Невалидный JSON: {html.escape(str(exc))}")
            return
        if not isinstance(payload, dict):
            await message.answer("В корне JSON должен быть объект.")
            return
        updates["metadata_json"] = json.dumps(payload, ensure_ascii=False)
    elif rec_type == "selected_user":
        first, separator, template = raw.partition("\n")
        if not separator or not first.strip().isdigit() or not template.strip():
            await message.answer("Нужны Telegram ID в первой строке и текст сообщения ниже.")
            return
        updates["recipient_user_id"] = int(first.strip())
        updates["message_template"] = template.strip()
    else:
        if not raw:
            await message.answer("Шаблон сообщения не должен быть пустым.")
            return
        updates["message_template"] = raw

    async with async_session_maker() as session:
        action = await session.get(AutomationAction, data["action_id"])
        if action:
            for key, val in updates.items():
                setattr(action, key, val)
            await session.commit()

    await state.clear()
    await message.answer("✅ Действие успешно обновлено!")
    await _show_handler(message, data["handler_id"], edit=False, state=state)


@router.callback_query(F.data.regexp(r"^automation_action_add_(\d+)_(admins|user|metadata)$"))
async def automation_action_add(callback: CallbackQuery, state: FSMContext):
    match = re.match(r"^automation_action_add_(\d+)_(admins|user|metadata)$", callback.data)
    handler_id, kind = int(match.group(1)), match.group(2)
    await state.set_state(AutomationAdminStates.action_value)
    await state.update_data(handler_id=handler_id, action_kind=kind)
    if kind == "metadata":
        hint = (
            "Введите дополнительные данные в формате JSON, например "
            "<code>{\"lead_status\":\"ready\"}</code>.\n"
            "Они добавятся к данным пользователя. <code>{}</code> сохранит данные, "
            "которые пришли вместе с событием."
        )
    elif kind == "user":
        hint = (
            "Первая строка — Telegram ID получателя, со второй строки — шаблон сообщения.\n\n"
            "<b>Пример:</b>\n"
            "<code>123456789\n"
            "Новая заявка от {name} ({username})!\n"
            "Телефон: {metadata.contact}\n"
            "Событие: {event}\n"
            "Шаг: {current_step}</code>\n\n"
            "<i>Можно использовать: {name}, {user}, {username}, {user_id}, {event}, "
            "{current_step}, например {metadata.contact}</i>"
        )
    else:
        hint = (
            "Введите шаблон сообщения для всех администраторов.\n\n"
            "<b>Пример:</b>\n"
            "<code>🚀 Событие: {event}\n"
            "Пользователь: {name} ({username}, ID: {user_id})\n"
            "Телефон: {metadata.contact}\n"
            "Шаг: {current_step}</code>\n\n"
            "<i>Можно использовать: {name}, {user}, {username}, {user_id}, {event}, "
            "{current_step}, например {metadata.contact}</i>"
        )
    await callback.message.edit_text(
        f"<b>Новое действие</b>\n\n{hint}",
        reply_markup=InlineKeyboardBuilder().row(_back(f"automation_actions_{handler_id}")).as_markup(),
    )


@router.message(AutomationAdminStates.action_value)
async def automation_action_received(message: Message, state: FSMContext):
    data = await state.get_data()
    raw = (message.text or "").strip()
    kind = data["action_kind"]
    values = {"handler_id": data["handler_id"]}
    if kind == "metadata":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            await message.answer(f"Невалидный JSON: {html.escape(str(exc))}")
            return
        if not isinstance(payload, dict):
            await message.answer("В корне JSON должен быть объект.")
            return
        values.update(action_type="save_metadata", metadata_json=json.dumps(payload, ensure_ascii=False))
    elif kind == "user":
        first, separator, template = raw.partition("\n")
        if not separator or not first.strip().isdigit() or not template.strip():
            await message.answer("Нужны Telegram ID в первой строке и текст сообщения ниже.")
            return
        values.update(
            action_type="send_message",
            recipient_type="selected_user",
            recipient_user_id=int(first.strip()),
            message_template=template.strip(),
        )
    else:
        if not raw:
            await message.answer("Шаблон сообщения не должен быть пустым.")
            return
        values.update(action_type="send_message", recipient_type="all_admins", message_template=raw)
    async with async_session_maker() as session:
        order = await session.scalar(
            select(func.count(AutomationAction.id)).where(AutomationAction.handler_id == data["handler_id"])
        ) or 0
        session.add(AutomationAction(sort_order=order, **values))
        await session.commit()
    return_topic_id = data.get("automation_return_topic_id")
    await _reset_navigation_context(state, "automation_return_topic_id", return_topic_id)
    await message.answer("✅ Действие добавлено.")
    await _show_handler(message, data["handler_id"], edit=False, state=state)


@router.callback_query(F.data.regexp(r"^automation_action_delete_(\d+)_(\d+)$"))
async def automation_action_delete(callback: CallbackQuery):
    parts = callback.data.split("_")
    handler_id_raw, action_id_raw = parts[3], parts[4]
    await _answer_callback(callback)
    async with async_session_maker() as session:
        action = await session.get(AutomationAction, int(action_id_raw))
        if action and action.handler_id == int(handler_id_raw):
            await session.delete(action)
            await session.commit()
    callback.data = f"automation_actions_{handler_id_raw}"
    await automation_actions(callback)


@router.callback_query(F.data.regexp(r"^automation_handler_delete_ask_(\d+)$"))
async def automation_handler_delete_ask(callback: CallbackQuery):
    handler_id = int(callback.data.rsplit("_", 1)[1])
    builder = InlineKeyboardBuilder()
    builder.button(text="Да, удалить", callback_data=f"automation_handler_delete_yes_{handler_id}")
    builder.row(_back(f"automation_handler_{handler_id}"))
    await callback.message.edit_text(
        "Удалить обработчик вместе с его условиями и действиями? История уже выполненных событий сохранится.",
        reply_markup=builder.as_markup(),
    )


@router.callback_query(F.data.regexp(r"^automation_handler_delete_yes_(\d+)$"))
async def automation_handler_delete_yes(callback: CallbackQuery, state: FSMContext | None = None):
    handler_id = int(callback.data.rsplit("_", 1)[1])
    return_topic_id = await _navigation_topic_id(state, "automation_return_topic_id")
    async with async_session_maker() as session:
        item = await session.get(AutomationHandler, handler_id)
        if item:
            await session.delete(item)
            await session.commit()
    await _show_automation_handlers(callback, state=state, topic_id=return_topic_id)


async def _campaign_with_relations(session, campaign_id: int):
    return await session.scalar(
        select(FollowupCampaign)
        .where(FollowupCampaign.id == campaign_id)
        .options(selectinload(FollowupCampaign.topics), selectinload(FollowupCampaign.steps))
    )


async def _reset_followup_navigation(state: FSMContext | None) -> int | None:
    return_topic_id = await _navigation_topic_id(state, "followup_return_topic_id")
    await _reset_navigation_context(state, "followup_return_topic_id", return_topic_id)
    return return_topic_id


def _campaign_name_value(raw_text: str | None) -> str | None:
    name = (raw_text or "").strip()
    return name if 2 <= len(name) <= 100 else None


def _campaign_stage_text(item: FollowupCampaign) -> str:
    mode = (getattr(item, "stage_mode", None) or "all").strip().lower()
    label = FOLLOWUP_STAGE_MODE_LABELS.get(mode, FOLLOWUP_STAGE_MODE_LABELS["all"])
    if mode in {"selected", "all_except"}:
        values = parse_followup_csv(getattr(item, "stage_values", ""))
        text = f"{html.escape(label)}: {html.escape(', '.join(values) or 'не заданы')}"
        if mode == "selected" and getattr(item, "stage_include_unset", False):
            text += "\n+ если этап не задан"
        return text
    return html.escape(label)


def _campaign_metadata_text(item: FollowupCampaign) -> str:
    field_path = (getattr(item, "metadata_field_path", None) or "").strip()
    if not field_path:
        return "не заданы"
    operator = getattr(item, "metadata_operator", None) or "equals"
    operator_label = FOLLOWUP_METADATA_OPERATOR_LABELS.get(operator, operator)
    value = "" if getattr(item, "metadata_expected_value", None) is None else str(item.metadata_expected_value)
    return (
        f"{html.escape(field_path)} {html.escape(operator_label)} "
        f"{html.escape(value)}"
    )


def _campaign_stop_events_text(item: FollowupCampaign) -> str:
    values = parse_followup_csv(getattr(item, "stop_events", ""))
    return html.escape(", ".join(values) or "не заданы")


def _campaign_has_metadata_condition(item: FollowupCampaign) -> bool:
    return bool((getattr(item, "metadata_field_path", None) or "").strip())


def _campaign_has_stop_events(item: FollowupCampaign) -> bool:
    return bool(parse_followup_csv(getattr(item, "stop_events", "")))


@router.callback_query(F.data == "followup_campaigns")
async def followup_campaigns(callback: CallbackQuery, state: FSMContext | None = None):
    await _show_followup_campaigns(callback, state=state)


@router.callback_query(F.data.regexp(r"^topic_followup_campaigns_(\d+)$"))
async def topic_followup_campaigns(callback: CallbackQuery, state: FSMContext | None = None):
    await _show_followup_campaigns(
        callback,
        state=state,
        topic_id=int(callback.data.rsplit("_", 1)[1]),
    )


def _campaign_applies_to_topic(campaign: FollowupCampaign, topic_id: int) -> bool:
    return bool(campaign.all_topics or topic_id in {topic.id for topic in campaign.topics})


async def _show_followup_campaigns(
    callback: CallbackQuery,
    *,
    state: FSMContext | None = None,
    topic_id: int | None = None,
):
    await _reset_navigation_context(state, "followup_return_topic_id", topic_id)
    async with async_session_maker() as session:
        campaigns = (
            await session.execute(
                select(FollowupCampaign)
                .options(selectinload(FollowupCampaign.topics))
                .order_by(FollowupCampaign.id)
            )
        ).scalars().all()
        topic = await session.get(Topic, topic_id) if topic_id is not None else None
    if topic_id is not None:
        campaigns = [item for item in campaigns if _campaign_applies_to_topic(item, topic_id)]
    builder = InlineKeyboardBuilder()
    for item in campaigns:
        view_button = InlineKeyboardButton(
            text=f"{'🌐' if item.all_topics else ('✅' if item.is_active else '⏸')} {item.name}",
            callback_data=f"followup_campaign_{item.id}",
        )
        if topic_id is not None and not item.all_topics:
            builder.row(
                view_button,
                InlineKeyboardButton(
                    text="✖️ Отвязать",
                    callback_data=f"topic_followup_campaign_unlink_{topic_id}_{item.id}",
                ),
            )
        else:
            builder.row(view_button)
    add_callback = f"topic_followup_campaign_add_{topic_id}" if topic_id is not None else "followup_campaign_add"
    back_callback = f"edit_topic_{topic_id}" if topic_id is not None else "automation_menu"
    builder.row(InlineKeyboardButton(text="➕ Новая цепочка", callback_data=add_callback))
    builder.row(_back(back_callback))
    await _safe_edit_text_or_markup(
        callback,
        f"💬 <b>Догоняющие сообщения ({len(campaigns)}){f' — {html.escape(topic.name)}' if topic else ''}</b>\n\n"
        "Цепочка начинается после действия пользователя и запускается заново после его нового сообщения или нажатия кнопки. "
        "При смене темы или создании нового диалога старая цепочка отменяется. "
        "Сообщения не отправляются в тихие часы."
        + ("\n\nЗдесь показаны цепочки этой темы. Новая цепочка привяжется к ней автоматически." if topic else ""),
        reply_markup=builder.as_markup(),
    )


@router.callback_query(F.data.regexp(r"^topic_followup_campaign_unlink_(\d+)_(\d+)$"))
async def topic_followup_campaign_unlink(callback: CallbackQuery):
    topic_id, campaign_id = map(int, callback.data.rsplit("_", 2)[-2:])
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, campaign_id)
        if item is not None and not item.all_topics:
            await session.execute(
                delete(followup_campaign_topic_association).where(
                    followup_campaign_topic_association.c.campaign_id == campaign_id,
                    followup_campaign_topic_association.c.topic_id == topic_id,
                )
            )
            remaining_topics = await session.scalar(
                select(func.count()).select_from(followup_campaign_topic_association).where(
                    followup_campaign_topic_association.c.campaign_id == campaign_id
                )
            )
            if not remaining_topics and not item.include_main_dialogue:
                item.is_active = False
            await session.commit()
    await _answer_callback(callback, "Цепочка отвязана от темы.")
    callback.data = f"topic_followup_campaigns_{topic_id}"
    await topic_followup_campaigns(callback)


@router.callback_query(F.data == "followup_campaign_add")
async def followup_campaign_add(callback: CallbackQuery, state: FSMContext):
    await _start_followup_campaign_add(callback, state)


@router.callback_query(F.data.regexp(r"^topic_followup_campaign_add_(\d+)$"))
async def topic_followup_campaign_add(callback: CallbackQuery, state: FSMContext):
    await _start_followup_campaign_add(
        callback,
        state,
        topic_id=int(callback.data.rsplit("_", 1)[1]),
    )


async def _start_followup_campaign_add(
    callback: CallbackQuery,
    state: FSMContext,
    topic_id: int | None = None,
):
    await state.set_state(AutomationAdminStates.campaign_name)
    await state.update_data(preset_topic_id=topic_id)
    back_callback = f"topic_followup_campaigns_{topic_id}" if topic_id is not None else "followup_campaigns"
    await callback.message.edit_text(
        "Введите название цепочки. После создания она будет выключена, пока вы не добавите шаги.",
        reply_markup=InlineKeyboardBuilder().row(_back(back_callback)).as_markup(),
    )


@router.message(AutomationAdminStates.campaign_name)
async def followup_campaign_name_received(message: Message, state: FSMContext):
    name = _campaign_name_value(message.text)
    if name is None:
        await message.answer("Название должно содержать от 2 до 100 символов.")
        return
    data = await state.get_data()
    preset_topic_id = data.get("preset_topic_id")
    async with async_session_maker() as session:
        item = FollowupCampaign(
            name=name,
            include_main_dialogue=preset_topic_id is None,
            is_active=False,
        )
        if preset_topic_id is not None:
            topic = await session.get(Topic, int(preset_topic_id))
            if topic is None:
                await message.answer("Тема больше не существует. Цепочка не создана.")
                await state.clear()
                return
            item.topics.append(topic)
        session.add(item)
        await session.commit()
        await session.refresh(item)
        campaign_id = item.id
    return_topic_id = int(preset_topic_id) if preset_topic_id is not None else None
    await _reset_navigation_context(state, "followup_return_topic_id", return_topic_id)
    await message.answer("✅ Цепочка создана выключенной.")
    await _show_campaign(message, campaign_id, edit=False, state=state)


async def _show_campaign(
    target,
    campaign_id: int,
    *,
    edit: bool = True,
    state: FSMContext | None = None,
):
    return_topic_id = await _reset_followup_navigation(state)
    back_callback = (
        f"topic_followup_campaigns_{return_topic_id}"
        if return_topic_id is not None
        else "followup_campaigns"
    )
    async with async_session_maker() as session:
        item = await _campaign_with_relations(session, campaign_id)
    if item is None:
        return
    scopes = "все темы" if item.all_topics else ", ".join(
        (["основной диалог"] if item.include_main_dialogue else []) + [topic.name for topic in item.topics]
    ) or "не выбраны"
    valid = bool(item.steps and (item.all_topics or item.include_main_dialogue or item.topics))
    builder = InlineKeyboardBuilder()
    builder.button(
        text=f"Статус: {'✅ включена' if item.is_active else '⏸ выключена'}",
        callback_data=f"followup_toggle_{item.id}",
    )
    builder.button(text="✏️ Переименовать", callback_data=f"followup_campaign_rename_{item.id}")
    builder.button(text="💬 Темы", callback_data=f"followup_topics_{item.id}")
    builder.button(text=f"🪜 Шаги ({len(item.steps)})", callback_data=f"followup_steps_{item.id}")
    builder.button(text="⚙️ Условия", callback_data=f"followup_conditions_{item.id}")
    builder.button(text="🧪 Проверить на себе", callback_data=f"followup_self_test_{item.id}")
    builder.button(text="🌙 Тихие часы", callback_data=f"followup_quiet_{item.id}")
    builder.button(text="🎲 Случайная задержка", callback_data=f"followup_jitter_{item.id}")
    builder.button(text="🗑 Удалить", callback_data=f"followup_delete_ask_{item.id}")
    builder.row(_back(back_callback))
    builder.adjust(1)
    warning = "" if valid else "\n\n⚠️ Для включения выберите область и добавьте хотя бы один шаг."
    text = (
        f"💬 <b>{html.escape(item.name)}</b>\n\n"
        "Цепочка отправляет несколько напоминаний, пока пользователь молчит.\n"
        f"Статус: {'✅ включена' if item.is_active else '⏸ выключена'}\n"
        f"Область: {html.escape(scopes)}\n"
        f"Шагов: {len(item.steps)}\n"
        f"Тихие часы: {item.quiet_start_minute // 60:02d}:{item.quiet_start_minute % 60:02d}–"
        f"{item.quiet_end_minute // 60:02d}:{item.quiet_end_minute % 60:02d} ({html.escape(item.timezone)})\n"
        f"Случайная задержка: {item.jitter_min_seconds}–{item.jitter_max_seconds} сек."
        f"{warning}"
    )
    if edit:
        await target.edit_text(text, reply_markup=builder.as_markup())
    else:
        await target.answer(text, reply_markup=builder.as_markup())


@router.callback_query(F.data.regexp(r"^followup_campaign_(\d+)$"))
async def followup_campaign_view(callback: CallbackQuery, state: FSMContext | None = None):
    await _show_campaign(callback.message, int(callback.data.rsplit("_", 1)[1]), state=state)


@router.callback_query(F.data.regexp(r"^followup_campaign_rename_(\d+)$"))
async def followup_campaign_rename(callback: CallbackQuery, state: FSMContext):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
    return_topic_id = await _reset_followup_navigation(state)
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, campaign_id)
    if item is None:
        await _answer_callback(callback, "Цепочка не найдена.", show_alert=True)
        return
    await state.set_state(AutomationAdminStates.followup_campaign_rename)
    await state.update_data(
        campaign_id=campaign_id,
        followup_return_topic_id=return_topic_id,
    )
    await callback.message.edit_text(
        f"✏️ <b>Переименовать цепочку</b>\n\n"
        f"Текущее название: <code>{html.escape(item.name)}</code>\n\n"
        "Введите новое название от 2 до 100 символов.",
        reply_markup=InlineKeyboardBuilder().row(_back(f"followup_campaign_{campaign_id}")).as_markup(),
    )


@router.message(AutomationAdminStates.followup_campaign_rename)
async def followup_campaign_rename_received(message: Message, state: FSMContext):
    name = _campaign_name_value(message.text)
    if name is None:
        await message.answer("Название должно содержать от 2 до 100 символов.")
        return
    data = await state.get_data()
    campaign_id = data.get("campaign_id")
    if campaign_id is None:
        await state.clear()
        await message.answer("Переименование устарело. Откройте карточку цепочки заново.")
        return
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, int(campaign_id))
        if item is None:
            await state.clear()
            await message.answer("Цепочка не найдена.")
            return
        item.name = name
        await session.commit()
    return_topic_id = data.get("followup_return_topic_id")
    await _reset_navigation_context(state, "followup_return_topic_id", return_topic_id)
    await message.answer("✅ Название цепочки изменено.")
    await _show_campaign(message, int(campaign_id), edit=False, state=state)


async def _show_followup_conditions(
    target,
    campaign_id: int,
    *,
    edit: bool = True,
):
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, campaign_id)
    if item is None:
        return
    builder = InlineKeyboardBuilder()
    builder.button(text="✏️ Изменить этапы", callback_data=f"followup_stage_edit_{campaign_id}")
    if (getattr(item, "stage_mode", None) or "all").strip().lower() == "selected":
        builder.button(
            text=f"{'✅' if getattr(item, 'stage_include_unset', False) else '☐'} Также если этап не задан",
            callback_data=f"followup_stage_include_unset_{campaign_id}",
        )
    builder.button(text="✏️ Изменить метаданные", callback_data=f"followup_metadata_edit_{campaign_id}")
    if _campaign_has_metadata_condition(item):
        builder.button(text="🧹 Очистить метаданные", callback_data=f"followup_metadata_clear_{campaign_id}")
    builder.button(text="✏️ Изменить события остановки", callback_data=f"followup_stop_events_edit_{campaign_id}")
    if _campaign_has_stop_events(item):
        builder.button(text="🧹 Очистить события", callback_data=f"followup_stop_events_clear_{campaign_id}")
    builder.row(_back(f"followup_campaign_{campaign_id}"))
    builder.adjust(1)
    text = (
        f"⚙️ <b>Условия цепочки</b>\n\n"
        f"Этапы:\n{_campaign_stage_text(item)}\n\n"
        f"Метаданные:\n{_campaign_metadata_text(item)}\n\n"
        f"События остановки:\n{_campaign_stop_events_text(item)}"
    )
    if edit:
        await _safe_edit_text_or_markup(target, text, reply_markup=builder.as_markup())
    else:
        await target.answer(text, reply_markup=builder.as_markup())


@router.callback_query(F.data.regexp(r"^followup_conditions_(\d+)$"))
async def followup_conditions(callback: CallbackQuery, state: FSMContext | None = None):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
    await _reset_followup_navigation(state)
    await _show_followup_conditions(callback, campaign_id)


@router.callback_query(F.data.regexp(r"^followup_stage_include_unset_(\d+)$"))
async def followup_stage_include_unset_toggle(callback: CallbackQuery, state: FSMContext | None = None):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
    await _reset_followup_navigation(state)
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, campaign_id)
        if item is None:
            await _answer_callback(callback, "Цепочка не найдена.", show_alert=True)
            return
        if (getattr(item, "stage_mode", None) or "all").strip().lower() != "selected":
            await _answer_callback(callback, "Сначала выберите режим «На выбранных этапах».", show_alert=True)
            return
        item.stage_include_unset = not item.stage_include_unset
        await session.commit()
    await _answer_callback(callback)
    await _show_followup_conditions(callback, campaign_id)


@router.callback_query(F.data.regexp(r"^followup_stage_edit_(\d+)$"))
async def followup_stage_edit(callback: CallbackQuery, state: FSMContext | None = None):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
    return_topic_id = await _reset_followup_navigation(state)
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, campaign_id)
    if item is None:
        return
    current_mode = (getattr(item, "stage_mode", None) or "all").strip().lower()
    builder = InlineKeyboardBuilder()
    for mode in FOLLOWUP_STAGE_MODES:
        prefix = "✅ " if mode == current_mode else ""
        builder.button(
            text=prefix + FOLLOWUP_STAGE_MODE_LABELS[mode],
            callback_data=f"followup_stage_mode_{campaign_id}_{mode}",
        )
    builder.row(_back(f"followup_conditions_{campaign_id}"))
    builder.adjust(1)
    await callback.message.edit_text(
        "🪜 <b>Этапы запуска</b>\n\nВыберите режим проверки текущего этапа.",
        reply_markup=builder.as_markup(),
    )
    if return_topic_id is not None:
        await state.update_data(followup_return_topic_id=return_topic_id)


@router.callback_query(F.data.regexp(r"^followup_stage_mode_(\d+)_(all_except|all|selected|not_set)$"))
async def followup_stage_mode(callback: CallbackQuery, state: FSMContext):
    match = re.match(r"^followup_stage_mode_(\d+)_(all_except|all|selected|not_set)$", callback.data)
    campaign_id, mode = int(match.group(1)), match.group(2)
    if mode in {"all", "not_set"}:
        async with async_session_maker() as session:
            item = await session.get(FollowupCampaign, campaign_id)
            if item is None:
                return
            item.stage_mode = mode
            item.stage_values = ""
            item.stage_include_unset = False
            await session.commit()
        await _reset_followup_navigation(state)
        await _show_followup_conditions(callback, campaign_id)
        return
    return_topic_id = await _reset_followup_navigation(state)
    await state.set_state(AutomationAdminStates.followup_stage_values)
    await state.update_data(
        campaign_id=campaign_id,
        pending_stage_mode=mode,
        followup_return_topic_id=return_topic_id,
    )
    await callback.message.edit_text(
        f"🪜 <b>{FOLLOWUP_STAGE_MODE_LABELS[mode]}</b>\n\n"
        "Введите точные названия этапов через запятую. Регистр сохраняется.",
        reply_markup=InlineKeyboardBuilder().row(_back(f"followup_stage_edit_{campaign_id}")).as_markup(),
    )


@router.message(AutomationAdminStates.followup_stage_values)
async def followup_stage_values_received(message: Message, state: FSMContext):
    values = parse_followup_csv(message.text)
    if not values:
        await message.answer("Укажите хотя бы один этап через запятую.")
        return
    data = await state.get_data()
    mode = data.get("pending_stage_mode")
    if mode not in {"selected", "all_except"}:
        await state.clear()
        await message.answer("Настройка этапов устарела. Откройте её заново.")
        return
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, data["campaign_id"])
        if item is None:
            await state.clear()
            await message.answer("Цепочка не найдена.")
            return
        item.stage_mode = mode
        item.stage_values = ", ".join(values)
        if mode != "selected":
            item.stage_include_unset = False
        await session.commit()
    return_topic_id = data.get("followup_return_topic_id")
    await _reset_navigation_context(state, "followup_return_topic_id", return_topic_id)
    await message.answer("✅ Условия этапов сохранены.")
    await _show_followup_conditions(message, data["campaign_id"], edit=False)


@router.callback_query(F.data.regexp(r"^followup_metadata_edit_(\d+)$"))
async def followup_metadata_edit(callback: CallbackQuery, state: FSMContext):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
    return_topic_id = await _reset_followup_navigation(state)
    await state.set_state(AutomationAdminStates.followup_metadata_field)
    await state.update_data(campaign_id=campaign_id, followup_return_topic_id=return_topic_id)
    await callback.message.edit_text(
        "🧩 <b>Метаданные</b>\n\nВведите путь поля, например <code>profile.outcome</code>.",
        reply_markup=InlineKeyboardBuilder().row(_back(f"followup_conditions_{campaign_id}")).as_markup(),
    )


@router.message(AutomationAdminStates.followup_metadata_field)
async def followup_metadata_field_received(message: Message, state: FSMContext):
    field_path = (message.text or "").strip()
    if not field_path or any(char.isspace() for char in field_path) or len(field_path) > 200:
        await message.answer("Введите непустой путь без пробелов, например <code>profile.outcome</code>.")
        return
    data = await state.get_data()
    await state.set_state(AutomationAdminStates.followup_metadata_operator)
    await state.update_data(metadata_field_path=field_path)
    await _show_followup_metadata_operator(message, data["campaign_id"], state, edit=False)


async def _show_followup_metadata_operator(
    target,
    campaign_id: int,
    state: FSMContext,
    *,
    edit: bool = True,
):
    data = await state.get_data()
    field_path = data.get("metadata_field_path")
    if data.get("campaign_id") != campaign_id or not field_path:
        await _reset_followup_navigation(state)
        await _show_followup_conditions(target, campaign_id, edit=edit)
        return
    await state.set_state(AutomationAdminStates.followup_metadata_operator)
    await state.update_data(metadata_operator=None)
    builder = InlineKeyboardBuilder()
    for operator, label in FOLLOWUP_METADATA_OPERATOR_LABELS.items():
        builder.button(text=label, callback_data=f"followup_metadata_operator_{campaign_id}_{operator}")
    builder.row(_back(f"followup_metadata_edit_{campaign_id}"))
    builder.adjust(1)
    text = f"Поле: <code>{html.escape(field_path)}</code>\n\nВыберите оператор:"
    if edit:
        await _safe_edit_text_or_markup(target, text, reply_markup=builder.as_markup())
    else:
        await target.answer(text, reply_markup=builder.as_markup())


@router.callback_query(F.data.regexp(r"^followup_metadata_operator_(\d+)_(equals|not_equals|contains)$"))
async def followup_metadata_operator(callback: CallbackQuery, state: FSMContext):
    match = re.match(r"^followup_metadata_operator_(\d+)_(equals|not_equals|contains)$", callback.data)
    campaign_id, operator = int(match.group(1)), match.group(2)
    data = await state.get_data()
    if data.get("campaign_id") != campaign_id or not data.get("metadata_field_path"):
        await _reset_followup_navigation(state)
        await _show_followup_conditions(callback, campaign_id)
        return
    await state.set_state(AutomationAdminStates.followup_metadata_value)
    await state.update_data(metadata_operator=operator)
    await callback.message.edit_text(
        f"🧩 Поле: <code>{html.escape(data['metadata_field_path'])}</code>\n"
        f"Оператор: <b>{html.escape(FOLLOWUP_METADATA_OPERATOR_LABELS[operator])}</b>\n\n"
        "Введите значение:",
        reply_markup=InlineKeyboardBuilder().row(_back(f"followup_metadata_operator_edit_{campaign_id}")).as_markup(),
    )


@router.callback_query(F.data.regexp(r"^followup_metadata_operator_edit_(\d+)$"))
async def followup_metadata_operator_edit(callback: CallbackQuery, state: FSMContext):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
    await _show_followup_metadata_operator(callback, campaign_id, state)


@router.message(AutomationAdminStates.followup_metadata_value)
async def followup_metadata_value_received(message: Message, state: FSMContext):
    value = (message.text or "").strip()
    if not value:
        await message.answer("Значение не должно быть пустым.")
        return
    data = await state.get_data()
    field_path = data.get("metadata_field_path")
    operator = data.get("metadata_operator")
    if not field_path or operator not in FOLLOWUP_METADATA_OPERATOR_LABELS:
        await state.clear()
        await message.answer("Настройка метаданных устарела. Откройте её заново.")
        return
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, data["campaign_id"])
        if item is None:
            await state.clear()
            await message.answer("Цепочка не найдена.")
            return
        item.metadata_field_path = field_path
        item.metadata_operator = operator
        item.metadata_expected_value = value
        await session.commit()
    return_topic_id = data.get("followup_return_topic_id")
    await _reset_navigation_context(state, "followup_return_topic_id", return_topic_id)
    await message.answer("✅ Условие метаданных сохранено.")
    await _show_followup_conditions(message, data["campaign_id"], edit=False)


@router.callback_query(F.data.regexp(r"^followup_metadata_clear_(\d+)$"))
async def followup_metadata_clear(callback: CallbackQuery, state: FSMContext | None = None):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, campaign_id)
        if item is not None:
            item.metadata_field_path = None
            item.metadata_operator = None
            item.metadata_expected_value = None
            await session.commit()
    await _reset_followup_navigation(state)
    await _show_followup_conditions(callback, campaign_id)


@router.callback_query(F.data.regexp(r"^followup_stop_events_edit_(\d+)$"))
async def followup_stop_events_edit(callback: CallbackQuery, state: FSMContext):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
    return_topic_id = await _reset_followup_navigation(state)
    await state.set_state(AutomationAdminStates.followup_stop_events)
    await state.update_data(campaign_id=campaign_id, followup_return_topic_id=return_topic_id)
    builder = InlineKeyboardBuilder()
    builder.button(text="🧹 Очистить список", callback_data=f"followup_stop_events_clear_{campaign_id}")
    builder.row(_back(f"followup_conditions_{campaign_id}"))
    await callback.message.edit_text(
        "🛑 <b>События остановки</b>\n\n"
        "Введите точные имена событий через запятую. Пустой список не останавливает цепочку.",
        reply_markup=builder.as_markup(),
    )


@router.message(AutomationAdminStates.followup_stop_events)
async def followup_stop_events_received(message: Message, state: FSMContext):
    data = await state.get_data()
    values = parse_followup_csv(message.text)
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, data["campaign_id"])
        if item is None:
            await state.clear()
            await message.answer("Цепочка не найдена.")
            return
        item.stop_events = ", ".join(values)
        await session.commit()
    return_topic_id = data.get("followup_return_topic_id")
    await _reset_navigation_context(state, "followup_return_topic_id", return_topic_id)
    await message.answer("✅ События остановки сохранены.")
    await _show_followup_conditions(message, data["campaign_id"], edit=False)


@router.callback_query(F.data.regexp(r"^followup_stop_events_clear_(\d+)$"))
async def followup_stop_events_clear(callback: CallbackQuery, state: FSMContext | None = None):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, campaign_id)
        if item is not None:
            item.stop_events = ""
            await session.commit()
    await _reset_followup_navigation(state)
    await _show_followup_conditions(callback, campaign_id)


async def _followup_self_test_snapshot(session, campaign_id: int, test_user_id: int):
    item = await _campaign_with_relations(session, campaign_id)
    user = await session.scalar(
        select(User)
        .options(selectinload(User.current_topic))
        .where(User.id == test_user_id)
    )
    snapshot = {
        "campaign": item,
        "user": user,
        "dialogue_id": None,
        "topic_id": 0,
        "eligibility": None,
        "run": None,
        "step": None,
        "step_index": None,
        "campaign_valid": False,
        "can_send": False,
        "reason": "campaign_not_found" if item is None else "user_not_found",
    }
    if item is None or user is None:
        return snapshot

    dialogue_id = user.current_dialogue_id if user.current_dialogue_id is not None else 1
    topic_id = user.current_topic_id or 0
    eligibility = await check_campaign_eligibility(
        session,
        item,
        user_id=test_user_id,
        dialogue_id=dialogue_id,
        topic_id=topic_id,
    )
    run = await session.scalar(
        select(FollowupRun).where(
            FollowupRun.campaign_id == campaign_id,
            FollowupRun.user_id == test_user_id,
            FollowupRun.dialogue_id == dialogue_id,
            FollowupRun.topic_id == topic_id,
            FollowupRun.status == "active",
        )
    )
    if run is not None:
        step_index = run.next_step_index
    elif eligibility.eligible:
        step_index = 0
    else:
        step_index = None
    step = item.steps[step_index] if step_index is not None and 0 <= step_index < len(item.steps) else None
    campaign_valid = bool(
        item.is_active
        and item.steps
        and _campaign_matches_scope(item, topic_id)
    )
    if not item.is_active:
        reason = "campaign_inactive"
    elif not _campaign_matches_scope(item, topic_id):
        reason = "scope_not_allowed"
    elif not eligibility.eligible:
        reason = eligibility.reason
    elif not item.steps or step is None:
        reason = "step_missing"
    elif step.message_type not in {"static", "ai"}:
        reason = "step_invalid"
    elif step.message_type == "static" and not (step.message_text or "").strip():
        reason = "step_invalid"
    else:
        reason = "eligible"
    snapshot.update({
        "dialogue_id": dialogue_id,
        "topic_id": topic_id,
        "eligibility": eligibility,
        "run": run,
        "step": step,
        "step_index": step_index,
        "campaign_valid": campaign_valid,
        "can_send": reason == "eligible" and campaign_valid,
        "reason": reason,
    })
    return snapshot


def _followup_self_test_keyboard(campaign_id: int, can_send: bool):
    builder = InlineKeyboardBuilder()
    if can_send:
        builder.button(
            text="▶️ Отправить следующий шаг сейчас",
            callback_data=f"followup_self_test_send_{campaign_id}",
        )
    builder.button(text="🔄 Проверить условия заново", callback_data=f"followup_self_test_{campaign_id}")
    builder.row(_back(f"followup_campaign_{campaign_id}"))
    builder.adjust(1)
    return builder.as_markup()


def _followup_self_test_text(snapshot, test_user_id: int) -> str:
    item = snapshot["campaign"]
    user = snapshot["user"]
    if item is None:
        return "Цепочка не найдена."
    identity = (
        f"@{html.escape(user.username)}"
        if user is not None and user.username
        else html.escape((user.first_name or user.name or "Не найден") if user is not None else "Не найден")
    )
    if user is None:
        return (
            "🧪 <b>Проверка цепочки на себе</b>\n\n"
            f"Пользователь: {identity} / ID {test_user_id}\n\n"
            "Итог:\n❌ Пользователь не найден в базе бота"
        )
    eligibility = snapshot["eligibility"]
    current_step = eligibility.current_step or "не задан"
    topic_id = snapshot["topic_id"]
    topic = user.current_topic
    topic_text = "Основной диалог" if topic_id == 0 else (
        topic.name if topic is not None else f"ID {topic_id}"
    )
    stage_status = "✅" if eligibility.stage_matches else "❌"
    metadata_status = "✅" if eligibility.metadata_matches else "❌"
    if eligibility.metadata_configured:
        metadata_text = _campaign_metadata_text(item)
    else:
        metadata_text = "не заданы"
    if eligibility.matched_stop_event:
        stop_text = f"❌ Событие остановки: {html.escape(eligibility.matched_stop_event)}"
    elif _campaign_has_stop_events(item):
        stop_text = "✅ События остановки: совпадений нет"
    else:
        stop_text = "✅ События остановки: не заданы"
    reason_labels = {
        "campaign_inactive": "цепочка выключена",
        "scope_not_allowed": "текущая тема не входит в область цепочки",
        "step_missing": "следующий шаг отсутствует",
        "step_invalid": "следующий шаг заполнен некорректно",
        "stage_not_allowed": "этап не подходит",
        "metadata_mismatch": "метаданные не подходят",
        "stop_event_found": "найдено событие остановки",
    }
    result = "✅ Цепочка сейчас может запуститься" if snapshot["can_send"] else (
        f"❌ Цепочка сейчас не запустится\nПричина: {reason_labels.get(snapshot['reason'], snapshot['reason'])}"
    )
    next_step = snapshot["step"]
    if next_step is None:
        next_text = "Следующий шаг:\nнет доступного шага"
    else:
        kind = "AI" if next_step.message_type == "ai" else "static"
        preview = (next_step.ai_instruction if next_step.message_type == "ai" else next_step.message_text) or ""
        preview = " ".join(preview.split())
        if len(preview) > 120:
            preview = preview[:119] + "…"
        next_text = (
            f"Следующий шаг:\n#{snapshot['step_index'] + 1} · через {next_step.delay_minutes} мин · {kind}\n"
            f"{html.escape(preview or 'не задано')}"
        )
    return (
        "🧪 <b>Проверка цепочки на себе</b>\n\n"
        f"Пользователь: {identity} / ID {test_user_id}\n"
        f"Диалог: {snapshot['dialogue_id']}\n"
        f"Тема: {html.escape(topic_text)}\n"
        f"Этап: {html.escape(current_step)}\n\n"
        "Условия:\n"
        f"{stage_status} Этап: {_campaign_stage_text(item)}\n"
        f"{metadata_status} Метаданные: {metadata_text}\n"
        f"{stop_text}\n\n"
        f"Итог:\n{result}\n\n"
        f"{next_text}"
    )


async def _show_followup_self_test(
    target,
    campaign_id: int,
    test_user_id: int,
    *,
    state: FSMContext | None = None,
):
    await _reset_followup_navigation(state)
    async with async_session_maker() as session:
        snapshot = await _followup_self_test_snapshot(session, campaign_id, test_user_id)
    text = _followup_self_test_text(snapshot, test_user_id)
    await _safe_edit_text_or_markup(
        target,
        text,
        reply_markup=_followup_self_test_keyboard(campaign_id, snapshot["can_send"]),
    )


@router.callback_query(F.data.regexp(r"^followup_self_test_(\d+)$"))
async def followup_self_test(callback: CallbackQuery, state: FSMContext | None = None):
    await _show_followup_self_test(
        callback,
        int(callback.data.rsplit("_", 1)[1]),
        callback.from_user.id,
        state=state,
    )


@router.callback_query(F.data.regexp(r"^followup_self_test_send_(\d+)$"))
async def followup_self_test_send(callback: CallbackQuery, bot: Bot, state: FSMContext | None = None):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
    test_user_id = callback.from_user.id
    key = (test_user_id, campaign_id)
    if key in _manual_followup_tests_inflight:
        await _answer_callback(callback, "Проверка уже выполняется.")
        return
    _manual_followup_tests_inflight.add(key)
    try:
        await _answer_callback(callback, "Проверяю условия…")
        edit_reply_markup = getattr(callback.message, "edit_reply_markup", None)
        if edit_reply_markup is not None:
            try:
                await edit_reply_markup(reply_markup=None)
            except Exception:
                pass
        async with async_session_maker() as session:
            snapshot = await _followup_self_test_snapshot(session, campaign_id, test_user_id)
        if snapshot["can_send"]:
            await send_followup_step(
                bot,
                user=snapshot["user"],
                step=snapshot["step"],
                dialogue_id=snapshot["dialogue_id"],
                topic_id=snapshot["topic_id"],
            )
        await _show_followup_self_test(
            callback,
            campaign_id,
            test_user_id,
            state=state,
        )
    except Exception:
        logging.exception("Manual follow-up self-test failed: campaign=%s user=%s", campaign_id, test_user_id)
        await _show_followup_self_test(
            callback,
            campaign_id,
            test_user_id,
            state=state,
        )
    finally:
        _manual_followup_tests_inflight.discard(key)


@router.callback_query(F.data.regexp(r"^followup_toggle_(\d+)$"))
async def followup_toggle(callback: CallbackQuery, state: FSMContext | None = None):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
    async with async_session_maker() as session:
        item = await _campaign_with_relations(session, campaign_id)
        if item is None:
            return
        if not item.is_active and not (item.steps and (item.all_topics or item.include_main_dialogue or item.topics)):
            await _answer_callback(callback, "Сначала выберите область и добавьте шаг.", show_alert=True)
            return
        item.is_active = not item.is_active
        await session.commit()
    await _show_campaign(callback.message, campaign_id, state=state)


@router.callback_query(F.data.regexp(r"^followup_topics_(\d+)$"))
async def followup_topics(callback: CallbackQuery, state: FSMContext | None = None):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
    await _reset_followup_navigation(state)
    async with async_session_maker() as session:
        item = await _campaign_with_relations(session, campaign_id)
        topics = (await session.execute(select(Topic).order_by(Topic.name))).scalars().all()
    if item is None:
        return
    selected = {topic.id for topic in item.topics}
    builder = InlineKeyboardBuilder()
    builder.button(text=f"{'✅' if item.all_topics else '❌'} Все темы", callback_data=f"followup_ftopic_all_{campaign_id}")
    builder.button(text=f"{'✅' if item.include_main_dialogue else '❌'} Основной диалог", callback_data=f"followup_ftopic_main_{campaign_id}")
    for topic in topics:
        builder.button(
            text=f"{'✅' if topic.id in selected else '❌'} {topic.name}",
            callback_data=f"followup_ftopic_{campaign_id}_{topic.id}",
        )
    builder.row(_back(f"followup_campaign_{campaign_id}"))
    builder.adjust(1)
    await callback.message.edit_text(
        "💬 <b>Темы цепочки</b>\n\n"
        "«Все темы» автоматически включает основной диалог и будущие темы. "
        "Иначе выберите нужные области отдельно.",
        reply_markup=builder.as_markup(),
    )


@router.callback_query(F.data.regexp(r"^followup_ftopic_(all|main)_(\d+)$"))
async def followup_scope_toggle(callback: CallbackQuery, state: FSMContext | None = None):
    scope, campaign_id_raw = callback.data.split("_")[-2:]
    campaign_id = int(campaign_id_raw)
    await _answer_callback(callback)
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, campaign_id)
        if item:
            if scope == "all":
                item.all_topics = not item.all_topics
            else:
                item.include_main_dialogue = not item.include_main_dialogue
            await session.commit()
    callback.data = f"followup_topics_{campaign_id}"
    await followup_topics(callback, state=state)


@router.callback_query(F.data.regexp(r"^followup_ftopic_(\d+)_(\d+)$"))
async def followup_topic_toggle(callback: CallbackQuery, state: FSMContext | None = None):
    campaign_id_raw, topic_id_raw = callback.data.split("_")[-2:]
    campaign_id, topic_id = int(campaign_id_raw), int(topic_id_raw)
    await _answer_callback(callback)
    async with async_session_maker() as session:
        exists = await session.scalar(
            select(followup_campaign_topic_association.c.campaign_id).where(
                followup_campaign_topic_association.c.campaign_id == campaign_id,
                followup_campaign_topic_association.c.topic_id == topic_id,
            )
        )
        clause = (
            followup_campaign_topic_association.c.campaign_id == campaign_id,
            followup_campaign_topic_association.c.topic_id == topic_id,
        )
        if exists:
            await session.execute(delete(followup_campaign_topic_association).where(*clause))
        else:
            await session.execute(followup_campaign_topic_association.insert().values(campaign_id=campaign_id, topic_id=topic_id))
        await session.commit()
    callback.data = f"followup_topics_{campaign_id}"
    await followup_topics(callback, state=state)


@router.callback_query(F.data.regexp(r"^followup_steps_(\d+)$"))
async def followup_steps(callback: CallbackQuery, state: FSMContext | None = None):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
    await _reset_followup_navigation(state)
    async with async_session_maker() as session:
        item = await _campaign_with_relations(session, campaign_id)
    if item is None:
        return
    builder = InlineKeyboardBuilder()
    for index, step in enumerate(item.steps, 1):
        kind = "AI" if step.message_type == "ai" else "текст"
        builder.button(
            text=f"{index}. через {step.delay_minutes} мин. — {kind}",
            callback_data=f"followup_step_{campaign_id}_{step.id}",
        )
    builder.button(text="➕ Обычный текст", callback_data=f"followup_step_add_{campaign_id}_static")
    builder.button(text="➕ Сгенерировать через AI", callback_data=f"followup_step_add_{campaign_id}_ai")
    builder.row(_back(f"followup_campaign_{campaign_id}"))
    builder.adjust(1)
    text = (
        f"🪜 <b>Шаги цепочки ({len(item.steps)})</b>\n\n"
        "Первое время считается от последнего действия пользователя, следующие — от предыдущего сообщения. "
        "Новое действие пользователя начинает цепочку заново.\n\n"
        "Шаг «Обычный текст» отправляет ваш текст. Шаг «Сгенерировать через AI» передаёт AI вашу инструкцию "
        "и текущий диалог. Нажмите на шаг, чтобы открыть его детали."
    )
    await _safe_edit_text_or_markup(callback, text, reply_markup=builder.as_markup())


async def _show_followup_step_detail(
    target,
    campaign_id: int,
    step_id: int,
    *,
    edit: bool = True,
    state: FSMContext | None = None,
):
    await _reset_followup_navigation(state)
    async with async_session_maker() as session:
        item = await _campaign_with_relations(session, campaign_id)
    if item is None:
        return
    step = next((candidate for candidate in item.steps if candidate.id == step_id), None)
    if step is None:
        return
    index = next(index for index, candidate in enumerate(item.steps, 1) if candidate.id == step_id)
    kind = "AI" if step.message_type == "ai" else "static"
    content_label = "Инструкция" if step.message_type == "ai" else "Текст"
    content = (step.ai_instruction if step.message_type == "ai" else step.message_text) or "не задано"
    content = content[:3000] + ("…" if len(content) > 3000 else "")
    builder = InlineKeyboardBuilder()
    builder.button(text="✏️ Редактировать", callback_data=f"followup_step_edit_{campaign_id}_{step_id}")
    builder.button(text="🗑 Удалить", callback_data=f"followup_step_delete_{campaign_id}_{step_id}")
    builder.row(_back(f"followup_steps_{campaign_id}"))
    builder.adjust(1)
    text = (
        f"🪜 <b>Шаг {index}</b>\n\n"
        f"Индекс: <b>{index}</b>\n"
        f"Задержка: <b>{step.delay_minutes} мин.</b>\n"
        f"Тип: <b>{kind}</b>\n"
        f"{content_label}:\n<code>{html.escape(content)}</code>"
    )
    if edit:
        await _safe_edit_text_or_markup(target, text, reply_markup=builder.as_markup())
    else:
        await target.answer(text, reply_markup=builder.as_markup())


@router.callback_query(F.data.regexp(r"^followup_step_(\d+)_(\d+)$"))
async def followup_step_detail(callback: CallbackQuery, state: FSMContext | None = None):
    match = re.match(r"^followup_step_(\d+)_(\d+)$", callback.data)
    await _show_followup_step_detail(
        callback,
        int(match.group(1)),
        int(match.group(2)),
        state=state,
    )


@router.callback_query(F.data.regexp(r"^followup_step_add_(\d+)_(static|ai)$"))
async def followup_step_add(callback: CallbackQuery, state: FSMContext):
    match = re.match(r"^followup_step_add_(\d+)_(static|ai)$", callback.data)
    campaign_id, kind = int(match.group(1)), match.group(2)
    return_topic_id = await _reset_followup_navigation(state)
    await state.set_state(AutomationAdminStates.followup_step)
    await state.update_data(
        campaign_id=campaign_id,
        step_kind=kind,
        followup_return_topic_id=return_topic_id,
    )
    label = "текст сообщения" if kind == "static" else "инструкцию для AI"
    await callback.message.edit_text(
        "<b>Новый шаг</b>\n\n"
        f"В первой строке укажите задержку в минутах, ниже — {label}.\n\n"
        "Пример:\n<code>60\nМягко напомни пользователю о незавершённом упражнении.</code>\n\n"
        + ("AI сам видит текущий диалог. Напишите только, что нужно сказать; DATA добавлять не нужно." if kind == "ai" else ""),
        reply_markup=InlineKeyboardBuilder().row(_back(f"followup_steps_{campaign_id}")).as_markup(),
    )


@router.callback_query(F.data.regexp(r"^followup_step_edit_(\d+)_(\d+)$"))
async def followup_step_edit(callback: CallbackQuery, state: FSMContext):
    match = re.match(r"^followup_step_edit_(\d+)_(\d+)$", callback.data)
    campaign_id, step_id = int(match.group(1)), int(match.group(2))
    return_topic_id = await _reset_followup_navigation(state)
    async with async_session_maker() as session:
        step = await session.get(FollowupStep, step_id)
    if step is None or step.campaign_id != campaign_id:
        await _answer_callback(callback, "Шаг не найден.", show_alert=True)
        return
    await state.set_state(AutomationAdminStates.followup_step_edit)
    await state.update_data(
        campaign_id=campaign_id,
        step_id=step_id,
        followup_return_topic_id=return_topic_id,
    )
    label = "текст сообщения" if step.message_type == "static" else "инструкцию для AI"
    await callback.message.edit_text(
        "<b>Редактирование шага</b>\n\n"
        f"В первой строке укажите задержку в минутах, ниже — {label}.",
        reply_markup=InlineKeyboardBuilder().row(_back(f"followup_step_{campaign_id}_{step_id}")).as_markup(),
    )


def _parse_followup_step_input(raw_text: str | None) -> tuple[int, str] | None:
    first, separator, body = (raw_text or "").partition("\n")
    if not separator or not first.strip().isdigit() or not body.strip():
        return None
    delay = int(first.strip())
    if not 1 <= delay <= 525600:
        return None
    return delay, body.strip()


@router.message(AutomationAdminStates.followup_step)
async def followup_step_received(message: Message, state: FSMContext):
    data = await state.get_data()
    parsed = _parse_followup_step_input(message.text)
    if parsed is None:
        first, separator, body = (message.text or "").partition("\n")
        if not separator or not first.strip().isdigit() or not body.strip():
            await message.answer("Нужны минуты в первой строке и текст ниже.")
            return
        await message.answer("Задержка должна быть от 1 минуты до 365 дней.")
        return
    delay, body = parsed
    if not body:
        await message.answer("Нужны минуты в первой строке и текст ниже.")
        return
    async with async_session_maker() as session:
        order = await session.scalar(
            select(func.count(FollowupStep.id)).where(FollowupStep.campaign_id == data["campaign_id"])
        ) or 0
        values = {
            "campaign_id": data["campaign_id"],
            "sort_order": order,
            "delay_minutes": delay,
            "message_type": data["step_kind"],
        }
        if data["step_kind"] == "ai":
            values["ai_instruction"] = body.strip()
        else:
            values["message_text"] = body.strip()
        session.add(FollowupStep(**values))
        await session.commit()
    return_topic_id = data.get("followup_return_topic_id")
    await _reset_navigation_context(state, "followup_return_topic_id", return_topic_id)
    await message.answer("✅ Шаг добавлен.")
    await _show_campaign(message, data["campaign_id"], edit=False, state=state)


@router.message(AutomationAdminStates.followup_step_edit)
async def followup_step_edit_received(message: Message, state: FSMContext):
    parsed = _parse_followup_step_input(message.text)
    if parsed is None:
        first, separator, body = (message.text or "").partition("\n")
        if not separator or not first.strip().isdigit() or not body.strip():
            await message.answer("Нужны минуты в первой строке и текст ниже.")
            return
        await message.answer("Задержка должна быть от 1 минуты до 365 дней.")
        return
    delay, body = parsed
    data = await state.get_data()
    async with async_session_maker() as session:
        step = await session.get(FollowupStep, data["step_id"])
        if step is None or step.campaign_id != data["campaign_id"]:
            await state.clear()
            await message.answer("Шаг не найден.")
            return
        step.delay_minutes = delay
        if step.message_type == "ai":
            step.ai_instruction = body
        else:
            step.message_text = body
        await session.commit()
    return_topic_id = data.get("followup_return_topic_id")
    await _reset_navigation_context(state, "followup_return_topic_id", return_topic_id)
    await message.answer("✅ Шаг обновлён.")
    await _show_followup_step_detail(
        message,
        data["campaign_id"],
        data["step_id"],
        edit=False,
        state=state,
    )


@router.callback_query(F.data.regexp(r"^followup_step_delete_(\d+)_(\d+)$"))
async def followup_step_delete(callback: CallbackQuery, state: FSMContext | None = None):
    campaign_id_raw, step_id_raw = callback.data.split("_")[-2:]
    blocked = False
    async with async_session_maker() as session:
        step = await session.scalar(
            select(FollowupStep)
            .where(
                FollowupStep.id == int(step_id_raw),
                FollowupStep.campaign_id == int(campaign_id_raw),
            )
            .with_for_update()
        )
        if step:
            sent_count = await session.scalar(
                select(func.count(FollowupDelivery.id)).where(FollowupDelivery.step_id == step.id)
            ) or 0
            protected_attempt_count = await session.scalar(
                select(func.count(FollowupDeliveryAttempt.id)).where(
                    FollowupDeliveryAttempt.step_id == step.id,
                    FollowupDeliveryAttempt.status.in_(
                        (
                            FOLLOWUP_ATTEMPT_CLAIMED,
                            FOLLOWUP_ATTEMPT_RETRYABLE,
                            FOLLOWUP_ATTEMPT_UNCERTAIN,
                            FOLLOWUP_ATTEMPT_DELIVERED,
                            FOLLOWUP_ATTEMPT_RETRY_EXHAUSTED,
                        )
                    ),
                )
            ) or 0
            if sent_count or protected_attempt_count:
                blocked = True
            else:
                await session.delete(step)
                await session.flush()
                remaining = (
                    await session.execute(
                        select(FollowupStep)
                        .where(FollowupStep.campaign_id == int(campaign_id_raw))
                        .order_by(FollowupStep.sort_order, FollowupStep.id)
                    )
                ).scalars().all()
                for index, item in enumerate(remaining):
                    item.sort_order = index
                await session.commit()
    if blocked:
        await _answer_callback(
            callback,
            "Шаг уже отправлялся или находится в процессе отправки, его нельзя удалить без потери истории.",
            show_alert=True,
        )
    callback.data = f"followup_steps_{campaign_id_raw}"
    await followup_steps(callback, state=state)


@router.callback_query(F.data.regexp(r"^followup_quiet_(\d+)$"))
async def followup_quiet(callback: CallbackQuery, state: FSMContext):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
    return_topic_id = await _reset_followup_navigation(state)
    await state.set_state(AutomationAdminStates.quiet_hours)
    await state.update_data(campaign_id=campaign_id, followup_return_topic_id=return_topic_id)
    await callback.message.edit_text(
        "🌙 <b>Тихие часы</b>\n\n"
        "Введите интервал и часовую зону в формате:\n"
        "<code>22:00-09:00 Europe/Moscow</code>\n\n"
        "Сообщение, попавшее в этот интервал, переносится на его окончание.",
        reply_markup=InlineKeyboardBuilder().row(_back(f"followup_campaign_{campaign_id}")).as_markup(),
    )


@router.message(AutomationAdminStates.quiet_hours)
async def followup_quiet_received(message: Message, state: FSMContext):
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})\s+([A-Za-z_]+/[A-Za-z_]+)\s*", message.text or "")
    if not match:
        await message.answer("Формат: <code>22:00-09:00 Europe/Moscow</code>")
        return
    sh, sm, eh, em = map(int, match.groups()[:4])
    if sh > 23 or eh > 23 or sm > 59 or em > 59:
        await message.answer("Проверьте время: часы 0–23, минуты 0–59.")
        return
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    try:
        ZoneInfo(match.group(5))
    except ZoneInfoNotFoundError:
        await message.answer("Неизвестная часовая зона. Пример: <code>Europe/Moscow</code>.")
        return
    data = await state.get_data()
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, data["campaign_id"])
        item.quiet_start_minute = sh * 60 + sm
        item.quiet_end_minute = eh * 60 + em
        item.timezone = match.group(5)
        await session.commit()
    return_topic_id = data.get("followup_return_topic_id")
    await _reset_navigation_context(state, "followup_return_topic_id", return_topic_id)
    await message.answer("✅ Тихие часы сохранены.")
    await _show_campaign(message, data["campaign_id"], edit=False, state=state)


@router.callback_query(F.data.regexp(r"^followup_jitter_(\d+)$"))
async def followup_jitter(callback: CallbackQuery, state: FSMContext):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
    return_topic_id = await _reset_followup_navigation(state)
    await state.set_state(AutomationAdminStates.jitter)
    await state.update_data(campaign_id=campaign_id, followup_return_topic_id=return_topic_id)
    await callback.message.edit_text(
        "🎲 <b>Случайная задержка</b>\n\n"
        "Введите диапазон в секундах, например <code>30-180</code>. "
        "Для точного времени отправки укажите <code>0-0</code>.",
        reply_markup=InlineKeyboardBuilder().row(_back(f"followup_campaign_{campaign_id}")).as_markup(),
    )


@router.message(AutomationAdminStates.jitter)
async def followup_jitter_received(message: Message, state: FSMContext):
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", message.text or "")
    if not match:
        await message.answer("Формат диапазона: <code>30-180</code>.")
        return
    low, high = map(int, match.groups())
    if low > high or high > 86400:
        await message.answer("Минимум не должен превышать максимум; максимум — 86400 секунд.")
        return
    data = await state.get_data()
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, data["campaign_id"])
        item.jitter_min_seconds = low
        item.jitter_max_seconds = high
        await session.commit()
    return_topic_id = data.get("followup_return_topic_id")
    await _reset_navigation_context(state, "followup_return_topic_id", return_topic_id)
    await message.answer("✅ Случайная задержка сохранена.")
    await _show_campaign(message, data["campaign_id"], edit=False, state=state)


@router.callback_query(F.data.regexp(r"^followup_delete_ask_(\d+)$"))
async def followup_delete_ask(callback: CallbackQuery):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
    builder = InlineKeyboardBuilder()
    builder.button(text="Да, удалить", callback_data=f"followup_delete_yes_{campaign_id}")
    builder.row(_back(f"followup_campaign_{campaign_id}"))
    await callback.message.edit_text(
        "Удалить цепочку, её шаги и все ожидающие отправки? Уже отправленные сообщения останутся у пользователей.",
        reply_markup=builder.as_markup(),
    )


@router.callback_query(F.data.regexp(r"^followup_delete_yes_(\d+)$"))
async def followup_delete_yes(callback: CallbackQuery, state: FSMContext | None = None):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
    return_topic_id = await _navigation_topic_id(state, "followup_return_topic_id")
    async with async_session_maker() as session:
        item = await session.get(FollowupCampaign, campaign_id)
        if item:
            await session.delete(item)
            await session.commit()
    await _show_followup_campaigns(callback, state=state, topic_id=return_topic_id)
