"""Admin UI for event handlers, state statistics and inactivity follow-ups."""

from __future__ import annotations

import html
import json
import re
from datetime import datetime

from aiogram import BaseMiddleware, F, Router
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
    FollowupStep,
    Topic,
    User,
    async_session_maker,
    event_handler_topic_association,
    followup_campaign_topic_association,
    get_all_admin_ids,
)
from time_helpers import format_msk


router = Router(name="automation_admin")
_answered_callback_ids: set[str] = set()


async def _answer_callback(callback: CallbackQuery, *args, **kwargs) -> None:
    await callback.answer(*args, **kwargs)
    callback_id = getattr(callback, "id", None)
    if callback_id:
        _answered_callback_ids.add(callback_id)


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
    action_value = State()
    campaign_name = State()
    followup_step = State()
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
        "<b>Обработчики событий</b> реагируют на события из блока DATA и выполняют несколько действий. "
        "Все условия одного обработчика проверяются вместе по правилу И (AND).\n\n"
        "<b>Догоняющие сообщения</b> запускаются после бездействия пользователя. Они привязываются к темам отдельно.\n\n"
        "<b>Статистика этапов</b> строится по фактическим сменам current_step, а не по числу ответов модели.",
        reply_markup=_automation_menu_keyboard(),
    )


@router.callback_query(F.data == "automation_data_help")
async def automation_data_help(callback: CallbackQuery):
    example = html.escape(
        'Ответ 1 — бот спрашивает об увлечениях:\n'
        '<DATA>\n'
        '{\n'
        '  "current_state": {"current_step": "STAGE_1_HOBBY"},\n'
        '  "events": [],\n'
        '  "metadata": {}\n'
        '}\n'
        '</DATA>\n\n'
        'Ответ 2 — увлечение получено, бот спрашивает о цели:\n'
        '<DATA>\n'
        '{\n'
        '  "current_state": {"current_step": "STAGE_2_GOAL"},\n'
        '  "events": ["HOBBY_RECEIVED"],\n'
        '  "metadata": {"interests": ["играть в компьютер"]}\n'
        '}\n'
        '</DATA>\n\n'
        'Ответ 3 — цель получена, бот готовит результат:\n'
        '<DATA>\n'
        '{\n'
        '  "current_state": {"current_step": "STAGE_3_RESULT"},\n'
        '  "events": ["GOAL_RECEIVED"],\n'
        '  "metadata": {"goal": "найти новое увлечение"}\n'
        '}\n'
        '</DATA>\n\n'
        'Ответ 4 — результат готов:\n'
        '<DATA>\n'
        '{\n'
        '  "current_state": {"current_step": "STAGE_4_COMPLETED"},\n'
        '  "events": ["RESULT_READY"],\n'
        '  "metadata": {"result": "подобран план занятий"}\n'
        '}\n'
        '</DATA>'
    )
    await callback.message.edit_text(
        "📋 <b>Единый формат DATA</b>\n\n"
        "Ниже четыре отдельных ответа модели. В каждом ответе должен быть только один блок <code>&lt;DATA&gt;</code>.\n\n"
        f"<pre>{example}</pre>\n\n"
        "Блок ставится один раз в конце ответа модели и пользователю не показывается.\n"
        "• <code>current_state.current_step</code> — текущий этап алгоритма; смена этапа попадает в статистику.\n"
        "• <code>events</code> — одноразовые сигналы для обработчиков.\n"
        "• <code>metadata</code> — данные текущего диалога и темы.\n"
        "• Метаданные по умолчанию дополняют уже сохранённые данные.\n"
        "• <code>save_mode: snapshot</code> указывается только когда нужен отдельный снимок.\n\n"
        "Старый <code>[DATA]</code> читается для совместимости, но новые промпты следует писать только в этом формате.",
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
    operators = {"equals": "=", "not_equals": "≠", "contains": "содержит", "exists": "существует"}
    if condition.condition_type == "event":
        source = "Событие"
    elif condition.condition_type == "current_step":
        source = "Этап"
    else:
        source = f"metadata.{condition.field_path}"
    return f"{source} {operators.get(condition.operator, condition.operator)} {condition.expected_value}"[:55]


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
    await callback.message.edit_text(
        f"⚡ <b>Обработчики событий{f' — {html.escape(topic.name)}' if topic else ''}</b>\n\n"
        "Обработчик срабатывает только когда совпали тема и все его условия. "
        "Действия выполняются по порядку и не повторяются для одного события."
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
            f"Условий: {len(item.conditions)} (логика И / AND)\n"
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
        "«Все темы» включает основной диалог и любые текущие/будущие темы. "
        "Иначе отметьте основной диалог и нужные темы отдельно.",
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
            text=f"🗑 {_condition_label(condition)}",
            callback_data=f"automation_condition_delete_{handler_id}_{condition.id}",
        )
    builder.button(text="➕ Событие", callback_data=f"automation_condition_add_{handler_id}_event")
    builder.button(text="➕ Текущий этап", callback_data=f"automation_condition_add_{handler_id}_current_step")
    builder.button(text="➕ Метаданные", callback_data=f"automation_condition_add_{handler_id}_metadata")
    builder.row(_back(f"automation_handler_{handler_id}"))
    builder.adjust(1)
    await callback.message.edit_text(
        "🔎 <b>Условия</b>\n\n"
        "Все добавленные условия должны выполниться одновременно (AND). "
        "Для обычного события достаточно условия «Событие = EVENT_NAME».\n\n"
        "Нажатие на существующее условие удаляет его.",
        reply_markup=builder.as_markup(),
    )


@router.callback_query(F.data.regexp(r"^automation_condition_add_(\d+)_(event|current_step|metadata)$"))
async def automation_condition_add(callback: CallbackQuery, state: FSMContext):
    match = re.match(r"^automation_condition_add_(\d+)_(event|current_step|metadata)$", callback.data)
    handler_id, condition_type = int(match.group(1)), match.group(2)
    await state.set_state(AutomationAdminStates.condition_value)
    await state.update_data(handler_id=handler_id, condition_type=condition_type)
    if condition_type == "metadata":
        hint = "Введите путь и значение: <code>profile.city = Москва</code>. Вложенные поля разделяются точкой."
    elif condition_type == "current_step":
        hint = "Введите точный ID этапа, например <code>STAGE_2_RESULT</code>."
    else:
        hint = "Введите точное имя события из массива events, например <code>LEAD_READY</code>."
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
    await message.answer("✅ Условие добавлено. Оно объединено с остальными по AND.")
    await _show_handler(message, data["handler_id"], edit=False, state=state)


@router.callback_query(F.data.regexp(r"^automation_condition_delete_(\d+)_(\d+)$"))
async def automation_condition_delete(callback: CallbackQuery):
    _, _, _, handler_id_raw, condition_id_raw = callback.data.split("_")
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
            text=f"🗑 {_action_label(action)}",
            callback_data=f"automation_action_delete_{handler_id}_{action.id}",
        )
    builder.button(text="➕ Сообщение всем админам", callback_data=f"automation_action_add_{handler_id}_admins")
    builder.button(text="➕ Сообщение выбранному ID", callback_data=f"automation_action_add_{handler_id}_user")
    builder.button(text="➕ Сохранить метаданные", callback_data=f"automation_action_add_{handler_id}_metadata")
    builder.row(_back(f"automation_handler_{handler_id}"))
    builder.adjust(1)
    await callback.message.edit_text(
        "⚙️ <b>Действия</b>\n\n"
        "Действия выполняются сверху вниз. Для одного события каждое действие выполняется не более одного раза.\n\n"
        "В шаблонах доступны: <code>{name}</code>, <code>{username}</code>, <code>{user_id}</code>, "
        "<code>{event}</code>, <code>{current_step}</code>, <code>{metadata.profile.city}</code>.\n\n"
        "Нажатие на существующее действие удаляет его.",
        reply_markup=builder.as_markup(),
    )


@router.callback_query(F.data.regexp(r"^automation_action_add_(\d+)_(admins|user|metadata)$"))
async def automation_action_add(callback: CallbackQuery, state: FSMContext):
    match = re.match(r"^automation_action_add_(\d+)_(admins|user|metadata)$", callback.data)
    handler_id, kind = int(match.group(1)), match.group(2)
    await state.set_state(AutomationAdminStates.action_value)
    await state.update_data(handler_id=handler_id, action_kind=kind)
    if kind == "metadata":
        hint = (
            "Отправьте JSON с дополнительными полями, например "
            "<code>{\"lead_status\":\"ready\"}</code>. Отправьте <code>{}</code>, чтобы сохранить metadata события как есть."
        )
    elif kind == "user":
        hint = "Первая строка — Telegram ID получателя, со второй строки — шаблон сообщения."
    else:
        hint = "Введите шаблон сообщения для всех администраторов."
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
    _, _, _, handler_id_raw, action_id_raw = callback.data.split("_")
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
    await callback.message.edit_text(
        f"💬 <b>Догоняющие сообщения{f' — {html.escape(topic.name)}' if topic else ''}</b>\n\n"
        "Таймер начинается заново после каждого сообщения или нажатия пользователя. "
        "При смене темы либо создании нового диалога старая цепочка отменяется. "
        "Шаги отправляются только вне тихих часов."
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
    name = (message.text or "").strip()
    if not 2 <= len(name) <= 100:
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
    return_topic_id = await _navigation_topic_id(state, "followup_return_topic_id")
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
    builder.button(text="💬 Темы", callback_data=f"followup_topics_{item.id}")
    builder.button(text=f"🪜 Шаги ({len(item.steps)})", callback_data=f"followup_steps_{item.id}")
    builder.button(text="🌙 Тихие часы", callback_data=f"followup_quiet_{item.id}")
    builder.button(text="🎲 Случайная задержка", callback_data=f"followup_jitter_{item.id}")
    builder.button(text="🗑 Удалить", callback_data=f"followup_delete_ask_{item.id}")
    builder.row(_back(back_callback))
    builder.adjust(1)
    warning = "" if valid else "\n\n⚠️ Для включения выберите область и добавьте хотя бы один шаг."
    text = (
        f"💬 <b>{html.escape(item.name)}</b>\n\n"
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
async def followup_topics(callback: CallbackQuery):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
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
async def followup_scope_toggle(callback: CallbackQuery):
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
    await followup_topics(callback)


@router.callback_query(F.data.regexp(r"^followup_ftopic_(\d+)_(\d+)$"))
async def followup_topic_toggle(callback: CallbackQuery):
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
    await followup_topics(callback)


@router.callback_query(F.data.regexp(r"^followup_steps_(\d+)$"))
async def followup_steps(callback: CallbackQuery):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
    async with async_session_maker() as session:
        item = await _campaign_with_relations(session, campaign_id)
    if item is None:
        return
    builder = InlineKeyboardBuilder()
    for index, step in enumerate(item.steps, 1):
        kind = "AI" if step.message_type == "ai" else "текст"
        builder.button(
            text=f"🗑 {index}. через {step.delay_minutes} мин. — {kind}",
            callback_data=f"followup_step_delete_{campaign_id}_{step.id}",
        )
    builder.button(text="➕ Обычный текст", callback_data=f"followup_step_add_{campaign_id}_static")
    builder.button(text="➕ Сгенерировать через AI", callback_data=f"followup_step_add_{campaign_id}_ai")
    builder.row(_back(f"followup_campaign_{campaign_id}"))
    builder.adjust(1)
    await callback.message.edit_text(
        "🪜 <b>Шаги цепочки</b>\n\n"
        "Задержка первого шага считается от последнего действия пользователя. "
        "Задержки следующих шагов — от предыдущей отправки. Новая активность начинает цепочку заново.\n\n"
        "Нажатие на существующий шаг удаляет его.",
        reply_markup=builder.as_markup(),
    )


@router.callback_query(F.data.regexp(r"^followup_step_add_(\d+)_(static|ai)$"))
async def followup_step_add(callback: CallbackQuery, state: FSMContext):
    match = re.match(r"^followup_step_add_(\d+)_(static|ai)$", callback.data)
    campaign_id, kind = int(match.group(1)), match.group(2)
    await state.set_state(AutomationAdminStates.followup_step)
    await state.update_data(campaign_id=campaign_id, step_kind=kind)
    label = "текст сообщения" if kind == "static" else "инструкцию для AI"
    await callback.message.edit_text(
        "<b>Новый шаг</b>\n\n"
        f"В первой строке укажите задержку в минутах, ниже — {label}.\n\n"
        "Пример:\n<code>60\nМягко напомни пользователю о незавершённом упражнении.</code>",
        reply_markup=InlineKeyboardBuilder().row(_back(f"followup_steps_{campaign_id}")).as_markup(),
    )


@router.message(AutomationAdminStates.followup_step)
async def followup_step_received(message: Message, state: FSMContext):
    data = await state.get_data()
    first, separator, body = (message.text or "").partition("\n")
    if not separator or not first.strip().isdigit() or not body.strip():
        await message.answer("Нужны минуты в первой строке и текст ниже.")
        return
    delay = int(first.strip())
    if not 1 <= delay <= 525600:
        await message.answer("Задержка должна быть от 1 минуты до 365 дней.")
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


@router.callback_query(F.data.regexp(r"^followup_step_delete_(\d+)_(\d+)$"))
async def followup_step_delete(callback: CallbackQuery):
    campaign_id_raw, step_id_raw = callback.data.split("_")[-2:]
    async with async_session_maker() as session:
        step = await session.get(FollowupStep, int(step_id_raw))
        if step and step.campaign_id == int(campaign_id_raw):
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
    callback.data = f"followup_steps_{campaign_id_raw}"
    await followup_steps(callback)


@router.callback_query(F.data.regexp(r"^followup_quiet_(\d+)$"))
async def followup_quiet(callback: CallbackQuery, state: FSMContext):
    campaign_id = int(callback.data.rsplit("_", 1)[1])
    await state.set_state(AutomationAdminStates.quiet_hours)
    await state.update_data(campaign_id=campaign_id)
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
    await state.set_state(AutomationAdminStates.jitter)
    await state.update_data(campaign_id=campaign_id)
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
