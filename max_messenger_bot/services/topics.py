from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from ..api import MaxApiClient
from ..keyboards import callback_button, inline_keyboard, main_menu_row, topics_keyboard
from ..legacy import AIConfig, Content, Topic, User, UserTopicState, async_session_maker
from memory_mode import apply_memory_mode_topic_switch, normalize_memory_mode


async def show_topics(client: MaxApiClient, chat_id: int, user_id: int) -> None:
    async with async_session_maker() as session:
        user = await session.get(User, user_id, options=[selectinload(User.current_topic)])
        config = await session.get(AIConfig, 1)
        is_admin_user = bool(user and user.is_admin)
        query = select(Topic).where(Topic.is_active == True, Topic.show_in_list == True)
        if not is_admin_user:
            query = query.where(Topic.admin_only == False)
        topics = (await session.execute(query.order_by(Topic.sort_order.asc(), Topic.id.asc()))).scalars().all()
        current_topic_id = user.current_topic_id if user else None
        current_status = "в <b>Основном диалоге</b>"
        if user and user.current_topic:
            current_status = f"в диалоге: <b>{user.current_topic.name}</b>"

    if not topics:
        await client.send_message(chat_id=chat_id, text="Сейчас нет доступных тем.")
        return

    text = (
        f"Вы находитесь {current_status}.\n"
        "Выберите тему для диалога."
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=topics_keyboard(topics, current_topic_id))


async def _apply_topic_switch(user: User, topic_id: int, memory_mode: str) -> bool:
    async with async_session_maker() as session:
        db_user = await session.get(User, user.id)
        if not db_user:
            return False
        restored = await apply_memory_mode_topic_switch(session, db_user, topic_id, memory_mode)
        db_user.current_topic_id = topic_id or None
        await session.commit()
    return restored


async def select_topic(client: MaxApiClient, chat_id: int, user_id: int, topic_id: int, states: StateStore | None = None) -> None:
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        topic = await session.get(Topic, topic_id)
        config = await session.get(AIConfig, 1)
    if (
        not user
        or not topic
        or not topic.is_active
        or (topic.admin_only and not user.is_admin)
    ):
        await client.send_message(chat_id=chat_id, text="Тема недоступна.")
        return

    if user.current_topic_id == topic_id:
        return

    current_memory_mode = normalize_memory_mode(config)
    restored = await _apply_topic_switch(user, topic_id, current_memory_mode)
    from ..formatting import translate_telegram_links_to_max
    if topic.start_message:
        text = translate_telegram_links_to_max(topic.start_message)
    elif current_memory_mode == "global":
        text = f"✅ Переключились на тему: <b>{topic.name}</b>.\n\nТекущий диалог продолжается. Память сохранена."
    elif restored:
        text = f"✅ Продолжаем тему: <b>{topic.name}</b>."
    else:
        text = f"✅ Переключились на тему: <b>{topic.name}</b>.\n\nПамять диалога очищена."

    auto_start = getattr(topic, "auto_start_dialogue", False)
    if auto_start:
        attachments = inline_keyboard([main_menu_row()])
    else:
        attachments = inline_keyboard([[
            callback_button("💬 Начать диалог", "topic_start_dialogue"),
        ], main_menu_row()])

    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=attachments,
    )

    if auto_start:
        from . import common
        async with async_session_maker() as session:
            fresh_user = await session.get(User, user_id, options=[selectinload(User.subscription)])

        if fresh_user and not fresh_user.name:
            if states is not None:
                await common.begin_onboarding(client, states, chat_id, user_id, resume_data={"pending_auto_start_topic_id": topic_id})
            return

        if fresh_user and states is not None:
            if await common.maybe_require_disclaimer(client, states, chat_id, fresh_user, resume_data={"pending_auto_start_topic_id": topic_id}):
                return

        if fresh_user and not await common.ensure_access_before_chat(client, chat_id, fresh_user):
            return

        from system_events import build_topic_auto_start_system_message
        synthetic_text = build_topic_auto_start_system_message(topic.name)
        await common.run_ai_dialogue(client, chat_id, user_id, synthetic_text, states=states)


async def reset_topic(client: MaxApiClient, chat_id: int, user_id: int) -> None:
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        config = await session.get(AIConfig, 1)
    if not user:
        return
    await _apply_topic_switch(user, 0, normalize_memory_mode(config))
    from .common import render_static_content
    await render_static_content(client, chat_id, user_id, "start_message", is_start=True)
    await client.send_message(chat_id=chat_id, text="✅ Тема сброшена.", attachments=inline_keyboard([main_menu_row()]))
