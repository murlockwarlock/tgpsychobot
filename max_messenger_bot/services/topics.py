from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from ..api import MaxApiClient
from ..keyboards import callback_button, inline_keyboard, main_menu_row, topics_keyboard
from ..legacy import AIConfig, Content, Topic, User, UserTopicState, async_session_maker
from ..storage import StateStore
from memory_mode import apply_memory_mode_topic_switch, normalize_memory_mode
from result_history import is_topic_welcome_shown, record_topic_welcome_shown, resolve_topic_entry_state


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
    async with async_session_maker() as session:
        db_user = await session.get(User, user.id)
        if not db_user:
            return
        restored = await apply_memory_mode_topic_switch(session, db_user, topic_id, current_memory_mode)
        db_user.current_topic_id = topic_id
        dialogue_id = db_user.current_dialogue_id
        welcome_shown = await resolve_topic_entry_state(session, user_id, dialogue_id, topic_id)

        from system_events import (
            build_topic_auto_start_system_message,
            build_topic_resume_system_message,
            record_navigation_system_event,
        )
        synthetic_text = (
            build_topic_resume_system_message(topic.name)
            if welcome_shown
            else build_topic_auto_start_system_message(topic.name)
        )
        nav_msg = await record_navigation_system_event(
            session,
            user_id=user_id,
            dialogue_id=dialogue_id,
            topic_id=topic_id,
            text=synthetic_text,
        )
        await session.commit()
        navigation_message_id = nav_msg.id

    if not welcome_shown:
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

        async with async_session_maker() as s_mark:
            await record_topic_welcome_shown(s_mark, user_id, dialogue_id, topic_id)
            await s_mark.commit()

        if auto_start:
            from . import common
            async with async_session_maker() as session:
                fresh_user = await session.get(User, user_id, options=[selectinload(User.subscription)])

            if fresh_user and not fresh_user.name:
                if states is not None:
                    await common.begin_onboarding(
                        client,
                        states,
                        chat_id,
                        user_id,
                        resume_data={
                            "pending_auto_start_topic_id": topic_id,
                            "pending_auto_start_dialogue_id": dialogue_id,
                            "pending_auto_start_kind": "first_entry",
                            "pending_auto_start_message_id": navigation_message_id,
                        },
                    )
                return

            if fresh_user and states is not None:
                if await common.maybe_require_disclaimer(
                    client,
                    states,
                    chat_id,
                    fresh_user,
                    resume_data={
                        "pending_auto_start_topic_id": topic_id,
                        "pending_auto_start_dialogue_id": dialogue_id,
                        "pending_auto_start_kind": "first_entry",
                        "pending_auto_start_message_id": navigation_message_id,
                    },
                ):
                    return

            if fresh_user and not await common.ensure_access_before_chat(client, chat_id, fresh_user):
                return

            await common.run_hidden_ai_kickoff(
                client,
                chat_id,
                user_id,
                synthetic_text,
                expected_dialogue_id=dialogue_id,
                expected_topic_id=topic_id,
                states=states,
                exclude_message_id=navigation_message_id,
            )
    else:
        from . import common
        async with async_session_maker() as session:
            fresh_user = await session.get(User, user_id, options=[selectinload(User.subscription)])

        if fresh_user and not fresh_user.name:
            if states is not None:
                await common.begin_onboarding(
                    client,
                    states,
                    chat_id,
                    user_id,
                    resume_data={
                        "pending_auto_start_topic_id": topic_id,
                        "pending_auto_start_dialogue_id": dialogue_id,
                        "pending_auto_start_kind": "resume",
                        "pending_auto_start_message_id": navigation_message_id,
                    },
                )
            return

        if fresh_user and states is not None:
            if await common.maybe_require_disclaimer(
                client,
                states,
                chat_id,
                fresh_user,
                resume_data={
                    "pending_auto_start_topic_id": topic_id,
                    "pending_auto_start_dialogue_id": dialogue_id,
                    "pending_auto_start_kind": "resume",
                    "pending_auto_start_message_id": navigation_message_id,
                },
            ):
                return

        if fresh_user and not await common.ensure_access_before_chat(client, chat_id, fresh_user):
            return

        await common.run_hidden_ai_kickoff(
            client,
            chat_id,
            user_id,
            synthetic_text,
            expected_dialogue_id=dialogue_id,
            expected_topic_id=topic_id,
            states=states,
            exclude_message_id=navigation_message_id,
        )


async def reset_topic(client: MaxApiClient, chat_id: int, user_id: int, states: StateStore | None = None) -> None:
    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        config = await session.get(AIConfig, 1)
        if not user or user.current_topic_id is None:
            return
        current_memory_mode = normalize_memory_mode(config)
        await apply_memory_mode_topic_switch(session, user, 0, current_memory_mode)
        user.current_topic_id = None
        dialogue_id = user.current_dialogue_id
        from system_events import build_main_dialogue_resume_system_message, record_navigation_system_event
        synthetic_text = build_main_dialogue_resume_system_message()
        nav_msg = await record_navigation_system_event(
            session,
            user_id=user_id,
            dialogue_id=dialogue_id,
            topic_id=None,
            text=synthetic_text,
        )
        await session.commit()
        navigation_message_id = nav_msg.id

    await client.send_message(
        chat_id=chat_id,
        text="✅ Мы вернулись в общий режим диалога.",
        attachments=inline_keyboard([main_menu_row()]),
    )

    from . import common
    async with async_session_maker() as session:
        fresh_user = await session.get(User, user_id, options=[selectinload(User.subscription)])

    if fresh_user and not fresh_user.name:
        if states is not None:
            await common.begin_onboarding(
                client,
                states,
                chat_id,
                user_id,
                resume_data={
                    "pending_auto_start_topic_id": None,
                    "pending_auto_start_dialogue_id": dialogue_id,
                    "pending_auto_start_kind": "main_resume",
                    "pending_auto_start_message_id": navigation_message_id,
                },
            )
        return

    if fresh_user and states is not None:
        if await common.maybe_require_disclaimer(
            client,
            states,
            chat_id,
            fresh_user,
            resume_data={
                "pending_auto_start_topic_id": None,
                "pending_auto_start_dialogue_id": dialogue_id,
                "pending_auto_start_kind": "main_resume",
                "pending_auto_start_message_id": navigation_message_id,
            },
        ):
            return

    if fresh_user and not await common.ensure_access_before_chat(client, chat_id, fresh_user):
        return

    await common.run_hidden_ai_kickoff(
        client,
        chat_id,
        user_id,
        synthetic_text,
        expected_dialogue_id=dialogue_id,
        expected_topic_id=None,
        states=states,
        exclude_message_id=navigation_message_id,
    )
