from __future__ import annotations

import asyncio
import io
import json
import os
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import ai_integration
import database
import handlers
import bot_commands
import keyboards as tg_kb
from database import (
    AIConfig,
    Base,
    BotGeneralConfig,
    Content,
    MediaCollection,
    MediaLibrary,
    Message as DBMessage,
    SubscriptionConfig,
    Topic,
    User,
    UserSubscription,
    UserTopicState,
    media_collection_items,
    topic_collection_association,
)
import max_messenger_bot.legacy as max_legacy
import max_messenger_bot.storage as max_storage
from max_messenger_bot import app as max_app
from max_messenger_bot.app import MaxBotApplication
from max_messenger_bot.models import IncomingCallback, IncomingMessage, Sender
from max_messenger_bot.services import (
    admin as max_admin,
    admin_clients as max_admin_clients,
    admin_export as max_admin_export,
    admin_mailing as max_admin_mailing,
    common as max_common,
    settings as max_settings,
    subscriptions as max_subscriptions,
    topics as max_topics,
)
from max_messenger_bot.storage import StateStore, StorageBase
import memory_mode
from memory_mode import (
    MEMORY_MODE_GLOBAL,
    MEMORY_MODE_RESET,
    MEMORY_MODE_TOPIC,
    apply_memory_mode_topic_switch,
)
from result_history import (
    TOPIC_WELCOME_ROLE,
    VISIBLE_HISTORY_ROLES,
    is_topic_welcome_shown,
    non_technical_role_filter,
    record_topic_welcome_shown,
    resolve_topic_entry_state,
    visible_history_role_filter,
)
from system_events import (
    build_main_dialogue_resume_system_message,
    build_topic_auto_start_system_message,
    build_topic_resume_system_message,
)


@pytest_asyncio.fixture
async def db_session(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test-welcome-resume.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(StorageBase.metadata.create_all)

    monkeypatch.setattr(database, "async_session_maker", sessions)
    monkeypatch.setattr(handlers, "async_session_maker", sessions)
    monkeypatch.setattr(bot_commands, "async_session_maker", sessions)
    monkeypatch.setattr(tg_kb, "async_session_maker", sessions)
    monkeypatch.setattr(ai_integration, "async_session_maker", sessions)
    monkeypatch.setattr(max_legacy, "async_session_maker", sessions)
    monkeypatch.setattr(max_storage, "async_session_maker", sessions)
    monkeypatch.setattr(max_common, "async_session_maker", sessions)
    monkeypatch.setattr(max_topics, "async_session_maker", sessions)
    monkeypatch.setattr(max_settings, "async_session_maker", sessions)
    monkeypatch.setattr(max_admin, "async_session_maker", sessions)
    monkeypatch.setattr(max_admin_mailing, "async_session_maker", sessions)
    monkeypatch.setattr(max_admin_clients, "async_session_maker", sessions)
    monkeypatch.setattr(max_admin_export, "async_session_maker", sessions)
    monkeypatch.setattr(max_app, "async_session_maker", sessions)
    monkeypatch.setattr(max_subscriptions, "async_session_maker", sessions)

    handlers.user_message_buffers.clear()
    handlers.user_isolated_turn_queues.clear()
    handlers.user_processing_tasks.clear()
    handlers.user_scheduling_locks.clear()
    handlers.user_locks.clear()
    handlers._ai_button_claims.clear()

    try:
        yield sessions
    finally:
        handlers.user_message_buffers.clear()
        handlers.user_isolated_turn_queues.clear()
        handlers.user_processing_tasks.clear()
        handlers.user_scheduling_locks.clear()
        handlers.user_locks.clear()
        handlers._ai_button_claims.clear()
        await engine.dispose()


def make_mock_bot():
    bot = AsyncMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=123, edit_text=AsyncMock()))
    bot.edit_message_text = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.send_document = AsyncMock()
    bot.get_chat = AsyncMock(return_value=SimpleNamespace(birthdate=None))
    return bot


def make_mock_state(data=None):
    state = MagicMock()
    stored_data = data.copy() if data else {}

    async def _get_data():
        return stored_data

    async def _update_data(**kwargs):
        stored_data.update(kwargs)

    async def _set_state(*args, **kwargs):
        pass

    async def _clear():
        stored_data.clear()

    state.get_data = AsyncMock(side_effect=_get_data)
    state.update_data = AsyncMock(side_effect=_update_data)
    state.set_state = AsyncMock(side_effect=_set_state)
    state.clear = AsyncMock(side_effect=_clear)
    return state


async def drain_tg_runner(user_id: int):
    while handlers._has_user_turn_work(user_id) or (user_id in handlers.user_processing_tasks and not handlers.user_processing_tasks[user_id].done()):
        task = handlers.user_processing_tasks.get(user_id)
        if task:
            await task
        await asyncio.sleep(0.01)


async def seed_env(sessions, memory_mode="topic", user_id=1001, username="testuser", topic_name="Психосоматика", auto_start=True):
    async with sessions() as session:
        session.add(AIConfig(id=1, memory_mode=memory_mode, provider="gemini", gemini_model="gemini-2.5-flash", gemini_api_key="test-key"))
        session.add(SubscriptionConfig(id=1, subscriptions_enabled=False))
        session.add(BotGeneralConfig(id=1, profile_collect_name=False, profile_collect_gender=False, profile_collect_age=False))
        user = User(
            id=user_id,
            username=username,
            first_name="Иван",
            name="Иван",
            gender="male",
            age="30",
            current_dialogue_id=1,
            current_topic_id=None,
            accepted_disclaimer=True,
        )
        session.add(user)
        topic = Topic(
            id=10,
            name=topic_name,
            start_message=f"Добро пожаловать в тему {topic_name}!",
            is_active=True,
            show_in_list=True,
            admin_only=False,
            auto_start_dialogue=auto_start,
        )
        session.add(topic)
        session.add(Content(key="start_message", is_visible=True, text_content="Добро пожаловать в бота!"))
        await session.commit()


# ==============================================================================
# JOURNEY 1: /start (TG) and Start Screen (MAX)
# ==============================================================================

@pytest.mark.asyncio
async def test_journey_1_start_screen_tg_and_max(db_session, monkeypatch):
    await seed_env(db_session)
    bot = make_mock_bot()
    state = make_mock_state()

    # TG /start
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        chat=SimpleNamespace(id=1001, type="private"),
        text="/start",
        reply_to_message=None,
        message_id=1,
        date=SimpleNamespace(timestamp=lambda: 1000),
        answer=AsyncMock(),
    )

    with patch.object(handlers, "is_admin", AsyncMock(return_value=False)), \
         patch.object(handlers, "render_static_content_telegram", new_callable=AsyncMock) as mock_render:
        await handlers.cmd_start(message, state, bot)
        mock_render.assert_awaited_once_with(bot, 1001, 1001, "start_message", is_start=True)

    # MAX Start Screen
    client = AsyncMock()
    app = MaxBotApplication(client=client)

    msg = IncomingMessage(
        raw={},
        message_id="msg_1",
        chat_id=1001,
        sender=Sender(user_id=1001, username="testuser", first_name="Иван", last_name=""),
        text="/start",
    )
    with patch.object(max_common, "is_admin", AsyncMock(return_value=False)), \
         patch.object(max_common, "show_start_screen", AsyncMock()) as mock_start:
        await app.handle_message(msg)
        mock_start.assert_awaited_once_with(client, 1001, 1001, None, app.states)


# ==============================================================================
# JOURNEY 2: Return to Main & Memory Modes (TOPIC, GLOBAL, RESET)
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("mem_mode", ["topic", "global", "reset"])
async def test_journey_2_return_to_main_memory_modes_tg(db_session, monkeypatch, mem_mode):
    await seed_env(db_session, memory_mode=mem_mode)
    bot = make_mock_bot()
    state = make_mock_state()

    async with db_session() as session:
        user = await session.get(User, 1001)
        user.current_topic_id = 10
        user.current_dialogue_id = 5
        if mem_mode == "topic":
            session.add(UserTopicState(user_id=1001, topic_id=0, dialogue_id=2))
            session.add(UserTopicState(user_id=1001, topic_id=10, dialogue_id=5))
        session.add(DBMessage(user_id=1001, role="user", content="msg in topic", dialogue_id=5, topic_id=10))
        if mem_mode == "topic":
            session.add(DBMessage(user_id=1001, role="user", content="old main msg", dialogue_id=2, topic_id=None))
        await session.commit()

    captured_prompt = None

    async def fake_generate_response(user_id, prompt_text, *args, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt_text
        return "Ответ в основном диалоге"

    monkeypatch.setattr("handlers.ai_integration.generate_response", fake_generate_response)
    monkeypatch.setattr("ai_integration.generate_response", fake_generate_response)

    cb = SimpleNamespace(
        id="cb_1",
        data="ai_btn:svc:topic:main",
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        message=SimpleNamespace(
            message_id=99,
            chat=SimpleNamespace(id=1001, type="private"),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="Основной диалог", callback_data="ai_btn:svc:topic:main")]]
            ),
        ),
        answer=AsyncMock(),
    )

    await handlers.process_response_button(cb, state, bot)
    await drain_tg_runner(1001)

    sent_texts = [call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "") for call in bot.send_message.call_args_list]
    assert any("✅ Мы вернулись в общий режим диалога." in t for t in sent_texts)

    async with db_session() as session:
        user = await session.get(User, 1001)
        assert user.current_topic_id is None
        if mem_mode == "topic":
            assert user.current_dialogue_id == 2
        elif mem_mode == "global":
            assert user.current_dialogue_id == 5
        elif mem_mode == "reset":
            assert user.current_dialogue_id == 6

        ai_msgs = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 1001, DBMessage.role == "assistant")
        )).scalars().all()
        assert len(ai_msgs) == 1
        assert ai_msgs[0].content == "Ответ в основном диалоге"
        assert ai_msgs[0].topic_id is None
        assert ai_msgs[0].dialogue_id == user.current_dialogue_id

    assert captured_prompt is not None
    assert "Пользователь вернулся в общий режим диалога" in captured_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("mem_mode", ["topic", "global", "reset"])
async def test_journey_2_return_to_main_memory_modes_max(db_session, monkeypatch, mem_mode):
    await seed_env(db_session, memory_mode=mem_mode)
    client = AsyncMock()
    states = StateStore()

    async with db_session() as session:
        user = await session.get(User, 1001)
        user.current_topic_id = 10
        user.current_dialogue_id = 5
        if mem_mode == "topic":
            session.add(UserTopicState(user_id=1001, topic_id=0, dialogue_id=2))
            session.add(UserTopicState(user_id=1001, topic_id=10, dialogue_id=5))
        await session.commit()

    captured_prompt = None

    async def fake_get_ai(user_id, prompt_text, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt_text
        return "MAX Ответ в основном диалоге"

    monkeypatch.setattr(max_common, "get_ai_response", fake_get_ai)

    await max_topics.reset_topic(client, chat_id=1001, user_id=1001, states=states)

    client.send_message.assert_any_await(
        chat_id=1001,
        text="✅ Мы вернулись в общий режим диалога.",
        attachments=ANY,
    )

    async with db_session() as session:
        user = await session.get(User, 1001)
        assert user.current_topic_id is None
        if mem_mode == "topic":
            assert user.current_dialogue_id == 2
        elif mem_mode == "global":
            assert user.current_dialogue_id == 5
        elif mem_mode == "reset":
            assert user.current_dialogue_id == 6

        ai_msgs = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 1001, DBMessage.role == "assistant")
        )).scalars().all()
        assert len(ai_msgs) == 1
        assert ai_msgs[0].content == "MAX Ответ в основном диалоге"
        assert ai_msgs[0].topic_id is None
        assert ai_msgs[0].dialogue_id == user.current_dialogue_id

    assert captured_prompt is not None
    assert "Пользователь вернулся в общий режим диалога" in captured_prompt


# ==============================================================================
# JOURNEY 3: Main Button Idempotency
# ==============================================================================

@pytest.mark.asyncio
async def test_journey_3_main_button_idempotency_tg(db_session, monkeypatch):
    await seed_env(db_session)
    bot = make_mock_bot()
    state = make_mock_state()

    async with db_session() as session:
        user = await session.get(User, 1001)
        user.current_topic_id = None
        user.current_dialogue_id = 1
        await session.commit()

    cb = SimpleNamespace(
        id="cb_idemp",
        data="ai_btn:svc:topic:main",
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        message=SimpleNamespace(
            message_id=99,
            chat=SimpleNamespace(id=1001, type="private"),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="Основной диалог", callback_data="ai_btn:svc:topic:main")]]
            ),
        ),
        answer=AsyncMock(),
    )

    await handlers.process_response_button(cb, state, bot)

    cb.answer.assert_awaited_once()
    bot.send_message.assert_not_called()
    assert 1001 not in handlers.user_processing_tasks


@pytest.mark.asyncio
async def test_journey_3_main_button_idempotency_max(db_session, monkeypatch):
    await seed_env(db_session)
    client = AsyncMock()
    states = StateStore()

    async with db_session() as session:
        user = await session.get(User, 1001)
        user.current_topic_id = None
        user.current_dialogue_id = 1
        await session.commit()

    await max_topics.reset_topic(client, chat_id=1001, user_id=1001, states=states)
    client.send_message.assert_not_called()


# ==============================================================================
# JOURNEY 4: First Topic Entry (auto_start=True vs False)
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("auto_start", [True, False])
async def test_journey_4_first_topic_entry_tg(db_session, monkeypatch, auto_start):
    await seed_env(db_session, auto_start=auto_start)
    bot = make_mock_bot()
    state = make_mock_state()

    captured_prompt = None

    async def fake_generate_response(user_id, prompt_text, *args, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt_text
        return "Приветствие в теме"

    monkeypatch.setattr("handlers.ai_integration.generate_response", fake_generate_response)
    monkeypatch.setattr("ai_integration.generate_response", fake_generate_response)

    cb = SimpleNamespace(
        id="cb_topic",
        data="select_topic_10",
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        message=SimpleNamespace(
            message_id=99,
            chat=SimpleNamespace(id=1001, type="private"),
            delete=AsyncMock(),
            answer=AsyncMock(),
        ),
        answer=AsyncMock(),
    )

    await handlers.process_topic_selection(cb, state, bot)
    await drain_tg_runner(1001)

    sent_texts = [call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "") for call in bot.send_message.call_args_list]
    assert any("Добро пожаловать в тему Психосоматика!" in t for t in sent_texts)

    async with db_session() as session:
        user = await session.get(User, 1001)
        is_shown = await is_topic_welcome_shown(session, 1001, user.current_dialogue_id, 10)
        assert is_shown is True

        ai_msgs = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 1001, DBMessage.role == "assistant")
        )).scalars().all()

        if auto_start:
            assert len(ai_msgs) == 1
            assert ai_msgs[0].content == "Приветствие в теме"
            assert ai_msgs[0].topic_id == 10
            assert captured_prompt is not None
            assert "Пользователь выбрал тему" in captured_prompt
        else:
            assert len(ai_msgs) == 0
            assert captured_prompt is None


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_start", [True, False])
async def test_journey_4_first_topic_entry_max(db_session, monkeypatch, auto_start):
    await seed_env(db_session, auto_start=auto_start)
    client = AsyncMock()
    states = StateStore()

    captured_prompt = None

    async def fake_get_ai(user_id, prompt_text, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt_text
        return "MAX Приветствие в теме"

    monkeypatch.setattr(max_common, "get_ai_response", fake_get_ai)

    await max_topics.select_topic(client, chat_id=1001, user_id=1001, topic_id=10, states=states)

    client.send_message.assert_any_await(
        chat_id=1001,
        text="Добро пожаловать в тему Психосоматика!",
        attachments=ANY,
    )

    async with db_session() as session:
        user = await session.get(User, 1001)
        is_shown = await is_topic_welcome_shown(session, 1001, user.current_dialogue_id, 10)
        assert is_shown is True

        ai_msgs = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 1001, DBMessage.role == "assistant")
        )).scalars().all()

        if auto_start:
            assert len(ai_msgs) == 1
            assert ai_msgs[0].content == "MAX Приветствие в теме"
            assert ai_msgs[0].topic_id == 10
        else:
            assert len(ai_msgs) == 0


# ==============================================================================
# JOURNEY 5: Technical topic_welcome Role Isolation
# ==============================================================================

@pytest.mark.asyncio
async def test_journey_5_technical_role_isolation(db_session, monkeypatch):
    await seed_env(db_session)

    async with db_session() as session:
        session.add(DBMessage(user_id=1001, role="user", content="Привет", dialogue_id=1, topic_id=10))
        session.add(DBMessage(user_id=1001, role="assistant", content="Здравствуйте!", dialogue_id=1, topic_id=10))
        await record_topic_welcome_shown(session, 1001, 1, 10, content="shown")
        await session.commit()

    max_uid = 1_000_000_000_000 + 1001
    async with db_session() as session:
        session.add(User(id=max_uid, username="max_user", first_name="Max", name="Max", current_dialogue_id=1))
        session.add(DBMessage(user_id=max_uid, role="user", content="Привет MAX", dialogue_id=1, topic_id=10))
        session.add(DBMessage(user_id=max_uid, role="assistant", content="Здравствуйте MAX!", dialogue_id=1, topic_id=10))
        await record_topic_welcome_shown(session, max_uid, 1, 10, content="shown")
        await session.commit()

    # 1. AI context assembly: topic_welcome excluded
    async with db_session() as session:
        from result_history import ai_history_role_filter
        ai_msgs = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 1001, ai_history_role_filter(DBMessage))
        )).scalars().all()
        roles = [m.role for m in ai_msgs]
        assert "topic_welcome" not in roles
        assert roles == ["user", "assistant"]

    # 2. MAX admin history: show_client_history
    client = AsyncMock()
    await max_admin_clients.show_client_history(client, chat_id=999, target_user_id=max_uid, page=0)
    client.send_message.assert_awaited_once()
    text = client.send_message.call_args[1]["text"]
    assert "shown" not in text
    assert "topic_welcome" not in text
    assert "Привет MAX" in text

    # 3. MAX single export
    with patch.object(max_admin_clients, "_send_history_file", new_callable=AsyncMock) as mock_send_single:
        await max_admin_clients.run_single_export(client, chat_id=999, target_user_id=max_uid, fmt="json", anonymize=False)
        mock_send_single.assert_awaited_once()
        file_bytes = mock_send_single.call_args[0][2]
        parsed = json.loads(file_bytes.decode("utf-8"))
        for m in parsed:
            assert m["role"] != "topic_welcome"
            assert m["role"] in ("user", "assistant", "test_result")

    # 4. MAX mass export
    states_mass = StateStore()
    await states_mass.set(999, 999, "admin_export", {"export_all": True})
    with patch.object(max_admin_export, "_send_export_file", new_callable=AsyncMock) as mock_send_mass:
        await max_admin_export.run_mass_export(client, states=states_mass, chat_id=999, user_id=999, fmt="json", anonymize=False)
        mock_send_mass.assert_awaited_once()
        file_bytes_mass = mock_send_mass.call_args[0][2]
        parsed_mass = json.loads(file_bytes_mass.decode("utf-8"))
        for u in parsed_mass:
            for m in u.get("messages", []):
                assert m["role"] != "topic_welcome"

    # 5. TG admin history paging
    callback = SimpleNamespace(
        id="cb_hist",
        from_user=SimpleNamespace(id=999, username="admin"),
        message=SimpleNamespace(
            chat=SimpleNamespace(id=999, type="private"),
            message_id=10,
            text="",
            edit_text=AsyncMock(),
            answer=AsyncMock(),
        ),
        data="admin_history_1001_page_1",
        answer=AsyncMock(),
    )
    with patch.object(handlers, "is_admin", AsyncMock(return_value=True)):
        await handlers.view_user_history_page(user_id=1001, page=0, original_message=callback, for_admin_view=True)
        callback.message.edit_text.assert_awaited_once()
        hist_text = callback.message.edit_text.call_args[0][0]
        assert "shown" not in hist_text
        assert "topic_welcome" not in hist_text

    # 6. TG single client export via process_single_export
    bot = make_mock_bot()
    thinking_msg = AsyncMock()
    thinking_msg.edit_text = AsyncMock()
    cb_export = SimpleNamespace(
        id="cb_exp",
        from_user=SimpleNamespace(id=999, username="admin"),
        message=SimpleNamespace(
            chat=SimpleNamespace(id=999, type="private"),
            message_id=10,
            answer=AsyncMock(return_value=thinking_msg),
            answer_document=AsyncMock(),
        ),
        data="run_single_json_no_1001_all",
        answer=AsyncMock(),
    )
    with patch.object(handlers, "is_admin", AsyncMock(return_value=True)):
        await handlers.process_single_export(cb_export, bot)
        assert cb_export.message.answer_document.call_count >= 1
        doc_arg = cb_export.message.answer_document.call_args[0][0]
        doc_bytes = doc_arg.data if hasattr(doc_arg, "data") else (doc_arg.file.read() if hasattr(doc_arg, "file") else doc_arg)
        parsed_tg = json.loads(doc_bytes.decode("utf-8"))
        assert len(parsed_tg) >= 1
        for m in parsed_tg:
            assert m["role"] != "topic_welcome"


# ==============================================================================
# JOURNEY 6: Welcome Shown -> Leave without user message -> Return
# ==============================================================================

@pytest.mark.asyncio
async def test_journey_6_welcome_leave_return_tg(db_session, monkeypatch):
    await seed_env(db_session, auto_start=True)
    bot = make_mock_bot()
    state = make_mock_state()

    async def fake_generate_response(user_id, prompt_text, *args, **kwargs):
        return "AI ответ"

    monkeypatch.setattr("handlers.ai_integration.generate_response", fake_generate_response)
    monkeypatch.setattr("ai_integration.generate_response", fake_generate_response)

    # 1. First entry into topic 10
    cb_topic = SimpleNamespace(
        id="cb_1",
        data="select_topic_10",
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        message=SimpleNamespace(
            message_id=99,
            chat=SimpleNamespace(id=1001, type="private"),
            delete=AsyncMock(),
            answer=AsyncMock(),
        ),
        answer=AsyncMock(),
    )
    await handlers.process_topic_selection(cb_topic, state, bot)
    await drain_tg_runner(1001)

    sent_texts_1 = [call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "") for call in bot.send_message.call_args_list]
    assert any("Добро пожаловать в тему" in t for t in sent_texts_1)
    bot.send_message.reset_mock()

    # 2. Leave to main without sending any user messages
    cb_main = SimpleNamespace(
        id="cb_2",
        data="ai_btn:svc:topic:main",
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        message=SimpleNamespace(
            message_id=99,
            chat=SimpleNamespace(id=1001, type="private"),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="Основной диалог", callback_data="ai_btn:svc:topic:main")]]
            ),
        ),
        answer=AsyncMock(),
    )
    await handlers.process_response_button(cb_main, state, bot)
    await drain_tg_runner(1001)
    bot.send_message.reset_mock()

    # 3. Return to topic 10
    cb_return = SimpleNamespace(
        id="cb_3",
        data="select_topic_10",
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        message=SimpleNamespace(
            message_id=99,
            chat=SimpleNamespace(id=1001, type="private"),
            delete=AsyncMock(),
            answer=AsyncMock(),
        ),
        answer=AsyncMock(),
    )
    await handlers.process_topic_selection(cb_return, state, bot)
    await drain_tg_runner(1001)

    # Welcome must NOT be repeated!
    sent_texts_return = [call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "") for call in bot.send_message.call_args_list]
    assert not any("Добро пожаловать в тему" in t for t in sent_texts_return)


@pytest.mark.asyncio
async def test_journey_6_welcome_leave_return_max(db_session, monkeypatch):
    await seed_env(db_session, auto_start=True)
    client = AsyncMock()
    states = StateStore()

    async def fake_get_ai(user_id, prompt_text, **kwargs):
        return "MAX AI ответ"

    monkeypatch.setattr(max_common, "get_ai_response", fake_get_ai)

    # 1. First entry into topic 10
    await max_topics.select_topic(client, chat_id=1001, user_id=1001, topic_id=10, states=states)
    welcome_calls = [c for c in client.send_message.call_args_list if "Добро пожаловать в тему" in str(c)]
    assert len(welcome_calls) == 1
    client.send_message.reset_mock()

    # 2. Leave to main
    await max_topics.reset_topic(client, chat_id=1001, user_id=1001, states=states)
    client.send_message.reset_mock()

    # 3. Return to topic 10
    await max_topics.select_topic(client, chat_id=1001, user_id=1001, topic_id=10, states=states)
    welcome_calls_return = [c for c in client.send_message.call_args_list if "Добро пожаловать в тему" in str(c)]
    assert len(welcome_calls_return) == 0


# ==============================================================================
# JOURNEY 7: Resume Existing Topic Dialogue
# ==============================================================================

@pytest.mark.asyncio
async def test_journey_7_resume_existing_topic_dialogue_tg(db_session, monkeypatch):
    await seed_env(db_session, auto_start=False)
    bot = make_mock_bot()
    state = make_mock_state()

    # Establish previous dialogue in topic 10
    async with db_session() as session:
        user = await session.get(User, 1001)
        user.current_topic_id = None
        user.current_dialogue_id = 1
        session.add(UserTopicState(user_id=1001, topic_id=10, dialogue_id=42))
        session.add(DBMessage(user_id=1001, role="user", content="Старое сообщение в теме", dialogue_id=42, topic_id=10))
        await record_topic_welcome_shown(session, 1001, 42, 10, content="shown")
        await session.commit()

    captured_prompt = None

    async def fake_generate_response(user_id, prompt_text, *args, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt_text
        return "Продолжение диалога"

    monkeypatch.setattr("handlers.ai_integration.generate_response", fake_generate_response)
    monkeypatch.setattr("ai_integration.generate_response", fake_generate_response)

    cb_topic = SimpleNamespace(
        id="cb_resume",
        data="select_topic_10",
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        message=SimpleNamespace(
            message_id=99,
            chat=SimpleNamespace(id=1001, type="private"),
            delete=AsyncMock(),
            answer=AsyncMock(),
        ),
        answer=AsyncMock(),
    )

    await handlers.process_topic_selection(cb_topic, state, bot)
    await drain_tg_runner(1001)

    async with db_session() as session:
        user = await session.get(User, 1001)
        assert user.current_dialogue_id == 42
        assert user.current_topic_id == 10

    # No welcome shown
    sent_texts = [call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "") for call in bot.send_message.call_args_list]
    assert not any("Добро пожаловать" in t for t in sent_texts)

    # Resume kickoff executed
    assert captured_prompt is not None
    assert "Пользователь вернулся к теме" in captured_prompt


# ==============================================================================
# JOURNEY 8: Hidden AI Kickoff Persistence Normalization
# ==============================================================================

@pytest.mark.asyncio
async def test_journey_8_hidden_kickoff_no_user_role_saved_tg(db_session, monkeypatch):
    await seed_env(db_session, auto_start=True)
    bot = make_mock_bot()
    state = make_mock_state()

    async def fake_generate_response(user_id, prompt_text, *args, **kwargs):
        return "Ответ ИИ на скрытый кикофф"

    monkeypatch.setattr("handlers.ai_integration.generate_response", fake_generate_response)
    monkeypatch.setattr("ai_integration.generate_response", fake_generate_response)

    cb_topic = SimpleNamespace(
        id="cb_kickoff",
        data="select_topic_10",
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        message=SimpleNamespace(
            message_id=99,
            chat=SimpleNamespace(id=1001, type="private"),
            delete=AsyncMock(),
            answer=AsyncMock(),
        ),
        answer=AsyncMock(),
    )

    await handlers.process_topic_selection(cb_topic, state, bot)
    await drain_tg_runner(1001)

    async with db_session() as session:
        user = await session.get(User, 1001)
        user_msgs = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 1001, DBMessage.role == "user")
        )).scalars().all()
        assert len(user_msgs) == 0

        ai_msgs = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 1001, DBMessage.role == "assistant")
        )).scalars().all()
        assert len(ai_msgs) == 1
        assert ai_msgs[0].content == "Ответ ИИ на скрытый кикофф"
        assert ai_msgs[0].dialogue_id == user.current_dialogue_id
        assert ai_msgs[0].topic_id == 10


@pytest.mark.asyncio
async def test_journey_8_hidden_kickoff_no_user_role_saved_max(db_session, monkeypatch):
    await seed_env(db_session, auto_start=True)
    client = AsyncMock()
    states = StateStore()

    async def fake_get_ai(user_id, prompt_text, **kwargs):
        return "MAX Ответ ИИ на скрытый кикофф"

    monkeypatch.setattr(max_common, "get_ai_response", fake_get_ai)

    await max_topics.select_topic(client, chat_id=1001, user_id=1001, topic_id=10, states=states)

    async with db_session() as session:
        user_msgs = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 1001, DBMessage.role == "user")
        )).scalars().all()
        assert len(user_msgs) == 0

        ai_msgs = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 1001, DBMessage.role == "assistant")
        )).scalars().all()
        assert len(ai_msgs) == 1
        assert ai_msgs[0].content == "MAX Ответ ИИ на скрытый кикофф"


# ==============================================================================
# JOURNEY 9: Memory Reset inside Topic vs Main
# ==============================================================================

@pytest.mark.asyncio
async def test_journey_9_memory_reset_in_topic_vs_main_tg(db_session, monkeypatch):
    await seed_env(db_session, auto_start=True)
    bot = make_mock_bot()
    token = "rst_tok_123"
    state = make_mock_state({
        "reset_token": token,
        "reset_dialogue_id": 1,
        "reset_topic_id": 10,
    })

    # 1. Reset in topic
    async with db_session() as session:
        user = await session.get(User, 1001)
        user.current_topic_id = 10
        user.current_dialogue_id = 1
        await record_topic_welcome_shown(session, 1001, 1, 10, content="shown")
        await session.commit()

    async def fake_generate_response(user_id, prompt_text, *args, **kwargs):
        return "Ответ после сброса"

    monkeypatch.setattr("handlers.ai_integration.generate_response", fake_generate_response)
    monkeypatch.setattr("ai_integration.generate_response", fake_generate_response)

    cb_reset_topic = SimpleNamespace(
        id="cb_rst_top",
        data=f"reset_topic_keep:{token}",
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        message=SimpleNamespace(
            message_id=99,
            chat=SimpleNamespace(id=1001, type="private"),
            delete=AsyncMock(),
            answer=AsyncMock(),
        ),
        answer=AsyncMock(),
    )

    await handlers.process_reset_topic_keep(cb_reset_topic, state, bot)
    await drain_tg_runner(1001)

    sent_texts = [call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "") for call in bot.send_message.call_args_list]
    assert any("Добро пожаловать в тему" in t for t in sent_texts)

    async with db_session() as session:
        user = await session.get(User, 1001)
        assert user.current_dialogue_id == 2
        assert await is_topic_welcome_shown(session, 1001, 2, 10) is True

    # 2. Reset to main
    bot.send_message.reset_mock()
    token_main = "rst_tok_main_456"
    state_main = make_mock_state({
        "reset_token": token_main,
        "reset_dialogue_id": 2,
        "reset_topic_id": 10,
    })
    cb_reset_main = SimpleNamespace(
        id="cb_rst_main",
        data=f"reset_topic_to_main:{token_main}",
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        message=SimpleNamespace(
            message_id=99,
            chat=SimpleNamespace(id=1001, type="private"),
            delete=AsyncMock(),
            answer=AsyncMock(),
        ),
        answer=AsyncMock(),
    )

    await handlers.process_reset_topic_to_main(cb_reset_main, state_main, bot)
    await drain_tg_runner(1001)

    sent_texts_main = [call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "") for call in bot.send_message.call_args_list]
    assert any("Мы вернулись в общий режим диалога" in t for t in sent_texts_main)
    assert not any("Добро пожаловать в тему" in t for t in sent_texts_main)


@pytest.mark.asyncio
async def test_journey_9_memory_reset_in_topic_vs_main_max(db_session, monkeypatch):
    await seed_env(db_session, auto_start=True)
    client = AsyncMock()
    states = StateStore()

    async def fake_get_ai(user_id, prompt_text, **kwargs):
        return "MAX Ответ после сброса"

    monkeypatch.setattr(max_common, "get_ai_response", fake_get_ai)

    # 1. Reset in topic
    async with db_session() as session:
        user = await session.get(User, 1001)
        user.current_topic_id = 10
        user.current_dialogue_id = 1
        await record_topic_welcome_shown(session, 1001, 1, 10, content="shown")
        await session.commit()

    token = "max_reset_tok"
    await states.set(1001, 1001, "confirm_reset", {"reset_token": token})
    await max_common.execute_dialogue_reset(client, states, chat_id=1001, user_id=1001, token=token, expected_dialogue_id=1, expected_topic_id=10)
    welcome_calls = [c for c in client.send_message.call_args_list if "Добро пожаловать в тему" in str(c)]
    assert len(welcome_calls) == 1

    async with db_session() as session:
        user = await session.get(User, 1001)
        assert user.current_dialogue_id == 2
        assert await is_topic_welcome_shown(session, 1001, 2, 10) is True

    # 2. Reset in main
    client.send_message.reset_mock()
    async with db_session() as session:
        user = await session.get(User, 1001)
        user.current_topic_id = None
        user.current_dialogue_id = 2
        await session.commit()

    token2 = "max_reset_tok_2"
    await states.set(1001, 1001, "confirm_reset", {"reset_token": token2})
    await max_common.execute_dialogue_reset(client, states, chat_id=1001, user_id=1001, token=token2, expected_dialogue_id=2, expected_topic_id=0)
    client.send_message.assert_any_await(
        chat_id=1001,
        text="✅ Память очищена.",
        attachments=ANY,
    )


# ==============================================================================
# JOURNEY 10: Access & Onboarding Gates on Resume
# ==============================================================================

@pytest.mark.asyncio
async def test_journey_10_onboarding_and_disclaimer_gates_tg(db_session, monkeypatch):
    await seed_env(db_session, auto_start=True)
    bot = make_mock_bot()
    state = make_mock_state()

    async with db_session() as session:
        user = await session.get(User, 1001)
        user.accepted_disclaimer = False
        session.add(Content(key="disclaimer", is_visible=True, text_content="Пожалуйста, примите правила."))
        await session.commit()

    cb_topic = SimpleNamespace(
        id="cb_gate",
        data="select_topic_10",
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        message=SimpleNamespace(
            message_id=99,
            chat=SimpleNamespace(id=1001, type="private"),
            delete=AsyncMock(),
            answer=AsyncMock(),
        ),
        answer=AsyncMock(),
    )

    await handlers.process_topic_selection(cb_topic, state, bot)
    await drain_tg_runner(1001)

    # Disclaimer sent, kickoff paused
    sent_texts = [call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "") for call in bot.send_message.call_args_list]
    assert any("Пожалуйста, примите правила." in t for t in sent_texts)
    assert 1001 not in handlers.user_processing_tasks

    # Now user accepts disclaimer
    captured_prompt = None

    async def fake_generate_response(user_id, prompt_text, *args, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt_text
        return "Ответ после принятия правил"

    monkeypatch.setattr("handlers.ai_integration.generate_response", fake_generate_response)
    monkeypatch.setattr("ai_integration.generate_response", fake_generate_response)

    cb_accept = SimpleNamespace(
        id="cb_acc",
        data="accept_disclaimer",
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        message=SimpleNamespace(
            message_id=100,
            chat=SimpleNamespace(id=1001, type="private"),
            delete=AsyncMock(),
            answer=AsyncMock(),
        ),
        answer=AsyncMock(),
    )

    await handlers.disclaimer_accepted_handler(cb_accept, state, bot)
    await drain_tg_runner(1001)

    assert captured_prompt is not None


@pytest.mark.asyncio
async def test_journey_10_onboarding_and_disclaimer_gates_max(db_session, monkeypatch):
    await seed_env(db_session, auto_start=True)
    client = AsyncMock()
    states = StateStore()

    async with db_session() as session:
        user = await session.get(User, 1001)
        user.accepted_disclaimer = False
        session.add(Content(key="disclaimer", is_visible=True, text_content="MAX Пожалуйста, примите правила."))
        await session.commit()

    await max_topics.select_topic(client, chat_id=1001, user_id=1001, topic_id=10, states=states)

    client.send_message.assert_any_await(
        chat_id=1001,
        text="MAX Пожалуйста, примите правила.",
        attachments=ANY,
    )
    state_rec = await states.get(1001)
    assert state_rec is not None
    assert state_rec.state == "awaiting_disclaimer_acceptance"


# ==============================================================================
# JOURNEY 11: TG Pre-Call Stale Kickoff Drop
# ==============================================================================

@pytest.mark.asyncio
async def test_journey_11_pre_call_stale_kickoff_drop_tg(db_session, monkeypatch):
    await seed_env(db_session, auto_start=True)
    bot = make_mock_bot()
    state = make_mock_state()

    ai_called_count = 0

    async def fake_generate_response(user_id, prompt_text, *args, **kwargs):
        nonlocal ai_called_count
        ai_called_count += 1
        return "Ответ"

    monkeypatch.setattr("handlers.ai_integration.generate_response", fake_generate_response)
    monkeypatch.setattr("ai_integration.generate_response", fake_generate_response)

    # User in DB is in main menu (topic None, dialogue 2)
    async with db_session() as session:
        user = await session.get(User, 1001)
        user.current_topic_id = None
        user.current_dialogue_id = 2
        await session.commit()

    # Enqueue kickoff with expected_topic_id = 10, dialogue_id = 1 (stale kickoff from prior scope)
    await handlers._start_telegram_hidden_kickoff(
        user_id=1001,
        bot=bot,
        state=state,
        synthetic_prompt="[СИСТЕМНОЕ СООБЩЕНИЕ: тест]",
        dialogue_id=1,
        topic_id=10,
    )

    await drain_tg_runner(1001)

    # Pre-call validation must drop the kickoff without calling AI provider
    assert ai_called_count == 0


# ==============================================================================
# JOURNEY 12: TG Provider-In-Flight Stale Navigation Drop
# ==============================================================================

@pytest.mark.asyncio
async def test_journey_12_in_flight_stale_navigation_drop_tg(db_session, monkeypatch):
    await seed_env(db_session, auto_start=True)
    bot = make_mock_bot()
    state = make_mock_state()

    provider_entered = asyncio.Event()
    provider_proceed = asyncio.Event()

    async def slow_generate_response(user_id, prompt_text, *args, **kwargs):
        provider_entered.set()
        await provider_proceed.wait()
        return "Запоздалый ответ ИИ для Топика 10"

    monkeypatch.setattr("handlers.ai_integration.generate_response", slow_generate_response)
    monkeypatch.setattr("ai_integration.generate_response", slow_generate_response)

    # User in topic 10, dialogue 1
    async with db_session() as session:
        user = await session.get(User, 1001)
        user.current_topic_id = 10
        user.current_dialogue_id = 1
        await session.commit()

    # Enqueue kickoff and launch drain_runner
    await handlers._start_telegram_hidden_kickoff(
        user_id=1001,
        bot=bot,
        state=state,
        synthetic_prompt="[СИСТЕМНОЕ СООБЩЕНИЕ: Тест инфлайт]",
        dialogue_id=1,
        topic_id=10,
    )

    # Wait until provider is entered
    await provider_entered.wait()

    # While provider is in flight, user switches to Topic 20
    async with db_session() as session:
        topic20 = Topic(id=20, name="Вторая тема", is_active=True, show_in_list=True, admin_only=False)
        session.add(topic20)
        user = await session.get(User, 1001)
        user.current_topic_id = 20
        user.current_dialogue_id = 3
        await session.commit()

    # Now let provider finish
    provider_proceed.set()
    await drain_tg_runner(1001)

    # Post-call check must discard the response:
    # 1. Not sent to user chat
    sent_late = [call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "") for call in bot.send_message.call_args_list if "Запоздалый ответ" in (call.args[1] if len(call.args) > 1 else call.kwargs.get("text", ""))]
    assert len(sent_late) == 0

    # 2. Not saved in DB
    async with db_session() as session:
        msgs = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 1001, DBMessage.role == "assistant")
        )).scalars().all()
        assert len(msgs) == 0


# ==============================================================================
# REMEDIATION REGRESSIONS (PR #14)
# ==============================================================================

async def drain_max_app(app: MaxBotApplication, user_id: int):
    while user_id in app.user_tasks and not app.user_tasks[user_id].done():
        await app.user_tasks[user_id]


def make_incoming_callback(chat_id: int, user_id: int, payload: str, callback_id: str = "cb1", message_id: str | None = "42") -> IncomingCallback:
    sender = Sender(user_id=user_id, username="testuser", first_name="Иван", last_name=None)
    return IncomingCallback(
        raw={},
        callback_id=callback_id,
        payload=payload,
        chat_id=chat_id,
        message_id=message_id,
        sender=sender,
    )


@pytest.mark.asyncio
async def test_remediation_1_max_list_clients_executes_without_name_error(db_session):
    """Regression 1: max_admin_clients.list_clients executes against DB without NameError for and_."""
    client = AsyncMock()
    max_uid1 = 1_000_000_000_001
    max_uid2 = 1_000_000_000_002

    async with db_session() as session:
        session.add(User(id=max_uid1, username="client1", first_name="Client1", name="Client1", current_dialogue_id=1))
        session.add(User(id=max_uid2, username="client2", first_name="Client2", name="Client2", current_dialogue_id=1))
        session.add(DBMessage(user_id=max_uid1, role="user", content="Привет", dialogue_id=1, topic_id=None))
        await session.commit()

    # Must execute cleanly without NameError
    await max_admin_clients.list_clients(client, chat_id=999, page=0)
    client.send_message.assert_awaited_once()
    assert "Список клиентов" in client.send_message.call_args[1]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    "svc:topic:main",
    "svc:topic:0",
    "svc:reset_topic",
    "reset_topic",
])
async def test_remediation_2_real_max_app_main_callbacks_pass_states(db_session, monkeypatch, payload):
    """Regression 2: Real MaxBotApplication.handle_callback passes StateStore to reset_topic."""
    await seed_env(db_session, auto_start=True)
    client = AsyncMock()
    app = MaxBotApplication(client)

    captured_prompt = None

    async def fake_get_ai(user_id, prompt_text, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt_text
        return "MAX AI ответ при возврате в основной диалог"

    monkeypatch.setattr(max_common, "get_ai_response", fake_get_ai)

    # Start user in topic 10
    async with db_session() as session:
        user = await session.get(User, 1001)
        user.current_topic_id = 10
        user.current_dialogue_id = 1
        await session.commit()

    cb = make_incoming_callback(chat_id=1001, user_id=1001, payload=payload)

    await app.handle_callback(cb)
    await drain_max_app(app, 1001)

    # 1. Callback ACKed
    client.answer_callback.assert_awaited()

    # 2. State transition committed
    async with db_session() as session:
        user = await session.get(User, 1001)
        assert user.current_topic_id is None

    # 3. Exactly one confirmation sent
    confirm_calls = [c for c in client.send_message.call_args_list if "Мы вернулись в общий режим диалога" in str(c)]
    assert len(confirm_calls) == 1

    # 4. Kickoff executed once
    assert captured_prompt is not None
    assert "Пользователь вернулся в общий режим диалога" in captured_prompt


@pytest.mark.asyncio
async def test_remediation_3_max_main_disclaimer_gates_and_continuation(db_session, monkeypatch):
    """Regression 3: MAX main resume with visible unaccepted disclaimer persists pending and executes kickoff after accept."""
    await seed_env(db_session, auto_start=True)
    client = AsyncMock()
    app = MaxBotApplication(client)
    states = app.states

    # User in topic 10, disclaimer visible and not accepted
    async with db_session() as session:
        user = await session.get(User, 1001)
        user.current_topic_id = 10
        user.current_dialogue_id = 1
        user.accepted_disclaimer = False
        session.add(Content(key="disclaimer", is_visible=True, text_content="MAX Правила и условия использования."))
        await session.commit()

    provider_called = False

    async def fake_get_ai(user_id, prompt_text, **kwargs):
        nonlocal provider_called
        provider_called = True
        return "MAX Ответ после согласия с правилами"

    monkeypatch.setattr(max_common, "get_ai_response", fake_get_ai)

    # 1. User clicks main dialogue
    cb_main = make_incoming_callback(chat_id=1001, user_id=1001, payload="svc:topic:main", callback_id="cb_disc_1")
    await app.handle_callback(cb_main)
    await drain_max_app(app, 1001)

    # Transition committed to DB
    expected_main_dialogue_id = None
    async with db_session() as session:
        user = await session.get(User, 1001)
        assert user.current_topic_id is None
        expected_main_dialogue_id = user.current_dialogue_id

    # Confirmation sent
    confirm_calls = [c for c in client.send_message.call_args_list if "Мы вернулись в общий режим диалога" in str(c)]
    assert len(confirm_calls) == 1

    # Disclaimer UI sent
    disc_calls = [c for c in client.send_message.call_args_list if "MAX Правила и условия" in str(c)]
    assert len(disc_calls) == 1

    # Provider NOT called yet
    assert not provider_called

    # Pending state in states store
    state_rec = await states.get(1001)
    assert state_rec is not None
    assert state_rec.state == "awaiting_disclaimer_acceptance"
    assert state_rec.data.get("pending_auto_start_kind") == "main_resume"
    assert state_rec.data.get("pending_auto_start_dialogue_id") == expected_main_dialogue_id
    assert state_rec.data.get("pending_auto_start_topic_id") is None

    # 2. User accepts disclaimer
    client.send_message.reset_mock()
    cb_accept = make_incoming_callback(chat_id=1001, user_id=1001, payload="disclaimer_accepted", callback_id="cb_disc_2")
    await app.handle_callback(cb_accept)
    await drain_max_app(app, 1001)

    # Kickoff executed once
    assert provider_called
    async with db_session() as session:
        user = await session.get(User, 1001)
        assert user.accepted_disclaimer is True
        ai_msgs = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 1001, DBMessage.role == "assistant")
        )).scalars().all()
        assert len(ai_msgs) == 1
        assert ai_msgs[0].content == "MAX Ответ после согласия с правилами"
        assert ai_msgs[0].topic_id is None
        assert ai_msgs[0].dialogue_id == expected_main_dialogue_id


@pytest.mark.asyncio
async def test_remediation_4_tg_main_disclaimer_continuation(db_session, monkeypatch):
    """Regression 4: TG main resume with visible unaccepted disclaimer persists pending and executes kickoff after accept."""
    await seed_env(db_session, auto_start=True)
    bot = make_mock_bot()
    state = make_mock_state()

    # User in topic 10, disclaimer visible and unaccepted
    async with db_session() as session:
        user = await session.get(User, 1001)
        user.current_topic_id = 10
        user.current_dialogue_id = 1
        user.accepted_disclaimer = False
        session.add(Content(key="disclaimer", is_visible=True, text_content="TG Правила и условия диалога."))
        await session.commit()

    captured_prompt = None

    async def fake_generate_response(user_id, prompt_text, *args, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt_text
        return "TG Ответ в основном диалоге после принятия дисклеймера"

    monkeypatch.setattr("handlers.ai_integration.generate_response", fake_generate_response)
    monkeypatch.setattr("ai_integration.generate_response", fake_generate_response)

    # 1. User clicks main dialogue button
    cb_main = SimpleNamespace(
        id="cb_tg_main",
        data="ai_btn:svc:topic:main",
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        message=SimpleNamespace(
            message_id=99,
            chat=SimpleNamespace(id=1001, type="private"),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="Основной диалог", callback_data="ai_btn:svc:topic:main")]]
            ),
        ),
        answer=AsyncMock(),
    )

    await handlers.process_response_button(cb_main, state, bot)
    await drain_tg_runner(1001)

    # Transition committed
    expected_main_dialogue_id = None
    async with db_session() as session:
        user = await session.get(User, 1001)
        assert user.current_topic_id is None
        expected_main_dialogue_id = user.current_dialogue_id

    # Confirmation + disclaimer UI sent
    sent_texts = [call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "") for call in bot.send_message.call_args_list]
    assert any("Мы вернулись в общий режим диалога" in t for t in sent_texts)
    assert any("TG Правила и условия диалога." in t for t in sent_texts)

    # Provider NOT called yet
    assert captured_prompt is None

    # Pending state in FSM
    fsm_data = await state.get_data()
    assert fsm_data.get("pending_auto_start_kind") == "main_resume"
    assert fsm_data.get("pending_auto_start_dialogue_id") == expected_main_dialogue_id
    assert fsm_data.get("pending_auto_start_topic_id") is None

    # 2. User accepts disclaimer
    bot.send_message.reset_mock()
    cb_accept = SimpleNamespace(
        id="cb_tg_acc",
        data="disclaimer_accepted",
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        message=SimpleNamespace(
            message_id=100,
            chat=SimpleNamespace(id=1001, type="private"),
            delete=AsyncMock(),
            answer=AsyncMock(),
        ),
        answer=AsyncMock(),
    )
    await handlers.disclaimer_accepted_handler(cb_accept, state, bot)
    await drain_tg_runner(1001)

    # Hidden kickoff executed once
    assert captured_prompt is not None
    assert "Пользователь вернулся в общий режим диалога" in captured_prompt

    async with db_session() as session:
        user = await session.get(User, 1001)
        assert user.accepted_disclaimer is True
        ai_msgs = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 1001, DBMessage.role == "assistant")
        )).scalars().all()
        assert len(ai_msgs) == 1
        assert ai_msgs[0].content == "TG Ответ в основном диалоге после принятия дисклеймера"
        assert ai_msgs[0].topic_id is None
        assert ai_msgs[0].dialogue_id == expected_main_dialogue_id

    # 3. Next normal user message continues same main dialogue
    msg = SimpleNamespace(
        message_id=101,
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        chat=SimpleNamespace(id=1001, type="private"),
        text="Продолжаем разговор в основном диалоге",
        answer=AsyncMock(),
        delete=AsyncMock(),
    )
    await handlers.handle_ai_chat(msg, state, bot)
    await drain_tg_runner(1001)

    async with db_session() as session:
        user = await session.get(User, 1001)
        assert user.current_topic_id is None
        assert user.current_dialogue_id == expected_main_dialogue_id
        user_msgs = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 1001, DBMessage.role == "user")
        )).scalars().all()
        assert len(user_msgs) == 1
        assert user_msgs[0].dialogue_id == expected_main_dialogue_id
        assert user_msgs[0].topic_id is None


@pytest.mark.asyncio
async def test_remediation_5_stale_main_pending_dropped_tg_and_max(db_session, monkeypatch):
    """Regression 5: Stale main resume pending kickoff is dropped if user changed scope before accepting disclaimer."""
    await seed_env(db_session, auto_start=True)

    # --- TG variant ---
    bot = make_mock_bot()
    state = make_mock_state()

    async with db_session() as session:
        user = await session.get(User, 1001)
        user.current_topic_id = 10
        user.current_dialogue_id = 1
        user.accepted_disclaimer = False
        session.add(Content(key="disclaimer", is_visible=True, text_content="Правила"))
        await session.commit()

    ai_calls_tg = 0

    async def fake_ai_tg(user_id, prompt_text, *args, **kwargs):
        nonlocal ai_calls_tg
        ai_calls_tg += 1
        return "Ответ"

    monkeypatch.setattr("handlers.ai_integration.generate_response", fake_ai_tg)
    monkeypatch.setattr("ai_integration.generate_response", fake_ai_tg)

    # Click main -> pending set
    cb_main = SimpleNamespace(
        id="cb_m",
        data="ai_btn:svc:topic:main",
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        message=SimpleNamespace(
            message_id=99,
            chat=SimpleNamespace(id=1001, type="private"),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="Основной диалог", callback_data="ai_btn:svc:topic:main")]]
            ),
        ),
        answer=AsyncMock(),
    )
    await handlers.process_response_button(cb_main, state, bot)
    await drain_tg_runner(1001)

    # User changes scope to Topic 20 in DB before accepting
    async with db_session() as session:
        topic20 = Topic(id=20, name="Тема 20", is_active=True, show_in_list=True, admin_only=False)
        session.add(topic20)
        user = await session.get(User, 1001)
        user.current_topic_id = 20
        user.current_dialogue_id = 5
        await session.commit()

    # Now accepts old disclaimer
    cb_acc = SimpleNamespace(
        id="cb_acc",
        data="disclaimer_accepted",
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        message=SimpleNamespace(
            message_id=100,
            chat=SimpleNamespace(id=1001, type="private"),
            delete=AsyncMock(),
            answer=AsyncMock(),
        ),
        answer=AsyncMock(),
    )
    await handlers.disclaimer_accepted_handler(cb_acc, state, bot)
    await drain_tg_runner(1001)

    # Main kickoff was dropped!
    assert ai_calls_tg == 0

    # --- MAX variant ---
    client = AsyncMock()
    app = MaxBotApplication(client)

    max_uid = 1_000_000_000_001
    async with db_session() as session:
        session.add(User(id=max_uid, username="max_stale", first_name="Max", name="Max", current_dialogue_id=1, current_topic_id=10, accepted_disclaimer=False))
        await session.commit()

    ai_calls_max = 0

    async def fake_ai_max(user_id, prompt_text, **kwargs):
        nonlocal ai_calls_max
        ai_calls_max += 1
        return "MAX Ответ"

    monkeypatch.setattr(max_common, "get_ai_response", fake_ai_max)

    cb_max = make_incoming_callback(chat_id=max_uid, user_id=max_uid, payload="svc:topic:main", callback_id="cb_max_1")
    await app.handle_callback(cb_max)
    await drain_max_app(app, max_uid)

    # Scope changed to Topic 20 before accepting
    async with db_session() as session:
        user = await session.get(User, max_uid)
        user.current_topic_id = 20
        user.current_dialogue_id = 6
        await session.commit()

    cb_acc_max = make_incoming_callback(chat_id=max_uid, user_id=max_uid, payload="disclaimer_accepted", callback_id="cb_max_2")
    await app.handle_callback(cb_acc_max)
    await drain_max_app(app, max_uid)

    # Main kickoff dropped in MAX too
    assert ai_calls_max == 0


@pytest.mark.asyncio
async def test_remediation_6_and_7_topic_welcome_activity_isolation_ordering(db_session):
    """Regressions 6 & 7: topic_welcome does not affect MAX normal client list or export-selection client list ordering."""
    from datetime import datetime
    client = AsyncMock()
    states = StateStore()

    max_uid_a = 1_000_000_000_001  # Recent topic_welcome only
    max_uid_b = 1_000_000_000_002  # Older real user/assistant conversation
    max_uid_c = 1_000_000_000_003  # No messages at all

    async with db_session() as session:
        session.add(User(id=max_uid_a, username="user_a", first_name="UserA", name="UserA", current_dialogue_id=1, created_at=datetime(2026, 9, 1, 10, 0, 0)))
        session.add(User(id=max_uid_b, username="user_b", first_name="UserB", name="UserB", current_dialogue_id=1, created_at=datetime(2026, 9, 1, 10, 0, 0)))
        session.add(User(id=max_uid_c, username="user_c", first_name="UserC", name="UserC", current_dialogue_id=1, created_at=datetime(2026, 9, 1, 10, 0, 0)))

        # User A has recent topic_welcome (timestamp = Sept 6)
        session.add(DBMessage(
            user_id=max_uid_a,
            role="topic_welcome",
            content="[TOPIC_WELCOME]",
            dialogue_id=1,
            topic_id=10,
            timestamp=datetime(2026, 9, 6, 12, 0, 0),
        ))

        # User B has older real conversation (timestamp = Sept 3)
        session.add(DBMessage(
            user_id=max_uid_b,
            role="user",
            content="Привет бот",
            dialogue_id=1,
            topic_id=None,
            timestamp=datetime(2026, 9, 3, 12, 0, 0),
        ))
        await session.commit()

    # 1. Normal MAX client list (Regression 6)
    await max_admin_clients.list_clients(client, chat_id=999, page=0)
    client.send_message.assert_awaited_once()
    keyboard_att = client.send_message.call_args[1]["attachments"]
    buttons = keyboard_att[0]["payload"]["buttons"]
    client_button_payloads = [btn["payload"] for row in buttons for btn in row if btn.get("payload", "").startswith("view_client_")]

    # User B (with real conversation) MUST be first, before User A and User C
    assert client_button_payloads[0] == f"view_client_{max_uid_b}"
    assert f"view_client_{max_uid_a}" in client_button_payloads
    assert f"view_client_{max_uid_c}" in client_button_payloads

    # 2. MAX export-selection client list (Regression 7)
    client.send_message.reset_mock()
    await max_admin_export.show_export_clients(client, states=states, chat_id=999, user_id=999, page=0)
    client.send_message.assert_awaited_once()
    export_keyboard_att = client.send_message.call_args[1]["attachments"]
    export_buttons = export_keyboard_att[0]["payload"]["buttons"]
    export_button_payloads = [btn["payload"] for row in export_buttons for btn in row if btn.get("payload", "").startswith("toggle_export_")]

    # User B MUST be first in export selection as well
    assert export_button_payloads[0] == f"toggle_export_{max_uid_b}_0"
    assert f"toggle_export_{max_uid_a}_0" in export_button_payloads
    assert f"toggle_export_{max_uid_c}_0" in export_button_payloads


@pytest.mark.asyncio
async def test_remediation_8_tg_post_provider_processing_race_drops_stale_response(db_session, monkeypatch):
    """Regression 8: TG navigation after provider return but before visible response delivery drops stale response."""
    await seed_env(db_session, auto_start=True)
    bot = make_mock_bot()
    state = make_mock_state()

    # Start user in Topic 10, dialogue 1
    async with db_session() as session:
        user = await session.get(User, 1001)
        user.current_topic_id = 10
        user.current_dialogue_id = 1
        await session.commit()

    async def fake_generate_response(user_id, prompt_text, *args, **kwargs):
        return "AI ответ для Топика 10 перед навигацией [SHOW_IMG:test_img]"

    monkeypatch.setattr("handlers.ai_integration.generate_response", fake_generate_response)
    monkeypatch.setattr("ai_integration.generate_response", fake_generate_response)

    # Provider returns successfully while user is still in Topic 10.
    # Immediate post-provider validator check PASSES normally with real function.
    # We hook a post-provider media preparation step to change DB scope to Topic 20 / dialogue 2.
    orig_media_handler = handlers.handle_ai_media_content

    async def hooked_media_handler(bot_instance, uid, raw_text):
        # User changes DB scope during post-provider processing
        async with db_session() as session:
            topic20 = Topic(id=20, name="Тема 20", is_active=True, show_in_list=True, admin_only=False)
            session.add(topic20)
            u = await session.get(User, uid)
            u.current_topic_id = 20
            u.current_dialogue_id = 2
            await session.commit()
        return await orig_media_handler(bot_instance, uid, raw_text)

    monkeypatch.setattr(handlers, "handle_ai_media_content", hooked_media_handler)

    # Launch hidden kickoff for Topic 10
    await handlers._start_telegram_hidden_kickoff(
        user_id=1001,
        bot=bot,
        state=state,
        synthetic_prompt="[СИСТЕМНОЕ СООБЩЕНИЕ: тест расы]",
        dialogue_id=1,
        topic_id=10,
    )
    await drain_tg_runner(1001)

    # 1. Stale AI text was dropped: not sent to chat
    sent_stale = [call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "") for call in bot.send_message.call_args_list if "AI ответ для Топика 10" in (call.args[1] if len(call.args) > 1 else call.kwargs.get("text", ""))]
    assert len(sent_stale) == 0

    # 2. Stale media/photos not sent
    assert bot.send_photo.call_count == 0

    # 3. Not persisted in DB under Topic 10 or Topic 20
    async with db_session() as session:
        msgs = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 1001, DBMessage.role == "assistant")
        )).scalars().all()
        assert len(msgs) == 0

    # 4. Topic 20 remains pristine; next ordinary message in Topic 20 works
    async def fake_normal_ai(user_id, prompt_text, *args, **kwargs):
        return "Нормальный ответ в теме 20"

    monkeypatch.setattr("handlers.ai_integration.generate_response", fake_normal_ai)
    monkeypatch.setattr("ai_integration.generate_response", fake_normal_ai)
    monkeypatch.setattr(handlers, "handle_ai_media_content", orig_media_handler)

    msg = SimpleNamespace(
        message_id=200,
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        chat=SimpleNamespace(id=1001, type="private"),
        text="Привет в теме 20",
        answer=AsyncMock(),
        delete=AsyncMock(),
    )
    await handlers.handle_ai_chat(msg, state, bot)
    await drain_tg_runner(1001)

    async with db_session() as session:
        user = await session.get(User, 1001)
        assert user.current_topic_id == 20
        assert user.current_dialogue_id == 2
        user_msgs = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 1001, DBMessage.role == "user")
        )).scalars().all()
        assert len(user_msgs) == 1
        assert user_msgs[0].topic_id == 20
        assert user_msgs[0].dialogue_id == 2

        ai_msgs = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 1001, DBMessage.role == "assistant")
        )).scalars().all()
        assert len(ai_msgs) == 1
        assert ai_msgs[0].content == "Нормальный ответ в теме 20"
        assert ai_msgs[0].topic_id == 20
        assert ai_msgs[0].dialogue_id == 2


@pytest.mark.asyncio
async def test_journey_tg_direct_topic_button_full_lifecycle(db_session, monkeypatch):
    """TG direct topic button:
    A. direct topic button, first entry, auto_start=False -> welcome once, marker exists, no hidden AI kickoff.
    B. leave topic -> return through SAME direct topic button -> saved dialogue restored where memory mode permits, no repeated welcome, immediate hidden resume kickoff.
    C. auto_start=True first entry -> welcome once, first-entry hidden kickoff.
    D. next ordinary message continues expected topic/dialogue.
    """
    await seed_env(db_session, memory_mode="topic", auto_start=False)
    bot = make_mock_bot()
    state = make_mock_state()

    captured_prompt = None

    async def fake_generate_response(user_id, prompt_text, *args, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt_text
        return "AI ответ по теме"

    monkeypatch.setattr("handlers.ai_integration.generate_response", fake_generate_response)
    monkeypatch.setattr("ai_integration.generate_response", fake_generate_response)

    # Add second topic with auto_start=True
    async with db_session() as session:
        session.add(Topic(
            id=20,
            name="Тема 20 Автостарт",
            start_message="Добро пожаловать в Тему 20!",
            is_active=True,
            show_in_main_menu=True,
            show_in_list=True,
            admin_only=False,
            auto_start_dialogue=True,
        ))
        # Ensure Topic 10 has show_in_main_menu=True
        topic10 = await session.get(Topic, 10)
        topic10.show_in_main_menu = True
        await session.commit()

    # --- A. Direct topic button, first entry, auto_start=False ---
    msg_topic10 = SimpleNamespace(
        message_id=101,
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        chat=SimpleNamespace(id=1001, type="private"),
        text="Психосоматика",
        answer=AsyncMock(),
        delete=AsyncMock(),
    )
    await handlers.handle_direct_topic_button(msg_topic10, 10, "Психосоматика", state, bot)
    await drain_tg_runner(1001)

    # Welcome sent once
    sent_texts = [call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "") for call in bot.send_message.call_args_list]
    assert any("Добро пожаловать в тему Психосоматика!" in t for t in sent_texts)
    bot.send_message.reset_mock()

    # Welcome marker recorded
    async with db_session() as session:
        user = await session.get(User, 1001)
        assert user.current_topic_id == 10
        topic10_dialogue_id = user.current_dialogue_id
        is_shown = await is_topic_welcome_shown(session, 1001, topic10_dialogue_id, 10)
        assert is_shown is True

    # No hidden kickoff because auto_start=False
    assert captured_prompt is None

    # Send a message in Topic 10 to establish history
    msg_user_in_10 = SimpleNamespace(
        message_id=102,
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        chat=SimpleNamespace(id=1001, type="private"),
        text="Вопрос в топике 10",
        answer=AsyncMock(),
        delete=AsyncMock(),
    )
    await handlers.handle_ai_chat(msg_user_in_10, state, bot)
    await drain_tg_runner(1001)
    bot.send_message.reset_mock()
    captured_prompt = None

    # --- B. Leave topic -> return through SAME direct topic button ---
    cb_main = SimpleNamespace(
        id="cb_m",
        data="ai_btn:svc:topic:main",
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        message=SimpleNamespace(
            message_id=103,
            chat=SimpleNamespace(id=1001, type="private"),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="Основной диалог", callback_data="ai_btn:svc:topic:main")]]
            ),
        ),
        answer=AsyncMock(),
    )
    await handlers.process_response_button(cb_main, state, bot)
    await drain_tg_runner(1001)
    bot.send_message.reset_mock()
    captured_prompt = None

    async with db_session() as session:
        user = await session.get(User, 1001)
        assert user.current_topic_id is None

    # Return through direct topic button
    msg_return_10 = SimpleNamespace(
        message_id=104,
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        chat=SimpleNamespace(id=1001, type="private"),
        text="Психосоматика",
        answer=AsyncMock(),
        delete=AsyncMock(),
    )
    await handlers.handle_direct_topic_button(msg_return_10, 10, "Психосоматика", state, bot)
    await drain_tg_runner(1001)

    # Assert: saved dialogue restored, no repeated welcome, immediate hidden resume kickoff
    async with db_session() as session:
        user = await session.get(User, 1001)
        assert user.current_topic_id == 10
        assert user.current_dialogue_id == topic10_dialogue_id

    sent_texts_ret = [call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "") for call in bot.send_message.call_args_list]
    assert not any("Добро пожаловать в тему" in t for t in sent_texts_ret)
    assert captured_prompt is not None
    assert "Пользователь вернулся к теме" in captured_prompt
    bot.send_message.reset_mock()
    captured_prompt = None

    # --- C. auto_start=True first entry via direct topic button ---
    msg_topic20 = SimpleNamespace(
        message_id=105,
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        chat=SimpleNamespace(id=1001, type="private"),
        text="Тема 20 Автостарт",
        answer=AsyncMock(),
        delete=AsyncMock(),
    )
    await handlers.handle_direct_topic_button(msg_topic20, 20, "Тема 20 Автостарт", state, bot)
    await drain_tg_runner(1001)

    # Welcome sent once and first-entry kickoff executed
    sent_texts_20 = [call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "") for call in bot.send_message.call_args_list]
    assert any("Добро пожаловать в Тему 20!" in t for t in sent_texts_20)
    assert captured_prompt is not None
    assert "Пользователь выбрал тему" in captured_prompt
    bot.send_message.reset_mock()
    captured_prompt = None

    # --- D. Next ordinary message continues expected topic/dialogue ---
    msg_next_in_20 = SimpleNamespace(
        message_id=106,
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        chat=SimpleNamespace(id=1001, type="private"),
        text="Вопрос в теме 20",
        answer=AsyncMock(),
        delete=AsyncMock(),
    )
    await handlers.handle_ai_chat(msg_next_in_20, state, bot)
    await drain_tg_runner(1001)

    async with db_session() as session:
        user = await session.get(User, 1001)
        assert user.current_topic_id == 20
        user_msgs = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 1001, DBMessage.role == "user", DBMessage.content == "Вопрос в теме 20")
        )).scalars().all()
        assert len(user_msgs) == 1
        assert user_msgs[0].topic_id == 20
        assert user_msgs[0].dialogue_id == user.current_dialogue_id


@pytest.mark.asyncio
async def test_journey_legacy_saved_topic_dialogue_tg_and_max(db_session, monkeypatch):
    """Legacy pre-PR dialogues without topic_welcome marker:
    - If exact scope has user/assistant/test_result history -> LEGACY RESUME (no welcome, immediate resume kickoff, marker lazily recorded).
    - If scope has test_result only -> LEGACY RESUME.
    - If scope has no conversation history -> FIRST ENTRY (welcome shown, marker recorded).
    """
    await seed_env(db_session, memory_mode="topic", auto_start=False)
    bot = make_mock_bot()
    state = make_mock_state()

    captured_prompt_tg = None

    async def fake_ai_tg(user_id, prompt_text, *args, **kwargs):
        nonlocal captured_prompt_tg
        captured_prompt_tg = prompt_text
        return "TG AI ответ"

    monkeypatch.setattr("handlers.ai_integration.generate_response", fake_ai_tg)
    monkeypatch.setattr("ai_integration.generate_response", fake_ai_tg)

    # 1. Setup legacy TG dialogue: user 1001 has topic 10 state at dialogue 5 with real messages but NO topic_welcome marker
    async with db_session() as session:
        session.add(UserTopicState(user_id=1001, topic_id=10, dialogue_id=5))
        session.add(DBMessage(user_id=1001, role="user", content="Старое сообщение пользователя", dialogue_id=5, topic_id=10))
        session.add(DBMessage(user_id=1001, role="assistant", content="Старый ответ ассистента", dialogue_id=5, topic_id=10))
        await session.commit()

    # User enters Topic 10 via select_topic callback
    cb_topic = SimpleNamespace(
        id="cb_leg_tg",
        data="select_topic_10",
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        message=SimpleNamespace(
            message_id=99,
            chat=SimpleNamespace(id=1001, type="private"),
            delete=AsyncMock(),
            answer=AsyncMock(),
        ),
        answer=AsyncMock(),
    )
    await handlers.process_topic_selection(cb_topic, state, bot)
    await drain_tg_runner(1001)

    # Assert: NO welcome shown
    sent_texts = [call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "") for call in bot.send_message.call_args_list]
    assert not any("Добро пожаловать в тему" in t for t in sent_texts)

    # Assert: Hidden resume kickoff executed
    assert captured_prompt_tg is not None
    assert "Пользователь вернулся к теме" in captured_prompt_tg

    # Assert: topic_welcome marker was lazily recorded
    async with db_session() as session:
        user = await session.get(User, 1001)
        assert user.current_topic_id == 10
        assert user.current_dialogue_id == 5
        marker = await session.scalar(
            select(DBMessage).where(
                DBMessage.user_id == 1001,
                DBMessage.dialogue_id == 5,
                DBMessage.topic_id == 10,
                DBMessage.role == TOPIC_WELCOME_ROLE,
            )
        )
        assert marker is not None
        assert marker.content == "legacy_resume"

    # 2. Setup legacy test_result-only scope: topic 30 at dialogue 7
    bot.send_message.reset_mock()
    captured_prompt_tg = None
    async with db_session() as session:
        session.add(Topic(id=30, name="Тема Тестов", start_message="Добро пожаловать в тесты!", is_active=True, show_in_list=True, admin_only=False))
        session.add(UserTopicState(user_id=1001, topic_id=30, dialogue_id=7))
        session.add(DBMessage(user_id=1001, role="test_result", content="Результаты теста", dialogue_id=7, topic_id=30))
        await session.commit()

    cb_test_topic = SimpleNamespace(
        id="cb_leg_test",
        data="select_topic_30",
        from_user=SimpleNamespace(id=1001, username="testuser", full_name="Иван"),
        message=SimpleNamespace(
            message_id=100,
            chat=SimpleNamespace(id=1001, type="private"),
            delete=AsyncMock(),
            answer=AsyncMock(),
        ),
        answer=AsyncMock(),
    )
    await handlers.process_topic_selection(cb_test_topic, state, bot)
    await drain_tg_runner(1001)

    # Assert: NO welcome shown for test_result-only legacy scope
    sent_texts_test = [call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "") for call in bot.send_message.call_args_list]
    assert not any("Добро пожаловать в тесты!" in t for t in sent_texts_test)
    assert captured_prompt_tg is not None
    assert "Пользователь вернулся к теме" in captured_prompt_tg

    # 3. Setup legacy MAX dialogue
    max_uid = 1_000_000_000_001
    client = AsyncMock()
    states = StateStore()

    captured_prompt_max = None

    async def fake_ai_max(user_id, prompt_text, **kwargs):
        nonlocal captured_prompt_max
        captured_prompt_max = prompt_text
        return "MAX AI ответ"

    monkeypatch.setattr(max_common, "get_ai_response", fake_ai_max)

    async with db_session() as session:
        session.add(User(id=max_uid, username="max_leg", first_name="Max", name="Max", current_dialogue_id=1, current_topic_id=None, accepted_disclaimer=True))
        session.add(UserTopicState(user_id=max_uid, topic_id=10, dialogue_id=3))
        session.add(DBMessage(user_id=max_uid, role="user", content="MAX старое сообщение", dialogue_id=3, topic_id=10))
        session.add(DBMessage(user_id=max_uid, role="assistant", content="MAX старый ответ", dialogue_id=3, topic_id=10))
        await session.commit()

    await max_topics.select_topic(client, chat_id=max_uid, user_id=max_uid, topic_id=10, states=states)

    # Assert: NO welcome sent in MAX
    welcome_calls = [c for c in client.send_message.call_args_list if "Добро пожаловать в тему" in str(c)]
    assert len(welcome_calls) == 0

    # Assert: MAX hidden resume kickoff executed
    assert captured_prompt_max is not None
    assert "Пользователь вернулся к теме" in captured_prompt_max

    # Assert: MAX marker lazily created
    async with db_session() as session:
        marker_max = await session.scalar(
            select(DBMessage).where(
                DBMessage.user_id == max_uid,
                DBMessage.dialogue_id == 3,
                DBMessage.topic_id == 10,
                DBMessage.role == TOPIC_WELCOME_ROLE,
            )
        )
        assert marker_max is not None
        assert marker_max.content == "legacy_resume"


@pytest.mark.asyncio
async def test_mailing_and_stats_technical_role_isolation(db_session):
    """Test that topic_welcome marker does not change mailing audience selection or total message stats:
    1. Shared / TG background worker no_dialogue audience: user with only topic_welcome is still selected.
    2. MAX admin mailing no_dialogue audience: user with only topic_welcome is still selected.
    3. MAX admin show_stats total_messages: user + assistant + topic_welcome -> total_messages increases by 2, not 3.
    """
    from datetime import datetime
    from database import Mailing
    from max_messenger_bot.services import admin as max_admin, admin_mailing as max_admin_mailing

    tg_user_id = 7001
    max_user_id = 1_000_000_000_701

    async with db_session() as session:
        # User 1 (TG): has only topic_welcome marker
        session.add(User(id=tg_user_id, username="tg_welcome_only", first_name="TG", name="TG", current_dialogue_id=1))
        session.add(DBMessage(user_id=tg_user_id, role=TOPIC_WELCOME_ROLE, content="shown", dialogue_id=1, topic_id=10))

        # User 2 (MAX): has only topic_welcome marker
        session.add(User(id=max_user_id, username="max_welcome_only", first_name="MAX", name="MAX", current_dialogue_id=1))
        session.add(DBMessage(user_id=max_user_id, role=TOPIC_WELCOME_ROLE, content="shown", dialogue_id=1, topic_id=10))

        # User 3 (MAX): has 1 user message, 1 assistant message, and 1 topic_welcome marker
        max_active_user_id = 1_000_000_000_702
        session.add(User(id=max_active_user_id, username="max_active", first_name="MAX Active", name="MAX Active", current_dialogue_id=1))
        session.add(DBMessage(user_id=max_active_user_id, role="user", content="Привет", dialogue_id=1, topic_id=10))
        session.add(DBMessage(user_id=max_active_user_id, role="assistant", content="Здравствуйте", dialogue_id=1, topic_id=10))
        session.add(DBMessage(user_id=max_active_user_id, role=TOPIC_WELCOME_ROLE, content="shown", dialogue_id=1, topic_id=10))

        await session.commit()

    # 1. TG / shared background worker no_dialogue audience query test
    async with db_session() as session:
        from background_worker import non_technical_role_filter as bg_non_tech
        from sqlalchemy import func
        subquery = select(DBMessage.user_id, func.count(DBMessage.id).label("msg_count")).where(
            bg_non_tech(DBMessage)
        ).group_by(
            DBMessage.user_id).subquery()
        target_users_stmt = select(User.id).outerjoin(subquery, User.id == subquery.c.user_id).where(
            (subquery.c.msg_count == None) | (subquery.c.msg_count <= 1))
        selected_tg_ids = (await session.execute(target_users_stmt)).scalars().all()

        # User with only topic_welcome has 0 conversational messages, so must be selected!
        assert tg_user_id in selected_tg_ids

    # 2. MAX admin mailing no_dialogue recipient resolution
    async with db_session() as session:
        max_no_dialogue_recipients = await max_admin_mailing._get_recipient_ids(session, "no_dialogue", 999)
        # User with only topic_welcome marker must be in "Кто не начал диалог"
        assert max_user_id in max_no_dialogue_recipients
        # User with real conversation must NOT be in "Кто не начал диалог"
        assert max_active_user_id not in max_no_dialogue_recipients

    # 3. MAX admin stats total_messages
    client = AsyncMock()
    await max_admin.show_stats(client, chat_id=999)
    client.send_message.assert_awaited_once()
    stats_text = client.send_message.call_args[1]["text"]
    # Total messages for MAX users: max_user_id (0 non-technical) + max_active_user_id (2 non-technical: user + assistant) = 2
    assert "<b>Сообщений всего:</b> 2" in stats_text


@pytest.mark.asyncio
async def test_choice_img_hidden_real_process_buffered_messages_and_stale_drop(db_session, monkeypatch):
    """
    A. Normal AI response with [CHOICE_IMG_HIDDEN: cards | 2]:
       -> send_card_album sends back cards
       -> selection prompt 'Выбери карту, которая тебе откликается:' sent
       -> card_selection_keyboard contains the actual card IDs
    B. Hidden scoped kickoff with scope change between album delivery and selection prompt:
       -> album sent
       -> scope check fails
       -> selection prompt NOT sent into new scope
    """
    await seed_env(db_session, auto_start=True)
    bot = make_mock_bot()
    state = make_mock_state()

    # Seed media library with back card and 2 cards in category "cards"
    async with db_session() as session:
        collection = MediaCollection(name="Test Cards")
        session.add(collection)
        await session.flush()
        cards = [
            MediaLibrary(id=200, category="cards", file_name="_back", file_id="file_back_123", media_type="photo"),
            MediaLibrary(id=201, category="cards", file_name="card1", file_id="file_card_201", media_type="photo"),
            MediaLibrary(id=202, category="cards", file_name="card2", file_id="file_card_202", media_type="photo"),
        ]
        session.add_all(cards)
        await session.flush()
        await session.execute(
            topic_collection_association.insert().values(topic_id=10, collection_id=collection.id)
        )
        await session.execute(
            media_collection_items.insert(),
            [{"collection_id": collection.id, "media_id": c.id} for c in cards],
        )
        await session.commit()

    # A. Normal AI response
    async with db_session() as session:
        user = await session.get(User, 1001)
        user.current_topic_id = 10
        user.current_dialogue_id = 1
        await session.commit()

    monkeypatch.setattr(
        "handlers.ai_integration.generate_response",
        AsyncMock(return_value="Вот расклад:\n[CHOICE_IMG_HIDDEN: cards | 2]"),
    )

    sent_albums = []
    async def mock_send_card_album(b, cid, file_ids, **kwargs):
        sent_albums.append((cid, file_ids))

    monkeypatch.setattr("handlers.send_card_album", mock_send_card_album)

    # Put a message in user_message_buffers to simulate normal turn
    handlers.user_message_buffers.setdefault(1001, []).append("Покажи карты")
    await handlers.process_buffered_messages(1001, bot, state)

    # Verify album sent with back card file_id
    assert len(sent_albums) == 1
    assert sent_albums[0][1] == ["file_back_123", "file_back_123"]

    # Verify selection prompt message sent with card IDs in reply_markup
    prompt_msgs = [call for call in bot.send_message.mock_calls if "Выбери карту, которая тебе откликается:" in str(call)]
    assert len(prompt_msgs) == 1
    # Check that inline keyboard contains buttons with callback_data containing card ids 201 and 202
    call_args, call_kwargs = prompt_msgs[0][1], prompt_msgs[0][2] if len(prompt_msgs[0]) > 2 else prompt_msgs[0].kwargs
    kb = call_kwargs.get("reply_markup")
    assert kb is not None
    inline_kbs = [btn.callback_data for row in kb.inline_keyboard for btn in row]
    assert any("201" in data for data in inline_kbs)
    assert any("202" in data for data in inline_kbs)

    # B. Hidden scoped kickoff: album completes, scope changes before selection prompt -> prompt dropped
    bot.reset_mock()
    sent_albums.clear()

    async with db_session() as session:
        user = await session.get(User, 1001)
        user.current_topic_id = 10
        user.current_dialogue_id = 1
        await session.commit()

    async def race_send_card_album(b, cid, file_ids, **kwargs):
        sent_albums.append((cid, file_ids))
        # Switch user scope in DB to main right after album completes before selection prompt
        async with db_session() as s:
            u = await s.get(User, 1001)
            u.current_topic_id = None
            u.current_dialogue_id = 2
            await s.commit()

    monkeypatch.setattr("handlers.send_card_album", race_send_card_album)

    kickoff = handlers.ScopedAIKickoff(
        user_id=1001,
        expected_dialogue_id=1,
        expected_topic_id=10,
        synthetic_prompt="[СИСТЕМНОЕ СООБЩЕНИЕ: kickoff]",
        is_hidden=True,
    )
    await handlers.process_buffered_messages(1001, bot, state, scoped_kickoff=kickoff)

    # Album was sent
    assert len(sent_albums) == 1
    # But selection prompt was NOT sent because scope check failed!
    prompt_msgs_race = [call for call in bot.send_message.mock_calls if "Выбери карту, которая тебе откликается:" in str(call)]
    assert len(prompt_msgs_race) == 0
