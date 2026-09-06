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
    Message as DBMessage,
    SubscriptionConfig,
    Topic,
    User,
    UserSubscription,
    UserTopicState,
)
import max_messenger_bot.legacy as max_legacy
import max_messenger_bot.storage as max_storage
from max_messenger_bot import app as max_app
from max_messenger_bot.app import MaxBotApplication
from max_messenger_bot.models import IncomingCallback, IncomingMessage, Sender
from max_messenger_bot.services import (
    admin_clients as max_admin_clients,
    admin_export as max_admin_export,
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

    # User starts in topic 10, dialogue 1
    async with db_session() as session:
        user = await session.get(User, 1001)
        user.current_topic_id = 10
        user.current_dialogue_id = 1
        await session.commit()

    # Enqueue kickoff with expected_topic_id = 10, dialogue_id = 1
    await handlers._start_telegram_hidden_kickoff(
        user_id=1001,
        bot=bot,
        state=state,
        synthetic_prompt="[СИСТЕМНОЕ СООБЩЕНИЕ: тест]",
        dialogue_id=1,
        topic_id=10,
    )

    # But change user scope in DB to main (None) before runner executes provider call
    async with db_session() as session:
        user = await session.get(User, 1001)
        user.current_topic_id = None
        user.current_dialogue_id = 2
        await session.commit()

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
