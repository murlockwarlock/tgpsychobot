from __future__ import annotations

import asyncio
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
import keyboards as tg_kb
from database import AIConfig, Base, Content, Message as DBMessage, SubscriptionConfig, Topic, User, UserTopicState
import max_messenger_bot.legacy as max_legacy
import max_messenger_bot.storage as max_storage
from max_messenger_bot import app as max_app
from max_messenger_bot.app import MaxBotApplication
from max_messenger_bot.models import IncomingCallback, IncomingMessage, Sender
from max_messenger_bot.services import (
    admin as max_admin,
    admin_content as max_admin_content,
    common as max_common,
    settings as max_settings,
    subscriptions as max_subscriptions,
    topics as max_topics,
)
from max_messenger_bot.storage import StorageBase
from memory_mode import MEMORY_MODE_GLOBAL, MEMORY_MODE_RESET, MEMORY_MODE_TOPIC
from response_buttons import (
    MAIN_TOPIC_ACTIONS,
    ResponseButton,
    build_action_callback_data,
    extract_response_buttons,
    split_action_callback_data,
)


@pytest_asyncio.fixture
async def db_session(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test-topic-main-btn.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(StorageBase.metadata.create_all)

    monkeypatch.setattr(database, "async_session_maker", sessions)
    monkeypatch.setattr(handlers, "async_session_maker", sessions)
    monkeypatch.setattr(tg_kb, "async_session_maker", sessions)
    monkeypatch.setattr(max_legacy, "async_session_maker", sessions)
    monkeypatch.setattr(max_storage, "async_session_maker", sessions)
    monkeypatch.setattr(max_common, "async_session_maker", sessions)
    monkeypatch.setattr(max_topics, "async_session_maker", sessions)
    monkeypatch.setattr(max_settings, "async_session_maker", sessions)
    monkeypatch.setattr(max_app, "async_session_maker", sessions)
    monkeypatch.setattr(max_subscriptions, "async_session_maker", sessions)
    monkeypatch.setattr(max_admin, "async_session_maker", sessions)
    monkeypatch.setattr(max_admin_content, "async_session_maker", sessions)

    handlers.user_message_buffers.clear()
    handlers.user_processing_tasks.clear()
    handlers._ai_button_claims.clear()

    try:
        yield sessions
    finally:
        handlers.user_message_buffers.clear()
        handlers.user_processing_tasks.clear()
        handlers._ai_button_claims.clear()
        await engine.dispose()


# ==============================================================================
# 1. PARSER / CALLBACK DATA
# ==============================================================================

def test_response_button_parser_canonical_and_aliases():
    assert "svc:topic:main" in MAIN_TOPIC_ACTIONS
    assert "svc:topic:0" in MAIN_TOPIC_ACTIONS
    assert "svc:reset_topic" in MAIN_TOPIC_ACTIONS

    # Canonical syntax
    clean, rows = extract_response_buttons("[Основной диалог](btn:svc:topic:main)")
    assert clean == ""
    assert len(rows) == 1
    assert rows[0][0] == ResponseButton(text="Основной диалог", kind="action", value="svc:topic:main")

    cb_canonical = build_action_callback_data(rows[0][0].value)
    assert cb_canonical == "ai_btn:svc:topic:main"
    action, idx = split_action_callback_data(cb_canonical)
    assert action == "svc:topic:main"
    assert idx is None

    # Indexed button callback
    cb_indexed = build_action_callback_data(rows[0][0].value, 1)
    assert cb_indexed == "ai_btn:svc:topic:main|01"
    action_idx, idx_val = split_action_callback_data(cb_indexed)
    assert action_idx == "svc:topic:main"
    assert idx_val == 1

    # Aliases
    clean_0, rows_0 = extract_response_buttons("[Основной диалог](btn:svc:topic:0)")
    assert rows_0[0][0] == ResponseButton(text="Основной диалог", kind="action", value="svc:topic:0")
    assert build_action_callback_data("svc:topic:0") == "ai_btn:svc:topic:0"
    assert split_action_callback_data("ai_btn:svc:topic:0") == ("svc:topic:0", None)

    clean_reset, rows_reset = extract_response_buttons("[Основной диалог](btn:svc:reset_topic)")
    assert rows_reset[0][0] == ResponseButton(text="Основной диалог", kind="action", value="svc:reset_topic")
    assert build_action_callback_data("svc:reset_topic") == "ai_btn:svc:reset_topic"
    assert split_action_callback_data("ai_btn:svc:reset_topic") == ("svc:reset_topic", None)


# ==============================================================================
# 2. TELEGRAM — PRIMARY JOURNEY
# ==============================================================================

@pytest.mark.asyncio
async def test_telegram_primary_journey_topic_to_main(db_session):
    async with db_session() as session:
        session.add(AIConfig(id=1, memory_mode=MEMORY_MODE_RESET))
        session.add(Topic(id=2, name="Topic B", is_active=True, admin_only=False))
        session.add(User(id=3001, username="test_tg", first_name="Tester", name="Tester", accepted_disclaimer=True, current_dialogue_id=1, current_topic_id=2, is_admin=True))
        session.add(DBMessage(user_id=3001, dialogue_id=1, topic_id=2, role="user", content="Вопрос в Topic B"))
        session.add(Content(key="start_message", text_content="Привет! Это главное меню.", is_visible=True))
        await session.commit()

    mock_bot = SimpleNamespace(
        send_message=AsyncMock(),
        send_chat_action=AsyncMock(),
    )
    mock_state = AsyncMock()

    cb = SimpleNamespace(
        data="ai_btn:svc:topic:main",
        from_user=SimpleNamespace(id=3001, username="test_tg", full_name="Tester"),
        message=SimpleNamespace(
            message_id=501,
            chat=SimpleNamespace(id=3001),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="Основной диалог", callback_data="ai_btn:svc:topic:main")]]
            ),
        ),
        answer=AsyncMock(),
    )

    # Press service button
    await handlers.process_response_button(cb, mock_state, mock_bot)

    # 1. Assert single ACK
    assert cb.answer.call_count == 1

    # 2. Assert user in DB: topic is None, dialogue_id advanced per RESET mode
    async with db_session() as session:
        user = await session.get(User, 3001)
        assert user.current_topic_id is None
        assert user.current_dialogue_id == 2

    # 3. Assert start/main interface and confirmation rendered
    assert mock_bot.send_message.call_count >= 1
    sent_texts = [call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "") for call in mock_bot.send_message.call_args_list]
    assert any("✅ Тема сброшена. Мы вернулись в общий режим диалога." in t for t in sent_texts)

    # 4. Next ordinary user message
    handlers.user_message_buffers.setdefault(3001, []).append("Новый вопрос в основном диалоге")

    with patch("ai_integration.generate_response", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "Ответ ассистента в основном диалоге"
        await handlers.process_buffered_messages(3001, mock_bot, mock_state)

    async with db_session() as session:
        messages = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 3001, DBMessage.dialogue_id == 2)
        )).scalars().all()
        assert len(messages) >= 2
        user_db_msg = next(m for m in messages if m.role == "user" and m.content == "Новый вопрос в основном диалоге")
        assert user_db_msg.topic_id is None
        assert user_db_msg.dialogue_id == 2


# ==============================================================================
# 3. TELEGRAM — ALIASES
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("alias_payload", ["ai_btn:svc:topic:0", "ai_btn:svc:reset_topic"])
async def test_telegram_aliases_topic_to_main(db_session, alias_payload):
    uid = 3002 if alias_payload.endswith("0") else 3003
    async with db_session() as session:
        session.add(AIConfig(id=1, memory_mode=MEMORY_MODE_RESET))
        session.add(Topic(id=3, name="Topic C", is_active=True, admin_only=False))
        session.add(User(id=uid, username=f"user_{uid}", first_name="Tester", name="Tester", accepted_disclaimer=True, current_dialogue_id=1, current_topic_id=3, is_admin=True))
        session.add(Content(key="start_message", text_content="Старт", is_visible=True))
        await session.commit()

    mock_bot = SimpleNamespace(send_message=AsyncMock())
    mock_state = AsyncMock()

    cb = SimpleNamespace(
        data=alias_payload,
        from_user=SimpleNamespace(id=uid, username=f"user_{uid}", full_name="Tester"),
        message=SimpleNamespace(
            message_id=503,
            chat=SimpleNamespace(id=uid),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="Основной диалог", callback_data=alias_payload)]]
            ),
        ),
        answer=AsyncMock(),
    )

    await handlers.process_response_button(cb, mock_state, mock_bot)
    assert cb.answer.call_count == 1

    async with db_session() as session:
        user = await session.get(User, uid)
        assert user.current_topic_id is None
        assert user.current_dialogue_id == 2


# ==============================================================================
# 4. TELEGRAM — ALREADY MAIN IDEMPOTENCY
# ==============================================================================

@pytest.mark.asyncio
async def test_telegram_already_main_idempotency(db_session):
    async with db_session() as session:
        session.add(AIConfig(id=1, memory_mode=MEMORY_MODE_RESET))
        session.add(User(id=3004, username="main_user", first_name="Tester", name="Tester", accepted_disclaimer=True, current_dialogue_id=5, current_topic_id=None, is_admin=True))
        session.add(DBMessage(user_id=3004, dialogue_id=5, topic_id=None, role="user", content="Старое сообщение в диалоге 5"))
        session.add(Content(key="start_message", text_content="Старт", is_visible=True))
        await session.commit()

    mock_bot = SimpleNamespace(
        send_message=AsyncMock(),
        send_chat_action=AsyncMock(),
    )
    mock_state = AsyncMock()

    cb = SimpleNamespace(
        data="ai_btn:svc:topic:main",
        from_user=SimpleNamespace(id=3004, username="main_user", full_name="Tester"),
        message=SimpleNamespace(
            message_id=504,
            chat=SimpleNamespace(id=3004),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="Основной диалог", callback_data="ai_btn:svc:topic:main")]]
            ),
        ),
        answer=AsyncMock(),
    )

    await handlers.process_response_button(cb, mock_state, mock_bot)

    # 1. Single ACK
    assert cb.answer.call_count == 1

    # 2. dialogue_id NOT changed, topic_id remains None
    async with db_session() as session:
        user = await session.get(User, 3004)
        assert user.current_topic_id is None
        assert user.current_dialogue_id == 5

    # 3. Next message stays in dialogue 5
    handlers.user_message_buffers.setdefault(3004, []).append("Вопрос после нажатия кнопки")

    with patch("ai_integration.generate_response", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "Ответ в диалоге 5"
        await handlers.process_buffered_messages(3004, mock_bot, mock_state)

    async with db_session() as session:
        messages = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 3004, DBMessage.dialogue_id == 5)
        )).scalars().all()
        assert any(m.content == "Вопрос после нажатия кнопки" and m.topic_id is None for m in messages)


# ==============================================================================
# 5. MAX — PRIMARY JOURNEY
# ==============================================================================

@pytest.mark.asyncio
async def test_max_primary_journey_topic_to_main(db_session):
    async with db_session() as session:
        session.add(AIConfig(id=1, memory_mode=MEMORY_MODE_RESET))
        session.add(Topic(id=4, name="Topic Max B", is_active=True, admin_only=False))
        session.add(User(id=4001, first_name="MaxUser", name="MaxUser", accepted_disclaimer=True, current_dialogue_id=1, current_topic_id=4, is_admin=True))
        session.add(DBMessage(user_id=4001, dialogue_id=1, topic_id=4, role="user", content="MAX вопрос в теме 4"))
        session.add(Content(key="start_message", text_content="Главный экран MAX", is_visible=True))
        await session.commit()

    mock_client = SimpleNamespace(
        send_message=AsyncMock(),
        answer_callback=AsyncMock(),
    )
    app = MaxBotApplication(client=mock_client)

    cb = IncomingCallback(
        raw={},
        callback_id="cb_max_main",
        payload="ai_btn:svc:topic:main",
        chat_id=4001,
        message_id="m4001",
        sender=Sender(user_id=4001, username="max_user", first_name="MaxUser", last_name=None),
    )

    await app.handle_callback(cb)
    # Await spawned user tasks
    user_task = app.user_tasks.get(4001)
    if user_task:
        await user_task

    # 1. Assert ACK once
    assert mock_client.answer_callback.call_count == 1

    # 2. Assert DB state: topic is None, dialogue incremented
    async with db_session() as session:
        user = await session.get(User, 4001)
        assert user.current_topic_id is None
        assert user.current_dialogue_id == 2

    # 3. Next ordinary MAX message
    msg = IncomingMessage(
        raw={},
        message_id="m4002",
        chat_id=4001,
        text="Следующий вопрос в основном диалоге MAX",
        sender=Sender(user_id=4001, username="max_user", first_name="MaxUser", last_name=None),
    )

    with patch("max_messenger_bot.services.common.get_ai_response", new_callable=AsyncMock) as mock_ai:
        mock_ai.return_value = "Ответ MAX в основном диалоге"
        await app.handle_message(msg)
        user_task = app.user_tasks.get(4001)
        if user_task:
            await user_task

    async with db_session() as session:
        messages = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == 4001, DBMessage.dialogue_id == 2)
        )).scalars().all()
        assert len(messages) >= 2
        user_msg_db = next(m for m in messages if m.role == "user")
        assert user_msg_db.topic_id is None
        assert user_msg_db.dialogue_id == 2


# ==============================================================================
# 6. MAX — ALIASES
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("alias_payload", ["ai_btn:svc:topic:0", "ai_btn:svc:reset_topic"])
async def test_max_aliases_topic_to_main(db_session, alias_payload):
    uid = 4002 if alias_payload.endswith("0") else 4003
    async with db_session() as session:
        session.add(AIConfig(id=1, memory_mode=MEMORY_MODE_RESET))
        session.add(Topic(id=6, name="Topic Max C", is_active=True, admin_only=False))
        session.add(User(id=uid, first_name=f"MaxUser_{uid}", name=f"MaxUser_{uid}", accepted_disclaimer=True, current_dialogue_id=1, current_topic_id=6, is_admin=True))
        session.add(Content(key="start_message", text_content="Главный экран", is_visible=True))
        await session.commit()

    mock_client = SimpleNamespace(
        send_message=AsyncMock(),
        answer_callback=AsyncMock(),
    )
    app = MaxBotApplication(client=mock_client)

    cb = IncomingCallback(
        raw={},
        callback_id=f"cb_{uid}",
        payload=alias_payload,
        chat_id=uid,
        message_id=f"m_{uid}",
        sender=Sender(user_id=uid, username=f"max_user_{uid}", first_name="MaxUser", last_name=None),
    )

    await app.handle_callback(cb)
    user_task = app.user_tasks.get(uid)
    if user_task:
        await user_task

    assert mock_client.answer_callback.call_count == 1
    async with db_session() as session:
        user = await session.get(User, uid)
        assert user.current_topic_id is None
        assert user.current_dialogue_id == 2


# ==============================================================================
# 7. MAX — ALREADY MAIN IDEMPOTENCY
# ==============================================================================

@pytest.mark.asyncio
async def test_max_already_main_idempotency(db_session):
    async with db_session() as session:
        session.add(AIConfig(id=1, memory_mode=MEMORY_MODE_RESET))
        session.add(User(id=4004, first_name="MaxMainUser", name="MaxMainUser", accepted_disclaimer=True, current_dialogue_id=4, current_topic_id=None, is_admin=True))
        session.add(Content(key="start_message", text_content="Главный экран", is_visible=True))
        await session.commit()

    mock_client = SimpleNamespace(
        send_message=AsyncMock(),
        answer_callback=AsyncMock(),
    )
    app = MaxBotApplication(client=mock_client)

    cb = IncomingCallback(
        raw={},
        callback_id="cb_4004",
        payload="ai_btn:svc:topic:main",
        chat_id=4004,
        message_id="m_4004",
        sender=Sender(user_id=4004, username="max_main", first_name="MaxMainUser", last_name=None),
    )

    await app.handle_callback(cb)
    user_task = app.user_tasks.get(4004)
    if user_task:
        await user_task

    assert mock_client.answer_callback.call_count == 1
    async with db_session() as session:
        user = await session.get(User, 4004)
        assert user.current_topic_id is None
        assert user.current_dialogue_id == 4  # Unchanged!


# ==============================================================================
# 8. REGRESSION: POSITIVE TOPIC IDS
# ==============================================================================

@pytest.mark.asyncio
async def test_positive_topic_ids_regression(db_session):
    async with db_session() as session:
        session.add(AIConfig(id=1, memory_mode=MEMORY_MODE_RESET))
        session.add(Topic(id=15, name="Real Topic 15", is_active=True, admin_only=False))
        session.add(User(id=5001, first_name="RegUser", name="RegUser", accepted_disclaimer=True, current_dialogue_id=1, current_topic_id=None, is_admin=True))
        await session.commit()

    mock_client = SimpleNamespace(
        send_message=AsyncMock(),
        answer_callback=AsyncMock(),
    )
    app = MaxBotApplication(client=mock_client)

    cb = IncomingCallback(
        raw={},
        callback_id="cb_pos_topic",
        payload="ai_btn:svc:topic:15",
        chat_id=5001,
        message_id="m_5001",
        sender=Sender(user_id=5001, username="reg_user", first_name="RegUser", last_name=None),
    )

    await app.handle_callback(cb)
    user_task = app.user_tasks.get(5001)
    if user_task:
        await user_task

    assert mock_client.answer_callback.call_count == 1
    async with db_session() as session:
        user = await session.get(User, 5001)
        assert user.current_topic_id == 15


# ==============================================================================
# 9. STARTUP SYMBOL REGRESSION
# ==============================================================================

def test_max_startup_symbols_regression():
    import max_messenger_bot.app as app
    assert callable(app.get_settings)
    assert callable(app.validate_webhook_runtime_settings)
