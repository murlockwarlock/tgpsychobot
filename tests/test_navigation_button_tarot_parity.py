from __future__ import annotations

import asyncio
import html
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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
from database import (
    AIConfig,
    Base,
    BotGeneralConfig,
    Content,
    MediaLibrary,
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
from max_messenger_bot.app import MaxBotApplication, resolve_max_ai_button_label
from max_messenger_bot.keyboards import callback_button, inline_keyboard, response_buttons_keyboard
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
from memory_mode import (
    MEMORY_MODE_GLOBAL,
    MEMORY_MODE_RESET,
    MEMORY_MODE_TOPIC,
    apply_memory_mode_topic_switch,
)
from response_buttons import (
    MAIN_TOPIC_ACTIONS,
    ResponseButton,
    _sanitize_ai_button_fragment,
    build_action_callback_data,
    build_ai_button_system_message,
    split_action_callback_data,
)
from result_history import (
    TOPIC_WELCOME_ROLE,
    is_topic_welcome_shown,
    record_topic_welcome_shown,
    resolve_topic_entry_state,
)
from system_events import (
    build_main_dialogue_resume_system_message,
    build_topic_auto_start_system_message,
    build_topic_resume_system_message,
)


@pytest_asyncio.fixture
async def db_session(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test-parity.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(StorageBase.metadata.create_all)

    monkeypatch.setattr(database, "async_session_maker", sessions)
    monkeypatch.setattr(handlers, "async_session_maker", sessions)
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


def _make_incoming_callback(chat_id: int, user_id: int, payload: str, message: dict | None = None, callback_id: str = "cb_123") -> IncomingCallback:
    return IncomingCallback(
        raw={},
        callback_id=callback_id,
        payload=payload,
        chat_id=chat_id,
        message_id="msg_1",
        sender=Sender(user_id=user_id, username="testuser", first_name="Test", last_name=None),
        message=message,
    )


# ==============================================================================
# 1. Shared Button Semantic Formatter Contract
# ==============================================================================

def test_shared_button_formatter_exact_contracts():
    # Canonical example from prompt
    res = build_ai_button_system_message("💔 Отношения", "pain_relations")
    assert res == '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку "💔 Отношения" (pain_relations)]'

    # Special characters: quotes, brackets, parens, backslashes, whitespace
    res_special = build_ai_button_system_message('Кнопка "1" [тест] (инфо) \\ путь', 'action "a" [b] (c) \\ d')
    assert res_special == '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку "Кнопка \\"1\\" \\[тест\\] \\(инфо\\) \\\\ путь" (action \\"a\\" \\[b\\] \\(c\\) \\\\ d)]'

    # Newlines and tabs normalized to spaces without stripping edges
    res_newlines = build_ai_button_system_message(" текст с\nпереносом\tи пробелами ", "action\r\nsub")
    assert res_newlines == '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку " текст с переносом и пробелами " (action sub)]'

    # None button text
    res_none = build_ai_button_system_message(None, "action")
    assert res_none == '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку "" (action)]'

    # Verify handlers import matches response_buttons builder
    assert handlers.build_ai_button_system_message is build_ai_button_system_message


# ==============================================================================
# 2. MAX Collision-Safe Identity & Label Recovery
# ==============================================================================

def test_max_keyboard_collision_safe_identity_and_label_recovery():
    # 1. Unique action produces unindexed ai_btn:{action}
    rows_unique = [[
        ResponseButton(text="Ссылка", kind="url", value="https://example.com"),
        ResponseButton(text="💔 Отношения", kind="action", value="pain_relations"),
    ]]
    kb_unique = response_buttons_keyboard(rows_unique)
    assert kb_unique[0]["payload"]["buttons"][0][1]["payload"] == "ai_btn:pain_relations"

    # 2. Duplicate actions use global ACTION-button indexes
    mixed_rows = [
        [
            ResponseButton(text="Продолжить 1", kind="action", value="continue"),
            ResponseButton(text="Другое", kind="action", value="other"),
        ],
        [
            ResponseButton(text="Продолжить 2", kind="action", value="continue"),
        ],
    ]
    kb_indexed = response_buttons_keyboard(mixed_rows)
    btn_0 = kb_indexed[0]["payload"]["buttons"][0][0]
    btn_1 = kb_indexed[0]["payload"]["buttons"][0][1]
    btn_2 = kb_indexed[0]["payload"]["buttons"][1][0]
    assert btn_0["payload"] == "ai_btn:continue|00"
    assert btn_1["payload"] == "ai_btn:other"  # Unique action remains unindexed!
    assert btn_2["payload"] == "ai_btn:continue|02"

    # 3. compose_max_keyboard parity
    composed_kb = max_common.compose_max_keyboard(mixed_rows)
    assert composed_kb[0]["payload"]["buttons"][0][0]["payload"] == "ai_btn:continue|00"
    assert composed_kb[0]["payload"]["buttons"][0][1]["payload"] == "ai_btn:other"
    assert composed_kb[0]["payload"]["buttons"][1][0]["payload"] == "ai_btn:continue|02"

    # 4. Label recovery from IncomingCallback.message.body.attachments
    cb_msg = {
        "body": {
            "attachments": kb_indexed,
        }
    }
    cb_0 = _make_incoming_callback(101, 101, "ai_btn:continue|00", message=cb_msg)
    cb_2 = _make_incoming_callback(101, 101, "ai_btn:continue|02", message=cb_msg)
    assert resolve_max_ai_button_label(cb_0) == "Продолжить 1"
    assert resolve_max_ai_button_label(cb_2) == "Продолжить 2"

    # 5. Ambiguous legacy unindexed duplicate returns None (fallback to action)
    ambiguous_kb = [
        {"type": "inline_keyboard", "payload": {"buttons": [
            [{"type": "callback", "text": "Вариант А", "payload": "ai_btn:duplicate"}],
            [{"type": "callback", "text": "Вариант Б", "payload": "ai_btn:duplicate"}],
        ]}}
    ]
    cb_ambiguous = _make_incoming_callback(101, 101, "ai_btn:duplicate", message={"body": {"attachments": ambiguous_kb}})
    assert resolve_max_ai_button_label(cb_ambiguous) is None


# ==============================================================================
# 3. MAX Legacy Action Normalization & Access Gate
# ==============================================================================

@pytest.mark.asyncio
async def test_max_legacy_action_normalization_and_access_gate(db_session, monkeypatch):
    async with db_session() as session:
        # User without active subscription
        user = User(id=777, first_name="MaxUser", name="MaxUser", gender="male", age="25", is_admin=False)
        session.add(user)
        session.add(SubscriptionConfig(id=1, subscriptions_enabled=True))
        session.add(AIConfig(id=1, provider="OpenAI", openai_api_key="sk-test"))
        await session.commit()

    client = AsyncMock()
    app = MaxBotApplication(client)

    # 1. Denied access on generic action:
    cb_denied = _make_incoming_callback(777, 777, "ai_btn:pain_relations")
    spawn_mock = MagicMock()
    monkeypatch.setattr(app, "spawn_user_task", spawn_mock)

    await app.handle_callback(cb_denied)

    # ACK called once
    assert client.answer_callback.await_count == 1
    # Access check rejected, no task spawned, no dialogue persisted
    assert spawn_mock.call_count == 0
    async with db_session() as session:
        msgs = (await session.execute(select(DBMessage).where(DBMessage.user_id == 777))).scalars().all()
        assert len(msgs) == 0

    # 2. Granted access with legacy action:pain_relations:
    client.answer_callback.reset_mock()
    monkeypatch.setattr(max_common, "ensure_access_before_chat", AsyncMock(return_value=True))

    captured_prompt = None
    async def fake_run_ai(client_arg, chat_id, user_id, prompt, states):
        nonlocal captured_prompt
        captured_prompt = prompt

    monkeypatch.setattr(max_common, "run_ai_dialogue", fake_run_ai)
    spawned_tasks = []
    def fake_spawn(uid, coro):
        task = asyncio.create_task(coro)
        spawned_tasks.append(task)
        return task
    monkeypatch.setattr(app, "spawn_user_task", fake_spawn)

    # Attach keyboard with visible label
    cb_msg = {
        "body": {
            "attachments": [
                {"type": "inline_keyboard", "payload": {"buttons": [[{"type": "callback", "text": "💔 Отношения", "payload": "action:pain_relations"}]]}}
            ]
        }
    }
    cb_legacy = _make_incoming_callback(777, 777, "action:pain_relations", message=cb_msg)
    await app.handle_callback(cb_legacy)
    if spawned_tasks:
        await asyncio.gather(*spawned_tasks)

    assert client.answer_callback.await_count == 1
    assert captured_prompt == '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку "💔 Отношения" (pain_relations)]'
    # Must NOT contain '(action:pain_relations)'
    assert "(action:pain_relations)" not in captured_prompt


# ==============================================================================
# 4. Telegram Single-Owner Claim & Service-vs-Generic Routing
# ==============================================================================

@pytest.mark.asyncio
async def test_telegram_service_vs_generic_claim_flow(db_session, monkeypatch):
    bot = SimpleNamespace(send_message=AsyncMock())
    state = AsyncMock()

    # Setup user
    async with db_session() as session:
        session.add(User(id=505, first_name="TG User", name="TG User", gender="male", age="25", is_admin=True))
        session.add(AIConfig(id=1, provider="OpenAI", openai_api_key="sk-test"))
        session.add(Topic(id=1, name="General Topic", is_active=True))
        await session.commit()

    handlers.user_message_buffers.clear()
    handlers._ai_button_claims.clear()

    # 1. Service action: svc:menu
    cb_menu = SimpleNamespace(
        data="ai_btn:svc:menu",
        from_user=SimpleNamespace(id=505),
        message=SimpleNamespace(
            message_id=10,
            chat=SimpleNamespace(id=505),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="В меню", callback_data="ai_btn:svc:menu")]]
            ),
        ),
        answer=AsyncMock(),
    )
    await handlers.process_response_button(cb_menu, state, bot)

    # ACK once
    assert cb_menu.answer.await_count == 1
    # Keyboard disabled
    assert cb_menu.message.edit_reply_markup.await_count == 1
    # Produces NO "Ответ принят" echo
    for call in cb_menu.message.answer.call_args_list:
        assert "Ответ принят" not in str(call)
    # Produces NO generic buffer
    assert 505 not in handlers.user_message_buffers

    # 2. Unknown service action: svc:unsupported
    cb_unknown = SimpleNamespace(
        data="ai_btn:svc:unsupported",
        from_user=SimpleNamespace(id=505),
        message=SimpleNamespace(
            message_id=11,
            chat=SimpleNamespace(id=505),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="Неизвестно", callback_data="ai_btn:svc:unsupported")]]
            ),
        ),
        answer=AsyncMock(),
    )
    with patch("logging.warning") as log_warn:
        await handlers.process_response_button(cb_unknown, state, bot)
        assert cb_unknown.answer.await_count == 1
        assert 505 not in handlers.user_message_buffers
        assert log_warn.called

    # 3. Generic action: ai_btn:pain_relations
    cb_generic = SimpleNamespace(
        data="ai_btn:pain_relations",
        from_user=SimpleNamespace(id=505),
        message=SimpleNamespace(
            message_id=12,
            chat=SimpleNamespace(id=505),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="💔 Отношения", callback_data="ai_btn:pain_relations")]]
            ),
        ),
        answer=AsyncMock(),
    )
    mock_process_buf = AsyncMock()
    monkeypatch.setattr(handlers, "process_buffered_messages", mock_process_buf)

    await handlers.process_response_button(cb_generic, state, bot)

    # Generic path: ACK once, keyboard disabled, echo label called, buffered and processed
    assert cb_generic.answer.await_count == 1
    assert cb_generic.message.edit_reply_markup.await_count == 1
    assert any("Ответ принят" in str(call) for call in cb_generic.message.answer.call_args_list)
    assert mock_process_buf.await_count == 1
    assert handlers.user_message_buffers[505] == ['[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку "💔 Отношения" (pain_relations)]']

    # 4. Duplicate click on same generic button is rejected
    cb_generic.answer.reset_mock()
    mock_process_buf.reset_mock()
    await handlers.process_response_button(cb_generic, state, bot)
    assert cb_generic.answer.await_count == 1
    assert mock_process_buf.await_count == 0


# ==============================================================================
# 5. Complete 8-Row Primary Topic Contract Matrix
# ==============================================================================

@pytest.mark.asyncio
async def test_complete_8_row_primary_topic_matrix(db_session, monkeypatch):
    """
    Verifies all 8 rows of the complete primary topic matrix:
    TG: select_topic_<id>, reset_topic
    MAX: select_topic, reset_topic
    """
    bot = SimpleNamespace(send_message=AsyncMock())
    state = AsyncMock()
    client = AsyncMock()
    max_app_instance = MaxBotApplication(client)

    # Setup topics: topic 10 (normal), topic 20 (special HTML name)
    async with db_session() as session:
        session.add(AIConfig(id=1, provider="OpenAI", openai_api_key="sk-test", memory_mode=MEMORY_MODE_TOPIC))
        session.add(BotGeneralConfig(id=1))
        session.add(Topic(id=10, name="Карьера и цели", is_active=True, auto_start_dialogue=True))
        session.add(Topic(id=20, name='Тема <Любовь & "Счастье">', is_active=True, auto_start_dialogue=True))
        # TG User 100 with completed profile onboarding & admin
        session.add(User(id=100, first_name="TG User", name="TG User", gender="male", age="25", is_admin=True, accepted_disclaimer=True, current_topic_id=None, current_dialogue_id=1))
        # MAX User 200 with completed profile onboarding & admin
        session.add(User(id=200, first_name="MAX User", name="MAX User", gender="male", age="25", is_admin=True, accepted_disclaimer=True, current_topic_id=None, current_dialogue_id=1))
        await session.commit()

    monkeypatch.setattr(handlers, "_start_telegram_hidden_kickoff", AsyncMock())
    monkeypatch.setattr(max_common, "run_hidden_ai_kickoff", AsyncMock())
    monkeypatch.setattr(max_common, "ensure_access_before_chat", AsyncMock(return_value=True))

    # --------------------------------------------------------------------------
    # ROW 1: TG First Entry
    # --------------------------------------------------------------------------
    cb_tg_first = SimpleNamespace(
        data="select_topic_10",
        from_user=SimpleNamespace(id=100),
        message=SimpleNamespace(
            message_id=1001,
            chat=SimpleNamespace(id=100),
            answer=AsyncMock(),
            delete=AsyncMock(),
        ),
        answer=AsyncMock(),
    )
    await handlers.process_topic_selection(cb_tg_first, state, bot)
    assert cb_tg_first.answer.await_count == 1
    async with db_session() as session:
        user_100 = await session.get(User, 100)
        assert user_100.current_topic_id == 10
        # Navigation system event delta = 1
        nav_msgs = (await session.execute(select(DBMessage).where(DBMessage.user_id == 100, DBMessage.topic_id == 10, DBMessage.role == "system_event"))).scalars().all()
        assert len(nav_msgs) == 1
        assert nav_msgs[0].content == build_topic_auto_start_system_message("Карьера и цели")
        # Topic welcome recorded
        assert await is_topic_welcome_shown(session, 100, user_100.current_dialogue_id, 10)

    # --------------------------------------------------------------------------
    # ROW 2: TG Direct Genuine Resume
    # --------------------------------------------------------------------------
    # Switch TG user to None, then back to 10 where welcome is already shown
    async with db_session() as session:
        user_100 = await session.get(User, 100)
        user_100.current_topic_id = None
        await session.commit()

    bot.send_message.reset_mock()
    handlers._start_telegram_hidden_kickoff.reset_mock()
    cb_tg_resume = SimpleNamespace(
        data="select_topic_10",
        from_user=SimpleNamespace(id=100),
        message=SimpleNamespace(
            message_id=1002,
            chat=SimpleNamespace(id=100),
            answer=AsyncMock(),
            delete=AsyncMock(),
        ),
        answer=AsyncMock(),
    )
    await handlers.process_topic_selection(cb_tg_resume, state, bot)
    assert cb_tg_resume.answer.await_count == 1
    # Visible resume notice sent with escaped topic name
    assert any("✅ Продолжаем тему: «Карьера и цели»." in str(call) for call in bot.send_message.call_args_list)
    # Hidden kickoff started
    assert handlers._start_telegram_hidden_kickoff.await_count == 1

    # --------------------------------------------------------------------------
    # ROW 3: TG Already Current
    # --------------------------------------------------------------------------
    bot.send_message.reset_mock()
    cb_tg_current = SimpleNamespace(
        data="select_topic_10",
        from_user=SimpleNamespace(id=100),
        message=SimpleNamespace(
            message_id=1003,
            chat=SimpleNamespace(id=100),
            answer=AsyncMock(),
            delete=AsyncMock(),
        ),
        answer=AsyncMock(),
    )
    await handlers.process_topic_selection(cb_tg_current, state, bot)
    assert cb_tg_current.answer.await_count == 1
    # Shows already current guidance
    assert any("Вы уже находитесь в теме «Карьера и цели»." in str(call) for call in cb_tg_current.message.answer.call_args_list)

    # --------------------------------------------------------------------------
    # ROW 4: TG Return to Main
    # --------------------------------------------------------------------------
    bot.send_message.reset_mock()
    cb_tg_reset = SimpleNamespace(
        data="reset_topic",
        from_user=SimpleNamespace(id=100),
        message=SimpleNamespace(
            message_id=1004,
            chat=SimpleNamespace(id=100),
            answer=AsyncMock(),
            delete=AsyncMock(),
        ),
        answer=AsyncMock(),
    )
    await handlers.process_topic_reset(cb_tg_reset, bot, state)
    assert cb_tg_reset.answer.await_count == 1
    async with db_session() as session:
        user_100 = await session.get(User, 100)
        assert user_100.current_topic_id is None
        # Verify hidden main return event
        main_nav = (await session.execute(select(DBMessage).where(DBMessage.user_id == 100, DBMessage.topic_id.is_(None)))).scalars().all()
        assert any(m.content == build_main_dialogue_resume_system_message() for m in main_nav)
    # Visible return message
    assert any("✅ Мы вернулись в основной диалог." in str(call) for call in bot.send_message.call_args_list)

    # --------------------------------------------------------------------------
    # ROW 5: MAX First Entry
    # --------------------------------------------------------------------------
    client.send_message.reset_mock()
    cb_max_first = _make_incoming_callback(200, 200, "topic_10")
    await max_topics.select_topic(client, 200, 200, 10, max_app_instance.states)
    async with db_session() as session:
        user_200 = await session.get(User, 200)
        assert user_200.current_topic_id == 10
        assert await is_topic_welcome_shown(session, 200, user_200.current_dialogue_id, 10)

    # --------------------------------------------------------------------------
    # ROW 6: MAX Direct Genuine Resume
    # --------------------------------------------------------------------------
    async with db_session() as session:
        user_200 = await session.get(User, 200)
        user_200.current_topic_id = None
        await session.commit()

    client.send_message.reset_mock()
    max_common.run_hidden_ai_kickoff.reset_mock()
    await max_topics.select_topic(client, 200, 200, 10, max_app_instance.states)
    # Visible resume notice sent
    assert any("✅ Продолжаем тему: <b>Карьера и цели</b>." in str(call) for call in client.send_message.call_args_list)
    assert max_common.run_hidden_ai_kickoff.await_count == 1

    # --------------------------------------------------------------------------
    # ROW 7: MAX Already Current
    # --------------------------------------------------------------------------
    client.send_message.reset_mock()
    max_common.run_hidden_ai_kickoff.reset_mock()
    await max_topics.select_topic(client, 200, 200, 10, max_app_instance.states)
    # Informational guidance sent
    assert any("Вы уже находитесь в теме «Карьера и цели»." in str(call) for call in client.send_message.call_args_list)
    assert max_common.run_hidden_ai_kickoff.await_count == 0

    # --------------------------------------------------------------------------
    # ROW 8: MAX Return to Main
    # --------------------------------------------------------------------------
    client.send_message.reset_mock()
    await max_topics.reset_topic(client, 200, 200, max_app_instance.states)
    assert any("✅ Мы вернулись в основной диалог." in str(call) for call in client.send_message.call_args_list)
    async with db_session() as session:
        user_200 = await session.get(User, 200)
        assert user_200.current_topic_id is None
        max_main_nav = (await session.execute(select(DBMessage).where(DBMessage.user_id == 200, DBMessage.topic_id.is_(None)))).scalars().all()
        assert any(m.content == build_main_dialogue_resume_system_message() for m in max_main_nav)


# ==============================================================================
# 6. Dynamic HTML-Sensitive Topic Name Contract
# ==============================================================================

@pytest.mark.asyncio
async def test_dynamic_html_escaped_topic_names(db_session, monkeypatch):
    bot = SimpleNamespace(send_message=AsyncMock())
    client = AsyncMock()
    state = AsyncMock()

    raw_topic_name = 'Тема <Любовь & "Счастье">'
    expected_escaped = 'Тема &lt;Любовь &amp; &quot;Счастье&quot;&gt;'

    async with db_session() as session:
        session.add(AIConfig(id=1, provider="OpenAI", openai_api_key="sk-test", memory_mode=MEMORY_MODE_TOPIC))
        session.add(Topic(id=30, name=raw_topic_name, is_active=True))
        session.add(User(id=301, first_name="TG User", name="TG User", gender="male", age="25", is_admin=True, accepted_disclaimer=True, current_topic_id=None, current_dialogue_id=1))
        session.add(User(id=302, first_name="MAX User", name="MAX User", gender="male", age="25", is_admin=True, accepted_disclaimer=True, current_topic_id=None, current_dialogue_id=1))
        await session.commit()
        # Pre-mark welcome shown and topic state so it resumes directly
        session.add(UserTopicState(user_id=301, topic_id=30, dialogue_id=1))
        session.add(UserTopicState(user_id=302, topic_id=30, dialogue_id=1))
        await record_topic_welcome_shown(session, 301, 1, 30)
        await record_topic_welcome_shown(session, 302, 1, 30)
        await session.commit()

    monkeypatch.setattr(handlers, "_start_telegram_hidden_kickoff", AsyncMock())
    monkeypatch.setattr(max_common, "run_hidden_ai_kickoff", AsyncMock())
    monkeypatch.setattr(max_common, "ensure_access_before_chat", AsyncMock(return_value=True))

    # 1. Telegram direct resume visible message
    cb_tg = SimpleNamespace(
        data="select_topic_30",
        from_user=SimpleNamespace(id=301),
        message=SimpleNamespace(message_id=1, chat=SimpleNamespace(id=301), answer=AsyncMock(), delete=AsyncMock()),
        answer=AsyncMock(),
    )
    await handlers.process_topic_selection(cb_tg, state, bot)
    tg_sent = [call[0][1] for call in bot.send_message.call_args_list if len(call[0]) > 1]
    assert f"✅ Продолжаем тему: «{expected_escaped}»." in tg_sent

    # 2. MAX direct resume visible message
    await max_topics.select_topic(client, 302, 302, 30)
    max_sent = [call.kwargs.get("text") for call in client.send_message.call_args_list]
    assert f"✅ Продолжаем тему: <b>{expected_escaped}</b>." in max_sent

    # 3. Canonical system event builder receives RAW topic name
    async with db_session() as session:
        events = (await session.execute(select(DBMessage).where(DBMessage.topic_id == 30, DBMessage.role == "system_event"))).scalars().all()
        assert len(events) >= 1
        for ev in events:
            assert ev.content == build_topic_resume_system_message(raw_topic_name)
            assert expected_escaped not in ev.content


# ==============================================================================
# 7. Delayed Resume Ordering Contract
# ==============================================================================

@pytest.mark.asyncio
async def test_delayed_resume_ordering(db_session, monkeypatch):
    bot = SimpleNamespace(send_message=AsyncMock())
    client = AsyncMock()
    state = AsyncMock()

    events_order = []

    async def fake_bot_send(*args, **kwargs):
        events_order.append("tg_visible_send")

    bot.send_message = fake_bot_send

    async def fake_tg_kickoff(*args, **kwargs):
        events_order.append("tg_hidden_kickoff")

    monkeypatch.setattr(handlers, "_start_telegram_hidden_kickoff", fake_tg_kickoff)
    monkeypatch.setattr(handlers, "_check_telegram_chat_access", AsyncMock(return_value=True))

    async with db_session() as session:
        session.add(Topic(id=40, name="Тема Отложенная", is_active=True))
        session.add(User(id=401, first_name="TG User", name="TG User", gender="male", age="25", is_admin=True, accepted_disclaimer=True, current_topic_id=40, current_dialogue_id=1))
        session.add(User(id=402, first_name="MAX User", name="MAX User", gender="male", age="25", is_admin=True, accepted_disclaimer=True, current_topic_id=40, current_dialogue_id=1))
        await session.commit()

    # TG delayed resume through _send_pending_topic_intro
    pending_data = {
        "topic_intro_after_onboarding": 40,
        "topic_intro_dialogue_id": 1,
        "topic_intro_welcome_needed": False,  # i.e. resume
    }
    await handlers._send_pending_topic_intro(pending_data, bot, 401, state)

    assert events_order == ["tg_visible_send", "tg_hidden_kickoff"]

    # MAX delayed resume through resume_pending_ai_turn
    max_events_order = []
    async def fake_client_send(*args, **kwargs):
        max_events_order.append("max_visible_send")
    client.send_message = fake_client_send

    async def fake_max_kickoff(*args, **kwargs):
        max_events_order.append("max_hidden_kickoff")
    monkeypatch.setattr(max_common, "run_hidden_ai_kickoff", fake_max_kickoff)
    monkeypatch.setattr(max_common, "ensure_access_before_chat", AsyncMock(return_value=True))

    max_state_data = {
        "pending_auto_start_topic_id": 40,
        "pending_auto_start_dialogue_id": 1,
        "pending_auto_start_kind": "resume",
    }
    await max_common.resume_pending_ai_turn(client, 402, 402, max_state_data)

    assert max_events_order == ["max_visible_send", "max_hidden_kickoff"]


# ==============================================================================
# 8. Tarot Deduplication Proof Across All 3 Production Paths
# ==============================================================================

@pytest.mark.asyncio
async def test_tarot_deduplication_production_request_construction(db_session, monkeypatch):
    """
    Verifies that the current card synthetic message appears EXACTLY ONCE in the
    final serialized OpenAI request messages across all 3 production paths:
    - process_card_selection (Round 2 and Round 3)
    - process_buffered_messages (Random card)
    - process_user_prompt (Random card)
    """
    bot = SimpleNamespace(
        send_message=AsyncMock(),
        send_photo=AsyncMock(),
        send_audio=AsyncMock(),
        send_chat_action=AsyncMock(),
        send_media_group=AsyncMock(),
    )
    state = AsyncMock()

    # Setup database with AIConfig and MediaLibrary cards
    async with db_session() as session:
        ai_config = AIConfig(
            id=1,
            provider="OpenAI",
            openai_api_key="sk-test",
            openai_model="gpt-5.6-terra",
            allow_fallback=False,
            fallback_provider=None,
            fallback_model=None,
            use_proxy=False,
            context_limit_first=10,
            context_limit_recent=20,
        )
        session.add(ai_config)
        session.add(BotGeneralConfig(id=1))
        session.add(User(id=801, first_name="Tarot User", name="Tarot User", gender="male", age="25", is_admin=True, accepted_disclaimer=True, current_topic_id=None, current_dialogue_id=1))
        session.add(User(id=802, first_name="Tarot User 2", name="Tarot User 2", gender="male", age="25", is_admin=True, accepted_disclaimer=True, current_topic_id=None, current_dialogue_id=1))
        session.add(User(id=803, first_name="Tarot User 3", name="Tarot User 3", gender="male", age="25", is_admin=True, accepted_disclaimer=True, current_topic_id=None, current_dialogue_id=1))
        # Add test cards
        session.add(MediaLibrary(id=1, file_name="card_1.jpg", file_id="f_1", category="tarot", media_type="photo", description="Карта 1 Маг"))
        session.add(MediaLibrary(id=2, file_name="card_2.jpg", file_id="f_2", category="tarot", media_type="photo", description="Карта 2 Жрица"))
        session.add(MediaLibrary(id=3, file_name="card_3.jpg", file_id="f_3", category="tarot", media_type="photo", description="Карта 3 Императрица"))
        session.add(MediaLibrary(id=4, file_name="_back", file_id="f_back", category="tarot", media_type="photo", description="Рубашка"))
        await session.commit()

    captured_openai_payloads = []

    # Mock AsyncOpenAI external transport
    class FakeChatCompletions:
        async def create(self, **kwargs):
            captured_openai_payloads.append(kwargs)
            choice = SimpleNamespace(message=SimpleNamespace(content="Интерпретация: Сила в твоих руках.", role="assistant"))
            return SimpleNamespace(choices=[choice], usage=SimpleNamespace(prompt_tokens=15, completion_tokens=15, total_tokens=30))

    class FakeAsyncOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(completions=FakeChatCompletions())

    class DummyMediaScope:
        def predicate(self, *args, **kwargs):
            return True

    monkeypatch.setattr(ai_integration, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(handlers, "load_media_scope", AsyncMock(return_value=DummyMediaScope()))
    monkeypatch.setattr(handlers, "send_card_album", AsyncMock())
    monkeypatch.setattr(handlers, "send_photo_or_document", AsyncMock())
    monkeypatch.setattr(handlers, "_send_generated_response", AsyncMock())

    # --------------------------------------------------------------------------
    # Path 1: process_card_selection (Round 2 and Round 3)
    # --------------------------------------------------------------------------
    # Initialize spread state: 3 rounds total, round 1 already completed
    await handlers._save_card_spread_state(801, {
        "category": "tarot",
        "topic_id": None,
        "rounds_left": 2,
        "total_rounds": 3,
        "cards_per_round": 1,
        "hidden": False,
        "chosen_card_ids": [1],
        "selected_file_ids": ["f_1"],
        "pending_card_ids": [2],
    })

    # Execute Round 2
    captured_openai_payloads.clear()
    cb_card_r2 = SimpleNamespace(
        data="card_select_2",
        from_user=SimpleNamespace(id=801),
        message=SimpleNamespace(message_id=2001, chat=SimpleNamespace(id=801), answer_photo=AsyncMock(), answer=AsyncMock(), delete=AsyncMock()),
        answer=AsyncMock(),
    )
    await handlers.process_card_selection(cb_card_r2, bot)

    assert len(captured_openai_payloads) == 1
    r2_messages = captured_openai_payloads[0]["messages"]
    r2_user_messages = [m for m in r2_messages if m["role"] == "user"]
    # Verify current round synthetic appears EXACTLY ONCE
    current_card_content = [m["content"] for m in r2_user_messages if "card_2.jpg" in str(m["content"])]
    assert len(current_card_content) == 1

    # Execute Round 3
    await handlers._save_card_spread_state(801, {
        "category": "tarot",
        "topic_id": None,
        "rounds_left": 1,
        "total_rounds": 3,
        "cards_per_round": 1,
        "hidden": False,
        "chosen_card_ids": [1, 2],
        "selected_file_ids": ["f_1", "f_2"],
        "pending_card_ids": [3],
    })
    captured_openai_payloads.clear()
    cb_card_r3 = SimpleNamespace(
        data="card_select_3",
        from_user=SimpleNamespace(id=801),
        message=SimpleNamespace(message_id=2002, chat=SimpleNamespace(id=801), answer_photo=AsyncMock(), answer=AsyncMock(), delete=AsyncMock()),
        answer=AsyncMock(),
    )
    await handlers.process_card_selection(cb_card_r3, bot)

    assert len(captured_openai_payloads) == 1
    r3_messages = captured_openai_payloads[0]["messages"]
    r3_user_messages = [m for m in r3_messages if m["role"] == "user"]
    # Current card 3 appears EXACTLY ONCE
    current_r3_content = [m["content"] for m in r3_user_messages if "card_3.jpg" in str(m["content"])]
    assert len(current_r3_content) == 1
    # Previous card 2 retained in history
    prev_r2_content = [m["content"] for m in r3_user_messages if "card_2.jpg" in str(m["content"])]
    assert len(prev_r2_content) == 1

    # --------------------------------------------------------------------------
    # Path 2: process_buffered_messages (Random Card)
    # --------------------------------------------------------------------------
    captured_openai_payloads.clear()
    handlers.user_message_buffers[802] = ["Вытяни мне случайную карту"]

    # First provider call returns a directive with RANDOM_IMG
    first_call_done = False
    class FakeChatCompletionsRandom:
        async def create(self, **kwargs):
            nonlocal first_call_done
            captured_openai_payloads.append(kwargs)
            if not first_call_done:
                first_call_done = True
                choice = SimpleNamespace(message=SimpleNamespace(content="Вот твоя карта [RANDOM_IMG:tarot,1]", role="assistant"))
            else:
                choice = SimpleNamespace(message=SimpleNamespace(content="Интерпретация случайной карты.", role="assistant"))
            return SimpleNamespace(choices=[choice], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20))

    class FakeAsyncOpenAIRandom:
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(completions=FakeChatCompletionsRandom())

    monkeypatch.setattr(ai_integration, "AsyncOpenAI", FakeAsyncOpenAIRandom)

    # Mock media lookup in process_buffered_messages
    async with db_session() as session:
        cards_res = (await session.execute(select(MediaLibrary).where(MediaLibrary.id == 1))).scalars().all()
    monkeypatch.setattr(handlers, "send_photo_or_document", AsyncMock())

    await handlers.process_buffered_messages(802, bot, state)

    # 2 provider calls: 1 originating dialogue + 1 card interpretation
    assert len(captured_openai_payloads) == 2
    # In the second call (interpretation):
    interp_messages = captured_openai_payloads[1]["messages"]
    interp_user_messages = [m for m in interp_messages if m["role"] == "user"]
    random_card_msgs = [m["content"] for m in interp_user_messages if "Случайно выпала карта" in str(m["content"])]
    assert len(random_card_msgs) == 1

    # --------------------------------------------------------------------------
    # Path 3: process_user_prompt (Random Card)
    # --------------------------------------------------------------------------
    captured_openai_payloads.clear()
    first_call_done = False

    msg_obj = SimpleNamespace(
        message_id=3001,
        chat=SimpleNamespace(id=803),
        from_user=SimpleNamespace(id=803),
        answer=AsyncMock(),
    )
    await handlers.process_user_prompt(msg_obj, 803, "Вытяни случайную карту снова", bot, state)

    assert len(captured_openai_payloads) == 2
    prompt_interp_messages = captured_openai_payloads[1]["messages"]
    prompt_interp_user_messages = [m for m in prompt_interp_messages if m["role"] == "user"]
    prompt_random_card_msgs = [m["content"] for m in prompt_interp_user_messages if "Случайно выпала карта" in str(m["content"])]
    assert len(prompt_random_card_msgs) == 1
