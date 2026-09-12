from __future__ import annotations

import asyncio
import html
import os
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import func, select
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
from max_messenger_bot.identity import MAX_ID_OFFSET
from max_messenger_bot.keyboards import callback_button, inline_keyboard, response_buttons_keyboard
from max_messenger_bot.models import IncomingCallback, IncomingMessage, Sender, parse_callback
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

RAW_USER_ID = 55
RAW_CHAT_ID = 555
INTERNAL_USER_ID = RAW_USER_ID + MAX_ID_OFFSET


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


def make_raw_max_callback_update(
    payload: str,
    *,
    user_id: int = RAW_USER_ID,
    chat_id: int = RAW_CHAT_ID,
    update_id: str = "upd_100",
    callback_id: str = "cb_100",
    inline_keyboard_buttons: list[list[dict]] | None = None,
    attachments: list[dict] | None = None,
    message_body_extra: dict | None = None,
    message_extra: dict | None = None,
) -> dict:
    if attachments is None:
        if inline_keyboard_buttons is not None:
            attachments = [{
                "type": "inline_keyboard",
                "payload": {
                    "buttons": inline_keyboard_buttons
                }
            }]
        else:
            attachments = []

    body = {
        "mid": "mid_100",
        "attachments": attachments,
    }
    if message_body_extra:
        body.update(message_body_extra)

    msg = {
        "recipient": {
            "chat_id": chat_id
        },
        "body": body,
    }
    if message_extra:
        msg.update(message_extra)

    return {
        "update_type": "message_callback",
        "update_id": update_id,
        "callback": {
            "callback_id": callback_id,
            "payload": payload,
            "user": {
                "user_id": user_id,
                "first_name": "MaxUser",
                "username": "maxuser",
            }
        },
        "message": msg,
    }


async def await_spawned_tasks(app: MaxBotApplication, user_id: int | None = None) -> None:
    if user_id and user_id in app.user_tasks:
        task = app.user_tasks[user_id]
        if not task.done():
            await task
    if app.background_tasks:
        pending = [t for t in app.background_tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending)


# ==============================================================================
# 1. Shared Button Semantic Formatter Contract
# ==============================================================================

def test_shared_button_formatter_exact_contracts():
    res = build_ai_button_system_message("💔 Отношения", "pain_relations")
    assert res == '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку "💔 Отношения" (pain_relations)]'

    res_special = build_ai_button_system_message('Кнопка "1" [тест] (инфо) \\ путь', 'action "a" [b] (c) \\ d')
    assert res_special == '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку "Кнопка \\"1\\" \\[тест\\] \\(инфо\\) \\\\ путь" (action \\"a\\" \\[b\\] \\(c\\) \\\\ d)]'

    res_newlines = build_ai_button_system_message(" текст с\nпереносом\tи пробелами ", "action\r\nsub")
    assert res_newlines == '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку " текст с переносом и пробелами " (action sub)]'

    res_none = build_ai_button_system_message(None, "action")
    assert res_none == '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку "" (action)]'

    assert handlers.build_ai_button_system_message is build_ai_button_system_message


# ==============================================================================
# 2. MAX Label Resolver & Non-Inline Attachment Regression
# ==============================================================================

def test_max_label_resolver_non_inline_attachment_regression():
    # 1. Non-inline attachment with fake buttons payload + real inline_keyboard attachment
    raw_cb_msg = {
        "body": {
            "attachments": [
                {
                    "type": "file",  # NON-inline attachment
                    "payload": {
                        "buttons": [[{"type": "callback", "text": "WRONG_TEXT", "payload": "ai_btn:pain_relations"}]]
                    }
                },
                {
                    "type": "inline_keyboard",  # Real inline keyboard
                    "payload": {
                        "buttons": [[{"type": "callback", "text": "💔 Отношения", "payload": "ai_btn:pain_relations"}]]
                    }
                }
            ]
        }
    }
    cb = IncomingCallback(
        raw={}, callback_id="cb_1", payload="ai_btn:pain_relations", chat_id=RAW_CHAT_ID,
        message_id="mid_1", sender=Sender(user_id=INTERNAL_USER_ID, username="testuser", first_name="Test", last_name=None), message=raw_cb_msg
    )
    # Resolver MUST return text from inline_keyboard only!
    assert resolve_max_ai_button_label(cb) == "💔 Отношения"

    # 2. Body exists but body.attachments is missing; top-level message.attachments exists
    raw_cb_legacy = {
        "body": {"mid": "mid_2"},
        "attachments": [
            {
                "type": "inline_keyboard",
                "payload": {
                    "buttons": [[{"type": "callback", "text": "Fallback Text", "payload": "ai_btn:fallback"}]]
                }
            }
        ]
    }
    cb_legacy = IncomingCallback(
        raw={}, callback_id="cb_2", payload="ai_btn:fallback", chat_id=RAW_CHAT_ID,
        message_id="mid_2", sender=Sender(user_id=INTERNAL_USER_ID, username="testuser", first_name="Test", last_name=None), message=raw_cb_legacy
    )
    assert resolve_max_ai_button_label(cb_legacy) == "Fallback Text"


# ==============================================================================
# 3. Canonical Raw MAX Callback Fixture & Parser Proof
# ==============================================================================

def test_canonical_raw_max_callback_fixture_and_parser():
    raw_update = make_raw_max_callback_update(
        "ai_btn:pain_relations",
        user_id=RAW_USER_ID,
        chat_id=RAW_CHAT_ID,
        inline_keyboard_buttons=[[{"type": "callback", "text": "💔 Отношения", "payload": "ai_btn:pain_relations"}]]
    )
    parsed = parse_callback(raw_update)
    assert parsed is not None
    # Raw webhook user ID is shifted by MAX_ID_OFFSET
    assert parsed.sender.user_id == INTERNAL_USER_ID
    # Chat ID stays unshifted
    assert parsed.chat_id == RAW_CHAT_ID
    # Message body and attachments preserved
    assert parsed.message == raw_update["message"]
    assert parsed.message["body"]["attachments"] == raw_update["message"]["body"]["attachments"]


# ==============================================================================
# 4. Raw MAX Generic Unique Button Execution via handle_update
# ==============================================================================

@pytest.mark.asyncio
async def test_raw_max_generic_unique_button(db_session, monkeypatch):
    async with db_session() as session:
        user = User(id=INTERNAL_USER_ID, first_name="MaxUser", name="MaxUser", gender="male", age="25", is_admin=True, accepted_disclaimer=True)
        session.add(user)
        session.add(AIConfig(id=1, provider="OpenAI", openai_api_key="sk-test"))
        session.add(SubscriptionConfig(id=1, subscriptions_enabled=True))
        await session.commit()

    client = AsyncMock()
    app = MaxBotApplication(client)

    # Render actual buttons
    rows = [[ResponseButton(text="💔 Отношения", kind="action", value="pain_relations")]]
    rendered_kb = response_buttons_keyboard(rows)
    raw_buttons = rendered_kb[0]["payload"]["buttons"]

    raw_update = make_raw_max_callback_update(
        "ai_btn:pain_relations",
        inline_keyboard_buttons=raw_buttons,
        update_id="upd_unique_1",
    )

    access_gate_called = False
    original_ensure_access = max_common.ensure_access_before_chat
    async def wrapped_ensure_access(client_arg, chat_id, user_arg):
        nonlocal access_gate_called
        access_gate_called = True
        return await original_ensure_access(client_arg, chat_id, user_arg)
    monkeypatch.setattr(max_common, "ensure_access_before_chat", wrapped_ensure_access)

    captured_prompt = None
    async def fake_run_ai(client_arg, chat_id, user_id, prompt, states):
        nonlocal captured_prompt
        assert access_gate_called is True
        captured_prompt = prompt

    monkeypatch.setattr(max_common, "run_ai_dialogue", fake_run_ai)

    await app.handle_update(raw_update)
    await await_spawned_tasks(app, INTERNAL_USER_ID)

    assert client.answer_callback.await_count == 1
    assert captured_prompt == '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку "💔 Отношения" (pain_relations)]'
    # No MAX original keyboard mutation
    assert client.edit_message.await_count == 0


# ==============================================================================
# 5. Raw MAX Duplicate-Action Label Test via handle_update
# ==============================================================================

@pytest.mark.asyncio
async def test_raw_max_duplicate_action_labels_via_handle_update(db_session, monkeypatch):
    async with db_session() as session:
        user = User(id=INTERNAL_USER_ID, first_name="MaxUser", name="MaxUser", gender="male", age="25", is_admin=True, accepted_disclaimer=True)
        session.add(user)
        session.add(AIConfig(id=1, provider="OpenAI", openai_api_key="sk-test"))
        session.add(SubscriptionConfig(id=1, subscriptions_enabled=True))
        await session.commit()

    client = AsyncMock()
    app = MaxBotApplication(client)

    mixed_rows = [
        [
            ResponseButton(text="Да", kind="action", value="continue"),
            ResponseButton(text="Продолжить", kind="action", value="continue"),
        ]
    ]
    rendered_kb = response_buttons_keyboard(mixed_rows)
    composed_kb = max_common.compose_max_keyboard(mixed_rows)

    # Verify compose_max_keyboard and response_buttons_keyboard contracts
    assert rendered_kb[0]["payload"]["buttons"][0][0]["payload"] == "ai_btn:continue|00"
    assert rendered_kb[0]["payload"]["buttons"][0][1]["payload"] == "ai_btn:continue|01"
    assert composed_kb[0]["payload"]["buttons"][0][0]["payload"] == "ai_btn:continue|00"
    assert composed_kb[0]["payload"]["buttons"][0][1]["payload"] == "ai_btn:continue|01"

    captured_prompts = []
    async def fake_run_ai(client_arg, chat_id, user_id, prompt, states):
        captured_prompts.append(prompt)

    monkeypatch.setattr(max_common, "run_ai_dialogue", fake_run_ai)

    # Click first duplicate: continue|00
    raw_update_1 = make_raw_max_callback_update(
        "ai_btn:continue|00",
        inline_keyboard_buttons=rendered_kb[0]["payload"]["buttons"],
        update_id="upd_dup_1",
        callback_id="cb_dup_1",
    )
    await app.handle_update(raw_update_1)
    await await_spawned_tasks(app, INTERNAL_USER_ID)

    assert len(captured_prompts) == 1
    assert captured_prompts[0] == '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку "Да" (continue)]'

    # Click second duplicate: continue|01
    raw_update_2 = make_raw_max_callback_update(
        "ai_btn:continue|01",
        inline_keyboard_buttons=rendered_kb[0]["payload"]["buttons"],
        update_id="upd_dup_2",
        callback_id="cb_dup_2",
    )
    await app.handle_update(raw_update_2)
    await await_spawned_tasks(app, INTERNAL_USER_ID)

    assert len(captured_prompts) == 2
    assert captured_prompts[1] == '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку "Продолжить" (continue)]'


# ==============================================================================
# 6. Raw MAX Legacy action: Prefix via handle_update
# ==============================================================================

@pytest.mark.asyncio
async def test_raw_max_legacy_action_normalization_via_handle_update(db_session, monkeypatch):
    async with db_session() as session:
        user = User(id=INTERNAL_USER_ID, first_name="MaxUser", name="MaxUser", gender="male", age="25", is_admin=True, accepted_disclaimer=True)
        session.add(user)
        session.add(AIConfig(id=1, provider="OpenAI", openai_api_key="sk-test"))
        session.add(SubscriptionConfig(id=1, subscriptions_enabled=True))
        await session.commit()

    client = AsyncMock()
    app = MaxBotApplication(client)

    captured_prompt = None
    async def fake_run_ai(client_arg, chat_id, user_id, prompt, states):
        nonlocal captured_prompt
        captured_prompt = prompt

    monkeypatch.setattr(max_common, "run_ai_dialogue", fake_run_ai)

    raw_update = make_raw_max_callback_update(
        "action:pain_relations",
        inline_keyboard_buttons=[[{"type": "callback", "text": "💔 Отношения", "payload": "action:pain_relations"}]],
        update_id="upd_legacy_1",
    )
    await app.handle_update(raw_update)
    await await_spawned_tasks(app, INTERNAL_USER_ID)

    assert client.answer_callback.await_count == 1
    assert captured_prompt == '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку "💔 Отношения" (pain_relations)]'
    assert "(action:pain_relations)" not in captured_prompt


# ==============================================================================
# 7. Raw MAX Access-Denied via handle_update
# ==============================================================================

@pytest.mark.asyncio
async def test_raw_max_access_denied_via_handle_update(db_session, monkeypatch):
    async with db_session() as session:
        user = User(id=INTERNAL_USER_ID, first_name="MaxUser", name="MaxUser", gender="male", age="25", is_admin=False)
        session.add(user)
        session.add(SubscriptionConfig(id=1, subscriptions_enabled=True))
        session.add(AIConfig(id=1, provider="OpenAI", openai_api_key="sk-test"))
        await session.commit()

    client = AsyncMock()
    app = MaxBotApplication(client)

    raw_update = make_raw_max_callback_update(
        "ai_btn:pain_relations",
        inline_keyboard_buttons=[[{"type": "callback", "text": "💔 Отношения", "payload": "ai_btn:pain_relations"}]],
        update_id="upd_denied_1",
    )

    run_ai_mock = AsyncMock()
    monkeypatch.setattr(max_common, "run_ai_dialogue", run_ai_mock)

    await app.handle_update(raw_update)
    await await_spawned_tasks(app, INTERNAL_USER_ID)

    # Callback ACKed
    assert client.answer_callback.await_count == 1
    # Access gate sent subscription offer
    assert any("активируйте подписку" in str(call) for call in client.send_message.call_args_list)
    # Dialogue never ran
    assert run_ai_mock.await_count == 0
    # No DB messages created
    async with db_session() as session:
        msgs = (await session.execute(select(DBMessage).where(DBMessage.user_id == INTERNAL_USER_ID))).scalars().all()
        assert len(msgs) == 0


# ==============================================================================
# 8. Raw MAX update_id Replay Test
# ==============================================================================

@pytest.mark.asyncio
async def test_raw_max_update_id_replay(db_session, monkeypatch):
    async with db_session() as session:
        user = User(id=INTERNAL_USER_ID, first_name="MaxUser", name="MaxUser", gender="male", age="25", is_admin=True, accepted_disclaimer=True)
        session.add(user)
        session.add(AIConfig(id=1, provider="OpenAI", openai_api_key="sk-test"))
        session.add(SubscriptionConfig(id=1, subscriptions_enabled=True))
        await session.commit()

    client = AsyncMock()
    app = MaxBotApplication(client)

    run_ai_count = 0
    async def fake_run_ai(client_arg, chat_id, user_id, prompt, states):
        nonlocal run_ai_count
        run_ai_count += 1

    monkeypatch.setattr(max_common, "run_ai_dialogue", fake_run_ai)

    raw_update = make_raw_max_callback_update(
        "ai_btn:pain_relations",
        inline_keyboard_buttons=[[{"type": "callback", "text": "💔 Отношения", "payload": "ai_btn:pain_relations"}]],
        update_id="upd_replay_999",
    )

    # First delivery
    await app.handle_update(raw_update)
    await await_spawned_tasks(app, INTERNAL_USER_ID)
    assert client.answer_callback.await_count == 1
    assert run_ai_count == 1

    # Second delivery (replay)
    await app.handle_update(raw_update)
    await await_spawned_tasks(app, INTERNAL_USER_ID)
    # No second ACK, no second AI run
    assert client.answer_callback.await_count == 1
    assert run_ai_count == 1


# ==============================================================================
# 9. Primary MAX 4-Row Topic Matrix via handle_update
# ==============================================================================

@pytest.mark.asyncio
async def test_primary_max_topic_matrix_via_handle_update(db_session, monkeypatch):
    async with db_session() as session:
        session.add(AIConfig(id=1, provider="OpenAI", openai_api_key="sk-test", memory_mode=MEMORY_MODE_TOPIC))
        session.add(BotGeneralConfig(id=1))
        session.add(Topic(id=10, name="Карьера и цели", is_active=True, auto_start_dialogue=True))
        session.add(User(id=INTERNAL_USER_ID, first_name="MAX User", name="MAX User", gender="male", age="25", is_admin=True, accepted_disclaimer=True, current_topic_id=None, current_dialogue_id=1))
        await session.commit()

    client = AsyncMock()
    app = MaxBotApplication(client)

    # Real kickoff orchestration, spy on kickoff, fake provider boundary
    max_kickoff_count = 0
    orig_max_kickoff = max_common.run_hidden_ai_kickoff
    async def spy_max_kickoff(*args, **kwargs):
        nonlocal max_kickoff_count
        max_kickoff_count += 1
        return await orig_max_kickoff(*args, **kwargs)
    monkeypatch.setattr(max_common, "run_hidden_ai_kickoff", spy_max_kickoff)

    max_provider_count = 0
    async def fake_max_get_ai_response(*args, **kwargs):
        nonlocal max_provider_count
        max_provider_count += 1
        return "Тестовый ответ MAX ИИ"
    monkeypatch.setattr(max_common, "get_ai_response", fake_max_get_ai_response)
    monkeypatch.setattr(max_common, "ensure_access_before_chat", AsyncMock(return_value=True))

    async def get_max_snapshot():
        async with db_session() as s:
            u = await s.get(User, INTERNAL_USER_ID)
            navs = (await s.execute(
                select(DBMessage).where(DBMessage.user_id == INTERNAL_USER_ID, DBMessage.role == "system_event").order_by(DBMessage.id.asc())
            )).scalars().all()
            welcomes = (await s.execute(
                select(func.count(DBMessage.id)).where(DBMessage.user_id == INTERNAL_USER_ID, DBMessage.role == TOPIC_WELCOME_ROLE)
            )).scalar_one()
            resume_notices = sum(1 for c in client.send_message.call_args_list if "✅ Продолжаем тему:" in str(c))
        return {
            "ack": client.answer_callback.await_count,
            "topic_id": u.current_topic_id if u else None,
            "dialogue_id": u.current_dialogue_id if u else None,
            "nav_count": len(navs),
            "nav_msgs": navs,
            "welcome_count": welcomes,
            "resume_notices": resume_notices,
            "kickoff_count": max_kickoff_count,
            "provider_count": max_provider_count,
        }

    # --------------------------------------------------------------------------
    # ROW 1: MAX First Entry
    # --------------------------------------------------------------------------
    s1_before = await get_max_snapshot()
    upd_first = make_raw_max_callback_update("select_topic_10", update_id="max_first_1")
    await app.handle_update(upd_first)
    await await_spawned_tasks(app, INTERNAL_USER_ID)
    s1_after = await get_max_snapshot()

    assert s1_after["ack"] - s1_before["ack"] == 1
    assert s1_after["topic_id"] == 10
    assert s1_after["nav_count"] - s1_before["nav_count"] == 1
    assert s1_after["nav_msgs"][-1].content == build_topic_auto_start_system_message("Карьера и цели")
    assert s1_after["welcome_count"] - s1_before["welcome_count"] == 1
    assert s1_after["resume_notices"] - s1_before["resume_notices"] == 0
    assert s1_after["kickoff_count"] - s1_before["kickoff_count"] == 1
    assert s1_after["provider_count"] - s1_before["provider_count"] == 1
    # Switched from main (1) to topic 10 (2) under MEMORY_MODE_TOPIC
    assert s1_after["dialogue_id"] == 2

    # --------------------------------------------------------------------------
    # ROW 2: MAX Genuine Resume
    # --------------------------------------------------------------------------
    # User switches to main first
    async with db_session() as s:
        u = await s.get(User, INTERNAL_USER_ID)
        u.current_topic_id = None
        u.current_dialogue_id = 1
        await s.commit()

    s2_before = await get_max_snapshot()
    upd_resume = make_raw_max_callback_update("select_topic_10", update_id="max_resume_2")
    await app.handle_update(upd_resume)
    await await_spawned_tasks(app, INTERNAL_USER_ID)
    s2_after = await get_max_snapshot()

    assert s2_after["ack"] - s2_before["ack"] == 1
    assert s2_after["topic_id"] == 10
    assert s2_after["nav_count"] - s2_before["nav_count"] == 1
    assert s2_after["nav_msgs"][-1].content == build_topic_resume_system_message("Карьера и цели")
    assert s2_after["welcome_count"] - s2_before["welcome_count"] == 0
    assert s2_after["resume_notices"] - s2_before["resume_notices"] == 1
    assert s2_after["kickoff_count"] - s2_before["kickoff_count"] == 1
    assert s2_after["provider_count"] - s2_before["provider_count"] == 1
    # Restored to saved topic dialogue 2 under MEMORY_MODE_TOPIC
    assert s2_after["dialogue_id"] == 2

    # --------------------------------------------------------------------------
    # ROW 3: MAX Already Current
    # --------------------------------------------------------------------------
    async with db_session() as s:
        uts_before = (await s.execute(select(UserTopicState).where(UserTopicState.user_id == INTERNAL_USER_ID))).scalars().all()
        uts_snapshot = [(x.topic_id, x.dialogue_id) for x in uts_before]

    s3_before = await get_max_snapshot()
    upd_current = make_raw_max_callback_update("select_topic_10", update_id="max_current_3")
    await app.handle_update(upd_current)
    await await_spawned_tasks(app, INTERNAL_USER_ID)
    s3_after = await get_max_snapshot()

    assert s3_after["ack"] - s3_before["ack"] == 1
    assert s3_after["topic_id"] == 10
    assert s3_after["dialogue_id"] == 2
    assert s3_after["nav_count"] - s3_before["nav_count"] == 0
    assert s3_after["welcome_count"] - s3_before["welcome_count"] == 0
    assert s3_after["resume_notices"] - s3_before["resume_notices"] == 0
    assert s3_after["kickoff_count"] - s3_before["kickoff_count"] == 0
    assert s3_after["provider_count"] - s3_before["provider_count"] == 0
    assert any("Вы уже находитесь в теме «Карьера и цели»." in str(call) for call in client.send_message.call_args_list)

    async with db_session() as s:
        uts_after = (await s.execute(select(UserTopicState).where(UserTopicState.user_id == INTERNAL_USER_ID))).scalars().all()
        assert [(x.topic_id, x.dialogue_id) for x in uts_after] == uts_snapshot

    # --------------------------------------------------------------------------
    # ROW 4: MAX Return to Main
    # --------------------------------------------------------------------------
    s4_before = await get_max_snapshot()
    upd_reset = make_raw_max_callback_update("reset_topic", update_id="max_reset_4")
    await app.handle_update(upd_reset)
    await await_spawned_tasks(app, INTERNAL_USER_ID)
    s4_after = await get_max_snapshot()

    assert s4_after["ack"] - s4_before["ack"] == 1
    assert s4_after["topic_id"] is None
    assert s4_after["nav_count"] - s4_before["nav_count"] == 1
    assert s4_after["nav_msgs"][-1].content == build_main_dialogue_resume_system_message()
    assert s4_after["welcome_count"] - s4_before["welcome_count"] == 0
    assert s4_after["resume_notices"] - s4_before["resume_notices"] == 0
    assert s4_after["kickoff_count"] - s4_before["kickoff_count"] == 1
    assert s4_after["provider_count"] - s4_before["provider_count"] == 1
    # Restored to saved main dialogue 1 under MEMORY_MODE_TOPIC
    assert s4_after["dialogue_id"] == 1
    assert any("✅ Мы вернулись в основной диалог." in str(call) for call in client.send_message.call_args_list)


# ==============================================================================
# 10. Telegram Primary 4-Row Topic Matrix via Handlers
# ==============================================================================

@pytest.mark.asyncio
async def test_telegram_primary_topic_matrix_via_handlers(db_session, monkeypatch):
    bot = SimpleNamespace(send_message=AsyncMock())
    state = AsyncMock()

    async with db_session() as session:
        session.add(AIConfig(id=1, provider="OpenAI", openai_api_key="sk-test", memory_mode=MEMORY_MODE_TOPIC))
        session.add(BotGeneralConfig(id=1))
        session.add(Topic(id=10, name="Карьера и цели", is_active=True, auto_start_dialogue=True))
        session.add(User(id=100, first_name="TG User", name="TG User", gender="male", age="25", is_admin=True, accepted_disclaimer=True, current_topic_id=None, current_dialogue_id=1))
        await session.commit()

    handlers.user_isolated_turn_queues.clear()
    handlers.user_processing_tasks.clear()
    handlers.user_message_buffers.clear()

    tg_provider_count = 0
    async def fake_tg_provider(user_id, prompt_text, *args, **kwargs):
        nonlocal tg_provider_count
        tg_provider_count += 1
        return "Тестовый ответ TG ИИ"
    monkeypatch.setattr("handlers.ai_integration.generate_response", fake_tg_provider)
    monkeypatch.setattr("ai_integration.generate_response", fake_tg_provider)

    tg_kickoff_count = 0
    orig_tg_kickoff = handlers._start_telegram_hidden_kickoff
    async def spy_tg_kickoff(*args, **kwargs):
        nonlocal tg_kickoff_count
        tg_kickoff_count += 1
        return await orig_tg_kickoff(*args, **kwargs)
    monkeypatch.setattr(handlers, "_start_telegram_hidden_kickoff", spy_tg_kickoff)

    async def drain_tg_runner(user_id: int):
        while handlers._has_user_turn_work(user_id) or (user_id in handlers.user_processing_tasks and not handlers.user_processing_tasks[user_id].done()):
            task = handlers.user_processing_tasks.get(user_id)
            if task:
                await task
            await asyncio.sleep(0.01)

    async def get_tg_snapshot():
        async with db_session() as s:
            u = await s.get(User, 100)
            navs = (await s.execute(
                select(DBMessage).where(DBMessage.user_id == 100, DBMessage.role == "system_event").order_by(DBMessage.id.asc())
            )).scalars().all()
            welcomes = (await s.execute(
                select(func.count(DBMessage.id)).where(DBMessage.user_id == 100, DBMessage.role == TOPIC_WELCOME_ROLE)
            )).scalar_one()
            resume_notices = sum(1 for c in bot.send_message.call_args_list if "✅ Продолжаем тему:" in str(c))
        return {
            "topic_id": u.current_topic_id if u else None,
            "dialogue_id": u.current_dialogue_id if u else None,
            "nav_count": len(navs),
            "nav_msgs": navs,
            "welcome_count": welcomes,
            "resume_notices": resume_notices,
            "kickoff_count": tg_kickoff_count,
            "provider_count": tg_provider_count,
        }

    # --------------------------------------------------------------------------
    # ROW 1: TG First Entry
    # --------------------------------------------------------------------------
    s1_before = await get_tg_snapshot()
    cb_first = SimpleNamespace(
        data="select_topic_10",
        from_user=SimpleNamespace(id=100),
        message=SimpleNamespace(message_id=1001, chat=SimpleNamespace(id=100), answer=AsyncMock(), delete=AsyncMock()),
        answer=AsyncMock(),
    )
    await handlers.process_topic_selection(cb_first, state, bot)
    await drain_tg_runner(100)
    s1_after = await get_tg_snapshot()

    assert cb_first.answer.await_count == 1
    assert s1_after["topic_id"] == 10
    assert s1_after["nav_count"] - s1_before["nav_count"] == 1
    assert s1_after["nav_msgs"][-1].content == build_topic_auto_start_system_message("Карьера и цели")
    assert s1_after["welcome_count"] - s1_before["welcome_count"] == 1
    assert s1_after["resume_notices"] - s1_before["resume_notices"] == 0
    assert s1_after["kickoff_count"] - s1_before["kickoff_count"] == 1
    assert s1_after["provider_count"] - s1_before["provider_count"] == 1
    # Switched from main (1) to topic 10 (2) under MEMORY_MODE_TOPIC
    assert s1_after["dialogue_id"] == 2

    # --------------------------------------------------------------------------
    # ROW 2: TG Genuine Resume
    # --------------------------------------------------------------------------
    # User switches to main first
    async with db_session() as s:
        u = await s.get(User, 100)
        u.current_topic_id = None
        u.current_dialogue_id = 1
        await s.commit()

    s2_before = await get_tg_snapshot()
    cb_resume = SimpleNamespace(
        data="select_topic_10",
        from_user=SimpleNamespace(id=100),
        message=SimpleNamespace(message_id=1002, chat=SimpleNamespace(id=100), answer=AsyncMock(), delete=AsyncMock()),
        answer=AsyncMock(),
    )
    await handlers.process_topic_selection(cb_resume, state, bot)
    await drain_tg_runner(100)
    s2_after = await get_tg_snapshot()

    assert cb_resume.answer.await_count == 1
    assert s2_after["topic_id"] == 10
    assert s2_after["nav_count"] - s2_before["nav_count"] == 1
    assert s2_after["nav_msgs"][-1].content == build_topic_resume_system_message("Карьера и цели")
    assert s2_after["welcome_count"] - s2_before["welcome_count"] == 0
    assert s2_after["resume_notices"] - s2_before["resume_notices"] == 1
    assert s2_after["kickoff_count"] - s2_before["kickoff_count"] == 1
    assert s2_after["provider_count"] - s2_before["provider_count"] == 1
    # Restored to saved topic dialogue 2 under MEMORY_MODE_TOPIC
    assert s2_after["dialogue_id"] == 2

    # --------------------------------------------------------------------------
    # ROW 3: TG Already Current
    # --------------------------------------------------------------------------
    async with db_session() as s:
        uts_before = (await s.execute(select(UserTopicState).where(UserTopicState.user_id == 100))).scalars().all()
        uts_snapshot = [(x.topic_id, x.dialogue_id) for x in uts_before]

    s3_before = await get_tg_snapshot()
    cb_current = SimpleNamespace(
        data="select_topic_10",
        from_user=SimpleNamespace(id=100),
        message=SimpleNamespace(message_id=1003, chat=SimpleNamespace(id=100), answer=AsyncMock(), delete=AsyncMock()),
        answer=AsyncMock(),
    )
    await handlers.process_topic_selection(cb_current, state, bot)
    await drain_tg_runner(100)
    s3_after = await get_tg_snapshot()

    assert cb_current.answer.await_count == 1
    assert s3_after["topic_id"] == 10
    assert s3_after["dialogue_id"] == 2
    assert s3_after["nav_count"] - s3_before["nav_count"] == 0
    assert s3_after["welcome_count"] - s3_before["welcome_count"] == 0
    assert s3_after["resume_notices"] - s3_before["resume_notices"] == 0
    assert s3_after["kickoff_count"] - s3_before["kickoff_count"] == 0
    assert s3_after["provider_count"] - s3_before["provider_count"] == 0
    assert any("Вы уже находитесь в теме «Карьера и цели»." in str(call) for call in cb_current.message.answer.call_args_list)

    async with db_session() as s:
        uts_after = (await s.execute(select(UserTopicState).where(UserTopicState.user_id == 100))).scalars().all()
        assert [(x.topic_id, x.dialogue_id) for x in uts_after] == uts_snapshot

    # --------------------------------------------------------------------------
    # ROW 4: TG Return to Main
    # --------------------------------------------------------------------------
    s4_before = await get_tg_snapshot()
    cb_reset = SimpleNamespace(
        data="reset_topic",
        from_user=SimpleNamespace(id=100),
        message=SimpleNamespace(message_id=1004, chat=SimpleNamespace(id=100), answer=AsyncMock(), delete=AsyncMock()),
        answer=AsyncMock(),
    )
    await handlers.process_topic_reset(cb_reset, bot, state)
    await drain_tg_runner(100)
    s4_after = await get_tg_snapshot()

    assert cb_reset.answer.await_count == 1
    assert s4_after["topic_id"] is None
    assert s4_after["nav_count"] - s4_before["nav_count"] == 1
    assert s4_after["nav_msgs"][-1].content == build_main_dialogue_resume_system_message()
    assert s4_after["welcome_count"] - s4_before["welcome_count"] == 0
    assert s4_after["resume_notices"] - s4_before["resume_notices"] == 0
    assert s4_after["kickoff_count"] - s4_before["kickoff_count"] == 1
    assert s4_after["provider_count"] - s4_before["provider_count"] == 1
    # Restored to saved main dialogue 1 under MEMORY_MODE_TOPIC
    assert s4_after["dialogue_id"] == 1
    assert any("✅ Мы вернулись в основной диалог." in str(call) for call in bot.send_message.call_args_list)


# ==============================================================================
# 11. Service Action vs Generic Button Routing & Matrix
# ==============================================================================

@pytest.mark.asyncio
async def test_telegram_and_max_button_service_matrix(db_session, monkeypatch):
    bot = SimpleNamespace(send_message=AsyncMock())
    state = AsyncMock()

    async with db_session() as session:
        session.add(User(id=505, first_name="TG User", name="TG User", gender="male", age="25", is_admin=True, accepted_disclaimer=True))
        session.add(AIConfig(id=1, provider="OpenAI", openai_api_key="sk-test", memory_mode=MEMORY_MODE_TOPIC))
        session.add(Topic(id=1, name="General Topic", is_active=True))
        session.add(Topic(id=2, name="Visited Topic", is_active=True))
        session.add(UserTopicState(user_id=505, topic_id=2, dialogue_id=1))
        await record_topic_welcome_shown(session, 505, 1, 2)
        await session.commit()

    handlers.user_message_buffers.clear()
    handlers._ai_button_claims.clear()

    # 1. TG svc:menu
    cb_menu = SimpleNamespace(
        data="ai_btn:svc:menu", from_user=SimpleNamespace(id=505),
        message=SimpleNamespace(message_id=10, chat=SimpleNamespace(id=505), answer=AsyncMock(), edit_reply_markup=AsyncMock(),
                                reply_markup=handlers.InlineKeyboardMarkup(inline_keyboard=[[handlers.InlineKeyboardButton(text="В меню", callback_data="ai_btn:svc:menu")]])),
        answer=AsyncMock(),
    )
    await handlers.process_response_button(cb_menu, state, bot)
    assert cb_menu.answer.await_count == 1
    assert 505 not in handlers.user_message_buffers
    assert cb_menu.message.answer.await_count == 1

    # TG svc:menu duplicate click: service not executed again, duplicate acknowledged, no generic processing
    cb_menu.answer.reset_mock()
    cb_menu.message.answer.reset_mock()
    await handlers.process_response_button(cb_menu, state, bot)
    assert cb_menu.message.answer.await_count == 0
    assert cb_menu.answer.await_count == 1
    assert 505 not in handlers.user_message_buffers

    # 2. TG svc:topics
    cb_topics = SimpleNamespace(
        data="ai_btn:svc:topics", from_user=SimpleNamespace(id=505),
        message=SimpleNamespace(message_id=11, chat=SimpleNamespace(id=505), answer=AsyncMock(), edit_reply_markup=AsyncMock(),
                                reply_markup=handlers.InlineKeyboardMarkup(inline_keyboard=[[handlers.InlineKeyboardButton(text="Темы", callback_data="ai_btn:svc:topics")]])),
        answer=AsyncMock(),
    )
    monkeypatch.setattr(handlers, "select_topic_menu", AsyncMock())
    await handlers.process_response_button(cb_topics, state, bot)
    assert cb_topics.answer.await_count == 1
    assert handlers.select_topic_menu.await_count == 1

    # 3. TG svc:topic:2 (visited genuine resume)
    monkeypatch.setattr(handlers, "_start_telegram_hidden_kickoff", AsyncMock())
    cb_topic_resume = SimpleNamespace(
        data="ai_btn:svc:topic:2", from_user=SimpleNamespace(id=505),
        message=SimpleNamespace(message_id=12, chat=SimpleNamespace(id=505), answer=AsyncMock(), edit_reply_markup=AsyncMock(), delete=AsyncMock(),
                                reply_markup=handlers.InlineKeyboardMarkup(inline_keyboard=[[handlers.InlineKeyboardButton(text="Тема 2", callback_data="ai_btn:svc:topic:2")]])),
        answer=AsyncMock(),
    )
    def _make_topic_copy(update=None):
        return SimpleNamespace(
            data=(update or {}).get("data", "select_topic_2"),
            from_user=SimpleNamespace(id=505),
            message=SimpleNamespace(message_id=12, chat=SimpleNamespace(id=505), answer=AsyncMock(), edit_reply_markup=AsyncMock(), delete=AsyncMock()),
            answer=cb_topic_resume.answer,
        )
    cb_topic_resume.model_copy = _make_topic_copy
    await handlers.process_response_button(cb_topic_resume, state, bot)
    assert cb_topic_resume.answer.await_count == 1
    assert any("✅ Продолжаем тему: «Visited Topic»." in str(call) for call in bot.send_message.call_args_list)

    # 4. TG svc:topic:main
    bot.send_message.reset_mock()
    cb_topic_main = SimpleNamespace(
        data="ai_btn:svc:topic:main", from_user=SimpleNamespace(id=505),
        message=SimpleNamespace(message_id=13, chat=SimpleNamespace(id=505), answer=AsyncMock(), edit_reply_markup=AsyncMock(),
                                reply_markup=handlers.InlineKeyboardMarkup(inline_keyboard=[[handlers.InlineKeyboardButton(text="Основной", callback_data="ai_btn:svc:topic:main")]])),
        answer=AsyncMock(),
    )
    await handlers.process_response_button(cb_topic_main, state, bot)
    assert cb_topic_main.answer.await_count == 1
    assert any("✅ Мы вернулись в основной диалог." in str(call) for call in bot.send_message.call_args_list)

    # 5. TG unknown svc:*
    cb_unknown = SimpleNamespace(
        data="ai_btn:svc:unknown_action", from_user=SimpleNamespace(id=505),
        message=SimpleNamespace(message_id=14, chat=SimpleNamespace(id=505), answer=AsyncMock(), edit_reply_markup=AsyncMock(),
                                reply_markup=handlers.InlineKeyboardMarkup(inline_keyboard=[[handlers.InlineKeyboardButton(text="Неизвестно", callback_data="ai_btn:svc:unknown_action")]])),
        answer=AsyncMock(),
    )
    await handlers.process_response_button(cb_unknown, state, bot)
    assert cb_unknown.answer.await_count == 1
    assert 505 not in handlers.user_message_buffers

    # 6. TG generic pain_relations & duplicate suppression
    mock_process_buf = AsyncMock()
    monkeypatch.setattr(handlers, "process_buffered_messages", mock_process_buf)
    cb_generic = SimpleNamespace(
        data="ai_btn:pain_relations", from_user=SimpleNamespace(id=505),
        message=SimpleNamespace(message_id=15, chat=SimpleNamespace(id=505), answer=AsyncMock(), edit_reply_markup=AsyncMock(),
                                reply_markup=handlers.InlineKeyboardMarkup(inline_keyboard=[[handlers.InlineKeyboardButton(text="💔 Отношения", callback_data="ai_btn:pain_relations")]])),
        answer=AsyncMock(),
    )
    await handlers.process_response_button(cb_generic, state, bot)
    assert cb_generic.answer.await_count == 1
    assert mock_process_buf.await_count == 1
    assert any("Ответ принят" in str(call) for call in cb_generic.message.answer.call_args_list)

    # Duplicate click on same generic button is suppressed
    mock_process_buf.reset_mock()
    cb_generic.answer.reset_mock()
    await handlers.process_response_button(cb_generic, state, bot)
    assert cb_generic.answer.await_count == 1
    assert mock_process_buf.await_count == 0

    # 7. MAX Service Matrix: svc:menu, svc:topics, svc:topic:2, svc:topic:main, unknown svc:*
    max_client = AsyncMock()
    max_app_inst = MaxBotApplication(max_client)
    monkeypatch.setattr(max_common, "show_menu", AsyncMock())
    monkeypatch.setattr(max_topics, "show_topics", AsyncMock())
    monkeypatch.setattr(max_topics, "select_topic", AsyncMock())
    monkeypatch.setattr(max_topics, "reset_topic", AsyncMock())

    run_ai_mock = AsyncMock()
    monkeypatch.setattr(max_common, "run_ai_dialogue", run_ai_mock)

    # MAX svc:menu
    max_client.answer_callback.reset_mock()
    await max_app_inst.handle_update(make_raw_max_callback_update("ai_btn:svc:menu", update_id="max_svc_menu"))
    await await_spawned_tasks(max_app_inst, INTERNAL_USER_ID)
    assert max_client.answer_callback.await_count == 1
    assert max_common.show_menu.await_count == 1
    assert run_ai_mock.await_count == 0

    # MAX svc:topics
    max_client.answer_callback.reset_mock()
    await max_app_inst.handle_update(make_raw_max_callback_update("ai_btn:svc:topics", update_id="max_svc_topics"))
    await await_spawned_tasks(max_app_inst, INTERNAL_USER_ID)
    assert max_client.answer_callback.await_count == 1
    assert max_topics.show_topics.await_count == 1
    assert run_ai_mock.await_count == 0

    # MAX svc:topic:2
    max_client.answer_callback.reset_mock()
    await max_app_inst.handle_update(make_raw_max_callback_update("ai_btn:svc:topic:2", update_id="max_svc_topic_2"))
    await await_spawned_tasks(max_app_inst, INTERNAL_USER_ID)
    assert max_client.answer_callback.await_count == 1
    assert max_topics.select_topic.await_count == 1
    assert run_ai_mock.await_count == 0

    # MAX svc:topic:main
    max_client.answer_callback.reset_mock()
    await max_app_inst.handle_update(make_raw_max_callback_update("ai_btn:svc:topic:main", update_id="max_svc_main"))
    await await_spawned_tasks(max_app_inst, INTERNAL_USER_ID)
    assert max_client.answer_callback.await_count == 1
    assert max_topics.reset_topic.await_count == 1
    assert run_ai_mock.await_count == 0

    # MAX unknown svc:*
    max_client.answer_callback.reset_mock()
    await max_app_inst.handle_update(make_raw_max_callback_update("ai_btn:svc:unknown_random", update_id="max_svc_unk"))
    await await_spawned_tasks(max_app_inst, INTERNAL_USER_ID)
    assert max_client.answer_callback.await_count == 1
    assert run_ai_mock.await_count == 0


# ==============================================================================
# 12. Delayed Resume Ordering & Ownership
# ==============================================================================

@pytest.mark.asyncio
async def test_delayed_resume_ordering_and_ownership(db_session, monkeypatch):
    bot = SimpleNamespace(send_message=AsyncMock())
    state = AsyncMock()
    client = AsyncMock()

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
        # Initial nav message
        nav_msg = DBMessage(user_id=401, dialogue_id=1, topic_id=40, role="system_event", content=build_topic_resume_system_message("Тема Отложенная"))
        session.add(nav_msg)
        await session.commit()
        nav_msg_id = nav_msg.id

    # 1. TG Disclaimer Delayed Resume via disclaimer_accepted_handler
    events_order.clear()
    state.get_data = AsyncMock(return_value={
        "pending_auto_start_topic_id": 40,
        "pending_auto_start_dialogue_id": 1,
        "pending_auto_start_kind": "resume",
        "pending_auto_start_message_id": nav_msg_id,
    })
    cb_disclaimer = SimpleNamespace(
        data="confirm_disclaimer",
        from_user=SimpleNamespace(id=401),
        message=SimpleNamespace(message_id=201, chat=SimpleNamespace(id=401), answer=AsyncMock(), delete=AsyncMock()),
    )
    await handlers.disclaimer_accepted_handler(cb_disclaimer, state, bot)

    assert events_order == ["tg_visible_send", "tg_hidden_kickoff"]
    # Reuses existing nav_msg_id, delta = 0
    async with db_session() as session:
        total_events = (await session.execute(select(DBMessage).where(DBMessage.user_id == 401, DBMessage.topic_id == 40, DBMessage.role == "system_event"))).scalars().all()
        assert len(total_events) == 1

    # 2. TG Profile Onboarding Delayed Resume via _resume_after_profile_onboarding
    events_order.clear()
    onboarding_data = {
        "topic_intro_after_onboarding": 40,
        "topic_intro_dialogue_id": 1,
        "topic_intro_welcome_needed": False,  # i.e. resume
    }
    dummy_msg = SimpleNamespace(chat=SimpleNamespace(id=401))
    await handlers._resume_after_profile_onboarding(onboarding_data, dummy_msg, state, bot, 401)
    assert events_order == ["tg_visible_send", "tg_hidden_kickoff"]

    # 3. MAX Delayed Resume via resume_pending_ai_turn
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
# 13. Dynamic HTML-Sensitive Topic Name Contract
# ==============================================================================

@pytest.mark.asyncio
async def test_dynamic_html_escaped_topic_names(db_session, monkeypatch):
    bot = SimpleNamespace(send_message=AsyncMock())
    client = AsyncMock()
    state = AsyncMock()

    raw_topic_name = 'Тема <A & "B">'
    expected_escaped = 'Тема &lt;A &amp; &quot;B&quot;&gt;'

    async with db_session() as session:
        session.add(AIConfig(id=1, provider="OpenAI", openai_api_key="sk-test", memory_mode=MEMORY_MODE_TOPIC))
        session.add(Topic(id=30, name=raw_topic_name, is_active=True))
        session.add(User(id=301, first_name="TG User", name="TG User", gender="male", age="25", is_admin=True, accepted_disclaimer=True, current_topic_id=None, current_dialogue_id=1))
        session.add(User(id=302, first_name="MAX User", name="MAX User", gender="male", age="25", is_admin=True, accepted_disclaimer=True, current_topic_id=None, current_dialogue_id=1))
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

    # 2. Telegram disclaimer-delayed resume visible message
    bot.send_message.reset_mock()
    state.get_data = AsyncMock(return_value={
        "pending_auto_start_topic_id": 30,
        "pending_auto_start_dialogue_id": 1,
        "pending_auto_start_kind": "resume",
        "pending_auto_start_message_id": 999,
    })
    cb_disc = SimpleNamespace(
        data="confirm_disclaimer",
        from_user=SimpleNamespace(id=301),
        message=SimpleNamespace(message_id=2, chat=SimpleNamespace(id=301), answer=AsyncMock(), delete=AsyncMock()),
    )
    await handlers.disclaimer_accepted_handler(cb_disc, state, bot)
    assert any(f"✅ Продолжаем тему: «{expected_escaped}»." in str(call) for call in bot.send_message.call_args_list)

    # 3. Telegram onboarding-delayed resume visible message
    bot.send_message.reset_mock()
    onboarding_data = {
        "topic_intro_after_onboarding": 30,
        "topic_intro_dialogue_id": 1,
        "topic_intro_welcome_needed": False,
    }
    dummy_msg = SimpleNamespace(chat=SimpleNamespace(id=301))
    await handlers._resume_after_profile_onboarding(onboarding_data, dummy_msg, state, bot, 301)
    assert any(f"✅ Продолжаем тему: «{expected_escaped}»." in str(call) for call in bot.send_message.call_args_list)

    # 4. MAX direct resume visible message
    await max_topics.select_topic(client, 302, 302, 30)
    assert any(f"✅ Продолжаем тему: <b>{expected_escaped}</b>." in str(call) for call in client.send_message.call_args_list)

    # 5. MAX delayed resume visible message
    client.send_message.reset_mock()
    max_state_data = {
        "pending_auto_start_topic_id": 30,
        "pending_auto_start_dialogue_id": 1,
        "pending_auto_start_kind": "resume",
    }
    await max_common.resume_pending_ai_turn(client, 302, 302, max_state_data)
    assert any(f"✅ Продолжаем тему: <b>{expected_escaped}</b>." in str(call) for call in client.send_message.call_args_list)

    # 6. Canonical system event builder receives RAW topic name
    async with db_session() as session:
        events = (await session.execute(select(DBMessage).where(DBMessage.topic_id == 30, DBMessage.role == "system_event"))).scalars().all()
        assert len(events) >= 1
        for ev in events:
            assert ev.content == build_topic_resume_system_message(raw_topic_name)
            assert expected_escaped not in ev.content


# ==============================================================================
# 14. Comprehensive Tarot Deduplication across All 3 Production Paths
# ==============================================================================

@pytest.mark.asyncio
async def test_tarot_deduplication_comprehensive_contract(db_session, monkeypatch):
    bot = SimpleNamespace(
        send_message=AsyncMock(),
        send_photo=AsyncMock(),
        send_audio=AsyncMock(),
        send_chat_action=AsyncMock(),
        send_media_group=AsyncMock(),
    )
    state = AsyncMock()

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
            context_limit_first=50,
            context_limit_recent=50,
        )
        session.add(ai_config)
        session.add(BotGeneralConfig(id=1))
        # Isolated test users
        session.add(User(id=801, first_name="Tarot Path 1", name="Tarot Path 1", gender="male", age="25", is_admin=True, accepted_disclaimer=True, current_topic_id=None, current_dialogue_id=1))
        session.add(User(id=802, first_name="Tarot Path 2", name="Tarot Path 2", gender="male", age="25", is_admin=True, accepted_disclaimer=True, current_topic_id=None, current_dialogue_id=1))
        session.add(User(id=803, first_name="Tarot Path 3", name="Tarot Path 3", gender="male", age="25", is_admin=True, accepted_disclaimer=True, current_topic_id=None, current_dialogue_id=1))
        # 3 Tarot cards + back
        session.add(MediaLibrary(id=1, file_name="card_1.jpg", file_id="f_1", category="tarot", media_type="photo", description="Маг"))
        session.add(MediaLibrary(id=2, file_name="card_2.jpg", file_id="f_2", category="tarot", media_type="photo", description="Жрица"))
        session.add(MediaLibrary(id=3, file_name="card_3.jpg", file_id="f_3", category="tarot", media_type="photo", description="Императрица"))
        session.add(MediaLibrary(id=4, file_name="_back", file_id="f_back", category="tarot", media_type="photo", description="Рубашка"))
        await session.commit()

    captured_openai_payloads = []
    fake_interp_texts = {}

    class FakeChatCompletions:
        async def create(self, **kwargs):
            captured_openai_payloads.append(kwargs)
            # Find matching response or generate
            call_idx = len(captured_openai_payloads)
            interp = fake_interp_texts.get(call_idx, f"Интерпретация ответа {call_idx}")
            choice = SimpleNamespace(message=SimpleNamespace(content=interp, role="assistant"))
            return SimpleNamespace(choices=[choice], usage=SimpleNamespace(prompt_tokens=20, completion_tokens=20, total_tokens=40))

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

    # ==========================================================================
    # PATH 1: process_card_selection (Round 1, Round 2, Round 3)
    # ==========================================================================
    # Round 1
    await handlers._save_card_spread_state(801, {
        "category": "tarot",
        "topic_id": None,
        "rounds_left": 2,
        "total_rounds": 3,
        "cards_per_round": 1,
        "hidden": False,
        "chosen_card_ids": [],
        "selected_file_ids": [],
        "pending_card_ids": [1],
    })
    fake_interp_texts[1] = "Интерпретация карты 1 (Маг)"

    cb_r1 = SimpleNamespace(
        data="card_select_1", from_user=SimpleNamespace(id=801),
        message=SimpleNamespace(message_id=2001, chat=SimpleNamespace(id=801), answer_photo=AsyncMock(), answer=AsyncMock(), delete=AsyncMock()),
        answer=AsyncMock(),
    )
    await handlers.process_card_selection(cb_r1, bot)

    r1_synthetic = handlers._card_selection_system_message("card_1.jpg: Маг", {
        "chosen_card_ids": [], "total_rounds": 3, "rounds_left": 2
    })
    r1_payload_msgs = captured_openai_payloads[0]["messages"]
    assert sum(1 for m in r1_payload_msgs if m["content"] == r1_synthetic) == 1
    async with db_session() as session:
        r1_user_msgs = (await session.execute(select(DBMessage).where(DBMessage.user_id == 801, DBMessage.role == "user", DBMessage.content == r1_synthetic))).scalars().all()
        assert len(r1_user_msgs) == 1
        r1_asst_msgs = (await session.execute(select(DBMessage).where(DBMessage.user_id == 801, DBMessage.role == "assistant", DBMessage.content == "Интерпретация карты 1 (Маг)"))).scalars().all()
        assert len(r1_asst_msgs) == 1

    # Round 2
    spread_r2 = await handlers._get_card_spread_state(801)
    spread_r2["pending_card_ids"] = [2]
    await handlers._save_card_spread_state(801, spread_r2)
    fake_interp_texts[2] = "Интерпретация карты 2 (Жрица)"
    cb_r2 = SimpleNamespace(
        data="card_select_2", from_user=SimpleNamespace(id=801),
        message=SimpleNamespace(message_id=2002, chat=SimpleNamespace(id=801), answer_photo=AsyncMock(), answer=AsyncMock(), delete=AsyncMock()),
        answer=AsyncMock(),
    )
    await handlers.process_card_selection(cb_r2, bot)

    r2_synthetic = handlers._card_selection_system_message("card_2.jpg: Жрица", {
        "chosen_card_ids": [1], "selected_file_ids": ["f_1"], "total_rounds": 3, "rounds_left": 1
    })
    r2_payload_msgs = captured_openai_payloads[1]["messages"]
    # Round 1 in history:
    assert sum(1 for m in r2_payload_msgs if m["content"] == r1_synthetic) == 1
    assert sum(1 for m in r2_payload_msgs if m["content"] == "Интерпретация карты 1 (Маг)") == 1
    # Current Round 2 synthetic appears EXACTLY ONCE:
    assert sum(1 for m in r2_payload_msgs if m["content"] == r2_synthetic) == 1
    async with db_session() as session:
        r2_user_msgs = (await session.execute(select(DBMessage).where(DBMessage.user_id == 801, DBMessage.role == "user", DBMessage.content == r2_synthetic))).scalars().all()
        assert len(r2_user_msgs) == 1
        r2_asst_msgs = (await session.execute(select(DBMessage).where(DBMessage.user_id == 801, DBMessage.role == "assistant", DBMessage.content == "Интерпретация карты 2 (Жрица)"))).scalars().all()
        assert len(r2_asst_msgs) == 1

    # Round 3
    spread_r3 = await handlers._get_card_spread_state(801)
    spread_r3["pending_card_ids"] = [3]
    await handlers._save_card_spread_state(801, spread_r3)
    fake_interp_texts[3] = "Интерпретация карты 3 (Императрица)"
    cb_r3 = SimpleNamespace(
        data="card_select_3", from_user=SimpleNamespace(id=801),
        message=SimpleNamespace(message_id=2003, chat=SimpleNamespace(id=801), answer_photo=AsyncMock(), answer=AsyncMock(), delete=AsyncMock()),
        answer=AsyncMock(),
    )
    await handlers.process_card_selection(cb_r3, bot)

    r3_synthetic = handlers._card_selection_system_message("card_3.jpg: Императрица", {
        "chosen_card_ids": [1, 2], "selected_file_ids": ["f_1", "f_2"], "total_rounds": 3, "rounds_left": 0
    })
    r3_payload_msgs = captured_openai_payloads[2]["messages"]
    # Prior rounds retained
    assert sum(1 for m in r3_payload_msgs if m["content"] == r1_synthetic) == 1
    assert sum(1 for m in r3_payload_msgs if m["content"] == "Интерпретация карты 1 (Маг)") == 1
    assert sum(1 for m in r3_payload_msgs if m["content"] == r2_synthetic) == 1
    assert sum(1 for m in r3_payload_msgs if m["content"] == "Интерпретация карты 2 (Жрица)") == 1
    # Current Round 3 synthetic appears EXACTLY ONCE
    assert sum(1 for m in r3_payload_msgs if m["content"] == r3_synthetic) == 1
    # Final round instruction appears in r3_synthetic exactly once
    final_instruction = "[СИСТЕМА: Это последний, раунд 3 из 3. Заверши расклад после интерпретации.]"
    assert final_instruction in r3_synthetic
    assert sum(1 for m in r3_payload_msgs if final_instruction in str(m["content"])) == 1
    async with db_session() as session:
        r3_user_msgs = (await session.execute(select(DBMessage).where(DBMessage.user_id == 801, DBMessage.role == "user", DBMessage.content == r3_synthetic))).scalars().all()
        assert len(r3_user_msgs) == 1
        r3_asst_msgs = (await session.execute(select(DBMessage).where(DBMessage.user_id == 801, DBMessage.role == "assistant", DBMessage.content == "Интерпретация карты 3 (Императрица)"))).scalars().all()
        assert len(r3_asst_msgs) == 1

    # Normal spread completion verification
    assert await handlers._get_card_spread_state(801) is None
    assert any("Твой расклад целиком:" in str(call) for call in bot.send_message.call_args_list)
    final_album_calls = [
        call for call in handlers.send_card_album.call_args_list
        if call.kwargs.get("context") == "process_card_selection.final_spread"
    ]
    assert len(final_album_calls) == 1
    assert final_album_calls[0].args[1] == 801
    assert final_album_calls[0].args[2] == ["f_1", "f_2", "f_3"]

    # ==========================================================================
    # PATH 2: process_buffered_messages (Random Card)
    # ==========================================================================
    captured_openai_payloads.clear()
    fake_interp_texts.clear()
    fake_interp_texts[1] = "Вот твоя карта [RANDOM_IMG:tarot,1]"
    fake_interp_texts[2] = "Интерпретация выпавшей карты Маг"

    handlers.user_message_buffers[802] = ["Вытяни мне случайную карту"]
    await handlers.process_buffered_messages(802, bot, state)

    # 1 originating + 1 interpretation request
    assert len(captured_openai_payloads) == 2
    interp_request_msgs = captured_openai_payloads[1]["messages"]

    async with db_session() as session:
        p2_user_msgs = (await session.execute(
            select(DBMessage).where(
                DBMessage.user_id == 802,
                DBMessage.role == "user",
                DBMessage.content.like("[СИСТЕМА: Случайно выпала карта: %]")
            )
        )).scalars().all()
        assert len(p2_user_msgs) == 1
        p2_synthetic = p2_user_msgs[0].content
        p2_asst_msgs = (await session.execute(
            select(DBMessage).where(
                DBMessage.user_id == 802,
                DBMessage.role == "assistant",
                DBMessage.content == "Интерпретация выпавшей карты Маг"
            )
        )).scalars().all()
        assert len(p2_asst_msgs) == 1

    assert sum(1 for m in interp_request_msgs if m["content"] == p2_synthetic) == 1

    # ==========================================================================
    # PATH 3: process_user_prompt (Random Card)
    # ==========================================================================
    captured_openai_payloads.clear()
    fake_interp_texts.clear()
    fake_interp_texts[1] = "Вот случайная карта [RANDOM_IMG:tarot,1]"
    fake_interp_texts[2] = "Интерпретация карты Маг из промпта"

    msg_obj = SimpleNamespace(
        message_id=3001,
        chat=SimpleNamespace(id=803),
        from_user=SimpleNamespace(id=803),
        answer=AsyncMock(),
    )
    await handlers.process_user_prompt(msg_obj, 803, "Вытяни случайную карту снова", bot, state)

    # 1 originating + 1 interpretation request
    assert len(captured_openai_payloads) == 2
    prompt_interp_request_msgs = captured_openai_payloads[1]["messages"]

    async with db_session() as session:
        p3_user_msgs = (await session.execute(
            select(DBMessage).where(
                DBMessage.user_id == 803,
                DBMessage.role == "user",
                DBMessage.content.like("[СИСТЕМА: Случайно выпала карта: %]")
            )
        )).scalars().all()
        assert len(p3_user_msgs) == 1
        p3_synthetic = p3_user_msgs[0].content
        p3_asst_msgs = (await session.execute(
            select(DBMessage).where(
                DBMessage.user_id == 803,
                DBMessage.role == "assistant",
                DBMessage.content == "Интерпретация карты Маг из промпта"
            )
        )).scalars().all()
        assert len(p3_asst_msgs) == 1

    assert sum(1 for m in prompt_interp_request_msgs if m["content"] == p3_synthetic) == 1
