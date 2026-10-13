from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("BOT_TOKEN", "123456:TEST_BOT_TOKEN")

from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.enums import ChatType
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.methods import AnswerCallbackQuery, EditMessageReplyMarkup, EditMessageText, SendMessage
from aiogram.types import CallbackQuery, Chat, InlineKeyboardMarkup, Message, Update, User as TgUser
import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from database import AIConfig, AILog, Base, BotGeneralConfig, Message as DBMessage, SubscriptionConfig, User
import handlers
import keyboards as kb
from max_messenger_bot.api import MaxApiClient
from max_messenger_bot.app import MaxBotApplication
import max_messenger_bot.ai as max_ai
from max_messenger_bot.models import MAX_ID_OFFSET
from max_messenger_bot.services import common as max_common
from max_messenger_bot.storage import StorageBase
import ai_integration
from ai_integration import AIServiceError
from ai_request_singleflight import AI_BUSY_MESSAGE, single_flight
from ai_retry_policy import (
    NON_RETRYABLE_ERROR_CLASSIFICATIONS,
    RETRYABLE_ERROR_CLASSIFICATIONS,
    claim_retry_token,
    clear_retry_claims,
    is_retryable_ai_error,
    is_retryable_classification,
)


class ValidatingTelegramSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def close(self):
        return None

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        # Aiogram Pydantic validation of all outgoing methods
        type(method).model_validate(method.model_dump())
        if isinstance(method, SendMessage):
            return Message(
                message_id=len(self.calls),
                date=datetime.now(timezone.utc),
                chat=Chat(id=method.chat_id, type=ChatType.PRIVATE),
                text=method.text or "",
                reply_markup=method.reply_markup,
            ).as_(bot)
        if isinstance(method, EditMessageText):
            return Message(
                message_id=method.message_id or 10,
                date=datetime.now(timezone.utc),
                chat=Chat(id=method.chat_id or 999, type=ChatType.PRIVATE),
                text=method.text or "",
                reply_markup=method.reply_markup,
            ).as_(bot)
        if isinstance(method, EditMessageReplyMarkup):
            return Message(
                message_id=method.message_id or 10,
                date=datetime.now(timezone.utc),
                chat=Chat(id=method.chat_id or 999, type=ChatType.PRIVATE),
                text="",
                reply_markup=method.reply_markup,
            ).as_(bot)
        return True

    async def stream_content(self, *args, **kwargs):
        if False:
            yield b""


@pytest_asyncio.fixture
async def async_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(StorageBase.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(async_engine):
    return async_sessionmaker(async_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def clean_runtime():
    clear_retry_claims()
    handlers.user_message_buffers.clear()
    handlers.user_processing_tasks.clear()
    handlers._ai_button_claims.clear()
    handlers.user_message_buffer_leases.clear()
    handlers.user_scheduling_locks.clear()
    single_flight._active_leases.clear()
    yield
    clear_retry_claims()
    handlers.user_message_buffers.clear()
    handlers.user_processing_tasks.clear()
    handlers._ai_button_claims.clear()
    handlers.user_message_buffer_leases.clear()
    handlers.user_scheduling_locks.clear()
    single_flight._active_leases.clear()


@pytest_asyncio.fixture
async def setup_tg_user(session_factory, monkeypatch):
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(
                id=1,
                provider="Gemini",
                gemini_model="gemini-2.5-flash",
                gemini_api_key="test-gemini-key",
                fallback_provider="OpenAI",
                fallback_model="gpt-5.6-terra",
                openai_model="gpt-5.6-terra",
                openai_api_key="test-openai-key",
                allow_fallback=True,
            )
            s.add(cfg)
        else:
            cfg.provider = "Gemini"
            cfg.gemini_model = "gemini-2.5-flash"
            cfg.gemini_api_key = "test-gemini-key"
            cfg.fallback_provider = "OpenAI"
            cfg.fallback_model = "gpt-5.6-terra"
            cfg.openai_model = "gpt-5.6-terra"
            cfg.openai_api_key = "test-openai-key"
            cfg.allow_fallback = True

        sub_cfg = await s.get(SubscriptionConfig, 1)
        if not sub_cfg:
            s.add(SubscriptionConfig(id=1, subscriptions_enabled=False))
        else:
            sub_cfg.subscriptions_enabled = False

        bg_cfg = await s.get(BotGeneralConfig, 1)
        if not bg_cfg:
            s.add(BotGeneralConfig(id=1, profile_collect_name=False, profile_collect_gender=False, profile_collect_age=False))
        else:
            bg_cfg.profile_collect_name = False
            bg_cfg.profile_collect_gender = False
            bg_cfg.profile_collect_age = False

        user = await s.get(User, 1001)
        if not user:
            user = User(
                id=1001,
                first_name="Alice",
                name="Alice",
                gender="female",
                age=25,
                accepted_disclaimer=True,
                current_dialogue_id=1,
                current_topic_id=None,
            )
            s.add(user)
        else:
            user.name = "Alice"
            user.gender = "female"
            user.age = 25
            user.accepted_disclaimer = True
        await s.commit()

    monkeypatch.setattr(handlers, "async_session_maker", session_factory)
    monkeypatch.setattr(ai_integration, "async_session_maker", session_factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=False))
    return 1001


def _extract_prompt_from_call(*args, **kwargs) -> str:
    layout = kwargs.get("request_layout")
    if layout and hasattr(layout, "current_user_content"):
        return layout.current_user_content or ""
    if len(args) > 4 and isinstance(args[4], str):
        return args[4]
    return ""


# ---------------------------------------------------------------------------
# 1. Classification & Contract Tests
# ---------------------------------------------------------------------------

def test_ai_retry_ui_contract_classification():
    """Verify visible button '🔄 Повторить запрос' contract classification as 'external action'."""
    markup = kb.ai_retry_keyboard(12345, locale="ru")
    assert isinstance(markup, InlineKeyboardMarkup)
    assert len(markup.inline_keyboard) == 1
    assert len(markup.inline_keyboard[0]) == 1

    btn = markup.inline_keyboard[0][0]
    assert btn.text == "🔄 Повторить запрос"
    assert btn.callback_data == "ai_retry:12345"
    assert len(btn.callback_data.encode("utf-8")) <= 64

    # Contract classifier
    def classify_button_action(callback_data: str) -> str:
        if callback_data.startswith("ai_retry:"):
            return "external action"
        if callback_data.startswith("ai_btn:"):
            return "ai response button"
        return "navigation"

    assert classify_button_action(btn.callback_data) == "external action"


def test_retryable_classification_policy():
    """Verify retryable vs non-retryable error classification sets."""
    for c in RETRYABLE_ERROR_CLASSIFICATIONS:
        assert is_retryable_classification(c) is True

    for c in NON_RETRYABLE_ERROR_CLASSIFICATIONS:
        assert is_retryable_classification(c) is False

    err_retryable = AIServiceError("transient timeout")
    err_retryable.classification = "timeout"
    assert is_retryable_ai_error(err_retryable) is True

    err_non_retryable = AIServiceError("invalid api key")
    err_non_retryable.classification = "auth"
    assert is_retryable_ai_error(err_non_retryable) is False


# ---------------------------------------------------------------------------
# 2. Telegram Mandatory A → Я → A Journey
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_telegram_mandatory_a_to_ya_to_a_journey(session_factory, setup_tg_user, monkeypatch):
    """Real Telegram E2E journey through Dispatcher:
    User sends text Update -> drain runner -> primary fails -> fallback fails ->
    terminal overloaded message with visible '🔄 Повторить запрос' button ->
    Pydantic validation on outgoing Telegram methods ->
    extract button from rendered reply_markup -> send callback via Dispatcher ->
    exact original prompt preserved -> primary succeeds ->
    DB history has exactly 1 user turn + 1 assistant turn -> old button disabled.
    """
    user_id = setup_tg_user
    session = ValidatingTelegramSession()
    bot = Bot(token="123456:TEST_BOT_TOKEN", session=session)

    handlers.router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)

    primary_calls = []
    fallback_calls = []

    # Phase 1: both providers fail with transient errors
    async def mock_primary_fail(*args, **kwargs):
        prompt = _extract_prompt_from_call(*args, **kwargs)
        primary_calls.append(prompt)
        err = AIServiceError("Primary gateway 502")
        err.classification = "provider_5xx"
        raise err

    async def mock_fallback_fail(*args, **kwargs):
        prompt = _extract_prompt_from_call(*args, **kwargs)
        fallback_calls.append(prompt)
        err = AIServiceError("Fallback gateway 504")
        err.classification = "timeout"
        raise err

    monkeypatch.setattr(ai_integration, "_call_gemini_api", mock_primary_fail)
    monkeypatch.setattr(ai_integration, "_call_openai_api", mock_fallback_fail)

    test_prompt = "Как мне справиться с тревогой? 🌿\nПосоветуй упражнения."

    # Send initial user text update
    msg_update = Update(
        update_id=1,
        message=Message(
            message_id=10,
            date=datetime.now(timezone.utc),
            chat=Chat(id=user_id, type=ChatType.PRIVATE),
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            text=test_prompt,
        ).as_(bot),
    )
    await dp.feed_update(bot, msg_update)

    # Await background processing task
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    assert len(primary_calls) == 1
    assert len(fallback_calls) == 1

    # Inspect outgoing Telegram calls (validated by ValidatingTelegramSession)
    send_messages = [c for c in session.calls if isinstance(c, SendMessage)]
    assert send_messages
    failure_msg = send_messages[-1]
    assert "перегружена" in failure_msg.text
    assert failure_msg.reply_markup is not None

    # Verify visible retry button in rendered markup
    rendered_buttons = [b for row in failure_msg.reply_markup.inline_keyboard for b in row]
    retry_btn = next((b for b in rendered_buttons if "Повторить запрос" in b.text), None)
    assert retry_btn is not None
    assert retry_btn.text == "🔄 Повторить запрос"
    assert retry_btn.callback_data.startswith("ai_retry:")
    assert len(retry_btn.callback_data.encode("utf-8")) <= 64

    # Verify DB history before retry: NO user turn inserted!
    async with session_factory() as s:
        user_msgs = (await s.scalars(select(DBMessage).where(DBMessage.user_id == user_id))).all()
        assert len(user_msgs) == 0

        # Verify AILog contains failure group
        logs = (await s.scalars(select(AILog).where(AILog.user_id == user_id))).all()
        assert len(logs) == 2
        assert all(l.status == "error" for l in logs)
        failed_log = logs[-1]
        assert failed_log.prompt_summary == test_prompt

    # Phase 2: User clicks visible Retry button. Primary will now succeed!
    async def mock_primary_success(*args, **kwargs):
        prompt = _extract_prompt_from_call(*args, **kwargs)
        primary_calls.append(prompt)
        return "Вот три простых упражнения для заземления."

    monkeypatch.setattr(ai_integration, "_call_gemini_api", mock_primary_success)

    # Send callback through real Dispatcher using the ACTUAL rendered button
    cb_update = Update(
        update_id=2,
        callback_query=CallbackQuery(
            id="cb_retry_1",
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            chat_instance="ci_1",
            message=Message(
                message_id=11,
                date=datetime.now(timezone.utc),
                chat=Chat(id=user_id, type=ChatType.PRIVATE),
                text=failure_msg.text,
                reply_markup=failure_msg.reply_markup,
            ).as_(bot),
            data=retry_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, cb_update)

    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    # Verify primary provider was called on retry with EXACT original prompt
    assert len(primary_calls) == 2
    assert primary_calls[-1] == test_prompt

    # Verify final response delivered to user
    final_msgs = [c for c in session.calls if isinstance(c, SendMessage) and "упражнения для заземления" in (c.text or "")]
    assert len(final_msgs) == 1

    # Verify old button was disabled
    edit_markups = [c for c in session.calls if isinstance(c, EditMessageReplyMarkup)]
    assert edit_markups
    assert edit_markups[-1].reply_markup is None

    # Verify DB history after successful retry: EXACTLY 1 user turn, 1 assistant turn
    async with session_factory() as s:
        history = (await s.scalars(select(DBMessage).where(DBMessage.user_id == user_id).order_by(DBMessage.id))).all()
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == test_prompt
        assert history[1].role == "assistant"
        assert "упражнения для заземления" in history[1].content

        # Verify audit: two separate request groups
        all_logs = (await s.scalars(select(AILog).where(AILog.user_id == user_id).order_by(AILog.id))).all()
        assert len(all_logs) == 3
        success_logs = [l for l in all_logs if l.status == "success"]
        assert len(success_logs) == 1
        assert success_logs[0].request_group_id != all_logs[0].request_group_id


# ---------------------------------------------------------------------------
# 3. Telegram Fallback on Retry Journey
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_telegram_fallback_on_retry_journey(session_factory, setup_tg_user, monkeypatch):
    """First request: primary & fallback fail.
    Retry request: primary fails, fallback succeeds -> user sees fallback response."""
    user_id = setup_tg_user
    session = ValidatingTelegramSession()
    bot = Bot(token="123456:TEST_BOT_TOKEN", session=session)

    handlers.router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)

    # Initial failure
    monkeypatch.setattr(ai_integration, "_call_gemini_api", AsyncMock(side_effect=AIServiceError("P1 500")))
    monkeypatch.setattr(ai_integration, "_call_openai_api", AsyncMock(side_effect=AIServiceError("FB 500")))

    msg_update = Update(
        update_id=1,
        message=Message(
            message_id=10,
            date=datetime.now(timezone.utc),
            chat=Chat(id=user_id, type=ChatType.PRIVATE),
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            text="Тестовый запрос для фолбэка",
        ).as_(bot),
    )
    await dp.feed_update(bot, msg_update)
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    send_messages = [c for c in session.calls if isinstance(c, SendMessage)]
    failure_msg = send_messages[-1]
    retry_btn = failure_msg.reply_markup.inline_keyboard[0][0]

    # On retry: primary fails, fallback succeeds
    monkeypatch.setattr(ai_integration, "_call_gemini_api", AsyncMock(side_effect=AIServiceError("P1 500")))
    monkeypatch.setattr(ai_integration, "_call_openai_api", AsyncMock(return_value="Успешный ответ от запасного провайдера!"))

    cb_update = Update(
        update_id=2,
        callback_query=CallbackQuery(
            id="cb_retry_2",
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            chat_instance="ci_2",
            message=Message(
                message_id=11,
                date=datetime.now(timezone.utc),
                chat=Chat(id=user_id, type=ChatType.PRIVATE),
                text=failure_msg.text,
                reply_markup=failure_msg.reply_markup,
            ).as_(bot),
            data=retry_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, cb_update)
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    final_msgs = [c for c in session.calls if isinstance(c, SendMessage) and "запасного провайдера" in (c.text or "")]
    assert len(final_msgs) == 1

    async with session_factory() as s:
        history = (await s.scalars(select(DBMessage).where(DBMessage.user_id == user_id))).all()
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"
        assert "запасного провайдера" in history[1].content


# ---------------------------------------------------------------------------
# 4. Telegram Retry Fails Again Journey
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_telegram_retry_fails_again_journey(session_factory, setup_tg_user, monkeypatch):
    """Request A: fail -> button A.
    Click Button A: fails again -> button B.
    Button A cannot trigger another AI call; Button B can."""
    user_id = setup_tg_user
    session = ValidatingTelegramSession()
    bot = Bot(token="123456:TEST_BOT_TOKEN", session=session)

    handlers.router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)

    primary_mock = AsyncMock(side_effect=AIServiceError("Fail"))
    fallback_mock = AsyncMock(side_effect=AIServiceError("Fail"))
    monkeypatch.setattr(ai_integration, "_call_gemini_api", primary_mock)
    monkeypatch.setattr(ai_integration, "_call_openai_api", fallback_mock)

    # 1. Initial request fails
    msg_update = Update(
        update_id=1,
        message=Message(
            message_id=10,
            date=datetime.now(timezone.utc),
            chat=Chat(id=user_id, type=ChatType.PRIVATE),
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            text="Первый запрос",
        ).as_(bot),
    )
    await dp.feed_update(bot, msg_update)
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    send_messages = [c for c in session.calls if isinstance(c, SendMessage)]
    failure_msg_a = send_messages[-1]
    btn_a = failure_msg_a.reply_markup.inline_keyboard[0][0]

    # 2. Click button A -> fails again
    cb_update_a = Update(
        update_id=2,
        callback_query=CallbackQuery(
            id="cb_a",
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            chat_instance="ci",
            message=Message(
                message_id=11,
                date=datetime.now(timezone.utc),
                chat=Chat(id=user_id, type=ChatType.PRIVATE),
                text=failure_msg_a.text,
                reply_markup=failure_msg_a.reply_markup,
            ).as_(bot),
            data=btn_a.callback_data,
        ),
    )
    await dp.feed_update(bot, cb_update_a)
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    send_messages_after_retry = [c for c in session.calls if isinstance(c, SendMessage)]
    failure_msg_b = send_messages_after_retry[-1]
    btn_b = failure_msg_b.reply_markup.inline_keyboard[0][0]

    assert btn_a.callback_data != btn_b.callback_data

    # 3. Clicking Button A again -> rejected, 0 new AI calls
    call_count_before = primary_mock.call_count
    await dp.feed_update(bot, cb_update_a)
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]
    assert primary_mock.call_count == call_count_before

    # 4. Clicking Button B -> valid new retry starts!
    primary_mock.side_effect = None
    primary_mock.return_value = "Наконец-то успех!"
    cb_update_b = Update(
        update_id=3,
        callback_query=CallbackQuery(
            id="cb_b",
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            chat_instance="ci",
            message=Message(
                message_id=12,
                date=datetime.now(timezone.utc),
                chat=Chat(id=user_id, type=ChatType.PRIVATE),
                text=failure_msg_b.text,
                reply_markup=failure_msg_b.reply_markup,
            ).as_(bot),
            data=btn_b.callback_data,
        ),
    )
    await dp.feed_update(bot, cb_update_b)
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    assert primary_mock.call_count == call_count_before + 1


# ---------------------------------------------------------------------------
# 5. Stale Scope & Ownership Tests (Telegram)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_telegram_stale_scope_journey(session_factory, setup_tg_user, monkeypatch):
    """User starts new dialogue after failure -> old retry rejected with localized alert, 0 provider calls."""
    user_id = setup_tg_user
    session = ValidatingTelegramSession()
    bot = Bot(token="123456:TEST_BOT_TOKEN", session=session)

    handlers.router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)

    primary_mock = AsyncMock(side_effect=AIServiceError("Fail"))
    monkeypatch.setattr(ai_integration, "_call_gemini_api", primary_mock)
    monkeypatch.setattr(ai_integration, "_call_openai_api", AsyncMock(side_effect=AIServiceError("Fail")))

    msg_update = Update(
        update_id=1,
        message=Message(
            message_id=10,
            date=datetime.now(timezone.utc),
            chat=Chat(id=user_id, type=ChatType.PRIVATE),
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            text="Вопрос в диалоге 1",
        ).as_(bot),
    )
    await dp.feed_update(bot, msg_update)
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    failure_msg = [c for c in session.calls if isinstance(c, SendMessage)][-1]
    retry_btn = failure_msg.reply_markup.inline_keyboard[0][0]

    # User advances to Dialogue 2
    async with session_factory() as s:
        u = await s.get(User, user_id)
        u.current_dialogue_id = 2
        await s.commit()

    primary_mock.reset_mock()

    cb_update = Update(
        update_id=2,
        callback_query=CallbackQuery(
            id="cb_stale",
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            chat_instance="ci",
            message=Message(
                message_id=11,
                date=datetime.now(timezone.utc),
                chat=Chat(id=user_id, type=ChatType.PRIVATE),
                text=failure_msg.text,
                reply_markup=failure_msg.reply_markup,
            ).as_(bot),
            data=retry_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, cb_update)
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    # Verify zero provider calls!
    assert primary_mock.call_count == 0

    # Verify stale notification was answered
    answers = [c for c in session.calls if isinstance(c, AnswerCallbackQuery)]
    assert answers
    assert "предыдущему диалогу" in (answers[-1].text or "")


@pytest.mark.asyncio
async def test_telegram_forged_ownership_journey(session_factory, setup_tg_user, monkeypatch):
    """User B sends forged callback targeting User A's failure log -> rejected, zero provider calls."""
    user_a_id = setup_tg_user
    user_b_id = 2002

    async with session_factory() as s:
        user_b = User(id=user_b_id, first_name="Bob", name="Bob", accepted_disclaimer=True, current_dialogue_id=1)
        s.add(user_b)
        await s.commit()

    session = ValidatingTelegramSession()
    bot = Bot(token="123456:TEST_BOT_TOKEN", session=session)

    handlers.router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)

    primary_mock = AsyncMock(side_effect=AIServiceError("Fail"))
    monkeypatch.setattr(ai_integration, "_call_gemini_api", primary_mock)
    monkeypatch.setattr(ai_integration, "_call_openai_api", AsyncMock(side_effect=AIServiceError("Fail")))

    # User A triggers failure
    msg_update = Update(
        update_id=1,
        message=Message(
            message_id=10,
            date=datetime.now(timezone.utc),
            chat=Chat(id=user_a_id, type=ChatType.PRIVATE),
            from_user=TgUser(id=user_a_id, is_bot=False, first_name="Alice"),
            text="Секретный вопрос пользователя А",
        ).as_(bot),
    )
    await dp.feed_update(bot, msg_update)
    if user_a_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_a_id]

    failure_msg = [c for c in session.calls if isinstance(c, SendMessage)][-1]
    retry_btn = failure_msg.reply_markup.inline_keyboard[0][0]

    primary_mock.reset_mock()

    # User B sends callback with User A's retry button data
    cb_update_forged = Update(
        update_id=2,
        callback_query=CallbackQuery(
            id="cb_forged",
            from_user=TgUser(id=user_b_id, is_bot=False, first_name="Bob"),
            chat_instance="ci_b",
            message=Message(
                message_id=20,
                date=datetime.now(timezone.utc),
                chat=Chat(id=user_b_id, type=ChatType.PRIVATE),
                text="Some message",
            ).as_(bot),
            data=retry_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, cb_update_forged)
    if user_b_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_b_id]

    # Verify zero provider calls!
    assert primary_mock.call_count == 0


# ---------------------------------------------------------------------------
# 6. Concurrency & Double-Click Tests (Telegram)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_telegram_double_click_journey(session_factory, setup_tg_user, monkeypatch):
    """Near-simultaneous double click on retry button -> exactly 1 retry runs."""
    user_id = setup_tg_user
    session = ValidatingTelegramSession()
    bot = Bot(token="123456:TEST_BOT_TOKEN", session=session)

    handlers.router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)

    monkeypatch.setattr(ai_integration, "_call_gemini_api", AsyncMock(side_effect=AIServiceError("Fail")))
    monkeypatch.setattr(ai_integration, "_call_openai_api", AsyncMock(side_effect=AIServiceError("Fail")))

    msg_update = Update(
        update_id=1,
        message=Message(
            message_id=10,
            date=datetime.now(timezone.utc),
            chat=Chat(id=user_id, type=ChatType.PRIVATE),
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            text="Запрос перед двойным кликом",
        ).as_(bot),
    )
    await dp.feed_update(bot, msg_update)
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    failure_msg = [c for c in session.calls if isinstance(c, SendMessage)][-1]
    retry_btn = failure_msg.reply_markup.inline_keyboard[0][0]

    # Switch primary to successful
    primary_call_count = 0

    async def mock_success(*args, **kwargs):
        nonlocal primary_call_count
        primary_call_count += 1
        await asyncio.sleep(0.05)
        return "Ответ после одного выполнения."

    monkeypatch.setattr(ai_integration, "_call_gemini_api", mock_success)

    cb_1 = Update(
        update_id=2,
        callback_query=CallbackQuery(
            id="cb_double_1",
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            chat_instance="ci",
            message=Message(
                message_id=11,
                date=datetime.now(timezone.utc),
                chat=Chat(id=user_id, type=ChatType.PRIVATE),
                text=failure_msg.text,
                reply_markup=failure_msg.reply_markup,
            ).as_(bot),
            data=retry_btn.callback_data,
        ),
    )
    cb_2 = Update(
        update_id=3,
        callback_query=CallbackQuery(
            id="cb_double_2",
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            chat_instance="ci",
            message=Message(
                message_id=11,
                date=datetime.now(timezone.utc),
                chat=Chat(id=user_id, type=ChatType.PRIVATE),
                text=failure_msg.text,
                reply_markup=failure_msg.reply_markup,
            ).as_(bot),
            data=retry_btn.callback_data,
        ),
    )

    # Launch both concurrently
    await asyncio.gather(dp.feed_update(bot, cb_1), dp.feed_update(bot, cb_2))
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    # Exactly 1 primary generation run!
    assert primary_call_count == 1


# ---------------------------------------------------------------------------
# 7. Deterministic Non-Retryable Errors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_deterministic_errors_do_not_show_retry_button(session_factory, setup_tg_user, monkeypatch):
    """Configuration / Auth / deterministic errors must NOT show Retry button."""
    user_id = setup_tg_user
    session = ValidatingTelegramSession()
    bot = Bot(token="123456:TEST_BOT_TOKEN", session=session)

    handlers.router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)

    async def mock_auth_err(*args, **kwargs):
        err = AIServiceError("Invalid API key")
        err.classification = "auth"
        raise err

    monkeypatch.setattr(ai_integration, "_call_gemini_api", mock_auth_err)

    # Disable fallback so primary auth error is terminal
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        cfg.allow_fallback = False
        await s.commit()

    msg_update = Update(
        update_id=1,
        message=Message(
            message_id=10,
            date=datetime.now(timezone.utc),
            chat=Chat(id=user_id, type=ChatType.PRIVATE),
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            text="Запрос с ошибкой конфигурации",
        ).as_(bot),
    )
    await dp.feed_update(bot, msg_update)
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    send_messages = [c for c in session.calls if isinstance(c, SendMessage)]
    failure_msg = send_messages[-1]
    assert "перегружена" in failure_msg.text
    # No reply_markup rendered!
    assert failure_msg.reply_markup is None


# ---------------------------------------------------------------------------
# 8. MAX Mandatory A → Я → A Journey
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_mandatory_a_to_ya_to_a_journey(session_factory, monkeypatch):
    """Real MAX E2E journey through MaxBotApplication.handle_update():
    message_created -> run_ai_dialogue -> user Message persisted once ->
    primary & fallback fail -> terminal failure message with visible Retry attachment ->
    API boundary payload verified -> message_callback sent to handle_update() ->
    retry validation -> primary succeeds -> original Message NOT duplicated ->
    DB: exactly 1 user Message, 1 assistant Message -> old button disabled.
    """
    monkeypatch.setattr("database.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.legacy.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.storage.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.services.common.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.app.async_session_maker", session_factory)
    monkeypatch.setattr(max_ai, "async_session_maker", session_factory)

    max_user_id = 777 + MAX_ID_OFFSET
    chat_id = 777

    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(
                id=1,
                provider="Gemini",
                gemini_model="gemini-2.5-flash",
                gemini_api_key="test-key",
                fallback_provider="OpenAI",
                fallback_model="gpt-5.6-terra",
                openai_model="gpt-5.6-terra",
                openai_api_key="test-key",
                allow_fallback=True,
            )
            s.add(cfg)
        else:
            cfg.provider = "Gemini"
            cfg.gemini_model = "gemini-2.5-flash"
            cfg.gemini_api_key = "test-key"
            cfg.fallback_provider = "OpenAI"
            cfg.fallback_model = "gpt-5.6-terra"
            cfg.openai_model = "gpt-5.6-terra"
            cfg.openai_api_key = "test-key"
            cfg.allow_fallback = True

        sub_cfg = await s.get(SubscriptionConfig, 1)
        if not sub_cfg:
            s.add(SubscriptionConfig(id=1, subscriptions_enabled=False))
        else:
            sub_cfg.subscriptions_enabled = False

        user = await s.get(User, max_user_id)
        if not user:
            user = User(
                id=max_user_id,
                first_name="MaxUser",
                name="MaxUser",
                gender="female",
                age=25,
                accepted_disclaimer=True,
                current_dialogue_id=1,
                current_topic_id=None,
            )
            s.add(user)
        else:
            user.first_name = "MaxUser"
            user.name = "MaxUser"
            user.accepted_disclaimer = True
        await s.commit()

    monkeypatch.setattr(max_common, "ensure_access_before_chat", AsyncMock(return_value=True))
    monkeypatch.setattr(max_common, "maybe_require_disclaimer", AsyncMock(return_value=False))

    api_calls = []

    async def fake_request(method, path, *, params=None, json_data=None, expected_status=200):
        call_entry = {"method": method, "path": path, "params": params or {}, "body": json_data or {}}
        api_calls.append(call_entry)
        if path == "/messages":
            return {"message": {"body": {"mid": f"mid_{len(api_calls)}"}}}
        return {}

    client = MaxApiClient(token="test_token", base_url="https://max.test")
    client._request = fake_request
    app = MaxBotApplication(client=client)

    primary_calls = []
    fallback_calls = []

    # Phase 1: Failures
    async def mock_primary_fail(*args, **kwargs):
        prompt = _extract_prompt_from_call(*args, **kwargs)
        primary_calls.append(prompt)
        err = AIServiceError("P1 unavailable")
        err.classification = "provider_5xx"
        raise err

    async def mock_fallback_fail(*args, **kwargs):
        prompt = _extract_prompt_from_call(*args, **kwargs)
        fallback_calls.append(prompt)
        err = AIServiceError("FB timeout")
        err.classification = "timeout"
        raise err

    monkeypatch.setattr(max_ai, "_call_gemini", mock_primary_fail)
    monkeypatch.setattr(max_ai, "_call_openai", mock_fallback_fail)

    test_prompt = "Как научиться говорить «нет»?"

    # 1. Ordinary MAX user sends text message
    user_msg_update = {
        "update_id": "max_up_1",
        "update_type": "message_created",
        "message": {
            "mid": "mid_user_1",
            "chat_id": chat_id,
            "sender": {"user_id": 777, "first_name": "MaxUser", "name": "MaxUser"},
            "recipient": {"chat_id": chat_id},
            "text": test_prompt,
        },
    }

    await app.handle_update(user_msg_update)
    if max_user_id in app.user_tasks:
        await app.user_tasks[max_user_id]

    assert len(primary_calls) == 1
    assert len(fallback_calls) == 1

    # Verify MAX API calls at wire boundary
    failure_api_call = next(
        (c for c in reversed(api_calls) if "Сервис ИИ временно недоступен" in c["body"].get("text", "")),
        None,
    )
    assert failure_api_call is not None
    attachments = failure_api_call["body"].get("attachments", [])
    assert attachments
    rendered_btn = attachments[0]["payload"]["buttons"][0][0]
    assert rendered_btn["text"] == "🔄 Повторить запрос"
    assert rendered_btn["payload"].startswith("ai_retry:")

    # Verify DB state: user message was persisted ONCE, assistant is 0
    async with session_factory() as s:
        db_msgs = (await s.scalars(select(DBMessage).where(DBMessage.user_id == max_user_id))).all()
        assert len(db_msgs) == 1
        assert db_msgs[0].role == "user"
        assert db_msgs[0].content == test_prompt
        original_msg_id = db_msgs[0].id

    # Phase 2: User clicks visible Retry button
    async def mock_primary_success(*args, **kwargs):
        prompt = _extract_prompt_from_call(*args, **kwargs)
        primary_calls.append(prompt)
        return "Границы — это забота о себе."

    monkeypatch.setattr(max_ai, "_call_gemini", mock_primary_success)

    retry_update = {
        "update_id": "max_up_2",
        "update_type": "message_callback",
        "callback": {
            "callback_id": "cb_max_1",
            "chat_id": chat_id,
            "payload": rendered_btn["payload"],
            "sender": {"user_id": 777, "first_name": "MaxUser"},
            "message": {
                "mid": "mid_failure_msg",
                "chat_id": chat_id,
            },
        },
    }

    await app.handle_update(retry_update)
    if max_user_id in app.user_tasks:
        await app.user_tasks[max_user_id]

    # Verify primary provider was called on retry with prompt
    assert len(primary_calls) == 2
    assert test_prompt in primary_calls[-1]

    # Verify DB state after retry: original user message NOT DUPLICATED!
    async with session_factory() as s:
        history = (await s.scalars(select(DBMessage).where(DBMessage.user_id == max_user_id).order_by(DBMessage.id))).all()
        assert len(history) == 2
        assert history[0].id == original_msg_id
        assert history[0].role == "user"
        assert history[1].role == "assistant"
        assert "Границы" in history[1].content


# ---------------------------------------------------------------------------
# 9. MAX Stale & Double-Click Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_stale_scope_rejected(session_factory, monkeypatch):
    """User changes dialogue in MAX -> old retry callback rejected, zero provider calls."""
    monkeypatch.setattr("database.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.legacy.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.storage.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.services.common.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.app.async_session_maker", session_factory)
    monkeypatch.setattr(max_ai, "async_session_maker", session_factory)

    max_user_id = 778 + MAX_ID_OFFSET
    chat_id = 778

    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="Gemini", gemini_model="gemini-2.5-flash", gemini_api_key="k", allow_fallback=False)
            s.add(cfg)
        else:
            cfg.provider = "Gemini"
            cfg.gemini_model = "gemini-2.5-flash"
            cfg.gemini_api_key = "k"
            cfg.allow_fallback = False

        sub_cfg = await s.get(SubscriptionConfig, 1)
        if not sub_cfg:
            s.add(SubscriptionConfig(id=1, subscriptions_enabled=False))
        else:
            sub_cfg.subscriptions_enabled = False

        user = User(id=max_user_id, first_name="MaxUser2", name="MaxUser2", accepted_disclaimer=True, current_dialogue_id=1)
        s.add(user)
        msg = DBMessage(id=501, user_id=max_user_id, role="user", content="Вопрос в старом диалоге", dialogue_id=1)
        s.add(msg)
        await s.commit()

    api_calls = []

    async def fake_request(method, path, *, params=None, json_data=None, expected_status=200):
        api_calls.append({"method": method, "path": path, "body": json_data or {}})
        return {}

    client = MaxApiClient(token="test_token", base_url="https://max.test")
    client._request = fake_request
    app = MaxBotApplication(client=client)

    primary_mock = AsyncMock()
    monkeypatch.setattr(max_ai, "_call_gemini", primary_mock)

    # Change user dialogue to 2
    async with session_factory() as s:
        u = await s.get(User, max_user_id)
        u.current_dialogue_id = 2
        await s.commit()

    retry_update = {
        "update_id": "max_stale_1",
        "update_type": "message_callback",
        "callback": {
            "callback_id": "cb_stale_max",
            "chat_id": chat_id,
            "payload": "ai_retry:501:token123",
            "sender": {"user_id": 778, "first_name": "MaxUser2"},
            "message": {"mid": "mid_failure_msg", "chat_id": chat_id},
        },
    }

    await app.handle_update(retry_update)
    if max_user_id in app.user_tasks:
        await app.user_tasks[max_user_id]

    # Verify zero provider calls!
    assert primary_mock.call_count == 0


@pytest.mark.asyncio
async def test_max_double_click_concurrency(session_factory, monkeypatch):
    """Two concurrent message_callback updates for same retry in MAX -> exactly 1 runs."""
    monkeypatch.setattr("database.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.legacy.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.storage.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.services.common.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.app.async_session_maker", session_factory)
    monkeypatch.setattr(max_ai, "async_session_maker", session_factory)

    max_user_id = 779 + MAX_ID_OFFSET
    chat_id = 779

    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="Gemini", gemini_model="gemini-2.5-flash", gemini_api_key="k", allow_fallback=False)
            s.add(cfg)
        else:
            cfg.provider = "Gemini"
            cfg.gemini_model = "gemini-2.5-flash"
            cfg.gemini_api_key = "k"
            cfg.allow_fallback = False

        sub_cfg = await s.get(SubscriptionConfig, 1)
        if not sub_cfg:
            s.add(SubscriptionConfig(id=1, subscriptions_enabled=False))
        else:
            sub_cfg.subscriptions_enabled = False

        user = User(id=max_user_id, first_name="MaxUser3", name="MaxUser3", accepted_disclaimer=True, current_dialogue_id=1)
        s.add(user)
        msg = DBMessage(id=601, user_id=max_user_id, role="user", content="Вопрос MAX", dialogue_id=1)
        s.add(msg)
        await s.commit()

    monkeypatch.setattr(max_common, "ensure_access_before_chat", AsyncMock(return_value=True))

    client = MaxApiClient(token="test_token", base_url="https://max.test")
    client._request = AsyncMock(return_value={})
    app = MaxBotApplication(client=client)

    primary_call_count = 0

    async def mock_primary(*args, **kwargs):
        nonlocal primary_call_count
        primary_call_count += 1
        await asyncio.sleep(0.05)
        return "Успех"

    monkeypatch.setattr(max_ai, "_call_gemini", mock_primary)

    cb_update_1 = {
        "update_id": "max_double_1",
        "update_type": "message_callback",
        "callback": {
            "callback_id": "cb1",
            "chat_id": chat_id,
            "payload": "ai_retry:601:token_same",
            "sender": {"user_id": 779, "first_name": "MaxUser3"},
            "message": {"mid": "mid_1", "chat_id": chat_id},
        },
    }
    cb_update_2 = {
        "update_id": "max_double_2",
        "update_type": "message_callback",
        "callback": {
            "callback_id": "cb2",
            "chat_id": chat_id,
            "payload": "ai_retry:601:token_same",
            "sender": {"user_id": 779, "first_name": "MaxUser3"},
            "message": {"mid": "mid_1", "chat_id": chat_id},
        },
    }

    await asyncio.gather(app.handle_update(cb_update_1), app.handle_update(cb_update_2))
    if max_user_id in app.user_tasks:
        await app.user_tasks[max_user_id]

    assert primary_call_count == 1


# ---------------------------------------------------------------------------
# 10. Existing Normal Paths Unchanged
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_existing_successful_primary_path_unchanged(session_factory, setup_tg_user, monkeypatch):
    """When primary succeeds normally, response is delivered without retry markup."""
    user_id = setup_tg_user
    session = ValidatingTelegramSession()
    bot = Bot(token="123456:TEST_BOT_TOKEN", session=session)

    handlers.router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)

    monkeypatch.setattr(ai_integration, "_call_gemini_api", AsyncMock(return_value="Обычный успешный ответ."))

    msg_update = Update(
        update_id=1,
        message=Message(
            message_id=10,
            date=datetime.now(timezone.utc),
            chat=Chat(id=user_id, type=ChatType.PRIVATE),
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            text="Обычный запрос",
        ).as_(bot),
    )
    await dp.feed_update(bot, msg_update)
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    delivered = [c for c in session.calls if isinstance(c, SendMessage) and "Обычный успешный ответ" in (c.text or "")]
    assert len(delivered) == 1
    assert delivered[0].reply_markup is None
