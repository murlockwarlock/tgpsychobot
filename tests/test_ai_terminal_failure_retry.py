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

from database import AIConfig, AILog, Base, BotGeneralConfig, Message as DBMessage, SubscriptionConfig, Topic, User
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
    SCOPE_UNSET,
    claim_retry_token,
    clear_retry_claims,
    is_retryable_ai_error,
    is_retryable_classification,
    validate_terminal_failure_retry_eligibility,
)


def transient_provider_error(message: str = "Fail") -> AIServiceError:
    error = AIServiceError(message)
    error.classification = "provider_5xx"
    return error


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
    """Verify visible button '🔄 Повторить запрос' contract classification as 'external'."""
    from tests.test_admin_ux_hardening import _classify_telegram_ai_callback
    from tests.test_max_ai_settings_dispatcher_journey import _classify_max_callback
    from max_messenger_bot.keyboards import ai_retry_keyboard_max

    # Telegram contract
    markup = kb.ai_retry_keyboard(12345, locale="ru")
    assert isinstance(markup, InlineKeyboardMarkup)
    assert len(markup.inline_keyboard) == 1
    assert len(markup.inline_keyboard[0]) == 1

    btn = markup.inline_keyboard[0][0]
    assert btn.text == "🔄 Повторить запрос"
    assert btn.callback_data == "ai_retry:12345"
    assert len(btn.callback_data.encode("utf-8")) <= 64
    assert _classify_telegram_ai_callback(btn.callback_data) == "external"

    # MAX contract
    max_markup = ai_retry_keyboard_max(54321, failed_log_id=98765, locale="ru")
    assert max_markup
    max_btn = max_markup[0]["payload"]["buttons"][0][0]
    assert max_btn["text"] == "🔄 Повторить запрос"
    assert max_btn["payload"] == "ai_retry:54321:98765"
    assert _classify_max_callback(max_btn["payload"]) == "external"

    # Contract failure on unclassified action
    with pytest.raises(AssertionError, match="unclassified"):
        _classify_telegram_ai_callback("unknown_button_action_xyz")
    with pytest.raises(AssertionError, match="unclassified"):
        _classify_max_callback("unknown_button_action_xyz")


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

    primary_mock = AsyncMock(side_effect=transient_provider_error())
    fallback_mock = AsyncMock(side_effect=transient_provider_error())
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

    primary_mock = AsyncMock(side_effect=transient_provider_error())
    monkeypatch.setattr(ai_integration, "_call_gemini_api", primary_mock)
    monkeypatch.setattr(ai_integration, "_call_openai_api", AsyncMock(side_effect=transient_provider_error()))

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

    primary_mock = AsyncMock(side_effect=transient_provider_error())
    monkeypatch.setattr(ai_integration, "_call_gemini_api", primary_mock)
    monkeypatch.setattr(ai_integration, "_call_openai_api", AsyncMock(side_effect=transient_provider_error()))

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


@pytest.mark.asyncio
async def test_telegram_forged_non_retryable_ai_log_rejected(session_factory, setup_tg_user, monkeypatch):
    """User owns AILog with classification 'auth' or 'configuration' ->
    crafting ai_retry:<id> through Dispatcher -> rejected -> 0 provider calls -> 0 DB messages.
    """
    user_id = setup_tg_user
    session = ValidatingTelegramSession()
    bot = Bot(token="123456:TEST_BOT_TOKEN", session=session)

    handlers.router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)

    primary_mock = AsyncMock()
    monkeypatch.setattr(ai_integration, "_call_gemini_api", primary_mock)

    async with session_factory() as s:
        auth_log = AILog(
            id=8801,
            user_id=user_id,
            platform="telegram",
            request_type="chat",
            status="error",
            error_classification="auth",
            prompt_summary="Секретный вопрос auth",
            dialogue_id=1,
            topic_id=None,
            provider="gemini",
            model="gemini-2.5-flash",
            raw_response="",
        )
        conf_log = AILog(
            id=8802,
            user_id=user_id,
            platform="telegram",
            request_type="chat",
            status="error",
            error_classification="configuration",
            prompt_summary="Секретный вопрос conf",
            dialogue_id=1,
            topic_id=None,
            provider="gemini",
            model="gemini-2.5-flash",
            raw_response="",
        )
        s.add_all([auth_log, conf_log])
        await s.commit()

    # 1. Send forged callback for auth log
    cb_auth = Update(
        update_id=10,
        callback_query=CallbackQuery(
            id="cb_forged_auth",
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            chat_instance="ci_auth",
            message=Message(
                message_id=20,
                date=datetime.now(timezone.utc),
                chat=Chat(id=user_id, type=ChatType.PRIVATE),
                text="Ошибка",
            ).as_(bot),
            data="ai_retry:8801",
        ),
    )
    await dp.feed_update(bot, cb_auth)
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    # 2. Send forged callback for configuration log
    cb_conf = Update(
        update_id=11,
        callback_query=CallbackQuery(
            id="cb_forged_conf",
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            chat_instance="ci_conf",
            message=Message(
                message_id=21,
                date=datetime.now(timezone.utc),
                chat=Chat(id=user_id, type=ChatType.PRIVATE),
                text="Ошибка",
            ).as_(bot),
            data="ai_retry:8802",
        ),
    )
    await dp.feed_update(bot, cb_conf)
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    # Verify zero provider calls!
    assert primary_mock.call_count == 0

    # Verify zero DB messages created!
    async with session_factory() as s:
        db_msgs = (await s.scalars(select(DBMessage).where(DBMessage.user_id == user_id))).all()
        assert len(db_msgs) == 0


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

    monkeypatch.setattr(ai_integration, "_call_gemini_api", AsyncMock(side_effect=transient_provider_error()))
    monkeypatch.setattr(ai_integration, "_call_openai_api", AsyncMock(side_effect=transient_provider_error()))

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


@pytest.mark.asyncio
async def test_telegram_busy_then_retry_journey(session_factory, setup_tg_user, monkeypatch):
    """Initial request fails -> visible retry button -> hold single-flight lease ->
    click retry -> gets busy response -> zero provider calls -> button remains usable ->
    release lease -> click SAME button again -> retry runs -> delivers final answer -> old button disabled.
    """
    user_id = setup_tg_user
    session = ValidatingTelegramSession()
    bot = Bot(token="123456:TEST_BOT_TOKEN", session=session)

    handlers.router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)

    primary_calls = []
    fallback_calls = []

    async def mock_primary_fail(*args, **kwargs):
        prompt = _extract_prompt_from_call(*args, **kwargs)
        primary_calls.append(prompt)
        err = AIServiceError("P1 503")
        err.classification = "provider_5xx"
        raise err

    async def mock_fallback_fail(*args, **kwargs):
        prompt = _extract_prompt_from_call(*args, **kwargs)
        fallback_calls.append(prompt)
        err = AIServiceError("FB timeout")
        err.classification = "timeout"
        raise err

    monkeypatch.setattr(ai_integration, "_call_gemini_api", mock_primary_fail)
    monkeypatch.setattr(ai_integration, "_call_openai_api", mock_fallback_fail)

    test_prompt = "Как мне справиться со стрессом?"

    # Initial text request fails on both providers
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
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    assert len(primary_calls) == 1
    assert len(fallback_calls) == 1

    failure_msg = [c for c in session.calls if isinstance(c, SendMessage)][-1]
    rendered_buttons = [b for row in failure_msg.reply_markup.inline_keyboard for b in row]
    retry_btn = next((b for b in rendered_buttons if "Повторить запрос" in b.text), None)
    assert retry_btn is not None

    # Switch primary to success for when retry is attempted
    async def mock_primary_success(*args, **kwargs):
        prompt = _extract_prompt_from_call(*args, **kwargs)
        primary_calls.append(prompt)
        return "Дыхание 4-7-8 помогает снизить уровень кортизола."

    monkeypatch.setattr(ai_integration, "_call_gemini_api", mock_primary_success)

    # 1. Artificially hold Telegram single-flight lease
    busy_lease = single_flight.try_claim("telegram", user_id)
    assert busy_lease is not None

    # 2. Click actual rendered Retry button while lease is busy
    cb_busy = Update(
        update_id=2,
        callback_query=CallbackQuery(
            id="cb_busy_click",
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            chat_instance="ci_busy",
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
    await dp.feed_update(bot, cb_busy)
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    # Verify zero provider calls during busy click!
    assert len(primary_calls) == 1
    assert len(fallback_calls) == 1

    # Verify busy answer was returned to callback
    answers = [c for c in session.calls if isinstance(c, AnswerCallbackQuery)]
    assert answers
    assert answers[-1].show_alert is True

    # Verify keyboard was NOT disabled during busy
    disabled_edits = [c for c in session.calls if isinstance(c, EditMessageReplyMarkup)]
    assert len(disabled_edits) == 0

    # 3. Release busy lease
    single_flight.release(busy_lease)

    # 4. Click the SAME Retry button again through real Dispatcher
    cb_retry_success = Update(
        update_id=3,
        callback_query=CallbackQuery(
            id="cb_retry_success",
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            chat_instance="ci_retry",
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
    await dp.feed_update(bot, cb_retry_success)
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    # Verify retry ran exactly once with exact original prompt!
    assert len(primary_calls) == 2
    assert primary_calls[-1] == test_prompt

    # Verify final assistant answer was delivered to user
    send_messages = [c for c in session.calls if isinstance(c, SendMessage)]
    final_answer = send_messages[-1]
    assert "Дыхание 4-7-8" in final_answer.text

    # Verify old button keyboard was disabled
    disabled_edits = [c for c in session.calls if isinstance(c, EditMessageReplyMarkup)]
    assert len(disabled_edits) >= 1

    # Verify DB history: exactly 1 user turn, 1 assistant turn
    async with session_factory() as s:
        db_history = (await s.scalars(select(DBMessage).where(DBMessage.user_id == user_id).order_by(DBMessage.id))).all()
        assert len(db_history) == 2
        assert db_history[0].role == "user"
        assert db_history[0].content == test_prompt
        assert db_history[1].role == "assistant"
        assert "Дыхание 4-7-8" in db_history[1].content


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
    assert "технической проблемы" in failure_msg.text
    # No reply_markup rendered!
    assert failure_msg.reply_markup is None


@pytest.mark.asyncio
async def test_request_group_terminal_outcome_eligibility(session_factory):
    """Test effective terminal failure classification policy on request groups:
    - primary provider_5xx + fallback timeout => retry allowed
    - primary timeout + fallback auth => retry allowed
    - primary provider_5xx + fallback configuration => retry allowed
    - primary auth + fallback provider_5xx => retry allowed
    - any attempt success => retry NOT allowed
    - non-retryable isolated errors => retry NOT allowed
    """
    user_id = 12345

    async with session_factory() as s:
        u = User(id=user_id, first_name="Test", current_dialogue_id=1, current_topic_id=None)
        s.add(u)

        # 1. provider_5xx (attempt 1) -> timeout (attempt 2): terminal is timeout (retryable)
        g1_1 = AILog(id=101, user_id=user_id, platform="telegram", request_type="chat", status="error",
                     error_classification="provider_5xx", prompt_summary="Q1", dialogue_id=1,
                     request_group_id="grp_1", attempt_no=1, provider="p", model="m", raw_response="")
        g1_2 = AILog(id=102, user_id=user_id, platform="telegram", request_type="chat", status="error",
                     error_classification="timeout", prompt_summary="Q1", dialogue_id=1,
                     request_group_id="grp_1", attempt_no=2, provider="p", model="m", raw_response="")

        # 2. timeout (attempt 1) -> auth (attempt 2): transient primary keeps group retryable
        g2_1 = AILog(id=201, user_id=user_id, platform="telegram", request_type="chat", status="error",
                     error_classification="timeout", prompt_summary="Q2", dialogue_id=1,
                     request_group_id="grp_2", attempt_no=1, provider="p", model="m", raw_response="")
        g2_2 = AILog(id=202, user_id=user_id, platform="telegram", request_type="chat", status="error",
                     error_classification="auth", prompt_summary="Q2", dialogue_id=1,
                     request_group_id="grp_2", attempt_no=2, provider="p", model="m", raw_response="")

        # 3. provider_5xx (attempt 1) -> configuration (attempt 2): executed transient primary keeps group retryable
        g3_1 = AILog(id=301, user_id=user_id, platform="telegram", request_type="chat", status="error",
                     error_classification="provider_5xx", prompt_summary="Q3", dialogue_id=1,
                     request_group_id="grp_3", attempt_no=1, provider="p", model="m", raw_response="")
        g3_2 = AILog(id=302, user_id=user_id, platform="telegram", request_type="chat", status="error",
                     error_classification="configuration", prompt_summary="Q3", dialogue_id=1,
                     request_group_id="grp_3", attempt_no=2, provider="p", model="m", raw_response="")

        # 4. auth (attempt 1) -> provider_5xx (attempt 2): terminal is provider_5xx (retryable)
        g4_1 = AILog(id=401, user_id=user_id, platform="telegram", request_type="chat", status="error",
                     error_classification="auth", prompt_summary="Q4", dialogue_id=1,
                     request_group_id="grp_4", attempt_no=1, provider="p", model="m", raw_response="")
        g4_2 = AILog(id=402, user_id=user_id, platform="telegram", request_type="chat", status="error",
                     error_classification="provider_5xx", prompt_summary="Q4", dialogue_id=1,
                     request_group_id="grp_4", attempt_no=2, provider="p", model="m", raw_response="")

        # 5. attempt 1 error -> attempt 2 success: group succeeded
        g5_1 = AILog(id=501, user_id=user_id, platform="telegram", request_type="chat", status="error",
                     error_classification="timeout", prompt_summary="Q5", dialogue_id=1,
                     request_group_id="grp_5", attempt_no=1, provider="p", model="m", raw_response="")
        g5_2 = AILog(id=502, user_id=user_id, platform="telegram", request_type="chat", status="success",
                     prompt_summary="Q5", dialogue_id=1,
                     request_group_id="grp_5", attempt_no=2, provider="p", model="m", raw_response="ok")

        s.add_all([g1_1, g1_2, g2_1, g2_2, g3_1, g3_2, g4_1, g4_2, g5_1, g5_2])
        await s.commit()

        # Test group 1: retry allowed
        ok, reason, term = await validate_terminal_failure_retry_eligibility(s, "telegram", user_id, 101, expected_dialogue_id=1)
        assert ok is True
        assert term.id == 102

        # Test group 2: transient primary -> retry remains allowed
        ok, reason, term = await validate_terminal_failure_retry_eligibility(s, "telegram", user_id, 201, expected_dialogue_id=1)
        assert ok is True
        assert term.id == 202

        # Test group 3: executed transient primary -> retry remains allowed
        ok, reason, term = await validate_terminal_failure_retry_eligibility(s, "telegram", user_id, 301, expected_dialogue_id=1)
        assert ok is True
        assert term.id == 302

        # Test group 4: terminal provider_5xx -> retry allowed
        ok, reason, term = await validate_terminal_failure_retry_eligibility(s, "telegram", user_id, 401, expected_dialogue_id=1)
        assert ok is True
        assert term.id == 402

        # Test group 5: contains success -> rejected
        ok, reason, term = await validate_terminal_failure_retry_eligibility(s, "telegram", user_id, 501, expected_dialogue_id=1)
        assert ok is False
        assert reason == "group_already_succeeded"

        # Topic scope exact matching tests
        topic_log = AILog(id=601, user_id=user_id, platform="telegram", request_type="chat", status="error",
                          error_classification="provider_5xx", prompt_summary="Промпт в теме 123", dialogue_id=1, topic_id=123,
                          provider="p", model="m", raw_response="")
        s.add(topic_log)
        await s.commit()

        # When current topic is None -> rejected as stale_topic!
        ok, reason, term = await validate_terminal_failure_retry_eligibility(s, "telegram", user_id, 601, expected_dialogue_id=1, expected_topic_id=None)
        assert ok is False
        assert reason == "stale_topic"

        # When current topic is 456 -> rejected as stale_topic!
        ok, reason, term = await validate_terminal_failure_retry_eligibility(s, "telegram", user_id, 601, expected_dialogue_id=1, expected_topic_id=456)
        assert ok is False
        assert reason == "stale_topic"

        # When current topic matches 123 -> accepted!
        ok, reason, term = await validate_terminal_failure_retry_eligibility(s, "telegram", user_id, 601, expected_dialogue_id=1, expected_topic_id=123)
        assert ok is True

        # When expected_topic_id is SCOPE_UNSET (omitted) -> topic not checked, accepted!
        ok, reason, term = await validate_terminal_failure_retry_eligibility(s, "telegram", user_id, 601, expected_dialogue_id=1)
        assert ok is True


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
        (c for c in reversed(api_calls) if "перегружена" in c["body"].get("text", "")),
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

    # Verify real MAX API boundary calls:
    # 1. Edit call that removed callback attachments:
    remove_button_call = next(
        (c for c in api_calls if c["method"] == "PUT" and c["path"] == "/messages" and c["params"].get("message_id") == "mid_failure_msg" and c["body"].get("attachments") == []),
        None,
    )
    assert remove_button_call is not None
    assert remove_button_call["body"]["attachments"] == []

    # 2. Final successful assistant delivery at wire boundary:
    delivery_call = next(
        (c for c in reversed(api_calls) if "Границы — это забота о себе." in c["body"].get("text", "")),
        None,
    )
    assert delivery_call is not None
    assert delivery_call["path"] == "/messages"
    assert delivery_call["method"] in {"PUT", "POST"}
    assert delivery_call["body"]["format"] == "html"
    assert "Границы — это забота о себе." in delivery_call["body"]["text"]

    # Verify DB state after retry: original user message NOT DUPLICATED!
    async with session_factory() as s:
        history = (await s.scalars(select(DBMessage).where(DBMessage.user_id == max_user_id).order_by(DBMessage.id))).all()
        assert len(history) == 2
        assert history[0].id == original_msg_id
        assert history[0].role == "user"
        assert history[1].role == "assistant"
        assert "Границы" in history[1].content

    # Resend old retry callback: prove zero new provider calls and zero duplicate assistant responses!
    await app.handle_update(retry_update)
    if max_user_id in app.user_tasks:
        await app.user_tasks[max_user_id]

    assert len(primary_calls) == 2
    async with session_factory() as s:
        history = (await s.scalars(select(DBMessage).where(DBMessage.user_id == max_user_id).order_by(DBMessage.id))).all()
        assert len(history) == 2


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
        failed_log = AILog(
            id=5011,
            user_id=max_user_id,
            platform="max",
            request_type="chat",
            status="error",
            error_classification="timeout",
            prompt_summary="Вопрос в старом диалоге",
            dialogue_id=1,
            provider="gemini",
            model="gemini-2.5-flash",
            raw_response="",
            diagnostics_json=json.dumps({"user_message_id": 501}),
        )
        s.add(failed_log)
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
            "payload": "ai_retry:501:5011",
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
        failed_log = AILog(
            id=6011,
            user_id=max_user_id,
            platform="max",
            request_type="chat",
            status="error",
            error_classification="timeout",
            prompt_summary="Вопрос MAX",
            dialogue_id=1,
            provider="gemini",
            model="gemini-2.5-flash",
            raw_response="",
            diagnostics_json=json.dumps({"user_message_id": 601}),
        )
        s.add(failed_log)
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
            "payload": "ai_retry:601:6011",
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
            "payload": "ai_retry:601:6011",
            "sender": {"user_id": 779, "first_name": "MaxUser3"},
            "message": {"mid": "mid_1", "chat_id": chat_id},
        },
    }

    await asyncio.gather(app.handle_update(cb_update_1), app.handle_update(cb_update_2))
    if max_user_id in app.user_tasks:
        await app.user_tasks[max_user_id]

    assert primary_call_count == 1


@pytest.mark.asyncio
async def test_max_forged_ordinary_message_callback_rejected(session_factory, monkeypatch):
    """User owns ordinary DBMessage #X that never failed -> crafted message_callback:
    ai_retry:<message_id>:<whatever> -> handle_update() -> rejected -> 0 provider calls -> 0 assistant messages.
    """
    monkeypatch.setattr("database.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.legacy.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.storage.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.services.common.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.app.async_session_maker", session_factory)
    monkeypatch.setattr(max_ai, "async_session_maker", session_factory)

    max_user_id = 888 + MAX_ID_OFFSET
    chat_id = 888

    async with session_factory() as s:
        cfg = AIConfig(id=1, provider="Gemini", gemini_model="gemini-2.5-flash", gemini_api_key="k", allow_fallback=False)
        s.add(cfg)
        sub_cfg = SubscriptionConfig(id=1, subscriptions_enabled=False)
        s.add(sub_cfg)
        user = User(id=max_user_id, first_name="MaxUser4", name="MaxUser4", accepted_disclaimer=True, current_dialogue_id=1)
        s.add(user)
        # Ordinary message that never failed:
        msg = DBMessage(id=701, user_id=max_user_id, role="user", content="Обычное сообщение без ошибок", dialogue_id=1)
        s.add(msg)
        await s.commit()

    monkeypatch.setattr(max_common, "ensure_access_before_chat", AsyncMock(return_value=True))

    client = MaxApiClient(token="test_token", base_url="https://max.test")
    client._request = AsyncMock(return_value={})
    app = MaxBotApplication(client=client)

    primary_mock = AsyncMock()
    monkeypatch.setattr(max_ai, "_call_gemini", primary_mock)

    # 1. Craft with arbitrary random token (unauthenticated token attack)
    cb_update_token = {
        "update_id": "max_forged_1",
        "update_type": "message_callback",
        "callback": {
            "callback_id": "cb_forged_token",
            "chat_id": chat_id,
            "payload": "ai_retry:701:random_token",
            "sender": {"user_id": 888, "first_name": "MaxUser4"},
            "message": {"mid": "mid_701", "chat_id": chat_id},
        },
    }
    await app.handle_update(cb_update_token)
    if max_user_id in app.user_tasks:
        await app.user_tasks[max_user_id]

    # 2. Craft with non-existent AILog ID
    cb_update_nonexistent = {
        "update_id": "max_forged_2",
        "update_type": "message_callback",
        "callback": {
            "callback_id": "cb_forged_nonexistent",
            "chat_id": chat_id,
            "payload": "ai_retry:701:99999",
            "sender": {"user_id": 888, "first_name": "MaxUser4"},
            "message": {"mid": "mid_701", "chat_id": chat_id},
        },
    }
    await app.handle_update(cb_update_nonexistent)
    if max_user_id in app.user_tasks:
        await app.user_tasks[max_user_id]

    # Verify zero provider calls!
    assert primary_mock.call_count == 0

    # Verify zero assistant messages in DB!
    async with session_factory() as s:
        msgs = (await s.scalars(select(DBMessage).where(DBMessage.user_id == max_user_id))).all()
        assert len(msgs) == 1
        assert msgs[0].role == "user"


@pytest.mark.asyncio
async def test_max_forged_non_retryable_failed_log_rejected(session_factory, monkeypatch):
    """User owns DBMessage #X and there is failed auth/config AILog ->
    crafted retry -> rejected -> 0 provider calls -> 0 assistant messages.
    """
    monkeypatch.setattr("database.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.legacy.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.storage.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.services.common.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.app.async_session_maker", session_factory)
    monkeypatch.setattr(max_ai, "async_session_maker", session_factory)

    max_user_id = 889 + MAX_ID_OFFSET
    chat_id = 889

    async with session_factory() as s:
        cfg = AIConfig(id=1, provider="Gemini", gemini_model="gemini-2.5-flash", gemini_api_key="k", allow_fallback=False)
        s.add(cfg)
        sub_cfg = SubscriptionConfig(id=1, subscriptions_enabled=False)
        s.add(sub_cfg)
        user = User(id=max_user_id, first_name="MaxUser5", name="MaxUser5", accepted_disclaimer=True, current_dialogue_id=1)
        s.add(user)
        msg = DBMessage(id=801, user_id=max_user_id, role="user", content="Вопрос с детерминированной ошибкой", dialogue_id=1)
        s.add(msg)
        auth_log = AILog(
            id=8011,
            user_id=max_user_id,
            platform="max",
            request_type="chat",
            status="error",
            error_classification="auth",
            prompt_summary="Вопрос с детерминированной ошибкой",
            dialogue_id=1,
            provider="gemini",
            model="gemini-2.5-flash",
            raw_response="",
        )
        conf_log = AILog(
            id=8012,
            user_id=max_user_id,
            platform="max",
            request_type="chat",
            status="error",
            error_classification="configuration",
            prompt_summary="Вопрос с детерминированной ошибкой",
            dialogue_id=1,
            provider="gemini",
            model="gemini-2.5-flash",
            raw_response="",
        )
        s.add_all([auth_log, conf_log])
        await s.commit()

    monkeypatch.setattr(max_common, "ensure_access_before_chat", AsyncMock(return_value=True))

    client = MaxApiClient(token="test_token", base_url="https://max.test")
    client._request = AsyncMock(return_value={})
    app = MaxBotApplication(client=client)

    primary_mock = AsyncMock()
    monkeypatch.setattr(max_ai, "_call_gemini", primary_mock)

    cb_auth = {
        "update_id": "max_forged_auth",
        "update_type": "message_callback",
        "callback": {
            "callback_id": "cb_auth",
            "chat_id": chat_id,
            "payload": "ai_retry:801:8011",
            "sender": {"user_id": 889, "first_name": "MaxUser5"},
            "message": {"mid": "mid_801", "chat_id": chat_id},
        },
    }
    await app.handle_update(cb_auth)
    if max_user_id in app.user_tasks:
        await app.user_tasks[max_user_id]

    cb_conf = {
        "update_id": "max_forged_conf",
        "update_type": "message_callback",
        "callback": {
            "callback_id": "cb_conf",
            "chat_id": chat_id,
            "payload": "ai_retry:801:8012",
            "sender": {"user_id": 889, "first_name": "MaxUser5"},
            "message": {"mid": "mid_801", "chat_id": chat_id},
        },
    }
    await app.handle_update(cb_conf)
    if max_user_id in app.user_tasks:
        await app.user_tasks[max_user_id]

    assert primary_mock.call_count == 0

    async with session_factory() as s:
        msgs = (await s.scalars(select(DBMessage).where(DBMessage.user_id == max_user_id))).all()
        assert len(msgs) == 1


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


# ---------------------------------------------------------------------------
# 14. Telegram Stale Topic Journey (Real Dispatcher route)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_telegram_terminal_failure_retry_stale_topic_journey(setup_tg_user, session_factory, monkeypatch):
    """Real Telegram stale topic journey:
    1. User is in dialogue 1 / topic 123.
    2. Sends normal text through Dispatcher -> primary fails -> fallback fails -> real terminal message -> real visible Retry button.
    3. User changes to: dialogue 1 / topic None.
    4. Press OLD Retry button through Dispatcher -> callback rejected as stale -> zero primary provider calls -> zero fallback provider calls -> no DB user/assistant retry turn -> stale localized response -> old button disabled.
    5. Also test: topic 123 -> topic 456 -> old Retry rejected.
    """
    user_id = setup_tg_user
    session = ValidatingTelegramSession()
    bot = Bot(token="123456:TEST_BOT_TOKEN", session=session)

    handlers.router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)

    async with session_factory() as s:
        topic123 = await s.get(Topic, 123)
        if not topic123:
            s.add(Topic(id=123, name="Тема 123", is_active=True))
        topic456 = await s.get(Topic, 456)
        if not topic456:
            s.add(Topic(id=456, name="Тема 456", is_active=True))
        user = await s.get(User, user_id)
        user.current_dialogue_id = 1
        user.current_topic_id = 123
        await s.commit()

    primary_calls = []
    fallback_calls = []

    async def mock_primary_fail(*args, **kwargs):
        prompt = _extract_prompt_from_call(*args, **kwargs)
        primary_calls.append(prompt)
        err = AIServiceError("P1 500")
        err.classification = "provider_5xx"
        raise err

    async def mock_fallback_fail(*args, **kwargs):
        prompt = _extract_prompt_from_call(*args, **kwargs)
        fallback_calls.append(prompt)
        err = AIServiceError("FB timeout")
        err.classification = "timeout"
        raise err

    monkeypatch.setattr(ai_integration, "_call_gemini_api", mock_primary_fail)
    monkeypatch.setattr(ai_integration, "_call_openai_api", mock_fallback_fail)

    test_prompt = "Как победить бессонницу в теме 123?"

    # 1. User sends message in topic 123
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
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    assert len(primary_calls) == 1
    assert len(fallback_calls) == 1

    # Verify failure message with retry button was delivered
    send_messages = [c for c in session.calls if isinstance(c, SendMessage)]
    failure_msg = send_messages[-1]
    assert failure_msg.reply_markup is not None
    retry_btn = failure_msg.reply_markup.inline_keyboard[0][0]
    assert retry_btn.text == "🔄 Повторить запрос"
    assert retry_btn.callback_data.startswith("ai_retry:")

    # Verify DB AILog was recorded with topic_id=123
    async with session_factory() as s:
        ailog = (await s.scalars(select(AILog).where(AILog.user_id == user_id, AILog.status == "error"))).first()
        assert ailog is not None
        assert ailog.topic_id == 123
        failed_log_id = ailog.id

    # 2. User changes topic to None (returns to main dialogue)
    async with session_factory() as s:
        user = await s.get(User, user_id)
        user.current_topic_id = None
        await s.commit()

    # 3. Press OLD Retry button through Dispatcher
    cb_stale = Update(
        update_id=2,
        callback_query=CallbackQuery(
            id="cb_stale_click",
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            chat_instance="ci_stale",
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
    await dp.feed_update(bot, cb_stale)
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    # Assertions for topic 123 -> None:
    # Zero new provider calls!
    assert len(primary_calls) == 1
    assert len(fallback_calls) == 1

    # Stale localized alert returned
    answers = [c for c in session.calls if isinstance(c, AnswerCallbackQuery)]
    assert answers
    assert answers[-1].show_alert is True
    assert "предыдущему диалогу" in answers[-1].text

    # Old button keyboard disabled
    disabled_edits = [c for c in session.calls if isinstance(c, EditMessageReplyMarkup)]
    assert len(disabled_edits) >= 1

    # No DB assistant turn created
    async with session_factory() as s:
        asst_msgs = (await s.scalars(select(DBMessage).where(DBMessage.user_id == user_id, DBMessage.role == "assistant"))).all()
        assert len(asst_msgs) == 0

    # 4. Now test topic 123 -> topic 456
    async with session_factory() as s:
        user = await s.get(User, user_id)
        user.current_topic_id = 456
        await s.commit()

    cb_stale_456 = Update(
        update_id=3,
        callback_query=CallbackQuery(
            id="cb_stale_456_click",
            from_user=TgUser(id=user_id, is_bot=False, first_name="Alice"),
            chat_instance="ci_stale_456",
            message=Message(
                message_id=11,
                date=datetime.now(timezone.utc),
                chat=Chat(id=user_id, type=ChatType.PRIVATE),
                text=failure_msg.text,
                reply_markup=failure_msg.reply_markup,
            ).as_(bot),
            data=f"ai_retry:{failed_log_id}",
        ),
    )
    await dp.feed_update(bot, cb_stale_456)
    if user_id in handlers.user_processing_tasks:
        await handlers.user_processing_tasks[user_id]

    # Zero new provider calls!
    assert len(primary_calls) == 1
    assert len(fallback_calls) == 1
    answers = [c for c in session.calls if isinstance(c, AnswerCallbackQuery)]
    assert "предыдущему диалогу" in answers[-1].text


# ---------------------------------------------------------------------------
# 15. MAX Stale Topic Journey (Real message_callback route)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_terminal_failure_retry_stale_topic_journey(session_factory, monkeypatch):
    """Real MAX stale topic journey:
    1. User is in dialogue 1 / topic 123.
    2. message_created in dialogue 1 / topic 123 -> terminal failure -> real Retry attachment.
    3. User current topic changes to None.
    4. Send actual message_callback through MaxBotApplication.handle_update() -> rejected -> zero provider calls -> no assistant DB row -> old attachment removed -> localized stale notification.
    5. Also test topic 123 -> 456.
    """
    monkeypatch.setattr("database.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.legacy.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.storage.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.services.common.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.app.async_session_maker", session_factory)
    monkeypatch.setattr(max_ai, "async_session_maker", session_factory)

    max_user_id = 888 + MAX_ID_OFFSET
    chat_id = 888

    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="Gemini", gemini_model="gemini-2.5-flash", gemini_api_key="key",
                           fallback_provider="OpenAI", fallback_model="gpt-5.6-terra", openai_model="gpt-5.6-terra", openai_api_key="key", allow_fallback=True)
            s.add(cfg)
        else:
            cfg.provider = "Gemini"
            cfg.fallback_provider = "OpenAI"
            cfg.allow_fallback = True

        sub_cfg = await s.get(SubscriptionConfig, 1)
        if not sub_cfg:
            s.add(SubscriptionConfig(id=1, subscriptions_enabled=False))
        else:
            sub_cfg.subscriptions_enabled = False

        topic123 = await s.get(Topic, 123)
        if not topic123:
            s.add(Topic(id=123, name="Тема 123", is_active=True))
        topic456 = await s.get(Topic, 456)
        if not topic456:
            s.add(Topic(id=456, name="Тема 456", is_active=True))

        user = await s.get(User, max_user_id)
        if not user:
            user = User(id=max_user_id, first_name="MaxAlice", name="MaxAlice", gender="female", age=25,
                        accepted_disclaimer=True, current_dialogue_id=1, current_topic_id=123)
            s.add(user)
        else:
            user.current_dialogue_id = 1
            user.current_topic_id = 123
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

    async def mock_primary_fail(*args, **kwargs):
        prompt = _extract_prompt_from_call(*args, **kwargs)
        primary_calls.append(prompt)
        err = AIServiceError("P1 500")
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

    test_prompt = "Как справиться с тревогой в теме 123?"

    # 1. User sends message in topic 123
    msg_update = {
        "update_id": "up_topic_1",
        "update_type": "message_created",
        "message": {
            "mid": "mid_u_123",
            "chat_id": chat_id,
            "sender": {"user_id": 888, "first_name": "MaxAlice", "name": "MaxAlice"},
            "recipient": {"chat_id": chat_id},
            "text": test_prompt,
        },
    }
    await app.handle_update(msg_update)
    if max_user_id in app.user_tasks:
        await app.user_tasks[max_user_id]

    assert len(primary_calls) == 1
    assert len(fallback_calls) == 1

    failure_call = next(
        (c for c in reversed(api_calls) if "перегружена" in c["body"].get("text", "")),
        None,
    )
    assert failure_call is not None
    rendered_btn = failure_call["body"]["attachments"][0]["payload"]["buttons"][0][0]
    callback_payload = rendered_btn["payload"]
    assert callback_payload.startswith("ai_retry:")

    # 2. User current topic changes to None
    async with session_factory() as s:
        user = await s.get(User, max_user_id)
        user.current_topic_id = None
        await s.commit()

    # 3. Send actual message_callback through MaxBotApplication.handle_update()
    cb_update_none = {
        "update_id": "up_cb_none",
        "update_type": "message_callback",
        "callback": {
            "callback_id": "cb_id_none",
            "payload": callback_payload,
            "user": {"user_id": 888, "first_name": "MaxAlice", "name": "MaxAlice"},
            "message": {"mid": "mid_fail_msg", "chat_id": chat_id},
        },
    }
    await app.handle_update(cb_update_none)
    if max_user_id in app.user_tasks:
        await app.user_tasks[max_user_id]

    # Assert: zero new provider calls
    assert len(primary_calls) == 1
    assert len(fallback_calls) == 1

    # Assert: old attachment removed
    edit_call = next(
        (c for c in reversed(api_calls) if c["method"] == "PUT" and c["body"].get("attachments") == []),
        None,
    )
    assert edit_call is not None

    # Assert: answer_callback called with notification ui.ai.retry_stale
    ans_call = next(
        (c for c in reversed(api_calls) if "/answers" in c["path"] and "предыдущему диалогу" in c["body"].get("notification", "")),
        None,
    )
    assert ans_call is not None

    # Assert: no assistant DB row
    async with session_factory() as s:
        asst = (await s.scalars(select(DBMessage).where(DBMessage.user_id == max_user_id, DBMessage.role == "assistant"))).all()
        assert len(asst) == 0

    # 4. Also test topic 123 -> 456
    async with session_factory() as s:
        user = await s.get(User, max_user_id)
        user.current_topic_id = 456
        await s.commit()

    cb_update_456 = {
        "update_id": "up_cb_456",
        "update_type": "message_callback",
        "callback": {
            "callback_id": "cb_id_456",
            "payload": callback_payload,
            "user": {"user_id": 888, "first_name": "MaxAlice", "name": "MaxAlice"},
            "message": {"mid": "mid_fail_msg", "chat_id": chat_id},
        },
    }
    await app.handle_update(cb_update_456)
    if max_user_id in app.user_tasks:
        await app.user_tasks[max_user_id]

    assert len(primary_calls) == 1
    assert len(fallback_calls) == 1


# ---------------------------------------------------------------------------
# 16. MAX Forged / Malformed Binding Validation Matrix
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_terminal_failure_retry_mandatory_binding_validation_matrix(session_factory, monkeypatch):
    """MAX forged/malformed binding tests:
    A. DBMessage exists, AILog is retryable, diagnostics_json = NULL -> rejected
    B. diagnostics_json = malformed JSON -> rejected
    C. diagnostics_json = {} -> rejected
    D. diagnostics_json.user_message_id points to another message -> rejected
    E. diagnostics_json.user_message_id is malformed non-integer -> rejected
    F. valid exact binding -> accepted
    All through MaxBotApplication.handle_update() with realistic message_callback.
    """
    monkeypatch.setattr("database.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.legacy.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.storage.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.services.common.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.app.async_session_maker", session_factory)
    monkeypatch.setattr(max_ai, "async_session_maker", session_factory)

    max_user_id = 770 + MAX_ID_OFFSET
    chat_id = 770

    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="Gemini", gemini_model="gemini-2.5-flash", gemini_api_key="key")
            s.add(cfg)
        sub_cfg = await s.get(SubscriptionConfig, 1)
        if not sub_cfg:
            s.add(SubscriptionConfig(id=1, subscriptions_enabled=False))
        else:
            sub_cfg.subscriptions_enabled = False
        user = await s.get(User, max_user_id)
        if not user:
            user = User(id=max_user_id, first_name="MaxBindingUser", current_dialogue_id=1, current_topic_id=None, accepted_disclaimer=True)
            s.add(user)
        else:
            user.current_dialogue_id = 1
            user.current_topic_id = None
        # User message id = 500
        msg_500 = await s.get(DBMessage, 500)
        if not msg_500:
            msg_500 = DBMessage(id=500, user_id=max_user_id, role="user", content="Мне очень трудно", dialogue_id=1, topic_id=None)
            s.add(msg_500)
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

    provider_calls = []
    async def mock_gemini(*args, **kwargs):
        provider_calls.append(args)
        return "Всё наладится."
    monkeypatch.setattr(max_ai, "_call_gemini", mock_gemini)

    async def make_failed_log(log_id: int, diagnostics_val: str | None):
        async with session_factory() as s:
            existing = await s.get(AILog, log_id)
            if existing:
                await s.delete(existing)
            log_entry = AILog(
                id=log_id,
                user_id=max_user_id,
                platform="max",
                request_type="chat",
                provider="Gemini",
                model="gemini-2.5-flash",
                raw_response="",
                status="error",
                error_classification="provider_5xx",
                prompt_summary="Мне очень трудно",
                dialogue_id=1,
                topic_id=None,
                diagnostics_json=diagnostics_val,
            )
            s.add(log_entry)
            await s.commit()

    async def send_cb(cb_id: str, payload_str: str):
        update = {
            "update_id": f"up_{cb_id}",
            "update_type": "message_callback",
            "callback": {
                "callback_id": cb_id,
                "payload": payload_str,
                "user": {"user_id": 770, "first_name": "MaxBindingUser"},
                "message": {"mid": "mid_fail", "chat_id": chat_id},
            },
        }
        await app.handle_update(update)
        if max_user_id in app.user_tasks:
            await app.user_tasks[max_user_id]

    # Test A: diagnostics_json = None (NULL) -> rejected
    await make_failed_log(801, None)
    await send_cb("cb_a", "ai_retry:500:801")
    assert len(provider_calls) == 0

    # Test B: diagnostics_json = malformed JSON -> rejected
    await make_failed_log(802, "bad_json{")
    await send_cb("cb_b", "ai_retry:500:802")
    assert len(provider_calls) == 0

    # Test C: diagnostics_json = {} -> rejected
    await make_failed_log(803, "{}")
    await send_cb("cb_c", "ai_retry:500:803")
    assert len(provider_calls) == 0

    # Test D: diagnostics_json.user_message_id points to another message (e.g. 99999) -> rejected
    await make_failed_log(804, json.dumps({"user_message_id": 99999}))
    await send_cb("cb_d", "ai_retry:500:804")
    assert len(provider_calls) == 0

    # Test E: diagnostics_json.user_message_id wrong type / malformed -> rejected
    await make_failed_log(805, json.dumps({"user_message_id": "not_digits"}))
    await send_cb("cb_e", "ai_retry:500:805")
    assert len(provider_calls) == 0

    # Test F: valid exact binding -> accepted
    await make_failed_log(806, json.dumps({"user_message_id": 500}))
    await send_cb("cb_f", "ai_retry:500:806")
    assert len(provider_calls) == 1


# ---------------------------------------------------------------------------
# 17. Identical Prompt Collision is Mandatory
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_identical_prompt_collision_identity_authorization(session_factory, monkeypatch):
    """Requirement 7: Identical prompt collision is mandatory.
    Create:
    Message A: id=101 text="Мне тревожно"
    Message B: id=102 text="Мне тревожно"
    Same: user, dialogue, topic, text.
    Failed AILog is durably bound to: user_message_id=101.
    Craft: ai_retry:102:<failed_log_id_for_101>
    Expected: REJECT, zero provider calls.
    Then: ai_retry:101:<failed_log_id_for_101>
    Expected: ACCEPT, exactly 1 provider call.
    Proves authorization is based on identity, not text coincidence.
    """
    monkeypatch.setattr("database.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.legacy.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.storage.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.services.common.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.app.async_session_maker", session_factory)
    monkeypatch.setattr(max_ai, "async_session_maker", session_factory)

    max_user_id = 660 + MAX_ID_OFFSET
    chat_id = 660

    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="Gemini", gemini_model="gemini-2.5-flash", gemini_api_key="key")
            s.add(cfg)
        sub_cfg = await s.get(SubscriptionConfig, 1)
        if not sub_cfg:
            s.add(SubscriptionConfig(id=1, subscriptions_enabled=False))
        else:
            sub_cfg.subscriptions_enabled = False
        user = await s.get(User, max_user_id)
        if not user:
            user = User(id=max_user_id, first_name="CollUser", current_dialogue_id=1, current_topic_id=None, accepted_disclaimer=True)
            s.add(user)
        else:
            user.current_dialogue_id = 1
            user.current_topic_id = None

        msg_a = await s.get(DBMessage, 101)
        if not msg_a:
            msg_a = DBMessage(id=101, user_id=max_user_id, role="user", content="Мне тревожно", dialogue_id=1, topic_id=None)
            s.add(msg_a)
        msg_b = await s.get(DBMessage, 102)
        if not msg_b:
            msg_b = DBMessage(id=102, user_id=max_user_id, role="user", content="Мне тревожно", dialogue_id=1, topic_id=None)
            s.add(msg_b)

        failed_log = await s.get(AILog, 701)
        if not failed_log:
            failed_log = AILog(
                id=701,
                user_id=max_user_id,
                platform="max",
                request_type="chat",
                provider="Gemini",
                model="gemini-2.5-flash",
                raw_response="",
                status="error",
                error_classification="timeout",
                prompt_summary="Мне тревожно",
                dialogue_id=1,
                topic_id=None,
                diagnostics_json=json.dumps({"user_message_id": 101}),
            )
            s.add(failed_log)
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

    provider_calls = []
    async def mock_gemini(*args, **kwargs):
        provider_calls.append(args)
        return "Дышите глубже, тревога отступит."
    monkeypatch.setattr(max_ai, "_call_gemini", mock_gemini)

    # 1. Craft callback referencing Message B (102) with Log 701 (which is bound to 101)
    cb_crafted_wrong = {
        "update_id": "up_col_1",
        "update_type": "message_callback",
        "callback": {
            "callback_id": "cb_col_wrong",
            "payload": "ai_retry:102:701",
            "user": {"user_id": 660, "first_name": "CollUser"},
            "message": {"mid": "mid_fail", "chat_id": chat_id},
        },
    }
    await app.handle_update(cb_crafted_wrong)
    if max_user_id in app.user_tasks:
        await app.user_tasks[max_user_id]

    # REJECTED: zero provider calls!
    assert len(provider_calls) == 0

    # Old attachment removed via edit_message
    edit_call = next(
        (c for c in reversed(api_calls) if c["method"] == "PUT" and c["body"].get("attachments") == []),
        None,
    )
    assert edit_call is not None

    # 2. Send correct callback referencing Message A (101)
    cb_correct = {
        "update_id": "up_col_2",
        "update_type": "message_callback",
        "callback": {
            "callback_id": "cb_col_correct",
            "payload": "ai_retry:101:701",
            "user": {"user_id": 660, "first_name": "CollUser"},
            "message": {"mid": "mid_fail", "chat_id": chat_id},
        },
    }
    await app.handle_update(cb_correct)
    if max_user_id in app.user_tasks:
        await app.user_tasks[max_user_id]

    # ACCEPTED: provider called exactly once!
    assert len(provider_calls) == 1

    # Assistant message saved in DB
    async with session_factory() as s:
        asst = (await s.scalars(select(DBMessage).where(DBMessage.user_id == max_user_id, DBMessage.role == "assistant"))).all()
        assert len(asst) == 1
        assert "Дышите глубже" in asst[0].content


# ---------------------------------------------------------------------------
# 10. MAX Fallback Configuration Parity Tests (PR #64 Blocker Fix)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_fallback_missing_api_key_configuration_outcome(session_factory, monkeypatch):
    """Test A: MAX missing fallback API key produces terminal configuration failure.
    Real journey: message_created -> MaxBotApplication.handle_update() ->
    primary provider fails transiently -> fallback enabled, provider/model configured ->
    fallback API key missing -> fallback provider function NOT called ->
    terminal failure message emitted -> NO Retry attachment on MAX API wire.
    AILog audit: 2 attempts (primary error, fallback configuration error).
    """
    monkeypatch.setattr("database.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.legacy.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.storage.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.services.common.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.app.async_session_maker", session_factory)
    monkeypatch.setattr(max_ai, "async_session_maker", session_factory)

    max_user_id = 801 + MAX_ID_OFFSET
    chat_id = 801

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
                openai_api_key="",  # Missing fallback API key!
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
            cfg.openai_api_key = ""  # Missing fallback API key!
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
                first_name="MaxUserA",
                name="MaxUserA",
                accepted_disclaimer=True,
                current_dialogue_id=1,
                current_topic_id=None,
            )
            s.add(user)
        else:
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

    async def mock_primary_fail(*args, **kwargs):
        prompt = _extract_prompt_from_call(*args, **kwargs)
        primary_calls.append(prompt)
        err = AIServiceError("P1 503 Service Unavailable")
        err.classification = "provider_5xx"
        raise err

    async def mock_fallback(*args, **kwargs):
        prompt = _extract_prompt_from_call(*args, **kwargs)
        fallback_calls.append(prompt)
        return "Should never be called"

    monkeypatch.setattr(max_ai, "_call_gemini", mock_primary_fail)
    monkeypatch.setattr(max_ai, "_call_openai", mock_fallback)

    user_msg_update = {
        "update_id": "max_up_fb_missing_key",
        "update_type": "message_created",
        "message": {
            "mid": "mid_user_fb_1",
            "chat_id": chat_id,
            "sender": {"user_id": 801, "first_name": "MaxUserA"},
            "recipient": {"chat_id": chat_id},
            "text": "Вопрос при отсутствии ключа фолбэка",
        },
    }

    await app.handle_update(user_msg_update)
    if max_user_id in app.user_tasks:
        await app.user_tasks[max_user_id]

    # Primary called once, fallback provider NEVER called
    assert len(primary_calls) == 1
    assert len(fallback_calls) == 0

    # A transient primary attempt keeps the group retryable despite missing fallback credentials.
    failure_api_call = next(
        (c for c in reversed(api_calls) if "перегружена" in c["body"].get("text", "")),
        None,
    )
    assert failure_api_call is not None
    attachments = failure_api_call["body"].get("attachments", [])
    assert any("ai_retry" in json.dumps(att) for att in attachments)

    # DB AILog verification
    async with session_factory() as s:
        logs = (await s.scalars(
            select(AILog).where(AILog.user_id == max_user_id).order_by(AILog.attempt_no.asc())
        )).all()
        assert len(logs) == 2

        # Attempt 1: primary error preserved
        log_1 = logs[0]
        assert log_1.attempt_no == 1
        assert log_1.attempt_role == "primary"
        assert log_1.status == "error"
        assert log_1.error_classification == "provider_5xx"
        assert log_1.provider.lower() == "gemini"

        # Attempt 2: fallback configuration error
        log_2 = logs[1]
        assert log_2.attempt_no == 2
        assert log_2.attempt_role == "fallback"
        assert log_2.status == "error"
        assert log_2.error_classification == "configuration"
        assert "API key for fallback AI provider 'OpenAI' is not configured" in log_2.error_message
        assert log_2.provider == "OpenAI"
        assert log_2.model == "gpt-5.6-terra"
        assert log_2.request_group_id == log_1.request_group_id
        assert log_2.diagnostics_json is not None
        diag_data = json.loads(log_2.diagnostics_json)
        assert "user_message_id" in diag_data


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_field", ["provider", "model"])
async def test_max_fallback_incomplete_provider_model_configuration_outcome(session_factory, monkeypatch, missing_field):
    """Test B: MAX incomplete fallback provider or model produces terminal configuration failure.
    allow_fallback=True, but fallback_provider or fallback_model is missing.
    Primary fails transiently (timeout). Fallback provider NOT called.
    Terminal failure message emitted with NO Retry attachment.
    AILog audit records attempt 1 primary error and attempt 2 fallback configuration error.
    """
    monkeypatch.setattr("database.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.legacy.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.storage.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.services.common.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.app.async_session_maker", session_factory)
    monkeypatch.setattr(max_ai, "async_session_maker", session_factory)

    user_seed = 810 if missing_field == "provider" else 820
    max_user_id = user_seed + MAX_ID_OFFSET
    chat_id = user_seed

    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        fb_prov = None if missing_field == "provider" else "OpenAI"
        fb_mod = None if missing_field == "model" else "gpt-5.6-terra"
        if not cfg:
            cfg = AIConfig(
                id=1,
                provider="Gemini",
                gemini_model="gemini-2.5-flash",
                gemini_api_key="test-key",
                fallback_provider=fb_prov,
                fallback_model=fb_mod,
                openai_model="gpt-5.6-terra",
                openai_api_key="test-openai-key",
                allow_fallback=True,
            )
            s.add(cfg)
        else:
            cfg.provider = "Gemini"
            cfg.gemini_model = "gemini-2.5-flash"
            cfg.gemini_api_key = "test-key"
            cfg.fallback_provider = fb_prov
            cfg.fallback_model = fb_mod
            cfg.openai_model = "gpt-5.6-terra"
            cfg.openai_api_key = "test-openai-key"
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
                first_name=f"MaxUserB_{missing_field}",
                name=f"MaxUserB_{missing_field}",
                accepted_disclaimer=True,
                current_dialogue_id=1,
                current_topic_id=None,
            )
            s.add(user)
        else:
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

    async def mock_primary_fail(*args, **kwargs):
        prompt = _extract_prompt_from_call(*args, **kwargs)
        primary_calls.append(prompt)
        err = AIServiceError("P1 timeout")
        err.classification = "timeout"
        raise err

    async def mock_fallback(*args, **kwargs):
        prompt = _extract_prompt_from_call(*args, **kwargs)
        fallback_calls.append(prompt)
        return "Should never be called"

    monkeypatch.setattr(max_ai, "_call_gemini", mock_primary_fail)
    monkeypatch.setattr(max_ai, "_call_openai", mock_fallback)

    user_msg_update = {
        "update_id": f"max_up_fb_incomplete_{missing_field}",
        "update_type": "message_created",
        "message": {
            "mid": f"mid_user_fb_b_{missing_field}",
            "chat_id": chat_id,
            "sender": {"user_id": user_seed, "first_name": "MaxUserB"},
            "recipient": {"chat_id": chat_id},
            "text": f"Вопрос при неполной конфигурации ({missing_field})",
        },
    }

    await app.handle_update(user_msg_update)
    if max_user_id in app.user_tasks:
        await app.user_tasks[max_user_id]

    assert len(primary_calls) == 1
    assert len(fallback_calls) == 0

    failure_api_call = next(
        (c for c in reversed(api_calls) if "перегружена" in c["body"].get("text", "")),
        None,
    )
    assert failure_api_call is not None
    attachments = failure_api_call["body"].get("attachments", [])
    assert any("ai_retry" in json.dumps(att) for att in attachments)

    async with session_factory() as s:
        logs = (await s.scalars(
            select(AILog).where(AILog.user_id == max_user_id).order_by(AILog.attempt_no.asc())
        )).all()
        assert len(logs) == 2

        log_1 = logs[0]
        assert log_1.attempt_no == 1
        assert log_1.attempt_role == "primary"
        assert log_1.status == "error"
        assert log_1.error_classification == "timeout"

        log_2 = logs[1]
        assert log_2.attempt_no == 2
        assert log_2.attempt_role == "fallback"
        assert log_2.status == "error"
        assert log_2.error_classification == "configuration"
        assert "Fallback AI provider configuration is incomplete" in log_2.error_message
        assert log_2.request_group_id == log_1.request_group_id
        assert log_2.diagnostics_json is not None
        diag_data = json.loads(log_2.diagnostics_json)
        assert "user_message_id" in diag_data


@pytest.mark.asyncio
async def test_max_fallback_transient_both_fail_still_retryable(session_factory, monkeypatch):
    """Test C: Regression guard — primary timeout + fallback timeout is retryable.
    Real journey: message_created -> MaxBotApplication.handle_update() ->
    primary timeout -> fallback timeout -> terminal failure message WITH visible Retry attachment.
    AILog audit records both attempts, and attempt 2 retains retryable classification and diagnostics.
    """
    monkeypatch.setattr("database.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.legacy.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.storage.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.services.common.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.app.async_session_maker", session_factory)
    monkeypatch.setattr(max_ai, "async_session_maker", session_factory)

    max_user_id = 830 + MAX_ID_OFFSET
    chat_id = 830

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
                openai_api_key="test-key-valid",
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
            cfg.openai_api_key = "test-key-valid"
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
                first_name="MaxUserC",
                name="MaxUserC",
                accepted_disclaimer=True,
                current_dialogue_id=1,
                current_topic_id=None,
            )
            s.add(user)
        else:
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

    async def mock_primary_fail(*args, **kwargs):
        prompt = _extract_prompt_from_call(*args, **kwargs)
        primary_calls.append(prompt)
        err = AIServiceError("Primary timeout")
        err.classification = "timeout"
        raise err

    async def mock_fallback_fail(*args, **kwargs):
        prompt = _extract_prompt_from_call(*args, **kwargs)
        fallback_calls.append(prompt)
        err = AIServiceError("Fallback timeout")
        err.classification = "timeout"
        raise err

    monkeypatch.setattr(max_ai, "_call_gemini", mock_primary_fail)
    monkeypatch.setattr(max_ai, "_call_openai", mock_fallback_fail)

    user_msg_update = {
        "update_id": "max_up_fb_transient_both",
        "update_type": "message_created",
        "message": {
            "mid": "mid_user_fb_c",
            "chat_id": chat_id,
            "sender": {"user_id": 830, "first_name": "MaxUserC"},
            "recipient": {"chat_id": chat_id},
            "text": "Вопрос при таймауте обоих провайдеров",
        },
    }

    await app.handle_update(user_msg_update)
    if max_user_id in app.user_tasks:
        await app.user_tasks[max_user_id]

    # Both primary and fallback were called
    assert len(primary_calls) == 1
    assert len(fallback_calls) == 1

    # Wire check: Retry button IS present
    failure_api_call = next(
        (c for c in reversed(api_calls) if "перегружена" in c["body"].get("text", "")),
        None,
    )
    assert failure_api_call is not None
    attachments = failure_api_call["body"].get("attachments", [])
    assert attachments
    rendered_btn = attachments[0]["payload"]["buttons"][0][0]
    assert rendered_btn["text"] == "🔄 Повторить запрос"
    assert rendered_btn["payload"].startswith("ai_retry:")

    # DB AILog check
    async with session_factory() as s:
        logs = (await s.scalars(
            select(AILog).where(AILog.user_id == max_user_id).order_by(AILog.attempt_no.asc())
        )).all()
        assert len(logs) == 2

        log_1 = logs[0]
        assert log_1.attempt_no == 1
        assert log_1.attempt_role == "primary"
        assert log_1.status == "error"
        assert log_1.error_classification == "timeout"

        log_2 = logs[1]
        assert log_2.attempt_no == 2
        assert log_2.attempt_role == "fallback"
        assert log_2.status == "error"
        assert log_2.error_classification == "timeout"
        assert log_2.request_group_id == log_1.request_group_id
        assert log_2.diagnostics_json is not None
        diag_data = json.loads(log_2.diagnostics_json)
        assert "user_message_id" in diag_data
