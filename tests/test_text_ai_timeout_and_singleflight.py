import asyncio
import os
import sys

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ai_request_singleflight import single_flight, AI_BUSY_MESSAGE, UserAISingleFlight
import ai_integration
from ai_integration import AIServiceError
import database
from database import Base, AIConfig, User, Topic, Message as DBMessage, SubscriptionConfig
import handlers
from max_messenger_bot.app import MaxBotApplication
from max_messenger_bot import ai as max_ai
from max_messenger_bot.storage import StorageBase
import max_messenger_bot.storage as max_storage
import max_messenger_bot.app as max_app_module
import max_messenger_bot.services.common as max_common
from max_messenger_bot.models import IncomingMessage, Sender


@pytest.fixture(autouse=True)
def clean_single_flight():
    single_flight.clear()
    yield
    single_flight.clear()


from sqlalchemy.pool import NullPool

@pytest_asyncio.fixture
async def db_session(tmp_path, monkeypatch):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'test_singleflight.db'}",
        poolclass=NullPool,
    )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(StorageBase.metadata.create_all)

    # Seed AIConfig and SubscriptionConfig
    async with sessions() as session:
        session.add(AIConfig(
            id=1,
            provider="Deepseek",
            deepseek_model="deepseek-chat",
            deepseek_api_key="ds-key",
            allow_fallback=True,
            fallback_provider="OpenAI",
            fallback_model="gpt-5.6-terra",
            openai_api_key="oai-key",
            fallback_timeout=0.05,
        ))
        session.add(Topic(id=1, name="Topic 1", is_active=True, start_button_payload="Action Payload"))
        # Seed test users
        session.add(User(id=999, username="user999", first_name="User 999", name="User 999", accepted_disclaimer=True))
        session.add(User(id=555, username="user555", first_name="User 555"))
        session.add(User(id=666, username="user666", first_name="User 666"))
        session.add(User(id=777, username="user777", first_name="User 777"))
        session.add(User(id=444, username="user444", first_name="User 444"))
        session.add(User(id=333, username="user333", first_name="User 333", current_topic_id=1))
        await session.commit()

    monkeypatch.setattr(database, "async_session_maker", sessions)
    monkeypatch.setattr(handlers, "async_session_maker", sessions)
    monkeypatch.setattr(ai_integration, "async_session_maker", sessions)
    monkeypatch.setattr(max_ai, "async_session_maker", sessions)
    monkeypatch.setattr(max_storage, "async_session_maker", sessions)
    monkeypatch.setattr(max_app_module, "async_session_maker", sessions)
    monkeypatch.setattr(max_common, "async_session_maker", sessions)

    handlers.user_message_buffers.clear()
    handlers.user_isolated_turn_queues.clear()
    handlers.user_processing_tasks.clear()
    handlers.user_scheduling_locks.clear()

    try:
        yield sessions
    finally:
        handlers.user_message_buffers.clear()
        handlers.user_isolated_turn_queues.clear()
        handlers.user_processing_tasks.clear()
        handlers.user_scheduling_locks.clear()
        await engine.dispose()


# ============================================================================
# 1. UserAISingleFlight Unit Tests
# ============================================================================

def test_single_flight_basic_flow():
    lease1 = single_flight.try_claim("telegram", 123)
    assert lease1 is not None
    assert lease1.platform == "telegram"
    assert lease1.user_id == "123"
    assert single_flight.is_busy("telegram", 123)

    # Second claim for same user & platform is rejected
    lease2 = single_flight.try_claim("telegram", 123)
    assert lease2 is None

    # Idempotent release
    assert single_flight.release(lease1) is True
    assert single_flight.release(lease1) is False
    assert not single_flight.is_busy("telegram", 123)

    # Can claim again after release
    lease3 = single_flight.try_claim("telegram", 123)
    assert lease3 is not None
    single_flight.release(lease3)


def test_single_flight_platform_and_user_isolation():
    lease_tg = single_flight.try_claim("telegram", 100)
    lease_max = single_flight.try_claim("max", 100)
    lease_tg2 = single_flight.try_claim("telegram", 200)

    assert lease_tg is not None
    assert lease_max is not None
    assert lease_tg2 is not None

    single_flight.release(lease_tg)
    single_flight.release(lease_max)
    single_flight.release(lease_tg2)


# ============================================================================
# 2. Telegram Text AI Hard Wall-Clock Timeout Tests
# ============================================================================

@pytest.mark.asyncio
async def test_telegram_ai_hard_timeout_primary_cancels_and_triggers_fallback(db_session, monkeypatch):
    """
    When primary provider hangs, asyncio.wait_for actively cancels it within fallback_timeout
    and triggers fallback provider.
    """
    primary_cancelled = False

    async def hanging_deepseek(*args, **kwargs):
        nonlocal primary_cancelled
        try:
            await asyncio.sleep(10.0)
            return "DeepSeek Finished"
        except asyncio.CancelledError:
            primary_cancelled = True
            raise

    fallback_called = False

    async def successful_openai(*args, **kwargs):
        nonlocal fallback_called
        fallback_called = True
        return "OpenAI Fallback Response"

    monkeypatch.setattr(ai_integration, "_call_deepseek_api", hanging_deepseek)
    monkeypatch.setattr(ai_integration, "_call_openai_api", successful_openai)
    monkeypatch.setattr(ai_integration, "record_ai_attempt_log", AsyncMock())

    start = asyncio.get_event_loop().time()
    response = await ai_integration.generate_response(
        user_id=999,
        user_prompt="hello",
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert response == "OpenAI Fallback Response"
    assert primary_cancelled is True
    assert fallback_called is True
    assert elapsed < 1.0  # Finished within milliseconds, did NOT wait 10s


@pytest.mark.asyncio
async def test_telegram_ai_hard_timeout_both_providers_fail(db_session, monkeypatch):
    """
    When both primary and fallback hang, both time out cleanly and AIServiceError
    with classification='timeout' is raised.
    """
    async def hanging_deepseek(*args, **kwargs):
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            raise

    async def hanging_openai(*args, **kwargs):
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr(ai_integration, "_call_deepseek_api", hanging_deepseek)
    monkeypatch.setattr(ai_integration, "_call_openai_api", hanging_openai)
    monkeypatch.setattr(ai_integration, "record_ai_attempt_log", AsyncMock())

    start = asyncio.get_event_loop().time()
    with pytest.raises(AIServiceError) as exc_info:
        await ai_integration.generate_response(
            user_id=999,
            user_prompt="hello",
        )
    elapsed = asyncio.get_event_loop().time() - start

    assert exc_info.value.classification == "timeout"
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_telegram_ai_cancelled_error_propagates_cleanly(db_session, monkeypatch):
    """
    External asyncio.CancelledError must be re-raised immediately and NOT
    swallowed or treated as fallback.
    """
    async def cancelling_provider(*args, **kwargs):
        raise asyncio.CancelledError("External cancel")

    monkeypatch.setattr(ai_integration, "_call_deepseek_api", cancelling_provider)
    fallback_mock = AsyncMock()
    monkeypatch.setattr(ai_integration, "_call_openai_api", fallback_mock)
    monkeypatch.setattr(ai_integration, "record_ai_attempt_log", AsyncMock())

    with pytest.raises(asyncio.CancelledError):
        await ai_integration.generate_response(
            user_id=999,
            user_prompt="hello",
        )

    # Fallback must NOT have been called on outer CancelledError
    assert fallback_mock.call_count == 0


# ============================================================================
# 3. MAX Hard Wall-Clock Timeout Tests
# ============================================================================

@pytest.mark.asyncio
async def test_max_dispatch_provider_timeout(monkeypatch):
    """
    max_messenger_bot.ai._dispatch_provider wraps the provider invocation in
    asyncio.wait_for(..., timeout=timeout). If it exceeds timeout, it must raise
    AIServiceError with classification='timeout' and cancel the underlying coroutine.
    """
    provider_cancelled = False

    async def hanging_call(*args, **kwargs):
        nonlocal provider_cancelled
        try:
            await asyncio.sleep(10.0)
            return "Never"
        except asyncio.CancelledError:
            provider_cancelled = True
            raise

    monkeypatch.setattr(max_ai, "_call_deepseek", hanging_call)

    ai_config = MagicMock()
    ai_config.provider = "deepseek"
    ai_config.deepseek_api_key = "test-key"
    ai_config.deepseek_model = "deepseek-chat"
    ai_config.fallback_timeout = 0.05

    start = asyncio.get_event_loop().time()
    with pytest.raises(max_ai.AIServiceError) as exc_info:
        await max_ai._dispatch_provider(
            ai_config=ai_config,
            request_layout="system prompt",
            messages=[],
        )
    elapsed = asyncio.get_event_loop().time() - start

    assert exc_info.value.classification == "timeout"
    assert provider_cancelled is True
    assert elapsed < 1.0


# ============================================================================
# 4. MAX Single-Flight Admission & Queue Bypass Tests
# ============================================================================

class MockMaxClient:
    def __init__(self):
        self.sent_messages = []

    async def send_message(self, chat_id: int, text: str, attachments=None):
        self.sent_messages.append((chat_id, text))
        return {"message": {"body": {"mid": f"mid.{len(self.sent_messages)}"}}}


@pytest.mark.asyncio
async def test_max_single_flight_bypasses_user_locks_queue(db_session):
    """
    When request 1 is in-flight for a user, request 2 MUST NOT queue behind
    request 1 in _user_locks. It must immediately send AI_BUSY_MESSAGE and return.
    """
    client = MockMaxClient()
    app = MaxBotApplication(client=client)

    user_id = 999
    chat_id = 999
    first_task_proceed = asyncio.Event()
    first_task_started = asyncio.Event()

    async def in_flight_ai_dialogue():
        first_task_started.set()
        await first_task_proceed.wait()

    # Request 1 claims the single-flight slot and runs under spawn_ai_user_task
    lease1 = single_flight.try_claim("max", user_id)
    assert lease1 is not None
    app.spawn_ai_user_task(user_id, in_flight_ai_dialogue(), lease1)
    await first_task_started.wait()

    # Verify: user is currently busy
    assert single_flight.is_busy("max", user_id)

    # Request 2 arrives: incoming message for user 999
    msg2 = IncomingMessage(
        raw={},
        message_id="msg_2",
        chat_id=chat_id,
        sender=Sender(user_id=user_id, username="u999", first_name="User", last_name="999"),
        text="Second message while request 1 is running",
    )

    # Dispatch Request 2
    await app.handle_message(msg2)

    # Request 2 MUST receive AI_BUSY_MESSAGE immediately!
    assert len(client.sent_messages) == 1
    assert client.sent_messages[0][0] == chat_id
    assert client.sent_messages[0][1] == AI_BUSY_MESSAGE

    # Release request 1
    first_task_proceed.set()
    await asyncio.sleep(0.02)  # allow runner finally to complete

    # Slot must now be free
    assert not single_flight.is_busy("max", user_id)


@pytest.mark.asyncio
async def test_max_spawn_ai_user_task_releases_lease_on_exception():
    """
    spawn_ai_user_task must release the lease in finally if the task raises an exception.
    """
    client = MockMaxClient()
    app = MaxBotApplication(client=client)

    user_id = 888
    lease = single_flight.try_claim("max", user_id)
    assert lease is not None

    async def crashing_coro():
        raise RuntimeError("Fatal crash in AI task")

    app.spawn_ai_user_task(user_id, crashing_coro(), lease)
    await asyncio.sleep(0.02)

    assert not single_flight.is_busy("max", user_id)


# ============================================================================
# 5. Telegram Single-Flight Admission & Cross-Modality Tests
# ============================================================================

@pytest.mark.asyncio
async def test_telegram_chat_blocks_incoming_message_without_buffering(db_session):
    """
    When a Telegram user has an active AI turn running, a second text message
    receives AI_BUSY_MESSAGE and is NOT added to user_message_buffers.
    """
    user_id = 555
    handlers.user_message_buffers.pop(user_id, None)

    # Claim slot for user
    lease = single_flight.try_claim("telegram", user_id)
    assert lease is not None

    mock_msg = MagicMock()
    mock_msg.from_user.id = user_id
    mock_msg.text = "Hello while busy"
    mock_msg.answer = AsyncMock()

    mock_state = AsyncMock()
    mock_bot = MagicMock()

    try:
        await handlers.handle_ai_chat(mock_msg, mock_state, mock_bot)

        # Immediate busy copy sent
        assert mock_msg.answer.call_count == 1
        assert mock_msg.answer.call_args[0][0] == AI_BUSY_MESSAGE

        # Buffer was NOT populated
        assert user_id not in handlers.user_message_buffers or len(handlers.user_message_buffers[user_id]) == 0
    finally:
        single_flight.release(lease)


@pytest.mark.asyncio
async def test_telegram_cross_modality_text_blocks_photo(db_session):
    """
    Active text turn blocks incoming photo message with AI_BUSY_MESSAGE.
    """
    user_id = 777
    lease = single_flight.try_claim("telegram", user_id)
    assert lease is not None

    mock_msg = MagicMock()
    mock_msg.from_user.id = user_id
    mock_msg.photo = [MagicMock()]
    mock_msg.caption = "Analyze this"
    mock_msg.answer = AsyncMock()

    mock_state = AsyncMock()
    mock_bot = MagicMock()

    try:
        await handlers.handle_photo_message(mock_msg, mock_state, mock_bot)
        assert mock_msg.answer.call_count == 1
        assert mock_msg.answer.call_args[0][0] == AI_BUSY_MESSAGE
    finally:
        single_flight.release(lease)


@pytest.mark.asyncio
async def test_telegram_cross_modality_photo_blocks_text(db_session):
    """
    Active photo turn blocks incoming text message with AI_BUSY_MESSAGE.
    """
    user_id = 666
    lease = single_flight.try_claim("telegram", user_id)
    assert lease is not None

    mock_msg = MagicMock()
    mock_msg.from_user.id = user_id
    mock_msg.text = "Text during photo"
    mock_msg.answer = AsyncMock()

    mock_state = AsyncMock()
    mock_bot = MagicMock()

    try:
        await handlers.handle_ai_chat(mock_msg, mock_state, mock_bot)
        assert mock_msg.answer.call_count == 1
        assert mock_msg.answer.call_args[0][0] == AI_BUSY_MESSAGE
    finally:
        single_flight.release(lease)


@pytest.mark.asyncio
async def test_telegram_cross_modality_text_blocks_voice(db_session):
    """
    Active text turn blocks incoming voice message with AI_BUSY_MESSAGE.
    """
    user_id = 444
    lease = single_flight.try_claim("telegram", user_id)
    assert lease is not None

    mock_msg = MagicMock()
    mock_msg.from_user.id = user_id
    mock_msg.voice = MagicMock()
    mock_msg.voice.duration = 10
    mock_msg.answer = AsyncMock()

    mock_state = AsyncMock()
    mock_bot = MagicMock()

    try:
        await handlers.handle_voice_message(mock_msg, mock_state, mock_bot)
        assert mock_msg.answer.call_count == 1
        assert mock_msg.answer.call_args[0][0] == AI_BUSY_MESSAGE
    finally:
        single_flight.release(lease)


@pytest.mark.asyncio
async def test_telegram_action_button_click_blocked_when_busy(db_session):
    """
    When user has an active AI turn, clicking action button returns AI_BUSY_MESSAGE.
    """
    user_id = 333
    lease = single_flight.try_claim("telegram", user_id)
    assert lease is not None

    mock_callback = MagicMock()
    mock_callback.from_user.id = user_id
    mock_callback.answer = AsyncMock()

    mock_state = AsyncMock()
    mock_bot = MagicMock()

    try:
        await handlers.handle_action_button_click(mock_callback, mock_state, mock_bot)

        # Verification: callback answered with busy message alert
        assert mock_callback.answer.call_count == 1
        assert mock_callback.answer.call_args[0][0] == AI_BUSY_MESSAGE
    finally:
        single_flight.release(lease)


# ============================================================================
# 6. Exact Copy Verbatim Assertions
# ============================================================================

def test_verbatim_copies():
    expected_busy = "Пожалуйста, не так быстро — я ещё разбираю твоё предыдущее сообщение. Дай мне немного времени, чтобы всё хорошенько обдумать."
    expected_error = "Ой. Нейросеть сейчас перегружена и не отвечает. Загляни через несколько минут и повтори запрос. Я буду ждать!"

    assert AI_BUSY_MESSAGE == expected_busy
    assert handlers.AI_BUSY_MESSAGE == expected_busy
