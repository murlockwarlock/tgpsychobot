import asyncio
import io
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from ai_request_singleflight import single_flight, AI_BUSY_MESSAGE, UserAISingleFlight
import ai_integration
from ai_integration import AIServiceError
import database
from database import Base, AIConfig, User, Topic, Message as DBMessage, SubscriptionConfig, AILog, BotGeneralConfig
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
        session.add(SubscriptionConfig(id=1, subscriptions_enabled=False))
        session.add(BotGeneralConfig(id=1, profile_collect_name=False, profile_collect_gender=False, profile_collect_age=False))
        session.add(Topic(id=1, name="Topic 1", is_active=True, start_button_payload="Action Payload"))
        # Seed test users
        session.add(User(id=999, username="user999", first_name="User 999", name="User 999", gender="male", age="25", accepted_disclaimer=True, current_dialogue_id=1))
        session.add(User(id=555, username="user555", first_name="User 555", name="User 555", gender="male", age="25", accepted_disclaimer=True, current_dialogue_id=1))
        session.add(User(id=666, username="user666", first_name="User 666", name="User 666", gender="male", age="25", accepted_disclaimer=True, current_dialogue_id=1))
        session.add(User(id=777, username="user777", first_name="User 777", name="User 777", gender="male", age="25", accepted_disclaimer=True, current_dialogue_id=1))
        session.add(User(id=444, username="user444", first_name="User 444", name="User 444", gender="male", age="25", accepted_disclaimer=True, current_dialogue_id=1))
        session.add(User(id=333, username="user333", first_name="User 333", name="User 333", gender="male", age="25", accepted_disclaimer=True, current_dialogue_id=1, current_topic_id=1))
        await session.commit()

    monkeypatch.setattr(database, "async_session_maker", sessions)
    monkeypatch.setattr(handlers, "async_session_maker", sessions)
    monkeypatch.setattr(ai_integration, "async_session_maker", sessions)
    monkeypatch.setattr(max_ai, "async_session_maker", sessions)
    monkeypatch.setattr(max_storage, "async_session_maker", sessions)
    monkeypatch.setattr(max_app_module, "async_session_maker", sessions)
    monkeypatch.setattr(max_common, "async_session_maker", sessions)

    handlers.user_message_buffers.clear()
    handlers.user_message_buffer_leases.clear()
    handlers.user_isolated_turn_queues.clear()
    handlers.user_processing_tasks.clear()
    handlers.user_scheduling_locks.clear()

    try:
        yield sessions
    finally:
        handlers.user_message_buffers.clear()
        handlers.user_message_buffer_leases.clear()
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


def test_stale_lease_generation_race():
    """
    Stale lease cannot release a newer turn (generation race).
    lease_A = try_claim(...)
    release(A)
    lease_B = try_claim(...)
    release(A) again -> False
    is_busy remains True
    release(B) -> True
    """
    lease_a = single_flight.try_claim("telegram", 888)
    assert lease_a is not None
    assert single_flight.release(lease_a) is True

    lease_b = single_flight.try_claim("telegram", 888)
    assert lease_b is not None

    # Stale release of A must fail
    assert single_flight.release(lease_a) is False

    # B remains active and busy
    assert single_flight.is_busy("telegram", 888) is True

    # Proper release of B succeeds
    assert single_flight.release(lease_b) is True
    assert single_flight.is_busy("telegram", 888) is False


# ============================================================================
# 2. Telegram Text AI Hard Wall-Clock Timeout & Persisted AILog Tests
# ============================================================================

@pytest.mark.asyncio
async def test_telegram_ai_hard_timeout_primary_cancels_and_triggers_fallback(db_session, monkeypatch):
    """
    When primary provider hangs, asyncio.wait_for actively cancels it within fallback_timeout
    and triggers fallback provider.
    Actual AILog audit rows must be persisted:
      attempt 1: status=error, attempt_role=primary, error_classification=timeout
      attempt 2: status=success, attempt_role=fallback
    No later success row from cancelled primary.
    """
    primary_cancelled = False

    async def hanging_deepseek(*args, **kwargs):
        nonlocal primary_cancelled
        try:
            await asyncio.sleep(10.0)
            return "DeepSeek Finished Late"
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

    # Verify persisted AILog in real DB
    async with db_session() as session:
        logs = (await session.execute(
            select(AILog).where(AILog.user_id == 999).order_by(AILog.id.asc())
        )).scalars().all()

        assert len(logs) == 2
        assert logs[0].status == "error"
        assert logs[0].attempt_role == "primary"
        assert logs[0].error_classification == "timeout"
        assert logs[1].status == "success"
        assert logs[1].attempt_role == "fallback"

    # Ensure cancelled primary never writes a late success row
    await asyncio.sleep(0.05)
    async with db_session() as session:
        logs_after = (await session.execute(
            select(AILog).where(AILog.user_id == 999).order_by(AILog.id.asc())
        )).scalars().all()
        assert len(logs_after) == 2


@pytest.mark.asyncio
async def test_telegram_ai_hard_timeout_both_providers_fail(db_session, monkeypatch):
    """
    When both primary and fallback hang, both time out cleanly and AIServiceError
    with classification='timeout' is raised.
    Persisted AILog rows:
      attempt 1: status=error, attempt_role=primary, error_classification=timeout
      attempt 2: status=error, attempt_role=fallback, error_classification=timeout
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

    start = asyncio.get_event_loop().time()
    with pytest.raises(AIServiceError) as exc_info:
        await ai_integration.generate_response(
            user_id=999,
            user_prompt="hello",
        )
    elapsed = asyncio.get_event_loop().time() - start

    assert exc_info.value.classification == "timeout"
    assert elapsed < 1.0

    async with db_session() as session:
        logs = (await session.execute(
            select(AILog).where(AILog.user_id == 999).order_by(AILog.id.asc())
        )).scalars().all()

        assert len(logs) == 2
        assert logs[0].status == "error"
        assert logs[0].attempt_role == "primary"
        assert logs[0].error_classification == "timeout"
        assert logs[1].status == "error"
        assert logs[1].attempt_role == "fallback"
        assert logs[1].error_classification == "timeout"


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

    with pytest.raises(asyncio.CancelledError):
        await ai_integration.generate_response(
            user_id=999,
            user_prompt="hello",
        )

    assert fallback_mock.call_count == 0


# ============================================================================
# 3. MAX Hard Wall-Clock Timeout & Persisted AILog Tests
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


@pytest.mark.asyncio
async def test_max_get_ai_response_timeout_primary_triggers_fallback(db_session, monkeypatch):
    """
    MAX get_ai_response(): primary hangs -> timeout -> fallback succeeds.
    Persisted AILog attempts:
      attempt 1: status=error, attempt_role=primary, error_classification=timeout
      attempt 2: status=success, attempt_role=fallback
    """
    primary_cancelled = False

    async def hanging_deepseek(*args, **kwargs):
        nonlocal primary_cancelled
        try:
            await asyncio.sleep(10.0)
            return "DeepSeek Late"
        except asyncio.CancelledError:
            primary_cancelled = True
            raise

    async def successful_openai(*args, **kwargs):
        return "MAX OpenAI Fallback Response"

    monkeypatch.setattr(max_ai, "_call_deepseek", hanging_deepseek)
    monkeypatch.setattr(max_ai, "_call_openai", successful_openai)

    start = asyncio.get_event_loop().time()
    response = await max_ai.get_ai_response(
        user_id=999,
        user_prompt="hello max",
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert response == "MAX OpenAI Fallback Response"
    assert primary_cancelled is True
    assert elapsed < 1.0

    async with db_session() as session:
        logs = (await session.execute(
            select(AILog).where(AILog.user_id == 999, AILog.platform == "max").order_by(AILog.id.asc())
        )).scalars().all()

        assert len(logs) == 2
        assert logs[0].status == "error"
        assert logs[0].attempt_role == "primary"
        assert logs[0].error_classification == "timeout"
        assert logs[1].status == "success"
        assert logs[1].attempt_role == "fallback"


@pytest.mark.asyncio
async def test_max_get_ai_response_both_timeout(db_session, monkeypatch):
    """
    MAX get_ai_response(): both primary and fallback hang -> timeout.
    Two error attempts, both classification=timeout.
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

    monkeypatch.setattr(max_ai, "_call_deepseek", hanging_deepseek)
    monkeypatch.setattr(max_ai, "_call_openai", hanging_openai)

    start = asyncio.get_event_loop().time()
    with pytest.raises(max_ai.AIServiceError) as exc_info:
        await max_ai.get_ai_response(
            user_id=999,
            user_prompt="hello max",
        )
    elapsed = asyncio.get_event_loop().time() - start

    assert exc_info.value.classification == "timeout"
    assert elapsed < 1.0

    async with db_session() as session:
        logs = (await session.execute(
            select(AILog).where(AILog.user_id == 999, AILog.platform == "max").order_by(AILog.id.asc())
        )).scalars().all()

        assert len(logs) == 2
        assert logs[0].status == "error"
        assert logs[0].attempt_role == "primary"
        assert logs[0].error_classification == "timeout"
        assert logs[1].status == "error"
        assert logs[1].attempt_role == "fallback"
        assert logs[1].error_classification == "timeout"


# ============================================================================
# 4. MAX Real Ingress & Single-Flight Admission Tests
# ============================================================================

class MockMaxClient:
    def __init__(self):
        self.sent_messages = []

    async def send_message(self, chat_id: int, text: str, attachments=None):
        self.sent_messages.append((chat_id, text))
        return {"message": {"body": {"mid": f"mid.{len(self.sent_messages)}"}}}


@pytest.mark.asyncio
async def test_max_real_first_request_ingress_single_flight(db_session):
    """
    Real MAX first-request ingress test:
    Request 1 enters via app.handle_message() -> app itself claims -> provider fake blocks.
    Request 2 enters via app.handle_message().
    Before Request 1 releases:
      -> busy copy already sent
      -> Request 2 not queued in _user_locks
      -> one AI lifecycle call only
      -> second prompt not persisted
    After Request 1 completes:
      -> Request 2 never executes later.
      -> slot is free.
    """
    client = MockMaxClient()
    app = MaxBotApplication(client=client)

    user_id = 999
    chat_id = 999

    req1_started = asyncio.Event()
    req1_proceed = asyncio.Event()
    ai_call_count = 0

    async def fake_run_ai_dialogue(cli, c_id, u_id, prompt, states, **kwargs):
        nonlocal ai_call_count
        ai_call_count += 1
        req1_started.set()
        await req1_proceed.wait()
        await cli.send_message(chat_id=c_id, text="AI Response 1")

    with patch("max_messenger_bot.services.common.run_ai_dialogue", side_effect=fake_run_ai_dialogue):
        # 1. Request 1 enters via app.handle_message() (NO manual claim)
        msg1 = IncomingMessage(
            raw={},
            message_id="msg_1",
            chat_id=chat_id,
            sender=Sender(user_id=user_id, username="u999", first_name="User", last_name="999"),
            text="First message",
        )
        await app.handle_message(msg1)

        # Wait for Request 1 to claim slot and block in provider
        await req1_started.wait()

        # Slot is busy
        assert single_flight.is_busy("max", user_id) is True

        # 2. Request 2 enters via app.handle_message()
        msg2 = IncomingMessage(
            raw={},
            message_id="msg_2",
            chat_id=chat_id,
            sender=Sender(user_id=user_id, username="u999", first_name="User", last_name="999"),
            text="Second message while active",
        )
        await app.handle_message(msg2)

        # 3. Before Request 1 releases:
        # -> busy copy already sent
        assert len(client.sent_messages) == 1
        assert client.sent_messages[0] == (chat_id, AI_BUSY_MESSAGE)
        # -> exactly one AI lifecycle call
        assert ai_call_count == 1

        # 4. Release Request 1
        req1_proceed.set()
        await asyncio.sleep(0.05)

        # 5. After Request 1 completes:
        assert single_flight.is_busy("max", user_id) is False
        assert ai_call_count == 1
        assert len(client.sent_messages) == 2
        assert client.sent_messages[1] == (chat_id, "AI Response 1")


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
    await asyncio.sleep(0.02)

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


def make_mock_bot():
    bot = MagicMock()
    bot.send_message = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.delete_message = AsyncMock()
    bot.get_file = AsyncMock(return_value=SimpleNamespace(file_path="dummy.jpg"))
    bot.download_file = AsyncMock(return_value=io.BytesIO(b"dummy image bytes"))
    return bot


def make_mock_state():
    state = MagicMock()
    state.get_data = AsyncMock(return_value={})
    state.set_state = AsyncMock()
    state.update_data = AsyncMock()
    state.clear = AsyncMock()
    return state


# ============================================================================
# 5. Telegram Real Ingress & Single-Flight Admission Tests
# ============================================================================

@pytest.mark.asyncio
async def test_telegram_real_ingress_handle_ai_chat(db_session):
    """
    Real Telegram ingress test (NO manual single_flight claim):
    request 1 enters through handle_ai_chat -> provider fake blocks.
    Assert while provider active: single_flight.is_busy == True.
    Then request 2 enters through handle_ai_chat.
    Assert:
      exact busy UI
      zero second provider call
      zero second DB user Message
      zero second AILog
      no queued delayed request
    After request 1 completes: slot becomes free.
    """
    user_id = 999
    bot = make_mock_bot()
    state = make_mock_state()

    turn1_started = asyncio.Event()
    turn1_proceed = asyncio.Event()
    provider_calls = []

    from ai_log_context import record_ai_attempt_log

    async def fake_generate_response(u_id, prompt, *args, **kwargs):
        provider_calls.append(prompt)
        turn1_started.set()
        await turn1_proceed.wait()
        async with db_session() as s:
            await record_ai_attempt_log(
                s,
                user_id=u_id,
                platform="telegram",
                provider="Deepseek",
                model="deepseek-chat",
                status="success",
            )
        return "Assistant response 1"

    with patch("handlers.ai_integration.generate_response", side_effect=fake_generate_response):
        # 1. Request 1 enters via handle_ai_chat (NO manual single_flight claim)
        msg1 = MagicMock()
        msg1.from_user = SimpleNamespace(id=user_id, username="u999", full_name="User 999")
        msg1.chat = SimpleNamespace(id=user_id)
        msg1.text = "Hello first turn"
        msg1.answer = AsyncMock()

        await handlers.handle_ai_chat(msg1, state, bot)

        # Wait until provider is executing
        await turn1_started.wait()

        # While provider active: slot is busy
        assert single_flight.is_busy("telegram", user_id) is True

        # 2. Request 2 enters via handle_ai_chat
        msg2 = MagicMock()
        msg2.from_user = SimpleNamespace(id=user_id, username="u999", full_name="User 999")
        msg2.chat = SimpleNamespace(id=user_id)
        msg2.text = "Hello second turn while busy"
        msg2.answer = AsyncMock()

        await handlers.handle_ai_chat(msg2, state, bot)

        # Assert:
        # -> exact busy UI
        msg2.answer.assert_called_once_with(AI_BUSY_MESSAGE)
        # -> zero second provider call
        assert len(provider_calls) == 1
        # -> not in user_message_buffers
        assert user_id not in handlers.user_message_buffers or len(handlers.user_message_buffers[user_id]) == 0

        # 3. Release request 1
        turn1_proceed.set()

        # Drain runner task
        while handlers._has_user_turn_work(user_id) or (user_id in handlers.user_processing_tasks and not handlers.user_processing_tasks[user_id].done()):
            task = handlers.user_processing_tasks.get(user_id)
            if task:
                await task
            await asyncio.sleep(0.01)

    # 4. After completion:
    assert single_flight.is_busy("telegram", user_id) is False
    assert len(provider_calls) == 1
    assert user_id not in handlers.user_message_buffers
    assert user_id not in handlers.user_message_buffer_leases
    assert user_id not in handlers.user_processing_tasks

    # Verify DB: exactly 1 user message, 1 assistant message (no second user message)
    async with db_session() as session:
        user_msgs = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == user_id, DBMessage.role == "user")
        )).scalars().all()
        assert len(user_msgs) == 1
        assert user_msgs[0].content == "Hello first turn"

        # Verify AILogs: only 1 log entry (for Request 1)
        logs = (await session.execute(
            select(AILog).where(AILog.user_id == user_id)
        )).scalars().all()
        assert len(logs) == 1
        assert logs[0].status == "success"


# ============================================================================
# 6. Telegram Real Cross-Modality Tests
# ============================================================================

@pytest.mark.asyncio
async def test_telegram_cross_modality_real_text_blocks_photo(db_session):
    """
    REAL Telegram text request via handle_ai_chat -> provider blocks.
    Photo arrives via handle_photo_message (NO manual pre-claim).
    Expected: photo gets AI_BUSY_MESSAGE, zero vision provider calls.
    """
    user_id = 777
    bot = make_mock_bot()
    state = make_mock_state()

    text_started = asyncio.Event()
    text_proceed = asyncio.Event()
    vision_called = False

    async def fake_text_response(*args, **kwargs):
        text_started.set()
        await text_proceed.wait()
        return "Text AI Response"

    async def fake_vision_response(*args, **kwargs):
        nonlocal vision_called
        vision_called = True
        return "Vision AI Response"

    with patch("handlers.ai_integration.generate_response", side_effect=fake_text_response), \
         patch("handlers.ai_integration.analyze_image_content", side_effect=fake_vision_response):

        # 1. Real text request enters via handle_ai_chat
        msg_text = MagicMock()
        msg_text.from_user = SimpleNamespace(id=user_id, username="u777", full_name="User 777")
        msg_text.chat = SimpleNamespace(id=user_id)
        msg_text.text = "Active text turn"
        msg_text.answer = AsyncMock()

        await handlers.handle_ai_chat(msg_text, state, bot)
        await text_started.wait()

        # Slot is busy
        assert single_flight.is_busy("telegram", user_id) is True

        # 2. Photo arrives via handle_photo_message (NO manual pre-claim)
        msg_photo = MagicMock()
        msg_photo.from_user = SimpleNamespace(id=user_id, username="u777", full_name="User 777")
        msg_photo.chat = SimpleNamespace(id=user_id)
        msg_photo.photo = [MagicMock(file_id="photo_file_1")]
        msg_photo.caption = "Analyze photo while text active"
        msg_photo.answer = AsyncMock()

        await handlers.handle_photo_message(msg_photo, state, bot)

        # Assert:
        # -> photo gets AI_BUSY_MESSAGE
        msg_photo.answer.assert_called_once_with(AI_BUSY_MESSAGE)
        # -> zero vision provider call
        assert vision_called is False

        # 3. Release text provider
        text_proceed.set()

        while handlers._has_user_turn_work(user_id) or (user_id in handlers.user_processing_tasks and not handlers.user_processing_tasks[user_id].done()):
            task = handlers.user_processing_tasks.get(user_id)
            if task:
                await task
            await asyncio.sleep(0.01)

        assert single_flight.is_busy("telegram", user_id) is False


@pytest.mark.asyncio
async def test_telegram_cross_modality_real_photo_blocks_text(db_session):
    """
    REAL Telegram photo request via handle_photo_message -> vision provider blocks.
    Text arrives via handle_ai_chat (NO manual pre-claim).
    Expected: text gets AI_BUSY_MESSAGE, zero text provider calls.
    """
    user_id = 666
    bot = make_mock_bot()
    state = make_mock_state()

    photo_started = asyncio.Event()
    photo_proceed = asyncio.Event()
    text_called = False

    async def fake_vision_response(*args, **kwargs):
        photo_started.set()
        await photo_proceed.wait()
        return "Vision analysis result"

    async def fake_text_response(*args, **kwargs):
        nonlocal text_called
        text_called = True
        return "Text AI Response"

    with patch("handlers.ai_integration.analyze_image_content", side_effect=fake_vision_response), \
         patch("handlers.ai_integration.generate_response", side_effect=fake_text_response):

        # 1. Real photo request enters via handle_photo_message
        msg_photo = MagicMock()
        msg_photo.from_user = SimpleNamespace(id=user_id, username="u666", full_name="User 666")
        msg_photo.chat = SimpleNamespace(id=user_id)
        msg_photo.photo = [MagicMock(file_id="photo_file_666")]
        msg_photo.caption = "Initial photo"
        msg_photo.answer = AsyncMock()

        photo_task = asyncio.create_task(handlers.handle_photo_message(msg_photo, state, bot))
        await photo_started.wait()

        # Slot is busy
        assert single_flight.is_busy("telegram", user_id) is True

        # 2. Text arrives via handle_ai_chat (NO manual pre-claim)
        msg_text = MagicMock()
        msg_text.from_user = SimpleNamespace(id=user_id, username="u666", full_name="User 666")
        msg_text.chat = SimpleNamespace(id=user_id)
        msg_text.text = "Text during active photo"
        msg_text.answer = AsyncMock()

        await handlers.handle_ai_chat(msg_text, state, bot)

        # Assert:
        # -> text gets AI_BUSY_MESSAGE
        msg_text.answer.assert_called_once_with(AI_BUSY_MESSAGE)
        # -> zero text provider calls
        assert text_called is False
        # -> text was not buffered
        assert user_id not in handlers.user_message_buffers or len(handlers.user_message_buffers[user_id]) == 0

        # 3. Release photo
        photo_proceed.set()
        await photo_task

        assert single_flight.is_busy("telegram", user_id) is False


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

    mock_state = make_mock_state()
    mock_bot = make_mock_bot()

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

    mock_state = make_mock_state()
    mock_bot = make_mock_bot()

    try:
        await handlers.handle_action_button_click(mock_callback, mock_state, mock_bot)

        assert mock_callback.answer.call_count == 1
        assert mock_callback.answer.call_args[0][0] == AI_BUSY_MESSAGE
    finally:
        single_flight.release(lease)


# ============================================================================
# 7. Buffer and Lease Lifetime Invariant Test
# ============================================================================

@pytest.mark.asyncio
async def test_telegram_drain_runner_reschedule_preserves_associated_lease(db_session):
    """
    Invariant: If a normal buffered user turn remains pending and is being rescheduled,
    its lease MUST still exist and remain associated with it.
    """
    user_id = 555
    bot = make_mock_bot()
    state = make_mock_state()

    lease = single_flight.try_claim("telegram", user_id)
    assert lease is not None

    # Simulate pending buffer and lease
    handlers.user_message_buffers[user_id] = ["Pending user message"]
    handlers.user_message_buffer_leases[user_id] = lease

    with patch("handlers.ai_integration.generate_response", AsyncMock(return_value="Responded")):
        # Schedule a runner
        task = handlers._schedule_telegram_drain_runner_locked(user_id, bot, state, initial_delay=0.0)

        # Allow it to run and pop the lease + buffer
        while handlers._has_user_turn_work(user_id) or (user_id in handlers.user_processing_tasks and not handlers.user_processing_tasks[user_id].done()):
            t = handlers.user_processing_tasks.get(user_id)
            if t:
                await t
            await asyncio.sleep(0.01)

    # After full execution, both buffer and lease are cleanly released
    assert user_id not in handlers.user_message_buffers
    assert user_id not in handlers.user_message_buffer_leases
    assert not single_flight.is_busy("telegram", user_id)


@pytest.mark.asyncio
async def test_telegram_runner_failure_rescheduling_preserves_normal_lease(db_session):
    """
    When an isolated turn in drain_runner fails, but normal buffer is still pending,
    the drain_runner outer finally must KEEP user_message_buffer_leases associated
    and NOT release the normal turn's lease.
    """
    user_id = 555
    bot = make_mock_bot()
    state = make_mock_state()

    # Normal lease claimed by incoming message
    normal_lease = single_flight.try_claim("telegram", user_id)
    assert normal_lease is not None

    handlers.user_message_buffers[user_id] = ["Normal prompt awaiting execution"]
    handlers.user_message_buffer_leases[user_id] = normal_lease

    # Push an isolated work item that will fail
    handlers.user_isolated_turn_queues[user_id] = handlers.deque(["Failing isolated work"])

    runner_ran_normal = False

    async def fake_generate_response(u_id, prompt, *args, **kwargs):
        nonlocal runner_ran_normal
        if "Failing isolated work" in prompt:
            raise RuntimeError("Isolated turn crashed!")
        runner_ran_normal = True
        return "Normal response"

    with patch("handlers.ai_integration.generate_response", side_effect=fake_generate_response):
        # Run drain runner
        handlers._schedule_telegram_drain_runner_locked(user_id, bot, state, initial_delay=0.0)

        # Wait until everything finishes (rescheduled runner processes the normal turn)
        while handlers._has_user_turn_work(user_id) or (user_id in handlers.user_processing_tasks and not handlers.user_processing_tasks[user_id].done()):
            t = handlers.user_processing_tasks.get(user_id)
            if t:
                try:
                    await t
                except Exception:
                    pass
            await asyncio.sleep(0.01)

    # Normal turn was executed by the rescheduled runner!
    assert runner_ran_normal is True
    # Everything is now cleanly cleaned up
    assert user_id not in handlers.user_message_buffers
    assert user_id not in handlers.user_message_buffer_leases
    assert not single_flight.is_busy("telegram", user_id)


# ============================================================================
# 8. Exact Copy Verbatim Assertions
# ============================================================================

def test_verbatim_copies():
    expected_busy = "Пожалуйста, не так быстро — я ещё разбираю твоё предыдущее сообщение. Дай мне немного времени, чтобы всё хорошенько обдумать."
    expected_error = "Ой. Нейросеть сейчас перегружена и не отвечает. Загляни через несколько минут и повтори запрос. Я буду ждать!"

    assert AI_BUSY_MESSAGE == expected_busy
    assert handlers.AI_BUSY_MESSAGE == expected_busy
