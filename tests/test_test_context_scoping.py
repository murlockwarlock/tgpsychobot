from __future__ import annotations

import json
import os
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import ai_integration
import database
from database import (
    AIConfig,
    Base,
    Message as DBMessage,
    SubscriptionConfig,
    TestConfig as DBTestConfig,
    TestSession as DBTestSession,
    Topic,
    User,
)
import max_messenger_bot.ai as max_ai
import max_messenger_bot.legacy as max_legacy
import memory_mode
from memory_mode import (
    MEMORY_MODE_GLOBAL,
    MEMORY_MODE_RESET,
    MEMORY_MODE_TOPIC,
    apply_memory_mode_topic_switch,
)
from result_history import TEST_RESULT_ROLE


class _CapturedCompletionClient:
    calls: list[dict] = []

    class _Completions:
        async def create(self, **payload):
            _CapturedCompletionClient.calls.append(payload)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Здравствуйте! Чем могу помочь?"))]
            )

    class _Chat:
        def __init__(self):
            self.completions = _CapturedCompletionClient._Completions()

    def __init__(self, *args, **kwargs):
        self.chat = self._Chat()

    async def close(self):
        return None


@pytest_asyncio.fixture
async def scoping_db(tmp_path, monkeypatch):
    _CapturedCompletionClient.calls.clear()
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test-scoping.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(database, "async_session_maker", sessions)
    monkeypatch.setattr(ai_integration, "async_session_maker", sessions)
    monkeypatch.setattr(max_legacy, "async_session_maker", sessions)
    monkeypatch.setattr(max_ai, "async_session_maker", sessions)

    # Seed AIConfig and SubscriptionConfig and TestConfig
    async with sessions() as session:
        session.add(
            AIConfig(
                id=1,
                provider="openai",
                openai_api_key="test-key",
                openai_model="gpt-5.6-terra",
                memory_mode=MEMORY_MODE_RESET,
                system_prompt="Ты эмпатичный психолог-помощник.",
            )
        )
        session.add(SubscriptionConfig(id=1))
        session.add(DBTestConfig(id=1, secret_test_enabled=True))
        await session.commit()

    # Mock AsyncOpenAI in both TG and MAX
    monkeypatch.setattr(ai_integration, "AsyncOpenAI", _CapturedCompletionClient)
    monkeypatch.setattr(max_ai, "AsyncOpenAI", _CapturedCompletionClient)
    monkeypatch.setattr(ai_integration, "search_relevant_chunks", AsyncMock(return_value=[]))
    monkeypatch.setattr(max_ai, "search_relevant_chunks", AsyncMock(return_value=[]))

    try:
        yield sessions
    finally:
        _CapturedCompletionClient.calls.clear()
        await engine.dispose()


@pytest.mark.asyncio
async def test_regression_1_telegram_reset_prod_incident(scoping_db):
    """Prod incident: user finished test in dialogue 2; after reset to dialogue 3,

    sending 'привет' in dialogue 3 must NOT contain any test context.
    """
    async with scoping_db() as session:
        user = User(
            id=101,
            first_name="Мария",
            current_dialogue_id=3,
            current_topic_id=2,
        )
        session.add(user)
        session.add(
            Topic(
                id=2,
                name="Тревожность",
                system_prompt="Ты специалист по тревожности.",
            )
        )
        # Finished TestSession from old dialogue 2
        session.add(
            DBTestSession(
                user_id=101,
                is_finished=True,
                invocation_dialogue_id=2,
                invocation_topic_id=2,
                answers=json.dumps([{"question": "Уровень тревоги", "answer": "9 из 10"}], ensure_ascii=False),
                secret_answers="Секретные ответы из диалога 2",
            )
        )
        # Old message in dialogue 2
        session.add(
            DBMessage(
                user_id=101,
                dialogue_id=2,
                topic_id=2,
                role=TEST_RESULT_ROLE,
                content="Уровень тревоги: 9 из 10",
                timestamp=datetime.utcnow(),
            )
        )
        await session.commit()

    _CapturedCompletionClient.calls.clear()
    reply = await ai_integration.generate_response(101, "привет")
    assert reply == "Здравствуйте! Чем могу помочь?"

    assert len(_CapturedCompletionClient.calls) == 1
    sent_payload = _CapturedCompletionClient.calls[0]
    all_text = json.dumps(sent_payload, ensure_ascii=False)

    assert "[КОНТЕКСТ ТЕСТА]" not in all_text
    assert "Уровень тревоги" not in all_text
    assert "9 из 10" not in all_text
    assert "[РЕЗУЛЬТАТЫ ПРОЙДЕННОГО ТЕСТА]" not in all_text
    assert "Секретные ответы из диалога 2" not in all_text


@pytest.mark.asyncio
async def test_regression_2_same_dialogue_persisted_test_result_injected_once(scoping_db):
    """Within the active dialogue, persisted test result is presented in history

    and does NOT duplicate with [КОНТЕКСТ ТЕСТА] injection.
    """
    async with scoping_db() as session:
        user = User(
            id=102,
            first_name="Алексей",
            current_dialogue_id=2,
            current_topic_id=None,
        )
        session.add(user)
        session.add(
            DBTestSession(
                user_id=102,
                is_finished=True,
                invocation_dialogue_id=2,
                invocation_topic_id=None,
                answers=json.dumps([{"question": "Шкала радости", "answer": "10"}], ensure_ascii=False),
            )
        )
        session.add(
            DBMessage(
                user_id=102,
                dialogue_id=2,
                topic_id=None,
                role=TEST_RESULT_ROLE,
                content="Шкала радости: 10",
                timestamp=datetime.utcnow(),
            )
        )
        await session.commit()

    _CapturedCompletionClient.calls.clear()
    await ai_integration.generate_response(102, "как мои результаты?")

    assert len(_CapturedCompletionClient.calls) == 1
    sent_payload = _CapturedCompletionClient.calls[0]
    all_text = json.dumps(sent_payload, ensure_ascii=False)

    assert "[РЕЗУЛЬТАТЫ ПРОЙДЕННОГО ТЕСТА]" in all_text
    assert "Шкала радости: 10" in all_text
    # No duplicate [КОНТЕКСТ ТЕСТА] block
    assert "[КОНТЕКСТ ТЕСТА]" not in all_text
    # Only 1 occurrence in the payload
    assert all_text.count("Шкала радости: 10") == 1


@pytest.mark.asyncio
async def test_regression_3_topic_mode_scoping(scoping_db):
    """In TOPIC mode:

    - main dialogue does not receive topic B test result
    - switching to topic B restores dialogue and receives its test result
    """
    async with scoping_db() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.memory_mode = MEMORY_MODE_TOPIC

        user = User(
            id=103,
            first_name="Елена",
            current_dialogue_id=1,
            current_topic_id=None,
        )
        session.add(user)
        session.add(Topic(id=5, name="Отношения"))
        await session.commit()

    # Switch to topic 5 -> creates dialogue 2
    async with scoping_db() as session:
        user = await session.get(User, 103)
        await apply_memory_mode_topic_switch(session, user, topic_id=5, memory_mode=MEMORY_MODE_TOPIC)
        user.current_topic_id = 5
        session.add(
            DBMessage(
                user_id=103,
                dialogue_id=user.current_dialogue_id,
                topic_id=5,
                role=TEST_RESULT_ROLE,
                content="Результат темы отношений: высокий",
                timestamp=datetime.utcnow(),
            )
        )
        await session.commit()

    # Switch back to main (topic None) -> restores dialogue 1
    async with scoping_db() as session:
        user = await session.get(User, 103)
        await apply_memory_mode_topic_switch(session, user, topic_id=None, memory_mode=MEMORY_MODE_TOPIC)
        user.current_topic_id = None
        await session.commit()

    _CapturedCompletionClient.calls.clear()
    await ai_integration.generate_response(103, "привет в основном диалоге")
    assert len(_CapturedCompletionClient.calls) == 1
    main_text = json.dumps(_CapturedCompletionClient.calls[0], ensure_ascii=False)
    assert "Результат темы отношений" not in main_text

    # Switch back to topic 5 -> restores dialogue 2
    async with scoping_db() as session:
        user = await session.get(User, 103)
        await apply_memory_mode_topic_switch(session, user, topic_id=5, memory_mode=MEMORY_MODE_TOPIC)
        user.current_topic_id = 5
        await session.commit()

    _CapturedCompletionClient.calls.clear()
    await ai_integration.generate_response(103, "привет в теме")
    assert len(_CapturedCompletionClient.calls) == 1
    topic_text = json.dumps(_CapturedCompletionClient.calls[0], ensure_ascii=False)
    assert "Результат темы отношений: высокий" in topic_text


@pytest.mark.asyncio
async def test_regression_4_global_mode_scoping_and_reset(scoping_db):
    """In GLOBAL mode:

    - switching topics preserves dialogue, keeping test result accessible
    - explicit dialogue reset removes test result from the new dialogue
    """
    async with scoping_db() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.memory_mode = MEMORY_MODE_GLOBAL

        user = User(
            id=104,
            first_name="Дмитрий",
            current_dialogue_id=1,
            current_topic_id=1,
        )
        session.add(user)
        session.add(Topic(id=1, name="Тема 1"))
        session.add(Topic(id=2, name="Тема 2"))
        session.add(
            DBMessage(
                user_id=104,
                dialogue_id=1,
                topic_id=1,
                role=TEST_RESULT_ROLE,
                content="Глобальный тест: 42 балла",
                timestamp=datetime.utcnow(),
            )
        )
        await session.commit()

    # Switch from topic 1 to topic 2 without resetting dialogue (preserves dialogue 1)
    async with scoping_db() as session:
        user = await session.get(User, 104)
        user.current_topic_id = 2
        await session.commit()

    _CapturedCompletionClient.calls.clear()
    await ai_integration.generate_response(104, "привет в теме 2")
    assert len(_CapturedCompletionClient.calls) == 1
    global_text = json.dumps(_CapturedCompletionClient.calls[0], ensure_ascii=False)
    assert "Глобальный тест: 42 балла" in global_text

    # Explicit reset to dialogue 2
    async with scoping_db() as session:
        user = await session.get(User, 104)
        user.current_dialogue_id = 2
        await session.commit()

    _CapturedCompletionClient.calls.clear()
    await ai_integration.generate_response(104, "привет после сброса")
    assert len(_CapturedCompletionClient.calls) == 1
    reset_text = json.dumps(_CapturedCompletionClient.calls[0], ensure_ascii=False)
    assert "Глобальный тест: 42 балла" not in reset_text
    assert "[КОНТЕКСТ ТЕСТА]" not in reset_text


@pytest.mark.asyncio
async def test_regression_5_stale_finished_testsession_without_message_not_injected(scoping_db):
    """Finished TestSession from old dialogue without a DBMessage is NOT injected

    into newer dialogue.
    """
    async with scoping_db() as session:
        user = User(
            id=105,
            first_name="Ольга",
            current_dialogue_id=2,
            current_topic_id=None,
        )
        session.add(user)
        session.add(
            DBTestSession(
                user_id=105,
                is_finished=True,
                invocation_dialogue_id=1,
                invocation_topic_id=None,
                answers=json.dumps([{"question": "Старый вопрос", "answer": "Старый ответ"}], ensure_ascii=False),
            )
        )
        await session.commit()

    _CapturedCompletionClient.calls.clear()
    await ai_integration.generate_response(105, "привет")
    assert len(_CapturedCompletionClient.calls) == 1
    sent_text = json.dumps(_CapturedCompletionClient.calls[0], ensure_ascii=False)
    assert "[КОНТЕКСТ ТЕСТА]" not in sent_text
    assert "Старый ответ" not in sent_text


@pytest.mark.asyncio
async def test_regression_6_isolated_testsession_without_message_never_injected(scoping_db):
    """TestSession without a persisted DBMessage(role="test_result") is NEVER injected

    at runtime, regardless of invocation_dialogue_id matching or null.
    """
    # Matching dialogue_id on TestSession, but no persisted test_result message
    async with scoping_db() as session:
        user = User(
            id=106,
            first_name="Кирилл",
            current_dialogue_id=2,
            current_topic_id=None,
        )
        session.add(user)
        session.add(
            DBTestSession(
                user_id=106,
                is_finished=True,
                invocation_dialogue_id=2,
                invocation_topic_id=None,
                answers=json.dumps([{"question": "Легаси вопрос", "answer": "Легаси ответ"}], ensure_ascii=False),
            )
        )
        await session.commit()

    _CapturedCompletionClient.calls.clear()
    await ai_integration.generate_response(106, "привет")
    assert len(_CapturedCompletionClient.calls) == 1
    sent_text = json.dumps(_CapturedCompletionClient.calls[0], ensure_ascii=False)
    assert "[КОНТЕКСТ ТЕСТА]" not in sent_text
    assert "Легаси ответ" not in sent_text

    # Null invocation_dialogue_id on TestSession
    async with scoping_db() as session:
        user_null = User(
            id=107,
            first_name="Анна",
            current_dialogue_id=2,
            current_topic_id=None,
        )
        session.add(user_null)
        session.add(
            DBTestSession(
                user_id=107,
                is_finished=True,
                invocation_dialogue_id=None,
                invocation_topic_id=None,
                answers=json.dumps([{"question": "Вопрос без диалога", "answer": "Ответ без диалога"}], ensure_ascii=False),
            )
        )
        await session.commit()

    _CapturedCompletionClient.calls.clear()
    await ai_integration.generate_response(107, "привет")
    assert len(_CapturedCompletionClient.calls) == 1
    null_text = json.dumps(_CapturedCompletionClient.calls[0], ensure_ascii=False)
    assert "[КОНТЕКСТ ТЕСТА]" not in null_text
    assert "Ответ без диалога" not in null_text


@pytest.mark.asyncio
async def test_regression_11_telegram_test_result_obeys_normal_history_window_selection(scoping_db):
    """Telegram: test_result lifetime equals ordinary configured history lifetime.

    - Outside selected first and recent window: test_result is not sent to AI.
    - Inside selected window: test_result is sent exactly once.
    """
    async with scoping_db() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.context_limit_first = 1
        cfg.context_limit_recent = 1

        # User 110: test_result placed in the MIDDLE of a long conversation
        user_outside = User(id=110, first_name="Пётр", current_dialogue_id=1)
        session.add(user_outside)

        # Pair 1 (First window):
        session.add(DBMessage(user_id=110, dialogue_id=1, role="user", content="1. Первое сообщение", timestamp=datetime.utcnow()))
        session.add(DBMessage(user_id=110, dialogue_id=1, role="assistant", content="1. Первый ответ", timestamp=datetime.utcnow()))

        # Pair 2 (Middle - outside window): contains test_result
        session.add(DBMessage(user_id=110, dialogue_id=1, role=TEST_RESULT_ROLE, content="Тест в середине истории: 77", timestamp=datetime.utcnow()))
        session.add(DBMessage(user_id=110, dialogue_id=1, role="user", content="2. Второй вопрос", timestamp=datetime.utcnow()))
        session.add(DBMessage(user_id=110, dialogue_id=1, role="assistant", content="2. Второй ответ", timestamp=datetime.utcnow()))

        # Pair 3 (Middle - outside window):
        session.add(DBMessage(user_id=110, dialogue_id=1, role="user", content="3. Третий вопрос", timestamp=datetime.utcnow()))
        session.add(DBMessage(user_id=110, dialogue_id=1, role="assistant", content="3. Третий ответ", timestamp=datetime.utcnow()))

        # Pair 4 (Recent window):
        session.add(DBMessage(user_id=110, dialogue_id=1, role="user", content="4. Четвертый вопрос", timestamp=datetime.utcnow()))
        session.add(DBMessage(user_id=110, dialogue_id=1, role="assistant", content="4. Четвертый ответ", timestamp=datetime.utcnow()))

        # User 111: test_result placed inside the FIRST window
        user_inside = User(id=111, first_name="Наталья", current_dialogue_id=1)
        session.add(user_inside)

        # Pair 1 (First window): contains test_result
        session.add(DBMessage(user_id=111, dialogue_id=1, role=TEST_RESULT_ROLE, content="Тест в первом окне: 88", timestamp=datetime.utcnow()))
        session.add(DBMessage(user_id=111, dialogue_id=1, role="user", content="1. Вопрос один", timestamp=datetime.utcnow()))
        session.add(DBMessage(user_id=111, dialogue_id=1, role="assistant", content="1. Ответ один", timestamp=datetime.utcnow()))

        # Pair 2 (Middle):
        session.add(DBMessage(user_id=111, dialogue_id=1, role="user", content="2. Вопрос два", timestamp=datetime.utcnow()))
        session.add(DBMessage(user_id=111, dialogue_id=1, role="assistant", content="2. Ответ два", timestamp=datetime.utcnow()))

        # Pair 3 (Recent window):
        session.add(DBMessage(user_id=111, dialogue_id=1, role="user", content="3. Вопрос три", timestamp=datetime.utcnow()))
        session.add(DBMessage(user_id=111, dialogue_id=1, role="assistant", content="3. Ответ три", timestamp=datetime.utcnow()))

        await session.commit()

    # Case A: outside window
    _CapturedCompletionClient.calls.clear()
    await ai_integration.generate_response(110, "5. Пятый вопрос")
    assert len(_CapturedCompletionClient.calls) == 1
    outside_payload = json.dumps(_CapturedCompletionClient.calls[0], ensure_ascii=False)
    assert "1. Первое сообщение" in outside_payload
    assert "4. Четвертый вопрос" in outside_payload
    assert "Тест в середине истории: 77" not in outside_payload
    assert "[РЕЗУЛЬТАТЫ ПРОЙДЕННОГО ТЕСТА]" not in outside_payload

    # Case B: inside window
    _CapturedCompletionClient.calls.clear()
    await ai_integration.generate_response(111, "4. Вопрос четыре")
    assert len(_CapturedCompletionClient.calls) == 1
    inside_payload = json.dumps(_CapturedCompletionClient.calls[0], ensure_ascii=False)
    assert "Тест в первом окне: 88" in inside_payload
    assert "[РЕЗУЛЬТАТЫ ПРОЙДЕННОГО ТЕСТА]" in inside_payload
    assert inside_payload.count("Тест в первом окне: 88") == 1


@pytest.mark.asyncio
async def test_regression_12_max_test_result_obeys_normal_history_window_selection(scoping_db):
    """MAX: test_result lifetime equals ordinary configured history lifetime.

    - Outside selected first and recent window: test_result is not sent to AI.
    - Inside selected window: test_result is sent exactly once.
    """
    async with scoping_db() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.context_limit_first = 1
        cfg.context_limit_recent = 1

        # User 210: test_result in the middle (outside window)
        user_outside = User(id=210, first_name="Константин", current_dialogue_id=1)
        session.add(user_outside)

        # Pair 1 (First window)
        session.add(DBMessage(user_id=210, dialogue_id=1, role="user", content="MAX 1. Первое сообщение", timestamp=datetime.utcnow()))
        session.add(DBMessage(user_id=210, dialogue_id=1, role="assistant", content="MAX 1. Первый ответ", timestamp=datetime.utcnow()))

        # Pair 2 (Middle - outside window)
        session.add(DBMessage(user_id=210, dialogue_id=1, role=TEST_RESULT_ROLE, content="MAX тест в середине: 99", timestamp=datetime.utcnow()))
        session.add(DBMessage(user_id=210, dialogue_id=1, role="user", content="MAX 2. Второй вопрос", timestamp=datetime.utcnow()))
        session.add(DBMessage(user_id=210, dialogue_id=1, role="assistant", content="MAX 2. Второй ответ", timestamp=datetime.utcnow()))

        # Pair 3 (Middle - outside window)
        session.add(DBMessage(user_id=210, dialogue_id=1, role="user", content="MAX 3. Третий вопрос", timestamp=datetime.utcnow()))
        session.add(DBMessage(user_id=210, dialogue_id=1, role="assistant", content="MAX 3. Третий ответ", timestamp=datetime.utcnow()))

        # Pair 4 (Recent window)
        session.add(DBMessage(user_id=210, dialogue_id=1, role="user", content="MAX 4. Четвертый вопрос", timestamp=datetime.utcnow()))
        session.add(DBMessage(user_id=210, dialogue_id=1, role="assistant", content="MAX 4. Четвертый ответ", timestamp=datetime.utcnow()))

        # User 211: test_result inside first window
        user_inside = User(id=211, first_name="Марина", current_dialogue_id=1)
        session.add(user_inside)

        # Pair 1 (First window): contains test_result
        session.add(DBMessage(user_id=211, dialogue_id=1, role=TEST_RESULT_ROLE, content="MAX тест в первом окне: 100", timestamp=datetime.utcnow()))
        session.add(DBMessage(user_id=211, dialogue_id=1, role="user", content="MAX 1. Вопрос один", timestamp=datetime.utcnow()))
        session.add(DBMessage(user_id=211, dialogue_id=1, role="assistant", content="MAX 1. Ответ один", timestamp=datetime.utcnow()))

        # Pair 2 (Middle)
        session.add(DBMessage(user_id=211, dialogue_id=1, role="user", content="MAX 2. Вопрос два", timestamp=datetime.utcnow()))
        session.add(DBMessage(user_id=211, dialogue_id=1, role="assistant", content="MAX 2. Ответ два", timestamp=datetime.utcnow()))

        # Pair 3 (Recent window)
        session.add(DBMessage(user_id=211, dialogue_id=1, role="user", content="MAX 3. Вопрос три", timestamp=datetime.utcnow()))
        session.add(DBMessage(user_id=211, dialogue_id=1, role="assistant", content="MAX 3. Ответ три", timestamp=datetime.utcnow()))

        await session.commit()

    # Case A: outside window
    _CapturedCompletionClient.calls.clear()
    await max_ai.get_ai_response(210, "MAX 5. Пятый вопрос")
    assert len(_CapturedCompletionClient.calls) == 1
    outside_payload = json.dumps(_CapturedCompletionClient.calls[0], ensure_ascii=False)
    assert "MAX 1. Первое сообщение" in outside_payload
    assert "MAX 4. Четвертый вопрос" in outside_payload
    assert "MAX тест в середине: 99" not in outside_payload
    assert "[РЕЗУЛЬТАТЫ ПРОЙДЕННОГО ТЕСТА]" not in outside_payload

    # Case B: inside window
    _CapturedCompletionClient.calls.clear()
    await max_ai.get_ai_response(211, "MAX 4. Вопрос четыре")
    assert len(_CapturedCompletionClient.calls) == 1
    inside_payload = json.dumps(_CapturedCompletionClient.calls[0], ensure_ascii=False)
    assert "MAX тест в первом окне: 100" in inside_payload
    assert "[РЕЗУЛЬТАТЫ ПРОЙДЕННОГО ТЕСТА]" in inside_payload
    assert inside_payload.count("MAX тест в первом окне: 100") == 1


@pytest.mark.asyncio
async def test_regression_7_max_regression_no_stale_test_context(scoping_db):
    """MAX AI request with an old finished TestSession from another dialogue

    does NOT acquire [КОНТЕКСТ ТЕСТА].
    """
    async with scoping_db() as session:
        user = User(
            id=108,
            first_name="Максим",
            current_dialogue_id=2,
            current_topic_id=None,
        )
        session.add(user)
        session.add(
            DBTestSession(
                user_id=108,
                is_finished=True,
                invocation_dialogue_id=1,
                invocation_topic_id=None,
                answers=json.dumps([{"question": "Вопрос MAX", "answer": "Ответ MAX"}], ensure_ascii=False),
            )
        )
        await session.commit()

    _CapturedCompletionClient.calls.clear()
    reply = await max_ai.get_ai_response(108, "привет")
    assert reply == "Здравствуйте! Чем могу помочь?"

    assert len(_CapturedCompletionClient.calls) == 1
    max_text = json.dumps(_CapturedCompletionClient.calls[0], ensure_ascii=False)
    assert "[КОНТЕКСТ ТЕСТА]" not in max_text
    assert "Ответ MAX" not in max_text


@pytest.mark.asyncio
async def test_regression_8_max_global_topic_switch_preserves_dialogue_and_reset_clears(scoping_db):
    """MAX GLOBAL mode journey:

    - Dialogue 5 in Topic A has ordinary history + test_result.
    - Switching active topic to Topic B without changing dialogue_id preserves Dialogue 5 context.
    - Explicit reset to Dialogue 6 removes old Dialogue 5 user/assistant history and test_result.
    """
    async with scoping_db() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.memory_mode = MEMORY_MODE_GLOBAL

        user = User(
            id=201,
            first_name="Виктор",
            current_dialogue_id=5,
            current_topic_id=1,
        )
        session.add(user)
        session.add(Topic(id=1, name="Тема A"))
        session.add(Topic(id=2, name="Тема B"))

        session.add(
            DBMessage(
                user_id=201,
                dialogue_id=5,
                topic_id=1,
                role="user",
                content="Вопрос в теме A",
                timestamp=datetime.utcnow(),
            )
        )
        session.add(
            DBMessage(
                user_id=201,
                dialogue_id=5,
                topic_id=1,
                role="assistant",
                content="Ответ в теме A",
                timestamp=datetime.utcnow(),
            )
        )
        session.add(
            DBMessage(
                user_id=201,
                dialogue_id=5,
                topic_id=1,
                role=TEST_RESULT_ROLE,
                content="Тест в теме A: 100 баллов",
                timestamp=datetime.utcnow(),
            )
        )
        await session.commit()

    # Switch active topic to Topic B WITHOUT changing dialogue_id (GLOBAL mode)
    async with scoping_db() as session:
        user = await session.get(User, 201)
        user.current_topic_id = 2
        await session.commit()

    _CapturedCompletionClient.calls.clear()
    reply = await max_ai.get_ai_response(201, "Привет в теме B")
    assert reply == "Здравствуйте! Чем могу помочь?"

    assert len(_CapturedCompletionClient.calls) == 1
    payload_text = json.dumps(_CapturedCompletionClient.calls[0], ensure_ascii=False)
    assert "Вопрос в теме A" in payload_text
    assert "Ответ в теме A" in payload_text
    assert "Тест в теме A: 100 баллов" in payload_text
    assert "[РЕЗУЛЬТАТЫ ПРОЙДЕННОГО ТЕСТА]" in payload_text

    # Explicit dialogue reset to dialogue 6
    async with scoping_db() as session:
        user = await session.get(User, 201)
        user.current_dialogue_id = 6
        await session.commit()

    _CapturedCompletionClient.calls.clear()
    reply = await max_ai.get_ai_response(201, "Привет после сброса")
    assert reply == "Здравствуйте! Чем могу помочь?"

    assert len(_CapturedCompletionClient.calls) == 1
    reset_payload_text = json.dumps(_CapturedCompletionClient.calls[0], ensure_ascii=False)
    assert "Вопрос в теме A" not in reset_payload_text
    assert "Ответ в теме A" not in reset_payload_text
    assert "Тест в теме A: 100 баллов" not in reset_payload_text
    assert "[РЕЗУЛЬТАТЫ ПРОЙДЕННОГО ТЕСТА]" not in reset_payload_text


@pytest.mark.asyncio
async def test_regression_9_max_topic_restored_topic_gets_only_its_saved_dialogue(scoping_db):
    """MAX TOPIC mode journey:

    - Topic A in dialogue 5 has A history + A test_result.
    - Topic B in dialogue 6 has B history + B test_result.
    - Active topic B / dialogue 6 receives only B content.
    - Restored topic A / dialogue 5 receives only A content.
    """
    async with scoping_db() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.memory_mode = MEMORY_MODE_TOPIC

        user = User(
            id=202,
            first_name="Светлана",
            current_dialogue_id=6,
            current_topic_id=20,
        )
        session.add(user)
        session.add(Topic(id=10, name="Тема A"))
        session.add(Topic(id=20, name="Тема B"))

        # Topic A / dialogue 5
        session.add(
            DBMessage(
                user_id=202,
                dialogue_id=5,
                topic_id=10,
                role="user",
                content="Сообщение темы A",
                timestamp=datetime.utcnow(),
            )
        )
        session.add(
            DBMessage(
                user_id=202,
                dialogue_id=5,
                topic_id=10,
                role="assistant",
                content="Ответ темы A",
                timestamp=datetime.utcnow(),
            )
        )
        session.add(
            DBMessage(
                user_id=202,
                dialogue_id=5,
                topic_id=10,
                role=TEST_RESULT_ROLE,
                content="Тест темы A: пройден успешно",
                timestamp=datetime.utcnow(),
            )
        )

        # Topic B / dialogue 6
        session.add(
            DBMessage(
                user_id=202,
                dialogue_id=6,
                topic_id=20,
                role="user",
                content="Сообщение темы B",
                timestamp=datetime.utcnow(),
            )
        )
        session.add(
            DBMessage(
                user_id=202,
                dialogue_id=6,
                topic_id=20,
                role="assistant",
                content="Ответ темы B",
                timestamp=datetime.utcnow(),
            )
        )
        session.add(
            DBMessage(
                user_id=202,
                dialogue_id=6,
                topic_id=20,
                role=TEST_RESULT_ROLE,
                content="Тест темы B: завершен успешно",
                timestamp=datetime.utcnow(),
            )
        )
        await session.commit()

    # Active: Topic B / dialogue 6
    _CapturedCompletionClient.calls.clear()
    reply = await max_ai.get_ai_response(202, "Запрос в теме B")
    assert reply == "Здравствуйте! Чем могу помочь?"

    assert len(_CapturedCompletionClient.calls) == 1
    b_payload = json.dumps(_CapturedCompletionClient.calls[0], ensure_ascii=False)
    assert "Сообщение темы B" in b_payload
    assert "Ответ темы B" in b_payload
    assert "Тест темы B: завершен успешно" in b_payload
    assert "Сообщение темы A" not in b_payload
    assert "Ответ темы A" not in b_payload
    assert "Тест темы A: пройден успешно" not in b_payload

    # Restore: Topic A / dialogue 5
    async with scoping_db() as session:
        user = await session.get(User, 202)
        user.current_topic_id = 10
        user.current_dialogue_id = 5
        await session.commit()

    _CapturedCompletionClient.calls.clear()
    reply = await max_ai.get_ai_response(202, "Запрос в теме A")
    assert reply == "Здравствуйте! Чем могу помочь?"

    assert len(_CapturedCompletionClient.calls) == 1
    a_payload = json.dumps(_CapturedCompletionClient.calls[0], ensure_ascii=False)
    assert "Сообщение темы A" in a_payload
    assert "Ответ темы A" in a_payload
    assert "Тест темы A: пройден успешно" in a_payload
    assert "Сообщение темы B" not in a_payload
    assert "Ответ темы B" not in a_payload
    assert "Тест темы B: завершен успешно" not in a_payload


@pytest.mark.asyncio
async def test_regression_10_max_topic_explicit_reset_inside_same_topic_clears_old_dialogue(scoping_db):
    """MAX TOPIC mode explicit reset journey:

    - Topic A in dialogue 5 has ordinary history + test_result.
    - Reset to Topic A in dialogue 6.
    - All old dialogue 5 content is absent even though topic_id (Topic A) is unchanged.
    """
    async with scoping_db() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.memory_mode = MEMORY_MODE_TOPIC

        user = User(
            id=203,
            first_name="Игорь",
            current_dialogue_id=5,
            current_topic_id=10,
        )
        session.add(user)
        session.add(Topic(id=10, name="Тема A"))

        session.add(
            DBMessage(
                user_id=203,
                dialogue_id=5,
                topic_id=10,
                role="user",
                content="Старый вопрос в теме A",
                timestamp=datetime.utcnow(),
            )
        )
        session.add(
            DBMessage(
                user_id=203,
                dialogue_id=5,
                topic_id=10,
                role="assistant",
                content="Старый ответ в теме A",
                timestamp=datetime.utcnow(),
            )
        )
        session.add(
            DBMessage(
                user_id=203,
                dialogue_id=5,
                topic_id=10,
                role=TEST_RESULT_ROLE,
                content="Старый тест в теме A: 55",
                timestamp=datetime.utcnow(),
            )
        )
        await session.commit()

    # Reset dialogue inside same Topic A to dialogue 6
    async with scoping_db() as session:
        user = await session.get(User, 203)
        user.current_dialogue_id = 6
        await session.commit()

    _CapturedCompletionClient.calls.clear()
    reply = await max_ai.get_ai_response(203, "Новый вопрос в теме A")
    assert reply == "Здравствуйте! Чем могу помочь?"

    assert len(_CapturedCompletionClient.calls) == 1
    reset_payload = json.dumps(_CapturedCompletionClient.calls[0], ensure_ascii=False)
    assert "Старый вопрос в теме A" not in reset_payload
    assert "Старый ответ в теме A" not in reset_payload
    assert "Старый тест в теме A: 55" not in reset_payload
    assert "[РЕЗУЛЬТАТЫ ПРОЙДЕННОГО ТЕСТА]" not in reset_payload
