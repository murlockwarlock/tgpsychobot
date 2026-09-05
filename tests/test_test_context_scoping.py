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
async def test_regression_6_legacy_matching_fallback_and_null_dialogue(scoping_db):
    """Legacy TestSession fallback:

    - Matching invocation_dialogue_id and topic injects [КОНТЕКСТ ТЕСТА] once
    - Null invocation_dialogue_id does NOT inject because scope cannot be proven
    """
    # Matching scope
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
    assert "[КОНТЕКСТ ТЕСТА]" in sent_text
    assert "Легаси ответ" in sent_text

    # Null invocation_dialogue_id case
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
