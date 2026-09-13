import asyncio
import json
import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("BOT_TOKEN", "test")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import ai_integration
from ai_request_builder import (
    build_conversational_request_layout,
    load_conversational_ai_history,
)
from ai_request_context import (
    AIRequestLayout,
    AIRequestMessage,
    build_gemini_contents,
    build_openai_chat_messages,
    build_responses_input,
)
import handlers
from database import (
    AIConfig,
    AILog,
    Base,
    Message as DBMessage,
    Topic,
    User,
)
from max_messenger_bot import ai as max_ai
from max_messenger_bot.services import common as max_common
from result_history import (
    SYSTEM_EVENT_ROLE,
    TEST_RESULT_ROLE,
    AIHistoryMessage,
    select_ai_history_messages,
)


class IncidentB1HistoryDedupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        self._orig_tg_sessions = ai_integration.async_session_maker
        self._orig_max_sessions = max_ai.async_session_maker
        self._orig_max_common_sessions = max_common.async_session_maker
        self._orig_handlers_sessions = handlers.async_session_maker
        ai_integration.async_session_maker = self.sessions
        max_ai.async_session_maker = self.sessions
        max_common.async_session_maker = self.sessions
        handlers.async_session_maker = self.sessions

        async with self.sessions() as session:
            self.topic = Topic(
                id=1,
                name="Психология",
                is_active=True,
                system_prompt="Ты эмпатичный психолог.",
            )
            self.other_topic = Topic(
                id=2,
                name="Карьера",
                is_active=True,
                system_prompt="Ты коуч.",
            )
            self.user = User(
                id=9001,
                first_name="Алексей",
                current_dialogue_id=1,
                current_topic_id=1,
                metadata_json="{}",
            )
            self.ai_config = AIConfig(
                id=1,
                provider="OpenAI",
                openai_api_key="test-openai-key",
                openai_model="gpt-5.6-terra",
                memory_mode="topic",
                system_prompt="Базовый промпт.",
            )
            session.add_all([self.topic, self.other_topic, self.user, self.ai_config])
            await session.commit()

    async def asyncTearDown(self):
        ai_integration.async_session_maker = self._orig_tg_sessions
        max_ai.async_session_maker = self._orig_max_sessions
        max_common.async_session_maker = self._orig_max_common_sessions
        handlers.async_session_maker = self._orig_handlers_sessions
        await self.engine.dispose()

    # -------------------------------------------------------------------------
    # Unit Tests for select_ai_history_messages
    # -------------------------------------------------------------------------

    def test_01_historical_raw_dedup_triple(self):
        base_ts = datetime.utcnow()
        messages = [
            DBMessage(id=1, role="user", content="27 мая 1988 года", timestamp=base_ts),
            DBMessage(id=2, role="user", content="27 мая 1988 года", timestamp=base_ts + timedelta(seconds=1)),
            DBMessage(id=3, role="user", content="27 мая 1988 года", timestamp=base_ts + timedelta(seconds=2)),
        ]
        result = select_ai_history_messages(messages, limit_first=2, limit_recent=10)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].role, "user")
        self.assertEqual(result[0].content, "27 мая 1988 года")

    def test_02_historical_raw_dedup_pairs(self):
        base_ts = datetime.utcnow()
        messages = [
            DBMessage(id=1, role="user", content="Вопрос А", timestamp=base_ts),
            DBMessage(id=2, role="user", content="Вопрос А", timestamp=base_ts + timedelta(seconds=1)),
            DBMessage(id=3, role="user", content="Вопрос Б", timestamp=base_ts + timedelta(seconds=2)),
            DBMessage(id=4, role="user", content="Вопрос Б", timestamp=base_ts + timedelta(seconds=3)),
        ]
        result = select_ai_history_messages(messages, limit_first=2, limit_recent=10)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].content, "Вопрос А")
        self.assertEqual(result[1].content, "Вопрос Б")

    def test_03_historical_raw_dedup_with_assistant_in_middle(self):
        base_ts = datetime.utcnow()
        messages = [
            DBMessage(id=1, role="user", content="Вопрос А", timestamp=base_ts),
            DBMessage(id=2, role="user", content="Вопрос А", timestamp=base_ts + timedelta(seconds=1)),
            DBMessage(id=3, role="assistant", content="Ответ 1", timestamp=base_ts + timedelta(seconds=2)),
            DBMessage(id=4, role="user", content="Вопрос А", timestamp=base_ts + timedelta(seconds=3)),
            DBMessage(id=5, role="user", content="Вопрос А", timestamp=base_ts + timedelta(seconds=4)),
        ]
        result = select_ai_history_messages(messages, limit_first=2, limit_recent=10)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].content, "Вопрос А")
        self.assertEqual(result[1].content, "Ответ 1")
        self.assertEqual(result[2].content, "Вопрос А")

    def test_04_visible_assistant_boundary(self):
        base_ts = datetime.utcnow()
        messages = [
            DBMessage(id=1, role="user", content="A", timestamp=base_ts),
            DBMessage(id=2, role="assistant", content="Ответ X", timestamp=base_ts + timedelta(seconds=1)),
            DBMessage(id=3, role="user", content="A", timestamp=base_ts + timedelta(seconds=2)),
        ]
        result = select_ai_history_messages(messages, limit_first=2, limit_recent=10)
        self.assertEqual(len(result), 3)
        self.assertEqual([m.content for m in result], ["A", "Ответ X", "A"])

    def test_05_test_a_data_only_assistant_is_dedup_boundary(self):
        base_ts = datetime.utcnow()
        messages = [
            DBMessage(id=1, role="user", content="A", timestamp=base_ts),
            DBMessage(id=2, role="user", content="A", timestamp=base_ts + timedelta(seconds=1)),
            DBMessage(id=3, role="assistant", content='<DATA>{"key": "val"}</DATA>', timestamp=base_ts + timedelta(seconds=2)),
            DBMessage(id=4, role="user", content="A", timestamp=base_ts + timedelta(seconds=3)),
            DBMessage(id=5, role="user", content="A", timestamp=base_ts + timedelta(seconds=4)),
        ]
        result = select_ai_history_messages(messages, limit_first=2, limit_recent=10)
        self.assertEqual(len(result), 2)
        self.assertEqual([m.content for m in result], ["A", "A"])

    def test_06_test_d_truncated_data_only_boundary(self):
        base_ts = datetime.utcnow()
        messages = [
            DBMessage(id=1, role="user", content="A", timestamp=base_ts),
            DBMessage(id=2, role="user", content="A", timestamp=base_ts + timedelta(seconds=1)),
            DBMessage(id=3, role="assistant", content='<DATA>{"key": "incomplete', timestamp=base_ts + timedelta(seconds=2)),
            DBMessage(id=4, role="user", content="A", timestamp=base_ts + timedelta(seconds=3)),
            DBMessage(id=5, role="user", content="A", timestamp=base_ts + timedelta(seconds=4)),
        ]
        result = select_ai_history_messages(messages, limit_first=2, limit_recent=10)
        self.assertEqual(len(result), 2)
        self.assertEqual([m.content for m in result], ["A", "A"])

    def test_07_pending_tail_dedup_simple(self):
        base_ts = datetime.utcnow()
        messages = [
            DBMessage(id=1, role="user", content="27 мая 1988", timestamp=base_ts),
            DBMessage(id=2, role="user", content="27 мая 1988", timestamp=base_ts + timedelta(seconds=1)),
            DBMessage(id=3, role="user", content="27 мая 1988", timestamp=base_ts + timedelta(seconds=2)),
        ]
        result = select_ai_history_messages(messages, limit_first=2, limit_recent=10, pending_user_content="27 мая 1988")
        self.assertEqual(len(result), 0)

    def test_08_test_b_data_only_blocks_pending_dedup(self):
        base_ts = datetime.utcnow()
        messages = [
            DBMessage(id=1, role="user", content="A", timestamp=base_ts),
            DBMessage(id=2, role="assistant", content='<DATA>{"status": "ok"}</DATA>', timestamp=base_ts + timedelta(seconds=1)),
        ]
        result = select_ai_history_messages(messages, limit_first=2, limit_recent=10, pending_user_content="A")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, "A")

    def test_09_test_c_visible_assistant_blocks_pending_dedup(self):
        base_ts = datetime.utcnow()
        messages = [
            DBMessage(id=1, role="user", content="A", timestamp=base_ts),
            DBMessage(id=2, role="assistant", content="Ответ X", timestamp=base_ts + timedelta(seconds=1)),
        ]
        result = select_ai_history_messages(messages, limit_first=2, limit_recent=10, pending_user_content="A")
        self.assertEqual(len(result), 2)
        self.assertEqual([m.content for m in result], ["A", "Ответ X"])

    def test_10_different_current_retains_history(self):
        base_ts = datetime.utcnow()
        messages = [
            DBMessage(id=1, role="user", content="A", timestamp=base_ts),
            DBMessage(id=2, role="user", content="A", timestamp=base_ts + timedelta(seconds=1)),
            DBMessage(id=3, role="user", content="A", timestamp=base_ts + timedelta(seconds=2)),
        ]
        result = select_ai_history_messages(messages, limit_first=2, limit_recent=10, pending_user_content="B")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, "A")

    def test_11_distinct_unanswered_retains_history(self):
        base_ts = datetime.utcnow()
        messages = [
            DBMessage(id=1, role="user", content="A", timestamp=base_ts),
            DBMessage(id=2, role="user", content="A", timestamp=base_ts + timedelta(seconds=1)),
            DBMessage(id=3, role="user", content="B", timestamp=base_ts + timedelta(seconds=2)),
            DBMessage(id=4, role="user", content="B", timestamp=base_ts + timedelta(seconds=3)),
        ]
        result = select_ai_history_messages(messages, limit_first=2, limit_recent=10, pending_user_content="C")
        self.assertEqual(len(result), 2)
        self.assertEqual([m.content for m in result], ["A", "B"])

    def test_12_system_event_boundary_not_deduped(self):
        base_ts = datetime.utcnow()
        messages = [
            DBMessage(id=1, role="user", content="A", timestamp=base_ts),
            DBMessage(id=2, role=SYSTEM_EVENT_ROLE, content="Пользователь перешел в тему", timestamp=base_ts + timedelta(seconds=1)),
            DBMessage(id=3, role="user", content="A", timestamp=base_ts + timedelta(seconds=2)),
        ]
        result = select_ai_history_messages(messages, limit_first=2, limit_recent=10)
        self.assertEqual(len(result), 3)
        self.assertEqual([m.content for m in result], ["A", "Пользователь перешел в тему", "A"])

    def test_13_test_result_boundary_not_deduped(self):
        base_ts = datetime.utcnow()
        messages = [
            DBMessage(id=1, role="user", content="A", timestamp=base_ts),
            DBMessage(id=2, role=TEST_RESULT_ROLE, content="Баллы: 10", timestamp=base_ts + timedelta(seconds=1)),
            DBMessage(id=3, role="user", content="A", timestamp=base_ts + timedelta(seconds=2)),
        ]
        result = select_ai_history_messages(messages, limit_first=2, limit_recent=10)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].content, "A")
        self.assertTrue("[РЕЗУЛЬТАТЫ ПРОЙДЕННОГО ТЕСТА]" in result[1].content)
        self.assertEqual(result[2].content, "A")

    def test_14_zero_context_limits(self):
        base_ts = datetime.utcnow()
        messages = [
            DBMessage(id=1, role="user", content="A", timestamp=base_ts),
            DBMessage(id=2, role="user", content="A", timestamp=base_ts + timedelta(seconds=1)),
        ]
        result = select_ai_history_messages(messages, limit_first=0, limit_recent=0)
        self.assertEqual(len(result), 0)

    def test_15_comparison_contract_whitespace_only(self):
        base_ts = datetime.utcnow()
        # "A" vs "A   " -> duplicate
        m1 = [
            DBMessage(id=1, role="user", content="A", timestamp=base_ts),
            DBMessage(id=2, role="user", content="A   ", timestamp=base_ts + timedelta(seconds=1)),
        ]
        self.assertEqual(len(select_ai_history_messages(m1, 2, 10)), 1)

        # "A" vs "a" -> distinct
        m2 = [
            DBMessage(id=1, role="user", content="A", timestamp=base_ts),
            DBMessage(id=2, role="user", content="a", timestamp=base_ts + timedelta(seconds=1)),
        ]
        self.assertEqual(len(select_ai_history_messages(m2, 2, 10)), 2)

        # "Привет" vs "Привет!" -> distinct
        m3 = [
            DBMessage(id=1, role="user", content="Привет", timestamp=base_ts),
            DBMessage(id=2, role="user", content="Привет!", timestamp=base_ts + timedelta(seconds=1)),
        ]
        self.assertEqual(len(select_ai_history_messages(m3, 2, 10)), 2)

    def test_16_orm_immutability_unit(self):
        base_ts = datetime.utcnow()
        msg1 = DBMessage(id=1, role="user", content="A", timestamp=base_ts)
        msg2 = DBMessage(id=2, role="user", content="A", timestamp=base_ts + timedelta(seconds=1))
        select_ai_history_messages([msg1, msg2], 2, 10)
        self.assertEqual(msg1.content, "A")
        self.assertEqual(msg1.id, 1)
        self.assertEqual(msg1.role, "user")
        self.assertEqual(msg2.content, "A")
        self.assertEqual(msg2.id, 2)
        self.assertEqual(msg2.role, "user")

    # -------------------------------------------------------------------------
    # Integration Tests with Database & Full Request Layout
    # -------------------------------------------------------------------------

    async def test_17_max_normal_text_retry_wire_payload_and_db_lifecycle(self):
        base_ts = datetime.utcnow()
        async with self.sessions() as session:
            # Three previous failed attempts
            m1 = DBMessage(id=101, user_id=9001, dialogue_id=1, topic_id=1, role="user", content="27 мая 1988", timestamp=base_ts)
            m2 = DBMessage(id=102, user_id=9001, dialogue_id=1, topic_id=1, role="user", content="27 мая 1988", timestamp=base_ts + timedelta(seconds=1))
            m3 = DBMessage(id=103, user_id=9001, dialogue_id=1, topic_id=1, role="user", content="27 мая 1988", timestamp=base_ts + timedelta(seconds=2))
            # Current turn persisted in DB by MAX before AI call
            m4 = DBMessage(id=104, user_id=9001, dialogue_id=1, topic_id=1, role="user", content="27 мая 1988", timestamp=base_ts + timedelta(seconds=3))
            session.add_all([m1, m2, m3, m4])
            await session.commit()

        async with self.sessions() as session:
            db_user = await session.get(User, 9001)
            ai_cfg = await session.get(AIConfig, 1)
            layout = await build_conversational_request_layout(
                session,
                user=db_user,
                ai_config=ai_cfg,
                dialogue_id=1,
                topic_id=1,
                exclude_message_id=104,
                current_user_content="27 мая 1988",
            )
            wire_messages = build_openai_chat_messages(layout)
            user_messages = [m for m in wire_messages if m["role"] == "user"]
            self.assertEqual(len(user_messages), 1)
            self.assertEqual(user_messages[0]["content"], "27 мая 1988")

            # Verify ORM DB rows unchanged
            rows = (await session.execute(select(DBMessage).where(DBMessage.user_id == 9001).order_by(DBMessage.id.asc()))).scalars().all()
            self.assertEqual(len(rows), 4)
            self.assertEqual([r.id for r in rows], [101, 102, 103, 104])

    async def test_18_max_normal_text_retry_enriched_prompt(self):
        base_ts = datetime.utcnow()
        async with self.sessions() as session:
            m1 = DBMessage(id=201, user_id=9001, dialogue_id=1, topic_id=1, role="user", content="Вопрос", timestamp=base_ts)
            m2 = DBMessage(id=202, user_id=9001, dialogue_id=1, topic_id=1, role="user", content="Вопрос", timestamp=base_ts + timedelta(seconds=1))
            m3 = DBMessage(id=203, user_id=9001, dialogue_id=1, topic_id=1, role="user", content="Вопрос", timestamp=base_ts + timedelta(seconds=2))
            session.add_all([m1, m2, m3])
            await session.commit()

        enriched_prompt = "Вопрос\n\nКонтекст для ответа: Психологическая метафора"
        async with self.sessions() as session:
            db_user = await session.get(User, 9001)
            ai_cfg = await session.get(AIConfig, 1)
            layout = await build_conversational_request_layout(
                session,
                user=db_user,
                ai_config=ai_cfg,
                dialogue_id=1,
                topic_id=1,
                exclude_message_id=203,
                current_user_content=enriched_prompt,
            )
            wire_messages = build_openai_chat_messages(layout)
            user_messages = [m for m in wire_messages if m["role"] == "user"]
            self.assertEqual(len(user_messages), 1)
            self.assertEqual(user_messages[0]["content"], enriched_prompt)

    async def test_19_max_vision_negative_discriminator(self):
        base_ts = datetime.utcnow()
        async with self.sessions() as session:
            m1 = DBMessage(id=301, user_id=9001, dialogue_id=1, topic_id=1, role="user", content="[Изображение] Описание", timestamp=base_ts)
            # Vision current row also has role="user"
            m_vision = DBMessage(id=302, user_id=9001, dialogue_id=1, topic_id=1, role="user", content="[Изображение] Описание", timestamp=base_ts + timedelta(seconds=1))
            session.add_all([m1, m_vision])
            await session.commit()

        async with self.sessions() as session:
            db_user = await session.get(User, 9001)
            ai_cfg = await session.get(AIConfig, 1)
            # Vision calls builder with current_user_content=None
            layout = await build_conversational_request_layout(
                session,
                user=db_user,
                ai_config=ai_cfg,
                dialogue_id=1,
                topic_id=1,
                exclude_message_id=302,
                current_user_content=None,
            )
            # current-tail dedup must be DISABLED: history should retain [Изображение] Описание
            self.assertEqual(len(layout.history), 1)
            self.assertEqual(layout.history[0].content, "[Изображение] Описание")

    async def test_20_max_hidden_kickoff_negative(self):
        base_ts = datetime.utcnow()
        async with self.sessions() as session:
            m_user = DBMessage(id=401, user_id=9001, dialogue_id=1, topic_id=1, role="user", content="Привет", timestamp=base_ts)
            nav_event = DBMessage(id=402, user_id=9001, dialogue_id=1, topic_id=1, role=SYSTEM_EVENT_ROLE, content="[НАВИГАЦИЯ]", timestamp=base_ts + timedelta(seconds=1))
            session.add_all([m_user, nav_event])
            await session.commit()

        async with self.sessions() as session:
            db_user = await session.get(User, 9001)
            ai_cfg = await session.get(AIConfig, 1)
            # Hidden kickoff with synthetic prompt and exclude_message_id pointing to navigation system_event
            layout = await build_conversational_request_layout(
                session,
                user=db_user,
                ai_config=ai_cfg,
                dialogue_id=1,
                topic_id=1,
                exclude_message_id=402,
                current_user_content="Привет",  # even if synthetic prompt coincides with historical text
            )
            # Historical user "Привет" must NOT be dropped
            history_user_messages = [m for m in layout.history if m.role == "user"]
            self.assertTrue(any(m.content == "Привет" for m in history_user_messages))

    async def test_21_telegram_scoped_kickoff_negative(self):
        base_ts = datetime.utcnow()
        async with self.sessions() as session:
            m_user = DBMessage(id=501, user_id=9001, dialogue_id=1, topic_id=1, role="user", content="Старт", timestamp=base_ts)
            nav_event = DBMessage(id=502, user_id=9001, dialogue_id=1, topic_id=1, role=SYSTEM_EVENT_ROLE, content="Автостарт", timestamp=base_ts + timedelta(seconds=1))
            session.add_all([m_user, nav_event])
            await session.commit()

        async with self.sessions() as session:
            db_user = await session.get(User, 9001)
            ai_cfg = await session.get(AIConfig, 1)
            layout = await build_conversational_request_layout(
                session,
                user=db_user,
                ai_config=ai_cfg,
                dialogue_id=1,
                topic_id=1,
                exclude_message_id=502,
                current_user_content="Старт",
            )
            history_user_messages = [m for m in layout.history if m.role == "user"]
            self.assertTrue(any(m.content == "Старт" for m in history_user_messages))

    async def test_22_telegram_failure_persistence_proof(self):
        # Verify that upon AI failure in Telegram, no DBMessage is inserted
        from handlers import process_buffered_messages, user_message_buffers
        bot_mock = MagicMock()
        bot_mock.send_chat_action = AsyncMock()
        bot_mock.send_message = AsyncMock()

        user_message_buffers[9001] = ["Тестовое сообщение Telegram"]

        with patch("ai_integration.generate_response", side_effect=ai_integration.AIServiceError("Simulated AI Failure")):
            await process_buffered_messages(9001, bot_mock)

        async with self.sessions() as session:
            msgs = (await session.execute(select(DBMessage).where(DBMessage.user_id == 9001))).scalars().all()
            self.assertEqual(len(msgs), 0)

    async def test_23_cross_scope_negatives(self):
        base_ts = datetime.utcnow()
        async with self.sessions() as session:
            m_target = DBMessage(id=601, user_id=9001, dialogue_id=1, topic_id=1, role="user", content="A", timestamp=base_ts)
            # Different user
            m_other_user = DBMessage(id=602, user_id=9999, dialogue_id=1, topic_id=1, role="user", content="A", timestamp=base_ts + timedelta(seconds=1))
            # Different dialogue
            m_other_dial = DBMessage(id=603, user_id=9001, dialogue_id=2, topic_id=1, role="user", content="A", timestamp=base_ts + timedelta(seconds=2))
            # Different topic
            m_other_topic = DBMessage(id=604, user_id=9001, dialogue_id=1, topic_id=2, role="user", content="A", timestamp=base_ts + timedelta(seconds=3))
            session.add_all([m_target, m_other_user, m_other_dial, m_other_topic])
            await session.commit()

        async with self.sessions() as session:
            db_user = await session.get(User, 9001)
            ai_cfg = await session.get(AIConfig, 1)

            # Case A: exclude_message_id of other user -> tail dedup disabled
            layout_a = await build_conversational_request_layout(
                session, user=db_user, ai_config=ai_cfg, dialogue_id=1, topic_id=1,
                exclude_message_id=602, current_user_content="A",
            )
            self.assertEqual(len([m for m in layout_a.history if m.role == "user"]), 1)

            # Case B: exclude_message_id of other dialogue -> tail dedup disabled
            layout_b = await build_conversational_request_layout(
                session, user=db_user, ai_config=ai_cfg, dialogue_id=1, topic_id=1,
                exclude_message_id=603, current_user_content="A",
            )
            self.assertEqual(len([m for m in layout_b.history if m.role == "user"]), 1)

            # Case C: exclude_message_id of other topic (in topic mode) -> tail dedup disabled
            layout_c = await build_conversational_request_layout(
                session, user=db_user, ai_config=ai_cfg, dialogue_id=1, topic_id=1,
                exclude_message_id=604, current_user_content="A",
            )
            self.assertEqual(len([m for m in layout_c.history if m.role == "user"]), 1)

    async def test_24_provider_serializers_parity_request_capture_and_ailog(self):
        base_ts = datetime.utcnow()
        async with self.sessions() as session:
            m1 = DBMessage(id=701, user_id=9001, dialogue_id=1, topic_id=1, role="user", content="Повтор", timestamp=base_ts)
            m2 = DBMessage(id=702, user_id=9001, dialogue_id=1, topic_id=1, role="user", content="Повтор", timestamp=base_ts + timedelta(seconds=1))
            session.add_all([m1, m2])
            await session.commit()

        captured_payloads = []

        async def fake_openai_create(**kwargs):
            captured_payloads.append(kwargs)
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "Успешный ответ ИИ"
            resp.choices = [choice]
            return resp

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_openai_create):
            response = await max_ai.get_ai_response(
                9001,
                "Повтор",
                exclude_message_id=702,
                track_user_activity=False,
            )
            self.assertEqual(response, "Успешный ответ ИИ")

        self.assertEqual(len(captured_payloads), 1)
        outbound_payload = captured_payloads[0]
        outbound_user_messages = [m for m in outbound_payload["messages"] if m["role"] == "user"]
        self.assertEqual(len(outbound_user_messages), 1)
        self.assertEqual(outbound_user_messages[0]["content"], "Повтор")

        # Serializer parity checks
        async with self.sessions() as session:
            db_user = await session.get(User, 9001)
            ai_cfg = await session.get(AIConfig, 1)
            layout = await build_conversational_request_layout(
                session, user=db_user, ai_config=ai_cfg, dialogue_id=1, topic_id=1,
                exclude_message_id=702, current_user_content="Повтор",
            )
            openai_msgs = build_openai_chat_messages(layout)
            gemini_contents = build_gemini_contents(layout)
            responses_input = build_responses_input(layout)

            self.assertEqual(len([m for m in openai_msgs if m["role"] == "user"]), 1)
            self.assertEqual(len([m for m in gemini_contents if m["role"] == "user"]), 1)
            self.assertEqual(len([m for m in responses_input if m["role"] == "user"]), 1)

            # AILog attempt-level audit verification
            ai_log = await session.scalar(select(AILog).order_by(AILog.id.desc()))
            self.assertIsNotNone(ai_log)
            capture = json.loads(ai_log.request_payload)
            self.assertEqual(capture["provider"], "OpenAI")
            self.assertIn("chat/completions", capture["endpoint"])
            self.assertEqual(capture["payload"]["messages"], outbound_payload["messages"])

    async def test_25_normal_text_flow_with_image_prefix_collapses_wire_to_single_turn(self):
        """Normal text flow where user typed '[Изображение] A' must deduplicate historical retries."""
        text_content = "[Изображение] A"
        base_ts = datetime.utcnow()
        async with self.sessions() as session:
            h1 = DBMessage(id=801, user_id=9001, dialogue_id=1, topic_id=1, role="user", content=text_content, timestamp=base_ts)
            h2 = DBMessage(id=802, user_id=9001, dialogue_id=1, topic_id=1, role="user", content=text_content, timestamp=base_ts + timedelta(seconds=1))
            current = DBMessage(id=803, user_id=9001, dialogue_id=1, topic_id=1, role="user", content=text_content, timestamp=base_ts + timedelta(seconds=2))
            session.add_all([h1, h2, current])
            await session.commit()

            db_user = await session.get(User, 9001)
            ai_cfg = await session.get(AIConfig, 1)

            layout = await build_conversational_request_layout(
                session,
                user=db_user,
                ai_config=ai_cfg,
                dialogue_id=1,
                topic_id=1,
                current_user_content=text_content,
                exclude_message_id=803,
            )

            # In normal text flow, history duplicates are superseded and collapsed
            self.assertEqual(len(layout.history), 0)

            # WIRE contains exactly one current user turn with "[Изображение] A"
            openai_wire = build_openai_chat_messages(layout)
            wire_user_msgs = [m for m in openai_wire if m["role"] == "user"]
            self.assertEqual(len(wire_user_msgs), 1)
            self.assertEqual(wire_user_msgs[0]["content"], text_content)
