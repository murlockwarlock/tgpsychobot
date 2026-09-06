import json
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ai_request_context import AIRequestLayout
from automation_engine import get_conversation_automation_state
from database import (
    AIConfig,
    AILog,
    AutomationConversationState,
    AutomationEvent,
    AutomationMetadataRecord,
    AutomationStepTransition,
    Base,
    Message as DBMessage,
    Topic,
    User,
)
from max_messenger_bot import ai as max_ai
from max_messenger_bot.services import common
from user_metadata import load_metadata, load_metadata_records


class _MockMaxApiClient:
    def __init__(self):
        self.sent_messages = []
        self.edited_messages = []
        self.mid_counter = 100

    async def send_message(self, chat_id, text, **kwargs):
        self.mid_counter += 1
        mid = f"mid_{self.mid_counter}"
        self.sent_messages.append({"chat_id": chat_id, "text": text, "mid": mid, "kwargs": kwargs})
        return {"message": {"mid": mid}}

    async def edit_message(self, message_id, text, **kwargs):
        self.edited_messages.append({"message_id": message_id, "text": text, "kwargs": kwargs})
        return {"ok": True}

    async def delete_message(self, message_id):
        return {"ok": True}

    async def upload_file(self, file_type, file_path):
        return {"token": "file_tok_123"}

    async def send_media_attachment(self, chat_id, media_type, token, **kwargs):
        return {"ok": True}


class MaxDataServiceParityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

        self._orig_max_ai_sessions = max_ai.async_session_maker
        self._orig_common_sessions = common.async_session_maker
        max_ai.async_session_maker = self.sessions
        common.async_session_maker = self.sessions

        async with self.sessions() as session:
            self.user_id = 10001
            user = User(
                id=self.user_id,
                first_name="Иван",
                current_dialogue_id=1,
                current_topic_id=None,
                metadata_json="{}",
            )
            ai_config = AIConfig(
                id=1,
                provider="OpenAI",
                openai_api_key="sk-test-openai",
                openai_model="gpt-5.6-terra",
                system_prompt="Ты полезный ИИ-помощник.",
                vision_provider="OpenAI",
                vision_model="gpt-5.6-terra",
            )
            session.add(user)
            session.add(ai_config)
            await session.commit()

    async def asyncTearDown(self):
        max_ai.async_session_maker = self._orig_max_ai_sessions
        common.async_session_maker = self._orig_common_sessions
        await self.engine.dispose()

    # -----------------------------------------------------------------------
    # 1. Unified <DATA> + visible text
    # -----------------------------------------------------------------------
    async def test_max_unified_data_block_extraction_and_application(self):
        mock_response = 'Здравствуйте!<DATA>{"current_state": {"step": "greeting"}}</DATA>'
        with patch.object(max_ai, "_dispatch_provider", AsyncMock(return_value=mock_response)):
            result = await max_ai.get_ai_response(self.user_id, "Привет")

        self.assertEqual(result, "Здравствуйте!")

        async with self.sessions() as session:
            ai_logs = (await session.execute(select(AILog).where(AILog.user_id == self.user_id))).scalars().all()
            self.assertEqual(len(ai_logs), 1)
            self.assertEqual(ai_logs[0].raw_response, mock_response)
            self.assertEqual(ai_logs[0].clean_text, "Здравствуйте!")

            state = await get_conversation_automation_state(session, user_id=self.user_id, dialogue_id=1, topic_id=None)
            self.assertIsNotNone(state)
            self.assertEqual(load_metadata(state.current_state_json), {"step": "greeting"})

    # -----------------------------------------------------------------------
    # 2. Legacy [DATA]
    # -----------------------------------------------------------------------
    async def test_max_legacy_data_block_extraction_and_application(self):
        mock_response = 'Ваш ответ.[DATA]{"mood":"calm"}[/DATA]'
        with patch.object(max_ai, "_dispatch_provider", AsyncMock(return_value=mock_response)):
            result = await max_ai.get_ai_response(self.user_id, "Как настроение?")

        self.assertEqual(result, "Ваш ответ.")

        async with self.sessions() as session:
            ai_logs = (await session.execute(select(AILog).where(AILog.user_id == self.user_id))).scalars().all()
            self.assertEqual(len(ai_logs), 1)
            self.assertEqual(ai_logs[0].raw_response, mock_response)
            self.assertEqual(ai_logs[0].clean_text, "Ваш ответ.")

            user = await session.get(User, self.user_id)
            records = load_metadata_records(user.metadata_json)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["data"], {"mood": "calm"})

    # -----------------------------------------------------------------------
    # 3. Malformed DATA
    # -----------------------------------------------------------------------
    async def test_max_malformed_data_block_handling(self):
        mock_response = "Текст ответа.<DATA>{invalid json</DATA>"
        with patch.object(max_ai, "_dispatch_provider", AsyncMock(return_value=mock_response)):
            result = await max_ai.get_ai_response(self.user_id, "Вопрос")

        self.assertEqual(result, "Текст ответа.")

        async with self.sessions() as session:
            ai_logs = (await session.execute(select(AILog).where(AILog.user_id == self.user_id))).scalars().all()
            self.assertEqual(len(ai_logs), 1)
            self.assertEqual(ai_logs[0].raw_response, mock_response)
            self.assertEqual(ai_logs[0].clean_text, "Текст ответа.")

            state = await get_conversation_automation_state(session, user_id=self.user_id, dialogue_id=1, topic_id=None)
            self.assertTrue(state is None or load_metadata(state.current_state_json) == {})

    # -----------------------------------------------------------------------
    # 4. DATA-only response
    # -----------------------------------------------------------------------
    async def test_max_data_only_response_returns_empty_string(self):
        mock_response = '<DATA>{"current_state": {"step": "analysis_done"}}</DATA>'
        with patch.object(max_ai, "_dispatch_provider", AsyncMock(return_value=mock_response)):
            result = await max_ai.get_ai_response(self.user_id, "Вопрос")

        self.assertEqual(result, "")

        async with self.sessions() as session:
            ai_logs = (await session.execute(select(AILog).where(AILog.user_id == self.user_id))).scalars().all()
            self.assertEqual(len(ai_logs), 1)
            self.assertEqual(ai_logs[0].raw_response, mock_response)
            self.assertEqual(ai_logs[0].clean_text, "")

            state = await get_conversation_automation_state(session, user_id=self.user_id, dialogue_id=1, topic_id=None)
            self.assertIsNotNone(state)
            self.assertEqual(load_metadata(state.current_state_json), {"step": "analysis_done"})

    # -----------------------------------------------------------------------
    # 5. Service-data persistence failure
    # -----------------------------------------------------------------------
    async def test_max_service_data_persistence_failure_raises_and_rolls_back(self):
        mock_response = 'Ответ.<DATA>{"current_state": {"step": "fail_state"}}</DATA>'
        with patch.object(max_ai, "_dispatch_provider", AsyncMock(return_value=mock_response)), \
             patch.object(max_ai, "apply_service_data_blocks", AsyncMock(side_effect=RuntimeError("Database lock failure"))):
            with self.assertRaises(max_ai.AIServiceError) as cm:
                await max_ai.get_ai_response(self.user_id, "Вопрос")

        self.assertIn("Ошибка сохранения метаданных диалога", str(cm.exception))

    # -----------------------------------------------------------------------
    # 6. No-service-block AILog persistence failure (best-effort preservation)
    # -----------------------------------------------------------------------
    async def test_max_no_service_block_ailog_commit_failure_preserves_best_effort(self):
        mock_response = "Обычный ответ без сервисных данных"

        class _FailingSession:
            def __init__(self, real_session):
                self._real = real_session

            def __getattr__(self, name):
                return getattr(self._real, name)

            async def commit(self):
                raise RuntimeError("AILog commit failed")

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return await self._real.__aexit__(exc_type, exc_val, exc_tb)

        def _failing_session_maker():
            return _FailingSession(self.sessions())

        with patch.object(max_ai, "_dispatch_provider", AsyncMock(return_value=mock_response)), \
             patch.object(max_ai, "async_session_maker", _failing_session_maker):
            result = await max_ai.get_ai_response(self.user_id, "Вопрос")

        self.assertEqual(result, "Обычный ответ без сервисных данных")

    # -----------------------------------------------------------------------
    # 7. AutomationEvent persistence only (no Telegram event dispatch)
    # -----------------------------------------------------------------------
    async def test_max_automation_event_persistence_without_telegram_dispatch(self):
        mock_response = 'Готово!<DATA>{"events": [{"name": "user_onboarded"}]}</DATA>'
        with patch.object(max_ai, "_dispatch_provider", AsyncMock(return_value=mock_response)):
            result = await max_ai.get_ai_response(self.user_id, "Старт")

        self.assertEqual(result, "Готово!")

        async with self.sessions() as session:
            events = (await session.execute(select(AutomationEvent).where(AutomationEvent.user_id == self.user_id))).scalars().all()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].name, "user_onboarded")
            self.assertEqual(events[0].dialogue_id, 1)
            self.assertIsNone(events[0].processed_at)

    # -----------------------------------------------------------------------
    # 8. Request-time scope isolation
    # -----------------------------------------------------------------------
    async def test_max_request_time_scope_isolation(self):
        async with self.sessions() as session:
            user = await session.get(User, self.user_id)
            user.current_dialogue_id = 1
            user.current_topic_id = 7
            session.add(Topic(id=7, name="Тема 7"))
            session.add(Topic(id=88, name="Тема 88"))
            await session.commit()

        async def _mock_dispatch_mutating_scope(ai_config, layout, request_capture=None):
            async with self.sessions() as s:
                u = await s.get(User, self.user_id)
                u.current_dialogue_id = 99
                u.current_topic_id = 88
                await s.commit()
            return 'Ответ в теме.<DATA>{"current_state": {"step": "scoped_step"}}</DATA>'

        with patch.object(max_ai, "_dispatch_provider", _mock_dispatch_mutating_scope):
            result = await max_ai.get_ai_response(self.user_id, "Вопрос")

        self.assertEqual(result, "Ответ в теме.")

        async with self.sessions() as session:
            state_captured = await get_conversation_automation_state(session, user_id=self.user_id, dialogue_id=1, topic_id=7)
            self.assertIsNotNone(state_captured)
            self.assertEqual(load_metadata(state_captured.current_state_json), {"step": "scoped_step"})

            state_mutated = await get_conversation_automation_state(session, user_id=self.user_id, dialogue_id=99, topic_id=88)
            self.assertTrue(state_mutated is None or load_metadata(state_mutated.current_state_json) == {})

    # -----------------------------------------------------------------------
    # 9. Ordinary run_ai_dialogue assistant message and client clean
    # -----------------------------------------------------------------------
    async def test_max_run_ai_dialogue_assistant_dbmessage_and_client_clean(self):
        client = _MockMaxApiClient()
        mock_response = 'Привет, Иван!<DATA>{"current_state": {"step": "s1"}}</DATA>'
        with patch.object(max_ai, "_dispatch_provider", AsyncMock(return_value=mock_response)):
            await common.run_ai_dialogue(client, chat_id=12345, user_id=self.user_id, prompt_text="Привет")

        async with self.sessions() as session:
            messages = (
                await session.execute(
                    select(DBMessage).where(DBMessage.user_id == self.user_id, DBMessage.role == "assistant")
                )
            ).scalars().all()
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].content, "Привет, Иван!")
            self.assertNotIn("<DATA>", messages[0].content)

        self.assertEqual(len(client.edited_messages), 1)
        self.assertIn("Привет, Иван!", client.edited_messages[0]["text"])
        self.assertNotIn("<DATA>", client.edited_messages[0]["text"])

    # -----------------------------------------------------------------------
    # 10. Next-turn history isolation
    # -----------------------------------------------------------------------
    async def test_max_next_turn_history_isolation(self):
        client = _MockMaxApiClient()

        mock_turn1 = 'Ответ 1.<DATA>{"current_state": {"step": "s1"}}</DATA>'
        with patch.object(max_ai, "_dispatch_provider", AsyncMock(return_value=mock_turn1)):
            await common.run_ai_dialogue(client, chat_id=12345, user_id=self.user_id, prompt_text="Вопрос 1")

        captured_layouts = []

        async def _capture_turn2(config, layout, request_capture=None):
            captured_layouts.append(layout)
            return "Ответ 2."

        with patch.object(max_ai, "_dispatch_provider", _capture_turn2):
            await common.run_ai_dialogue(client, chat_id=12345, user_id=self.user_id, prompt_text="Вопрос 2")

        self.assertEqual(len(captured_layouts), 1)
        history = captured_layouts[0].history
        assistant_history = [msg.content for msg in history if getattr(msg, "role", None) == "assistant"]
        self.assertEqual(assistant_history, ["Ответ 1."])
        self.assertTrue(all("<DATA>" not in content for content in assistant_history))

    # -----------------------------------------------------------------------
    # 11. run_hidden_ai_kickoff data leak prevention
    # -----------------------------------------------------------------------
    async def test_max_hidden_ai_kickoff_data_leak_prevention(self):
        client = _MockMaxApiClient()

        mock_response = 'Продолжаем.<DATA>{"current_state": {"step": "resumed"}}</DATA>'
        with patch.object(max_ai, "_dispatch_provider", AsyncMock(return_value=mock_response)):
            await common.run_hidden_ai_kickoff(
                client,
                chat_id=12345,
                user_id=self.user_id,
                synthetic_prompt="[system kickoff]",
                expected_dialogue_id=1,
                expected_topic_id=None,
            )

        async with self.sessions() as session:
            messages = (
                await session.execute(
                    select(DBMessage).where(DBMessage.user_id == self.user_id, DBMessage.role == "assistant")
                )
            ).scalars().all()
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].content, "Продолжаем.")
            self.assertNotIn("<DATA>", messages[0].content)

            state = await get_conversation_automation_state(session, user_id=self.user_id, dialogue_id=1, topic_id=None)
            self.assertIsNotNone(state)
            self.assertEqual(load_metadata(state.current_state_json), {"step": "resumed"})

        self.assertEqual(len(client.edited_messages), 1)
        self.assertIn("Продолжаем.", client.edited_messages[0]["text"])
        self.assertNotIn("<DATA>", client.edited_messages[0]["text"])

    # -----------------------------------------------------------------------
    # 12. get_ai_response_direct text+DATA, DATA-only, and error handling
    # -----------------------------------------------------------------------
    async def test_max_direct_ai_data_only_and_data_bearing(self):
        # 1. Text + DATA
        mock_response_1 = 'Прямой ответ.<DATA>{"current_state": {"step": "direct_1"}}</DATA>'
        with patch.object(max_ai, "_dispatch_provider", AsyncMock(return_value=mock_response_1)):
            res1 = await max_ai.get_ai_response_direct(self.user_id, "Системный", "Промпт")
        self.assertEqual(res1, "Прямой ответ.")

        async with self.sessions() as session:
            state = await get_conversation_automation_state(session, user_id=self.user_id, dialogue_id=1, topic_id=None)
            self.assertIsNotNone(state)
            self.assertEqual(load_metadata(state.current_state_json), {"step": "direct_1"})

        # 2. DATA-only
        mock_response_2 = '<DATA>{"current_state": {"step": "direct_2"}}</DATA>'
        with patch.object(max_ai, "_dispatch_provider", AsyncMock(return_value=mock_response_2)):
            res2 = await max_ai.get_ai_response_direct(self.user_id, "Системный", "Промпт")
        self.assertEqual(res2, "")

        async with self.sessions() as session:
            state = await get_conversation_automation_state(session, user_id=self.user_id, dialogue_id=1, topic_id=None)
            self.assertIsNotNone(state)
            self.assertEqual(load_metadata(state.current_state_json), {"step": "direct_2"})

        # 3. Persistence error rolls back and raises AIServiceError
        mock_response_3 = 'Ответ 3.<DATA>{"current_state": {"step": "direct_3"}}</DATA>'
        with patch.object(max_ai, "_dispatch_provider", AsyncMock(return_value=mock_response_3)), \
             patch.object(max_ai, "apply_service_data_blocks", AsyncMock(side_effect=RuntimeError("Direct metadata save error"))):
            with self.assertRaises(max_ai.AIServiceError) as cm:
                await max_ai.get_ai_response_direct(self.user_id, "Системный", "Промпт")
        self.assertIn("Ошибка сохранения метаданных диалога", str(cm.exception))

    # -----------------------------------------------------------------------
    # 13. analyze_image and run_ai_dialogue_with_image
    # -----------------------------------------------------------------------
    async def test_max_analyze_image_and_run_ai_dialogue_with_image_data_handling(self):
        client = _MockMaxApiClient()

        # 1. analyze_image with metadata
        mock_vision_1 = 'Разбор фото.<DATA>{"metadata": {"photo_topic": "nature"}}</DATA>'
        with patch.object(max_ai, "_analyze_openai", AsyncMock(return_value=mock_vision_1)):
            res_vis = await max_ai.analyze_image(self.user_id, b"fake_bytes", "Опиши фото")
        self.assertEqual(res_vis, "Разбор фото.")

        async with self.sessions() as session:
            user = await session.get(User, self.user_id)
            records = load_metadata_records(user.metadata_json)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["data"]["metadata"], {"photo_topic": "nature"})

            state = await get_conversation_automation_state(session, user_id=self.user_id, dialogue_id=1, topic_id=None)
            self.assertIsNotNone(state)
            self.assertEqual(load_metadata(state.metadata_json), {"photo_topic": "nature"})

        # 2. run_ai_dialogue_with_image end-to-end clean DB & UI
        mock_vision_2 = 'Разбор фото 2.<DATA>{"current_state": {"step": "photo_done"}}</DATA>'
        with patch.object(max_ai, "_analyze_openai", AsyncMock(return_value=mock_vision_2)):
            await common.run_ai_dialogue_with_image(client, chat_id=12345, user_id=self.user_id, image_bytes=b"fake_bytes", caption="Опиши")

        async with self.sessions() as session:
            messages = (
                await session.execute(
                    select(DBMessage).where(DBMessage.user_id == self.user_id, DBMessage.role == "assistant")
                )
            ).scalars().all()
            self.assertGreaterEqual(len(messages), 1)
            self.assertEqual(messages[-1].content, "Разбор фото 2.")
            self.assertNotIn("<DATA>", messages[-1].content)

        # 3. DATA-only vision response
        mock_vision_3 = '<DATA>{"current_state": {"step": "silent_vision"}}</DATA>'
        with patch.object(max_ai, "_analyze_openai", AsyncMock(return_value=mock_vision_3)):
            res_vis_empty = await max_ai.analyze_image(self.user_id, b"fake_bytes", "Опиши фото")
        self.assertEqual(res_vis_empty, "")
