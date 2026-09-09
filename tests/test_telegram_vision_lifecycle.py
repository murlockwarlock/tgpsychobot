import asyncio
import io
import json
import os
import tempfile
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("BOT_TOKEN", "test")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import ai_integration
import automation_engine
from database import (
    AIConfig,
    AILog,
    AutomationConversationState,
    AutomationMetadataRecord,
    Base,
    Message as DBMessage,
    SubscriptionConfig,
    Topic,
    User,
    UserAIActivity,
)
import handlers
from prompt_blocks import TELEGRAM_CAPABILITIES, DEFAULT_SERVICE_PROMPT_TEMPLATE
from ai_request_builder import build_conversational_request_layout, ActivityTracker


class TelegramVisionLifecycleIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        self._orig_handlers_sessions = handlers.async_session_maker
        self._orig_ai_sessions = ai_integration.async_session_maker
        handlers.async_session_maker = self.sessions
        ai_integration.async_session_maker = self.sessions

        async with self.sessions() as session:
            self.topic = Topic(
                id=1,
                name="Арт-терапия",
                is_active=True,
                system_prompt="Инструкция арт-терапевта.",
            )
            self.user = User(
                id=7001,
                first_name="Мария",
                gender="female",
                age=28,
                current_dialogue_id=1,
                current_topic_id=1,
                metadata_json="{}",
                accepted_disclaimer=True,
            )
            self.sub_config = SubscriptionConfig(
                id=1,
                subscriptions_enabled=False,
            )
            self.ai_config = AIConfig(
                id=1,
                provider="OpenAI",
                openai_api_key="sk-test-vision-key",
                openai_model="gpt-5.6-terra",
                vision_provider="OpenAI",
                vision_model="gpt-5.6-terra",
                gemini_api_key="sk-test-gemini",
                gemini_model="gemini-3.7-flash",
                claude_api_key="sk-test-claude",
                claude_model="claude-sonnet-5",
                kie_api_key="sk-test-kie",
                shared_prompt_block="Общий блок правил.",
                service_prompt_block=DEFAULT_SERVICE_PROMPT_TEMPLATE + "\n\nПРОТОКОЛ: <DATA>...",
                system_prompt="Общий системный промпт.",
                memory_mode="reset",
            )
            session.add_all([self.topic, self.user, self.sub_config, self.ai_config])
            await session.commit()

    async def asyncTearDown(self):
        handlers.async_session_maker = self._orig_handlers_sessions
        ai_integration.async_session_maker = self._orig_ai_sessions
        await self.engine.dispose()

    def _make_mock_bot(self):
        bot = MagicMock()
        bot.send_chat_action = AsyncMock()
        bot.get_file = AsyncMock(return_value=MagicMock(file_path="photos/sample.jpg"))
        bot.download_file = AsyncMock(return_value=io.BytesIO(b"fake_jpeg_image_bytes"))
        bot.delete_message = AsyncMock()
        bot.send_message = AsyncMock()
        return bot

    def _make_mock_msg(self, user_id=7001, caption="Разбери символ дерева и скажи, что он может означать"):
        msg = MagicMock()
        msg.chat.id = user_id
        msg.from_user.id = user_id
        msg.from_user.username = "maria_user"
        msg.from_user.full_name = "Мария"
        msg.caption = caption
        msg.photo = [MagicMock(file_id="photo_file_id_123")]
        msg.answer = AsyncMock(return_value=MagicMock(message_id=999))
        msg.answer_photo = AsyncMock()
        return msg

    # 1. Telegram vision sends real user caption, caption occurs once, ID excluded from history, identical past remains
    async def test_telegram_vision_outbound_real_caption_and_id_exclusion(self):
        caption = "Разбери символ дерева и скажи, что он может означать"

        # Pre-seed history with an identical past caption from a previous turn
        async with self.sessions() as session:
            past_msg = DBMessage(
                user_id=7001,
                dialogue_id=1,
                topic_id=1,
                role="user",
                content=f"[Фото для анализа] {caption}",
                timestamp=datetime(2026, 1, 1, 10, 0, 0),
            )
            session.add(past_msg)
            await session.commit()

        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption=caption)

        captured_calls = []
        async def fake_create(**kwargs):
            captured_calls.append(kwargs)
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "Разбор символа дерева: это символ жизни и роста."
            resp.choices = [choice]
            return resp

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create), \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        self.assertEqual(len(captured_calls), 1)
        wire_messages = captured_calls[0]["messages"]

        # Current user turn must be at the end, with actual user caption
        current_turn = wire_messages[-1]
        self.assertEqual(current_turn["role"], "user")
        self.assertIsInstance(current_turn["content"], list)
        self.assertEqual(current_turn["content"][0]["text"], caption)
        self.assertEqual(current_turn["content"][1]["type"], "image_url")

        # Historical identical caption remains in history
        history_msgs = [m for m in wire_messages if m["role"] != "system"][:-1]
        self.assertEqual(len(history_msgs), 1)
        self.assertIn(caption, history_msgs[0]["content"])

        # Entire outbound payload contains the caption exactly twice (1 historical, 1 current, NOT 3+)
        all_text_contents = []
        for m in wire_messages:
            if isinstance(m["content"], str):
                all_text_contents.append(m["content"])
            elif isinstance(m["content"], list):
                for p in m["content"]:
                    if isinstance(p, dict) and p.get("type") == "text":
                        all_text_contents.append(p.get("text", ""))
        caption_occurrences = sum(c.count(caption) for c in all_text_contents)
        self.assertEqual(caption_occurrences, 2)

        # Database rows: 2 user messages (1 historical + 1 current) and 1 assistant message
        async with self.sessions() as session:
            user_msgs = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 7001, DBMessage.role == "user"))).all()
            self.assertEqual(len(user_msgs), 2)
            assistant_msgs = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 7001, DBMessage.role == "assistant"))).all()
            self.assertEqual(len(assistant_msgs), 1)

    # 2. Telegram vision provider serializers: Gemini, Claude, KIE all receive caption as current user text
    async def test_telegram_vision_serializers_gemini_claude_kie(self):
        caption = "Разбери символ дерева и скажи, что он может означать"
        image_bytes = b"fake_bytes"

        # Build canonical layout
        async with self.sessions() as session:
            user = await session.get(User, 7001)
            cfg = await session.get(AIConfig, 1)
            layout = await build_conversational_request_layout(
                session,
                user=user,
                ai_config=cfg,
                dialogue_id=1,
                topic_id=1,
                current_user_content=caption,
                stable_system_prompt="Инструкция арт-терапевта.",
                service_capabilities=TELEGRAM_CAPABILITIES,
            )

        # Gemini serializer verification
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = MagicMock(status_code=200, json=lambda: {"candidates": [{"content": {"parts": [{"text": "Gemini response"}]}}]})
            await ai_integration._call_gemini_vision(
                "gemini-key", "gemini-3.7-flash", image_bytes, "prompt",
                request_layout=layout, effective_user_prompt=caption
            )
            gemini_payload = mock_post.call_args.kwargs["json"]
            gemini_contents = gemini_payload["contents"]
            self.assertEqual(gemini_contents[-1]["role"], "user")
            self.assertEqual(gemini_contents[-1]["parts"][0]["text"], caption)

        # Claude serializer verification
        with patch("anthropic.resources.messages.AsyncMessages.create", new_callable=AsyncMock) as mock_claude_create:
            mock_claude_create.return_value = MagicMock(content=[MagicMock(type="text", text="Claude response")])
            await ai_integration._call_claude_vision(
                "claude-key", "claude-sonnet-5", image_bytes, "prompt",
                request_layout=layout, effective_user_prompt=caption
            )
            claude_kwargs = mock_claude_create.call_args.kwargs
            claude_messages = claude_kwargs["messages"]
            self.assertEqual(claude_messages[-1]["role"], "user")
            # Text part must be caption
            user_parts = claude_messages[-1]["content"]
            text_part = next(p for p in user_parts if p.get("type") == "text")
            self.assertEqual(text_part["text"], caption)

        # KIE serializer verification
        with patch("ai_integration._upload_file_to_kie", new_callable=AsyncMock, return_value="https://files.kie.ai/image.jpg"), \
             patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_kie_post:
            mock_kie_post.return_value = MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "KIE response"}}]})
            await ai_integration._call_kie_vision(
                "kie-key", "https://api.kie.ai/v1", "https://upload.kie.ai", "gemini-3-flash",
                image_bytes, "prompt", request_layout=layout, effective_user_prompt=caption
            )
            kie_payload = mock_kie_post.call_args.kwargs["json"]
            kie_messages = kie_payload["messages"]
            self.assertEqual(kie_messages[-1]["role"], "user")
            self.assertEqual(kie_messages[-1]["content"][0]["text"], caption)

    # 3. Provider local validation failure: user photo Message durably persisted, activity = 0
    async def test_durable_user_photo_turn_local_validation_failure(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_model = "unsupported-model-x"
            await session.commit()

        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="Фото с локальной ошибкой модели")

        with patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        async with self.sessions() as session:
            # User photo turn committed before provider validation
            user_msgs = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 7001, DBMessage.role == "user"))).all()
            self.assertEqual(len(user_msgs), 1)
            self.assertIn("Фото с локальной ошибкой модели", user_msgs[0].content)

            # Assistant message was not created
            assistant_msgs = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 7001, DBMessage.role == "assistant"))).all()
            self.assertEqual(len(assistant_msgs), 0)

            # Activity was not recorded
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 7001))).all()
            self.assertEqual(len(acts), 0)

    # 4. Provider network timeout: user photo Message durably persisted, activity recorded = 1
    async def test_durable_user_photo_turn_network_timeout(self):
        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="Фото с сетевым таймаутом")

        async def fake_timeout(**kwargs):
            raise asyncio.TimeoutError("Network timeout during vision inference")

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_timeout), \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        async with self.sessions() as session:
            # User photo turn committed
            user_msgs = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 7001, DBMessage.role == "user"))).all()
            self.assertEqual(len(user_msgs), 1)
            self.assertIn("Фото с сетевым таймаутом", user_msgs[0].content)

            # Assistant message not created
            assistant_msgs = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 7001, DBMessage.role == "assistant"))).all()
            self.assertEqual(len(assistant_msgs), 0)

            # Activity was recorded once on outbound boundary before timeout
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 7001))).all()
            self.assertEqual(len(acts), 2)

    # 5. Telegram vision: valid DATA + AILog success commits both in the same transaction
    async def test_telegram_vision_data_and_ai_log_atomicity_success(self):
        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="Анализ с данными")

        ai_response_with_data = (
            "Разбор завершен отлично.\n"
            '<DATA>{"current_state":{"current_step":"analyzed"},"metadata":{"tree_type":"oak"}}</DATA>'
        )

        async def fake_create(**kwargs):
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = ai_response_with_data
            resp.choices = [choice]
            return resp

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create), \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        async with self.sessions() as session:
            # Both AILog and AutomationConversationState committed
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001))).all()
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0].prompt_summary, "Анализ с данными")
            self.assertEqual(logs[0].clean_text, "Разбор завершен отлично.")

            conv_state = await automation_engine.get_conversation_automation_state(
                session, user_id=7001, dialogue_id=1, topic_id=1
            )
            self.assertIsNotNone(conv_state)
            self.assertIn("analyzed", conv_state.current_state_json)

            # Assistant message exists outside the DATA transaction
            assistant_msgs = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 7001, DBMessage.role == "assistant"))).all()
            self.assertEqual(len(assistant_msgs), 1)

    # 6. Telegram vision: DATA apply failure rolls back DATA + AILog, raises clean AIServiceError, leaves assistant uncommitted
    async def test_telegram_vision_data_apply_failure_rolls_back_both(self):
        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="Анализ с падающим DATA")

        ai_response_with_data = (
            "Разбор завершен.\n"
            '<DATA>{"current_state":{"current_step":"broken"}}</DATA>'
        )

        async def fake_create(**kwargs):
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = ai_response_with_data
            resp.choices = [choice]
            return resp

        # Patch apply_service_data_blocks to simulate database error
        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create), \
             patch("handlers.apply_service_data_blocks", side_effect=RuntimeError("Simulated DB integrity failure")), \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        async with self.sessions() as session:
            # Neither AILog nor AutomationConversationState committed
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001))).all()
            self.assertEqual(len(logs), 0)

            conv_state = await automation_engine.get_conversation_automation_state(
                session, user_id=7001, dialogue_id=1, topic_id=1
            )
            self.assertIsNone(conv_state)

            # Assistant message was NOT committed
            assistant_msgs = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 7001, DBMessage.role == "assistant"))).all()
            self.assertEqual(len(assistant_msgs), 0)

            # User photo turn remains persisted
            user_msgs = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 7001, DBMessage.role == "user"))).all()
            self.assertEqual(len(user_msgs), 1)

    # 7. Telegram vision: stale scope guard between Transaction A commit and Transaction B builder
    async def test_telegram_vision_stale_scope_guard_aborts(self):
        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="Фото со сменой диалога")

        # Simulate user switching dialogue immediately after Transaction A commit
        orig_download = mock_bot.download_file
        async def switch_scope_during_download(*args, **kwargs):
            async with self.sessions() as session:
                u = await session.get(User, 7001)
                u.current_dialogue_id = 2  # Switched dialogue!
                await session.commit()
            return io.BytesIO(b"fake_jpeg")

        mock_bot.download_file = AsyncMock(side_effect=switch_scope_during_download)

        call_made = False
        async def fake_create(**kwargs):
            nonlocal call_made
            call_made = True
            resp = MagicMock()
            resp.choices = [MagicMock(message=MagicMock(content="Outbound reply"))]
            return resp

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create), \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        # Provider must NOT have been called
        self.assertFalse(call_made)

        # Assistant message must NOT have been created in dialogue 2
        async with self.sessions() as session:
            assistant_msgs = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 7001, DBMessage.role == "assistant"))).all()
            self.assertEqual(len(assistant_msgs), 0)

    # 8. File-backed SQLite concurrency: commit user photo turn, then ActivityTracker writes from separate session without "database is locked"
    async def test_sqlite_file_backed_photo_turn_and_activity_no_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_file_backed.db")
            file_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
            file_sessions = async_sessionmaker(file_engine, expire_on_commit=False)

            async with file_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            async with file_sessions() as session:
                topic = Topic(id=1, name="Арт", is_active=True, system_prompt="Инструкция.")
                user = User(id=8001, first_name="Иван", current_dialogue_id=1, current_topic_id=1, accepted_disclaimer=True)
                sub_cfg = SubscriptionConfig(id=1, subscriptions_enabled=False)
                cfg = AIConfig(id=1, provider="OpenAI", openai_api_key="sk-test", openai_model="gpt-5.6-terra", vision_provider="OpenAI", vision_model="gpt-5.6-terra")
                session.add_all([topic, user, sub_cfg, cfg])
                await session.commit()

            handlers.async_session_maker = file_sessions
            ai_integration.async_session_maker = file_sessions

            mock_bot = self._make_mock_bot()
            mock_msg = self._make_mock_msg(user_id=8001, caption="Тест файловой SQLite БД")

            async def fake_create(**kwargs):
                resp = MagicMock()
                choice = MagicMock()
                choice.message.content = "Успешный ответ на файловой БД"
                resp.choices = [choice]
                return resp

            with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create), \
                 patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
                # This must NOT raise "OperationalError: database is locked"
                await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

            async with file_sessions() as session:
                user_msgs = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 8001, DBMessage.role == "user"))).all()
                self.assertEqual(len(user_msgs), 1)
                assistant_msgs = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 8001, DBMessage.role == "assistant"))).all()
                self.assertEqual(len(assistant_msgs), 1)
                acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 8001))).all()
                self.assertEqual(len(acts), 2)

            await file_engine.dispose()

    # 9. Empty caption uses canonical default "Опиши это изображение подробно."
    async def test_telegram_vision_empty_caption_uses_canonical_default(self):
        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption=None)

        captured_calls = []
        async def fake_create(**kwargs):
            captured_calls.append(kwargs)
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "Разбор без подписи."
            resp.choices = [choice]
            return resp

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create), \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        self.assertEqual(len(captured_calls), 1)
        wire_messages = captured_calls[0]["messages"]
        current_turn = wire_messages[-1]
        self.assertEqual(current_turn["role"], "user")
        self.assertEqual(current_turn["content"][0]["text"], "Опиши это изображение подробно.")
        self.assertNotIn("You are a professional expert analyst", current_turn["content"][0]["text"])
        self.assertNotIn("Проанализируй это изображение", current_turn["content"][0]["text"])

    # 10. Successful request: exactly one user photo row and exactly one assistant row
    async def test_successful_photo_turn_one_user_one_assistant(self):
        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="Один пользовательский запрос")

        async def fake_create(**kwargs):
            resp = MagicMock()
            resp.choices = [MagicMock(message=MagicMock(content="Один ассистентский ответ"))]
            return resp

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create), \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        async with self.sessions() as session:
            user_msgs = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 7001, DBMessage.role == "user"))).all()
            self.assertEqual(len(user_msgs), 1)
            self.assertEqual(user_msgs[0].content, "[Фото для анализа] Один пользовательский запрос")

            assistant_msgs = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 7001, DBMessage.role == "assistant"))).all()
            self.assertEqual(len(assistant_msgs), 1)
            self.assertEqual(assistant_msgs[0].content, "Один ассистентский ответ")

    # 11. Transaction failure during commit with valid DATA: neither DATA nor AILog partially committed
    async def test_telegram_vision_commit_failure_with_valid_data_rolls_back_both(self):
        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="Анализ с ошибкой коммита")

        ai_response_with_data = (
            "Разбор завершен.\n"
            '<DATA>{"current_state":{"current_step":"commit_fail"}}</DATA>'
        )

        async def fake_create(**kwargs):
            resp = MagicMock()
            resp.choices = [MagicMock(message=MagicMock(content=ai_response_with_data))]
            return resp

        # Patch session.commit in Transaction C to fail
        orig_session_maker = self.sessions
        class FailingSessionWrapper:
            def __init__(self, real_session):
                self._real = real_session
                self.commit_attempts = 0
            def __getattr__(self, name):
                return getattr(self._real, name)
            async def commit(self):
                self.commit_attempts += 1
                # Fail specifically during Transaction C (when AILog/DATA are in session.new)
                for obj in self._real.new:
                    if isinstance(obj, AILog):
                        raise RuntimeError("Simulated commit failure on AILog+DATA")
                return await self._real.commit()

        def custom_session_maker():
            real_sess = orig_session_maker()
            wrapper = FailingSessionWrapper(real_sess)
            async def _enter():
                await real_sess.__aenter__()
                return wrapper
            async def _exit(*args):
                return await real_sess.__aexit__(*args)
            ctx = MagicMock()
            ctx.__aenter__ = _enter
            ctx.__aexit__ = _exit
            return ctx

        handlers.async_session_maker = custom_session_maker
        try:
            with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create), \
                 patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
                await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)
        finally:
            handlers.async_session_maker = orig_session_maker

        async with self.sessions() as session:
            # Neither DATA nor AILog partially committed
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001))).all()
            self.assertEqual(len(logs), 0)

            conv_state = await automation_engine.get_conversation_automation_state(
                session, user_id=7001, dialogue_id=1, topic_id=1
            )
            self.assertIsNone(conv_state)

            # Assistant message was not committed
            assistant_msgs = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 7001, DBMessage.role == "assistant"))).all()
            self.assertEqual(len(assistant_msgs), 0)

    # 12. Canonical automation context: TG normal, MAX normal, TG vision, MAX vision all use automation_engine
    async def test_canonical_automation_context_across_platforms(self):
        from max_messenger_bot import ai as max_ai
        mock_ctx = AsyncMock(return_value="СЦЕНАРИЙ_АВТОМАТИЗАЦИИ_КАНОНИЧЕСКИЙ")

        with patch("automation_engine.build_runtime_automation_context", mock_ctx):
            # TG normal
            async with self.sessions() as session:
                u = await session.get(User, 7001)
                cfg = await session.get(AIConfig, 1)
                tg_norm_layout = await build_conversational_request_layout(
                    session, user=u, ai_config=cfg, dialogue_id=1, topic_id=1,
                    service_capabilities=TELEGRAM_CAPABILITIES,
                )
                self.assertIn("СЦЕНАРИЙ_АВТОМАТИЗАЦИИ_КАНОНИЧЕСКИЙ", tg_norm_layout.scenario_context)

                # TG vision
                tg_vis_layout = await build_conversational_request_layout(
                    session, user=u, ai_config=cfg, dialogue_id=1, topic_id=1,
                    current_user_content="Тестовое фото",
                    service_capabilities=TELEGRAM_CAPABILITIES,
                )
                self.assertIn("СЦЕНАРИЙ_АВТОМАТИЗАЦИИ_КАНОНИЧЕСКИЙ", tg_vis_layout.scenario_context)

                # MAX normal
                from prompt_blocks import MAX_CAPABILITIES
                max_norm_layout = await build_conversational_request_layout(
                    session, user=u, ai_config=cfg, dialogue_id=1, topic_id=1,
                    service_capabilities=MAX_CAPABILITIES,
                )
                self.assertIn("СЦЕНАРИЙ_АВТОМАТИЗАЦИИ_КАНОНИЧЕСКИЙ", max_norm_layout.scenario_context)

                # MAX vision
                max_vis_layout = await build_conversational_request_layout(
                    session, user=u, ai_config=cfg, dialogue_id=1, topic_id=1,
                    current_user_content="Тестовое фото MAX",
                    service_capabilities=MAX_CAPABILITIES,
                )
                self.assertIn("СЦЕНАРИЙ_АВТОМАТИЗАЦИИ_КАНОНИЧЕСКИЙ", max_vis_layout.scenario_context)

        self.assertEqual(mock_ctx.call_count, 4)

    # 13. KIE preferred transient failure -> fallback model succeeds -> AILog.model matches fallback request payload
    async def test_telegram_vision_ailog_kie_fallback_model_matches_successful_attempt(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "KIE"
            cfg.vision_model = "gemini-3-flash"
            cfg.kie_api_key = "sk-test-kie-key"
            await session.commit()

        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="KIE фолбэк тест")

        attempt_models = []
        async def fake_kie_post(url, headers=None, json=None, **kwargs):
            model_called = json.get("model") if json else None
            attempt_models.append(model_called)
            if model_called == "gemini-3-flash":
                return MagicMock(status_code=503, json=lambda: {"error": {"message": "Server temporarily unavailable - 503"}})
            return MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "Успешный ответ от KIE fallback"}}]})

        with patch("ai_integration._upload_file_to_kie", new_callable=AsyncMock, return_value="https://files.kie.ai/test.jpg"), \
             patch("httpx.AsyncClient.post", side_effect=fake_kie_post), \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        self.assertIn("gemini-3-flash", attempt_models)
        self.assertIn("gemini-2.5-flash", attempt_models)

        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001))).all()
            self.assertEqual(len(logs), 1)
            ai_log = logs[0]
            self.assertEqual(ai_log.provider, "KIE")
            self.assertEqual(ai_log.model, "gemini-2.5-flash")
            captured = json.loads(ai_log.request_payload)
            self.assertEqual(captured["provider"], "KIE")
            self.assertEqual(captured["payload"]["model"], "gemini-2.5-flash")

    # 14. OpenAI vision_model="" (empty) -> default model used -> AILog.model matches request payload model
    async def test_telegram_vision_ailog_openai_empty_model_uses_actual_default(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "OpenAI"
            cfg.vision_model = ""  # Empty string in production schema
            await session.commit()

        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="OpenAI дефолтная модель тест")

        captured_wire = []
        async def fake_openai_create(**kwargs):
            captured_wire.append(kwargs)
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "Успешный ответ от OpenAI default"
            resp.choices = [choice]
            return resp

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_openai_create), \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        self.assertEqual(len(captured_wire), 1)
        self.assertEqual(captured_wire[0]["model"], "gpt-5.6-terra")

        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001))).all()
            self.assertEqual(len(logs), 1)
            ai_log = logs[0]
            self.assertEqual(ai_log.provider, "OpenAI")
            self.assertEqual(ai_log.model, "gpt-5.6-terra")
            self.assertNotEqual(ai_log.model, "Vision")
            captured = json.loads(ai_log.request_payload)
            self.assertEqual(captured["provider"], "OpenAI")
            self.assertEqual(captured["payload"]["model"], "gpt-5.6-terra")

    # 15. Gemini vision_model="" (empty) -> actual default model logged from endpoint
    async def test_telegram_vision_ailog_gemini_empty_model_uses_actual_default_from_endpoint(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "Gemini"
            cfg.vision_model = ""  # Empty string in production schema
            await session.commit()

        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="Gemini дефолтная модель тест")

        captured_urls = []
        async def fake_gemini_post(url, headers=None, json=None, **kwargs):
            captured_urls.append(url)
            return MagicMock(status_code=200, json=lambda: {"candidates": [{"content": {"parts": [{"text": "Ответ Gemini default"}]}}]})

        with patch("httpx.AsyncClient.post", side_effect=fake_gemini_post), \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        self.assertEqual(len(captured_urls), 1)
        self.assertIn("models/gemini-3.7-flash:generateContent", captured_urls[0])

        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001))).all()
            self.assertEqual(len(logs), 1)
            ai_log = logs[0]
            self.assertEqual(ai_log.provider, "Gemini")
            self.assertEqual(ai_log.model, "gemini-3.7-flash")
            self.assertNotEqual(ai_log.model, "Vision")
            captured = json.loads(ai_log.request_payload)
            self.assertEqual(captured["provider"], "Gemini")
            self.assertIn("models/gemini-3.7-flash", captured["endpoint"])

    # 16. Claude vision_model="" (empty) -> actual default model logged
    async def test_telegram_vision_ailog_claude_empty_model_uses_actual_default(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "Claude"
            cfg.vision_model = ""  # Empty string in production schema
            await session.commit()

        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="Claude дефолтная модель тест")

        captured_claude = []
        async def fake_claude_create(**kwargs):
            captured_claude.append(kwargs)
            return MagicMock(content=[MagicMock(type="text", text="Ответ Claude default")])

        with patch("anthropic.resources.messages.AsyncMessages.create", side_effect=fake_claude_create), \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        self.assertEqual(len(captured_claude), 1)
        self.assertEqual(captured_claude[0]["model"], "claude-sonnet-5")

        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001))).all()
            self.assertEqual(len(logs), 1)
            ai_log = logs[0]
            self.assertEqual(ai_log.provider, "Claude")
            self.assertEqual(ai_log.model, "claude-sonnet-5")
            self.assertNotEqual(ai_log.model, "Vision")
            captured = json.loads(ai_log.request_payload)
            self.assertEqual(captured["provider"], "Claude")
            self.assertEqual(captured["payload"]["model"], "claude-sonnet-5")

    # 16b. Defensive in-memory None model handling across handler and resolver without schema modification
    async def test_defensive_in_memory_none_model_resolver_and_handler(self):
        from ai_request_context import extract_effective_provider_and_model

        # Unit level: None, "", and "Vision" all resolve to canonical provider defaults
        for missing_val in (None, "", "Vision"):
            p, m = extract_effective_provider_and_model(None, default_provider="OpenAI", default_model=missing_val, channel="vision")
            self.assertEqual(p, "OpenAI")
            self.assertEqual(m, "gpt-5.6-terra")

            p, m = extract_effective_provider_and_model(None, default_provider="Gemini", default_model=missing_val, channel="vision")
            self.assertEqual(p, "Gemini")
            self.assertEqual(m, "gemini-3.7-flash")

            p, m = extract_effective_provider_and_model(None, default_provider="Claude", default_model=missing_val, channel="vision")
            self.assertEqual(p, "Claude")
            self.assertEqual(m, "claude-sonnet-5")

            p, m = extract_effective_provider_and_model(None, default_provider="KIE", default_model=missing_val, channel="vision")
            self.assertEqual(p, "KIE")
            self.assertEqual(m, "gemini-3-flash")

        # Handler level: in-memory mock config with vision_model=None
        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="OpenAI in-memory None model тест")

        # Intercept AIConfig load in handler to return object with vision_model=None
        orig_session_maker = handlers.async_session_maker
        class InterceptingSessionWrapper:
            def __init__(self, real_session):
                self._real = real_session
            def __getattr__(self, name):
                return getattr(self._real, name)
            async def get(self, entity, ident, **kwargs):
                obj = await self._real.get(entity, ident, **kwargs)
                if entity is AIConfig and obj is not None:
                    # Return clone/proxy with in-memory vision_model=None
                    proxy = SimpleNamespace(**{c.key: getattr(obj, c.key) for c in obj.__table__.columns})
                    proxy.vision_provider = "OpenAI"
                    proxy.vision_model = None
                    proxy.current_topic = getattr(obj, "current_topic", None)
                    return proxy
                return obj

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def intercepting_maker():
            async with orig_session_maker() as real_sess:
                yield InterceptingSessionWrapper(real_sess)

        handlers.async_session_maker = intercepting_maker
        try:
            async def fake_create(**kwargs):
                resp = MagicMock()
                choice = MagicMock()
                choice.message.content = "Ответ от OpenAI с in-memory None моделью"
                resp.choices = [choice]
                return resp

            with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create), \
                 patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
                await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)
        finally:
            handlers.async_session_maker = orig_session_maker

        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001))).all()
            self.assertEqual(len(logs), 1)
            ai_log = logs[0]
            self.assertEqual(ai_log.provider, "OpenAI")
            self.assertEqual(ai_log.model, "gpt-5.6-terra")
            self.assertNotEqual(ai_log.model, "Vision")

    # 17. Stale scope during inference: user photo Message durably persisted, activity recorded, no visible text, no buttons, no assistant Message
    async def test_telegram_vision_stale_scope_during_inference_skips_assistant_and_buttons(self):
        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="Фото со сменой темы во время инференса")

        call_made = False
        async def fake_openai_create(**kwargs):
            nonlocal call_made
            call_made = True
            async with self.sessions() as session:
                u = await session.get(User, 7001)
                u.current_topic_id = 2
                u.current_dialogue_id = 2
                await session.commit()

            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "Разбор рисунка во время смены темы\n[BTN: Подробнее | act_details]"
            resp.choices = [choice]
            return resp

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_openai_create), \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        self.assertTrue(call_made)

        async with self.sessions() as session:
            # Activity recorded because real outbound call happened
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 7001))).all()
            self.assertEqual(len(acts), 2)
            self.assertTrue(any(a.scope_key == "topic:1" for a in acts))
            self.assertTrue(any(a.scope_key == "global" for a in acts))

            # Inbound user photo Message remains durably persisted in original scope (topic 1, dialogue 1)
            user_msgs = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 7001, DBMessage.role == "user"))).all()
            self.assertEqual(len(user_msgs), 1)
            self.assertEqual(user_msgs[0].topic_id, 1)
            self.assertEqual(user_msgs[0].dialogue_id, 1)

            # NO assistant Message persisted in any scope
            asst_msgs = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 7001, DBMessage.role == "assistant"))).all()
            self.assertEqual(len(asst_msgs), 0)

            # AILog was persisted with request-time scope
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001))).all()
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0].topic_id, 1)

        # NO visible assistant text or response buttons sent
        for call_args in mock_msg.answer.call_args_list:
            text_arg = call_args[0][0] if call_args[0] else call_args[1].get("text", "")
            self.assertNotIn("Разбор рисунка во время смены темы", text_arg)
            reply_markup = call_args[1].get("reply_markup")
            self.assertIsNone(reply_markup)

    # 18. Secondary generation race: AI response contains GEN_IMG, user switches topic during generate_image -> media and assistant suppressed
    async def test_telegram_vision_stale_scope_during_secondary_gen_img_skips_media_and_assistant(self):
        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="Фото с директивой генерации")

        ai_response_text = "Вот подробный разбор вашего фото.\nGEN_IMG: a tranquil mountain lake"
        async def fake_openai_create(**kwargs):
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = ai_response_text
            resp.choices = [choice]
            return resp

        gen_called = False
        async def fake_generate_image(prompt):
            nonlocal gen_called
            gen_called = True
            # User switches topic in a separate session while generate_image is running
            async with self.sessions() as session:
                u = await session.get(User, 7001)
                u.current_topic_id = 2
                await session.commit()
            return b"fake_png_generated_data"

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_openai_create), \
             patch("ai_integration.generate_image", side_effect=fake_generate_image), \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        self.assertTrue(gen_called)

        # Generated photo must NOT have been sent
        mock_msg.answer_photo.assert_not_called()

        # Assistant DBMessage must NOT have been persisted
        async with self.sessions() as session:
            asst_msgs = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 7001, DBMessage.role == "assistant"))).all()
            self.assertEqual(len(asst_msgs), 0)

