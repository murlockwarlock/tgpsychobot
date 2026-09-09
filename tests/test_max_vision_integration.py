import asyncio
import json
import os
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("BOT_TOKEN", "test")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from database import (
    AIConfig,
    Base,
    Message as DBMessage,
    Topic,
    User,
    UserAIActivity,
)
from max_messenger_bot import ai as max_ai
from max_messenger_bot.services import common as max_common
from prompt_blocks import DEFAULT_SERVICE_PROMPT_TEMPLATE
from system_events import record_navigation_system_event


class MaxVisionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        self._orig_max_ai_sessions = max_ai.async_session_maker
        self._orig_max_common_sessions = max_common.async_session_maker
        max_ai.async_session_maker = self.sessions
        max_common.async_session_maker = self.sessions

        async with self.sessions() as session:
            self.topic = Topic(
                id=1,
                name="Арт-терапия",
                is_active=True,
                system_prompt="Инструкция арт-терапевта.",
            )
            self.user = User(
                id=6001,
                first_name="Алексей",
                gender="male",
                age=35,
                current_dialogue_id=1,
                current_topic_id=1,
                metadata_json="{}",
            )
            service_template = (
                DEFAULT_SERVICE_PROMPT_TEMPLATE + "\n\n"
                "ПРОТОКОЛ ДАННЫХ:\n"
                "Используй блок <DATA>{\"art_analyzed\": true}</DATA>."
            )
            self.ai_config = AIConfig(
                id=1,
                provider="OpenAI",
                vision_provider="OpenAI",
                vision_model="gpt-5.6-terra",
                openai_api_key="sk-openai-vision-test",
                kie_api_key="sk-kie-vision-test",
                kie_model="gemini-3-flash",
                shared_prompt_block="Общие правила vision.",
                service_prompt_block=service_template,
                system_prompt="Общий системный промпт.",
                memory_mode="global",
            )
            session.add_all([self.topic, self.user, self.ai_config])
            await session.commit()

            # Add previous user message that starts with "[Изображение]" to prove string dedup is gone
            prev_user_img_msg = DBMessage(
                user_id=6001,
                dialogue_id=1,
                topic_id=1,
                role="user",
                content="[Изображение] Предыдущее фото рисунка",
                timestamp=datetime(2026, 1, 1, 10, 0, 0),
            )
            prev_ai_msg = DBMessage(
                user_id=6001,
                dialogue_id=1,
                topic_id=1,
                role="assistant",
                content="Вижу ваш рисунок, очень выразительно.",
                timestamp=datetime(2026, 1, 1, 10, 0, 10),
            )
            session.add_all([prev_user_img_msg, prev_ai_msg])
            await session.commit()

            # Record a persistent navigation system event
            await record_navigation_system_event(
                session,
                user_id=6001,
                dialogue_id=1,
                topic_id=1,
                text="[СИСТЕМНОЕ СОБЫТИЕ: Переход в тему 'Арт-терапия']",
            )
            await session.commit()

            # Prior activity timestamp
            act_topic = UserAIActivity(
                user_id=6001, scope_key="topic:1", last_request_at=datetime(2026, 1, 1, 10, 0, 0)
            )
            act_global = UserAIActivity(
                user_id=6001, scope_key="global", last_request_at=datetime(2026, 1, 1, 10, 0, 0)
            )
            session.add_all([act_topic, act_global])
            await session.commit()

    async def asyncTearDown(self):
        max_ai.async_session_maker = self._orig_max_ai_sessions
        max_common.async_session_maker = self._orig_max_common_sessions
        await self.engine.dispose()

    async def test_max_vision_full_flow_through_run_ai_dialogue_with_image(self):
        mock_client = MagicMock()
        mock_client.send_message = AsyncMock(return_value={"message": {"mid": "mid_123"}})
        mock_client.edit_message = AsyncMock(return_value={"ok": True})
        mock_client.delete_message = AsyncMock(return_value={"ok": True})

        captured_payloads = []

        async def fake_create(**kwargs):
            captured_payloads.append(kwargs)
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "Анализ изображения завершен успешно."
            resp.choices = [choice]
            return resp

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create):
            await max_common.run_ai_dialogue_with_image(
                client=mock_client,
                chat_id=888,
                user_id=6001,
                image_bytes=b"raw_jpeg_bytes_12345",
                caption="Опиши метафору этого рисунка",
            )

        # 1. Assert current image Message was persisted in DB
        async with self.sessions() as session:
            messages = (
                await session.scalars(
                    select(DBMessage).where(DBMessage.user_id == 6001).order_by(DBMessage.id.asc())
                )
            ).all()
            user_messages = [m for m in messages if m.role == "user"]
            self.assertEqual(len(user_messages), 2)
            latest_user_msg = user_messages[-1]
            self.assertEqual(latest_user_msg.content, "[Изображение] Опиши метафору этого рисунка")

        # 2. Inspect captured payload
        self.assertEqual(len(captured_payloads), 1)
        payload = captured_payloads[0]
        outbound_messages = payload["messages"]

        all_system_content = "\n\n".join(m["content"] for m in outbound_messages if m["role"] == "system")

        # 3. Assert MAX_CAPABILITIES applied
        self.assertNotIn("SEND_AUDIO", all_system_content)
        self.assertNotIn("CHOICE_IMG", all_system_content)
        self.assertNotIn("RANDOM_IMG", all_system_content)
        self.assertNotIn("SHOW_IMG", all_system_content)
        self.assertIn("GEN_IMG", all_system_content)
        self.assertIn("<DATA>", all_system_content)
        self.assertIn("ИНСТРУКЦИЯ ПО АНАЛИЗУ ФОТО:", all_system_content)
        self.assertIn("EDIT_IMG:", all_system_content)

        # 4. Assert temporal variables present
        self.assertIn("ВРЕМЕННОЙ КОНТЕКСТ:", all_system_content)
        self.assertIn("minutes_since_last_visit:", all_system_content)

        # 5. Assert exact ID excluded & string dedup absent
        # History messages sent to model (excluding system messages and the current user message)
        history_in_payload = [m for m in outbound_messages if m["role"] != "system"][:-1]
        history_texts = [m["content"] for m in history_in_payload]

        # The previous user message starting with "[Изображение]" is retained (no string dedup!)
        self.assertTrue(any("[Изображение] Предыдущее фото рисунка" in t for t in history_texts))
        # The current user message is NOT in history (exact ID was excluded!)
        self.assertFalse(any("[Изображение] Опиши метафору этого рисунка" in t for t in history_texts))
        # System event is retained in history
        self.assertTrue(any("[СИСТЕМНОЕ СОБЫТИЕ" in t for t in history_texts))

        # 6. Current multimodal content once at end
        current_content = outbound_messages[-1]
        self.assertEqual(current_content["role"], "user")
        self.assertIsInstance(current_content["content"], list)
        self.assertEqual(current_content["content"][0]["type"], "text")
        self.assertEqual(current_content["content"][0]["text"], "Опиши метафору этого рисунка")
        self.assertEqual(current_content["content"][1]["type"], "image_url")

        # 7. Outbound activity was recorded
        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 6001))).all()
            self.assertEqual(len(acts), 2)
            # Timestamp updated to after 10:00:00
            for act in acts:
                self.assertGreater(act.last_request_at, datetime(2026, 1, 1, 10, 0, 0))

    async def test_max_vision_local_validation_failure_records_zero_activity(self):
        # Reset activity rows to test clean slate
        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 6001))).all()
            for act in acts:
                await session.delete(act)
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "OpenAI"
            cfg.vision_model = "invalid-model-name"
            await session.commit()

        mock_client = MagicMock()
        mock_client.send_message = AsyncMock(return_value={"message": {"mid": "mid_123"}})

        with self.assertRaises(Exception):
            await max_common.run_ai_dialogue_with_image(
                client=mock_client,
                chat_id=888,
                user_id=6001,
                image_bytes=b"raw_bytes",
                caption="Разбери фото",
            )

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 6001))).all()
            self.assertEqual(len(acts), 0)

    async def test_max_vision_kie_upload_failure_records_zero_activity(self):
        # Clear activity
        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 6001))).all()
            for act in acts:
                await session.delete(act)
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "KIE"
            cfg.vision_model = "gemini-3-flash"
            await session.commit()

        with patch("max_messenger_bot.ai._upload_file_to_kie", side_effect=RuntimeError("KIE upload connection refused")):
            with self.assertRaises(Exception):
                await max_ai.analyze_image(
                    user_id=6001,
                    image_bytes=b"fake_image_bytes",
                    prompt="Анализируй KIE",
                )

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 6001))).all()
            self.assertEqual(len(acts), 0)

    async def test_max_vision_kie_inference_attempt_records_activity(self):
        # Clear activity
        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 6001))).all()
            for act in acts:
                await session.delete(act)
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "KIE"
            cfg.vision_model = "gemini-3-flash"
            await session.commit()

        # Upload succeeds, but model inference endpoint times out
        with patch("max_messenger_bot.ai._upload_file_to_kie", new_callable=AsyncMock, return_value="https://kie.upload/file.jpg"), \
             patch("httpx.AsyncClient.post", side_effect=TimeoutError("KIE model inference network timeout")):
            with self.assertRaises(Exception):
                await max_ai.analyze_image(
                    user_id=6001,
                    image_bytes=b"fake_image_bytes",
                    prompt="Анализируй KIE",
                )

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 6001))).all()
            self.assertEqual(len(acts), 2)
