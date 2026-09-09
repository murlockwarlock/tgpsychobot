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

import ai_integration
from database import (
    AIConfig,
    AILog,
    Base,
    Message as DBMessage,
    Topic,
    User,
    UserAIActivity,
)
from max_messenger_bot import ai as max_ai
from prompt_blocks import DEFAULT_SERVICE_PROMPT_TEMPLATE, build_media_instruction_block
from system_events import record_navigation_system_event


class PlatformPayloadParityIntegrationTests(unittest.IsolatedAsyncioTestCase):
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
        ai_integration.async_session_maker = self.sessions
        max_ai.async_session_maker = self.sessions

        async with self.sessions() as session:
            self.topic = Topic(
                id=1,
                name="Психология",
                is_active=True,
                system_prompt="Инструкция психолога.",
            )
            self.user = User(
                id=5001,
                first_name="Мария",
                gender="female",
                age=28,
                current_dialogue_id=1,
                current_topic_id=1,
                metadata_json="{}",
            )
            # Service prompt template with both media rules and DATA instructions
            service_template = (
                DEFAULT_SERVICE_PROMPT_TEMPLATE + "\n\n"
                "ПРОТОКОЛ ДАННЫХ:\n"
                "Используй блок <DATA>{\"lead\": true}</DATA> при выявлении интереса."
            )
            self.ai_config = AIConfig(
                id=1,
                provider="OpenAI",
                openai_api_key="sk-openai-parity",
                openai_model="gpt-5.6-terra",
                claude_api_key="sk-claude-parity",
                claude_model="claude-sonnet-5",
                gemini_api_key="sk-gemini-parity",
                gemini_model="gemini-3.7-flash",
                shared_prompt_block="Общий блок правил для всех платформ.",
                service_prompt_block=service_template,
                system_prompt="Общий системный промпт.",
                memory_mode="global",
            )
            session.add_all([self.topic, self.user, self.ai_config])
            await session.commit()

            # Add previous conversational history
            msg1 = DBMessage(
                user_id=5001, dialogue_id=1, topic_id=1, role="user",
                content="Привет, хочу консультацию", timestamp=datetime(2026, 1, 1, 10, 0, 0)
            )
            msg2 = DBMessage(
                user_id=5001, dialogue_id=1, topic_id=1, role="assistant",
                content="Здравствуйте, Мария!", timestamp=datetime(2026, 1, 1, 10, 0, 5)
            )
            session.add_all([msg1, msg2])
            await session.commit()

            # Record a persistent navigation system event
            await record_navigation_system_event(
                session,
                user_id=5001,
                dialogue_id=1,
                topic_id=1,
                text="[СИСТЕМНОЕ СОБЫТИЕ: Переход в тему 'Психология']",
            )
            await session.commit()

            # Prior activity timestamp for temporal context
            act_global = UserAIActivity(
                user_id=5001, scope_key="global", last_request_at=datetime(2026, 1, 1, 10, 0, 0)
            )
            act_topic = UserAIActivity(
                user_id=5001, scope_key="topic:1", last_request_at=datetime(2026, 1, 1, 10, 0, 0)
            )
            session.add_all([act_global, act_topic])
            await session.commit()

    async def asyncTearDown(self):
        ai_integration.async_session_maker = self._orig_tg_sessions
        max_ai.async_session_maker = self._orig_max_sessions
        await self.engine.dispose()

    async def test_telegram_normal_openai_payload_and_ailog(self):
        from types import SimpleNamespace

        captured_payloads = []

        async def fake_create(**kwargs):
            captured_payloads.append(kwargs)
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "Ответ ассистента Telegram"
            resp.choices = [choice]
            return resp

        mock_media = [
            SimpleNamespace(media_type="audio", file_name="audio_track_1", description="", category="Музыка")
        ]
        with patch("ai_integration.load_available_media", AsyncMock(return_value=([], mock_media))), \
             patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create):
            resp = await ai_integration.generate_response(
                user_id=5001,
                user_prompt="Что посоветуете?",
                track_user_activity=True,
            )
            self.assertEqual(resp, "Ответ ассистента Telegram")

        self.assertEqual(len(captured_payloads), 1)
        payload = captured_payloads[0]
        messages = payload["messages"]
        all_system_content = "\n\n".join(m["content"] for m in messages if m["role"] == "system")

        # Assert canonical order and components in Telegram normal
        self.assertIn("Инструкция психолога.", all_system_content)
        self.assertIn("Общий блок правил для всех платформ.", all_system_content)
        # Telegram preserves media instructions
        self.assertIn("SEND_AUDIO", all_system_content)
        self.assertIn("audio_track_1", all_system_content)
        self.assertIn("GEN_IMG", all_system_content)
        self.assertIn("<DATA>", all_system_content)
        self.assertIn("ДАННЫЕ КЛИЕНТА:", all_system_content)
        self.assertIn("ИМЯ: Мария", all_system_content)
        self.assertIn("ВРЕМЕННОЙ КОНТЕКСТ:", all_system_content)
        self.assertIn("minutes_since_last_visit:", all_system_content)
        self.assertIn("minutes_since_last_message:", all_system_content)

        # Chronological history & system event
        history_roles_contents = [(m["role"], m["content"]) for m in messages if m["role"] != "system"]
        self.assertEqual(history_roles_contents[-1], ("user", "Что посоветуете?"))
        texts = [m["content"] for m in messages if m["role"] != "system"]
        self.assertTrue(any("[СИСТЕМНОЕ СОБЫТИЕ" in t for t in texts))

        # Verify persisted AILog matches outbound request payload
        async with self.sessions() as session:
            ai_log = await session.scalar(select(AILog).order_by(AILog.id.desc()))
            self.assertIsNotNone(ai_log)
            logged_payload = json.loads(ai_log.request_payload)
            self.assertEqual(logged_payload["provider"], "OpenAI")
            self.assertNotIn("sk-", logged_payload.get("endpoint", ""))
            captured_body = logged_payload.get("payload", logged_payload)
            self.assertEqual(captured_body["model"], payload["model"])
            self.assertEqual(captured_body["messages"], payload["messages"])

    async def test_max_normal_openai_payload_and_ailog(self):
        captured_payloads = []

        async def fake_create(**kwargs):
            captured_payloads.append(kwargs)
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "Ответ ассистента MAX"
            resp.choices = [choice]
            return resp

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create):
            resp = await max_ai.get_ai_response(
                5001,
                "Как мне справиться со стрессом?",
                track_user_activity=True,
            )
            self.assertEqual(resp, "Ответ ассистента MAX")

        self.assertEqual(len(captured_payloads), 1)
        payload = captured_payloads[0]
        messages = payload["messages"]
        all_system_content = "\n\n".join(m["content"] for m in messages if m["role"] == "system")

        # Assert canonical order and MAX capability sanitization
        self.assertIn("Инструкция психолога.", all_system_content)
        self.assertIn("Общий блок правил для всех платформ.", all_system_content)

        # MAX hard invariants: media commands stripped, GEN_IMG and DATA retained
        self.assertNotIn("SEND_AUDIO", all_system_content)
        self.assertNotIn("RANDOM_IMG", all_system_content)
        self.assertNotIn("CHOICE_IMG", all_system_content)
        self.assertNotIn("SHOW_IMG", all_system_content)
        self.assertNotIn("МЕДИА-ФАЙЛЫ:", all_system_content)
        self.assertNotIn("audio_track_1", all_system_content)
        self.assertIn("GEN_IMG", all_system_content)
        self.assertIn("<DATA>", all_system_content)

        # Client and temporal variables
        self.assertIn("ДАННЫЕ КЛИЕНТА:", all_system_content)
        self.assertIn("ИМЯ: Мария", all_system_content)
        self.assertIn("ВРЕМЕННОЙ КОНТЕКСТ:", all_system_content)
        self.assertIn("minutes_since_last_visit:", all_system_content)
        self.assertIn("minutes_since_last_message:", all_system_content)

        # History and current content
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "Как мне справиться со стрессом?")
        texts = [m["content"] for m in messages if m["role"] != "system"]
        self.assertTrue(any("[СИСТЕМНОЕ СОБЫТИЕ" in t for t in texts))

        # Verify persisted MAX AILog matches outbound request payload
        async with self.sessions() as session:
            ai_log = await session.scalar(select(AILog).where(AILog.platform == "max").order_by(AILog.id.desc()))
            self.assertIsNotNone(ai_log)
            logged_payload = json.loads(ai_log.request_payload)
            self.assertEqual(logged_payload["provider"], "OpenAI")
            self.assertNotIn("sk-", logged_payload.get("endpoint", ""))
            captured_body = logged_payload.get("payload", logged_payload)
            self.assertEqual(captured_body["model"], payload["model"])
            self.assertEqual(captured_body["messages"], payload["messages"])

    async def test_max_vision_shared_builder_payload(self):
        captured_payloads = []

        async def fake_create(**kwargs):
            captured_payloads.append(kwargs)
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "Анализ изображения MAX"
            resp.choices = [choice]
            return resp

        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "OpenAI"
            cfg.vision_model = "gpt-5.6-terra"
            await session.commit()

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create):
            resp = await max_ai.analyze_image(
                5001,
                image_bytes=b"fake_jpeg_data",
                prompt="Что изображено на этом рисунке?",
            )
            self.assertEqual(resp, "Анализ изображения MAX")

        self.assertEqual(len(captured_payloads), 1)
        payload = captured_payloads[0]
        messages = payload["messages"]
        all_system_content = "\n\n".join(m["content"] for m in messages if m["role"] == "system")

        # Assert MAX vision received shared prompt, capability-aware service prompt, and photo instructions
        self.assertIn("Инструкция психолога.", all_system_content)
        self.assertIn("Общий блок правил для всех платформ.", all_system_content)
        self.assertIn("ИНСТРУКЦИЯ ПО АНАЛИЗУ ФОТО:", all_system_content)
        self.assertIn("EDIT_IMG:", all_system_content)
        self.assertNotIn("SEND_AUDIO", all_system_content)
        self.assertNotIn("CHOICE_IMG", all_system_content)
        self.assertIn("GEN_IMG", all_system_content)
        self.assertIn("<DATA>", all_system_content)

        # Temporal activity variables present
        self.assertIn("ВРЕМЕННОЙ КОНТЕКСТ:", all_system_content)
        self.assertIn("minutes_since_last_visit:", all_system_content)

        # Multimodal current user content once at end
        last_msg = messages[-1]
        self.assertEqual(last_msg["role"], "user")
        self.assertIsInstance(last_msg["content"], list)
        types = [part["type"] for part in last_msg["content"]]
        self.assertEqual(types, ["text", "image_url"])
        self.assertEqual(last_msg["content"][0]["text"], "Что изображено на этом рисунке?")

    async def test_max_direct_openai_payload(self):
        captured_payloads = []

        async def fake_create(**kwargs):
            captured_payloads.append(kwargs)
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "Прямой ответ MAX"
            resp.choices = [choice]
            return resp

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create):
            resp = await max_ai.get_ai_response_direct(
                5001,
                system_prompt="Прямой системный промпт.",
                user_prompt="Краткий вопрос",
                track_user_activity=True,
            )
            self.assertEqual(resp, "Прямой ответ MAX")

        self.assertEqual(len(captured_payloads), 1)
        payload = captured_payloads[0]
        messages = payload["messages"]
        all_system_content = "\n\n".join(m["content"] for m in messages if m["role"] == "system")

        self.assertIn("Прямой системный промпт.", all_system_content)
        # MAX direct gets MAX_CAPABILITIES: media commands stripped, GEN_IMG and DATA retained
        self.assertNotIn("SEND_AUDIO", all_system_content)
        self.assertNotIn("CHOICE_IMG", all_system_content)
        self.assertIn("GEN_IMG", all_system_content)
        self.assertIn("<DATA>", all_system_content)
        self.assertIn("ВРЕМЕННОЙ КОНТЕКСТ:", all_system_content)

    async def test_max_normal_claude_family_payload(self):
        captured_payloads = []

        async def fake_messages_create(**kwargs):
            captured_payloads.append(kwargs)
            resp = MagicMock()
            block = MagicMock()
            block.text = "Ответ Claude на MAX"
            resp.content = [block]
            return resp

        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "Claude"
            await session.commit()

        with patch("anthropic.resources.messages.AsyncMessages.create", side_effect=fake_messages_create):
            resp = await max_ai.get_ai_response(
                5001,
                "Тест Claude на MAX",
                track_user_activity=True,
            )
            self.assertEqual(resp, "Ответ Claude на MAX")

        self.assertEqual(len(captured_payloads), 1)
        payload = captured_payloads[0]
        system_prompt = payload["system"]
        all_system_content = "\n\n".join(b["text"] if isinstance(b, dict) else str(b) for b in system_prompt)

        self.assertIn("Инструкция психолога.", all_system_content)
        self.assertIn("Общий блок правил для всех платформ.", all_system_content)
        self.assertNotIn("SEND_AUDIO", all_system_content)
        self.assertIn("GEN_IMG", all_system_content)
        self.assertIn("<DATA>", all_system_content)
        self.assertIn("ВРЕМЕННОЙ КОНТЕКСТ:", all_system_content)
        self.assertEqual(payload["messages"][-1]["content"], "Тест Claude на MAX")

    async def test_max_normal_gemini_family_payload(self):
        captured_payloads = []

        async def fake_post(url, *args, **kwargs):
            captured_payloads.append(kwargs.get("json", {}))
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "candidates": [
                    {"content": {"parts": [{"text": "Ответ Gemini на MAX"}]}}
                ]
            }
            return resp

        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "Gemini"
            await session.commit()

        with patch("httpx.AsyncClient.post", side_effect=fake_post):
            resp = await max_ai.get_ai_response(
                5001,
                "Тест Gemini на MAX",
                track_user_activity=True,
            )
            self.assertEqual(resp, "Ответ Gemini на MAX")

        self.assertEqual(len(captured_payloads), 1)
        payload = captured_payloads[0]
        system_text = "\n\n".join(p["text"] for p in payload["systemInstruction"]["parts"])

        self.assertIn("Инструкция психолога.", system_text)
        self.assertIn("Общий блок правил для всех платформ.", system_text)
        self.assertNotIn("SEND_AUDIO", system_text)
        self.assertIn("GEN_IMG", system_text)
        self.assertIn("<DATA>", system_text)
        self.assertIn("ВРЕМЕННОЙ КОНТЕКСТ:", system_text)

    async def test_deepseek_platform_payload_parity_and_ailog(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "DeepSeek"
            cfg.deepseek_api_key = "sk-deepseek-parity"
            cfg.deepseek_model = "deepseek-v4-flash"
            await session.commit()

        captured_tg = []
        captured_max = []

        async def fake_create_tg(**kwargs):
            captured_tg.append(kwargs)
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "Ответ DeepSeek TG"
            resp.choices = [choice]
            return resp

        async def fake_create_max(**kwargs):
            captured_max.append(kwargs)
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "Ответ DeepSeek MAX"
            resp.choices = [choice]
            return resp

        # TG actual get_ai_response
        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create_tg):
            resp_tg = await ai_integration.generate_response(
                user_id=5001,
                user_prompt="Как мне справиться со стрессом?",
                track_user_activity=True,
            )
            self.assertEqual(resp_tg, "Ответ DeepSeek TG")

        # Verify persisted Telegram AILog for DeepSeek
        async with self.sessions() as session:
            tg_ai_log = await session.scalar(select(AILog).where(AILog.platform == "telegram").order_by(AILog.id.desc()))
            self.assertIsNotNone(tg_ai_log)
            self.assertEqual(tg_ai_log.provider, "DeepSeek")
            self.assertEqual(tg_ai_log.model, "deepseek-v4-flash")
            tg_log_data = json.loads(tg_ai_log.request_payload)
            self.assertEqual(tg_log_data["provider"], "Deepseek")
            self.assertNotIn("sk-", tg_log_data.get("endpoint", ""))
            self.assertEqual(tg_log_data["payload"]["model"], "deepseek-v4-flash")
            self.assertEqual(tg_log_data["payload"]["messages"], captured_tg[0]["messages"])

        # MAX actual get_ai_response
        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create_max):
            resp_max = await max_ai.get_ai_response(
                5001,
                "Как мне справиться со стрессом?",
                track_user_activity=True,
            )
            self.assertEqual(resp_max, "Ответ DeepSeek MAX")

        # Verify persisted MAX AILog for DeepSeek
        async with self.sessions() as session:
            max_ai_log = await session.scalar(select(AILog).where(AILog.platform == "max").order_by(AILog.id.desc()))
            self.assertIsNotNone(max_ai_log)
            self.assertIn(max_ai_log.provider, ("DeepSeek", "Deepseek"))
            self.assertEqual(max_ai_log.model, "deepseek-v4-flash")
            max_log_data = json.loads(max_ai_log.request_payload)
            self.assertEqual(max_log_data["provider"], "Deepseek")
            self.assertNotIn("sk-", max_log_data.get("endpoint", ""))
            self.assertEqual(max_log_data["payload"]["model"], "deepseek-v4-flash")
            self.assertEqual(max_log_data["payload"]["messages"], captured_max[0]["messages"])

        # Compare normalized common semantic request between TG and MAX
        payload_tg = captured_tg[0]
        payload_max = captured_max[0]

        self.assertEqual(payload_tg["model"], "deepseek-v4-flash")
        self.assertEqual(payload_max["model"], "deepseek-v4-flash")

        msgs_tg = payload_tg["messages"]
        msgs_max = payload_max["messages"]

        self.assertEqual(len(msgs_tg), len(msgs_max))
        self.assertEqual([m["role"] for m in msgs_tg], [m["role"] for m in msgs_max])

        all_sys_tg = "\n\n".join(m["content"] for m in msgs_tg if m["role"] == "system")
        all_sys_max = "\n\n".join(m["content"] for m in msgs_max if m["role"] == "system")

        for common_element in [
            "Инструкция психолога.",
            "Общий блок правил для всех платформ.",
            "<DATA>",
            "ДАННЫЕ КЛИЕНТА:",
            "ИМЯ: Мария",
            "ПОЛ: female",
            "ВРЕМЕННОЙ КОНТЕКСТ:",
            "minutes_since_last_visit:",
            "minutes_since_last_message:",
        ]:
            self.assertIn(common_element, all_sys_tg)
            self.assertIn(common_element, all_sys_max)

        self.assertIn("GEN_IMG", all_sys_tg)
        self.assertIn("GEN_IMG", all_sys_max)
        self.assertNotIn("SEND_AUDIO", all_sys_max)

        non_sys_tg = [m for m in msgs_tg if m["role"] != "system"]
        non_sys_max = [m for m in msgs_max if m["role"] != "system"]
        self.assertEqual(len(non_sys_tg), len(non_sys_max))
        for m_tg, m_max in zip(non_sys_tg, non_sys_max):
            self.assertEqual(m_tg["role"], m_max["role"])
            self.assertEqual(m_tg["content"], m_max["content"])

    async def test_kie_platform_payload_parity_and_ailog(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "KIE"
            cfg.kie_api_key = "sk-kie-parity"
            cfg.kie_model = "gemini-2.5-flash"
            await session.commit()

        captured_posts = []

        async def fake_post(url, *args, **kwargs):
            captured_posts.append({
                "url": url,
                "headers": kwargs.get("headers", {}),
                "json": kwargs.get("json", {}),
            })
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "choices": [{"message": {"content": "Ответ KIE Chat"}}]
            }
            return resp

        # TG actual get_ai_response
        with patch("httpx.AsyncClient.post", side_effect=fake_post):
            resp_tg = await ai_integration.generate_response(
                user_id=5001,
                user_prompt="Как мне справиться со стрессом?",
                track_user_activity=True,
            )
            self.assertEqual(resp_tg, "Ответ KIE Chat")

        # Verify persisted Telegram AILog for KIE
        async with self.sessions() as session:
            tg_ai_log = await session.scalar(select(AILog).where(AILog.platform == "telegram").order_by(AILog.id.desc()))
            self.assertIsNotNone(tg_ai_log)
            self.assertEqual(tg_ai_log.provider, "KIE")
            self.assertEqual(tg_ai_log.model, "gemini-2.5-flash")
            tg_log_data = json.loads(tg_ai_log.request_payload)
            self.assertEqual(tg_log_data["provider"], "KIE")
            self.assertNotIn("sk-", tg_log_data.get("endpoint", ""))
            self.assertEqual(tg_log_data["payload"]["model"], "gemini-2.5-flash")
            self.assertEqual(tg_log_data["payload"], captured_posts[0]["json"])

        # MAX actual get_ai_response
        with patch("httpx.AsyncClient.post", side_effect=fake_post):
            resp_max = await max_ai.get_ai_response(
                5001,
                "Как мне справиться со стрессом?",
                track_user_activity=True,
            )
            self.assertEqual(resp_max, "Ответ KIE Chat")

        # Verify persisted MAX AILog for KIE
        async with self.sessions() as session:
            max_ai_log = await session.scalar(select(AILog).where(AILog.platform == "max").order_by(AILog.id.desc()))
            self.assertIsNotNone(max_ai_log)
            self.assertEqual(max_ai_log.provider, "KIE")
            self.assertEqual(max_ai_log.model, "gemini-2.5-flash")
            max_log_data = json.loads(max_ai_log.request_payload)
            self.assertEqual(max_log_data["provider"], "KIE")
            self.assertNotIn("sk-", max_log_data.get("endpoint", ""))
            self.assertEqual(max_log_data["payload"]["model"], "gemini-2.5-flash")
            self.assertEqual(max_log_data["payload"], captured_posts[1]["json"])

        # Compare normalized common semantic request between TG and MAX
        post_tg = captured_posts[0]["json"]
        post_max = captured_posts[1]["json"]

        self.assertEqual(post_tg["model"], "gemini-2.5-flash")
        self.assertEqual(post_max["model"], "gemini-2.5-flash")

        msgs_tg = post_tg["messages"]
        msgs_max = post_max["messages"]

        self.assertEqual(len(msgs_tg), len(msgs_max))
        self.assertEqual([m["role"] for m in msgs_tg], [m["role"] for m in msgs_max])

        all_sys_tg = "\n\n".join(m["content"] for m in msgs_tg if m["role"] == "system")
        all_sys_max = "\n\n".join(m["content"] for m in msgs_max if m["role"] == "system")

        for common_element in [
            "Инструкция психолога.",
            "Общий блок правил для всех платформ.",
            "<DATA>",
            "ДАННЫЕ КЛИЕНТА:",
            "ИМЯ: Мария",
            "ПОЛ: female",
            "ВРЕМЕННОЙ КОНТЕКСТ:",
            "minutes_since_last_visit:",
            "minutes_since_last_message:",
        ]:
            self.assertIn(common_element, all_sys_tg)
            self.assertIn(common_element, all_sys_max)

        self.assertNotIn("SEND_AUDIO", all_sys_max)

        non_sys_tg = [m for m in msgs_tg if m["role"] != "system"]
        non_sys_max = [m for m in msgs_max if m["role"] != "system"]
        self.assertEqual(len(non_sys_tg), len(non_sys_max))
        for m_tg, m_max in zip(non_sys_tg, non_sys_max):
            self.assertEqual(m_tg["role"], m_max["role"])
            self.assertEqual(m_tg["content"], m_max["content"])
