import asyncio
import json
import logging
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("BOT_TOKEN", "test")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import ai_integration
from database import AIConfig, AILog, Base, Topic, User
from max_messenger_bot import ai as max_ai
from provider_models import (
    CLAUDE_CHAT_MAX_TOKENS,
    DEEPSEEK_CHAT_MAX_TOKENS,
    GEMINI_CHAT_MAX_TOKENS,
    OPENAI_CHAT_MAX_TOKENS,
    PROVIDER_CLAUDE,
    PROVIDER_DEEPSEEK,
    PROVIDER_GEMINI,
    PROVIDER_KIE,
    PROVIDER_OPENAI,
)


class FakeSDKMessage:
    def __init__(self, content=None, reasoning_content=None, extra=None):
        self.content = content
        self.reasoning_content = reasoning_content
        if extra:
            self.model_extra = extra


class FakeSDKChoice:
    def __init__(self, finish_reason="stop", message=None):
        self.finish_reason = finish_reason
        self.message = message


class FakeSDKCompletion:
    def __init__(self, choices=None, model="test-model"):
        self.choices = choices or []
        self.model = model


class PhaseA2DirectChatBudgetsTests(unittest.IsolatedAsyncioTestCase):
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
                id=7001,
                first_name="Иван",
                gender="male",
                age=30,
                current_dialogue_id=1,
                current_topic_id=1,
                metadata_json="{}",
            )
            self.ai_config = AIConfig(
                id=1,
                provider="OpenAI",
                openai_api_key="sk-openai-test-key",
                openai_model="gpt-5.6-terra",
                claude_api_key="sk-claude-test-key",
                claude_model="claude-sonnet-5",
                gemini_api_key="sk-gemini-test-key",
                gemini_model="gemini-3.7-flash",
                deepseek_api_key="sk-deepseek-test-key",
                deepseek_model="deepseek-v4-flash",
                kie_api_key="sk-kie-test-key",
                kie_model="gemini-3-flash",
                vision_provider="OpenAI",
                vision_model="gpt-5.6-terra",
                fallback_provider="Claude",
                fallback_model="claude-sonnet-5",
                allow_fallback=True,
                shared_prompt_block="Общие правила диалога.",
                system_prompt="Общий системный промпт.",
                memory_mode="global",
            )
            session.add(self.topic)
            session.add(self.user)
            session.add(self.ai_config)
            await session.commit()

    async def asyncTearDown(self):
        ai_integration.async_session_maker = self._orig_tg_sessions
        max_ai.async_session_maker = self._orig_max_sessions
        await self.engine.dispose()

    # -------------------------------------------------------------------------
    # 1. Telegram OpenAI Direct: max_completion_tokens == 16384, no max_tokens
    # -------------------------------------------------------------------------
    async def test_telegram_openai_direct_chat_budget(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "OpenAI"
            await session.commit()

        captured = []
        async def fake_create(**kwargs):
            captured.append(kwargs)
            return FakeSDKCompletion(choices=[FakeSDKChoice(message=FakeSDKMessage(content="TG OpenAI OK"))])

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create):
            resp = await ai_integration.generate_response(user_id=7001, user_prompt="Вопрос к OpenAI TG")
            self.assertEqual(resp, "TG OpenAI OK")

        self.assertEqual(len(captured), 1)
        wire_payload = captured[0]
        self.assertEqual(wire_payload["max_completion_tokens"], 16384)
        self.assertEqual(wire_payload["max_completion_tokens"], OPENAI_CHAT_MAX_TOKENS)
        self.assertNotIn("max_tokens", wire_payload)

        # AILog verification
        async with self.sessions() as session:
            log_entry = await session.scalar(select(AILog).where(AILog.platform == "telegram").order_by(AILog.id.desc()))
            self.assertIsNotNone(log_entry)
            parsed = json.loads(log_entry.request_payload)
            self.assertEqual(parsed["payload"]["max_completion_tokens"], 16384)
            self.assertNotIn("max_tokens", parsed["payload"])

    # -------------------------------------------------------------------------
    # 2. MAX OpenAI Direct: max_completion_tokens == 16384, no max_tokens
    # -------------------------------------------------------------------------
    async def test_max_openai_direct_chat_budget(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "OpenAI"
            await session.commit()

        captured = []
        async def fake_create(**kwargs):
            captured.append(kwargs)
            return FakeSDKCompletion(choices=[FakeSDKChoice(message=FakeSDKMessage(content="MAX OpenAI OK"))])

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create):
            resp = await max_ai.get_ai_response(7001, "Вопрос к OpenAI MAX", track_user_activity=True)
            self.assertEqual(resp, "MAX OpenAI OK")

        self.assertEqual(len(captured), 1)
        wire_payload = captured[0]
        self.assertEqual(wire_payload["max_completion_tokens"], 16384)
        self.assertEqual(wire_payload["max_completion_tokens"], OPENAI_CHAT_MAX_TOKENS)
        self.assertNotIn("max_tokens", wire_payload)

        # AILog verification
        async with self.sessions() as session:
            log_entry = await session.scalar(select(AILog).where(AILog.platform == "max").order_by(AILog.id.desc()))
            self.assertIsNotNone(log_entry)
            parsed = json.loads(log_entry.request_payload)
            self.assertEqual(parsed["payload"]["max_completion_tokens"], 16384)
            self.assertNotIn("max_tokens", parsed["payload"])

    # -------------------------------------------------------------------------
    # 3. Telegram Claude Direct: max_tokens == 16384, no max_completion_tokens
    # -------------------------------------------------------------------------
    async def test_telegram_claude_direct_chat_budget(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "Claude"
            await session.commit()

        captured = []
        mock_client = AsyncMock()
        mock_msg = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "TG Claude OK"
        mock_msg.content = [mock_part]

        async def fake_create(**kwargs):
            captured.append(kwargs)
            return mock_msg

        mock_client.messages.create.side_effect = fake_create

        with patch.object(ai_integration.anthropic, "AsyncAnthropic", return_value=mock_client):
            resp = await ai_integration.generate_response(user_id=7001, user_prompt="Вопрос к Claude TG")
            self.assertEqual(resp, "TG Claude OK")

        self.assertEqual(len(captured), 1)
        wire_payload = captured[0]
        self.assertEqual(wire_payload["max_tokens"], 16384)
        self.assertEqual(wire_payload["max_tokens"], CLAUDE_CHAT_MAX_TOKENS)
        self.assertNotIn("max_completion_tokens", wire_payload)

        # AILog verification
        async with self.sessions() as session:
            log_entry = await session.scalar(select(AILog).where(AILog.platform == "telegram").order_by(AILog.id.desc()))
            self.assertIsNotNone(log_entry)
            parsed = json.loads(log_entry.request_payload)
            self.assertEqual(parsed["payload"]["max_tokens"], 16384)
            self.assertNotIn("max_completion_tokens", parsed["payload"])

    # -------------------------------------------------------------------------
    # 4. MAX Claude Direct: max_tokens == 16384, no max_completion_tokens
    # -------------------------------------------------------------------------
    async def test_max_claude_direct_chat_budget(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "Claude"
            await session.commit()

        captured = []
        async def fake_create(**kwargs):
            captured.append(kwargs)
            resp = MagicMock()
            part = MagicMock()
            part.text = "MAX Claude OK"
            resp.content = [part]
            return resp

        with patch("anthropic.resources.messages.AsyncMessages.create", side_effect=fake_create):
            resp = await max_ai.get_ai_response(7001, "Вопрос к Claude MAX", track_user_activity=True)
            self.assertEqual(resp, "MAX Claude OK")

        self.assertEqual(len(captured), 1)
        wire_payload = captured[0]
        self.assertEqual(wire_payload["max_tokens"], 16384)
        self.assertEqual(wire_payload["max_tokens"], CLAUDE_CHAT_MAX_TOKENS)
        self.assertNotIn("max_completion_tokens", wire_payload)

        # AILog verification
        async with self.sessions() as session:
            log_entry = await session.scalar(select(AILog).where(AILog.platform == "max").order_by(AILog.id.desc()))
            self.assertIsNotNone(log_entry)
            parsed = json.loads(log_entry.request_payload)
            self.assertEqual(parsed["payload"]["max_tokens"], 16384)
            self.assertNotIn("max_completion_tokens", parsed["payload"])

    # -------------------------------------------------------------------------
    # 5. Telegram Gemini Direct: generationConfig.maxOutputTokens == 16384
    # -------------------------------------------------------------------------
    async def test_telegram_gemini_direct_chat_budget(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "Gemini"
            await session.commit()

        captured = []
        class FakeGeminiClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, **kwargs):
                captured.append(kwargs)
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {
                    "candidates": [{"content": {"parts": [{"text": "TG Gemini OK"}]}}]
                }
                return resp

        with patch.object(ai_integration.httpx, "AsyncClient", return_value=FakeGeminiClient()):
            resp = await ai_integration.generate_response(user_id=7001, user_prompt="Вопрос к Gemini TG")
            self.assertEqual(resp, "TG Gemini OK")

        self.assertEqual(len(captured), 1)
        payload = captured[0]["json"]
        self.assertIn("generationConfig", payload)
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 16384)
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], GEMINI_CHAT_MAX_TOKENS)
        self.assertNotIn("maxOutputTokens", payload)
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("max_completion_tokens", payload)

        # AILog verification
        async with self.sessions() as session:
            log_entry = await session.scalar(select(AILog).where(AILog.platform == "telegram").order_by(AILog.id.desc()))
            self.assertIsNotNone(log_entry)
            parsed = json.loads(log_entry.request_payload)
            self.assertEqual(parsed["payload"]["generationConfig"]["maxOutputTokens"], 16384)

    # -------------------------------------------------------------------------
    # 6. MAX Gemini Direct: generationConfig.maxOutputTokens == 16384
    # -------------------------------------------------------------------------
    async def test_max_gemini_direct_chat_budget(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "Gemini"
            await session.commit()

        captured = []
        async def fake_post(url, *args, **kwargs):
            captured.append(kwargs)
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "candidates": [{"content": {"parts": [{"text": "MAX Gemini OK"}]}}]
            }
            return resp

        with patch("httpx.AsyncClient.post", side_effect=fake_post):
            resp = await max_ai.get_ai_response(7001, "Вопрос к Gemini MAX", track_user_activity=True)
            self.assertEqual(resp, "MAX Gemini OK")

        self.assertEqual(len(captured), 1)
        payload = captured[0]["json"]
        self.assertIn("generationConfig", payload)
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 16384)
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], GEMINI_CHAT_MAX_TOKENS)
        self.assertNotIn("maxOutputTokens", payload)
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("max_completion_tokens", payload)

        # AILog verification
        async with self.sessions() as session:
            log_entry = await session.scalar(select(AILog).where(AILog.platform == "max").order_by(AILog.id.desc()))
            self.assertIsNotNone(log_entry)
            parsed = json.loads(log_entry.request_payload)
            self.assertEqual(parsed["payload"]["generationConfig"]["maxOutputTokens"], 16384)

    # -------------------------------------------------------------------------
    # 7, 8, 9: Provider-Specific Field Discipline Confirmed Above (and explicit tests)
    # -------------------------------------------------------------------------
    def test_provider_budget_constants(self):
        self.assertEqual(OPENAI_CHAT_MAX_TOKENS, 16384)
        self.assertEqual(CLAUDE_CHAT_MAX_TOKENS, 16384)
        self.assertEqual(GEMINI_CHAT_MAX_TOKENS, 16384)
        self.assertEqual(DEEPSEEK_CHAT_MAX_TOKENS, 65536)

    # -------------------------------------------------------------------------
    # 10. Telegram Fallback -> OpenAI (actual outbound payload uses 16384)
    # -------------------------------------------------------------------------
    async def test_telegram_fallback_to_openai(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "Claude"
            cfg.fallback_provider = "OpenAI"
            cfg.fallback_model = "gpt-5.6-terra"
            cfg.allow_fallback = True
            await session.commit()

        captured_fallback = []
        async def fake_claude_fail(**kwargs):
            raise RuntimeError("Claude primary network error")

        async def fake_openai_fallback(**kwargs):
            captured_fallback.append(kwargs)
            return FakeSDKCompletion(choices=[FakeSDKChoice(message=FakeSDKMessage(content="Fallback OpenAI OK"))])

        with (
            patch("anthropic.resources.messages.AsyncMessages.create", side_effect=fake_claude_fail),
            patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_openai_fallback),
        ):
            resp = await ai_integration.generate_response(user_id=7001, user_prompt="Тест фолбэк на OpenAI TG")
            self.assertEqual(resp, "Fallback OpenAI OK")

        self.assertEqual(len(captured_fallback), 1)
        self.assertEqual(captured_fallback[0]["max_completion_tokens"], 16384)
        self.assertNotIn("max_tokens", captured_fallback[0])

    # -------------------------------------------------------------------------
    # 11. Telegram Fallback -> Claude (actual outbound payload uses 16384)
    # -------------------------------------------------------------------------
    async def test_telegram_fallback_to_claude(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "OpenAI"
            cfg.fallback_provider = "Claude"
            cfg.fallback_model = "claude-sonnet-5"
            cfg.allow_fallback = True
            await session.commit()

        captured_fallback = []
        async def fake_openai_fail(**kwargs):
            raise RuntimeError("OpenAI primary network error")

        mock_client = AsyncMock()
        mock_msg = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Fallback Claude OK"
        mock_msg.content = [mock_part]

        async def fake_claude_fallback(**kwargs):
            captured_fallback.append(kwargs)
            return mock_msg

        mock_client.messages.create.side_effect = fake_claude_fallback

        with (
            patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_openai_fail),
            patch.object(ai_integration.anthropic, "AsyncAnthropic", return_value=mock_client),
        ):
            resp = await ai_integration.generate_response(user_id=7001, user_prompt="Тест фолбэк на Claude TG")
            self.assertEqual(resp, "Fallback Claude OK")

        self.assertEqual(len(captured_fallback), 1)
        self.assertEqual(captured_fallback[0]["max_tokens"], 16384)
        self.assertNotIn("max_completion_tokens", captured_fallback[0])

    # -------------------------------------------------------------------------
    # 12. Telegram Fallback -> Gemini (actual outbound payload uses 16384)
    # -------------------------------------------------------------------------
    async def test_telegram_fallback_to_gemini(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "OpenAI"
            cfg.fallback_provider = "Gemini"
            cfg.fallback_model = "gemini-3.7-flash"
            cfg.allow_fallback = True
            await session.commit()

        captured_fallback = []
        async def fake_openai_fail(**kwargs):
            raise RuntimeError("OpenAI primary network error")

        class FakeGeminiClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, **kwargs):
                captured_fallback.append(kwargs)
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {
                    "candidates": [{"content": {"parts": [{"text": "Fallback Gemini OK"}]}}]
                }
                return resp

        with (
            patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_openai_fail),
            patch.object(ai_integration.httpx, "AsyncClient", return_value=FakeGeminiClient()),
        ):
            resp = await ai_integration.generate_response(user_id=7001, user_prompt="Тест фолбэк на Gemini TG")
            self.assertEqual(resp, "Fallback Gemini OK")

        self.assertEqual(len(captured_fallback), 1)
        self.assertEqual(captured_fallback[0]["json"]["generationConfig"]["maxOutputTokens"], 16384)

    # -------------------------------------------------------------------------
    # 13. MAX Fallback -> OpenAI (actual outbound payload uses 16384)
    # -------------------------------------------------------------------------
    async def test_max_fallback_to_openai(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "Claude"
            cfg.fallback_provider = "OpenAI"
            cfg.fallback_model = "gpt-5.6-terra"
            cfg.allow_fallback = True
            await session.commit()

        captured_fallback = []
        async def fake_claude_fail(**kwargs):
            raise RuntimeError("Claude primary network error")

        async def fake_openai_fallback(**kwargs):
            captured_fallback.append(kwargs)
            return FakeSDKCompletion(choices=[FakeSDKChoice(message=FakeSDKMessage(content="MAX Fallback OpenAI OK"))])

        with (
            patch("anthropic.resources.messages.AsyncMessages.create", side_effect=fake_claude_fail),
            patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_openai_fallback),
        ):
            resp = await max_ai.get_ai_response(7001, "Тест фолбэк на OpenAI MAX", track_user_activity=True)
            self.assertEqual(resp, "MAX Fallback OpenAI OK")

        self.assertEqual(len(captured_fallback), 1)
        self.assertEqual(captured_fallback[0]["max_completion_tokens"], 16384)
        self.assertNotIn("max_tokens", captured_fallback[0])

    # -------------------------------------------------------------------------
    # 14. MAX Fallback -> Claude (actual outbound payload uses 16384)
    # -------------------------------------------------------------------------
    async def test_max_fallback_to_claude(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "OpenAI"
            cfg.fallback_provider = "Claude"
            cfg.fallback_model = "claude-sonnet-5"
            cfg.allow_fallback = True
            await session.commit()

        captured_fallback = []
        async def fake_openai_fail(**kwargs):
            raise RuntimeError("OpenAI primary network error")

        async def fake_claude_fallback(**kwargs):
            captured_fallback.append(kwargs)
            resp = MagicMock()
            part = MagicMock()
            part.text = "MAX Fallback Claude OK"
            resp.content = [part]
            return resp

        with (
            patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_openai_fail),
            patch("anthropic.resources.messages.AsyncMessages.create", side_effect=fake_claude_fallback),
        ):
            resp = await max_ai.get_ai_response(7001, "Тест фолбэк на Claude MAX", track_user_activity=True)
            self.assertEqual(resp, "MAX Fallback Claude OK")

        self.assertEqual(len(captured_fallback), 1)
        self.assertEqual(captured_fallback[0]["max_tokens"], 16384)
        self.assertNotIn("max_completion_tokens", captured_fallback[0])

    # -------------------------------------------------------------------------
    # 15. MAX Fallback -> Gemini (actual outbound payload uses 16384)
    # -------------------------------------------------------------------------
    async def test_max_fallback_to_gemini(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "OpenAI"
            cfg.fallback_provider = "Gemini"
            cfg.fallback_model = "gemini-3.7-flash"
            cfg.allow_fallback = True
            await session.commit()

        captured_fallback = []
        async def fake_openai_fail(**kwargs):
            raise RuntimeError("OpenAI primary network error")

        async def fake_gemini_fallback(url, *args, **kwargs):
            captured_fallback.append(kwargs)
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "candidates": [{"content": {"parts": [{"text": "MAX Fallback Gemini OK"}]}}]
            }
            return resp

        with (
            patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_openai_fail),
            patch("httpx.AsyncClient.post", side_effect=fake_gemini_fallback),
        ):
            resp = await max_ai.get_ai_response(7001, "Тест фолбэк на Gemini MAX", track_user_activity=True)
            self.assertEqual(resp, "MAX Fallback Gemini OK")

        self.assertEqual(len(captured_fallback), 1)
        self.assertEqual(captured_fallback[0]["json"]["generationConfig"]["maxOutputTokens"], 16384)

    # -------------------------------------------------------------------------
    # 16, 17, 18. DeepSeek Direct Telegram & MAX remain 16384
    # -------------------------------------------------------------------------
    async def test_deepseek_direct_chat_budget_telegram_and_max(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "DeepSeek"
            await session.commit()

        captured_tg = []
        async def fake_deepseek_tg(**kwargs):
            captured_tg.append(kwargs)
            return FakeSDKCompletion(choices=[FakeSDKChoice(message=FakeSDKMessage(content="TG DeepSeek OK"))])

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_deepseek_tg):
            resp = await ai_integration.generate_response(user_id=7001, user_prompt="Вопрос к DeepSeek TG")
            self.assertEqual(resp, "TG DeepSeek OK")

        self.assertEqual(captured_tg[0]["max_tokens"], 65536)
        self.assertEqual(captured_tg[0]["max_tokens"], DEEPSEEK_CHAT_MAX_TOKENS)

        captured_max = []
        async def fake_deepseek_max(**kwargs):
            captured_max.append(kwargs)
            return FakeSDKCompletion(choices=[FakeSDKChoice(message=FakeSDKMessage(content="MAX DeepSeek OK"))])

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_deepseek_max):
            resp = await max_ai.get_ai_response(7001, "Вопрос к DeepSeek MAX", track_user_activity=True)
            self.assertEqual(resp, "MAX DeepSeek OK")

        self.assertEqual(captured_max[0]["max_tokens"], 65536)
        self.assertEqual(captured_max[0]["max_tokens"], DEEPSEEK_CHAT_MAX_TOKENS)

    # -------------------------------------------------------------------------
    # 19. KIE Normal Chat Remains 4096
    # -------------------------------------------------------------------------
    async def test_kie_normal_chat_budget_remains_4096(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "KIE"
            await session.commit()

        captured = []
        class FakeKieClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, **kwargs):
                captured.append(kwargs)
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {
                    "choices": [{"message": {"content": "KIE OK"}}]
                }
                return resp

        with patch.object(ai_integration.httpx, "AsyncClient", return_value=FakeKieClient()):
            resp = await ai_integration.generate_response(user_id=7001, user_prompt="Вопрос к KIE")
            self.assertEqual(resp, "KIE OK")

        self.assertEqual(captured[0]["json"]["max_tokens"], 4096)

    # -------------------------------------------------------------------------
    # 20, 21, 22. Telegram Vision Remains 4096
    # -------------------------------------------------------------------------
    async def test_telegram_vision_budgets_remain_4096(self):
        # Telegram OpenAI Vision: 4096
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "OpenAI"
            cfg.vision_model = "gpt-5.6-terra"
            await session.commit()

        captured_tg_v_openai = []
        async def fake_v_openai(**kwargs):
            captured_tg_v_openai.append(kwargs)
            return FakeSDKCompletion(choices=[FakeSDKChoice(message=FakeSDKMessage(content="TG V OpenAI OK"))])

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_v_openai):
            resp = await ai_integration.analyze_image_content(b"fake_image", prompt="Describe")
            self.assertEqual(resp, "TG V OpenAI OK")

        self.assertEqual(captured_tg_v_openai[0]["max_completion_tokens"], 4096)

        # Telegram Claude Vision: 4096
        mock_client = AsyncMock()
        mock_msg = MagicMock()
        mock_part = MagicMock()
        mock_part.type = "text"
        mock_part.text = "TG V Claude OK"
        mock_msg.content = [mock_part]

        captured_tg_v_claude = []
        async def fake_v_claude(**kwargs):
            captured_tg_v_claude.append(kwargs)
            return mock_msg

        mock_client.messages.create.side_effect = fake_v_claude

        with patch.object(ai_integration.anthropic, "AsyncAnthropic", return_value=mock_client):
            resp = await ai_integration._call_claude_vision(
                api_key="sk-test",
                model="claude-sonnet-5",
                image_bytes=b"fake_image",
                prompt="Describe",
            )
            self.assertEqual(resp, "TG V Claude OK")

        self.assertEqual(captured_tg_v_claude[0]["max_tokens"], 4096)

        # Telegram Gemini Vision: 4096
        captured_tg_v_gemini = []
        class FakeGeminiClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, **kwargs):
                captured_tg_v_gemini.append(kwargs)
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {
                    "candidates": [{"content": {"parts": [{"text": "TG V Gemini OK"}]}}]
                }
                return resp

        with patch.object(ai_integration.httpx, "AsyncClient", return_value=FakeGeminiClient()):
            resp = await ai_integration._call_gemini_vision(
                api_key="sk-test",
                model="gemini-3.7-flash",
                image_bytes=b"fake_image",
                prompt="Describe",
            )
            self.assertEqual(resp, "TG V Gemini OK")

        self.assertEqual(captured_tg_v_gemini[0]["json"]["generationConfig"]["maxOutputTokens"], 4096)

    # -------------------------------------------------------------------------
    # 23, 24, 25. MAX Vision Remains 4096
    # -------------------------------------------------------------------------
    async def test_max_vision_budgets_remain_4096(self):
        # MAX OpenAI Vision: 4096
        captured_max_v_openai = []
        async def fake_v_openai(**kwargs):
            captured_max_v_openai.append(kwargs)
            return FakeSDKCompletion(choices=[FakeSDKChoice(message=FakeSDKMessage(content="MAX V OpenAI OK"))])

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_v_openai):
            resp = await max_ai._analyze_openai(
                api_key="sk-test",
                model="gpt-5.6-terra",
                image_bytes=b"fake_image",
                system_prompt="sys",
                prompt="Describe",
                temperature=0.7,
            )
            self.assertEqual(resp, "MAX V OpenAI OK")

        self.assertEqual(captured_max_v_openai[0]["max_completion_tokens"], 4096)

        # MAX Claude Vision: 4096
        captured_max_v_claude = []
        async def fake_v_claude(**kwargs):
            captured_max_v_claude.append(kwargs)
            resp = MagicMock()
            part = MagicMock()
            part.text = "MAX V Claude OK"
            resp.content = [part]
            return resp

        with patch("anthropic.resources.messages.AsyncMessages.create", side_effect=fake_v_claude):
            resp = await max_ai._analyze_claude(
                api_key="sk-test",
                model="claude-sonnet-5",
                image_bytes=b"fake_image",
                system_prompt="sys",
                prompt="Describe",
                temperature=0.7,
            )
            self.assertEqual(resp, "MAX V Claude OK")

        self.assertEqual(captured_max_v_claude[0]["max_tokens"], 4096)

        # MAX Gemini Vision: 4096
        captured_max_v_gemini = []
        async def fake_v_gemini(url, *args, **kwargs):
            captured_max_v_gemini.append(kwargs)
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "candidates": [{"content": {"parts": [{"text": "MAX V Gemini OK"}]}}]
            }
            return resp

        with patch("httpx.AsyncClient.post", side_effect=fake_v_gemini):
            resp = await max_ai._analyze_gemini(
                api_key="sk-test",
                model="gemini-3.7-flash",
                image_bytes=b"fake_image",
                prompt="Describe",
                system_prompt="sys",
                temperature=0.7,
            )
            self.assertEqual(resp, "MAX V Gemini OK")

        self.assertEqual(captured_max_v_gemini[0]["json"]["generationConfig"]["maxOutputTokens"], 4096)

    # -------------------------------------------------------------------------
    # 26, 27, 28. request_capture & AILog Schemas Unchanged & Value Reflection
    # -------------------------------------------------------------------------
    async def test_request_capture_and_ailog_schema_unchanged_reflects_16384(self):
        capture = {}
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "OpenAI"
            await session.commit()

        async def fake_create(**kwargs):
            return FakeSDKCompletion(choices=[FakeSDKChoice(message=FakeSDKMessage(content="Capture OK"))])

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create):
            await ai_integration.generate_response(
                user_id=7001,
                user_prompt="Проверка захвата",
                response_capture=capture,
            )

        # Verify AILog schema and content
        async with self.sessions() as session:
            log_entry = await session.scalar(select(AILog).order_by(AILog.id.desc()))
            self.assertIsNotNone(log_entry)
            self.assertEqual(log_entry.provider, "OpenAI")
            self.assertEqual(log_entry.model, "gpt-5.6-terra")

            parsed = json.loads(log_entry.request_payload)
            # Schema unchanged: keys are 'provider', 'endpoint', 'payload'
            self.assertEqual(set(parsed.keys()), {"provider", "endpoint", "payload"})
            self.assertEqual(parsed["provider"], "OpenAI")
            self.assertEqual(parsed["payload"]["max_completion_tokens"], 16384)
            self.assertNotIn("sk-", parsed["endpoint"])
            self.assertNotIn("sk-", log_entry.request_payload)

    # -------------------------------------------------------------------------
    # 29. Telegram xAI Direct Chat Budget Remains 4096 (xAI safety isolation)
    # -------------------------------------------------------------------------
    async def test_telegram_xai_direct_chat_budget_remains_4096(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "xAI"
            await session.commit()

        captured = []
        async def fake_create(**kwargs):
            captured.append(kwargs)
            return FakeSDKCompletion(choices=[FakeSDKChoice(message=FakeSDKMessage(content="TG xAI OK"))])

        with (
            patch.object(AIConfig, "xai_api_key", "sk-xai-test-key", create=True),
            patch.object(AIConfig, "xai_model", "gpt-5.6-terra", create=True),
            patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create),
        ):
            resp = await ai_integration.generate_response(user_id=7001, user_prompt="Вопрос к xAI TG")
            self.assertEqual(resp, "TG xAI OK")

        self.assertEqual(len(captured), 1)
        wire_payload = captured[0]
        self.assertEqual(wire_payload["max_completion_tokens"], 4096)
        self.assertNotIn("max_tokens", wire_payload)

        # AILog verification
        async with self.sessions() as session:
            log_entry = await session.scalar(select(AILog).where(AILog.platform == "telegram").order_by(AILog.id.desc()))
            self.assertIsNotNone(log_entry)
            parsed = json.loads(log_entry.request_payload)
            self.assertEqual(parsed["payload"]["max_completion_tokens"], 4096)
            self.assertNotIn("max_tokens", parsed["payload"])

    # -------------------------------------------------------------------------
    # 30. Telegram Fallback -> xAI Remains 4096 (does not inherit 16384)
    # -------------------------------------------------------------------------
    async def test_telegram_fallback_to_xai_remains_4096(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "OpenAI"
            cfg.allow_fallback = True
            cfg.fallback_provider = "xAI"
            cfg.fallback_model = "gpt-5.6-terra"
            await session.commit()

        captured_calls = []
        async def fake_create(**kwargs):
            captured_calls.append(kwargs)
            if len(captured_calls) == 1:
                raise RuntimeError("OpenAI primary failure")
            return FakeSDKCompletion(choices=[FakeSDKChoice(message=FakeSDKMessage(content="xAI fallback OK"))])

        with (
            patch.object(AIConfig, "xai_api_key", "sk-xai-test-key", create=True),
            patch.object(AIConfig, "xai_model", "gpt-5.6-terra", create=True),
            patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create),
        ):
            resp = await ai_integration.generate_response(user_id=7001, user_prompt="Тест фоллбэка на xAI")
            self.assertEqual(resp, "xAI fallback OK")

        self.assertEqual(len(captured_calls), 2)
        # Primary OpenAI got 16384
        self.assertEqual(captured_calls[0]["max_completion_tokens"], 16384)
        # Fallback xAI got 4096
        self.assertEqual(captured_calls[1]["max_completion_tokens"], 4096)

    # -------------------------------------------------------------------------
    # 31. Telegram Fallback from xAI -> OpenAI (OpenAI gets 16384)
    # -------------------------------------------------------------------------
    async def test_telegram_fallback_from_xai_to_openai(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "xAI"
            cfg.allow_fallback = True
            cfg.fallback_provider = "OpenAI"
            cfg.fallback_model = "gpt-5.6-terra"
            cfg.openai_api_key = "sk-openai-key"
            await session.commit()

        captured_calls = []
        async def fake_create(**kwargs):
            captured_calls.append(kwargs)
            if len(captured_calls) == 1:
                raise RuntimeError("xAI primary failure")
            return FakeSDKCompletion(choices=[FakeSDKChoice(message=FakeSDKMessage(content="OpenAI fallback OK"))])

        with (
            patch.object(AIConfig, "xai_api_key", "sk-xai-test-key", create=True),
            patch.object(AIConfig, "xai_model", "gpt-5.6-terra", create=True),
            patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create),
        ):
            resp = await ai_integration.generate_response(user_id=7001, user_prompt="Тест фоллбэка с xAI на OpenAI")
            self.assertEqual(resp, "OpenAI fallback OK")

        self.assertEqual(len(captured_calls), 2)
        # Primary xAI was 4096
        self.assertEqual(captured_calls[0]["max_completion_tokens"], 4096)
        # Fallback OpenAI received 16384
        self.assertEqual(captured_calls[1]["max_completion_tokens"], 16384)

    # -------------------------------------------------------------------------
    # 32. _call_openai_api Defaults Safely to 4096
    # -------------------------------------------------------------------------
    async def test_call_openai_api_defaults_to_4096(self):
        captured = []
        async def fake_create(**kwargs):
            captured.append(kwargs)
            return FakeSDKCompletion(choices=[FakeSDKChoice(message=FakeSDKMessage(content="Default OK"))])

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create):
            resp = await ai_integration._call_openai_api(
                api_key="sk-test",
                model="gpt-5.6-terra",
                history=[],
                context="",
                system_prompt="sys",
            )
            self.assertEqual(resp, "Default OK")

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["max_completion_tokens"], 4096)

    # -------------------------------------------------------------------------
    # 33. request_capture Reflects Final Successful Attempt Only (no failed attempt leakage)
    # -------------------------------------------------------------------------
    async def test_fallback_request_capture_reflects_final_successful_attempt_only(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "OpenAI"
            cfg.allow_fallback = True
            cfg.fallback_provider = "Claude"
            cfg.fallback_model = "claude-sonnet-5"
            cfg.claude_api_key = "sk-claude-test-key"
            await session.commit()

        async def fake_openai_fail(**kwargs):
            raise RuntimeError("OpenAI outage")

        mock_client = AsyncMock()
        mock_msg = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Claude fallback success"
        mock_msg.content = [mock_part]

        async def fake_claude_success(**kwargs):
            return mock_msg

        mock_client.messages.create.side_effect = fake_claude_success

        with (
            patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_openai_fail),
            patch.object(ai_integration.anthropic, "AsyncAnthropic", return_value=mock_client),
        ):
            resp = await ai_integration.generate_response(
                user_id=7001,
                user_prompt="Тест захвата запроса при фоллбэке",
            )
            self.assertEqual(resp, "Claude fallback success")

        # AILog / request_capture reflects final successful provider (Claude), NOT failed OpenAI
        async with self.sessions() as session:
            log_entry = await session.scalar(select(AILog).where(AILog.platform == "telegram").order_by(AILog.id.desc()))
            self.assertIsNotNone(log_entry)
            parsed = json.loads(log_entry.request_payload)
            self.assertEqual(parsed["provider"], "Claude")
            self.assertEqual(parsed["endpoint"], "https://api.anthropic.com/v1/messages")
            self.assertEqual(parsed["payload"]["max_tokens"], 16384)
            self.assertNotIn("max_completion_tokens", parsed["payload"])

