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
    DEEPSEEK_CHAT_MAX_TOKENS,
    DeepSeekDiagnostics,
    PROVIDER_DEEPSEEK,
    inspect_deepseek_response,
    normalize_deepseek_model,
)

SENTINEL_LEAK_SECRET = "SECRET_REASONING_MUST_NOT_LEAK"


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
    def __init__(self, choices=None, model="deepseek-v4-flash"):
        self.choices = choices or []
        self.model = model


class DeepSeekHotfixBudgetAndDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
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
                provider="DeepSeek",
                deepseek_api_key="sk-deepseek-test-key",
                deepseek_model="deepseek-v4-flash",
                openai_api_key="sk-openai-test-key",
                openai_model="gpt-5.6-terra",
                claude_api_key="sk-claude-fallback-key",
                claude_model="claude-sonnet-5",
                gemini_api_key="sk-gemini-test-key",
                gemini_model="gemini-3.7-flash",
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
    # 1. Pure Helper Unit Tests (SDK object shapes, dict shapes, security invariant)
    # -------------------------------------------------------------------------

    def test_inspect_deepseek_response_case_a_normal_stop(self):
        resp = FakeSDKCompletion(
            choices=[FakeSDKChoice(finish_reason="stop", message=FakeSDKMessage(content="Normal visible answer"))]
        )
        content, diag = inspect_deepseek_response(resp, model="deepseek-v4-flash", platform="telegram")
        self.assertEqual(content, "Normal visible answer")
        self.assertEqual(diag.provider, "Deepseek")
        self.assertEqual(diag.model, "deepseek-v4-flash")
        self.assertEqual(diag.platform, "telegram")
        self.assertEqual(diag.finish_reason, "stop")
        self.assertTrue(diag.visible_content_present)
        self.assertEqual(diag.visible_content_length, len("Normal visible answer"))
        self.assertFalse(diag.reasoning_content_present)
        self.assertEqual(diag.reasoning_content_length, 0)
        self.assertFalse(diag.output_budget_exhausted)

    def test_inspect_deepseek_response_case_b_length_with_partial_visible_content(self):
        resp = FakeSDKCompletion(
            choices=[
                FakeSDKChoice(
                    finish_reason="length",
                    message=FakeSDKMessage(
                        content="Partial visible text before cutoff",
                        reasoning_content=SENTINEL_LEAK_SECRET,
                    ),
                )
            ]
        )
        content, diag = inspect_deepseek_response(resp, model="deepseek-v4-flash", platform="telegram")
        self.assertEqual(content, "Partial visible text before cutoff")
        self.assertEqual(diag.finish_reason, "length")
        self.assertTrue(diag.visible_content_present)
        self.assertEqual(diag.visible_content_length, len("Partial visible text before cutoff"))
        self.assertTrue(diag.reasoning_content_present)
        self.assertEqual(diag.reasoning_content_length, len(SENTINEL_LEAK_SECRET))
        self.assertTrue(diag.output_budget_exhausted)
        # Security invariant: Sentinel string is nowhere in diag
        self.assertNotIn(SENTINEL_LEAK_SECRET, str(diag))

    def test_inspect_deepseek_response_case_c_length_with_empty_visible_content(self):
        resp = FakeSDKCompletion(
            choices=[
                FakeSDKChoice(
                    finish_reason="length",
                    message=FakeSDKMessage(
                        content="",
                        reasoning_content=SENTINEL_LEAK_SECRET,
                    ),
                )
            ]
        )
        content, diag = inspect_deepseek_response(resp, model="deepseek-v4-flash", platform="max")
        self.assertIsNone(content)
        self.assertEqual(diag.platform, "max")
        self.assertEqual(diag.finish_reason, "length")
        self.assertFalse(diag.visible_content_present)
        self.assertEqual(diag.visible_content_length, 0)
        self.assertTrue(diag.reasoning_content_present)
        self.assertEqual(diag.reasoning_content_length, len(SENTINEL_LEAK_SECRET))
        self.assertTrue(diag.output_budget_exhausted)
        self.assertNotIn(SENTINEL_LEAK_SECRET, str(diag))

    def test_inspect_deepseek_response_case_d_non_length_empty_content(self):
        resp = FakeSDKCompletion(
            choices=[FakeSDKChoice(finish_reason="stop", message=FakeSDKMessage(content="   "))]
        )
        content, diag = inspect_deepseek_response(resp, model="deepseek-v4-flash", platform="telegram")
        self.assertIsNone(content)
        self.assertFalse(diag.visible_content_present)
        self.assertFalse(diag.output_budget_exhausted)

    def test_inspect_deepseek_response_pydantic_model_extra_support(self):
        resp = FakeSDKCompletion(
            choices=[
                FakeSDKChoice(
                    finish_reason="length",
                    message=FakeSDKMessage(
                        content="Visible text",
                        reasoning_content=None,
                        extra={"reasoning_content": SENTINEL_LEAK_SECRET},
                    ),
                )
            ]
        )
        content, diag = inspect_deepseek_response(resp, model="deepseek-v4-flash", platform="telegram")
        self.assertEqual(content, "Visible text")
        self.assertTrue(diag.reasoning_content_present)
        self.assertEqual(diag.reasoning_content_length, len(SENTINEL_LEAK_SECRET))
        self.assertNotIn(SENTINEL_LEAK_SECRET, str(diag))

    def test_inspect_deepseek_response_dict_shape(self):
        dict_resp = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": "Dict visible text",
                        "reasoning_content": SENTINEL_LEAK_SECRET,
                    },
                }
            ]
        }
        content, diag = inspect_deepseek_response(dict_resp, model="deepseek-v4-pro", platform="max")
        self.assertEqual(content, "Dict visible text")
        self.assertEqual(diag.model, "deepseek-v4-pro")
        self.assertEqual(diag.platform, "max")
        self.assertTrue(diag.output_budget_exhausted)
        self.assertTrue(diag.reasoning_content_present)
        self.assertEqual(diag.reasoning_content_length, len(SENTINEL_LEAK_SECRET))
        self.assertNotIn(SENTINEL_LEAK_SECRET, str(diag))

    def test_inspect_deepseek_response_none_and_empty_choices(self):
        content, diag = inspect_deepseek_response(None, model="deepseek-v4-flash", platform="telegram")
        self.assertIsNone(content)
        self.assertFalse(diag.visible_content_present)
        self.assertFalse(diag.output_budget_exhausted)

        content2, diag2 = inspect_deepseek_response({"choices": []}, model="deepseek-v4-flash", platform="telegram")
        self.assertIsNone(content2)
        self.assertFalse(diag2.visible_content_present)

    # -------------------------------------------------------------------------
    # 2. Wire Payload & Model Tests (Telegram & MAX: flash, pro, aliases)
    # -------------------------------------------------------------------------

    async def test_telegram_deepseek_v4_flash_outbound_max_tokens_is_16384(self):
        captured_payloads = []

        async def mock_create(**kwargs):
            captured_payloads.append(kwargs)
            return FakeSDKCompletion(
                choices=[FakeSDKChoice(finish_reason="stop", message=FakeSDKMessage(content="Ответ Flash TG"))]
            )

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_create):
            resp = await ai_integration.generate_response(user_id=7001, user_prompt="Тест Flash TG")
            self.assertEqual(resp, "Ответ Flash TG")

        self.assertEqual(len(captured_payloads), 1)
        wire_payload = captured_payloads[0]
        self.assertEqual(wire_payload["model"], "deepseek-v4-flash")
        self.assertEqual(wire_payload["max_tokens"], 16384)
        self.assertEqual(wire_payload.get("extra_body"), {"thinking": {"type": "disabled"}})
        self.assertNotIn("max_completion_tokens", wire_payload)

    async def test_telegram_deepseek_v4_pro_outbound_max_tokens_is_16384(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.deepseek_model = "deepseek-v4-pro"
            await session.commit()

        captured_payloads = []

        async def mock_create(**kwargs):
            captured_payloads.append(kwargs)
            return FakeSDKCompletion(
                choices=[FakeSDKChoice(finish_reason="stop", message=FakeSDKMessage(content="Ответ Pro TG"))]
            )

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_create):
            resp = await ai_integration.generate_response(user_id=7001, user_prompt="Тест Pro TG")
            self.assertEqual(resp, "Ответ Pro TG")

        self.assertEqual(len(captured_payloads), 1)
        wire_payload = captured_payloads[0]
        self.assertEqual(wire_payload["model"], "deepseek-v4-pro")
        self.assertEqual(wire_payload["max_tokens"], 16384)
        self.assertEqual(wire_payload.get("extra_body"), {"thinking": {"type": "disabled"}})

    async def test_max_deepseek_v4_flash_outbound_max_tokens_is_16384(self):
        captured_payloads = []

        async def mock_create(**kwargs):
            captured_payloads.append(kwargs)
            return FakeSDKCompletion(
                choices=[FakeSDKChoice(finish_reason="stop", message=FakeSDKMessage(content="Ответ Flash MAX"))]
            )

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_create):
            resp = await max_ai.get_ai_response(7001, "Тест Flash MAX")
            self.assertEqual(resp, "Ответ Flash MAX")

        self.assertEqual(len(captured_payloads), 1)
        wire_payload = captured_payloads[0]
        self.assertEqual(wire_payload["model"], "deepseek-v4-flash")
        self.assertEqual(wire_payload["max_tokens"], 16384)
        self.assertEqual(wire_payload.get("extra_body"), {"thinking": {"type": "disabled"}})
        self.assertNotIn("max_completion_tokens", wire_payload)

    async def test_max_deepseek_v4_pro_outbound_max_tokens_is_16384(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.deepseek_model = "deepseek-v4-pro"
            await session.commit()

        captured_payloads = []

        async def mock_create(**kwargs):
            captured_payloads.append(kwargs)
            return FakeSDKCompletion(
                choices=[FakeSDKChoice(finish_reason="stop", message=FakeSDKMessage(content="Ответ Pro MAX"))]
            )

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_create):
            resp = await max_ai.get_ai_response(7001, "Тест Pro MAX")
            self.assertEqual(resp, "Ответ Pro MAX")

        self.assertEqual(len(captured_payloads), 1)
        wire_payload = captured_payloads[0]
        self.assertEqual(wire_payload["model"], "deepseek-v4-pro")
        self.assertEqual(wire_payload["max_tokens"], 16384)
        self.assertEqual(wire_payload.get("extra_body"), {"thinking": {"type": "disabled"}})

    async def test_deepseek_legacy_aliases_normalized_and_receive_16384(self):
        self.assertEqual(normalize_deepseek_model("deepseek-chat"), "deepseek-v4-flash")
        self.assertEqual(normalize_deepseek_model("deepseek-reasoner"), "deepseek-v4-flash")
        self.assertEqual(normalize_deepseek_model("deepseek-coder"), "deepseek-v4-flash")

        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.deepseek_model = "deepseek-chat"
            await session.commit()

        captured = []

        async def mock_create(**kwargs):
            captured.append(kwargs)
            return FakeSDKCompletion(
                choices=[FakeSDKChoice(finish_reason="stop", message=FakeSDKMessage(content="Ответ Legacy Alias"))]
            )

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_create):
            resp = await ai_integration.generate_response(user_id=7001, user_prompt="Тест Alias")
            self.assertEqual(resp, "Ответ Legacy Alias")

        self.assertEqual(captured[0]["model"], "deepseek-v4-flash")
        self.assertEqual(captured[0]["max_tokens"], 16384)
        self.assertEqual(captured[0].get("extra_body"), {"thinking": {"type": "disabled"}})

    # -------------------------------------------------------------------------
    # 3. Partial Content Semantics (finish_reason == "length" with non-empty content)
    # -------------------------------------------------------------------------

    async def test_telegram_finish_reason_length_returns_partial_visible_content_without_fallback(self):
        captured_logs = []

        class LogHandler(logging.Handler):
            def emit(self, record):
                captured_logs.append(record.getMessage())

        handler = LogHandler()
        logging.getLogger().addHandler(handler)

        try:
            async def mock_create(**kwargs):
                return FakeSDKCompletion(
                    choices=[
                        FakeSDKChoice(
                            finish_reason="length",
                            message=FakeSDKMessage(
                                content="Частичный ответ клиенту, который не должен быть сброшен",
                                reasoning_content="Скрытые рассуждения модели",
                            ),
                        )
                    ]
                )

            with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_create):
                resp = await ai_integration.generate_response(user_id=7001, user_prompt="Дай длинный ответ")
                self.assertEqual(resp, "Частичный ответ клиенту, который не должен быть сброшен")

            # Verify exhaustion warning was logged
            exhaustion_logs = [l for l in captured_logs if "DeepSeek output budget exhausted" in l]
            self.assertTrue(len(exhaustion_logs) > 0)
            self.assertIn("finish_reason=length", exhaustion_logs[0])
            self.assertIn("visible_content_present=True", exhaustion_logs[0])
            self.assertIn("output_budget_exhausted=True", exhaustion_logs[0])

            # Verify AILog contains the partial content, not fallback
            async with self.sessions() as session:
                ai_log = await session.scalar(
                    select(AILog).where(AILog.platform == "telegram").order_by(AILog.id.desc())
                )
                self.assertIsNotNone(ai_log)
                self.assertEqual(ai_log.provider, "DeepSeek")
                self.assertEqual(ai_log.clean_text, "Частичный ответ клиенту, который не должен быть сброшен")
        finally:
            logging.getLogger().removeHandler(handler)

    async def test_max_finish_reason_length_returns_partial_visible_content_without_fallback(self):
        captured_logs = []

        class LogHandler(logging.Handler):
            def emit(self, record):
                captured_logs.append(record.getMessage())

        handler = LogHandler()
        logging.getLogger().addHandler(handler)
        max_ai.log.addHandler(handler)

        try:
            async def mock_create(**kwargs):
                return FakeSDKCompletion(
                    choices=[
                        FakeSDKChoice(
                            finish_reason="length",
                            message=FakeSDKMessage(
                                content="Частичный ответ MAX клиенту",
                                reasoning_content="MAX рассуждения",
                            ),
                        )
                    ]
                )

            with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_create):
                resp = await max_ai.get_ai_response(7001, "Дай длинный ответ MAX")
                self.assertEqual(resp, "Частичный ответ MAX клиенту")

            exhaustion_logs = [l for l in captured_logs if "DeepSeek output budget exhausted" in l]
            self.assertTrue(len(exhaustion_logs) > 0)
            self.assertIn("platform=max", exhaustion_logs[0])
            self.assertIn("visible_content_present=True", exhaustion_logs[0])

            async with self.sessions() as session:
                ai_log = await session.scalar(
                    select(AILog).where(AILog.platform == "max").order_by(AILog.id.desc())
                )
                self.assertIsNotNone(ai_log)
                self.assertEqual(ai_log.clean_text, "Частичный ответ MAX клиенту")
        finally:
            logging.getLogger().removeHandler(handler)
            max_ai.log.removeHandler(handler)

    # -------------------------------------------------------------------------
    # 4. Fallback Execution & Anti-Leak Sentinel Proof (Telegram & MAX)
    # -------------------------------------------------------------------------

    async def test_telegram_empty_visible_length_triggers_fallback_and_anti_leak(self):
        """When DeepSeek hits length with empty visible content, existing fallback executes,
        and SECRET_REASONING_MUST_NOT_LEAK never leaks to any observable surface."""
        captured_logs = []

        class LogHandler(logging.Handler):
            def emit(self, record):
                captured_logs.append(record.getMessage())

        handler = LogHandler()
        logging.getLogger().addHandler(handler)

        try:
            async def mock_deepseek(**kwargs):
                return FakeSDKCompletion(
                    choices=[
                        FakeSDKChoice(
                            finish_reason="length",
                            message=FakeSDKMessage(
                                content="",
                                reasoning_content=SENTINEL_LEAK_SECRET,
                            ),
                        )
                    ]
                )

            async def mock_claude_create(**kwargs):
                mock_msg = MagicMock()
                mock_part = MagicMock()
                mock_part.text = "Надёжный ответ от резервного Claude"
                mock_msg.content = [mock_part]
                return mock_msg

            with (
                patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_deepseek),
                patch("anthropic.resources.messages.AsyncMessages.create", side_effect=mock_claude_create),
            ):
                user_visible_resp = await ai_integration.generate_response(
                    user_id=7001,
                    user_prompt="Вопрос с исчерпанием бюджета",
                )

            # 1. Fallback returned to user
            self.assertEqual(user_visible_resp, "Надёжный ответ от резервного Claude")

            # 2. Assert sentinel is NOT in returned text
            self.assertNotIn(SENTINEL_LEAK_SECRET, user_visible_resp)

            # 3. Assert sentinel is NOT in captured logs
            all_logs_text = " ".join(captured_logs)
            self.assertNotIn(SENTINEL_LEAK_SECRET, all_logs_text)
            self.assertIn("DeepSeek output budget exhausted", all_logs_text)
            self.assertIn("visible_content_present=False", all_logs_text)
            self.assertIn("falling back to 'Claude'", all_logs_text)

            # 4. Assert sentinel is NOT in AILog
            async with self.sessions() as session:
                ai_log = await session.scalar(
                    select(AILog).where(AILog.platform == "telegram").order_by(AILog.id.desc())
                )
                self.assertIsNotNone(ai_log)
                self.assertEqual(ai_log.provider, "Claude")
                self.assertEqual(ai_log.model, "claude-sonnet-5")
                self.assertNotIn(SENTINEL_LEAK_SECRET, ai_log.clean_text or "")
                self.assertNotIn(SENTINEL_LEAK_SECRET, ai_log.raw_response or "")
                self.assertNotIn(SENTINEL_LEAK_SECRET, ai_log.request_payload or "")
        finally:
            logging.getLogger().removeHandler(handler)

    async def test_max_empty_visible_length_triggers_fallback_and_anti_leak(self):
        """When MAX DeepSeek hits length with empty visible content, existing fallback executes,
        and SECRET_REASONING_MUST_NOT_LEAK never leaks to any observable surface."""
        captured_logs = []

        class LogHandler(logging.Handler):
            def emit(self, record):
                captured_logs.append(record.getMessage())

        handler = LogHandler()
        logging.getLogger().addHandler(handler)
        max_ai.log.addHandler(handler)

        try:
            async def mock_deepseek(**kwargs):
                return FakeSDKCompletion(
                    choices=[
                        FakeSDKChoice(
                            finish_reason="length",
                            message=FakeSDKMessage(
                                content="",
                                reasoning_content=SENTINEL_LEAK_SECRET,
                            ),
                        )
                    ]
                )

            async def mock_claude_create(**kwargs):
                mock_msg = MagicMock()
                mock_part = MagicMock()
                mock_part.text = "MAX ответ от резервного Claude"
                mock_msg.content = [mock_part]
                return mock_msg

            with (
                patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_deepseek),
                patch("anthropic.resources.messages.AsyncMessages.create", side_effect=mock_claude_create),
            ):
                user_visible_resp = await max_ai.get_ai_response(
                    7001,
                    "MAX вопрос с исчерпанием бюджета",
                )

            # 1. Fallback returned to user
            self.assertEqual(user_visible_resp, "MAX ответ от резервного Claude")

            # 2. Assert sentinel is NOT in returned text
            self.assertNotIn(SENTINEL_LEAK_SECRET, user_visible_resp)

            # 3. Assert sentinel is NOT in captured logs
            all_logs_text = " ".join(captured_logs)
            self.assertNotIn(SENTINEL_LEAK_SECRET, all_logs_text)
            self.assertIn("DeepSeek output budget exhausted", all_logs_text)
            self.assertIn("platform=max", all_logs_text)
            self.assertIn("falling back to 'Claude'", all_logs_text)

            # 4. Assert sentinel is NOT in AILog
            async with self.sessions() as session:
                ai_log = await session.scalar(
                    select(AILog).where(AILog.platform == "max").order_by(AILog.id.desc())
                )
                self.assertIsNotNone(ai_log)
                self.assertEqual(ai_log.provider, "Claude")
                self.assertEqual(ai_log.model, "claude-sonnet-5")
                self.assertNotIn(SENTINEL_LEAK_SECRET, ai_log.clean_text or "")
                self.assertNotIn(SENTINEL_LEAK_SECRET, ai_log.raw_response or "")
                self.assertNotIn(SENTINEL_LEAK_SECRET, ai_log.request_payload or "")
        finally:
            logging.getLogger().removeHandler(handler)
            max_ai.log.removeHandler(handler)

    # -------------------------------------------------------------------------
    # 5. DeepSeek as Fallback Provider (receives 16384 through direct adapter)
    # -------------------------------------------------------------------------

    async def test_deepseek_as_fallback_provider_receives_16384(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "Claude"
            cfg.fallback_provider = "DeepSeek"
            cfg.fallback_model = "deepseek-v4-flash"
            await session.commit()

        captured_deepseek = []

        async def mock_claude_fail(**kwargs):
            raise RuntimeError("Claude primary network timeout")

        async def mock_deepseek_fallback(**kwargs):
            captured_deepseek.append(kwargs)
            return FakeSDKCompletion(
                choices=[FakeSDKChoice(finish_reason="stop", message=FakeSDKMessage(content="Ответ DeepSeek фолбэк"))]
            )

        with (
            patch("anthropic.resources.messages.AsyncMessages.create", side_effect=mock_claude_fail),
            patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_deepseek_fallback),
        ):
            resp = await ai_integration.generate_response(user_id=7001, user_prompt="Тест DeepSeek фолбэк")
            self.assertEqual(resp, "Ответ DeepSeek фолбэк")

        self.assertEqual(len(captured_deepseek), 1)
        self.assertEqual(captured_deepseek[0]["model"], "deepseek-v4-flash")
        self.assertEqual(captured_deepseek[0]["max_tokens"], 16384)

    # -------------------------------------------------------------------------
    # 6. Negative / Invariant Confirmation: Direct Chat 16384 & KIE / Vision 4096
    # -------------------------------------------------------------------------

    async def test_direct_normal_chat_budgets_are_16384_and_kie_vision_remain_4096(self):
        # 1. OpenAI: max_completion_tokens == 16384
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "OpenAI"
            await session.commit()

        openai_captured = []

        async def mock_openai(**kwargs):
            openai_captured.append(kwargs)
            return FakeSDKCompletion(
                choices=[FakeSDKChoice(finish_reason="stop", message=FakeSDKMessage(content="OpenAI ok"))]
            )

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_openai):
            await ai_integration.generate_response(user_id=7001, user_prompt="Тест OpenAI")

        self.assertEqual(openai_captured[0]["max_completion_tokens"], 16384)
        self.assertNotIn("max_tokens", openai_captured[0])

        # 2. Claude: max_tokens == 16384
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "Claude"
            await session.commit()

        claude_captured = []
        mock_claude_client = AsyncMock()
        mock_msg = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Claude ok"
        mock_msg.content = [mock_part]

        async def mock_claude(**kwargs):
            claude_captured.append(kwargs)
            return mock_msg

        mock_claude_client.messages.create.side_effect = mock_claude_create = mock_claude

        with patch.object(ai_integration.anthropic, "AsyncAnthropic", return_value=mock_claude_client):
            await ai_integration.generate_response(user_id=7001, user_prompt="Тест Claude")

        self.assertEqual(claude_captured[0]["max_tokens"], 16384)
        self.assertNotIn("max_completion_tokens", claude_captured[0])

        # 3. Gemini: maxOutputTokens == 16384
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "Gemini"
            await session.commit()

        gemini_captured = []

        class FakeGeminiResponse:
            status_code = 200
            def json(self):
                return {"candidates": [{"content": {"parts": [{"text": "Gemini ok"}]}}]}

        class FakeGeminiClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, **kwargs):
                gemini_captured.append(kwargs)
                return FakeGeminiResponse()

        with patch.object(ai_integration.httpx, "AsyncClient", return_value=FakeGeminiClient()):
            await ai_integration.generate_response(user_id=7001, user_prompt="Тест Gemini")

        self.assertEqual(gemini_captured[0]["json"]["generationConfig"]["maxOutputTokens"], 16384)

        # 4. DeepSeek: max_tokens == 16384
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "DeepSeek"
            await session.commit()

        deepseek_captured = []

        async def mock_deepseek(**kwargs):
            deepseek_captured.append(kwargs)
            return FakeSDKCompletion(
                choices=[FakeSDKChoice(finish_reason="stop", message=FakeSDKMessage(content="DeepSeek ok"))]
            )

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_deepseek):
            await ai_integration.generate_response(user_id=7001, user_prompt="Тест DeepSeek")

        self.assertEqual(deepseek_captured[0]["max_tokens"], 16384)
        self.assertNotIn("max_completion_tokens", deepseek_captured[0])

        # 5. KIE: max_tokens == 4096 (NON-TARGET remains 4096)
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "KIE"
            cfg.kie_api_key = "sk-kie-test-key"
            cfg.kie_model = "gemini-3-flash"
            await session.commit()

        kie_captured = []

        class FakeKieResponse:
            status_code = 200
            def json(self):
                return {"choices": [{"message": {"content": "KIE ok"}}]}

        class FakeKieClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, **kwargs):
                kie_captured.append(kwargs)
                return FakeKieResponse()

        with patch.object(ai_integration.httpx, "AsyncClient", return_value=FakeKieClient()):
            await ai_integration.generate_response(user_id=7001, user_prompt="Тест KIE")

        self.assertEqual(kie_captured[0]["json"]["max_tokens"], 4096)

    async def test_vision_budgets_remain_4096_across_providers(self):
        # Explicit test proving vision paths remain 4096
        # 1. Telegram OpenAI Vision: max_completion_tokens == 4096
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "OpenAI"
            cfg.openai_api_key = "sk-openai-test-key"
            cfg.vision_model = "gpt-5.6-terra"
            await session.commit()

        openai_vision_captured = []
        async def mock_v_openai(**kwargs):
            openai_vision_captured.append(kwargs)
            return FakeSDKCompletion(
                choices=[FakeSDKChoice(finish_reason="stop", message=FakeSDKMessage(content="OpenAI Vision ok"))]
            )
        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_v_openai):
            await ai_integration.analyze_image_content(
                image_bytes=b"fake_image_bytes",
                prompt="Describe",
            )
        self.assertEqual(openai_vision_captured[0]["max_completion_tokens"], 4096)

        # 2. Telegram Claude Vision: max_tokens == 4096
        claude_vision_captured = []
        mock_v_claude_client = AsyncMock()
        mock_v_msg = MagicMock()
        mock_v_part = MagicMock()
        mock_v_part.type = "text"
        mock_v_part.text = "Claude Vision ok"
        mock_v_msg.content = [mock_v_part]
        async def mock_v_claude(**kwargs):
            claude_vision_captured.append(kwargs)
            return mock_v_msg
        mock_v_claude_client.messages.create.side_effect = mock_v_claude
        with patch.object(ai_integration.anthropic, "AsyncAnthropic", return_value=mock_v_claude_client):
            await ai_integration._call_claude_vision(
                api_key="sk-test",
                model="claude-sonnet-5",
                image_bytes=b"fake_image_bytes",
                prompt="Describe",
            )
        self.assertEqual(claude_vision_captured[0]["max_tokens"], 4096)

        # 3. Telegram Gemini Vision: generationConfig.maxOutputTokens == 4096
        gemini_vision_captured = []
        class FakeGeminiVisionResponse:
            status_code = 200
            def json(self):
                return {"candidates": [{"content": {"parts": [{"text": "Gemini Vision ok"}]}}]}
        class FakeGeminiVisionClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, **kwargs):
                gemini_vision_captured.append(kwargs)
                return FakeGeminiVisionResponse()
        with patch.object(ai_integration.httpx, "AsyncClient", return_value=FakeGeminiVisionClient()):
            await ai_integration._call_gemini_vision(
                api_key="sk-test",
                model="gemini-3.7-flash",
                image_bytes=b"fake_image_bytes",
                prompt="Describe",
            )
        self.assertEqual(gemini_vision_captured[0]["json"]["generationConfig"]["maxOutputTokens"], 4096)
        self.assertNotIn("extra_body", gemini_vision_captured[0]["json"])

    # -------------------------------------------------------------------------
    # 8. Hotfix B1: Thinking Disabled & Request Capture Regression Tests
    # -------------------------------------------------------------------------

    async def test_deepseek_thinking_disabled_in_wire_payload_and_request_capture_both_platforms(self):
        # 1. Telegram DeepSeek
        tg_captured = []
        async def mock_tg_deepseek(**kwargs):
            tg_captured.append(kwargs)
            return FakeSDKCompletion(
                choices=[FakeSDKChoice(finish_reason="stop", message=FakeSDKMessage(content="Ответ TG DeepSeek"))]
            )

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_tg_deepseek):
            resp = await ai_integration.generate_response(user_id=7001, user_prompt="Тест TG Thinking Disabled")
            self.assertEqual(resp, "Ответ TG DeepSeek")

        self.assertEqual(len(tg_captured), 1)
        tg_payload = tg_captured[0]
        self.assertEqual(tg_payload["max_tokens"], DEEPSEEK_CHAT_MAX_TOKENS)
        self.assertEqual(tg_payload["max_tokens"], 16384)
        self.assertEqual(tg_payload.get("extra_body"), {"thinking": {"type": "disabled"}})

        # Verify Telegram request_payload in AILog
        async with self.sessions() as session:
            tg_ai_log = await session.scalar(
                select(AILog).where(AILog.platform == "telegram").order_by(AILog.id.desc())
            )
            self.assertIsNotNone(tg_ai_log)
            self.assertIsNotNone(tg_ai_log.request_payload)
            tg_parsed = json.loads(tg_ai_log.request_payload)
            self.assertEqual(tg_parsed["provider"], "Deepseek")
            self.assertEqual(tg_parsed["payload"]["max_tokens"], 16384)
            self.assertEqual(tg_parsed["payload"]["extra_body"], {"thinking": {"type": "disabled"}})

        # 2. MAX DeepSeek
        max_captured = []
        async def mock_max_deepseek(**kwargs):
            max_captured.append(kwargs)
            return FakeSDKCompletion(
                choices=[FakeSDKChoice(finish_reason="stop", message=FakeSDKMessage(content="Ответ MAX DeepSeek"))]
            )

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_max_deepseek):
            resp = await max_ai.get_ai_response(7001, "Тест MAX Thinking Disabled")
            self.assertEqual(resp, "Ответ MAX DeepSeek")

        self.assertEqual(len(max_captured), 1)
        max_payload = max_captured[0]
        self.assertEqual(max_payload["max_tokens"], DEEPSEEK_CHAT_MAX_TOKENS)
        self.assertEqual(max_payload["max_tokens"], 16384)
        self.assertEqual(max_payload.get("extra_body"), {"thinking": {"type": "disabled"}})

        # Verify MAX request_payload in AILog
        async with self.sessions() as session:
            max_ai_log = await session.scalar(
                select(AILog).where(AILog.platform == "max").order_by(AILog.id.desc())
            )
            self.assertIsNotNone(max_ai_log)
            self.assertIsNotNone(max_ai_log.request_payload)
            max_parsed = json.loads(max_ai_log.request_payload)
            self.assertEqual(max_parsed["provider"], "Deepseek")
            self.assertEqual(max_parsed["payload"]["max_tokens"], 16384)
            self.assertEqual(max_parsed["payload"]["extra_body"], {"thinking": {"type": "disabled"}})

    async def test_other_providers_isolated_from_deepseek_thinking_parameter(self):
        # Verify OpenAI, Claude, Gemini, KIE do not have extra_body/thinking passed
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "OpenAI"
            await session.commit()

        # 1. Telegram OpenAI
        openai_captured = []
        async def mock_openai(**kwargs):
            openai_captured.append(kwargs)
            return FakeSDKCompletion(
                choices=[FakeSDKChoice(finish_reason="stop", message=FakeSDKMessage(content="OpenAI TG ok"))]
            )

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_openai):
            resp = await ai_integration.generate_response(user_id=7001, user_prompt="Тест OpenAI TG")
            self.assertEqual(resp, "OpenAI TG ok")

        self.assertEqual(len(openai_captured), 1)
        self.assertNotIn("extra_body", openai_captured[0])
        self.assertNotIn("thinking", openai_captured[0])

        # 2. MAX OpenAI
        max_openai_captured = []
        async def mock_max_openai(**kwargs):
            max_openai_captured.append(kwargs)
            return FakeSDKCompletion(
                choices=[FakeSDKChoice(finish_reason="stop", message=FakeSDKMessage(content="OpenAI MAX ok"))]
            )

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_max_openai):
            resp = await max_ai.get_ai_response(7001, "Тест OpenAI MAX")
            self.assertEqual(resp, "OpenAI MAX ok")

        self.assertEqual(len(max_openai_captured), 1)
        self.assertNotIn("extra_body", max_openai_captured[0])
        self.assertNotIn("thinking", max_openai_captured[0])

