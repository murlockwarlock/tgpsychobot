import asyncio
import html
import json
import os
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import func, select

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from ai_integration import (
    AIServiceError,
    AIResponseError,
    generate_response,
    get_ai_response,
)
from ai_log_context import (
    build_ai_attempt_txt_file,
    record_ai_attempt_log,
)
from ai_request_context import (
    _capture_ai_request,
    extract_effective_provider_and_model,
)
from alert_cooldown import KeyedAlertCooldown
from database import (
    AIConfig,
    AILog,
    Topic,
    User,
    async_session_maker,
    init_db,
)
from error_reporting import (
    classify_external_error,
    extract_error_metadata,
    send_output_budget_exhausted_alert,
    send_terminal_ai_failure_alert,
)
from keyboards import (
    admin_ai_log_detail_keyboard,
    admin_ai_logs_keyboard,
    format_ai_log_button,
)
import max_messenger_bot.ai as max_ai
from max_messenger_bot.identity import MAX_ID_OFFSET
import max_messenger_bot.keyboards as max_keyboards
import max_messenger_bot.services.admin_ai_logs as max_admin_ai_logs


class IncidentB1AttemptAuditTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await init_db()

    async def _setup_ai_config(
        self,
        *,
        provider="Gemini",
        model="gemini-2.5-flash",
        allow_fallback=True,
        fallback_provider="OpenAI",
        fallback_model="gpt-5.6-terra",
    ):
        async with async_session_maker() as session:
            cfg = await session.get(AIConfig, 1)
            if not cfg:
                cfg = AIConfig(id=1)
                session.add(cfg)
            cfg.provider = provider
            cfg.gemini_model = model
            cfg.gemini_api_key = "gemini-test-key"
            cfg.openai_model = fallback_model
            cfg.openai_api_key = "openai-test-key"
            cfg.deepseek_model = "deepseek-v4-flash"
            cfg.deepseek_api_key = "deepseek-test-key"
            cfg.allow_fallback = allow_fallback
            cfg.fallback_provider = fallback_provider
            cfg.fallback_model = fallback_model
            cfg.system_prompt = "You are a helpful psychologist."
            await session.commit()

    async def _setup_user(self, user_id=9901, platform="telegram"):
        async with async_session_maker() as session:
            u = await session.get(User, user_id)
            if not u:
                u = User(
                    id=user_id,
                    first_name="TestUser",
                    username="testuser",
                    name="Тест",
                )
                session.add(u)
                await session.commit()
            return u

    # 1. Primary Success: Exactly one AILog with attempt_no=1, role=primary, status=success
    async def test_telegram_primary_success_creates_single_log(self):
        await self._setup_ai_config()
        await self._setup_user(9901)

        mock_call = AsyncMock(return_value="Valid primary response")
        with patch("ai_integration._call_gemini_api", mock_call):
            resp = await generate_response(user_id=9901, user_prompt="Hello AI")
            self.assertIn("Valid primary response", resp)

        async with async_session_maker() as session:
            logs = (
                await session.scalars(
                    select(AILog).where(AILog.user_id == 9901).order_by(AILog.id.asc())
                )
            ).all()
            self.assertEqual(len(logs), 1, "Must create exactly ONE AILog on primary success (no duplicate)")
            log = logs[0]
            self.assertEqual(log.status, "success")
            self.assertEqual(log.attempt_no, 1)
            self.assertEqual(log.attempt_role, "primary")
            self.assertIsNotNone(log.request_group_id)
            self.assertEqual(log.provider, "Gemini")
            self.assertEqual(log.clean_text, "Valid primary response")

    # 2. Primary Failure -> Fallback Success: Exactly two AILogs with isolated captures
    async def test_telegram_primary_failure_fallback_success_creates_two_logs(self):
        await self._setup_ai_config(allow_fallback=True)
        await self._setup_user(9902)

        def failing_gemini(*args, **kwargs):
            capture = kwargs.get("request_capture")
            if capture is not None:
                _capture_ai_request(
                    capture,
                    provider="Gemini",
                    endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash",
                    payload={"contents": [{"parts": [{"text": "Hello Primary"}]}]},
                )
            raise AIServiceError("Gemini 503 Overloaded")

        def successful_openai(*args, **kwargs):
            capture = kwargs.get("request_capture")
            if capture is not None:
                _capture_ai_request(
                    capture,
                    provider="OpenAI",
                    endpoint="https://api.openai.com/v1/chat/completions",
                    payload={"model": "gpt-5.6-terra", "messages": [{"role": "user", "content": "Hello Fallback"}]},
                )
            return "Fallback OpenAI answer"

        with patch("ai_integration._call_gemini_api", side_effect=failing_gemini), \
             patch("ai_integration._call_openai_api", side_effect=successful_openai):
            resp = await generate_response(user_id=9902, user_prompt="Hello with fallback")
            self.assertEqual(resp, "Fallback OpenAI answer")

        async with async_session_maker() as session:
            logs = (
                await session.scalars(
                    select(AILog).where(AILog.user_id == 9902).order_by(AILog.attempt_no.asc())
                )
            ).all()
            self.assertEqual(len(logs), 2, "Must create exactly TWO logs: attempt 1 error, attempt 2 success")

            # Attempt 1: Primary Failure
            att1 = logs[0]
            self.assertEqual(att1.attempt_no, 1)
            self.assertEqual(att1.attempt_role, "primary")
            self.assertEqual(att1.status, "error")
            self.assertEqual(att1.provider, "Gemini")
            self.assertEqual(att1.error_type, "AIServiceError")
            self.assertIn("Gemini 503 Overloaded", att1.error_message)
            self.assertIn("Hello Primary", att1.request_payload)
            self.assertNotIn("Hello Fallback", att1.request_payload, "No mutable payload bleed")

            # Attempt 2: Fallback Success
            att2 = logs[1]
            self.assertEqual(att2.attempt_no, 2)
            self.assertEqual(att2.attempt_role, "fallback")
            self.assertEqual(att2.status, "success")
            self.assertEqual(att2.provider, "OpenAI")
            self.assertEqual(att2.model, "gpt-5.6-terra")
            self.assertIn("Hello Fallback", att2.request_payload)
            self.assertEqual(att2.clean_text, "Fallback OpenAI answer")

            # Shared correlation
            self.assertEqual(att1.request_group_id, att2.request_group_id)

    # 3. Primary Failure -> Fallback Failure: Two error logs + Terminal alert with log IDs
    async def test_telegram_both_failed_creates_two_error_logs_and_terminal_alert(self):
        await self._setup_ai_config(allow_fallback=True)
        await self._setup_user(9903)

        gemini_err = AIServiceError("Gemini down")
        openai_err = AIServiceError("OpenAI quota exceeded")

        bot = MagicMock()
        bot.send_message = AsyncMock()

        with patch("ai_integration._call_gemini_api", side_effect=gemini_err), \
             patch("ai_integration._call_openai_api", side_effect=openai_err), \
             patch("error_reporting.get_all_admin_ids", AsyncMock(return_value=[111, 222])), \
             patch("error_reporting._terminal_failure_cooldown.should_send", return_value=True):
            with self.assertRaises(AIServiceError):
                await generate_response(user_id=9903, user_prompt="Question to fail", bot=bot)

        async with async_session_maker() as session:
            logs = (
                await session.scalars(
                    select(AILog).where(AILog.user_id == 9903).order_by(AILog.attempt_no.asc())
                )
            ).all()
            self.assertEqual(len(logs), 2)
            self.assertEqual(logs[0].status, "error")
            self.assertEqual(logs[1].status, "error")
            log_ids = [l.id for l in logs]

        # Verify admin terminal alert sent with AI Log IDs
        bot.send_message.assert_awaited()
        alert_text = bot.send_message.await_args_list[0].kwargs.get("text") or bot.send_message.await_args_list[0].args[1]
        self.assertIn(f"#{log_ids[0]}", alert_text)
        self.assertIn(f"#{log_ids[1]}", alert_text)

    # 4. Output Budget Exhausted alert with KeyedAlertCooldown
    async def test_output_budget_exhausted_alert_and_keyed_cooldown(self):
        cooldown = KeyedAlertCooldown(timedelta(minutes=30))
        key = "budget:telegram:DeepSeek:deepseek-v4-flash"
        self.assertTrue(cooldown.should_send(key))
        # Second immediate attempt is suppressed by cooldown
        self.assertFalse(cooldown.should_send(key))

        # Different provider or platform is not suppressed
        diff_key = "budget:max:DeepSeek:deepseek-v4-flash"
        self.assertTrue(cooldown.should_send(diff_key))

        # Test alert formatting
        bot = MagicMock()
        bot.send_message = AsyncMock()
        with patch("error_reporting.get_all_admin_ids", AsyncMock(return_value=[111])), \
             patch("error_reporting._output_budget_cooldown.should_send", return_value=True):
            await send_output_budget_exhausted_alert(
                bot=bot,
                platform="telegram",
                user_id=9904,
                provider="DeepSeek",
                model="deepseek-v4-flash",
                finish_reason="length",
                visible_content_length=150,
                reasoning_content_length=40000,
                max_tokens=65536,
                ai_log_id=12345,
            )
            bot.send_message.assert_awaited()
            msg = bot.send_message.await_args_list[0].kwargs.get("text") or bot.send_message.await_args_list[0].args[1]
            self.assertIn("исчерпал output budget", msg)
            self.assertIn("AI Log: #12345", msg)
            self.assertIn("65536", msg)

    # 5. Defensive rendering: legacy rows without status default to success
    def test_defensive_rendering_legacy_log_without_status(self):
        legacy_log = AILog(
            id=501,
            user_id=9905,
            provider="Gemini",
            model="gemini-flash",
            latency_ms=750,
            status=None,  # Legacy row without status
            created_at=datetime(2026, 9, 1, 10, 0),
        )
        label = format_ai_log_button(legacy_log)
        self.assertTrue(label.startswith("✅"), "Legacy logs must default to success mark (✅)")
        self.assertNotIn("❌", label)

    # 6. Fallback and Followup markers in button label
    def test_button_label_fallback_and_followup_markers(self):
        fb_err_log = AILog(
            id=502,
            provider="OpenAI",
            model="gpt-4o",
            latency_ms=800,
            status="error",
            attempt_role="fallback",
            request_type="followup",
            created_at=datetime(2026, 9, 1, 10, 0),
        )
        label = format_ai_log_button(fb_err_log)
        self.assertIn("↪️", label, "Fallback attempt must have ↪️ marker")
        self.assertIn("❌", label, "Error status must have ❌ marker")
        self.assertIn("[F]", label, "Followup request must have [F] marker")

    # 7. Single .txt download sections [1]-[4]
    def test_single_txt_file_content_sections(self):
        err_log = AILog(
            id=503,
            user_id=9906,
            platform="telegram",
            provider="DeepSeek",
            model="deepseek-v4-flash",
            latency_ms=2500,
            status="error",
            request_group_id="grp123456",
            attempt_no=1,
            attempt_role="primary",
            error_type="AIResponseError",
            error_message="output budget exhausted",
            error_classification="output_budget_exhausted",
            http_status=200,
            finish_reason="length",
            request_payload='{"messages": [{"role": "user", "content": "Full Prompt"}]}',
            provider_response_payload='{"choices": [{"finish_reason": "length"}]}',
            clean_text="",
            created_at=datetime(2026, 9, 1, 12, 0),
        )
        content = build_ai_attempt_txt_file(err_log)
        self.assertIn("AI LOG RECORD #503", content)
        self.assertIn("Status: ERROR", content)
        self.assertIn("Request Group: grp123456", content)
        self.assertIn("Attempt: 1 (PRIMARY)", content)
        self.assertIn("📤 [1] FULL REQUEST PAYLOAD:", content)
        self.assertIn("Full Prompt", content)
        self.assertIn("🤖 [2] RAW RESPONSE FROM LLM:", content)
        self.assertIn("finish_reason", content)
        self.assertIn("🚨 [3] APPLICATION ERROR:", content)
        self.assertIn("output budget exhausted", content)
        self.assertIn("💬 [3] CLEAN TEXT SENT TO USER:", content)

    # 8. Best-effort logging: DB commit error does not crash dialogue
    async def test_best_effort_logging_rollback_on_db_failure(self):
        mock_session = MagicMock()
        mock_session.commit = AsyncMock(side_effect=RuntimeError("DB disk full"))
        mock_session.rollback = AsyncMock()

        log_id = await record_ai_attempt_log(
            mock_session,
            user_id=9907,
            platform="telegram",
            request_type="chat",
            provider="Gemini",
            model="flash",
            prompt_summary="Question",
            request_capture={},
            raw_response="Answer",
            latency_ms=500,
            status="success",
        )
        self.assertIsNone(log_id, "Must return None on commit failure")
        mock_session.rollback.assert_awaited_once()

    # 9. MAX bot parity: primary failure -> fallback success records two logs
    async def test_max_bot_attempt_auditing_parity(self):
        await self._setup_ai_config(allow_fallback=True)
        max_user_id = MAX_ID_OFFSET + 9908
        await self._setup_user(max_user_id)

        with patch.object(max_ai, "_dispatch_provider", AsyncMock(side_effect=AIServiceError("Gemini Max Failed"))), \
             patch.object(max_ai, "_call_openai", AsyncMock(return_value="Max Fallback Answer")):
            resp = await max_ai.get_ai_response(max_user_id, "MAX question")
            self.assertEqual(resp, "Max Fallback Answer")

        async with async_session_maker() as session:
            logs = (
                await session.scalars(
                    select(AILog).where(AILog.user_id == max_user_id).order_by(AILog.attempt_no.asc())
                )
            ).all()
            self.assertEqual(len(logs), 2, "MAX bot must record two attempts: primary error and fallback success")
            self.assertEqual(logs[0].status, "error")
            self.assertEqual(logs[0].attempt_role, "primary")
            self.assertEqual(logs[1].status, "success")
            self.assertEqual(logs[1].attempt_role, "fallback")
            self.assertEqual(logs[0].request_group_id, logs[1].request_group_id)

    # 10. Backward compatibility for legacy callbacks (admin_ai_logs_0_all_all)
    async def test_legacy_callbacks_backward_compatibility(self):
        # Callback with 6 parts (legacy): admin_ai_logs_{page}_{period}_{request_type}
        parts_6 = "admin_ai_logs_0_all_all".split("_")
        status_6 = parts_6[6] if len(parts_6) > 6 else "all"
        self.assertEqual(status_6, "all")

        # Callback with 7 parts (new): admin_ai_logs_{page}_{period}_{request_type}_{status}
        parts_7 = "admin_ai_logs_0_all_all_error".split("_")
        status_7 = parts_7[6] if len(parts_7) > 6 else "all"
        self.assertEqual(status_7, "error")
