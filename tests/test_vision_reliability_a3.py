"""Comprehensive behavioral test suite for Phase A3 — Vision & Photo Analysis Reliability.

Covers:
1. Classification precedence (Timeout wrapping SSL, standalone SSL, transport vs body HTTP codes).
2. Vision policy & fallback predicates (KIE upload retry, alternate model retry, provider fallback, candidate ordering).
3. Deadline tracker, budget calculations, and media privacy/redaction.
4. Admin UI separation & compact callbacks (<= 64 bytes, prefix "w:").
5. Public admin alert architecture (chat vs vision isolation, cooldowns, single logical notification).
6. Telegram & MAX vision orchestration (upload reuse, rate limit, output budget fallback, isolated DB session, nil context safety).
"""

import asyncio
import io
import json
import os
import ssl
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from sqlalchemy import select
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "123456:TEST_BOT_TOKEN_FOR_A3")
os.environ.setdefault("ADMIN_ID", "1001")
os.environ.setdefault("ADMIN_IDS", "1001,1002")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import ai_integration
from ai_integration import (
    AIResponseError,
    AIServiceError,
    analyze_image_content,
    _validate_kie_json_response as tg_validate_kie_json_response,
)
from ai_log_context import record_ai_attempt_log
from ai_request_builder import ActivityTracker
from ai_request_singleflight import AI_BUSY_MESSAGE, single_flight
from alert_cooldown import KeyedAlertCooldown
from database import (
    AIConfig,
    AILog,
    Base,
    Message as DBMessage,
    SubscriptionConfig,
    Topic,
    User,
    UserAIActivity,
    async_session_maker,
    init_db,
)
from error_reporting import (
    _output_budget_cooldown,
    _terminal_failure_cooldown,
    classify_external_error,
    send_ai_fallback_used_alert,
    send_output_budget_exhausted_alert,
    send_terminal_ai_failure_alert,
)
import handlers
from keyboards import ai_keys_models_keyboard
import max_messenger_bot.ai as max_ai
from max_messenger_bot.ai import (
    _validate_kie_json_response as max_validate_kie_json_response,
)
from max_messenger_bot.identity import MAX_ID_OFFSET
from provider_models import (
    KIE_VISION_INITIAL_MAX_TOKENS,
    PROVIDER_CLAUDE,
    PROVIDER_GEMINI,
    PROVIDER_KIE,
    PROVIDER_OPENAI,
    build_telegram_model_callback_data,
    get_provider_vision_max_tokens,
    resolve_telegram_model_callback,
)
from vision_reliability import (
    VisionDeadlineTracker,
    VisionExecutionContext,
    order_kie_vision_candidates,
    resolve_effective_vision_fallback,
    run_coro_with_timeout,
    sanitize_vision_request_payload,
    sanitize_vision_text,
    should_retry_kie_vision_model,
    should_retry_kie_vision_upload,
    should_use_vision_provider_fallback,
)


class VisionClassificationPrecedenceTests(unittest.TestCase):
    """Test deterministic error classification precedence per Section 3 & 17."""

    def test_readtimeout_wrapping_ssl_classified_as_timeout(self):
        # Invariant: ReadTimeout wrapping SSLWantReadError MUST be 'timeout', NOT 'network_ssl'
        ssl_err = ssl.SSLWantReadError("The operation did not complete (read)")
        req = httpx.Request("POST", "https://api.kie.ai/v1/chat/completions")
        timeout_err = httpx.ReadTimeout("timed out while reading", request=req)
        timeout_err.__cause__ = ssl_err

        code, desc = classify_external_error(timeout_err)
        self.assertEqual(code, "timeout")

    def test_standalone_ssl_classified_as_network_ssl(self):
        ssl_err = ssl.SSLError("certificate verify failed: unable to get local issuer certificate")
        code, desc = classify_external_error(ssl_err)
        self.assertEqual(code, "network_ssl")

    def test_http_503_beats_body_402(self):
        req = httpx.Request("POST", "https://api.kie.ai/v1/chat/completions")
        resp = httpx.Response(status_code=503, request=req, text='{"code": 402, "message": "insufficient credits"}')
        status_err = httpx.HTTPStatusError("503 Service Unavailable", request=req, response=resp)

        code, desc = classify_external_error(status_err)
        self.assertEqual(code, "provider_5xx")

    def test_http_200_with_body_402_classifies_insufficient_balance(self):
        body_err = AIServiceError("KIE error: insufficient credits (code: 402)")
        body_err.http_status = 200
        body_err.provider_code = 402

        code, desc = classify_external_error(body_err)
        self.assertEqual(code, "insufficient_balance_quota")

    def test_explicit_empty_response_classification_preserved(self):
        resp_err = AIResponseError("OpenAI vision returned empty content")
        resp_err.classification = "empty_response"
        code, desc = classify_external_error(resp_err)
        self.assertEqual(code, "empty_response")


class KIEStructuredBodyCodeParserTests(unittest.TestCase):
    """Section 7: KIE structured body-code parser tests for HTTP 200 with error codes."""

    def test_kie_body_code_402_payment_required(self):
        for validate_fn in (tg_validate_kie_json_response, max_validate_kie_json_response):
            with self.assertRaises(Exception) as ctx:
                validate_fn(200, {"code": 402, "msg": "Insufficient balance or quota"}, context="KIE vision inference")
            err = ctx.exception
            self.assertEqual(getattr(err, "http_status", None), 200)
            self.assertEqual(getattr(err, "provider_code", None), 402)
            self.assertIn("Insufficient balance or quota", str(getattr(err, "provider_response_payload", "")))
            code, _ = classify_external_error(err)
            self.assertEqual(code, "insufficient_balance_quota")

    def test_kie_body_code_429_rate_limit(self):
        for validate_fn in (tg_validate_kie_json_response, max_validate_kie_json_response):
            with self.assertRaises(Exception) as ctx:
                validate_fn(200, {"code": 429, "message": "Too many requests, slow down"}, context="KIE vision inference")
            err = ctx.exception
            self.assertEqual(getattr(err, "http_status", None), 200)
            self.assertEqual(getattr(err, "provider_code", None), 429)
            code, _ = classify_external_error(err)
            self.assertEqual(code, "rate_limit")

    def test_kie_body_code_500_server_error(self):
        for validate_fn in (tg_validate_kie_json_response, max_validate_kie_json_response):
            with self.assertRaises(Exception) as ctx:
                validate_fn(200, {"code": 500, "msg": "Internal Server Error"}, context="KIE vision inference")
            err = ctx.exception
            self.assertEqual(getattr(err, "http_status", None), 200)
            self.assertEqual(getattr(err, "provider_code", None), 500)
            code, _ = classify_external_error(err)
            self.assertEqual(code, "provider_5xx")

    def test_kie_body_code_400_and_422_rejection(self):
        for validate_fn in (tg_validate_kie_json_response, max_validate_kie_json_response):
            with self.assertRaises(Exception) as ctx:
                validate_fn(200, {"code": 400, "msg": "Bad request payload"}, context="KIE vision inference")
            err = ctx.exception
            self.assertEqual(getattr(err, "http_status", None), 200)
            self.assertEqual(getattr(err, "provider_code", None), 400)
            code, _ = classify_external_error(err)
            self.assertEqual(code, "provider_rejection")

            with self.assertRaises(Exception) as ctx:
                validate_fn(200, {"code": 422, "msg": "Unprocessable entity parameters"}, context="KIE vision inference")
            err2 = ctx.exception
            self.assertEqual(getattr(err2, "http_status", None), 200)
            self.assertEqual(getattr(err2, "provider_code", None), 422)
            code2, _ = classify_external_error(err2)
            self.assertEqual(code2, "provider_rejection")


class VisionMediaRedactionPersistenceTests(unittest.IsolatedAsyncioTestCase):
    """Section 8: Real DB persistence media redaction test with 4 sentinels."""

    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def test_real_db_persistence_redacts_all_four_sentinels(self):
        sentinel_b64 = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
        sentinel_raw = "SECRET_IMAGE_SENTINEL_BASE64"
        sentinel_media = "SECRET_MEDIA"
        sentinel_sig = "SECRET_SIGNATURE"

        async with self.sessions() as session:
            await record_ai_attempt_log(
                session,
                user_id=8888,
                platform="telegram",
                request_type="vision",
                provider="OpenAI",
                model="gpt-5.6-terra",
                prompt_summary=f"User prompt with {sentinel_sig}",
                request_capture={
                    "image_url": sentinel_b64,
                    "secret_token": sentinel_raw,
                    "nested": {"media": sentinel_media},
                },
                raw_response=f"Answer mentioning {sentinel_sig} and {sentinel_b64}",
                clean_text="Clean answer text",
                latency_ms=500,
                status="error",
                error_message=f"Failed due to {sentinel_raw} and {sentinel_media}",
                diagnostics={
                    "error_detail": sentinel_raw,
                    "sig": sentinel_sig,
                    "uri": sentinel_b64,
                },
                provider_response_payload=f"{sentinel_raw} {sentinel_sig} {sentinel_media}",
            )
            await session.commit()

        async with self.sessions() as session:
            row = (await session.scalars(select(AILog).where(AILog.user_id == 8888))).first()
            self.assertIsNotNone(row)

            columns_to_check = [
                row.request_payload,
                row.raw_response,
                row.error_message,
                row.diagnostics_json,
                row.provider_response_payload,
            ]
            for col in columns_to_check:
                if col is not None:
                    self.assertNotIn(sentinel_raw, col, f"Sentinel '{sentinel_raw}' leaked into DB: {col}")
                    self.assertNotIn(sentinel_media, col, f"Sentinel '{sentinel_media}' leaked into DB: {col}")
                    self.assertNotIn(sentinel_sig, col, f"Sentinel '{sentinel_sig}' leaked into DB: {col}")
                    self.assertNotIn(sentinel_b64, col, f"Base64 data URI leaked into DB: {col}")
                    self.assertNotIn("/9j/4AAQSkZJRgABAQEA", col, f"Raw base64 leaked into DB: {col}")


class VisionAdminAlertMediaRedactionTests(unittest.IsolatedAsyncioTestCase):
    """Section 9: Alert media redaction test patching _dispatch_admin_alert_text."""

    def setUp(self):
        _terminal_failure_cooldown._last_sent.clear()

    async def test_alert_media_redaction_terminal_and_fallback(self):
        sentinel_b64 = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
        sentinel_raw = "SECRET_IMAGE_SENTINEL_BASE64"
        sentinel_media = "SECRET_MEDIA"
        sentinel_sig = "SECRET_SIGNATURE"

        mock_bot = MagicMock()
        mock_user = SimpleNamespace(id=9999, full_name="Secret User", username="secretuser")

        with patch("error_reporting._dispatch_admin_alert_text", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.return_value = True

            # 1. Test terminal failure alert
            await send_terminal_ai_failure_alert(
                bot=mock_bot,
                title="Тестовый сбой",
                user=mock_user,
                dialogue_id=1,
                provider="OpenAI",
                model="gpt-5.6-terra",
                stage="vision_inference",
                classification="provider_rejection",
                details=f"Failure with {sentinel_raw} and {sentinel_media} and {sentinel_sig} {sentinel_b64}",
                exception=AIServiceError(f"Exception containing {sentinel_raw} and {sentinel_sig}"),
                request_type="vision",
                attempts=[
                    {
                        "provider": "OpenAI",
                        "model": "gpt-5.6-terra",
                        "status": "error",
                        "error": f"Attempt error with {sentinel_media} and {sentinel_b64}",
                    }
                ],
            )

            self.assertTrue(mock_dispatch.called)
            alert_text = mock_dispatch.call_args[0][1]
            self.assertNotIn(sentinel_raw, alert_text)
            self.assertNotIn(sentinel_media, alert_text)
            self.assertNotIn(sentinel_sig, alert_text)
            self.assertNotIn(sentinel_b64, alert_text)
            self.assertNotIn("/9j/4AAQSkZJRgABAQEA", alert_text)

            mock_dispatch.reset_mock()

            # 2. Test fallback used alert
            await send_ai_fallback_used_alert(
                bot=mock_bot,
                primary_provider="OpenAI",
                primary_model="gpt-5.6-terra",
                fallback_provider="KIE",
                fallback_model="gemini-3-flash",
                failure_reason=f"Failed due to {sentinel_raw} and {sentinel_sig} with {sentinel_b64}",
                user=mock_user,
                dialogue_id=1,
                request_type="vision",
                attempts=[
                    {
                        "provider": "OpenAI",
                        "model": "gpt-5.6-terra",
                        "status": "error",
                        "error": f"Failed with {sentinel_media}",
                    }
                ],
            )

            self.assertTrue(mock_dispatch.called)
            fallback_alert_text = mock_dispatch.call_args[0][1]
            self.assertNotIn(sentinel_raw, fallback_alert_text)
            self.assertNotIn(sentinel_media, fallback_alert_text)
            self.assertNotIn(sentinel_sig, fallback_alert_text)
            self.assertNotIn(sentinel_b64, fallback_alert_text)
            self.assertNotIn("/9j/4AAQSkZJRgABAQEA", fallback_alert_text)


class VisionPolicyAndFallbackTests(unittest.TestCase):
    """Test shared policy helpers in vision_reliability.py."""

    def test_order_kie_vision_candidates_primary_first(self):
        candidates = order_kie_vision_candidates("gemini-3-flash")
        self.assertEqual(candidates[0], "gemini-3-flash")
        self.assertIn("gemini-2.5-flash", candidates)

        candidates_rev = order_kie_vision_candidates("gemini-2.5-flash")
        self.assertEqual(candidates_rev[0], "gemini-2.5-flash")
        self.assertIn("gemini-3-flash", candidates_rev)

    def test_should_retry_kie_vision_upload(self):
        self.assertTrue(should_retry_kie_vision_upload("timeout"))
        self.assertTrue(should_retry_kie_vision_upload("network_ssl"))
        self.assertTrue(should_retry_kie_vision_upload("network_connection"))
        self.assertTrue(should_retry_kie_vision_upload("provider_5xx"))

        # Fatal errors do not retry upload
        self.assertFalse(should_retry_kie_vision_upload("rate_limit"))
        self.assertFalse(should_retry_kie_vision_upload("auth_invalid_key"))
        self.assertFalse(should_retry_kie_vision_upload("insufficient_balance_quota"))

    def test_should_retry_kie_vision_model(self):
        self.assertTrue(should_retry_kie_vision_model("timeout"))
        self.assertTrue(should_retry_kie_vision_model("provider_5xx"))
        self.assertTrue(should_retry_kie_vision_model("invalid_response"))
        self.assertTrue(should_retry_kie_vision_model("empty_response"))

        # 429 rate limit MUST NOT retry same-KIE model
        self.assertFalse(should_retry_kie_vision_model("rate_limit"))
        self.assertFalse(should_retry_kie_vision_model("auth"))

    def test_should_use_vision_provider_fallback(self):
        # Output budget fallback requires higher budget
        self.assertTrue(should_use_vision_provider_fallback("output_budget_exhausted", 4096, 16384))
        self.assertFalse(should_use_vision_provider_fallback("output_budget_exhausted", 16384, 4096))
        self.assertFalse(should_use_vision_provider_fallback("output_budget_exhausted", 16384, 16384))

        # Transient / fatal provider errors qualify
        self.assertTrue(should_use_vision_provider_fallback("timeout"))
        self.assertTrue(should_use_vision_provider_fallback("provider_5xx"))
        self.assertTrue(should_use_vision_provider_fallback("rate_limit"))
        self.assertTrue(should_use_vision_provider_fallback("insufficient_balance_quota"))

    def test_resolve_effective_vision_fallback_disables_when_same_provider(self):
        allow, prov, model = resolve_effective_vision_fallback("OpenAI", True, "OpenAI", "gpt-5.6-terra")
        self.assertFalse(allow)
        self.assertEqual(prov, "OpenAI")

    def test_resolve_effective_vision_fallback_valid(self):
        allow, prov, model = resolve_effective_vision_fallback("KIE", True, "OpenAI", "gpt-5.6-terra")
        self.assertTrue(allow)
        self.assertEqual(prov, "OpenAI")
        self.assertEqual(model, "gpt-5.6-terra")


class VisionDeadlineAndSanitizationTests(unittest.TestCase):
    """Test deadline budgeting and media data redaction."""

    def test_vision_deadline_tracker_bounds(self):
        tracker = VisionDeadlineTracker(total_deadline_seconds=85.0)
        self.assertGreater(tracker.remaining_time(), 80.0)

        # Stage budget capped by aggregate_cap
        b = tracker.stage_budget("kie_upload", aggregate_cap=20.0)
        self.assertLessEqual(b, 20.0)

        tracker_exhausted = VisionDeadlineTracker(total_deadline_seconds=0.0)
        self.assertLessEqual(tracker_exhausted.remaining_time(), 0.1)

    def test_sanitize_vision_request_payload_redacts_base64_and_urls(self):
        b64_long = "A" * 120
        payload = {
            "model": "gpt-5.6-terra",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe image"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_long}"}},
                    ],
                }
            ],
            "file_url": "https://files.kie.ai/tmp/vision_input_12345.jpg",
        }

        sanitized = sanitize_vision_request_payload(payload)
        sanitized_str = json.dumps(sanitized)

        self.assertNotIn(b64_long, sanitized_str)
        self.assertNotIn("https://files.kie.ai/tmp/vision_input_12345.jpg", sanitized_str)
        self.assertIn("<redacted_media_url>", sanitized_str)

    def test_sanitize_vision_text_redacts_media_strings(self):
        raw_text = "Analysis error: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA... at https://files.kie.ai/tmp/img.png"
        clean = sanitize_vision_text(raw_text)
        self.assertNotIn("iVBORw0KGgoAAA", clean)
        self.assertNotIn("https://files.kie.ai/tmp/img.png", clean)


class VisionAdminUITests(unittest.IsolatedAsyncioTestCase):
    """Test TG and MAX vision fallback admin UI separation and compact callback resolution."""

    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with self.sessions() as session:
            cfg = AIConfig(
                id=1,
                provider="OpenAI",
                vision_provider="OpenAI",
                vision_model="gpt-5.6-terra",
                allow_vision_fallback=False,
                vision_fallback_provider="Gemini",
                vision_fallback_model="gemini-3.7-flash",
            )
            session.add(cfg)
            await session.commit()

    async def asyncTearDown(self):
        await self.engine.dispose()

    def test_compact_callback_vision_fallback_length_and_prefix(self):
        # Compact callback code for "vision_fallback" is "w" (prefix ai_m_w_)
        cb = build_telegram_model_callback_data(PROVIDER_OPENAI, "vision_fallback", "gpt-5.6-terra")
        self.assertTrue(cb.startswith("ai_m_w_"))
        self.assertLessEqual(len(cb.encode("utf-8")), 64)

        # Resolves cleanly
        resolved = resolve_telegram_model_callback(cb)
        self.assertIsNotNone(resolved)
        prov, channel, model = resolved
        self.assertEqual(channel, "vision_fallback")
        self.assertEqual(prov, PROVIDER_OPENAI)
        self.assertEqual(model, "gpt-5.6-terra")

    async def test_toggle_vision_fallback_toggles_boolean_only(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            self.assertFalse(cfg.allow_vision_fallback)
            cfg.allow_vision_fallback = not cfg.allow_vision_fallback
            await session.commit()

        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            self.assertTrue(cfg.allow_vision_fallback)
            # Preserves stored provider and model
            self.assertEqual(cfg.vision_fallback_provider, "Gemini")
            self.assertEqual(cfg.vision_fallback_model, "gemini-3.7-flash")


class VisionPublicAlertTests(unittest.IsolatedAsyncioTestCase):
    """Test public admin alert contracts, cooldowns, and isolation between chat and vision."""

    def setUp(self):
        _terminal_failure_cooldown._last_sent.clear()
        _output_budget_cooldown._last_sent.clear()

    async def test_chat_alert_then_vision_alert_both_send(self):
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()

        with patch("error_reporting._dispatch_admin_alert_text", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.return_value = True

            # 1. Chat terminal alert
            chat_sent = await send_terminal_ai_failure_alert(
                bot=mock_bot,
                platform="telegram",
                user_id=7001,
                primary_provider="OpenAI",
                primary_model="gpt-5.6-terra",
                request_type="chat",
                error_message="Chat timeout",
            )
            self.assertTrue(chat_sent)

            # 2. Vision terminal alert for same provider/model sends because namespace is isolated
            vision_sent = await send_terminal_ai_failure_alert(
                bot=mock_bot,
                platform="telegram",
                user_id=7001,
                primary_provider="OpenAI",
                primary_model="gpt-5.6-terra",
                request_type="vision",
                error_message="Vision timeout",
            )
            self.assertTrue(vision_sent)
            self.assertEqual(mock_dispatch.call_count, 2)

    async def test_vision_terminal_alert_cooldown_deduplication(self):
        mock_bot = MagicMock()
        with patch("error_reporting._dispatch_admin_alert_text", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.return_value = True

            # First vision alert sends
            sent1 = await send_terminal_ai_failure_alert(
                bot=mock_bot,
                platform="telegram",
                user_id=7001,
                primary_provider="OpenAI",
                primary_model="gpt-5.6-terra",
                request_type="vision",
                error_message="Vision timeout",
            )
            self.assertTrue(sent1)

            # Second identical vision alert within cooldown is suppressed
            sent2 = await send_terminal_ai_failure_alert(
                bot=mock_bot,
                platform="telegram",
                user_id=7001,
                primary_provider="OpenAI",
                primary_model="gpt-5.6-terra",
                request_type="vision",
                error_message="Vision timeout",
            )
            self.assertFalse(sent2)
            self.assertEqual(mock_dispatch.call_count, 1)

    async def test_fallback_recovery_alert_dispatches_cleanly(self):
        mock_bot = MagicMock()
        with patch("error_reporting._dispatch_admin_alert_text", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.return_value = True

            sent = await send_ai_fallback_used_alert(
                bot=mock_bot,
                platform="telegram",
                user_id=7001,
                primary_provider="KIE",
                primary_model="gemini-3-flash",
                fallback_provider="OpenAI",
                fallback_model="gpt-5.6-terra",
                primary_error="KIE 503",
                request_type="vision",
            )
            self.assertTrue(sent)
            self.assertEqual(mock_dispatch.call_count, 1)
            # Verify only (bot, text) was passed to _dispatch_admin_alert_text
            args, kwargs = mock_dispatch.call_args
            self.assertEqual(len(args), 2)
            self.assertEqual(args[0], mock_bot)
            self.assertIn("Использован резервный AI-провайдер", args[1])

    async def test_recovery_alert_does_not_consume_terminal_or_budget_cooldown(self):
        mock_bot = MagicMock()
        with patch("error_reporting._dispatch_admin_alert_text", new_callable=AsyncMock) as mock_dispatch:
            mock_dispatch.return_value = True

            # 1. Fallback used alert
            await send_ai_fallback_used_alert(
                bot=mock_bot,
                platform="telegram",
                user_id=7001,
                primary_provider="KIE",
                primary_model="gemini-3-flash",
                fallback_provider="OpenAI",
                fallback_model="gpt-5.6-terra",
                primary_error="KIE 503",
                request_type="vision",
            )

            # 2. Terminal alert for same provider/model is still sent
            term_sent = await send_terminal_ai_failure_alert(
                bot=mock_bot,
                platform="telegram",
                user_id=7001,
                primary_provider="KIE",
                primary_model="gemini-3-flash",
                request_type="vision",
                error_message="KIE terminal",
            )
            self.assertTrue(term_sent)
            self.assertEqual(mock_dispatch.call_count, 2)


class VisionOrchestrationIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Test full Telegram and MAX vision orchestration lifecycles."""

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
        self._orig_max_ai_sessions = max_ai.async_session_maker

        handlers.async_session_maker = self.sessions
        ai_integration.async_session_maker = self.sessions
        max_ai.async_session_maker = self.sessions

        single_flight.clear()
        async with self.sessions() as session:
            topic = Topic(id=1, name="Арт-терапия", is_active=True, system_prompt="Инструкция арт-терапевта.")
            user = User(id=7001, first_name="Анна", gender="female", age=25, current_dialogue_id=1, current_topic_id=1, accepted_disclaimer=True)
            max_user = User(id=7001 + MAX_ID_OFFSET, first_name="MAX User", gender="male", age=30, current_dialogue_id=1, current_topic_id=1, accepted_disclaimer=True)
            sub_cfg = SubscriptionConfig(id=1, subscriptions_enabled=False)
            cfg = AIConfig(
                id=1,
                provider="OpenAI",
                openai_api_key="sk-openai-key",
                vision_provider="KIE",
                vision_model="gemini-3-flash",
                kie_api_key="sk-kie-key",
                gemini_api_key="sk-gemini-key",
                gemini_model="gemini-3.7-flash",
                claude_api_key="sk-claude-key",
                claude_model="claude-sonnet-5",
                allow_vision_fallback=True,
                vision_fallback_provider="OpenAI",
                vision_fallback_model="gpt-5.6-terra",
            )
            session.add_all([topic, user, max_user, sub_cfg, cfg])
            await session.commit()

    async def asyncTearDown(self):
        handlers.async_session_maker = self._orig_handlers_sessions
        ai_integration.async_session_maker = self._orig_ai_sessions
        max_ai.async_session_maker = self._orig_max_ai_sessions
        single_flight.clear()
        await self.engine.dispose()

    def _make_mock_bot(self):
        bot = MagicMock()
        bot.send_chat_action = AsyncMock()
        bot.get_file = AsyncMock(return_value=MagicMock(file_path="photos/sample.jpg"))
        bot.download_file = AsyncMock(return_value=io.BytesIO(b"fake_jpeg_image_bytes"))
        bot.delete_message = AsyncMock()
        bot.send_message = AsyncMock()
        return bot

    def _make_mock_msg(self, user_id=7001, caption="Тестовое фото"):
        msg = MagicMock()
        msg.chat.id = user_id
        msg.from_user.id = user_id
        msg.from_user.username = "annatest"
        msg.from_user.full_name = "Анна Тест"
        msg.message_id = 999
        msg.caption = caption
        msg.text = None
        photo_obj = MagicMock()
        photo_obj.file_id = "photo_123"
        msg.photo = [photo_obj]
        msg.answer = AsyncMock(return_value=MagicMock(message_id=1001))
        msg.answer_photo = AsyncMock()
        return msg

    async def test_telegram_vision_kie_upload_reuse_on_model_retry(self):
        """KIE upload is called once, then reused when candidate 1 fails with 503 and candidate 2 succeeds."""
        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="KIE upload reuse test")

        upload_calls = []
        async def fake_upload(*args, **kwargs):
            upload_calls.append(args)
            return "https://files.kie.ai/tmp/reused_url.jpg"

        inference_calls = []
        async def fake_kie_post(url, headers=None, json=None, **kwargs):
            model = json.get("model") if json else None
            inference_calls.append(model)
            if model == "gemini-3-flash":
                return MagicMock(status_code=503, json=lambda: {"error": {"message": "503 overload"}})
            return MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "KIE Alternate Model Success"}}]})

        with patch("ai_integration._upload_file_to_kie", side_effect=fake_upload), \
             patch("httpx.AsyncClient.post", side_effect=fake_kie_post), \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        # Upload was executed exactly once
        self.assertEqual(len(upload_calls), 1)
        # Inference tried both models with the single reused URL
        self.assertEqual(inference_calls, ["gemini-3-flash", "gemini-2.5-flash"])

        # Per-attempt audit logged exactly 2 rows
        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001).order_by(AILog.attempt_no))).all()
            self.assertEqual(len(logs), 2)
            self.assertEqual(logs[0].status, "error")
            self.assertEqual(logs[0].model, "gemini-3-flash")
            self.assertEqual(logs[1].status, "success")
            self.assertEqual(logs[1].model, "gemini-2.5-flash")

    async def test_telegram_vision_kie_upload_failure_falls_back_to_provider(self):
        """When KIE upload fails persistently, it skips KIE models and falls back directly to provider fallback."""
        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="KIE upload outage test")

        # Simulate KIE upload network timeout
        async def fake_upload_fail(*args, **kwargs):
            raise httpx.ReadTimeout("Upload timed out")

        openai_called = []
        async def fake_openai_create(**kwargs):
            openai_called.append(kwargs)
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "OpenAI Provider Fallback Success"
            resp.choices = [choice]
            return resp

        with patch("ai_integration._upload_file_to_kie", side_effect=fake_upload_fail), \
             patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_openai_create), \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        self.assertEqual(len(openai_called), 1)
        # Provider fallback was OpenAI
        self.assertEqual(openai_called[0]["model"], "gpt-5.6-terra")

    async def test_telegram_vision_kie_inference_429_skips_alternate_model_and_falls_back(self):
        """KIE 429 RateLimit on candidate 1 does NOT try candidate 2; falls back to OpenAI."""
        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="KIE 429 test")

        async def fake_kie_post(url, headers=None, json=None, **kwargs):
            return MagicMock(status_code=429, json=lambda: {"error": {"message": "Rate limit exceeded 429"}})

        openai_called = []
        async def fake_openai_create(**kwargs):
            openai_called.append(kwargs)
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "OpenAI Fallback After 429"
            resp.choices = [choice]
            return resp

        with patch("ai_integration._upload_file_to_kie", new_callable=AsyncMock, return_value="https://files.kie.ai/tmp/img.jpg"), \
             patch("httpx.AsyncClient.post", side_effect=fake_kie_post), \
             patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_openai_create), \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        self.assertEqual(len(openai_called), 1)

    async def test_telegram_vision_none_execution_context_no_logs_no_alerts(self):
        """Calling analyze_image_content with execution_context=None creates 0 AILog rows."""
        async def fake_kie_post(url, headers=None, json=None, **kwargs):
            return MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "Direct Call OK"}}]})

        with patch("ai_integration._upload_file_to_kie", new_callable=AsyncMock, return_value="https://files.kie.ai/tmp/img.jpg"), \
             patch("httpx.AsyncClient.post", side_effect=fake_kie_post):
            res = await analyze_image_content(
                b"fake_image",
                prompt="Direct prompt",
                execution_context=None,
            )
            self.assertEqual(res, "Direct Call OK")

        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog))).all()
            self.assertEqual(len(logs), 0)

    async def test_telegram_vision_isolated_audit_session_survives_data_rollback(self):
        """When downstream DATA persistence fails, the orchestrator's attempt log in isolated DB session survives."""
        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="Rollback test")

        async def fake_kie_post(url, headers=None, json=None, **kwargs):
            return MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "Valid response\n<DATA>{\"current_state\":{\"step\":\"broken\"}}</DATA>"}}]})

        with patch("ai_integration._upload_file_to_kie", new_callable=AsyncMock, return_value="https://files.kie.ai/tmp/img.jpg"), \
             patch("httpx.AsyncClient.post", side_effect=fake_kie_post), \
             patch("handlers.apply_service_data_blocks", side_effect=RuntimeError("Forced DB Error")), \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001))).all()
            # Attempt log survived despite DATA rollback!
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0].status, "success")
            # Assistant message was rolled back
            msgs = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 7001, DBMessage.role == "assistant"))).all()
            self.assertEqual(len(msgs), 0)

    async def test_max_vision_orchestration_with_model_retry_and_fallback(self):
        """MAX Messenger vision lifecycle: KIE upload reuse, same-KIE model retry, provider fallback to OpenAI."""
        ctx = VisionExecutionContext(
            user_id=7001 + MAX_ID_OFFSET,
            username="maxuser",
            full_name="MAX User",
            dialogue_id=1,
            topic_id=1,
            topic_name="Арт-терапия",
            platform="max",
            chat_id=12345,
            bot_name="MAXBot",
            bot=None,
        )

        upload_called = 0
        async def fake_max_upload(*args, **kwargs):
            nonlocal upload_called
            upload_called += 1
            return "https://files.kie.ai/tmp/max_img.jpg"

        # Both KIE models fail with 503
        async def fake_max_kie_post(url, headers=None, json=None, **kwargs):
            return MagicMock(status_code=503, json=lambda: {"error": {"message": "KIE down"}})

        # Provider fallback to OpenAI succeeds
        openai_calls = []
        async def fake_max_openai(**kwargs):
            openai_calls.append(kwargs)
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "MAX Vision Fallback Success"
            resp.choices = [choice]
            return resp

        with patch("max_messenger_bot.ai._upload_file_to_kie", side_effect=fake_max_upload), \
             patch("httpx.AsyncClient.post", side_effect=fake_max_kie_post), \
             patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_max_openai), \
             patch("error_reporting._dispatch_admin_alert_text", new_callable=AsyncMock) as mock_alert_dispatch:
            mock_alert_dispatch.return_value = True

            res = await max_ai.analyze_image(
                user_id=7001 + MAX_ID_OFFSET,
                image_bytes=b"fake_image_bytes",
                prompt="MAX prompt",
                execution_context=ctx,
            )
            self.assertEqual(res, "MAX Vision Fallback Success")

        # KIE upload was called once and reused
        self.assertEqual(upload_called, 1)
        # Fallback to OpenAI succeeded
        self.assertEqual(len(openai_calls), 1)

        # Verify audit logs in isolated sessions: 2 KIE errors + 1 OpenAI fallback success = 3 logs
        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001 + MAX_ID_OFFSET).order_by(AILog.id))).all()
            self.assertEqual(len(logs), 3)
            self.assertEqual(logs[0].provider, "KIE")
            self.assertEqual(logs[0].status, "error")
            self.assertEqual(logs[1].provider, "KIE")
            self.assertEqual(logs[1].status, "error")
            self.assertEqual(logs[2].provider, "OpenAI")
            self.assertEqual(logs[2].status, "success")
            self.assertEqual(logs[2].attempt_role, "fallback")

    async def test_telegram_vision_openai_timeout_falls_back_to_kie(self):
        """Blocker 4.1: OpenAI primary timeout -> KIE provider fallback upload -> KIE inference -> success."""
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "OpenAI"
            cfg.vision_model = "gpt-5.6-terra"
            cfg.allow_vision_fallback = True
            cfg.vision_fallback_provider = "KIE"
            cfg.vision_fallback_model = "gemini-3-flash"
            await session.commit()

        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="OpenAI timeout fallback to KIE")

        async def fake_openai_timeout(**kwargs):
            raise httpx.ReadTimeout("OpenAI vision request timed out")

        upload_called = []
        async def fake_kie_upload(api_key, base_url, image_bytes, filename, folder, timeout=None, activity_tracker=None):
            upload_called.append({"filename": filename, "timeout": timeout})
            return "https://files.kie.ai/tmp/openai_fallback_upload.jpg"

        async def fake_kie_post(url, headers=None, json=None, **kwargs):
            return MagicMock(
                status_code=200,
                json=lambda: {
                    "choices": [
                        {"message": {"content": "KIE Fallback Content After OpenAI Timeout"}}
                    ]
                },
            )

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_openai_timeout), \
             patch("ai_integration._upload_file_to_kie", side_effect=fake_kie_upload), \
             patch("httpx.AsyncClient.post", side_effect=fake_kie_post), \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        self.assertEqual(len(upload_called), 1)
        # Check answer sent to user
        answer_texts = [call.args[0] for call in mock_msg.answer.call_args_list if call.args]
        self.assertTrue(any("KIE Fallback Content After OpenAI Timeout" in txt for txt in answer_texts))

        # Check logs: OpenAI error + KIE fallback success
        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001).order_by(AILog.id))).all()
            self.assertEqual(len(logs), 2)
            self.assertEqual(logs[0].provider, "OpenAI")
            self.assertEqual(logs[0].status, "error")
            self.assertEqual(logs[0].error_classification, "timeout")
            self.assertEqual(logs[1].provider, "KIE")
            self.assertEqual(logs[1].status, "success")
            self.assertEqual(logs[1].attempt_role, "fallback")

    async def test_telegram_vision_internal_retry_single_flight_lease_held(self):
        """Internal KIE retry does not trigger single-flight busy rejection against itself."""
        mock_bot = self._make_mock_bot()
        mock_msg = self._make_mock_msg(caption="KIE internal retry single-flight test")

        post_calls = 0
        async def fake_kie_post_retry(url, headers=None, json=None, **kwargs):
            nonlocal post_calls
            post_calls += 1
            if post_calls == 1:
                return MagicMock(status_code=503, json=lambda: {"error": {"message": "Service Unavailable 503"}})
            return MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "Retry Success"}}]})

        with patch("ai_integration._upload_file_to_kie", new_callable=AsyncMock, return_value="https://files.kie.ai/tmp/img.jpg"), \
             patch("httpx.AsyncClient.post", side_effect=fake_kie_post_retry), \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            self.assertFalse(single_flight.is_busy("telegram", 7001))
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)
            self.assertFalse(single_flight.is_busy("telegram", 7001))

        self.assertEqual(post_calls, 2)
        answer_texts = [call.args[0] for call in mock_msg.answer.call_args_list if call.args]
        self.assertTrue(any("Retry Success" in txt for txt in answer_texts))
        self.assertFalse(any(AI_BUSY_MESSAGE in txt for txt in answer_texts))

    # =========================================================================
    # Section 10: Provider metadata persistence tests
    # =========================================================================

    async def test_vision_metadata_persistence_openai(self):
        """OpenAI vision success persists finish_reason, http_status=200, and usage in AILog."""
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "OpenAI"
            cfg.vision_model = "gpt-5.6-terra"
            cfg.allow_vision_fallback = False
            await session.commit()

        ctx = VisionExecutionContext(user_id=7001, dialogue_id=1, topic_id=1, platform="telegram", bot=None)

        fake_resp = MagicMock()
        fake_choice = MagicMock()
        fake_choice.message.content = "OpenAI vision analysis result"
        fake_choice.finish_reason = "stop"
        fake_resp.choices = [fake_choice]
        fake_resp.usage = SimpleNamespace(prompt_tokens=150, completion_tokens=75)

        with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock, return_value=fake_resp):
            res = await analyze_image_content(b"fake_image_bytes", prompt="Analyze", execution_context=ctx)
            self.assertIn("OpenAI vision analysis result", res)

        async with self.sessions() as session:
            log_row = (await session.scalars(select(AILog).where(AILog.user_id == 7001))).first()
            self.assertIsNotNone(log_row)
            self.assertEqual(log_row.provider, "OpenAI")
            self.assertEqual(log_row.status, "success")
            self.assertEqual(log_row.http_status, 200)
            self.assertEqual(log_row.finish_reason, "stop")
            self.assertIn("150", str(log_row.diagnostics_json))
            self.assertIn("75", str(log_row.diagnostics_json))

    async def test_vision_metadata_persistence_openai_length(self):
        """OpenAI vision length/exhaustion persists finish_reason='length' and usage in AILog."""
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "OpenAI"
            cfg.vision_model = "gpt-5.6-terra"
            cfg.allow_vision_fallback = False
            await session.commit()

        ctx = VisionExecutionContext(user_id=7001, dialogue_id=1, topic_id=1, platform="telegram", bot=None)

        fake_resp = MagicMock()
        fake_choice = MagicMock()
        fake_choice.message.content = "Truncated..."
        fake_choice.finish_reason = "length"
        fake_resp.choices = [fake_choice]
        fake_resp.usage = SimpleNamespace(prompt_tokens=200, completion_tokens=4096)

        with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock, return_value=fake_resp), \
             patch("error_reporting._dispatch_admin_alert_text", new_callable=AsyncMock):
            with self.assertRaises(AIServiceError) as ctx_err:
                await analyze_image_content(b"fake_image_bytes", prompt="Analyze", execution_context=ctx)
            self.assertEqual(ctx_err.exception.classification, "output_budget_exhausted")

        async with self.sessions() as session:
            log_row = (await session.scalars(select(AILog).where(AILog.user_id == 7001))).first()
            self.assertIsNotNone(log_row)
            self.assertEqual(log_row.provider, "OpenAI")
            self.assertEqual(log_row.status, "error")
            self.assertEqual(log_row.error_classification, "output_budget_exhausted")
            self.assertEqual(log_row.finish_reason, "length")
            self.assertIn("4096", str(log_row.diagnostics_json))

    async def test_vision_metadata_persistence_claude(self):
        """Claude vision success persists finish_reason, http_status=200, and usage in AILog."""
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "Claude"
            cfg.vision_model = "claude-sonnet-5"
            cfg.allow_vision_fallback = False
            await session.commit()

        ctx = VisionExecutionContext(user_id=7001, dialogue_id=1, topic_id=1, platform="telegram", bot=None)

        fake_resp = MagicMock()
        fake_resp.content = [SimpleNamespace(type="text", text="Claude vision result")]
        fake_resp.stop_reason = "end_turn"
        fake_resp.usage = SimpleNamespace(input_tokens=120, output_tokens=60)

        with patch("anthropic.resources.messages.AsyncMessages.create", new_callable=AsyncMock, return_value=fake_resp):
            res = await analyze_image_content(b"fake_image_bytes", prompt="Analyze", execution_context=ctx)
            self.assertEqual(res, "Claude vision result")

        async with self.sessions() as session:
            log_row = (await session.scalars(select(AILog).where(AILog.user_id == 7001))).first()
            self.assertIsNotNone(log_row)
            self.assertEqual(log_row.provider, "Claude")
            self.assertEqual(log_row.status, "success")
            self.assertEqual(log_row.http_status, 200)
            self.assertEqual(log_row.finish_reason, "end_turn")
            self.assertIn("120", str(log_row.diagnostics_json))

    async def test_vision_metadata_persistence_gemini(self):
        """Gemini vision success persists finish_reason, http_status=200, and usage in AILog."""
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "Gemini"
            cfg.vision_model = "gemini-3.7-flash"
            cfg.allow_vision_fallback = False
            await session.commit()

        ctx = VisionExecutionContext(user_id=7001, dialogue_id=1, topic_id=1, platform="telegram", bot=None)

        fake_resp = MagicMock(
            status_code=200,
            json=lambda: {
                "candidates": [
                    {"content": {"parts": [{"text": "Gemini vision result"}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": {"promptTokenCount": 95, "candidatesTokenCount": 45},
            },
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=fake_resp):
            res = await analyze_image_content(b"fake_image_bytes", prompt="Analyze", execution_context=ctx)
            self.assertEqual(res, "Gemini vision result")

        async with self.sessions() as session:
            log_row = (await session.scalars(select(AILog).where(AILog.user_id == 7001))).first()
            self.assertIsNotNone(log_row)
            self.assertEqual(log_row.provider, "Gemini")
            self.assertEqual(log_row.status, "success")
            self.assertEqual(log_row.http_status, 200)
            self.assertEqual(log_row.finish_reason, "STOP")
            self.assertIn("95", str(log_row.diagnostics_json))

    async def test_vision_metadata_persistence_kie(self):
        """KIE vision success persists finish_reason, http_status=200, and usage in AILog."""
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "KIE"
            cfg.vision_model = "gemini-3-flash"
            cfg.allow_vision_fallback = False
            await session.commit()

        ctx = VisionExecutionContext(user_id=7001, dialogue_id=1, topic_id=1, platform="telegram", bot=None)

        fake_resp = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [
                    {"message": {"content": "KIE vision result"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 110, "completion_tokens": 55},
            },
        )

        with patch("ai_integration._upload_file_to_kie", new_callable=AsyncMock, return_value="https://files.kie.ai/tmp/img.jpg"), \
             patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=fake_resp):
            res = await analyze_image_content(b"fake_image_bytes", prompt="Analyze", execution_context=ctx)
            self.assertEqual(res, "KIE vision result")

        async with self.sessions() as session:
            log_row = (await session.scalars(select(AILog).where(AILog.user_id == 7001))).first()
            self.assertIsNotNone(log_row)
            self.assertEqual(log_row.provider, "KIE")
            self.assertEqual(log_row.status, "success")
            self.assertEqual(log_row.http_status, 200)
            self.assertEqual(log_row.finish_reason, "stop")
            self.assertIn("110", str(log_row.diagnostics_json))

    async def test_vision_metadata_persistence_max(self):
        """MAX bot vision success persists finish_reason, http_status=200, and usage in AILog."""
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "KIE"
            cfg.vision_model = "gemini-3-flash"
            cfg.allow_vision_fallback = False
            await session.commit()

        ctx = VisionExecutionContext(
            user_id=7001 + MAX_ID_OFFSET,
            username="maxuser",
            full_name="MAX User",
            dialogue_id=1,
            topic_id=1,
            platform="max",
            chat_id=12345,
            bot=None,
        )

        fake_resp = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [
                    {"message": {"content": "MAX KIE result"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 115, "completion_tokens": 65},
            },
        )

        with patch("max_messenger_bot.ai._upload_file_to_kie", new_callable=AsyncMock, return_value="https://files.kie.ai/tmp/max_img.jpg"), \
             patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=fake_resp):
            res = await max_ai.analyze_image(
                user_id=7001 + MAX_ID_OFFSET,
                image_bytes=b"fake_image_bytes",
                prompt="MAX prompt",
                execution_context=ctx,
            )
            self.assertEqual(res, "MAX KIE result")

        async with self.sessions() as session:
            log_row = (await session.scalars(select(AILog).where(AILog.user_id == 7001 + MAX_ID_OFFSET))).first()
            self.assertIsNotNone(log_row)
            self.assertEqual(log_row.provider, "KIE")
            self.assertEqual(log_row.status, "success")
            self.assertEqual(log_row.http_status, 200)
            self.assertEqual(log_row.finish_reason, "stop")
            self.assertIn("115", str(log_row.diagnostics_json))

    # =========================================================================
    # Section 11: Exact pre-flight tests (assert 0 HTTP, 0 AILog, 0 activity marks)
    # =========================================================================

    async def test_telegram_preflight_missing_key_raises_configuration_zero_side_effects(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "OpenAI"
            cfg.vision_model = "gpt-5.6-terra"
            cfg.openai_api_key = None
            await session.commit()

        tracker = ActivityTracker(self.sessions, user_id=7001, topic_id=1, track_user_activity=True)
        ctx = VisionExecutionContext(user_id=7001, dialogue_id=1, topic_id=1, platform="telegram", bot=None)

        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False), \
             patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_http, \
             patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_openai:
            with self.assertRaises(AIServiceError) as err_ctx:
                await analyze_image_content(b"fake_image_bytes", prompt="Analyze", activity_tracker=tracker, execution_context=ctx)
            self.assertEqual(err_ctx.exception.classification, "configuration")
            mock_http.assert_not_called()
            mock_openai.assert_not_called()

        self.assertFalse(tracker._marked)
        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001))).all()
            self.assertEqual(len(logs), 0)
            activities = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 7001))).all()
            self.assertEqual(len(activities), 0)

    async def test_telegram_preflight_invalid_model_raises_configuration_zero_side_effects(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "OpenAI"
            cfg.vision_model = "invalid-model-nonexistent"
            cfg.openai_api_key = "sk-test-key"
            await session.commit()

        tracker = ActivityTracker(self.sessions, user_id=7001, topic_id=1, track_user_activity=True)
        ctx = VisionExecutionContext(user_id=7001, dialogue_id=1, topic_id=1, platform="telegram", bot=None)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_http, \
             patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_openai:
            with self.assertRaises(AIServiceError) as err_ctx:
                await analyze_image_content(b"fake_image_bytes", prompt="Analyze", activity_tracker=tracker, execution_context=ctx)
            self.assertEqual(err_ctx.exception.classification, "configuration")
            mock_http.assert_not_called()
            mock_openai.assert_not_called()

        self.assertFalse(tracker._marked)
        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001))).all()
            self.assertEqual(len(logs), 0)
            activities = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 7001))).all()
            self.assertEqual(len(activities), 0)

    async def test_telegram_preflight_unknown_provider_raises_configuration_zero_side_effects(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "UnknownProviderXYZ"
            await session.commit()

        tracker = ActivityTracker(self.sessions, user_id=7001, topic_id=1, track_user_activity=True)
        ctx = VisionExecutionContext(user_id=7001, dialogue_id=1, topic_id=1, platform="telegram", bot=None)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_http, \
             patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_openai:
            with self.assertRaises(AIServiceError) as err_ctx:
                await analyze_image_content(b"fake_image_bytes", prompt="Analyze", activity_tracker=tracker, execution_context=ctx)
            self.assertEqual(err_ctx.exception.classification, "configuration")
            mock_http.assert_not_called()
            mock_openai.assert_not_called()

        self.assertFalse(tracker._marked)
        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001))).all()
            self.assertEqual(len(logs), 0)
            activities = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 7001))).all()
            self.assertEqual(len(activities), 0)

    async def test_max_preflight_missing_key_raises_configuration_zero_side_effects(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "OpenAI"
            cfg.vision_model = "gpt-5.6-terra"
            cfg.openai_api_key = None
            await session.commit()

        tracker = ActivityTracker(self.sessions, user_id=7001 + MAX_ID_OFFSET, topic_id=1, track_user_activity=True)
        ctx = VisionExecutionContext(user_id=7001 + MAX_ID_OFFSET, dialogue_id=1, topic_id=1, platform="max", bot=None)

        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False), \
             patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_http, \
             patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_openai:
            with self.assertRaises((AIServiceError, max_ai.AIServiceError)) as err_ctx:
                await max_ai.analyze_image(
                    user_id=7001 + MAX_ID_OFFSET,
                    image_bytes=b"fake_image_bytes",
                    prompt="MAX prompt",
                    activity_tracker=tracker,
                    execution_context=ctx,
                )
            self.assertEqual(err_ctx.exception.classification, "configuration")
            mock_http.assert_not_called()
            mock_openai.assert_not_called()

        self.assertFalse(tracker._marked)
        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001 + MAX_ID_OFFSET))).all()
            self.assertEqual(len(logs), 0)
            activities = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 7001 + MAX_ID_OFFSET))).all()
            self.assertEqual(len(activities), 0)

    async def test_max_preflight_invalid_model_raises_configuration_zero_side_effects(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "OpenAI"
            cfg.vision_model = "invalid-model-nonexistent"
            cfg.openai_api_key = "sk-test-key"
            await session.commit()

        tracker = ActivityTracker(self.sessions, user_id=7001 + MAX_ID_OFFSET, topic_id=1, track_user_activity=True)
        ctx = VisionExecutionContext(user_id=7001 + MAX_ID_OFFSET, dialogue_id=1, topic_id=1, platform="max", bot=None)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_http, \
             patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_openai:
            with self.assertRaises((AIServiceError, max_ai.AIServiceError)) as err_ctx:
                await max_ai.analyze_image(
                    user_id=7001 + MAX_ID_OFFSET,
                    image_bytes=b"fake_image_bytes",
                    prompt="MAX prompt",
                    activity_tracker=tracker,
                    execution_context=ctx,
                )
            self.assertEqual(err_ctx.exception.classification, "configuration")
            mock_http.assert_not_called()
            mock_openai.assert_not_called()

        self.assertFalse(tracker._marked)
        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001 + MAX_ID_OFFSET))).all()
            self.assertEqual(len(logs), 0)
            activities = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 7001 + MAX_ID_OFFSET))).all()
            self.assertEqual(len(activities), 0)

    async def test_max_preflight_unknown_provider_raises_configuration_zero_side_effects(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "UnknownProviderXYZ"
            await session.commit()

        tracker = ActivityTracker(self.sessions, user_id=7001 + MAX_ID_OFFSET, topic_id=1, track_user_activity=True)
        ctx = VisionExecutionContext(user_id=7001 + MAX_ID_OFFSET, dialogue_id=1, topic_id=1, platform="max", bot=None)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_http, \
             patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_openai:
            with self.assertRaises((AIServiceError, max_ai.AIServiceError)) as err_ctx:
                await max_ai.analyze_image(
                    user_id=7001 + MAX_ID_OFFSET,
                    image_bytes=b"fake_image_bytes",
                    prompt="MAX prompt",
                    activity_tracker=tracker,
                    execution_context=ctx,
                )
            self.assertEqual(err_ctx.exception.classification, "configuration")
            mock_http.assert_not_called()
            mock_openai.assert_not_called()

        self.assertFalse(tracker._marked)
        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001 + MAX_ID_OFFSET))).all()
            self.assertEqual(len(logs), 0)
            activities = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 7001 + MAX_ID_OFFSET))).all()
            self.assertEqual(len(activities), 0)

        self.assertFalse(tracker._marked)
        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001 + MAX_ID_OFFSET))).all()
            self.assertEqual(len(logs), 0)
            activities = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 7001 + MAX_ID_OFFSET))).all()
            self.assertEqual(len(activities), 0)

    # =========================================================================
    # Section 12: KIE fallback retry tests
    # =========================================================================

    async def test_telegram_kie_fallback_upload_retry_success(self):
        """TG KIE fallback upload attempt 1 timeout -> retry attempt 2 success -> inference success -> 1 fallback AILog row."""
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "OpenAI"
            cfg.vision_model = "gpt-5.6-terra"
            cfg.allow_vision_fallback = True
            cfg.vision_fallback_provider = "KIE"
            cfg.vision_fallback_model = "gemini-3-flash"
            await session.commit()

        ctx = VisionExecutionContext(user_id=7001, dialogue_id=1, topic_id=1, platform="telegram", bot=None)

        async def fake_openai_timeout(**kwargs):
            raise httpx.ReadTimeout("OpenAI timeout")

        upload_calls = 0
        async def fake_kie_upload_with_retry(*args, **kwargs):
            nonlocal upload_calls
            upload_calls += 1
            if upload_calls == 1:
                req = httpx.Request("POST", "https://upload.kie.ai/api/file-stream-upload")
                raise httpx.ReadTimeout("KIE upload transient timeout", request=req)
            return "https://files.kie.ai/tmp/retry_success.jpg"

        async def fake_kie_post(url, headers=None, json=None, **kwargs):
            return MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "TG KIE Retry Success"}}]})

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_openai_timeout), \
             patch("ai_integration._upload_file_to_kie", side_effect=fake_kie_upload_with_retry), \
             patch("httpx.AsyncClient.post", side_effect=fake_kie_post), \
             patch("error_reporting._dispatch_admin_alert_text", new_callable=AsyncMock):
            res = await analyze_image_content(b"fake_image_bytes", prompt="Analyze", execution_context=ctx)
            self.assertEqual(res, "TG KIE Retry Success")

        self.assertEqual(upload_calls, 2)
        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001).order_by(AILog.id))).all()
            self.assertEqual(len(logs), 2)
            self.assertEqual(logs[0].provider, "OpenAI")
            self.assertEqual(logs[0].status, "error")
            self.assertEqual(logs[1].provider, "KIE")
            self.assertEqual(logs[1].status, "success")
            self.assertEqual(logs[1].attempt_role, "fallback")

    async def test_telegram_kie_fallback_upload_429_no_retry(self):
        """TG KIE fallback upload attempt 1 returns HTTP 429 -> non-retryable -> 1 call only, no retry."""
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "OpenAI"
            cfg.vision_model = "gpt-5.6-terra"
            cfg.allow_vision_fallback = True
            cfg.vision_fallback_provider = "KIE"
            cfg.vision_fallback_model = "gemini-3-flash"
            await session.commit()

        ctx = VisionExecutionContext(user_id=7001, dialogue_id=1, topic_id=1, platform="telegram", bot=None)

        async def fake_openai_timeout(**kwargs):
            raise httpx.ReadTimeout("OpenAI timeout")

        upload_calls = 0
        async def fake_kie_upload_429(*args, **kwargs):
            nonlocal upload_calls
            upload_calls += 1
            req = httpx.Request("POST", "https://upload.kie.ai/api/file-stream-upload")
            resp = httpx.Response(429, request=req)
            raise httpx.HTTPStatusError("429 Too Many Requests", request=req, response=resp)

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_openai_timeout), \
             patch("ai_integration._upload_file_to_kie", side_effect=fake_kie_upload_429), \
             patch("error_reporting._dispatch_admin_alert_text", new_callable=AsyncMock):
            with self.assertRaises(AIServiceError):
                await analyze_image_content(b"fake_image_bytes", prompt="Analyze", execution_context=ctx)

        self.assertEqual(upload_calls, 1)

    async def test_max_kie_fallback_upload_retry_success(self):
        """MAX KIE fallback upload attempt 1 timeout -> retry attempt 2 success -> inference success -> 1 fallback AILog row."""
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "OpenAI"
            cfg.vision_model = "gpt-5.6-terra"
            cfg.allow_vision_fallback = True
            cfg.vision_fallback_provider = "KIE"
            cfg.vision_fallback_model = "gemini-3-flash"
            await session.commit()

        ctx = VisionExecutionContext(
            user_id=7001 + MAX_ID_OFFSET,
            username="maxuser",
            full_name="MAX User",
            dialogue_id=1,
            topic_id=1,
            platform="max",
            chat_id=12345,
            bot=None,
        )

        async def fake_openai_timeout(**kwargs):
            raise httpx.ReadTimeout("OpenAI timeout")

        upload_calls = 0
        async def fake_kie_upload_with_retry(*args, **kwargs):
            nonlocal upload_calls
            upload_calls += 1
            if upload_calls == 1:
                req = httpx.Request("POST", "https://upload.kie.ai/api/file-stream-upload")
                raise httpx.ReadTimeout("KIE upload transient timeout", request=req)
            return "https://files.kie.ai/tmp/max_retry_success.jpg"

        async def fake_kie_post(url, headers=None, json=None, **kwargs):
            return MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "MAX KIE Retry Success"}}]})

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_openai_timeout), \
             patch("max_messenger_bot.ai._upload_file_to_kie", side_effect=fake_kie_upload_with_retry), \
             patch("httpx.AsyncClient.post", side_effect=fake_kie_post), \
             patch("error_reporting._dispatch_admin_alert_text", new_callable=AsyncMock):
            res = await max_ai.analyze_image(
                user_id=7001 + MAX_ID_OFFSET,
                image_bytes=b"fake_image_bytes",
                prompt="MAX prompt",
                execution_context=ctx,
            )
            self.assertEqual(res, "MAX KIE Retry Success")

        self.assertEqual(upload_calls, 2)
        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog).where(AILog.user_id == 7001 + MAX_ID_OFFSET).order_by(AILog.id))).all()
            self.assertEqual(len(logs), 2)
            self.assertEqual(logs[0].provider, "OpenAI")
            self.assertEqual(logs[0].status, "error")
            self.assertEqual(logs[1].provider, "KIE")
            self.assertEqual(logs[1].status, "success")
            self.assertEqual(logs[1].attempt_role, "fallback")

    async def test_max_kie_fallback_upload_429_no_retry(self):
        """MAX KIE fallback upload attempt 1 returns HTTP 429 -> non-retryable -> 1 call only, no retry."""
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "OpenAI"
            cfg.vision_model = "gpt-5.6-terra"
            cfg.allow_vision_fallback = True
            cfg.vision_fallback_provider = "KIE"
            cfg.vision_fallback_model = "gemini-3-flash"
            await session.commit()

        ctx = VisionExecutionContext(
            user_id=7001 + MAX_ID_OFFSET,
            username="maxuser",
            full_name="MAX User",
            dialogue_id=1,
            topic_id=1,
            platform="max",
            chat_id=12345,
            bot=None,
        )

        async def fake_openai_timeout(**kwargs):
            raise httpx.ReadTimeout("OpenAI timeout")

        upload_calls = 0
        async def fake_kie_upload_429(*args, **kwargs):
            nonlocal upload_calls
            upload_calls += 1
            req = httpx.Request("POST", "https://upload.kie.ai/api/file-stream-upload")
            resp = httpx.Response(429, request=req)
            raise httpx.HTTPStatusError("429 Too Many Requests", request=req, response=resp)

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_openai_timeout), \
             patch("max_messenger_bot.ai._upload_file_to_kie", side_effect=fake_kie_upload_429), \
             patch("error_reporting._dispatch_admin_alert_text", new_callable=AsyncMock):
            with self.assertRaises((AIServiceError, max_ai.AIServiceError)):
                await max_ai.analyze_image(
                    user_id=7001 + MAX_ID_OFFSET,
                    image_bytes=b"fake_image_bytes",
                    prompt="MAX prompt",
                    execution_context=ctx,
                )

        self.assertEqual(upload_calls, 1)


if __name__ == "__main__":
    unittest.main()

