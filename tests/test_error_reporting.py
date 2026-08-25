import os
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import ai_integration
import error_reporting
from alert_cooldown import AlertCooldown


class EmptyMessageError(Exception):
    def __str__(self):
        return ""


class FakeSSLError(Exception):
    pass


class ErrorReportingTests(unittest.IsolatedAsyncioTestCase):
    def test_empty_exception_message_uses_exception_class(self):
        self.assertEqual(error_reporting.exception_summary(EmptyMessageError()), "EmptyMessageError")

    def test_classify_ai_error_401_auth(self):
        code, msg = error_reporting.classify_ai_error(Exception("AuthenticationError: Invalid API Key provided"))
        self.assertEqual(code, "auth_invalid_key")
        self.assertIn("401", msg)

    def test_classify_ai_error_402_credits(self):
        code, msg = error_reporting.classify_ai_error(Exception("Insufficient credits: purchase credits to continue (code: 402)"))
        self.assertEqual(code, "insufficient_balance")
        self.assertIn("402", msg)

    def test_classify_ai_error_429_rate_limit(self):
        code, msg = error_reporting.classify_ai_error(Exception("RateLimitError: 429 Too Many Requests"))
        self.assertEqual(code, "rate_limited")
        self.assertIn("429", msg)

    def test_classify_ai_error_timeout(self):
        code, msg = error_reporting.classify_ai_error(Exception("APITimeoutError: Request timed out after 60 seconds"))
        self.assertEqual(code, "timeout")
        self.assertIn("Timeout", msg)

    def test_classify_ai_error_5xx(self):
        code, msg = error_reporting.classify_ai_error(Exception("InternalServerError: 503 Service Unavailable / server overloaded"))
        self.assertEqual(code, "provider_5xx")
        self.assertIn("5xx", msg)

    def test_classify_ai_error_network(self):
        code, msg = error_reporting.classify_ai_error(Exception("ConnectError: Connection error while contacting upstream"))
        self.assertEqual(code, "network_error")
        self.assertIn("сетевого", msg)

    def test_classify_ai_error_empty_response(self):
        code, msg = error_reporting.classify_ai_error(Exception("Провайдер вернул пустой ответ (no response data)"))
        self.assertEqual(code, "empty_response")
        self.assertIn("пустой ответ", msg)

    def test_classify_ai_error_retired_model(self):
        code, msg = error_reporting.classify_ai_error(Exception("Модель 'gemini-2.0-flash' отключена провайдером"))
        self.assertEqual(code, "retired_model_unsupported")
        self.assertIn("отключена провайдером", msg)

    def test_classify_ai_error_geo_blocked(self):
        code, msg = error_reporting.classify_ai_error(Exception("User location is not supported in this region / 403"))
        self.assertEqual(code, "geo_blocked")
        self.assertIn("Geo-Block", msg)

    def test_classify_ai_error_missing_config(self):
        code, msg = error_reporting.classify_ai_error(Exception("Не указан API ключ OpenAI"))
        self.assertEqual(code, "missing_config")

    def test_chained_ssl_error_uses_deep_root_cause(self):
        ssl_error = FakeSSLError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF")
        sdk_error = AttributeError("'NoneType' object has no attribute 'status_code'")
        sdk_error.__cause__ = ssl_error

        self.assertEqual(error_reporting.classify_external_error(sdk_error)[0], "network_ssl")
        self.assertEqual(error_reporting.exception_summary(sdk_error), str(ssl_error))

    async def test_admin_alert_exposes_root_classification_not_secondary_sdk_error(self):
        bot = SimpleNamespace(send_message=AsyncMock())
        ssl_error = FakeSSLError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF")
        sdk_error = AttributeError("'NoneType' object has no attribute 'status_code'")
        sdk_error.__cause__ = ssl_error

        with patch.object(error_reporting, "get_all_admin_ids", AsyncMock(return_value=[123])):
            await error_reporting.notify_admins_about_error(
                bot,
                title="Сбой создания платежа YooKassa",
                provider="YooKassa",
                stage="create_payment",
                exception=sdk_error,
                include_traceback=False,
            )

        text = bot.send_message.await_args.args[1]
        self.assertIn("network_ssl", text)
        self.assertIn("FakeSSLError", text)
        self.assertIn("UNEXPECTED_EOF_WHILE_READING", text)
        self.assertNotIn("NoneType", text)

    async def test_yookassa_persistence_alert_uses_application_classification_override(self):
        bot = SimpleNamespace(send_message=AsyncMock())

        with patch.object(error_reporting, "get_all_admin_ids", AsyncMock(return_value=[123])):
            await error_reporting.notify_admins_about_error(
                bot,
                title="Сбой сохранения платежа YooKassa",
                provider="YooKassa",
                stage="persist_payment",
                exception=RuntimeError("database commit failed"),
                classification_override="application_internal",
                extra={"payment_id": "payment-1", "provider_status": "SUCCESS"},
                include_traceback=False,
            )

        text = bot.send_message.await_args.args[1]
        self.assertIn("persist_payment", text)
        self.assertIn("application_internal", text)
        self.assertIn("provider_status=SUCCESS", text)
        self.assertNotIn("network_ssl", text)

    async def test_chained_exception_secrets_are_redacted(self):
        bot = SimpleNamespace(send_message=AsyncMock())
        root = FakeSSLError("TLS failed password=pass1 signature=deadbeef")
        outer = RuntimeError("SDK wrapper password=pass2")
        outer.__cause__ = root

        with patch.object(error_reporting, "get_all_admin_ids", AsyncMock(return_value=[123])):
            await error_reporting.notify_admins_about_error(
                bot,
                title="External failure",
                provider="Robokassa",
                exception=outer,
                include_traceback=True,
                extra={"password": "pass3", "safe": "ok"},
            )

        text = bot.send_message.await_args.args[1]
        for secret in ("pass1", "pass2", "pass3", "deadbeef"):
            self.assertNotIn(secret, text)
        self.assertIn("[REDACTED]", text)

    def test_sanitize_secret_values_redacts_keys_and_tokens(self):
        sample = (
            "Error on url https://api.openai.com/v1?key=AIzaSySecretKey1234567890 "
            "with Authorization: Bearer sk-proj-superSecretToken12345678 and api_key='sk-abcdef1234567890'"
        )
        cleaned = error_reporting.sanitize_secret_values(sample)
        self.assertNotIn("AIzaSySecretKey1234567890", cleaned)
        self.assertNotIn("sk-proj-superSecretToken12345678", cleaned)
        self.assertNotIn("sk-abcdef1234567890", cleaned)
        self.assertIn("[REDACTED]", cleaned)

    async def test_provider_attempts_are_rendered_in_order_without_traceback(self):
        bot = SimpleNamespace(send_message=AsyncMock())
        attempts = (
            {"provider": "KIE", "model": "elevenlabs/speech-to-text", "error": "ReadError: connection dropped"},
            {"provider": "OpenAI", "model": "whisper-1", "error": "credit_balance_exhausted on sk-secret12345678"},
        )

        with patch.object(error_reporting, "get_all_admin_ids", AsyncMock(return_value=[123])):
            await error_reporting.notify_admins_about_error(
                bot,
                title="Сбой транскрибации",
                provider_attempts=attempts,
                exception=RuntimeError("fallback failed"),
                include_traceback=False,
            )

        text = bot.send_message.await_args.args[1]
        self.assertLess(text.index("KIE"), text.index("OpenAI"))
        self.assertIn("ReadError", text)
        self.assertIn("credit_balance_exhausted", text)
        self.assertNotIn("sk-secret12345678", text)  # Secret redacted in attempts
        self.assertNotIn("Traceback:", text)

    async def test_non_ai_error_includes_traceback_when_requested(self):
        bot = SimpleNamespace(send_message=AsyncMock())
        try:
            raise ValueError("Test unexpected runtime error")
        except ValueError as e:
            test_exc = e

        with patch.object(error_reporting, "get_all_admin_ids", AsyncMock(return_value=[123])):
            await error_reporting.notify_admins_about_error(
                bot,
                title="Системная ошибка",
                exception=test_exc,
                include_traceback=True,
            )

        text = bot.send_message.await_args.args[1]
        self.assertIn("Traceback:", text)
        self.assertIn("Test unexpected runtime error", text)


class TranscriptionProviderChainTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_kie_multimodal_exception_is_not_reported_as_blank_reason(self):
        class EmptyProviderError(Exception):
            def __str__(self):
                return ""

        response = SimpleNamespace(status_code=200, json=Mock(side_effect=EmptyProviderError()))
        client = SimpleNamespace(post=AsyncMock(return_value=response))
        client_context = MagicMock()
        client_context.__aenter__.return_value = client
        client_context.__aexit__.return_value = False

        with patch.object(ai_integration.httpx, "AsyncClient", return_value=client_context):
            with self.assertRaises(ai_integration.AIServiceError) as raised:
                await ai_integration._call_kie_multimodal(
                    "kie-key",
                    "https://kie.example",
                    "gemini-3-flash",
                    "system",
                    [{"type": "text", "text": "hello"}],
                )

        self.assertIn("EmptyProviderError", str(raised.exception))
        self.assertFalse(str(raised.exception).endswith(":"))

    async def test_kie_stt_converts_telegram_ogg_to_wav_before_upload(self):
        with (
            patch.object(
                ai_integration,
                "_prepare_kie_transcription_audio",
                AsyncMock(return_value=(b"wav-audio", "voice.wav")),
            ) as prepare,
            patch.object(
                ai_integration,
                "_upload_file_to_kie",
                AsyncMock(return_value="https://files.example/voice.wav"),
            ) as upload,
            patch.object(ai_integration, "_create_kie_task", AsyncMock(return_value="task-1")) as create,
            patch.object(
                ai_integration,
                "_poll_kie_task",
                AsyncMock(return_value={"resultJson": '{"text":"готово"}'}),
            ),
        ):
            result = await ai_integration._call_kie_transcribe(
                "key",
                "https://api.example",
                "https://upload.example",
                "elevenlabs/speech-to-text",
                b"ogg-audio",
                "voice.ogg",
            )

        self.assertEqual(result, "готово")
        prepare.assert_awaited_once_with(b"ogg-audio", "voice.ogg")
        upload.assert_awaited_once_with(
            "key",
            "https://upload.example",
            b"wav-audio",
            "voice.wav",
            "audio",
        )
        self.assertEqual(create.await_args.args[3]["language_code"], "ru")

    async def test_transient_kie_step_is_retried(self):
        operation = AsyncMock(side_effect=[ai_integration.AIServiceError("ReadError"), "ok"])

        with patch.object(ai_integration.asyncio, "sleep", AsyncMock()) as sleep:
            result = await ai_integration._retry_kie_transcription_step("upload", operation)

        self.assertEqual(result, "ok")
        self.assertEqual(operation.await_count, 2)
        sleep.assert_awaited_once_with(1)

    async def test_kie_and_openai_failures_are_preserved_in_order(self):
        config = SimpleNamespace(
            transcription_provider="KIE",
            kie_api_key="kie-key",
            kie_transcription_model="elevenlabs/speech-to-text",
            kie_model="gemini-3-flash",
            kie_base_url="https://kie.example",
            kie_upload_base_url="https://upload.example",
            openai_api_key="openai-key",
        )
        session = AsyncMock()
        session.get.return_value = config
        session_context = MagicMock()
        session_context.__aenter__.return_value = session
        session_context.__aexit__.return_value = False

        kie_error = ai_integration.AIServiceError("KIE upload: ReadError")
        openai_error = ai_integration.InsufficientBalanceError("OpenAI: credit_balance_exhausted")

        with (
            patch.object(ai_integration, "async_session_maker", return_value=session_context),
            patch.object(ai_integration, "_call_kie_transcribe", AsyncMock(side_effect=kie_error)),
            patch.object(ai_integration, "_call_openai_transcribe", AsyncMock(side_effect=openai_error)),
        ):
            with self.assertRaises(ai_integration.InsufficientBalanceError) as raised:
                await ai_integration.transcribe_voice_message(b"audio", "voice.ogg")

        attempts = raised.exception.provider_attempts
        self.assertEqual([attempt["provider"] for attempt in attempts], ["KIE", "OpenAI"])
        self.assertEqual(attempts[0]["error"], "KIE upload: ReadError")
        self.assertEqual(attempts[1]["error"], "OpenAI: credit_balance_exhausted")


class KieCreditCheckTests(unittest.IsolatedAsyncioTestCase):
    async def test_transport_failure_is_retried_before_returning_balance(self):
        response = SimpleNamespace(status_code=200, text='{"data": 464.13}', json=lambda: {"data": 464.13})
        client = SimpleNamespace(
            get=AsyncMock(side_effect=[ai_integration.httpx.ConnectError("connection failed"), response])
        )
        client_context = MagicMock()
        client_context.__aenter__.return_value = client
        client_context.__aexit__.return_value = False

        with (
            patch.object(ai_integration.httpx, "AsyncClient", return_value=client_context),
            patch.object(ai_integration.asyncio, "sleep", AsyncMock()) as sleep,
        ):
            balance = await ai_integration.get_kie_remaining_credits("key", "https://api.example")

        self.assertEqual(balance, 464.13)
        self.assertEqual(client.get.await_count, 2)
        sleep.assert_awaited_once_with(1)

    def test_credit_error_alert_has_three_hour_cooldown(self):
        cooldown = AlertCooldown(timedelta(hours=3))
        started = datetime(2026, 8, 1, 12, 0, 0)
        self.assertTrue(cooldown.should_send(started))
        self.assertFalse(cooldown.should_send(started + timedelta(minutes=15)))
        self.assertTrue(cooldown.should_send(started + timedelta(hours=3)))


if __name__ == "__main__":
    unittest.main()
