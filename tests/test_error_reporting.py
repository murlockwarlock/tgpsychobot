import os
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import ai_integration
import error_reporting
from alert_cooldown import AlertCooldown


class EmptyMessageError(Exception):
    def __str__(self):
        return ""


class ErrorReportingTests(unittest.IsolatedAsyncioTestCase):
    def test_empty_exception_message_uses_exception_class(self):
        self.assertEqual(error_reporting.exception_summary(EmptyMessageError()), "EmptyMessageError")

    async def test_provider_attempts_are_rendered_in_order_without_traceback(self):
        bot = SimpleNamespace(send_message=AsyncMock())
        attempts = (
            {"provider": "KIE", "model": "elevenlabs/speech-to-text", "error": "ReadError"},
            {"provider": "OpenAI", "model": "whisper-1", "error": "credit_balance_exhausted"},
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
        self.assertNotIn("Traceback:", text)


class TranscriptionProviderChainTests(unittest.IsolatedAsyncioTestCase):
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
