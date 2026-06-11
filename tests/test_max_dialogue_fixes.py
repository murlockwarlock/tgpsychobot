import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import sqlalchemy.ext.asyncio as sqlalchemy_asyncio


os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

original_create_async_engine = sqlalchemy_asyncio.create_async_engine


def _sqlite_compatible_engine(*args, **kwargs):
    kwargs.pop("pool_recycle", None)
    kwargs.pop("pool_use_lifo", None)
    return original_create_async_engine(*args, **kwargs)


with patch.object(sqlalchemy_asyncio, "create_async_engine", _sqlite_compatible_engine):
    from max_messenger_bot import ai
    from max_messenger_bot.services.common import _send_ai_text
    from memory_mode import MEMORY_MODE_TOPIC


class MaxHistoryScopeTests(unittest.TestCase):
    def test_general_topic_memory_scope_uses_current_dialogue(self):
        user = SimpleNamespace(id=123, current_dialogue_id=7, current_topic_id=None)

        scope = ai._build_max_history_scope(user, MEMORY_MODE_TOPIC)
        sql = str(scope.compile(compile_kwargs={"literal_binds": True}))

        self.assertIn("messages.user_id = 123", sql)
        self.assertIn("messages.dialogue_id = 7", sql)
        self.assertIn("messages.topic_id IS NULL", sql)


class MaxChunkedResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_multichunk_response_sends_separate_main_menu(self):
        client = SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock())

        await _send_ai_text(client, 123, "thinking-id", ["first", "second"])

        client.edit_message.assert_awaited_once_with(
            "thinking-id",
            text="first",
            attachments=None,
        )
        self.assertEqual(client.send_message.await_count, 2)
        final_call = client.send_message.await_args_list[-1]
        self.assertEqual(final_call.kwargs["text"], "Главное меню:")
        self.assertTrue(final_call.kwargs["attachments"])

    async def test_single_chunk_keeps_menu_on_answer(self):
        client = SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock())

        await _send_ai_text(client, 123, "thinking-id", ["answer"])

        self.assertTrue(client.edit_message.await_args.kwargs["attachments"])
        client.send_message.assert_not_awaited()


class MaxTranscriptionFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_kie_failure_falls_back_to_gemini(self):
        config = SimpleNamespace(
            transcription_provider="KIE",
            kie_api_key="kie-key",
            kie_transcription_model="elevenlabs/speech-to-text",
            kie_base_url="https://kie.example",
            kie_upload_base_url="https://upload.example",
            gemini_api_key="gemini-key",
            gemini_model="gemini-model",
        )
        session = AsyncMock()
        session.get.return_value = config
        session_context = MagicMock()
        session_context.__aenter__.return_value = session
        session_context.__aexit__.return_value = False

        with (
            patch.object(ai, "async_session_maker", return_value=session_context),
            patch.object(ai, "_transcribe_kie", AsyncMock(side_effect=ai.AIServiceError("timeout"))),
            patch.object(ai, "_transcribe_gemini", AsyncMock(return_value="готовый текст")) as gemini,
        ):
            result = await ai.transcribe_audio(b"audio", "voice.ogg")

        self.assertEqual(result, "готовый текст")
        gemini.assert_awaited_once_with(
            "gemini-key",
            "gemini-model",
            b"audio",
            "voice.ogg",
        )


if __name__ == "__main__":
    unittest.main()
