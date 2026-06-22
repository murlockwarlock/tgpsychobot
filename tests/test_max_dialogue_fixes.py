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


class MaxBotDeduplicationTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_updates_are_ignored(self):
        from max_messenger_bot.app import MaxBotApplication
        client = MagicMock()
        app = MaxBotApplication(client)

        update = {
            "type": "message_created",
            "event_id": "unique-event-id-1",
            "message": {
                "chat_id": 123,
                "sender": {"user_id": 456, "username": "user"},
                "body": {"text": "hello"}
            }
        }

        with patch.object(app, "handle_message", AsyncMock()) as mock_handle:
            # First time: processes update
            await app.handle_update(update)
            self.assertEqual(mock_handle.call_count, 1)

            # Second time: skips duplicate update
            await app.handle_update(update)
            self.assertEqual(mock_handle.call_count, 1)


class MaxBotHistorySlicingTests(unittest.TestCase):
    def test_qa_pairs_slicing_logic(self):
        # Setup mock messages
        class MockMsg:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        history_rows = [
            MockMsg('user', 'U1'),
            MockMsg('assistant', 'A1'),
            MockMsg('user', 'U2'),
            MockMsg('user', 'U2_extra'),
            MockMsg('assistant', 'A2'),
            MockMsg('user', 'U3'),
            MockMsg('assistant', 'A3'),
            MockMsg('user', 'U4'),
            MockMsg('assistant', 'A4'),
        ]

        # Slicing logic simulation (extracted from ai.py)
        pairs = []
        current_pair = []
        for msg in history_rows:
            if msg.role == 'user' and any(m.role == 'assistant' for m in current_pair):
                pairs.append(current_pair)
                current_pair = [msg]
            else:
                current_pair.append(msg)
        if current_pair:
            pairs.append(current_pair)

        # Expected pairs:
        # Pair 1: U1, A1
        # Pair 2: U2, U2_extra, A2
        # Pair 3: U3, A3
        # Pair 4: U4, A4
        self.assertEqual(len(pairs), 4)
        self.assertEqual([m.content for m in pairs[0]], ['U1', 'A1'])
        self.assertEqual([m.content for m in pairs[1]], ['U2', 'U2_extra', 'A2'])

        limit_first = 1
        limit_recent = 2

        if len(pairs) <= limit_first + limit_recent:
            selected_pairs = pairs
        else:
            selected_pairs = pairs[:limit_first] + pairs[-limit_recent:]

        # Expected selected pairs:
        # Pair 1 (first 1) + Pairs 3, 4 (recent 2)
        self.assertEqual(len(selected_pairs), 3)
        self.assertEqual([m.content for m in selected_pairs[0]], ['U1', 'A1'])
        self.assertEqual([m.content for m in selected_pairs[1]], ['U3', 'A3'])
        self.assertEqual([m.content for m in selected_pairs[2]], ['U4', 'A4'])


class MaxBotUploadStateRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_admin_kb_upload_file_state_routes_correctly(self):
        from max_messenger_bot.app import MaxBotApplication
        from max_messenger_bot.models import IncomingMessage, Sender

        client = MagicMock()
        client.send_message = AsyncMock()
        app = MaxBotApplication(client)

        message = IncomingMessage(
            raw={},
            message_id="msg1",
            chat_id=123,
            sender=Sender(user_id=456, username="user", first_name="User", last_name=""),
            text="Done",
            media_type=None,
            media_token=None,
            media_url=None
        )

        with (
            patch.object(app.states, "get", AsyncMock(return_value=SimpleNamespace(state="admin_kb_upload_file", data={}))),
            patch("max_messenger_bot.services.admin_kb.receive_upload_file", AsyncMock()) as mock_receive_file,
            patch("max_messenger_bot.services.common.ensure_user", AsyncMock())
        ):
            await app.handle_message(message)
            # Since message has no file attachment, it should prompt the user instead of doing run_ai_dialogue
            client.send_message.assert_awaited_once_with(
                chat_id=123,
                text="Пожалуйста, отправьте файл (txt, md, pdf, docx, xlsx) или завершите загрузку кнопкой."
            )
            mock_receive_file.assert_not_awaited()

        # Test when file attachment is present
        message_with_file = IncomingMessage(
            raw={},
            message_id="msg2",
            chat_id=123,
            sender=Sender(user_id=456, username="user", first_name="User", last_name=""),
            text="my_kb.txt",
            media_type="file",
            media_token="token123",
            media_url="http://example.com"
        )

        with (
            patch.object(app.states, "get", AsyncMock(return_value=SimpleNamespace(state="admin_kb_upload_file", data={}))),
            patch("max_messenger_bot.services.admin_kb.receive_upload_file", AsyncMock()) as mock_receive_file,
            patch("max_messenger_bot.services.common.ensure_user", AsyncMock())
        ):
            await app.handle_message(message_with_file)
            # Should route to receive_upload_file since it is in upload state and has a file attachment
            mock_receive_file.assert_awaited_once_with(client, app.states, message_with_file)


class MaxBotMediaTests(unittest.IsolatedAsyncioTestCase):
    def test_extract_directive_payload(self):
        from max_messenger_bot.services.common import _extract_ai_directive_payload
        text = "Hello world\nEDIT_IMG: a cute cat"
        payload, clean = _extract_ai_directive_payload(text, "EDIT_IMG")
        self.assertEqual(payload, "a cute cat")
        self.assertEqual(clean, "Hello world")

        text2 = "Some response\nGEN_IMG: blue sky"
        payload2, clean2 = _extract_ai_directive_payload(text2, "GEN_IMG")
        self.assertEqual(payload2, "blue sky")
        self.assertEqual(clean2, "Some response")

    async def test_analyze_image_filters_out_last_image_message(self):
        from max_messenger_bot import ai
        
        # Mock database session and queries
        config = SimpleNamespace(
            vision_provider="Gemini",
            gemini_api_key="gemini-key",
            vision_model="gemini-2.0-flash",
            temperature=0.7,
            system_prompt="some system prompt",
        )
        
        class MockMsg:
            def __init__(self, role, content):
                self.role = role
                self.content = content
                self.timestamp = 100

        user = SimpleNamespace(id=123, current_topic=None, current_dialogue_id=1, current_topic_id=None)
        
        history_rows = [
            MockMsg("user", "Hello"),
            MockMsg("assistant", "Hi there"),
            MockMsg("user", "[Изображение] test caption"),  # Current message saved to DB beforehand
        ]
        
        session = MagicMock()
        session.get = AsyncMock(return_value=config)
        session.scalar = AsyncMock(return_value=user)
        
        # Mocking execute().scalars().all()
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = history_rows
        execute_mock = MagicMock()
        execute_mock.scalars.return_value = scalars_mock
        session.execute = AsyncMock(return_value=execute_mock)
        
        session_context = MagicMock()
        session_context.__aenter__.return_value = session
        session_context.__aexit__.return_value = False
        
        with (
            patch.object(ai, "async_session_maker", return_value=session_context),
            patch.object(ai, "_analyze_gemini", AsyncMock(return_value="analyzed result")) as gemini,
        ):
            res = await ai.analyze_image(123, b"image_data", "test caption")
            
        self.assertEqual(res, "analyzed result")
        
        # Verify history passed to _analyze_gemini does NOT include the [Изображение] message (which was filtered out)
        history_passed = gemini.call_args.kwargs["history"]
        self.assertEqual(len(history_passed), 2)
        self.assertEqual(history_passed[0]["content"], "Hello")
        self.assertEqual(history_passed[1]["content"], "Hi there")


if __name__ == "__main__":
    unittest.main()
