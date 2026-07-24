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
    from max_messenger_bot.services.common import _notify_referrer_about_registration, _send_ai_text
    from memory_mode import MEMORY_MODE_TOPIC
    from response_buttons import ResponseButton


class MaxHistoryScopeTests(unittest.TestCase):
    def test_general_topic_memory_scope_uses_current_dialogue(self):
        user = SimpleNamespace(id=123, current_dialogue_id=7, current_topic_id=None)

        scope = ai._build_max_history_scope(user, MEMORY_MODE_TOPIC)
        sql = str(scope.compile(compile_kwargs={"literal_binds": True}))

        self.assertIn("messages.user_id = 123", sql)
        self.assertIn("messages.dialogue_id = 7", sql)
        self.assertIn("messages.topic_id IS NULL", sql)


class MaxChunkedResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_multichunk_response_appends_menu_to_last_chunk(self):
        client = SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock())

        await _send_ai_text(client, 123, "thinking-id", ["first", "second"])

        client.edit_message.assert_awaited_once_with(
            "thinking-id",
            text="first",
            attachments=None,
        )
        self.assertEqual(client.send_message.await_count, 1)
        final_call = client.send_message.await_args_list[-1]
        self.assertEqual(final_call.kwargs["text"], "second")
        self.assertTrue(final_call.kwargs["attachments"])

    async def test_single_chunk_keeps_menu_on_answer(self):
        client = SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock())

        await _send_ai_text(client, 123, "thinking-id", ["answer"])

        self.assertTrue(client.edit_message.await_args.kwargs["attachments"])
        client.send_message.assert_not_awaited()

    async def test_generated_buttons_are_attached_to_last_chunk(self):
        client = SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock())
        buttons = [[
            ResponseButton("YouTube", "url", "https://youtube.com/watch?v=1"),
            ResponseButton("Да", "action", "yes"),
        ]]

        await _send_ai_text(client, 123, "thinking-id", ["first", "second"], buttons)

        attachment = client.send_message.await_args.kwargs["attachments"][0]
        rows = attachment["payload"]["buttons"]
        self.assertEqual(rows[0][0]["type"], "link")
        self.assertEqual(rows[0][0]["url"], "https://youtube.com/watch?v=1")
        self.assertEqual(rows[0][1]["payload"], "ai_btn:yes")


class MaxReferralNotificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_referrer_receives_registration_bonus_notification(self):
        client = SimpleNamespace(send_message=AsyncMock())

        await _notify_referrer_about_registration(client, 100_123_456_789, 5)

        client.send_message.assert_awaited_once()
        call = client.send_message.await_args.kwargs
        self.assertEqual(call["user_id"], 123_456_789)
        self.assertIn("зарегистрировался новый пользователь", call["text"])
        self.assertIn("5 бонусных дн.", call["text"])


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


class MaxMailingInputTests(unittest.IsolatedAsyncioTestCase):
    def test_shared_message_attachment_is_parsed(self):
        from max_messenger_bot.models import parse_message

        message = parse_message({
            "message": {
                "timestamp": 2000,
                "sender": {"user_id": 55},
                "recipient": {"chat_id": 321, "user_id": 999},
                "body": {
                    "mid": "shared-1",
                    "text": "Текст пересланной публикации",
                    "attachments": [{
                        "type": "share",
                        "payload": {"token": "share-token", "url": "https://max.ru/example"},
                    }],
                },
            },
        })

        self.assertIsNotNone(message)
        self.assertEqual(message.media_type, "share")
        self.assertEqual(message.media_token, "share-token")

    def test_forwarded_message_link_text_is_parsed(self):
        from max_messenger_bot.models import parse_message

        message = parse_message({
            "message": {
                "timestamp": 2000,
                "sender": {"user_id": 55},
                "recipient": {"chat_id": 321, "user_id": 999},
                "body": {"mid": "forward-1", "seq": 10, "text": ""},
                "link": {
                    "type": "forward",
                    "message": {"mid": "source-1", "seq": 2, "text": "Текст пересылки"},
                },
            },
        })

        self.assertIsNotNone(message)
        self.assertEqual(message.text, "Текст пересылки")

    async def test_recovers_forwarded_message_from_chat_history(self):
        from max_messenger_bot.services import admin_mailing

        user_id = 100_000_000_055
        snapshot = SimpleNamespace(
            state="admin_mailing_text",
            data={"audience": "self", "input_after_ms": 1000, "input_request_id": "request-1"},
        )
        states = SimpleNamespace(get=AsyncMock(return_value=snapshot), set=AsyncMock())
        client = SimpleNamespace(
            get_messages=AsyncMock(return_value={
                "messages": [
                    {
                        "timestamp": 2100,
                        "sender": {"user_id": 999},
                        "recipient": {"chat_id": 321, "user_id": 55},
                        "body": {"text": "Подсказка бота", "attachments": []},
                    },
                    {
                        "timestamp": 2000,
                        "sender": {"user_id": 55},
                        "recipient": {"chat_id": 321, "user_id": 999},
                        "body": {
                            "mid": "shared-2",
                            "text": "Готовая публикация",
                            "attachments": [{"type": "share", "payload": {"token": "share-token"}}],
                        },
                    },
                ],
            }),
            send_message=AsyncMock(),
        )

        captured = await admin_mailing.capture_latest_input(client, states, 321, user_id)

        self.assertTrue(captured)
        state_call = states.set.await_args
        self.assertEqual(state_call.args[2], "admin_mailing_preview")
        self.assertEqual(state_call.args[3]["text"], "Готовая публикация")
        self.assertEqual(state_call.args[3]["media_type"], "share")
        preview_attachments = client.send_message.await_args.kwargs["attachments"]
        self.assertEqual(preview_attachments[0], {"type": "share", "payload": {"token": "share-token"}})

    async def test_watcher_recovers_forward_link_from_chat_history(self):
        from max_messenger_bot.services import admin_mailing

        user_id = 100_000_000_055
        snapshot = SimpleNamespace(
            state="admin_mailing_text",
            data={"audience": "self", "input_after_ms": 1000, "input_request_id": "request-2"},
        )
        states = SimpleNamespace(get=AsyncMock(return_value=snapshot), set=AsyncMock())
        client = SimpleNamespace(
            get_messages=AsyncMock(return_value={
                "messages": [{
                    "timestamp": 2000,
                    "sender": {"user_id": 55},
                    "recipient": {"chat_id": 321, "user_id": 999},
                    "body": {"mid": "forward-2", "seq": 10, "text": ""},
                    "link": {
                        "type": "forward",
                        "message": {"mid": "source-2", "seq": 2, "text": "Текст пересылки"},
                    },
                }],
            }),
            send_message=AsyncMock(),
        )

        captured = await admin_mailing.capture_latest_input(
            client,
            states,
            321,
            user_id,
            shares_only=True,
        )

        self.assertTrue(captured)
        state_call = states.set.await_args
        self.assertEqual(state_call.args[2], "admin_mailing_preview")
        self.assertEqual(state_call.args[3]["text"], "Текст пересылки")

    async def test_audience_step_exposes_manual_history_fallback(self):
        from max_messenger_bot.services import admin_mailing

        states = SimpleNamespace(set=AsyncMock())
        client = SimpleNamespace(send_message=AsyncMock())

        request_id = await admin_mailing.choose_audience(client, states, 321, 100_000_000_055, "self")

        self.assertTrue(request_id)
        payload = states.set.await_args.args[3]
        self.assertEqual(payload["audience"], "self")
        self.assertEqual(payload["input_request_id"], request_id)
        rows = client.send_message.await_args.kwargs["attachments"][0]["payload"]["buttons"]
        callbacks = [button.get("payload") for row in rows for button in row]
        self.assertIn("mailing_use_latest", callbacks)

    async def test_watcher_automatically_captures_shared_message(self):
        from max_messenger_bot.services import admin_mailing

        snapshot = SimpleNamespace(
            state="admin_mailing_text",
            data={"input_request_id": "request-1"},
        )
        states = SimpleNamespace(get=AsyncMock(return_value=snapshot))
        client = SimpleNamespace()

        with patch.object(admin_mailing, "capture_latest_input", AsyncMock(return_value=True)) as capture:
            await admin_mailing.watch_for_shared_input(
                client,
                states,
                321,
                100_000_000_055,
                "request-1",
                poll_interval=0,
                max_checks=1,
            )

        capture.assert_awaited_once_with(
            client,
            states,
            321,
            100_000_000_055,
            notify_if_missing=False,
            shares_only=True,
        )


class MaxGeneratedButtonActionTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_test_action_uses_existing_test_flow(self):
        from max_messenger_bot.app import MaxBotApplication
        from max_messenger_bot.models import IncomingCallback, Sender

        client = SimpleNamespace(answer_callback=AsyncMock())
        app = MaxBotApplication(client)
        callback = IncomingCallback(
            raw={},
            callback_id="callback-1",
            payload="ai_btn:start_test",
            chat_id=321,
            message_id="message-1",
            sender=Sender(user_id=100_123, username="user", first_name="User", last_name=""),
        )

        with patch("max_messenger_bot.app.tests_service.start_test", AsyncMock()) as start_test:
            await app.handle_callback(callback)

        client.answer_callback.assert_awaited_once_with("callback-1")
        start_test.assert_awaited_once_with(client, 321, 100_123, app.states)

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


class MaxFormattingTests(unittest.TestCase):
    def test_markdown_to_html_converts_lists_and_bold(self):
        from max_messenger_bot.formatting import markdown_to_html
        
        # Test bullet lists and bold text
        text = "* **Item 1**\n- **Item 2**\n+ Item 3"
        html_out = markdown_to_html(text)
        self.assertEqual(html_out, "• <b>Item 1</b>\n• <b>Item 2</b>\n• Item 3")

        # Test headings and heading spacing
        text_headings = "### Heading 3\nSome text"
        html_headings = markdown_to_html(text_headings)
        self.assertEqual(html_headings, "<b>Heading 3</b>\nSome text")

        text_spacing = "Some text.\n### Heading"
        html_spacing = markdown_to_html(text_spacing)
        self.assertEqual(html_spacing, "Some text.\n\n<b>Heading</b>")

        # Test bullet lists remain close together but headings get spacing
        text_complex = "Intro text.\n* Bullet 1\n* Bullet 2\n### Subheading\nSome other text."
        html_complex = markdown_to_html(text_complex)
        self.assertEqual(
            html_complex,
            "Intro text.\n• Bullet 1\n• Bullet 2\n\n<b>Subheading</b>\nSome other text."
        )

        # Test standard bold and italic text
        text_bold_italic = "This is **bold** and *italic* text."
        html_bold_italic = markdown_to_html(text_bold_italic)
        self.assertEqual(html_bold_italic, "This is <b>bold</b> and <i>italic</i> text.")

    def test_render_markup_html_utf16(self):
        from max_messenger_bot.models import _render_markup_html
        
        # Test surrogate pair emoji causes offset shift.
        # "👥" is 2 UTF-16 code units.
        # " " is 1 UTF-16 code unit.
        # So "НАШИ ЭКСПЕРТЫ" starts at index 3 in UTF-16 and has length 13.
        text = "👥 НАШИ ЭКСПЕРТЫ"
        markups = [{'from': 3, 'length': 13, 'type': 'strong'}]
        
        rendered = _render_markup_html(text, markups)
        self.assertEqual(rendered, "👥 <b>НАШИ ЭКСПЕРТЫ</b>")

    def test_translate_telegram_links_to_max(self):
        from max_messenger_bot.formatting import translate_telegram_links_to_max
        from unittest.mock import patch, MagicMock
        
        # Test rewriting telegram links to max platform links using settings bot name
        settings_mock = MagicMock()
        settings_mock.bot_name = "test_bot_name"
        
        with patch("max_messenger_bot.settings.get_settings", return_value=settings_mock):
             # Test tg://resolve format with start payload
            text_tg = "Click here: tg://resolve?domain=yourself_way_bot&start=about"
            text_tg_amp = "Click here: tg://resolve?domain=yourself_way_bot&amp;start=about"
            # Test http/https format
            text_http = "Click here: https://t.me/yourself_way_bot?start=about_me"
            text_hyphen = "Click here: tg://resolve?domain=yourself_way_bot&start=about-me-now"
            text_http_hyphen = "Click here: https://t.me/yourself_way_bot?start=about-me-now"
            
            self.assertEqual(translate_telegram_links_to_max(text_tg), "Click here: https://max.ru/test_bot_name?start=about")
            self.assertEqual(translate_telegram_links_to_max(text_tg_amp), "Click here: https://max.ru/test_bot_name?start=about")
            self.assertEqual(translate_telegram_links_to_max(text_http), "Click here: https://max.ru/test_bot_name?start=about_me")
            self.assertEqual(translate_telegram_links_to_max(text_hyphen), "Click here: https://max.ru/test_bot_name?start=about-me-now")
            self.assertEqual(translate_telegram_links_to_max(text_http_hyphen), "Click here: https://max.ru/test_bot_name?start=about-me-now")


class MockStateStore:
    def __init__(self):
        self.states = {}

    async def get(self, user_id):
        from max_messenger_bot.storage import StateSnapshot
        if user_id in self.states:
            return StateSnapshot(state=self.states[user_id]["state"], data=self.states[user_id]["data"])
        return None

    async def set(self, user_id, chat_id, state, data=None):
        self.states[user_id] = {"chat_id": chat_id, "state": state, "data": data or {}}

    async def clear(self, user_id):
        self.states.pop(user_id, None)


class MaxAdminContentEditorTests(unittest.IsolatedAsyncioTestCase):
    async def test_admin_content_editor_flow(self):
        from max_messenger_bot.services import admin_content
        from max_messenger_bot.legacy import Content
        from max_messenger_bot.storage import MaxContentMedia

        # Mock database session
        session = MagicMock()
        session.commit = AsyncMock()
        content_item = Content(key="about_me", text_content="old text", content_order="media_top", is_visible=True)
        session.get = AsyncMock(return_value=content_item)

        media_rows = [
            MaxContentMedia(content_key="about_me", media_type="photo", token="tok123"),
        ]
        content_list_rows = [
            content_item,
        ]

        def mock_execute(query):
            query_str = str(query)
            scalars_mock = MagicMock()
            if "max_content_media" in query_str:
                scalars_mock.all.return_value = media_rows
            else:
                scalars_mock.all.return_value = content_list_rows
            execute_mock = MagicMock()
            execute_mock.scalars.return_value = scalars_mock
            return execute_mock

        session.execute = AsyncMock(side_effect=mock_execute)

        session_context = MagicMock()
        session_context.__aenter__.return_value = session
        session_context.__aexit__.return_value = False

        client = MagicMock()
        client.send_message = AsyncMock()

        states = MockStateStore()

        with patch("max_messenger_bot.services.admin_content.async_session_maker", return_value=session_context):
            # 1. Test show_content_editor initializes state
            await admin_content.show_content_editor(client, states, chat_id=111, user_id=222, content_key="about_me")

            # Verify state was initialized
            snapshot = await states.get(222)
            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot.state, "admin_edit_content")
            self.assertEqual(snapshot.data["content_key"], "about_me")
            self.assertEqual(snapshot.data["text_content"], "old text")
            self.assertEqual(len(snapshot.data["media_files"]), 1)
            self.assertEqual(snapshot.data["media_files"][0]["token"], "tok123")

            # 2. Test receive_message updates state with text and media
            await admin_content.receive_message(client, states, chat_id=111, user_id=222, text="new text", media_token=None, media_type=None)
            snapshot = await states.get(222)
            self.assertEqual(snapshot.data["text_content"], "new text")

            await admin_content.receive_message(client, states, chat_id=111, user_id=222, text=None, media_token="tok456", media_type="image")
            snapshot = await states.get(222)
            self.assertEqual(len(snapshot.data["media_files"]), 2)
            self.assertEqual(snapshot.data["media_files"][1]["token"], "tok456")
            self.assertEqual(snapshot.data["media_files"][1]["type"], "photo")

            # 3. Test handle_order_toggle toggles ordering
            await admin_content.handle_order_toggle(client, states, chat_id=111, user_id=222, content_key="about_me")
            snapshot = await states.get(222)
            self.assertEqual(snapshot.data["content_order"], "text_top")

            # 4. Test handle_media_delete deletes media file
            await admin_content.handle_media_delete(client, states, chat_id=111, user_id=222, content_key="about_me", index=0)
            snapshot = await states.get(222)
            self.assertEqual(len(snapshot.data["media_files"]), 1)
            self.assertEqual(snapshot.data["media_files"][0]["token"], "tok456")

            # 5. Test handle_save_content commits to db
            await admin_content.handle_save_content(client, states, chat_id=111, user_id=222, content_key="about_me")
            # Verify state was cleared
            snapshot = await states.get(222)
            self.assertIsNone(snapshot)

            # Verify session calls to update Content and MaxContentMedia
            self.assertEqual(content_item.text_content, "new text")
            self.assertEqual(content_item.content_order, "text_top")
            session.commit.assert_called()


if __name__ == "__main__":
    unittest.main()
