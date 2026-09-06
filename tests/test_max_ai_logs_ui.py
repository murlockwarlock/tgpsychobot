import html
import io
import json
import os
import unittest
import zipfile
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from database import AILog, Base, User, async_session_maker, init_db
from max_messenger_bot.api import MaxApiClient, MaxApiError
from max_messenger_bot.app import MaxBotApplication
from max_messenger_bot import keyboards as max_keyboards
from max_messenger_bot.models import MAX_ID_OFFSET, IncomingCallback, IncomingMessage, Sender
from max_messenger_bot.services import admin_ai_logs
from max_messenger_bot.services.admin_ai_logs import (
    MAX_AI_LOG_DETAIL_TEXT_LIMIT,
    MAX_EXPORT_LOGS_LIMIT,
    MAX_EXPORT_UNCOMPRESSED_BYTES,
)


def _create_mock_client():
    client = SimpleNamespace(
        send_message=AsyncMock(return_value={"message_id": "123"}),
        send_text_file=AsyncMock(return_value={"message_id": "124"}),
        send_media_attachment=AsyncMock(return_value={"message_id": "125"}),
        upload_file=AsyncMock(return_value={"token": "fake_file_token_123"}),
        answer_callback=AsyncMock(return_value={"result": "ok"}),
        delete_message=AsyncMock(return_value={"result": "ok"}),
    )
    return client


def _create_sender(user_id: int, name: str = "Admin") -> Sender:
    return Sender(user_id=user_id, username=None, first_name=name, last_name=None)


class MaxAILogsUITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await init_db()
        async with async_session_maker() as session:
            await session.execute(AILog.__table__.delete())
            await session.execute(User.__table__.delete())
            await session.commit()

    # 1. Non-Admin Authorization Denial
    async def test_non_admin_denied_access_to_all_ai_logs_callbacks(self):
        non_admin_user_id = 99999
        with patch("max_messenger_bot.services.common.is_admin", AsyncMock(return_value=False)):
            app = MaxBotApplication.__new__(MaxBotApplication)
            app.client = _create_mock_client()
            app.states = SimpleNamespace(get=AsyncMock(return_value=None), clear=AsyncMock())

            test_callbacks = [
                "admin_ai_logs_0_all_all",
                f"admin_user_ai_logs_{MAX_ID_OFFSET + 55}_0_all_all",
                "admin_ai_log_1_0_0_all_all",
                "admin_ai_log_file_1",
                "export_ai_logs_0_all_all",
            ]

            for cb_data in test_callbacks:
                cb = IncomingCallback(
                    raw={},
                    callback_id="cb_test",
                    payload=cb_data,
                    sender=_create_sender(non_admin_user_id, "Attacker"),
                    chat_id=12345,
                    message_id="msg_1",
                )
                await app.handle_callback(cb)

            for call in app.client.answer_callback.await_args_list:
                self.assertEqual(call.kwargs.get("notification"), "Команда пока не реализована в MAX-версии")
            self.assertEqual(app.client.send_message.call_count, 0)
            self.assertEqual(app.client.send_text_file.call_count, 0)
            self.assertEqual(app.client.send_media_attachment.call_count, 0)

    # 2. Admin Authorization Callback ACK Exactly Once
    async def test_admin_callback_acknowledged_exactly_once_before_processing(self):
        admin_user_id = 1001
        with patch("max_messenger_bot.services.common.is_admin", AsyncMock(return_value=True)):
            app = MaxBotApplication.__new__(MaxBotApplication)
            app.client = _create_mock_client()
            app.states = SimpleNamespace(get=AsyncMock(return_value=None), clear=AsyncMock())

            cb = IncomingCallback(
                raw={},
                callback_id="cb_admin_123",
                payload="admin_ai_logs_0_all_all",
                sender=_create_sender(admin_user_id, "Admin"),
                chat_id=12345,
                message_id="msg_1",
            )
            await app.handle_callback(cb)

            app.client.answer_callback.assert_awaited_once_with("cb_admin_123")
            self.assertEqual(app.client.send_message.call_count, 1)

    # 3. AI Settings Keyboard Entry Point
    def test_admin_ai_settings_keyboard_contains_ai_logs_button(self):
        kb = max_keyboards.admin_ai_settings_keyboard("Gemini")
        buttons = [btn for row in kb[0]["payload"]["buttons"] for btn in row]
        ai_log_btn = next((btn for btn in buttons if btn["payload"] == "admin_ai_logs_0_all_all"), None)
        self.assertIsNotNone(ai_log_btn)
        self.assertIn("Логи запросов ИИ", ai_log_btn["text"])

    # 4. Client Profile Keyboard Entry Point (passes DB User.id)
    def test_admin_client_profile_keyboard_contains_user_ai_logs_button(self):
        db_user_id = MAX_ID_OFFSET + 100018792559
        kb = max_keyboards.admin_client_profile_keyboard(db_user_id)
        buttons = [btn for row in kb[0]["payload"]["buttons"] for btn in row]
        user_ai_log_btn = next((btn for btn in buttons if btn["payload"] == f"admin_user_ai_logs_{db_user_id}_0_all_all"), None)
        self.assertIsNotNone(user_ai_log_btn)
        self.assertIn("Логи запросов ИИ", user_ai_log_btn["text"])

    # 5. Global List Shared Telemetry (both Telegram and MAX logs)
    async def test_global_logs_list_shows_both_telegram_and_max_logs(self):
        tg_user_id = 777
        max_db_user_id = MAX_ID_OFFSET + 55

        async with async_session_maker() as session:
            session.add(AILog(
                id=1,
                user_id=tg_user_id,
                platform="telegram",
                provider="OpenAI",
                model="gpt-4o",
                latency_ms=800,
                request_payload='{"prompt": "Hello TG"}',
                raw_response="Hi TG user",
                clean_text="Hi TG user",
                created_at=datetime.utcnow() - timedelta(minutes=5),
            ))
            session.add(AILog(
                id=2,
                user_id=max_db_user_id,
                platform="max",
                provider="Gemini",
                model="gemini-2.5-flash",
                latency_ms=1200,
                request_payload='{"prompt": "Hello MAX"}',
                raw_response="Hi MAX user",
                clean_text="Hi MAX user",
                created_at=datetime.utcnow() - timedelta(minutes=2),
            ))
            await session.commit()

        client = _create_mock_client()
        await admin_ai_logs.show_ai_logs_list(client, chat_id=123, page=0, period="all", request_type="all")

        client.send_message.assert_awaited_once()
        kwargs = client.send_message.await_args.kwargs
        buttons = [btn for row in kwargs["attachments"][0]["payload"]["buttons"] for btn in row]
        callbacks = [btn["payload"] for btn in buttons]

        self.assertTrue(any(cb.startswith("admin_ai_log_1_") for cb in callbacks))
        self.assertTrue(any(cb.startswith("admin_ai_log_2_") for cb in callbacks))

    # 6. MAX Identity in List / Detail
    async def test_max_identity_detail_uses_raw_max_id_and_never_internal_offset(self):
        raw_id = 100018792559
        db_id = MAX_ID_OFFSET + raw_id

        async with async_session_maker() as session:
            user = User(
                id=db_id,
                name="Мама",
                first_name="Зоя Александровна",
                username="mama_max",
                tg_user_id=888888,
            )
            session.add(user)
            log_entry = AILog(
                id=10,
                user_id=db_id,
                platform="max",
                provider="Deepseek",
                model="deepseek-chat",
                latency_ms=1500,
                request_payload='{"message": "secret question"}',
                raw_response="secret answer",
                clean_text="secret answer",
                created_at=datetime(2026, 9, 6, 12, 0, 0),
            )
            session.add(log_entry)
            await session.commit()

        client = _create_mock_client()
        await admin_ai_logs.show_ai_log_detail(client, chat_id=123, log_id=10)

        client.send_message.assert_awaited_once()
        text = client.send_message.await_args.kwargs["text"]

        self.assertIn(f"🆔 <b>ID max:</b> <code>{raw_id}</code>", text)
        self.assertNotIn(str(db_id), text)
        self.assertIn("💬 <b>Имя для общения:</b> Мама", text)
        self.assertIn("👤 <b>Имя в max:</b> Зоя Александровна", text)
        self.assertIn("<b>Username:</b> @mama_max", text)
        self.assertIn("🔗 <b>Telegram ID:</b> <code>888888</code>", text)
        self.assertIn("📱 <b>Платформа:</b> MAX", text)

    # 7. Telegram Identity in Detail
    async def test_telegram_identity_detail_uses_telegram_formatting(self):
        tg_id = 777999

        async with async_session_maker() as session:
            user = User(
                id=tg_id,
                name="Иван",
                first_name="Ivan",
                username="ivan_tg",
            )
            session.add(user)
            log_entry = AILog(
                id=11,
                user_id=tg_id,
                platform="telegram",
                provider="Claude",
                model="claude-3-5-sonnet",
                latency_ms=900,
                request_payload='{"prompt": "TG request"}',
                raw_response="TG response",
                clean_text="TG response",
                created_at=datetime(2026, 9, 6, 12, 0, 0),
            )
            session.add(log_entry)
            await session.commit()

        client = _create_mock_client()
        await admin_ai_logs.show_ai_log_detail(client, chat_id=123, log_id=11)

        text = client.send_message.await_args.kwargs["text"]
        self.assertIn("📱 <b>Платформа:</b> Telegram", text)
        self.assertIn(f"🆔 <b>Telegram ID:</b> <code>{tg_id}</code>", text)
        self.assertIn("@ivan_tg", text)
        self.assertIn("💬 <b>Имя для общения:</b> Иван", text)

    # 8. Period Filters
    async def test_period_filtering_in_sql(self):
        now = datetime.utcnow()
        async with async_session_maker() as session:
            session.add(AILog(id=1, user_id=1, raw_response="R", created_at=now, provider="P", model="M"))
            session.add(AILog(id=2, user_id=1, raw_response="R", created_at=now - timedelta(days=2), provider="P", model="M"))
            session.add(AILog(id=3, user_id=1, raw_response="R", created_at=now - timedelta(days=10), provider="P", model="M"))
            session.add(AILog(id=4, user_id=1, raw_response="R", created_at=now - timedelta(days=40), provider="P", model="M"))
            await session.commit()

        client = _create_mock_client()

        # Today
        await admin_ai_logs.show_ai_logs_list(client, chat_id=123, page=0, period="today")
        text = client.send_message.await_args.kwargs["text"]
        self.assertIn("Всего вызовов: <b>1</b>", text)

        # 7 days
        await admin_ai_logs.show_ai_logs_list(client, chat_id=123, page=0, period="7d")
        text = client.send_message.await_args.kwargs["text"]
        self.assertIn("Всего вызовов: <b>2</b>", text)

        # 30 days
        await admin_ai_logs.show_ai_logs_list(client, chat_id=123, page=0, period="30d")
        text = client.send_message.await_args.kwargs["text"]
        self.assertIn("Всего вызовов: <b>3</b>", text)

        # All
        await admin_ai_logs.show_ai_logs_list(client, chat_id=123, page=0, period="all")
        text = client.send_message.await_args.kwargs["text"]
        self.assertIn("Всего вызовов: <b>4</b>", text)

    # 9. Request Type Filters
    async def test_request_type_filtering_in_sql(self):
        async with async_session_maker() as session:
            session.add(AILog(id=1, user_id=1, raw_response="R", request_type="chat", provider="P", model="M"))
            session.add(AILog(id=2, user_id=1, raw_response="R", request_type="chat", provider="P", model="M"))
            session.add(AILog(id=3, user_id=1, raw_response="R", request_type="followup", provider="P", model="M"))
            await session.commit()

        client = _create_mock_client()

        await admin_ai_logs.show_ai_logs_list(client, chat_id=123, page=0, request_type="chat")
        text = client.send_message.await_args.kwargs["text"]
        self.assertIn("Всего вызовов: <b>2</b>", text)

        await admin_ai_logs.show_ai_logs_list(client, chat_id=123, page=0, request_type="followup")
        text = client.send_message.await_args.kwargs["text"]
        self.assertIn("Всего вызовов: <b>1</b>", text)

    # 10. Combined Filters (User + Period + Type)
    async def test_combined_filters(self):
        now = datetime.utcnow()
        user_a = MAX_ID_OFFSET + 100
        user_b = MAX_ID_OFFSET + 200

        async with async_session_maker() as session:
            session.add(AILog(id=1, user_id=user_a, raw_response="R", request_type="chat", created_at=now, provider="P", model="M"))
            session.add(AILog(id=2, user_id=user_a, raw_response="R", request_type="followup", created_at=now, provider="P", model="M"))
            session.add(AILog(id=3, user_id=user_a, raw_response="R", request_type="chat", created_at=now - timedelta(days=20), provider="P", model="M"))
            session.add(AILog(id=4, user_id=user_b, raw_response="R", request_type="chat", created_at=now, provider="P", model="M"))
            await session.commit()

        client = _create_mock_client()
        await admin_ai_logs.show_ai_logs_list(client, chat_id=123, page=0, filter_user_id=user_a, period="7d", request_type="chat")
        text = client.send_message.await_args.kwargs["text"]
        self.assertIn("Всего вызовов: <b>1</b>", text)
        self.assertIn("ID max: 100", text)

    # 11. Pagination Boundaries and Stale Page Clamping
    async def test_pagination_and_stale_page_clamping(self):
        async with async_session_maker() as session:
            for i in range(1, 13):  # 12 records = 2 pages of 8
                session.add(AILog(id=i, user_id=1, raw_response="R", created_at=datetime.utcnow() - timedelta(minutes=i), provider="P", model="M"))
            await session.commit()

        client = _create_mock_client()

        # Page 0
        await admin_ai_logs.show_ai_logs_list(client, chat_id=123, page=0)
        text = client.send_message.await_args.kwargs["text"]
        self.assertIn("(Стр. 1/2)", text)

        # Page 1
        await admin_ai_logs.show_ai_logs_list(client, chat_id=123, page=1)
        text = client.send_message.await_args.kwargs["text"]
        self.assertIn("(Стр. 2/2)", text)

        # Stale Page 999 -> Clamps to Page 1
        await admin_ai_logs.show_ai_logs_list(client, chat_id=123, page=999)
        text = client.send_message.await_args.kwargs["text"]
        self.assertIn("(Стр. 2/2)", text)

    # 12. State Preservation (List -> Detail -> Back)
    def test_detail_screen_back_button_restores_exact_list_state(self):
        log_id = 42
        page = 2
        filter_user_id = MAX_ID_OFFSET + 55
        period = "7d"
        request_type = "followup"

        # User filtered detail
        kb = max_keyboards.admin_ai_log_detail_keyboard(log_id, page, filter_user_id, period, request_type)
        buttons = [btn for row in kb[0]["payload"]["buttons"] for btn in row]
        back_btn = next((b for b in buttons if b["text"] == "⬅️ Назад к логам"), None)
        self.assertIsNotNone(back_btn)
        self.assertEqual(back_btn["payload"], f"admin_user_ai_logs_{filter_user_id}_{page}_{period}_{request_type}")

        # Global detail
        kb_global = max_keyboards.admin_ai_log_detail_keyboard(log_id, page, None, period, request_type)
        buttons_global = [btn for row in kb_global[0]["payload"]["buttons"] for btn in row]
        back_btn_global = next((b for b in buttons_global if b["text"] == "⬅️ Назад к логам"), None)
        self.assertIsNotNone(back_btn_global)
        self.assertEqual(back_btn_global["payload"], f"admin_ai_logs_{page}_{period}_{request_type}")

    # 13. Missing/Deleted Log Resilience
    async def test_missing_or_deleted_log_gives_safe_admin_message(self):
        client = _create_mock_client()
        await admin_ai_logs.show_ai_log_detail(client, chat_id=123, log_id=999999)
        client.send_message.assert_awaited_once_with(chat_id=123, text="Запись лога не найдена.")

    # 14. Detail Message Hard Upper Bound (3800 chars)
    async def test_detail_message_hard_upper_bound_with_huge_payload(self):
        huge_payload = "PAYLOAD_" * 10000  # 80,000 chars
        huge_raw = "RAW_RESPONSE_" * 10000  # 130,000 chars

        async with async_session_maker() as session:
            log_entry = AILog(
                id=50,
                user_id=1,
                platform="max",
                provider="Gemini",
                model="gemini-flash",
                request_payload=huge_payload,
                raw_response=huge_raw,
                created_at=datetime.utcnow(),
            )
            session.add(log_entry)
            await session.commit()

        client = _create_mock_client()
        await admin_ai_logs.show_ai_log_detail(client, chat_id=123, log_id=50)

        text = client.send_message.await_args.kwargs["text"]
        self.assertLessEqual(len(text), MAX_AI_LOG_DETAIL_TEXT_LIMIT)
        self.assertGreater(len(text), 500)
        self.assertIn("📄 <b>Детали лога ИИ #50</b>", text)
        self.assertIn("Полный сырой файл скачайте по кнопке ниже", text)

    # 15. HTML Escaping Expansion Safety & Tag Validity
    async def test_html_escaping_expansion_safety_and_valid_tags(self):
        extreme_chars = "<script>alert('test' & \"hello\");</script>" * 200

        async with async_session_maker() as session:
            log_entry = AILog(
                id=51,
                user_id=1,
                platform="max",
                provider="Gemini",
                model="gemini-flash",
                request_payload=extreme_chars,
                raw_response=extreme_chars,
                created_at=datetime.utcnow(),
            )
            session.add(log_entry)
            await session.commit()

        client = _create_mock_client()
        await admin_ai_logs.show_ai_log_detail(client, chat_id=123, log_id=51)

        text = client.send_message.await_args.kwargs["text"]
        self.assertLessEqual(len(text), MAX_AI_LOG_DETAIL_TEXT_LIMIT)
        self.assertNotIn("<script>", text)
        self.assertIn("&lt;script&gt;", text)
        self.assertEqual(text.count("<code>"), text.count("</code>"))
        self.assertEqual(text.count("<b>"), text.count("</b>"))

    # 16. Metadata Preservation Under Truncation
    async def test_metadata_preserved_under_aggressive_truncation(self):
        raw_max_id = 987654321
        db_user_id = MAX_ID_OFFSET + raw_max_id

        async with async_session_maker() as session:
            user = User(id=db_user_id, name="Тестовый", first_name="Тест")
            session.add(user)
            log_entry = AILog(
                id=52,
                user_id=db_user_id,
                platform="max",
                provider="KIE",
                model="kie-vision-1",
                latency_ms=2540,
                context_kind="topic",
                topic_name_snapshot="Медитация",
                request_type="followup",
                request_payload="A" * 50000,
                raw_response="B" * 50000,
                created_at=datetime(2026, 9, 6, 15, 30, 0),
            )
            session.add(log_entry)
            await session.commit()

        client = _create_mock_client()
        await admin_ai_logs.show_ai_log_detail(client, chat_id=123, log_id=52)

        text = client.send_message.await_args.kwargs["text"]
        self.assertIn("Детали лога ИИ #52", text)
        self.assertIn(f"ID max:</b> <code>{raw_max_id}</code>", text)
        self.assertIn("Тип запроса:</b> догоняющие", text)
        self.assertIn("Контекст:</b> Тема диалога — «Медитация»", text)
        self.assertIn("Провайдер:</b> <b>KIE</b>", text)
        self.assertIn("Модель:</b> <code>kie-vision-1</code>", text)
        self.assertIn("2.54 сек", text)
        self.assertLessEqual(len(text), MAX_AI_LOG_DETAIL_TEXT_LIMIT)

    # 17. Full Untruncated Single TXT File Download
    async def test_download_single_txt_file_untruncated(self):
        full_payload = '{"messages": [{"role": "user", "content": "Full payload ' + 'X' * 5000 + '"}]}'
        full_raw = "FULL RAW RESPONSE " + "Y" * 5000
        full_clean = "FULL CLEAN TEXT " + "Z" * 5000

        async with async_session_maker() as session:
            log_entry = AILog(
                id=60,
                user_id=MAX_ID_OFFSET + 77,
                platform="max",
                provider="OpenAI",
                model="gpt-4o",
                latency_ms=1100,
                request_payload=full_payload,
                raw_response=full_raw,
                clean_text=full_clean,
                created_at=datetime(2026, 9, 6, 10, 0, 0),
            )
            session.add(log_entry)
            await session.commit()

        client = _create_mock_client()
        await admin_ai_logs.download_ai_log_file(client, chat_id=123, log_id=60)

        client.send_text_file.assert_awaited_once()
        kwargs = client.send_text_file.await_args.kwargs
        self.assertEqual(kwargs["filename"], "ai_log_60_OpenAI_gpt-4o.txt")
        content = kwargs["content"]

        self.assertIn("AI LOG RECORD #60", content)
        self.assertIn("User ID: 77", content)
        self.assertNotIn(str(MAX_ID_OFFSET + 77), content)
        self.assertIn("📤 [1] FULL REQUEST PAYLOAD:", content)
        self.assertIn(full_payload, content)
        self.assertIn("🤖 [2] RAW RESPONSE FROM LLM:", content)
        self.assertIn(full_raw, content)
        self.assertIn("💬 [3] CLEAN TEXT SENT TO USER:", content)
        self.assertIn(full_clean, content)

    # 18. Missing Request Payload Representation ("не зафиксирован")
    async def test_missing_request_payload_rendered_as_not_captured(self):
        async with async_session_maker() as session:
            log_entry = AILog(
                id=61,
                user_id=1,
                platform="telegram",
                provider="Gemini",
                model="gemini-flash",
                prompt_summary="legacy prompt summary that should not be used",
                request_payload=None,
                raw_response="hello",
                created_at=datetime.utcnow(),
            )
            session.add(log_entry)
            await session.commit()

        client = _create_mock_client()

        # In detail view
        await admin_ai_logs.show_ai_log_detail(client, chat_id=123, log_id=61)
        text = client.send_message.await_args.kwargs["text"]
        self.assertIn("<code>не зафиксирован</code>", text)
        self.assertNotIn("legacy prompt summary", text)

        # In TXT download
        await admin_ai_logs.download_ai_log_file(client, chat_id=123, log_id=61)
        txt = client.send_text_file.await_args.kwargs["content"]
        self.assertIn("📤 [1] FULL REQUEST PAYLOAD:\n----------------------------------------\nне зафиксирован", txt)
        self.assertNotIn("legacy prompt summary", txt)

    # 19. Direct-Filesystem ZIP Export (Creation & Cleanup on Success)
    async def test_export_ai_logs_package_creates_direct_filesystem_zip_and_cleans_up(self):
        created_tmp_paths = []
        original_named_temp = admin_ai_logs.tempfile.NamedTemporaryFile

        def spy_named_temp(*args, **kwargs):
            res = original_named_temp(*args, **kwargs)
            created_tmp_paths.append(res.name)
            return res

        async with async_session_maker() as session:
            session.add(AILog(id=101, user_id=777, raw_response="R", platform="telegram", provider="OpenAI", model="gpt-4o", created_at=datetime.utcnow()))
            session.add(AILog(id=102, user_id=MAX_ID_OFFSET + 888, raw_response="R", platform="max", provider="Gemini", model="gemini", created_at=datetime.utcnow()))
            await session.commit()

        client = _create_mock_client()

        with patch.object(admin_ai_logs.tempfile, "NamedTemporaryFile", side_effect=spy_named_temp):
            await admin_ai_logs.export_ai_logs_package(client, chat_id=123, period="all", request_type="all")

        client.upload_file.assert_awaited_once()
        self.assertEqual(client.upload_file.await_args.args[0], "file")
        uploaded_path = client.upload_file.await_args.args[1]

        client.send_media_attachment.assert_awaited_once()
        self.assertEqual(client.send_media_attachment.await_args.kwargs["media_type"], "file")
        self.assertEqual(client.send_media_attachment.await_args.kwargs["token"], "fake_file_token_123")

        # Verify temp file was cleaned up on disk
        for path in created_tmp_paths:
            self.assertFalse(os.path.exists(path))
        self.assertFalse(os.path.exists(uploaded_path))

    # 20. ZIP Manifest Platform Identity (No offset leakage)
    async def test_zip_manifest_has_platform_aware_identity_without_offset_leakage(self):
        captured_manifest = []

        async with async_session_maker() as session:
            session.add(AILog(
                id=201,
                user_id=MAX_ID_OFFSET + 100018792559,
                platform="max",
                provider="Deepseek",
                model="deepseek-v4-flash",
                raw_response="R",
                latency_ms=1400,
                created_at=datetime(2026, 9, 6, 12, 0, 0),
            ))
            session.add(AILog(
                id=202,
                user_id=999,
                platform="telegram",
                provider="OpenAI",
                model="gpt-4o-mini",
                raw_response="R",
                latency_ms=800,
                created_at=datetime(2026, 9, 6, 12, 5, 0),
            ))
            await session.commit()

        client = _create_mock_client()

        async def mock_upload(media_type, file_path):
            with zipfile.ZipFile(file_path, "r") as archive:
                manifest_data = json.loads(archive.read("manifest.json").decode("utf-8"))
                captured_manifest.extend(manifest_data)
            return {"token": "valid_token"}

        client.upload_file = AsyncMock(side_effect=mock_upload)

        await admin_ai_logs.export_ai_logs_package(client, chat_id=123, period="all", request_type="all")

        self.assertEqual(len(captured_manifest), 2)
        max_entry = next(m for m in captured_manifest if m["id"] == 201)
        tg_entry = next(m for m in captured_manifest if m["id"] == 202)

        self.assertEqual(max_entry["platform"], "MAX")
        self.assertEqual(max_entry["max_id"], 100018792559)
        self.assertNotIn("telegram_id", max_entry)
        self.assertNotIn(str(MAX_ID_OFFSET + 100018792559), json.dumps(max_entry))

        self.assertEqual(tg_entry["platform"], "Telegram")
        self.assertEqual(tg_entry["telegram_id"], 999)
        self.assertNotIn("max_id", tg_entry)

    # 21. ZIP Export Record Count Limit (500 records)
    async def test_zip_export_record_count_limit_with_transparent_caption(self):
        async with async_session_maker() as session:
            for i in range(1, 15):
                session.add(AILog(id=i, user_id=1, raw_response="R", provider="P", model="M", created_at=datetime.utcnow() - timedelta(minutes=i)))
            await session.commit()

        client = _create_mock_client()

        with patch.object(admin_ai_logs, "MAX_EXPORT_LOGS_LIMIT", 5):
            await admin_ai_logs.export_ai_logs_package(client, chat_id=123, period="all", request_type="all")

        caption = client.send_media_attachment.await_args.kwargs["caption"]
        self.assertIn("лимит экспорта по количеству: 5 из 14", caption)

    # 22. ZIP Export Byte Size Limit (20 MB)
    async def test_zip_export_byte_size_limit_with_transparent_caption(self):
        async with async_session_maker() as session:
            for i in range(1, 10):
                session.add(AILog(
                    id=i,
                    user_id=1,
                    provider="P",
                    model="M",
                    request_payload="X" * 1000,
                    raw_response="Y" * 1000,
                    created_at=datetime.utcnow() - timedelta(minutes=i),
                ))
            await session.commit()

        client = _create_mock_client()

        with patch.object(admin_ai_logs, "MAX_EXPORT_UNCOMPRESSED_BYTES", 3500):
            await admin_ai_logs.export_ai_logs_package(client, chat_id=123, period="all", request_type="all")

        caption = client.send_media_attachment.await_args.kwargs["caption"]
        self.assertIn("лимит экспорта по объёму данных:", caption)
        self.assertIn("из 9", caption)

    # 23. Temp File Cleanup on Upload and Send Failures
    async def test_temp_file_cleanup_on_upload_failure(self):
        created_tmp_paths = []
        original_named_temp = admin_ai_logs.tempfile.NamedTemporaryFile

        def spy_named_temp(*args, **kwargs):
            res = original_named_temp(*args, **kwargs)
            created_tmp_paths.append(res.name)
            return res

        async with async_session_maker() as session:
            session.add(AILog(id=1, user_id=1, raw_response="R", provider="P", model="M", created_at=datetime.utcnow()))
            await session.commit()

        client = _create_mock_client()
        client.upload_file = AsyncMock(side_effect=Exception("MAX Upload Network Error"))

        with patch.object(admin_ai_logs.tempfile, "NamedTemporaryFile", side_effect=spy_named_temp):
            await admin_ai_logs.export_ai_logs_package(client, chat_id=123, period="all", request_type="all")

        client.send_message.assert_awaited_once()
        self.assertIn("Не удалось отправить архив логов: MAX Upload Network Error", client.send_message.await_args.kwargs["text"])

        for path in created_tmp_paths:
            self.assertFalse(os.path.exists(path))

    async def test_temp_file_cleanup_on_send_failure(self):
        created_tmp_paths = []
        original_named_temp = admin_ai_logs.tempfile.NamedTemporaryFile

        def spy_named_temp(*args, **kwargs):
            res = original_named_temp(*args, **kwargs)
            created_tmp_paths.append(res.name)
            return res

        async with async_session_maker() as session:
            session.add(AILog(id=1, user_id=1, raw_response="R", provider="P", model="M", created_at=datetime.utcnow()))
            await session.commit()

        client = _create_mock_client()
        client.send_media_attachment = AsyncMock(side_effect=Exception("MAX Send Media Error"))

        with patch.object(admin_ai_logs.tempfile, "NamedTemporaryFile", side_effect=spy_named_temp):
            await admin_ai_logs.export_ai_logs_package(client, chat_id=123, period="all", request_type="all")

        client.send_message.assert_awaited_once()
        self.assertIn("Не удалось отправить архив логов: MAX Send Media Error", client.send_message.await_args.kwargs["text"])

        for path in created_tmp_paths:
            self.assertFalse(os.path.exists(path))

    # 24. Missing Upload Token Failure Handling
    async def test_missing_upload_token_handled_safely(self):
        async with async_session_maker() as session:
            session.add(AILog(id=1, user_id=1, raw_response="R", provider="P", model="M", created_at=datetime.utcnow()))
            await session.commit()

        client = _create_mock_client()
        client.upload_file = AsyncMock(return_value={})  # Missing token

        await admin_ai_logs.export_ai_logs_package(client, chat_id=123, period="all", request_type="all")

        client.send_message.assert_awaited_once()
        self.assertIn("Не удалось отправить архив логов:", client.send_message.await_args.kwargs["text"])

    # 25. Malformed Callbacks Handled Safely Without Worker Crash
    async def test_malformed_callbacks_do_not_crash_worker(self):
        with patch("max_messenger_bot.services.common.is_admin", AsyncMock(return_value=True)):
            app = MaxBotApplication.__new__(MaxBotApplication)
            app.client = _create_mock_client()
            app.states = SimpleNamespace(get=AsyncMock(return_value=None), clear=AsyncMock())

            malformed_payloads = [
                "admin_ai_log_file_notanumber",
                "admin_ai_log_notanumber",
                "admin_user_ai_logs_bad_page_notint",
                "admin_ai_logs_bad_page",
                "export_ai_logs_bad_user",
            ]

            for payload in malformed_payloads:
                cb = IncomingCallback(
                    raw={},
                    callback_id="cb_malformed",
                    payload=payload,
                    sender=_create_sender(1001, "Admin"),
                    chat_id=12345,
                    message_id="msg_1",
                )
                await app.handle_callback(cb)

            self.assertEqual(app.client.answer_callback.call_count, len(malformed_payloads))
