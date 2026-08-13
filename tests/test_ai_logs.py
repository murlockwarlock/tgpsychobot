import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from database import AILog, User, async_session_maker, engine, init_db
from keyboards import (
    admin_ai_log_detail_keyboard,
    admin_ai_logs_keyboard,
    admin_panel_keyboard,
    ai_settings_keyboard,
    fixed_pagination_rows,
)


class AILogFeatureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await init_db()

    async def test_ai_log_creation_and_retrieval(self):
        async with async_session_maker() as session:
            user = User(id=777, first_name="Тест")
            session.add(user)
            log = AILog(
                user_id=777,
                provider="KIE",
                model="gemini-3-flash",
                prompt_summary="Какой типаж?",
                request_payload='{"payload":{"temperature":0.1,"messages":[]}}',
                raw_response="[Дальше](btn:after_photo)\n<DATA>{}</DATA>",
                clean_text="Дальше",
                latency_ms=1200,
            )
            session.add(log)
            await session.commit()

            fetched = await session.get(AILog, log.id)
            self.assertIsNotNone(fetched)
            self.assertEqual(fetched.provider, "KIE")
            self.assertEqual(fetched.model, "gemini-3-flash")
            self.assertEqual(fetched.latency_ms, 1200)
            self.assertIn('"temperature":0.1', fetched.request_payload)

    def test_ai_log_keyboards_structure(self):
        fake_log = AILog(id=1, provider="Deepseek", model="deepseek-chat", latency_ms=850)
        markup = admin_ai_logs_keyboard([fake_log], page=0, total_pages=1)
        buttons = [b.text for row in markup.inline_keyboard for b in row]
        self.assertTrue(any("#1 | Deepseek: deepseek-chat" in btn for btn in buttons))

        detail_markup = admin_ai_log_detail_keyboard(1, page=0)
        detail_buttons = [b.text for row in detail_markup.inline_keyboard for b in row]
        self.assertIn("📄 Скачать .txt файл лога", detail_buttons)

    def test_log_navigation_filters_export_and_settings_location(self):
        fake_log = AILog(id=9, provider="KIE", model="gemini", latency_ms=100)
        markup = admin_ai_logs_keyboard([fake_log], page=2, total_pages=5, filter_user_id=777, period="7d")
        pairs = [(button.text, button.callback_data) for row in markup.inline_keyboard for button in row]
        self.assertIn(("⏮ В начало", "admin_user_ai_logs_777_0_7d"), pairs)
        self.assertIn(("В конец ⏭", "admin_user_ai_logs_777_4_7d"), pairs)
        self.assertIn(("📦 Скачать пакет логов", "export_ai_logs_777_7d"), pairs)
        self.assertIn(("⬅️ Назад", "view_client_777"), pairs)
        self.assertTrue(any(text == "✅ 7 дней" for text, _ in pairs))

        ai_settings_buttons = [button.text for row in ai_settings_keyboard("KIE").inline_keyboard for button in row]
        admin_panel_buttons = [button.text for row in admin_panel_keyboard().inline_keyboard for button in row]
        self.assertIn("📜 Логи запросов ИИ", ai_settings_buttons)
        self.assertNotIn("📜 Логи ИИ", admin_panel_buttons)

    def test_followup_log_filter_is_available(self):
        markup = admin_ai_logs_keyboard(
            [AILog(id=10, request_type="followup", provider="Gemini", model="flash", latency_ms=100)],
            page=0,
            total_pages=1,
            period="7d",
            request_type="followup",
        )
        pairs = [(button.text, button.callback_data) for row in markup.inline_keyboard for button in row]
        self.assertIn(("#10 | Gemini: flash (0.1s)", "admin_ai_log_10_0_0_7d_followup"), pairs)
        self.assertIn(("✅ Догоняющие", "admin_ai_logs_0_7d_followup"), pairs)
        self.assertIn(("Сегодня", "admin_ai_logs_0_today_followup"), pairs)
        self.assertIn(("📦 Скачать пакет логов", "export_ai_logs_0_7d_followup"), pairs)

    def test_fixed_navigation_has_same_two_rows_on_every_page(self):
        callback = lambda page: f"page_{page}"
        first_page = fixed_pagination_rows(0, 5, callback)
        middle_page = fixed_pagination_rows(2, 5, callback)
        last_page = fixed_pagination_rows(4, 5, callback)

        for rows in (first_page, middle_page, last_page):
            self.assertEqual([len(row) for row in rows], [3, 2])
            self.assertEqual([button.text for button in rows[0]][::2], ["⬅️ Назад", "Вперёд ➡️"])
            self.assertEqual([button.text for button in rows[1]], ["⏮ В начало", "В конец ⏭"])
        self.assertEqual(first_page[0][0].callback_data, "page_4")
        self.assertEqual(last_page[0][2].callback_data, "page_0")
