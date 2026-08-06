import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from database import AILog, User, async_session_maker, engine, init_db
from keyboards import admin_ai_log_detail_keyboard, admin_ai_logs_keyboard


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

    def test_ai_log_keyboards_structure(self):
        fake_log = AILog(id=1, provider="Deepseek", model="deepseek-chat", latency_ms=850)
        markup = admin_ai_logs_keyboard([fake_log], page=0, total_pages=1)
        buttons = [b.text for row in markup.inline_keyboard for b in row]
        self.assertTrue(any("#1 | Deepseek: deepseek-chat" in btn for btn in buttons))

        detail_markup = admin_ai_log_detail_keyboard(1, page=0)
        detail_buttons = [b.text for row in detail_markup.inline_keyboard for b in row]
        self.assertIn("📄 Скачать .txt файл лога", detail_buttons)
