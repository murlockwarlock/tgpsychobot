import os
import unittest

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from response_buttons import extract_response_buttons
from keyboards import ai_keys_models_keyboard, mask_api_key


class ResponseButtonsTests(unittest.TestCase):
    def test_extracts_links_and_actions_with_rows(self):
        text, rows = extract_response_buttons(
            "Выберите действие:\n"
            "[Да](btn:yes) | [Нет](btn:no)\n"
            "[Смотреть](https://youtube.com/watch?v=123)"
        )

        self.assertEqual(text, "Выберите действие:")
        self.assertEqual([(b.text, b.kind, b.value) for b in rows[0]], [
            ("Да", "action", "yes"),
            ("Нет", "action", "no"),
        ])
        self.assertEqual(rows[1][0].kind, "url")
        self.assertEqual(rows[1][0].value, "https://youtube.com/watch?v=123")

    def test_keeps_regular_markdown_link_inside_text(self):
        source = "Посмотрите [наш сайт](https://example.com) и возвращайтесь."
        text, rows = extract_response_buttons(source)
        self.assertEqual(text, source)
        self.assertEqual(rows, [])

    def test_keeps_malformed_action_visible(self):
        source = "[Кнопка](btn:код с пробелом)"
        text, rows = extract_response_buttons(source)
        self.assertEqual(text, source)
        self.assertEqual(rows, [])

    def test_requires_pipe_between_buttons_in_same_row(self):
        source = "[Да](btn:yes) [Нет](btn:no)"
        text, rows = extract_response_buttons(source)
        self.assertEqual(text, source)
        self.assertEqual(rows, [])

    def test_accepts_valid_url_with_parentheses(self):
        source = "[Документ](https://example.com/path_(version))"
        text, rows = extract_response_buttons(source)
        self.assertEqual(text, "")
        self.assertEqual(rows[0][0].value, "https://example.com/path_(version)")

    def test_rejects_non_http_link(self):
        source = "[Опасная ссылка](javascript:alert(1))"
        text, rows = extract_response_buttons(source)
        self.assertEqual(text, source)
        self.assertEqual(rows, [])


class ApiKeyDisplayTests(unittest.TestCase):
    def test_masks_api_keys_in_telegram_admin_keyboard(self):
        secret = "abcd12345678wxyz"
        self.assertEqual(mask_api_key(secret), "abcd...wxyz")

        markup = ai_keys_models_keyboard(
            current_transcription_provider="OpenAI",
            context_first=2,
            context_recent=10,
            current_vision_provider="Gemini",
            current_vision_model="vision-model",
            image_generation_provider="Gemini",
            image_generation_model="image-model",
            image_edit_provider="Gemini",
            image_edit_model="edit-model",
            kie_credit_alert_threshold=0,
            api_keys={"Deepseek": secret},
        )
        labels = [button.text for row in markup.inline_keyboard for button in row]
        self.assertIn("🔑 Deepseek: abcd...wxyz", labels)
        self.assertNotIn(secret, "\n".join(labels))
