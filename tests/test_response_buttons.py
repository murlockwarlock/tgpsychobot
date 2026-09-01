import os
import unittest

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from response_buttons import extract_response_buttons, extract_test_start_directive
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

    def test_accepts_cyrillic_multiword_action(self):
        source = "[Готова. Начинаем](btn:я готова)"
        text, rows = extract_response_buttons(source)
        self.assertEqual(text, "")
        self.assertEqual(rows[0][0].value, "я готова")

    def test_keeps_empty_action_visible(self):
        source = "[Кнопка](btn:)"
        text, rows = extract_response_buttons(source)
        self.assertEqual(text, source)
        self.assertEqual(rows, [])

    def test_rejects_action_over_telegram_callback_byte_limit(self):
        source = f"[Кнопка](btn:{'я' * 29})"
        text, rows = extract_response_buttons(source)
        self.assertEqual(text, source)
        self.assertEqual(rows, [])

    def test_denis_mixed_layout_creates_rows_from_whitespace(self):
        source = (
            "[Спереди](btn:front_shoulder) | [Сбоку](btn:side_shoulder) "
            "[Одинаково](btn:both_shoulder)"
        )
        text, rows = extract_response_buttons(source)
        self.assertEqual(text, "")
        self.assertEqual(
            [[button.text for button in row] for row in rows],
            [["Спереди", "Сбоку"], ["Одинаково"]],
        )

    def test_single_space_between_buttons_creates_next_row(self):
        text, rows = extract_response_buttons("[Да](btn:yes) [Нет](btn:no)")
        self.assertEqual(text, "")
        self.assertEqual([[button.text for button in row] for row in rows], [["Да"], ["Нет"]])

    def test_multiple_spaces_between_buttons_create_next_row(self):
        text, rows = extract_response_buttons("[Да](btn:yes)    [Нет](btn:no)")
        self.assertEqual(text, "")
        self.assertEqual([[button.text for button in row] for row in rows], [["Да"], ["Нет"]])

    def test_pipe_between_buttons_keeps_same_row(self):
        text, rows = extract_response_buttons("[Да](btn:yes) | [Нет](btn:no)")
        self.assertEqual(text, "")
        self.assertEqual([[button.text for button in row] for row in rows], [["Да", "Нет"]])

    def test_mixed_whitespace_and_pipe_layout_keeps_canonical_rows(self):
        text, rows = extract_response_buttons("[A](btn:a) [B](btn:b) | [C](btn:c)")
        self.assertEqual(text, "")
        self.assertEqual([[button.text for button in row] for row in rows], [["A"], ["B", "C"]])

    def test_spaces_inside_labels_and_actions_are_preserved(self):
        source = "[Первый вариант](btn:first choice) | [Второй вариант](btn:second choice)"
        text, rows = extract_response_buttons(source)
        self.assertEqual(text, "")
        self.assertEqual(
            [(button.text, button.value) for button in rows[0]],
            [("Первый вариант", "first choice"), ("Второй вариант", "second choice")],
        )

    def test_url_buttons_support_whitespace_row_separators(self):
        source = (
            "[Документ](https://example.com/path_(version)) "
            "[Сайт](https://example.org)"
        )
        text, rows = extract_response_buttons(source)
        self.assertEqual(text, "")
        self.assertEqual(
            [(button.text, button.kind, button.value) for button in rows[0]],
            [("Документ", "url", "https://example.com/path_(version)")],
        )
        self.assertEqual(
            [(button.text, button.kind, button.value) for button in rows[1]],
            [("Сайт", "url", "https://example.org")],
        )

    def test_malformed_button_line_is_kept_without_partial_extraction(self):
        source = "[Да](btn:yes) [Сломано](btn:) [Нет](btn:no)"
        text, rows = extract_response_buttons(source)
        self.assertEqual(text, source)
        self.assertEqual(rows, [])

    def test_plain_and_bracket_buttons_remain_compatible(self):
        text, rows = extract_response_buttons("[Дальше] | готово")
        self.assertEqual(text, "")
        self.assertEqual(
            [(button.text, button.kind, button.value) for button in rows[0]],
            [("Дальше", "action", "Дальше"), ("готово", "action", "готово")],
        )

    def test_button_row_and_total_row_limits_are_preserved(self):
        too_many_buttons = " | ".join(f"[{index}](btn:button_{index})" for index in range(9))
        text, rows = extract_response_buttons(too_many_buttons)
        self.assertEqual(text, too_many_buttons)
        self.assertEqual(rows, [])

        too_many_inline_rows = " ".join(f"[{index}](btn:button_{index})" for index in range(21))
        text, rows = extract_response_buttons(too_many_inline_rows)
        self.assertEqual(text, too_many_inline_rows)
        self.assertEqual(rows, [])

        too_many_lines = "\n".join(f"[{index}](btn:button_{index})" for index in range(21))
        text, rows = extract_response_buttons(too_many_lines)
        self.assertEqual(text, "[20](btn:button_20)")
        self.assertEqual(len(rows), 20)

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

    def test_start_test_button_is_not_executed_as_directive(self):
        source = (
            "Проверка кнопок:\n\n"
            "[YouTube](https://www.youtube.com/) | [Пройти тест](btn:start_test)"
        )

        should_start_test, clean_text = extract_test_start_directive(source)
        visible_text, rows = extract_response_buttons(clean_text)

        self.assertFalse(should_start_test)
        self.assertEqual(visible_text, "Проверка кнопок:")
        self.assertEqual([button.text for button in rows[0]], ["YouTube", "Пройти тест"])

    def test_standalone_start_test_directive_still_works(self):
        should_start_test, clean_text = extract_test_start_directive(
            "Можно начинать.\n\n[START_TEST]"
        )

        self.assertTrue(should_start_test)
        self.assertEqual(clean_text, "Можно начинать.")

    def test_extracts_buttons_wrapped_in_bold_or_code_markdown(self):
        source = "Текст сообщения.\n\n**[Дальше](btn:after_photo)**\n`[Готов](btn:ready)`\n[**Дальше 2**](btn:after_photo_2)"
        text, rows = extract_response_buttons(source)
        self.assertEqual(text, "Текст сообщения.")
        self.assertEqual(len(rows), 3)
        self.assertEqual((rows[0][0].text, rows[0][0].value), ("Дальше", "after_photo"))
        self.assertEqual((rows[1][0].text, rows[1][0].value), ("Готов", "ready"))
        self.assertEqual((rows[2][0].text, rows[2][0].value), ("Дальше 2", "after_photo_2"))

    def test_unescapes_literal_newlines_in_text(self):
        source = r"Первая строка.\n\nВторая строка.\n\n[Готов](btn:ready)"
        text, rows = extract_response_buttons(source)
        self.assertEqual(text, "Первая строка.\n\nВторая строка.")
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0][0].text, rows[0][0].value), ("Готов", "ready"))


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

    def test_fallback_status_is_visible_in_telegram_admin_keyboard(self):
        common = {
            "current_transcription_provider": "OpenAI",
            "context_first": 2,
            "context_recent": 10,
            "current_vision_provider": "Gemini",
            "current_vision_model": "vision-model",
            "image_generation_provider": "Gemini",
            "image_generation_model": "image-model",
            "image_edit_provider": "Gemini",
            "image_edit_model": "edit-model",
            "kie_credit_alert_threshold": 0,
            "fallback_provider": "Deepseek",
            "fallback_model": "deepseek-v4-flash",
        }

        disabled = ai_keys_models_keyboard(**common, allow_fallback=False)
        enabled = ai_keys_models_keyboard(**common, allow_fallback=True)

        disabled_labels = [button.text for row in disabled.inline_keyboard for button in row]
        enabled_labels = [button.text for row in enabled.inline_keyboard for button in row]
        self.assertIn("🔄 Резерв: Deepseek — ❌ ВЫКЛ", disabled_labels)
        self.assertIn("🔄 Резерв: Deepseek — ✅ ВКЛ", enabled_labels)


class RemoveMarkdownTests(unittest.TestCase):
    def test_remove_markdown_preserves_all_buttons_in_row(self):
        from handlers import remove_markdown
        from max_messenger_bot.services.admin_clients import _remove_markdown

        source = (
            "Чем любишь заниматься?\n\n"
            "[1](btn:interest_1) | [2](btn:interest_2) | [3](btn:interest_3) | "
            "[4](btn:interest_4) | [5](btn:interest_5) | [6](btn:interest_6)\n"
            "[7](btn:interest_7) | [8](btn:interest_8) | [9](btn:interest_9) | "
            "[10](btn:interest_10) | [11](btn:interest_11) | [12](btn:interest_12)"
        )
        expected = (
            "Чем любишь заниматься?\n\n"
            "1 | 2 | 3 | 4 | 5 | 6\n"
            "7 | 8 | 9 | 10 | 11 | 12"
        )
        self.assertEqual(remove_markdown(source), expected)
        self.assertEqual(_remove_markdown(source), expected)

    def test_remove_markdown_handles_duels_and_levels(self):
        from handlers import remove_markdown
        from max_messenger_bot.services.admin_clients import _remove_markdown

        duels = "[1](btn:duel_1) | [2](btn:duel_2)"
        levels = "[1](btn:level_1) | [2](btn:level_2) | [3](btn:level_3) | [4](btn:level_4)"

        self.assertEqual(remove_markdown(duels), "1 | 2")
        self.assertEqual(_remove_markdown(duels), "1 | 2")
        self.assertEqual(remove_markdown(levels), "1 | 2 | 3 | 4")
        self.assertEqual(_remove_markdown(levels), "1 | 2 | 3 | 4")

    def test_remove_markdown_preserves_formatting_and_links(self):
        from handlers import remove_markdown
        from max_messenger_bot.services.admin_clients import _remove_markdown

        text = "**Жирный** и _курсив_, а также `код` и [Ссылка](https://example.com)"
        expected = "Жирный и курсив, а также код и Ссылка"
        self.assertEqual(remove_markdown(text), expected)
        self.assertEqual(_remove_markdown(text), expected)
