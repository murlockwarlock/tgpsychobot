import unittest
import json

from user_metadata import (
    append_metadata_records,
    build_metadata_context,
    extend_system_prompt_with_metadata,
    extract_data_blocks,
    load_metadata_records,
)


class UserMetadataTests(unittest.TestCase):
    def test_saved_records_are_rendered_as_hidden_model_context(self):
        raw = append_metadata_records(
            None,
            [{"data": {"stage": "test"}, "raw_json": '{"stage":"test"}'}],
            saved_at="2026-07-19T10:00:00+00:00",
        )

        context = build_metadata_context(raw)

        self.assertIn("СЛУЖЕБНАЯ ИСТОРИЯ МЕТАДАННЫХ", context)
        self.assertIn("2026-07-19T10:00:00+00:00", context)
        self.assertIn('[DATA]\n{"stage":"test"}\n[/DATA]', context)
        self.assertIn("сформируй его один раз", context)

    def test_metadata_context_keeps_only_latest_records(self):
        raw = None
        for value in range(3):
            raw = append_metadata_records(
                raw,
                [{"data": {"value": value}, "raw_json": json.dumps({"value": value})}],
                saved_at=f"2026-07-19T10:0{value}:00+00:00",
            )

        context = build_metadata_context(raw, record_limit=2)

        self.assertIn("не включённых в контекст: 1", context)
        self.assertNotIn('{"value":0}', context)
        self.assertIn('{"value":1}', context)
        self.assertIn('{"value":2}', context)

    def test_metadata_context_includes_all_records_by_default(self):
        raw = None
        for value in range(3):
            raw = append_metadata_records(
                raw,
                [{"data": {"value": value}, "raw_json": json.dumps({"value": value})}],
                saved_at=f"2026-07-19T10:0{value}:00+00:00",
            )

        context = build_metadata_context(raw)

        self.assertNotIn("не включённых в контекст", context)
        self.assertLess(context.index('{"value":0}'), context.index('{"value":1}'))
        self.assertLess(context.index('{"value":1}'), context.index('{"value":2}'))

    def test_empty_metadata_adds_no_context(self):
        self.assertEqual(build_metadata_context(None), "")
        self.assertEqual(build_metadata_context("{}"), "")
        self.assertEqual(extend_system_prompt_with_metadata("Основной промпт", "{}"), "Основной промпт")

    def test_metadata_context_is_appended_to_existing_system_prompt(self):
        raw = append_metadata_records(
            None,
            [{"data": {"stage": "test"}, "raw_json": '{"stage":"test"}'}],
            saved_at="2026-07-19T10:00:00+00:00",
        )

        prompt = extend_system_prompt_with_metadata("Основной промпт", raw)

        self.assertTrue(prompt.startswith("Основной промпт\n\n"))
        self.assertIn('[DATA]\n{"stage":"test"}\n[/DATA]', prompt)

    def test_extracts_json_and_keeps_only_visible_answer(self):
        visible, blocks, invalid = extract_data_blocks(
            "Твой результат готов.\n[DATA]\n{\"profile\": {\"name\": \"Максим\"}, \"score\": 7}\n[/DATA]"
        )

        self.assertEqual(visible, "Твой результат готов.")
        self.assertEqual(blocks[0]["data"], {"profile": {"name": "Максим"}, "score": 7})
        self.assertEqual(blocks[0]["raw_json"], '{"profile": {"name": "Максим"}, "score": 7}')
        self.assertEqual(invalid, 0)

    def test_hides_invalid_data_block(self):
        visible, blocks, invalid = extract_data_blocks("Ответ\n[DATA]{not json}[/DATA]\nПродолжение")

        self.assertEqual(visible, "Ответ\n\nПродолжение")
        self.assertEqual(blocks, [])
        self.assertEqual(invalid, 1)

    def test_loads_old_object_as_one_legacy_record(self):
        self.assertEqual(load_metadata_records('["not", "an", "object"]'), [])
        records = load_metadata_records('{"saved": true}')
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0]["saved_at"])
        self.assertEqual(records[0]["data"], {"saved": True})

    def test_futuro_analyzer_response_keeps_reveal_and_hides_full_profile(self):
        response = """Ты — Штурман: тебе важно самому выбирать направление. Твоя сила — идти своим путём.

[DATA]
{
  "profile": {
    "name": "Максим", "age": 14, "city": "Казань",
    "interests": ["Игры", "Программирование/техника"],
    "has_friends_by_interest": "Пара человек"
  },
  "driver_scores": {
    "autonomy": 13, "mastery": 0, "belonging": 0,
    "recognition": 0, "exploration": 6, "meaning": 0
  },
  "dominant_driver": "autonomy",
  "secondary_driver": "exploration",
  "is_hybrid": false,
  "archetype": "Штурман",
  "confidence": "high",
  "free_answer_notes": "Ценит свободу и новые проекты.",
  "attention_flag": false
}
[/DATA]"""

        visible, blocks, invalid = extract_data_blocks(response)

        self.assertEqual(
            visible,
            "Ты — Штурман: тебе важно самому выбирать направление. Твоя сила — идти своим путём.",
        )
        self.assertEqual(invalid, 0)
        metadata = blocks[0]["data"]
        self.assertEqual(metadata["profile"]["name"], "Максим")
        self.assertEqual(metadata["driver_scores"]["autonomy"], 13)
        self.assertEqual(metadata["archetype"], "Штурман")
        self.assertFalse(metadata["attention_flag"])

    def test_follow_up_block_is_appended_without_merging(self):
        saved = {
            "profile": {"name": "Максим", "age": 14, "city": "Казань"},
            "driver_scores": {"autonomy": 13},
            "confidence": "high",
        }
        _, blocks, invalid = extract_data_blocks(
            "Хорошо. [DATA]{\"profile\": {\"city\": \"Самара\"}, \"attention_flag\": true}[/DATA]"
        )

        self.assertEqual(invalid, 0)
        stored = append_metadata_records(
            json.dumps(saved, ensure_ascii=False),
            blocks,
            saved_at="2026-07-19T10:00:00+00:00",
        )
        records = load_metadata_records(stored)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["data"], saved)
        self.assertIsNone(records[0]["saved_at"])
        self.assertEqual(records[1]["data"], {
            "profile": {"city": "Самара"},
            "attention_flag": True,
        })
        self.assertEqual(records[1]["saved_at"], "2026-07-19T10:00:00+00:00")

    def test_multiple_blocks_keep_source_order(self):
        _, blocks, invalid = extract_data_blocks(
            '[DATA]{"test": 1}[/DATA]\n[DATA]{"test": 2}[/DATA]'
        )
        stored = append_metadata_records(None, blocks, saved_at="2026-07-19T10:00:00+00:00")
        records = load_metadata_records(stored)

        self.assertEqual(invalid, 0)
        self.assertEqual([record["data"]["test"] for record in records], [1, 2])
