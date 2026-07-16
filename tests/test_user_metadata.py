import unittest

from user_metadata import extract_data_blocks, load_metadata, merge_metadata


class UserMetadataTests(unittest.TestCase):
    def test_extracts_json_and_keeps_only_visible_answer(self):
        visible, metadata, invalid = extract_data_blocks(
            "Твой результат готов.\n[DATA]\n{\"profile\": {\"name\": \"Максим\"}, \"score\": 7}\n[/DATA]"
        )

        self.assertEqual(visible, "Твой результат готов.")
        self.assertEqual(metadata, {"profile": {"name": "Максим"}, "score": 7})
        self.assertEqual(invalid, 0)

    def test_hides_invalid_data_block(self):
        visible, metadata, invalid = extract_data_blocks("Ответ\n[DATA]{not json}[/DATA]\nПродолжение")

        self.assertEqual(visible, "Ответ\n\nПродолжение")
        self.assertEqual(metadata, {})
        self.assertEqual(invalid, 1)

    def test_merges_nested_metadata_without_losing_other_fields(self):
        previous = {"profile": {"name": "Аня", "city": "Томск"}, "old_flag": True}
        incoming = {"profile": {"age": 15, "city": "Омск"}, "drivers": {"autonomy": 4}}

        self.assertEqual(
            merge_metadata(previous, incoming),
            {
                "profile": {"name": "Аня", "city": "Омск", "age": 15},
                "old_flag": True,
                "drivers": {"autonomy": 4},
            },
        )

    def test_loads_only_json_objects(self):
        self.assertEqual(load_metadata('["not", "an", "object"]'), {})
        self.assertEqual(load_metadata('{"saved": true}'), {"saved": True})

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

        visible, metadata, invalid = extract_data_blocks(response)

        self.assertEqual(
            visible,
            "Ты — Штурман: тебе важно самому выбирать направление. Твоя сила — идти своим путём.",
        )
        self.assertEqual(invalid, 0)
        self.assertEqual(metadata["profile"]["name"], "Максим")
        self.assertEqual(metadata["driver_scores"]["autonomy"], 13)
        self.assertEqual(metadata["archetype"], "Штурман")
        self.assertFalse(metadata["attention_flag"])

    def test_follow_up_prompt_can_update_only_part_of_profile(self):
        saved = {
            "profile": {"name": "Максим", "age": 14, "city": "Казань"},
            "driver_scores": {"autonomy": 13},
            "confidence": "high",
        }
        _, update, invalid = extract_data_blocks(
            "Хорошо. [DATA]{\"profile\": {\"city\": \"Самара\"}, \"attention_flag\": true}[/DATA]"
        )

        self.assertEqual(invalid, 0)
        self.assertEqual(
            merge_metadata(saved, update),
            {
                "profile": {"name": "Максим", "age": 14, "city": "Самара"},
                "driver_scores": {"autonomy": 13},
                "confidence": "high",
                "attention_flag": True,
            },
        )
