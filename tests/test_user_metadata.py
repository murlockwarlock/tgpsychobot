import unittest
import json

from user_metadata import (
    append_metadata_records,
    extract_data_blocks,
    extract_service_data,
    load_metadata_records,
)


class UserMetadataTests(unittest.TestCase):

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

    def test_unified_xml_envelope_is_parsed_atomically(self):
        visible, blocks, invalid = extract_service_data("""Ответ пользователю.
<DATA>
{
  "current_state": {"current_step": "STAGE_1_HOBBY", "attempt": 2},
  "events": ["HOBBY_RECEIVED", {"name": "LEAD_UPDATED"}],
  "save_mode": "snapshot",
  "metadata": {"profile": {"hobby": "игры"}}
}
</DATA>""")

        self.assertEqual(visible, "Ответ пользователю.")
        self.assertEqual(invalid, 0)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].current_state["current_step"], "STAGE_1_HOBBY")
        self.assertEqual(blocks[0].events, ["HOBBY_RECEIVED", "LEAD_UPDATED"])
        self.assertEqual(blocks[0].save_mode, "snapshot")
        self.assertEqual(blocks[0].metadata, {"profile": {"hobby": "игры"}})
        self.assertFalse(blocks[0].legacy)

    def test_legacy_events_field_is_not_executed(self):
        _, blocks, invalid = extract_service_data(
            '[DATA]{"events": ["this is ordinary legacy metadata"], "score": 5}[/DATA]'
        )

        self.assertEqual(invalid, 0)
        self.assertTrue(blocks[0].legacy)
        self.assertEqual(blocks[0].events, [])
        self.assertEqual(blocks[0].metadata["score"], 5)

    def test_markdown_fence_around_xml_data_is_hidden(self):
        visible, blocks, invalid = extract_service_data(
            'Готово\n```json\n<DATA>{"current_state":{"current_step":"DONE"}}</DATA>\n```'
        )

        self.assertEqual(visible, "Готово")
        self.assertEqual(invalid, 0)
        self.assertEqual(blocks[0].current_state["current_step"], "DONE")

    def test_visible_code_fence_is_preserved(self):
        visible, blocks, invalid = extract_service_data(
            'Пример:\n```python\nprint("ok")\n```\n\n<DATA>{"events":["CODE_SHOWN"]}</DATA>'
        )

        self.assertEqual(visible, 'Пример:\n```python\nprint("ok")\n```')
        self.assertEqual(blocks[0].events, ["CODE_SHOWN"])
        self.assertEqual(invalid, 0)
