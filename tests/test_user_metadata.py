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
