import json

import pytest

from user_metadata import extract_service_data, extract_service_data_details, service_data_diagnostics


def _compact(text: str) -> str:
    return " ".join(text.split())


def test_literal_data_mention_does_not_hide_later_complete_block():
    raw = 'Начало. Упоминание <DATA> в тексте. <DATA>{"current_step":"ok"}</DATA> Продолжение.'

    visible, blocks, invalid = extract_service_data(raw)

    assert _compact(visible) == "Начало. Упоминание <DATA> в тексте. Продолжение."
    assert [block.current_state for block in blocks] == [{"current_step": "ok"}]
    assert invalid == 0


def test_literal_data_mention_does_not_join_with_later_block():
    raw = (
        "Раздел содержит <DATA> как обычное слово.\n"
        '<DATA>{"metadata":{"answer":42}}</DATA>\n'
        "Текст после блока."
    )

    visible, blocks, invalid = extract_service_data(raw)

    assert "Раздел содержит <DATA> как обычное слово." in visible
    assert "Текст после блока." in visible
    assert len(blocks) == 1
    assert blocks[0].metadata == {"answer": 42}
    assert invalid == 0


def test_service_json_without_tags_is_removed_and_normalized():
    raw = (
        "Текст ответа.\n"
        "Служебный блок данных:\n"
        '{"current_step":"retest","event":"","metadata":{"ws":{"ok":true}}}'
    )

    visible, blocks, invalid = extract_service_data(raw)

    assert visible == "Текст ответа."
    assert len(blocks) == 1
    assert blocks[0].current_state == {"current_step": "retest"}
    assert blocks[0].events == []
    assert blocks[0].metadata == {"ws": {"ok": True}}
    assert invalid == 0


@pytest.mark.parametrize(
    ("payload", "expected_step", "expected_events"),
    [
        ('{"current_step":"step-a"}', "step-a", []),
        ('{"current_step":"step-b","event":"EVENT_B","metadata":{"x":1}}', "step-b", ["EVENT_B"]),
    ],
)
def test_top_level_aliases_are_canonicalized(payload, expected_step, expected_events):
    visible, blocks, invalid = extract_service_data(f"Ответ.\n<DATA>{payload}</DATA>")

    assert visible == "Ответ."
    assert len(blocks) == 1
    assert blocks[0].current_state == {"current_step": expected_step}
    assert blocks[0].events == expected_events
    assert invalid == 0


def test_truncated_real_service_json_is_hidden_without_partial_block():
    raw = (
        "Человеческий ответ.\n"
        '<DATA>{"current_state":{"current_step":"secret"},"events":["SHOULD_NOT_RUN"],'
        '"metadata":{"private":"value"}'
    )

    visible, blocks, invalid = extract_service_data(raw)

    assert visible == "Человеческий ответ."
    assert blocks == []
    assert invalid == 1


def test_multiple_complete_blocks_keep_order_and_do_not_merge_boundaries():
    raw = (
        '<DATA>{"metadata":{"first":1}}</DATA>\n'
        '<DATA>{"metadata":{"second":2}}</DATA>'
    )

    visible, blocks, invalid = extract_service_data(raw)

    assert visible == ""
    assert [block.metadata for block in blocks] == [{"first": 1}, {"second": 2}]
    assert invalid == 0


def test_markdown_service_json_handles_nested_braces_and_escaped_quotes():
    payload = {
        "current_state": {"current_step": "done"},
        "metadata": {"note": 'brace } and escaped "quote"'},
    }
    raw = "Текст.\nЗаполненный current_state:\n```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"

    visible, blocks, invalid = extract_service_data(raw)

    assert visible == "Текст."
    assert len(blocks) == 1
    assert blocks[0].current_state == {"current_step": "done"}
    assert blocks[0].metadata == payload["metadata"]
    assert invalid == 0


def test_user_facing_json_example_with_metadata_key_is_not_removed():
    raw = 'Пример JSON для пользователя:\n{"metadata":{"name":"Alice"}}\nЭто обычное объяснение.'

    visible, blocks, invalid = extract_service_data(raw)

    assert visible == raw
    assert blocks == []
    assert invalid == 0


def test_user_facing_ws_example_without_service_signature_is_not_removed():
    raw = 'Пример структуры рабочего листа: {"ws":{"title":"Demo"}}.'

    visible, blocks, invalid = extract_service_data(raw)

    assert visible == raw
    assert blocks == []
    assert invalid == 0


def test_legacy_square_block_remains_supported():
    visible, blocks, invalid = extract_service_data('[DATA]{"legacy":true}[/DATA]')

    assert visible == ""
    assert len(blocks) == 1
    assert blocks[0].legacy is True
    assert blocks[0].metadata == {"legacy": True}
    assert invalid == 0


def test_line_start_literal_does_not_join_with_later_block():
    raw = "Начало.\n<DATA> как слово.\n<DATA>{\"current_step\":\"ok\"}</DATA>\nКонец"

    visible, blocks, invalid = extract_service_data(raw)

    assert "<DATA> как слово." in visible
    assert blocks[0].current_state == {"current_step": "ok"}
    assert invalid == 0


def test_fallback_metadata_requires_service_signature():
    ordinary = '```json\n{"metadata":{"name":"Alice"}}\n```'
    visible, blocks, invalid = extract_service_data(ordinary)

    assert visible == ordinary
    assert blocks == []
    assert invalid == 0


def test_signed_metadata_in_markdown_fence_is_service_data():
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        "```json\n"
        '{"metadata":{"ws":{"ok":true}}}\n'
        "```"
    )

    visible, blocks, invalid = extract_service_data(raw)

    assert visible == "Ответ."
    assert blocks[0].metadata == {"ws": {"ok": True}}
    assert invalid == 0


def test_truncated_untagged_service_json_is_hidden_without_partial_state():
    raw = 'Ответ.\nСлужебный блок данных:\n{"current_step":"secret","metadata":{"private":'

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert extraction.blocks == []
    assert extraction.invalid_count == 1
    assert extraction.classification == "parse_error"


def test_signed_service_json_can_be_followed_by_visible_explanation():
    raw = (
        "Текст.\n"
        "Заполненный current_state:\n"
        "JSON\n"
        '{"current_state":{"current_step":"done"}}\n'
        "Пояснение для пользователя."
    )

    visible, blocks, invalid = extract_service_data(raw)

    assert visible == "Текст.\n\nПояснение для пользователя."
    assert blocks[0].current_state == {"current_step": "done"}
    assert invalid == 0


def test_extraction_classification_is_safe_and_does_not_include_metadata():
    cases = [
        ('Ответ <DATA>{"metadata":{"private":"value"}}</DATA>', "tags"),
        ('Ответ.\n{"current_step":"step"}', "fallback"),
        ("Обычный ответ.", "none"),
        ('Ответ.\n<DATA>{"metadata":', "parse_error"),
    ]

    for raw, expected in cases:
        extraction = extract_service_data_details(raw)
        assert extraction.classification == expected
        assert "private" not in str(service_data_diagnostics(extraction))
