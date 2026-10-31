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


def test_multiple_signed_fallback_objects_in_one_fence_are_all_hidden():
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        "```json\n"
        '{"current_state":{"current_step":"first"}}\n'
        '{"current_state":{"current_step":"second"},"metadata":{"private":"secret"}}\n'
        "```"
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert [block.current_state for block in extraction.blocks] == [
        {"current_step": "first"},
        {"current_step": "second"},
    ]
    assert extraction.blocks[1].metadata == {"private": "secret"}
    assert "private" not in extraction.visible_text
    assert extraction.invalid_count == 0


def test_mixed_tagged_and_fallback_blocks_in_one_fence_keep_source_order():
    raw = (
        "Служебный блок данных:\n"
        "```json\n"
        '<DATA>{"event":"FIRST"}</DATA>\n'
        '{"event":"SECOND"}\n'
        "```"
    )

    extraction = extract_service_data_details(raw)

    assert [block.events for block in extraction.blocks] == [["FIRST"], ["SECOND"]]
    assert extraction.visible_text == ""
    assert extraction.invalid_count == 0


def test_visible_code_before_data_inside_fence_is_preserved():
    raw = (
        "Ответ.\n"
        "```python\n"
        'print("before")\n'
        '<DATA>{"metadata":{"x":1}}</DATA>\n'
        "```\n"
        "Конец."
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text.startswith("Ответ.\n```python\n")
    assert 'print("before")' in extraction.visible_text
    assert extraction.visible_text.endswith("```\nКонец.")
    assert extraction.blocks[0].metadata == {"x": 1}
    assert extraction.invalid_count == 0


def test_malformed_fallback_payload_after_valid_payload_is_hidden():
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        "```json\n"
        '{"current_state":{"current_step":"ok"}}\n'
        '{"metadata":{"private":"secret",}}\n'
        "```"
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert [block.current_state for block in extraction.blocks] == [{"current_step": "ok"}]
    assert extraction.invalid_count == 1
    assert extraction.classification == "parse_error"
    assert "private" not in extraction.visible_text


def test_malformed_untagged_service_json_is_hidden_at_response_tail():
    raw = 'Ответ.\n{"current_state":{"current_step":"secret",}}'

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert extraction.blocks == []
    assert extraction.invalid_count == 1
    assert extraction.classification == "parse_error"


def test_malformed_fallback_payload_before_valid_payload_is_diagnosed():
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        "```json\n"
        '{"metadata":{"private":"secret",}}\n'
        '{"current_state":{"current_step":"good"}}\n'
        "```"
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert [block.current_state for block in extraction.blocks] == [{"current_step": "good"}]
    assert extraction.invalid_count == 1
    assert extraction.classification == "parse_error"


def test_untagged_malformed_service_payload_before_valid_payload_is_hidden():
    raw = (
        'Ответ.\n{"metadata":{"private":"secret",}}\n'
        '{"current_state":{"current_step":"good"}}'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert [block.current_state for block in extraction.blocks] == [{"current_step": "good"}]
    assert extraction.invalid_count == 1
    assert extraction.classification == "parse_error"


def test_signed_non_json_fallback_is_hidden_as_parse_error():
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        "```json\n"
        "not json\n"
        "```"
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert extraction.blocks == []
    assert extraction.invalid_count == 1
    assert extraction.classification == "parse_error"


def test_fallback_data_preserves_visible_braced_code_inside_signed_fence():
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        "```python\n"
        'print({"before": 1})\n'
        '{"current_state":{"current_step":"ok"}}\n'
        'print({"after": 2})\n'
        "```\n"
        "Конец."
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == raw
    assert extraction.blocks == []
    assert extraction.invalid_count == 0


def test_malformed_untagged_service_json_removes_explicit_heading():
    raw = 'Ответ.\nСлужебный блок данных:\n{"current_state":{"current_step":"secret",}}'

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert extraction.blocks == []
    assert extraction.invalid_count == 1
    assert extraction.classification == "parse_error"


def test_untagged_malformed_payload_before_tagged_service_is_hidden():
    raw = 'Ответ.\n{"metadata":{"private":"secret",}}\n<DATA>{"event":"valid"}</DATA>'

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert [block.events for block in extraction.blocks] == [["valid"]]
    assert extraction.invalid_count == 1
    assert extraction.classification == "parse_error"


def test_signed_json_fence_hides_malformed_tail_after_valid_payload():
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        "```json\n"
        '{"current_state":{"current_step":"good"}}\n'
        "not json\n"
        "```\n"
        "Конец."
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ.\nКонец."
    assert [block.current_state for block in extraction.blocks] == [{"current_step": "good"}]
    assert extraction.invalid_count == 1
    assert extraction.classification == "parse_error"


def test_malformed_user_json_example_before_service_payload_is_preserved():
    raw = (
        'Ответ. Пример JSON: {"metadata":{"name":"Alice",}}\n'
        '{"current_state":{"current_step":"good"}}'
    )

    extraction = extract_service_data_details(raw)

    assert 'Пример JSON: {"metadata":{"name":"Alice",}}' in extraction.visible_text
    assert extraction.blocks[0].current_state == {"current_step": "good"}
    assert extraction.invalid_count == 0
    assert extraction.classification == "fallback"


def test_service_metadata_literal_inside_visible_python_is_preserved():
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        "```python\n"
        'print({"metadata": {"name": "Alice"}})\n'
        '{"current_state":{"current_step":"ok"}}\n'
        'print("after")\n'
        "```"
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == raw
    assert extraction.blocks == []
    assert extraction.invalid_count == 0


def test_multiline_malformed_user_json_example_before_service_payload_is_preserved():
    raw = (
        "Ответ.\n"
        "Пример JSON:\n"
        '{"metadata":{"name":"Alice",}}\n'
        '{"current_state":{"current_step":"good"}}'
    )

    extraction = extract_service_data_details(raw)

    assert "Пример JSON:" in extraction.visible_text
    assert '{"metadata":{"name":"Alice",}}' in extraction.visible_text
    assert extraction.blocks[0].current_state == {"current_step": "good"}
    assert extraction.invalid_count == 0
    assert extraction.classification == "fallback"


def test_blank_line_after_json_example_marker_preserves_malformed_example():
    raw = (
        "Ответ.\n"
        "Пример JSON:\n"
        "\n"
        '{"metadata":{"name":"Alice",}}\n'
        '{"current_state":{"current_step":"good"}}'
    )

    extraction = extract_service_data_details(raw)

    assert "Пример JSON:" in extraction.visible_text
    assert '{"metadata":{"name":"Alice",}}' in extraction.visible_text
    assert extraction.blocks[0].current_state == {"current_step": "good"}
    assert extraction.invalid_count == 0
    assert extraction.classification == "fallback"


@pytest.mark.parametrize("fence", ["```python", "```"])
def test_metadata_only_literal_in_non_json_fence_is_not_service_data(fence):
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        f"{fence}\n"
        '{"metadata":{"name":"Alice"}}\n'
        '{"current_state":{"current_step":"ok"}}\n'
        "```"
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == raw
    assert extraction.blocks == []
    assert extraction.invalid_count == 0


@pytest.mark.parametrize("fence", ["```python", "```"])
def test_service_shaped_literal_in_non_json_fence_is_not_service_data(fence):
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        f"{fence}\n"
        '{"current_state":{"current_step":"example"}}\n'
        "```"
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == raw
    assert extraction.blocks == []
    assert extraction.invalid_count == 0


def test_unrelated_empty_code_fence_is_preserved():
    raw = "Обычный ответ.\n```python\n```\nПродолжение."

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == raw
    assert extraction.blocks == []
    assert extraction.invalid_count == 0


def test_unrelated_empty_fence_after_data_is_preserved():
    raw = 'Ответ.\n<DATA>{"metadata":{"x":1}}</DATA>\n```python\n```\nПродолжение.'

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == 'Ответ.\n\n```python\n```\nПродолжение.'
    assert extraction.blocks[0].metadata == {"x": 1}
    assert extraction.invalid_count == 0


def test_fallback_data_does_not_remove_visible_code_inside_signed_fence():
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        "```python\n"
        'print("before")\n'
        '{"current_state":{"current_step":"ok"}}\n'
        'print("after")\n'
        "```\n"
        "Конец."
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == raw
    assert extraction.blocks == []
    assert extraction.invalid_count == 0


@pytest.mark.parametrize(
    ("opening", "closing"),
    [("<DATA>", "</DATA>"), ("[DATA]", "[/DATA]")],
)
def test_pure_service_fence_is_removed_when_closing_fence_is_missing(opening, closing):
    raw = (
        "Ответ.\n"
        "```json\n"
        f'{opening}{{"metadata":{{"private":"secret"}}}}{closing}'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert extraction.blocks
    assert "```" not in extraction.visible_text
    assert extraction.invalid_count == 0


@pytest.mark.parametrize(
    ("opening", "closing"),
    [("<DATA>", "</DATA>"), ("[DATA]", "[/DATA]")],
)
def test_multiple_tagged_service_blocks_remove_their_wrapper_fence(opening, closing):
    raw = (
        "Ответ.\n"
        "```json\n"
        f'{opening}{{"metadata":{{"first":1}}}}{closing}\n'
        f'{opening}{{"metadata":{{"second":2}}}}{closing}\n'
        "```"
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert [block.raw_json for block in extraction.blocks] == [
        '{"metadata":{"first":1}}',
        '{"metadata":{"second":2}}',
    ]
    assert "```" not in extraction.visible_text
    assert extraction.invalid_count == 0


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
        ('Ответ.\n{"current_state":{"current_step":"step"}}', "fallback"),
        ("Обычный ответ.", "none"),
        ('Ответ.\n<DATA>{"metadata":', "parse_error"),
    ]

    for raw, expected in cases:
        extraction = extract_service_data_details(raw)
        assert extraction.classification == expected
        assert "private" not in str(service_data_diagnostics(extraction))


def test_inline_truncated_xml_opener_hides_service_suffix_but_keeps_prefix():
    raw = 'Ответ. <DATA version="1"\n{"metadata":{"private":"value"}}'

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert extraction.blocks == []
    assert extraction.invalid_count == 1


def test_multiple_signed_fallback_blocks_are_removed_and_preserved_in_order():
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        '{"metadata":{"first":1}}\n'
        "Заполненный current_state:\n"
        '{"current_state":{"current_step":"second"}}'
    )

    visible, blocks, invalid = extract_service_data(raw)

    assert visible == "Ответ."
    assert [block.metadata for block in blocks] == [{"first": 1}, {}]
    assert [block.current_state for block in blocks] == [{}, {"current_step": "second"}]
    assert invalid == 0


def test_valid_fallback_is_still_extracted_after_invalid_tagged_block():
    raw = (
        "Ответ.\n"
        "<DATA>{not-json}</DATA>\n"
        "Служебный блок данных:\n"
        '{"current_step":"safe"}'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert extraction.blocks[0].current_state == {"current_step": "safe"}
    assert extraction.invalid_count == 1
    assert extraction.classification == "parse_error"


def test_malformed_tagged_event_does_not_execute_alias_or_create_block():
    raw = '<DATA>{"event":"SHOULD_NOT_RUN"</DATA>'

    visible, blocks, invalid = extract_service_data(raw)

    assert visible == ""
    assert blocks == []
    assert invalid == 1


def test_ordinary_event_json_example_is_not_treated_as_service_data():
    raw = 'Ответ.\n{"event":"MEETING","title":"Demo"}'

    visible, blocks, invalid = extract_service_data(raw)

    assert visible == raw
    assert blocks == []
    assert invalid == 0


def test_generic_json_heading_does_not_erase_ordinary_json():
    raw = 'Ответ.\nJSON:\n{"name":"Alice"}\nПояснение.'

    visible, blocks, invalid = extract_service_data(raw)

    assert visible == raw
    assert blocks == []
    assert invalid == 0


def test_line_start_literal_data_without_json_is_not_hidden():
    raw = "Начало.\n<DATA> это обычное упоминание и продолжение."

    visible, blocks, invalid = extract_service_data(raw)

    assert visible == raw
    assert blocks == []
    assert invalid == 0


def test_valid_fallback_does_not_expose_later_truncated_service_tail():
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        '{"metadata":{"public":"value"}}\n'
        "Служебный блок данных:\n"
        '{"metadata":{"private":"secret"}'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert [block.metadata for block in extraction.blocks] == [{"public": "value"}]
    assert extraction.invalid_count == 1


def test_tagged_and_fallback_blocks_are_applied_in_source_order():
    raw = (
        "Служебный блок данных:\n"
        '{"current_state":{"current_step":"first"}}\n'
        '<DATA>{"current_state":{"current_step":"second"},"events":["SECOND"]}</DATA>'
    )

    visible, blocks, invalid = extract_service_data(raw)

    assert visible == ""
    assert [block.current_state for block in blocks] == [
        {"current_step": "first"},
        {"current_step": "second"},
    ]
    assert [block.events for block in blocks] == [[], ["SECOND"]]
    assert invalid == 0


def test_rejected_literal_truncated_opener_does_not_mask_later_service_opener():
    raw = (
        "Упоминание <DATA как слово\n"
        "Пояснение.\n"
        '<DATA version="1"\n'
        '{"metadata":{"private":"secret"}}'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Упоминание <DATA как слово\nПояснение."
    assert extraction.blocks == []
    assert extraction.invalid_count == 1


def test_multiple_truncated_fallback_sections_are_all_hidden():
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        '{"metadata":{"private":"secret_one"}\n'
        "Служебный блок данных:\n"
        '{"metadata":{"private":"secret_two"}'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert extraction.blocks == []
    assert extraction.invalid_count >= 1


def test_data_tags_inside_fallback_json_strings_are_preserved():
    note = "Keep <DATA>{}</DATA> literally"
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        + json.dumps({"metadata": {"note": note}})
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert extraction.blocks[0].metadata == {"note": note}
    assert extraction.invalid_count == 0


@pytest.mark.parametrize(
    "note",
    [
        "Keep <DATA/> literally",
        "Keep <DATA literally",
    ],
)
def test_all_tag_cleanup_passes_ignore_tags_inside_fallback_json_strings(note):
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        + json.dumps({"metadata": {"note": note}})
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert extraction.blocks[0].metadata == {"note": note}
    assert extraction.invalid_count == 0


def test_quote_in_visible_prose_does_not_hide_valid_fallback_json():
    raw = (
        'Ответ содержит символ ".\n'
        "Служебный блок данных:\n"
        '{"current_state":{"current_step":"done"},"event":"DONE"}'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == 'Ответ содержит символ ".'
    assert extraction.blocks[0].current_state == {"current_step": "done"}
    assert extraction.blocks[0].events == ["DONE"]
    assert extraction.invalid_count == 0


def test_overlapping_fallback_removals_hide_all_same_line_service_objects():
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        '{"metadata":{"first":1}} {"metadata":{"private":"secret"}}'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert [block.metadata for block in extraction.blocks] == [
        {"first": 1},
        {"private": "secret"},
    ]
    assert "secret" not in extraction.visible_text
    assert extraction.invalid_count == 0


def test_invalid_bracketed_prose_does_not_shield_tagged_service_block():
    raw = 'Ответ [не JSON "\n<DATA>{"metadata":{"private":"secret"}}</DATA>\n"]'

    extraction = extract_service_data_details(raw)

    assert extraction.blocks[0].metadata == {"private": "secret"}
    assert "secret" not in extraction.visible_text
    assert extraction.invalid_count == 0


def test_unmatched_prose_bracket_does_not_shield_fallback_service_block():
    raw = (
        "Оцените тревогу в диапазоне [0, 10).\n"
        "Служебный блок данных:\n"
        '{"current_state":{"current_step":"done"},"event":"DONE"}'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Оцените тревогу в диапазоне [0, 10)."
    assert extraction.blocks[0].current_state == {"current_step": "done"}
    assert extraction.blocks[0].events == ["DONE"]
    assert extraction.invalid_count == 0


def test_malformed_service_envelope_does_not_recover_nested_event_object():
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        '{"metadata":{"event":"SHOULD_NOT_RUN"}'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert extraction.blocks == []
    assert extraction.invalid_count == 1


def test_json_array_literals_protect_embedded_data_tags():
    raw = '["Keep <DATA>{}</DATA> literally", "Keep <DATA/> too"]'

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == raw
    assert extraction.blocks == []
    assert extraction.invalid_count == 0


def test_multiline_malformed_service_envelope_does_not_recover_nested_event():
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        "{\n"
        '  "metadata": {\n'
        '    "current_state": {"current_step":"bad"},\n'
        '    "event":"SHOULD_NOT_RUN"\n'
        ""
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert extraction.blocks == []
    assert extraction.invalid_count == 1


def test_complete_nested_metadata_in_truncated_envelope_is_not_recovered():
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        "{\n"
        '  "metadata": {\n'
        '    "current_state": {"current_step":"bad"},\n'
        '    "event":"SHOULD_NOT_RUN"\n'
        "  }\n"
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert extraction.blocks == []
    assert extraction.invalid_count == 1


def test_multiline_malformed_envelope_without_heading_does_not_recover_nested_data():
    raw = (
        "Ответ.\n"
        "{\n"
        '  "metadata": {\n'
        '    "current_state": {"current_step":"bad"},\n'
        '    "event":"SHOULD_NOT_RUN"\n'
        "  }\n"
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert extraction.blocks == []
    assert extraction.invalid_count == 1


def test_prose_bracket_does_not_shield_inline_fallback_service_json():
    raw = (
        'Ответ содержит символ [. {"current_state":{"current_step":"done"},'
        '"metadata":{"private":"secret"}}'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ содержит символ [."
    assert extraction.blocks[0].current_state == {"current_step": "done"}
    assert extraction.blocks[0].metadata == {"private": "secret"}
    assert extraction.invalid_count == 0


def test_prose_brace_does_not_shield_inline_fallback_service_json():
    raw = (
        'Ответ содержит символ {. {"current_state":{"current_step":"done"},'
        '"metadata":{"private":"secret"}}'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ содержит символ {."
    assert extraction.blocks[0].current_state == {"current_step": "done"}
    assert extraction.blocks[0].metadata == {"private": "secret"}
    assert extraction.invalid_count == 0


def test_multiple_service_headings_do_not_remove_following_explanation():
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        "Заполненный current_state:\n"
        '{"metadata":{"ok":true}}\n'
        "Пояснение для пользователя."
    )

    extraction = extract_service_data_details(raw)

    assert "Пояснение для пользователя." in extraction.visible_text
    assert extraction.blocks[0].metadata == {"ok": True}
    assert extraction.invalid_count == 0


def test_completed_service_block_does_not_shield_later_inline_fallback():
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        '{"current_state":{"current_step":"first"}}\n'
        'Пояснение содержит {. {"current_state":{"current_step":"second"},'
        '"metadata":{"private":"secret"}}'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.blocks[0].current_state == {"current_step": "first"}
    assert extraction.blocks[1].current_state == {"current_step": "second"}
    assert extraction.blocks[1].metadata == {"private": "secret"}
    assert "secret" not in extraction.visible_text
    assert extraction.invalid_count == 0


def test_prose_quote_does_not_enable_partial_recovery_from_truncated_envelope():
    raw = (
        'Ответ содержит символ ".\n'
        "{\n"
        '  "metadata": {"current_state":{"current_step":"bad"},'
        '"event":"SHOULD_NOT_RUN"}\n'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == 'Ответ содержит символ ".'
    assert extraction.blocks == []
    assert extraction.invalid_count == 1


def test_combined_prose_delimiters_do_not_enable_partial_recovery():
    raw = (
        'Ответ содержит символы { и ".\n'
        "{\n"
        '  "metadata": {"current_state":{"current_step":"bad"},'
        '"event":"SHOULD_NOT_RUN"}\n'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == 'Ответ содержит символы { и ".'
    assert extraction.blocks == []
    assert extraction.invalid_count == 1


def test_unclosed_array_does_not_recover_nested_service_object():
    raw = (
        "Ответ.\n"
        "[\n"
        '{"current_state":{"current_step":"bad"},"event":"SHOULD_NOT_RUN"}\n'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert extraction.blocks == []
    assert extraction.invalid_count == 1


def test_indented_ordinary_json_example_is_not_deleted_as_unbalanced_service_tail():
    raw = 'Пример JSON:\n  {"metadata":{"name":"Alice"}}\nПояснение.'

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == raw
    assert extraction.blocks == []
    assert extraction.invalid_count == 0


def test_interval_prose_does_not_shield_or_delete_following_service_data():
    raw = (
        "[0, 10) — допустимый диапазон.\n"
        "Служебный блок данных:\n"
        '{"current_state":{"current_step":"done"},"event":"DONE"}'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "[0, 10) — допустимый диапазон."
    assert extraction.blocks[0].current_state == {"current_step": "done"}
    assert extraction.blocks[0].events == ["DONE"]
    assert extraction.invalid_count == 0


@pytest.mark.parametrize(
    "prefix",
    [
        "[0,\n",
        '["ordinary",\n',
    ],
)
def test_truncated_array_does_not_recover_nested_service_object(prefix):
    raw = (
        "Ответ.\n"
        + prefix
        + '{"current_state":{"current_step":"bad"},"event":"SHOULD_NOT_RUN"}\n'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert extraction.blocks == []
    assert extraction.invalid_count == 1


def test_inline_truncated_json_envelope_does_not_recover_nested_service_object():
    raw = (
        'Ответ. {"metadata":{"current_state":{"current_step":"bad"},'
        '"event":"SHOULD_NOT_RUN"}'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert extraction.blocks == []
    assert extraction.invalid_count == 1


def test_inline_truncated_array_does_not_recover_nested_service_object():
    raw = (
        'Ответ. [0, {"current_state":{"current_step":"bad"},'
        '"event":"SHOULD_NOT_RUN"}'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ."
    assert extraction.blocks == []
    assert extraction.invalid_count == 1


@pytest.mark.parametrize("interval", ["[0.0, 10.0)", "[0; 10)", "[0, +∞)"])
def test_decimal_and_semicolon_intervals_do_not_shield_service_data(interval):
    raw = (
        f"{interval} — допустимый диапазон.\n"
        '{"current_state":{"current_step":"done"},"event":"DONE"}'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == f"{interval} — допустимый диапазон."
    assert extraction.blocks[0].current_state == {"current_step": "done"}
    assert extraction.blocks[0].events == ["DONE"]
    assert extraction.invalid_count == 0


def test_scientific_interval_does_not_shield_following_service_data():
    raw = (
        "[0, 1e3) — допустимый диапазон.\n"
        '{"current_state":{"current_step":"done"},"event":"DONE"}'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "[0, 1e3) — допустимый диапазон."
    assert extraction.blocks[0].current_state == {"current_step": "done"}
    assert extraction.blocks[0].events == ["DONE"]
    assert extraction.invalid_count == 0


@pytest.mark.parametrize(
    "interval",
    [
        "[0,5; 1,5)",
        "[0; 1,5)",
        "[0,5; 1)",
        "[0,5; +∞)",
        "[-∞; 1,5)",
        "[0,5; 1e3)",
    ],
)
def test_decimal_comma_interval_does_not_shield_following_service_data(interval):
    raw = (
        f"{interval} — допустимый диапазон.\n"
        '{"current_state":{"current_step":"done"},"event":"DONE"}'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == f"{interval} — допустимый диапазон."
    assert extraction.blocks[0].current_state == {"current_step": "done"}
    assert extraction.blocks[0].events == ["DONE"]
    assert extraction.invalid_count == 0


def test_unfenced_data_does_not_remove_following_code_fence():
    raw = (
        '<DATA>{"current_state":{"current_step":"done"},"event":"DONE"}</DATA>\n'
        '```python\n'
        'print("ok")\n'
        '```'
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == '```python\nprint("ok")\n```'
    assert extraction.blocks[0].events == ["DONE"]
    assert extraction.invalid_count == 0


@pytest.mark.parametrize(
    ("opening", "closing"),
    [
        ("<DATA>", "</DATA>"),
        ("[DATA]", "[/DATA]"),
    ],
)
def test_inline_data_does_not_remove_code_before_tag(opening, closing):
    raw = (
        "Ответ.\n"
        "```python\n"
        f'print("hello") {opening}{{"metadata":{{"x":1}}}}{closing}\n'
        "```"
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text.startswith("Ответ.\n```python\n")
    assert extraction.visible_text.endswith('print("hello")\n```')
    assert 'print("hello")' in extraction.visible_text
    assert extraction.blocks[0].raw_json == '{"metadata":{"x":1}}'
    assert extraction.invalid_count == 0


@pytest.mark.parametrize(
    ("opening", "closing", "suffix"),
    [
        ("<DATA>", "</DATA>", ' print("hello")\n'),
        ("<DATA>", "</DATA>", '\nprint("hello")\n'),
        ("[DATA]", "[/DATA]", ' print("hello")\n'),
        ("[DATA]", "[/DATA]", '\nprint("hello")\n'),
    ],
)
def test_data_with_following_code_preserves_wrapper_fences(opening, closing, suffix):
    raw = (
        "Ответ.\n"
        "```python\n"
        f'{opening}{{"metadata":{{"x":1}}}}{closing}'
        f"{suffix}"
        "```"
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text.startswith("Ответ.\n```python\n")
    assert extraction.visible_text.endswith('print("hello")\n```')
    assert 'print("hello")' in extraction.visible_text
    assert extraction.blocks[0].raw_json == '{"metadata":{"x":1}}'
    assert extraction.invalid_count == 0


def test_signed_fallback_removes_closing_fence_after_blank_line():
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        "```json\n"
        '{"metadata":{"private":"secret"}}\n'
        "\n"
        "```\n"
        "Пояснение."
    )

    extraction = extract_service_data_details(raw)

    assert extraction.visible_text == "Ответ.\nПояснение."
    assert extraction.blocks[0].metadata == {"private": "secret"}
    assert "```" not in extraction.visible_text
    assert extraction.invalid_count == 0


@pytest.mark.parametrize(
    ("opening", "closing"),
    [
        ("```json\n\n", "\n```\n"),
        ("````json\n\n", "\n````\n"),
    ],
)
def test_signed_fallback_handles_fence_whitespace_and_lengths(opening, closing):
    raw = (
        "Ответ.\n"
        "Служебный блок данных:\n"
        + opening
        + '{"metadata":{"private":"secret"}}'
        + closing
        + "Пояснение для пользователя."
    )

    extraction = extract_service_data_details(raw)

    assert "Пояснение для пользователя." in extraction.visible_text
    assert extraction.blocks[0].metadata == {"private": "secret"}
    assert extraction.invalid_count == 0
