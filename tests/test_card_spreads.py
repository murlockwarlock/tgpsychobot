from card_spreads import extract_numbered_spread_definition


PROMPT = """
### Расклад 1. Карта дня

[CHOICE_IMG_HIDDEN: tarot | 4]

### Расклад 10. Специальный — «Союз душ» (7 карт)

[CHOICE_IMG_HIDDEN: tarot | 8 | 7]

### Расклад 11. Путь и предназначение

[CHOICE_IMG_HIDDEN: tarot | 10 | 9]
"""


def test_extracts_multiround_definition_from_selected_section():
    assert extract_numbered_spread_definition(PROMPT, 10) == {
        "hidden": True,
        "category": "tarot",
        "cards_per_round": 8,
        "rounds": 7,
    }


def test_does_not_take_command_from_neighboring_section():
    assert extract_numbered_spread_definition(PROMPT, 1) == {
        "hidden": True,
        "category": "tarot",
        "cards_per_round": 4,
        "rounds": 1,
    }


def test_returns_none_for_unknown_layout():
    assert extract_numbered_spread_definition(PROMPT, 8) is None
