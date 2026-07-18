import re


CARD_CHOICE_DIRECTIVE_RE = re.compile(
    r"\[(CHOICE_IMG_HIDDEN|CHOICE_IMG):\s*(.+?)\s*\|\s*(\d+)"
    r"(?:\s*\|\s*(\d+))?\s*\]",
    re.IGNORECASE,
)


def extract_numbered_spread_definition(system_prompt: str, choice_number: int) -> dict | None:
    if not system_prompt or choice_number < 1:
        return None

    section_pattern = re.compile(
        rf"(?ims)^\s*#{{1,6}}\s*Расклад\s*(?:№\s*)?{choice_number}(?!\d).*?"
        rf"(?=^\s*#{{1,6}}\s*Расклад\s*(?:№\s*)?\d+|\Z)"
    )
    section_match = section_pattern.search(system_prompt)
    if not section_match:
        return None

    command_match = CARD_CHOICE_DIRECTIVE_RE.search(section_match.group(0))
    if not command_match:
        return None

    return {
        "hidden": command_match.group(1).upper() == "CHOICE_IMG_HIDDEN",
        "category": command_match.group(2).strip(),
        "cards_per_round": int(command_match.group(3)),
        "rounds": int(command_match.group(4)) if command_match.group(4) else 1,
    }
