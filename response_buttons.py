"""Parse inline button declarations from generated responses."""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urlsplit


MAX_BUTTONS_PER_ROW = 8
MAX_BUTTON_ROWS = 20
MAX_ACTION_CHARS = 30
MAX_CALLBACK_DATA_BYTES = 64
ACTION_CALLBACK_PREFIX = "ai_btn:"
ACTION_CALLBACK_ID_SEPARATOR = "|"
ACTION_CALLBACK_ID_WIDTH = 2
ACTION_CALLBACK_ID_SUFFIX = (
    f"{ACTION_CALLBACK_ID_SEPARATOR}{'0' * ACTION_CALLBACK_ID_WIDTH}"
)
BUTTON_RE = re.compile(r"\[([^\]\n]{1,64})\]\((.+)\)")
TEST_START_DIRECTIVE_RE = re.compile(
    r"(?<![:\w])\[?\s*(?:START|RUN)\\?_TEST\s*\]?(?!\w)",
    re.IGNORECASE,
)
_ACTION_CALLBACK_ID_RE = re.compile(
    rf"{re.escape(ACTION_CALLBACK_ID_SEPARATOR)}"
    rf"([0-9a-f]{{{ACTION_CALLBACK_ID_WIDTH}}})$"
)


@dataclass(frozen=True)
class ResponseButton:
    text: str
    kind: str
    value: str


def build_action_callback_data(action: str, button_index: int | None = None) -> str:
    """Build a legacy or collision-safe Telegram callback payload."""
    callback_data = f"{ACTION_CALLBACK_PREFIX}{action}"
    if button_index is None:
        if len(callback_data.encode("utf-8")) > MAX_CALLBACK_DATA_BYTES:
            raise ValueError("action callback data exceeds Telegram's 64-byte limit")
        return callback_data
    if not isinstance(button_index, int) or not 0 <= button_index <= 0xFF:
        raise ValueError("button_index must fit into the two-digit callback identity")
    callback_data = (
        f"{callback_data}{ACTION_CALLBACK_ID_SEPARATOR}"
        f"{button_index:0{ACTION_CALLBACK_ID_WIDTH}x}"
    )
    if len(callback_data.encode("utf-8")) > MAX_CALLBACK_DATA_BYTES:
        raise ValueError("action callback data exceeds Telegram's 64-byte limit")
    return callback_data


def split_action_callback_data(callback_data: str) -> tuple[str, int | None]:
    """Return the action and optional generated per-button identity."""
    if not isinstance(callback_data, str) or not callback_data.startswith(ACTION_CALLBACK_PREFIX):
        return callback_data, None
    payload = callback_data[len(ACTION_CALLBACK_PREFIX):]
    match = _ACTION_CALLBACK_ID_RE.search(payload)
    if not match:
        return payload, None
    return payload[:match.start()], int(match.group(1), 16)


def _is_valid_action(action: str) -> bool:
    if not 1 <= len(action) <= MAX_ACTION_CHARS:
        return False
    if not all(char.isprintable() for char in action):
        return False
    if any(char in "[]()|" for char in action):
        return False
    return len(
        f"{ACTION_CALLBACK_PREFIX}{action}{ACTION_CALLBACK_ID_SUFFIX}"
        .encode("utf-8")
    ) <= MAX_CALLBACK_DATA_BYTES


def _clean_part(part: str) -> str:
    p = part.strip()
    while len(p) >= 2:
        if (p.startswith("**") and p.endswith("**")) or (p.startswith("__") and p.endswith("__")):
            p = p[2:-2].strip()
        elif (p.startswith("*") and p.endswith("*")) or (p.startswith("_") and p.endswith("_")) or (p.startswith("`") and p.endswith("`")) or (p.startswith("~") and p.endswith("~")):
            p = p[1:-1].strip()
        else:
            break
    return p


COMMON_PLAIN_BUTTON_WORDS = frozenset({
    "дальше", "далее", "продолжить", "начать", "погнали", "вперед", "вперёд", "готово", "понятно", "ок", "хорошо"
})


def _button_from_target(text: str, target: str) -> ResponseButton | None:
    cleaned_target = target.strip()
    if not cleaned_target or "|" in cleaned_target:
        return None
    if cleaned_target.lower().startswith("btn:"):
        action = cleaned_target[4:]
        if not _is_valid_action(action):
            return None
        return ResponseButton(text=text, kind="action", value=action)

    try:
        parsed_url = urlsplit(cleaned_target)
    except ValueError:
        return None
    if (
        parsed_url.scheme.lower() not in {"http", "https"}
        or not parsed_url.netloc
        or any(char.isspace() for char in cleaned_target)
    ):
        return None
    return ResponseButton(text=text, kind="url", value=cleaned_target)


def _parse_button_part(part: str) -> ResponseButton | None:
    cleaned_part = _clean_part(part)
    if not cleaned_part or "|" in cleaned_part:
        return None

    match = BUTTON_RE.fullmatch(cleaned_part)
    if match:
        text = _clean_part(match.group(1))
        if not text:
            return None
        return _button_from_target(text, match.group(2))

    bracket_match = re.fullmatch(r"\[([^\]\n]{1,64})\]", cleaned_part)
    if bracket_match:
        text = _clean_part(bracket_match.group(1))
        if text and _is_valid_action(text):
            return ResponseButton(text=text, kind="action", value=text)
        return None

    word = cleaned_part.strip()
    if word.lower() in COMMON_PLAIN_BUTTON_WORDS and _is_valid_action(word):
        return ResponseButton(text=word, kind="action", value=word)
    return None


def _split_button_line(line: str) -> tuple[list[str], list[str]]:
    parts: list[str] = []
    separators: list[str] = []
    part_start = 0
    position = 0
    in_label = False
    target_depth = 0
    while position < len(line):
        char = line[position]
        if in_label:
            if char == "]":
                in_label = False
            position += 1
            continue
        if target_depth:
            if char == "(":
                target_depth += 1
            elif char == ")":
                target_depth -= 1
            position += 1
            continue
        if char == "[":
            in_label = True
            position += 1
            continue
        if char == "(":
            target_depth = 1
            position += 1
            continue
        if char.isspace():
            separator_start = position
            while position < len(line) and line[position].isspace():
                position += 1
            if position == len(line):
                break
            if line[position] == "|":
                parts.append(line[part_start:separator_start])
                separators.append("|")
                position += 1
                while position < len(line) and line[position].isspace():
                    position += 1
                part_start = position
            else:
                parts.append(line[part_start:separator_start])
                separators.append(" ")
                part_start = position
            continue
        if char == "|":
            parts.append(line[part_start:position])
            separators.append("|")
            position += 1
            while position < len(line) and line[position].isspace():
                position += 1
            part_start = position
            continue
        position += 1
    parts.append(line[part_start:])
    return parts, separators


def _parse_button_row(line: str) -> list[list[ResponseButton]] | None:
    cleaned_line = _clean_part(line)
    if not cleaned_line:
        return None

    parts, separators = _split_button_line(cleaned_line)
    if len(parts) != len(separators) + 1:
        return None
    if any(
        separators[index] == " "
        and (
            not _clean_part(parts[index]).startswith("[")
            or not _clean_part(parts[index + 1]).startswith("[")
        )
        for index in range(len(separators))
    ):
        return None
    rows: list[list[ResponseButton]] = [[]]
    for index, part in enumerate(parts):
        button = _parse_button_part(part)
        if button is None:
            return None
        rows[-1].append(button)
        if len(rows[-1]) > MAX_BUTTONS_PER_ROW:
            return None
        if index < len(separators) and separators[index] == " ":
            if len(rows) >= MAX_BUTTON_ROWS:
                return None
            rows.append([])
    return rows


def extract_response_buttons(text: str | None) -> tuple[str, list[list[ResponseButton]]]:
    """Remove standalone button rows and return their platform-neutral description."""
    source = text or ""
    if r"\n" in source:
        source = source.replace(r"\r\n", "\n").replace(r"\n", "\n")
    clean_lines: list[str] = []
    rows: list[list[ResponseButton]] = []
    for line in source.splitlines():
        parsed = _parse_button_row(line) if len(rows) < MAX_BUTTON_ROWS else None
        if parsed and len(rows) + len(parsed) <= MAX_BUTTON_ROWS:
            rows.extend(parsed)
        else:
            clean_lines.append(line)

    clean_text = "\n".join(clean_lines).strip()
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    return clean_text, rows


def extract_test_start_directive(text: str | None) -> tuple[bool, str]:
    """Extract a standalone test directive without executing btn:start_test buttons."""
    raw = text or ""
    has_directive = bool(TEST_START_DIRECTIVE_RE.search(raw))
    if not has_directive:
        return False, raw.strip()
    clean_text = TEST_START_DIRECTIVE_RE.sub("", raw)
    clean_text = re.sub(r"\s+([.,!?;:])", r"\1", clean_text)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    return True, clean_text.strip(" \t\r\n-—–:;")
