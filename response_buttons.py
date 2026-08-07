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
BUTTON_RE = re.compile(r"\[([^\]\n]{1,64})\]\((.+)\)")
TEST_START_DIRECTIVE_RE = re.compile(
    r"(?<![:\w])\[?\s*(?:START|RUN)\\?_TEST\s*\]?(?!\w)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ResponseButton:
    text: str
    kind: str
    value: str


def _is_valid_action(action: str) -> bool:
    if not 1 <= len(action) <= MAX_ACTION_CHARS:
        return False
    if not all(char.isprintable() for char in action):
        return False
    if any(char in "[]()|" for char in action):
        return False
    return len(f"{ACTION_CALLBACK_PREFIX}{action}".encode("utf-8")) <= MAX_CALLBACK_DATA_BYTES


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


def _parse_button_row(line: str) -> list[ResponseButton] | None:
    cleaned_line = _clean_part(line)
    parts = cleaned_line.split("|")
    if not parts or len(parts) > MAX_BUTTONS_PER_ROW:
        return None

    row: list[ResponseButton] = []
    for part in parts:
        cleaned_part = _clean_part(part)
        match = BUTTON_RE.fullmatch(cleaned_part)
        if match:
            text = _clean_part(match.group(1))
            target = match.group(2).strip()
            if not text:
                return None
            if target.lower().startswith("btn:"):
                action = target[4:]
                if not _is_valid_action(action):
                    return None
                row.append(ResponseButton(text=text, kind="action", value=action))
            else:
                parsed_url = urlsplit(target)
                if parsed_url.scheme.lower() not in {"http", "https"} or not parsed_url.netloc or any(char.isspace() for char in target):
                    return None
                row.append(ResponseButton(text=text, kind="url", value=target))
        else:
            bracket_match = re.fullmatch(r"\[([^\]\n]{1,64})\]", cleaned_part)
            if bracket_match:
                b_text = _clean_part(bracket_match.group(1))
                if b_text and _is_valid_action(b_text):
                    row.append(ResponseButton(text=b_text, kind="action", value=b_text))
                    continue

            word = cleaned_part.strip()
            if word.lower() in COMMON_PLAIN_BUTTON_WORDS and _is_valid_action(word):
                row.append(ResponseButton(text=word, kind="action", value=word))
                continue

            return None
    return row


def extract_response_buttons(text: str | None) -> tuple[str, list[list[ResponseButton]]]:
    """Remove standalone button rows and return their platform-neutral description."""
    source = text or ""
    if r"\n" in source:
        source = source.replace(r"\r\n", "\n").replace(r"\n", "\n")
    clean_lines: list[str] = []
    rows: list[list[ResponseButton]] = []
    for line in source.splitlines():
        parsed = _parse_button_row(line) if len(rows) < MAX_BUTTON_ROWS else None
        if parsed:
            rows.append(parsed)
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
