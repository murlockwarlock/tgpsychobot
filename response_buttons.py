"""Parse inline button declarations from generated responses."""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urlsplit


MAX_BUTTONS_PER_ROW = 8
MAX_BUTTON_ROWS = 20
BUTTON_RE = re.compile(r"\[([^\]\n]{1,64})\]\((.+)\)")
ACTION_RE = re.compile(r"[A-Za-z0-9_.-]{1,30}")


@dataclass(frozen=True)
class ResponseButton:
    text: str
    kind: str
    value: str


def _parse_button_row(line: str) -> list[ResponseButton] | None:
    parts = line.split("|")
    if not parts or len(parts) > MAX_BUTTONS_PER_ROW:
        return None

    row: list[ResponseButton] = []
    for part in parts:
        match = BUTTON_RE.fullmatch(part.strip())
        if not match:
            return None
        text = match.group(1).strip()
        target = match.group(2).strip()
        if not text:
            return None
        if target.lower().startswith("btn:"):
            action = target[4:]
            if not ACTION_RE.fullmatch(action):
                return None
            row.append(ResponseButton(text=text, kind="action", value=action))
        else:
            parsed_url = urlsplit(target)
            if parsed_url.scheme.lower() not in {"http", "https"} or not parsed_url.netloc or any(char.isspace() for char in target):
                return None
            row.append(ResponseButton(text=text, kind="url", value=target))
    return row


def extract_response_buttons(text: str | None) -> tuple[str, list[list[ResponseButton]]]:
    """Remove standalone button rows and return their platform-neutral description."""
    source = text or ""
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
