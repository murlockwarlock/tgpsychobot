from __future__ import annotations

import html
import re


def markdown_to_html(text: str | None) -> str:
    if not text:
        return ""
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"__(.+?)__", r"<b>\1</b>", escaped)
    escaped = re.sub(r"\*(.+?)\*", r"<i>\1</i>", escaped)
    escaped = re.sub(r"_(.+?)_", r"<i>\1</i>", escaped)
    escaped = escaped.replace("\n", "<br>")
    return escaped


def split_text(text: str, max_len: int = 3900) -> list[str]:
    if len(text) <= max_len:
        return [text]
    parts: list[str] = []
    rest = text
    while rest:
        if len(rest) <= max_len:
            parts.append(rest)
            break
        boundary = rest.rfind("\n", 0, max_len)
        if boundary == -1:
            boundary = max_len
        parts.append(rest[:boundary].strip())
        rest = rest[boundary:].strip()
    return [part for part in parts if part]

