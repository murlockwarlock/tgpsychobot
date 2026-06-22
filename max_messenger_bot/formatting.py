from __future__ import annotations

import html
import re

MAX_MESSAGE_TEXT_LEN = 3900


def markdown_to_html(text: str | None) -> str:
    if not text:
        return ""

    # 1. Escape HTML special characters first
    escaped = html.escape(text)

    # 2. Convert bullet lists to bullet points using • (U+2022)
    # e.g., "- Item" or "* Item" -> "• Item"
    escaped = re.sub(r'^\s*[-*+]\s+', '• ', escaped, flags=re.MULTILINE)

    # 3. Convert headers to bold
    # e.g., "### Header" -> "<b>Header</b>"
    escaped = re.sub(r'^\s*#{1,6}\s+(.+)$', r'<b>\1</b>', escaped, flags=re.MULTILINE)

    # 4. Convert bold and italic markdown tags to HTML
    # Bold first: **text** or __text__ -> <b>text</b>
    escaped = re.sub(r'\*\*(?=[^<>]*\*\*)((?:(?!\n\n)[^<>])+?)\*\*', r'<b>\1</b>', escaped)
    escaped = re.sub(r'__(?=[^<>]*__)((?:(?!\n\n)[^<>])+?)__', r'<b>\1</b>', escaped)
    
    # Italic next: *text* or _text_ -> <i>text</i>
    escaped = re.sub(r'(?<!\w)\*(?!\s)([^<>\n]+?)(?<!\s)\*(?!\w)', r'<i>\1</i>', escaped)
    escaped = re.sub(r'(?<!\w)_(?!\s)([^<>\n]+?)(?<!\s)_(?!\w)', r'<i>\1</i>', escaped)

    # Clean up unmatched bold/italic markers
    escaped = escaped.replace('**', '').replace('__', '')
    escaped = escaped.replace('<b></b>', '').replace('<i></i>', '')

    return escaped


def _open_tags(text: str) -> list[str]:
    tags: list[str] = []
    for match in re.finditer(r"</?(b|i)>", text):
        tag = match.group(1)
        if match.group(0).startswith("</"):
            if tag in tags[::-1]:
                tags.pop(len(tags) - 1 - tags[::-1].index(tag))
        else:
            tags.append(tag)
    return tags


def _safe_boundary(text: str, max_len: int) -> int:
    boundary = text.rfind("\n", 0, max_len)
    if boundary == -1 or boundary < max_len // 2:
        boundary = text.rfind(" ", 0, max_len)
    if boundary == -1 or boundary < max_len // 2:
        boundary = max_len

    # Do not cut in the middle of a simple HTML tag.
    lt = text.rfind("<", 0, boundary)
    gt = text.rfind(">", 0, boundary)
    if lt > gt:
        next_gt = text.find(">", boundary)
        if next_gt != -1 and next_gt < max_len:
            boundary = next_gt + 1
        else:
            boundary = lt
    return max(1, boundary)


def split_text(text: str, max_len: int = MAX_MESSAGE_TEXT_LEN) -> list[str]:
    if len(text) <= max_len:
        return [text]
    parts: list[str] = []
    rest = text
    prefix = ""
    while rest:
        if len(prefix) + len(rest) <= max_len:
            parts.append(prefix + rest + "".join(f"</{tag}>" for tag in reversed(_open_tags(prefix + rest))))
            break
        usable_len = max_len - len(prefix) - 16
        boundary = _safe_boundary(rest, usable_len)
        chunk_body = rest[:boundary].strip()
        chunk = prefix + chunk_body
        active = _open_tags(chunk)
        parts.append(chunk + "".join(f"</{tag}>" for tag in reversed(active)))
        prefix = "".join(f"<{tag}>" for tag in active)
        rest = rest[boundary:].strip()
    return [part for part in parts if part]
