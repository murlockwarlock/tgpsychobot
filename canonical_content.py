from __future__ import annotations

import html
import re
from urllib.parse import urlsplit


_TAG_RE = re.compile(
    r"</?(?:b|strong|i|em|u|ins|s|strike|del|code|pre|blockquote|a|br|mark|h[1-6])(?:\s+[^<>]*)?\s*/?>",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(r"(?<!\\)\[([^\]\n]{1,64})\]\((btn:[^\)\n]*)\)")
_TOKEN_PREFIX = "\ue000"
_TOKEN_SUFFIX = "\ue001"


def _canonical_tag(raw_tag: str) -> str:
    closing = raw_tag.startswith("</")
    match = re.match(r"</?([a-z][a-z0-9-]*)", raw_tag, re.IGNORECASE)
    if not match:
        return html.escape(raw_tag, quote=False)
    name = match.group(1).lower()
    name = {
        "strong": "b",
        "em": "i",
        "ins": "u",
        "strike": "s",
        "del": "s",
        "mark": "u",
        "h1": "b",
        "h2": "b",
        "h3": "b",
        "h4": "b",
        "h5": "b",
        "h6": "b",
    }.get(name, name)
    if closing:
        return f"</{name}>"
    if name == "br":
        return "<br>"
    if name == "a":
        href_match = re.search(r"\bhref\s*=\s*(['\"])(.*?)\1", raw_tag, re.IGNORECASE | re.DOTALL)
        href = html.unescape(href_match.group(2).strip()) if href_match else ""
        parsed = urlsplit(href)
        if parsed.scheme.lower() not in {"http", "https", "tg", "max"} or any(char.isspace() for char in href):
            return html.escape(raw_tag, quote=False)
        return f'<a href="{html.escape(href, quote=True)}">'
    return f"<{name}>"


def _markdown_to_html(text: str) -> str:
    escaped = html.escape(text, quote=False)
    placeholders: dict[str, str] = {}

    def placeholder(value: str) -> str:
        token = f"\ue010{len(placeholders)}\ue011"
        placeholders[token] = value
        return token

    escaped = re.sub(
        r"```(.*?)```",
        lambda match: placeholder(f"<pre><code>{match.group(1)}</code></pre>"),
        escaped,
        flags=re.DOTALL,
    )
    escaped = re.sub(
        r"`(.*?)`",
        lambda match: placeholder(f"<code>{match.group(1)}</code>"),
        escaped,
    )
    escaped = re.sub(r"^\s*[-*+]\s+", "• ", escaped, flags=re.MULTILINE)
    escaped = re.sub(r"^\s*#{1,6}\s+(.+)$", r"<b>\1</b>", escaped, flags=re.MULTILINE)
    escaped = re.sub(r"\*\*(?=[^<>]*\*\*)((?:(?!\n\n)[^<>])+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"__(?=[^<>]*__)((?:(?!\n\n)[^<>])+?)__", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\w)\*(?!\s)([^<>\n]+?)(?<!\s)\*(?!\w)", r"<i>\1</i>", escaped)
    escaped = re.sub(r"(?<!\w)_(?!\s)([^<>\n]+?)(?<!\s)_(?!\w)", r"<i>\1</i>", escaped)
    escaped = re.sub(r"~~(?=[^<>\n]+~~)([^<>\n]+?)~~", r"<s>\1</s>", escaped)
    escaped = re.sub(r"\[(.*?)\]\((.*?)\)", r'<a href="\2">\1</a>', escaped)
    escaped = re.sub(r"\n{3,}", "\n\n", escaped)
    for token, value in placeholders.items():
        escaped = escaped.replace(token, value)
    return escaped.strip()


def canonical_markup_to_html(text: str | None) -> str:
    if not text:
        return ""

    protected: dict[str, str] = {}

    def protect(replacement: str) -> str:
        token = f"{_TOKEN_PREFIX}{len(protected)}{_TOKEN_SUFFIX}"
        protected[token] = replacement
        return token

    source = _TAG_RE.sub(lambda match: protect(_canonical_tag(match.group(0))), text)
    source = html.unescape(source)
    source = _ACTION_RE.sub(lambda match: protect(html.escape(match.group(0), quote=False)), source)
    rendered = _markdown_to_html(source)
    for token, value in protected.items():
        rendered = rendered.replace(token, value)
    return rendered


def split_canonical_html(text: str, max_length: int = 4090) -> list[str]:
    if not text:
        return []
    if len(text) <= max_length:
        return [text]

    tag_re = re.compile(r"(</?[a-z][a-z0-9-]*(?:\s+[^>]*)?>)", re.IGNORECASE)
    parts = tag_re.split(text)
    chunks: list[str] = []
    current = ""
    opened: list[tuple[str, str]] = []

    def suffix() -> str:
        return "".join(f"</{name}>" for name, _ in reversed(opened))

    def reopen() -> str:
        return "".join(raw for _, raw in opened)

    for part in parts:
        if not part:
            continue
        if part.startswith("<"):
            match = re.match(r"</?([a-z][a-z0-9-]*)", part, re.IGNORECASE)
            if match:
                name = match.group(1).lower()
                if part.startswith("</"):
                    for index in range(len(opened) - 1, -1, -1):
                        if opened[index][0] == name:
                            opened.pop(index)
                            break
                elif name not in {"br", "hr", "img"}:
                    opened.append((name, part))
            if len(current) + len(part) + len(suffix()) > max_length and current.strip():
                chunks.append(current + suffix())
                current = reopen()
            current += part
            continue

        remaining = part
        while len(current) + len(remaining) + len(suffix()) > max_length:
            available = max_length - len(current) - len(suffix())
            if available <= 0:
                if current.strip():
                    chunks.append(current + suffix())
                current = reopen()
                available = max_length - len(current) - len(suffix())
            split_at = remaining.rfind("\n", 0, available)
            if split_at <= 0:
                split_at = remaining.rfind(" ", 0, available)
            if split_at <= 0:
                split_at = available
            current += remaining[:split_at]
            if current.strip():
                chunks.append(current + suffix())
            current = reopen()
            remaining = remaining[split_at:].lstrip()
        current += remaining

    if current.strip():
        chunks.append(current)
    return chunks
