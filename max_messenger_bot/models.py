from __future__ import annotations

from dataclasses import dataclass
import html
from typing import Any


MAX_ID_OFFSET = 100_000_000_000


def _nested_get(data: dict[str, Any] | None, *path: str) -> Any:
    current: Any = data or {}
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


@dataclass(slots=True)
class Sender:
    user_id: int
    username: str | None
    first_name: str | None
    last_name: str | None

    @property
    def full_name(self) -> str:
        parts = [self.first_name or "", self.last_name or ""]
        text = " ".join(part for part in parts if part).strip()
        return text or self.username or str(self.user_id)


@dataclass(slots=True)
class IncomingMessage:
    raw: dict[str, Any]
    message_id: str | None
    chat_id: int
    sender: Sender
    text: str | None
    html_text: str | None = None
    attachments: list[dict[str, Any]] | None = None
    media_type: str | None = None
    media_token: str | None = None
    media_url: str | None = None
    start_payload: str | None = None


@dataclass(slots=True)
class IncomingCallback:
    raw: dict[str, Any]
    callback_id: str
    payload: str
    chat_id: int
    message_id: str | None
    sender: Sender


def parse_sender(raw: dict[str, Any]) -> Sender:
    sender = (
        raw.get("sender")
        or raw.get("user")
        or raw.get("from")
        or {}
    )
    return Sender(
        user_id=int(
            sender.get("user_id")
            or sender.get("id")
            or raw.get("user_id")
            or 0
        ),
        username=sender.get("username"),
        first_name=sender.get("first_name") or sender.get("name"),
        last_name=sender.get("last_name"),
    )


def _markup_type(markup: dict[str, Any]) -> str:
    raw_type = markup.get("type")
    if isinstance(raw_type, dict):
        raw_type = raw_type.get("type")
    return str(raw_type or "").lower()


def _markup_bounds(markup: dict[str, Any]) -> tuple[int, int] | None:
    start = markup.get("from")
    length = markup.get("length") or markup.get("len")
    if start is None or length is None:
        return None
    try:
        start_int = max(0, int(start))
        end_int = max(start_int, start_int + int(length))
    except (TypeError, ValueError):
        return None
    return start_int, end_int


def _markup_tags(markup: dict[str, Any]) -> tuple[str, str] | None:
    kind = _markup_type(markup)
    if kind in {"strong", "bold"}:
        return "<b>", "</b>"
    if kind in {"emphasized", "italic"}:
        return "<i>", "</i>"
    if kind in {"monospaced", "code"}:
        return "<code>", "</code>"
    if kind in {"strikethrough", "strike"}:
        return "<s>", "</s>"
    if kind == "underline":
        return "<u>", "</u>"
    if kind in {"highlighted", "mark"}:
        return "<mark>", "</mark>"
    if kind in {"quote", "blockquote"}:
        return "<blockquote>", "</blockquote>"
    if kind in {"heading", "header"}:
        return "<h1>", "</h1>"
    if kind == "link":
        url = markup.get("url")
        if isinstance(markup.get("link"), dict):
            url = markup["link"].get("url") or url
        if url:
            return f'<a href="{html.escape(str(url), quote=True)}">', "</a>"
    if kind in {"user_mention", "user"}:
        user_id = markup.get("user_id") or markup.get("userId")
        if user_id:
            return f'<a href="max://user/{html.escape(str(user_id), quote=True)}">', "</a>"
    return None


def _render_markup_html(text: str | None, markups: Any) -> str | None:
    if not text or not isinstance(markups, list):
        return None
    # MAX platform uses UTF-16 code unit offsets (Telegram-compatible)
    utf16_bytes = text.encode('utf-16-le')
    code_units = [utf16_bytes[i:i+2].decode('utf-16-le', errors='surrogatepass') for i in range(0, len(utf16_bytes), 2)]
    openings: dict[int, list[str]] = {}
    closings: dict[int, list[str]] = {}
    for markup in markups:
        if not isinstance(markup, dict):
            continue
        bounds = _markup_bounds(markup)
        tags = _markup_tags(markup)
        if not bounds or not tags:
            continue
        start, end = bounds
        if start >= len(code_units):
            continue
        end = min(end, len(code_units))
        opening, closing = tags
        openings.setdefault(start, []).append(opening)
        closings.setdefault(end, []).insert(0, closing)
    parts: list[str] = []
    for index, cu in enumerate(code_units):
        parts.extend(openings.get(index, []))
        parts.append(html.escape(cu))
        parts.extend(closings.get(index + 1, []))
    raw_html = "".join(parts)
    return raw_html.encode('utf-16-le', errors='surrogatepass').decode('utf-16-le')


def parse_message(update: dict[str, Any]) -> IncomingMessage | None:
    raw = update.get("message") or update.get("body") or update.get("data") or update
    body = raw.get("body") or raw
    link = raw.get("link") if isinstance(raw, dict) else None
    forwarded = link.get("message") if isinstance(link, dict) and link.get("type") == "forward" else None
    forwarded_body = (forwarded.get("body") or forwarded) if isinstance(forwarded, dict) else None
    recipient = raw.get("recipient") or raw.get("chat") or {}
    sender = parse_sender(raw)
    sender = Sender(
        user_id=sender.user_id + MAX_ID_OFFSET,
        username=sender.username,
        first_name=sender.first_name,
        last_name=sender.last_name,
    )
    attachments = None
    media_type = None
    media_token = None
    media_url = None

    chat_id = (
        recipient.get("chat_id")
        or recipient.get("user_id")
        or raw.get("chat_id")
        or sender.user_id
    )
    if not chat_id:
        return None

    possible_attachments = (
        body.get("attachments")
        if isinstance(body, dict)
        else None
    ) or raw.get("attachments") or []
    if not possible_attachments and isinstance(forwarded_body, dict):
        possible_attachments = forwarded_body.get("attachments") or []
    if isinstance(possible_attachments, list):
        attachments = possible_attachments
        for item in possible_attachments:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type not in {"image", "video", "audio", "file", "share"}:
                continue
            payload = item.get("payload") or {}
            token = (
                payload.get("token")
                or payload.get("file_token")
                or payload.get("id")
                or item.get("token")
            )
            url = (
                payload.get("url")
                or payload.get("file_url")
                or payload.get("download_url")
            )
            if token:
                media_type = str(item_type)
                media_token = str(token)
                media_url = str(url) if url else None
                break

    text = body.get("text") if isinstance(body, dict) else raw.get("text")
    content_body = body
    if not text and isinstance(forwarded_body, dict):
        text = forwarded_body.get("text")
        content_body = forwarded_body
    markups = None
    if isinstance(content_body, dict):
        markups = content_body.get("markup") or content_body.get("markups")
    html_text = _render_markup_html(text, markups)
    if not html_text and isinstance(content_body, dict) and content_body.get("format") == "html":
        html_text = text

    return IncomingMessage(
        raw=raw,
        message_id=raw.get("mid") or raw.get("message_id") or raw.get("id"),
        chat_id=int(chat_id),
        sender=sender,
        text=text,
        html_text=html_text,
        attachments=attachments,
        media_type=media_type,
        media_token=media_token,
        media_url=media_url,
        start_payload=update.get("start_payload")
        or update.get("payload")
        or _nested_get(update, "bot_started", "payload"),
    )


def parse_callback(update: dict[str, Any]) -> IncomingCallback | None:
    callback = update.get("callback") or update.get("message_callback") or update
    sender = parse_sender(callback)
    sender = Sender(
        user_id=sender.user_id + MAX_ID_OFFSET,
        username=sender.username,
        first_name=sender.first_name,
        last_name=sender.last_name,
    )
    # MAX puts "message" at the top level of the update, not inside "callback"
    message = update.get("message") or callback.get("message") or {}
    recipient = message.get("recipient") or callback.get("recipient") or {}
    chat_id = (
        recipient.get("chat_id")
        or recipient.get("user_id")
        or message.get("chat_id")
        or callback.get("chat_id")
        or sender.user_id
    )
    callback_id = callback.get("callback_id") or callback.get("id")
    payload = callback.get("payload") or callback.get("data") or ""
    if not chat_id or not callback_id:
        return None

    return IncomingCallback(
        raw=callback,
        callback_id=str(callback_id),
        payload=str(payload),
        chat_id=int(chat_id),
        message_id=message.get("mid") or message.get("message_id") or message.get("id"),
        sender=sender,
    )
