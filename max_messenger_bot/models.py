from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


def parse_message(update: dict[str, Any]) -> IncomingMessage | None:
    raw = update.get("message") or update.get("body") or update.get("data") or update
    body = raw.get("body") or raw
    recipient = raw.get("recipient") or raw.get("chat") or {}
    sender = parse_sender(raw)
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
    if isinstance(possible_attachments, list):
        attachments = possible_attachments
        for item in possible_attachments:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type not in {"image", "video", "audio", "file"}:
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

    return IncomingMessage(
        raw=raw,
        message_id=raw.get("mid") or raw.get("message_id") or raw.get("id"),
        chat_id=int(chat_id),
        sender=sender,
        text=body.get("text") if isinstance(body, dict) else raw.get("text"),
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
    message = callback.get("message") or {}
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
