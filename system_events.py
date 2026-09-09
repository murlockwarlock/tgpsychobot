"""Canonical synthetic system messages for AI dialogue history."""
from __future__ import annotations

import re

SYSTEM_EVENT_ROLE = "system_event"


async def record_navigation_system_event(
    session,
    *,
    user_id: int,
    dialogue_id: int,
    topic_id: int | None,
    text: str,
):
    """Persist a navigation system event into messages table with role='system_event'."""
    from database import Message
    msg = Message(
        user_id=user_id,
        dialogue_id=dialogue_id,
        topic_id=topic_id,
        role=SYSTEM_EVENT_ROLE,
        content=text,
    )
    session.add(msg)
    await session.flush()
    return msg


def sanitize_synthetic_text_fragment(value: str | None) -> str:
    """Sanitize arbitrary text for inclusion in synthetic system events.

    Escapes backslashes, quotes, square brackets, parentheses and replaces
    newlines/tabs with spaces to prevent prompt injection / markup corruption.
    """
    value = value if isinstance(value, str) else ""
    value = value.replace("\\", "\\\\")
    value = re.sub(r"[\r\n\t]+", " ", value)
    for character in ('"', "[", "]", "(", ")"):
        value = value.replace(character, f"\\{character}")
    return value.strip()


def build_topic_auto_start_system_message(topic_name: str | None) -> str:
    """Build canonical hidden event message for topic auto-start."""
    sanitized_name = sanitize_synthetic_text_fragment(topic_name)
    return f'[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь выбрал тему "{sanitized_name}"]'


def build_topic_resume_system_message(topic_name: str | None) -> str:
    """Build canonical hidden event message for topic resume."""
    sanitized_name = sanitize_synthetic_text_fragment(topic_name)
    return f'[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь вернулся к теме "{sanitized_name}"]'


def build_main_dialogue_resume_system_message() -> str:
    """Build canonical hidden event message for main dialogue return/resume."""
    return '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь вернулся в общий режим диалога]'
