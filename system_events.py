"""Canonical synthetic system messages for AI dialogue history."""
from __future__ import annotations

import re


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
