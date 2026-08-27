from __future__ import annotations

from typing import Any


AI_PLATFORM_TELEGRAM = "telegram"
AI_PLATFORM_MAX = "max"
AI_CONTEXT_LEGACY = "legacy"
AI_CONTEXT_MAIN = "main"
AI_CONTEXT_TOPIC = "topic"


def apply_ai_log_context(
    log: Any,
    *,
    platform: str,
    topic_id: int | None,
    topic_name: str | None,
) -> None:
    normalized_topic_id = topic_id if topic_id not in (None, 0) else None
    log.platform = platform
    log.context_kind = AI_CONTEXT_TOPIC if normalized_topic_id is not None else AI_CONTEXT_MAIN
    log.topic_id = normalized_topic_id
    log.topic_name_snapshot = topic_name if normalized_topic_id is not None else None


def ai_log_context_label(log: Any) -> str:
    context_kind = getattr(log, "context_kind", None)
    if context_kind == AI_CONTEXT_MAIN:
        return "Основной диалог"
    if context_kind == AI_CONTEXT_TOPIC:
        topic_name = (getattr(log, "topic_name_snapshot", None) or "").strip()
        if topic_name:
            return f"Тема диалога — «{topic_name}»"
        topic_id = getattr(log, "topic_id", None)
        if topic_id is not None:
            return f"Тема диалога — ID {topic_id}"
    return "не зафиксирован"
