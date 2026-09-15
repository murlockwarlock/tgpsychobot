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


async def record_ai_attempt_log(
    session: Any,
    *,
    user_id: int | None,
    platform: str,
    dialogue_id: int | None = None,
    topic_id: int | None = None,
    topic_name: str | None = None,
    request_type: str = "chat",
    provider: str,
    model: str,
    prompt_summary: str | None = None,
    request_capture: dict | None = None,
    raw_response: str | None = None,
    clean_text: str | None = None,
    latency_ms: int | None = None,
    status: str = "success",  # "success" | "error"
    request_group_id: str | None = None,
    attempt_no: int = 1,
    attempt_role: str = "primary",  # "primary" | "fallback"
    error_type: str | None = None,
    error_message: str | None = None,
    error_classification: str | None = None,
    http_status: int | None = None,
    finish_reason: str | None = None,
    diagnostics: Any = None,
    provider_response_payload: str | None = None,
) -> int | None:
    """Safely record an individual AI provider attempt (success or error) in AILog.

    This function is strictly best-effort: failure to persist will log an error
    and rollback without raising or breaking the conversational flow.
    """
    import json
    import logging
    from database import AILog
    from error_reporting import sanitize_secret_values

    log = logging.getLogger("ai_log")

    payload_str: str | None = None
    if request_capture:
        try:
            raw_capture_json = json.dumps(request_capture, ensure_ascii=False, indent=2, default=str)
            payload_str = sanitize_secret_values(raw_capture_json)
        except Exception:
            payload_str = sanitize_secret_values(str(request_capture))

    resp_str: str | None = None
    if provider_response_payload is not None:
        resp_str = sanitize_secret_values(str(provider_response_payload))

    err_msg_str: str | None = None
    if error_message is not None:
        err_msg_str = sanitize_secret_values(str(error_message))

    diag_str: str | None = None
    if diagnostics is not None:
        try:
            if hasattr(diagnostics, "__dict__"):
                diag_data = {k: v for k, v in diagnostics.__dict__.items() if not k.startswith("_")}
            elif isinstance(diagnostics, dict):
                diag_data = diagnostics
            else:
                diag_data = {"repr": str(diagnostics)}
            diag_str = json.dumps(diag_data, ensure_ascii=False, default=str)
        except Exception:
            diag_str = str(diagnostics)

    if isinstance(error_classification, (tuple, list)):
        error_classification = str(error_classification[0]) if error_classification else None
    elif error_classification is not None and not isinstance(error_classification, str):
        error_classification = str(error_classification)

    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)

    req_type_norm = (request_type or "chat").strip().lower()
    if req_type_norm == "vision":
        from vision_reliability import sanitize_vision_text
        if payload_str:
            payload_str = sanitize_vision_text(payload_str)
        if resp_str:
            resp_str = sanitize_vision_text(resp_str)
        if err_msg_str:
            err_msg_str = sanitize_vision_text(err_msg_str)
        if diag_str:
            diag_str = sanitize_vision_text(diag_str)
        if raw_response:
            raw_response = sanitize_vision_text(raw_response)
        if clean_text:
            clean_text = sanitize_vision_text(clean_text)

    ai_log = AILog(
        user_id=user_id,
        request_type=(request_type or "chat").strip().lower(),
        provider=provider,
        model=model,
        prompt_summary=prompt_summary if prompt_summary else None,
        request_payload=payload_str,
        raw_response=raw_response or "",
        clean_text=clean_text,
        latency_ms=latency_ms,
        status=status,
        request_group_id=request_group_id,
        attempt_no=attempt_no,
        attempt_role=attempt_role,
        dialogue_id=dialogue_id,
        error_type=error_type,
        error_message=err_msg_str,
        error_classification=error_classification,
        http_status=http_status,
        finish_reason=finish_reason,
        diagnostics_json=diag_str,
        provider_response_payload=resp_str,
    )
    apply_ai_log_context(
        ai_log,
        platform=platform,
        topic_id=topic_id,
        topic_name=topic_name,
    )

    try:
        session.add(ai_log)
        await session.commit()
        return ai_log.id
    except Exception as exc:
        try:
            if hasattr(session, "rollback"):
                await session.rollback()
        except Exception:
            pass
        log.error("Failed to persist AI attempt log (best-effort): %s", exc, exc_info=True)
        return None


def build_ai_attempt_txt_file(log_entry: Any) -> str:
    """Format full, downloadable raw diagnostic log file matching Section 10 specification."""
    status = (getattr(log_entry, "status", None) or "success").upper()
    req_group = getattr(log_entry, "request_group_id", None) or "не зафиксирован"
    attempt = getattr(log_entry, "attempt_no", None) or 1
    role = (getattr(log_entry, "attempt_role", None) or "primary").upper()

    created_at = getattr(log_entry, "created_at", None)
    ts_str = str(created_at) if created_at else "не зафиксирован"

    platform = getattr(log_entry, "platform", None) or "не зафиксирована"
    user_id = getattr(log_entry, "user_id", None)

    try:
        from max_messenger_bot.identity import is_max_user_id, raw_max_user_id
        if platform.lower() == "max" and user_id is not None and is_max_user_id(user_id):
            disp_user_id = str(raw_max_user_id(user_id))
        else:
            disp_user_id = str(user_id) if user_id is not None else "не указан"
    except Exception:
        disp_user_id = str(user_id) if user_id is not None else "не указан"

    dialogue_id = getattr(log_entry, "dialogue_id", None) or "не зафиксирован"
    topic_str = ai_log_context_label(log_entry)

    provider = getattr(log_entry, "provider", None) or "не указан"
    model = getattr(log_entry, "model", None) or "не указан"
    lat = getattr(log_entry, "latency_ms", None)
    lat_str = f"{lat} ms ({lat / 1000:.2f} сек)" if lat is not None else "не измерялось"

    http_status = getattr(log_entry, "http_status", None)
    http_str = str(http_status) if http_status is not None else "не зафиксирован"

    error_type = getattr(log_entry, "error_type", None) or "не зафиксировано"
    error_cls = getattr(log_entry, "error_classification", None) or "не зафиксировано"
    error_msg = getattr(log_entry, "error_message", None) or "не зафиксировано"
    finish_reason = getattr(log_entry, "finish_reason", None) or "не зафиксирован"

    request_payload = getattr(log_entry, "request_payload", None) or "<none>"
    raw_provider_resp = (
        getattr(log_entry, "provider_response_payload", None)
        or getattr(log_entry, "raw_response", None)
        or "<none>"
    )

    if status == "ERROR":
        app_error = f"{error_type}: {error_msg}"
        clean_text = getattr(log_entry, "clean_text", None) or "не зафиксирован"
    else:
        app_error = "не зафиксировано"
        clean_text = getattr(log_entry, "clean_text", None) or getattr(log_entry, "raw_response", None) or "не зафиксирован"

    req_payload_content = (
        request_payload
        if (request_payload and request_payload.strip() and request_payload != "<none>")
        else "не зафиксирован"
    )

    return (
        f"========================================\n"
        f"AI LOG RECORD #{getattr(log_entry, 'id', 0)}\n"
        f"========================================\n\n"
        f"Status: {status}\n"
        f"Request Group: {req_group}\n"
        f"Attempt: {attempt} ({role})\n"
        f"Platform: {platform}\n"
        f"User ID: {disp_user_id}\n"
        f"Timestamp: {ts_str}\n"
        f"Dialogue: {dialogue_id}\n"
        f"Topic: {topic_str}\n\n"
        f"Provider: {provider}\n"
        f"Model: {model}\n"
        f"Latency: {lat_str}\n\n"
        f"HTTP Status: {http_str}\n"
        f"Error Type: {error_type}\n"
        f"Error Classification: {error_cls}\n"
        f"Error Message: {error_msg}\n"
        f"Finish Reason: {finish_reason}\n\n"
        f"========================================\n"
        f"📤 [1] FULL REQUEST PAYLOAD:\n"
        f"----------------------------------------\n"
        f"{req_payload_content}\n\n"
        f"========================================\n"
        f"🤖 [2] RAW RESPONSE FROM LLM:\n"
        f"----------------------------------------\n"
        f"{raw_provider_resp}\n\n"
        f"========================================\n"
        f"🚨 [3] APPLICATION ERROR:\n"
        f"----------------------------------------\n"
        f"{app_error}\n\n"
        f"========================================\n"
        f"💬 [3] CLEAN TEXT SENT TO USER:\n"
        f"----------------------------------------\n"
        f"{clean_text}\n"
    )

