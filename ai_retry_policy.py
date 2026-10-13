from __future__ import annotations

from collections import OrderedDict
from typing import Any

from error_reporting import classify_external_error

RETRYABLE_ERROR_CLASSIFICATIONS = frozenset({
    "timeout",
    "network_ssl",
    "network_connection",
    "provider_5xx",
    "rate_limit",
    "empty_response",
    "invalid_response",
    "unknown",
})

NON_RETRYABLE_ERROR_CLASSIFICATIONS = frozenset({
    "configuration",
    "auth",
    "forbidden_geo",
    "insufficient_balance_quota",
    "provider_rejection",
    "output_budget_exhausted",
    "application_internal",
})

_MAX_RETRY_TOKEN_CLAIMS = 4096
_claimed_retry_tokens: OrderedDict[tuple[str, Any], None] = OrderedDict()


def is_retryable_classification(classification: str | None) -> bool:
    if not classification:
        return False
    norm = classification.strip().lower()
    return norm in RETRYABLE_ERROR_CLASSIFICATIONS and norm not in NON_RETRYABLE_ERROR_CLASSIFICATIONS


def is_retryable_ai_error(exception: Exception | None, provider: str | None = None) -> bool:
    if exception is None:
        return False
    explicit = getattr(exception, "classification", None)
    if explicit and isinstance(explicit, str):
        return is_retryable_classification(explicit)
    attempts = getattr(exception, "provider_attempts", None)
    if attempts:
        for attempt in reversed(attempts):
            attempt_cls = getattr(attempt, "classification", None)
            if attempt_cls and isinstance(attempt_cls, str):
                return is_retryable_classification(attempt_cls)
    code, _ = classify_external_error(exception, provider=provider)
    return is_retryable_classification(code)


def claim_retry_token(platform: str, token: Any) -> bool:
    key = (platform, str(token))
    if key in _claimed_retry_tokens:
        return False
    _claimed_retry_tokens[key] = None
    _claimed_retry_tokens.move_to_end(key)
    while len(_claimed_retry_tokens) > _MAX_RETRY_TOKEN_CLAIMS:
        _claimed_retry_tokens.popitem(last=False)
    return True


def unclaim_retry_token(platform: str, token: Any) -> None:
    _claimed_retry_tokens.pop((platform, str(token)), None)


def is_retry_token_claimed(platform: str, token: Any) -> bool:
    return (platform, str(token)) in _claimed_retry_tokens


def clear_retry_claims() -> None:
    _claimed_retry_tokens.clear()


async def validate_terminal_failure_retry_eligibility(
    session: Any,
    platform: str,
    user_id: int,
    failed_log_id: int,
    *,
    expected_dialogue_id: int | None = None,
    expected_topic_id: int | None = None,
) -> tuple[bool, str, Any | None]:
    """Validate that the referenced AILog represents a genuine, unfulfilled, retryable terminal failure."""
    from database import AILog
    from sqlalchemy import select

    failed_log = await session.get(AILog, failed_log_id)
    if not failed_log:
        return False, "log_not_found", None

    if failed_log.user_id != user_id:
        return False, "user_mismatch", None

    log_platform = (failed_log.platform or "").strip().lower()
    norm_platform = (platform or "").strip().lower()
    if norm_platform == "telegram" and log_platform not in {"telegram", ""}:
        return False, "platform_mismatch", None
    if norm_platform == "max" and log_platform != "max":
        return False, "platform_mismatch", None

    if failed_log.request_type != "chat":
        return False, "invalid_request_type", None

    if failed_log.status != "error":
        return False, "status_not_error", None

    prompt = (failed_log.prompt_summary or "").strip()
    if not prompt:
        return False, "empty_prompt", None

    if expected_dialogue_id is not None and failed_log.dialogue_id != expected_dialogue_id:
        return False, "stale_dialogue", None

    if expected_topic_id is not None and failed_log.topic_id != expected_topic_id:
        return False, "stale_topic", None

    # Determine effective terminal failure outcome across the entire request group
    terminal_row = failed_log
    if failed_log.request_group_id:
        stmt = (
            select(AILog)
            .where(AILog.request_group_id == failed_log.request_group_id)
            .order_by(AILog.attempt_no.desc(), AILog.id.desc())
        )
        group_rows = (await session.scalars(stmt)).all()
        if any(r.status == "success" for r in group_rows):
            return False, "group_already_succeeded", None
        if group_rows:
            terminal_row = group_rows[0]

    # Effective terminal classification must be retryable
    term_classification = terminal_row.error_classification
    if not is_retryable_classification(term_classification):
        return False, f"non_retryable_classification:{term_classification}", terminal_row

    # Check if a later chat turn with the exact same prompt already succeeded in this dialogue/topic
    stmt_later = select(AILog.id).where(
        AILog.user_id == user_id,
        AILog.dialogue_id == failed_log.dialogue_id,
        AILog.topic_id == failed_log.topic_id,
        AILog.request_type == "chat",
        AILog.status == "success",
        AILog.prompt_summary == failed_log.prompt_summary,
        AILog.id > failed_log.id,
    ).limit(1)
    later_success = (await session.scalar(stmt_later)) is not None
    if later_success:
        return False, "already_fulfilled", terminal_row

    return True, "ok", terminal_row
