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


def clear_retry_claims() -> None:
    _claimed_retry_tokens.clear()
