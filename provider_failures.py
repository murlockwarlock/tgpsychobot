from __future__ import annotations

import json
import math
import re
import ssl
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping

import httpx


RETRYABLE_CLASSES = frozenset({
    "timeout", "network_connection", "network_ssl", "proxy_error", "rate_limit",
    "provider_overloaded", "provider_unavailable", "provider_5xx", "empty_response",
    "invalid_response",
})
OPERATOR_CLASSES = frozenset({
    "configuration", "auth", "permission", "forbidden_geo",
    "insufficient_balance_quota", "invalid_model_or_resource",
})
CANONICAL_CLASSES = RETRYABLE_CLASSES | OPERATOR_CLASSES | frozenset({
    "invalid_request", "context_limit", "output_budget_exhausted", "content_policy",
    "provider_rejection", "cancelled", "application_internal", "unknown",
})
LEGACY_ALIASES = {
    "auth_invalid_key": "auth", "auth_forbidden": "permission", "geo_blocked": "forbidden_geo",
    "rate_limited": "rate_limit", "insufficient_balance": "insufficient_balance_quota",
    "quota": "insufficient_balance_quota", "server_error": "provider_5xx",
    "network": "network_connection", "network_error": "network_connection",
    "invalid_model": "invalid_model_or_resource", "malformed_response": "invalid_response",
    "provider": "unknown",
}


def canonical_classification(value: Any) -> str:
    value = str(value or "unknown").strip().lower()
    return LEGACY_ALIASES.get(value, value if value in CANONICAL_CLASSES else "unknown")


def safe_message(value: Any, limit: int = 1000) -> str:
    text = str(value or "")
    text = re.sub(r"(?i)authorization\s*:\s*(?:bearer\s+)?[^\s,;}]+", "Authorization: [REDACTED]", text)
    text = re.sub(r"(?i)\bbearer\s+[^\s,;}]+", "Bearer [REDACTED]", text)
    try:
        from error_reporting import sanitize_secret_values
        text = sanitize_secret_values(text)
    except Exception:
        text = re.sub(r"(?i)(api[_-]?key|authorization|token|secret|password)([\"']?\s*[:=]\s*[\"']?)[^\s,\"'}]+", r"\1\2[REDACTED]", text)
    text = re.sub(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", text)
    text = re.sub(r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?)(?:bearer\s+)?[^\s,\"'}]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)\bbearer\s+[^\s,\"'}]+", "Bearer [REDACTED]", text)
    return text[:limit]


@dataclass(frozen=True)
class ProviderFailure:
    provider: str
    operation: str
    classification: str
    detail_code: str
    provider_error_code: str | None = None
    provider_error_type: str | None = None
    provider_error_message: str = ""
    http_status: int | None = None
    request_id: str | None = None
    retryable: bool = False
    retry_after_seconds: float | None = None
    network_attempted: bool = True
    transient: bool = False
    operator_action_required: bool = False
    classification_source: str = "provider_structured"

    def diagnostics(self) -> dict[str, Any]:
        return asdict(self)


def parse_retry_after(headers: Mapping[str, Any] | None, *, now: datetime | None = None) -> float | None:
    values = {str(k).lower(): v for k, v in (headers or {}).items()}
    raw = values.get("retry-after")
    if raw is None:
        return None
    try:
        delay = float(raw)
    except (ValueError, TypeError):
        try:
            date = parsedate_to_datetime(str(raw))
            delay = (date - (now or datetime.now(timezone.utc))).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, delay) if math.isfinite(delay) else None


_CODE_GROUPS = {
    "auth": "invalid_api_key authentication authentication_error unauthenticated invalid_auth",
    "permission": "permission_denied permission_error insufficient_permissions",
    "forbidden_geo": "ip_not_authorized unsupported_country_region_territory",
    "insufficient_balance_quota": "payment_required billing_error insufficient_quota quota_exceeded credit_balance_exhausted organization_spend_limit_exceeded project_spend_limit_exceeded organization_usage_limit_exceeded asr_payment_required",
    "rate_limit": "rate_limit_error rate_limit_exceeded too_many_requests slow_down",
    "provider_overloaded": "provider_overloaded overloaded_error server_is_overloaded",
    "provider_unavailable": "provider_unavailable service_unavailable service_unavailable_error unavailable",
    "provider_5xx": "server server_error api_error internal",
    "timeout": "timeout timeout_error deadline_exceeded",
    "invalid_request": "invalid_request invalid_request_error invalid_argument invalid_prompt precondition_failed payload_too_large unprocessable asr_unprocessable_entity unimplemented already_exists conflict_error invalid_image image_too_large image_too_small unsupported_image_format image_not_found image_download_failed",
    "invalid_model_or_resource": "not_found not_found_error model_not_found",
    "configuration": "failed_precondition",
    "context_limit": "context_length_exceeded token_limit_exceeded string_too_long request_too_large",
    "output_budget_exhausted": "max_tokens_exceeded max_output_tokens",
    "content_policy": "content_policy_violation refusal safety recitation language prohibited_content spii blocklist image_safety image_prohibited_content image_recitation content_blocked",
    "provider_rejection": "aborted malformed_function_call malformed_tool_call unexpected_tool_call too_many_tool_calls missing_thought_signature no_image image_other",
    "cancelled": "cancelled",
    "unknown": "unmapped",
}
CODE_CLASSES = {code: cls for cls, codes in _CODE_GROUPS.items() for code in codes.split()}
OPENROUTER_TYPES = frozenset("context_length_exceeded max_tokens_exceeded token_limit_exceeded string_too_long authentication permission_denied payment_required rate_limit_exceeded provider_overloaded provider_unavailable invalid_request invalid_prompt not_found precondition_failed payload_too_large unprocessable content_policy_violation refusal invalid_image image_too_large image_too_small unsupported_image_format image_not_found image_download_failed server timeout unmapped".split())


def _status_class(status: int | None, provider: str) -> str:
    if status == 503 and provider in {"deepseek", "openai"}:
        return "provider_overloaded"
    if status in {408, 504}:
        return "timeout"
    if status == 501:
        return "invalid_request"
    if status is not None and 500 <= status < 600:
        return "provider_5xx"
    return {400: "invalid_request", 401: "auth", 402: "insufficient_balance_quota",
            403: "permission", 404: "invalid_model_or_resource", 409: "invalid_request",
            413: "context_limit", 422: "invalid_request", 429: "rate_limit",
            499: "cancelled"}.get(status, "unknown")


def parse_provider_failure(
    provider: str, operation: str, http_status: int | None, body: Any = None,
    headers: Mapping[str, Any] | None = None,
) -> ProviderFailure | None:
    if isinstance(operation, int) and isinstance(http_status, dict):
        if body is not None and headers is None and isinstance(body, Mapping):
            headers = body
        operation, http_status, body = "chat", operation, http_status
    name = provider.lower()
    data = body if isinstance(body, dict) else {}
    error = data.get("error")
    if name == "openrouter" and not error:
        for choice in data.get("choices", []) if isinstance(data.get("choices"), list) else []:
            if isinstance(choice, dict) and (choice.get("error") or choice.get("finish_reason") == "error"):
                error = choice.get("error") or {"type": "unmapped"}
                break
    if name == "kie" and not error:
        task = data.get("data") if isinstance(data.get("data"), dict) else data
        if task.get("state") in {"fail", "failed", "error"}:
            error = {"code": task.get("failCode"), "message": task.get("failMsg"), "type": "task_failed"}
    err = error if isinstance(error, dict) else {}
    code = err.get("code")
    kind = err.get("type") or err.get("status")
    message = err.get("message") or data.get("msg") or data.get("message") or (error if isinstance(error, str) else "")
    metadata = err.get("metadata") if isinstance(err.get("metadata"), dict) else {}
    if name == "deepgram":
        code = data.get("err_code") or code
        kind = data.get("category") or kind
        message = data.get("err_msg") or message
    if name == "kie":
        code = code or data.get("code")
    if name == "openrouter":
        kind = metadata.get("error_type") or kind
    reason = None
    if name == "gemini":
        feedback = data.get("promptFeedback") or {}
        reason = feedback.get("blockReason") if isinstance(feedback, dict) else None
        for candidate in data.get("candidates", []) if isinstance(data.get("candidates"), list) else []:
            finish = candidate.get("finishReason") if isinstance(candidate, dict) else None
            if str(finish).lower() in CODE_CLASSES:
                reason = finish
                break
        kind = reason or kind
        for detail in err.get("details", []) if isinstance(err.get("details"), list) else []:
            if isinstance(detail, dict) and detail.get("reason"):
                code = detail["reason"]
    failed = bool(error or reason or data.get("err_code")) or (http_status is not None and http_status >= 400)
    failed = failed or (name == "kie" and code not in (None, 200, "200"))
    status = data.get("status") if name == "perplexity" else None
    failed = failed or status in {"failed", "incomplete", "cancelled", "queued", "in_progress"}
    if not failed:
        return None
    normalized_code = str(code or "").lower()
    normalized_type = str(kind or "").lower()
    signal = normalized_type if name == "openrouter" and kind else normalized_code
    if signal not in CODE_CLASSES and not (name == "openrouter" and kind):
        signal = normalized_type
    source = "provider_structured"
    classification = CODE_CLASSES.get(signal)
    retry_after = parse_retry_after(headers)
    effective_status = http_status
    if name == "kie" and str(code).isdigit() and code not in (200, "200"):
        effective_status = int(code)
        classification = classification or _status_class(effective_status, name)
        signal = str(code)
    if classification is None:
        classification = "unknown" if kind and name == "openrouter" else _status_class(effective_status, name)
        source = "provider_structured" if kind and name == "openrouter" else "http_status_fallback"
        signal = str(kind or code or effective_status or "unknown")
    lowered = str(message).lower()
    if name in {"claude", "anthropic"} and effective_status in {400, 429} and any(
        phrase in lowered for phrase in ("spend limit", "spend cap", "spending limit", "usage tier's monthly")
    ):
        classification, signal, source = "insufficient_balance_quota", "spend_limit", "message_fallback"
    if name in {"kie", "openai"} and effective_status == 401:
        if "access permissions" in lowered or "ip allowlist" in lowered or "ip not authorized" in lowered:
            classification = "permission" if name == "kie" else "forbidden_geo"
            signal, source = "access_policy", "message_fallback"
    if name == "openai" and effective_status == 403 and any(x in lowered for x in ("country", "region", "territory")):
        classification, signal, source = "forbidden_geo", "unsupported_region", "message_fallback"
    if name == "openai" and effective_status in {400, 429} and classification == "invalid_request" and any(
        marker in lowered for marker in (
            "credit balance", "credit_balance", "insufficient quota", "insufficient_quota",
            "spend limit", "spend_limit", "usage limit", "billing", "balance exhausted",
        )
    ):
        classification, signal, source = "insufficient_balance_quota", "billing_or_spend_limit", "message_fallback"
    if name == "perplexity" and effective_status in {400, 422} and any(
        marker in lowered for marker in (
            "model not found", "unknown model", "invalid model", "does not exist",
            "max_output_tokens", "max_tokens",
        )
    ):
        classification, signal, source = "configuration", "invalid_model_or_parameter", "message_fallback"
    if name == "perplexity" and effective_status == 404:
        classification, signal, source = "configuration", "not_found", "provider_structured"
    if name in {"claude", "anthropic"} and effective_status in {400, 429} and classification == "rate_limit" and any(
        marker in lowered for marker in (
            "spend limit", "spend cap", "spending limit", "monthly", "billing", "credit balance",
            "credits exhausted", "payment required",
        )
    ):
        classification, signal, source = "insufficient_balance_quota", "spend_limit", "message_fallback"
    if name == "openrouter" and effective_status == 403 and any(
        marker in lowered for marker in ("security policy", "cloudflare", "access denied", "blocked")
    ):
        classification, signal, source = "provider_rejection", "security_policy", "message_fallback"
    if name == "gemini" and normalized_type == "resource_exhausted":
        details = err.get("details") or []
        quota_ids = [str(v.get("quotaId", "")).lower() for d in details if isinstance(d, dict)
                     for v in d.get("violations", []) if isinstance(v, dict)]
        short_window = bool(quota_ids) and all("perminute" in q or "persecond" in q for q in quota_ids)
        classification = "rate_limit" if short_window else "insufficient_balance_quota"
        signal = "rate_limit_exceeded" if short_window else "quota_exceeded"
        for detail in details:
            if isinstance(detail, dict) and detail.get("retryDelay"):
                retry_after = parse_retry_after({"retry-after": str(detail["retryDelay"]).removesuffix("s")})
    if status in {"failed", "incomplete", "cancelled", "queued", "in_progress"}:
        reason = (data.get("incomplete_details") or {}).get("reason")
        if status == "failed" and (error or code or kind):
            signal = str(kind or code or status)
            source = "provider_structured"
        else:
            classification = "output_budget_exhausted" if reason == "max_output_tokens" else "cancelled" if status == "cancelled" else "invalid_response"
            signal, source = str(reason or status), "provider_structured"
    retryable = classification in RETRYABLE_CLASSES
    if status in {"incomplete", "cancelled", "queued", "in_progress"}:
        retryable = False
    operator = classification in OPERATOR_CLASSES
    if name == "openrouter" and effective_status == 402 and metadata.get("limit_source") == "openrouter_in_flight_budget" and retry_after is not None:
        signal, retryable, operator = "openrouter_in_flight_budget", True, False
    header_values = {str(k).lower(): v for k, v in (headers or {}).items()}
    task_data = data.get("data") if isinstance(data.get("data"), dict) else data
    request_id = (
        data.get("request_id")
        or data.get("requestId")
        or data.get("id")
        or err.get("request_id")
        or metadata.get("request_id")
        or header_values.get("x-request-id")
        or header_values.get("request-id")
        or header_values.get("cf-ray")
        or task_data.get("taskId")
    )
    return ProviderFailure(
        provider=provider, operation=operation, classification=classification, detail_code=safe_message(signal, 160),
        provider_error_code=safe_message(code, 160) if code is not None else None,
        provider_error_type=safe_message(kind, 160) if kind is not None else None,
        provider_error_message=safe_message(message), http_status=http_status,
        request_id=safe_message(request_id, 200) if request_id else None,
        retryable=retryable, transient=retryable, retry_after_seconds=retry_after,
        operator_action_required=operator, classification_source=source,
    )


def normalize_exception(exception: Exception, provider: str = "", operation: str = "chat") -> ProviderFailure:
    chain, seen = [], set()
    item = exception
    while item is not None and id(item) not in seen:
        seen.add(id(item))
        chain.append(item)
        item = item.__cause__
    for item in chain:
        existing = getattr(item, "failure", None)
        if isinstance(existing, ProviderFailure):
            return existing
    for item in chain:
        cls = None
        if isinstance(item, (TimeoutError, httpx.TimeoutException)) or type(item).__name__ == "APITimeoutError":
            cls = "timeout"
        elif isinstance(item, ssl.SSLError) or "sslerror" in type(item).__name__.lower() or "ssl" in str(item).lower() and "unexpected_eof" in str(item).lower():
            cls = "network_ssl"
        elif isinstance(item, httpx.ProxyError):
            cls = "proxy_error"
        elif isinstance(item, (httpx.TransportError, ConnectionError)) or type(item).__name__ == "APIConnectionError":
            cls = "network_connection"
        if cls:
            return ProviderFailure(provider, operation, cls, cls, provider_error_message=safe_message(item), retryable=True, transient=True, classification_source="transport")
    for item in chain:
        response = getattr(item, "response", None)
        body = getattr(item, "body", None)
        headers = getattr(response, "headers", None)
        status = getattr(response, "status_code", None) or getattr(item, "http_status", None) or getattr(item, "status_code", None)
        if response is not None and body is None:
            try:
                body = response.json()
            except (ValueError, TypeError, AttributeError):
                pass
        if body is not None or status is not None:
            if isinstance(body, dict) and "error" not in body and any(k in body for k in ("type", "message")):
                body = {"error": body}
            failure = parse_provider_failure(provider, operation, status, body, headers)
            if failure:
                if failure.classification_source == "http_status_fallback":
                    response_headers = getattr(getattr(item, "response", None), "headers", None)
                    enriched = parse_provider_failure(
                        provider,
                        operation,
                        status,
                        {"error": {"message": safe_message(item)}},
                        response_headers,
                    )
                    if enriched is not None and enriched.classification_source != "http_status_fallback":
                        return enriched
                return failure
    if any(isinstance(item, (TypeError, AttributeError, AssertionError, KeyError)) for item in chain):
        item = chain[-1]
        return ProviderFailure(
            provider,
            operation,
            "application_internal",
            type(item).__name__,
            provider_error_message=safe_message(item),
            retryable=False,
            transient=False,
            network_attempted=False,
            operator_action_required=False,
            classification_source="message_fallback",
        )
    for item in chain:
        explicit = getattr(item, "classification", None)
        if explicit:
            cls = canonical_classification(explicit)
            network = getattr(item, "network_attempted", cls != "configuration")
            if cls == "configuration":
                network = False
            retryable = bool(getattr(item, "retryable", cls in RETRYABLE_CLASSES)) and network
            return ProviderFailure(provider, operation, cls, str(explicit), provider_error_message=safe_message(item), retryable=retryable, transient=retryable, network_attempted=network, operator_action_required=cls in OPERATOR_CLASSES, classification_source="sdk_typed_exception")
    fallback_text = " ".join(str(item) for item in chain).lower()
    if any(marker in fallback_text for marker in ("timed out", "timeout", "status code 408")):
        cls = "timeout"
    elif any(marker in fallback_text for marker in ("status code 5", "http 5", " 500", " 502", " 503", " 504", "server error", "overloaded")):
        cls = "provider_5xx"
    elif any(marker in fallback_text for marker in ("rate limit", "too many requests", "status code 429")):
        cls = "rate_limit"
    elif any(marker in fallback_text for marker in ("invalid api key", "authentication", "unauthorized", "status code 401")):
        cls = "auth"
    elif any(marker in fallback_text for marker in ("quota", "credit", "billing", "status code 402")):
        cls = "insufficient_balance_quota"
    elif any(marker in fallback_text for marker in ("configuration", "not configured", "api key for")):
        cls = "configuration"
    elif any(isinstance(e, (TypeError, AttributeError, AssertionError, KeyError)) for e in chain):
        cls = "application_internal"
    else:
        cls = "unknown"
    retryable = cls in RETRYABLE_CLASSES
    network_attempted = cls != "configuration"
    return ProviderFailure(provider, operation, cls, type(chain[-1]).__name__, provider_error_message=safe_message(chain[-1]), retryable=retryable, transient=retryable, network_attempted=network_attempted, operator_action_required=cls in OPERATOR_CLASSES, classification_source="message_fallback")


def attach_failure(exception: Exception, failure: ProviderFailure) -> Exception:
    exception.failure = failure
    exception.classification = failure.classification
    exception.retryable = failure.retryable
    exception.network_attempted = failure.network_attempted
    exception.http_status = failure.http_status
    exception.provider_code = failure.provider_error_code
    exception.provider_message = failure.provider_error_message
    exception.provider_error_type = failure.provider_error_type
    exception.detail_code = failure.detail_code
    exception.retry_after_seconds = failure.retry_after_seconds
    exception.request_id = failure.request_id
    exception.diagnostics = failure.diagnostics()
    return exception


def evaluate_request_group_retryability(attempts: Any) -> bool:
    eligible = False
    for attempt in attempts:
        get = attempt.get if isinstance(attempt, dict) else lambda k, default=None: getattr(attempt, k, default)
        if str(get("status", "")).lower() == "success":
            return False
        diagnostics = get("diagnostics") or get("diagnostics_json") or {}
        if isinstance(diagnostics, str):
            try:
                diagnostics = json.loads(diagnostics)
            except (ValueError, TypeError):
                diagnostics = {}
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        cls = canonical_classification(get("classification") or get("error_classification"))
        network = get("network_attempted", diagnostics.get("network_attempted", cls != "configuration"))
        retryable = get("retryable", diagnostics.get("retryable"))
        if retryable is None:
            retryable = cls in RETRYABLE_CLASSES
        eligible = eligible or (network is not False and retryable is True)
    return eligible
