import asyncio

import httpx
import pytest

from provider_failures import (
    evaluate_request_group_retryability,
    parse_provider_failure,
    safe_message,
)
from provider_adapters import ProviderAdapterError, call_openrouter
from ai_request_context import AIRequestLayout


@pytest.mark.parametrize(
    ("code", "expected", "retryable"),
    [
        ("rate_limit_error", "rate_limit", True),
        ("slow_down", "rate_limit", True),
        ("credit_balance_exhausted", "insufficient_balance_quota", False),
        ("organization_spend_limit_exceeded", "insufficient_balance_quota", False),
        ("project_spend_limit_exceeded", "insufficient_balance_quota", False),
        ("organization_usage_limit_exceeded", "insufficient_balance_quota", False),
        ("invalid_api_key", "auth", False),
        ("ip_not_authorized", "forbidden_geo", False),
    ],
)
def test_openai_structured_codes_beat_http_status(code, expected, retryable):
    failure = parse_provider_failure(
        "OpenAI", 429 if code not in {"invalid_api_key", "ip_not_authorized"} else 401,
        {"error": {"type": "insufficient_quota" if "limit" in code or "balance" in code else "rate_limit_error", "code": code, "message": code}},
    )
    assert failure.classification == expected
    assert failure.detail_code == code
    assert failure.retryable is retryable


@pytest.mark.parametrize(
    ("code", "expected", "retryable"),
    [
        ("invalid_request_error", "invalid_request", False),
        ("authentication_error", "auth", False),
        ("billing_error", "insufficient_balance_quota", False),
        ("permission_error", "permission", False),
        ("not_found_error", "invalid_model_or_resource", False),
        ("request_too_large", "context_limit", False),
        ("rate_limit_error", "rate_limit", True),
        ("api_error", "provider_5xx", True),
        ("timeout_error", "timeout", True),
        ("overloaded_error", "provider_overloaded", True),
    ],
)
def test_anthropic_structured_types(code, expected, retryable):
    failure = parse_provider_failure("Claude", 429 if "rate" in code else 500, {"error": {"type": code, "message": code}})
    assert (failure.classification, failure.retryable) == (expected, retryable)


@pytest.mark.parametrize(
    ("code", "expected", "retryable"),
    [
        ("invalid_request", "invalid_request", False),
        ("failed_precondition", "configuration", False),
        ("authentication", "auth", False),
        ("payment_required", "insufficient_balance_quota", False),
        ("permission_denied", "permission", False),
        ("model_not_found", "invalid_model_or_resource", False),
        ("rate_limit_exceeded", "rate_limit", True),
        ("quota_exceeded", "insufficient_balance_quota", False),
        ("service_unavailable", "provider_unavailable", True),
        ("deadline_exceeded", "timeout", True),
        ("safety", "content_policy", False),
        ("no_image", "provider_rejection", False),
    ],
)
def test_gemini_codes_are_explicit(code, expected, retryable):
    failure = parse_provider_failure("Gemini", 400, {"error": {"code": code, "status": code, "message": code}})
    assert (failure.classification, failure.retryable) == (expected, retryable)


@pytest.mark.parametrize("status,expected,retryable", [(400, "invalid_request", False), (401, "auth", False), (402, "insufficient_balance_quota", False), (422, "invalid_request", False), (429, "rate_limit", True), (500, "provider_5xx", True), (503, "provider_overloaded", True)])
def test_deepseek_statuses(status, expected, retryable):
    failure = parse_provider_failure("DeepSeek", status, {"error": {"message": "documented"}})
    assert (failure.classification, failure.retryable) == (expected, retryable)


@pytest.mark.parametrize("error_type,expected,retryable", [
    ("authentication", "auth", False),
    ("permission_denied", "permission", False),
    ("payment_required", "insufficient_balance_quota", False),
    ("rate_limit_exceeded", "rate_limit", True),
    ("provider_overloaded", "provider_overloaded", True),
    ("provider_unavailable", "provider_unavailable", True),
    ("invalid_prompt", "invalid_request", False),
    ("context_length_exceeded", "context_limit", False),
    ("content_policy_violation", "content_policy", False),
    ("server", "provider_5xx", True),
    ("timeout", "timeout", True),
    ("unmapped", "unknown", False),
])
def test_openrouter_embedded_200_errors(error_type, expected, retryable):
    failure = parse_provider_failure("OpenRouter", 200, {"error": {"code": 502, "message": "provider result", "metadata": {"error_type": error_type}}})
    assert (failure.classification, failure.detail_code, failure.retryable) == (expected, error_type, retryable)


def test_openrouter_choice_error_and_inflight_budget():
    choice = parse_provider_failure("OpenRouter", 200, {"choices": [{"finish_reason": "error", "error": {"metadata": {"error_type": "provider_overloaded"}}}]})
    assert choice.classification == "provider_overloaded"
    budget = parse_provider_failure("OpenRouter", 402, {"error": {"metadata": {"error_type": "payment_required", "limit_source": "openrouter_in_flight_budget"}}}, {"Retry-After": "3"})
    assert budget.retryable is True
    assert budget.retry_after_seconds == 3


def test_openrouter_adapter_rejects_http_200_structured_error_before_empty_response():
    def handler(request):
        return httpx.Response(
            200,
            headers={"x-request-id": "or-request-123"},
            json={
                "error": {
                    "code": 503,
                    "message": "upstream provider overloaded",
                    "metadata": {"error_type": "provider_overloaded"},
                }
            },
        )

    layout = AIRequestLayout(current_user_content="hello")
    capture = {}
    with pytest.raises(ProviderAdapterError) as raised:
        asyncio.run(
            call_openrouter(
                "secret",
                layout,
                "openai/gpt-5.6-terra",
                transport=httpx.MockTransport(handler),
                request_capture=capture,
            )
        )

    error = raised.value
    assert error.failure.classification == "provider_overloaded"
    assert error.failure.retryable is True
    assert error.failure.request_id == "or-request-123"
    assert error.failure.provider_error_type == "provider_overloaded"
    assert capture["classification"] == "provider_overloaded"
    assert capture["request_id"] == "or-request-123"


@pytest.mark.parametrize("code,expected", [(401, "auth"), (402, "insufficient_balance_quota"), (422, "invalid_request"), (429, "rate_limit"), (500, "provider_5xx")])
def test_kie_http_200_body_codes(code, expected):
    failure = parse_provider_failure("KIE", 200, {"code": code, "msg": "provider message"})
    assert failure.http_status == 200
    assert failure.provider_error_code == str(code)
    assert failure.classification == expected


def test_kie_error_envelope_and_task_id_are_preserved():
    failure = parse_provider_failure("KIE", 200, {"error": {"type": "task_error", "message": "failed"}, "data": {"taskId": "task-123"}})
    assert failure.classification == "unknown"
    assert failure.request_id == "task-123"


def test_deepgram_auth_permission_and_request_id():
    auth = parse_provider_failure("Deepgram", 401, {"err_code": "INVALID_AUTH", "err_msg": "bad", "request_id": "req-auth"})
    permission = parse_provider_failure("Deepgram", 401, {"err_code": "INSUFFICIENT_PERMISSIONS", "err_msg": "forbidden", "request_id": "req-perm"})
    assert (auth.classification, auth.request_id) == ("auth", "req-auth")
    assert (permission.classification, permission.request_id) == ("permission", "req-perm")


@pytest.mark.parametrize("status,expected", [("failed", "unknown"), ("cancelled", "cancelled"), ("incomplete", "invalid_response"), ("queued", "invalid_response"), ("in_progress", "invalid_response")])
def test_perplexity_agent_terminal_statuses(status, expected):
    body = {"status": status, "error": {"type": "provider_error", "code": "undocumented", "message": "status"}}
    failure = parse_provider_failure("Perplexity", "agent", 200, body)
    assert failure.classification == expected
    assert failure.retryable is False


def test_group_retry_uses_any_executed_attempt_and_success_wins():
    assert evaluate_request_group_retryability([{"classification": "timeout"}, {"classification": "auth"}]) is True
    assert evaluate_request_group_retryability([{"classification": "auth"}, {"classification": "permission"}]) is False
    assert evaluate_request_group_retryability([{"classification": "timeout", "network_attempted": False}, {"classification": "auth"}]) is False
    assert evaluate_request_group_retryability([{"classification": "timeout"}, {"status": "success"}]) is False


def test_diagnostics_redact_secrets():
    message = safe_message("Authorization: Bearer sk-secret api_key=secret-value https://user:pass@example.test")
    assert "sk-secret" not in message
    assert "secret-value" not in message
    assert "user:pass" not in message


def test_application_type_errors_and_unknown_provider_errors_fail_closed():
    from provider_failures import normalize_exception

    app_failure = normalize_exception(TypeError("quota parser invariant"), "OpenAI")
    assert app_failure.classification == "application_internal"
    assert app_failure.retryable is False
    unknown_failure = normalize_exception(RuntimeError("provider returned an undocumented envelope"), "OpenAI")
    assert unknown_failure.classification == "unknown"
    assert unknown_failure.retryable is False
