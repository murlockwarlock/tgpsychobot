import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("BOT_TOKEN", "123456:TEST_BOT_TOKEN")

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
        ("image_other", "content_policy", False),
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


@pytest.mark.parametrize(
    ("code", "expected", "retryable"),
    [
        (401, "auth", False),
        (402, "insufficient_balance_quota", False),
        (429, "rate_limit", True),
        (500, "provider_5xx", True),
        (503, "provider_5xx", True),
    ],
)
def test_openrouter_embedded_numeric_code_uses_provider_code_without_replacing_transport_status(code, expected, retryable):
    failure = parse_provider_failure(
        "OpenRouter",
        200,
        {"error": {"code": code, "message": "upstream unavailable"}},
    )
    assert failure is not None
    assert failure.http_status == 200
    assert failure.provider_error_code == str(code)
    assert failure.classification == expected
    assert failure.classification_source == "provider_structured"
    assert failure.retryable is retryable


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


def test_openrouter_embedded_numeric_transient_code_retries_through_real_adapter():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                json={"error": {"code": 503, "message": "upstream unavailable"}},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "recovered"}, "finish_reason": "stop"}]},
        )

    with patch("provider_adapters.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(
            call_openrouter(
                "secret",
                AIRequestLayout(current_user_content="hello"),
                "openai/gpt-5.6-terra",
                transport=httpx.MockTransport(handler),
            )
        )

    assert result == "recovered"
    assert calls == 2


def test_perplexity_429_credits_exhausted_beats_generic_rate_limit():
    failure = parse_provider_failure(
        "Perplexity",
        429,
        {"error": {"message": "credits exhausted"}},
    )
    assert failure is not None
    assert failure.classification == "insufficient_balance_quota"
    assert failure.retryable is False
    assert failure.operator_action_required is True
    assert failure.classification_source == "message_fallback"


def test_new_provider_failures_reach_shared_group_retry_evaluator():
    openrouter = parse_provider_failure(
        "OpenRouter",
        200,
        {"error": {"code": 503, "message": "upstream unavailable"}},
    )
    perplexity = parse_provider_failure(
        "Perplexity",
        429,
        {"error": {"message": "credits exhausted"}},
    )
    assert evaluate_request_group_retryability([{
        "status": "error",
        "classification": openrouter.classification,
        "retryable": openrouter.retryable,
        "network_attempted": openrouter.network_attempted,
    }]) is True
    assert evaluate_request_group_retryability([{
        "status": "error",
        "classification": perplexity.classification,
        "retryable": perplexity.retryable,
        "network_attempted": perplexity.network_attempted,
    }]) is False


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


@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        ({"promptFeedback": {"blockReason": "SAFETY"}}, "content_policy"),
        ({"candidates": [{"finishReason": "MAX_TOKENS"}]}, "output_budget_exhausted"),
    ),
)
def test_gemini_generate_content_structured_http_200_semantics(payload, expected):
    failure = parse_provider_failure("Gemini", "chat", 200, payload, {"x-goog-request-id": "gemini-chat-1"})
    assert failure.classification == expected
    assert failure.retryable is False
    assert failure.request_id == "gemini-chat-1"
    assert failure.classification_source == "provider_structured"


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        ({"promptFeedback": {"blockReason": "SAFETY"}}, "content_policy"),
        ({"candidates": [{"finishReason": "MAX_TOKENS"}]}, "output_budget_exhausted"),
    ),
)
async def test_gemini_generate_content_real_runtime_uses_canonical_failure(monkeypatch, surface, payload, expected):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"x-goog-request-id": "gemini-runtime-1"}, json=payload)

    if surface == "telegram":
        import ai_integration as module

        monkeypatch.setattr(module, "_build_async_transport_from_env", lambda *_args, **_kwargs: httpx.MockTransport(handler))
        with pytest.raises((module.AIServiceError, module.AIResponseError)) as raised:
            await module._call_gemini_api(
                "key",
                "gemini-3.7-flash",
                [],
                "",
                "system",
                request_layout=AIRequestLayout(current_user_content="hello"),
            )
    else:
        from max_messenger_bot import ai as module

        monkeypatch.setattr(module, "_build_gemini_proxy_transport", lambda: httpx.MockTransport(handler))
        with pytest.raises((module.AIServiceError, module.AIResponseError)) as raised:
            await module._call_gemini(
                "key",
                "gemini-3.7-flash",
                [{"role": "user", "content": "hello"}],
                "system",
                0.7,
            )

    assert raised.value.failure.classification == expected
    assert raised.value.failure.retryable is False
    assert raised.value.failure.request_id == "gemini-runtime-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        ({"promptFeedback": {"blockReason": "SAFETY"}}, "content_policy"),
        ({"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": []}}]}, "output_budget_exhausted"),
    ),
)
async def test_gemini_vision_real_runtime_uses_same_canonical_failure(monkeypatch, surface, payload, expected):
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"x-goog-request-id": "gemini-vision-runtime-1"}, json=payload)

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    image_bytes = b"\x89PNG\r\n\x1a\n" + b"fixture"
    if surface == "telegram":
        import ai_integration as module

        monkeypatch.setattr(module.httpx, "AsyncClient", client_factory)
        with pytest.raises((module.AIServiceError, module.AIResponseError)) as raised:
            await module._call_gemini_vision("key", "gemini-3.7-flash", image_bytes, "describe")
    else:
        from max_messenger_bot import ai as module

        monkeypatch.setattr(module.httpx, "AsyncClient", client_factory)
        with pytest.raises((module.AIServiceError, module.AIResponseError)) as raised:
            await module._analyze_gemini(
                "key",
                "gemini-3.7-flash",
                image_bytes,
                "system",
                "describe",
                0.7,
            )

    assert raised.value.failure.classification == expected
    assert raised.value.failure.retryable is False
    assert raised.value.failure.request_id == "gemini-vision-runtime-1"


class _ImageConfigSession:
    def __init__(self, config):
        self.config = config

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, model, key):
        return self.config


class _OpenAIImageSDKError(Exception):
    def __init__(self, status: int, body: dict):
        super().__init__(body.get("error", {}).get("message", "OpenAI image error"))
        self.response = httpx.Response(status, json=body)
        self.body = body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body", "expected"),
    (
        (401, {"error": {"code": "invalid_api_key", "message": "invalid key"}}, "auth"),
        (429, {"error": {"code": "credit_balance_exhausted", "message": "credits exhausted"}}, "insufficient_balance_quota"),
        (500, {"error": {"type": "server_error", "message": "provider failed"}}, "provider_5xx"),
    ),
)
async def test_openai_image_generation_deterministic_physical_call_count(monkeypatch, status, body, expected):
    import ai_integration as module

    calls = []
    config = SimpleNamespace(openai_api_key="openai-key")
    monkeypatch.setattr(module, "async_session_maker", lambda: _ImageConfigSession(config))

    class Images:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            raise _OpenAIImageSDKError(status, body)

    class Client:
        def __init__(self, **kwargs):
            self.images = Images()

    monkeypatch.setattr(module, "AsyncOpenAI", Client)

    with pytest.raises((module.AIServiceError, module.InsufficientBalanceError)) as raised:
        await module.generate_openai_image("portrait illustration")

    assert len(calls) == 1
    assert raised.value.failure.classification == expected
    assert raised.value.failure.retryable is (expected == "provider_5xx")


@pytest.mark.asyncio
async def test_openai_image_generation_allows_only_documented_size_correction(monkeypatch):
    import ai_integration as module

    calls = []
    config = SimpleNamespace(openai_api_key="openai-key")
    monkeypatch.setattr(module, "async_session_maker", lambda: _ImageConfigSession(config))

    class Images:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise _OpenAIImageSDKError(
                    400,
                    {"error": {"code": "invalid_request", "message": "size 1024x1536 is unsupported"}},
                )
            return SimpleNamespace(data=[SimpleNamespace(url="https://image.example/result.png", b64_json=None)])

    class Client:
        def __init__(self, **kwargs):
            self.images = Images()

    monkeypatch.setattr(module, "AsyncOpenAI", Client)
    monkeypatch.setattr(module, "_select_image_generation_shape", lambda prompt: ("portrait", "1024x1536"))

    result = await module.generate_openai_image("portrait illustration")

    assert result == "https://image.example/result.png"
    assert [call["size"] for call in calls] == ["1024x1536", "1024x1024"]


@pytest.mark.asyncio
async def test_kie_image_generation_never_recreates_created_task(monkeypatch):
    import ai_integration as module

    real_client = httpx.AsyncClient
    calls = {"create": 0, "poll": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/jobs/createTask"):
            calls["create"] += 1
            return httpx.Response(200, json={"data": {"taskId": "task-image-1"}})
        if request.method == "GET" and request.url.path.endswith("/jobs/recordInfo"):
            calls["poll"] += 1
            return httpx.Response(200, json={"data": {"state": "fail", "failCode": 500, "failMsg": "internal error"}})
        raise AssertionError(f"unexpected KIE request: {request.method} {request.url}")

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", client_factory)

    with pytest.raises((module.AIServiceError, module.InsufficientBalanceError)) as raised:
        await module._call_kie_image_generation(
            "kie-key",
            "https://kie.example",
            "bytedance/seedream-v4-text-to-image",
            "draw a forest",
        )

    assert calls == {"create": 1, "poll": 1}
    assert raised.value.task_id == "task-image-1"
    assert raised.value.failure.classification == "provider_5xx"
    assert raised.value.diagnostics["task_id"] == "task-image-1"
    assert raised.value.diagnostics["retryable"] is True
    assert raised.value.diagnostics["classification_source"] == "provider_structured"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "message", "expected"),
    (
        (401, "invalid key", "auth"),
        (402, "insufficient credits", "insufficient_balance_quota"),
        (400, "content policy violation", "content_policy"),
    ),
)
async def test_kie_image_creation_failures_have_no_duplicate_task(monkeypatch, code, message, expected):
    import ai_integration as module

    real_client = httpx.AsyncClient
    create_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal create_calls
        if request.method == "POST" and request.url.path.endswith("/jobs/createTask"):
            create_calls += 1
            return httpx.Response(200, json={"code": code, "msg": message})
        raise AssertionError("poll/download must not run after task creation failure")

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", client_factory)

    with pytest.raises((module.AIServiceError, module.InsufficientBalanceError)) as raised:
        await module._call_kie_image_generation(
            "kie-key",
            "https://kie.example",
            "bytedance/seedream-v4-text-to-image",
            "draw a forest",
        )

    assert create_calls == 1
    assert raised.value.failure.classification == expected
