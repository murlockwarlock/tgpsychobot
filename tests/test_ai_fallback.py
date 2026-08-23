import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")


class _Result:
    def __init__(self, *, scalar=None, rows=()):
        self._scalar = scalar
        self._rows = list(rows)

    def scalar_one_or_none(self):
        return self._scalar

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)


class _Session:
    def __init__(self, module, user, ai_config, *, telegram):
        self.module = module
        self.user = user
        self.ai_config = ai_config
        self.telegram = telegram
        self._execute_results = (
            [_Result(scalar=user), _Result(rows=[]), _Result(rows=[])]
            if telegram
            else [_Result(rows=[]), _Result(rows=[])]
        )
        self._scalar_calls = 0
        self.added = []

    async def execute(self, statement):
        assert self._execute_results, "unexpected database execute in fallback test"
        return self._execute_results.pop(0)

    async def scalar(self, statement):
        self._scalar_calls += 1
        if not self.telegram and self._scalar_calls == 1:
            return self.user
        return None

    async def get(self, model, key):
        if model is self.module.AIConfig:
            return self.ai_config
        return None

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        return None


class _SessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *args):
        return False


class _KieResponse:
    def __init__(self, payload, status_code):
        self.status_code = status_code
        self.text = json.dumps(payload)
        self._payload = payload

    def json(self):
        return self._payload


class _KieClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        return self.response


def _module(surface):
    if surface == "telegram":
        import ai_integration

        return ai_integration
    from max_messenger_bot import ai

    return ai


def _failure(module, kind):
    if kind == "service":
        return module.AIServiceError("primary provider failed")
    if kind == "balance":
        return module.InsufficientBalanceError("KIE API Error: Insufficient credits")
    if kind == "timeout":
        return TimeoutError("primary provider timed out")
    raise AssertionError(f"unknown failure kind: {kind}")


def _config(*, fallback_enabled):
    return SimpleNamespace(
        provider="gemini",
        system_prompt="SYSTEM",
        prompt_mode="text",
        prompt_filename=None,
        gemini_api_key="primary-key",
        gemini_model="primary-model",
        openai_api_key="fallback-key",
        openai_model="fallback-model",
        fallback_provider="openai" if fallback_enabled else None,
        fallback_model="fallback-model" if fallback_enabled else None,
        allow_fallback=fallback_enabled,
        temperature=0.7,
        context_limit_first=2,
        context_limit_recent=10,
        memory_mode="reset",
        preserve_topic_context=False,
        shared_prompt_block="",
        service_prompt_block=None,
    )


def _user():
    return SimpleNamespace(
        id=123,
        username="tester",
        full_name="Test User",
        name="Test User",
        first_name="Test",
        gender="unknown",
        age=None,
        response_length="normal",
        current_dialogue_id=1,
        current_topic_id=None,
        current_topic=None,
        subscription=None,
        ai_debug_enabled=False,
    )


async def _run_telegram(monkeypatch, primary_outcome, *, fallback_enabled):
    import ai_integration

    config = _config(fallback_enabled=fallback_enabled)
    session = _Session(ai_integration, _user(), config, telegram=True)
    primary = (
        AsyncMock(side_effect=primary_outcome)
        if isinstance(primary_outcome, BaseException)
        else AsyncMock(return_value=primary_outcome)
    )
    fallback = AsyncMock(return_value="fallback answer")
    monkeypatch.setattr(ai_integration, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(ai_integration, "build_runtime_context", AsyncMock(return_value="runtime"))
    monkeypatch.setattr(ai_integration, "_call_gemini_api", primary)
    monkeypatch.setattr(ai_integration, "_call_openai_api", fallback)
    result = await ai_integration.get_ai_response(
        123,
        "question",
        "Test User",
        "unknown",
        include_test_context=False,
    )
    return result, primary, fallback


async def _run_max(monkeypatch, primary_outcome, *, fallback_enabled):
    from max_messenger_bot import ai

    config = _config(fallback_enabled=fallback_enabled)
    session = _Session(ai, _user(), config, telegram=False)
    primary = (
        AsyncMock(side_effect=primary_outcome)
        if isinstance(primary_outcome, BaseException)
        else AsyncMock(return_value=primary_outcome)
    )
    fallback = AsyncMock(return_value="fallback answer")
    monkeypatch.setattr(ai, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(ai, "_dispatch_provider", primary)
    monkeypatch.setattr(ai, "_call_openai", fallback)
    result = await ai.get_ai_response(123, "question")
    return result, primary, fallback


async def _run_kie_primary_with_fallback(monkeypatch, surface, payload, status_code):
    module = _module(surface)
    config = _config(fallback_enabled=True)
    config.provider = "kie"
    config.kie_api_key = "primary-key"
    config.kie_model = "grok-4-3"
    config.kie_base_url = "https://api.example"
    user = _user()
    session = _Session(module, user, config, telegram=surface == "telegram")
    client = _KieClient(_KieResponse(payload, status_code))
    fallback = AsyncMock(return_value="fallback answer")

    monkeypatch.setattr(module, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(module, "_call_openai_api" if surface == "telegram" else "_call_openai", fallback)
    if surface == "telegram":
        monkeypatch.setattr(module, "build_runtime_context", AsyncMock(return_value="runtime"))
        result = await module.get_ai_response(
            123,
            "question",
            "Test User",
            "unknown",
            include_test_context=False,
        )
    else:
        result = await module.get_ai_response(123, "question")
    return result, client, fallback


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
@pytest.mark.parametrize("failure_kind", ("service", "balance", "timeout"))
async def test_any_primary_failure_attempts_fallback_and_returns_its_answer(
    monkeypatch, surface, failure_kind
):
    module = _module(surface)
    runner = _run_telegram if surface == "telegram" else _run_max

    result, primary, fallback = await runner(
        monkeypatch,
        _failure(module, failure_kind),
        fallback_enabled=True,
    )

    assert result == "fallback answer"
    primary.assert_awaited_once()
    fallback.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
@pytest.mark.parametrize(
    ("status_code", "payload"),
    (
        (402, {"msg": "Insufficient credits"}),
        (200, {"code": 402, "msg": "Insufficient credits"}),
    ),
)
async def test_kie_credit_failure_is_classified_and_still_uses_fallback(
    monkeypatch, surface, status_code, payload
):
    result, client, fallback = await _run_kie_primary_with_fallback(
        monkeypatch,
        surface,
        payload,
        status_code,
    )

    assert result == "fallback answer"
    assert len(client.calls) == 1
    fallback.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
async def test_primary_success_does_not_invoke_fallback(monkeypatch, surface):
    runner = _run_telegram if surface == "telegram" else _run_max

    result, primary, fallback = await runner(
        monkeypatch,
        "primary answer",
        fallback_enabled=True,
    )

    assert result == "primary answer"
    primary.assert_awaited_once()
    fallback.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
async def test_without_configured_fallback_primary_error_is_preserved(monkeypatch, surface):
    module = _module(surface)
    runner = _run_telegram if surface == "telegram" else _run_max
    primary_error = module.AIServiceError("primary provider failed")

    with pytest.raises(module.AIServiceError, match="primary provider failed") as raised:
        await runner(monkeypatch, primary_error, fallback_enabled=False)

    assert raised.value is primary_error
