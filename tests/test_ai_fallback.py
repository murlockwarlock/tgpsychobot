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


def _config(*, fallback_enabled, fallback_configured=None):
    if fallback_configured is None:
        fallback_configured = fallback_enabled
    return SimpleNamespace(
        provider="gemini",
        system_prompt="SYSTEM",
        prompt_mode="text",
        prompt_filename=None,
        gemini_api_key="primary-key",
        gemini_model="primary-model",
        openai_api_key="fallback-key",
        openai_model="fallback-model",
        fallback_provider="openai" if fallback_configured else None,
        fallback_model="fallback-model" if fallback_configured else None,
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


async def _run_telegram(
    monkeypatch,
    primary_outcome,
    *,
    fallback_enabled,
    fallback_configured=None,
    fallback_call=None,
):
    import ai_integration

    config = _config(
        fallback_enabled=fallback_enabled,
        fallback_configured=fallback_configured,
    )
    session = _Session(ai_integration, _user(), config, telegram=True)
    primary = (
        AsyncMock(side_effect=primary_outcome)
        if isinstance(primary_outcome, BaseException)
        else AsyncMock(return_value=primary_outcome)
    )
    fallback = fallback_call if fallback_call is not None else AsyncMock(return_value="fallback answer")
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


async def _run_max(
    monkeypatch,
    primary_outcome,
    *,
    fallback_enabled,
    fallback_configured=None,
    fallback_call=None,
):
    from max_messenger_bot import ai

    config = _config(
        fallback_enabled=fallback_enabled,
        fallback_configured=fallback_configured,
    )
    session = _Session(ai, _user(), config, telegram=False)
    primary = (
        AsyncMock(side_effect=primary_outcome)
        if isinstance(primary_outcome, BaseException)
        else AsyncMock(return_value=primary_outcome)
    )
    fallback = fallback_call if fallback_call is not None else AsyncMock(return_value="fallback answer")
    monkeypatch.setattr(ai, "async_session_maker", lambda: _SessionContext(session))
    monkeypatch.setattr(ai, "_call_gemini", primary)
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
@pytest.mark.parametrize("primary_text", ("primary answer", "Ошибка в вычислении пользователя..."))
async def test_primary_success_does_not_invoke_fallback(monkeypatch, surface, primary_text):
    runner = _run_telegram if surface == "telegram" else _run_max

    result, primary, fallback = await runner(
        monkeypatch,
        primary_text,
        fallback_enabled=True,
    )

    assert result == primary_text
    primary.assert_awaited_once()
    fallback.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
@pytest.mark.parametrize("primary_text", (None, "", " \t\n"))
async def test_empty_primary_response_attempts_fallback(monkeypatch, surface, primary_text):
    runner = _run_telegram if surface == "telegram" else _run_max

    result, primary, fallback = await runner(
        monkeypatch,
        primary_text,
        fallback_enabled=True,
    )

    assert result == "fallback answer"
    primary.assert_awaited_once()
    fallback.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
@pytest.mark.parametrize("fallback_text", (None, "", " \t\n"))
async def test_empty_fallback_response_is_a_final_failure(monkeypatch, surface, fallback_text):
    module = _module(surface)
    runner = _run_telegram if surface == "telegram" else _run_max
    fallback = AsyncMock(return_value=fallback_text)

    with pytest.raises(module.AIServiceError, match="недоступны"):
        await runner(
            monkeypatch,
            module.AIServiceError("primary provider failed"),
            fallback_enabled=True,
            fallback_call=fallback,
        )

    fallback.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
async def test_empty_primary_response_respects_disabled_fallback(monkeypatch, surface):
    module = _module(surface)
    runner = _run_telegram if surface == "telegram" else _run_max
    fallback = AsyncMock(return_value="should not be used")

    with pytest.raises(module.AIServiceError):
        await runner(
            monkeypatch,
            " \t\n",
            fallback_enabled=False,
            fallback_configured=True,
            fallback_call=fallback,
        )

    fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_telegram_gemini_empty_or_blocked_response_raises(monkeypatch):
    import ai_integration

    client = _KieClient(_KieResponse({"candidates": []}, 200))
    monkeypatch.setattr(ai_integration.httpx, "AsyncClient", lambda *args, **kwargs: client)

    with pytest.raises(ai_integration.AIServiceError):
        await ai_integration._call_gemini_api(
            "gemini-key",
            "gemini-model",
            [SimpleNamespace(role="user", content="question")],
            "",
            "SYSTEM",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
async def test_fallback_toggle_controls_configured_fallback(monkeypatch, surface):
    module = _module(surface)
    runner = _run_telegram if surface == "telegram" else _run_max
    primary_error = module.AIServiceError("primary provider failed")
    disabled_fallback = AsyncMock(return_value="should not be used")

    with pytest.raises(module.AIServiceError, match="primary provider failed"):
        await runner(
            monkeypatch,
            primary_error,
            fallback_enabled=False,
            fallback_configured=True,
            fallback_call=disabled_fallback,
        )
    disabled_fallback.assert_not_awaited()

    enabled_fallback = AsyncMock(return_value="fallback answer")
    result, _, fallback = await runner(
        monkeypatch,
        primary_error,
        fallback_enabled=True,
        fallback_configured=True,
        fallback_call=enabled_fallback,
    )

    assert result == "fallback answer"
    assert fallback is enabled_fallback
    enabled_fallback.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
async def test_without_configured_fallback_primary_error_is_preserved(monkeypatch, surface):
    module = _module(surface)
    runner = _run_telegram if surface == "telegram" else _run_max
    primary_error = module.AIServiceError("primary provider failed")

    with pytest.raises(module.AIServiceError, match="primary provider failed") as raised:
        await runner(monkeypatch, primary_error, fallback_enabled=False)

    assert raised.value is primary_error
