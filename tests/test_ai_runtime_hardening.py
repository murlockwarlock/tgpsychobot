from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from provider_models import (
    APP_DISABLED_OR_MIGRATED_MODELS,
    PROVIDER_CLAUDE,
    PROVIDER_DEEPSEEK,
    PROVIDER_GEMINI,
    PROVIDER_KIE,
    PROVIDER_OPENAI,
    RETIRED_UPSTREAM_MODELS,
    SELECTABLE_VISION_MODELS,
    ModelUnavailableError,
    ensure_model_available,
    get_selectable_models,
    validate_model_selection,
)


class _SessionContext:
    def __init__(self, config=None, user=None):
        self.config = config
        self.user = user
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, model, key):
        from database import AIConfig

        return self.config if model is AIConfig else None

    async def scalar(self, statement):
        return self.user

    async def execute(self, statement):
        return _Rows([])

    async def commit(self):
        self.committed = True


class _Rows:
    def __init__(self, rows):
        self._rows = list(rows)

    def all(self):
        return list(self._rows)

    def scalars(self):
        return self


class _MaxClient:
    def __init__(self):
        self.send_message = AsyncMock()


class _TelegramCallback:
    def __init__(self, data):
        self.data = data
        self.answers = []

    async def answer(self, text="", **kwargs):
        self.answers.append((text, kwargs))


class _HttpResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


class _HttpClient:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, *args, **kwargs):
        return self.response


def test_runtime_catalog_rejects_retired_disabled_unknown_and_cross_provider_models():
    invalid_values = (
        (PROVIDER_GEMINI, "gemini-2.0-flash"),
        (PROVIDER_OPENAI, "gpt-4o"),
        (PROVIDER_OPENAI, "gemini-3.7-flash"),
        (PROVIDER_KIE, "not-a-real-kie-model"),
    )
    for provider, model in invalid_values:
        with pytest.raises(ModelUnavailableError):
            ensure_model_available(provider, model, channel="chat")

    assert RETIRED_UPSTREAM_MODELS
    assert APP_DISABLED_OR_MIGRATED_MODELS


def test_deepseek_legacy_alias_is_normalized_before_validation():
    assert validate_model_selection(PROVIDER_DEEPSEEK, "deepseek-reasoner") == "deepseek-v4-flash"
    ensure_model_available(PROVIDER_DEEPSEEK, "deepseek-chat", channel="chat")


def test_gemini_preview_cannot_return_to_the_vision_picker_after_migration():
    assert "gemini-3-flash-preview" not in SELECTABLE_VISION_MODELS[PROVIDER_GEMINI]
    assert "gemini-3-flash-preview" not in get_selectable_models(PROVIDER_GEMINI, channel="vision")

    from sqlalchemy import create_engine, text
    from database import _migrate_ai_config_models

    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE ai_config (
                id INTEGER PRIMARY KEY,
                openai_model VARCHAR,
                claude_model VARCHAR,
                gemini_model VARCHAR,
                deepseek_model VARCHAR,
                vision_provider VARCHAR,
                vision_model VARCHAR,
                fallback_provider VARCHAR,
                fallback_model VARCHAR
            )
        """))
        conn.execute(text("""
            INSERT INTO ai_config (
                id, openai_model, claude_model, gemini_model, deepseek_model,
                vision_provider, vision_model, fallback_provider, fallback_model
            ) VALUES (
                1, 'gpt-5.6-terra', 'claude-sonnet-5', 'gemini-3.7-flash',
                'deepseek-v4-flash', 'Gemini', 'gemini-3-flash-preview', NULL, NULL
            )
        """))
        _migrate_ai_config_models(conn)
        migrated = conn.execute(text("SELECT vision_model FROM ai_config WHERE id = 1")).scalar_one()

    assert migrated == "gemini-3.7-flash"
    assert migrated in get_selectable_models(PROVIDER_GEMINI, channel="vision")


@pytest.mark.asyncio
async def test_telegram_legacy_callback_fails_closed_against_current_provider(monkeypatch):
    import handlers

    config = SimpleNamespace(
        fallback_provider=PROVIDER_OPENAI,
        fallback_model="gpt-5.6-terra",
    )
    session = _SessionContext(config=config)
    monkeypatch.setattr(handlers, "async_session_maker", lambda: session)

    callback = _TelegramCallback("save_fallback_model_gemini-3.7-flash")
    await handlers.save_fallback_model(callback)

    assert not session.committed
    assert config.fallback_provider == PROVIDER_OPENAI
    assert config.fallback_model == "gpt-5.6-terra"
    assert callback.answers[0][1]["show_alert"] is True


@pytest.mark.asyncio
async def test_max_provider_bound_and_legacy_callbacks_fail_closed(monkeypatch):
    from max_messenger_bot.services import admin_ai

    config = SimpleNamespace(
        fallback_provider=PROVIDER_GEMINI,
        fallback_model="gemini-3.7-flash",
        vision_provider=PROVIDER_CLAUDE,
        vision_model="claude-sonnet-5",
    )
    session = _SessionContext(config=config)
    client = _MaxClient()
    monkeypatch.setattr(admin_ai, "async_session_maker", lambda: session)
    monkeypatch.setattr(admin_ai, "show_keys", AsyncMock())

    await admin_ai.save_fallback_model(client, 1, PROVIDER_OPENAI, "gemini-3.7-flash")
    await admin_ai.set_vision_model(client, 1, "gemini-3.7-flash")

    assert not session.committed
    assert config.fallback_provider == PROVIDER_GEMINI
    assert config.fallback_model == "gemini-3.7-flash"
    assert config.vision_provider == PROVIDER_CLAUDE
    assert config.vision_model == "claude-sonnet-5"
    assert client.send_message.await_count == 2


@pytest.mark.asyncio
async def test_max_temperature_zero_reaches_primary_and_fallback(monkeypatch):
    from max_messenger_bot import ai

    user = SimpleNamespace(
        id=1,
        current_topic_id=None,
        current_dialogue_id=1,
        current_topic=None,
        response_length="normal",
        name="Test",
        first_name="Test",
        gender="unknown",
    )
    config = SimpleNamespace(
        provider="Gemini",
        gemini_api_key="primary-key",
        gemini_model="gemini-3.7-flash",
        openai_api_key="fallback-key",
        openai_model="gpt-5.6-terra",
        fallback_provider="OpenAI",
        fallback_model="gpt-5.6-terra",
        allow_fallback=True,
        temperature=0.0,
        system_prompt="SYSTEM",
        shared_prompt_block="",
        context_limit_first=2,
        context_limit_recent=10,
        memory_mode="reset",
        preserve_topic_context=False,
    )
    session = _SessionContext(config=config, user=user)
    monkeypatch.setattr(ai, "async_session_maker", lambda: session)
    monkeypatch.setattr(ai, "build_runtime_automation_context", AsyncMock(return_value=""))
    primary = AsyncMock(side_effect=ai.AIServiceError("primary failed"))
    fallback = AsyncMock(return_value="fallback answer")
    monkeypatch.setattr(ai, "_call_gemini", primary)
    monkeypatch.setattr(ai, "_call_openai", fallback)

    result = await ai.get_ai_response(1, "question")

    assert result == "fallback answer"
    assert primary.await_args.args[4] == 0.0
    assert fallback.await_args.args[3] == 0.0


@pytest.mark.asyncio
async def test_timeout_reaches_openai_gemini_deepseek_claude_and_kie(monkeypatch):
    import ai_integration

    openai_client = AsyncMock()
    openai_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
    )
    with patch.object(ai_integration, "AsyncOpenAI", return_value=openai_client) as factory:
        await ai_integration._call_openai_api(
            "key", "gpt-5.6-terra", [], "", "", timeout=37.0
        )
    assert factory.call_args.kwargs["timeout"] == 37.0

    gemini_http = _HttpClient(_HttpResponse({
        "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
    }))
    with patch.object(ai_integration.httpx, "AsyncClient", return_value=gemini_http) as factory:
        await ai_integration._call_gemini_api(
            "key", "gemini-3.7-flash", [{"role": "user", "content": "question"}], "", "", timeout=37.0
        )
    assert factory.call_args.kwargs["timeout"] == 37.0

    deepseek_transport = Mock()
    deepseek_client = AsyncMock()
    deepseek_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
    )
    with (
        patch.object(ai_integration.httpx, "AsyncClient", return_value=deepseek_transport) as http_factory,
        patch.object(ai_integration, "AsyncOpenAI", return_value=deepseek_client),
    ):
        await ai_integration._call_deepseek_api(
            "key", "deepseek-v4-flash", [], "", "", timeout=37.0
        )
    assert http_factory.call_args.kwargs["timeout"] == 37.0

    claude_client = AsyncMock()
    claude_client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text="ok")]
    )
    with patch.object(ai_integration.anthropic, "AsyncAnthropic", return_value=claude_client) as factory:
        await ai_integration._call_claude_api(
            "key", "claude-haiku-4-5-20251001", [], "", "", timeout=37.0
        )
    assert factory.call_args.kwargs["timeout"] == 37.0

    kie_http = _HttpClient(_HttpResponse({
        "choices": [{"message": {"content": "ok"}}],
    }))
    with patch.object(ai_integration.httpx, "AsyncClient", return_value=kie_http) as factory:
        await ai_integration._call_kie_chat(
            "key", "https://kie.example", "gemini-3-flash", [], "", "", timeout=37.0
        )
    assert factory.call_args.kwargs["timeout"] == 37.0


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
async def test_kie_chat_rejects_cross_provider_model_before_http(surface):
    if surface == "telegram":
        import ai_integration as module

        call = module._call_kie_chat
        args = ("key", "https://kie.example", "gpt-5.6-terra", [], "", "")
    else:
        from max_messenger_bot import ai as module

        call = module._call_kie_text_chat
        args = ("key", "https://kie.example", "gpt-5.6-terra", [], "", 0.0)

    with patch.object(module.httpx, "AsyncClient", side_effect=AssertionError("HTTP must not start")) as factory:
        with pytest.raises(ModelUnavailableError):
            await call(*args)
    assert factory.call_count == 0
