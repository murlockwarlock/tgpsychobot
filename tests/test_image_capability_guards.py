from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from provider_models import (
    ModelUnavailableError,
    PROVIDER_GEMINI,
    PROVIDER_KIE,
    PROVIDER_OPENAI,
    get_default_model,
    get_selectable_models,
)


class _SessionContext:
    def __init__(self, config):
        self.config = config

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, model, key):
        return self.config


def _image_config(*, provider: str, model: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        image_generation_provider=provider,
        image_generation_model=model,
        image_edit_provider=provider,
        image_edit_model=model,
        vision_provider=PROVIDER_GEMINI,
        gemini_api_key="gemini-test-key",
        kie_api_key="kie-test-key",
        openai_api_key="openai-test-key",
    )


def _surface_module(surface: str):
    if surface == "telegram":
        import ai_integration

        return ai_integration
    from max_messenger_bot import ai

    return ai


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
async def test_gemini_image_generation_invalid_model_fails_before_http(monkeypatch, surface):
    module = _surface_module(surface)
    config = _image_config(provider=PROVIDER_GEMINI, model="gemini-3.7-flash")
    http_client_factory = Mock(side_effect=AssertionError("HTTP client must not be created"))

    monkeypatch.setattr(module, "async_session_maker", lambda: _SessionContext(config))
    monkeypatch.setattr(module.httpx, "AsyncClient", http_client_factory)

    with pytest.raises(ModelUnavailableError):
        await module.generate_image("test prompt")

    assert http_client_factory.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
async def test_gemini_image_edit_invalid_model_fails_before_http(monkeypatch, surface):
    module = _surface_module(surface)
    config = _image_config(provider=PROVIDER_GEMINI, model="gemini-3.7-flash")
    http_client_factory = Mock(side_effect=AssertionError("HTTP client must not be created"))

    monkeypatch.setattr(module, "async_session_maker", lambda: _SessionContext(config))
    monkeypatch.setattr(module.httpx, "AsyncClient", http_client_factory)

    with pytest.raises(ModelUnavailableError):
        await module.edit_image("test prompt", b"image-bytes")

    assert http_client_factory.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
@pytest.mark.parametrize(
    ("operation", "model"),
    (
        ("generate", "imagen-4.0-generate-001"),
        ("edit", "gemini-3-pro-image-preview"),
    ),
)
async def test_retired_stored_google_image_model_fails_before_http(
    monkeypatch, surface, operation, model
):
    module = _surface_module(surface)
    config = _image_config(provider=PROVIDER_KIE, model=model)
    http_client_factory = Mock(side_effect=AssertionError("HTTP client must not be created"))

    monkeypatch.setattr(module, "async_session_maker", lambda: _SessionContext(config))
    monkeypatch.setattr(module.httpx, "AsyncClient", http_client_factory)

    with pytest.raises(ModelUnavailableError):
        if operation == "generate":
            await module.generate_image("test prompt")
        else:
            await module.edit_image("test prompt", b"image-bytes")

    assert http_client_factory.call_count == 0


@pytest.mark.parametrize("channel", ("image_gen", "image_edit"))
def test_gemini_image_defaults_use_direct_image_model(channel):
    assert get_default_model(PROVIDER_GEMINI, channel=channel) == "gemini-3.1-flash-image"


def test_supported_image_defaults_are_catalog_defaults():
    assert get_default_model(PROVIDER_OPENAI, channel="image_gen") == get_selectable_models(
        PROVIDER_OPENAI, channel="image_gen"
    )[0]
    assert get_default_model(PROVIDER_KIE, channel="image_gen") == get_selectable_models(
        PROVIDER_KIE, channel="image_gen"
    )[0]
    assert get_default_model(PROVIDER_KIE, channel="image_edit") == get_selectable_models(
        PROVIDER_KIE, channel="image_edit"
    )[0]


def _telegram_callbacks(markup) -> set[str]:
    return {
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if button.callback_data
    }


def _max_callbacks(markup) -> set[str]:
    return {
        button["payload"]
        for row in markup[0]["payload"]["buttons"]
        for button in row
        if button.get("type") == "callback"
    }


def test_telegram_and_max_image_admin_buttons_still_render():
    import keyboards
    from max_messenger_bot.services import admin_ai

    telegram_markup = keyboards.ai_keys_models_keyboard(
        current_transcription_provider="OpenAI",
        context_first=2,
        context_recent=10,
        current_vision_provider=PROVIDER_GEMINI,
        current_vision_model=get_default_model(PROVIDER_GEMINI, channel="vision"),
        image_generation_provider=PROVIDER_OPENAI,
        image_generation_model=get_default_model(PROVIDER_OPENAI, channel="image_gen"),
        image_edit_provider=PROVIDER_KIE,
        image_edit_model=get_default_model(PROVIDER_KIE, channel="image_edit"),
        kie_credit_alert_threshold=0,
    )
    telegram_callbacks = _telegram_callbacks(telegram_markup)
    assert {
        "admin_toggle_image_generation",
        "admin_change_image_generation_model",
        "admin_toggle_image_edit",
        "admin_change_image_edit_model",
    } <= telegram_callbacks

    max_config = SimpleNamespace(
        deepseek_api_key=None,
        claude_api_key=None,
        gemini_api_key=None,
        openai_api_key=None,
        kie_api_key=None,
        kie_credit_alert_threshold=0,
        transcription_provider="OpenAI",
        max_voice_duration_sec=180,
        vision_provider=PROVIDER_GEMINI,
        vision_model=get_default_model(PROVIDER_GEMINI, channel="vision"),
        allow_image_generation=True,
        image_generation_provider=PROVIDER_OPENAI,
        image_generation_model=get_default_model(PROVIDER_OPENAI, channel="image_gen"),
        allow_image_edit=True,
        image_edit_provider=PROVIDER_KIE,
        image_edit_model=get_default_model(PROVIDER_KIE, channel="image_edit"),
        allow_fallback=False,
        fallback_provider=None,
        fallback_model=None,
        context_limit_first=2,
        context_limit_recent=10,
        temperature=0.7,
        memory_mode="reset",
        use_proxy=True,
    )
    max_callbacks = _max_callbacks(admin_ai._build_keys_keyboard(max_config))
    assert {
        "admin_ai_toggle_image_generation",
        "admin_ai_image_generation_models",
        "admin_ai_toggle_image_edit",
        "admin_ai_image_edit_models",
    } <= max_callbacks
