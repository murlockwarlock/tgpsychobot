from __future__ import annotations

import base64
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import gemini_image
from provider_models import (
    ModelUnavailableError,
    PROVIDER_GEMINI,
    PROVIDER_KIE,
    PROVIDER_OPENAI,
    get_default_model,
    get_selectable_models,
)


API_KEY = "gemini-test-secret"
GEN_MODEL = "gemini-3.1-flash-image"
EDIT_MODEL = "gemini-3-pro-image"
SOURCE_PNG = b"\x89PNG\r\n\x1a\nsource-image"
RESULT_BYTES = b"exact-generated-image-bytes"


class _Response:
    def __init__(self, status_code=200, payload=None, json_error: Exception | None = None):
        self.status_code = status_code
        self.payload = payload
        self.json_error = json_error

    def json(self):
        if self.json_error:
            raise self.json_error
        return self.payload


class _AsyncClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def _image_response(*, image_data=RESULT_BYTES, include_text=True):
    content = []
    if include_text:
        content.append({"type": "text", "text": "I created the image."})
    content.append({"type": "image", "mime_type": "image/png", "data": base64.b64encode(image_data).decode()})
    return {"status": "completed", "steps": [{"type": "model_output", "content": content}]}


def _install_client(monkeypatch, response):
    client = _AsyncClient(response)
    monkeypatch.setattr(gemini_image.httpx, "AsyncClient", lambda **kwargs: client)
    return client


@pytest.mark.asyncio
async def test_gemini_generation_uses_interactions_endpoint_auth_and_text_input(monkeypatch):
    client = _install_client(monkeypatch, _Response(payload=_image_response(include_text=False)))

    result = await gemini_image.generate_image(API_KEY, GEN_MODEL, "draw a moonlit forest")

    assert result == RESULT_BYTES
    url, request = client.calls[0]
    assert url == gemini_image.GEMINI_INTERACTIONS_ENDPOINT
    assert ":predict" not in url
    assert "generateContent" not in url
    assert request["headers"] == {
        "x-goog-api-key": API_KEY,
        "Content-Type": "application/json",
    }
    assert "key=" not in url
    assert request["json"] == {
        "model": GEN_MODEL,
        "input": [{"type": "text", "text": "draw a moonlit forest"}],
    }


@pytest.mark.asyncio
async def test_gemini_edit_payload_contains_source_base64_mime_and_prompt(monkeypatch):
    client = _install_client(monkeypatch, _Response(payload=_image_response(include_text=False)))

    result = await gemini_image.edit_image(API_KEY, EDIT_MODEL, "turn it into a watercolor", SOURCE_PNG)

    assert result == RESULT_BYTES
    _, request = client.calls[0]
    payload = request["json"]
    assert payload["model"] == EDIT_MODEL
    assert payload["input"][0] == {
        "type": "image",
        "data": base64.b64encode(SOURCE_PNG).decode(),
        "mime_type": "image/png",
    }
    assert payload["input"][1] == {"type": "text", "text": "turn it into a watercolor"}


@pytest.mark.asyncio
async def test_gemini_image_extraction_returns_exact_bytes_and_skips_text_blocks(monkeypatch):
    _install_client(monkeypatch, _Response(payload=_image_response(include_text=True)))

    assert await gemini_image.generate_image(API_KEY, GEN_MODEL, "prompt") == RESULT_BYTES


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"steps": []},
        {"steps": [{"type": "model_output", "content": [{"type": "text", "text": "no image"}]}]},
    ),
)
async def test_gemini_malformed_or_no_image_response_fails_safely(monkeypatch, payload):
    _install_client(monkeypatch, _Response(payload=payload))

    with pytest.raises(gemini_image.GeminiImageResponseError):
        await gemini_image.generate_image(API_KEY, GEN_MODEL, "prompt")


@pytest.mark.asyncio
async def test_gemini_invalid_base64_fails_safely(monkeypatch):
    payload = {"steps": [{"type": "model_output", "content": [{"type": "image", "data": "not-base64"}]}]}
    _install_client(monkeypatch, _Response(payload=payload))

    with pytest.raises(gemini_image.GeminiImageResponseError, match="invalid base64"):
        await gemini_image.generate_image(API_KEY, GEN_MODEL, "prompt")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "payload", "json_error"),
    (
        (503, {"error": {"status": "UNAVAILABLE", "message": "temporarily unavailable"}}, None),
        (502, None, ValueError("bad json")),
    ),
)
async def test_gemini_provider_errors_are_safe(monkeypatch, status_code, payload, json_error):
    _install_client(monkeypatch, _Response(status_code, payload, json_error))

    with pytest.raises(gemini_image.GeminiImageError):
        await gemini_image.generate_image(API_KEY, GEN_MODEL, "prompt")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "model"),
    (
        ("generation", "imagen-4.0-generate-001"),
        ("generation", "gemini-3-pro-image-preview"),
        ("edit", "imagen-4.0-generate-001"),
        ("edit", "gemini-3-pro-image-preview"),
    ),
)
async def test_retired_direct_gemini_image_models_fail_before_http(monkeypatch, operation, model):
    client_factory = Mock(side_effect=AssertionError("HTTP must not be created"))
    monkeypatch.setattr(gemini_image.httpx, "AsyncClient", client_factory)

    with pytest.raises(ModelUnavailableError):
        if operation == "generation":
            await gemini_image.generate_image(API_KEY, model, "prompt")
        else:
            await gemini_image.edit_image(API_KEY, model, "prompt", SOURCE_PNG)

    assert client_factory.call_count == 0


def test_gemini_image_catalogs_and_defaults_are_current():
    expected = ("gemini-3.1-flash-image", "gemini-3-pro-image")
    assert get_selectable_models(PROVIDER_GEMINI, channel="image_gen") == expected
    assert get_selectable_models(PROVIDER_GEMINI, channel="image_edit") == expected
    assert get_default_model(PROVIDER_GEMINI, channel="image_gen") == GEN_MODEL
    assert get_default_model(PROVIDER_GEMINI, channel="image_edit") == GEN_MODEL


class _Session:
    def __init__(self, config):
        self.config = config

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, model, key):
        return self.config


def _image_config(*, gen_provider=PROVIDER_GEMINI, gen_model=GEN_MODEL, edit_provider=PROVIDER_GEMINI, edit_model=EDIT_MODEL):
    return SimpleNamespace(
        image_generation_provider=gen_provider,
        image_generation_model=gen_model,
        image_edit_provider=edit_provider,
        image_edit_model=edit_model,
        vision_provider=PROVIDER_GEMINI,
        gemini_api_key=API_KEY,
        openai_api_key="openai-test-key",
        kie_api_key="kie-test-key",
        kie_base_url="https://kie.example",
        kie_upload_base_url="https://kie-upload.example",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
@pytest.mark.parametrize("operation", ("generation", "edit"))
async def test_both_surfaces_route_configured_gemini_image_operations_to_shared_adapter(
    monkeypatch, surface, operation
):
    if surface == "telegram":
        import ai_integration as module
    else:
        from max_messenger_bot import ai as module

    config = _image_config()
    monkeypatch.setattr(module, "async_session_maker", lambda: _Session(config))
    adapter_call = AsyncMock(return_value=RESULT_BYTES)
    adapter_name = "generate_image" if operation == "generation" else "edit_image"
    monkeypatch.setattr(gemini_image, adapter_name, adapter_call)

    if operation == "generation":
        result = await module.generate_image("prompt")
        assert result == RESULT_BYTES
        assert adapter_call.await_args.args == (API_KEY, GEN_MODEL, "prompt")
    else:
        result = await module.edit_image("edit prompt", SOURCE_PNG)
        assert result == RESULT_BYTES
        assert adapter_call.await_args.args == (API_KEY, EDIT_MODEL, "edit prompt", SOURCE_PNG)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
@pytest.mark.parametrize(
    ("provider", "operation"),
    (
        (PROVIDER_OPENAI, "generation"),
        (PROVIDER_KIE, "generation"),
        (PROVIDER_KIE, "edit"),
    ),
)
async def test_existing_image_provider_routing_remains_authoritative(monkeypatch, surface, provider, operation):
    if surface == "telegram":
        import ai_integration as module
        provider_call = AsyncMock(return_value="openai-result" if provider == PROVIDER_OPENAI else RESULT_BYTES)
        if operation == "generation":
            target = "generate_openai_image" if provider == PROVIDER_OPENAI else "_call_kie_image_generation"
        else:
            target = "_call_kie_image_edit"
    else:
        from max_messenger_bot import ai as module
        provider_call = AsyncMock(return_value="openai-result" if provider == PROVIDER_OPENAI else RESULT_BYTES)
        if operation == "generation":
            target = "_generate_openai" if provider == PROVIDER_OPENAI else "_generate_kie"
        else:
            target = "_edit_kie"

    config = _image_config(
        gen_provider=provider,
        gen_model="gpt-image-2" if provider == PROVIDER_OPENAI else "bytedance/seedream-v4-text-to-image",
        edit_provider=provider,
        edit_model="bytedance/seedream-v4-edit",
    )
    monkeypatch.setattr(module, "async_session_maker", lambda: _Session(config))
    monkeypatch.setattr(module, target, provider_call)
    gemini_call = AsyncMock(side_effect=AssertionError("Gemini must not be selected"))
    monkeypatch.setattr(gemini_image, "generate_image", gemini_call)
    monkeypatch.setattr(gemini_image, "edit_image", gemini_call)

    if operation == "generation":
        result = await module.generate_image("prompt")
    else:
        result = await module.edit_image("prompt", SOURCE_PNG)

    assert result == ("openai-result" if provider == PROVIDER_OPENAI else RESULT_BYTES)
    assert not gemini_call.await_args_list


class _TelegramMessage:
    def __init__(self):
        self.text = None
        self.reply_markup = None

    async def edit_text(self, text, reply_markup=None):
        self.text = text
        self.reply_markup = reply_markup


class _TelegramCallback:
    def __init__(self):
        self.message = _TelegramMessage()


class _MaxClient:
    def __init__(self):
        self.kwargs = None

    async def send_message(self, **kwargs):
        self.kwargs = kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ("image_gen", "image_edit"))
async def test_telegram_admin_selector_renders_gemini_models(monkeypatch, channel):
    import handlers

    config = _image_config()
    monkeypatch.setattr(handlers, "async_session_maker", lambda: _Session(config))
    callback = _TelegramCallback()
    if channel == "image_gen":
        await handlers.admin_change_image_generation_model_list(callback)
    else:
        await handlers.admin_change_image_edit_model_list(callback)

    labels = {
        button.text
        for row in callback.message.reply_markup.inline_keyboard
        for button in row
    }
    assert set(get_selectable_models(PROVIDER_GEMINI, channel=channel)) <= labels


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ("image_gen", "image_edit"))
async def test_max_admin_selector_renders_gemini_models(monkeypatch, channel):
    from max_messenger_bot.services import admin_ai

    config = _image_config()
    monkeypatch.setattr(admin_ai, "_get_config", AsyncMock(return_value=config))
    client = _MaxClient()
    if channel == "image_gen":
        await admin_ai.show_image_generation_models(client, 1)
    else:
        await admin_ai.show_image_edit_models(client, 1)

    rows = client.kwargs["attachments"][0]["payload"]["buttons"]
    labels = {button[0]["text"].removeprefix("✅ ") for button in rows[:-1]}
    assert set(get_selectable_models(PROVIDER_GEMINI, channel=channel)) <= labels
