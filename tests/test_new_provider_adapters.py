import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ai_request_context import AIRequestLayout, AIRequestMessage
from provider_adapters import (
    ProviderAdapterError,
    build_openrouter_payload,
    build_openrouter_vision_layout,
    build_perplexity_payload,
    call_deepgram,
    call_openrouter,
    call_perplexity,
    format_perplexity_response,
    normalize_provider_error_classification,
)
from provider_models import (
    OPENROUTER_MODEL_SPECS,
    OPENROUTER_MODELS,
    OPENROUTER_VISION_MODELS,
    PROVIDER_DEEPGRAM,
    PROVIDER_OPENROUTER,
    PROVIDER_PERPLEXITY,
    ModelUnavailableError,
    effective_chat_output_tokens,
    get_default_model,
    get_selectable_models,
    validate_chat_output_tokens,
    validate_model_selection,
)


def layout() -> AIRequestLayout:
    return AIRequestLayout(
        stable_system_prompt="Отвечай на языке пользователя.",
        history=(AIRequestMessage(role="user", content="Предыдущий вопрос"),),
        current_user_content="Текущий вопрос",
    )


def test_openrouter_catalog_is_curated_and_capability_driven():
    assert len(OPENROUTER_MODELS) == 15
    assert set(get_selectable_models(PROVIDER_OPENROUTER)) == set(OPENROUTER_MODEL_SPECS)
    assert set(get_selectable_models(PROVIDER_OPENROUTER, "vision")) == set(OPENROUTER_VISION_MODELS)
    assert "deepseek/deepseek-v3.2" not in OPENROUTER_VISION_MODELS
    assert all(spec.text for spec in OPENROUTER_MODEL_SPECS.values())
    assert OPENROUTER_MODEL_SPECS["google/gemini-3.7-flash"].audio_input is True
    assert validate_model_selection(PROVIDER_OPENROUTER, "openai/gpt-5.6-terra") == "openai/gpt-5.6-terra"
    assert validate_model_selection(PROVIDER_PERPLEXITY, "medium") == "medium"
    assert get_selectable_models(PROVIDER_DEEPGRAM, "transcription") == ("nova-3",)
    assert validate_model_selection(PROVIDER_DEEPGRAM, "nova-3", channel="transcription") == "nova-3"
    assert get_default_model(PROVIDER_DEEPGRAM, channel="transcription") == "nova-3"


def test_openrouter_text_payload_preserves_history_and_output_budget():
    payload = build_openrouter_payload(
        layout(),
        "openai/gpt-5.6-terra",
        temperature=0.7,
        max_output_tokens=1234,
    )
    assert payload["model"] == "openai/gpt-5.6-terra"
    assert payload["max_tokens"] == 1234
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][-1] == {"role": "user", "content": "Текущий вопрос"}


def test_openrouter_vision_uses_data_uri_and_rejects_text_only_model():
    vision = build_openrouter_vision_layout(layout(), b"image", mime_type="image/png", user_instruction="Что на фото?")
    content = vision.current_user_content
    assert content[0] == {"type": "text", "text": "Что на фото?"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    with pytest.raises(ModelUnavailableError):
        validate_model_selection(PROVIDER_OPENROUTER, "deepseek/deepseek-v3.2", channel="vision")


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeClient:
    response = FakeResponse(200, {})
    requests = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return self.response


class SequenceClient(FakeClient):
    responses = []

    async def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return self.responses.pop(0)


def test_openrouter_response_and_perplexity_citations_are_safe():
    FakeClient.response = FakeResponse(200, {"choices": [{"message": {"content": [{"type": "text", "text": "Готово"}]}}], "usage": {"total_tokens": 3}})
    capture = {}
    with patch("provider_adapters.httpx.AsyncClient", FakeClient):
        text = asyncio.run(call_openrouter("secret", layout(), "openai/gpt-5.6-terra", max_output_tokens=512, request_capture=capture))
    assert text == "Готово"
    assert capture["provider"] == "OpenRouter"
    assert capture["payload"]["max_tokens"] == 512

    payload = {
        "output_text": "Ответ [1]",
        "output": [{"type": "search_results", "results": [{"title": "Источник", "url": "https://example.com"}]}],
    }
    assert "Источники:" in format_perplexity_response(payload)
    assert "https://example.com" in format_perplexity_response(payload)


def test_perplexity_payload_uses_official_preset_and_web_search_without_reasoning_control():
    payload = build_perplexity_payload(layout(), "medium", max_output_tokens=2048)
    assert payload["preset"] == "medium"
    assert payload["tools"] == [{"type": "web_search"}]
    assert payload["max_output_tokens"] == 2048
    assert payload["instructions"].startswith("Отвечай на языке пользователя.")
    assert "web_search" in payload["instructions"]
    assert "inline citations" in payload["instructions"]
    assert "system:" not in payload["input"]
    assert "user: Текущий вопрос" in payload["input"]
    assert "reasoning" not in payload


def test_deepgram_payload_and_transcript_support_multilingual_audio():
    FakeClient.response = FakeResponse(200, {"results": {"channels": [{"alternatives": [{"transcript": "Olá, мир"}]}]}})
    capture = {}
    with patch("provider_adapters.httpx.AsyncClient", FakeClient):
        transcript = asyncio.run(call_deepgram("secret", b"audio", "voice.ogg", request_capture=capture))
    assert transcript == "Olá, мир"
    assert "language=multi" in capture["endpoint"]
    request_url, request_kwargs = FakeClient.requests[-1]
    assert request_url.endswith("language=multi")
    assert request_kwargs["headers"]["Content-Type"] == "audio/ogg"


def test_new_provider_output_budget_respects_model_capability_metadata():
    assert effective_chat_output_tokens(PROVIDER_OPENROUTER, "qwen/qwen3-vl-235b-a22b-instruct", 1000) == 1000
    assert effective_chat_output_tokens(PROVIDER_OPENROUTER, "qwen/qwen3-vl-235b-a22b-instruct", 50000) == 32768
    assert effective_chat_output_tokens(PROVIDER_PERPLEXITY, "fast", None) == 8192
    assert effective_chat_output_tokens(PROVIDER_PERPLEXITY, "low", None) == 32768
    assert effective_chat_output_tokens(PROVIDER_PERPLEXITY, "medium", None) == 128000
    with pytest.raises(ValueError):
        validate_chat_output_tokens(PROVIDER_PERPLEXITY, "fast", 9000)


def test_provider_errors_are_normalized_and_retries_are_bounded():
    SequenceClient.responses = [FakeResponse(429, {}), FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})]
    with patch("provider_adapters.httpx.AsyncClient", SequenceClient):
        assert asyncio.run(call_openrouter("secret", layout(), "openai/gpt-5.6-terra")) == "ok"
    assert len(SequenceClient.requests) >= 2

    FakeClient.response = FakeResponse(401, {"error": "bad key"})
    with patch("provider_adapters.httpx.AsyncClient", FakeClient):
        with pytest.raises(ProviderAdapterError) as error:
            asyncio.run(call_openrouter("secret", layout(), "openai/gpt-5.6-terra"))
    assert error.value.classification == "auth"
    assert error.value.http_status == 401

    FakeClient.response = FakeResponse(402, {})
    with patch("provider_adapters.httpx.AsyncClient", FakeClient):
        with pytest.raises(ProviderAdapterError) as error:
            asyncio.run(call_openrouter("secret", layout(), "openai/gpt-5.6-terra"))
    assert error.value.classification == "quota"
    assert error.value.http_status == 402
    assert normalize_provider_error_classification("quota") == "insufficient_balance_quota"
    assert normalize_provider_error_classification("server_error") == "provider_5xx"
    assert normalize_provider_error_classification("network") == "network_connection"
    assert normalize_provider_error_classification("invalid_model") == "configuration"


def test_new_provider_activity_tracker_marks_only_once():
    class Tracker:
        def __init__(self):
            self.calls = 0

        async def mark_outbound_attempt_once(self):
            self.calls += 1

    FakeClient.response = FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})
    tracker = Tracker()
    with patch("provider_adapters.httpx.AsyncClient", FakeClient):
        assert asyncio.run(call_openrouter("secret", layout(), "openai/gpt-5.6-terra", activity_tracker=tracker)) == "ok"
    assert tracker.calls == 1


def test_provider_rejects_invalid_models_and_malformed_responses():
    with pytest.raises(ProviderAdapterError, match="модель"):
        build_openrouter_payload(layout(), "unknown/model")
    with pytest.raises(ProviderAdapterError, match="режим"):
        build_perplexity_payload(layout(), "unknown")

    FakeClient.response = FakeResponse(200, {"choices": []})
    with patch("provider_adapters.httpx.AsyncClient", FakeClient):
        with pytest.raises(ProviderAdapterError) as error:
            asyncio.run(call_openrouter("secret", layout(), "openai/gpt-5.6-terra"))
    assert error.value.classification == "empty_response"

    with pytest.raises(ProviderAdapterError) as error:
        asyncio.run(call_deepgram("secret", b"audio", "voice.ogg", model="unsupported"))
    assert error.value.classification == "invalid_model"
