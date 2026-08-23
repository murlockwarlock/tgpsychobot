import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from kie_chat import build_kie_chat_request
from provider_models import (
    KIE_CHAT_MODELS,
    KIE_CHAT_PROTOCOL_ANTHROPIC,
    KIE_CHAT_PROTOCOL_GEMINI,
    KIE_CHAT_PROTOCOL_RESPONSES,
    KIE_DEFAULT_CHAT_MODEL,
    get_kie_chat_model_spec,
)


NEW_KIE_CHAT_MODELS = (
    "claude-haiku-4-5",
    "grok-4-3",
    "gemini-3-7-flash",
    "gpt-5-6-luna",
)
KIE_API_KEY = "kie-test-secret"
HISTORY = [
    SimpleNamespace(role="user", content="предыдущий вопрос"),
    SimpleNamespace(role="assistant", content="предыдущий ответ"),
    SimpleNamespace(role="user", content="текущий вопрос"),
]


def _telegram_primary_models():
    module = ast.parse(Path("handlers.py").read_text(encoding="utf-8"))
    assignment = next(
        node for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "MODELS_INFO" for target in node.targets)
    )
    return [name for name in ast.literal_eval(assignment.value)["KIE"] if name != "pricing"]


def _telegram_fallback_models():
    module = ast.parse(Path("handlers.py").read_text(encoding="utf-8"))
    handler = next(
        node for node in ast.walk(module)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "admin_change_fallback_model_list"
    )
    assignment = next(
        node for node in ast.walk(handler)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "model_map" for target in node.targets)
    )
    kie_value = next(
        value
        for key, value in zip(assignment.value.keys, assignment.value.values)
        if isinstance(key, ast.Constant) and key.value == "KIE"
    )
    assert isinstance(kie_value, ast.Call)
    assert isinstance(kie_value.func, ast.Name) and kie_value.func.id == "list"
    assert isinstance(kie_value.args[0], ast.Name) and kie_value.args[0].id == "KIE_CHAT_MODELS"
    return list(KIE_CHAT_MODELS)


def test_all_requested_ids_are_in_authoritative_catalog():
    assert KIE_DEFAULT_CHAT_MODEL in KIE_CHAT_MODELS
    assert set(NEW_KIE_CHAT_MODELS).issubset(KIE_CHAT_MODELS)


def test_telegram_primary_and_fallback_expose_authoritative_catalog():
    assert _telegram_primary_models() == list(KIE_CHAT_MODELS)
    assert _telegram_fallback_models() == list(KIE_CHAT_MODELS)


def test_max_primary_and_fallback_expose_authoritative_catalog():
    from max_messenger_bot.services import admin_ai

    assert admin_ai.PROVIDER_MODELS["KIE"] == list(KIE_CHAT_MODELS)
    assert admin_ai.FALLBACK_MODELS["KIE"] == list(KIE_CHAT_MODELS)


def test_telegram_and_max_model_lists_stay_in_parity():
    from max_messenger_bot.services import admin_ai

    telegram_primary = _telegram_primary_models()
    telegram_fallback = _telegram_fallback_models()
    assert telegram_primary == telegram_fallback == admin_ai.PROVIDER_MODELS["KIE"]
    assert telegram_fallback == admin_ai.FALLBACK_MODELS["KIE"]


@pytest.mark.parametrize("model_id", NEW_KIE_CHAT_MODELS)
def test_documented_endpoint_and_protocol_are_authoritative(model_id):
    request = build_kie_chat_request(
        KIE_API_KEY,
        "https://api.example",
        model_id,
        HISTORY,
        "SYSTEM PROMPT",
        temperature=0.35,
    )
    spec = get_kie_chat_model_spec(model_id)

    assert request.endpoint == f"https://api.example{spec.endpoint_path.format(model=model_id)}"
    assert request.protocol == spec.protocol
    assert request.stream == spec.stream
    assert request.headers["Authorization"] == f"Bearer {KIE_API_KEY}"
    assert KIE_API_KEY not in json.dumps(request.payload)


def test_claude_messages_request_shape_and_headers():
    request = build_kie_chat_request(KIE_API_KEY, "https://api.example", "claude-haiku-4-5", HISTORY, "SYSTEM", 0.4)

    assert request.protocol == KIE_CHAT_PROTOCOL_ANTHROPIC
    assert request.endpoint == "https://api.example/claude/v1/messages"
    assert request.payload == {
        "model": "claude-haiku-4-5",
        "system": "SYSTEM",
        "messages": [
            {"role": "user", "content": "предыдущий вопрос"},
            {"role": "assistant", "content": "предыдущий ответ"},
            {"role": "user", "content": "текущий вопрос"},
        ],
        "max_tokens": 4096,
        "temperature": 0.4,
        "stream": False,
    }
    assert request.headers["X-Api-Key"] == KIE_API_KEY
    assert request.headers["anthropic-version"] == "2023-06-01"


def test_responses_request_shape_for_grok_and_luna():
    grok = build_kie_chat_request(KIE_API_KEY, "https://api.example", "grok-4-3", HISTORY, "SYSTEM", 0.4)
    luna = build_kie_chat_request(KIE_API_KEY, "https://api.example", "gpt-5-6-luna", HISTORY, "SYSTEM", 0.4)

    assert grok.protocol == luna.protocol == KIE_CHAT_PROTOCOL_RESPONSES
    assert grok.endpoint == "https://api.example/grok/v1/responses"
    assert luna.endpoint == "https://api.example/codex/v1/responses"
    assert grok.payload["model"] == "grok-4-3"
    assert luna.payload["model"] == "gpt-5-6-luna"
    assert grok.payload["stream"] is True
    assert luna.payload["stream"] is False
    assert grok.payload["instructions"] == luna.payload["instructions"] == "SYSTEM"
    assert grok.payload["input"][0] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "предыдущий вопрос"}],
    }
    assert "temperature" not in grok.payload
    assert "temperature" not in luna.payload


def test_gemini_native_request_shape():
    request = build_kie_chat_request(KIE_API_KEY, "https://api.example", "gemini-3-7-flash", HISTORY, "SYSTEM", 0.4)

    assert request.protocol == KIE_CHAT_PROTOCOL_GEMINI
    assert request.endpoint == "https://api.example/gemini/v1/models/gemini-3-7-flash:streamGenerateContent"
    assert request.stream is True
    assert request.payload["stream"] is True
    assert request.payload["systemInstruction"] == {"parts": [{"text": "SYSTEM"}]}
    assert request.payload["contents"] == [
        {"role": "user", "parts": [{"text": "предыдущий вопрос"}]},
        {"role": "model", "parts": [{"text": "предыдущий ответ"}]},
        {"role": "user", "parts": [{"text": "текущий вопрос"}]},
    ]
    assert request.payload["generationConfig"]["temperature"] == 0.4
    assert request.headers["X-Goog-Api-Key"] == KIE_API_KEY


class _Response:
    status_code = 200

    def __init__(self, payload=None, text=""):
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("response is SSE")
        return self._payload


class _HttpClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, endpoint, **kwargs):
        self.calls.append({"endpoint": endpoint, "kwargs": kwargs})
        return self.response


def _provider_response(model_id):
    if model_id == "claude-haiku-4-5":
        return _Response({"content": [{"type": "text", "text": "Claude answer"}]})
    if model_id == "grok-4-3":
        return _Response(
            text=(
                'event: response.output_text.delta\n'
                'data: {"type":"response.output_text.delta","delta":"Grok "}\n\n'
                'event: response.output_text.delta\n'
                'data: {"type":"response.output_text.delta","delta":"answer"}\n\n'
                'data: [DONE]\n\n'
            )
        )
    if model_id == "gemini-3-7-flash":
        return _Response(
            text=(
                'data: {"candidates":[{"content":{"parts":[{"text":"Gemini "}]}}]}\n\n'
                'data: {"candidates":[{"content":{"parts":[{"text":"answer"}]}}]}\n\n'
            )
        )
    return _Response({
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Luna answer"}],
        }],
    })


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_id", "expected"),
    (
        ("claude-haiku-4-5", "Claude answer"),
        ("grok-4-3", "Grok answer"),
        ("gemini-3-7-flash", "Gemini answer"),
        ("gpt-5-6-luna", "Luna answer"),
    ),
)
async def test_telegram_routes_and_parses_each_new_protocol(model_id, expected):
    import ai_integration

    client = _HttpClient(_provider_response(model_id))
    capture = {}
    with patch.object(ai_integration.httpx, "AsyncClient", return_value=client):
        result = await ai_integration._call_kie_chat(
            KIE_API_KEY,
            "https://api.example",
            model_id,
            HISTORY,
            "KB context",
            "SYSTEM",
            temperature=0.25,
            request_capture=capture,
        )

    assert result == expected
    assert client.calls[0]["endpoint"] == capture["endpoint"]
    assert client.calls[0]["kwargs"]["json"] == capture["payload"]
    captured_payload = json.dumps(capture["payload"], ensure_ascii=False)
    assert "SYSTEM" in captured_payload
    assert "KB context" in captured_payload
    assert "текущий вопрос" in captured_payload
    assert KIE_API_KEY not in json.dumps(capture)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", NEW_KIE_CHAT_MODELS)
async def test_max_routes_each_new_protocol_through_shared_builder(model_id):
    from max_messenger_bot import ai

    client = _HttpClient(_provider_response(model_id))
    with patch.object(ai.httpx, "AsyncClient", return_value=client):
        result = await ai._call_kie_text_chat(
            KIE_API_KEY,
            "https://api.example",
            model_id,
            [{"role": "user", "content": "текущий вопрос"}],
            "SYSTEM",
            0.25,
        )

    assert result
    assert client.calls[0]["endpoint"] == build_kie_chat_request(
        KIE_API_KEY,
        "https://api.example",
        model_id,
        [{"role": "user", "content": "текущий вопрос"}],
        "SYSTEM",
        0.25,
    ).endpoint


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", ("gemini-3-flash", "gemini-2.5-flash"))
async def test_existing_kie_chat_models_keep_openai_compatible_route(model_id):
    import ai_integration

    client = _HttpClient(_Response({"choices": [{"message": {"content": "legacy answer"}}]}))
    capture = {}
    with patch.object(ai_integration.httpx, "AsyncClient", return_value=client):
        result = await ai_integration._call_kie_chat(
            KIE_API_KEY,
            "https://api.example",
            model_id,
            HISTORY,
            "",
            "SYSTEM",
            request_capture=capture,
        )

    assert result == "legacy answer"
    assert client.calls[0]["endpoint"] == f"https://api.example/{model_id}/v1/chat/completions"
    assert client.calls[0]["kwargs"]["json"]["model"] == model_id
    assert KIE_API_KEY not in json.dumps(capture)
