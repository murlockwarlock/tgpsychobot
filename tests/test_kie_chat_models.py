import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from kie_chat import build_kie_chat_request
from ai_request_context import AIRequestLayout, AIRequestMessage
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
KIE_CANONICAL_LAYOUT_MODELS = NEW_KIE_CHAT_MODELS + ("gemini-3-flash",)
KIE_API_KEY = "kie-test-secret"
HISTORY = [
    SimpleNamespace(role="user", content="предыдущий вопрос"),
    SimpleNamespace(role="assistant", content="предыдущий ответ"),
    SimpleNamespace(role="user", content="текущий вопрос"),
]


def _telegram_primary_models():
    from handlers import MODELS_INFO

    return [name for name in MODELS_INFO["KIE"] if name != "pricing"]


def _telegram_fallback_models():
    """Previously inspected a hardcoded model_map dict; after the SSOT refactor,
    admin_change_fallback_model_list calls get_selectable_models(provider, 'fallback').
    We verify the handler uses that call and that the result matches KIE_CHAT_MODELS."""
    from provider_models import get_selectable_models
    # Structural check: ensure model_map is gone and get_selectable_models is referenced
    source = Path("handlers.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    handler = next(
        node for node in ast.walk(module)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "admin_change_fallback_model_list"
    )
    # After refactor: no model_map assignment should remain in this handler
    has_model_map = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "model_map" for t in node.targets)
        for node in ast.walk(handler)
    )
    assert not has_model_map, (
        "admin_change_fallback_model_list still contains a hardcoded model_map dict. "
        "It should call get_selectable_models() instead."
    )
    # Functional: get_selectable_models KIE/fallback must equal canonical list
    return list(get_selectable_models("KIE", channel="fallback"))


def test_all_requested_ids_are_in_authoritative_catalog():
    assert KIE_DEFAULT_CHAT_MODEL in KIE_CHAT_MODELS
    assert set(NEW_KIE_CHAT_MODELS).issubset(KIE_CHAT_MODELS)


def test_telegram_primary_and_fallback_expose_authoritative_catalog():
    """Both the Telegram primary picker (MODELS_INFO) and fallback picker
    (get_selectable_models) must expose the full KIE_CHAT_MODELS catalog."""
    assert _telegram_primary_models() == list(KIE_CHAT_MODELS)
    assert _telegram_fallback_models() == list(KIE_CHAT_MODELS)


def test_max_primary_and_fallback_expose_authoritative_catalog():
    """MAX admin now calls get_selectable_models() instead of using PROVIDER_MODELS.
    Verify the SSOT returns KIE_CHAT_MODELS for both chat and fallback channels."""
    from provider_models import get_selectable_models
    assert list(get_selectable_models("KIE", channel="chat")) == list(KIE_CHAT_MODELS)
    assert list(get_selectable_models("KIE", channel="fallback")) == list(KIE_CHAT_MODELS)


def test_telegram_and_max_model_lists_stay_in_parity():
    """Telegram and MAX both call get_selectable_models(), so they are always in parity.
    Verify get_selectable_models returns the same KIE list for both channels used by each UI."""
    from provider_models import get_selectable_models
    telegram_primary = _telegram_primary_models()
    telegram_fallback = _telegram_fallback_models()
    max_primary = list(get_selectable_models("KIE", channel="chat"))
    max_fallback = list(get_selectable_models("KIE", channel="fallback"))
    assert telegram_primary == list(KIE_CHAT_MODELS)
    assert telegram_fallback == list(KIE_CHAT_MODELS)
    assert max_primary == list(KIE_CHAT_MODELS)
    assert max_fallback == list(KIE_CHAT_MODELS)


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
        "system": [{"type": "text", "text": "SYSTEM"}],
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
    def __init__(self, payload=None, text="", status_code=200):
        self.status_code = status_code
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
    if model_id == "gemini-3-flash":
        return _Response({"choices": [{"message": {"content": "OpenAI-compatible answer"}}]})
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


def _canonical_kie_layout():
    return AIRequestLayout(
        stable_system_prompt="STABLE CONFIGURED TOPIC PROMPT",
        shared_instructions=("SHARED TOPIC INSTRUCTIONS",),
        runtime_context=("ДАННЫЕ КЛИЕНТА: ИМЯ: Ясна",),
        scenario_context=('current_state=photo metadata=kie-meta',),
        request_context=("TEST RESULTS", "KB RAG RESULTS", "GLOBAL MEMORY"),
        history=(
            AIRequestMessage("user", "предыдущий вопрос"),
            AIRequestMessage("assistant", "предыдущий ответ"),
        ),
        current_user_content="текущий вопрос",
    )


def _assert_kie_layout_payload(payload, protocol):
    stable = "STABLE CONFIGURED TOPIC PROMPT\n\nSHARED TOPIC INSTRUCTIONS"
    if protocol == "openai_chat":
        blocks = [item["content"] for item in payload["messages"] if item["role"] == "system"]
        history = payload["messages"][-3:-1]
        current = payload["messages"][-1]
    elif protocol == "anthropic_messages":
        blocks = [item["text"] for item in payload["system"]]
        history = payload["messages"][-3:-1]
        current = payload["messages"][-1]
    elif protocol == "responses":
        assert payload["instructions"] == stable
        blocks = [
            item["content"][0]["text"]
            for item in payload["input"]
            if item.get("role") == "developer"
        ]
        history = payload["input"][-3:-1]
        current = payload["input"][-1]
    else:
        blocks = [item["text"] for item in payload["systemInstruction"]["parts"]]
        history = payload["contents"][-3:-1]
        current = payload["contents"][-1]

    if protocol == "responses":
        assert "Ясна" not in payload["instructions"]
        assert "photo" not in payload["instructions"]
        assert "kie-meta" not in payload["instructions"]
        assert payload["instructions"] == stable
    else:
        assert blocks[0] == stable
        assert "Ясна" not in blocks[0]
        assert "photo" not in blocks[0]
        assert "kie-meta" not in blocks[0]
    dynamic_index = next(i for i, block in enumerate(blocks) if "Ясна" in block)
    assert "photo" in blocks[dynamic_index]
    assert "kie-meta" in blocks[dynamic_index]
    assert "SHARED TOPIC INSTRUCTIONS" in (payload["instructions"] if protocol == "responses" else blocks[0])
    if protocol == "responses":
        assert dynamic_index == 0
    else:
        assert dynamic_index == 1
    request_indices = [
        i for i, block in enumerate(blocks)
        if any(value in block for value in ("TEST RESULTS", "KB RAG RESULTS", "GLOBAL MEMORY"))
    ]
    assert request_indices
    assert min(request_indices) > dynamic_index
    assert all(value in "\n".join(blocks) for value in ("TEST RESULTS", "KB RAG RESULTS", "GLOBAL MEMORY"))
    assert "предыдущий вопрос" in json.dumps(history[0], ensure_ascii=False)
    assert "предыдущий ответ" in json.dumps(history[1], ensure_ascii=False)
    assert "текущий вопрос" in json.dumps(current, ensure_ascii=False)
    assert current["role"] == "user"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", KIE_CANONICAL_LAYOUT_MODELS)
async def test_kie_final_payload_preserves_canonical_layout(monkeypatch, model_id):
    import ai_integration

    client = _HttpClient(_provider_response(model_id))
    capture = {}
    layout = _canonical_kie_layout()
    with patch.object(ai_integration.httpx, "AsyncClient", return_value=client):
        result = await ai_integration._call_kie_chat(
            KIE_API_KEY,
            "https://api.example",
            model_id,
            [],
            "",
            "LEGACY SYSTEM MUST NOT BE USED",
            temperature=0.25,
            request_capture=capture,
            request_layout=layout,
        )

    assert result
    assert client.calls[0]["kwargs"]["json"] == capture["payload"]
    _assert_kie_layout_payload(capture["payload"], get_kie_chat_model_spec(model_id).protocol)


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


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
async def test_kie_http_402_is_classified_as_insufficient_balance(surface):
    if surface == "telegram":
        import ai_integration as module

        call = module._call_kie_chat
        args = (KIE_API_KEY, "https://api.example", "claude-haiku-4-5", HISTORY, "", "SYSTEM")
    else:
        from max_messenger_bot import ai as module

        call = module._call_kie_text_chat
        args = (
            KIE_API_KEY,
            "https://api.example",
            "claude-haiku-4-5",
            [{"role": "user", "content": "текущий вопрос"}],
            "SYSTEM",
            0.25,
        )

    client = _HttpClient(_Response({"code": 402, "msg": "Insufficient credits"}, status_code=402))
    with patch.object(module.httpx, "AsyncClient", return_value=client):
        with pytest.raises(module.InsufficientBalanceError, match="Insufficient credits"):
            await call(*args)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
async def test_kie_http_200_credit_error_envelope_is_rejected_before_stream_parsing(surface):
    if surface == "telegram":
        import ai_integration as module

        call = module._call_kie_chat
        args = (KIE_API_KEY, "https://api.example", "grok-4-3", HISTORY, "", "SYSTEM")
    else:
        from max_messenger_bot import ai as module

        call = module._call_kie_text_chat
        args = (
            KIE_API_KEY,
            "https://api.example",
            "grok-4-3",
            [{"role": "user", "content": "текущий вопрос"}],
            "SYSTEM",
            0.25,
        )

    error_payload = {"code": 402, "msg": "Insufficient credits"}
    client = _HttpClient(_Response(error_payload, text=json.dumps(error_payload)))
    with patch.object(module.httpx, "AsyncClient", return_value=client):
        with pytest.raises(module.InsufficientBalanceError, match="Insufficient credits"):
            await call(*args)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
async def test_kie_http_200_channel_rejection_is_provider_error_with_full_metadata(surface):
    if surface == "telegram":
        import ai_integration as module

        call = module._call_kie_chat
        args = (KIE_API_KEY, "https://api.example", "gemini-3-flash", HISTORY, "", "SYSTEM")
    else:
        from max_messenger_bot import ai as module

        call = module._call_kie_text_chat
        args = (
            KIE_API_KEY,
            "https://api.example",
            "gemini-3-flash",
            [{"role": "user", "content": "текущий вопрос"}],
            "SYSTEM",
            0.25,
        )

    error_payload = {"code": 422, "msg": "The channel is not supported", "data": None}
    client = _HttpClient(_Response(error_payload, text=json.dumps(error_payload)))
    with patch.object(module.httpx, "AsyncClient", return_value=client):
        with pytest.raises(module.AIServiceError) as raised:
            await call(*args)

    error = raised.value
    assert not isinstance(error, module.InsufficientBalanceError)
    assert error.classification == "provider_rejection"
    assert error.provider == "KIE"
    assert error.model == "gemini-3-flash"
    assert error.provider_code == 422
    assert error.provider_message == "The channel is not supported"
    assert error.http_status == 200
    assert error.diagnostics == {
        "provider_code": 422,
        "provider_message": "The channel is not supported",
    }
    assert json.loads(error.provider_response_payload) == error_payload


@pytest.mark.asyncio
async def test_kie_rejection_is_visible_in_admin_filters_details_and_exports(tmp_path, monkeypatch):
    from ai_integration import AIServiceError
    from ai_log_context import record_ai_attempt_log
    from database import AILog, Base, User
    import handlers
    from max_messenger_bot.services import admin_ai_logs as max_admin_ai_logs
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'kie-admin-log.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    error = AIServiceError("Ошибка при обращении к KIE Chat API: The channel is not supported")
    error.classification = "provider_rejection"
    error.provider = "KIE"
    error.model = "gemini-3-flash"
    error.provider_code = 422
    error.provider_message = "The channel is not supported"
    error.http_status = 200
    error.diagnostics = {
        "provider_code": 422,
        "provider_message": "The channel is not supported",
    }
    provider_payload = {"code": 422, "msg": "The channel is not supported", "data": None}
    error.provider_response_payload = json.dumps(provider_payload)
    error_meta = {
        "error_type": "AIServiceError",
        "error_message": str(error),
        "error_classification": "provider_rejection",
        "http_status": 200,
        "finish_reason": None,
        "diagnostics": error.diagnostics,
        "provider_response_payload": error.provider_response_payload,
    }
    async with sessions() as session:
        session.add(User(id=100_000_000_321, first_name="MAX", current_dialogue_id=1))
        await session.commit()
        log_id = await record_ai_attempt_log(
            session,
            user_id=100_000_000_321,
            platform="max",
            provider="KIE",
            model="gemini-3-flash",
            raw_response="",
            status="error",
            error_type=error_meta["error_type"],
            error_message=error_meta["error_message"],
            error_classification=error_meta["error_classification"],
            http_status=error_meta["http_status"],
            diagnostics=error_meta["diagnostics"],
            provider_response_payload=error_meta["provider_response_payload"],
        )
    assert log_id is not None

    monkeypatch.setattr(handlers, "async_session_maker", sessions)
    tg_event = SimpleNamespace(answer=AsyncMock())
    await handlers.show_ai_log_detail(tg_event, log_id)
    tg_detail = tg_event.answer.await_args.args[0]
    assert "KIE" in tg_detail
    assert "gemini-3-flash" in tg_detail
    assert "provider_rejection" in tg_detail
    assert "The channel is not supported" in tg_detail

    monkeypatch.setattr(max_admin_ai_logs, "async_session_maker", sessions)
    max_client = SimpleNamespace(send_message=AsyncMock(), send_text_file=AsyncMock())
    await max_admin_ai_logs.show_ai_logs_list(max_client, 321, status="error")
    assert "Всего вызовов: <b>1</b>" in max_client.send_message.await_args.kwargs["text"]
    await max_admin_ai_logs.show_ai_logs_list(max_client, 321, status="all")
    assert "Всего вызовов: <b>1</b>" in max_client.send_message.await_args.kwargs["text"]
    await max_admin_ai_logs.show_ai_log_detail(max_client, 321, log_id, status="error")
    max_detail = max_client.send_message.await_args.kwargs["text"]
    assert "KIE" in max_detail
    assert "gemini-3-flash" in max_detail
    assert "provider_rejection" in max_detail
    assert "The channel is not supported" in max_detail
    await max_admin_ai_logs.download_ai_log_file(max_client, 321, log_id)
    downloaded = max_client.send_text_file.await_args.kwargs["content"]
    assert "Provider: KIE" in downloaded
    assert "HTTP Status: 200" in downloaded
    assert "provider_code" in downloaded
    assert "The channel is not supported" in downloaded

    class ExportClient:
        def __init__(self):
            self.archive = None
            self.sent = None

        async def upload_file(self, _kind, path):
            self.archive = Path(path).read_bytes()
            return {"token": "kie-log-export"}

        async def send_media_attachment(self, **kwargs):
            self.sent = kwargs

        async def send_message(self, **kwargs):
            self.sent = kwargs

    export_client = ExportClient()
    await max_admin_ai_logs.export_ai_logs_package(export_client, 321, status="error")
    assert export_client.archive is not None
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(export_client.archive)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        log_file = archive.read(manifest[0]["file"]).decode("utf-8")
    assert manifest[0]["provider"] == "KIE"
    assert manifest[0]["error_classification"] == "provider_rejection"
    assert manifest[0]["provider_code"] == 422
    assert manifest[0]["provider_message"] == "The channel is not supported"
    assert "The channel is not supported" in log_file

    async with sessions() as session:
        stored = await session.get(AILog, log_id)
        assert stored.provider == "KIE"
        assert stored.model == "gemini-3-flash"
        assert stored.http_status == 200
        assert stored.error_classification == "provider_rejection"
    await engine.dispose()
