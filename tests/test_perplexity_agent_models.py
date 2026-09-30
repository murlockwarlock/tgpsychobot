import os
import json
import time
import importlib
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("BOT_TOKEN", "123456:TEST_BOT_TOKEN")

import httpx
import pytest
import pytest_asyncio
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.enums import ChatType
from aiogram.types import CallbackQuery, Chat, Message, Update, User as TgUser
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import ai_integration
import keyboards as kb
from database import AIConfig, AIModelSettings, Base, User
from max_messenger_bot.api import MaxApiClient
from max_messenger_bot.app import MaxBotApplication
from max_messenger_bot.keyboards import (
    admin_ai_perplexity_category_keyboard,
    admin_ai_perplexity_presets_keyboard,
    admin_ai_perplexity_models_keyboard,
)
from max_messenger_bot.storage import StateStore
from provider_adapters import (
    ProviderAdapterError,
    _classify_perplexity_error_dict,
    _classify_perplexity_http_error,
    _extract_perplexity_citations,
    _extract_perplexity_text,
    _post_perplexity_json,
    build_perplexity_payload,
    call_perplexity,
    format_perplexity_response,
)
import provider_models
from provider_models import (
    PROVIDER_PERPLEXITY,
    PERPLEXITY_MODES,
    PERPLEXITY_MODE_INFO,
    PERPLEXITY_MODE_OUTPUT_LIMITS,
    PERPLEXITY_STATIC_DIRECT_MODELS,
    PerplexityCatalogState,
    build_telegram_model_callback_data,
    canonical_provider_name,
    ensure_model_available,
    get_chat_output_token_limit,
    get_perplexity_catalog_state,
    get_perplexity_model_label,
    get_perplexity_selectable_models,
    get_selectable_models,
    is_perplexity_preset,
    refresh_perplexity_catalog,
    resolve_telegram_model_callback,
    validate_model_selection,
    ModelUnavailableError,
)
from ai_request_context import AIRequestLayout, AIRequestMessage


# ---------------------------------------------------------------------------
# Validating Session for aiogram (ensures all markups pass Pydantic validation)
# ---------------------------------------------------------------------------

class ValidatingTelegramSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def close(self):
        return None

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        # Real Pydantic / aiogram schema validation
        type(method).model_validate(method.model_dump())
        return True

    async def stream_content(self, *args, **kwargs):
        if False:
            yield b""


# ---------------------------------------------------------------------------
# Test Fixtures & Database Setup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_perplexity_catalog_state():
    provider_models._current_perplexity_catalog_state = None
    provider_models._previous_generation_perplexity_models = ()
    yield
    provider_models._current_perplexity_catalog_state = None
    provider_models._previous_generation_perplexity_models = ()


@pytest_asyncio.fixture
async def ppx_db():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with sessions() as session:
        session.add(User(id=1, first_name="Admin", username="admin", is_admin=True))
        session.add(
            AIConfig(
                id=1,
                provider="Perplexity",
                perplexity_api_key="test-ppx-key",
                perplexity_model="low",
                system_prompt="Test system prompt",
            )
        )
        await session.commit()

    import handlers
    import keyboards
    import database
    import max_messenger_bot.services.admin_ai as max_admin_ai
    import max_messenger_bot.app as max_app_module

    with patch.object(handlers, "async_session_maker", sessions), \
         patch.object(keyboards, "async_session_maker", sessions), \
         patch.object(database, "async_session_maker", sessions), \
         patch.object(max_admin_ai, "async_session_maker", sessions), \
         patch.object(max_app_module, "async_session_maker", sessions):
        yield sessions

    await engine.dispose()


def make_layout(text: str = "Текущий вопрос", system: str = "Отвечай кратко.") -> AIRequestLayout:
    return AIRequestLayout(
        stable_system_prompt=system,
        history=(),
        current_user_content=text,
    )


# ---------------------------------------------------------------------------
# 1. Canonical Provider ID & Normalization
# ---------------------------------------------------------------------------

def test_canonical_provider_id():
    assert PROVIDER_PERPLEXITY == "Perplexity"
    assert canonical_provider_name("perplexity") == "Perplexity"
    assert canonical_provider_name("Perplexity") == "Perplexity"
    assert canonical_provider_name("PERPLEXITY") == "Perplexity"
    assert canonical_provider_name("  perplexity  ") == "Perplexity"


# ---------------------------------------------------------------------------
# 2. Presets & Capabilities
# ---------------------------------------------------------------------------

def test_five_official_presets():
    expected_presets = ("fast", "low", "medium", "high", "xhigh")
    assert PERPLEXITY_MODES == expected_presets
    for preset in expected_presets:
        assert is_perplexity_preset(preset) is True
        assert preset in PERPLEXITY_MODE_INFO
        assert preset in PERPLEXITY_MODE_OUTPUT_LIMITS
        assert get_chat_output_token_limit(PROVIDER_PERPLEXITY, preset) == PERPLEXITY_MODE_OUTPUT_LIMITS[preset]

    assert is_perplexity_preset("wide-research") is False
    assert is_perplexity_preset("anthropic/claude-sonnet-4-6") is False
    assert is_perplexity_preset("openai/gpt-4o") is False


# ---------------------------------------------------------------------------
# 3. Payload Construction
# ---------------------------------------------------------------------------

def test_preset_payload_no_tools():
    layout = make_layout("Hello", "Be concise")
    payload = build_perplexity_payload(layout, "low")
    assert payload["preset"] == "low"
    assert "tools" not in payload
    assert "model" not in payload
    assert "Hello" in payload["input"]


def test_direct_model_payload_no_tools():
    layout = make_layout("Hello", "Be concise")
    payload = build_perplexity_payload(layout, "openai/gpt-4o", temperature=0.7, max_output_tokens=4000)
    assert payload["model"] == "openai/gpt-4o"
    assert "tools" not in payload
    assert "preset" not in payload
    assert payload["temperature"] == 0.7
    assert payload["max_output_tokens"] == 4000
    assert "Hello" in payload["input"]


def test_anthropic_mandatory_8192_tokens():
    layout = make_layout("Hello", "Be concise")
    payload = build_perplexity_payload(layout, "anthropic/claude-sonnet-4-6", max_output_tokens=None)
    assert payload["model"] == "anthropic/claude-sonnet-4-6"
    assert payload["max_output_tokens"] == 8192

    payload2 = build_perplexity_payload(layout, "anthropic/claude-3-5-haiku", max_output_tokens=2048)
    assert payload2["max_output_tokens"] == 2048


def test_preset_plus_model_override():
    layout = make_layout("Hello", "Be concise")
    payload = build_perplexity_payload(layout, preset="low", model="anthropic/claude-sonnet-4-6")
    assert payload["preset"] == "low"
    assert payload["model"] == "anthropic/claude-sonnet-4-6"
    assert payload["max_output_tokens"] == 8192
    assert "Hello" in payload["input"]


@pytest.mark.asyncio
async def test_no_reachable_anthropic_call_without_tokens():
    """Contract test proving anthropic/* always sends max_output_tokens."""
    captured_payload = None

    async def mock_post(url, *, headers, payload, timeout, request_capture, activity_tracker=None, retries=1):
        nonlocal captured_payload
        captured_payload = payload
        return {
            "id": "resp-123",
            "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "text", "text": "Anthropic response"}]}],
        }

    layout = make_layout("Hi", "")
    with patch("provider_adapters._post_perplexity_json", mock_post):
        text = await call_perplexity("test-key", layout, "anthropic/claude-sonnet-4-6", max_output_tokens=None)
        assert "Anthropic response" in text
        assert captured_payload is not None
        assert "max_output_tokens" in captured_payload
        assert captured_payload["max_output_tokens"] == 8192


# ---------------------------------------------------------------------------
# 4. Response Parsing & Text Extraction
# ---------------------------------------------------------------------------

def test_extract_perplexity_text_clean():
    data = {
        "id": "agent_123",
        "status": "completed",
        "output": [
            {"type": "search_results", "results": [{"title": "Search Result 1", "snippet": "Snippet 1", "url": "https://example.com/1"}]},
            {"type": "fetch_url_results", "url": "https://example.com", "content": "HTML body"},
            {"type": "finance", "data": {"ticker": "AAPL"}},
            {"type": "sandbox", "stdout": "print(1)"},
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "This is the final clean assistant text."},
                ],
            },
        ],
    }
    extracted = _extract_perplexity_text(data)
    assert extracted == "This is the final clean assistant text."
    assert "Search Result 1" not in extracted
    assert "HTML body" not in extracted
    assert "AAPL" not in extracted
    assert "print(1)" not in extracted

    formatted = format_perplexity_response(data)
    assert "This is the final clean assistant text." in formatted
    assert "https://example.com/1" in formatted


def test_extract_perplexity_citations_dedup():
    data = {
        "status": "completed",
        "output": [
            {
                "type": "search_results",
                "results": [
                    {"title": "Page 1", "url": "https://example.com/1"},
                    {"title": "Page 1 duplicate", "url": "https://example.com/1"},
                    {"title": "Page 2", "url": "https://example.com/2"},
                ],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Answer text"}],
            },
        ],
        "citations": ["https://example.com/1", "https://example.com/3"],
    }
    citations = _extract_perplexity_citations(data)
    urls = [c.url for c in citations]
    assert len(urls) == 3
    assert urls == ["https://example.com/1", "https://example.com/2", "https://example.com/3"]


@pytest.mark.parametrize("bad_status", ["failed", "cancelled", "incomplete", "queued", "in_progress"])
def test_http_200_bad_status_raises(bad_status):
    data = {
        "status": bad_status,
        "output": [],
    }
    with pytest.raises(ProviderAdapterError):
        _extract_perplexity_text(data)


def test_http_200_with_error_raises():
    data = {
        "status": "completed",
        "error": {"type": "insufficient_quota", "message": "Credit exhausted"},
    }
    with pytest.raises(ProviderAdapterError) as exc_info:
        _extract_perplexity_text(data)
    assert exc_info.value.category == "insufficient_balance_quota"


# ---------------------------------------------------------------------------
# 5. Error Classification & Retry Policy
# ---------------------------------------------------------------------------

def test_error_classification_table():
    assert _classify_perplexity_http_error(400, {"error": {"message": "max_output_tokens invalid"}}, "") == "configuration"
    assert _classify_perplexity_http_error(401, {}, "Unauthorized") == "auth"
    assert _classify_perplexity_http_error(403, {"error": {"message": "permission denied"}}, "") == "auth"
    assert _classify_perplexity_http_error(404, {"error": {"message": "model not found"}}, "") == "configuration"
    assert _classify_perplexity_http_error(429, {}, "Too many requests") == "rate_limit"
    assert _classify_perplexity_http_error(500, {}, "Internal server error") == "provider_5xx"
    assert _classify_perplexity_http_error(402, {"error": {"message": "insufficient credit"}}, "") == "insufficient_balance_quota"


@pytest.mark.asyncio
async def test_no_retries_on_400_validation():
    attempts = 0

    async def mock_post(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        resp = MagicMock()
        resp.status_code = 400
        resp.json.return_value = {"error": {"message": "model not found"}}
        resp.text = '{"error": {"message": "model not found"}}'
        resp.headers = {}
        return resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        with pytest.raises(ProviderAdapterError) as exc_info:
            await _post_perplexity_json(
                "https://api.perplexity.ai/v1/agent",
                headers={},
                payload={},
                timeout=5.0,
                request_capture=None,
                retries=2,
            )

    assert attempts == 1
    assert exc_info.value.category == "configuration"


@pytest.mark.asyncio
async def test_retry_on_429_with_retry_after():
    attempts = 0

    async def mock_post(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        resp = MagicMock()
        if attempts == 1:
            resp.status_code = 429
            resp.json.return_value = {"error": {"message": "Rate limited"}}
            resp.text = "Rate limited"
            resp.headers = {"retry-after": "0"}
            return resp
        resp.status_code = 200
        resp.json.return_value = {"id": "res1", "status": "completed", "output": []}
        resp.text = '{"id": "res1"}'
        resp.headers = {}
        return resp

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        resp = await _post_perplexity_json(
            "https://api.perplexity.ai/v1/agent",
            headers={},
            payload={},
            timeout=5.0,
            request_capture=None,
            retries=2,
        )
    assert attempts == 2
    assert resp["id"] == "res1"


# ---------------------------------------------------------------------------
# 6. Dynamic Model Catalog & Authority Metadata
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_public_catalog_get_no_auth():
    request_headers = None

    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            nonlocal request_headers
            request_headers = dict(request.headers)
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "anthropic/claude-sonnet-4-6", "object": "model"},
                        {"id": "openai/gpt-4o", "object": "model"},
                    ],
                },
                request=request,
            )

    client = httpx.AsyncClient(transport=MockTransport())
    state = await refresh_perplexity_catalog(force=True, http_client=client)
    assert request_headers is not None
    assert "authorization" not in request_headers
    assert state.is_fresh is True
    assert state.is_authoritative is True
    assert state.source == "live"
    assert "anthropic/claude-sonnet-4-6" in state.models
    assert "openai/gpt-4o" in state.models


@pytest.mark.asyncio
async def test_authoritative_negative_rejection():
    """Fresh successful live GET /v1/models: absence IS authoritative."""
    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "anthropic/claude-sonnet-4-6", "object": "model"},
                    ],
                },
                request=request,
            )

    client = httpx.AsyncClient(transport=MockTransport())
    await refresh_perplexity_catalog(force=True, http_client=client)

    # In live catalog: ok
    ensure_model_available(PROVIDER_PERPLEXITY, "anthropic/claude-sonnet-4-6")

    # Absent from fresh live catalog: MUST REJECT
    with pytest.raises(ModelUnavailableError):
        ensure_model_available(PROVIDER_PERPLEXITY, "openai/non-existent-model")


@pytest.mark.asyncio
async def test_non_authoritative_static_fallback_allows_model():
    """Static or stale catalog: absence is NOT authoritative for models with '/'."""
    class FailingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise httpx.ConnectError("Network down")

    client = httpx.AsyncClient(transport=FailingTransport())
    provider_models._current_perplexity_catalog_state = None
    state = await refresh_perplexity_catalog(force=True, http_client=client)
    assert state.source in {"static_fallback", "fallback"}
    assert state.is_authoritative is False

    # Persisted dynamic model absent from static snapshot: MUST BE ALLOWED to attempt runtime call
    ensure_model_available(PROVIDER_PERPLEXITY, "openai/future-model-x")


# ---------------------------------------------------------------------------
# 7. Telegram Compact Callbacks (<= 64 bytes)
# ---------------------------------------------------------------------------

def test_telegram_callback_length_under_64_bytes():
    long_models = [
        "anthropic/claude-sonnet-4-6",
        "perplexity/nemotron-3-ultra-550b-a55b",
        "google/gemini-3.1-pro-preview",
        "xai/grok-4.20-reasoning",
    ]
    for m in long_models:
        cb_data = build_telegram_model_callback_data(PROVIDER_PERPLEXITY, "chat", m)
        assert len(cb_data.encode("utf-8")) <= 64
        resolved = resolve_telegram_model_callback(cb_data)
        assert resolved is not None
        provider, channel, resolved_model = resolved
        assert provider == PROVIDER_PERPLEXITY
        assert channel == "chat"
        assert resolved_model == m


# ---------------------------------------------------------------------------
# 8. Stale Callbacks: Refresh & Restart
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stale_callback_after_cache_refresh():
    # Advance to Gen A
    class GenATransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": "anthropic/claude-gen-a", "object": "model"}]},
                request=request,
            )

    clientA = httpx.AsyncClient(transport=GenATransport())
    await refresh_perplexity_catalog(force=True, http_client=clientA)
    cb_data = build_telegram_model_callback_data(PROVIDER_PERPLEXITY, "chat", "anthropic/claude-gen-a")

    # Advance to Gen B with authoritative live response (Gen A model retired)
    class GenBTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": "openai/gpt-4o", "object": "model"}]},
                request=request,
            )

    clientB = httpx.AsyncClient(transport=GenBTransport())
    await refresh_perplexity_catalog(force=True, http_client=clientB)

    # Callback resolves safely from previous generation
    resolved = resolve_telegram_model_callback(cb_data)
    assert resolved is not None
    assert resolved[2] == "anthropic/claude-gen-a"

    # But validation rejects selection because fresh Gen B does NOT have model X
    with pytest.raises(ModelUnavailableError):
        validate_model_selection(PROVIDER_PERPLEXITY, resolved[2], channel="chat")


def test_stale_callback_after_restart():
    # Build a callback for a model not in the static snapshot
    # so after a restart with a new catalog it cannot resolve
    dynamic_model = "custom-vendor/dynamic-model-yesterday"
    provider_models._previous_generation_perplexity_models = (dynamic_model,)
    cb_data = build_telegram_model_callback_data(PROVIDER_PERPLEXITY, "chat", dynamic_model)

    # Simulate restart: previous generation wiped, new catalog does not have it
    provider_models._previous_generation_perplexity_models = ()
    provider_models._current_perplexity_catalog_state = PerplexityCatalogState(
        models=("openai/gpt-4o",),
        source="live",
        is_fresh=True,
        is_authoritative=True,
        fetched_at=time.monotonic(),
    )

    resolved = resolve_telegram_model_callback(cb_data)
    assert resolved is None


# ---------------------------------------------------------------------------
# 9. Telegram Admin Full Journey
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_telegram_admin_presets_journey(ppx_db, monkeypatch):
    import handlers
    if handlers.router.parent_router is not None:
        handlers = importlib.reload(handlers)

    monkeypatch.setattr(handlers, "async_session_maker", ppx_db)
    monkeypatch.setattr(kb, "async_session_maker", ppx_db)

    session = ValidatingTelegramSession()
    bot = Bot(token="123456:TEST_BOT_TOKEN", session=session)

    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)

    user = TgUser(id=1, is_bot=False, first_name="Admin", username="admin")
    chat = Chat(id=1, type=ChatType.PRIVATE)
    message = Message(message_id=1, date=datetime.now(timezone.utc), chat=chat, from_user=user, text="test").as_(bot)

    # 1. Open Perplexity settings
    cb1 = CallbackQuery(id="cb1", from_user=user, chat_instance="1", message=message, data="view_models_Perplexity").as_(bot)
    await dp.feed_update(bot, Update(update_id=1, callback_query=cb1))

    # Verify outgoing edit message
    edit_call = next(call for call in reversed(session.calls) if type(call).__name__ == "EditMessageText")
    assert "Perplexity" in edit_call.text
    markup = edit_call.reply_markup
    buttons = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert "ai_ppx_presets" in buttons
    assert "ai_ppx_models:0" in buttons

    # 2. Click "⚡ Пресеты поиска"
    cb2 = CallbackQuery(id="cb2", from_user=user, chat_instance="1", message=message, data="ai_ppx_presets").as_(bot)
    await dp.feed_update(bot, Update(update_id=2, callback_query=cb2))
    edit_presets = next(call for call in reversed(session.calls) if type(call).__name__ == "EditMessageText")
    markup2 = edit_presets.reply_markup
    preset_buttons = [b for row in markup2.inline_keyboard for b in row if b.callback_data.startswith("ai_m_")]
    assert len(preset_buttons) == 5
    back_button = next(b for row in markup2.inline_keyboard for b in row if b.text == "⬅️ Назад")
    assert back_button.callback_data == "view_models_Perplexity"

    # Select 'high' preset by resolving its compact callback
    high_btn = next(b for b in preset_buttons if resolve_telegram_model_callback(b.callback_data)[2] == "high")
    high_cb_data = high_btn.callback_data

    # 3. Select 'high' preset and verify persistence
    cb3 = CallbackQuery(id="cb3", from_user=user, chat_instance="1", message=message, data=high_cb_data).as_(bot)
    await dp.feed_update(bot, Update(update_id=3, callback_query=cb3))
    async with ppx_db() as session_db:
        config = await session_db.get(AIConfig, 1)
        assert config.perplexity_model == "high"


@pytest.mark.asyncio
async def test_telegram_admin_direct_models_pagination_journey(ppx_db, monkeypatch):
    import handlers
    if handlers.router.parent_router is not None:
        handlers = importlib.reload(handlers)

    monkeypatch.setattr(handlers, "async_session_maker", ppx_db)
    monkeypatch.setattr(kb, "async_session_maker", ppx_db)

    # Mock refresh_perplexity_catalog to return offline snapshot
    async def mock_refresh(*args, **kwargs):
        return get_perplexity_catalog_state()

    monkeypatch.setattr(handlers, "refresh_perplexity_catalog", mock_refresh)

    session = ValidatingTelegramSession()
    bot = Bot(token="123456:TEST_BOT_TOKEN", session=session)

    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)

    user = TgUser(id=1, is_bot=False, first_name="Admin", username="admin")
    chat = Chat(id=1, type=ChatType.PRIVATE)
    message = Message(message_id=1, date=datetime.now(timezone.utc), chat=chat, from_user=user, text="test").as_(bot)

    # Open direct models page 0
    cb1 = CallbackQuery(id="cb1", from_user=user, chat_instance="1", message=message, data="ai_ppx_models:0").as_(bot)
    await dp.feed_update(bot, Update(update_id=1, callback_query=cb1))
    edit_call1 = next(call for call in reversed(session.calls) if type(call).__name__ == "EditMessageText")
    buttons1 = [b for row in edit_call1.reply_markup.inline_keyboard for b in row]
    model_buttons1 = [b for b in buttons1 if b.callback_data.startswith("ai_m_")]
    assert len(model_buttons1) == 6
    next_btn = next(b for b in buttons1 if b.text == "След ➡️")
    assert next_btn.callback_data == "ai_ppx_models:1"

    # Navigate to page 1
    cb2 = CallbackQuery(id="cb2", from_user=user, chat_instance="1", message=message, data="ai_ppx_models:1").as_(bot)
    await dp.feed_update(bot, Update(update_id=2, callback_query=cb2))
    edit_call2 = next(call for call in reversed(session.calls) if type(call).__name__ == "EditMessageText")
    buttons2 = [b for row in edit_call2.reply_markup.inline_keyboard for b in row]
    model_buttons2 = [b for b in buttons2 if b.callback_data.startswith("ai_m_")]
    assert len(model_buttons2) == 6

    # Select model from page 1
    selected_cb = model_buttons2[0].callback_data
    cb3 = CallbackQuery(id="cb3", from_user=user, chat_instance="1", message=message, data=selected_cb).as_(bot)
    await dp.feed_update(bot, Update(update_id=3, callback_query=cb3))
    async with ppx_db() as session_db:
        config = await session_db.get(AIConfig, 1)
        resolved = resolve_telegram_model_callback(selected_cb)
        assert config.perplexity_model == resolved[2]


# ---------------------------------------------------------------------------
# 10. MAX Admin Full Journey
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_admin_presets_and_models_journey(ppx_db, monkeypatch):
    import max_messenger_bot.app as max_app_module
    import max_messenger_bot.services.admin_ai as max_admin_ai

    async def _true_async(*args, **kwargs):
        return True

    monkeypatch.setattr(max_app_module.common, "is_admin", _true_async)
    monkeypatch.setattr(max_app_module.common, "ensure_user", _true_async)
    monkeypatch.setattr(max_admin_ai, "async_session_maker", ppx_db)
    monkeypatch.setattr(max_app_module, "async_session_maker", ppx_db)

    async def mock_refresh(*args, **kwargs):
        return get_perplexity_catalog_state()

    monkeypatch.setattr(max_admin_ai, "refresh_perplexity_catalog", mock_refresh)

    captured = []

    async def fake_request(method, path, *, params=None, json_data=None, expected_status=200):
        captured.append({"method": method, "path": path, "body": json_data or {}})
        if path == "/messages":
            return {"message": {"body": {"mid": str(len(captured))}}}
        return {}

    def get_last_message_body():
        messages = [item for item in captured if item["path"] == "/messages"]
        assert messages
        return messages[-1]["body"]

    client = MaxApiClient(token="test", base_url="http://max.test")
    client._request = fake_request
    app = MaxBotApplication(client=client)

    # 1. Open Perplexity presets via MAX
    await app.handle_update({
        "update_type": "message_callback",
        "update_id": 1,
        "callback": {
            "callback_id": "cb-1",
            "payload": "admin_ai_ppx_presets",
            "sender": {"user_id": 1001, "name": "Admin"},
        },
        "message": {
            "mid": "mid-1",
            "recipient": {"chat_id": 1001},
            "body": {},
        },
    })
    last_msg = get_last_message_body()
    assert "Пресеты поиска Perplexity" in last_msg["text"]
    buttons = [btn["payload"] for row in last_msg["attachments"][0]["payload"]["buttons"] for btn in row]
    assert "admin_ai_set_model_Perplexity_xhigh" in buttons

    # 2. Select preset 'xhigh' via MAX
    await app.handle_update({
        "update_type": "message_callback",
        "update_id": 2,
        "callback": {
            "callback_id": "cb-2",
            "payload": "admin_ai_set_model_Perplexity_xhigh",
            "sender": {"user_id": 1001, "name": "Admin"},
        },
        "message": {
            "mid": "mid-2",
            "recipient": {"chat_id": 1001},
            "body": {},
        },
    })
    async with ppx_db() as session:
        config = await session.get(AIConfig, 1)
        assert config.perplexity_model == "xhigh"

    # 3. Open Direct Models page 0
    await app.handle_update({
        "update_type": "message_callback",
        "update_id": 3,
        "callback": {
            "callback_id": "cb-3",
            "payload": "admin_ai_ppx_models_0",
            "sender": {"user_id": 1001, "name": "Admin"},
        },
        "message": {
            "mid": "mid-3",
            "recipient": {"chat_id": 1001},
            "body": {},
        },
    })
    last_msg = get_last_message_body()
    model_buttons = [btn["payload"] for row in last_msg["attachments"][0]["payload"]["buttons"] for btn in row]
    assert any(b.startswith("admin_ai_set_model_Perplexity_") for b in model_buttons)
    assert "admin_ai_ppx_models_1" in model_buttons

    # 4. Select direct model via MAX with partition('_')
    await app.handle_update({
        "update_type": "message_callback",
        "update_id": 4,
        "callback": {
            "callback_id": "cb-4",
            "payload": "admin_ai_set_model_Perplexity_anthropic/claude-sonnet-4-6",
            "sender": {"user_id": 1001, "name": "Admin"},
        },
        "message": {
            "mid": "mid-4",
            "recipient": {"chat_id": 1001},
            "body": {},
        },
    })
    async with ppx_db() as session:
        config = await session.get(AIConfig, 1)
        assert config.perplexity_model == "anthropic/claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# 11. Visible Button Contract Classification
# ---------------------------------------------------------------------------

def test_button_contract_classification():
    all_callbacks = [
        "view_models_Perplexity",
        "view_provider_models_Perplexity",
        "ai_ppx_presets",
        "ai_ppx_models:0",
        "ai_ppx_models:1",
        "noop",
        "admin_ai_ppx_presets",
        "admin_ai_ppx_models_0",
        "admin_ai_ppx_models_1",
        "admin_ai_models_Perplexity",
        build_telegram_model_callback_data(PROVIDER_PERPLEXITY, "chat", "low"),
        build_telegram_model_callback_data(PROVIDER_PERPLEXITY, "chat", "anthropic/claude-sonnet-4-6"),
        "admin_ai_set_model_Perplexity_low",
        "admin_ai_set_model_Perplexity_anthropic/claude-sonnet-4-6",
    ]

    classified_count = 0
    missing_count = 0

    for cb in all_callbacks:
        category = None
        if cb.startswith((
            "view_models_",
            "view_provider_models_",
            "ai_ppx_",
            "admin_ai_ppx_",
            "admin_ai_models_",
            "noop",
        )):
            category = "navigation"
        elif cb.startswith(("ai_m_", "admin_ai_set_model_")):
            category = "mutation"

        if category is not None:
            classified_count += 1
        else:
            missing_count += 1

    assert missing_count == 0
    assert classified_count == len(all_callbacks)


# ---------------------------------------------------------------------------
# 12. Backward Compatibility
# ---------------------------------------------------------------------------

def test_legacy_presets_backward_compatibility():
    for legacy in ("fast", "low", "medium"):
        validated = validate_model_selection(PROVIDER_PERPLEXITY, legacy, channel="chat")
        assert validated == legacy
        # Must not raise ModelUnavailableError
        ensure_model_available(PROVIDER_PERPLEXITY, legacy, channel="chat")
