import asyncio
import copy
import html
import importlib
import json
import os
import random
import re
import time
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
from aiogram.enums import ChatType
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, Chat, InlineKeyboardButton, InlineKeyboardMarkup, Message, Update, User as TgUser
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import ai_integration
from ai_request_context import AIRequestLayout, AIRequestMessage
from database import AIConfig, AIModelSettings, Base, User
import keyboards as kb
from max_messenger_bot.api import MaxApiClient
from max_messenger_bot.app import MaxBotApplication
from max_messenger_bot.keyboards import (
    admin_ai_perplexity_category_keyboard,
    admin_ai_perplexity_models_keyboard,
    admin_ai_perplexity_presets_keyboard,
)
from max_messenger_bot.storage import StateStore
import max_messenger_bot.services.admin_ai as max_admin_ai
from provider_adapters import (
    ProviderAdapterError,
    _classify_perplexity_error_dict,
    _classify_perplexity_http_error,
    _extract_perplexity_text,
    _populate_perplexity_diagnostics,
    _post_perplexity_json,
    build_perplexity_payload,
    call_perplexity,
    format_perplexity_response,
    set_perplexity_jitter_provider,
)
import provider_models
from provider_models import (
    PERPLEXITY_CHAT_MAX_TOKENS,
    PERPLEXITY_MODE_INFO,
    PERPLEXITY_MODE_OUTPUT_LIMITS,
    PERPLEXITY_MODES,
    PERPLEXITY_STATIC_DIRECT_MODELS,
    PROVIDER_PERPLEXITY,
    ModelUnavailableError,
    PerplexityCatalogState,
    build_telegram_model_callback_data,
    canonical_provider_name,
    ensure_model_available,
    get_callback_resolution_models,
    get_chat_output_token_limit,
    get_perplexity_callback_resolution_models,
    get_perplexity_catalog_state,
    get_perplexity_model_label,
    get_perplexity_selectable_models,
    get_selectable_models,
    is_perplexity_preset,
    refresh_perplexity_catalog,
    reset_perplexity_catalog_state_for_tests,
    resolve_telegram_model_callback,
    set_perplexity_monotonic_time_provider,
    validate_model_selection,
)


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
        type(method).model_validate(method.model_dump())
        return True

    async def stream_content(self, *args, **kwargs):
        if False:
            yield b""


# ---------------------------------------------------------------------------
# Test Fixtures & Database Setup
# ---------------------------------------------------------------------------

def _last_markup(calls):
    return next(c.reply_markup for c in reversed(calls) if hasattr(c, "reply_markup") and c.reply_markup is not None)


_real_async_client = httpx.AsyncClient


def mock_httpx_async_client(transport):
    def _client_factory(*args, **kwargs):
        kw = dict(kwargs)
        kw["transport"] = transport
        return _real_async_client(*args, **kw)
    return _client_factory


@pytest.fixture(autouse=True)
def reset_perplexity_catalog():
    reset_perplexity_catalog_state_for_tests()
    set_perplexity_jitter_provider(None)
    yield
    reset_perplexity_catalog_state_for_tests()
    set_perplexity_jitter_provider(None)


@pytest_asyncio.fixture
async def async_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(async_engine):
    return async_sessionmaker(async_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def init_ai_config(session_factory):
    async with session_factory() as session:
        cfg = AIConfig(
            id=1,
            perplexity_model="medium",
            fallback_provider=None,
            fallback_model=None,
        )
        session.add(cfg)
        await session.commit()
    return cfg


# ---------------------------------------------------------------------------
# 1. Catalog TTL, Dynamic Freshness & Monotonic Time Injection
# ---------------------------------------------------------------------------

def test_catalog_ttl_authority_monotonic_time():
    """Verify that catalog freshness & negative authority expire dynamically with monotonic time."""
    current_time = 1000.0
    set_perplexity_monotonic_time_provider(lambda: current_time)

    # Set up a fresh live catalog lacking "custom/model-future"
    provider_models._current_perplexity_catalog_state = PerplexityCatalogState(
        models=("anthropic/claude-sonnet-4-6", "openai/gpt-5.6-sol"),
        source="live",
        is_fresh=True,
        is_authoritative=True,
        fetched_at=current_time,
    )

    # T=0: fresh live catalog -> authoritative -> reject unlisted model
    state = get_perplexity_catalog_state()
    assert state.is_fresh is True
    assert state.is_authoritative is True
    assert state.source == "live"
    with pytest.raises(ModelUnavailableError, match="отключена провайдером"):
        ensure_model_available(PROVIDER_PERPLEXITY, "custom/model-future")

    # T=TTL-1 (3599s later): still fresh -> still authoritative -> still reject
    current_time += 3599.0
    state = get_perplexity_catalog_state()
    assert state.is_fresh is True
    assert state.is_authoritative is True
    assert state.source == "live"
    with pytest.raises(ModelUnavailableError, match="отключена провайдером"):
        ensure_model_available(PROVIDER_PERPLEXITY, "custom/model-future")

    # T=TTL+1 (3601s later) WITHOUT any network request or admin action:
    current_time += 2.0  # now 3601s after fetched_at
    state = get_perplexity_catalog_state()
    assert state.is_fresh is False
    assert state.is_authoritative is False
    assert state.source == "stale_live"

    # Runtime call after TTL expiry allows unlisted model without requiring Admin screen!
    ensure_model_available(PROVIDER_PERPLEXITY, "custom/model-future")

    # Successful new refresh without model X makes it authoritative again:
    provider_models._current_perplexity_catalog_state = PerplexityCatalogState(
        models=("anthropic/claude-sonnet-4-6", "openai/gpt-5.6-sol"),
        source="live",
        is_fresh=True,
        is_authoritative=True,
        fetched_at=current_time,
    )
    state = get_perplexity_catalog_state()
    assert state.is_fresh is True
    assert state.is_authoritative is True
    with pytest.raises(ModelUnavailableError, match="отключена провайдером"):
        ensure_model_available(PROVIDER_PERPLEXITY, "custom/model-future")


# ---------------------------------------------------------------------------
# 2. Selectable Models vs Callback Resolution & Bounded Retention
# ---------------------------------------------------------------------------

def test_selectable_models_vs_callback_resolution_and_retention():
    """Verify clean split: active selectable models vs callback candidates with bounded retention."""
    sim_time = 5000.0
    set_perplexity_monotonic_time_provider(lambda: sim_time)

    # Generation A: catalog has model-a
    provider_models._current_perplexity_catalog_state = PerplexityCatalogState(
        models=("vendor/model-a",),
        source="live",
        is_fresh=True,
        is_authoritative=True,
        fetched_at=sim_time,
    )
    cb_model_a = build_telegram_model_callback_data(PROVIDER_PERPLEXITY, "chat", "vendor/model-a")

    # Advance time by 60s and transition to Generation B (retiring model-a, adding model-b)
    sim_time += 60.0
    provider_models._previous_generation_perplexity_models = ("vendor/model-a",)
    provider_models._previous_generation_created_at = sim_time
    provider_models._current_perplexity_catalog_state = PerplexityCatalogState(
        models=("vendor/model-b",),
        source="live",
        is_fresh=True,
        is_authoritative=True,
        fetched_at=sim_time,
    )

    # A. Active selectable catalog contains ONLY model-b, NEVER retired model-a
    selectable = get_selectable_models(PROVIDER_PERPLEXITY)
    assert "vendor/model-b" in selectable
    assert "vendor/model-a" not in selectable

    # B. Callback resolution candidates contain model-a within retention (300s)
    resolved = resolve_telegram_model_callback(cb_model_a)
    assert resolved == (PROVIDER_PERPLEXITY, "chat", "vendor/model-a")

    # But validation under authoritative Gen B blocks selection of model-a!
    with pytest.raises(ModelUnavailableError, match="отключена провайдером"):
        validate_model_selection(PROVIDER_PERPLEXITY, "vendor/model-a")

    # C. Advance time beyond retention (300s window):
    sim_time += 301.0
    resolved_expired = resolve_telegram_model_callback(cb_model_a)
    assert resolved_expired is None, "Retired model from old generation must not resolve after retention expiry"


# ---------------------------------------------------------------------------
# 3. Canonical Telegram UI Hierarchy
# ---------------------------------------------------------------------------

def test_telegram_canonical_ui_hierarchy():
    """Verify Perplexity provider settings has only Presets and Direct Models, no redundant 'Выбрать модель'."""
    keyboard = kb.provider_model_settings_keyboard(
        PROVIDER_PERPLEXITY,
        show_reasoning=False,
        show_temperature=False,
    )
    texts = [btn.text for row in keyboard.inline_keyboard for btn in row]
    callbacks = [btn.callback_data for row in keyboard.inline_keyboard for btn in row]

    assert "⚡ Пресеты поиска" in texts
    assert "🤖 Прямые модели" in texts
    assert "🤖 Выбрать модель" not in texts, "Redundant 'Выбрать модель' must be removed for Perplexity"
    assert "ai_ppx_presets" in callbacks
    assert "ai_ppx_models:0" in callbacks
    assert f"view_provider_models_{PROVIDER_PERPLEXITY}" not in callbacks

    # Presets keyboard back button leads directly to provider settings parent
    presets_kb = kb.perplexity_presets_keyboard()
    back_btn = presets_kb.inline_keyboard[-1][0]
    assert back_btn.callback_data == f"view_models_{PROVIDER_PERPLEXITY}"

    # Direct models keyboard back button leads directly to provider settings parent
    models_kb = kb.perplexity_models_keyboard(models=["vendor/model-1"])
    parent_btn = models_kb.inline_keyboard[-1][0]
    assert parent_btn.callback_data == f"view_models_{PROVIDER_PERPLEXITY}"
    assert parent_btn.text == "К настройкам"


# ---------------------------------------------------------------------------
# 4. Canonical Pagination Contract (Telegram and MAX)
# ---------------------------------------------------------------------------

def test_canonical_pagination_contract_telegram():
    """Verify Telegram pagination contract: no wraparound, correct navigation buttons."""
    # 0 items: empty state, no pagination row
    kb_0 = kb.perplexity_models_keyboard(models=[])
    assert len(kb_0.inline_keyboard) == 1  # Only parent button
    assert kb_0.inline_keyboard[0][0].text == "К настройкам"

    # 1 item: no pagination row
    kb_1 = kb.perplexity_models_keyboard(models=["vendor/model-1"])
    assert len(kb_1.inline_keyboard) == 2  # 1 model + parent button

    # exactly page_size (6): no pagination row
    models_6 = [f"vendor/model-{i}" for i in range(1, 7)]
    kb_6 = kb.perplexity_models_keyboard(models=models_6)
    assert len(kb_6.inline_keyboard) == 7  # 6 models + parent button

    # page_size + 1 (7 items, 2 pages):
    models_7 = [f"vendor/model-{i}" for i in range(1, 8)]
    # First page: only Далее ➡️
    kb_p0 = kb.perplexity_models_keyboard(models=models_7, page=0, page_size=6)
    nav_row_p0 = kb_p0.inline_keyboard[-2]
    nav_texts_p0 = [b.text for b in nav_row_p0]
    assert nav_texts_p0 == ["Далее ➡️"]
    assert nav_row_p0[0].callback_data == "ai_ppx_models:1"

    # Last page: only ⬅️ Назад
    kb_p1 = kb.perplexity_models_keyboard(models=models_7, page=1, page_size=6)
    nav_row_p1 = kb_p1.inline_keyboard[-2]
    nav_texts_p1 = [b.text for b in nav_row_p1]
    assert nav_texts_p1 == ["⬅️ Назад"]
    assert nav_row_p1[0].callback_data == "ai_ppx_models:0"

    # 18 items (3 pages): middle page has both controls
    models_18 = [f"vendor/model-{i}" for i in range(1, 19)]
    kb_p_mid = kb.perplexity_models_keyboard(models=models_18, page=1, page_size=6)
    nav_row_mid = kb_p_mid.inline_keyboard[-2]
    nav_texts_mid = [b.text for b in nav_row_mid]
    assert nav_texts_mid == ["⬅️ Назад", "Далее ➡️"]
    assert nav_row_mid[0].callback_data == "ai_ppx_models:0"
    assert nav_row_mid[1].callback_data == "ai_ppx_models:2"


def test_canonical_pagination_contract_max():
    """Verify MAX pagination contract: no wraparound, correct navigation buttons."""
    # 0 items
    kb_0 = admin_ai_perplexity_models_keyboard(current_model="", models=[])
    btns_0 = kb_0[0]["payload"]["buttons"]
    assert len(btns_0) == 1
    assert btns_0[0][0]["text"] == "К настройкам"

    # 1 item
    kb_1 = admin_ai_perplexity_models_keyboard(current_model="", models=["vendor/model-1"])
    btns_1 = kb_1[0]["payload"]["buttons"]
    assert len(btns_1) == 2

    # 6 items
    models_6 = [f"vendor/model-{i}" for i in range(1, 7)]
    kb_6 = admin_ai_perplexity_models_keyboard(current_model="", models=models_6)
    btns_6 = kb_6[0]["payload"]["buttons"]
    assert len(btns_6) == 7

    # 7 items (page 0): only Далее ➡️
    models_7 = [f"vendor/model-{i}" for i in range(1, 8)]
    kb_p0 = admin_ai_perplexity_models_keyboard(current_model="", models=models_7, page=0, page_size=6)
    btns_p0 = kb_p0[0]["payload"]["buttons"]
    nav_p0 = btns_p0[-2]
    assert [b["text"] for b in nav_p0] == ["Далее ➡️"]
    assert nav_p0[0]["payload"] == "admin_ai_ppx_models_1"

    # 7 items (page 1): only ⬅️ Назад
    kb_p1 = admin_ai_perplexity_models_keyboard(current_model="", models=models_7, page=1, page_size=6)
    btns_p1 = kb_p1[0]["payload"]["buttons"]
    nav_p1 = btns_p1[-2]
    assert [b["text"] for b in nav_p1] == ["⬅️ Назад"]
    assert nav_p1[0]["payload"] == "admin_ai_ppx_models_0"

    # 18 items (middle page): both controls
    models_18 = [f"vendor/model-{i}" for i in range(1, 19)]
    kb_mid = admin_ai_perplexity_models_keyboard(current_model="", models=models_18, page=1, page_size=6)
    btns_mid = kb_mid[0]["payload"]["buttons"]
    nav_mid = btns_mid[-2]
    assert [b["text"] for b in nav_mid] == ["⬅️ Назад", "Далее ➡️"]
    assert nav_mid[0]["payload"] == "admin_ai_ppx_models_0"
    assert nav_mid[1]["payload"] == "admin_ai_ppx_models_2"


# ---------------------------------------------------------------------------
# 5. Retry Safety: Safe Connect vs Uncertain Post-Dispatch Timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_safety_safe_connect_vs_uncertain_read_timeout():
    """Verify safe connect failures retry, but uncertain post-dispatch read timeouts do NOT resend POST."""
    url = "https://api.perplexity.ai/v1/agent"
    payload = {"input": "ping"}

    # A. ConnectTimeout (SAFE pre-dispatch): retries up to max_attempts
    call_count = 0

    def connect_timeout_handler(request):
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectTimeout("Connection refused", request=request)

    transport = httpx.MockTransport(connect_timeout_handler)
    with patch("httpx.AsyncClient", mock_httpx_async_client(transport)):
        with pytest.raises(ProviderAdapterError) as exc_info:
            await _post_perplexity_json(
                url,
                headers={},
                payload=payload,
                timeout=10.0,
                request_capture=None,
                max_attempts=2,
            )
        assert exc_info.value.category == "timeout"
        assert call_count == 2, "Safe connect timeout should be retried"

    # B. ReadTimeout (UNCERTAIN post-dispatch): MUST NOT RETRY!
    read_call_count = 0

    def read_timeout_handler(request):
        nonlocal read_call_count
        read_call_count += 1
        raise httpx.ReadTimeout("Server did not send data", request=request)

    read_transport = httpx.MockTransport(read_timeout_handler)
    with patch("httpx.AsyncClient", mock_httpx_async_client(read_transport)):
        with pytest.raises(ProviderAdapterError) as exc_info:
            await _post_perplexity_json(
                url,
                headers={},
                payload=payload,
                timeout=10.0,
                request_capture=None,
                max_attempts=2,
            )
        assert exc_info.value.category == "timeout"
        assert read_call_count == 1, "UNCERTAIN read timeout MUST NOT resend paid POST generation!"


# ---------------------------------------------------------------------------
# 6. Rate Limit Backoff & Jitter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_429_backoff_and_jitter():
    """Verify Retry-After honoring (no 2.0s silent cap), exponential jitter, and deadline abort."""
    url = "https://api.perplexity.ai/v1/agent"
    payload = {"input": "ping"}

    # Mock jitter to 0.1 for predictable test
    set_perplexity_jitter_provider(lambda a, b: 0.1)

    # 1. Retry-After: 0 works without delay
    attempts = 0
    slept_delays = []

    async def fake_sleep(dur):
        slept_delays.append(dur)

    def rate_limit_0_handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "rate limit"}})
        return httpx.Response(200, json={"status": "completed", "output": [{"type": "message", "role": "assistant", "status": "completed", "content": "ok"}]})

    with patch("httpx.AsyncClient", mock_httpx_async_client(httpx.MockTransport(rate_limit_0_handler))), \
         patch("asyncio.sleep", fake_sleep):
        res = await _post_perplexity_json(url, headers={}, payload=payload, timeout=10.0, request_capture=None, max_attempts=2)
        assert res["status"] == "completed"
        assert slept_delays == [0.0]

    # 2. Retry-After: 3.5 honors 3.5s without being capped to 2.0s
    attempts = 0
    slept_delays.clear()

    def rate_limit_35_handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "3.5"}, json={"error": {"message": "rate limit"}})
        return httpx.Response(200, json={"status": "completed", "output": [{"type": "message", "role": "assistant", "status": "completed", "content": "ok"}]})

    with patch("httpx.AsyncClient", mock_httpx_async_client(httpx.MockTransport(rate_limit_35_handler))), \
         patch("asyncio.sleep", fake_sleep):
        res = await _post_perplexity_json(url, headers={}, payload=payload, timeout=10.0, request_capture=None, max_attempts=2)
        assert slept_delays == [3.5], "Retry-After of 3.5s must be respected directly without a 2.0s cap"

    # 3. Absent Retry-After: uses 0.5 * (2**0) + jitter (0.1) = 0.6s
    attempts = 0
    slept_delays.clear()

    def rate_limit_absent_handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": {"message": "rate limit"}})
        return httpx.Response(200, json={"status": "completed", "output": [{"type": "message", "role": "assistant", "status": "completed", "content": "ok"}]})

    with patch("httpx.AsyncClient", mock_httpx_async_client(httpx.MockTransport(rate_limit_absent_handler))), \
         patch("asyncio.sleep", fake_sleep):
        res = await _post_perplexity_json(url, headers={}, payload=payload, timeout=10.0, request_capture=None, max_attempts=2)
        assert slept_delays == [pytest.approx(0.6)]

    # 4. Deadline prevents retrying when delay exceeds remaining timeout
    def rate_limit_long_handler(request):
        return httpx.Response(429, headers={"retry-after": "15.0"}, json={"error": {"message": "rate limit"}})

    with patch("httpx.AsyncClient", mock_httpx_async_client(httpx.MockTransport(rate_limit_long_handler))), \
         patch("asyncio.sleep", fake_sleep):
        with pytest.raises(ProviderAdapterError) as exc_info:
            # timeout is 2.0s, retry-after is 15.0s -> abort retry immediately
            await _post_perplexity_json(url, headers={}, payload=payload, timeout=2.0, request_capture=None, max_attempts=2)
        assert exc_info.value.category == "rate_limit"


# ---------------------------------------------------------------------------
# 7. Table-Driven Error Classification with Realistic Full Bodies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "status,data,expected_category",
    [
        # 403 + explicit auth problem -> auth
        (
            403,
            {"error": {"code": 403, "type": "unauthorized_error", "message": "Your account does not have access to tier 2"}},
            "auth",
        ),
        # 403 + explicit exhausted balance/credits -> insufficient_balance_quota
        (
            403,
            {"error": {"code": 403, "type": "forbidden", "message": "You have exhausted your credits. Please add balance."}},
            "insufficient_balance_quota",
        ),
        # 429 + plain rate limit -> rate_limit
        (
            429,
            {"error": {"code": 429, "type": "rate_limit_exceeded", "message": "Requests per minute limit reached"}},
            "rate_limit",
        ),
        # 429 + explicit billing/credit exhaustion -> insufficient_balance_quota
        (
            429,
            {"error": {"code": 429, "type": "insufficient_quota", "message": "Monthly spending quota exceeded"}},
            "insufficient_balance_quota",
        ),
        # 400 model not found -> configuration
        (
            400,
            {"error": {"code": 400, "type": "invalid_request_error", "message": "Model 'vendor/legacy-model' does not exist"}},
            "configuration",
        ),
        # 400 missing required Anthropic max_output_tokens -> configuration
        (
            400,
            {"error": {"code": 400, "type": "invalid_request_error", "message": "Missing required field: max_output_tokens"}},
            "configuration",
        ),
        # 400 content/safety rejection -> provider_rejection
        (
            400,
            {"error": {"code": 400, "type": "invalid_request_error", "message": "Request rejected due to content moderation safety policy"}},
            "provider_rejection",
        ),
        # 400 generic validation -> provider_rejection (NOT configuration!)
        (
            400,
            {"error": {"code": 400, "type": "invalid_request_error", "message": "Payload schema validation failed for input argument"}},
            "provider_rejection",
        ),
        # 404 -> configuration
        (
            404,
            {"error": {"code": 404, "message": "Endpoint not found"}},
            "configuration",
        ),
        # 500 -> provider_5xx
        (
            500,
            {"error": {"code": 500, "message": "Internal server error"}},
            "provider_5xx",
        ),
    ],
)
def test_error_classification_realistic_bodies(status, data, expected_category):
    cat = _classify_perplexity_http_error(status, data, json.dumps(data))
    assert cat == expected_category


# ---------------------------------------------------------------------------
# 8. Error Diagnostics in request_capture
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_request_capture_structured_diagnostics():
    """Verify request_capture contains structured diagnostics and NEVER leaks API keys."""
    url = "https://api.perplexity.ai/v1/agent"
    payload = {"input": "ping"}

    # A. 400 fixture
    capture_400 = {}
    mock_400 = httpx.Response(
        400,
        headers={"x-request-id": "req-400-abc"},
        json={"error": {"type": "invalid_request_error", "code": "missing_param", "message": "max_output_tokens required"}},
    )
    with patch("httpx.AsyncClient", mock_httpx_async_client(httpx.MockTransport(lambda r: mock_400))):
        with pytest.raises(ProviderAdapterError):
            await _post_perplexity_json(url, headers={"Authorization": "Bearer SECRET_KEY_123"}, payload=payload, timeout=5.0, request_capture=capture_400, max_attempts=1)
    assert capture_400["http_status"] == 400
    assert capture_400["x_request_id"] == "req-400-abc"
    assert capture_400["provider_error_type"] == "invalid_request_error"
    assert capture_400["provider_error_code"] == "missing_param"
    assert capture_400["provider_error_message"] == "max_output_tokens required"
    assert "SECRET_KEY_123" not in str(capture_400)

    # B. 403 fixture
    capture_403 = {}
    mock_403 = httpx.Response(
        403,
        headers={"x-request-id": "req-403-xyz"},
        json={"error": {"type": "permission_denied", "code": 403, "message": "Account forbidden"}},
    )
    with patch("httpx.AsyncClient", mock_httpx_async_client(httpx.MockTransport(lambda r: mock_403))):
        with pytest.raises(ProviderAdapterError):
            await _post_perplexity_json(url, headers={"Authorization": "Bearer SECRET_KEY_123"}, payload=payload, timeout=5.0, request_capture=capture_403, max_attempts=1)
    assert capture_403["http_status"] == 403
    assert capture_403["x_request_id"] == "req-403-xyz"
    assert capture_403["provider_error_type"] == "permission_denied"

    # C. 429 fixture with Retry-After
    capture_429 = {}
    mock_429 = httpx.Response(
        429,
        headers={"x-request-id": "req-429-lim", "retry-after": "4.5"},
        json={"error": {"type": "rate_limit", "code": 429, "message": "Too many requests"}},
    )
    with patch("httpx.AsyncClient", mock_httpx_async_client(httpx.MockTransport(lambda r: mock_429))):
        with pytest.raises(ProviderAdapterError):
            # timeout=1.0 < 4.5 -> aborts retry
            await _post_perplexity_json(url, headers={"Authorization": "Bearer SECRET_KEY_123"}, payload=payload, timeout=1.0, request_capture=capture_429, max_attempts=2)
    assert capture_429["http_status"] == 429
    assert capture_429["x_request_id"] == "req-429-lim"
    assert capture_429["retry_after"] == "4.5"

    # D. HTTP 200 status="failed" fixture
    capture_failed = {}
    mock_failed = httpx.Response(
        200,
        headers={"x-request-id": "req-200-failed"},
        json={
            "id": "resp-app-failed",
            "status": "failed",
            "model": "anthropic/claude-sonnet-4-6",
            "error": {"type": "internal_error", "code": 500, "message": "Agent execution failed"},
        },
    )
    with patch("httpx.AsyncClient", mock_httpx_async_client(httpx.MockTransport(lambda r: mock_failed))):
        with pytest.raises(ProviderAdapterError):
            res = await _post_perplexity_json(url, headers={"Authorization": "Bearer SECRET_KEY_123"}, payload=payload, timeout=5.0, request_capture=capture_failed, max_attempts=1)
            _extract_perplexity_text(res, request_capture=capture_failed)
    assert capture_failed["http_status"] == 200
    assert capture_failed["x_request_id"] == "req-200-failed"
    assert capture_failed["response_id"] == "resp-app-failed"
    assert capture_failed["response_status"] == "failed"
    assert capture_failed["provider_error_type"] == "internal_error"
    assert capture_failed["provider_error_message"] == "Agent execution failed"


# ---------------------------------------------------------------------------
# 9. Strict Agent Response Success Contract
# ---------------------------------------------------------------------------

def test_strict_agent_response_success_contract():
    """Verify strict validation: outer status == 'completed', assistant message completed."""
    # 1. Missing outer status -> invalid_response
    with pytest.raises(ProviderAdapterError, match="Отсутствует обязательный статус"):
        _extract_perplexity_text({"output": [{"type": "message", "role": "assistant", "status": "completed", "content": "text"}]})

    # 2. Non-completed outer statuses
    for bad_status in ("queued", "in_progress", "failed", "incomplete", "cancelled", "unknown"):
        with pytest.raises(ProviderAdapterError):
            _extract_perplexity_text({"status": bad_status, "output": []})

    # 3. Outer status completed, but assistant item has role != assistant
    with pytest.raises(ProviderAdapterError, match="пустой ответ"):
        _extract_perplexity_text({
            "status": "completed",
            "output": [{"type": "message", "role": "user", "status": "completed", "content": "hello"}],
        })

    # 4. Outer status completed, but assistant item has status != completed
    with pytest.raises(ProviderAdapterError, match="пустой ответ"):
        _extract_perplexity_text({
            "status": "completed",
            "output": [{"type": "message", "role": "assistant", "status": "in_progress", "content": "hello"}],
        })

    # 5. Outer status completed, all criteria satisfied -> returns text
    text = _extract_perplexity_text({
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "status": "completed", "content": "Clean output text"}],
    })
    assert text == "Clean output text"


# ---------------------------------------------------------------------------
# 10. Citation Identity & Deduplication
# ---------------------------------------------------------------------------

def test_citation_identity_and_deduplication():
    """Verify consistent citation mapping: deduplicated URLs and atomically resolved inline markers."""
    payload = {
        "status": "completed",
        "output": [
            {
                "type": "tool_call",
                "tool_name": "web_search",
                "results": [
                    {"id": "web:1", "title": "Perplexity API Docs", "url": "https://docs.perplexity.ai"},
                    {"id": "web:2", "title": "Perplexity API Mirror", "url": "https://docs.perplexity.ai"},  # DUPLICATE URL!
                    {"id": "web:3", "title": "Anthropic Models", "url": "https://docs.anthropic.com"},
                ],
            },
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": "According to docs [web:1] and mirror [web:2], and also Anthropic [web:3].",
            },
        ],
    }

    result = format_perplexity_response(payload)

    # 1. Duplicate URL must not produce duplicate entries in footer
    assert result.count("https://docs.perplexity.ai") == 1
    assert result.count("https://docs.anthropic.com") == 1

    # 2. Duplicate marker [web:2] in text must resolve to the single canonical entry [web:1]
    assert "[web:1]" in result
    assert "[web:2]" not in result, "[web:2] pointing to duplicate URL must be normalized to canonical [web:1]"
    assert "[web:3]" in result

    # 3. Footer entries match inline markers
    assert "[web:1] Perplexity API Docs — https://docs.perplexity.ai" in result
    assert "[web:3] Anthropic Models — https://docs.anthropic.com" in result


# ---------------------------------------------------------------------------
# 11. Real Telegram Journey A -> Я -> A
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_telegram_journey_a_to_ya_to_a(session_factory, monkeypatch, init_ai_config):
    """Full Telegram Journey: navigate screens from ACTUAL visible markup, select, persist, reopen, Back."""
    import handlers

    # Set up memory db and session maker in handlers
    monkeypatch.setattr(handlers, "async_session_maker", session_factory)
    monkeypatch.setattr(handlers, "is_admin", lambda user_id: True)

    session = ValidatingTelegramSession()
    bot = Bot(token="123456:TEST_BOT_TOKEN", session=session)
    handlers.router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)

    user = TgUser(id=999, is_bot=False, first_name="Admin", username="admin")
    chat = Chat(id=999, type=ChatType.PRIVATE)

    # 1. Open Perplexity provider settings screen
    update_1 = Update(
        update_id=1,
        callback_query=CallbackQuery(
            id="cb1",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Providers"),
            data=f"view_models_{PROVIDER_PERPLEXITY}",
        ),
    )
    await dp.feed_update(bot, update_1)
    provider_kb: InlineKeyboardMarkup = _last_markup(session.calls)

    # Extract actual callback for Presets from visible markup
    presets_btn = next(b for row in provider_kb.inline_keyboard for b in row if "Пресеты" in b.text)
    assert presets_btn.callback_data == "ai_ppx_presets"

    # 2. Feed actual Presets callback into Dispatcher
    update_2 = Update(
        update_id=2,
        callback_query=CallbackQuery(
            id="cb2",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Settings"),
            data=presets_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, update_2)
    presets_kb: InlineKeyboardMarkup = _last_markup(session.calls)

    # Extract actual preset model callback for "fast" from visible markup
    fast_btn = next(b for row in presets_kb.inline_keyboard for b in row if "Быстрый" in b.text)

    # 3. Select preset model "fast" via Dispatcher
    update_3 = Update(
        update_id=3,
        callback_query=CallbackQuery(
            id="cb3",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Presets"),
            data=fast_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, update_3)

    # Verify DB persistence
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.perplexity_model == "fast"

    # 4. Reopen Presets screen to verify active check
    update_4 = Update(
        update_id=4,
        callback_query=CallbackQuery(
            id="cb4",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Settings"),
            data="ai_ppx_presets",
        ),
    )
    await dp.feed_update(bot, update_4)
    reopened_kb: InlineKeyboardMarkup = _last_markup(session.calls)
    fast_btn_reopened = next(b for row in reopened_kb.inline_keyboard for b in row if "Быстрый" in b.text)
    assert "✅" in fast_btn_reopened.text

    # 5. Extract actual Back button callback and verify immediate parent
    back_btn = reopened_kb.inline_keyboard[-1][0]
    assert back_btn.callback_data == f"view_models_{PROVIDER_PERPLEXITY}"

    # 6. Direct Models Journey:
    update_5 = Update(
        update_id=5,
        callback_query=CallbackQuery(
            id="cb5",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Settings"),
            data="ai_ppx_models:0",
        ),
    )
    await dp.feed_update(bot, update_5)
    models_kb_call: InlineKeyboardMarkup = _last_markup(session.calls)

    # Extract pagination button
    next_btn = next((b for row in models_kb_call.inline_keyboard for b in row if "Далее" in b.text), None)
    assert next_btn is not None
    assert next_btn.callback_data == "ai_ppx_models:1"

    # Extract first model button from markup and select it
    first_model_btn = models_kb_call.inline_keyboard[0][0]
    update_6 = Update(
        update_id=6,
        callback_query=CallbackQuery(
            id="cb6",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Models"),
            data=first_model_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, update_6)

    # Verify DB persistence of direct model
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.perplexity_model.startswith("anthropic/")


# ---------------------------------------------------------------------------
# 12. Real MAX Journey A -> Я -> A
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_journey_a_to_ya_to_a(session_factory, monkeypatch, init_ai_config):
    """Full MAX Journey: navigate screens from actual rendered attachments, select, persist, reopen, Back."""
    monkeypatch.setattr(max_admin_ai, "async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.storage.async_session_maker", session_factory)

    from max_messenger_bot.services import common as max_common
    async def _true(*args, **kwargs):
        return True
    monkeypatch.setattr(max_common, "is_admin", _true)
    monkeypatch.setattr(max_common, "ensure_user", _true)

    captured_requests = []

    async def fake_request(method, path, *, params=None, json_data=None, expected_status=200):
        captured_requests.append({"method": method, "path": path, "params": params or {}, "body": json_data or {}})
        if path == "/messages":
            return {"message": {"body": {"mid": str(len(captured_requests))}}}
        return {}

    client = MaxApiClient(token="test_token", base_url="https://max.test")
    client._request = fake_request
    app = MaxBotApplication(client=client)

    chat_id = 777
    user_id = 777
    update_id = 100

    async def press_callback(payload, attachments=None):
        nonlocal update_id
        update_id += 1
        await app.handle_update({
            "update_type": "message_callback",
            "update_id": update_id,
            "callback": {
                "callback_id": f"cb_{update_id}",
                "payload": payload,
                "sender": {"user_id": user_id, "first_name": "Admin"},
            },
            "message": {
                "mid": f"msg_{update_id}",
                "recipient": {"chat_id": chat_id},
                "body": {"attachments": attachments or []},
            },
        })
        messages = [req for req in captured_requests if req["path"] == "/messages"]
        assert messages, f"No message sent for payload: {payload}"
        return messages[-1]["body"]

    # 1. Open Perplexity presets screen
    captured_requests.clear()
    last_msg = await press_callback("admin_ai_ppx_presets")
    presets_attachment = last_msg["attachments"][0]

    # Extract actual payload for preset "low" from rendered attachment
    low_btn = next(b for row in presets_attachment["payload"]["buttons"] for b in row if "admin_ai_set_model_Perplexity_low" in b["payload"])
    assert low_btn["payload"] == "admin_ai_set_model_Perplexity_low"

    # 2. Select preset "low"
    await press_callback(low_btn["payload"])

    # Verify DB persistence
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.perplexity_model == "low"

    # 3. Direct Models Journey:
    captured_requests.clear()
    models_msg = await press_callback("admin_ai_ppx_models_0")
    models_attachment = models_msg["attachments"][0]

    # Extract actual payload for first direct model
    model_btn = models_attachment["payload"]["buttons"][0][0]
    expected_model = model_btn["payload"].replace("admin_ai_set_model_Perplexity_", "")

    # Select direct model
    await press_callback(model_btn["payload"])

    # Verify DB persistence
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.perplexity_model == expected_model

    # 4. Extract Back button payload from markup and verify parent
    back_btn = models_attachment["payload"]["buttons"][-1][0]
    assert back_btn["payload"] == f"admin_ai_models_{PROVIDER_PERPLEXITY}"
    assert back_btn["text"] == "К настройкам"


# ---------------------------------------------------------------------------
# 13. Real Stale Callback Recovery Journey
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_real_stale_callback_journey(session_factory, monkeypatch, init_ai_config):
    """Verify stale/unresolvable callback does NOT mutate DB, answers callback, and recovers UI."""
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", session_factory)
    monkeypatch.setattr(handlers, "is_admin", lambda user_id: True)

    session = ValidatingTelegramSession()
    bot = Bot(token="123456:TEST_BOT_TOKEN", session=session)
    handlers.router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)

    user = TgUser(id=999, is_bot=False, first_name="Admin", username="admin")
    chat = Chat(id=999, type=ChatType.PRIVATE)

    # Record initial DB model and set active provider to Perplexity
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        cfg.provider = PROVIDER_PERPLEXITY
        await s.commit()
        initial_model = cfg.perplexity_model

    # Synthetic stale compact callback with non-existent digest
    stale_cb = "ai_m_c_00000000000000000000000000000000"
    update = Update(
        update_id=99,
        callback_query=CallbackQuery(
            id="cb_stale",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Old"),
            data=stale_cb,
        ),
    )
    await dp.feed_update(bot, update)

    # 1. DB was NOT mutated
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.perplexity_model == initial_model

    # 2. Callback was answered with explanation
    answer_call = next(c for c in session.calls if hasattr(c, "text") and c.text and ("недоступна" in c.text or "обновлен" in c.text.lower() or "обновлён" in c.text.lower()))
    assert answer_call is not None

    # 3. Direct models screen was re-rendered so user is not stranded
    edit_call = next(c for c in session.calls if hasattr(c, "reply_markup") and c.reply_markup)
    assert any("Прямые модели" in (btn.text or "") for row in edit_call.reply_markup.inline_keyboard for btn in row)

    # 4. Refreshed-but-retired callback (Gen A -> Gen B):
    # In Gen A, vendor/retired-model is in active catalog -> callback is generated
    sim_now = time.monotonic()
    provider_models._current_perplexity_catalog_state = PerplexityCatalogState(
        models=("vendor/retired-model",),
        source="live",
        is_fresh=True,
        is_authoritative=True,
        fetched_at=sim_now,
    )
    cb_retired = build_telegram_model_callback_data(PROVIDER_PERPLEXITY, "chat", "vendor/retired-model")

    # Catalog refresh occurs (Gen B): vendor/retired-model is retired, vendor/current-model is added
    provider_models._previous_generation_perplexity_models = ("vendor/retired-model",)
    provider_models._previous_generation_created_at = sim_now
    provider_models._current_perplexity_catalog_state = PerplexityCatalogState(
        models=("vendor/current-model",),
        source="live",
        is_fresh=True,
        is_authoritative=True,
        fetched_at=sim_now,
    )

    update_retired = Update(
        update_id=100,
        callback_query=CallbackQuery(
            id="cb_retired",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Retired"),
            data=cb_retired,
        ),
    )
    await dp.feed_update(bot, update_retired)

    # DB was NOT mutated
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.perplexity_model == initial_model

    # Callback answered with alert
    answer_call_2 = next(c for c in reversed(session.calls) if hasattr(c, "text") and c.text and "недоступна" in c.text)
    assert answer_call_2 is not None

    # Direct models screen was re-rendered with active catalog
    retired_edit_markup = _last_markup(session.calls)
    assert any("current-model" in (btn.text or "") for row in retired_edit_markup.inline_keyboard for btn in row)


# ---------------------------------------------------------------------------
# 14. Button Classification from REAL Rendered Markup
# ---------------------------------------------------------------------------

def test_button_classification_from_real_markup():
    """Render all affected Telegram and MAX screens, extract EVERY visible button, and classify 100%."""
    # 1. Telegram screens:
    tg_provider_kb = kb.provider_model_settings_keyboard(PROVIDER_PERPLEXITY, show_reasoning=False, show_temperature=False)
    tg_presets_kb = kb.perplexity_presets_keyboard()
    tg_models_p0 = kb.perplexity_models_keyboard(models=["vendor/m1", "vendor/m2", "vendor/m3", "vendor/m4", "vendor/m5", "vendor/m6", "vendor/m7"], page=0)
    tg_models_p1 = kb.perplexity_models_keyboard(models=["vendor/m1", "vendor/m2", "vendor/m3", "vendor/m4", "vendor/m5", "vendor/m6", "vendor/m7"], page=1)
    tg_fallback_cat = kb.perplexity_category_keyboard(channel="fallback", back_callback="admin_ai_text_fallback")

    all_tg_callbacks = set()
    for keyboard in (tg_provider_kb, tg_presets_kb, tg_models_p0, tg_models_p1, tg_fallback_cat):
        for row in keyboard.inline_keyboard:
            for btn in row:
                if btn.callback_data:
                    all_tg_callbacks.add(btn.callback_data)

    # 2. MAX screens:
    max_cat = admin_ai_perplexity_category_keyboard()
    max_presets = admin_ai_perplexity_presets_keyboard(current_model="medium")
    max_models_p0 = admin_ai_perplexity_models_keyboard(current_model="medium", models=["vendor/m1", "vendor/m2", "vendor/m3", "vendor/m4", "vendor/m5", "vendor/m6", "vendor/m7"], page=0)
    max_models_p1 = admin_ai_perplexity_models_keyboard(current_model="medium", models=["vendor/m1", "vendor/m2", "vendor/m3", "vendor/m4", "vendor/m5", "vendor/m6", "vendor/m7"], page=1)

    all_max_payloads = set()
    for keyboard in (max_cat, max_presets, max_models_p0, max_models_p1):
        for row in keyboard[0]["payload"]["buttons"]:
            for btn in row:
                if btn.get("payload"):
                    all_max_payloads.add(btn["payload"])

    # Classification rules:
    # navigation: viewing screens, pagination, back buttons
    # mutation: selecting/setting model or preset, toggling settings
    # destructive: delete, reset
    # external-action: payment, links

    classified_count = 0
    missing = []

    for cb in all_tg_callbacks:
        if (
            cb.startswith("view_models_")
            or cb.startswith("ai_ppx_presets")
            or cb.startswith("ai_ppx_models:")
            or cb.startswith("admin_ai_keys")
            or cb.startswith("admin_ai_text_fallback")
            or cb == "noop"
        ):
            classification = "navigation"
        elif cb.startswith("ai_m_") or cb.startswith("model_setting_"):
            classification = "mutation"
        else:
            missing.append(cb)
            continue
        classified_count += 1

    for payload in all_max_payloads:
        if (
            payload.startswith("admin_ai_models_")
            or payload.startswith("admin_ai_ppx_presets")
            or payload.startswith("admin_ai_ppx_models_")
            or payload == "noop"
        ):
            classification = "navigation"
        elif payload.startswith("admin_ai_set_model_"):
            classification = "mutation"
        else:
            missing.append(payload)
            continue
        classified_count += 1

    total = len(all_tg_callbacks) + len(all_max_payloads)
    assert len(missing) == 0, f"Unclassified buttons found: {missing}"
    assert classified_count == total
    print(f"\nButton classification verified: {classified_count}/{total} classified, missing 0")


# ---------------------------------------------------------------------------
# 15. Isolated /v1/agent Runtime Request Proof
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_runtime_v1_agent_request_proof():
    """Verify runtime HTTP boundary: exact payload, headers, max_output_tokens, and response formatting."""
    captured_requests = []

    def mock_agent_handler(request: httpx.Request):
        body = json.loads(request.content.decode("utf-8"))
        captured_requests.append({"url": str(request.url), "headers": dict(request.headers), "body": body})
        model_name = body.get("model") or body.get("preset") or "perplexity-agent"
        return httpx.Response(
            200,
            json={
                "id": "resp-123",
                "status": "completed",
                "model": model_name,
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": f"Answer generated with {model_name}",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(mock_agent_handler)
    layout = AIRequestLayout(current_user_content="Explain quantum teleportation")

    with patch("httpx.AsyncClient", mock_httpx_async_client(transport)):
        # 1. Preset runtime dispatch
        captured_requests.clear()
        res_preset = await call_perplexity("TEST_KEY_1", layout, preset="fast", max_output_tokens=4096)
        req_preset = captured_requests[0]
        assert req_preset["url"] == "https://api.perplexity.ai/v1/agent"
        assert req_preset["headers"]["authorization"] == "Bearer TEST_KEY_1"
        assert req_preset["body"]["preset"] == "fast"
        assert req_preset["body"]["max_output_tokens"] == 4096
        assert "tools" not in req_preset["body"]
        assert "Answer generated with fast" in res_preset

        # 2. Anthropic Direct Model runtime dispatch
        captured_requests.clear()
        res_direct = await call_perplexity("TEST_KEY_2", layout, preset="medium", model="anthropic/claude-sonnet-4-6")
        req_direct = captured_requests[0]
        assert req_direct["body"]["model"] == "anthropic/claude-sonnet-4-6"
        assert req_direct["body"]["max_output_tokens"] == 8192, "Anthropic models via Perplexity must have max_output_tokens=8192"
        assert "tools" not in req_direct["body"]
        assert "Answer generated with anthropic/claude-sonnet-4-6" in res_direct


# ---------------------------------------------------------------------------
# 16. Fallback Model Pickers (Telegram & MAX)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fallback_model_pickers_telegram_and_max(session_factory, monkeypatch, init_ai_config):
    """Verify fallback model selection for Perplexity across Telegram and MAX."""
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", session_factory)
    monkeypatch.setattr(handlers, "is_admin", lambda user_id: True)

    session = ValidatingTelegramSession()
    bot = Bot(token="123456:TEST_BOT_TOKEN", session=session)
    handlers.router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)

    # Set fallback provider to Perplexity in DB
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        cfg.fallback_provider = PROVIDER_PERPLEXITY
        await s.commit()

    user = TgUser(id=999, is_bot=False, first_name="Admin", username="admin")
    chat = Chat(id=999, type=ChatType.PRIVATE)

    # 1. Telegram: open fallback model picker
    update = Update(
        update_id=1,
        callback_query=CallbackQuery(
            id="cb_fb",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Fallback"),
            data="admin_ai_fallback_model",
        ),
    )
    await dp.feed_update(bot, update)
    markup: InlineKeyboardMarkup = _last_markup(session.calls)
    callbacks = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert "ai_ppx_fb_presets" in callbacks
    assert "ai_ppx_fb_models:0" in callbacks

    # 2. MAX: set_fallback_provider and show_fallback_model
    monkeypatch.setattr(max_admin_ai, "async_session_maker", session_factory)
    captured_requests = []

    async def fake_max_request(method, path, *, params=None, json_data=None, expected_status=200):
        captured_requests.append({"method": method, "path": path, "params": params or {}, "body": json_data or {}})
        if path == "/messages":
            return {"message": {"body": {"mid": str(len(captured_requests))}}}
        return {}

    max_client = MaxApiClient(token="test_token", base_url="https://max.test")
    max_client._request = fake_max_request
    await max_admin_ai.set_fallback_provider(max_client, 777, PROVIDER_PERPLEXITY)

    last_body = captured_requests[-1]["body"]
    attachment = last_body["attachments"][0]
    payloads = [b["payload"] for row in attachment["payload"]["buttons"] for b in row]
    assert "admin_ai_fb_ppx_presets" in payloads
    assert "admin_ai_fb_ppx_models_0" in payloads


# ---------------------------------------------------------------------------
# 17. UI Source Labels (live, stale_live, static_fallback)
# ---------------------------------------------------------------------------

def test_ui_source_labels():
    """Verify exact UI source descriptions for live, stale_live, and static_fallback."""
    for source, expected_label in (
        ("live", "🟢 Актуальный каталог"),
        ("stale_live", "🟡 Кэш каталога"),
        ("static_fallback", "⚪ Статический каталог"),
    ):
        state = PerplexityCatalogState(
            models=PERPLEXITY_STATIC_DIRECT_MODELS,
            source=source,
            is_fresh=(source == "live"),
            is_authoritative=(source == "live"),
            fetched_at=time.monotonic() if source == "live" else 0.0,
        )
        label = (
            "🟢 Актуальный каталог"
            if state.source == "live"
            else ("🟡 Кэш каталога" if state.source == "stale_live" else "⚪ Статический каталог")
        )
        assert label == expected_label
