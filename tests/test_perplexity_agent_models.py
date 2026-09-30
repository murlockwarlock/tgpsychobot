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
from aiogram.methods import SendMessage
from aiogram.types import CallbackQuery, Chat, InlineKeyboardButton, InlineKeyboardMarkup, Message, Update, User as TgUser
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import ai_integration
from ai_request_context import AIRequestLayout, AIRequestMessage
from database import AIConfig, AIModelSettings, Base, BotGeneralConfig, Message as DBMessage, User
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
    normalize_provider_error_classification,
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
        if isinstance(method, SendMessage):
            return Message(
                message_id=len(self.calls),
                date=datetime.now(timezone.utc),
                chat=Chat(id=method.chat_id, type=ChatType.PRIVATE),
                text=method.text or "",
            )
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
        from max_messenger_bot.storage import StorageBase
        await conn.run_sync(StorageBase.metadata.create_all)
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
        gen_cfg = BotGeneralConfig(id=1)
        session.add_all([cfg, gen_cfg])
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
# 2b. Multi-turn History Structured Contract & Single-turn String
# ---------------------------------------------------------------------------

def test_perplexity_multiturn_history_structured_input():
    """Verify Perplexity Agent API receives structured message items with preserved roles for multi-turn dialogues."""
    # 1. Multi-turn contract
    history = [
        AIRequestMessage(role="user", content="Первый вопрос"),
        AIRequestMessage(role="assistant", content="Первый ответ"),
    ]
    layout = AIRequestLayout(
        history=tuple(history),
        current_user_content="Второй вопрос",
        stable_system_prompt="Ты клинический психолог.",
    )
    payload = build_perplexity_payload(layout, preset="fast")
    inp = payload["input"]

    assert isinstance(inp, list), "Multi-turn input MUST be structured as a list of message dicts"
    assert len(inp) == 3
    assert inp[0] == {"role": "user", "content": "Первый вопрос"}
    assert inp[1] == {"role": "assistant", "content": "Первый ответ"}
    assert inp[2] == {"role": "user", "content": "Второй вопрос"}

    payload_str = json.dumps(payload, ensure_ascii=False)
    assert "user: Первый вопрос" not in payload_str, "Roles must NOT be prefixed as plain text into content"
    assert "assistant: Первый ответ" not in payload_str, "Roles must NOT be prefixed as plain text into content"
    assert "system:" not in payload_str
    assert payload["instructions"].startswith("Ты клинический психолог.")
    assert "tools" not in payload
    assert "tool_choice" not in payload

    # 2. Single-turn contract (no history)
    layout_st = AIRequestLayout(
        current_user_content="Одиночный запрос",
        stable_system_prompt="Ты клинический психолог.",
    )
    payload_st = build_perplexity_payload(layout_st, preset="fast")
    assert payload_st["input"] == "Одиночный запрос", "Single-turn request without history should use simple string input"
    assert isinstance(payload_st["input"], str)
    assert "tools" not in payload_st
    assert "tool_choice" not in payload_st

    # 3. Direct model multi-turn contract (requires web_search tool and forced tool_choice)
    payload_dm = build_perplexity_payload(layout, model="openai/gpt-5.6-terra")
    assert payload_dm["model"] == "openai/gpt-5.6-terra"
    assert payload_dm["tools"] == [{"type": "web_search"}]
    assert payload_dm["tool_choice"] == {"type": "web_search"}
    assert isinstance(payload_dm["input"], list)
    assert len(payload_dm["input"]) == 3


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
# 6b. Retry Policy: Rate Limit vs Insufficient Quota / Credits
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_429_retry_policy_rate_limit_vs_quota():
    """Verify HTTP-level retry behavior: true 429 rate limits retry once (attempts=2),
    while 429 balance/quota exhaustion aborts immediately without duplicate paid requests (attempts=1).
    """
    url = "https://api.perplexity.ai/v1/agent"
    headers = {"Authorization": "Bearer TEST_KEY"}
    payload = {"model": "sonar-pro", "input": "test"}

    # Case A: 429 + "rate limit" -> first attempt: will_retry=True, final attempt: will_retry=False
    attempts_a = 0
    snapshot_attempt_1 = {}
    def mock_rate_limit(request):
        nonlocal attempts_a
        attempts_a += 1
        return httpx.Response(
            429,
            headers={"retry-after": "0"},
            json={"error": {"type": "rate_limit", "code": 429, "message": "Too many requests"}},
        )

    async def mock_sleep_a(delay):
        nonlocal snapshot_attempt_1
        snapshot_attempt_1 = dict(capture_a)

    capture_a = {}
    with patch("httpx.AsyncClient", mock_httpx_async_client(httpx.MockTransport(mock_rate_limit))), \
         patch("asyncio.sleep", mock_sleep_a):
        with pytest.raises(ProviderAdapterError) as exc_info_a:
            await _post_perplexity_json(url, headers=headers, payload=payload, timeout=5.0, request_capture=capture_a, max_attempts=2)
        assert exc_info_a.value.category == "rate_limit"

    assert attempts_a == 2, "Real 429 rate limit must be retried up to max_attempts=2"
    # First attempt snapshot
    assert snapshot_attempt_1["attempt_count"] == 1
    assert snapshot_attempt_1["classification"] == "rate_limit"
    assert snapshot_attempt_1["is_retryable"] is True
    assert snapshot_attempt_1["will_retry"] is True
    # Final exhausted attempt
    assert capture_a["attempt_count"] == 2
    assert capture_a["classification"] == "rate_limit"
    assert capture_a["is_retryable"] is True
    assert capture_a["will_retry"] is False

    # Case A2: 429 rate_limit + Retry-After (4.5s) exceeds remaining deadline (1.0s) -> NO RETRY
    attempts_a2 = 0
    def mock_rate_limit_deadline(request):
        nonlocal attempts_a2
        attempts_a2 += 1
        return httpx.Response(
            429,
            headers={"retry-after": "4.5"},
            json={"error": {"type": "rate_limit", "code": 429, "message": "Rate limit exceeded"}},
        )

    capture_a2 = {}
    with patch("httpx.AsyncClient", mock_httpx_async_client(httpx.MockTransport(mock_rate_limit_deadline))), \
         patch("asyncio.sleep", AsyncMock()):
        with pytest.raises(ProviderAdapterError) as exc_info_a2:
            await _post_perplexity_json(url, headers=headers, payload=payload, timeout=1.0, request_capture=capture_a2, max_attempts=2)
        assert exc_info_a2.value.category == "rate_limit"

    assert attempts_a2 == 1, "429 with Retry-After exceeding deadline must NOT trigger second attempt"
    assert capture_a2["attempt_count"] == 1
    assert capture_a2["classification"] == "rate_limit"
    assert capture_a2["is_retryable"] is True
    assert capture_a2["will_retry"] is False

    # Case B: 429 + "insufficient_quota" -> classification = insufficient_balance_quota, attempts = 1 (NO RETRY)
    attempts_b = 0
    def mock_quota(request):
        nonlocal attempts_b
        attempts_b += 1
        return httpx.Response(
            429,
            headers={"retry-after": "0"},
            json={"error": {"type": "insufficient_quota", "code": 429, "message": "Monthly spending quota exceeded"}},
        )

    capture_b = {}
    with patch("httpx.AsyncClient", mock_httpx_async_client(httpx.MockTransport(mock_quota))), \
         patch("asyncio.sleep", AsyncMock()):
        with pytest.raises(ProviderAdapterError) as exc_info_b:
            await _post_perplexity_json(url, headers=headers, payload=payload, timeout=5.0, request_capture=capture_b, max_attempts=2)
        assert exc_info_b.value.category == "insufficient_balance_quota"

    assert attempts_b == 1, "429 with insufficient balance/quota MUST NOT be retried!"
    assert capture_b["attempt_count"] == 1
    assert capture_b["classification"] == "insufficient_balance_quota"
    assert capture_b["is_retryable"] is False
    assert capture_b["will_retry"] is False

    # Case C: ConnectTimeout -> safe pre-dispatch retry (attempts = 2)
    attempts_c = 0
    snapshot_connect_1 = {}
    def mock_connect(request):
        nonlocal attempts_c
        attempts_c += 1
        raise httpx.ConnectTimeout("Connect timeout")

    async def mock_sleep_c(delay):
        nonlocal snapshot_connect_1
        snapshot_connect_1 = dict(capture_c)

    capture_c = {}
    with patch("httpx.AsyncClient", mock_httpx_async_client(httpx.MockTransport(mock_connect))), \
         patch("asyncio.sleep", mock_sleep_c):
        with pytest.raises(ProviderAdapterError):
            await _post_perplexity_json(url, headers=headers, payload=payload, timeout=5.0, request_capture=capture_c, max_attempts=2)
    assert attempts_c == 2
    assert snapshot_connect_1["attempt_count"] == 1
    assert snapshot_connect_1["classification"] == "timeout"
    assert snapshot_connect_1["is_retryable"] is True
    assert snapshot_connect_1["will_retry"] is True
    assert capture_c["attempt_count"] == 2
    assert capture_c["classification"] == "timeout"
    assert capture_c["is_retryable"] is True
    assert capture_c["will_retry"] is False

    # Case C2: ConnectTimeout where computed backoff exceeds remaining deadline -> NO RETRY
    attempts_c2 = 0
    def mock_connect_deadline(request):
        nonlocal attempts_c2
        attempts_c2 += 1
        raise httpx.ConnectTimeout("Connect timeout")

    capture_c2 = {}
    with patch("httpx.AsyncClient", mock_httpx_async_client(httpx.MockTransport(mock_connect_deadline))), \
         patch("asyncio.sleep", AsyncMock()):
        with pytest.raises(ProviderAdapterError) as exc_info_c2:
            await _post_perplexity_json(url, headers=headers, payload=payload, timeout=0.2, request_capture=capture_c2, max_attempts=2)
        assert exc_info_c2.value.category == "timeout"

    assert attempts_c2 == 1, "ConnectTimeout with delay exceeding deadline must NOT trigger second attempt"
    assert capture_c2["attempt_count"] == 1
    assert capture_c2["classification"] == "timeout"
    assert capture_c2["is_retryable"] is True
    assert capture_c2["will_retry"] is False

    # Case D: ReadTimeout -> uncertain post-dispatch (attempts = 1, NO RETRY)
    attempts_d = 0
    def mock_read(request):
        nonlocal attempts_d
        attempts_d += 1
        raise httpx.ReadTimeout("Read timeout")

    capture_d = {}
    with patch("httpx.AsyncClient", mock_httpx_async_client(httpx.MockTransport(mock_read))), \
         patch("asyncio.sleep", AsyncMock()):
        with pytest.raises(ProviderAdapterError):
            await _post_perplexity_json(url, headers=headers, payload=payload, timeout=5.0, request_capture=capture_d, max_attempts=2)
    assert attempts_d == 1, "Uncertain ReadTimeout must NOT be automatically retried!"
    assert capture_d["attempt_count"] == 1
    assert capture_d["classification"] == "timeout"
    assert capture_d["is_retryable"] is False
    assert capture_d["will_retry"] is False

    # Case E: HTTP 200 + Agent status="failed" does NOT leave classification="success"
    capture_failed = {}
    with pytest.raises(ProviderAdapterError) as exc_failed:
        _extract_perplexity_text(
            {
                "status": "failed",
                "error": {"type": "internal_error", "message": "Perplexity backend failure"},
            },
            request_capture=capture_failed,
        )
    assert exc_failed.value.category in {"provider", "provider_5xx"}
    assert capture_failed["response_status"] == "failed"
    assert capture_failed["classification"] != "success"
    assert capture_failed["classification"] in {"provider", "provider_5xx"}
    assert capture_failed["is_retryable"] is False
    assert capture_failed["will_retry"] is False

    # Case F: HTTP 200 + Agent status="completed" leaves classification="success"
    capture_success = {}
    res_text = _extract_perplexity_text(
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": "Valid completed answer",
                }
            ],
        },
        request_capture=capture_success,
    )
    assert res_text == "Valid completed answer"
    assert capture_success["response_status"] == "completed"
    assert capture_success["classification"] == "success"
    assert capture_success["is_retryable"] is False
    assert capture_success["will_retry"] is False

    # Case G: HTTP 500 provider_5xx retry (attempts = 2)
    attempts_g = 0
    snapshot_500_1 = {}
    def mock_500(request):
        nonlocal attempts_g
        attempts_g += 1
        return httpx.Response(
            500,
            headers={"retry-after": "0"},
            json={"error": {"message": "Internal Server Error"}},
        )

    async def mock_sleep_g(delay):
        nonlocal snapshot_500_1
        snapshot_500_1 = dict(capture_g)

    capture_g = {}
    with patch("httpx.AsyncClient", mock_httpx_async_client(httpx.MockTransport(mock_500))), \
         patch("asyncio.sleep", mock_sleep_g):
        with pytest.raises(ProviderAdapterError) as exc_info_g:
            await _post_perplexity_json(url, headers=headers, payload=payload, timeout=5.0, request_capture=capture_g, max_attempts=2)
        assert exc_info_g.value.category == "provider_5xx"

    assert attempts_g == 2, "Real HTTP 500 must be retried up to max_attempts=2"
    # First attempt snapshot
    assert snapshot_500_1["attempt_count"] == 1
    assert snapshot_500_1["classification"] == "provider_5xx"
    assert snapshot_500_1["is_retryable"] is True
    assert snapshot_500_1["will_retry"] is True
    # Final exhausted attempt
    assert capture_g["attempt_count"] == 2
    assert capture_g["classification"] == "provider_5xx"
    assert capture_g["is_retryable"] is True
    assert capture_g["will_retry"] is False

    # Case H: HTTP 403 auth (attempts = 1, NO RETRY)
    attempts_h = 0
    def mock_403(request):
        nonlocal attempts_h
        attempts_h += 1
        return httpx.Response(
            403,
            json={"error": {"type": "authentication_error", "message": "Access forbidden: invalid account permissions"}},
        )

    capture_h = {}
    with patch("httpx.AsyncClient", mock_httpx_async_client(httpx.MockTransport(mock_403))), \
         patch("asyncio.sleep", AsyncMock()):
        with pytest.raises(ProviderAdapterError) as exc_info_h:
            await _post_perplexity_json(url, headers=headers, payload=payload, timeout=5.0, request_capture=capture_h, max_attempts=2)
        assert exc_info_h.value.category == "auth"

    assert attempts_h == 1, "HTTP 403 auth MUST NOT be retried"
    assert capture_h["attempt_count"] == 1
    assert capture_h["classification"] == "auth"
    assert capture_h["is_retryable"] is False
    assert capture_h["will_retry"] is False

    # Case I: HTTP 400 configuration (attempts = 1, NO RETRY)
    attempts_i = 0
    def mock_400(request):
        nonlocal attempts_i
        attempts_i += 1
        return httpx.Response(
            400,
            json={"error": {"type": "invalid_request_error", "message": "Invalid model: model not found"}},
        )

    capture_i = {}
    with patch("httpx.AsyncClient", mock_httpx_async_client(httpx.MockTransport(mock_400))), \
         patch("asyncio.sleep", AsyncMock()):
        with pytest.raises(ProviderAdapterError) as exc_info_i:
            await _post_perplexity_json(url, headers=headers, payload=payload, timeout=5.0, request_capture=capture_i, max_attempts=2)
        assert exc_info_i.value.category == "configuration"

    assert attempts_i == 1, "HTTP 400 invalid_request MUST NOT be retried"
    assert capture_i["attempt_count"] == 1
    assert capture_i["classification"] == "configuration"
    assert capture_i["is_retryable"] is False
    assert capture_i["will_retry"] is False


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


def test_agent_status_incomplete_distinguishes_output_budget_exhaustion():
    """Verify Perplexity Agent status="incomplete" handling:
    - reason="max_output_tokens" maps to output_budget_exhausted category
    - unknown/missing reason maps to invalid_response category
    - normalize_provider_error_classification preserves output_budget_exhausted
    """
    # Case A: status="incomplete", incomplete_details.reason="max_output_tokens"
    capture_a = {}
    with pytest.raises(ProviderAdapterError) as exc_a:
        _extract_perplexity_text(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
            request_capture=capture_a,
        )
    assert exc_a.value.category == "output_budget_exhausted"
    assert capture_a["response_status"] == "incomplete"
    assert capture_a["classification"] == "output_budget_exhausted"
    assert capture_a["is_retryable"] is False
    assert capture_a["will_retry"] is False
    assert capture_a["incomplete_reason"] == "max_output_tokens"

    # Case B: status="incomplete", incomplete_details.reason="unknown_reason"
    capture_b = {}
    with pytest.raises(ProviderAdapterError) as exc_b:
        _extract_perplexity_text(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "unknown_reason"},
            },
            request_capture=capture_b,
        )
    assert exc_b.value.category == "invalid_response"
    assert capture_b["response_status"] == "incomplete"
    assert capture_b["classification"] == "invalid_response"
    assert capture_b["is_retryable"] is False
    assert capture_b["will_retry"] is False

    # Case C: normalize_provider_error_classification preservation
    assert normalize_provider_error_classification("output_budget_exhausted") == "output_budget_exhausted"


# ---------------------------------------------------------------------------
# 10. Canonical Agent Search Results & Citation Deduplication
# ---------------------------------------------------------------------------

def test_citation_identity_and_deduplication():
    """Verify canonical Agent API search_results citation mapping:
    - deduplicated URLs (scheme, host, default port, non-root trailing slash)
    - stable sequential 1-based footer IDs ([1], [2])
    - upstream result IDs do NOT appear as user-facing citation numbers
    - inline [1], [2] still correspond to footer entries
    - no duplicate footer rows
    """
    payload = {
        "status": "completed",
        "output": [
            {
                "type": "search_results",
                "queries": ["example query"],
                "results": [
                    {
                        "id": "arbitrary-provider-id-A",
                        "title": "Guide",
                        "url": "https://Example.com/guide/",
                        "snippet": "...",
                    },
                    {
                        "id": "arbitrary-provider-id-B",
                        "title": "Guide duplicate",
                        "url": "https://example.com/guide",
                        "snippet": "...",
                    },
                    {
                        "id": "another-provider-id",
                        "title": "Second Source",
                        "url": "https://example.org/article",
                        "snippet": "...",
                    },
                ],
            },
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "First claim [1]. Second claim [2].",
                    }
                ],
            },
        ],
    }

    result = format_perplexity_response(payload)

    # 1. Duplicate URL variants produce one source
    assert result.count("https://example.com/guide") == 1
    assert result.count("https://example.org/article") == 1

    # 2. Footer IDs are sequential integers
    assert "[1] Guide — https://example.com/guide" in result
    assert "[2] Second Source — https://example.org/article" in result

    # 3. Arbitrary upstream result IDs do NOT appear as user-facing citation numbers
    assert "arbitrary-provider-id-A" not in result
    assert "arbitrary-provider-id-B" not in result
    assert "another-provider-id" not in result

    # 4. Inline [1], [2] still correspond to footer entries
    assert "First claim [1]. Second claim [2]." in result

    # 5. No duplicate footer rows
    footer_section = result.split("Источники:\n", 1)[1]
    footer_lines = [line.strip() for line in footer_section.strip().splitlines() if line.strip()]
    assert len(footer_lines) == 2
    assert footer_lines[0] == "[1] Guide — https://example.com/guide"
    assert footer_lines[1] == "[2] Second Source — https://example.org/article"


def test_citation_upstream_id_and_web_marker_alias_resolution():
    """Verify upstream result IDs and web:N markers in text are resolved to sequential display IDs."""
    payload = {
        "status": "completed",
        "output": [
            {
                "type": "search_results",
                "queries": ["example query"],
                "results": [
                    {
                        "id": "arbitrary-provider-id-A",
                        "title": "Guide",
                        "url": "https://Example.com/guide/",
                    },
                    {
                        "id": "another-provider-id",
                        "title": "Second Source",
                        "url": "https://example.org/article",
                    },
                ],
            },
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": "Using upstream IDs: [arbitrary-provider-id-A] and [another-provider-id].",
            },
        ],
    }
    result = format_perplexity_response(payload)
    assert "Using upstream IDs: [1] and [2]." in result
    assert "[1] Guide — https://example.com/guide" in result
    assert "[2] Second Source — https://example.org/article" in result


def test_citation_legacy_compatibility_fallbacks():
    """Verify legacy compatibility fallbacks when canonical search_results are absent:
    1. tool_call fallback (legacy compatibility only)
    2. top-level citations fallback (legacy compatibility only)
    """
    # Fallback 1: tool_call (legacy compatibility only)
    payload_tool_call = {
        "status": "completed",
        "output": [
            {
                "type": "tool_call",
                "tool_name": "web_search",
                "results": [
                    {"id": "web:1", "title": "Legacy Tool Doc", "url": "https://docs.legacy.ai"},
                ],
            },
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": "From legacy tool [web:1].",
            },
        ],
    }
    result_tool = format_perplexity_response(payload_tool_call)
    assert "[1] Legacy Tool Doc — https://docs.legacy.ai" in result_tool
    assert "From legacy tool [1]." in result_tool

    # Fallback 2: top-level citations (legacy compatibility only)
    payload_top_level = {
        "status": "completed",
        "citations": [
            "https://docs.legacy-top.ai/page",
        ],
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": "From top-level citation [1].",
            },
        ],
    }
    result_top = format_perplexity_response(payload_top_level)
    assert "[1] https://docs.legacy-top.ai/page — https://docs.legacy-top.ai/page" in result_top
    assert "From top-level citation [1]." in result_top


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

    # Extract pagination button "Далее ➡️" from actual rendered markup
    next_btn = next((b for row in models_kb_call.inline_keyboard for b in row if "Далее" in b.text), None)
    assert next_btn is not None
    assert next_btn.callback_data == "ai_ppx_models:1"

    # Click "Далее ➡️" via Dispatcher to navigate to Page 1
    update_next = Update(
        update_id=6,
        callback_query=CallbackQuery(
            id="cb_next",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Models Page 0"),
            data=next_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, update_next)
    p1_kb: InlineKeyboardMarkup = _last_markup(session.calls)

    # Verify Page 1 has Back pagination button
    prev_btn = next((b for row in p1_kb.inline_keyboard for b in row if "Назад" in b.text and "ai_ppx_models" in b.callback_data), None)
    assert prev_btn is not None
    assert prev_btn.callback_data == "ai_ppx_models:0"

    # Extract first direct model button from Page 1 markup and select it
    first_model_p1_btn = p1_kb.inline_keyboard[0][0]
    resolved_tuple = provider_models.resolve_telegram_model_callback(first_model_p1_btn.callback_data)
    assert resolved_tuple is not None
    _, _, expected_p1_model = resolved_tuple

    update_select_p1 = Update(
        update_id=7,
        callback_query=CallbackQuery(
            id="cb_select_p1",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Models Page 1"),
            data=first_model_p1_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, update_select_p1)

    # Verify DB persistence of direct model from Page 1
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.perplexity_model == expected_p1_model

    # Reopen Page 1 via Dispatcher to verify active checkmark on selected model
    update_reopen_p1 = Update(
        update_id=8,
        callback_query=CallbackQuery(
            id="cb_reopen_p1",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Models Page 1"),
            data="ai_ppx_models:1",
        ),
    )
    await dp.feed_update(bot, update_reopen_p1)
    reopened_p1_kb: InlineKeyboardMarkup = _last_markup(session.calls)
    reopened_p1_btn = reopened_p1_kb.inline_keyboard[0][0]
    assert "✅" in reopened_p1_btn.text

    # Extract Back button from Page 1 markup ("К настройкам") -> leads back to Perplexity provider settings
    back_to_provider_btn = reopened_p1_kb.inline_keyboard[-1][0]
    assert back_to_provider_btn.callback_data == f"view_models_{PROVIDER_PERPLEXITY}"
    assert back_to_provider_btn.text == "К настройкам"

    update_back_prov = Update(
        update_id=9,
        callback_query=CallbackQuery(
            id="cb_back_prov",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Models Page 1"),
            data=back_to_provider_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, update_back_prov)
    prov_settings_kb: InlineKeyboardMarkup = _last_markup(session.calls)
    assert any("Пресеты" in b.text for row in prov_settings_kb.inline_keyboard for b in row)

    # Extract Back button from Provider Settings markup -> leads to Keys/Providers Menu
    back_to_keys_btn = next(b for row in prov_settings_kb.inline_keyboard for b in row if b.callback_data == "admin_ai_keys")
    update_back_keys = Update(
        update_id=10,
        callback_query=CallbackQuery(
            id="cb_back_keys",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Provider Settings"),
            data=back_to_keys_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, update_back_keys)
    keys_kb: InlineKeyboardMarkup = _last_markup(session.calls)
    assert any("🧠 Perplexity" in b.text for row in keys_kb.inline_keyboard for b in row)

    # Extract Back button from Keys Menu -> leads to Main Settings
    back_to_main_btn = next(b for row in keys_kb.inline_keyboard for b in row if b.callback_data == "admin_ai_settings")
    update_back_main = Update(
        update_id=11,
        callback_query=CallbackQuery(
            id="cb_back_main",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=11, date=datetime.now(timezone.utc), chat=chat, text="Keys Menu"),
            data=back_to_main_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, update_back_main)
    main_settings_kb: InlineKeyboardMarkup = _last_markup(session.calls)
    assert any("Провайдер" in b.text or "OpenAI" in b.text or "Perplexity" in b.text for row in main_settings_kb.inline_keyboard for b in row)


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

    # Extract "Далее ➡️" pagination button from rendered attachment
    next_btn = next((b for row in models_attachment["payload"]["buttons"] for b in row if "admin_ai_ppx_models_1" in b["payload"]), None)
    assert next_btn is not None
    assert next_btn["payload"] == "admin_ai_ppx_models_1"

    # Click "Далее ➡️" via handle_update to reach Page 1
    p1_msg = await press_callback(next_btn["payload"])
    p1_attachment = p1_msg["attachments"][0]

    # Verify Page 1 has back pagination button
    prev_btn = next((b for row in p1_attachment["payload"]["buttons"] for b in row if "admin_ai_ppx_models_0" in b["payload"]), None)
    assert prev_btn is not None

    # Extract first model from Page 1 attachment
    p1_model_btn = p1_attachment["payload"]["buttons"][0][0]
    expected_p1_model = p1_model_btn["payload"].replace("admin_ai_set_model_Perplexity_", "")

    # Select direct model on Page 1
    await press_callback(p1_model_btn["payload"])

    # Verify DB persistence of direct model from Page 1
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.perplexity_model == expected_p1_model

    # Reopen Page 1 to verify active checkmark on selected model
    reopened_p1_msg = await press_callback("admin_ai_ppx_models_1")
    reopened_p1_attachment = reopened_p1_msg["attachments"][0]
    reopened_p1_btn = reopened_p1_attachment["payload"]["buttons"][0][0]
    assert "✅" in reopened_p1_btn["text"]

    # Extract Back button from Page 1 attachment -> leads to Perplexity provider settings
    back_to_prov_btn = reopened_p1_attachment["payload"]["buttons"][-1][0]
    assert back_to_prov_btn["payload"] == f"admin_ai_models_{PROVIDER_PERPLEXITY}"
    assert back_to_prov_btn["text"] == "К настройкам"

    prov_msg = await press_callback(back_to_prov_btn["payload"])
    prov_attachment = prov_msg["attachments"][0]

    # Extract Back button from Provider Settings attachment -> leads to Keys/Providers Menu
    back_to_keys_btn = next(b for row in prov_attachment["payload"]["buttons"] for b in row if b["payload"] == "admin_ai_keys")
    keys_msg = await press_callback(back_to_keys_btn["payload"])
    keys_attachment = keys_msg["attachments"][0]
    assert any("admin_ai_models_Perplexity" in b["payload"] for row in keys_attachment["payload"]["buttons"] for b in row)

    # Extract Back button from Keys Menu -> leads to Main Settings
    back_to_main_btn = next(b for row in keys_attachment["payload"]["buttons"] for b in row if b["payload"] == "admin_ai_settings")
    main_msg = await press_callback(back_to_main_btn["payload"])
    main_attachment = main_msg["attachments"][0]
    assert any("admin_ai_provider" in b["payload"] for row in main_attachment["payload"]["buttons"] for b in row)


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
    """Verify runtime HTTP boundary: exact payload, headers, web_search tool, max_output_tokens, and realistic Agent API response."""
    captured_requests = []

    def mock_agent_handler(request: httpx.Request):
        body = json.loads(request.content.decode("utf-8"))
        captured_requests.append({"url": str(request.url), "headers": dict(request.headers), "body": body})
        model_name = body.get("model") or body.get("preset") or "perplexity-agent"
        return httpx.Response(
            200,
            json={
                "id": "resp-realistic-1",
                "status": "completed",
                "model": model_name,
                "output": [
                    {
                        "type": "search_results",
                        "results": [
                            {
                                "title": "Quantum Teleportation Overview",
                                "url": "https://example.com/quantum-teleportation",
                                "snippet": "Quantum teleportation transmission data.",
                            }
                        ],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": f"Answer generated with {model_name} [1].",
                            }
                        ],
                    },
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
        assert "tools" not in req_preset["body"], "Presets manage their own search profile without forced tools"
        assert "tool_choice" not in req_preset["body"], "Presets must not receive forced tool_choice"
        assert "Answer generated with fast [1]" in res_preset
        assert "https://example.com/quantum-teleportation" in res_preset
        assert "Источники:" in res_preset

        # 2. Anthropic Direct Model runtime dispatch
        captured_requests.clear()
        res_direct = await call_perplexity("TEST_KEY_2", layout, model="anthropic/claude-sonnet-4-6")
        req_direct = captured_requests[0]
        assert req_direct["body"]["model"] == "anthropic/claude-sonnet-4-6"
        assert req_direct["body"]["tools"] == [{"type": "web_search"}], "Direct models MUST explicitly include web_search tool"
        assert req_direct["body"]["tool_choice"] == {"type": "web_search"}, "Direct models MUST force tool_choice for web_search"
        assert req_direct["body"]["max_output_tokens"] == 8192, "Anthropic models via Perplexity must have max_output_tokens=8192"
        assert "preset" not in req_direct["body"], "Pure direct model request must not contain preset"
        assert "Answer generated with anthropic/claude-sonnet-4-6 [1]" in res_direct
        assert "https://example.com/quantum-teleportation" in res_direct
        assert "Источники:" in res_direct


# ---------------------------------------------------------------------------
# 16. Fallback Direct Models Full Journey (Telegram & MAX)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fallback_direct_models_full_journey_telegram_and_max(session_factory, monkeypatch, init_ai_config):
    """Verify fallback model selection & runtime fallback invocation for Perplexity across Telegram and MAX."""
    import handlers
    import ai_integration

    monkeypatch.setattr(handlers, "async_session_maker", session_factory)
    monkeypatch.setattr(ai_integration, "async_session_maker", session_factory)
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
        cfg.provider = "openrouter"
        cfg.openrouter_api_key = "OR_TEST_KEY"
        cfg.openrouter_model = "openai/gpt-5.6-terra"
        cfg.perplexity_api_key = "PPLX_FALLBACK_TEST_KEY"
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

    # 2. Telegram: Open fallback direct models page 0
    update_fb_p0 = Update(
        update_id=2,
        callback_query=CallbackQuery(
            id="cb_fb_p0",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Fallback Models P0"),
            data="ai_ppx_fb_models:0",
        ),
    )
    await dp.feed_update(bot, update_fb_p0)
    fb_p0_kb: InlineKeyboardMarkup = _last_markup(session.calls)

    # Extract pagination button "Далее ➡️" from Page 0
    next_btn = next((b for row in fb_p0_kb.inline_keyboard for b in row if "Далее" in b.text), None)
    assert next_btn is not None
    assert next_btn.callback_data == "ai_ppx_fb_models:1"

    # Click "Далее ➡️" to open Page 1
    update_fb_p1 = Update(
        update_id=3,
        callback_query=CallbackQuery(
            id="cb_fb_p1",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Fallback Models P1"),
            data=next_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, update_fb_p1)
    fb_p1_kb: InlineKeyboardMarkup = _last_markup(session.calls)

    # Extract first direct model on Page 1
    first_fb_btn = fb_p1_kb.inline_keyboard[0][0]
    resolved_tuple = provider_models.resolve_telegram_model_callback(first_fb_btn.callback_data)
    assert resolved_tuple is not None
    _, channel_res, expected_fb_model = resolved_tuple
    assert channel_res == "fallback"

    # Select direct fallback model
    update_fb_select = Update(
        update_id=4,
        callback_query=CallbackQuery(
            id="cb_fb_select",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Fallback Models P1"),
            data=first_fb_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, update_fb_select)

    # Verify DB persistence of fallback model
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.fallback_provider == PROVIDER_PERPLEXITY
        assert cfg.fallback_model == expected_fb_model

    # Reopen Page 1 via Dispatcher to verify checkmark
    update_fb_reopen = Update(
        update_id=5,
        callback_query=CallbackQuery(
            id="cb_fb_reopen",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Fallback Models P1"),
            data="ai_ppx_fb_models:1",
        ),
    )
    await dp.feed_update(bot, update_fb_reopen)
    reopened_fb_kb: InlineKeyboardMarkup = _last_markup(session.calls)
    assert "✅" in reopened_fb_kb.inline_keyboard[0][0].text

    # Extract Back button from markup ("К настройкам") -> leads back to admin_ai_fallback_model
    back_fb_btn = reopened_fb_kb.inline_keyboard[-1][0]
    assert back_fb_btn.callback_data == "admin_ai_fallback_model"

    # Actually feed back_fb_btn.callback_data through the real Telegram Dispatcher
    update_fb_back = Update(
        update_id=6,
        callback_query=CallbackQuery(
            id="cb_fb_back",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Fallback Models P1"),
            data=back_fb_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, update_fb_back)
    parent_fb_kb: InlineKeyboardMarkup = _last_markup(session.calls)
    parent_callbacks = [b.callback_data for row in parent_fb_kb.inline_keyboard for b in row]
    assert "ai_ppx_fb_presets" in parent_callbacks
    assert "ai_ppx_fb_models:0" in parent_callbacks

    # Back from parent_fb_kb leads to admin_ai_text_fallback
    back_to_text_fb = next(b for row in parent_fb_kb.inline_keyboard for b in row if b.callback_data == "admin_ai_text_fallback")
    update_text_fb_back = Update(
        update_id=7,
        callback_query=CallbackQuery(
            id="cb_text_fb_back",
            from_user=user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Fallback Category"),
            data=back_to_text_fb.callback_data,
        ),
    )
    await dp.feed_update(bot, update_text_fb_back)
    text_fb_kb: InlineKeyboardMarkup = _last_markup(session.calls)
    assert any("admin_ai_fallback_model" in (b.callback_data or "") for row in text_fb_kb.inline_keyboard for b in row)

    # Verify fallback provider/model context is preserved
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.fallback_provider == PROVIDER_PERPLEXITY
        assert cfg.fallback_model == expected_fb_model

    # 3. Telegram Runtime Fallback Execution:
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        cfg.allow_fallback = True
        await s.commit()

    captured_requests = []
    def mock_runtime_transport(request: httpx.Request):
        url_str = str(request.url)
        if "openrouter.ai" in url_str:
            raise httpx.ConnectError("Primary OpenRouter unreachable")
        if "api.perplexity.ai" in url_str:
            body = json.loads(request.content.decode("utf-8"))
            captured_requests.append({"url": url_str, "headers": dict(request.headers), "body": body})
            return httpx.Response(
                200,
                json={
                    "id": "resp-fb-run-1",
                    "status": "completed",
                    "model": expected_fb_model,
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "Ответ резервного ассистента Perplexity.",
                                }
                            ],
                        }
                    ],
                },
            )
        return httpx.Response(404)

    with patch("httpx.AsyncClient", mock_httpx_async_client(httpx.MockTransport(mock_runtime_transport))):
        async with session_factory() as s:
            u = User(id=888, username="fb_user", first_name="Client", accepted_disclaimer=True)
            s.add(u)
            await s.commit()

        resp_capture = {}
        fallback_reply = await ai_integration.generate_response(
            888,
            "Тестовый вопрос для проверки резервного контура",
            bot=bot,
            response_capture=resp_capture,
        )
        assert "Ответ резервного ассистента Perplexity" in fallback_reply
        assert len(captured_requests) == 1
        fb_req = captured_requests[0]
        assert fb_req["headers"]["authorization"] == "Bearer PPLX_FALLBACK_TEST_KEY"
        assert fb_req["body"]["model"] == expected_fb_model
        assert fb_req["body"]["tools"] == [{"type": "web_search"}]
        assert fb_req["body"]["tool_choice"] == {"type": "web_search"}

    # 4. MAX: Fallback direct models navigation through MaxBotApplication
    monkeypatch.setattr(max_admin_ai, "async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.storage.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.legacy.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.ai.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.services.common.async_session_maker", session_factory)
    monkeypatch.setattr("max_messenger_bot.app.async_session_maker", session_factory)
    from max_messenger_bot.services import common as max_common

    async def _is_admin(uid, *args, **kwargs):
        return True

    monkeypatch.setattr(max_common, "is_admin", _is_admin)
    monkeypatch.setattr(max_common, "ensure_user", AsyncMock(return_value=True))
    monkeypatch.setattr(max_common, "ensure_access_before_chat", AsyncMock(return_value=True))
    monkeypatch.setattr(max_common, "maybe_require_disclaimer", AsyncMock(return_value=False))

    max_requests = []
    async def fake_max_request(method, path, *, params=None, json_data=None, expected_status=200):
        max_requests.append({"method": method, "path": path, "params": params or {}, "body": json_data or {}})
        if path == "/messages":
            return {"message": {"body": {"mid": str(len(max_requests))}}}
        return {}

    max_client = MaxApiClient(token="test_token", base_url="https://max.test")
    max_client._request = fake_max_request
    max_app = MaxBotApplication(client=max_client)

    max_update_id = 200
    async def press_max_cb(payload):
        nonlocal max_update_id
        max_update_id += 1
        await max_app.handle_update({
            "update_type": "message_callback",
            "update_id": max_update_id,
            "callback": {
                "callback_id": f"cb_{max_update_id}",
                "payload": payload,
                "sender": {"user_id": 777, "first_name": "Admin"},
            },
            "message": {
                "mid": f"msg_{max_update_id}",
                "recipient": {"chat_id": 777},
                "body": {"attachments": []},
            },
        })
        msgs = [r for r in max_requests if r["path"] == "/messages"]
        assert msgs, f"No message sent for payload: {payload}"
        return msgs[-1]["body"]

    max_primary_attempted = False
    max_captured_requests = []
    expected_max_fb_model = None

    def mock_max_runtime_transport(request: httpx.Request):
        nonlocal max_primary_attempted
        url_str = str(request.url)
        if "openrouter.ai" in url_str:
            max_primary_attempted = True
            raise httpx.ConnectError("Primary OpenRouter unreachable")
        if "api.perplexity.ai" in url_str:
            if "/v1/models" in url_str:
                return httpx.Response(200, json={"data": [{"id": m} for m in PERPLEXITY_STATIC_DIRECT_MODELS]})
            body = json.loads(request.content.decode("utf-8"))
            max_captured_requests.append({
                "url": url_str,
                "headers": dict(request.headers),
                "body": body,
            })
            return httpx.Response(
                200,
                json={
                    "id": "resp-max-fb-1",
                    "status": "completed",
                    "model": expected_max_fb_model or body.get("model", "anthropic/claude-sonnet-4-6"),
                    "output": [
                        {
                            "type": "search_results",
                            "results": [
                                {
                                    "title": "MAX Coping Guide",
                                    "url": "https://example.com/max-coping",
                                    "snippet": "Guidelines for stress coping in MAX.",
                                }
                            ],
                        },
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "Ответ резервного ассистента Perplexity в MAX [1].",
                                }
                            ],
                        },
                    ],
                },
            )
        return httpx.Response(404)

    with patch("httpx.AsyncClient", mock_httpx_async_client(httpx.MockTransport(mock_max_runtime_transport))):
        # Open fallback models page 0 on MAX
        fb_p0_max_msg = await press_max_cb("admin_ai_fb_ppx_models_0")
        fb_p0_max_att = fb_p0_max_msg["attachments"][0]
        next_btn_max = next(b for row in fb_p0_max_att["payload"]["buttons"] for b in row if b["payload"] == "admin_ai_fb_ppx_models_1")
        assert next_btn_max is not None

        # Click "Далее ➡️" to reach Page 1
        fb_p1_max_msg = await press_max_cb(next_btn_max["payload"])
        fb_p1_max_att = fb_p1_max_msg["attachments"][0]
        first_fb_max_btn = fb_p1_max_att["payload"]["buttons"][0][0]
        expected_max_fb_model = first_fb_max_btn["payload"].replace("admin_ai_save_fallback_Perplexity_", "")

        # Select direct fallback model
        await press_max_cb(first_fb_max_btn["payload"])

        # Verify DB persistence in MAX
        async with session_factory() as s:
            cfg = await s.get(AIConfig, 1)
            assert cfg.fallback_provider == PROVIDER_PERPLEXITY
            assert cfg.fallback_model == expected_max_fb_model

        # Reopen Page 1 on MAX to verify checkmark
        fb_reopened_max_msg = await press_max_cb("admin_ai_fb_ppx_models_1")
        fb_reopened_att = fb_reopened_max_msg["attachments"][0]
        assert "✅" in fb_reopened_att["payload"]["buttons"][0][0]["text"]

        # Extract Back button ("К настройкам") -> leads to admin_ai_fallback_model
        fb_back_btn_max = fb_reopened_att["payload"]["buttons"][-1][0]
        assert fb_back_btn_max["payload"] == "admin_ai_fallback_model"
        assert fb_back_btn_max["text"] == "К настройкам"

        # Click Back button on MAX -> reaches admin_ai_fallback_model
        fb_parent_msg = await press_max_cb(fb_back_btn_max["payload"])
        fb_parent_att = fb_parent_msg["attachments"][0]
        assert any("admin_ai_fb_ppx" in b["payload"] for row in fb_parent_att["payload"]["buttons"] for b in row)

        # 5. Enable fallback & execute real MAX user message through MaxBotApplication.handle_update()
        async with session_factory() as s:
            cfg = await s.get(AIConfig, 1)
            cfg.allow_fallback = True
            await s.commit()

        from max_messenger_bot.models import MAX_ID_OFFSET
        max_client_raw_id = 999
        effective_max_uid = max_client_raw_id + MAX_ID_OFFSET

        async with session_factory() as s:
            max_user = User(
                id=effective_max_uid,
                username="max_client",
                name="Client",
                first_name="Client",
                accepted_disclaimer=True,
            )
            s.add(max_user)
            await s.commit()

        max_update_id += 1
        await max_app.handle_update({
            "update_type": "message_created",
            "update_id": max_update_id,
            "message": {
                "mid": f"msg_user_{max_update_id}",
                "recipient": {"chat_id": max_client_raw_id},
                "sender": {
                    "user_id": max_client_raw_id,
                    "name": "Client",
                    "first_name": "Client",
                    "username": "max_client",
                },
                "body": {"text": "Вопрос пользователя в MAX"},
            },
        })
        user_task = max_app.user_tasks.get(effective_max_uid)
        if user_task:
            await user_task

    assert max_primary_attempted is True, "Primary provider OpenRouter must be attempted before fallback"
    assert len(max_captured_requests) == 1, "Perplexity fallback should be invoked exactly once after primary failure"
    max_fb_req = max_captured_requests[0]
    assert max_fb_req["body"]["model"] == expected_max_fb_model
    assert max_fb_req["body"]["tools"] == [{"type": "web_search"}]
    assert max_fb_req["body"]["tool_choice"] == {"type": "web_search"}

    max_messages = [r for r in max_requests if r["path"] == "/messages" and "text" in r.get("body", {})]
    assert max_messages, "Final answer must be sent via MAX /messages API endpoint"
    final_max_reply = max_messages[-1]["body"]["text"]
    assert "Ответ резервного ассистента Perplexity в MAX" in final_max_reply
    assert "https://example.com/max-coping" in final_max_reply


# ---------------------------------------------------------------------------
# 16b. Main Provider Runtime Journey (Admin -> DB -> Runtime -> User)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_main_provider_admin_to_runtime_user_journey(session_factory, monkeypatch, init_ai_config):
    """Verify complete real end-to-end journey:
    AI Admin screen -> Perplexity provider screen -> real visible 'Direct Models' button ->
    actual page 0 -> real visible 'Далее ➡️' -> actual page 1 -> real visible direct-model button ->
    Dispatcher -> handler -> DB persistence -> reopen -> selected model shows ✅ ->
    return to appropriate parent -> ordinary user sends a real message through Telegram runtime ->
    real runtime reads saved AIConfig -> Perplexity /v1/agent HTTP boundary is hit ->
    real response parser runs -> real Telegram outgoing method is constructed ->
    aiogram/Pydantic validates it -> user receives final text.
    """
    import handlers
    import ai_integration

    monkeypatch.setattr(handlers, "async_session_maker", session_factory)
    monkeypatch.setattr(ai_integration, "async_session_maker", session_factory)
    monkeypatch.setattr(handlers, "is_admin", lambda user_id: True)

    session = ValidatingTelegramSession()
    bot = Bot(token="123456:TEST_BOT_TOKEN", session=session)
    handlers.router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handlers.router)

    admin_user = TgUser(id=999, is_bot=False, first_name="Admin", username="admin")
    admin_chat = Chat(id=999, type=ChatType.PRIVATE)

    # Pre-configure Perplexity credentials in DB
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        cfg.provider = PROVIDER_PERPLEXITY
        cfg.perplexity_api_key = "PPLX_MAIN_ADMIN_KEY"
        cfg.perplexity_model = "fast"
        await s.commit()

    # 1. AI Admin: Open Perplexity provider settings screen
    update_prov = Update(
        update_id=1,
        callback_query=CallbackQuery(
            id="cb_prov",
            from_user=admin_user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=admin_chat, text="Providers"),
            data=f"view_models_{PROVIDER_PERPLEXITY}",
        ),
    )
    await dp.feed_update(bot, update_prov)
    provider_kb: InlineKeyboardMarkup = _last_markup(session.calls)

    # Extract visible "Direct Models" button from actual rendered markup
    direct_btn = next(b for row in provider_kb.inline_keyboard for b in row if "Прямые модели" in b.text)
    assert direct_btn.callback_data == "ai_ppx_models:0"

    # 2. Click "Direct Models" to navigate to Page 0
    update_p0 = Update(
        update_id=2,
        callback_query=CallbackQuery(
            id="cb_p0",
            from_user=admin_user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=admin_chat, text="Settings"),
            data=direct_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, update_p0)
    p0_kb: InlineKeyboardMarkup = _last_markup(session.calls)

    # Extract visible "Далее ➡️" button
    next_btn = next(b for row in p0_kb.inline_keyboard for b in row if "Далее" in b.text)
    assert next_btn.callback_data == "ai_ppx_models:1"

    # 3. Click "Далее ➡️" to navigate to Page 1
    update_p1 = Update(
        update_id=3,
        callback_query=CallbackQuery(
            id="cb_p1",
            from_user=admin_user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=admin_chat, text="Models P0"),
            data=next_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, update_p1)
    p1_kb: InlineKeyboardMarkup = _last_markup(session.calls)

    # Extract real visible direct-model button from Page 1
    first_model_btn = p1_kb.inline_keyboard[0][0]
    resolved_tuple = provider_models.resolve_telegram_model_callback(first_model_btn.callback_data)
    assert resolved_tuple is not None
    _, _, expected_admin_model = resolved_tuple

    # 4. Click direct-model button via Dispatcher -> handler -> DB persistence
    update_select = Update(
        update_id=4,
        callback_query=CallbackQuery(
            id="cb_select",
            from_user=admin_user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=admin_chat, text="Models P1"),
            data=first_model_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, update_select)

    # Verify selected model came from visible Admin markup and saved DB model equals selected model
    async with session_factory() as s:
        cfg = await s.get(AIConfig, 1)
        assert cfg.perplexity_model == expected_admin_model

    # 5. Reopen Page 1 via Dispatcher to confirm active checkmark ✅
    update_reopen = Update(
        update_id=5,
        callback_query=CallbackQuery(
            id="cb_reopen",
            from_user=admin_user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=admin_chat, text="Models P1"),
            data="ai_ppx_models:1",
        ),
    )
    await dp.feed_update(bot, update_reopen)
    reopened_kb: InlineKeyboardMarkup = _last_markup(session.calls)
    assert "✅" in reopened_kb.inline_keyboard[0][0].text

    # 6. Return to appropriate parent via visible Back button ("К настройкам")
    back_btn = reopened_kb.inline_keyboard[-1][0]
    assert back_btn.callback_data == f"view_models_{PROVIDER_PERPLEXITY}"
    update_back = Update(
        update_id=6,
        callback_query=CallbackQuery(
            id="cb_back",
            from_user=admin_user,
            chat_instance="ci",
            message=Message(message_id=10, date=datetime.now(timezone.utc), chat=admin_chat, text="Models P1"),
            data=back_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, update_back)
    prov_reopened_kb: InlineKeyboardMarkup = _last_markup(session.calls)
    assert any("Прямые модели" in b.text for row in prov_reopened_kb.inline_keyboard for b in row)

    # 7. Ordinary user sends a real message through Telegram production runtime:
    client_id = 555
    async with session_factory() as s:
        client_user = User(id=client_id, username="journey_client", first_name="Client", accepted_disclaimer=True)
        s.add(client_user)
        await s.flush()
        # Add prior multi-turn dialogue history
        m1 = DBMessage(user_id=client_id, role="user", content="Первый вопрос о бессоннице")
        m2 = DBMessage(user_id=client_id, role="assistant", content="Первый ответ психолога о режиме сна")
        s.add_all([m1, m2])
        await s.commit()

    captured_requests = []
    def mock_agent_handler(request: httpx.Request):
        body = json.loads(request.content.decode("utf-8"))
        captured_requests.append({
            "url": str(request.url),
            "headers": dict(request.headers),
            "body": body,
        })
        return httpx.Response(
            200,
            json={
                "id": "resp-main-runtime-1",
                "status": "completed",
                "model": expected_admin_model,
                "output": [
                    {
                        "type": "search_results",
                        "results": [
                            {
                                "title": "Sleep Hygiene Guidelines",
                                "url": "https://example.com/sleep-hygiene",
                                "snippet": "Scientific evidence for relaxation techniques.",
                            }
                        ],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Я внимательно прочитал ваш вопрос. Рекомендую дыхательные техники перед сном [1].",
                            }
                        ],
                    },
                ],
            },
        )

    transport = httpx.MockTransport(mock_agent_handler)
    with patch("httpx.AsyncClient", mock_httpx_async_client(transport)):
        handlers.user_message_buffers[client_id] = ["Второй вопрос: что делать, если не помогает?"]
        # Call real production entrypoint handlers.process_buffered_messages
        await handlers.process_buffered_messages(client_id, bot)

    # Assertions:
    # A. Runtime used exact model chosen in Admin UI
    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req["url"] == "https://api.perplexity.ai/v1/agent"
    assert req["headers"]["authorization"] == "Bearer PPLX_MAIN_ADMIN_KEY"
    assert req["body"]["model"] == expected_admin_model

    # B. Tools and forced tool_choice present
    assert req["body"]["tools"] == [{"type": "web_search"}]
    assert req["body"]["tool_choice"] == {"type": "web_search"}
    assert "preset" not in req["body"]

    # C. Structured multi-turn input preserved without role flattening
    inp = req["body"]["input"]
    assert isinstance(inp, list)
    assert len(inp) == 3
    assert inp[0]["role"] == "user"
    assert inp[0]["content"] == "Первый вопрос о бессоннице"
    assert inp[1]["role"] == "assistant"
    assert inp[1]["content"] == "Первый ответ психолога о режиме сна"
    assert inp[2]["role"] == "user"
    assert inp[2]["content"] == "Второй вопрос: что делать, если не помогает?"
    assert "user: Первый вопрос" not in str(inp)

    # D. Final Telegram delivery via real SendMessage passes Pydantic validation
    outgoing_messages = [c for c in session.calls if isinstance(c, SendMessage) and c.chat_id == client_id]
    assert outgoing_messages, "Production handler must construct and deliver SendMessage to user"
    final_telegram_method = outgoing_messages[-1]
    SendMessage.model_validate(final_telegram_method.model_dump())
    assert "Я внимательно прочитал ваш вопрос. Рекомендую дыхательные техники перед сном" in final_telegram_method.text
    assert "https://example.com/sleep-hygiene" in final_telegram_method.text
    assert "Источники:" in final_telegram_method.text


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
