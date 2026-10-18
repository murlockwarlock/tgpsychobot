import asyncio
import os

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("BOT_TOKEN", "test")

from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import ai_integration
from database import AIConfig, Base, Topic, User
from max_messenger_bot import ai as max_ai
from ai_request_context import AIRequestLayout
from provider_adapters import (
    ProviderAdapterError,
    build_provider_http_client,
    call_openrouter,
    normalize_provider_error_classification,
    redact_proxy_url,
    resolve_provider_network_route,
    verify_openrouter_catalog,
)


def _make_layout(prompt: str = "Hello") -> AIRequestLayout:
    return AIRequestLayout(
        history=(),
        current_user_content=prompt,
        stable_system_prompt="You are an assistant.",
    )


# ---------------------------------------------------------------------------
# 1. Proxy Route Resolution and Precedence
# ---------------------------------------------------------------------------

def test_openrouter_proxy_precedence(monkeypatch):
    """Test OPENROUTER_PROXY > HTTPS_PROXY > HTTP_PROXY > direct."""
    # A. OPENROUTER_PROXY takes precedence
    monkeypatch.setenv("OPENROUTER_PROXY", "http://user:pass@openrouter-proxy:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://https-proxy:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://http-proxy:3128")
    route = resolve_provider_network_route("OpenRouter")
    assert route.is_proxied is True
    assert route.proxy_source == "OPENROUTER_PROXY"
    assert route.proxy_url == "http://user:pass@openrouter-proxy:3128"

    # B. Falls back to HTTPS_PROXY
    monkeypatch.delenv("OPENROUTER_PROXY", raising=False)
    route = resolve_provider_network_route("OpenRouter")
    assert route.is_proxied is True
    assert route.proxy_source == "HTTPS_PROXY"
    assert route.proxy_url == "http://https-proxy:3128"

    # C. Falls back to HTTP_PROXY
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    route = resolve_provider_network_route("OpenRouter")
    assert route.is_proxied is True
    assert route.proxy_source == "HTTP_PROXY"
    assert route.proxy_url == "http://http-proxy:3128"

    # D. Direct when no proxy env
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    route = resolve_provider_network_route("OpenRouter")
    assert route.is_proxied is False
    assert route.proxy_source == "direct"
    assert route.proxy_url is None


def test_perplexity_and_deepgram_routes(monkeypatch):
    """Perplexity and Deepgram use provider-specific proxy or explicit direct."""
    monkeypatch.delenv("PERPLEXITY_PROXY", raising=False)
    monkeypatch.delenv("DEEPGRAM_PROXY", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://general-proxy:3128")

    # Perplexity defaults to direct
    p_route = resolve_provider_network_route("Perplexity")
    assert p_route.is_proxied is False
    assert p_route.proxy_source == "direct"

    # Deepgram defaults to direct
    d_route = resolve_provider_network_route("Deepgram")
    assert d_route.is_proxied is False
    assert d_route.proxy_source == "direct"

    # Perplexity with explicit PERPLEXITY_PROXY
    monkeypatch.setenv("PERPLEXITY_PROXY", "http://pplx-proxy:8080")
    p_route = resolve_provider_network_route("Perplexity")
    assert p_route.is_proxied is True
    assert p_route.proxy_source == "PERPLEXITY_PROXY"
    assert p_route.proxy_url == "http://pplx-proxy:8080"

    # Deepgram with explicit DEEPGRAM_PROXY
    monkeypatch.setenv("DEEPGRAM_PROXY", "http://dg-proxy:8080")
    d_route = resolve_provider_network_route("Deepgram")
    assert d_route.is_proxied is True
    assert d_route.proxy_source == "DEEPGRAM_PROXY"
    assert d_route.proxy_url == "http://dg-proxy:8080"


# ---------------------------------------------------------------------------
# 2. Credential Redaction
# ---------------------------------------------------------------------------

def test_redact_proxy_url():
    assert redact_proxy_url("http://user:secret123@proxy.host:3128") == "http://***:***@proxy.host:3128"
    assert redact_proxy_url("http://proxy.host:3128") == "http://proxy.host:3128"
    assert redact_proxy_url(None) is None
    assert redact_proxy_url("") is None


# ---------------------------------------------------------------------------
# 3. Client and Transport Construction
# ---------------------------------------------------------------------------

def test_build_provider_http_client_proxied(monkeypatch):
    monkeypatch.setenv("OPENROUTER_PROXY", "http://proxy.example.com:3128")
    client, route = build_provider_http_client("OpenRouter", timeout=30.0)
    assert route.is_proxied is True
    assert client.trust_env is False
    # Check that transport is an AsyncHTTPTransport configured with proxy
    transport = getattr(client, "_transport", None)
    assert isinstance(transport, httpx.AsyncHTTPTransport)
    # httpx._transport._pool._proxy
    pool = getattr(transport, "_pool", None)
    proxy_url = getattr(pool, "_proxy_url", None)
    if proxy_url:
        assert "proxy.example.com" in str(proxy_url)


def test_build_provider_http_client_direct(monkeypatch):
    monkeypatch.delenv("OPENROUTER_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    client, route = build_provider_http_client("OpenRouter", timeout=30.0)
    assert route.is_proxied is False
    assert client.trust_env is False


# ---------------------------------------------------------------------------
# 4. OpenRouter Catalog Uses Same Egress Policy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openrouter_catalog_uses_configured_proxy(monkeypatch):
    monkeypatch.setenv("OPENROUTER_PROXY", "http://test-proxy:3128")
    captured_transport = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "deepseek/deepseek-v4.1-flash"}]})

    original_build = build_provider_http_client

    def spying_build(provider, timeout=20.0, route=None, transport=None):
        c, r = original_build(provider, timeout=timeout, route=route, transport=httpx.MockTransport(mock_handler))
        captured_transport.append((provider, r))
        return c, r

    with patch("provider_adapters.build_provider_http_client", side_effect=spying_build):
        result = await verify_openrouter_catalog(timeout=5.0)

    assert len(captured_transport) == 1
    provider_name, route = captured_transport[0]
    assert provider_name == "OpenRouter"
    assert route.is_proxied is True
    assert route.proxy_source == "OPENROUTER_PROXY"


# ---------------------------------------------------------------------------
# 5. Fail Closed on Proxy Connection Failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openrouter_proxy_failure_fails_closed_no_silent_direct(monkeypatch):
    """When proxy is configured and fails, it must fail closed without direct retry."""
    monkeypatch.setenv("OPENROUTER_PROXY", "http://unreachable-proxy.local:3128")
    attempts = 0

    def mock_proxy_failure(request: httpx.Request):
        nonlocal attempts
        attempts += 1
        raise httpx.ProxyError("Unable to connect to proxy", request=request)

    capture = {}
    mock_transport = httpx.MockTransport(mock_proxy_failure)
    with pytest.raises(ProviderAdapterError) as exc_info:
        await call_openrouter(
            "test-key",
            _make_layout(),
            "deepseek/deepseek-v4.1-flash",
            request_capture=capture,
            transport=mock_transport,
        )

    err = exc_info.value
    assert "Ошибка прокси" in str(err)
    assert normalize_provider_error_classification(err.classification) == "network_connection"
    assert err.proxy_used is True
    assert err.proxy_source == "OPENROUTER_PROXY"
    assert attempts == 1  # ZERO silent direct retries!


# ---------------------------------------------------------------------------
# 6. Error Body Preservation and 403 Security Policy Classification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_403_security_policy_classification_and_body_preservation(monkeypatch):
    """HTTP 403 with 'Access denied by security policy' must be classified as provider_rejection, NOT auth."""
    monkeypatch.delenv("OPENROUTER_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)

    def mock_403(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            headers={"cf-ray": "cf-ray-12345"},
            json={"success": False, "error": "Access denied by security policy."},
        )

    capture = {}
    mock_transport = httpx.MockTransport(mock_403)
    with pytest.raises(ProviderAdapterError) as exc_info:
        await call_openrouter(
            "test-key",
            _make_layout(),
            "deepseek/deepseek-v4.1-flash",
            request_capture=capture,
            transport=mock_transport,
        )

    err = exc_info.value
    # 1. Error message preserves the provider's exact message
    assert "Access denied by security policy." in str(err)
    assert err.provider_error_message == "Access denied by security policy."
    assert err.request_id == "cf-ray-12345"

    # 2. Classification must NOT be auth! It must be provider_rejection
    assert err.classification != "auth"
    assert err.classification == "provider_rejection"
    assert normalize_provider_error_classification(err.classification) == "provider_rejection"

    # 3. request_capture contains structured diagnostics
    assert capture["http_status"] == 403
    assert capture["classification"] == "provider_rejection"
    assert capture["provider_error_message"] == "Access denied by security policy."
    assert capture["request_id"] == "cf-ray-12345"
    assert capture["proxy_used"] is False


@pytest.mark.asyncio
async def test_401_invalid_key_classification(monkeypatch):
    """HTTP 401 with invalid API key must be classified as auth."""
    def mock_401(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            headers={"x-request-id": "req-auth-fail"},
            json={"error": {"message": "Invalid API key provided", "code": 401}},
        )

    capture = {}
    mock_transport = httpx.MockTransport(mock_401)
    with pytest.raises(ProviderAdapterError) as exc_info:
        await call_openrouter(
            "test-key",
            _make_layout(),
            "deepseek/deepseek-v4.1-flash",
            request_capture=capture,
            transport=mock_transport,
        )

    err = exc_info.value
    assert "Invalid API key provided" in str(err)
    assert err.classification == "auth"
    assert normalize_provider_error_classification(err.classification) == "auth"
    assert capture["http_status"] == 401
    assert capture["classification"] == "auth"
    assert capture["request_id"] == "req-auth-fail"


# ---------------------------------------------------------------------------
# 7. Diagnostics Masking in request_capture
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_diagnostics_redacts_proxy_credentials(monkeypatch):
    """Verify that credentials in proxy URLs are redacted in diagnostics."""
    monkeypatch.setenv("OPENROUTER_PROXY", "http://secretuser:supersecretpass@proxy.example.com:3128")

    def mock_200(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Hello from OpenRouter"}}], "usage": {"total_tokens": 10}},
        )

    capture = {}
    mock_transport = httpx.MockTransport(mock_200)
    res = await call_openrouter(
        "test-key",
        _make_layout(),
        "deepseek/deepseek-v4.1-flash",
        request_capture=capture,
        transport=mock_transport,
    )

    assert res == "Hello from OpenRouter"
    assert capture["proxy_used"] is True
    assert capture["proxy_source"] == "OPENROUTER_PROXY"
    assert "supersecretpass" not in str(capture)
    assert "secretuser" not in str(capture)
    assert capture["proxy_url"] == "http://***:***@proxy.example.com:3128"


# ---------------------------------------------------------------------------
# 8. Real Application Call Paths: Telegram and MAX Parity
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def egress_db():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with sessions() as session:
        session.add(Topic(id=1, name="Тема", is_active=True, system_prompt="Система"))
        session.add(User(id=101, first_name="Тест", current_topic_id=1, current_dialogue_id=1, metadata_json="{}"))
        session.add(
            AIConfig(
                id=1,
                provider="OpenRouter",
                openrouter_api_key="sk-or-test-key",
                openrouter_model="deepseek/deepseek-v4.1-flash",
                system_prompt="Ты ассистент.",
                shared_prompt_block="",
                memory_mode="global",
            )
        )
        await session.commit()
    old_tg = ai_integration.async_session_maker
    old_max = max_ai.async_session_maker
    ai_integration.async_session_maker = sessions
    max_ai.async_session_maker = sessions
    try:
        yield sessions
    finally:
        ai_integration.async_session_maker = old_tg
        max_ai.async_session_maker = old_max
        await engine.dispose()


@pytest.mark.asyncio
async def test_telegram_openrouter_call_uses_proxy(egress_db, monkeypatch):
    """Telegram real call path -> OpenRouter -> verifies proxy transport is used."""
    monkeypatch.setenv("OPENROUTER_PROXY", "http://tg-proxy.example.com:3128")
    intercepted = []

    orig_post = httpx.AsyncClient.post

    async def spy_post(self, url, *args, **kwargs):
        # Inspect the AsyncClient instance created by the provider adapter
        transport = getattr(self, "_transport", None)
        pool = getattr(transport, "_pool", None) if transport else None
        proxy_url = getattr(pool, "_proxy_url", None) if pool else None
        intercepted.append({
            "client_trust_env": self.trust_env,
            "transport_type": type(transport).__name__,
            "proxy_url": str(proxy_url) if proxy_url else None,
            "url": str(url),
            "auth": kwargs.get("headers", {}).get("Authorization"),
        })
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Ответ из Telegram OpenRouter"}}], "usage": {"total_tokens": 12}},
        )

    with patch.object(httpx.AsyncClient, "post", spy_post):
        response = await ai_integration.generate_response(101, "Привет из Telegram!")

    assert response == "Ответ из Telegram OpenRouter"
    assert len(intercepted) == 1
    call_info = intercepted[0]
    assert call_info["client_trust_env"] is False
    assert call_info["transport_type"] == "AsyncHTTPTransport"
    assert "tg-proxy.example.com" in (call_info["proxy_url"] or "")
    assert "3128" in (call_info["proxy_url"] or "")
    assert "https://openrouter.ai/api/v1/chat/completions" in call_info["url"]
    assert call_info["auth"] == "Bearer sk-or-test-key"


@pytest.mark.asyncio
async def test_max_openrouter_call_uses_proxy(egress_db, monkeypatch):
    """MAX real call path -> OpenRouter -> verifies identical proxy transport is used."""
    monkeypatch.setenv("OPENROUTER_PROXY", "http://max-proxy.example.com:3128")
    intercepted = []

    async def spy_post(self, url, *args, **kwargs):
        transport = getattr(self, "_transport", None)
        pool = getattr(transport, "_pool", None) if transport else None
        proxy_url = getattr(pool, "_proxy_url", None) if pool else None
        intercepted.append({
            "client_trust_env": self.trust_env,
            "transport_type": type(transport).__name__,
            "proxy_url": str(proxy_url) if proxy_url else None,
            "url": str(url),
            "auth": kwargs.get("headers", {}).get("Authorization"),
        })
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Ответ из MAX OpenRouter"}}], "usage": {"total_tokens": 15}},
        )

    with patch.object(httpx.AsyncClient, "post", spy_post):
        response = await max_ai.get_ai_response(101, "Привет из MAX!")

    assert response == "Ответ из MAX OpenRouter"
    assert len(intercepted) == 1
    call_info = intercepted[0]
    assert call_info["client_trust_env"] is False
    assert call_info["transport_type"] == "AsyncHTTPTransport"
    assert "max-proxy.example.com" in (call_info["proxy_url"] or "")
    assert "3128" in (call_info["proxy_url"] or "")
    assert "https://openrouter.ai/api/v1/chat/completions" in call_info["url"]
    assert call_info["auth"] == "Bearer sk-or-test-key"

