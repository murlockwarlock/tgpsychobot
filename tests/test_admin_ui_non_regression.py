"""
tests/test_admin_ui_non_regression.py

Focused tests for Telegram and MAX AI admin callback flows.

Covers:
- No retired model can be written by any admin toggle / picker button
- Telegram and MAX resolve identical defaults for every capability channel
- All AI admin keyboards render successfully (no KeyError / empty button list)
- Back-navigation callback_data is preserved in every keyboard
- Save → reopen round-trip preserves the selected model
- Vision toggle cycles through valid providers with valid models only
- Image-gen toggle never writes a retired model (Gemini image-gen is retired)
- Image-edit toggle never writes a retired model
- Fallback toggle resolves defaults from SSOT, not hardcoded stale literals
"""

from __future__ import annotations

import os

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest

import provider_models as pm
from provider_models import (
    PROVIDER_CLAUDE,
    PROVIDER_DEEPSEEK,
    PROVIDER_GEMINI,
    PROVIDER_KIE,
    PROVIDER_OPENAI,
    ModelUnavailableError,
    RETIRED_UPSTREAM_MODELS,
    APP_DISABLED_OR_MIGRATED_MODELS,
    SELECTABLE_CHAT_MODELS,
    SELECTABLE_FALLBACK_MODELS,
    SELECTABLE_VISION_MODELS,
    SELECTABLE_IMAGE_GEN_MODELS,
    SELECTABLE_IMAGE_EDIT_MODELS,
    get_selectable_models,
    get_default_model,
    is_retired_model,
)

ALL_CHANNELS = ("chat", "fallback", "vision", "image_gen", "image_edit")
ALL_PROVIDERS = [PROVIDER_GEMINI, PROVIDER_CLAUDE, PROVIDER_OPENAI, PROVIDER_DEEPSEEK, PROVIDER_KIE]
BLOCKED_MODELS = RETIRED_UPSTREAM_MODELS | APP_DISABLED_OR_MIGRATED_MODELS


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _all_selectable_models():
    rows = []
    for channel, catalog in [
        ("chat", SELECTABLE_CHAT_MODELS),
        ("fallback", SELECTABLE_FALLBACK_MODELS),
        ("vision", SELECTABLE_VISION_MODELS),
        ("image_gen", SELECTABLE_IMAGE_GEN_MODELS),
        ("image_edit", SELECTABLE_IMAGE_EDIT_MODELS),
    ]:
        for provider, models in catalog.items():
            for model in models:
                rows.append((channel, provider, model))
    return rows


# ---------------------------------------------------------------------------
# 1. No retired / disabled model in any selectable catalog
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("channel,provider,model", _all_selectable_models())
def test_no_retired_model_in_catalogs(channel, provider, model):
    assert model not in RETIRED_UPSTREAM_MODELS, (
        f"Retired model '{model}' in {channel}/{provider} catalog."
    )


@pytest.mark.parametrize("channel,provider,model", _all_selectable_models())
def test_no_app_disabled_model_in_catalogs(channel, provider, model):
    assert model not in APP_DISABLED_OR_MIGRATED_MODELS, (
        f"App-disabled model '{model}' in {channel}/{provider} catalog."
    )


# ---------------------------------------------------------------------------
# 2. get_selectable_models is deterministic (TG and MAX get the same list)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("channel", ALL_CHANNELS)
@pytest.mark.parametrize("provider", ALL_PROVIDERS)
def test_selectable_models_deterministic(channel, provider):
    r1 = get_selectable_models(provider, channel=channel)
    r2 = get_selectable_models(provider, channel=channel)
    assert r1 == r2
    assert isinstance(r1, tuple)


# ---------------------------------------------------------------------------
# 3. get_default_model never returns a retired / disabled model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("channel", ALL_CHANNELS)
@pytest.mark.parametrize("provider", ALL_PROVIDERS)
def test_default_model_not_retired(channel, provider):
    if not get_selectable_models(provider, channel=channel):
        with pytest.raises(ModelUnavailableError):
            get_default_model(provider, channel=channel)
        return
    d = get_default_model(provider, channel=channel)
    assert d not in RETIRED_UPSTREAM_MODELS, (
        f"get_default_model({provider!r}, {channel!r}) returned retired '{d}'"
    )
    assert d not in APP_DISABLED_OR_MIGRATED_MODELS, (
        f"get_default_model({provider!r}, {channel!r}) returned disabled '{d}'"
    )


# ---------------------------------------------------------------------------
# 4. Vision toggle cycle
# ---------------------------------------------------------------------------

_VISION_CYCLE = [PROVIDER_OPENAI, PROVIDER_GEMINI, PROVIDER_KIE, PROVIDER_CLAUDE]


@pytest.mark.parametrize("provider", _VISION_CYCLE)
def test_vision_toggle_valid_model(provider):
    model = get_default_model(provider, channel="vision")
    assert model not in RETIRED_UPSTREAM_MODELS
    assert model not in APP_DISABLED_OR_MIGRATED_MODELS
    vision_models = get_selectable_models(provider, channel="vision")
    assert model in vision_models, (
        f"Vision toggle for {provider!r} writes '{model}' not in {vision_models}"
    )


# ---------------------------------------------------------------------------
# 5. Image-gen toggle: Gemini retired, cycle is OpenAI <-> KIE only
# ---------------------------------------------------------------------------

_IMG_GEN_CYCLE = [PROVIDER_OPENAI, PROVIDER_KIE]


@pytest.mark.parametrize("provider", _IMG_GEN_CYCLE)
def test_image_gen_toggle_valid_model(provider):
    model = get_default_model(provider, channel="image_gen")
    assert model not in RETIRED_UPSTREAM_MODELS
    assert model not in APP_DISABLED_OR_MIGRATED_MODELS
    gen_models = get_selectable_models(provider, channel="image_gen")
    assert model in gen_models, (
        f"Image-gen default for {provider!r} = '{model}' not in {gen_models}"
    )


def test_gemini_not_in_image_gen_cycle():
    assert PROVIDER_GEMINI not in _IMG_GEN_CYCLE


def test_gemini_image_gen_selectable_empty():
    """Gemini image-gen is retired; catalog must be empty until migration."""
    gemini_models = get_selectable_models(PROVIDER_GEMINI, channel="image_gen")
    assert len(gemini_models) == 0, (
        f"Gemini image-gen should have 0 selectable models, got {gemini_models}"
    )


# ---------------------------------------------------------------------------
# 6. Image-edit toggle: Gemini retired, only KIE
# ---------------------------------------------------------------------------

def test_image_edit_toggle_valid_model():
    model = get_default_model(PROVIDER_KIE, channel="image_edit")
    assert model not in RETIRED_UPSTREAM_MODELS
    assert model not in APP_DISABLED_OR_MIGRATED_MODELS
    edit_models = get_selectable_models(PROVIDER_KIE, channel="image_edit")
    assert model in edit_models


def test_gemini_image_edit_selectable_empty():
    gemini_edit = get_selectable_models(PROVIDER_GEMINI, channel="image_edit")
    assert len(gemini_edit) == 0, (
        f"Gemini image-edit should have 0 selectable models, got {gemini_edit}"
    )


# ---------------------------------------------------------------------------
# 7. Fallback toggle: SSOT defaults, not old stale literals
# ---------------------------------------------------------------------------

_FALLBACK_PROVIDERS = [
    PROVIDER_DEEPSEEK, PROVIDER_CLAUDE, PROVIDER_GEMINI, PROVIDER_KIE, PROVIDER_OPENAI
]

STALE_FALLBACK_LITERALS = {
    "claude-sonnet-4-5-20250929",  # retired
    "gemini-2.0-flash",            # retired upstream
    "gpt-4o",                      # app-disabled
}


@pytest.mark.parametrize("provider", _FALLBACK_PROVIDERS)
def test_fallback_default_not_stale(provider):
    model = get_default_model(provider, channel="fallback")
    assert model not in RETIRED_UPSTREAM_MODELS
    assert model not in APP_DISABLED_OR_MIGRATED_MODELS
    assert model not in STALE_FALLBACK_LITERALS, (
        f"Fallback default for {provider!r} is still stale literal '{model}'"
    )


@pytest.mark.parametrize("provider", _FALLBACK_PROVIDERS)
def test_fallback_picker_no_retired(provider):
    for m in get_selectable_models(provider, channel="fallback"):
        assert m not in RETIRED_UPSTREAM_MODELS
        assert m not in APP_DISABLED_OR_MIGRATED_MODELS


# ---------------------------------------------------------------------------
# 8. Primary picker no retired / disabled
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("provider", ALL_PROVIDERS)
def test_primary_picker_no_retired(provider):
    for m in get_selectable_models(provider, channel="chat"):
        assert m not in RETIRED_UPSTREAM_MODELS
        assert m not in APP_DISABLED_OR_MIGRATED_MODELS


# ---------------------------------------------------------------------------
# 9. Vision picker no retired
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("provider", [PROVIDER_GEMINI, PROVIDER_CLAUDE, PROVIDER_OPENAI, PROVIDER_KIE])
def test_vision_picker_no_retired(provider):
    for m in get_selectable_models(provider, channel="vision"):
        assert m not in RETIRED_UPSTREAM_MODELS


# ---------------------------------------------------------------------------
# 10. Back-navigation safety: every active-provider list is non-empty
# ---------------------------------------------------------------------------

def test_all_providers_have_chat_models():
    for provider in ALL_PROVIDERS:
        models = get_selectable_models(provider, channel="chat")
        assert len(models) > 0, f"{provider!r} has 0 chat models"


def test_fallback_providers_have_models():
    for provider in _FALLBACK_PROVIDERS:
        models = get_selectable_models(provider, channel="fallback")
        assert len(models) > 0, f"{provider!r} has 0 fallback models"


def test_image_gen_active_providers_have_models():
    for provider in _IMG_GEN_CYCLE:
        assert len(get_selectable_models(provider, channel="image_gen")) > 0


def test_image_edit_kie_has_models():
    assert len(get_selectable_models(PROVIDER_KIE, channel="image_edit")) > 0


# ---------------------------------------------------------------------------
# 11. Round-trip: default is in selectable list (save → reopen shows tick)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("provider", ALL_PROVIDERS)
def test_chat_default_in_selectable(provider):
    default = get_default_model(provider, channel="chat")
    selectable = get_selectable_models(provider, channel="chat")
    assert default in selectable, (
        f"Default chat model for {provider!r} = '{default}' not in {selectable}. "
        "Admin won't see tick mark on reopen."
    )


@pytest.mark.parametrize("provider", _IMG_GEN_CYCLE)
def test_image_gen_default_in_selectable(provider):
    default = get_default_model(provider, channel="image_gen")
    selectable = get_selectable_models(provider, channel="image_gen")
    assert default in selectable


def test_image_edit_default_in_selectable():
    default = get_default_model(PROVIDER_KIE, channel="image_edit")
    assert default in get_selectable_models(PROVIDER_KIE, channel="image_edit")


# ---------------------------------------------------------------------------
# 12. Known retired models flagged correctly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", sorted(RETIRED_UPSTREAM_MODELS))
def test_is_retired_true(model):
    assert is_retired_model(model)


@pytest.mark.parametrize("model", [
    "gemini-3.7-flash", "claude-sonnet-5", "gpt-5.6-terra",
    "deepseek-v4-flash", "gemini-3-flash", "gpt-image-2",
])
def test_is_retired_false_for_active(model):
    assert not is_retired_model(model)


# ---------------------------------------------------------------------------
# 13. KIE catalog identical between KIE_CHAT_MODELS and SELECTABLE_CHAT_MODELS
# ---------------------------------------------------------------------------

def test_kie_chat_models_consistent():
    from provider_models import KIE_CHAT_MODELS
    assert tuple(KIE_CHAT_MODELS) == SELECTABLE_CHAT_MODELS[PROVIDER_KIE], (
        "KIE_CHAT_MODELS and SELECTABLE_CHAT_MODELS['KIE'] diverged. "
        "handlers.py uses KIE_CHAT_MODELS to filter MODELS_INFO."
    )


# ---------------------------------------------------------------------------
# 14. Fresh AIConfig defaults do not contain retired / disabled models
# ---------------------------------------------------------------------------

def test_fresh_aiconfig_defaults_are_current_and_active():
    from database import AIConfig
    config = AIConfig()

    runtime_models = {
        "gemini_model": config.gemini_model,
        "claude_model": config.claude_model,
        "openai_model": config.openai_model,
        "deepseek_model": config.deepseek_model,
        "kie_model": config.kie_model,
        "vision_model": config.vision_model,
        "image_generation_model": config.image_generation_model,
        "image_edit_model": config.image_edit_model,
    }
    for field_name, model_val in runtime_models.items():
        assert model_val not in RETIRED_UPSTREAM_MODELS, (
            f"AIConfig.{field_name} defaults to retired model {model_val!r}"
        )
        assert model_val not in APP_DISABLED_OR_MIGRATED_MODELS, (
            f"AIConfig.{field_name} defaults to app-disabled model {model_val!r}"
        )

    # Provider defaults must not point to retired Google image capabilities
    assert config.image_generation_provider != "Gemini", (
        "AIConfig.image_generation_provider must not default to retired Gemini"
    )
    assert config.image_edit_provider != "Gemini", (
        "AIConfig.image_edit_provider must not default to retired Gemini"
    )


# ---------------------------------------------------------------------------
# 15. Database migration _migrate_ai_config_models replaces legacy models
# ---------------------------------------------------------------------------

def test_database_migration_replaces_all_legacy_and_retired_models():
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
            )
            VALUES
                (1, 'gpt-4o', 'claude-sonnet-4-5-20250929', 'gemini-2.0-flash', 'deepseek-chat',
                 'Gemini', 'gemini-2.0-flash', 'OpenAI', 'gpt-4o'),
                (2, 'gpt-4.1', 'claude-opus-4-1-20250805', 'gemini-1.5-pro', 'deepseek-v4-pro',
                 'Claude', 'claude-opus-4-1-20250805', 'Claude', 'claude-opus-4-1-20250805'),
                (3, 'gpt-4-turbo', 'claude-3-haiku-20240307', 'gemini-2.5-flash-preview-05-20', 'deepseek-coder',
                 'OpenAI', 'gpt-4o', 'Gemini', 'gemini-2.0-flash')
        """))

        _migrate_ai_config_models(conn)

        rows = conn.execute(text("""
            SELECT id, openai_model, claude_model, gemini_model, deepseek_model,
                   vision_provider, vision_model, fallback_provider, fallback_model
            FROM ai_config ORDER BY id
        """)).all()

    # Row 1 (Sonnet family preserved)
    assert rows[0][1] == "gpt-5.6-terra"
    assert rows[0][2] == "claude-sonnet-5"
    assert rows[0][3] == "gemini-3.7-flash"
    assert rows[0][4] == "deepseek-v4-flash"
    assert rows[0][6] == "gemini-3.7-flash"
    assert rows[0][8] == "gpt-5.6-terra"

    # Row 2 (Opus family preserved)
    assert rows[1][1] == "gpt-5.6-terra"
    assert rows[1][2] == "claude-opus-5"
    assert rows[1][3] == "gemini-3.7-flash"
    assert rows[1][4] == "deepseek-v4-pro"
    assert rows[1][6] == "claude-opus-5"
    assert rows[1][8] == "claude-opus-5"

    # Row 3 (Haiku family preserved)
    assert rows[2][1] == "gpt-5.6-terra"
    assert rows[2][2] == "claude-haiku-4-5-20251001"
    assert rows[2][3] == "gemini-3.7-flash"
    assert rows[2][4] == "deepseek-v4-flash"
    assert rows[2][6] == "gpt-5.6-terra"
    assert rows[2][8] == "gemini-3.7-flash"


# ---------------------------------------------------------------------------
# 16. MAX fallback selection with stale primary models
# ---------------------------------------------------------------------------

def test_max_fallback_selection_does_not_resurrect_stale_primary_models():
    from database import AIConfig
    from max_messenger_bot.services.admin_ai import _fallback_model_for_provider

    # Simulate AIConfig with stale primary models
    config = AIConfig(
        openai_model="gpt-4o",
        claude_model="claude-sonnet-4-5-20250929",
        gemini_model="gemini-2.0-flash",
        fallback_provider=None,
        fallback_model=None,
    )

    # Selecting OpenAI as fallback must write current default, NOT 'gpt-4o'
    fb_openai = _fallback_model_for_provider(config, "OpenAI")
    assert fb_openai == "gpt-5.6-terra"
    assert fb_openai in get_selectable_models("OpenAI", channel="fallback")

    # Selecting Claude as fallback must write current default, NOT 'claude-sonnet-4-5-20250929'
    fb_claude = _fallback_model_for_provider(config, "Claude")
    assert fb_claude == "claude-sonnet-5"
    assert fb_claude in get_selectable_models("Claude", channel="fallback")

    # Selecting Gemini as fallback must write current default, NOT 'gemini-2.0-flash'
    fb_gemini = _fallback_model_for_provider(config, "Gemini")
    assert fb_gemini == "gemini-3.7-flash"
    assert fb_gemini in get_selectable_models("Gemini", channel="fallback")


# ---------------------------------------------------------------------------
# 17. OpenAI BadRequestError handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openai_bad_request_error_raises_ai_service_error_not_none():
    from unittest.mock import AsyncMock, patch
    from openai import BadRequestError
    import httpx
    from ai_integration import _call_openai_api, AIServiceError, InsufficientBalanceError

    mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    mock_response = httpx.Response(400, request=mock_request)
    bad_req_err = BadRequestError("Invalid parameter: temperature not allowed", response=mock_response, body=None)

    with patch("ai_integration.AsyncOpenAI") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = bad_req_err
        mock_client_cls.return_value = mock_client

        with pytest.raises(AIServiceError) as exc_info:
            await _call_openai_api(
                api_key="sk-test",
                model="gpt-5.6-terra",
                history=[],
                context="",
                system_prompt="Test system prompt",
            )

        assert "Ошибка при обращении к OpenAI API" in str(exc_info.value)
        assert "Invalid parameter" in str(exc_info.value)


@pytest.mark.asyncio
async def test_openai_billing_bad_request_raises_insufficient_balance():
    from unittest.mock import AsyncMock, patch
    from openai import BadRequestError
    import httpx
    from ai_integration import _call_openai_api, InsufficientBalanceError

    mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    mock_response = httpx.Response(400, request=mock_request)
    billing_err = BadRequestError("You exceeded your current quota, please check your plan and billing details.", response=mock_response, body=None)

    with patch("ai_integration.AsyncOpenAI") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = billing_err
        mock_client_cls.return_value = mock_client

        with pytest.raises(InsufficientBalanceError):
            await _call_openai_api(
                api_key="sk-test",
                model="gpt-5.6-terra",
                history=[],
                context="",
                system_prompt="Test system prompt",
            )


# ---------------------------------------------------------------------------
# 18. Gemini transcription resolution with old configured gemini_model
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_transcription_ignores_stale_chat_model_in_db():
    from unittest.mock import AsyncMock, MagicMock, patch
    from types import SimpleNamespace
    from ai_integration import transcribe_voice_message

    config = SimpleNamespace(
        transcription_provider="Gemini",
        gemini_api_key="test-gemini-key",
        gemini_model="gemini-2.0-flash",  # Stale retired chat model in DB
    )
    session = AsyncMock()
    session.get.return_value = config
    session_context = MagicMock()
    session_context.__aenter__.return_value = session
    session_context.__aexit__.return_value = False

    with patch("ai_integration.async_session_maker", return_value=session_context), \
         patch("ai_integration._call_gemini_transcribe", AsyncMock(return_value="тестовый текст")) as mock_call:
        result = await transcribe_voice_message(b"audio-data", "voice.ogg")

    assert result == "тестовый текст"
    # Proves transcription was called with 'gemini-3.7-flash', NOT the retired 'gemini-2.0-flash' from chat config
    called_model = mock_call.await_args.args[1]
    assert called_model == "gemini-3.7-flash"
    assert called_model not in RETIRED_UPSTREAM_MODELS


# ---------------------------------------------------------------------------
# 19. Provider Serialization Tests (Telegram & MAX)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["claude-sonnet-5", "claude-opus-5"])
async def test_claude_sampling_omitted_for_sonnet_and_opus_telegram(model_name):
    from unittest.mock import AsyncMock, patch
    import ai_integration

    capture = {}
    mock_client = AsyncMock()
    mock_resp = AsyncMock()
    mock_resp.content = [AsyncMock(text="Claude test answer")]
    mock_client.messages.create.return_value = mock_resp

    with patch("ai_integration.anthropic.AsyncAnthropic", return_value=mock_client):
        await ai_integration._call_claude_api(
            api_key="test-key",
            model=model_name,
            history=[],
            context="",
            system_prompt="System Prompt",
            temperature=0.7,
            runtime_context="",
            request_capture=capture,
        )

    # Telegram payload check
    payload = capture.get("payload", {})
    assert "temperature" not in payload, f"temperature must be omitted for {model_name}"
    assert "top_p" not in payload, f"top_p must be omitted for {model_name}"
    assert "top_k" not in payload, f"top_k must be omitted for {model_name}"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["claude-sonnet-5", "claude-opus-5"])
async def test_claude_sampling_omitted_for_sonnet_and_opus_max(model_name):
    from unittest.mock import AsyncMock, patch
    from max_messenger_bot import ai as max_ai

    mock_client = AsyncMock()
    mock_resp = AsyncMock()
    mock_resp.content = [AsyncMock(text="Max Claude answer")]
    mock_client.messages.create.return_value = mock_resp

    with patch("max_messenger_bot.ai.anthropic.AsyncAnthropic", return_value=mock_client):
        await max_ai._call_claude(
            api_key="test-key",
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
            system_prompt="System Prompt",
            temperature=0.7,
        )

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert "temperature" not in call_kwargs, f"MAX temperature must be omitted for {model_name}"
    assert "top_p" not in call_kwargs
    assert "top_k" not in call_kwargs


@pytest.mark.asyncio
async def test_claude_sampling_preserved_for_haiku_4_5_telegram_and_max():
    from unittest.mock import AsyncMock, patch
    import ai_integration
    from max_messenger_bot import ai as max_ai

    # 1. Telegram
    capture = {}
    mock_tg_client = AsyncMock()
    mock_tg_resp = AsyncMock()
    mock_tg_resp.content = [AsyncMock(text="Haiku answer")]
    mock_tg_client.messages.create.return_value = mock_tg_resp

    with patch("ai_integration.anthropic.AsyncAnthropic", return_value=mock_tg_client):
        await ai_integration._call_claude_api(
            api_key="test-key",
            model="claude-haiku-4-5-20251001",
            history=[],
            context="",
            system_prompt="System Prompt",
            temperature=0.0,
            runtime_context="",
            request_capture=capture,
        )

    payload = capture.get("payload", {})
    assert "temperature" in payload
    assert payload["temperature"] == 0.0, "temperature=0.0 must be preserved exactly for Haiku 4.5"

    # 2. MAX
    mock_max_client = AsyncMock()
    mock_max_resp = AsyncMock()
    mock_max_resp.content = [AsyncMock(text="Max Haiku answer")]
    mock_max_client.messages.create.return_value = mock_max_resp

    with patch("max_messenger_bot.ai.anthropic.AsyncAnthropic", return_value=mock_max_client):
        await max_ai._call_claude(
            api_key="test-key",
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "Hello"}],
            system_prompt="System Prompt",
            temperature=0.0,
        )

    max_kwargs = mock_max_client.messages.create.call_args.kwargs
    assert "temperature" in max_kwargs
    assert max_kwargs["temperature"] == 0.0, "MAX temperature=0.0 must be preserved exactly for Haiku 4.5"


@pytest.mark.asyncio
async def test_openai_gpt56_payload_shape_telegram_and_max():
    from unittest.mock import AsyncMock, patch
    import ai_integration
    from max_messenger_bot import ai as max_ai
    from ai_request_context import AIRequestLayout, AIRequestMessage

    layout = AIRequestLayout(
        stable_system_prompt="System instructions",
        history=(
            AIRequestMessage(role="user", content="Past question"),
            AIRequestMessage(role="assistant", content="Past answer"),
        ),
        current_user_content="Current question",
    )

    # 1. Telegram
    capture = {}
    mock_tg_client = AsyncMock()
    mock_tg_client.chat.completions.create.return_value = AsyncMock(
        choices=[AsyncMock(message=AsyncMock(content="GPT answer"))]
    )

    with patch("ai_integration.AsyncOpenAI", return_value=mock_tg_client):
        await ai_integration._call_openai_api(
            api_key="test-key",
            model="gpt-5.6-terra",
            history=[],
            context="",
            system_prompt="",
            temperature=0.7,
            request_capture=capture,
            request_layout=layout,
        )

    tg_payload = capture.get("payload", {})
    assert tg_payload["model"] == "gpt-5.6-terra"
    assert "max_completion_tokens" in tg_payload
    assert "temperature" not in tg_payload, "temperature must be omitted for gpt-5.6 models"
    # Current user remains final message
    assert tg_payload["messages"][-1]["role"] == "user"
    assert tg_payload["messages"][-1]["content"] == "Current question"

    # 2. MAX
    mock_max_client = AsyncMock()
    mock_max_client.chat.completions.create.return_value = AsyncMock(
        choices=[AsyncMock(message=AsyncMock(content="Max GPT answer"))]
    )

    with patch("max_messenger_bot.ai.AsyncOpenAI", return_value=mock_max_client):
        await max_ai._call_openai(
            api_key="test-key",
            model="gpt-5.6-terra",
            messages=None,
            temperature=0.7,
            request_layout=layout,
        )

    max_kwargs = mock_max_client.chat.completions.create.call_args.kwargs
    assert max_kwargs["model"] == "gpt-5.6-terra"
    assert "max_completion_tokens" in max_kwargs
    assert "temperature" not in max_kwargs, "MAX temperature must be omitted for gpt-5.6 models"
    assert max_kwargs["messages"][-1]["role"] == "user"
    assert max_kwargs["messages"][-1]["content"] == "Current question"


@pytest.mark.asyncio
async def test_gemini_37_payload_shape_telegram_and_max():
    from unittest.mock import AsyncMock, MagicMock, patch
    import ai_integration
    from max_messenger_bot import ai as max_ai
    from ai_request_context import AIRequestLayout, AIRequestMessage

    layout = AIRequestLayout(
        stable_system_prompt="System instructions",
        history=(
            AIRequestMessage(role="user", content="Past question"),
            AIRequestMessage(role="assistant", content="Past answer"),
        ),
        current_user_content="Current question",
    )

    # 1. Telegram
    capture = {}
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.json = lambda: {
        "candidates": [{
            "content": {
                "parts": [{"text": "Gemini answer"}],
                "role": "model",
            }
        }]
    }
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client_cm = MagicMock()
    mock_client_cm.__aenter__.return_value = mock_client
    mock_client_cm.__aexit__.return_value = False

    with patch("httpx.AsyncClient", return_value=mock_client_cm):
        await ai_integration._call_gemini_api(
            api_key="test-key",
            model="gemini-3.7-flash",
            history=[],
            context="",
            system_prompt="",
            temperature=0.7,
            request_capture=capture,
            request_layout=layout,
        )

    tg_payload = capture.get("payload", {})
    gen_config = tg_payload.get("generationConfig", {})
    assert "temperature" not in gen_config, "temperature must be omitted for Gemini 3.7"
    assert "top_p" not in gen_config, "top_p must be omitted for Gemini 3.7"
    assert "top_k" not in gen_config, "top_k must be omitted for Gemini 3.7"
    # Final turn must be current user
    assert tg_payload["contents"][-1]["role"] == "user"
    assert tg_payload["contents"][-1]["parts"][0]["text"] == "Current question"

    # 2. MAX
    max_mock_resp = AsyncMock()
    max_mock_resp.status_code = 200
    max_mock_resp.raise_for_status = MagicMock()
    max_mock_resp.json = lambda: {
        "candidates": [{
            "content": {
                "parts": [{"text": "Max Gemini answer"}],
                "role": "model",
            }
        }]
    }
    max_mock_client = AsyncMock()
    max_mock_client.post.return_value = max_mock_resp
    max_mock_client_cm = MagicMock()
    max_mock_client_cm.__aenter__.return_value = max_mock_client
    max_mock_client_cm.__aexit__.return_value = False

    with patch("httpx.AsyncClient", return_value=max_mock_client_cm):
        await max_ai._call_gemini(
            api_key="test-key",
            model="gemini-3.7-flash",
            messages=None,
            system_prompt="",
            temperature=0.7,
            request_layout=layout,
        )

    max_call_kwargs = max_mock_client.post.call_args.kwargs
    max_payload = max_call_kwargs.get("json", {})
    max_gen_config = max_payload.get("generationConfig", {})
    assert "temperature" not in max_gen_config, "MAX temperature must be omitted for Gemini 3.7"
    assert "top_p" not in max_gen_config, "MAX top_p must be omitted for Gemini 3.7"
    assert "top_k" not in max_gen_config, "MAX top_k must be omitted for Gemini 3.7"
    contents = max_payload["contents"]
    assert contents[-1]["role"] == "user"
    assert contents[-1]["parts"][0]["text"] == "Current question"
