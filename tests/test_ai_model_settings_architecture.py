import os
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from unittest.mock import AsyncMock

import ai_integration
import handlers
import keyboards
from max_messenger_bot import ai as max_ai
from max_messenger_bot.services import admin_ai as max_admin_ai
from max_messenger_bot.services.admin_ai import _build_keys_keyboard
from ai_model_settings import (
    REASONING_AUTO,
    REASONING_HIGH,
    REASONING_MAX,
    REASONING_NONE,
    get_generation_capabilities,
    get_or_create_model_settings,
    resolve_model_settings,
    validate_model_setting,
    wire_max_output_tokens,
    wire_reasoning_effort,
    wire_temperature,
)
from ai_request_context import AIRequestLayout
from database import AIConfig, Base
from provider_models import (
    ModelUnavailableError,
    PROVIDER_DEEPGRAM,
    PROVIDER_DEEPSEEK,
    PROVIDER_OPENAI,
    PROVIDER_OPENROUTER,
    build_telegram_model_callback_data,
    get_default_model,
    get_capability_providers,
    get_selectable_models,
    resolve_telegram_model_callback,
)
from vision_reliability import sanitize_vision_request_payload


@pytest_asyncio.fixture
async def settings_db():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with sessions() as session:
        session.add(
            AIConfig(
                id=1,
                provider=PROVIDER_DEEPSEEK,
                deepseek_model="deepseek-v4-pro",
                deepseek_thinking_enabled=None,
                max_output_tokens=None,
                temperature=0.7,
            )
        )
        await session.commit()
    yield sessions
    await engine.dispose()


def test_capabilities_are_model_driven():
    deepseek = get_generation_capabilities(PROVIDER_DEEPSEEK, "deepseek-v4-pro")
    assert deepseek.reasoning_effort == (REASONING_AUTO, REASONING_NONE, "low", "high", REASONING_MAX)
    assert deepseek.temperature_ignored_for_reasoning is True
    assert get_generation_capabilities(PROVIDER_OPENAI, "gpt-5.6-terra").reasoning_effort == ()
    assert get_generation_capabilities(PROVIDER_OPENROUTER, "openai/gpt-5.6-terra").reasoning_effort == ()


@pytest.mark.asyncio
async def test_model_scoped_settings_survive_switching(settings_db):
    async with settings_db() as session:
        deepseek = await get_or_create_model_settings(session, PROVIDER_DEEPSEEK, "deepseek-v4-pro")
        deepseek.reasoning_effort = REASONING_MAX
        deepseek.max_output_tokens = None
        await session.commit()
        openai = await get_or_create_model_settings(session, PROVIDER_OPENAI, "gpt-5.6-terra")
        openai.max_output_tokens = 12000
        await session.commit()

        deepseek_resolved = await resolve_model_settings(session, PROVIDER_DEEPSEEK, "deepseek-v4-pro")
        openai_resolved = await resolve_model_settings(session, PROVIDER_OPENAI, "gpt-5.6-terra")

    assert deepseek_resolved.reasoning_effort == REASONING_MAX
    assert deepseek_resolved.max_output_tokens is None
    assert openai_resolved.max_output_tokens == 12000


@pytest.mark.asyncio
async def test_new_model_does_not_inherit_legacy_global_override_after_scoping(settings_db):
    async with settings_db() as session:
        config = await session.get(AIConfig, 1)
        config.max_output_tokens = 12000
        await session.commit()
        legacy_model = await get_or_create_model_settings(session, PROVIDER_DEEPSEEK, "deepseek-v4-pro", config=config)
        assert legacy_model.max_output_tokens == 12000
        config.deepseek_model = "deepseek-flash"
        await session.commit()
        resolved = await resolve_model_settings(session, PROVIDER_DEEPSEEK, "deepseek-flash", config=config)

    assert resolved.max_output_tokens is None
    assert resolved.reasoning_effort == REASONING_AUTO


def test_auto_wire_semantics_and_validation():
    deepseek = SimpleNamespace(
        provider=PROVIDER_DEEPSEEK,
        model="deepseek-v4-pro",
        channel="chat",
        max_output_tokens=None,
        temperature=0.7,
        reasoning_effort=REASONING_AUTO,
    )
    assert wire_max_output_tokens(deepseek) is None
    assert wire_reasoning_effort(deepseek) is None
    assert wire_temperature(deepseek) is None
    assert validate_model_setting(PROVIDER_DEEPSEEK, "deepseek-v4-pro", "max_output_tokens", 393216, reasoning_effort=REASONING_AUTO) == 393216

    deepseek.reasoning_effort = REASONING_MAX
    assert wire_max_output_tokens(deepseek) is None
    assert wire_reasoning_effort(deepseek) == REASONING_MAX
    assert wire_temperature(deepseek) is None

    deepseek.reasoning_effort = REASONING_NONE
    assert wire_reasoning_effort(deepseek) == REASONING_NONE
    assert wire_temperature(deepseek) == 0.7
    assert validate_model_setting(PROVIDER_DEEPSEEK, "deepseek-v4-pro", "max_output_tokens", 393216, reasoning_effort=REASONING_MAX) == 393216
    with pytest.raises(ValueError):
        validate_model_setting(PROVIDER_DEEPSEEK, "deepseek-v4-pro", "max_output_tokens", 393217, reasoning_effort=REASONING_MAX)
    assert validate_model_setting(PROVIDER_DEEPSEEK, "deepseek-v4-pro", "temperature", "Авто") is None


class _Completion:
    choices = [SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reasoning", "field", "value"),
    [
        (REASONING_AUTO, "reasoning_effort", None),
        (REASONING_NONE, "reasoning_effort", REASONING_NONE),
        ("low", "reasoning_effort", "low"),
        ("high", "reasoning_effort", "high"),
        (REASONING_MAX, "reasoning_effort", REASONING_MAX),
    ],
)
async def test_deepseek_wire_reasoning_payloads(reasoning, field, value):
    capture = {}

    async def create(**kwargs):
        capture.update(kwargs)
        return _Completion()

    with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=create):
        await ai_integration._call_deepseek_api(
            "key",
            "deepseek-v4-pro",
            [],
            "",
            "system",
            temperature=0.7,
            use_proxy=False,
            request_layout=AIRequestLayout(stable_system_prompt="system", current_user_content="hello"),
            reasoning_effort=None if reasoning == REASONING_AUTO else reasoning,
        )

    assert capture.get("model") == "deepseek-v4-pro"
    if reasoning == REASONING_AUTO:
        assert field not in capture
        assert "extra_body" not in capture
    elif reasoning == REASONING_NONE:
        assert "reasoning_effort" not in capture
        assert capture["extra_body"] == {"thinking": {"type": "disabled"}}
    else:
        assert capture[field] == value
        assert capture["extra_body"] == {"thinking": {"type": "enabled"}}


@pytest.mark.asyncio
async def test_deepseek_flash_vision_payload_parity_and_log_redaction():
    captures = []

    async def create(**kwargs):
        captures.append(kwargs)
        return _Completion()

    image_bytes = b"\xff\xd8" + (b"vision-fixture" * 32)
    layout = AIRequestLayout(stable_system_prompt="system", current_user_content="describe")
    with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=create):
        await ai_integration._call_deepseek_vision(
            "key",
            "deepseek-flash",
            image_bytes,
            "system",
            request_layout=layout,
        )
        await max_ai._analyze_deepseek(
            "key",
            "deepseek-flash",
            image_bytes,
            "system",
            "describe",
            request_layout=layout,
        )

    assert len(captures) == 2
    first_content = captures[0]["messages"][-1]["content"]
    second_content = captures[1]["messages"][-1]["content"]
    assert first_content == second_content
    image_url = first_content[1]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    assert image_url == second_content[1]["image_url"]["url"]
    sanitized = sanitize_vision_request_payload(captures[0])
    assert image_url not in str(sanitized)
    assert "<redacted_media_url>" in str(sanitized)


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ("telegram", "max"))
async def test_deepseek_pro_is_rejected_for_vision(surface):
    image_bytes = b"\xff\xd8vision"
    with pytest.raises(ModelUnavailableError):
        if surface == "telegram":
            await ai_integration._call_deepseek_vision("key", "deepseek-v4-pro", image_bytes, "system")
        else:
            await max_ai._analyze_deepseek("key", "deepseek-v4-pro", image_bytes, "system", "describe")


def test_admin_generation_controls_are_only_on_model_card():
    labels = [button.text for row in keyboards.ai_settings_keyboard(PROVIDER_DEEPSEEK).inline_keyboard for button in row]
    assert not any("Max" in label or "Thinking" in label or "Reasoning" in label for label in labels)
    card_labels = [button.text for row in keyboards.model_settings_keyboard(show_reasoning=True, show_temperature=True).inline_keyboard for button in row]
    assert "📏 Max tokens" in card_labels
    assert "🧠 Reasoning" in card_labels
    openai_labels = [button.text for row in keyboards.model_settings_keyboard(show_reasoning=False, show_temperature=True).inline_keyboard for button in row]
    assert "🧠 Reasoning" not in openai_labels


def test_provider_key_and_model_layouts_are_four_by_two():
    markup = keyboards.ai_keys_models_keyboard(
        "OpenAI", 2, 10, "Gemini", "gemini-3.7-flash", "OpenAI", "gpt-image-2", "KIE", "seedream/4.5-edit", 0,
        api_keys={name: None for name in ("Deepseek", "Claude", "Gemini", "KIE", "OpenAI", "OpenRouter", "Perplexity", "Deepgram")},
        current_provider=PROVIDER_DEEPSEEK,
        current_model="deepseek-v4-pro",
    )
    rows = markup.inline_keyboard
    assert all(len(rows[index]) == 2 for index in range(8))
    assert rows[3][1].text.startswith("🔑 Deepgram")
    assert rows[7][1].text.startswith("🗣️ Deepgram")
    assert "nova-3" in rows[7][1].text or "nova-3" in rows[3][1].text


def test_deepgram_transcription_model_callback_resolves_without_chat_selection():
    callback = build_telegram_model_callback_data(PROVIDER_DEEPGRAM, "transcription", "nova-3")
    assert resolve_telegram_model_callback(callback) == (PROVIDER_DEEPGRAM, "transcription", "nova-3")


def test_capability_picker_catalog_has_only_supported_providers_and_models():
    assert PROVIDER_DEEPGRAM in get_capability_providers("transcription")
    assert PROVIDER_DEEPGRAM not in get_capability_providers("chat")
    assert PROVIDER_DEEPSEEK in get_capability_providers("vision")
    assert "deepseek-flash" in get_selectable_models(PROVIDER_DEEPSEEK, channel="vision")
    assert "deepseek-v4-pro" not in get_selectable_models(PROVIDER_DEEPSEEK, channel="vision")
    assert get_capability_providers("vision_fallback") == get_capability_providers("vision")
    assert get_capability_providers("image_generation") == get_capability_providers("image_gen")


def test_max_admin_key_and_model_cards_are_four_by_two():
    config = AIConfig(id=1, provider=PROVIDER_DEEPSEEK, deepseek_model="deepseek-v4-pro")
    attachments = _build_keys_keyboard(config)
    rows = attachments[0]["payload"]["buttons"]
    assert all(len(row) == 2 for row in rows[:4])
    assert all(len(row) == 2 for row in rows[5:9])
    assert "Deepgram" in rows[3][1]["text"]
    assert rows[8][1]["text"].startswith("🗣️ Deepgram · nova-3")


def test_capability_controls_open_explicit_pickers_not_cycles():
    markup = keyboards.ai_keys_models_keyboard(
        "OpenAI", 2, 10, "Gemini", "gemini-3.7-flash", "OpenAI", "gpt-image-2", "KIE", "seedream/4.5-edit", 0,
        api_keys={}, current_provider=PROVIDER_DEEPSEEK, current_model="deepseek-flash",
    )
    callbacks = {
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if button.callback_data
    }
    assert "admin_select_transcription_provider" in callbacks
    assert "admin_select_vision_provider" in callbacks
    assert "admin_select_image_generation_provider" in callbacks
    assert "admin_select_image_edit_provider" in callbacks
    assert "admin_toggle_transcription" not in callbacks
    assert "admin_toggle_vision" not in callbacks
    assert "admin_toggle_image_generation" not in callbacks
    assert "admin_toggle_image_edit" not in callbacks


PICKER_CASES = tuple(
    (channel, provider)
    for channel in ("transcription", "vision", "image_gen", "image_edit")
    for provider in get_capability_providers(channel)
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("channel", "provider"), PICKER_CASES)
async def test_telegram_capability_picker_persists_provider_and_model(
    settings_db, monkeypatch, channel, provider
):
    callback = SimpleNamespace(
        data=f"admin_choose_capability_{channel}_{provider}",
        message=SimpleNamespace(edit_text=AsyncMock()),
        answer=AsyncMock(),
    )
    monkeypatch.setattr(handlers, "async_session_maker", settings_db)
    monkeypatch.setattr(handlers, "admin_ai_keys_models", AsyncMock())

    await handlers.choose_capability_provider(callback)

    expected_model = get_default_model(provider, channel=channel)
    async with settings_db() as session:
        config = await session.get(AIConfig, 1)
        if channel == "transcription":
            assert config.transcription_provider == provider
            if provider == PROVIDER_DEEPGRAM:
                assert config.deepgram_model == expected_model
        elif channel == "vision":
            assert (config.vision_provider, config.vision_model) == (provider, expected_model)
        elif channel == "image_gen":
            assert (config.image_generation_provider, config.image_generation_model) == (provider, expected_model)
        else:
            assert (config.image_edit_provider, config.image_edit_model) == (provider, expected_model)

    callback.answer.assert_awaited_once()
    callback.message.edit_text.assert_awaited_once()
    rendered_markup = callback.message.edit_text.await_args.kwargs["reply_markup"]
    rendered_callbacks = {
        button.callback_data
        for row in rendered_markup.inline_keyboard
        for button in row
        if button.callback_data
    }
    assert any(callback_data.startswith("ai_m_") for callback_data in rendered_callbacks)


@pytest.mark.asyncio
@pytest.mark.parametrize(("channel", "provider"), PICKER_CASES)
async def test_max_capability_picker_persists_provider_and_returns_to_model_list(
    settings_db, monkeypatch, channel, provider
):
    client = SimpleNamespace(send_message=AsyncMock())
    monkeypatch.setattr(max_admin_ai, "async_session_maker", settings_db)

    await max_admin_ai.choose_capability_provider(client, 1001, channel, provider)

    expected_model = get_default_model(provider, channel=channel)
    async with settings_db() as session:
        config = await session.get(AIConfig, 1)
        if channel == "transcription":
            assert config.transcription_provider == provider
            if provider == PROVIDER_DEEPGRAM:
                assert config.deepgram_model == expected_model
        elif channel == "vision":
            assert (config.vision_provider, config.vision_model) == (provider, expected_model)
        elif channel == "image_gen":
            assert (config.image_generation_provider, config.image_generation_model) == (provider, expected_model)
        else:
            assert (config.image_edit_provider, config.image_edit_model) == (provider, expected_model)

    client.send_message.assert_awaited_once()
    rendered = client.send_message.await_args.kwargs
    assert rendered["chat_id"] == 1001
    assert "Выберите модель" in rendered["text"]
    callbacks = {
        button["payload"]
        for row in rendered["attachments"][0]["payload"]["buttons"]
        for button in row
        if button.get("type") == "callback"
    }
    assert any(callback_data.startswith("admin_ai_set_channel_model_") for callback_data in callbacks)


@pytest.mark.asyncio
@pytest.mark.parametrize(("channel", "provider"), PICKER_CASES)
async def test_telegram_capability_model_choice_uses_same_stable_callback(
    settings_db, monkeypatch, channel, provider
):
    model = get_default_model(provider, channel=channel)
    callback = SimpleNamespace(
        data=build_telegram_model_callback_data(provider, channel, model),
        message=SimpleNamespace(edit_text=AsyncMock()),
        answer=AsyncMock(),
    )
    monkeypatch.setattr(handlers, "async_session_maker", settings_db)
    monkeypatch.setattr(handlers, "admin_ai_keys_models", AsyncMock())

    await handlers.handle_compact_model_callback(callback)

    async with settings_db() as session:
        config = await session.get(AIConfig, 1)
        if channel == "transcription":
            assert config.transcription_provider == provider
            if provider == PROVIDER_DEEPGRAM:
                assert config.deepgram_model == model
        elif channel == "vision":
            assert (config.vision_provider, config.vision_model) == (provider, model)
        elif channel == "image_gen":
            assert (config.image_generation_provider, config.image_generation_model) == (provider, model)
        else:
            assert (config.image_edit_provider, config.image_edit_model) == (provider, model)

    callback.answer.assert_awaited_once()
