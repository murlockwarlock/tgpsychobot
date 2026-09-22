import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("BOT_TOKEN", "test")

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import ai_integration
import keyboards
from database import AIConfig, Base, Topic, User
from max_messenger_bot import ai as max_ai
from provider_models import (
    DEEPSEEK_CHAT_MAX_TOKENS,
    PROVIDER_DEEPSEEK,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    PROVIDER_OPENROUTER,
    PROVIDER_PERPLEXITY,
    effective_chat_output_tokens,
    get_default_model,
    get_chat_output_token_limit,
    validate_chat_output_tokens,
)


class _Completion:
    choices = [SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")]


@pytest.fixture
async def generation_db():
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
                provider="Deepseek",
                deepseek_api_key="deepseek-key",
                deepseek_model="deepseek-v4-flash",
                openai_api_key="openai-key",
                openai_model="gpt-5.6-terra",
                gemini_api_key="gemini-key",
                gemini_model="gemini-3.7-flash",
                system_prompt="Система",
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


def test_output_budget_defaults_and_validation():
    assert get_chat_output_token_limit(PROVIDER_DEEPSEEK, "deepseek-v4-flash") == DEEPSEEK_CHAT_MAX_TOKENS
    assert effective_chat_output_tokens(PROVIDER_DEEPSEEK, "deepseek-v4-flash", None) == DEEPSEEK_CHAT_MAX_TOKENS
    assert effective_chat_output_tokens(PROVIDER_DEEPSEEK, "deepseek-v4-flash", 12000) == 12000
    assert effective_chat_output_tokens(PROVIDER_DEEPSEEK, "deepseek-v4-flash", 999999) == DEEPSEEK_CHAT_MAX_TOKENS
    assert validate_chat_output_tokens(PROVIDER_OPENAI, "gpt-5.6-terra", None) is None
    assert validate_chat_output_tokens(PROVIDER_OPENAI, "gpt-5.6-terra", "12000") == 12000
    with pytest.raises(ValueError, match="максимальное значение"):
        validate_chat_output_tokens(PROVIDER_GEMINI, "gemini-3.7-flash", "20000")
    with pytest.raises(ValueError):
        validate_chat_output_tokens(PROVIDER_OPENAI, "gpt-5.6-terra", "0")
    with pytest.raises(ValueError):
        validate_chat_output_tokens(PROVIDER_OPENAI, "gpt-5.6-terra", "not-a-number")
    assert effective_chat_output_tokens(PROVIDER_PERPLEXITY, "fast", 50000) == 8192


def test_admin_exposes_deepseek_thinking_only_for_deepseek():
    provider_labels = [
        button.text
        for row in keyboards.ai_settings_keyboard("OpenRouter").inline_keyboard
        for button in row
    ]
    assert any("OpenRouter" in label for label in provider_labels)
    assert any("Perplexity" in label for label in provider_labels)

    common = dict(
        current_transcription_provider="OpenAI",
        context_first=2,
        context_recent=10,
        current_vision_provider="Gemini",
        current_vision_model="gemini-3.7-flash",
        image_generation_provider="OpenAI",
        image_generation_model="gpt-image-2",
        image_edit_provider="KIE",
        image_edit_model="seedream/4.5-edit",
        kie_credit_alert_threshold=0,
        current_model="deepseek-v4-flash",
        max_output_tokens=None,
        deepseek_thinking_enabled=True,
    )
    deepseek_markup = keyboards.ai_keys_models_keyboard(current_provider="Deepseek", **common)
    deepseek_labels = [button.text for row in deepseek_markup.inline_keyboard for button in row]
    assert any("Thinking DeepSeek: Включён" in label for label in deepseek_labels)
    assert any("По умолчанию" in label for label in deepseek_labels)
    common["deepseek_thinking_enabled"] = None
    default_markup = keyboards.ai_keys_models_keyboard(current_provider="Deepseek", **common)
    default_labels = [button.text for row in default_markup.inline_keyboard for button in row]
    assert any("Thinking DeepSeek: По умолчанию" in label for label in default_labels)
    common["deepseek_thinking_enabled"] = False
    disabled_markup = keyboards.ai_keys_models_keyboard(current_provider="Deepseek", **common)
    disabled_labels = [button.text for row in disabled_markup.inline_keyboard for button in row]
    assert any("Thinking DeepSeek: Выключен" in label for label in disabled_labels)
    openai_markup = keyboards.ai_keys_models_keyboard(current_provider="OpenAI", **common)
    openai_labels = [button.text for row in openai_markup.inline_keyboard for button in row]
    assert not any("Thinking" in label for label in openai_labels)
    for provider in ("OpenRouter", "Perplexity"):
        provider_markup = keyboards.ai_keys_models_keyboard(current_provider=provider, **common)
        provider_labels = [button.text for row in provider_markup.inline_keyboard for button in row]
        assert any(provider in label for label in provider_labels)
        assert not any("Thinking" in label for label in provider_labels)
    deepgram_common = dict(common)
    deepgram_common["current_transcription_provider"] = "Deepgram"
    deepgram_markup = keyboards.ai_keys_models_keyboard(current_provider="OpenRouter", **deepgram_common)
    deepgram_labels = [button.text for row in deepgram_markup.inline_keyboard for button in row]
    assert any("Deepgram" in label for label in deepgram_labels)
    choice_markup = keyboards.deepseek_thinking_keyboard()
    choices = {
        button.text: button.callback_data
        for row in choice_markup.inline_keyboard
        for button in row
    }
    assert choices["По умолчанию"] == "set_deepseek_thinking_default"
    assert choices["Включён"] == "set_deepseek_thinking_on"
    assert choices["Выключен"] == "set_deepseek_thinking_off"


@pytest.mark.asyncio
async def test_deepseek_thinking_and_custom_budget_are_per_config(generation_db):
    sessions = generation_db
    async with sessions() as session:
        config = await session.get(AIConfig, 1)
        config.deepseek_thinking_enabled = True
        config.max_output_tokens = 12000
        await session.commit()

    captured = []

    async def create(**kwargs):
        captured.append(kwargs)
        return _Completion()

    with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=create):
        assert await ai_integration.generate_response(101, "Проверка") == "ok"

    assert captured[0]["max_tokens"] == 12000
    assert captured[0]["extra_body"] == {"thinking": {"type": "enabled"}}

    async with sessions() as session:
        config = await session.get(AIConfig, 1)
        config.deepseek_thinking_enabled = False
        config.max_output_tokens = None
        await session.commit()

    captured.clear()
    with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=create):
        assert await ai_integration.generate_response(101, "Проверка 2") == "ok"
    assert captured[0]["max_tokens"] == DEEPSEEK_CHAT_MAX_TOKENS
    assert captured[0]["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
async def test_deepseek_default_is_distinct_from_explicit_disabled(generation_db):
    sessions = generation_db
    captured = []

    async def create(**kwargs):
        captured.append(kwargs)
        return _Completion()

    with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=create):
        assert await ai_integration.generate_response(101, "Проверка default") == "ok"

    assert "extra_body" not in captured[0]

    async with sessions() as session:
        config = await session.get(AIConfig, 1)
        config.deepseek_thinking_enabled = False
        await session.commit()

    captured.clear()
    with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=create):
        assert await ai_integration.generate_response(101, "Проверка disabled") == "ok"

    assert captured[0]["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
async def test_openai_uses_shared_budget_without_reasoning(generation_db):
    sessions = generation_db
    async with sessions() as session:
        config = await session.get(AIConfig, 1)
        config.provider = "OpenAI"
        config.max_output_tokens = 7000
        await session.commit()

    captured = []

    async def create(**kwargs):
        captured.append(kwargs)
        return _Completion()

    with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=create):
        assert await ai_integration.generate_response(101, "Проверка OpenAI") == "ok"

    assert captured[0]["max_completion_tokens"] == 7000
    assert "extra_body" not in captured[0]
    assert "thinking" not in captured[0]


@pytest.mark.asyncio
async def test_openrouter_uses_shared_budget_without_thinking_control(generation_db):
    sessions = generation_db
    async with sessions() as session:
        config = await session.get(AIConfig, 1)
        config.provider = "OpenRouter"
        config.openrouter_api_key = "openrouter-key"
        config.openrouter_model = "openai/gpt-5.6-terra"
        config.max_output_tokens = 7000
        await session.commit()

    with patch("ai_integration.call_openrouter", AsyncMock(return_value="ok")) as call:
        assert await ai_integration.generate_response(101, "Проверка OpenRouter") == "ok"

    assert call.await_args.kwargs["max_output_tokens"] == 7000
    assert "thinking_enabled" not in call.await_args.kwargs


@pytest.mark.asyncio
async def test_perplexity_uses_shared_budget_and_default_when_reset(generation_db):
    sessions = generation_db
    async with sessions() as session:
        config = await session.get(AIConfig, 1)
        config.provider = "Perplexity"
        config.perplexity_api_key = "perplexity-key"
        config.perplexity_model = "medium"
        config.max_output_tokens = 6000
        await session.commit()

    with patch("ai_integration.call_perplexity", AsyncMock(return_value="Ответ\n\nИсточники:")) as call:
        assert await ai_integration.generate_response(101, "Проверка Perplexity") == "Ответ\n\nИсточники:"

    assert call.await_args.kwargs["max_output_tokens"] == 6000

    async with sessions() as session:
        config = await session.get(AIConfig, 1)
        config.max_output_tokens = None
        await session.commit()

    with patch("ai_integration.call_perplexity", AsyncMock(return_value="Ответ")) as call:
        assert await ai_integration.generate_response(101, "Проверка Perplexity default") == "Ответ"

    assert call.await_args.kwargs["max_output_tokens"] == 128000


@pytest.mark.asyncio
async def test_new_provider_model_defaults_are_effective_without_backfill(generation_db):
    sessions = generation_db
    async with sessions() as session:
        config = await session.get(AIConfig, 1)
        config.provider = PROVIDER_OPENROUTER
        config.openrouter_api_key = "openrouter-key"
        config.openrouter_model = None
        await session.commit()

    with patch("ai_integration.call_openrouter", AsyncMock(return_value="ok")) as call:
        assert await ai_integration.generate_response(101, "Проверка default модели") == "ok"

    assert call.await_args.args[2] == get_default_model(PROVIDER_OPENROUTER)
    async with sessions() as session:
        config = await session.get(AIConfig, 1)
        assert config.openrouter_model is None


@pytest.mark.asyncio
async def test_max_deepseek_uses_same_per_bot_settings(generation_db):
    sessions = generation_db
    async with sessions() as session:
        config = await session.get(AIConfig, 1)
        config.max_output_tokens = 9000
        config.deepseek_thinking_enabled = True
        await session.commit()

    captured = []

    async def create(**kwargs):
        captured.append(kwargs)
        return _Completion()

    with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=create):
        assert await max_ai.get_ai_response(101, "Проверка MAX") == "ok"

    assert captured[0]["max_tokens"] == 9000
    assert captured[0]["extra_body"] == {"thinking": {"type": "enabled"}}


@pytest.mark.asyncio
async def test_max_openrouter_uses_shared_budget(generation_db):
    sessions = generation_db
    async with sessions() as session:
        config = await session.get(AIConfig, 1)
        config.provider = "OpenRouter"
        config.openrouter_api_key = "openrouter-key"
        config.openrouter_model = "anthropic/claude-sonnet-4.6"
        config.max_output_tokens = 5000
        await session.commit()

    with patch("max_messenger_bot.ai.call_openrouter", AsyncMock(return_value="ok")) as call:
        assert await max_ai.get_ai_response(101, "Проверка MAX OpenRouter") == "ok"

    assert call.await_args.kwargs["max_output_tokens"] == 5000
