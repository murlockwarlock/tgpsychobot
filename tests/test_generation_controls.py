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
    effective_chat_output_tokens,
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


def test_admin_exposes_deepseek_thinking_only_for_deepseek():
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
    assert any("Thinking: Включён" in label for label in deepseek_labels)
    assert any("По умолчанию" in label for label in deepseek_labels)
    openai_markup = keyboards.ai_keys_models_keyboard(current_provider="OpenAI", **common)
    openai_labels = [button.text for row in openai_markup.inline_keyboard for button in row]
    assert not any("Thinking" in label for label in openai_labels)


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
