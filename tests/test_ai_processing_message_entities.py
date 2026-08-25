import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aiogram.types import MessageEntity

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import handlers
from database import DEFAULT_AI_PROCESSING_MESSAGE_TEXT


class _Session:
    def __init__(self, config):
        self.config = config
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, model, key):
        return self.config

    def add(self, value):
        return None

    async def commit(self):
        self.commits += 1


class _Bot:
    def __init__(self):
        self.send_message = AsyncMock(return_value=SimpleNamespace())


def _custom(offset: int, length: int, emoji_id: str) -> MessageEntity:
    return MessageEntity(
        type="custom_emoji",
        offset=offset,
        length=length,
        custom_emoji_id=emoji_id,
    )


@pytest.mark.asyncio
async def test_plain_text_and_normal_unicode_emoji_remain_legacy_values():
    assert handlers.serialize_ai_processing_message_text("Думаю...") == "Думаю..."
    assert handlers.serialize_ai_processing_message_text("🙂 Жду") == "🙂 Жду"

    bot = _Bot()
    config = SimpleNamespace(
        ai_processing_message_enabled=True,
        ai_processing_message_text="🙂 Жду",
    )
    await handlers._send_ai_processing_message(bot, 42, config)
    kwargs = bot.send_message.await_args.kwargs
    assert kwargs["text"] == "🙂 Жду"
    assert kwargs["parse_mode"] is None
    assert "entities" not in kwargs


@pytest.mark.asyncio
async def test_admin_save_preserves_exact_custom_emoji_id(monkeypatch):
    config = SimpleNamespace(
        ai_processing_message_enabled=True,
        ai_processing_message_text="Думаю...",
    )
    session = _Session(config)
    message = SimpleNamespace(
        text="Жду 🙂",
        entities=[_custom(5, 2, "987654321")],
        chat=SimpleNamespace(id=42),
        delete=AsyncMock(),
        answer=AsyncMock(),
    )
    state = SimpleNamespace(get_data=AsyncMock(return_value={}), clear=AsyncMock())
    bot = SimpleNamespace(edit_message_text=AsyncMock())

    monkeypatch.setattr(handlers, "async_session_maker", lambda: session)
    monkeypatch.setattr(handlers, "admin_general_settings", AsyncMock())

    await handlers.admin_save_ai_processing_message_text(message, state, bot)

    assert config.ai_processing_message_text.startswith(handlers._AI_PROCESSING_ENTITIES_PREFIX)
    decoded_text, entities, encoded = handlers._decode_ai_processing_message_text(
        config.ai_processing_message_text
    )
    assert encoded is True
    assert decoded_text == "Жду 🙂"
    assert [entity.custom_emoji_id for entity in entities] == ["987654321"]


@pytest.mark.asyncio
async def test_multiple_mixed_custom_emoji_are_sent_as_aiogram_entities():
    text = "🙂 A 😀"
    entities = [
        _custom(0, 2, "first-id"),
        MessageEntity(type="bold", offset=3, length=1),
        _custom(5, 2, "second-id"),
    ]
    stored = handlers.serialize_ai_processing_message_text(text, entities)
    bot = _Bot()
    config = SimpleNamespace(ai_processing_message_enabled=True, ai_processing_message_text=stored)

    await handlers._send_ai_processing_message(bot, 42, config)

    kwargs = bot.send_message.await_args.kwargs
    assert kwargs["text"] == text
    assert [entity.custom_emoji_id for entity in kwargs["entities"] if entity.custom_emoji_id] == [
        "first-id",
        "second-id",
    ]
    assert [entity.type for entity in kwargs["entities"]] == [
        "custom_emoji",
        "bold",
        "custom_emoji",
    ]
    assert kwargs["parse_mode"] is None


def test_preview_escapes_arbitrary_legacy_html_text():
    assert handlers._ai_processing_message_html("<b>& unsafe") == "&lt;b&gt;&amp; unsafe"


@pytest.mark.asyncio
async def test_legacy_plain_text_value_is_sent_unchanged():
    bot = _Bot()
    config = SimpleNamespace(
        ai_processing_message_enabled=True,
        ai_processing_message_text="legacy <text> & value",
    )

    await handlers._send_ai_processing_message(bot, 42, config)

    kwargs = bot.send_message.await_args.kwargs
    assert kwargs["text"] == "legacy <text> & value"
    assert kwargs["parse_mode"] is None
    assert "entities" not in kwargs


@pytest.mark.asyncio
async def test_malformed_serialized_value_uses_safe_default_and_does_not_raise():
    bot = _Bot()
    config = SimpleNamespace(
        ai_processing_message_enabled=True,
        ai_processing_message_text=handlers._AI_PROCESSING_ENTITIES_PREFIX + "not-valid",
    )

    await handlers._send_ai_processing_message(bot, 42, config)

    kwargs = bot.send_message.await_args.kwargs
    assert kwargs["text"] == DEFAULT_AI_PROCESSING_MESSAGE_TEXT
    assert kwargs["parse_mode"] is None
    assert "entities" not in kwargs
