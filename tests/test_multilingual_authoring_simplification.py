import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

os.environ.setdefault("BOT_TOKEN", "123456:test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from aiogram.types import Chat, Message, Update, User
from aiogram.client.session.base import BaseSession
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from content_authoring import (
    AUTHORING_RESOURCES,
    authoring_locales,
    multilingual_authoring_enabled,
    read_content_value,
    save_content_value,
)
from database import Base, BotGeneralConfig, BotTranslation, Content, ContentMedia, SubscriptionPlan, TestQuestion as Question, Topic, User as DBUser
from translation_pack_manager import audit_translation_readiness
from translation_registry import TranslationRegistry, TranslationSource
from translation_service import source_hash


@pytest_asyncio.fixture
async def factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions() as session:
        session.add(BotGeneralConfig(id=1, telegram_enabled_languages='["ru", "en", "pt"]'))
        await session.commit()
    yield sessions
    await engine.dispose()


def test_authoring_whitelist_is_explicit():
    assert set(AUTHORING_RESOURCES) == {
        "topic",
        "content",
        "plan",
        "referral_template",
        "subscription_config",
        "media_library",
    }
    assert "test_question" not in AUTHORING_RESOURCES
    assert "automation_action" not in AUTHORING_RESOURCES
    assert "followup_step" not in AUTHORING_RESOURCES


@pytest.mark.asyncio
async def test_multilingual_authoring_is_per_bot_and_locales_follow_enabled_languages(factory):
    async with factory() as session:
        config = await session.get(BotGeneralConfig, 1)
        assert not await multilingual_authoring_enabled(session)
        config.multilingual_authoring_enabled = True
        await session.commit()
    async with factory() as session:
        assert await multilingual_authoring_enabled(session)
        assert await authoring_locales(session) == ("ru", "en", "pt")
        config = await session.get(BotGeneralConfig, 1)
        config.telegram_enabled_languages = '["ru"]'
        await session.commit()
    async with factory() as session:
        assert await authoring_locales(session) == ("ru",)


@pytest.mark.asyncio
async def test_object_local_values_reuse_one_topic_and_preserve_russian(factory):
    async with factory() as session:
        topic = Topic(id=17, name="Отношения", description="Русское описание")
        session.add(topic)
        await session.flush()
        await save_content_value(session, "topic", topic, "name", "en", "Relationships")
        await save_content_value(session, "topic", topic, "name", "pt", "Relacionamentos")
        await session.commit()
    async with factory() as session:
        topic = await session.get(Topic, 17)
        assert topic.name == "Отношения"
        assert (await read_content_value(session, "topic", topic, "name", "en")).text == "Relationships"
        assert (await read_content_value(session, "topic", topic, "name", "pt")).text == "Relacionamentos"
        assert await session.scalar(select(Topic.id).where(Topic.name == "Отношения")) == 17


@pytest.mark.asyncio
async def test_system_readiness_ignores_dynamic_content(factory):
    registry = TranslationRegistry([
        TranslationSource("ui.example", "Система"),
        TranslationSource("topic.17.name", "Отношения"),
    ])
    async with factory() as session:
        session.add(BotTranslation(
            locale="en",
            translation_key="topic.17.name",
            text="Relationships",
            source_hash=source_hash("Отношения"),
        ))
        await session.commit()
        readiness = await audit_translation_readiness(session, registry, locales=("en",))
    report = readiness["locales"]["en"]
    assert report["required"] == 1
    assert report["missing"] == ["ui.example"]
    assert readiness.get("content") == {}


@pytest.mark.asyncio
async def test_content_media_variant_uses_one_content_identity(factory):
    async with factory() as session:
        content = Content(key="welcome", text_content="Русский текст")
        content.media.append(ContentMedia(file_type="photo", file_id="ru-photo"))
        session.add(content)
        await session.flush()
        await save_content_value(session, "content", content, "text_content", "pt", "Texto português")
        await save_content_value(
            session,
            "content",
            content,
            "media",
            "pt",
            json.dumps([{"type": "photo", "file_id": "pt-photo"}], ensure_ascii=False),
        )
        await session.commit()
    async with factory() as session:
        content = await session.get(Content, "welcome")
        assert content.text_content == "Русский текст"
        assert (await read_content_value(session, "content", content, "text_content", "pt")).text == "Texto português"
        media = await read_content_value(session, "content", content, "media", "pt")
        assert json.loads(media.text) == [{"type": "photo", "file_id": "pt-photo"}]
        assert await session.scalar(select(Content.key).where(Content.key == "welcome")) == "welcome"
    from translation_service import refresh_translation_cache
    snapshot = await refresh_translation_cache(factory, force=True)
    assert json.loads(snapshot.get("content.welcome.media", "pt")) == [
        {"type": "photo", "file_id": "pt-photo"}
    ]


@pytest.mark.asyncio
async def test_content_runtime_uses_locale_media_and_russian_fallback(factory, monkeypatch):
    from database import User
    from handlers import get_content_from_db
    import handlers
    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers.kb, "content_management_keyboard", AsyncMock(return_value=None))
    async with factory() as session:
        config = await session.get(BotGeneralConfig, 1)
        config.telegram_language_selection_enabled = True
        content = Content(key="fallback", text_content="Русский")
        content.media.append(ContentMedia(file_type="photo", file_id="ru-photo"))
        session.add_all([content, User(id=41, telegram_language_code="pt")])
        await session.flush()
        await save_content_value(session, "content", content, "text_content", "pt", "Português")
        await save_content_value(
            session,
            "content",
            content,
            "media",
            "pt",
            json.dumps([{"type": "photo", "file_id": "pt-photo"}], ensure_ascii=False),
        )
        await session.commit()
    from translation_service import refresh_translation_cache
    await refresh_translation_cache(factory, force=True)
    localized = await get_content_from_db("fallback", user_id=41)
    assert localized["text"] == "Português"
    assert localized["media"] == [{"type": "photo", "file_id": "pt-photo"}]
    async with factory() as session:
        config = await session.get(BotGeneralConfig, 1)
        config.telegram_enabled_languages = '["ru", "en", "pt"]'
        await session.commit()


@pytest.mark.asyncio
async def test_content_renderer_uses_locale_media_and_preserves_explicit_empty_override(factory, monkeypatch):
    import handlers
    from handlers import render_static_content_telegram
    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers.kb, "main_client_keyboard", AsyncMock(return_value=None))
    async with factory() as session:
        config = await session.get(BotGeneralConfig, 1)
        config.telegram_language_selection_enabled = True
        content = Content(key="render_media", text_content="Русский текст")
        content.media.append(ContentMedia(file_type="photo", file_id="ru-photo"))
        session.add_all([content, DBUser(id=42, telegram_language_code="pt")])
        await session.flush()
        await save_content_value(session, "content", content, "text_content", "pt", "Texto português")
        await save_content_value(
            session,
            "content",
            content,
            "media",
            "pt",
            json.dumps([{"type": "photo", "file_id": "pt-photo"}], ensure_ascii=False),
        )
        await session.commit()
    from translation_service import refresh_translation_cache
    await refresh_translation_cache(factory, force=True)
    bot = SimpleNamespace(
        send_photo=AsyncMock(),
        send_video=AsyncMock(),
        send_media_group=AsyncMock(),
        send_message=AsyncMock(),
    )
    assert await render_static_content_telegram(bot, 42, 42, "render_media")
    assert bot.send_photo.call_args.args[1] == "pt-photo"

    async with factory() as session:
        content = await session.get(Content, "render_media")
        await save_content_value(session, "content", content, "media", "pt", "[]")
        await session.commit()
    await refresh_translation_cache(factory, force=True)
    bot.send_photo.reset_mock()
    bot.send_message.reset_mock()
    assert await render_static_content_telegram(bot, 42, 42, "render_media")
    bot.send_photo.assert_not_called()
    assert any("Texto português" in call.args[1] for call in bot.send_message.call_args_list if len(call.args) > 1)


@pytest.mark.asyncio
async def test_content_settings_save_does_not_materialize_russian_media_as_missing_locale_variant(factory, monkeypatch):
    import handlers
    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers.kb, "content_management_keyboard", AsyncMock(return_value=None))
    async with factory() as session:
        content = Content(key="settings_media", text_content="Русский текст")
        content.media.append(ContentMedia(file_type="photo", file_id="ru-photo"))
        session.add(content)
        await session.commit()

    class State:
        async def get_data(self):
            return {
                "content_key": "settings_media",
                "text_content": "Texto português",
                "media_files": [{"type": "photo", "file_id": "ru-photo"}],
                "content_order": "media_top",
                "authoring_locale": "pt",
                "media_variant_present": False,
                "media_variant_touched": False,
            }

        async def clear(self):
            return None

    callback = SimpleNamespace(
        message=SimpleNamespace(edit_text=AsyncMock()),
        answer=AsyncMock(),
    )
    await handlers.save_content(callback, State())

    async with factory() as session:
        assert await session.scalar(
            select(BotTranslation.text).where(
                BotTranslation.locale == "pt",
                BotTranslation.translation_key == "content.settings_media.media",
            )
        ) is None
        media_rows = (await session.scalars(
            select(ContentMedia).where(ContentMedia.content_key == "settings_media")
        )).all()
        assert [(item.file_type, item.file_id) for item in media_rows] == [("photo", "ru-photo")]


class _RecordingSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def close(self):
        return None

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        return True

    async def stream_content(self, *args, **kwargs):
        if False:
            yield b""


@pytest.mark.asyncio
async def test_object_card_exposes_locale_tabs_only_when_enabled(factory, monkeypatch):
    import admin_content_authoring as module
    monkeypatch.setattr(module, "async_session_maker", factory)
    monkeypatch.setattr(module, "is_admin", None, raising=False)
    async with factory() as session:
        session.add(Topic(id=17, name="Отношения"))
        await session.commit()
    recording = _RecordingSession()
    bot = Bot("123456:TEST", session=recording)
    user = User(id=11, is_bot=False, first_name="Admin")
    message = Message(message_id=1, date=__import__("datetime").datetime.now(__import__("datetime").timezone.utc), chat=Chat(id=11, type="private"), from_user=user, text="x").as_(bot)
    callback = __import__("types").SimpleNamespace(message=message, bot=bot, from_user=user, answer=__import__("unittest").mock.AsyncMock())
    await module.resource_card(callback, "topic", 17)
    first = recording.calls[-1]
    assert not any("English" in button.text for row in first.reply_markup.inline_keyboard for button in row)
    async with factory() as session:
        config = await session.get(BotGeneralConfig, 1)
        config.multilingual_authoring_enabled = True
        await session.commit()
    await module.resource_card(callback, "topic", 17)
    second = recording.calls[-1]
    callback_data = [button.callback_data for row in second.reply_markup.inline_keyboard for button in row]
    assert "ca:locale:topic:17:en" in callback_data
    assert "ca:locale:topic:17:pt" in callback_data
    async with factory() as session:
        session.add(Question(id=31, text="Вопрос", category="custom"))
        await session.commit()
    assert module.ENTRY_LISTS["admin_test_questions"] == "test_question"
    await module.resource_card(callback, "test_question", 31)
    question_card = recording.calls[-1]
    assert "Язык:" not in question_card.text
    assert not any(
        button.callback_data and button.callback_data.startswith("ca:locale:")
        for row in question_card.reply_markup.inline_keyboard
        for button in row
    )
