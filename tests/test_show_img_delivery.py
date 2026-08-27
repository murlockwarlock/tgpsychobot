from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiogram.types import InlineKeyboardMarkup
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import handlers
from database import (
    Base,
    MediaCollection,
    MediaLibrary,
    Topic,
    TopicMediaDeck,
    User,
    main_dialogue_collection_association,
    media_collection_items,
    topic_collection_association,
)
from media_scope import load_topic_media_scope, make_topic_media_scope
from response_buttons import extract_response_buttons


@pytest_asyncio.fixture
async def media_store(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'show-img.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(handlers, "async_session_maker", sessions)
    try:
        yield sessions
    finally:
        await engine.dispose()


class _Bot:
    def __init__(self):
        self.events = []

    async def send_message(self, **kwargs):
        self.events.append(("message", kwargs))

    async def send_photo(self, **kwargs):
        self.events.append(("photo", kwargs))

    async def send_document(self, **kwargs):
        self.events.append(("document", kwargs))

    async def send_audio(self, **kwargs):
        self.events.append(("audio", kwargs))

    async def send_chat_action(self, **kwargs):
        return None


class _BrokenBot:
    async def send_photo(self, **kwargs):
        raise RuntimeError("wrong file identifier for this bot")

    async def send_document(self, **kwargs):
        raise RuntimeError("wrong file identifier for this bot")


class _CommandMessage:
    def __init__(self):
        self.chat = SimpleNamespace(id=42)
        self.events = []

    async def answer(self, text, **kwargs):
        self.events.append((text, kwargs))


@pytest.mark.asyncio
async def test_show_img_parser_removes_tag_and_preserves_other_directives():
    clean, audios, random_images, choices, hidden_choices, show_images = await handlers.handle_ai_media_content(
        None,
        42,
        "Текст\n[SHOW_IMG: photo_AQADOiNrG-dgiUh-]\n[SEND_AUDIO: relax]\n[RANDOM_IMG: tarot]",
    )

    assert "[SHOW_IMG:" not in clean
    assert "Текст" in clean
    assert audios == ["relax"]
    assert random_images == [("tarot", "")]
    assert choices == []
    assert hidden_choices == []
    assert show_images == ["photo_AQADOiNrG-dgiUh-"]


@pytest.mark.asyncio
async def test_show_img_resolution_uses_topic_and_collection_scope(media_store):
    async with media_store() as session:
        topic = Topic(id=1, name="Topic one")
        other_topic = Topic(id=2, name="Topic two")
        collection = MediaCollection(name="Shared cards")
        direct = MediaLibrary(
            topic_id=1,
            file_id="direct-file",
            file_name="direct-photo",
            category="cards",
            media_type="photo",
        )
        legacy_image = MediaLibrary(
            topic_id=None,
            file_id="legacy-image-file",
            file_name="legacy-image",
            category="cards",
            media_type="image",
        )
        foreign = MediaLibrary(
            topic_id=2,
            file_id="foreign-file",
            file_name="foreign-photo",
            category="cards",
            media_type="photo",
        )
        other_collection = MediaCollection(name="Other cards")
        session.add_all([topic, other_topic, collection, other_collection, direct, legacy_image, foreign])
        await session.flush()
        await session.execute(
            topic_collection_association.insert().values(topic_id=1, collection_id=collection.id)
        )
        await session.execute(
            media_collection_items.insert().values(collection_id=collection.id, media_id=legacy_image.id)
        )
        await session.execute(
            topic_collection_association.insert().values(topic_id=2, collection_id=other_collection.id)
        )
        await session.execute(
            media_collection_items.insert().values(collection_id=other_collection.id, media_id=foreign.id)
        )
        await session.commit()

        scope = await load_topic_media_scope(session, 1)

        assert await handlers.resolve_show_image(session, scope, "direct-photo") is None
        assert (await handlers.resolve_show_image(session, scope, "legacy-image")).file_id == "legacy-image-file"
        assert await handlers.resolve_show_image(session, scope, "foreign-photo") is None


@pytest.mark.asyncio
async def test_show_img_resolution_does_not_use_legacy_deck_scope(media_store):
    async with media_store() as session:
        session.add_all([
            Topic(id=1, name="Topic one"),
            Topic(id=2, name="Topic two"),
            TopicMediaDeck(topic_id=1, deck_name="cards"),
            MediaLibrary(
                topic_id=2,
                file_id="deck-file",
                file_name="deck-photo",
                category="cards",
                media_type="photo",
            ),
            MediaLibrary(
                topic_id=2,
                file_id="other-file",
                file_name="other-photo",
                category="other",
                media_type="photo",
            ),
        ])
        await session.commit()

        scope = await load_topic_media_scope(session, 1)

        assert await handlers.resolve_show_image(session, scope, "deck-photo") is None
        assert await handlers.resolve_show_image(session, scope, "other-photo") is None


@pytest.mark.asyncio
async def test_text_buttons_and_show_img_send_in_order_with_one_image(media_store):
    async with media_store() as session:
        collection = MediaCollection(name="Cards")
        session.add_all([Topic(id=1, name="Topic one"), collection])
        media = MediaLibrary(
            topic_id=1,
            file_id="photo-file",
            file_name="show-photo",
            category="cards",
            media_type="photo",
            description="Описание",
        )
        session.add(media)
        await session.flush()
        await session.execute(topic_collection_association.insert().values(topic_id=1, collection_id=collection.id))
        await session.execute(media_collection_items.insert().values(collection_id=collection.id, media_id=media.id))
        await session.commit()

        clean, _, _, _, _, show_images = await handlers.handle_ai_media_content(
            None,
            42,
            "Ответ\n[Продолжить](btn:next)\n[SHOW_IMG: show-photo]",
        )
        clean, button_rows = extract_response_buttons(clean)
        markup = handlers._telegram_response_buttons_markup(button_rows)
        bot = _Bot()
        await bot.send_message(chat_id=42, text=clean, reply_markup=markup)
        sent_count = await handlers.send_show_images(
            bot,
            42,
            session,
            await load_topic_media_scope(session, 1),
            show_images,
        )

        assert sent_count == 1
        assert [event[0] for event in bot.events] == ["message", "photo"]
        assert "[SHOW_IMG:" not in bot.events[0][1]["text"]
        assert isinstance(bot.events[0][1]["reply_markup"], InlineKeyboardMarkup)
        assert bot.events[1][1]["photo"] == "photo-file"


@pytest.mark.asyncio
async def test_process_buffered_handler_keeps_buttons_and_sends_show_img_once(media_store, monkeypatch):
    async with media_store() as session:
        topic = Topic(id=1, name="Topic one")
        collection = MediaCollection(name="Cards")
        media = MediaLibrary(
            file_id="photo-file",
            file_name="show-photo",
            category="cards",
            media_type="photo",
        )
        session.add_all([User(id=42, current_topic_id=1), topic, collection, media])
        await session.flush()
        await session.execute(topic_collection_association.insert().values(topic_id=1, collection_id=collection.id))
        await session.execute(media_collection_items.insert().values(collection_id=collection.id, media_id=media.id))
        await session.commit()

    monkeypatch.setattr(
        handlers.ai_integration,
        "generate_response",
        AsyncMock(return_value="Ответ\n[Продолжить](btn:wish_next)\n[SHOW_IMG: show-photo]"),
    )
    bot = _Bot()
    handlers.user_message_buffers[42] = ["Покажи карту"]
    await handlers.process_buffered_messages(42, bot)

    assert [event[0] for event in bot.events] == ["message", "photo"]
    assert "[SHOW_IMG:" not in bot.events[0][1]["text"]
    assert isinstance(bot.events[0][1]["reply_markup"], InlineKeyboardMarkup)
    assert bot.events[1][1]["photo"] == "photo-file"


@pytest.mark.asyncio
async def test_duplicate_show_img_tags_emit_one_image(media_store):
    async with media_store() as session:
        collection = MediaCollection(name="Cards")
        session.add_all([Topic(id=1, name="Topic one"), collection])
        media = MediaLibrary(
            topic_id=1,
            file_id="photo-file",
            file_name="show-photo",
            category="cards",
            media_type="photo",
        )
        session.add(media)
        await session.flush()
        await session.execute(topic_collection_association.insert().values(topic_id=1, collection_id=collection.id))
        await session.execute(media_collection_items.insert().values(collection_id=collection.id, media_id=media.id))
        await session.commit()
        bot = _Bot()

        sent_count = await handlers.send_show_images(
            bot,
            42,
            session,
            await load_topic_media_scope(session, 1),
            ["show-photo", "show-photo"],
        )

        assert sent_count == 1
        assert [event[0] for event in bot.events] == ["photo"]


@pytest.mark.asyncio
async def test_random_choice_and_audio_use_collection_scope(media_store):
    async with media_store() as session:
        topic = Topic(id=1, name="Topic one")
        collection = MediaCollection(name="Cards")
        media = [
            MediaLibrary(file_id="audio-file", file_name="relax", media_type="audio"),
            MediaLibrary(file_id="card-file", file_name="card", category="cards", media_type="photo"),
        ]
        session.add_all([User(id=42, current_topic_id=1), topic, collection, *media])
        await session.flush()
        await session.execute(topic_collection_association.insert().values(topic_id=1, collection_id=collection.id))
        await session.execute(
            media_collection_items.insert(),
            [{"collection_id": collection.id, "media_id": item.id} for item in media],
        )
        await session.commit()

    bot = _Bot()
    message = _CommandMessage()
    clean = await handlers.execute_media_commands(
        message,
        "[SEND_AUDIO: relax]\n[RANDOM_IMG: cards]\n[CHOICE_IMG: cards | 1]",
        42,
        bot,
    )

    assert clean == ""
    assert [event[0] for event in bot.events] == ["audio", "photo", "photo"]
    assert len(message.events) == 1


async def _seed_spread_scope(sessions, *, topic_id: int | None, include_back: bool = False):
    async with sessions() as session:
        collection = MediaCollection(name=f"Spread {topic_id or 'main'}")
        user = User(id=42, current_topic_id=topic_id)
        session.add(collection)
        if topic_id is not None:
            session.add(Topic(id=topic_id, name=f"Topic {topic_id}"))
        session.add(user)
        cards = [
            MediaLibrary(file_id=f"card-{index}", file_name=f"card-{index}", category="cards", media_type="photo")
            for index in range(2)
        ]
        if include_back:
            cards.append(MediaLibrary(file_id="back", file_name="_back", category="cards", media_type="photo"))
        session.add_all(cards)
        await session.flush()
        if topic_id is None:
            await session.execute(main_dialogue_collection_association.insert().values(collection_id=collection.id))
        else:
            await session.execute(
                topic_collection_association.insert().values(topic_id=topic_id, collection_id=collection.id)
            )
        await session.execute(
            media_collection_items.insert(),
            [{"collection_id": collection.id, "media_id": media.id} for media in cards],
        )
        await session.commit()


@pytest.mark.asyncio
async def test_main_dialogue_visible_spread_accepts_none_and_survives_reload(media_store):
    handlers.user_spread_state.clear()
    await _seed_spread_scope(media_store, topic_id=None)
    bot = _Bot()

    await handlers.execute_media_commands(
        _CommandMessage(),
        "[CHOICE_IMG: cards | 1 | 2]",
        42,
        bot,
    )
    spread = await handlers._get_card_spread_state(42)
    assert spread["topic_id"] is None

    handlers.user_spread_state.clear()
    reloaded = await handlers._get_card_spread_state(42)
    assert reloaded["topic_id"] is None
    first_card_id = reloaded["pending_card_ids"][0]
    await handlers._advance_card_spread_after_selection(
        bot=bot,
        user_id=42,
        card_id=first_card_id,
        selected_file_id="card-0",
    )
    next_spread = await handlers._get_card_spread_state(42)
    assert next_spread["topic_id"] is None
    assert next_spread["pending_card_ids"]
    assert await handlers._resend_active_spread_choice(bot, 42)
    await handlers._clear_card_spread_state(42)


@pytest.mark.asyncio
async def test_main_dialogue_hidden_spread_uses_none_for_next_round(media_store):
    handlers.user_spread_state.clear()
    await _seed_spread_scope(media_store, topic_id=None, include_back=True)
    bot = _Bot()

    await handlers.execute_media_commands(
        _CommandMessage(),
        "[CHOICE_IMG_HIDDEN: cards | 1 | 2]",
        42,
        bot,
    )
    spread = await handlers._get_card_spread_state(42)
    assert spread["topic_id"] is None
    await handlers._advance_card_spread_after_selection(
        bot=bot,
        user_id=42,
        card_id=spread["pending_card_ids"][0],
        selected_file_id="card-0",
    )
    next_spread = await handlers._get_card_spread_state(42)
    assert next_spread["topic_id"] is None
    assert any(event[0] == "photo" and event[1]["photo"] == "back" for event in bot.events)
    await handlers._clear_card_spread_state(42)


@pytest.mark.asyncio
async def test_topic_scoped_spread_keeps_topic_id(media_store):
    handlers.user_spread_state.clear()
    await _seed_spread_scope(media_store, topic_id=7)
    bot = _Bot()

    await handlers.execute_media_commands(
        _CommandMessage(),
        "[CHOICE_IMG: cards | 1 | 2]",
        42,
        bot,
    )
    spread = await handlers._get_card_spread_state(42)
    assert spread["topic_id"] == 7
    handlers.user_spread_state.clear()
    assert (await handlers._get_card_spread_state(42))["topic_id"] == 7
    await handlers._clear_card_spread_state(42)


@pytest.mark.asyncio
async def test_unavailable_and_wrong_bot_media_fail_safely(media_store, caplog):
    async with media_store() as session:
        session.add(Topic(id=1, name="Topic one"))
        session.add(MediaLibrary(
            topic_id=1,
            file_id="video-file",
            file_name="video-name",
            category="cards",
            media_type="video",
        ))
        await session.commit()
        scope = make_topic_media_scope(1)

        with caplog.at_level(logging.WARNING):
            assert await handlers.resolve_show_image(session, scope, "missing-name") is None
            assert await handlers.resolve_show_image(session, scope, "video-name") is None
            delivered = await handlers.send_photo_or_document(
                _BrokenBot(),
                42,
                "bot-specific-file-id",
                context="SHOW_IMG",
            )

    assert delivered is False
    assert "reason=not_found" in caplog.text
    assert "reason=unsupported_media_type:video" in caplog.text
    assert "Media delivery failed context=SHOW_IMG" in caplog.text
    assert "wrong file identifier" in caplog.text
    assert "bot-specific-file-id" not in caplog.text
