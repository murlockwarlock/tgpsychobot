from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import database
from database import Base, MediaCollection, MediaLibrary, Topic, TopicMediaDeck, media_collection_items, topic_collection_association
from max_messenger_bot import app as max_app
from max_messenger_bot.models import IncomingCallback, Sender
from max_messenger_bot.services import admin_collections as collection_service
from max_messenger_bot.services import admin_topic_media as service
from media_scope import load_available_media, load_media_scope


@pytest_asyncio.fixture
async def max_store(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'max-media-admin.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(service, "async_session_maker", sessions)
    monkeypatch.setattr(collection_service, "async_session_maker", sessions)
    try:
        yield sessions
    finally:
        await engine.dispose()


class FakeState:
    def __init__(self):
        self.state = None
        self.data = {}

    async def get(self, user_id):
        if self.state is None:
            return None
        return SimpleNamespace(state=self.state, data=dict(self.data))

    async def set(self, user_id, chat_id, state, data=None):
        self.state = state
        self.data = dict(data or {})

    async def clear(self, user_id):
        self.state = None
        self.data.clear()


class FakeClient:
    def __init__(self):
        self.messages = []

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)

    async def answer_callback(self, *args, **kwargs):
        return None


def _buttons(attachments):
    return [
        button
        for row in attachments[0]["payload"]["buttons"]
        for button in row
    ]


async def _create_topic_collection(sessions, topic_id=1, collection_name="Cards"):
    async with sessions() as session:
        topic = Topic(id=topic_id, name=f"Topic {topic_id}")
        collection = MediaCollection(name=collection_name)
        session.add_all([topic, collection])
        await session.flush()
        await session.execute(
            topic_collection_association.insert().values(
                topic_id=topic.id,
                collection_id=collection.id,
            )
        )
        await session.commit()
        return collection.id


@pytest.mark.asyncio
async def test_max_listing_after_legacy_migration_uses_canonical_scope(max_store):
    async with max_store() as session:
        session.add_all([
            Topic(id=1, name="Topic one"),
            TopicMediaDeck(topic_id=1, deck_name="tarot"),
            MediaLibrary(file_id="direct", file_name="direct", media_type="photo", topic_id=1),
            MediaLibrary(file_id="deck", file_name="deck", media_type="photo", category="tarot"),
        ])
        await session.commit()

    async with max_store() as session:
        await session.run_sync(database._migrate_legacy_media_ownership)
        await session.commit()

    client = FakeClient()
    await service.show_list(client, 100, 1)
    assert "Файлов: 2" in client.messages[-1]["text"]
    assert {button["text"].split(" ", 1)[-1] for button in _buttons(client.messages[-1]["attachments"]) if button["payload"].startswith("admin_media_view_")} == {"direct", "deck"}


@pytest.mark.asyncio
async def test_max_add_media_is_canonical_and_visible_without_remigration(max_store):
    collection_id = await _create_topic_collection(max_store)
    client = FakeClient()
    state = FakeState()

    await service.start_add_media(client, state, 100, 7, 1, page=2)
    assert state.state == "admin_media_add_file"
    assert state.data == {"topic_id": 1, "page": 2, "collection_id": collection_id}
    await service.receive_add_file(client, state, 100, 7, media_token="max-photo", media_type="photo")
    await service.save_add_name(client, state, 100, 7, "new_card")
    await service.save_add_category(client, state, 100, 7, "tarot")
    await service.save_add_description(client, state, 100, 7, "New card")

    assert state.state is None
    async with max_store() as session:
        media = await session.scalar(select(MediaLibrary).where(MediaLibrary.file_name == "new_card"))
        assert media is not None
        assert media.topic_id is None
        assert await session.scalar(
            select(media_collection_items.c.media_id).where(
                media_collection_items.c.collection_id == collection_id,
                media_collection_items.c.media_id == media.id,
            )
        ) == media.id
        scope, available = await load_available_media(session, 1)

    assert media.id in scope.collection_media_ids
    assert [item.file_name for item in available] == ["new_card"]
    assert "Файлов: 1" in client.messages[-1]["text"]


@pytest.mark.asyncio
async def test_max_add_with_multiple_collections_requires_and_preserves_collection_context(max_store):
    first_collection = await _create_topic_collection(max_store, collection_name="First")
    async with max_store() as session:
        second = MediaCollection(name="Second")
        session.add(second)
        await session.flush()
        await session.execute(
            topic_collection_association.insert().values(topic_id=1, collection_id=second.id)
        )
        await session.commit()
        second_collection = second.id

    client = FakeClient()
    state = FakeState()
    await service.start_add_media(client, state, 100, 7, 1, page=4)
    assert state.state is None
    selector_buttons = _buttons(client.messages[-1]["attachments"])
    collection_payload = next(
        button["payload"]
        for button in selector_buttons
        if button["payload"].endswith(f"_4_{second_collection}")
    )
    assert collection_payload == f"admin_media_add_collection_1_4_{second_collection}"
    await service.start_add_media(client, state, 100, 7, 1, page=4, collection_id=second_collection)
    assert state.state == "admin_media_add_file"
    assert state.data["collection_id"] == second_collection
    assert state.data["page"] == 4


@pytest.mark.asyncio
async def test_max_shared_media_edit_and_scoped_membership_never_delete_globally(max_store):
    first_collection = await _create_topic_collection(max_store, topic_id=1, collection_name="First")
    async with max_store() as session:
        topic_two = Topic(id=2, name="Topic 2")
        second_collection = MediaCollection(name="Second")
        media = MediaLibrary(file_id="shared", file_name="shared", media_type="photo")
        session.add_all([topic_two, second_collection, media])
        await session.flush()
        await session.execute(
            topic_collection_association.insert(),
            [{"topic_id": 2, "collection_id": second_collection.id}],
        )
        await session.execute(
            media_collection_items.insert(),
            [
                {"collection_id": first_collection, "media_id": media.id},
                {"collection_id": second_collection.id, "media_id": media.id},
            ],
        )
        await session.commit()
        media_id = media.id

    client = FakeClient()
    state = FakeState()
    await service.show_list(client, 100, 1, page=3)
    view_button = next(
        button for button in _buttons(client.messages[-1]["attachments"])
        if button["payload"].startswith("admin_media_view_")
    )
    assert view_button["payload"] == f"admin_media_view_1_0_{media_id}"

    await service.show_media_detail(client, 100, media_id, topic_id=1, page=3)
    detail_buttons = _buttons(client.messages[-1]["attachments"])
    edit_button = next(button for button in detail_buttons if button["payload"].startswith("admin_media_editname_"))
    assert edit_button["payload"] == f"admin_media_editname_1_3_{media_id}"
    assert not any(button["payload"].startswith("admin_media_delete_") for button in detail_buttons)
    await service.start_edit_name(client, state, 100, 7, media_id, topic_id=1, page=3)
    await service.save_edit_name(client, state, 100, 7, "renamed")

    assert state.state is None
    assert client.messages[-1]["attachments"][0]["payload"]["buttons"][-1][0]["payload"] == "admin_topic_media_1_3"
    async with max_store() as session:
        updated = await session.get(MediaLibrary, media_id)
        assert updated.file_name == "renamed"
        assert updated.topic_id is None
        assert (await session.execute(select(TopicMediaDeck))).scalars().all() == []

    await service.delete_media(client, 100, media_id, topic_id=1, page=3)
    async with max_store() as session:
        assert await session.get(MediaLibrary, media_id) is not None
        assert (await session.execute(
            select(media_collection_items.c.collection_id)
            .where(media_collection_items.c.media_id == media_id)
            .order_by(media_collection_items.c.collection_id)
        )).scalars().all() == [first_collection, second_collection.id]
        assert (await session.execute(select(TopicMediaDeck))).scalars().all() == []

    await collection_service.toggle_file(client, 100, "remove", first_collection, media_id, 0)
    async with max_store() as session:
        assert await session.get(MediaLibrary, media_id) is not None
        remaining_collections = (await session.execute(
            select(media_collection_items.c.collection_id)
            .where(media_collection_items.c.media_id == media_id)
            .order_by(media_collection_items.c.collection_id)
        )).scalars().all()
        assert remaining_collections == [second_collection.id]
        first_scope = await load_media_scope(session, 1)
        second_scope = await load_media_scope(session, 2)
        assert media_id not in first_scope.collection_media_ids
        assert media_id in second_scope.collection_media_ids


@pytest.mark.asyncio
async def test_max_listing_paginates_in_sql_and_summarizes_all_categories(max_store):
    collection_id = await _create_topic_collection(max_store)
    async with max_store() as session:
        media_rows = [
            MediaLibrary(
                file_id=f"file-{index}",
                file_name=f"card-{index}",
                category="first" if index < 10 else "second",
                media_type="photo",
            )
            for index in range(25)
        ]
        media_rows[14].file_name = "_back"
        session.add_all(media_rows)
        await session.flush()
        await session.execute(
            media_collection_items.insert(),
            [{"collection_id": collection_id, "media_id": media.id} for media in media_rows],
        )
        await session.commit()

    client = FakeClient()
    await service.show_list(client, 100, 1, page=1)
    message = client.messages[-1]
    buttons = _buttons(message["attachments"])
    page_media = [
        button["text"]
        for button in buttons
        if button["payload"].startswith("admin_media_view_")
    ]
    assert len(page_media) == 10
    expected_names = ["_back" if index == 14 else f"card-{index}" for index in range(10, 20)]
    assert all(expected in actual for expected, actual in zip(expected_names, page_media))
    assert "Файлов: 25" in message["text"]
    assert "<code>first</code> — ⚠️ нет рубашки" in message["text"]
    assert "<code>second</code> — 🃏" in message["text"]


@pytest.mark.asyncio
async def test_max_topic_scope_does_not_show_foreign_collection_media(max_store):
    await _create_topic_collection(max_store, topic_id=1, collection_name="Topic one")
    async with max_store() as session:
        topic = Topic(id=2, name="Topic two")
        collection = MediaCollection(name="Topic two collection")
        media = MediaLibrary(file_id="foreign", file_name="foreign", media_type="photo")
        session.add_all([topic, collection, media])
        await session.flush()
        await session.execute(topic_collection_association.insert().values(topic_id=2, collection_id=collection.id))
        await session.execute(media_collection_items.insert().values(collection_id=collection.id, media_id=media.id))
        await session.commit()
        scope = await load_media_scope(session, 1)
        assert media.id not in scope.collection_media_ids

    client = FakeClient()
    await service.show_media_detail(client, 100, media.id, topic_id=1)
    assert "не найден в медиатеке этой темы" in client.messages[-1]["text"]


@pytest.mark.asyncio
async def test_max_media_callbacks_preserve_context_and_clear_state_on_back(max_store, monkeypatch):
    collection_id = await _create_topic_collection(max_store)
    async with max_store() as session:
        media = MediaLibrary(file_id="photo", file_name="card", media_type="photo")
        session.add(media)
        await session.flush()
        await session.execute(
            media_collection_items.insert().values(collection_id=collection_id, media_id=media.id)
        )
        await session.commit()

    client = FakeClient()
    state = FakeState()
    application = max_app.MaxBotApplication(client)
    application.states = state
    monkeypatch.setattr(max_app.common, "is_admin", AsyncMock(return_value=True))

    def callback(payload):
        return IncomingCallback(
            raw={},
            callback_id="callback",
            payload=payload,
            chat_id=100,
            message_id=None,
            sender=Sender(user_id=7, username=None, first_name=None, last_name=None),
        )

    await application.handle_callback(callback("admin_topic_media_1_0"))
    list_buttons = _buttons(client.messages[-1]["attachments"])
    view_payload = next(button["payload"] for button in list_buttons if button["payload"].startswith("admin_media_view_"))
    await application.handle_callback(callback(view_payload))
    edit_payload = next(
        button["payload"]
        for button in _buttons(client.messages[-1]["attachments"])
        if button["payload"].startswith("admin_media_editname_")
    )
    await application.handle_callback(callback(edit_payload))
    assert state.state == "admin_media_edit_name"
    cancel_payload = _buttons(client.messages[-1]["attachments"])[0]["payload"]
    assert cancel_payload == "admin_topic_media_1_0"
    await application.handle_callback(callback(cancel_payload))
    assert state.state is None
    assert "Файлов: 1" in client.messages[-1]["text"]

    add_payload = next(
        button["payload"]
        for button in _buttons(client.messages[-1]["attachments"])
        if button["payload"].startswith("admin_media_add_")
    )
    await application.handle_callback(callback(add_payload))
    assert state.state == "admin_media_add_file"
    await application.handle_callback(callback(_buttons(client.messages[-1]["attachments"])[0]["payload"]))
    assert state.state is None
    assert "Файлов: 1" in client.messages[-1]["text"]
