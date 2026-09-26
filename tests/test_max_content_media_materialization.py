from __future__ import annotations

import io
import importlib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.methods import AnswerCallbackQuery, DeleteMessage, EditMessageText, GetMe, SendMessage
from aiogram.types import Chat, Message, PhotoSize, Update, User as TelegramUser, Video
from sqlalchemy import delete, inspect, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database import Base, Content, ContentMedia, User as DBUser
from max_messenger_bot.content_media import ContentMediaMaterializationError, materialize_content_media
from max_messenger_bot.services import common as max_common
from max_messenger_bot.storage import MaxContentMedia, StorageBase


@pytest_asyncio.fixture
async def media_db(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(StorageBase.metadata.create_all)
    monkeypatch.setattr(max_common, "async_session_maker", sessions)
    yield sessions
    await engine.dispose()


@pytest_asyncio.fixture
async def telegram_media_db(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(StorageBase.metadata.create_all)
    import database
    import handlers
    import keyboards
    from max_messenger_bot import legacy as max_legacy
    from max_messenger_bot import storage as max_storage
    for module in (database, handlers, keyboards, max_legacy, max_storage, max_common):
        monkeypatch.setattr(module, "async_session_maker", sessions)
    async with sessions() as session:
        session.add(Content(key="about_me", button_title="Об авторе", text_content="Текст", is_visible=True))
        session.add(DBUser(id=11, first_name="Admin", is_admin=True))
        await session.commit()
    yield sessions
    await engine.dispose()


class ValidatingSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def close(self):
        return None

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        type(method).model_validate(method.model_dump())
        if isinstance(method, (EditMessageText, SendMessage)):
            return Message(
                message_id=getattr(method, "message_id", 1),
                date=datetime.now(timezone.utc),
                chat=Chat(id=int(method.chat_id), type="private"),
                from_user=TelegramUser(id=999, is_bot=True, first_name="TestBot", username="testbot"),
                text=getattr(method, "text", ""),
            ).as_(bot)
        if isinstance(method, GetMe):
            return TelegramUser(id=999, is_bot=True, first_name="TestBot", username="testbot").as_(bot)
        if isinstance(method, DeleteMessage):
            return True
        return True

    async def stream_content(self, *args, **kwargs):
        if False:
            yield b""


def _admin_message(bot, *, media=None, video=None, text=None):
    return Message(
        message_id=100,
        date=datetime.now(timezone.utc),
        chat=Chat(id=11, type="private"),
        from_user=TelegramUser(id=11, is_bot=False, first_name="Admin"),
        text=text,
        photo=media,
        video=video,
    ).as_(bot)


async def _feed(dispatcher, bot, update):
    return await dispatcher.feed_update(bot, update)


async def _feed_callback(dispatcher, bot, data):
    from aiogram.types import CallbackQuery

    message = _admin_message(bot)
    callback = CallbackQuery(
        id=f"cb-{data}",
        from_user=message.from_user,
        chat_instance="admin",
        message=message,
        data=data,
    ).as_(bot)
    return await _feed(dispatcher, bot, Update(update_id=hash(data) & 0x7FFFFFFF, callback_query=callback))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("media_type", "file_id"),
    (("photo", "dispatcher-photo"), ("video", "dispatcher-video")),
)
async def test_telegram_admin_media_save_materializes_max_artifact(telegram_media_db, monkeypatch, media_type, file_id):
    import handlers
    if handlers.router.parent_router is not None:
        handlers = importlib.reload(handlers)
    dispatcher = Dispatcher(storage=MemoryStorage())
    dispatcher.include_router(handlers.router)
    session = ValidatingSession()
    bot = Bot("123456:TEST", session=session)
    uploader = FakeMaxUploader()
    replacement_id = f"{file_id}-replacement"
    fake_telegram = FakeTelegramMedia({file_id: b"dispatcher-bytes", replacement_id: b"replacement-bytes"})

    async def fake_materialize(content_key, telegram_bot, *, session_factory=None):
        return await materialize_content_media(
            content_key,
            fake_telegram,
            uploader,
            session_factory=session_factory,
            force_refresh=True,
        )

    import max_messenger_bot.content_media as content_media
    monkeypatch.setattr(content_media, "materialize_content_media_from_telegram", fake_materialize)

    await _feed_callback(dispatcher, bot, "edit_content_about_me")
    incoming = _admin_message(
        bot,
        media=[PhotoSize(file_id=file_id, file_unique_id=f"unique-{file_id}", width=10, height=10)] if media_type == "photo" else None,
        video=Video(file_id=file_id, file_unique_id=f"unique-{file_id}", width=10, height=10, duration=1) if media_type == "video" else None,
    )
    await _feed(dispatcher, bot, Update(update_id=1000, message=incoming))
    await _feed_callback(dispatcher, bot, "save_content_about_me")

    async with telegram_media_db() as db:
        canonical = (await db.execute(select(ContentMedia).where(ContentMedia.content_key == "about_me"))).scalars().all()
        max_rows = (await db.execute(select(MaxContentMedia).where(MaxContentMedia.content_key == "about_me"))).scalars().all()
    assert [(item.file_type, item.file_id) for item in canonical] == [(media_type, file_id)]
    assert len(max_rows) == 1
    assert max_rows[0].media_type == ("image" if media_type == "photo" else "video")
    assert max_rows[0].source_media_id == canonical[0].id
    assert max_rows[0].source_file_id == file_id
    assert uploader.uploads == [("image" if media_type == "photo" else "video", b"dispatcher-bytes")]

    await _feed_callback(dispatcher, bot, "edit_content_about_me")
    await _feed_callback(dispatcher, bot, "delete_media_0")
    replacement = _admin_message(
        bot,
        media=[PhotoSize(file_id=replacement_id, file_unique_id=f"unique-{replacement_id}", width=10, height=10)] if media_type == "photo" else None,
        video=Video(file_id=replacement_id, file_unique_id=f"unique-{replacement_id}", width=10, height=10, duration=1) if media_type == "video" else None,
    )
    await _feed(dispatcher, bot, Update(update_id=2000, message=replacement))
    await _feed_callback(dispatcher, bot, "save_content_about_me")
    async with telegram_media_db() as db:
        replaced_rows = (await db.execute(select(MaxContentMedia).where(MaxContentMedia.content_key == "about_me"))).scalars().all()
    assert [row.token for row in replaced_rows] == ["max-token-2"]
    assert uploader.uploads[-1] == ("image" if media_type == "photo" else "video", b"replacement-bytes")

    await _feed_callback(dispatcher, bot, "edit_content_about_me")
    await _feed_callback(dispatcher, bot, "delete_media_0")
    await _feed_callback(dispatcher, bot, "save_content_about_me")
    async with telegram_media_db() as db:
        assert (await db.execute(select(MaxContentMedia).where(MaxContentMedia.content_key == "about_me"))).scalars().all() == []
    runtime_client = _runtime_client()
    assert await max_common.render_static_content(runtime_client, 100, 100, "about_me") is True
    assert all(not call.kwargs.get("attachments") or call.kwargs["attachments"][0].get("type") != ("image" if media_type == "photo" else "video") for call in runtime_client.send_message.await_args_list)


@pytest.mark.asyncio
async def test_telegram_media_failure_is_visible_and_preserves_canonical(telegram_media_db, monkeypatch):
    import handlers
    if handlers.router.parent_router is not None:
        handlers = importlib.reload(handlers)
    dispatcher = Dispatcher(storage=MemoryStorage())
    dispatcher.include_router(handlers.router)
    session = ValidatingSession()
    bot = Bot("123456:TEST", session=session)

    import max_messenger_bot.content_media as content_media

    async def fail_materialize(*_args, **_kwargs):
        raise ContentMediaMaterializationError("MAX unavailable")

    monkeypatch.setattr(content_media, "materialize_content_media_from_telegram", fail_materialize)
    await _feed_callback(dispatcher, bot, "edit_content_about_me")
    incoming = _admin_message(
        bot,
        media=[PhotoSize(file_id="failed-photo", file_unique_id="unique-failed-photo", width=10, height=10)],
    )
    await _feed(dispatcher, bot, Update(update_id=3000, message=incoming))
    await _feed_callback(dispatcher, bot, "save_content_about_me")

    async with telegram_media_db() as db:
        canonical = (await db.execute(select(ContentMedia).where(ContentMedia.content_key == "about_me"))).scalars().all()
        max_rows = (await db.execute(select(MaxContentMedia).where(MaxContentMedia.content_key == "about_me"))).scalars().all()
    assert [(item.file_type, item.file_id) for item in canonical] == [("photo", "failed-photo")]
    assert max_rows == []
    alerts = [call for call in session.calls if isinstance(call, AnswerCallbackQuery) and call.show_alert]
    assert alerts
    assert "Медиа MAX не обновлено" in alerts[-1].text


class FakeTelegramMedia:
    def __init__(self, payloads):
        self.payloads = payloads

    async def get_file(self, file_id):
        return SimpleNamespace(file_path=f"{file_id}.bin")

    async def download_file(self, file_path):
        return io.BytesIO(self.payloads[Path(file_path).stem])


class FakeMaxUploader:
    def __init__(self):
        self.uploads = []
        self.counter = 0

    async def upload_file(self, media_type, path):
        self.counter += 1
        self.uploads.append((media_type, Path(path).read_bytes()))
        return {"token": f"max-token-{self.counter}"}


class ValidatingMaxClient:
    def __init__(self):
        self.send_message = AsyncMock(side_effect=self._send_message)

    async def _send_message(self, **kwargs):
        assert isinstance(kwargs.get("text"), str)
        attachments = kwargs.get("attachments")
        if attachments is not None:
            assert isinstance(attachments, list)
            for attachment in attachments:
                assert isinstance(attachment, dict)
                attachment_type = attachment.get("type")
                assert attachment_type in {"image", "video", "audio", "file", "inline_keyboard"}
                if attachment_type in {"image", "video", "audio", "file"}:
                    payload = attachment.get("payload")
                    assert isinstance(payload, dict)
                    assert isinstance(payload.get("token"), str) and payload["token"]
                if attachment_type == "inline_keyboard":
                    payload = attachment.get("payload")
                    assert isinstance(payload, dict)
                    assert isinstance(payload.get("buttons"), list)
        return {"ok": True}


def _runtime_client():
    return ValidatingMaxClient()


async def _max_rows(factory, key):
    async with factory() as session:
        return (
            await session.execute(select(MaxContentMedia).where(MaxContentMedia.content_key == key).order_by(MaxContentMedia.id.asc()))
        ).scalars().all()


@pytest.mark.asyncio
async def test_veda_static_photo_materializes_and_renders_in_max(media_db):
    async with media_db() as session:
        session.add(Content(key="about_me", text_content="О методе\n[Подробнее](btn:svc:topics)", content_order="media_top", is_visible=True))
        session.add(ContentMedia(content_key="about_me", file_type="photo", file_id="veda-photo"))
        await session.commit()

    uploader = FakeMaxUploader()
    telegram = FakeTelegramMedia({"veda-photo": b"photo-bytes"})
    assert await materialize_content_media("about_me", telegram, uploader, session_factory=media_db, force_refresh=True) == 1
    rows = await _max_rows(media_db, "about_me")
    assert [row.token for row in rows] == ["max-token-1"]
    assert uploader.uploads == [("image", b"photo-bytes")]

    runtime_client = _runtime_client()
    rendered = await max_common.render_static_content(runtime_client, 100, 100, "about_me")
    assert rendered is True
    assert runtime_client.send_message.await_args_list[0].kwargs["attachments"] == [
        {"type": "image", "payload": {"token": "max-token-1"}}
    ]
    assert "О методе" in runtime_client.send_message.await_args_list[1].kwargs["text"]
    assert runtime_client.send_message.await_args_list[1].kwargs["attachments"]
    assert any(
        attachment["type"] == "inline_keyboard"
        for attachment in runtime_client.send_message.await_args_list[1].kwargs["attachments"]
    )


@pytest.mark.asyncio
async def test_failed_refresh_keeps_old_artifact_and_runtime_retries(media_db, monkeypatch):
    async with media_db() as session:
        session.add(Content(key="retry_me", text_content="Текст", is_visible=True))
        session.add(ContentMedia(content_key="retry_me", file_type="photo", file_id="photo-a"))
        await session.commit()

    telegram = FakeTelegramMedia({"photo-a": b"a", "photo-b": b"b"})
    uploader = FakeMaxUploader()
    await materialize_content_media("retry_me", telegram, uploader, session_factory=media_db, force_refresh=True)

    async with media_db() as session:
        row = await session.scalar(select(ContentMedia).where(ContentMedia.content_key == "retry_me"))
        row.file_id = "photo-b"
        await session.commit()

    class FailingUploader:
        async def upload_file(self, *_args):
            raise RuntimeError("upload failed")

    with pytest.raises(ContentMediaMaterializationError):
        await materialize_content_media("retry_me", telegram, FailingUploader(), session_factory=media_db, force_refresh=True)
    old_rows = await _max_rows(media_db, "retry_me")
    assert [row.token for row in old_rows] == ["max-token-1"]
    assert old_rows[0].source_media_id == 1
    assert old_rows[0].source_file_id == "photo-a"

    import max_messenger_bot.content_media as content_media

    async def recover(content_key, *, session_factory=None):
        return await materialize_content_media(
            content_key,
            telegram,
            uploader,
            session_factory=session_factory,
        )

    monkeypatch.setattr(content_media, "materialize_missing_content_media", recover)
    runtime_client = _runtime_client()
    assert await max_common.render_static_content(runtime_client, 100, 100, "retry_me") is True
    assert runtime_client.send_message.await_args_list[0].kwargs["attachments"] == [
        {"type": "image", "payload": {"token": "max-token-2"}}
    ]


@pytest.mark.asyncio
async def test_veda_method_video_materializes_and_renders_in_max(media_db):
    async with media_db() as session:
        session.add(Content(key="about", text_content="О методе", content_order="text_top", is_visible=True))
        session.add(ContentMedia(content_key="about", file_type="video", file_id="veda-video"))
        await session.commit()

    uploader = FakeMaxUploader()
    telegram = FakeTelegramMedia({"veda-video": b"video-bytes"})
    assert await materialize_content_media("about", telegram, uploader, session_factory=media_db, force_refresh=True) == 1
    assert uploader.uploads == [("video", b"video-bytes")]

    runtime_client = _runtime_client()
    assert await max_common.render_static_content(runtime_client, 100, 100, "about") is True
    calls = runtime_client.send_message.await_args_list
    assert "О методе" in calls[0].kwargs["text"]
    assert calls[0].kwargs["attachments"] is not None
    assert calls[-1].kwargs["attachments"][0]["type"] == "video"


@pytest.mark.asyncio
async def test_content_media_replace_delete_and_cached_recovery(media_db):
    async with media_db() as session:
        session.add(Content(key="replace_me", text_content="Текст", content_order="media_top", is_visible=True))
        session.add(ContentMedia(content_key="replace_me", file_type="photo", file_id="photo-a"))
        await session.commit()

    uploader = FakeMaxUploader()
    telegram = FakeTelegramMedia({"photo-a": b"a", "photo-b": b"b"})
    await materialize_content_media("replace_me", telegram, uploader, session_factory=media_db, force_refresh=True)
    assert [row.token for row in await _max_rows(media_db, "replace_me")] == ["max-token-1"]

    class NoUpload:
        async def upload_file(self, *_args):
            raise AssertionError("cached MAX artifact must be reused")

    assert await materialize_content_media("replace_me", telegram, NoUpload(), session_factory=media_db) == 1
    assert uploader.counter == 1

    async with media_db() as session:
        row = await session.get(ContentMedia, 1)
        row.file_id = "photo-b"
        await session.commit()
    await materialize_content_media("replace_me", telegram, uploader, session_factory=media_db, force_refresh=True)
    assert [row.token for row in await _max_rows(media_db, "replace_me")] == ["max-token-2"]
    replacement_client = _runtime_client()
    assert await max_common.render_static_content(replacement_client, 100, 100, "replace_me") is True
    assert replacement_client.send_message.await_args_list[0].kwargs["attachments"] == [
        {"type": "image", "payload": {"token": "max-token-2"}}
    ]

    async with media_db() as session:
        await session.execute(delete(ContentMedia).where(ContentMedia.content_key == "replace_me"))
        await session.commit()
    assert await materialize_content_media("replace_me", telegram, uploader, session_factory=media_db, force_refresh=True) == 0
    assert await _max_rows(media_db, "replace_me") == []
    deleted_client = _runtime_client()
    assert await max_common.render_static_content(deleted_client, 100, 100, "replace_me") is True
    assert all(not call.kwargs.get("attachments") or call.kwargs["attachments"][0].get("type") != "image" for call in deleted_client.send_message.await_args_list)


@pytest.mark.asyncio
async def test_missing_max_artifact_is_materialized_once_by_runtime(media_db, monkeypatch):
    async with media_db() as session:
        session.add(Content(key="recover_me", text_content="Текст", content_order="media_top", is_visible=True))
        session.add(ContentMedia(content_key="recover_me", file_type="photo", file_id="recover-photo"))
        await session.commit()

    uploader = FakeMaxUploader()
    telegram = FakeTelegramMedia({"recover-photo": b"recover-bytes"})
    import max_messenger_bot.content_media as content_media
    calls = 0

    async def recover(content_key, *, session_factory=None):
        nonlocal calls
        calls += 1
        return await materialize_content_media(
            content_key,
            telegram,
            uploader,
            session_factory=session_factory,
        )

    monkeypatch.setattr(content_media, "materialize_missing_content_media", recover)
    first = _runtime_client()
    assert await max_common.render_static_content(first, 100, 100, "recover_me") is True
    assert calls == 1
    assert uploader.counter == 1
    assert first.send_message.await_args_list[0].kwargs["attachments"] == [
        {"type": "image", "payload": {"token": "max-token-1"}}
    ]

    second = _runtime_client()
    assert await max_common.render_static_content(second, 100, 100, "recover_me") is True
    assert calls == 1
    assert uploader.counter == 1


@pytest.mark.asyncio
async def test_put_k_sebe_existing_max_media_token_remains_usable(media_db):
    async with media_db() as session:
        session.add(Content(key="put_k_sebe", text_content="Готово", content_order="media_top", is_visible=True))
        session.add(ContentMedia(content_key="put_k_sebe", file_type="photo", file_id="legacy-photo"))
        session.add(MaxContentMedia(content_key="put_k_sebe", media_type="photo", token="existing-token"))
        await session.commit()

    runtime_client = _runtime_client()
    assert await max_common.render_static_content(runtime_client, 100, 100, "put_k_sebe") is True
    assert runtime_client.send_message.await_args_list[0].kwargs["attachments"] == [
        {"type": "image", "payload": {"token": "existing-token"}}
    ]
    async with media_db() as session:
        bound = await session.scalar(select(MaxContentMedia).where(MaxContentMedia.content_key == "put_k_sebe"))
    assert bound.source_media_id == 1
    assert bound.source_file_id == "legacy-photo"


@pytest.mark.asyncio
async def test_max_storage_migration_adds_canonical_source_columns(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "CREATE TABLE max_content_media ("
                "id INTEGER PRIMARY KEY, content_key VARCHAR(255) NOT NULL, "
                "media_type VARCHAR(32) NOT NULL, token VARCHAR(512) NOT NULL, "
                "description TEXT, created_at DATETIME NOT NULL)"
            )
        )
    import max_messenger_bot.legacy as legacy
    from max_messenger_bot.storage import init_storage

    monkeypatch.setattr(legacy, "engine", engine)
    await init_storage()
    async with engine.begin() as connection:
        columns = await connection.run_sync(
            lambda sync_connection: {item["name"] for item in inspect(sync_connection).get_columns("max_content_media")}
        )
    assert {"source_media_id", "source_file_id"}.issubset(columns)
    await engine.dispose()
