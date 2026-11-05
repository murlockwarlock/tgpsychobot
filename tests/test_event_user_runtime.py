import html
import importlib
import json
import os
from datetime import datetime, timedelta, timezone

os.environ.setdefault("BOT_TOKEN", "test_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.methods import EditMessageText, SendMessage
from aiogram.types import CallbackQuery, Chat, Message, Update, User as TelegramUser
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from unittest.mock import AsyncMock

from database import Base, EventRegistration, EventScenario, User
from max_messenger_bot.api import MaxApiClient
from max_messenger_bot.app import MaxBotApplication
from max_messenger_bot.formatting import MAX_MESSAGE_TEXT_LEN
from max_messenger_bot.models import MAX_ID_OFFSET


@pytest_asyncio.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    await engine.dispose()


class _ValidatedTelegramSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls = []
        self._next_message_id = 1

    async def close(self):
        return None

    async def make_request(self, bot, method, timeout=None):
        type(method).model_validate(method.model_dump())
        if isinstance(method, (SendMessage, EditMessageText)) and method.text is not None:
            assert len(method.text.encode("utf-16-le")) // 2 <= 4096
        self.calls.append(method)
        if isinstance(method, (SendMessage, EditMessageText)):
            message_id = getattr(method, "message_id", None) or self._next_message_id
            self._next_message_id += 1
            return Message(
                message_id=int(message_id),
                date=datetime.now(timezone.utc),
                chat=Chat(id=int(method.chat_id), type="private"),
                from_user=TelegramUser(id=999, is_bot=True, first_name="Test bot"),
                text=method.text,
                reply_markup=method.reply_markup,
            ).as_(bot)
        return True

    async def stream_content(self, *args, **kwargs):
        if False:
            yield b""


class _RecordedMaxResponse:
    status = 200
    content_type = "application/json"

    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def text(self):
        return json.dumps(self.payload, ensure_ascii=False)

    async def json(self):
        return self.payload


class _RecordedMaxSession:
    def __init__(self):
        self.requests = []

    def request(self, method, url, **kwargs):
        body = kwargs.get("json")
        if body is not None:
            body = json.loads(json.dumps(body, ensure_ascii=False))
        self.requests.append(
            {
                "method": method,
                "url": url,
                "params": dict(kwargs.get("params") or {}),
                "body": body,
            }
        )
        return _RecordedMaxResponse({"message": {"body": {"mid": f"max-runtime-{len(self.requests)}"}}})


def _long_title() -> str:
    prefix = '<&>"«'
    title = prefix + "Я" * (1000 - len(prefix))
    assert len(title) == 1000
    return title


async def _create_event(
    session_factory,
    *,
    code,
    title,
    starts_at,
    user_id=None,
    link="https://example.test/event",
):
    async with session_factory() as session:
        if user_id is not None:
            session.add(User(id=user_id, first_name="Пользователь", name="Пользователь", gender="other"))
        scenario = EventScenario(
            code=code,
            event_type="webinar",
            title=title,
            starts_at=starts_at,
            duration_minutes=120,
            link=link,
            is_active=True,
            is_archived=False,
        )
        session.add(scenario)
        await session.flush()
        event_id = scenario.id
        await session.commit()
    return event_id


def _telegram_callback_data(method):
    for row in method.reply_markup.inline_keyboard:
        for button in row:
            if button.callback_data:
                return button.callback_data
    raise AssertionError("Telegram event message has no callback button")


def _max_callback_data(body):
    for attachment in body.get("attachments") or []:
        for row in attachment.get("payload", {}).get("buttons", []):
            for button in row:
                if button.get("payload"):
                    return button["payload"]
    raise AssertionError("MAX event message has no callback button")


def _assert_full_escaped_title(text, title):
    assert html.escape(title, quote=False) in text
    assert title in html.unescape(text)
    assert "&amp;lt;" not in text


@pytest.mark.asyncio
async def test_telegram_user_runtime_uses_dispatcher_and_escapes_long_title(
    session_factory,
    monkeypatch,
):
    import database
    import handlers

    if handlers.router.parent_router is not None:
        handlers = importlib.reload(handlers)
    monkeypatch.setattr(database, "async_session_maker", session_factory)
    monkeypatch.setattr(handlers, "async_session_maker", session_factory)
    monkeypatch.setattr(handlers, "_sync_user_birthdate_from_telegram", AsyncMock())
    monkeypatch.setattr(handlers, "refresh_commands_for_user", AsyncMock())
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=False))

    title = _long_title()
    now = datetime.utcnow()
    active_user_id = 9301
    ended_user_id = 9302
    active_link = 'https://example.test/event?a=1&b="quoted"'
    active_event_id = await _create_event(
        session_factory,
        code="runtime_tg_active",
        title=title,
        starts_at=now - timedelta(minutes=30),
        user_id=active_user_id,
        link=active_link,
    )
    ended_event_id = await _create_event(
        session_factory,
        code="runtime_tg_ended",
        title=title,
        starts_at=now - timedelta(days=1),
        user_id=ended_user_id,
    )

    dispatcher = Dispatcher(storage=MemoryStorage())
    dispatcher.include_router(handlers.router)
    transport = _ValidatedTelegramSession()
    bot = Bot("123456:test", session=transport)

    async def feed_start(user_id, code, update_id):
        await dispatcher.feed_update(
            bot,
            Update(
                update_id=update_id,
                message=Message(
                    message_id=update_id,
                    date=datetime.now(timezone.utc),
                    chat=Chat(id=user_id, type="private"),
                    from_user=TelegramUser(id=user_id, is_bot=False, first_name="Пользователь"),
                    text=f"/start {code}",
                ),
            ),
        )

    try:
        await feed_start(active_user_id, "runtime_tg_active", 1)
        first = next(method for method in reversed(transport.calls) if isinstance(method, SendMessage))
        _assert_full_escaped_title(first.text, title)
        assert first.parse_mode == "HTML"
        assert _telegram_callback_data(first) == f"ev_cancel:{active_event_id}"
        assert len(first.text.encode("utf-16-le")) // 2 <= 4096

        async with session_factory() as session:
            registration = await session.scalar(
                select(EventRegistration).where(
                    EventRegistration.event_id == active_event_id,
                    EventRegistration.user_id == active_user_id,
                )
            )
            assert registration.status == "started"

        await feed_start(active_user_id, "runtime_tg_active", 2)
        repeated = next(method for method in reversed(transport.calls) if isinstance(method, SendMessage))
        _assert_full_escaped_title(repeated.text, title)
        assert html.escape(active_link, quote=True) in repeated.text
        assert active_link in html.unescape(repeated.text)
        assert repeated.parse_mode == "HTML"
        assert _telegram_callback_data(repeated) == f"ev_cancel:{active_event_id}"

        callback = CallbackQuery(
            id="tg-event-cancel",
            from_user=TelegramUser(id=active_user_id, is_bot=False, first_name="Пользователь"),
            chat_instance="runtime",
            message=Message(
                message_id=2,
                date=datetime.now(timezone.utc),
                chat=Chat(id=active_user_id, type="private"),
                from_user=TelegramUser(id=999, is_bot=True, first_name="Test bot"),
                text=repeated.text,
                reply_markup=repeated.reply_markup,
            ),
            data=_telegram_callback_data(repeated),
        )
        await dispatcher.feed_update(bot, Update(update_id=3, callback_query=callback))

        async with session_factory() as session:
            registration = await session.scalar(
                select(EventRegistration).where(
                    EventRegistration.event_id == active_event_id,
                    EventRegistration.user_id == active_user_id,
                )
            )
            assert registration.status == "cancelled"
        assert await dispatcher.fsm.get_context(
            bot=bot,
            chat_id=active_user_id,
            user_id=active_user_id,
        ).get_state() is None

        await feed_start(ended_user_id, "runtime_tg_ended", 4)
        ended = next(method for method in reversed(transport.calls) if isinstance(method, SendMessage))
        _assert_full_escaped_title(ended.text, title)
        assert ended.parse_mode == "HTML"
        assert len(ended.text.encode("utf-16-le")) // 2 <= 4096
        async with session_factory() as session:
            assert await session.scalar(
                select(EventRegistration).where(
                    EventRegistration.event_id == ended_event_id,
                    EventRegistration.user_id == ended_user_id,
                )
            ) is None
    finally:
        dispatcher.sub_routers.clear()
        handlers.router._parent_router = None
        await bot.session.close()


@pytest.mark.asyncio
async def test_max_user_runtime_uses_application_and_escapes_long_title(
    session_factory,
    monkeypatch,
):
    import database
    import max_messenger_bot.app as max_app_module
    import max_messenger_bot.services.common as max_common

    monkeypatch.setattr(database, "async_session_maker", session_factory)
    monkeypatch.setattr(max_app_module, "async_session_maker", session_factory)
    monkeypatch.setattr(max_common, "async_session_maker", session_factory)
    monkeypatch.setattr(max_common, "is_admin", AsyncMock(return_value=False))

    title = _long_title()
    now = datetime.utcnow()
    active_link = 'https://example.test/event?a=1&b="quoted"'
    active_event_id = await _create_event(
        session_factory,
        code="runtime_max_active",
        title=title,
        starts_at=now - timedelta(minutes=30),
        link=active_link,
    )
    ended_event_id = await _create_event(
        session_factory,
        code="runtime_max_ended",
        title=title,
        starts_at=now - timedelta(days=1),
    )

    transport = _RecordedMaxSession()
    client = MaxApiClient(token="test-token", base_url="https://max.test")
    client._session = transport
    application = MaxBotApplication(client)
    raw_user_id = 9303
    max_user_id = MAX_ID_OFFSET + raw_user_id

    async def feed_start(raw_id, code, update_id):
        await application.handle_update(
            {
                "update_type": "bot_started",
                "update_id": update_id,
                "start_payload": code,
                "message": {
                    "sender": {"user_id": raw_id, "name": "Пользователь"},
                    "recipient": {"chat_id": raw_id},
                    "body": {"mid": f"incoming-{update_id}"},
                },
            }
        )

    await feed_start(raw_user_id, "runtime_max_active", "max-start-1")
    first = transport.requests[-1]
    assert first["method"] == "POST"
    assert first["url"] == "https://max.test/messages"
    first_body = first["body"]
    _assert_full_escaped_title(first_body["text"], title)
    assert first_body["format"] == "html"
    assert len(first_body["text"]) <= MAX_MESSAGE_TEXT_LEN
    callback_data = _max_callback_data(first_body)
    assert callback_data == f"ev_cancel:{active_event_id}"

    async with session_factory() as session:
        registration = await session.scalar(
            select(EventRegistration).where(
                EventRegistration.event_id == active_event_id,
                EventRegistration.user_id == max_user_id,
            )
        )
        assert registration.status == "started"

    await feed_start(raw_user_id, "runtime_max_active", "max-start-2")
    repeated_body = transport.requests[-1]["body"]
    _assert_full_escaped_title(repeated_body["text"], title)
    assert html.escape(active_link, quote=True) in repeated_body["text"]
    assert active_link in html.unescape(repeated_body["text"])
    assert repeated_body["format"] == "html"
    assert _max_callback_data(repeated_body) == f"ev_cancel:{active_event_id}"

    await application.handle_update(
        {
            "update_type": "message_callback",
            "update_id": "max-callback-1",
            "callback": {
                "callback_id": "max-event-cancel",
                "sender": {"user_id": raw_user_id, "name": "Пользователь"},
                "payload": callback_data,
            },
            "message": {
                "mid": "max-message-1",
                "recipient": {"chat_id": raw_user_id},
                "body": repeated_body,
            },
        }
    )
    async with session_factory() as session:
        registration = await session.scalar(
            select(EventRegistration).where(
                EventRegistration.event_id == active_event_id,
                EventRegistration.user_id == max_user_id,
            )
        )
        assert registration.status == "cancelled"

    ended_raw_user_id = raw_user_id + 1
    ended_max_user_id = MAX_ID_OFFSET + ended_raw_user_id
    await feed_start(ended_raw_user_id, "runtime_max_ended", "max-start-3")
    ended_body = transport.requests[-1]["body"]
    _assert_full_escaped_title(ended_body["text"], title)
    assert ended_body["format"] == "html"
    assert len(ended_body["text"]) <= MAX_MESSAGE_TEXT_LEN
    async with session_factory() as session:
        assert await session.scalar(
            select(EventRegistration).where(
                EventRegistration.event_id == ended_event_id,
                EventRegistration.user_id == ended_max_user_id,
            )
        ) is None
