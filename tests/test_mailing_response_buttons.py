from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.methods import EditMessageText, SendMessage
from aiogram.types import CallbackQuery, Chat, Message, MessageEntity, Update, User as TelegramUser
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import handlers
import max_messenger_bot.storage as max_storage
from database import Base, Mailing, User
from max_messenger_bot import app as max_app
from max_messenger_bot.app import MaxBotApplication
from max_messenger_bot.api import MaxApiClient
from max_messenger_bot.identity import MAX_ID_OFFSET
from max_messenger_bot.models import IncomingMessage, Sender
from max_messenger_bot.services import admin_mailing as max_admin_mailing
from max_messenger_bot.services import common as max_common
from max_messenger_bot.storage import StorageBase
from mailing_utils import send_mailing_content
from response_buttons import extract_response_buttons


class _MaxBoundaryResponse:
    status = 200
    content_type = "application/json"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def text(self):
        return '{"message":{"body":{"mid":"mailing-boundary"}}}'

    async def json(self):
        return {"message": {"body": {"mid": "mailing-boundary"}}}


class _MaxBoundaryTransport:
    def __init__(self):
        self.requests = []

    def request(self, method, url, *, params=None, json=None):
        self.requests.append({"method": method, "url": url, "params": params or {}, "body": json})
        return _MaxBoundaryResponse()


class _MaxBoundaryClient(MaxApiClient):
    def __init__(self):
        super().__init__("test-token", "https://max.test")
        self.transport = _MaxBoundaryTransport()
        self._session = self.transport


class _TelegramBoundarySession(BaseSession):
    def __init__(self):
        super().__init__()
        self.methods = []
        self.last_message = None

    async def close(self):
        return None

    async def make_request(self, bot, method, timeout=None):
        type(method).model_validate(method.model_dump())
        self.methods.append(method)
        if isinstance(method, (SendMessage, EditMessageText)):
            self.last_message = Message(
                message_id=len(self.methods),
                date=datetime.now(timezone.utc),
                chat=Chat(id=int(method.chat_id), type="private"),
                from_user=TelegramUser(id=999, is_bot=True, first_name="TestBot"),
                text=getattr(method, "text", ""),
                reply_markup=getattr(method, "reply_markup", None),
            ).as_(bot)
            return self.last_message
        return True

    async def stream_content(self, *args, **kwargs):
        if False:
            yield b""


class _RecordedStates:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.cleared = False

    async def get(self, _user_id):
        return self.snapshot

    async def set(self, _user_id, _chat_id, state, data):
        self.snapshot = SimpleNamespace(state=state, data=data)

    async def clear(self, _user_id):
        self.cleared = True
        self.snapshot = None


def _max_callback_update(payload: str, attachments: list[dict]) -> dict:
    return {
        "update_type": "message_callback",
        "update_id": "mailing-callback-1",
        "callback": {
            "callback_id": "callback-1",
            "payload": payload,
            "user": {
                "user_id": 55,
                "first_name": "Получатель",
            },
        },
        "message": {
            "recipient": {"chat_id": 555},
            "body": {"mid": "mailing-message-1", "attachments": attachments},
        },
    }


def _max_admin_callback(update_id: str, payload: str, attachments: list[dict]) -> dict:
    return {
        "update_type": "message_callback",
        "update_id": update_id,
        "callback": {
            "callback_id": f"callback-{update_id}",
            "payload": payload,
            "sender": {"user_id": 99, "name": "Администратор"},
        },
        "message": {
            "recipient": {"chat_id": 99},
            "body": {"attachments": attachments},
        },
    }


def _max_request_buttons(request: dict) -> list[dict]:
    return [
        button
        for attachment in request["body"].get("attachments", [])
        if attachment.get("type") == "inline_keyboard"
        for row in attachment["payload"].get("buttons", [])
        for button in row
    ]


async def _press_max_admin_button(app, client, update_id: str, label: str) -> dict:
    screen = client.transport.requests[-1]
    button = next(button for button in _max_request_buttons(screen) if button.get("text") == label)
    await app.handle_update(_max_admin_callback(update_id, button["payload"], screen["body"].get("attachments", [])))
    return client.transport.requests[-1]


@pytest.mark.asyncio
async def test_telegram_mailing_recipient_press_uses_normal_ai_button_handler():
    mailing = SimpleNamespace(
        text="Привет!\n\n[Продолжить](btn:continue)\n[Сайт](https://example.com)",
        media_file_id=None,
        media_file_type=None,
        media_position="media_top",
    )
    bot = SimpleNamespace(send_message=AsyncMock())

    await send_mailing_content(bot, 42, mailing)

    delivery = bot.send_message.await_args
    assert delivery.args[1] == "Привет!"
    markup = delivery.kwargs["reply_markup"]
    buttons = [button for row in markup.inline_keyboard for button in row]
    assert [button.text for button in buttons] == ["Продолжить", "Сайт"]
    assert buttons[0].callback_data == "ai_btn:continue"
    assert buttons[1].url == "https://example.com"
    assert "[Продолжить]" not in delivery.args[1]
    assert "btn:continue" not in delivery.args[1]

    recipient_message = SimpleNamespace(
        message_id=901,
        chat=SimpleNamespace(id=42),
        reply_markup=markup,
        edit_reply_markup=AsyncMock(),
        answer=AsyncMock(),
    )
    callback = SimpleNamespace(
        data=buttons[0].callback_data,
        from_user=SimpleNamespace(id=42),
        message=recipient_message,
        answer=AsyncMock(),
    )
    state = SimpleNamespace()
    process = AsyncMock()
    with patch.object(handlers, "_get_user_locale", AsyncMock(return_value="ru")), patch.object(
        handlers, "process_buffered_messages", process
    ):
        handlers._ai_button_claims.clear()
        await handlers.process_response_button(callback, state, bot)

    process.assert_awaited_once_with(42, bot, state, visible_user_text="Продолжить")
    assert handlers.user_message_buffers[42][0].endswith("(continue)]")
    handlers.user_message_buffers.clear()
    handlers._ai_button_claims.clear()


@pytest.mark.asyncio
async def test_escaped_llm_buttons_render_as_generated_response_buttons_on_both_platforms():
    source = (
        "\\- [Что-то случилось]\\(btn:start_event)\n"
        "\\- [Просто тяжело]\\(btn:start_heavy)\n"
        "\\- [Хочу разобраться в себе]\\(btn:start_self)"
    )

    telegram_bot = SimpleNamespace(send_message=AsyncMock())
    with patch.object(handlers, "_get_user_locale", AsyncMock(return_value="ru")):
        await handlers._send_generated_response(telegram_bot, 42, source)

    telegram_delivery = telegram_bot.send_message.await_args
    assert telegram_delivery.kwargs["text"] == "Выберите действие:"
    telegram_buttons = [
        button
        for row in telegram_delivery.kwargs["reply_markup"].inline_keyboard
        for button in row
    ]
    assert [(button.text, button.callback_data) for button in telegram_buttons] == [
        ("Что-то случилось", "ai_btn:start_event"),
        ("Просто тяжело", "ai_btn:start_heavy"),
        ("Хочу разобраться в себе", "ai_btn:start_self"),
    ]
    assert all("btn:" not in button.text for button in telegram_buttons)

    clean_text, rows = extract_response_buttons(source)
    max_client = SimpleNamespace(send_message=AsyncMock())
    if not clean_text and rows:
        clean_text = "Выберите действие:"
    await max_common._send_ai_text(
        max_client,
        555,
        None,
        [max_common.markdown_to_html(clean_text)],
        rows,
    )

    max_delivery = max_client.send_message.await_args
    assert max_delivery.kwargs["text"] == "Выберите действие:"
    max_buttons = [
        button
        for attachment in max_delivery.kwargs["attachments"]
        if attachment["type"] == "inline_keyboard"
        for row in attachment["payload"]["buttons"]
        for button in row
        if button["type"] == "callback"
    ]
    assert [(button["text"], button["payload"]) for button in max_buttons[:3]] == [
        ("Что-то случилось", "ai_btn:start_event"),
        ("Просто тяжело", "ai_btn:start_heavy"),
        ("Хочу разобраться в себе", "ai_btn:start_self"),
    ]
    assert all("btn:" not in button["text"] for button in max_buttons)


@pytest_asyncio.fixture
async def mailing_db(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'mailing-buttons.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(StorageBase.metadata.create_all)
    monkeypatch.setattr(max_admin_mailing, "async_session_maker", sessions)
    monkeypatch.setattr(max_common, "async_session_maker", sessions)
    monkeypatch.setattr(max_app, "async_session_maker", sessions)
    monkeypatch.setattr(max_storage, "async_session_maker", sessions)
    async with sessions() as session:
        session.add(User(
            id=MAX_ID_OFFSET + 99,
            first_name="Администратор",
            name="Администратор",
            is_admin=True,
            accepted_disclaimer=True,
        ))
        await session.commit()
    try:
        yield sessions
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def telegram_mailing_db(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'telegram-mailing-buttons.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with sessions() as session:
        session.add(User(
            id=11,
            first_name="Администратор",
            name="Администратор",
            is_admin=True,
            accepted_disclaimer=True,
        ))
        await session.commit()
    monkeypatch.setattr(handlers, "async_session_maker", sessions)
    try:
        yield sessions
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_max_mailing_admin_preview_delivery_and_recipient_press(mailing_db):
    user_id = MAX_ID_OFFSET + 99
    states = _RecordedStates(SimpleNamespace(
        state="admin_mailing_text",
        data={"audience": "self"},
    ))
    client = SimpleNamespace(send_message=AsyncMock(), answer_callback=AsyncMock())
    message = IncomingMessage(
        raw={},
        message_id="mailing-authoring-1",
        chat_id=99,
        sender=Sender(user_id=user_id, username=None, first_name="Администратор", last_name=None),
        text="Привет!\n\n[Продолжить](btn:continue)\n[Сайт](https://example.com)",
        html_text="Привет!\n\n[Продолжить](btn:continue)\n[Сайт](https://example.com)",
    )

    await max_admin_mailing.save_input(client, states, 99, user_id, message)

    preview = client.send_message.await_args
    preview_buttons = [
        button
        for attachment in preview.kwargs["attachments"]
        if attachment.get("type") == "inline_keyboard"
        for row in attachment["payload"]["buttons"]
        for button in row
        if button.get("payload", "").startswith("ai_btn:") or button.get("type") == "link"
    ]
    assert [button["text"] for button in preview_buttons] == ["Продолжить", "Сайт"]
    assert preview.kwargs["text"].find("[Продолжить]") == -1
    assert preview.kwargs["text"].find("btn:continue") == -1

    await max_admin_mailing.confirm_send(client, states, 99, user_id)

    delivery = next(
        call for call in client.send_message.await_args_list
        if call.kwargs.get("user_id") == 99
    )
    assert delivery.kwargs["text"] == "Привет!"
    assert delivery.kwargs["format_"] == "html"
    attachments = delivery.kwargs["attachments"]
    delivered_buttons = [
        button
        for attachment in attachments
        if attachment.get("type") == "inline_keyboard"
        for row in attachment["payload"]["buttons"]
        for button in row
    ]
    assert delivered_buttons == [
        {"type": "callback", "text": "Продолжить", "payload": "ai_btn:continue"},
        {"type": "link", "text": "Сайт", "url": "https://example.com"},
    ]

    app = MaxBotApplication(client)
    run_ai = AsyncMock()
    with patch.object(max_common, "ensure_access_before_chat", AsyncMock(return_value=True)), patch.object(
        max_common, "run_ai_dialogue", run_ai
    ):
        await app.handle_update(_max_callback_update("ai_btn:continue", attachments))
        if app.background_tasks:
            await asyncio.gather(*app.background_tasks)

    run_ai.assert_awaited_once()
    assert run_ai.await_args.args[1:] == (
        555,
        MAX_ID_OFFSET + 55,
        "[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку \"Продолжить\" (continue)]",
        app.states,
    )


@pytest.mark.asyncio
async def test_max_formatted_mailing_round_trip_media_buttons_and_reopen(mailing_db):
    user_id = MAX_ID_OFFSET + 99
    states = _RecordedStates(SimpleNamespace(
        state="admin_mailing_text",
        data={"audience": "self"},
    ))
    client = _MaxBoundaryClient()
    source = '<b>Привет</b>\n\n[Продолжить](btn:continue)\n[Сайт](https://example.com)'
    message = IncomingMessage(
        raw={},
        message_id="formatted-mailing-authoring",
        chat_id=99,
        sender=Sender(user_id=user_id, username=None, first_name="Администратор", last_name=None),
        text="Привет\n\n[Продолжить](btn:continue)\n[Сайт](https://example.com)",
        html_text=source,
        media_type="image",
        media_token="photo-token",
    )

    await max_admin_mailing.save_input(client, states, 99, user_id, message)
    assert states.snapshot.data["canonical_text"] == source
    preview_body = client.transport.requests[-1]["body"]
    preview_buttons = [
        button
        for attachment in preview_body["attachments"]
        if attachment.get("type") == "inline_keyboard"
        for row in attachment["payload"]["buttons"]
        for button in row
    ]
    assert [button["text"] for button in preview_buttons[:2]] == ["Продолжить", "Сайт"]

    await max_admin_mailing.confirm_send(client, states, 99, user_id)
    delivery_request = next(
        request
        for request in client.transport.requests
        if request["params"].get("user_id") == 99 and request["body"].get("text") == "<b>Привет</b>"
    )
    assert delivery_request["body"]["format"] == "html"
    assert delivery_request["body"]["attachments"] == [
        {"type": "image", "payload": {"token": "photo-token"}},
        {
            "type": "inline_keyboard",
            "payload": {
                "buttons": [
                    [{"type": "callback", "text": "Продолжить", "payload": "ai_btn:continue"}],
                    [{"type": "link", "text": "Сайт", "url": "https://example.com"}],
                ]
            },
        },
    ]

    async with mailing_db() as session:
        mailing = await session.scalar(select(Mailing).where(Mailing.creator_id == user_id))
        assert mailing is not None
        assert mailing.text == source

    await max_admin_mailing.show_details(client, 99, mailing.id)
    details_body = client.transport.requests[-1]["body"]
    detail_buttons = [
        button
        for attachment in details_body["attachments"]
        if attachment.get("type") == "inline_keyboard"
        for row in attachment["payload"]["buttons"]
        for button in row
        if button.get("payload") == "ai_btn:continue" or button.get("type") == "link"
    ]
    assert detail_buttons == [
        {"type": "callback", "text": "Продолжить", "payload": "ai_btn:continue"},
        {"type": "link", "text": "Сайт", "url": "https://example.com"},
    ]

    app = MaxBotApplication(client)
    run_ai = AsyncMock()
    with patch.object(max_common, "ensure_access_before_chat", AsyncMock(return_value=True)), patch.object(
        max_common, "run_ai_dialogue", run_ai
    ):
        await app.handle_update(_max_callback_update("ai_btn:continue", delivery_request["body"]["attachments"]))
        if app.background_tasks:
            await asyncio.gather(*app.background_tasks)
    run_ai.assert_awaited_once()


@pytest.mark.asyncio
async def test_max_formatted_mailing_real_admin_journey(mailing_db):
    client = _MaxBoundaryClient()
    app = MaxBotApplication(client)

    await app.handle_update({
        "update_type": "message_created",
        "update_id": "journey-admin-start",
        "message": {
            "recipient": {"chat_id": 99},
            "sender": {"user_id": 99, "name": "Администратор"},
            "body": {"text": "/admin"},
        },
    })
    await _press_max_admin_button(app, client, "journey-mailing-menu", "✉️ Рассылки")
    await _press_max_admin_button(app, client, "journey-mailing-create", "🚀 Создать рассылку")
    await _press_max_admin_button(app, client, "journey-mailing-audience", "👤 Только себе")

    text = "Привет\n\n[Продолжить](btn:continue)\n[Сайт](https://example.com)"
    await app.handle_update({
        "update_type": "message_created",
        "update_id": "journey-mailing-text",
        "message": {
            "recipient": {"chat_id": 99},
            "sender": {"user_id": 99, "name": "Администратор"},
            "body": {
                "text": text,
                "markup": [{"type": "strong", "from": 0, "length": 6}],
                "attachments": [{"type": "image", "payload": {"token": "photo-token"}}],
            },
        },
    })
    preview = client.transport.requests[-1]
    preview_buttons = _max_request_buttons(preview)
    assert {button["text"] for button in preview_buttons} >= {"Продолжить", "Сайт", "✅ Отправить"}

    await _press_max_admin_button(app, client, "journey-mailing-confirm", "✅ Отправить")
    delivery = next(
        request
        for request in client.transport.requests
        if request["params"].get("user_id") == 99 and request["body"].get("text") == "<b>Привет</b>"
    )
    assert {button["text"] for button in _max_request_buttons(delivery)} == {"Продолжить", "Сайт"}

    summary = client.transport.requests[-1]
    await _press_max_admin_button(app, client, "journey-mailing-history", "📜 История рассылок")
    history = client.transport.requests[-1]
    history_button = next(
        button for button in _max_request_buttons(history) if button["payload"].startswith("mailing_details_")
    )
    await app.handle_update(_max_admin_callback("journey-mailing-details", history_button["payload"], history["body"]["attachments"]))
    details = client.transport.requests[-1]
    assert {button["text"] for button in _max_request_buttons(details)} >= {"Продолжить", "Сайт"}

    run_ai = AsyncMock()
    with patch.object(max_common, "ensure_access_before_chat", AsyncMock(return_value=True)), patch.object(
        max_common, "run_ai_dialogue", run_ai
    ):
        await app.handle_update(_max_callback_update("ai_btn:continue", delivery["body"]["attachments"]))
        if app.background_tasks:
            await asyncio.gather(*app.background_tasks)
    run_ai.assert_awaited_once()

    for task in list(app.background_tasks):
        task.cancel()
    if app.background_tasks:
        await asyncio.gather(*app.background_tasks, return_exceptions=True)


def _telegram_inline_buttons(message):
    return [
        button
        for row in (message.reply_markup.inline_keyboard if message.reply_markup else [])
        for button in row
        if button.callback_data or button.url
    ]


async def _feed_telegram_button(dispatcher, bot, message, label, update_id):
    button = next(button for button in _telegram_inline_buttons(message) if button.text == label)
    callback = CallbackQuery(
        id=f"telegram-callback-{update_id}",
        from_user=TelegramUser(id=11, is_bot=False, first_name="Администратор"),
        chat_instance="mailing",
        message=message,
        data=button.callback_data,
    ).as_(bot)
    await dispatcher.feed_update(bot, Update(update_id=update_id, callback_query=callback))


@pytest.mark.asyncio
async def test_telegram_formatted_mailing_real_admin_journey(telegram_mailing_db, monkeypatch):
    telegram_handlers = handlers
    if telegram_handlers.router.parent_router is not None:
        telegram_handlers = importlib.reload(telegram_handlers)
        monkeypatch.setattr(telegram_handlers, "async_session_maker", telegram_mailing_db)

    dispatcher = Dispatcher(storage=MemoryStorage())
    dispatcher.include_router(telegram_handlers.router)
    session = _TelegramBoundarySession()
    bot = Bot("123456:test", session=session)
    admin = TelegramUser(id=11, is_bot=False, first_name="Администратор")
    await dispatcher.feed_update(
        bot,
        Update(
            update_id=1,
            message=Message(
                message_id=1,
                date=datetime.now(timezone.utc),
                chat=Chat(id=11, type="private"),
                from_user=admin,
                text="/admin",
            ).as_(bot),
        ),
    )
    await _feed_telegram_button(dispatcher, bot, session.last_message, "✉️ Рассылка", 2)
    await _feed_telegram_button(dispatcher, bot, session.last_message, "🚀 Создать рассылку", 3)
    await _feed_telegram_button(dispatcher, bot, session.last_message, "👤 Только себе (тест)", 4)

    text = "Привет\n\n[Продолжить](btn:continue)\n[Сайт](https://example.com)"
    await dispatcher.feed_update(
        bot,
        Update(
            update_id=5,
            message=Message(
                message_id=5,
                date=datetime.now(timezone.utc),
                chat=Chat(id=11, type="private"),
                from_user=admin,
                text=text,
                entities=[MessageEntity(type="bold", offset=0, length=6)],
            ).as_(bot),
        ),
    )
    preview = session.last_message
    preview_buttons = _telegram_inline_buttons(preview)
    assert {button.text for button in preview_buttons} >= {"Продолжить", "Сайт", "✅ Отправить"}
    await _feed_telegram_button(dispatcher, bot, preview, "✅ Отправить", 6)

    async with telegram_mailing_db() as db_session:
        mailing = await db_session.scalar(select(Mailing).where(Mailing.creator_id == 11))
        assert mailing is not None
        assert mailing.text == '<b>Привет</b>\n\n[Продолжить](btn:continue)\n[Сайт](https://example.com)'

    await send_mailing_content(bot, 11, mailing)
    delivery = session.last_message
    assert delivery.text == "<b>Привет</b>"
    delivery_buttons = _telegram_inline_buttons(delivery)
    assert [(button.text, button.callback_data, button.url) for button in delivery_buttons] == [
        ("Продолжить", "ai_btn:continue", None),
        ("Сайт", None, "https://example.com"),
    ]

    process = AsyncMock()
    with patch.object(telegram_handlers, "_get_user_locale", AsyncMock(return_value="ru")), patch.object(
        telegram_handlers, "process_buffered_messages", process
    ):
        callback = CallbackQuery(
            id="telegram-recipient-callback",
            from_user=admin,
            chat_instance="mailing",
            message=delivery,
            data=delivery_buttons[0].callback_data,
        ).as_(bot)
        await dispatcher.feed_update(bot, Update(update_id=7, callback_query=callback))
    process.assert_awaited_once()
