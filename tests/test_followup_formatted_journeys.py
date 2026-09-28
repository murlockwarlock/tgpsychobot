from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib
import os

import pytest
import pytest_asyncio
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.methods import EditMessageText, SendMessage
from aiogram.types import CallbackQuery, Chat, Message, MessageEntity, Update, User as TelegramUser
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "123456:test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import admin_content_authoring
import automation_admin
import database as database_module
import followups
import handlers
from database import Base, FollowupCampaign, FollowupDelivery, FollowupRun, FollowupStep, User
from max_messenger_bot import app as max_app_module
from max_messenger_bot import storage as max_storage
from max_messenger_bot.api import MaxApiClient
from max_messenger_bot.app import MaxBotApplication
from max_messenger_bot.identity import MAX_ID_OFFSET
from max_messenger_bot.storage import StorageBase


class TelegramJourneySession(BaseSession):
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
                message_id=getattr(method, "message_id", len(self.methods)),
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


class MaxJourneyResponse:
    status = 200
    content_type = "application/json"

    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def text(self):
        import json

        return json.dumps(self.payload)

    async def json(self):
        return self.payload


class MaxJourneyTransport:
    def __init__(self):
        self.requests = []

    def request(self, method, url, *, params=None, json=None):
        path = url.rsplit("/", 1)[-1]
        request = {"method": method, "path": f"/{path}", "params": params or {}, "body": json}
        if path == "messages":
            self._validate_message(json)
        self.requests.append(request)
        if path == "messages":
            return MaxJourneyResponse({"message": {"body": {"mid": f"max-journey-{len(self.requests)}"}}})
        return MaxJourneyResponse({})

    @staticmethod
    def _validate_message(body):
        assert isinstance(body, dict)
        assert isinstance(body.get("text"), str)
        assert body.get("format") == "html"
        for attachment in body.get("attachments", []):
            assert attachment["type"] == "inline_keyboard"
            for row in attachment["payload"]["buttons"]:
                for button in row:
                    assert isinstance(button["text"], str)
                    if button["type"] == "callback":
                        assert isinstance(button["payload"], str)
                        assert button["payload"]


@pytest_asyncio.fixture
async def journey_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(StorageBase.metadata.create_all)
    try:
        yield sessions
    finally:
        await engine.dispose()


def _telegram_buttons(message):
    markup = message.reply_markup
    return [
        button
        for row in (markup.inline_keyboard if markup else [])
        for button in row
        if button.callback_data
    ]


async def _press_telegram_button(dispatcher, bot, session, label, update_id, user):
    screen = session.last_message
    button = next(button for button in _telegram_buttons(screen) if button.text == label)
    callback = CallbackQuery(
        id=f"telegram-journey-{update_id}",
        from_user=user,
        chat_instance="followup-journey",
        message=screen,
        data=button.callback_data,
    ).as_(bot)
    await dispatcher.feed_update(bot, Update(update_id=update_id, callback_query=callback))
    return session.last_message


def _telegram_input(bot, user, update_id):
    text = "Жирный текст\n\n[Продолжить](btn:continue)\n[Сайт](https://example.com)"
    return Update(
        update_id=update_id,
        message=Message(
            message_id=update_id,
            date=datetime.now(timezone.utc),
            chat=Chat(id=user.id, type="private"),
            from_user=user,
            text=text,
            entities=[MessageEntity(type="bold", offset=0, length=len("Жирный текст"))],
        ).as_(bot),
    )


def _max_buttons(request):
    return [
        button
        for attachment in request["body"].get("attachments", [])
        if attachment.get("type") == "inline_keyboard"
        for row in attachment["payload"]["buttons"]
        for button in row
    ]


def _max_messages(client):
    return [request for request in client.transport.requests if request["path"] == "/messages"]


async def _press_max_button(app, client, raw_user_id, label, update_id):
    screen = _max_messages(client)[-1]
    button = next(button for button in _max_buttons(screen) if button["text"] == label)
    await app.handle_update(
        {
            "update_type": "message_callback",
            "update_id": update_id,
            "callback": {
                "callback_id": f"callback-{update_id}",
                "payload": button["payload"],
                "sender": {"user_id": raw_user_id, "name": "MAX Admin"},
            },
            "message": {
                "recipient": {"chat_id": raw_user_id},
                "body": {"attachments": screen["body"].get("attachments", [])},
            },
        }
    )
    return _max_messages(client)[-1]


async def _send_max_formatted_text(app, raw_user_id, update_id):
    text = "Жирный текст\n\n[Продолжить](btn:continue)\n[Сайт](https://example.com)"
    await app.handle_update(
        {
            "update_type": "message_created",
            "update_id": update_id,
            "message": {
                "recipient": {"chat_id": raw_user_id},
                "sender": {"user_id": raw_user_id, "name": "MAX Admin"},
                "body": {
                    "text": text,
                    "markup": [{"type": "strong", "from": 0, "length": len("Жирный текст")}],
                },
            },
        }
    )


async def _create_campaign(sessions, user_id, name):
    async with sessions() as session:
        user = User(
            id=user_id,
            name="Admin",
            first_name="Admin",
            is_admin=True,
            accepted_disclaimer=True,
            current_dialogue_id=1,
            current_topic_id=None,
        )
        campaign = FollowupCampaign(
            name=name,
            is_active=True,
            include_main_dialogue=True,
            quiet_start_minute=0,
            quiet_end_minute=0,
            jitter_min_seconds=0,
            jitter_max_seconds=0,
        )
        campaign.steps.append(
            FollowupStep(
                sort_order=0,
                delay_minutes=1,
                message_type="static",
                message_text="Начальный текст",
            )
        )
        session.add_all([user, campaign])
        await session.flush()
        step_id = campaign.steps[0].id
        campaign_id = campaign.id
        await session.commit()
    return campaign_id, step_id


@pytest.mark.asyncio
async def test_telegram_formatted_static_followup_admin_runtime_journey(journey_db, monkeypatch):
    telegram_handlers = handlers
    telegram_automation = automation_admin
    telegram_authoring = admin_content_authoring
    modules = [telegram_handlers, telegram_automation, telegram_authoring]
    refreshed = []
    for module in modules:
        if module.router.parent_router is not None:
            module = importlib.reload(module)
        refreshed.append(module)
    telegram_handlers, telegram_automation, telegram_authoring = refreshed

    monkeypatch.setattr(database_module, "async_session_maker", journey_db)
    monkeypatch.setattr(telegram_handlers, "async_session_maker", journey_db)
    monkeypatch.setattr(telegram_automation, "async_session_maker", journey_db)
    monkeypatch.setattr(telegram_authoring, "async_session_maker", journey_db)
    monkeypatch.setattr(followups, "async_session_maker", journey_db)
    campaign_id, step_id = await _create_campaign(journey_db, 42, "Formatted Telegram")

    dispatcher = Dispatcher(storage=MemoryStorage())
    dispatcher.include_router(telegram_handlers.router)
    dispatcher.include_router(telegram_automation.router)
    dispatcher.include_router(telegram_authoring.router)
    session = TelegramJourneySession()
    bot = Bot("123456:test", session=session)
    admin = TelegramUser(id=42, is_bot=False, first_name="Admin")

    try:
        await dispatcher.feed_update(
            bot,
            Update(
                update_id=1,
                message=Message(
                    message_id=1,
                    date=datetime.now(timezone.utc),
                    chat=Chat(id=42, type="private"),
                    from_user=admin,
                    text="/admin",
                ).as_(bot),
            ),
        )
        await _press_telegram_button(dispatcher, bot, session, "⚙️ Автоматизации", 2, admin)
        await _press_telegram_button(dispatcher, bot, session, "💬 Догоняющие сообщения", 3, admin)
        await _press_telegram_button(dispatcher, bot, session, "✅ Formatted Telegram", 4, admin)
        await _press_telegram_button(dispatcher, bot, session, "🪜 Шаги (1)", 5, admin)
        await _press_telegram_button(dispatcher, bot, session, "1. через 1 мин. — текст", 6, admin)
        await _press_telegram_button(dispatcher, bot, session, "Текст сообщения", 7, admin)
        await _press_telegram_button(dispatcher, bot, session, "Изменить: Сообщение", 8, admin)
        await dispatcher.feed_update(bot, _telegram_input(bot, admin, 9))
        assert "Сохранено." in session.last_message.text
        await _press_telegram_button(dispatcher, bot, session, "Открыть материал", 10, admin)

        assert "Жирный текст" in session.last_message.text
        assert "Продолжить" in session.last_message.text
        assert "btn:continue" in session.last_message.text
        async with journey_db() as db_session:
            stored_step = await db_session.get(FollowupStep, step_id)
            assert stored_step.message_text == (
                "<b>Жирный текст</b>\n\n[Продолжить](btn:continue)\n[Сайт](https://example.com)"
            )

        await _press_telegram_button(dispatcher, bot, session, "⬅️ Назад", 11, admin)
        assert "Шаг 1" in session.last_message.text
        await _press_telegram_button(dispatcher, bot, session, "Текст сообщения", 12, admin)
        assert "Жирный текст" in session.last_message.text
        assert "Продолжить" in session.last_message.text
        assert "btn:continue" in session.last_message.text
        await _press_telegram_button(dispatcher, bot, session, "⬅️ Назад", 13, admin)
        assert "Шаг 1" in session.last_message.text
        await _press_telegram_button(dispatcher, bot, session, "⬅️ Назад", 14, admin)
        assert "Шаги цепочки" in session.last_message.text
        await _press_telegram_button(dispatcher, bot, session, "⬅️ Назад", 15, admin)
        assert "Formatted Telegram" in session.last_message.text

        async with journey_db() as db_session:
            db_session.add(
                FollowupRun(
                    campaign_id=campaign_id,
                    user_id=42,
                    dialogue_id=1,
                    topic_id=0,
                    due_at=datetime.utcnow() - timedelta(minutes=2),
                )
            )
            await db_session.commit()

        assert await followups.process_due_followups(followups.FollowupTransportRegistry(telegram=bot)) == 1
        final_method = session.methods[-1]
        assert isinstance(final_method, SendMessage)
        assert final_method.text == "<b>Жирный текст</b>"
        final_buttons = [button for row in final_method.reply_markup.inline_keyboard for button in row]
        assert [(button.text, button.callback_data, button.url) for button in final_buttons] == [
            ("Продолжить", "ai_btn:continue", None),
            ("Сайт", None, "https://example.com"),
        ]
        async with journey_db() as db_session:
            delivery = await db_session.scalar(select(FollowupDelivery))
            assert delivery.platform == "telegram"
            assert delivery.telegram_message_id is not None
    finally:
        await dispatcher.storage.close()
        await bot.session.close()


@pytest.mark.asyncio
async def test_max_formatted_static_followup_admin_runtime_journey(journey_db, monkeypatch):
    import max_messenger_bot.services.admin_followups as max_admin_followups
    import max_messenger_bot.services.common as max_common

    raw_user_id = 55
    user_id = MAX_ID_OFFSET + raw_user_id
    campaign_id, step_id = await _create_campaign(journey_db, user_id, "Formatted MAX")
    monkeypatch.setattr(max_app_module, "async_session_maker", journey_db)
    monkeypatch.setattr(max_common, "async_session_maker", journey_db)
    monkeypatch.setattr(max_admin_followups, "async_session_maker", journey_db)
    monkeypatch.setattr(max_storage, "async_session_maker", journey_db)
    monkeypatch.setattr(followups, "async_session_maker", journey_db)

    transport = MaxJourneyTransport()
    client = MaxApiClient("test-token", "https://max.test")
    client.transport = transport
    client._session = transport
    app = MaxBotApplication(client)

    await app.handle_update(
        {
            "update_type": "message_created",
            "update_id": "max-journey-admin",
            "message": {
                "recipient": {"chat_id": raw_user_id},
                "sender": {"user_id": raw_user_id, "name": "MAX Admin"},
                "body": {"text": "/admin"},
            },
        }
    )
    await _press_max_button(app, client, raw_user_id, "💬 Догоняющие сообщения", "max-journey-1")
    await _press_max_button(app, client, raw_user_id, "✅ Formatted MAX", "max-journey-2")
    await _press_max_button(app, client, raw_user_id, "🪜 Шаги (1)", "max-journey-3")
    await _press_max_button(app, client, raw_user_id, "1. через 1 мин. — текст", "max-journey-4")
    await _press_max_button(app, client, raw_user_id, "Текст сообщения", "max-journey-5")
    await _press_max_button(app, client, raw_user_id, "Изменить: Сообщение", "max-journey-6")
    await _send_max_formatted_text(app, raw_user_id, "max-journey-text")
    assert any(
        "сохранено" in request["body"]["text"].lower()
        for request in _max_messages(client)[-2:]
    )
    await _press_max_button(app, client, raw_user_id, "⬅️ Назад", "max-journey-7")
    assert "Жирный текст" in _max_messages(client)[-1]["body"]["text"]
    await _press_max_button(app, client, raw_user_id, "Текст сообщения", "max-journey-8")
    assert "Жирный текст" in _max_messages(client)[-1]["body"]["text"]
    assert "Продолжить" in _max_messages(client)[-1]["body"]["text"]
    assert "btn:continue" in _max_messages(client)[-1]["body"]["text"]
    await _press_max_button(app, client, raw_user_id, "⬅️ Назад", "max-journey-9")
    assert "Жирный текст" in _max_messages(client)[-1]["body"]["text"]
    await _press_max_button(app, client, raw_user_id, "⬅️ Назад", "max-journey-10")
    assert "Шаги цепочки" in _max_messages(client)[-1]["body"]["text"]
    await _press_max_button(app, client, raw_user_id, "⬅️ Назад", "max-journey-11")
    assert "Formatted MAX" in _max_messages(client)[-1]["body"]["text"]

    async with journey_db() as db_session:
        stored_step = await db_session.get(FollowupStep, step_id)
        assert stored_step.message_text == (
            "<b>Жирный текст</b>\n\n[Продолжить](btn:continue)\n[Сайт](https://example.com)"
        )
        db_session.add(
            FollowupRun(
                campaign_id=campaign_id,
                user_id=user_id,
                dialogue_id=1,
                topic_id=0,
                due_at=datetime.utcnow() - timedelta(minutes=2),
            )
        )
        await db_session.commit()

    assert await followups.process_due_followups(followups.FollowupTransportRegistry(max_client=client)) == 1
    final_request = next(
        request
        for request in reversed(_max_messages(client))
        if request["params"].get("user_id") == raw_user_id
    )
    assert final_request["body"]["text"] == "<b>Жирный текст</b>"
    assert final_request["body"]["format"] == "html"
    final_buttons = _max_buttons(final_request)
    assert final_buttons == [
        {"type": "callback", "text": "Продолжить", "payload": "ai_btn:continue"},
        {"type": "link", "text": "Сайт", "url": "https://example.com"},
    ]
    async with journey_db() as db_session:
        delivery = await db_session.scalar(select(FollowupDelivery))
        assert delivery.platform == "max"
        assert delivery.external_message_id

    for task in list(app.background_tasks):
        task.cancel()
    if app.background_tasks:
        await __import__("asyncio").gather(*app.background_tasks, return_exceptions=True)
