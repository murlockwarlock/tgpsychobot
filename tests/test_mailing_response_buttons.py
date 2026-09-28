from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import handlers
import max_messenger_bot.storage as max_storage
from database import Base, User
from max_messenger_bot import app as max_app
from max_messenger_bot.app import MaxBotApplication
from max_messenger_bot.identity import MAX_ID_OFFSET
from max_messenger_bot.models import IncomingMessage, Sender
from max_messenger_bot.services import admin_mailing as max_admin_mailing
from max_messenger_bot.services import common as max_common
from max_messenger_bot.storage import StorageBase
from mailing_utils import send_mailing_content
from response_buttons import extract_response_buttons


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
