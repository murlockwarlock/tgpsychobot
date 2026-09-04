from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest
import pytest_asyncio
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import handlers
from database import Base, TelegramPendingAIReply, User
from response_buttons import extract_response_buttons


@pytest_asyncio.fixture
async def pending_store(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'reply-buttons.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with sessions() as session:
        session.add(User(
            id=42,
            first_name="Тест",
            name="Тест",
            gender="unknown",
            current_dialogue_id=1,
            is_admin=True,
            accepted_disclaimer=True,
        ))
        await session.commit()

    monkeypatch.setattr(handlers, "async_session_maker", sessions)
    handlers.user_message_buffers.clear()
    handlers.user_processing_tasks.clear()
    handlers._ai_button_claims.clear()
    try:
        yield engine, sessions
    finally:
        handlers.user_message_buffers.clear()
        handlers.user_processing_tasks.clear()
        handlers._ai_button_claims.clear()
        await engine.dispose()


class _ReplyMessage:
    def __init__(self, text: str = "ответ", message_id: int = 101, reply_markup=None):
        self.text = text
        self.message_id = message_id
        self.from_user = SimpleNamespace(id=42, username="tester", full_name="Тест")
        self.chat = SimpleNamespace(id=42)
        self.answer = AsyncMock()
        self.reply_markup = reply_markup
        self.edit_calls = []

    async def edit_reply_markup(self, *, reply_markup=None):
        self.edit_calls.append(reply_markup)
        self.reply_markup = reply_markup


class _ReplyBot:
    def __init__(self):
        self.sent = []

    async def send_chat_action(self, **kwargs):
        return None

    async def send_message(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(delete=AsyncMock())


def _button_callback(message, data="ai_btn:some_new_action"):
    return SimpleNamespace(
        data=data,
        from_user=SimpleNamespace(id=42),
        message=message,
        answer=AsyncMock(),
    )


def _ai_settings_config(*, allow_fallback):
    return SimpleNamespace(
        provider="KIE",
        kie_model="gemini-2.5-flash",
        transcription_provider="OpenAI",
        vision_provider="Gemini",
        vision_model="gemini-3.7-flash",
        image_generation_provider="OpenAI",
        image_generation_model="gpt-image-2",
        image_edit_provider="KIE",
        image_edit_model="seedream/4.5-edit",
        kie_credit_alert_threshold=0,
        fallback_provider="Deepseek",
        fallback_model="deepseek-v4-flash",
        allow_fallback=allow_fallback,
        max_voice_duration_sec=180,
        prompt_mode="text",
        prompt_filename=None,
        shared_prompt_block="shared",
        service_prompt_block="service",
    )


def test_all_ai_response_buttons_are_inline_and_keep_rows_and_urls():
    _, rows = extract_response_buttons(
        "[Новый ответ](btn:some_new_action) | [Сайт](https://example.com) | [Темы](btn:topics)\n"
        "[Тест](btn:start_test) | [Тема](btn:topic_42) | [Меню](btn:main_menu)"
    )

    markup = handlers._telegram_response_buttons_markup(rows)

    assert isinstance(markup, InlineKeyboardMarkup)
    assert len(markup.inline_keyboard) == 2
    buttons = [button for row in markup.inline_keyboard for button in row]
    assert [button.text for button in buttons] == [
        "Новый ответ",
        "Сайт",
        "Темы",
        "Тест",
        "Тема",
        "Меню",
    ]
    assert buttons[0].callback_data == "ai_btn:some_new_action"
    assert buttons[1].url == "https://example.com"
    assert buttons[2].callback_data == "ai_btn:topics"
    assert buttons[3].callback_data == "ai_btn:start_test"
    assert buttons[4].callback_data == "ai_btn:topic_42"
    assert buttons[5].callback_data == "ai_btn:main_menu"


def test_generated_response_buttons_use_whitespace_for_next_rows():
    _, rows = extract_response_buttons(
        "[A](btn:a) [B](btn:b) | [Сайт](https://example.com)"
    )

    markup = handlers._telegram_response_buttons_markup(rows)

    assert [[button.text for button in row] for row in markup.inline_keyboard] == [
        ["A"],
        ["B", "Сайт"],
    ]
    assert markup.inline_keyboard[0][0].callback_data == "ai_btn:a"
    assert markup.inline_keyboard[1][0].callback_data == "ai_btn:b"
    assert markup.inline_keyboard[1][1].url == "https://example.com"


@pytest.mark.asyncio
async def test_generated_response_uses_one_inline_markup_and_does_not_touch_legacy_rows(pending_store):
    _, sessions = pending_store
    async with sessions() as session:
        session.add(TelegramPendingAIReply(
            user_id=42,
            mapping_json=json.dumps({"Старая кнопка": "old_action"}, ensure_ascii=False),
            consumed_message_id=None,
        ))
        await session.commit()

    bot = _ReplyBot()
    await handlers._send_generated_response(
        bot,
        42,
        "Текст ответа\n[Новый ответ](btn:some_new_action) | [Документация](https://example.com)",
    )

    assert len(bot.sent) == 1
    assert bot.sent[0]["text"] == "Текст ответа"
    markup = bot.sent[0]["reply_markup"]
    assert isinstance(markup, InlineKeyboardMarkup)
    assert [button.text for row in markup.inline_keyboard for button in row] == [
        "Новый ответ",
        "Документация",
    ]
    assert not hasattr(markup, "keyboard")

    async with sessions() as session:
        pending = await session.get(TelegramPendingAIReply, 42)
    assert json.loads(pending.mapping_json) == {"Старая кнопка": "old_action"}


@pytest.mark.asyncio
async def test_mixed_answer_url_and_navigation_buttons_stay_under_the_same_message():
    bot = _ReplyBot()

    await handlers._send_generated_response(
        bot,
        42,
        "Текст ответа\n"
        "[Да](btn:yes) | [Документация](https://example.com) | [Подписка](btn:subscription)",
    )

    assert len(bot.sent) == 1
    assert bot.sent[0]["text"] == "Текст ответа"
    markup = bot.sent[0]["reply_markup"]
    assert isinstance(markup, InlineKeyboardMarkup)
    assert [button.callback_data for button in markup.inline_keyboard[0]] == [
        "ai_btn:yes",
        None,
        "ai_btn:subscription",
    ]
    assert markup.inline_keyboard[0][1].url == "https://example.com"


@pytest.mark.asyncio
async def test_inline_click_disables_first_echoes_once_and_runs_one_continuation(pending_store):
    message = _ReplyMessage(
        message_id=601,
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="Да", callback_data="ai_btn:yes"),
        ]]),
    )
    events = []

    async def echo(*args, **kwargs):
        events.append("echo")

    message.answer = AsyncMock(side_effect=echo)
    callback = _button_callback(message, "ai_btn:yes")

    async def acknowledge(*args, **kwargs):
        events.append("callback_ack")

    callback.answer = AsyncMock(side_effect=acknowledge)
    process = AsyncMock(side_effect=lambda *args, **kwargs: events.append("continue"))
    original_edit = message.edit_reply_markup

    async def disable(*args, **kwargs):
        events.append("disable")
        await original_edit(*args, **kwargs)

    message.edit_reply_markup = disable
    original_process = handlers.process_buffered_messages
    handlers.process_buffered_messages = process
    try:
        await handlers.process_response_button(callback, None, _ReplyBot())
        await handlers.process_response_button(callback, None, _ReplyBot())
    finally:
        handlers.process_buffered_messages = original_process

    assert events.index("disable") < events.index("echo") < events.index("continue")
    assert message.edit_calls == [None]
    assert message.answer.await_count == 1
    assert message.answer.await_args.args == ("Ответ принят: Да",)
    assert message.answer.await_args.kwargs == {"parse_mode": None}
    assert process.await_count == 1
    process.assert_awaited_once_with(
        42,
        ANY,
        None,
        visible_user_text="Да",
    )
    assert handlers.user_message_buffers[42] == [
        '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку "Да" (yes)]'
    ]


@pytest.mark.asyncio
async def test_stale_legacy_rows_do_not_intercept_ordinary_typed_text(pending_store, monkeypatch):
    _, sessions = pending_store
    async with sessions() as session:
        session.add(TelegramPendingAIReply(
            user_id=42,
            mapping_json=json.dumps({"Старая кнопка": "old_action"}, ensure_ascii=False),
            consumed_message_id=None,
        ))
        await session.commit()

    message = _ReplyMessage("обычный пользовательский текст", message_id=701)
    state = SimpleNamespace()
    bot = SimpleNamespace()
    async def fake_process(user_id, *args, **kwargs):
        assert handlers.user_message_buffers.get(user_id) == ["обычный пользовательский текст"]
        handlers.user_message_buffers.pop(user_id, None)

    process = AsyncMock(side_effect=fake_process)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(handlers, "_sync_user_birthdate_from_telegram", AsyncMock())
    monkeypatch.setattr(handlers, "_request_profile_onboarding_if_needed", AsyncMock(return_value=False))
    monkeypatch.setattr(handlers, "process_buffered_messages", process)

    await handlers.handle_ai_chat(message, state, bot)
    await handlers.user_processing_tasks[42]

    process.assert_awaited_once_with(42, bot, state)
    assert 42 not in handlers.user_message_buffers
    async with sessions() as session:
        pending = await session.get(TelegramPendingAIReply, 42)
    assert json.loads(pending.mapping_json) == {"Старая кнопка": "old_action"}


@pytest.mark.asyncio
async def test_legacy_ai_button_callback_still_uses_inline_label_and_action():
    handlers._ai_button_claims.clear()
    handlers.user_message_buffers.clear()
    message = _ReplyMessage(
        message_id=702,
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="Продолжить", callback_data="ai_btn:legacy_action"),
        ]]),
    )
    callback = _button_callback(message, "ai_btn:legacy_action")
    process = AsyncMock()
    original_process = handlers.process_buffered_messages
    handlers.process_buffered_messages = process
    try:
        await handlers.process_response_button(callback, None, _ReplyBot())
    finally:
        handlers.process_buffered_messages = original_process

    assert message.answer.await_args.args == ("Ответ принят: Продолжить",)
    process.assert_awaited_once()
    assert handlers.user_message_buffers[42] == [
        '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку "Продолжить" (legacy_action)]'
    ]


@pytest.mark.asyncio
async def test_telegram_fallback_picker_enables_the_fallback_runtime(monkeypatch):
    config = SimpleNamespace(fallback_provider=None, fallback_model=None, allow_fallback=False)

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def get(self, _model, _key):
            return config

        async def commit(self):
            return None

    callback = SimpleNamespace(answer=AsyncMock())
    monkeypatch.setattr(handlers, "async_session_maker", lambda: Session())
    monkeypatch.setattr(handlers, "admin_ai_keys_models", AsyncMock())

    await handlers.admin_toggle_fallback(callback)

    assert config.fallback_provider is not None
    assert config.fallback_model
    assert config.allow_fallback is True
    assert "включен" in callback.answer.await_args_list[0].args[0]

    legacy_config = SimpleNamespace(
        fallback_provider="KIE",
        fallback_model="claude-haiku-4-5",
        allow_fallback=False,
    )

    class LegacySession(Session):
        async def get(self, _model, _key):
            return legacy_config

    monkeypatch.setattr(handlers, "async_session_maker", lambda: LegacySession())
    await handlers.admin_toggle_fallback(callback)
    assert legacy_config.fallback_provider == "KIE"
    assert legacy_config.allow_fallback is True
    assert "включен" in callback.answer.await_args_list[1].args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("allow_fallback", "status"),
    ((False, "❌ выключен"), (True, "✅ включён")),
)
async def test_telegram_ai_settings_info_shows_fallback_status(monkeypatch, allow_fallback, status):
    config = _ai_settings_config(allow_fallback=allow_fallback)

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def get(self, _model, _key):
            return config

    target_message = SimpleNamespace(
        text=None,
        reply_markup=None,
        edit_text=AsyncMock(),
    )
    callback = SimpleNamespace(message=target_message)
    monkeypatch.setattr(handlers, "async_session_maker", lambda: Session())

    await handlers.admin_ai_settings(callback)

    rendered_text = target_message.edit_text.await_args.args[0]
    assert f"▫️ Статус: <b>{status}</b>" in rendered_text
