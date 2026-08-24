from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import handlers
from database import Base, Message as DBMessage, TelegramPendingAIReply, User
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
    try:
        yield engine, sessions
    finally:
        handlers.user_message_buffers.clear()
        await engine.dispose()


class _ReplyMessage:
    def __init__(self, text: str, message_id: int = 101):
        self.text = text
        self.message_id = message_id
        self.from_user = SimpleNamespace(id=42)
        self.chat = SimpleNamespace(id=42)
        self.answer = AsyncMock()
        self.reply_markup = None
        self.edit_reply_markup = AsyncMock()


class _ReplyBot:
    def __init__(self):
        self.sent = []

    async def send_chat_action(self, **kwargs):
        return None

    async def send_message(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(delete=AsyncMock())


def test_conversational_actions_render_as_reply_keyboard_and_navigation_stays_inline():
    _, rows = extract_response_buttons(
        "[Синтетический ответ](btn:some_new_action) | [Другой ответ](btn:another_new_action)\n"
        "[Документация](https://example.com) | [Навигация](btn:topics)"
    )

    reply_rows, inline_rows, pending_mapping = handlers._partition_telegram_response_buttons(rows)
    reply_markup = handlers._telegram_reply_keyboard_markup(reply_rows)
    inline_markup = handlers._telegram_response_buttons_markup(inline_rows)

    assert isinstance(reply_markup, ReplyKeyboardMarkup)
    assert reply_markup.resize_keyboard is True
    assert reply_markup.one_time_keyboard is True
    assert [button.text for button in reply_markup.keyboard[0]] == ["Синтетический ответ", "Другой ответ"]
    assert pending_mapping == {
        "Синтетический ответ": "some_new_action",
        "Другой ответ": "another_new_action",
    }
    assert inline_markup is not None
    assert inline_markup.inline_keyboard[0][0].url == "https://example.com"
    assert inline_markup.inline_keyboard[0][1].callback_data.startswith("ai_btn:topics")


def test_ambiguous_visible_label_stays_inline_but_same_action_with_different_labels_is_reply_keyboard():
    _, ambiguous_rows = extract_response_buttons(
        "[Неоднозначный ответ](btn:first) | [Неоднозначный ответ](btn:second)"
    )
    reply_rows, inline_rows, pending_mapping = handlers._partition_telegram_response_buttons(ambiguous_rows)
    assert reply_rows == []
    assert pending_mapping == {}
    assert len(inline_rows[0]) == 2

    _, valid_rows = extract_response_buttons(
        "[Ответ A](btn:continue) | [Ответ B](btn:continue)"
    )
    reply_rows, inline_rows, pending_mapping = handlers._partition_telegram_response_buttons(valid_rows)
    assert len(reply_rows[0]) == 2
    assert inline_rows == []
    assert pending_mapping == {"Ответ A": "continue", "Ответ B": "continue"}

    _, url_collision_rows = extract_response_buttons(
        "[Одинаковый текст](btn:some_new_action) | "
        "[Одинаковый текст](https://example.com)"
    )
    reply_rows, inline_rows, pending_mapping = handlers._partition_telegram_response_buttons(url_collision_rows)
    assert reply_rows == []
    assert pending_mapping == {}
    assert len(inline_rows[0]) == 2


@pytest.mark.asyncio
async def test_generated_response_persists_arbitrary_new_reply_button_and_sends_reply_keyboard(pending_store):
    bot = _ReplyBot()

    await handlers._send_generated_response(
        bot,
        42,
        "[Совершенно новый ответ](btn:some_new_action)",
    )

    assert len(bot.sent) == 1
    assert bot.sent[0]["text"] == "Выберите действие:"
    assert isinstance(bot.sent[0]["reply_markup"], ReplyKeyboardMarkup)
    assert bot.sent[0]["reply_markup"].keyboard[0][0].text == "Совершенно новый ответ"

    async with pending_store[1]() as session:
        pending = await session.get(TelegramPendingAIReply, 42)
    assert json.loads(pending.mapping_json) == {"Совершенно новый ответ": "some_new_action"}


@pytest.mark.asyncio
async def test_mixed_response_keeps_reply_and_inline_controls(pending_store):
    bot = _ReplyBot()

    await handlers._send_generated_response(
        bot,
        42,
        "Текст ответа\n"
        "[Синтетический ответ](btn:some_new_action) | "
        "[Документация](https://example.com)",
    )

    assert len(bot.sent) == 2
    assert isinstance(bot.sent[0]["reply_markup"], ReplyKeyboardMarkup)
    assert bot.sent[0]["reply_markup"].keyboard[0][0].text == "Синтетический ответ"
    assert bot.sent[1]["text"] == "Дополнительные действия:"
    assert bot.sent[1]["reply_markup"].inline_keyboard[0][0].url == "https://example.com"


@pytest.mark.asyncio
async def test_accepted_inline_control_invalidates_mixed_reply_keyboard(pending_store, monkeypatch):
    handlers._ai_button_claims.clear()
    handlers.user_message_buffers.clear()
    await handlers._replace_telegram_pending_replies(
        42,
        {"Старая синтетическая кнопка": "old_answer_action"},
    )
    markup = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="Панель действий", callback_data="ai_btn:control_action"),
    ]])
    message = _ReplyMessage("mixed", message_id=601)
    message.reply_markup = markup
    callback = SimpleNamespace(
        data="ai_btn:control_action",
        from_user=SimpleNamespace(id=42),
        message=message,
        answer=AsyncMock(),
    )
    process = AsyncMock()
    monkeypatch.setattr(handlers, "process_buffered_messages", process)

    await handlers.process_response_button(callback, None, _ReplyBot())

    async with pending_store[1]() as session:
        pending = await session.get(TelegramPendingAIReply, 42)
    assert json.loads(pending.mapping_json) == {}
    assert isinstance(
        message.answer.await_args.kwargs["reply_markup"],
        ReplyKeyboardRemove,
    )
    assert process.await_count == 1

    stale_reply = await handlers._consume_telegram_pending_reply(
        42,
        "Старая синтетическая кнопка",
        602,
    )
    assert stale_reply.accepted is False
    assert stale_reply.stale_cleared is False
    assert process.await_count == 1


@pytest.mark.asyncio
async def test_special_inline_control_invalidates_pending_reply_keyboard(pending_store, monkeypatch):
    handlers._ai_button_claims.clear()
    await handlers._replace_telegram_pending_replies(
        42,
        {"Ещё один синтетический ответ": "another_answer_action"},
    )
    markup = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="Открыть раздел", callback_data="ai_btn:topics"),
    ]])
    message = _ReplyMessage("mixed", message_id=603)
    message.reply_markup = markup
    callback = SimpleNamespace(
        data="ai_btn:topics",
        from_user=SimpleNamespace(id=42),
        message=message,
        answer=AsyncMock(),
    )
    topics = AsyncMock()
    monkeypatch.setattr(handlers, "select_topic_menu", topics)

    await handlers.process_response_button(callback, None, _ReplyBot())

    async with pending_store[1]() as session:
        pending = await session.get(TelegramPendingAIReply, 42)
    assert json.loads(pending.mapping_json) == {}
    assert isinstance(message.answer.await_args.kwargs["reply_markup"], ReplyKeyboardRemove)
    topics.assert_awaited_once()


@pytest.mark.asyncio
async def test_unresolved_inline_control_does_not_invalidate_pending_reply_keyboard(pending_store):
    handlers._ai_button_claims.clear()
    await handlers._replace_telegram_pending_replies(
        42,
        {"Сохрани этот ответ": "preserve_answer_action"},
    )
    message = _ReplyMessage("mixed", message_id=604)
    message.reply_markup = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="Действие", callback_data="ai_btn:known_action"),
    ]])
    callback = SimpleNamespace(
        data="ai_btn:missing_action",
        from_user=SimpleNamespace(id=42),
        message=message,
        answer=AsyncMock(),
    )

    await handlers.process_response_button(callback, None, _ReplyBot())

    async with pending_store[1]() as session:
        pending = await session.get(TelegramPendingAIReply, 42)
    assert json.loads(pending.mapping_json) == {"Сохрани этот ответ": "preserve_answer_action"}
    assert message.answer.await_count == 0


@pytest.mark.asyncio
async def test_reply_mapping_persists_across_session_reload_and_replaces_previous(pending_store):
    engine, sessions = pending_store
    assert await handlers._replace_telegram_pending_replies(42, {"Архивный вариант": "old"}) is False
    assert await handlers._replace_telegram_pending_replies(42, {"Свежий вариант": "new"}) is True

    async with sessions() as session:
        pending = await session.get(TelegramPendingAIReply, 42)
        assert json.loads(pending.mapping_json) == {"Свежий вариант": "new"}

    reloaded_sessions = async_sessionmaker(engine, expire_on_commit=False)
    handlers.async_session_maker = reloaded_sessions
    result = await handlers._consume_telegram_pending_reply(42, "Свежий вариант", 201)
    assert result.accepted is True
    assert result.label == "Свежий вариант"
    assert result.action == "new"


@pytest.mark.asyncio
async def test_pending_reply_is_consumed_once_with_exact_context_and_no_duplicate_user_message(pending_store, monkeypatch):
    await handlers._replace_telegram_pending_replies(42, {"Синтетический ответ": "some_new_action"})
    message = _ReplyMessage("Синтетический ответ", message_id=301)
    bot = _ReplyBot()
    generated = AsyncMock(return_value="Принято")

    monkeypatch.setattr(handlers, "_resend_active_spread_choice", AsyncMock(return_value=False))
    monkeypatch.setattr(
        handlers,
        "handle_ai_media_content",
        AsyncMock(return_value=("Принято", [], [], [], [], [])),
    )
    monkeypatch.setattr(handlers.ai_integration, "generate_response", generated)

    result = await handlers._consume_telegram_pending_reply(42, message.text, message.message_id)
    assert result.accepted is True
    await handlers._handle_pending_telegram_reply(message, None, bot, result)

    replay = await handlers._consume_telegram_pending_reply(42, message.text, message.message_id)
    assert replay.duplicate is True
    assert generated.await_count == 1
    assert handlers.user_message_buffers.get(42) is None

    assert message.answer.await_count == 1
    acknowledgement = message.answer.await_args
    assert acknowledgement.args == ("Ответ принят: Синтетический ответ",)
    assert isinstance(acknowledgement.kwargs["reply_markup"], ReplyKeyboardRemove)

    async with pending_store[1]() as session:
        user_messages = (
            await session.execute(
                select(DBMessage).where(DBMessage.user_id == 42, DBMessage.role == "user")
            )
        ).scalars().all()
    assert len(user_messages) == 1
    assert user_messages[0].content == "Синтетический ответ"
    assert user_messages[0].ai_context_content == (
        '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку "Синтетический ответ" (some_new_action)]'
    )


@pytest.mark.asyncio
async def test_arbitrary_typed_text_clears_stale_mapping(pending_store):
    await handlers._replace_telegram_pending_replies(42, {"Синтетический ответ": "some_new_action"})

    stale = await handlers._consume_telegram_pending_reply(42, "обычный текст", 401)
    assert stale.stale_cleared is True

    old_button = await handlers._consume_telegram_pending_reply(42, "Синтетический ответ", 402)
    assert old_button.accepted is False
    assert old_button.stale_cleared is False


@pytest.mark.asyncio
async def test_arbitrary_typed_text_remains_normal_input_and_removes_stale_keyboard(
    pending_store,
    monkeypatch,
):
    await handlers._replace_telegram_pending_replies(42, {"Синтетический ответ": "some_new_action"})
    message = _ReplyMessage("обычный пользовательский текст", message_id=501)
    message.from_user.username = "tester"
    message.from_user.full_name = "Тест"
    state = SimpleNamespace()
    bot = SimpleNamespace()
    process = AsyncMock()

    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(handlers, "_sync_user_birthdate_from_telegram", AsyncMock())
    monkeypatch.setattr(handlers, "_request_profile_onboarding_if_needed", AsyncMock(return_value=False))
    monkeypatch.setattr(handlers, "process_buffered_messages", process)
    handlers.user_processing_tasks.clear()

    await handlers.handle_ai_chat(message, state, bot)
    await handlers.user_processing_tasks[42]

    process.assert_awaited_once_with(
        42,
        bot,
        state,
        remove_reply_keyboard=True,
    )
    assert handlers.user_message_buffers[42] == ["обычный пользовательский текст"]
    handlers.user_processing_tasks.clear()


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
