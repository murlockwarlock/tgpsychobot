import asyncio
import os
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import handlers
import keyboards
from database import (
    AI_PROCESSING_MESSAGE_MAX_LENGTH,
    BotGeneralConfig,
    DEFAULT_AI_PROCESSING_MESSAGE_TEXT,
)
from result_history import select_ai_history_messages
from response_buttons import extract_response_buttons, split_action_callback_data


class _ButtonMessage:
    def __init__(self, message_id=77, reply_markup=None):
        self.chat = SimpleNamespace(id=1001)
        self.message_id = message_id
        self.reply_markup = reply_markup if reply_markup is not None else InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="Пропустить", callback_data="ai_btn:hero_skip"),
            InlineKeyboardButton(text="Открыть сайт", url="https://example.com"),
        ]])
        self.answer = AsyncMock()
        self.edit_calls = []

    async def edit_reply_markup(self, *, reply_markup=None):
        self.edit_calls.append(reply_markup)
        self.reply_markup = reply_markup


def _button_callback(message, data="ai_btn:hero_skip"):
    return SimpleNamespace(
        data=data,
        from_user=SimpleNamespace(id=42),
        message=message,
        answer=AsyncMock(),
    )


@pytest.fixture(autouse=True)
def _disable_durable_reply_cleanup_for_legacy_button_tests(monkeypatch):
    monkeypatch.setattr(
        handlers,
        "_invalidate_telegram_pending_reply",
        AsyncMock(return_value=False),
    )


def _button_callback_with_model_copy(message, data):
    callback = _button_callback(message, data)

    def model_copy(*, update):
        return _button_callback(message, update["data"])

    callback.model_copy = model_copy
    return callback


def test_button_payload_uses_visible_label_and_sanitizes_framing():
    assert handlers.build_ai_button_system_message("Пропустить", "hero_skip") == (
        '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку "Пропустить" (hero_skip)]'
    )
    escaped = handlers.build_ai_button_system_message('bad"]\nnext', 'a](b)')
    assert "\n" not in escaped
    assert escaped.startswith("[СИСТЕМНОЕ СООБЩЕНИЕ:")
    assert escaped.endswith("]")


def test_indexed_action_callbacks_are_compact_and_legacy_callbacks_parse():
    _, rows = extract_response_buttons(
        "[Да](btn:continue) | [Продолжить](btn:continue) | "
        "[Сайт](https://example.com)"
    )
    markup = handlers._telegram_response_buttons_markup(rows)
    callbacks = [button.callback_data for button in markup.inline_keyboard[0]]

    assert callbacks[:2] == ["ai_btn:continue|00", "ai_btn:continue|01"]
    assert all(len(value.encode("utf-8")) <= 64 for value in callbacks[:2])
    assert callbacks[2] is None
    assert split_action_callback_data("ai_btn:continue") == ("continue", None)
    assert split_action_callback_data("ai_btn:continue|01") == ("continue", 1)


@pytest.mark.asyncio
async def test_duplicate_action_buttons_echo_clicked_label_and_share_action():
    handlers._ai_button_claims.clear()
    handlers.user_message_buffers.clear()
    _, rows = extract_response_buttons(
        "[Да](btn:continue) | [Продолжить](btn:continue)"
    )
    markup = handlers._telegram_response_buttons_markup(rows)
    bot = SimpleNamespace(send_message=AsyncMock())
    process = AsyncMock()

    with patch.object(handlers, "process_buffered_messages", process):
        first_message = _ButtonMessage(101, markup)
        await handlers.process_response_button(
            _button_callback(first_message, "ai_btn:continue|00"),
            None,
            bot,
        )
        first_payload = handlers.user_message_buffers[42][0]
        handlers.user_message_buffers.clear()

        second_message = _ButtonMessage(102, markup)
        await handlers.process_response_button(
            _button_callback(second_message, "ai_btn:continue|01"),
            None,
            bot,
        )
        second_payload = handlers.user_message_buffers[42][0]

    assert first_message.answer.await_args.args == ("Ответ принят: Да",)
    assert second_message.answer.await_args.args == ("Ответ принят: Продолжить",)
    assert '"Да" (continue)' in first_payload
    assert '"Продолжить" (continue)' in second_payload
    assert [call.kwargs["visible_user_text"] for call in process.await_args_list] == [
        "Да",
        "Продолжить",
    ]


@pytest.mark.asyncio
async def test_legacy_action_callback_remains_accepted():
    handlers._ai_button_claims.clear()
    handlers.user_message_buffers.clear()
    bot = SimpleNamespace(send_message=AsyncMock())
    process = AsyncMock()

    with patch.object(handlers, "process_buffered_messages", process):
        await handlers.process_response_button(_button_callback(_ButtonMessage()), None, bot)

    assert process.await_count == 1
    assert process.await_args.kwargs["visible_user_text"] == "Пропустить"


@pytest.mark.asyncio
async def test_unresolved_action_label_is_acknowledged_and_rejected(caplog):
    handlers._ai_button_claims.clear()
    handlers.user_message_buffers.clear()
    bot = SimpleNamespace(send_message=AsyncMock())
    process = AsyncMock()
    callback = _button_callback(_ButtonMessage(), "ai_btn:not_in_markup")

    with patch.object(handlers, "process_buffered_messages", process):
        with caplog.at_level("WARNING", logger=handlers.log.name):
            await handlers.process_response_button(callback, None, bot)

    assert callback.answer.await_count == 1
    assert process.await_count == 0
    assert callback.message.answer.await_count == 0
    assert 42 not in handlers.user_message_buffers
    assert "Could not resolve AI button label" in caplog.text
    assert "Кнопка" not in caplog.text


@pytest.mark.asyncio
async def test_indexed_special_actions_still_dispatch(monkeypatch):
    handlers._ai_button_claims.clear()
    handlers.user_message_buffers.clear()
    bot = SimpleNamespace(send_message=AsyncMock())
    targets = {
        "start_test": AsyncMock(),
        "topics": AsyncMock(),
        "topic_42": AsyncMock(),
        "new_dialogue": AsyncMock(),
        "main_menu": AsyncMock(),
        "subscription": AsyncMock(),
        "referral": AsyncMock(),
    }
    monkeypatch.setattr(handlers, "_start_test_from_ai_directive", targets["start_test"])
    monkeypatch.setattr(handlers, "select_topic_menu", targets["topics"])
    monkeypatch.setattr(handlers, "process_topic_selection", targets["topic_42"])
    monkeypatch.setattr(handlers, "ask_delete_history", targets["new_dialogue"])
    monkeypatch.setattr(handlers.kb, "main_client_keyboard", AsyncMock(return_value=None))
    monkeypatch.setattr(handlers, "show_subscription_info", targets["subscription"])
    monkeypatch.setattr(handlers, "show_referral_info", targets["referral"])

    for index, action in enumerate(targets):
        callback_data = f"ai_btn:{action}|{index:02x}"
        markup = InlineKeyboardMarkup(inline_keyboard=[[
            InlineKeyboardButton(text="Действие", callback_data=callback_data),
        ]])
        message = _ButtonMessage(200 + index, markup)
        if action == "main_menu":
            message.answer = targets["main_menu"]
        callback = (
            _button_callback_with_model_copy(message, callback_data)
            if action == "topic_42"
            else _button_callback(message, callback_data)
        )
        await handlers.process_response_button(callback, None, bot)

    for target in targets.values():
        expected_count = 2 if target is targets["main_menu"] else 1
        assert target.await_count == expected_count


@pytest.mark.asyncio
async def test_ai_button_is_claimed_once_echoes_label_and_removes_keyboard():
    handlers._ai_button_claims.clear()
    handlers.user_message_buffers.clear()
    message = _ButtonMessage()
    callback = _button_callback(message)
    bot = SimpleNamespace(send_message=AsyncMock())

    process = AsyncMock()
    with patch.object(handlers, "process_buffered_messages", process):
        await handlers.process_response_button(callback, None, bot)
        await handlers.process_response_button(callback, None, bot)

    assert process.await_count == 1
    process.assert_awaited_once_with(
        42,
        bot,
        None,
        visible_user_text="Пропустить",
    )
    assert callback.answer.await_count == 2
    assert message.edit_calls == [None]
    message.answer.assert_awaited_once_with("Ответ принят: Пропустить", parse_mode=None)
    assert handlers.user_message_buffers[42] == [
        '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку "Пропустить" (hero_skip)]'
    ]


@pytest.mark.asyncio
async def test_different_originating_button_messages_are_independent():
    handlers._ai_button_claims.clear()
    handlers.user_message_buffers.clear()
    bot = SimpleNamespace(send_message=AsyncMock())
    process = AsyncMock()

    with patch.object(handlers, "process_buffered_messages", process):
        await handlers.process_response_button(_button_callback(_ButtonMessage(77)), None, bot)
        await handlers.process_response_button(_button_callback(_ButtonMessage(78)), None, bot)

    assert process.await_count == 2


def test_url_buttons_still_use_url_markup():
    markup = handlers._telegram_response_buttons_markup([
        [
            SimpleNamespace(text="Сайт", kind="url", value="https://example.com"),
            SimpleNamespace(text="Действие", kind="action", value="hero_skip"),
        ]
    ])
    assert markup.inline_keyboard[0][0].url == "https://example.com"
    assert markup.inline_keyboard[0][1].callback_data == "ai_btn:hero_skip"


def test_processing_defaults_and_validation():
    assert BotGeneralConfig.ai_processing_message_enabled.default.arg is False
    assert BotGeneralConfig.ai_processing_message_text.default.arg == DEFAULT_AI_PROCESSING_MESSAGE_TEXT
    assert handlers.normalize_ai_processing_message_text("  Думаю...  ") == "Думаю..."
    with pytest.raises(ValueError):
        handlers.normalize_ai_processing_message_text(" \t\n")
    with pytest.raises(ValueError):
        handlers.normalize_ai_processing_message_text("x" * (AI_PROCESSING_MESSAGE_MAX_LENGTH + 1))

    markup = keyboards.admin_general_settings_keyboard(SimpleNamespace(
        profile_collect_name=True,
        profile_collect_gender=True,
        profile_collect_age=False,
        ai_processing_message_enabled=False,
        ai_processing_message_text=DEFAULT_AI_PROCESSING_MESSAGE_TEXT,
    ))
    callbacks = {
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if button.callback_data
    }
    labels = [button.text for row in markup.inline_keyboard for button in row]
    assert "admin_general_toggle_ai_processing_message" in callbacks
    assert "admin_general_edit_ai_processing_message_text" in callbacks
    assert any("Показывать сообщение во время ответа ИИ" in label for label in labels)
    assert any("Текст сообщения ожидания" in label for label in labels)


def test_button_ai_context_is_used_without_changing_visible_history():
    enriched = '[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь нажал кнопку "Пропустить" (hero_skip)]'
    message = SimpleNamespace(
        role="user",
        content="Пропустить",
        ai_context_content=enriched,
        topic_id=None,
        topic=None,
    )
    selected = select_ai_history_messages([message], limit_first=5, limit_recent=15)
    assert selected[0].content == enriched
    assert message.content == "Пропустить"


class _AdminSession:
    def __init__(self, config):
        self.config = config
        self.commits = 0

    async def get(self, _model, _key):
        return self.config

    async def commit(self):
        self.commits += 1


class _AdminSessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *_args):
        return False


@pytest.mark.asyncio
async def test_admin_processing_toggle_persists(monkeypatch):
    config = SimpleNamespace(
        ai_processing_message_enabled=False,
        ai_processing_message_text="Думаю...",
    )
    session = _AdminSession(config)
    callback = SimpleNamespace(answer=AsyncMock())
    monkeypatch.setattr(handlers, "async_session_maker", lambda: _AdminSessionContext(session))
    monkeypatch.setattr(handlers, "admin_general_settings", AsyncMock())

    await handlers.admin_toggle_ai_processing_message(callback)

    assert config.ai_processing_message_enabled is True
    assert session.commits == 1


@pytest.mark.asyncio
async def test_admin_processing_text_back_returns_to_general_settings(monkeypatch):
    callback = SimpleNamespace(
        data="cancel_state_admin_general_settings",
        message=SimpleNamespace(),
        from_user=SimpleNamespace(id=42),
        bot=SimpleNamespace(),
        answer=AsyncMock(),
    )
    state = SimpleNamespace(clear=AsyncMock())
    general_settings = AsyncMock()
    monkeypatch.setattr(handlers, "admin_general_settings", general_settings)

    await handlers.cancel_handler(callback, state)

    state.clear.assert_awaited_once_with()
    general_settings.assert_awaited_once_with(callback)
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_topic_prompt_back_returns_to_topic_editor_menu(monkeypatch):
    callback = SimpleNamespace(
        data="cancel_state_edit_topic_17",
        message=SimpleNamespace(
            chat=SimpleNamespace(id=42),
            message_id=314,
        ),
        from_user=SimpleNamespace(id=42),
        bot=SimpleNamespace(),
        answer=AsyncMock(),
    )
    state = SimpleNamespace(clear=AsyncMock())
    show_topic_menu = AsyncMock()
    monkeypatch.setattr(handlers, "_show_edit_topic_menu", show_topic_menu)

    await handlers.cancel_handler(callback, state)

    state.clear.assert_awaited_once_with()
    show_topic_menu.assert_awaited_once_with(
        bot=callback.bot,
        chat_id=42,
        message_id=314,
        topic_id=17,
    )
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_admin_processing_text_persists_and_rejects_empty(monkeypatch):
    config = SimpleNamespace(
        ai_processing_message_enabled=True,
        ai_processing_message_text="Думаю...",
    )
    session = _AdminSession(config)
    state = SimpleNamespace(
        get_data=AsyncMock(return_value={}),
        clear=AsyncMock(),
    )
    message = SimpleNamespace(
        text="  Подождите немного  ",
        chat=SimpleNamespace(id=99),
        delete=AsyncMock(),
        answer=AsyncMock(),
    )
    bot = SimpleNamespace(edit_message_text=AsyncMock())
    monkeypatch.setattr(handlers, "async_session_maker", lambda: _AdminSessionContext(session))
    monkeypatch.setattr(handlers, "admin_general_settings", AsyncMock())

    await handlers.admin_save_ai_processing_message_text(message, state, bot)

    assert config.ai_processing_message_text == "Подождите немного"
    assert session.commits == 1
    assert state.clear.await_count == 1

    empty_message = SimpleNamespace(text=" \n", answer=AsyncMock())
    await handlers.admin_save_ai_processing_message_text(empty_message, state, bot)
    assert config.ai_processing_message_text == "Подождите немного"
    empty_message.answer.assert_awaited_once()


class _ProcessingMessage:
    def __init__(self, events, *, fail_delete=False):
        self.events = events
        self.fail_delete = fail_delete

    async def delete(self):
        self.events.append("delete")
        if self.fail_delete:
            raise RuntimeError("cleanup race")


class _FakeSession:
    def __init__(self, module, general_config):
        self.module = module
        self.general_config = general_config
        self.user = SimpleNamespace(id=42, current_topic_id=None, current_dialogue_id=1)
        self.added = []

    async def get(self, model, _key):
        if model is self.module.AIConfig:
            return SimpleNamespace(provider="Gemini", gemini_model="test-model")
        if model is self.module.BotGeneralConfig:
            return self.general_config
        if model is self.module.User:
            return self.user
        return None

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        return None


class _FakeSessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *_args):
        return False


class _FakeBot:
    def __init__(self, *, fail_cleanup=False):
        self.events = []
        self.fail_cleanup = fail_cleanup

    async def send_chat_action(self, **_kwargs):
        self.events.append("typing")
        await asyncio.sleep(0)

    async def send_message(self, *, text, **_kwargs):
        self.events.append(text)
        await asyncio.sleep(0)
        if text in {"Думаю...", "Подождите"}:
            return _ProcessingMessage(self.events, fail_delete=self.fail_cleanup)
        return SimpleNamespace()


async def _run_buffered(monkeypatch, *, enabled, text="Думаю...", response="Ответ готов", fail_cleanup=False):
    config = SimpleNamespace(
        ai_processing_message_enabled=enabled,
        ai_processing_message_text=text,
    )
    session = _FakeSession(handlers, config)
    bot = _FakeBot(fail_cleanup=fail_cleanup)
    handlers.user_message_buffers[42] = ["hello"]
    monkeypatch.setattr(
        handlers,
        "async_session_maker",
        lambda: _FakeSessionContext(session),
    )
    monkeypatch.setattr(handlers, "_resend_active_spread_choice", AsyncMock(return_value=False))
    monkeypatch.setattr(handlers, "handle_ai_media_content", AsyncMock(return_value=(response, [], [], [], [], [])))
    generate = AsyncMock(return_value=response)
    monkeypatch.setattr(handlers.ai_integration, "generate_response", generate)
    await handlers.process_buffered_messages(42, bot)
    return bot, generate


@pytest.mark.asyncio
async def test_processing_message_lifecycle_preserves_typing_and_final_response(monkeypatch):
    bot, generate = await _run_buffered(monkeypatch, enabled=True, text="Подождите")
    assert generate.await_count == 1
    assert bot.events.index("Подождите") < bot.events.index("Ответ готов")
    assert bot.events[-1] == "delete"
    assert "typing" in bot.events


@pytest.mark.asyncio
async def test_processing_message_disabled_keeps_typing_without_extra_text(monkeypatch):
    bot, _ = await _run_buffered(monkeypatch, enabled=False)
    assert "Думаю..." not in bot.events
    assert "typing" in bot.events
    assert "Ответ готов" in bot.events


@pytest.mark.asyncio
async def test_processing_cleanup_failure_does_not_hide_success(monkeypatch):
    bot, _ = await _run_buffered(monkeypatch, enabled=True, fail_cleanup=True)
    assert "Ответ готов" in bot.events
    assert bot.events[-1] == "delete"


@pytest.mark.asyncio
async def test_processing_message_is_cleaned_on_ai_failure(monkeypatch):
    config = SimpleNamespace(ai_processing_message_enabled=True, ai_processing_message_text="Думаю...")
    session = _FakeSession(handlers, config)
    bot = _FakeBot()
    handlers.user_message_buffers[42] = ["hello"]
    monkeypatch.setattr(handlers, "async_session_maker", lambda: _FakeSessionContext(session))
    monkeypatch.setattr(handlers, "_resend_active_spread_choice", AsyncMock(return_value=False))
    monkeypatch.setattr(handlers, "_report_ai_failure", AsyncMock())
    monkeypatch.setattr(
        handlers.ai_integration,
        "generate_response",
        AsyncMock(side_effect=handlers.AIServiceError("provider failed")),
    )

    await handlers.process_buffered_messages(42, bot)

    assert "Думаю..." in bot.events
    assert bot.events[-1] == "delete"


def test_init_db_additively_migrates_general_settings():
    script = textwrap.dedent(
        """
        import asyncio
        from sqlalchemy import select
        from database import BotGeneralConfig, async_session_maker, engine, init_db

        async def main():
            async with engine.begin() as connection:
                await connection.exec_driver_sql(
                    "CREATE TABLE bot_general_config ("
                    "id INTEGER PRIMARY KEY, "
                    "profile_collect_name BOOLEAN NOT NULL, "
                    "profile_collect_gender BOOLEAN NOT NULL, "
                    "profile_collect_age BOOLEAN NOT NULL)"
                )
                await connection.exec_driver_sql(
                    "INSERT INTO bot_general_config VALUES (1, 0, 1, 0)"
                )
            await init_db()
            async with async_session_maker() as session:
                config = await session.scalar(select(BotGeneralConfig).where(BotGeneralConfig.id == 1))
                assert config.profile_collect_name is False
                assert config.profile_collect_gender is True
                assert config.ai_processing_message_enabled is False
                assert config.ai_processing_message_text == "Думаю..."

        asyncio.run(main())
        """
    )
    env = os.environ.copy()
    env["BOT_TOKEN"] = "test"
    with tempfile.TemporaryDirectory() as tmpdir:
        env["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmpdir}/migration.db"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    assert completed.returncode == 0, completed.stderr or completed.stdout
