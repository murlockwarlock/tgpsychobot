import json
import os
import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

os.environ.setdefault("BOT_TOKEN", "123456:test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from aiogram import Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.methods import AnswerCallbackQuery, DeleteMessage, EditMessageText, GetChat, GetMe, SendMessage
from aiogram.types import CallbackQuery, Chat, Message, MessageEntity, Update, User
from aiogram.client.session.base import BaseSession
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database import (
    AIConfig,
    Base,
    BotGeneralConfig,
    BotAddressingConfig,
    BotSystemMessage,
    DEFAULT_AI_PROCESSING_MESSAGE_TEXT,
    BotTranslation,
    Content,
    ContentMedia,
    SubscriptionBenefitGrant,
    SubscriptionConfig,
    TelegramStartIntent,
    TestConfig as DBTestConfig,
    TrialUsageHistory,
    User as DBUser,
    UserMenuBinding,
    UserSubscription,
)
from translation_service import source_hash
from system_messages import (
    SYSTEM_MESSAGE_CATALOG,
    SYSTEM_MESSAGE_PAGE_SIZE,
    parse_system_message_callback,
    system_message_page_count,
    system_message_callback_data,
)


@pytest_asyncio.fixture
async def factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions() as session:
        session.add(
            BotGeneralConfig(
                id=1,
                telegram_enabled_languages='["ru", "en", "pt"]',
                multilingual_authoring_enabled=True,
            )
        )
        await session.commit()
    yield sessions
    await engine.dispose()


@pytest.fixture(scope="module")
def admin_dispatcher():
    import admin_content_authoring
    import automation_admin
    import handlers

    if handlers.router.parent_router is not None:
        handlers = importlib.reload(handlers)
    if automation_admin.router.parent_router is not None:
        automation_admin = importlib.reload(automation_admin)
    if admin_content_authoring.router.parent_router is not None:
        admin_content_authoring = importlib.reload(admin_content_authoring)

    dispatcher = Dispatcher(storage=MemoryStorage())
    dispatcher.include_router(handlers.router)
    dispatcher.include_router(automation_admin.router)
    dispatcher.include_router(admin_content_authoring.router)
    yield dispatcher
    for module in (admin_content_authoring, automation_admin, handlers):
        if module.router.parent_router is not None:
            importlib.reload(module)


class RecordingSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def close(self):
        return None

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        return True

    async def stream_content(self, *args, **kwargs):
        if False:
            yield b""


class ValidatingTelegramSession(BaseSession):
    def __init__(self, *, stateful=False):
        super().__init__()
        self.calls = []
        self.stateful = stateful
        self._screens = {}
        self._active_screen_keys = {}
        self._next_message_ids = {}
        self.next_edit_error = None

    async def close(self):
        return None

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        type(method).model_validate(method.model_dump())

        if isinstance(method, EditMessageText) and len(method.text or "") > 4096:
            raise TelegramBadRequest(method, "Bad Request: message is too long")

        if not self.stateful and isinstance(method, (EditMessageText, SendMessage)):
            return Message(
                message_id=getattr(method, "message_id", 1),
                date=datetime.now(timezone.utc),
                chat=Chat(id=int(method.chat_id), type="private"),
                from_user=User(id=999, is_bot=True, first_name="TestBot", username="testbot"),
                text=getattr(method, "text", ""),
            ).as_(bot)
        if isinstance(method, EditMessageText):
            chat_id = int(method.chat_id)
            message_id = int(method.message_id)
            screen_key = (chat_id, message_id)
            if self.next_edit_error:
                message = self.next_edit_error
                self.next_edit_error = None
                raise TelegramBadRequest(method, message)
            requested = self._screen_payload(method.text, method.entities, method.reply_markup)
            if self._screens.get(screen_key) == requested:
                raise TelegramBadRequest(method, "Bad Request: message is not modified")
            self._screens[screen_key] = requested
            self._active_screen_keys[chat_id] = screen_key
            return self._screen_message(bot, chat_id, message_id, requested)
        if isinstance(method, SendMessage):
            chat_id = int(method.chat_id)
            message_id = self._next_message_ids.get(chat_id, 1)
            self._next_message_ids[chat_id] = message_id + 1
            requested = self._screen_payload(method.text, method.entities, method.reply_markup)
            screen_key = (chat_id, message_id)
            self._screens[screen_key] = requested
            self._active_screen_keys[chat_id] = screen_key
            return self._screen_message(bot, chat_id, message_id, requested)
        if isinstance(method, GetMe):
            return User(id=999, is_bot=True, first_name="TestBot", username="testbot").as_(bot)
        if isinstance(method, GetChat):
            return Chat(id=method.chat_id, type="private", first_name="Admin", username="admin").as_(bot)
        return True

    async def stream_content(self, *args, **kwargs):
        if False:
            yield b""

    @staticmethod
    def _model_value(value):
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json", exclude_none=True)
        return value

    @classmethod
    def _screen_payload(cls, text, entities, reply_markup):
        return (
            text,
            cls._model_value(entities),
            cls._model_value(reply_markup),
        )

    @staticmethod
    def _screen_message(bot, chat_id, message_id, payload):
        text, entities, reply_markup = payload
        return Message(
            message_id=message_id,
            date=datetime.now(timezone.utc),
            chat=Chat(id=chat_id, type="private"),
            from_user=User(id=999, is_bot=True, first_name="TestBot", username="testbot"),
            text=text,
            entities=entities,
            reply_markup=reply_markup,
        ).as_(bot)

    def current_message(self, message):
        if not self.stateful:
            return message
        chat_id = message.chat.id
        screen_key = (chat_id, message.message_id)
        if screen_key not in self._screens:
            screen_key = self._active_screen_keys.get(chat_id)
        if screen_key is None:
            return message
        message_id = screen_key[1]
        payload = self._screens[screen_key]
        current = self._screen_message(message.bot, chat_id, message_id, payload)
        return current.model_copy(update={"from_user": message.from_user, "chat": message.chat}).as_(message.bot)

    def message_for(self, bot, chat_id, message_id, user):
        payload = self._screens[(int(chat_id), int(message_id))]
        current = self._screen_message(bot, int(chat_id), int(message_id), payload)
        return current.model_copy(update={"from_user": user, "chat": current.chat}).as_(bot)

    def latest_message_with_keyboard(self, bot, chat_id, user):
        candidates = [
            key for key, payload in self._screens.items()
            if key[0] == int(chat_id) and payload[2] is not None
        ]
        message_id = max(candidates, key=lambda item: item[1])[1]
        return self.message_for(bot, chat_id, message_id, user)


def _callback(bot, message, data):
    return SimpleNamespace(
        bot=bot,
        message=message,
        from_user=message.from_user,
        data=data,
        answer=AsyncMock(),
    )


def _admin_message(bot, user_id=11, message_id=1, text="Админ-панель"):
    user = User(id=user_id, is_bot=False, first_name="Admin")
    return Message(
        message_id=message_id,
        date=datetime.now(timezone.utc),
        chat=Chat(id=user_id, type="private"),
        from_user=user,
        text=text,
    ).as_(bot)


async def _feed_callback(dispatcher, bot, message, data, update_id):
    session = getattr(bot, "session", None)
    if isinstance(session, ValidatingTelegramSession) and session.stateful:
        message = session.current_message(message)
    user = message.from_user
    callback = CallbackQuery(
        id=f"callback-{update_id}",
        from_user=user,
        chat_instance="admin",
        message=message,
        data=data,
    ).as_(bot)
    return await dispatcher.feed_update(
        bot,
        Update(update_id=update_id, callback_query=callback),
    )


def _ai_callback_buttons(markup):
    return [
        button
        for row in (markup.inline_keyboard if markup else [])
        for button in row
        if button.callback_data
    ]


def _classify_telegram_ai_callback(callback_data):
    navigation = {
        "admin_ai_settings",
        "admin_ai_keys",
        "admin_ai_main_chat",
        "admin_ai_main_chat_provider",
        "admin_ai_main_chat_model",
        "admin_ai_text_fallback",
        "admin_ai_fallback_provider",
        "admin_ai_fallback_model",
        "admin_ai_audio",
        "admin_audio_model",
        "admin_ai_vision",
        "admin_change_vision_model",
        "admin_ai_vision_fallback",
        "admin_ai_vision_fallback_provider",
        "admin_ai_vision_fallback_model",
        "admin_ai_image_generation",
        "admin_change_image_generation_model",
        "admin_ai_image_edit",
        "admin_change_image_edit_model",
        "admin_ai_common",
        "admin_edit_system_prompt",
        "admin_edit_shared_prompt_block",
        "admin_edit_service_prompt_block",
        "admin_ai_logs_0_all",
        "admin_panel",
        "admin_select_transcription_provider",
        "admin_select_vision_provider",
        "admin_select_image_generation_provider",
        "admin_select_image_edit_provider",
    }
    if callback_data in navigation or callback_data.startswith((
        "view_models_",
        "view_provider_models_",
        "cancel_state_",
        "ai_ppx_",
        "noop",
    )):
        return "navigation"
    if callback_data.startswith((
        "ai_provider_",
        "set_key_",
        "model_setting_",
        "model_reasoning_",
        "ai_m_",
        "admin_choose_capability_",
        "admin_ai_fallback_set_provider_",
        "admin_ai_vision_fallback_set_provider_",
        "admin_ai_main_chat_set_",
    )):
        return "mutation"
    if callback_data in {
        "admin_ai_fallback_toggle",
        "admin_ai_vision_fallback_toggle",
        "admin_ai_image_generation_toggle",
        "admin_ai_image_edit_toggle",
        "admin_ai_deepseek_proxy",
        "set_audio_limit",
        "set_context_first",
        "set_context_recent",
        "toggle_preserve_topic_context",
        "set_ai_timeout",
        "set_kie_credit_threshold",
    }:
        return "mutation"
    if callback_data.startswith("ai_retry:"):
        return "external"
    raise AssertionError(f"unclassified changed Telegram AI callback: {callback_data}")


def _classify_telegram_ai_markup(markup, contracts):
    buttons = _ai_callback_buttons(markup)
    assert buttons
    for button in buttons:
        category = _classify_telegram_ai_callback(button.callback_data)
        contracts[button.callback_data] = category
    return buttons


@pytest.mark.asyncio
async def test_content_resource_fields_and_card_have_one_primary_editor(factory, monkeypatch):
    import admin_content_authoring as module

    monkeypatch.setattr(module, "async_session_maker", factory)
    async with factory() as session:
        session.add(Content(key="about_me", button_title="Об авторе", text_content="<b>Текст</b>"))
        await session.commit()

    assert [field[0] for field in module.AUTHORING_RESOURCES["content"].fields] == [
        "button_title",
        "text_content",
    ]

    recording = RecordingSession()
    bot = Bot("123456:TEST", session=recording)
    user = User(id=11, is_bot=False, first_name="Admin")
    message = Message(
        message_id=1,
        date=datetime.now(timezone.utc),
        chat=Chat(id=11, type="private"),
        from_user=user,
        text="x",
    ).as_(bot)
    callback = _callback(bot, message, "ca:view:content:about_me:ru:0")
    await module.resource_card(callback, "content", "about_me", "ru", 0)
    request = recording.calls[-1]
    texts = [button.text for row in request.reply_markup.inline_keyboard for button in row]
    assert texts.count("✏️ Изменить: контент") == 1
    assert not any(text == "Медиа и общие настройки" for text in texts)


@pytest.mark.asyncio
async def test_content_list_has_fifteen_items_and_excludes_technical_rows(factory, monkeypatch):
    import admin_content_authoring as module

    monkeypatch.setattr(module, "async_session_maker", factory)
    async with factory() as session:
        session.add_all(
            [
                Content(key=f"content_{index:02d}", button_title=f"Кнопка {index}")
                for index in range(16)
            ]
            + [
                Content(key="test_button", button_title="Тест"),
                Content(key="test_intro", button_title="Вступление"),
                Content(key="test_results", button_title="Результаты"),
                Content(key="secret_test_outro", button_title="Финал"),
            ]
        )
        await session.commit()

    recording = RecordingSession()
    bot = Bot("123456:TEST", session=recording)
    user = User(id=11, is_bot=False, first_name="Admin")
    message = Message(
        message_id=1,
        date=datetime.now(timezone.utc),
        chat=Chat(id=11, type="private"),
        from_user=user,
        text="x",
    ).as_(bot)
    callback = _callback(bot, message, "ca:list:content:0")
    await module.resource_list(callback, "content", 0)
    request = recording.calls[-1]
    rows = request.reply_markup.inline_keyboard
    content_buttons = [button for row in rows for button in row if button.callback_data.startswith("ca:view:content:")]
    assert len(content_buttons) == 15
    assert any(button.callback_data == "ca:list:content:1" for row in rows for button in row)
    assert not any("test_" in button.text or "secret_test_outro" in button.text for row in rows for button in row)


@pytest.mark.asyncio
async def test_content_list_uses_human_first_labels_and_card_keeps_machine_id(factory, monkeypatch):
    import admin_content_authoring as module

    monkeypatch.setattr(module, "async_session_maker", factory)
    async with factory() as session:
        session.add_all([
            Content(key="menu", button_title="Старое меню", text_content="menu"),
            Content(key="start_message", button_title="Старт", text_content="start"),
            Content(key="disclaimer", button_title="Правила", text_content="rules"),
            Content(key="about_me", button_title="Об авторе", text_content="about"),
            Content(key="instruction_bot", button_title="Инструкция", text_content="instruction"),
            Content(key="btn_4d49f0df8b", button_title="Записаться на сессию", text_content="book"),
        ])
        await session.commit()

    recording = RecordingSession()
    bot = Bot("123456:TEST", session=recording)
    message = _admin_message(bot)
    await module.resource_list(_callback(bot, message, "ca:list:content:0"), "content", 0)
    rows = recording.calls[-1].reply_markup.inline_keyboard
    labels = [button.text for row in rows for button in row]
    assert "Меню" in labels
    assert "Приветствие (/start)" in labels
    assert "Дисклеймер" in labels
    assert "Об авторе" in labels
    assert "Инструкция" in labels
    assert "Записаться на сессию" in labels
    assert "about_me" not in labels
    assert "instruction_bot" not in labels
    assert "btn_4d49f0df8b" not in labels
    assert "menu: Меню" not in labels
    assert next(
        button.callback_data
        for row in rows
        for button in row
        if button.text == "Об авторе"
    ) == "ca:view:content:about_me:0"
    assert next(
        button.callback_data
        for row in rows
        for button in row
        if button.text == "Записаться на сессию"
    ) == "ca:view:content:btn_4d49f0df8b:0"

    await module.resource_card(
        _callback(bot, message, "ca:view:content:menu:ru:0"),
        "content",
        "menu",
        "ru",
        0,
    )
    assert "ID: <code>menu</code>" in recording.calls[-1].text

    await module.resource_card(
        _callback(bot, message, "ca:view:content:about_me:0"),
        "content",
        "about_me",
        "ru",
        0,
    )
    assert "ID: <code>about_me</code>" in recording.calls[-1].text
    await module.resource_card(
        _callback(bot, message, "ca:view:content:btn_4d49f0df8b:0"),
        "content",
        "btn_4d49f0df8b",
        "ru",
        0,
    )
    assert "ID: <code>btn_4d49f0df8b</code>" in recording.calls[-1].text
    await module.resource_list(
        _callback(bot, message, "ca:list:content:0"),
        "content",
        0,
    )
    listed_again = recording.calls[-1]
    assert any(
        button.text == "Об авторе"
        and button.callback_data == "ca:view:content:about_me:0"
        for row in listed_again.reply_markup.inline_keyboard
        for button in row
    )


@pytest.mark.asyncio
async def test_perplexity_model_screen_omits_unrepresented_rub_pricing(factory, monkeypatch):
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    async with factory() as db:
        db.add(AIConfig(id=1, provider="Perplexity", perplexity_model="low"))
        await db.commit()
    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    message = _admin_message(bot)
    await handlers.view_models_by_provider(_callback(bot, message, "view_models_Perplexity"))
    rendered = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    assert "Провайдер:" in rendered.text
    assert "Прайсинг" not in rendered.text
    assert "руб" not in rendered.text.lower()
    assert "Perplexity" in rendered.text
    assert isinstance(rendered.text, str)


@pytest.mark.asyncio
async def test_test_content_is_reachable_from_test_management_and_saves(factory, admin_dispatcher, monkeypatch):
    import admin_content_authoring
    import automation_admin
    import handlers
    import keyboards

    for module in (admin_content_authoring, automation_admin, handlers, keyboards):
        monkeypatch.setattr(module, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(automation_admin, "get_all_admin_ids", AsyncMock(return_value={11}))
    monkeypatch.setattr(handlers, "send_temp_notification", AsyncMock())

    async with factory() as session:
        session.add_all([
            DBUser(id=11, is_admin=True, first_name="Admin"),
            DBTestConfig(id=1, is_enabled=True, secret_test_enabled=True),
            Content(key="test_intro", text_content="Intro", is_visible=True),
            Content(key="test_results", text_content="Results", is_visible=True),
            Content(key="secret_test_outro", text_content="Final", is_visible=True),
        ])
        await session.commit()

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    message = _admin_message(bot)
    dispatcher = admin_dispatcher
    await _feed_callback(dispatcher, bot, message, "admin_test_menu", 1100)
    test_menu = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    callbacks = [
        button.callback_data
        for row in test_menu.reply_markup.inline_keyboard
        for button in row
        if button.callback_data
    ]
    for index, callback_data in enumerate((
        "edit_content_test_intro",
        "edit_content_test_results",
        "edit_content_secret_test_outro",
    ), start=1):
        assert callback_data in callbacks
        await _feed_callback(dispatcher, bot, message, callback_data, 1100 + index)
        editor = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
        assert "Редактирование:" in editor.text
        content_key = callback_data.removeprefix("edit_content_")
        if index < 3:
            await dispatcher.feed_update(
                bot,
                Update(
                    update_id=1110 + index,
                    message=_admin_message(bot, text=f"Обновлено {content_key}"),
                ),
            )
            await _feed_callback(dispatcher, bot, message, f"save_content_{content_key}", 1120 + index)
        else:
            await _feed_callback(dispatcher, bot, message, f"cancel_content_edit_{content_key}", 1120 + index)
        returned = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
        assert "Управление разделом 'Тест'" in returned.text


@pytest.mark.asyncio
async def test_test_button_uses_ui_translation_source_for_keyboard_and_filter(factory, monkeypatch):
    import handlers
    import keyboards

    monkeypatch.setattr(keyboards, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "async_session_maker", factory)
    async with factory() as session:
        session.add_all([
            DBUser(id=42, telegram_language_code="ru"),
            DBTestConfig(id=1, is_enabled=True),
            Content(key="test_button", button_title="LEGACY TEST LABEL", is_visible=True),
            SubscriptionConfig(id=1, subscriptions_enabled=False),
        ])
        await session.commit()

    markup = await keyboards.main_client_keyboard(42)
    labels = [button.text for row in markup.keyboard for button in row]
    assert "📝 Пройти тест" in labels
    assert "LEGACY TEST LABEL" not in labels

    message = _admin_message(Bot("123456:TEST", session=RecordingSession()), user_id=42, text="📝 Пройти тест")
    assert await handlers.TestButtonFilter()(message)

    from translation_registry import build_translation_registry
    async with factory() as session:
        registry = await build_translation_registry(session)
        assert registry.get("ui.button.test") is not None
        assert registry.get("content.test_button.button_title") is None


@pytest.mark.asyncio
async def test_action_button_editor_is_scoped_to_start_message(factory, monkeypatch):
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    async with factory() as session:
        session.add_all([
            Content(key="about_me", button_title="Об авторе", text_content="Текст"),
            Content(
                key="start_message",
                button_title="Приветствие",
                text_content="Старт",
                action_btn_text="Начать",
                action_btn_payload="начать",
            ),
        ])
        await session.commit()

    ordinary_state = SimpleNamespace(get_data=AsyncMock(return_value={"content_key": "about_me", "text_content": "Текст"}))
    ordinary_text, _ = await handlers.get_content_display(ordinary_state)
    assert "Кнопка действия" not in ordinary_text

    start_state = SimpleNamespace(get_data=AsyncMock(return_value={"content_key": "start_message", "text_content": "Старт"}))
    start_text, _ = await handlers.get_content_display(start_state)
    assert "Кнопка действия" in start_text


@pytest.mark.asyncio
async def test_content_list_hides_pagination_at_fifteen(factory, monkeypatch):
    import admin_content_authoring as module

    monkeypatch.setattr(module, "async_session_maker", factory)
    async with factory() as session:
        session.add_all(
            [Content(key=f"content_{index:02d}", button_title=f"Кнопка {index}") for index in range(15)]
        )
        await session.commit()

    recording = RecordingSession()
    bot = Bot("123456:TEST", session=recording)
    user = User(id=11, is_bot=False, first_name="Admin")
    message = Message(
        message_id=1,
        date=datetime.now(timezone.utc),
        chat=Chat(id=11, type="private"),
        from_user=user,
        text="x",
    ).as_(bot)
    await module.resource_list(_callback(bot, message, "ca:list:content:0"), "content", 0)
    rows = recording.calls[-1].reply_markup.inline_keyboard
    callbacks = [button.callback_data for row in rows for button in row]
    assert "ca:list:content:1" not in callbacks
    assert not any(button.text in {"⬅️ Назад", "Далее ➡️"} for row in rows for button in row)


@pytest.mark.asyncio
async def test_content_dependency_report_blocks_unsafe_delete_and_rename(factory):
    from admin_content_authoring import content_dependency_report

    async with factory() as session:
        session.add_all([
            Content(key="about_me", button_title="Об авторе", text_content="Основной текст"),
            Content(key="other", button_title="Другое", text_content="[Открыть](btn:svc:content:about_me)"),
        ])
        session.add(UserMenuBinding(user_id=7, label="Об авторе", resource_kind="content", resource_id="about_me"))
        session.add(BotTranslation(
            locale="pt",
            translation_key="content.about_me.text_content",
            text="Sobre",
            source_hash=source_hash("Русский текст"),
        ))
        await session.commit()
        report = await content_dependency_report(session, "about_me")

    assert report
    assert any("пользовател" in item.lower() for item in report)
    assert any("текст" in item.lower() or "ссыл" in item.lower() for item in report)


@pytest.mark.asyncio
async def test_core_content_delete_reports_system_dependency(factory):
    from admin_content_authoring import content_dependency_report

    async with factory() as session:
        session.add(Content(key="start_message", text_content="Привет"))
        await session.commit()
        report = await content_dependency_report(session, "start_message")

    assert any("системным маршрутом" in item for item in report)


@pytest.mark.asyncio
async def test_menu_label_card_stays_ru_only_when_multilingual_off(factory, monkeypatch):
    import admin_content_authoring as module
    from database import SubscriptionConfig

    monkeypatch.setattr(module, "async_session_maker", factory)
    async with factory() as session:
        config = await session.get(BotGeneralConfig, 1)
        config.multilingual_authoring_enabled = False
        session.add(SubscriptionConfig(id=1, topics_btn_name="Темы", referral_btn_name="Рефералы"))
        await session.commit()

    recording = RecordingSession()
    bot = Bot("123456:TEST", session=recording)
    user = User(id=11, is_bot=False, first_name="Admin")
    message = Message(
        message_id=1,
        date=datetime.now(timezone.utc),
        chat=Chat(id=11, type="private"),
        from_user=user,
        text="x",
    ).as_(bot)
    callback = _callback(bot, message, "admin_menu_labels")
    await module.menu_labels_card(callback, SimpleNamespace(clear=AsyncMock()))
    markup = recording.calls[-1].reply_markup
    callbacks = [button.callback_data for row in markup.inline_keyboard for button in row]
    assert not any(item.startswith("ca:locale:") for item in callbacks)
    assert any(item.startswith("ca:edit:subscription_config:1:ru:") for item in callbacks)


def test_admin_preview_preserves_supported_html_and_escapes_invalid_markup():
    from handlers import render_admin_content_preview

    assert render_admin_content_preview("<b>Жирный</b> <i>текст</i>") == "<b>Жирный</b> <i>текст</i>"
    assert "&lt;b&gt;" in render_admin_content_preview("<b>незакрытый")


def test_admin_preview_truncates_formatted_html_at_safe_boundaries():
    from handlers import render_admin_content_preview
    from admin_html_preview import _HTMLPreviewParser

    source = (
        '<tg-emoji emoji-id="5404421747296455061">🎅</tg-emoji> Добро пожаловать в мир кинематографа\n\n'
        "Приглашаем погрузиться в увлекательное путешествие, где мы вместе с проводником составим "
        'уникальный кино-портрет твоего ребёнка <tg-emoji emoji-id="5384316232289769088">🙏</tg-emoji>\n\n'
        "Удели 10 минут"
    )

    preview = render_admin_content_preview(source, max_length=250)
    parser = _HTMLPreviewParser()
    parser.feed(preview)
    parser.finish()

    assert len(preview) <= 250
    assert "<tg-emoji emoji-id=\"5384316232289769088\">" not in preview
    assert preview.endswith("…")


def test_admin_preview_escapes_incomplete_and_unescaped_html():
    from handlers import render_admin_content_preview

    assert "&lt;tg-emoji" in render_admin_content_preview('<tg-emoji emoji-id="123')
    assert "&amp;" in render_admin_content_preview("a & b")


@pytest.mark.asyncio
async def test_content_card_never_sends_truncated_html(factory, monkeypatch):
    import admin_content_authoring as module
    from translation_registry import _TelegramHTMLParser

    monkeypatch.setattr(module, "async_session_maker", factory)
    source = (
        '<tg-emoji emoji-id="5404421747296455061">🎅</tg-emoji> Добро пожаловать в мир кинематографа\n\n'
        "Приглашаем погрузиться в увлекательное путешествие, где мы вместе с проводником составим "
        'уникальный кино-портрет твоего ребёнка <tg-emoji emoji-id="5384316232289769088">🙏</tg-emoji>\n\n'
        "Удели 10 минут"
    )
    async with factory() as session:
        session.add(Content(key="start_message", button_title="Старт", text_content=source))
        await session.commit()

    recording = RecordingSession()
    bot = Bot("123456:TEST", session=recording)
    await module.resource_card(
        _callback(bot, _admin_message(bot), "ca:view:content:start_message:ru:0"),
        "content",
        "start_message",
        "ru",
        0,
    )

    request = recording.calls[-1]
    parser = _TelegramHTMLParser()
    parser.feed(request.text)
    parser.finish()


def test_general_settings_exposes_existing_language_request_setting():
    import keyboards

    config = SimpleNamespace(
        profile_collect_name=True,
        profile_collect_gender=False,
        profile_collect_age=False,
        telegram_language_selection_enabled=True,
        multilingual_authoring_enabled=True,
        ai_processing_message_enabled=False,
    )
    markup = keyboards.admin_general_settings_keyboard(config)
    buttons = [button for row in markup.inline_keyboard for button in row]
    language = next(button for button in buttons if button.text.startswith("Язык:"))
    assert language.text == "Язык: ✅ запрашивать"
    assert language.callback_data == "admin_general_toggle_language_selection"


def test_system_message_catalog_uses_admin_friendly_metadata_and_platform_scope():
    import system_messages

    busy = system_messages.SYSTEM_MESSAGE_DEFINITIONS["ui.ai.busy"]
    assert busy.title == "Предыдущий запрос ещё выполняется"
    assert busy.description
    assert "ui.ai.busy" not in busy.title
    assert all(definition.description for definition in system_messages.SYSTEM_MESSAGE_CATALOG)
    assert all("ui." not in definition.title for definition in system_messages.SYSTEM_MESSAGE_CATALOG)
    assert "ui.ai.internal_failed" not in {
        definition.key
        for definition in system_messages.system_message_catalog_for_platform("telegram")
    }
    assert "ui.access.subscription_required" not in {
        definition.key
        for definition in system_messages.system_message_catalog_for_platform("max")
    }


def test_system_message_keyboard_is_friendly_and_keeps_canonical_rows():
    import keyboards

    markup = keyboards.admin_system_messages_keyboard(locale="pt", page=1)
    rows = markup.inline_keyboard
    buttons = [button for row in rows for button in row]
    assert all(button.callback_data != "noop" for button in buttons)
    assert all("ui." not in button.text for button in buttons)
    assert any(button.text == "⬅️ Назад" for button in buttons)
    assert any(button.text == "Далее ➡️" for button in buttons)
    assert any({button.text for button in row} == {"RU", "EN", "✅ PT"} for row in rows)
    assert all(len(button.callback_data.encode("utf-8")) <= 64 for button in buttons)


def test_system_message_keyboard_hides_locale_switch_when_multilingual_is_off():
    import keyboards

    markup = keyboards.admin_system_messages_keyboard(
        locale="pt",
        page=0,
        multilingual_enabled=False,
    )
    buttons = [button for row in markup.inline_keyboard for button in row]
    assert not any(
        (parse_system_message_callback(button.callback_data or "") or {}).get("action") == "l"
        for button in buttons
    )

    enabled_markup = keyboards.admin_system_messages_keyboard(
        locale="ru",
        page=0,
        multilingual_enabled=True,
        active_locales=("ru",),
    )
    locale_row = next(
        row
        for row in enabled_markup.inline_keyboard
        if any((parse_system_message_callback(button.callback_data or "") or {}).get("action") == "l" for button in row)
    )
    assert [button.text for button in locale_row] == ["✅ RU", "⚪ EN", "⚪ PT"]


def test_system_message_keyboard_hides_pagination_for_single_page(monkeypatch):
    import keyboards
    import system_messages

    monkeypatch.setattr(
        keyboards,
        "system_message_catalog_for_platform",
        lambda _platform: (system_messages.SYSTEM_MESSAGE_CATALOG[0],),
    )
    monkeypatch.setattr(keyboards, "system_message_page_count", lambda _platform: 1)
    markup = keyboards.admin_system_messages_keyboard(locale="ru", page=0)
    buttons = [button for row in markup.inline_keyboard for button in row]
    assert not any(button.text in {"⬅️ Назад", "Далее ➡️"} for button in buttons)
    assert not any(
        (parse_system_message_callback(button.callback_data or "") or {}).get("action") == "p"
        for button in buttons
    )


@pytest.mark.asyncio
async def test_telegram_system_message_locale_scope_survives_toggle_and_reopen(
    factory,
    admin_dispatcher,
    monkeypatch,
):
    import admin_content_authoring
    import automation_admin
    import handlers
    import keyboards
    from system_messages import resolve_system_text

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(admin_content_authoring, "async_session_maker", factory)
    monkeypatch.setattr(automation_admin, "async_session_maker", factory)
    monkeypatch.setattr(keyboards, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(automation_admin, "get_all_admin_ids", AsyncMock(return_value={11}))

    async with factory() as session:
        config = await session.get(BotGeneralConfig, 1)
        config.multilingual_authoring_enabled = False
        session.add_all([
            DBUser(id=11, is_admin=True, first_name="Admin"),
            AIConfig(id=1, provider="Deepseek"),
            SubscriptionConfig(id=1),
            BotSystemMessage(
                bot_scope="telegram",
                translation_key="ui.ai.thinking",
                locale="pt",
                addressing_style="formal",
                text="Сохранённый PT текст",
            ),
            BotTranslation(
                locale="pt",
                translation_key="content.saved_translation",
                text="Сохранённый перевод PT",
                source_hash=source_hash("Исходный текст"),
            ),
        ])
        await session.commit()

    telegram_session = ValidatingTelegramSession(stateful=True)
    bot = Bot("123456:TEST", session=telegram_session)
    root = _admin_message(bot, text="/admin")
    update_id = 7000
    await admin_dispatcher.feed_update(
        bot,
        Update(update_id=update_id, message=root),
    )

    def buttons(message):
        return [
            button
            for row in (message.reply_markup.inline_keyboard if message.reply_markup else [])
            for button in row
            if button.callback_data
        ]

    def parsed_buttons(message, action=None):
        result = []
        for button in buttons(message):
            parsed = parse_system_message_callback(button.callback_data)
            if parsed is not None and (action is None or parsed["action"] == action):
                result.append((button, parsed))
        return result

    async def press(message, predicate):
        nonlocal update_id
        visible = telegram_session.current_message(message)
        button = next(button for button in buttons(visible) if predicate(button))
        update_id += 1
        await _feed_callback(admin_dispatcher, bot, visible, button.callback_data, update_id)
        return telegram_session.current_message(visible)

    screen = await press(root, lambda button: button.callback_data == "admin_general_settings")
    system_list = await press(screen, lambda button: button.callback_data == "admin_system_messages")
    assert not parsed_buttons(system_list, "l")

    before_hidden_callback = telegram_session.current_message(system_list)
    calls_before_hidden_callback = len(telegram_session.calls)
    update_id += 1
    await _feed_callback(
        admin_dispatcher,
        bot,
        before_hidden_callback,
        "admin_system_messages_locale:pt:0",
        update_id,
    )
    hidden_calls = telegram_session.calls[calls_before_hidden_callback:]
    assert any(isinstance(method, AnswerCallbackQuery) for method in hidden_calls)
    assert not any(isinstance(method, EditMessageText) for method in hidden_calls)
    assert telegram_session.current_message(system_list).text == before_hidden_callback.text

    calls_before_hidden_compact_callback = len(telegram_session.calls)
    update_id += 1
    await _feed_callback(
        admin_dispatcher,
        bot,
        before_hidden_callback,
        "sm:l:pt:0",
        update_id,
    )
    hidden_compact_calls = telegram_session.calls[calls_before_hidden_compact_callback:]
    assert any(isinstance(method, AnswerCallbackQuery) for method in hidden_compact_calls)
    assert not any(isinstance(method, EditMessageText) for method in hidden_compact_calls)
    assert telegram_session.current_message(system_list).text == before_hidden_callback.text

    screen = await press(system_list, lambda button: button.callback_data == "admin_general_settings")
    language = await press(screen, lambda button: button.callback_data == "admin_language_settings")
    assert any(
        button.callback_data == "admin_toggle_multilingual_authoring"
        for button in buttons(language)
    )
    language = await press(language, lambda button: button.callback_data == "admin_toggle_multilingual_authoring")
    screen = await press(language, lambda button: button.callback_data == "admin_general_settings")
    system_list = await press(screen, lambda button: button.callback_data == "admin_system_messages")
    locale_buttons = parsed_buttons(system_list, "l")
    assert {parsed["locale"] for _, parsed in locale_buttons} == {"ru", "en", "pt"}
    assert any(button.text.startswith("✅ ") for button, _ in locale_buttons)
    assert all(len(button.callback_data.encode("utf-8")) <= 64 for button in buttons(system_list))

    system_list = await press(
        system_list,
        lambda button: (parse_system_message_callback(button.callback_data) or {}).get("locale") == "pt"
        and (parse_system_message_callback(button.callback_data) or {}).get("action") == "l",
    )
    current_page = 0
    while True:
        target = [
            (button, parsed)
            for button, parsed in parsed_buttons(system_list, "d")
            if parsed["key"] == "ui.ai.thinking"
        ]
        if target:
            break
        next_page = next(
            button
            for button, parsed in parsed_buttons(system_list, "p")
            if int(parsed["page"]) == current_page + 1
        )
        system_list = await press(system_list, lambda button, target=next_page: button.callback_data == target.callback_data)
        current_page += 1

    detail = await press(system_list, lambda button, target=target[0][0]: button.callback_data == target.callback_data)
    editor = await press(
        detail,
        lambda button: (parse_system_message_callback(button.callback_data) or {}).get("action") == "e"
        and (parse_system_message_callback(button.callback_data) or {}).get("style") == "formal",
    )
    assert "Язык: <b>Português</b>" in editor.text
    assert "<b>Вы</b>" in editor.text
    telegram_session._next_message_ids[11] = 100
    user = root.from_user
    draft_message = Message(
        message_id=90,
        date=datetime.now(timezone.utc),
        chat=Chat(id=11, type="private"),
        from_user=user,
        text="Сохранённый PT текст после выключения",
    ).as_(bot)
    update_id += 1
    await admin_dispatcher.feed_update(
        bot,
        Update(update_id=update_id, message=draft_message),
    )
    draft_screen = telegram_session.latest_message_with_keyboard(bot, 11, user)
    detail = await press(
        draft_screen,
        lambda button: (parse_system_message_callback(button.callback_data) or {}).get("action") == "s",
    )
    system_list = await press(
        detail,
        lambda button: (parse_system_message_callback(button.callback_data) or {}).get("action") == "p",
    )
    screen = await press(system_list, lambda button: button.callback_data == "admin_general_settings")
    language = await press(screen, lambda button: button.callback_data == "admin_language_settings")
    language = await press(language, lambda button: button.callback_data == "admin_toggle_multilingual_authoring")
    screen = await press(language, lambda button: button.callback_data == "admin_general_settings")
    system_list = await press(screen, lambda button: button.callback_data == "admin_system_messages")
    assert not parsed_buttons(system_list, "l")

    screen = await press(system_list, lambda button: button.callback_data == "admin_general_settings")
    language = await press(screen, lambda button: button.callback_data == "admin_language_settings")
    language = await press(language, lambda button: button.callback_data == "admin_toggle_multilingual_authoring")
    screen = await press(language, lambda button: button.callback_data == "admin_general_settings")
    system_list = await press(screen, lambda button: button.callback_data == "admin_system_messages")
    system_list = await press(
        system_list,
        lambda button: (parse_system_message_callback(button.callback_data) or {}).get("locale") == "pt"
        and (parse_system_message_callback(button.callback_data) or {}).get("action") == "l",
    )
    current_page = 0
    while not any(
        parsed["key"] == "ui.ai.thinking"
        for _, parsed in parsed_buttons(system_list, "d")
    ):
        current_page += 1
        system_list = await press(
            system_list,
            lambda button, page=current_page: (parse_system_message_callback(button.callback_data) or {}).get("action") == "p"
            and int((parse_system_message_callback(button.callback_data) or {}).get("page", -1)) == page,
        )
    detail = await press(
        system_list,
        lambda button: (parse_system_message_callback(button.callback_data) or {}).get("action") == "d"
        and (parse_system_message_callback(button.callback_data) or {}).get("key") == "ui.ai.thinking",
    )
    assert "Сохранённый PT текст после выключения" in detail.text

    async with factory() as session:
        assert await resolve_system_text(
            session,
            "ui.ai.thinking",
            "pt",
            "formal",
            bot_scope="telegram",
        ) == "Сохранённый PT текст после выключения"
        translation = await session.scalar(
            select(BotTranslation).where(
                BotTranslation.locale == "pt",
                BotTranslation.translation_key == "content.saved_translation",
            )
        )
        assert translation.text == "Сохранённый перевод PT"


@pytest.mark.parametrize("hidden_locale", ["en", "pt"])
@pytest.mark.asyncio
async def test_telegram_hidden_locale_cancel_returns_to_ru_list_and_clears_state(
    factory,
    admin_dispatcher,
    monkeypatch,
    hidden_locale,
):
    import admin_content_authoring
    import automation_admin
    import handlers
    import keyboards
    from handlers import AdminStates

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(admin_content_authoring, "async_session_maker", factory)
    monkeypatch.setattr(automation_admin, "async_session_maker", factory)
    monkeypatch.setattr(keyboards, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(automation_admin, "get_all_admin_ids", AsyncMock(return_value={11, 12}))

    key = "ui.ai.thinking"
    original_text = "Сохранённый PT текст"
    original_translation = "Сохранённый перевод PT"
    async with factory() as session:
        session.add_all([
            DBUser(id=11, is_admin=True, first_name="Admin 11"),
            DBUser(id=12, is_admin=True, first_name="Admin 12"),
            AIConfig(id=1, provider="Deepseek"),
            SubscriptionConfig(id=1),
            BotSystemMessage(
                bot_scope="telegram",
                translation_key=key,
                locale=hidden_locale,
                addressing_style="formal",
                text=original_text,
            ),
            BotTranslation(
                locale=hidden_locale,
                translation_key="content.saved_translation",
                text=original_translation,
                source_hash=source_hash("Исходный текст"),
            ),
        ])
        await session.commit()

    telegram_session = ValidatingTelegramSession(stateful=True)
    bot = Bot("123456:TEST", session=telegram_session)
    update_id = 7100
    root_11 = _admin_message(bot, user_id=11, text="/admin")
    root_12 = _admin_message(bot, user_id=12, text="/admin")
    await admin_dispatcher.feed_update(bot, Update(update_id=update_id, message=root_11))
    update_id += 1
    await admin_dispatcher.feed_update(bot, Update(update_id=update_id, message=root_12))

    def buttons(message):
        visible = telegram_session.current_message(message)
        return [
            button
            for row in (visible.reply_markup.inline_keyboard if visible.reply_markup else [])
            for button in row
            if button.callback_data
        ]

    def parsed_buttons(message, action=None):
        result = []
        for button in buttons(message):
            parsed = parse_system_message_callback(button.callback_data)
            if parsed is not None and (action is None or parsed["action"] == action):
                result.append((button, parsed))
        return result

    async def press(message, predicate):
        nonlocal update_id
        visible = telegram_session.current_message(message)
        button = next(button for button in buttons(visible) if predicate(button))
        update_id += 1
        calls_before = len(telegram_session.calls)
        await _feed_callback(admin_dispatcher, bot, visible, button.callback_data, update_id)
        return (
            telegram_session.current_message(visible),
            telegram_session.calls[calls_before:],
        )

    async def feed(message, callback_data):
        nonlocal update_id
        update_id += 1
        visible = telegram_session.current_message(message)
        calls_before = len(telegram_session.calls)
        await _feed_callback(admin_dispatcher, bot, visible, callback_data, update_id)
        return telegram_session.current_message(visible), telegram_session.calls[calls_before:]

    screen_11, _ = await press(root_11, lambda button: button.callback_data == "admin_general_settings")
    system_list_11, _ = await press(screen_11, lambda button: button.callback_data == "admin_system_messages")
    system_list_11, _ = await press(
        system_list_11,
        lambda button: (parse_system_message_callback(button.callback_data) or {}).get("action") == "l"
        and (parse_system_message_callback(button.callback_data) or {}).get("locale") == hidden_locale,
    )
    while not any(parsed["key"] == key for _, parsed in parsed_buttons(system_list_11, "d")):
        system_list_11, _ = await press(
            system_list_11,
            lambda button: (parse_system_message_callback(button.callback_data) or {}).get("action") == "p"
            and int((parse_system_message_callback(button.callback_data) or {}).get("page", -1))
            == max(
                int(parsed["page"])
                for _, parsed in parsed_buttons(system_list_11, "p")
            ),
        )
    detail_11, _ = await press(
        system_list_11,
        lambda button: (parse_system_message_callback(button.callback_data) or {}).get("action") == "d"
        and (parse_system_message_callback(button.callback_data) or {}).get("key") == key,
    )
    editor_11, _ = await press(
        detail_11,
        lambda button: (parse_system_message_callback(button.callback_data) or {}).get("action") == "e"
        and (parse_system_message_callback(button.callback_data) or {}).get("style") == "formal",
    )
    context_11 = admin_dispatcher.fsm.get_context(bot=bot, chat_id=11, user_id=11)
    assert (await context_11.get_state()).endswith("set_system_message_text")
    assert (await context_11.get_data())["system_message_locale"] == hidden_locale

    screen_12, _ = await press(root_12, lambda button: button.callback_data == "admin_general_settings")
    language_12, _ = await press(screen_12, lambda button: button.callback_data == "admin_language_settings")
    language_12, _ = await press(
        language_12,
        lambda button: button.callback_data == "admin_toggle_multilingual_authoring",
    )
    async with factory() as session:
        assert not (await session.get(BotGeneralConfig, 1)).multilingual_authoring_enabled

    _, blocked_mutation_calls = await feed(
        editor_11,
        system_message_callback_data("r", key, hidden_locale, "formal", 0),
    )
    assert any(isinstance(method, AnswerCallbackQuery) for method in blocked_mutation_calls)
    assert not any(isinstance(method, EditMessageText) for method in blocked_mutation_calls)
    assert (await context_11.get_data())["system_message_locale"] == hidden_locale

    ru_list, cancel_calls = await press(
        editor_11,
        lambda button: (parse_system_message_callback(button.callback_data) or {}).get("action") == "c",
    )
    assert any(isinstance(method, EditMessageText) for method in cancel_calls)
    assert any(isinstance(method, AnswerCallbackQuery) for method in cancel_calls)
    assert "Язык: <b>Русский</b>" in ru_list.text
    assert not parsed_buttons(ru_list, "l")
    assert await context_11.get_state() is None
    assert await context_11.get_data() == {}

    ru_page, legacy_page_calls = await feed(ru_list, f"admin_system_messages_page:1:{hidden_locale}")
    assert any(isinstance(method, EditMessageText) for method in legacy_page_calls)
    assert "страница 2/" in ru_page.text
    await context_11.set_state(AdminStates.set_system_message_text)
    await context_11.update_data(
        system_message_key=key,
        system_message_locale=hidden_locale,
        system_message_style="formal",
        system_message_page=0,
        system_message_draft="Не сохранять",
    )
    ru_list, legacy_cancel_calls = await feed(ru_page, f"admin_system_cancel:{key}:{hidden_locale}:0")
    assert any(isinstance(method, EditMessageText) for method in legacy_cancel_calls)
    assert "Язык: <b>Русский</b>" in ru_list.text
    assert await context_11.get_state() is None
    assert await context_11.get_data() == {}

    screen_11, _ = await press(ru_list, lambda button: button.callback_data == "admin_general_settings")
    language_11, _ = await press(screen_11, lambda button: button.callback_data == "admin_language_settings")
    language_11, _ = await press(
        language_11,
        lambda button: button.callback_data == "admin_toggle_multilingual_authoring",
    )
    screen_11, _ = await press(language_11, lambda button: button.callback_data == "admin_general_settings")
    system_list_11, _ = await press(screen_11, lambda button: button.callback_data == "admin_system_messages")
    assert {
        parsed["locale"]
        for _, parsed in parsed_buttons(system_list_11, "l")
    } == {"ru", "en", "pt"}
    system_list_11, _ = await press(
        system_list_11,
        lambda button: (parse_system_message_callback(button.callback_data) or {}).get("action") == "l"
        and (parse_system_message_callback(button.callback_data) or {}).get("locale") == hidden_locale,
    )
    while not any(parsed["key"] == key for _, parsed in parsed_buttons(system_list_11, "d")):
        next_page = max(
            int(parsed["page"])
            for _, parsed in parsed_buttons(system_list_11, "p")
        )
        system_list_11, _ = await press(
            system_list_11,
            lambda button, next_page=next_page: (parse_system_message_callback(button.callback_data) or {}).get("action") == "p"
            and int((parse_system_message_callback(button.callback_data) or {}).get("page", -1)) == next_page,
        )
    detail_11, _ = await press(
        system_list_11,
        lambda button: (parse_system_message_callback(button.callback_data) or {}).get("action") == "d"
        and (parse_system_message_callback(button.callback_data) or {}).get("key") == key,
    )
    assert original_text in detail_11.text
    async with factory() as session:
        row = await session.scalar(
            select(BotSystemMessage).where(
                BotSystemMessage.bot_scope == "telegram",
                BotSystemMessage.translation_key == key,
                BotSystemMessage.locale == hidden_locale,
                BotSystemMessage.addressing_style == "formal",
            )
        )
        translation = await session.scalar(
            select(BotTranslation).where(
                BotTranslation.locale == hidden_locale,
                BotTranslation.translation_key == "content.saved_translation",
            )
        )
        assert row.text == original_text
        assert translation.text == original_translation


@pytest.mark.asyncio
async def test_general_settings_handler_builds_validated_telegram_method(factory, monkeypatch):
    import handlers
    from aiogram import Bot
    from aiogram.types import CallbackQuery, Update
    from aiogram.methods import EditMessageText

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    user = User(id=11, is_bot=False, first_name="Admin")
    message = Message(
        message_id=1,
        date=datetime.now(timezone.utc),
        chat=Chat(id=11, type="private"),
        from_user=user,
        text="Админ-панель",
    ).as_(bot)
    callback = CallbackQuery(
        id="general-settings",
        from_user=user,
        chat_instance="admin",
        message=message,
        data="admin_general_settings",
    ).as_(bot)

    await handlers.admin_general_settings(callback)

    edit = next(method for method in session.calls if isinstance(method, EditMessageText))
    assert isinstance(edit.text, str)
    assert "Общие настройки" in edit.text
    assert any(method.__class__.__name__ == "AnswerCallbackQuery" for method in session.calls)


@pytest.mark.asyncio
async def test_admin_dispatcher_general_settings_journey_uses_validated_methods(factory, admin_dispatcher, monkeypatch):
    import admin_content_authoring
    import automation_admin
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(admin_content_authoring, "async_session_maker", factory)
    monkeypatch.setattr(automation_admin, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(automation_admin, "get_all_admin_ids", AsyncMock(return_value={11}))

    async with factory() as session:
        session.add(DBUser(id=11, is_admin=True, first_name="Admin"))
        await session.commit()

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    dispatcher = admin_dispatcher

    root_message = _admin_message(bot, text="/admin")
    await dispatcher.feed_update(
        bot,
        Update(update_id=100, message=root_message),
    )
    root = next(method for method in reversed(session.calls) if isinstance(method, SendMessage))
    assert root.text == "Добро пожаловать в админ-панель!"
    assert any(
        button.callback_data == "admin_general_settings"
        for row in root.reply_markup.inline_keyboard
        for button in row
    )

    await _feed_callback(dispatcher, bot, root_message, "admin_general_settings", 101)
    general = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    assert isinstance(general.text, str)
    assert "Общие настройки" in general.text
    general_button_text = " ".join(
        button.text
        for row in general.reply_markup.inline_keyboard
        for button in row
    )
    assert all(label in general_button_text for label in ("Имя", "Пол", "Возраст", "Язык"))
    general_buttons = {
        button.callback_data
        for row in general.reply_markup.inline_keyboard
        for button in row
    }
    assert "admin_panel" in general_buttons

    await _feed_callback(dispatcher, bot, root_message, "admin_panel", 102)
    back = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    assert back.text == "Добро пожаловать в админ-панель!"


@pytest.mark.asyncio
async def test_addressing_style_system_message_journey_uses_real_dispatcher(factory, admin_dispatcher, monkeypatch):
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers.ai_integration, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))

    async with factory() as db_session:
        db_session.add(DBUser(id=11, is_admin=True, first_name="Admin"))
        await db_session.commit()

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    root_message = _admin_message(bot, text="/admin")

    async def press(callback_data, update_id):
        session.calls.clear()
        await _feed_callback(admin_dispatcher, bot, root_message, callback_data, update_id)
        rendered = [
            method
            for method in session.calls
            if isinstance(method, (EditMessageText, SendMessage))
        ]
        assert rendered, callback_data
        return rendered[-1]

    await press("admin_general_settings", 500)
    general = await press("admin_general_toggle_addressing_style", 501)
    assert "Обращение: <b>Вы</b>" in general.text
    async with factory() as db_session:
        assert (await db_session.get(BotGeneralConfig, 1)).addressing_style == "formal"
    general = await press("admin_general_toggle_addressing_style", 502)
    assert "Обращение: <b>Ты</b>" in general.text
    async with factory() as db_session:
        assert (await db_session.get(BotGeneralConfig, 1)).addressing_style == "informal"
    general = await press("admin_general_toggle_addressing_style", 503)
    assert "Обращение: <b>Вы</b>" in general.text
    async with factory() as db_session:
        assert (await db_session.get(BotGeneralConfig, 1)).addressing_style == "formal"

    system_list = await press("admin_system_messages", 504)
    assert "Сообщения бота" in system_list.text
    system_list = await press(system_message_callback_data("p", locale="ru", page=1), 505)
    assert any(
        button.callback_data == system_message_callback_data("d", "ui.ai.busy", "ru", page=1)
        for row in system_list.reply_markup.inline_keyboard
        for button in row
    )

    detail = await press(system_message_callback_data("d", "ui.ai.busy", "ru", page=1), 506)
    assert "Предыдущий запрос ещё выполняется" in detail.text
    assert "Когда появляется:" in detail.text
    assert "ui.ai.busy" not in detail.text
    await press(system_message_callback_data("e", "ui.ai.busy", "ru", "formal", 1), 507)

    draft_message = _admin_message(
        bot,
        message_id=2,
        text="Пожалуйста, подождите — Ваш предыдущий запрос ещё выполняется.",
    )
    await admin_dispatcher.feed_update(
        bot,
        Update(update_id=508, message=draft_message),
    )
    assert any(
        isinstance(method, SendMessage) and "Черновик сохранён" in method.text
        for method in session.calls
    )

    saved = await press(system_message_callback_data("s", "ui.ai.busy", "ru", "formal", 1), 509)
    assert "Ваш предыдущий запрос" in saved.text
    async with factory() as db_session:
        formal_row = await db_session.scalar(
            select(BotSystemMessage).where(
                BotSystemMessage.translation_key == "ui.ai.busy",
                BotSystemMessage.addressing_style == "formal",
            )
        )
        assert formal_row.text == "Пожалуйста, подождите — Ваш предыдущий запрос ещё выполняется."

    reopened = await press(system_message_callback_data("d", "ui.ai.busy", "ru", page=1), 510)
    assert "Ваш предыдущий запрос" in reopened.text
    await press(system_message_callback_data("e", "ui.ai.busy", "ru", "formal", 1), 511)
    await admin_dispatcher.feed_update(
        bot,
        Update(
            update_id=512,
            message=_admin_message(
                bot,
                message_id=3,
                text="Этот текст должен быть отменён.",
            ),
        ),
    )
    cancelled = await press(system_message_callback_data("c", "ui.ai.busy", "ru", "formal", 1), 513)
    assert "Ваш предыдущий запрос" in cancelled.text
    await press(system_message_callback_data("r", "ui.ai.busy", "ru", "formal", 1), 514)
    reset_detail = await press(system_message_callback_data("d", "ui.ai.busy", "ru", page=1), 515)
    assert "Пожалуйста, не так быстро" in reset_detail.text

    await press(system_message_callback_data("p", locale="ru", page=1), 516)
    english_list = await press(system_message_callback_data("l", locale="en", page=1), 517)
    assert "Язык: <b>English</b>" in english_list.text
    returned = await press("admin_general_settings", 519)
    assert "Общие настройки" in returned.text

    async with factory() as db_session:
        formal_row = await db_session.scalar(
            select(BotSystemMessage).where(
                BotSystemMessage.translation_key == "ui.ai.busy",
                BotSystemMessage.addressing_style == "formal",
            )
        )
        informal_row = await db_session.scalar(
            select(BotSystemMessage).where(
                BotSystemMessage.translation_key == "ui.ai.busy",
                BotSystemMessage.addressing_style == "informal",
            )
        )
        assert formal_row is None
        assert informal_row is None

    import httpx

    async with factory() as db_session:
        db_session.add(DBUser(
            id=12,
            first_name="Иван",
            name="Иван",
            gender="male",
            accepted_disclaimer=True,
            current_dialogue_id=1,
        ))
        db_session.add(AIConfig(
            id=1,
            provider="OpenRouter",
            openrouter_api_key="sk-test-key",
            openrouter_model="qwen/qwen3.7-plus",
            system_prompt="Система администратора",
            memory_mode="global",
        ))
        db_session.add(SubscriptionConfig(id=1, subscriptions_enabled=False))
        await db_session.commit()

    provider_payloads = []

    async def fake_provider_post(self, url, *args, **kwargs):
        provider_payloads.append(kwargs["json"])
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "Ответ пользователю"}}]},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_provider_post)
    handlers.user_message_buffers[12] = ["Привет"]
    session.calls.clear()
    await handlers.process_buffered_messages(12, bot)
    outgoing = [method for method in session.calls if isinstance(method, SendMessage) and method.chat_id == 12]
    assert any(method.text == "Ответ пользователю" for method in outgoing)
    assert provider_payloads
    assert "ОБРАЩЕНИЕ: обращайся к пользователю на «Вы»" in str(provider_payloads[-1])


@pytest.mark.asyncio
async def test_telegram_system_message_ux_journey_uses_visible_stateful_keyboard(factory, admin_dispatcher, monkeypatch):
    import handlers
    import httpx

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers.ai_integration, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))

    async with factory() as db_session:
        db_session.add_all([
            DBUser(
                id=21,
                is_admin=True,
                first_name="Admin",
                name="Admin",
                accepted_disclaimer=True,
                current_dialogue_id=1,
            ),
            AIConfig(
                id=1,
                provider="OpenRouter",
                openrouter_api_key="sk-test-key",
                openrouter_model="qwen/qwen3.7-plus",
                system_prompt="Система администратора",
                memory_mode="global",
            ),
            SubscriptionConfig(id=1, subscriptions_enabled=False),
        ])
        await db_session.commit()

    provider_payloads = []

    async def fake_provider_post(self, url, *args, **kwargs):
        provider_payloads.append(kwargs["json"])
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "Ответ"}}]},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_provider_post)
    session = ValidatingTelegramSession(stateful=True)
    bot = Bot("123456:TEST", session=session)
    root_message = _admin_message(bot, user_id=21, text="/admin")
    await admin_dispatcher.feed_update(bot, Update(update_id=2000, message=root_message))
    update_id = 2000

    def visible_buttons(message):
        buttons = [
            button
            for row in (message.reply_markup.inline_keyboard if message.reply_markup else [])
            for button in row
            if button.callback_data
        ]
        assert buttons
        for button in buttons:
            assert len(button.callback_data.encode("utf-8")) <= 64
            assert "ui." not in button.text
        return buttons

    async def press(message, predicate):
        nonlocal update_id
        current = session.current_message(message)
        button = next(button for button in visible_buttons(current) if predicate(button))
        update_id += 1
        session.calls.clear()
        await _feed_callback(admin_dispatcher, bot, current, button.callback_data, update_id)
        assert any(isinstance(method, AnswerCallbackQuery) for method in session.calls)
        return session.current_message(current)

    general = await press(root_message, lambda button: button.callback_data == "admin_general_settings")
    formal = await press(general, lambda button: button.callback_data == "admin_general_toggle_addressing_style")
    assert "Обращение: <b>Вы</b>" in formal.text
    general_callbacks = {
        button.callback_data
        for button in visible_buttons(formal)
    }
    assert "admin_general_edit_ai_processing_message_text" not in general_callbacks
    assert "admin_general_open_ai_processing_variants" not in general_callbacks

    await _feed_callback(
        admin_dispatcher,
        bot,
        formal,
        "admin_general_edit_ai_processing_message_text",
        2003,
    )
    legacy_detail = session.current_message(formal)
    assert any(
        parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "e"
        for button in visible_buttons(legacy_detail)
    )
    back_to_list = next(
        button
        for button in visible_buttons(legacy_detail)
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "p"
    )
    await _feed_callback(admin_dispatcher, bot, legacy_detail, back_to_list.callback_data, 2004)
    system_list = session.current_message(formal)
    general_button = next(
        button
        for button in visible_buttons(system_list)
        if button.callback_data == "admin_general_settings"
    )
    await _feed_callback(admin_dispatcher, bot, system_list, general_button.callback_data, 2005)
    formal = session.current_message(formal)
    await _feed_callback(
        admin_dispatcher,
        bot,
        formal,
        "admin_general_open_ai_processing_variants",
        2006,
    )
    legacy_detail = session.current_message(formal)
    assert any(
        parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "e"
        for button in visible_buttons(legacy_detail)
    )
    back_to_list = next(
        button
        for button in visible_buttons(legacy_detail)
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "p"
    )
    await _feed_callback(admin_dispatcher, bot, legacy_detail, back_to_list.callback_data, 2007)
    system_list = session.current_message(formal)
    general_button = next(
        button
        for button in visible_buttons(system_list)
        if button.callback_data == "admin_general_settings"
    )
    await _feed_callback(admin_dispatcher, bot, system_list, general_button.callback_data, 2008)
    formal = session.current_message(formal)
    update_id = 2008

    system_list = await press(formal, lambda button: button.callback_data == "admin_system_messages")
    waiting_button = next(
        button
        for button in visible_buttons(system_list)
        if parse_system_message_callback(button.callback_data)
        and parse_system_message_callback(button.callback_data)["action"] == "d"
        and parse_system_message_callback(button.callback_data)["key"] == "ui.ai.processing_default"
    )
    waiting_detail = await press(system_list, lambda button: button.callback_data == waiting_button.callback_data)
    waiting_editor = await press(
        waiting_detail,
        lambda button: (
            parse_system_message_callback(button.callback_data) is not None
            and parse_system_message_callback(button.callback_data)["action"] == "e"
            and parse_system_message_callback(button.callback_data)["style"] == "formal"
        ),
    )
    assert "Сообщение во время ожидания ответа" in waiting_editor.text
    assert "Язык: <b>Русский</b>" in waiting_editor.text
    assert "Обращение: <b>Вы</b>" in waiting_editor.text
    assert {button.text for button in visible_buttons(waiting_editor)} >= {"💾 Сохранить", "Отмена"}
    update_id += 1
    await admin_dispatcher.feed_update(
        bot,
        Update(
            update_id=update_id,
            message=_admin_message(bot, user_id=21, message_id=2, text="Custom waiting"),
        ),
    )
    waiting_draft = session.latest_message_with_keyboard(bot, 21, User(id=21, is_bot=False, first_name="Admin"))
    save_button = next(
        button
        for button in visible_buttons(waiting_draft)
        if parse_system_message_callback(button.callback_data)
        and parse_system_message_callback(button.callback_data)["action"] == "s"
    )
    update_id += 1
    await _feed_callback(admin_dispatcher, bot, waiting_draft, save_button.callback_data, update_id)
    waiting_detail = session.current_message(waiting_draft)
    assert "Custom waiting" in waiting_detail.text
    async with factory() as db_session:
        config = await db_session.get(BotGeneralConfig, 1)
        assert config.ai_processing_message_text == DEFAULT_AI_PROCESSING_MESSAGE_TEXT
        assert config.ai_processing_message_text_explicit is False
        saved_waiting = await db_session.scalar(
            select(BotSystemMessage).where(
                BotSystemMessage.translation_key == "ui.ai.processing_default",
                BotSystemMessage.locale == "ru",
                BotSystemMessage.addressing_style == "formal",
            )
        )
        assert saved_waiting.text == "Custom waiting"

    system_list = await press(
        waiting_detail,
        lambda button: (
            parse_system_message_callback(button.callback_data) is not None
            and parse_system_message_callback(button.callback_data)["action"] == "p"
        ),
    )
    formal = await press(system_list, lambda button: button.callback_data == "admin_general_settings")
    assert "Действующий текст: Custom waiting" in formal.text

    system_list = await press(formal, lambda button: button.callback_data == "admin_system_messages")
    assert "Сообщения бота · страница 1/" in system_list.text
    system_list = await press(
        system_list,
        lambda button: (
            parse_system_message_callback(button.callback_data) is not None
            and parse_system_message_callback(button.callback_data)["action"] == "p"
            and parse_system_message_callback(button.callback_data)["page"] == "1"
        ),
    )
    assert "страница 2/" in system_list.text
    assert any(button.text == "⬅️ Назад" for button in visible_buttons(system_list))
    assert any(button.text == "Далее ➡️" for button in visible_buttons(system_list))
    system_list = await press(
        system_list,
        lambda button: (
            parse_system_message_callback(button.callback_data) is not None
            and parse_system_message_callback(button.callback_data)["action"] == "l"
            and parse_system_message_callback(button.callback_data)["locale"] == "pt"
        ),
    )
    assert "Язык: <b>Português</b>" in system_list.text
    assert any(
        {button.text for button in row} == {"🟢 RU", "🟢 EN", "✅ PT"}
        for row in system_list.reply_markup.inline_keyboard
    )
    target_button = next(
        button
        for button in visible_buttons(system_list)
        if parse_system_message_callback(button.callback_data)
        and parse_system_message_callback(button.callback_data)["action"] == "d"
        and parse_system_message_callback(button.callback_data)["key"] == "ui.ai.busy"
    )
    detail = await press(system_list, lambda button: button.callback_data == target_button.callback_data)
    assert "Предыдущий запрос ещё выполняется" in detail.text
    assert "Когда появляется:" in detail.text
    assert "ui.ai.busy" not in detail.text
    editor = await press(
        detail,
        lambda button: (
            parse_system_message_callback(button.callback_data) is not None
            and parse_system_message_callback(button.callback_data)["action"] == "e"
            and parse_system_message_callback(button.callback_data)["style"] == "formal"
        ),
    )
    assert "Язык: <b>Português</b>" in editor.text
    assert "Обращение: <b>Вы</b>" in editor.text
    assert "ui.ai.busy" not in editor.text

    await admin_dispatcher.feed_update(
        bot,
        Update(
            update_id=update_id + 1,
            message=_admin_message(
                bot,
                user_id=21,
                message_id=2,
                text="PT formal custom busy",
            ),
        ),
    )
    draft = session.latest_message_with_keyboard(bot, 21, User(id=21, is_bot=False, first_name="Admin"))
    save_button = next(
        button
        for button in visible_buttons(draft)
        if parse_system_message_callback(button.callback_data)
        and parse_system_message_callback(button.callback_data)["action"] == "s"
    )
    update_id += 2
    await _feed_callback(admin_dispatcher, bot, draft, save_button.callback_data, update_id)
    detail = session.current_message(draft)
    assert "PT formal custom busy" in detail.text
    async with factory() as db_session:
        saved = await db_session.scalar(
            select(BotSystemMessage).where(
                BotSystemMessage.bot_scope == "telegram",
                BotSystemMessage.translation_key == "ui.ai.busy",
                BotSystemMessage.locale == "pt",
                BotSystemMessage.addressing_style == "formal",
            )
        )
        assert saved.text == "PT formal custom busy"

    system_list = await press(detail, lambda button: parse_system_message_callback(button.callback_data) and parse_system_message_callback(button.callback_data)["action"] == "p")
    assert "страница 2/" in system_list.text
    assert "Язык: <b>Português</b>" in system_list.text
    detail = await press(system_list, lambda button: button.callback_data == target_button.callback_data)
    assert "PT formal custom busy" in detail.text
    system_list = await press(
        detail,
        lambda button: parse_system_message_callback(button.callback_data)
        and parse_system_message_callback(button.callback_data)["action"] == "p",
    )
    assert "страница 2/" in system_list.text
    general = await press(system_list, lambda button: button.callback_data == "admin_general_settings")
    assert "Общие настройки" in general.text
    assert "Сообщение во время ожидания ответа" in general.text
    assert "Действующий текст: Custom waiting" in general.text

    handlers.user_message_buffers[21] = ["Привет"]
    session.calls.clear()
    await handlers.process_buffered_messages(21, bot)
    assert any(isinstance(method, SendMessage) and method.chat_id == 21 and method.text == "Ответ" for method in session.calls)
    assert provider_payloads
    assert "ОБРАЩЕНИЕ: обращайся к пользователю на «Вы»" in str(provider_payloads[-1])


@pytest.mark.asyncio
async def test_addressing_style_toggle_updates_and_finishes_callback(factory, admin_dispatcher, monkeypatch):
    import handlers
    import httpx

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers.ai_integration, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))

    async with factory() as db_session:
        db_session.add_all([
            DBUser(
                id=18,
                is_admin=True,
                first_name="Admin",
                name="Admin",
                accepted_disclaimer=True,
                current_dialogue_id=1,
            ),
            AIConfig(
                id=1,
                provider="OpenRouter",
                openrouter_api_key="sk-test-key",
                openrouter_model="qwen/qwen3.7-plus",
                system_prompt="Система администратора",
                memory_mode="global",
            ),
            SubscriptionConfig(id=1, subscriptions_enabled=False),
        ])
        await db_session.commit()

    provider_payloads = []

    async def fake_provider_post(self, url, *args, **kwargs):
        provider_payloads.append(kwargs["json"])
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "Ответ"}}]},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_provider_post)
    session = ValidatingTelegramSession(stateful=True)
    bot = Bot("123456:TEST", session=session)
    root_message = _admin_message(bot, user_id=18, text="/admin")
    await admin_dispatcher.feed_update(bot, Update(update_id=560, message=root_message))
    session.calls.clear()
    await _feed_callback(admin_dispatcher, bot, root_message, "admin_general_settings", 561)
    current = session.current_message(root_message)
    toggle_button = next(button for row in current.reply_markup.inline_keyboard for button in row if button.callback_data == "admin_general_toggle_addressing_style")
    assert "Ты" in toggle_button.text
    session.calls.clear()
    await _feed_callback(
        admin_dispatcher,
        bot,
        current,
        toggle_button.callback_data,
        562,
    )
    assert any(isinstance(method, EditMessageText) for method in session.calls)
    answers = [method for method in session.calls if isinstance(method, AnswerCallbackQuery)]
    assert len(answers) == 1
    assert answers[0].text == "Обращение: Вы"
    async with factory() as db_session:
        assert (await db_session.get(BotGeneralConfig, 1)).addressing_style == "formal"

    current = session.current_message(root_message)
    session.calls.clear()
    await _feed_callback(admin_dispatcher, bot, current, "admin_general_toggle_addressing_style", 563)
    assert any(isinstance(method, EditMessageText) for method in session.calls)
    assert any(
        isinstance(method, AnswerCallbackQuery) and method.text == "Обращение: Ты"
        for method in session.calls
    )
    assert "Обращение: <b>Ты</b>" in session.current_message(root_message).text
    async with factory() as db_session:
        assert (await db_session.get(BotGeneralConfig, 1)).addressing_style == "informal"

    session.calls.clear()
    await _feed_callback(
        admin_dispatcher,
        bot,
        session.current_message(root_message),
        "admin_panel",
        564,
    )
    assert session.current_message(root_message).text == "Добро пожаловать в админ-панель!"
    session.calls.clear()
    await _feed_callback(
        admin_dispatcher,
        bot,
        session.current_message(root_message),
        "admin_general_settings",
        565,
    )
    reopened = session.current_message(root_message)
    assert "Обращение: <b>Ты</b>" in reopened.text

    handlers.user_message_buffers[18] = ["Привет"]
    session.calls.clear()
    await handlers.process_buffered_messages(18, bot)
    assert any(
        isinstance(method, SendMessage) and method.chat_id == 18 and method.text == "Ответ"
        for method in session.calls
    )
    assert provider_payloads
    assert "ОБРАЩЕНИЕ: обращайся к пользователю на «ты»" in str(provider_payloads[-1])

    session.calls.clear()
    await _feed_callback(
        admin_dispatcher,
        bot,
        reopened,
        "admin_panel",
        566,
    )
    assert session.current_message(root_message).text == "Добро пожаловать в админ-панель!"


@pytest.mark.asyncio
async def test_addressing_style_edit_failure_finishes_callback(factory, admin_dispatcher, monkeypatch):
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))

    async with factory() as db_session:
        db_session.add(DBUser(id=19, is_admin=True, first_name="Admin"))
        await db_session.commit()

    session = ValidatingTelegramSession(stateful=True)
    bot = Bot("123456:TEST", session=session)
    root_message = _admin_message(bot, user_id=19, text="/admin")
    await admin_dispatcher.feed_update(bot, Update(update_id=570, message=root_message))
    await _feed_callback(admin_dispatcher, bot, root_message, "admin_general_settings", 571)
    session.next_edit_error = "Bad Request: simulated edit failure"
    session.calls.clear()

    await _feed_callback(
        admin_dispatcher,
        bot,
        session.current_message(root_message),
        "admin_general_toggle_addressing_style",
        572,
    )

    answers = [method for method in session.calls if isinstance(method, AnswerCallbackQuery)]
    assert len(answers) == 1
    assert answers[0].show_alert is True
    assert answers[0].text == "Не удалось обновить экран. Попробуйте ещё раз."
    async with factory() as db_session:
        assert (await db_session.get(BotGeneralConfig, 1)).addressing_style == "formal"


@pytest.mark.asyncio
async def test_addressing_style_general_settings_uses_one_toggle_button(factory, admin_dispatcher, monkeypatch):
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))

    async with factory() as db_session:
        db_session.add(DBUser(id=20, is_admin=True, first_name="Admin"))
        await db_session.commit()

    session = ValidatingTelegramSession(stateful=True)
    bot = Bot("123456:TEST", session=session)
    root_message = _admin_message(bot, user_id=20, text="/admin")
    await admin_dispatcher.feed_update(bot, Update(update_id=580, message=root_message))
    await _feed_callback(admin_dispatcher, bot, root_message, "admin_general_settings", 581)
    current = session.current_message(root_message)
    buttons = [
        button
        for row in current.reply_markup.inline_keyboard
        for button in row
        if button.callback_data
    ]
    addressing_buttons = [button for button in buttons if button.callback_data.startswith("admin_general_set_addressing_")]
    toggle_buttons = [button for button in buttons if button.callback_data == "admin_general_toggle_addressing_style"]
    assert not addressing_buttons
    assert len(toggle_buttons) == 1
    assert "Ты" in toggle_buttons[0].text

    await _feed_callback(admin_dispatcher, bot, current, toggle_buttons[0].callback_data, 582)
    current = session.current_message(root_message)
    assert "Обращение: <b>Вы</b>" in current.text
    async with factory() as db_session:
        assert (await db_session.get(BotGeneralConfig, 1)).addressing_style == "formal"

    toggle_button = next(
        button
        for row in current.reply_markup.inline_keyboard
        for button in row
        if button.callback_data == "admin_general_toggle_addressing_style"
    )
    await _feed_callback(admin_dispatcher, bot, current, toggle_button.callback_data, 583)
    assert "Обращение: <b>Ты</b>" in session.current_message(root_message).text
    async with factory() as db_session:
        assert (await db_session.get(BotGeneralConfig, 1)).addressing_style == "informal"


@pytest.mark.asyncio
async def test_system_message_catalog_uses_valid_callback_data_on_every_detail_page(factory, admin_dispatcher, monkeypatch):
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))

    async with factory() as db_session:
        db_session.add(DBUser(id=13, is_admin=True, first_name="Admin"))
        await db_session.commit()

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    root_message = _admin_message(bot, user_id=13, text="/admin")
    update_id = 600
    categories = {}

    def classify(callback_data):
        if callback_data in {
            "admin_general_settings",
            "admin_system_messages",
            "admin_panel",
            "admin_language_settings",
            "noop",
        }:
            return "navigation"
        if callback_data.startswith(("admin_general_toggle_", "admin_general_edit_", "admin_general_set_")):
            return "mutation"
        if callback_data.startswith("admin_main_collections_page_"):
            return "navigation"
        parsed = parse_system_message_callback(callback_data)
        assert parsed is not None, callback_data
        action = parsed["action"]
        if action == "r":
            return "destructive"
        if action in {"e", "s"}:
            return "mutation"
        if action in {"c", "d", "l", "p", "v"}:
            return "navigation"
        raise AssertionError(f"unclassified system-message callback: {callback_data}")

    def inspect_markup(markup):
        buttons = [
            button
            for row in (markup.inline_keyboard if markup else [])
            for button in row
            if button.callback_data
        ]
        assert buttons
        for button in buttons:
            assert len(button.callback_data.encode("utf-8")) <= 64
            assert "ui." not in button.text
            categories[button.callback_data] = classify(button.callback_data)
        return buttons

    async def press(callback_data, *, expect_render=True):
        nonlocal update_id
        update_id += 1
        session.calls.clear()
        await _feed_callback(admin_dispatcher, bot, root_message, callback_data, update_id)
        for method in session.calls:
            if isinstance(method, AnswerCallbackQuery):
                assert len((method.text or "").encode("utf-8")) <= 200
        if not expect_render:
            return None
        rendered = [method for method in session.calls if isinstance(method, (EditMessageText, SendMessage))]
        assert rendered, f"{callback_data}: {[(type(method).__name__, getattr(method, 'text', None)) for method in session.calls]}"
        return rendered[-1]

    await press("admin_general_settings")
    system_list = await press("admin_system_messages")
    total_pages = system_message_page_count("telegram")

    async def restore_page(target_page, current_list):
        current_page = 0
        while current_page < target_page:
            next_button = next(
                button
                for button in inspect_markup(current_list.reply_markup)
                if parse_system_message_callback(button.callback_data) is not None
                and parse_system_message_callback(button.callback_data)["action"] == "p"
                and int(parse_system_message_callback(button.callback_data)["page"]) == current_page + 1
            )
            current_list = await press(next_button.callback_data)
            current_page += 1
        return current_list

    for page in range(total_pages):
        if page:
            system_list = await restore_page(page, system_list)
        list_buttons = inspect_markup(system_list.reply_markup)
        locale_buttons = [
            button
            for button in list_buttons
            if parse_system_message_callback(button.callback_data)
            and parse_system_message_callback(button.callback_data)["action"] == "l"
        ]
        for locale_button in locale_buttons:
            system_list = await press(locale_button.callback_data)
        ru_button = next(
            button
            for button in inspect_markup(system_list.reply_markup)
            if parse_system_message_callback(button.callback_data)
            and parse_system_message_callback(button.callback_data)["action"] == "l"
            and parse_system_message_callback(button.callback_data)["locale"] == "ru"
        )
        system_list = await press(ru_button.callback_data)
        list_buttons = inspect_markup(system_list.reply_markup)

        item_buttons = [
            button
            for button in list_buttons
            if parse_system_message_callback(button.callback_data)
            and parse_system_message_callback(button.callback_data)["action"] == "d"
        ]
        for item_index, item_button in enumerate(item_buttons):
            detail = await press(item_button.callback_data)
            detail_buttons = inspect_markup(detail.reply_markup)
            for detail_button in detail_buttons:
                parsed = parse_system_message_callback(detail_button.callback_data)
                if not parsed or parsed["action"] != "v":
                    continue
                preview = await press(detail_button.callback_data)
                back_button = next(
                    button
                    for button in inspect_markup(preview.reply_markup)
                    if parse_system_message_callback(button.callback_data)
                    and parse_system_message_callback(button.callback_data)["action"] == "d"
                )
                detail = await press(back_button.callback_data)
            detail_buttons = inspect_markup(detail.reply_markup)
            for detail_button in detail_buttons:
                parsed = parse_system_message_callback(detail_button.callback_data)
                if not parsed or parsed["action"] != "e":
                    continue
                await press(detail_button.callback_data)
                update_id += 1
                await admin_dispatcher.feed_update(
                    bot,
                    Update(
                        update_id=update_id,
                        message=_admin_message(
                            bot,
                            user_id=13,
                            message_id=update_id,
                            text=f"CATALOG DRAFT {page} {item_index} {parsed['style']}",
                        ),
                    ),
                )
                draft_rendered = [
                    method for method in session.calls if isinstance(method, SendMessage)
                ]
                assert draft_rendered
                draft_method = next(
                    method for method in reversed(draft_rendered) if method.reply_markup
                )
                draft_buttons = inspect_markup(draft_method.reply_markup)
                draft_action = "c" if parsed["style"] == "informal" else "s"
                draft_button = next(
                    button
                    for button in draft_buttons
                    if parse_system_message_callback(button.callback_data)
                    and parse_system_message_callback(button.callback_data)["action"] == draft_action
                )
                detail = await press(draft_button.callback_data)
            detail_buttons = inspect_markup(detail.reply_markup)
            for detail_button in detail_buttons:
                parsed = parse_system_message_callback(detail_button.callback_data)
                if not parsed or parsed["action"] != "r":
                    continue
                detail = await press(detail_button.callback_data)
            back_button = next(
                button
                for button in inspect_markup(detail.reply_markup)
                if parse_system_message_callback(button.callback_data)
                and parse_system_message_callback(button.callback_data)["action"] == "p"
            )
            system_list = await press(back_button.callback_data)

        list_buttons = inspect_markup(system_list.reply_markup)
        page_buttons = [
            button
            for button in list_buttons
            if parse_system_message_callback(button.callback_data)
            and parse_system_message_callback(button.callback_data)["action"] == "p"
        ]
        for page_button in page_buttons:
            target = int(parse_system_message_callback(page_button.callback_data)["page"])
            if target == page:
                continue
            target_list = await press(page_button.callback_data)
            return_button = next(
                button
                for button in inspect_markup(target_list.reply_markup)
                if parse_system_message_callback(button.callback_data)
                and parse_system_message_callback(button.callback_data)["action"] == "p"
                and int(parse_system_message_callback(button.callback_data)["page"]) == page
            )
            system_list = await press(return_button.callback_data)
        general_button = next(
            button for button in inspect_markup(system_list.reply_markup)
            if button.callback_data == "admin_general_settings"
        )
        general = await press(general_button.callback_data)
        system_button = next(
            button for button in inspect_markup(general.reply_markup)
            if button.callback_data == "admin_system_messages"
        )
        system_list = await press(system_button.callback_data)

    assert set(categories.values()) == {"navigation", "mutation", "destructive"}


@pytest.mark.asyncio
async def test_system_message_save_rejects_stale_telegram_edit_context(factory, admin_dispatcher, monkeypatch):
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))

    async with factory() as db_session:
        db_session.add(DBUser(id=14, is_admin=True, first_name="Admin"))
        await db_session.commit()

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    root_message = _admin_message(bot, text="/admin")
    await _feed_callback(admin_dispatcher, bot, root_message, "admin_system_message:ui.ai.busy:ru:2", 700)
    await _feed_callback(admin_dispatcher, bot, root_message, "admin_system_edit:ui.ai.busy:ru:formal:2", 701)
    await admin_dispatcher.feed_update(
        bot,
        Update(
            update_id=702,
            message=_admin_message(bot, message_id=10, text="STALE DRAFT"),
        ),
    )
    await _feed_callback(admin_dispatcher, bot, root_message, "admin_system_edit:ui.ai.thinking:ru:informal:0", 703)
    await _feed_callback(admin_dispatcher, bot, root_message, "admin_system_save:ui.ai.busy:ru:formal:2", 704)

    async with factory() as db_session:
        assert await db_session.scalar(
            select(BotSystemMessage).where(
                BotSystemMessage.translation_key == "ui.ai.busy",
                BotSystemMessage.addressing_style == "formal",
            )
        ) is None


@pytest.mark.asyncio
async def test_legacy_telegram_system_message_mutations_are_non_mutating(factory, admin_dispatcher, monkeypatch):
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))

    async with factory() as db_session:
        db_session.add_all([
            DBUser(id=17, is_admin=True, first_name="Admin"),
            BotSystemMessage(
                bot_scope="telegram",
                translation_key="ui.ai.busy",
                locale="ru",
                addressing_style="formal",
                text="ORIGINAL TELEGRAM TEXT",
            ),
        ])
        await db_session.commit()

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    root_message = _admin_message(bot, user_id=17, text="/admin")
    await _feed_callback(
        admin_dispatcher,
        bot,
        root_message,
        "admin_system_edit:ui.ai.busy:ru:formal:0",
        710,
    )
    await admin_dispatcher.feed_update(
        bot,
        Update(
            update_id=711,
            message=_admin_message(bot, user_id=17, message_id=11, text="LEGACY SAVE MUST NOT APPLY"),
        ),
    )
    await _feed_callback(
        admin_dispatcher,
        bot,
        root_message,
        "admin_system_save:ui.ai.busy:ru:formal:0",
        712,
    )

    async with factory() as db_session:
        row = await db_session.scalar(
            select(BotSystemMessage).where(
                BotSystemMessage.bot_scope == "telegram",
                BotSystemMessage.translation_key == "ui.ai.busy",
                BotSystemMessage.locale == "ru",
                BotSystemMessage.addressing_style == "formal",
            )
        )
        assert row.text == "ORIGINAL TELEGRAM TEXT"

    await _feed_callback(
        admin_dispatcher,
        bot,
        root_message,
        "admin_system_reset:ui.ai.busy:ru:formal:0",
        713,
    )
    async with factory() as db_session:
        row = await db_session.scalar(
            select(BotSystemMessage).where(
                BotSystemMessage.bot_scope == "telegram",
                BotSystemMessage.translation_key == "ui.ai.busy",
                BotSystemMessage.locale == "ru",
                BotSystemMessage.addressing_style == "formal",
            )
        )
        assert row.text == "ORIGINAL TELEGRAM TEXT"


@pytest.mark.asyncio
async def test_system_message_telegram_stale_save_after_cancel_locale_and_back(factory, admin_dispatcher, monkeypatch):
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))

    async with factory() as db_session:
        db_session.add(DBUser(id=16, is_admin=True, first_name="Admin"))
        await db_session.commit()

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    root_message = _admin_message(bot, user_id=16, text="/admin")
    update_id = 750

    async def feed(data):
        nonlocal update_id
        update_id += 1
        session.calls.clear()
        await _feed_callback(admin_dispatcher, bot, root_message, data, update_id)
        for method in session.calls:
            if isinstance(method, AnswerCallbackQuery):
                assert len((method.text or "").encode("utf-8")) <= 200

    async def draft(text_value):
        nonlocal update_id
        update_id += 1
        await admin_dispatcher.feed_update(
            bot,
            Update(
                update_id=update_id,
                message=_admin_message(bot, user_id=16, message_id=update_id, text=text_value),
            ),
        )

    key = "ui.ai.thinking"
    edit = system_message_callback_data("e", key, "ru", "formal", 0)
    save = system_message_callback_data("s", key, "ru", "formal", 0)
    cancel = system_message_callback_data("c", key, "ru", "formal", 0)

    await feed(edit)
    await draft("CANCELLED DRAFT")
    await feed(cancel)
    await feed(save)

    await feed(edit)
    await draft("LOCALE DRAFT")
    await feed(system_message_callback_data("l", locale="en", page=0))
    await feed(save)

    await feed(edit)
    await draft("BACK DRAFT")
    await feed("admin_general_settings")
    await feed(save)

    async with factory() as db_session:
        assert await db_session.scalar(
            select(BotSystemMessage).where(
                BotSystemMessage.translation_key == key,
                BotSystemMessage.addressing_style == "formal",
            )
        ) is None


@pytest.mark.asyncio
async def test_system_message_detail_and_preview_fit_telegram_text_limit(factory, admin_dispatcher, monkeypatch):
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))

    async with factory() as db_session:
        db_session.add_all([
            DBUser(id=15, is_admin=True, first_name="Admin"),
            BotSystemMessage(
                translation_key="ui.access.subscription_required",
                locale="ru",
                addressing_style="informal",
                text="<b>" + "A" * 4092 + "</b>",
            ),
            BotSystemMessage(
                translation_key="ui.access.subscription_required",
                locale="ru",
                addressing_style="formal",
                text="<b>" + "B" * 4092 + "</b>",
            ),
        ])
        await db_session.commit()

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    root_message = _admin_message(bot, text="/admin")
    await _feed_callback(
        admin_dispatcher,
        bot,
        root_message,
        "admin_system_message:ui.access.subscription_required:ru:4",
        705,
    )
    detail = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    assert len(detail.text) <= 4096
    assert "&lt;b&gt;" in detail.text

    await _feed_callback(
        admin_dispatcher,
        bot,
        root_message,
        "admin_system_edit:ui.access.subscription_required:ru:formal:4",
        7051,
    )
    editor = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    assert len(editor.text) <= 4096

    await _feed_callback(
        admin_dispatcher,
        bot,
        root_message,
        "admin_system_preview:ui.access.subscription_required:ru:formal:4",
        706,
    )
    preview = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    assert len(preview.text) <= 4096


@pytest.mark.asyncio
async def test_admin_audio_deepgram_model_selection_returns_to_audio_section(factory, admin_dispatcher, monkeypatch):
    import admin_content_authoring
    import automation_admin
    import handlers

    for module in (handlers, admin_content_authoring, automation_admin):
        monkeypatch.setattr(module, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(automation_admin, "get_all_admin_ids", AsyncMock(return_value={11}))

    async with factory() as session:
        session.add_all([
            DBUser(id=11, is_admin=True, first_name="Admin"),
            AIConfig(id=1, provider="Deepseek", transcription_provider="OpenAI"),
            SubscriptionConfig(id=1),
        ])
        await session.commit()

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    dispatcher = admin_dispatcher
    message = _admin_message(bot, user_id=11, text="/admin")

    async def press(data, update_id):
        session.calls.clear()
        await _feed_callback(dispatcher, bot, message, data, update_id)
        return next(
            method
            for method in reversed(session.calls)
            if isinstance(method, (EditMessageText, SendMessage))
        )

    await dispatcher.feed_update(bot, Update(update_id=600, message=message))
    await press("admin_ai_settings", 601)
    await press("admin_ai_keys", 602)
    await press("admin_ai_audio", 603)
    picker = await press("admin_select_transcription_provider", 604)
    deepgram_provider = next(
        button
        for row in picker.reply_markup.inline_keyboard
        for button in row
        if button.callback_data == "admin_choose_capability_transcription_Deepgram"
    )
    models = await press(deepgram_provider.callback_data, 605)
    model_button = next(
        button
        for row in models.reply_markup.inline_keyboard
        for button in row
        if button.callback_data and button.callback_data.startswith("ai_m_")
    )
    audio = await press(model_button.callback_data, 606)

    assert "🎙 <b>Аудио</b>" in audio.text
    assert "Deepgram" in audio.text


@pytest.mark.asyncio
async def test_admin_dispatcher_crawls_every_visible_top_level_button(factory, admin_dispatcher, monkeypatch):
    import admin_content_authoring
    import automation_admin
    import handlers
    import keyboards

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(admin_content_authoring, "async_session_maker", factory)
    monkeypatch.setattr(automation_admin, "async_session_maker", factory)
    monkeypatch.setattr(keyboards, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(automation_admin, "get_all_admin_ids", AsyncMock(return_value={11}))
    monkeypatch.setattr(
        handlers,
        "get_current_pm2_identity",
        AsyncMock(return_value={"pm2_id": 1, "name": "test", "app_port": 8080}),
    )

    async with factory() as session:
        session.add_all([
            DBUser(id=11, is_admin=True, first_name="Admin"),
            AIConfig(id=1),
            SubscriptionConfig(id=1),
        ])
        await session.commit()

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    dispatcher = admin_dispatcher

    root_message = _admin_message(bot, text="/admin")
    await dispatcher.feed_update(bot, Update(update_id=200, message=root_message))
    root = next(method for method in reversed(session.calls) if isinstance(method, SendMessage))
    visible = [
        button
        for row in root.reply_markup.inline_keyboard
        for button in row
        if button.callback_data
    ]
    expected = {
        "admin_stats": ("navigation", "Статистика"),
        "admin_clients_page_0": ("navigation", "Список клиентов"),
        "admin_general_settings": ("navigation", "Общие настройки"),
        "admin_test_menu": ("navigation", "Управление разделом"),
        "automation_menu": ("navigation", "Автоматизации"),
        "admin_ai_settings": ("navigation", "Настройки ИИ"),
        "admin_subscriptions": ("navigation", "Управление подписками"),
        "admin_kb_page_0": ("navigation", "База знаний"),
        "admin_collections_page_0": ("navigation", "Медиа-коллекции"),
        "admin_content": ("navigation", "Контент"),
        "admin_topics_page_0": ("navigation", "Управление темами"),
        "admin_manage_buttons": ("navigation", "Управление кнопками"),
        "admin_manage_admins": ("navigation", "Управление администраторами"),
        "admin_mailing_menu": ("navigation", "Управление рассылками"),
        "admin_restart_bot": ("destructive", "Перезагрузить текущего бота"),
    }
    discovered = {button.callback_data for button in visible}
    assert discovered == set(expected), "new visible admin button requires navigation contract entry"
    assert len(visible) == len(expected) == 15

    for index, button in enumerate(visible, start=1):
        session.calls.clear()
        context = dispatcher.fsm.get_context(bot=bot, chat_id=11, user_id=11)
        await context.clear()
        await _feed_callback(dispatcher, bot, root_message, button.callback_data, 200 + index)
        text_methods = [
            method
            for method in session.calls
            if isinstance(method, (EditMessageText, SendMessage))
        ]
        assert text_methods, button.callback_data
        destination = text_methods[-1]
        assert expected[button.callback_data][1] in destination.text, button.callback_data
        if expected[button.callback_data][0] == "destructive":
            callbacks = {
                item.callback_data
                for row in destination.reply_markup.inline_keyboard
                for item in row
                if item.callback_data
            }
            assert "admin_panel" in callbacks
        else:
            await _feed_callback(dispatcher, bot, root_message, "admin_panel", 300 + index)
            returned = next(
                method
                for method in reversed(session.calls)
                if isinstance(method, EditMessageText)
            )
            assert returned.text == "Добро пожаловать в админ-панель!"


@pytest.mark.asyncio
async def test_admin_dispatcher_crawls_general_languages_and_ai_sections(factory, admin_dispatcher, monkeypatch):
    import admin_content_authoring
    import automation_admin
    import handlers
    import keyboards

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(admin_content_authoring, "async_session_maker", factory)
    monkeypatch.setattr(automation_admin, "async_session_maker", factory)
    monkeypatch.setattr(keyboards, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(automation_admin, "get_all_admin_ids", AsyncMock(return_value={11}))

    async with factory() as session:
        session.add_all([
            DBUser(id=11, is_admin=True, first_name="Admin"),
            AIConfig(id=1, provider="Deepseek"),
            SubscriptionConfig(id=1),
        ])
        await session.commit()

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    dispatcher = admin_dispatcher
    root_message = _admin_message(bot, text="/admin")
    await dispatcher.feed_update(bot, Update(update_id=400, message=root_message))

    async def screen(callback_data, heading, update_id):
        session.calls.clear()
        context = dispatcher.fsm.get_context(bot=bot, chat_id=11, user_id=11)
        await context.clear()
        await _feed_callback(dispatcher, bot, root_message, callback_data, update_id)
        methods = [
            method
            for method in session.calls
            if isinstance(method, (EditMessageText, SendMessage))
        ]
        assert methods, callback_data
        rendered = methods[-1]
        assert isinstance(rendered.text, str)
        assert heading in rendered.text, callback_data
        return rendered

    general = await screen("admin_general_settings", "Общие настройки", 401)
    general_callbacks = {
        button.callback_data
        for row in general.reply_markup.inline_keyboard
        for button in row
        if button.callback_data
    }
    assert {
        "admin_general_toggle_profile_name",
        "admin_general_toggle_profile_gender",
        "admin_general_toggle_profile_age",
        "admin_general_toggle_language_selection",
        "admin_general_toggle_ai_processing_message",
        "admin_system_messages",
        "admin_general_toggle_addressing_style",
        "admin_main_collections_page_0",
        "admin_language_settings",
        "admin_panel",
    } <= general_callbacks
    assert "admin_general_edit_ai_processing_message_text" not in general_callbacks
    assert "admin_general_open_ai_processing_variants" not in general_callbacks

    toggled = await screen("admin_general_toggle_profile_name", "Общие настройки", 402)
    assert "Имя:" in " ".join(
        button.text
        for row in toggled.reply_markup.inline_keyboard
        for button in row
    )
    await screen("admin_general_toggle_profile_gender", "Общие настройки", 403)
    await screen("admin_general_toggle_profile_age", "Общие настройки", 404)
    session.calls.clear()
    await _feed_callback(dispatcher, bot, root_message, "admin_general_toggle_language_selection", 405)
    assert any(method.__class__.__name__ == "AnswerCallbackQuery" for method in session.calls)
    await screen("admin_main_collections_page_0", "Медиаколлекции основного диалога", 406)
    await screen("admin_language_settings", "Языки", 407)
    language = await screen("admin_toggle_multilingual_authoring", "Языки", 408)
    language_callbacks = {
        button.callback_data
        for row in language.reply_markup.inline_keyboard
        for button in row
        if button.callback_data
    }
    assert "admin_language_locale_en" in language_callbacks
    await screen("admin_language_locale_en", "English", 409)
    await screen("admin_language_settings", "Языки", 410)
    await screen("admin_general_settings", "Общие настройки", 411)
    await screen("admin_panel", "Добро пожаловать в админ-панель", 412)

    ai = await screen("admin_ai_settings", "Настройки ИИ", 413)
    ai_callbacks = {
        button.callback_data
        for row in ai.reply_markup.inline_keyboard
        for button in row
        if button.callback_data
    }
    assert "admin_ai_keys" in ai_callbacks
    assert "toggle_deepseek_thinking" not in ai_callbacks
    assert "set_max_output_tokens" not in ai_callbacks

    keys = await screen("admin_ai_keys", "Провайдеры и модели", 414)
    key_buttons = [
        button for row in keys.reply_markup.inline_keyboard for button in row
        if button.callback_data and button.callback_data.startswith("set_key_")
    ]
    model_buttons = [
        button for row in keys.reply_markup.inline_keyboard for button in row
        if button.callback_data and button.callback_data.startswith("view_models_")
    ]
    assert len(key_buttons) == 8
    assert len(model_buttons) == 8
    assert all(len(row) == 2 for row in keys.reply_markup.inline_keyboard[:8])
    assert {"admin_ai_text_fallback", "admin_ai_audio", "admin_ai_vision", "admin_ai_common"} <= {
        button.callback_data
        for row in keys.reply_markup.inline_keyboard
        for button in row
        if button.callback_data
    }

    card = await screen("view_models_Deepseek", "Параметры модели", 415)
    assert "Провайдер:" in card.text and "Deepseek" in card.text
    card_callbacks = {
        button.callback_data
        for row in card.reply_markup.inline_keyboard
        for button in row
        if button.callback_data
    }
    assert "model_setting_max_tokens_Deepseek" in card_callbacks
    assert "model_setting_reasoning_Deepseek" in card_callbacks
    assert "model_setting_api_key_Deepseek" in card_callbacks
    assert "admin_ai_deepseek_proxy" in card_callbacks
    await screen("model_setting_reasoning_Deepseek", "Reasoning", 416)
    await screen("model_reasoning_Deepseek_max", "Параметры модели", 417)
    await screen("model_setting_max_tokens_Deepseek", "Max tokens", 418)
    prompt = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    cancel_callback = next(
        button.callback_data
        for row in prompt.reply_markup.inline_keyboard
        for button in row
        if button.callback_data and button.callback_data.startswith("cancel_state_")
    )
    session.calls.clear()
    await _feed_callback(dispatcher, bot, root_message, cancel_callback, 419)
    canceled = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    assert isinstance(canceled.text, str)
    assert "Параметры модели" in canceled.text
    await screen("admin_ai_keys", "Провайдеры и модели", 420)
    await screen("admin_ai_settings", "Настройки ИИ", 421)
    await screen("admin_panel", "Добро пожаловать в админ-панель", 422)


@pytest.mark.asyncio
async def test_telegram_openrouter_qwen_reasoning_is_a_real_model_scoped_journey(factory, admin_dispatcher, monkeypatch):
    import ai_integration
    import admin_content_authoring
    import automation_admin
    import handlers
    import keyboards
    import httpx
    from database import AIModelSettings, Topic
    from provider_models import resolve_reasoning_callback_token, resolve_telegram_model_callback

    for module in (handlers, admin_content_authoring, automation_admin, keyboards):
        monkeypatch.setattr(module, "async_session_maker", factory)
    monkeypatch.setattr(ai_integration, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(automation_admin, "get_all_admin_ids", AsyncMock(return_value={11}))

    async with factory() as session:
        session.add_all([
            DBUser(id=11, is_admin=True, first_name="Admin", current_topic_id=1, current_dialogue_id=1),
            Topic(id=1, name="Тестовая тема", is_active=True, system_prompt="Система"),
            AIConfig(
                id=1,
                provider="OpenRouter",
                openrouter_api_key="sk-or-test-key",
                openrouter_model="qwen/qwen3.8-27b",
                system_prompt="Система",
                shared_prompt_block="",
                memory_mode="global",
            ),
            SubscriptionConfig(id=1),
        ])
        await session.commit()

    wire_payloads = []

    async def spy_post(self, url, *args, **kwargs):
        wire_payloads.append(kwargs["json"])
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", spy_post)

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    message = _admin_message(bot, user_id=11, text="/admin")
    update_id = 5000
    classified_callbacks = {}
    executed_callbacks = set()

    async def press(current_message, callback_data):
        nonlocal update_id
        visible = {
            button.callback_data
            for button in _ai_callback_buttons(current_message.reply_markup)
        }
        assert callback_data in visible
        executed_callbacks.add(callback_data)
        session.calls.clear()
        update_id += 1
        await _feed_callback(admin_dispatcher, bot, current_message, callback_data, update_id)
        rendered = next(
            method
            for method in reversed(session.calls)
            if isinstance(method, (EditMessageText, SendMessage))
        )
        _classify_telegram_ai_markup(rendered.reply_markup, classified_callbacks)
        return message.model_copy(update={
            "text": rendered.text,
            "reply_markup": rendered.reply_markup,
        })

    await admin_dispatcher.feed_update(bot, Update(update_id=update_id, message=message))
    initial = next(
        method
        for method in reversed(session.calls)
        if isinstance(method, (EditMessageText, SendMessage))
    )
    message = message.model_copy(update={
        "text": initial.text,
        "reply_markup": initial.reply_markup,
    })
    message = await press(message, "admin_ai_settings")
    message = await press(message, "admin_ai_keys")
    message = await press(message, "view_models_OpenRouter")
    message = await press(message, "view_provider_models_OpenRouter")

    qwen_button = next(
        button
        for button in _ai_callback_buttons(message.reply_markup)
        if "Qwen3.8 27B" in button.text
    )
    assert len(qwen_button.callback_data.encode()) <= 64
    resolved_model = resolve_telegram_model_callback(qwen_button.callback_data)
    assert resolved_model == ("OpenRouter", "chat", "qwen/qwen3.8-27b")
    message = await press(message, qwen_button.callback_data)
    assert "Модель: <code>qwen/qwen3.8-27b</code>" in message.text

    reasoning_entry = next(
        button
        for button in _ai_callback_buttons(message.reply_markup)
        if button.callback_data.startswith("model_setting_reasoning_r_")
    )
    message = await press(message, reasoning_entry.callback_data)
    assert "🧠 <b>Reasoning</b>" in message.text
    mode_buttons = _ai_callback_buttons(message.reply_markup)
    assert [button.text for button in mode_buttons[:-1]] == ["Авто", "Выкл", "Low", "Medium", "Max"]
    assert all(len(button.callback_data.encode()) <= 64 for button in mode_buttons)
    parent = mode_buttons[-1].callback_data
    assert parent == "view_models_OpenRouter"
    visible_modes = [(button.text, button.callback_data) for button in mode_buttons[:-1]]
    assert resolve_reasoning_callback_token(
        dict(visible_modes)["Max"].replace("model_reasoning_", "", 1)
    ) == ("OpenRouter", "qwen/qwen3.8-27b", "max")
    for index, (label, mode_callback) in enumerate(visible_modes):
        if index:
            message = await press(message, reasoning_entry.callback_data)
            assert [button.text for button in _ai_callback_buttons(message.reply_markup)[:-1]] == [
                "Авто", "Выкл", "Low", "Medium", "Max"
            ]
        picker_buttons = _ai_callback_buttons(message.reply_markup)
        assert next(button for button in picker_buttons if button.text == label).callback_data == mode_callback
        assert picker_buttons[-1].callback_data == parent
        message = await press(message, mode_callback)
        assert f"Reasoning: <b>{label}</b>" in message.text
        async with factory() as session_check:
            stored = await session_check.scalar(
                select(AIModelSettings).where(
                    AIModelSettings.provider == "OpenRouter",
                    AIModelSettings.model == "qwen/qwen3.8-27b",
                    AIModelSettings.channel == "chat",
                )
            )
            assert stored.reasoning_effort == {"Авто": "auto", "Выкл": "none", "Low": "low", "Medium": "medium", "Max": "max"}[label]
        assert await ai_integration.generate_response(
            11,
            f"Telegram {label}",
            track_user_activity=False,
            minutes_since_last_visit=1,
            minutes_since_last_message=1,
        ) == "ok"
        payload = wire_payloads[-1]
        assert payload["model"] == "qwen/qwen3.8-27b"
        expected_reasoning = {
            "Авто": None,
            "Выкл": {"effort": "none", "exclude": True},
            "Low": {"effort": "low", "exclude": True},
            "Medium": {"effort": "medium", "exclude": True},
            "Max": {"effort": "xhigh", "exclude": True},
        }[label]
        if expected_reasoning is None:
            assert "reasoning" not in payload
            assert "provider" not in payload
        else:
            assert payload["reasoning"] == expected_reasoning
            assert payload["provider"] == {"require_parameters": True}
        assert "extra_body" not in payload
        assert "reasoning_effort" not in payload
        message = await press(message, reasoning_entry.callback_data)
        back = next(button.callback_data for button in _ai_callback_buttons(message.reply_markup) if button.text == "⬅️ Назад")
        assert back == parent
        message = await press(message, back)
        assert f"Reasoning: <b>{label}</b>" in message.text

    assert {callback for _, callback in visible_modes} <= executed_callbacks
    assert reasoning_entry.callback_data in executed_callbacks
    assert parent in executed_callbacks
    assert classified_callbacks[reasoning_entry.callback_data] == "mutation"
    assert all(classified_callbacks[callback] == "mutation" for _, callback in visible_modes)
    assert classified_callbacks[parent] == "navigation"

    message = await press(message, "view_provider_models_OpenRouter")
    qwen_vl_button = next(
        button
        for button in _ai_callback_buttons(message.reply_markup)
        if "Qwen3 VL 235B Instruct" in button.text
    )
    message = await press(message, qwen_vl_button.callback_data)
    assert "Модель: <code>qwen/qwen3-vl-235b-a22b-instruct</code>" in message.text
    assert not any(
        button.callback_data.startswith("model_setting_reasoning")
        for button in _ai_callback_buttons(message.reply_markup)
    )
    assert "Reasoning:" not in message.text
    assert await ai_integration.generate_response(
        11,
        "Telegram Qwen VL",
        track_user_activity=False,
        minutes_since_last_visit=1,
        minutes_since_last_message=1,
    ) == "ok"
    assert wire_payloads[-1]["model"] == "qwen/qwen3-vl-235b-a22b-instruct"
    assert "reasoning" not in wire_payloads[-1]
    assert "provider" not in wire_payloads[-1]

    message = await press(message, "view_provider_models_OpenRouter")
    qwen_back_button = next(
        button
        for button in _ai_callback_buttons(message.reply_markup)
        if "Qwen3.8 27B" in button.text
    )
    message = await press(message, qwen_back_button.callback_data)
    assert "Reasoning: <b>Max</b>" in message.text
    assert await ai_integration.generate_response(
        11,
        "Telegram Qwen restored",
        track_user_activity=False,
        minutes_since_last_visit=1,
        minutes_since_last_message=1,
    ) == "ok"
    assert wire_payloads[-1]["model"] == "qwen/qwen3.8-27b"
    assert wire_payloads[-1]["reasoning"] == {"effort": "xhigh", "exclude": True}
    assert wire_payloads[-1]["provider"] == {"require_parameters": True}


@pytest.mark.parametrize(
    ("model", "friendly_name", "modes"),
    [
        ("qwen/qwen3.7-plus", "Qwen3.7 Plus", [("Авто", "auto"), ("Выкл", "none"), ("Вкл", "enabled")]),
        ("qwen/qwen3.8-27b", "Qwen3.8 27B", [("Авто", "auto"), ("Выкл", "none"), ("Low", "low"), ("Medium", "medium"), ("Max", "max")]),
        ("deepseek/deepseek-v4-flash", "DeepSeek V4 Flash", [("Авто", "auto"), ("Выкл", "none"), ("High", "high"), ("Max", "max")]),
        ("deepseek/deepseek-v4-pro", "DeepSeek V4 Pro", [("Авто", "auto"), ("Выкл", "none"), ("High", "high"), ("Max", "max")]),
        ("deepseek/deepseek-v4.1-flash", "DeepSeek V4.1 Flash", [("Авто", "auto"), ("Выкл", "none"), ("Low", "low"), ("High", "high"), ("Max", "max")]),
    ],
)
@pytest.mark.asyncio
async def test_telegram_openrouter_reasoning_modes_have_real_scoped_wire_journeys(
    factory,
    admin_dispatcher,
    monkeypatch,
    model,
    friendly_name,
    modes,
):
    import ai_integration
    import admin_content_authoring
    import automation_admin
    import handlers
    import httpx
    import keyboards
    from database import AIModelSettings, Topic
    from provider_models import resolve_reasoning_callback_token

    for module in (handlers, admin_content_authoring, automation_admin, keyboards):
        monkeypatch.setattr(module, "async_session_maker", factory)
    monkeypatch.setattr(ai_integration, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(automation_admin, "get_all_admin_ids", AsyncMock(return_value={11}))

    async with factory() as session:
        session.add_all([
            DBUser(id=11, is_admin=True, first_name="Admin", current_topic_id=1, current_dialogue_id=1),
            Topic(id=1, name="Тестовая тема", is_active=True, system_prompt="Система"),
            AIConfig(
                id=1,
                provider="OpenRouter",
                openrouter_api_key="sk-or-test-key",
                openrouter_model=model,
                system_prompt="Система",
                shared_prompt_block="",
                memory_mode="global",
            ),
            SubscriptionConfig(id=1),
        ])
        await session.commit()

    wire_payloads = []

    async def spy_post(self, url, *args, **kwargs):
        wire_payloads.append(kwargs["json"])
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", spy_post)
    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    message = _admin_message(bot, user_id=11, text="/admin")
    update_id = 7000
    classified_callbacks = {}
    executed_callbacks = set()

    async def press(current_message, callback_data):
        nonlocal update_id
        visible = {button.callback_data for button in _ai_callback_buttons(current_message.reply_markup)}
        assert callback_data in visible
        executed_callbacks.add(callback_data)
        session.calls.clear()
        update_id += 1
        await _feed_callback(admin_dispatcher, bot, current_message, callback_data, update_id)
        rendered = next(
            method
            for method in reversed(session.calls)
            if isinstance(method, (EditMessageText, SendMessage))
        )
        return message.model_copy(update={
            "text": rendered.text,
            "reply_markup": rendered.reply_markup,
        })

    await admin_dispatcher.feed_update(bot, Update(update_id=update_id, message=message))
    initial = next(
        method
        for method in reversed(session.calls)
        if isinstance(method, (EditMessageText, SendMessage))
    )
    message = message.model_copy(update={"text": initial.text, "reply_markup": initial.reply_markup})
    message = await press(message, "admin_ai_settings")
    message = await press(message, "admin_ai_keys")
    message = await press(message, "view_models_OpenRouter")
    message = await press(message, "view_provider_models_OpenRouter")
    for button in _ai_callback_buttons(message.reply_markup):
        classified_callbacks[button.callback_data] = _classify_telegram_ai_callback(button.callback_data)
    model_button = next(button for button in _ai_callback_buttons(message.reply_markup) if friendly_name in button.text)
    message = await press(message, model_button.callback_data)
    assert f"Модель: <code>{model}</code>" in message.text
    for button in _ai_callback_buttons(message.reply_markup):
        classified_callbacks[button.callback_data] = _classify_telegram_ai_callback(button.callback_data)
    reasoning_entry = next(
        button
        for button in _ai_callback_buttons(message.reply_markup)
        if button.callback_data.startswith("model_setting_reasoning_r_")
    )
    message = await press(message, reasoning_entry.callback_data)
    for button in _ai_callback_buttons(message.reply_markup):
        classified_callbacks[button.callback_data] = _classify_telegram_ai_callback(button.callback_data)
    picker_buttons = _ai_callback_buttons(message.reply_markup)
    assert [button.text for button in picker_buttons[:-1]] == [label for label, _ in modes]
    back_callback = picker_buttons[-1].callback_data
    assert back_callback == "view_models_OpenRouter"
    mode_callbacks = {button.text: button.callback_data for button in picker_buttons[:-1]}
    assert all(len(callback.encode()) <= 64 for callback in mode_callbacks.values())
    for index, (label, value) in enumerate(modes):
        if index:
            message = await press(message, reasoning_entry.callback_data)
            visible_picker = _ai_callback_buttons(message.reply_markup)
            assert [button.text for button in visible_picker[:-1]] == [label for label, _ in modes]
            for button in visible_picker:
                classified_callbacks[button.callback_data] = _classify_telegram_ai_callback(button.callback_data)
            assert visible_picker[-1].callback_data == back_callback
        message = await press(message, mode_callbacks[label])
        assert f"Reasoning: <b>{label}</b>" in message.text
        async with factory() as session_check:
            stored = await session_check.scalar(
                select(AIModelSettings).where(
                    AIModelSettings.provider == "OpenRouter",
                    AIModelSettings.model == model,
                    AIModelSettings.channel == "chat",
                )
            )
            assert stored.reasoning_effort == value
        assert await ai_integration.generate_response(
            11,
            f"Telegram {model} {value}",
            track_user_activity=False,
            minutes_since_last_visit=1,
            minutes_since_last_message=1,
        ) == "ok"
        payload = wire_payloads[-1]
        assert payload["model"] == model
        if value == "auto":
            assert "reasoning" not in payload
            assert "provider" not in payload
        elif model == "qwen/qwen3.7-plus" and value == "none":
            assert payload["reasoning"] == {"enabled": False}
            assert payload["provider"] == {"require_parameters": True}
        elif model == "qwen/qwen3.7-plus" and value == "enabled":
            assert payload["reasoning"] == {"enabled": True, "exclude": True}
            assert payload["provider"] == {"require_parameters": True}
        else:
            assert payload["reasoning"] == {
                "effort": (
                    "xhigh"
                    if value == "max" and model in {
                        "qwen/qwen3.8-27b",
                        "deepseek/deepseek-v4-flash",
                        "deepseek/deepseek-v4-pro",
                    }
                    else value
                ),
                "exclude": True,
            }
            assert payload["provider"] == {"require_parameters": True}
        assert "extra_body" not in payload
        assert "reasoning_effort" not in payload
        message = await press(message, reasoning_entry.callback_data)
        visible_picker = _ai_callback_buttons(message.reply_markup)
        assert [button.text for button in visible_picker[:-1]] == [label for label, _ in modes]
        for button in visible_picker:
            classified_callbacks[button.callback_data] = _classify_telegram_ai_callback(button.callback_data)
        assert visible_picker[-1].callback_data == back_callback
        message = await press(message, back_callback)
        assert f"Reasoning: <b>{label}</b>" in message.text
        assert classified_callbacks[back_callback] == "navigation"

    assert set(mode_callbacks.values()) <= executed_callbacks
    assert reasoning_entry.callback_data in executed_callbacks
    assert back_callback in executed_callbacks
    assert classified_callbacks[reasoning_entry.callback_data] == "mutation"
    assert all(classified_callbacks[callback] == "mutation" for callback in mode_callbacks.values())


@pytest.mark.asyncio
async def test_telegram_unresolved_reasoning_token_does_not_mutate_scoped_settings(factory, admin_dispatcher, monkeypatch):
    import automation_admin
    import handlers
    from database import AIModelSettings
    from provider_models import build_reasoning_callback_token

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(automation_admin, "get_all_admin_ids", AsyncMock(return_value={11}))
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))

    async with factory() as session:
        session.add_all([
            DBUser(id=11, is_admin=True, first_name="Admin"),
            AIConfig(
                id=1,
                provider="OpenRouter",
                deepseek_model="deepseek-v4-pro",
                openrouter_model="qwen/qwen3.8-27b",
            ),
            SubscriptionConfig(id=1),
            AIModelSettings(
                provider="Deepseek",
                model="deepseek-v4-pro",
                channel="chat",
                reasoning_effort="high",
            ),
            AIModelSettings(
                provider="OpenRouter",
                model="qwen/qwen3.8-27b",
                channel="chat",
                reasoning_effort="max",
            ),
        ])
        await session.commit()

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    message = _admin_message(bot, user_id=11)
    valid_token = build_reasoning_callback_token("OpenRouter", "qwen/qwen3.8-27b", "max")
    unresolved_token = "r_" + "0" * 32
    assert valid_token != unresolved_token

    await _feed_callback(
        admin_dispatcher,
        bot,
        message,
        f"model_reasoning_{unresolved_token}",
        5300,
    )

    async with factory() as session_check:
        deepseek = await session_check.scalar(
            select(AIModelSettings).where(
                AIModelSettings.provider == "Deepseek",
                AIModelSettings.model == "deepseek-v4-pro",
                AIModelSettings.channel == "chat",
            )
        )
        openrouter = await session_check.scalar(
            select(AIModelSettings).where(
                AIModelSettings.provider == "OpenRouter",
                AIModelSettings.model == "qwen/qwen3.8-27b",
                AIModelSettings.channel == "chat",
            )
        )
        config = await session_check.get(AIConfig, 1)
        assert deepseek.reasoning_effort == "high"
        assert openrouter.reasoning_effort == "max"
        assert config.provider == "OpenRouter"
        assert config.openrouter_model == "qwen/qwen3.8-27b"

    answer = next(method for method in session.calls if method.__class__.__name__ == "AnswerCallbackQuery")
    assert answer.show_alert is True

    async with factory() as session_update:
        config = await session_update.get(AIConfig, 1)
        config.openrouter_model = "qwen/qwen3.7-plus"
        await session_update.commit()
    session.calls.clear()
    await _feed_callback(
        admin_dispatcher,
        bot,
        message,
        f"model_reasoning_{valid_token}",
        5302,
    )
    async with factory() as session_check:
        openrouter = await session_check.scalar(
            select(AIModelSettings).where(
                AIModelSettings.provider == "OpenRouter",
                AIModelSettings.model == "qwen/qwen3.8-27b",
                AIModelSettings.channel == "chat",
            )
        )
        config = await session_check.get(AIConfig, 1)
        assert openrouter.reasoning_effort == "max"
        assert config.openrouter_model == "qwen/qwen3.7-plus"
    answer = next(method for method in session.calls if method.__class__.__name__ == "AnswerCallbackQuery")
    assert answer.show_alert is True

    session.calls.clear()
    await _feed_callback(
        admin_dispatcher,
        bot,
        message,
        "model_reasoning_NotAProvider_max",
        5301,
    )
    async with factory() as session_check:
        deepseek = await session_check.scalar(
            select(AIModelSettings).where(
                AIModelSettings.provider == "Deepseek",
                AIModelSettings.model == "deepseek-v4-pro",
                AIModelSettings.channel == "chat",
            )
        )
        openrouter = await session_check.scalar(
            select(AIModelSettings).where(
                AIModelSettings.provider == "OpenRouter",
                AIModelSettings.model == "qwen/qwen3.8-27b",
                AIModelSettings.channel == "chat",
            )
        )
        assert deepseek.reasoning_effort == "high"
        assert openrouter.reasoning_effort == "max"
    answer = next(method for method in session.calls if method.__class__.__name__ == "AnswerCallbackQuery")
    assert answer.show_alert is True


@pytest.mark.asyncio
async def test_ai_settings_changed_buttons_use_real_dispatcher_journeys(factory, admin_dispatcher, monkeypatch):
    import database
    import admin_content_authoring
    import automation_admin
    import handlers
    import keyboards

    for module in (handlers, admin_content_authoring, automation_admin, keyboards):
        monkeypatch.setattr(module, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(automation_admin, "get_all_admin_ids", AsyncMock(return_value={11}))

    async with factory() as session:
        session.add_all([
            DBUser(id=11, is_admin=True, first_name="Admin"),
            AIConfig(id=1, provider="Deepseek", deepseek_model="deepseek-flash", transcription_provider="OpenAI", vision_provider="Gemini", vision_model="gemini-3.7-flash", image_generation_provider="OpenAI", image_generation_model="gpt-image-2", image_edit_provider="KIE", image_edit_model="seedream/4.5-edit", allow_fallback=False, allow_vision_fallback=False),
            SubscriptionConfig(id=1),
        ])
        await session.commit()

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    dispatcher = admin_dispatcher
    current_message = _admin_message(bot, user_id=11, text="/admin")
    await dispatcher.feed_update(bot, Update(update_id=1200, message=current_message))
    current_markup = next(method for method in reversed(session.calls) if isinstance(method, SendMessage)).reply_markup
    update_id = 1200

    def buttons(markup):
        return [button for row in (markup.inline_keyboard if markup else []) for button in row if button.callback_data]

    def rendered():
        methods = [method for method in session.calls if isinstance(method, (EditMessageText, SendMessage))]
        assert methods
        assert isinstance(methods[-1].text, str)
        return methods[-1]

    async def press(data, expected=None):
        nonlocal current_message, current_markup, update_id
        await dispatcher.fsm.get_context(bot=bot, chat_id=11, user_id=11).clear()
        session.calls.clear()
        callback_message = current_message.model_copy(update={"reply_markup": current_markup, "text": getattr(current_message, "text", "Админ-панель")})
        update_id += 1
        await _feed_callback(dispatcher, bot, callback_message, data, update_id)
        result = rendered()
        if expected:
            assert expected in result.text
        current_markup = result.reply_markup
        current_message = current_message.model_copy(update={"reply_markup": current_markup, "text": result.text})
        return result

    await press("admin_ai_settings", "Настройки ИИ")
    keys = await press("admin_ai_keys", "Провайдеры и модели")
    key_buttons = buttons(keys.reply_markup)
    assert len([b for b in key_buttons if b.callback_data.startswith("set_key_")]) == 8
    providers = [b for b in key_buttons if b.callback_data.startswith("view_models_")]
    assert len(providers) == 8
    assert all(len(row) == 2 for row in keys.reply_markup.inline_keyboard[:8])

    for card in providers:
        detail = await press(card.callback_data, "Провайдер:")
        assert any(b.callback_data.startswith("model_setting_api_key_") for b in buttons(detail.reply_markup))
        if card.callback_data != "view_models_Deepgram":
            max_button = next(b for b in buttons(detail.reply_markup) if b.callback_data.startswith("model_setting_max_tokens_"))
            await press(max_button.callback_data, "Max tokens")
            prompt = rendered()
            update_id += 1
            await dispatcher.feed_update(bot, Update(update_id=update_id, message=_admin_message(bot, user_id=11, text="Авто").model_copy(update={"reply_markup": current_markup})))
            result = rendered()
            assert "Max tokens" in result.text
            current_markup = result.reply_markup
            current_message = current_message.model_copy(update={"reply_markup": current_markup, "text": result.text})
        async with factory() as verify:
            config = await verify.get(AIConfig, 1)
            assert config.provider == "Deepseek"
        await press("admin_ai_keys", "Провайдеры и модели")

    fallback = await press("admin_ai_text_fallback", "Резерв текста")
    await press("admin_ai_fallback_provider", "провайдера")
    provider_button = next(b for b in buttons(current_markup) if b.callback_data.startswith("admin_ai_fallback_set_provider_"))
    await press(provider_button.callback_data, "Резерв текста")
    await press("admin_ai_fallback_toggle", "Резерв текста")
    await press("admin_ai_fallback_toggle", "Резерв текста")
    await press("admin_ai_keys", "Провайдеры и модели")

    for callback_data, heading in (("admin_ai_audio", "Аудио"), ("admin_ai_vision", "Vision"), ("admin_ai_image_generation", "Генерация"), ("admin_ai_image_edit", "Редактирование"), ("admin_ai_common", "Общие настройки")):
        await press(callback_data, heading)
    await press("admin_ai_keys", "Провайдеры и модели")

    await press("admin_panel", "Добро пожаловать в админ-панель")


@pytest.mark.asyncio
async def test_telegram_ai_visible_button_contracts_use_complete_dispatcher_journeys(factory, admin_dispatcher, monkeypatch):
    import admin_content_authoring
    import automation_admin
    import handlers
    from database import AIModelSettings
    from provider_models import (
        ALL_PROVIDERS,
        PROVIDER_DEEPGRAM,
        PROVIDER_PERPLEXITY,
        resolve_telegram_model_callback,
    )

    for module in (handlers, admin_content_authoring, automation_admin):
        monkeypatch.setattr(module, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(automation_admin, "get_all_admin_ids", AsyncMock(return_value={11}))
    monkeypatch.setattr(handlers, "send_temp_notification", AsyncMock())

    async with factory() as session:
        session.add_all([
            DBUser(id=11, is_admin=True, first_name="Admin"),
            AIConfig(
                id=1,
                provider="Deepseek",
                deepseek_model="deepseek-flash",
                transcription_provider="OpenAI",
                vision_provider="Gemini",
                vision_model="gemini-3.7-flash",
                image_generation_provider="OpenAI",
                image_generation_model="gpt-image-2",
                image_edit_provider="KIE",
                image_edit_model="seedream/4.5-edit",
                fallback_provider=None,
                fallback_model=None,
                allow_fallback=False,
                allow_vision_fallback=False,
                vision_fallback_provider=None,
                vision_fallback_model=None,
                use_proxy=True,
            ),
            SubscriptionConfig(id=1),
        ])
        await session.commit()

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    dispatcher = admin_dispatcher
    message = _admin_message(bot, user_id=11, text="/admin")
    update_id = 2000
    current_markup = None
    current_text = "/admin"
    contracts = {}
    nested_back_contracts = {
        "provider_model_picker": [0, 8],
        "provider_api_key_cancel": [0, 8],
        "deepseek_reasoning": [0, 1],
        "text_fallback": [0, 8],
        "vision_fallback": [0, 7],
        "capability_pickers": [0, 19],
    }
    completed_flows = {
        "text_fallback": False,
        "vision_fallback": False,
        "transcription": False,
        "vision": False,
        "image_gen": False,
        "image_edit": False,
    }

    async def render_latest(expected=None, *, classify=True):
        nonlocal current_markup, current_text
        methods = [method for method in session.calls if isinstance(method, (EditMessageText, SendMessage))]
        assert methods
        rendered = methods[-1]
        assert isinstance(rendered.text, str)
        if expected:
            assert expected in rendered.text, rendered.text
        if classify:
            _classify_telegram_ai_markup(rendered.reply_markup, contracts)
        current_markup = rendered.reply_markup
        current_text = rendered.text
        return rendered

    async def press(callback_data, expected=None, *, classify=True):
        nonlocal update_id
        visible = {button.callback_data for button in _ai_callback_buttons(current_markup)}
        assert callback_data in visible, (callback_data, visible, current_text)
        session.calls.clear()
        update_id += 1
        callback_message = message.model_copy(update={"text": current_text, "reply_markup": current_markup})
        await _feed_callback(dispatcher, bot, callback_message, callback_data, update_id)
        return await render_latest(expected, classify=classify)

    async def send_text(value, expected=None):
        nonlocal update_id
        session.calls.clear()
        update_id += 1
        input_message = _admin_message(bot, user_id=11, message_id=message.message_id, text=value)
        input_message = input_message.model_copy(update={"reply_markup": current_markup})
        await dispatcher.feed_update(bot, Update(update_id=update_id, message=input_message))
        return await render_latest(expected)

    await dispatcher.feed_update(bot, Update(update_id=update_id, message=message))
    await render_latest("Добро пожаловать", classify=False)
    await press("admin_ai_settings", "Настройки ИИ", classify=False)
    keys = await press("admin_ai_keys", "Провайдеры и модели")
    key_buttons = _ai_callback_buttons(keys.reply_markup)
    assert [len(row) for row in keys.reply_markup.inline_keyboard[:8]] == [2] * 8
    assert len([button for button in key_buttons if button.callback_data.startswith("set_key_")]) == 8
    assert not {
        "admin_toggle_fallback",
        "admin_toggle_transcription",
        "admin_toggle_vision",
        "admin_toggle_image_generation",
        "admin_toggle_image_edit",
    } & {button.callback_data for button in key_buttons}
    provider_buttons = [button for button in key_buttons if button.callback_data.startswith("view_models_")]
    assert len(provider_buttons) == 8
    assert {button.callback_data.replace("view_models_", "", 1) for button in provider_buttons} == {*ALL_PROVIDERS, PROVIDER_DEEPGRAM}

    model_picker_passes = 0
    api_key_passes = 0
    max_token_passes = 0
    for provider_button in provider_buttons:
        provider = provider_button.callback_data.replace("view_models_", "", 1)
        detail = await press(provider_button.callback_data, "Провайдер:")
        api_button = next(button for button in _ai_callback_buttons(detail.reply_markup) if button.callback_data == f"model_setting_api_key_{provider}")
        prompt = await press(api_button.callback_data, "Отправьте новый API-ключ")
        cancel = next(button.callback_data for button in _ai_callback_buttons(prompt.reply_markup) if button.callback_data.startswith("cancel_state_view_models_"))
        detail = await press(cancel, "Параметры модели" if provider != PROVIDER_DEEPGRAM else provider)
        nested_back_contracts["provider_api_key_cancel"][0] += 1
        api_button = next(button for button in _ai_callback_buttons(detail.reply_markup) if button.callback_data == f"model_setting_api_key_{provider}")
        await press(api_button.callback_data, "Отправьте новый API-ключ")
        detail = await send_text(f"isolated-{provider.lower()}-key", provider)
        async with factory() as verify:
            config = await verify.get(AIConfig, 1)
            key_field = "deepgram_api_key" if provider == PROVIDER_DEEPGRAM else f"{provider.lower()}_api_key"
            assert getattr(config, key_field) == f"isolated-{provider.lower()}-key"
            assert config.provider == "Deepseek"
        api_key_passes += 1

        if provider == PROVIDER_PERPLEXITY:
            model_picker = await press("ai_ppx_models:0", "Прямые модели Perplexity")
            model_back = next(button.callback_data for button in _ai_callback_buttons(model_picker.reply_markup) if button.text == "К настройкам")
            detail = await press(model_back, "Параметры модели")
            nested_back_contracts["provider_model_picker"][0] += 1
            model_picker = await press("ai_ppx_models:0", "Прямые модели Perplexity")
            model_button = next(button for button in _ai_callback_buttons(model_picker.reply_markup) if button.callback_data.startswith("ai_m_"))
            resolved = resolve_telegram_model_callback(model_button.callback_data)
            assert resolved and resolved[0] == provider
            detail = await press(model_button.callback_data, "Провайдер:")
            assert resolved[2] in detail.text
            model_picker_passes += 1
        else:
            model_picker = await press(f"view_provider_models_{provider}", "Выберите модель")
            model_back = next(button.callback_data for button in _ai_callback_buttons(model_picker.reply_markup) if button.text == "⬅️ Назад")
            detail = await press(model_back, "Параметры модели" if provider != PROVIDER_DEEPGRAM else provider)
            nested_back_contracts["provider_model_picker"][0] += 1
            model_picker = await press(f"view_provider_models_{provider}", "Выберите модель")
            model_button = next(button for button in _ai_callback_buttons(model_picker.reply_markup) if button.callback_data.startswith("ai_m_"))
            resolved = resolve_telegram_model_callback(model_button.callback_data)
            assert resolved and resolved[0] == provider
            detail = await press(model_button.callback_data, "Провайдер:")
            assert resolved[2] in detail.text
            model_picker_passes += 1

        if provider != PROVIDER_DEEPGRAM:
            max_button = next(button for button in _ai_callback_buttons(detail.reply_markup) if button.callback_data == f"model_setting_max_tokens_{provider}")
            await press(max_button.callback_data, "Max tokens")
            detail = await send_text("1", "Max tokens")
            assert "Max tokens: <b>1</b>" in detail.text
            async with factory() as verify:
                settings = await verify.scalar(
                    select(AIModelSettings).where(
                        AIModelSettings.provider == provider,
                        AIModelSettings.model == resolved[2],
                        AIModelSettings.channel == "chat",
                    )
                )
                assert settings.max_output_tokens == 1
            max_token_passes += 1

        if provider == "Deepseek":
            proxy_button = next(button for button in _ai_callback_buttons(detail.reply_markup) if button.callback_data == "admin_ai_deepseek_proxy")
            proxy_before = None
            async with factory() as verify:
                proxy_before = (await verify.get(AIConfig, 1)).use_proxy
            detail = await press(proxy_button.callback_data, "Proxy:")
            async with factory() as verify:
                assert (await verify.get(AIConfig, 1)).use_proxy is not proxy_before
            await press("admin_ai_keys", "Провайдеры и модели")
            detail = await press("view_models_Deepseek", "Параметры модели")
            proxy_button = next(button for button in _ai_callback_buttons(detail.reply_markup) if button.callback_data == "admin_ai_deepseek_proxy")
            await press(proxy_button.callback_data, "Proxy:")
            async with factory() as verify:
                assert (await verify.get(AIConfig, 1)).use_proxy is proxy_before
            detail = await press("model_setting_reasoning_Deepseek", "Reasoning")
            reasoning_back = next(button.callback_data for button in _ai_callback_buttons(detail.reply_markup) if button.text == "⬅️ Назад")
            detail = await press(reasoning_back, "Параметры модели")
            nested_back_contracts["deepseek_reasoning"][0] += 1
            detail = await press("model_setting_reasoning_Deepseek", "Reasoning")
            reasoning_max = next(button.callback_data for button in _ai_callback_buttons(detail.reply_markup) if button.callback_data == "model_reasoning_Deepseek_max")
            detail = await press(reasoning_max, "Параметры модели")
            assert "Reasoning: <b>Max</b>" in detail.text
            await press("admin_ai_keys", "Провайдеры и модели")
            detail = await press("view_models_Deepseek", "Параметры модели")
            assert "Reasoning: <b>Max</b>" in detail.text

        await press("admin_ai_keys", "Провайдеры и модели")
        keys = await render_latest("Провайдеры и модели")

    assert model_picker_passes == 8
    assert api_key_passes == 8
    assert max_token_passes == len(ALL_PROVIDERS)

    async def return_to_keys():
        nonlocal keys
        keys = await press("admin_ai_keys", "Провайдеры и модели")

    fallback = await press("admin_ai_text_fallback", "Резерв текста")
    fallback_picker = await press("admin_ai_fallback_provider", "провайдера")
    await press("admin_ai_text_fallback", "Резерв текста")
    nested_back_contracts["text_fallback"][0] += 1
    fallback_picker = await press("admin_ai_fallback_provider", "провайдера")
    fallback_provider_buttons = [button for button in _ai_callback_buttons(fallback_picker.reply_markup) if button.callback_data.startswith("admin_ai_fallback_set_provider_")]
    assert fallback_provider_buttons
    for provider_button in fallback_provider_buttons:
        provider = provider_button.callback_data.replace("admin_ai_fallback_set_provider_", "")
        fallback = await press(provider_button.callback_data, "Резерв текста")
        model_picker = await press("admin_ai_fallback_model", "Выберите модель")
        if provider == PROVIDER_PERPLEXITY:
            model_back = next(button.callback_data for button in _ai_callback_buttons(model_picker.reply_markup) if button.text == "⬅️ Назад")
            fallback = await press(model_back, "Резерв текста")
            nested_back_contracts["text_fallback"][0] += 1
            model_picker = await press("admin_ai_fallback_model", "Выберите модель")
            presets_screen = await press("ai_ppx_fb_presets", "Пресеты")
            model_button = next(button for button in _ai_callback_buttons(presets_screen.reply_markup) if button.callback_data.startswith("ai_m_"))
            fallback = await press(model_button.callback_data, "Резерв текста")
        else:
            model_back = next(button.callback_data for button in _ai_callback_buttons(model_picker.reply_markup) if button.text == "⬅️ Назад")
            fallback = await press(model_back, "Резерв текста")
            nested_back_contracts["text_fallback"][0] += 1
            model_picker = await press("admin_ai_fallback_model", "Выберите модель")
            model_button = next(button for button in _ai_callback_buttons(model_picker.reply_markup) if button.callback_data.startswith("ai_m_"))
            fallback = await press(model_button.callback_data, "Резерв текста")
        await return_to_keys()
        fallback = await press("admin_ai_text_fallback", "Резерв текста")
        fallback_picker = await press("admin_ai_fallback_provider", "провайдера")
    fallback = await press(fallback_provider_buttons[-1].callback_data, "Резерв текста")
    last_provider = fallback_provider_buttons[-1].callback_data.replace("admin_ai_fallback_set_provider_", "")
    if last_provider == PROVIDER_PERPLEXITY:
        await press("admin_ai_fallback_model", "Выберите модель")
        await press("ai_ppx_fb_presets", "Пресеты")
        model_button = next(button for button in _ai_callback_buttons(current_markup) if button.callback_data.startswith("ai_m_"))
        fallback = await press(model_button.callback_data, "Резерв текста")
    else:
        await press("admin_ai_fallback_model", "Выберите модель")
        model_button = next(button for button in _ai_callback_buttons(current_markup) if button.callback_data.startswith("ai_m_"))
        fallback = await press(model_button.callback_data, "Резерв текста")
    fallback = await press("admin_ai_fallback_toggle", "Резерв текста")
    fallback = await press("admin_ai_fallback_toggle", "Резерв текста")
    await return_to_keys()
    fallback = await press("admin_ai_text_fallback", "Резерв текста")
    assert "Статус: <b>Выключен</b>" in fallback.text
    assert "Провайдер:" in fallback.text and "Модель:" in fallback.text
    fallback = await press("admin_ai_fallback_toggle", "Резерв текста")
    assert "Статус: <b>Включён</b>" in fallback.text
    completed_flows["text_fallback"] = True
    await return_to_keys()

    vision_fallback = await press("admin_ai_vision_fallback", "Vision резерв")
    vision_picker = await press("admin_ai_vision_fallback_provider", "провайдера")
    await press("admin_ai_vision_fallback", "Vision резерв")
    nested_back_contracts["vision_fallback"][0] += 1
    vision_picker = await press("admin_ai_vision_fallback_provider", "провайдера")
    vision_provider_buttons = [button for button in _ai_callback_buttons(vision_picker.reply_markup) if button.callback_data.startswith("admin_ai_vision_fallback_set_provider_")]
    assert vision_provider_buttons
    for provider_button in vision_provider_buttons:
        vision_fallback = await press(provider_button.callback_data, "Vision резерв")
        model_picker = await press("admin_ai_vision_fallback_model", "модель Vision")
        model_back = next(button.callback_data for button in _ai_callback_buttons(model_picker.reply_markup) if button.text == "⬅️ Назад")
        vision_fallback = await press(model_back, "Vision резерв")
        nested_back_contracts["vision_fallback"][0] += 1
        model_picker = await press("admin_ai_vision_fallback_model", "модель Vision")
        model_button = next(button for button in _ai_callback_buttons(model_picker.reply_markup) if button.callback_data.startswith("ai_m_"))
        vision_fallback = await press(model_button.callback_data, "Vision резерв")
        await return_to_keys()
        vision_fallback = await press("admin_ai_vision_fallback", "Vision резерв")
        vision_picker = await press("admin_ai_vision_fallback_provider", "провайдера")
    vision_fallback = await press(vision_provider_buttons[-1].callback_data, "Vision резерв")
    await press("admin_ai_vision_fallback_toggle", "Vision резерв")
    await press("admin_ai_vision_fallback_toggle", "Vision резерв")
    await return_to_keys()
    vision_fallback = await press("admin_ai_vision_fallback", "Vision резерв")
    assert "Статус: <b>Выключен</b>" in vision_fallback.text
    assert "Провайдер:" in vision_fallback.text and "Модель:" in vision_fallback.text
    vision_fallback = await press("admin_ai_vision_fallback_toggle", "Vision резерв")
    assert "Статус: <b>Включён</b>" in vision_fallback.text
    completed_flows["vision_fallback"] = True
    await return_to_keys()

    capability_journeys = {
        "transcription": ("admin_ai_audio", "admin_select_transcription_provider", "🎙 <b>Аудио</b>", "admin_ai_audio"),
        "vision": ("admin_ai_vision", "admin_select_vision_provider", "🖼 <b>Vision</b>", "admin_ai_vision"),
        "image_gen": ("admin_ai_image_generation", "admin_select_image_generation_provider", "Генерация изображений", "admin_ai_image_generation"),
        "image_edit": ("admin_ai_image_edit", "admin_select_image_edit_provider", "Редактирование изображений", "admin_ai_image_edit"),
    }
    for channel, (section_callback, provider_callback, heading, back_callback) in capability_journeys.items():
        section = await press(section_callback, heading)
        picker = await press(provider_callback, "провайдера")
        provider_buttons = [button for button in _ai_callback_buttons(picker.reply_markup) if button.callback_data.startswith(f"admin_choose_capability_{channel}_")]
        assert provider_buttons
        for provider_button in provider_buttons:
            if provider_button.callback_data.endswith("_None"):
                selected = await press(provider_button.callback_data, heading)
                assert not any(button.callback_data.startswith("ai_m_") for button in _ai_callback_buttons(selected.reply_markup))
                await press("admin_ai_keys", "Провайдеры и модели")
                section = await press(section_callback, heading)
                picker = await press(provider_callback, "провайдера")
                continue
            selected = await press(provider_button.callback_data, "Выберите модель")
            model_buttons = [button for button in _ai_callback_buttons(selected.reply_markup) if button.callback_data.startswith("ai_m_")]
            if model_buttons:
                model_picker_back = next(button.callback_data for button in _ai_callback_buttons(selected.reply_markup) if button.text == "⬅️ Назад")
                await press(model_picker_back, heading)
                nested_back_contracts["capability_pickers"][0] += 1
                picker = await press(provider_callback, "провайдера")
                selected = await press(provider_button.callback_data, "Выберите модель")
                model_button = model_buttons[0]
                selected = await press(model_button.callback_data, heading)
                assert provider_button.callback_data.rsplit("_", 1)[-1] in selected.text
            else:
                assert "выключ" in selected.text.lower()
            await press("admin_ai_keys", "Провайдеры и модели")
            section = await press(section_callback, heading)
            picker = await press(provider_callback, "провайдера")
        await press(back_callback, heading)
        nested_back_contracts["capability_pickers"][0] += 1
        completed_flows[channel] = True
        await return_to_keys()

    await press("admin_ai_settings", "Настройки ИИ", classify=False)
    session.calls.clear()
    update_id += 1
    callback_message = message.model_copy(update={"text": current_text, "reply_markup": current_markup})
    await _feed_callback(dispatcher, bot, callback_message, "admin_panel", update_id)
    await render_latest("Добро пожаловать", classify=False)

    assert model_picker_passes == 8
    assert api_key_passes == 8
    assert max_token_passes == 7
    assert nested_back_contracts == {
        "provider_model_picker": [8, 8],
        "provider_api_key_cancel": [8, 8],
        "deepseek_reasoning": [1, 1],
        "text_fallback": [8, 8],
        "vision_fallback": [7, 7],
        "capability_pickers": [19, 19],
    }
    assert all(completed_flows.values())
    assert len(contracts) > 0
    assert all(category in {"navigation", "mutation", "destructive", "external"} for category in contracts.values())


@pytest.mark.asyncio
async def test_telegram_legacy_ai_callbacks_redirect_without_cycling(factory, admin_dispatcher, monkeypatch):
    import admin_content_authoring
    import automation_admin
    import handlers

    for module in (handlers, admin_content_authoring, automation_admin):
        monkeypatch.setattr(module, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(automation_admin, "get_all_admin_ids", AsyncMock(return_value={11}))

    async with factory() as session:
        session.add_all([
            DBUser(id=11, is_admin=True, first_name="Admin"),
            AIConfig(
                id=1,
                provider="Deepseek",
                transcription_provider="OpenAI",
                vision_provider="Gemini",
                image_generation_provider="OpenAI",
                image_edit_provider="KIE",
                allow_fallback=False,
            ),
            SubscriptionConfig(id=1),
        ])
        await session.commit()

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    dispatcher = admin_dispatcher
    message = _admin_message(bot, user_id=11, text="/admin")
    await dispatcher.feed_update(bot, Update(update_id=3000, message=message))
    root = next(method for method in reversed(session.calls) if isinstance(method, SendMessage))
    await _feed_callback(dispatcher, bot, message.model_copy(update={"reply_markup": root.reply_markup}), "admin_ai_settings", 3001)
    ai = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    await _feed_callback(dispatcher, bot, message.model_copy(update={"reply_markup": ai.reply_markup, "text": ai.text}), "admin_ai_keys", 3002)
    keys = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))

    stale = {
        "admin_toggle_fallback": ("Резерв текста", "admin_ai_keys", "fallback_provider", "fallback_model"),
        "admin_toggle_transcription": ("Выберите провайдера", "admin_ai_audio", "transcription_provider", None),
        "admin_toggle_vision": ("Выберите провайдера", "admin_ai_vision", "vision_provider", "vision_model"),
        "admin_toggle_image_generation": ("Выберите провайдера", "admin_ai_image_generation", "image_generation_provider", "image_generation_model"),
        "admin_toggle_image_edit": ("Выберите провайдера", "admin_ai_image_edit", "image_edit_provider", "image_edit_model"),
    }
    for index, (callback_data, (heading, parent, *fields)) in enumerate(stale.items(), start=1):
        async with factory() as verify:
            before = {field: getattr(await verify.get(AIConfig, 1), field) for field in fields if field}
        session.calls.clear()
        await _feed_callback(
            dispatcher,
            bot,
            message.model_copy(update={"reply_markup": keys.reply_markup, "text": keys.text}),
            callback_data,
            3010 + index,
        )
        rendered = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
        assert heading in rendered.text
        assert callback_data not in {button.callback_data for button in _ai_callback_buttons(rendered.reply_markup)}
        async with factory() as verify:
            after = {field: getattr(await verify.get(AIConfig, 1), field) for field in fields if field}
        assert before == after
        back = next(
            button.callback_data
            for button in _ai_callback_buttons(rendered.reply_markup)
            if button.text in {"⬅️ Назад", "◀️ Назад"}
        )
        await _feed_callback(
            dispatcher,
            bot,
            message.model_copy(update={"reply_markup": rendered.reply_markup, "text": rendered.text}),
            back,
            3020 + index,
        )
        parent_screen = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
        if parent == "admin_ai_keys":
            assert "Провайдеры и модели" in parent_screen.text
        else:
            assert parent_screen.reply_markup
        keys = parent_screen if parent == "admin_ai_keys" else keys
        if parent != "admin_ai_keys":
            session.calls.clear()
            await _feed_callback(
                dispatcher,
                bot,
                message.model_copy(update={"reply_markup": parent_screen.reply_markup, "text": parent_screen.text}),
                "admin_ai_keys",
                3030 + index,
            )
            keys = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))


@pytest.mark.asyncio
async def test_admin_dispatcher_content_journey_validates_rendering_and_runtime(factory, admin_dispatcher, monkeypatch):
    import admin_content_authoring
    import automation_admin
    import content_menu
    import handlers
    import keyboards
    from content_authoring import read_content_value, save_content_value
    from translation_service import refresh_translation_cache

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(admin_content_authoring, "async_session_maker", factory)
    monkeypatch.setattr(automation_admin, "async_session_maker", factory)
    monkeypatch.setattr(keyboards, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(automation_admin, "get_all_admin_ids", AsyncMock(return_value={11}))
    monkeypatch.setattr(handlers, "send_temp_notification", AsyncMock())

    async with factory() as session:
        config = await session.get(BotGeneralConfig, 1)
        config.telegram_language_selection_enabled = True
        content = Content(key="about_me", button_title="Об авторе", text_content="<b>Русский</b>")
        session.add_all([
            DBUser(id=11, is_admin=True, first_name="Admin"),
            DBUser(id=42, telegram_language_code="pt", first_name="User"),
            content,
        ])
        await session.flush()
        await save_content_value(session, "content", content, "button_title", "en", "About")
        await save_content_value(session, "content", content, "button_title", "pt", "Sobre")
        await save_content_value(session, "content", content, "text_content", "en", "<b>English</b>")
        await save_content_value(session, "content", content, "text_content", "pt", "<b>Português</b>")
        await session.commit()

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    dispatcher = admin_dispatcher
    root_message = _admin_message(bot, text="/admin")
    await dispatcher.feed_update(bot, Update(update_id=500, message=root_message))

    await _feed_callback(dispatcher, bot, root_message, "admin_content", 501)
    content_list = next(
        method for method in reversed(session.calls) if isinstance(method, EditMessageText)
    )
    assert "Контент" in content_list.text
    content_buttons = [
        button
        for row in content_list.reply_markup.inline_keyboard
        for button in row
        if button.callback_data and button.callback_data.startswith("ca:view:content:")
    ]
    assert any(button.text == "Об авторе" for button in content_buttons)
    assert not any("about_me" in button.text for button in content_buttons)
    content_view = next(
        button.callback_data
        for row in content_list.reply_markup.inline_keyboard
        for button in row
        if button.callback_data and button.callback_data.startswith("ca:view:content:about_me")
    )
    await _feed_callback(dispatcher, bot, root_message, content_view, 502)
    card = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    assert "Контент #about_me" in card.text
    assert "ID: <code>about_me</code>" in card.text
    assert "Изменить: контент" in " ".join(
        button.text
        for row in card.reply_markup.inline_keyboard
        for button in row
    )
    await _feed_callback(dispatcher, bot, root_message, "ca:locale:content:about_me:pt:0", 503)
    pt_card = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    assert "Português" in pt_card.text
    await _feed_callback(dispatcher, bot, root_message, "ca:edit_content:content:about_me:pt:0", 504)
    await dispatcher.feed_update(
        bot,
        Update(
            update_id=505,
            message=_admin_message(bot, text="Новый PT").model_copy(
                update={"entities": [MessageEntity(type="bold", offset=0, length=8)]}
            ),
        ),
    )
    await _feed_callback(dispatcher, bot, root_message, "save_content_about_me", 506)
    saved_card = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    assert "Новый PT" in saved_card.text
    async with factory() as verify:
        content = await verify.get(Content, "about_me")
        assert content.text_content == "<b>Русский</b>"
        assert (await read_content_value(verify, "content", content, "text_content", "pt")).text == "<b>Новый PT</b>"

    await refresh_translation_cache(factory, force=True)
    user_keyboard = await keyboards.main_client_keyboard(42)
    labels = [button.text for row in user_keyboard.keyboard for button in row]
    assert "Sobre" in labels
    assert await content_menu.resolve_menu(42, "Sobre", "content", session_maker=factory) == "about_me"

    await _feed_callback(dispatcher, bot, root_message, "ca:list:content:0", 507)
    listed_again = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    assert "Контент" in listed_again.text
    await _feed_callback(dispatcher, bot, root_message, "admin_panel", 508)
    returned = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    assert returned.text == "Добро пожаловать в админ-панель!"


@pytest.mark.asyncio
async def test_subscription_reset_then_real_start_regrants_welcome_bonus(factory, admin_dispatcher, monkeypatch):
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "OWNER_IDS", [11])
    monkeypatch.setattr(handlers, "refresh_commands_for_user", AsyncMock())
    monkeypatch.setattr(handlers, "_sync_user_birthdate_from_telegram", AsyncMock(return_value=False))
    monkeypatch.setattr(handlers, "render_static_content_telegram", AsyncMock(return_value=True))

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    async with factory() as session:
        config = await session.get(BotGeneralConfig, 1)
        config.profile_collect_name = False
        config.profile_collect_gender = False
        config.profile_collect_age = False
        session.add_all([
            DBUser(id=11, is_admin=True, first_name="Admin"),
            DBUser(id=42, first_name="User", accepted_disclaimer=True),
            SubscriptionConfig(id=1, welcome_bonus_days=3),
            UserSubscription(
                user_id=42,
                start_date=now - timedelta(days=1),
                end_date=now + timedelta(days=1),
                payment_provider="Trial Welcome",
            ),
            TrialUsageHistory(user_id=42, plan_id=None, used_at=now - timedelta(days=1)),
            SubscriptionBenefitGrant(
                grant_key="welcome:42",
                grant_type="welcome",
                beneficiary_user_id=42,
                days=3,
                created_at=now - timedelta(days=1),
            ),
            TelegramStartIntent(user_id=42, new_user_eligible=False, status="completed"),
        ])
        await session.commit()

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    dispatcher = admin_dispatcher
    admin_message = _admin_message(bot, user_id=11)
    await _feed_callback(dispatcher, bot, admin_message, "admin_reset_sub_42", 599)
    confirmation = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    assert "Сбросить подписку клиента" in confirmation.text
    await _feed_callback(dispatcher, bot, admin_message, "admin_reset_sub_confirm_42", 600)
    async with factory() as verify:
        assert await verify.scalar(select(UserSubscription.id).where(UserSubscription.user_id == 42)) is None
        assert await verify.scalar(select(TrialUsageHistory.id).where(TrialUsageHistory.user_id == 42)) is None
        assert await verify.scalar(
            select(SubscriptionBenefitGrant.id).where(SubscriptionBenefitGrant.grant_key == "welcome:42")
        ) is None
        intent = await verify.get(TelegramStartIntent, 42)
        assert intent.new_user_eligible is True
        assert intent.status == "awaiting_language"

    user_message = _admin_message(bot, user_id=42, text="/start")
    await dispatcher.feed_update(bot, Update(update_id=601, message=user_message))
    bonus = [
        method.text
        for method in session.calls
        if isinstance(method, SendMessage) and method.text and "бонус" in method.text.lower()
    ]
    assert bonus and "3" in bonus[-1]
    async with factory() as verify:
        subscription = await verify.scalar(
            select(UserSubscription).where(UserSubscription.user_id == 42)
        )
        assert subscription is not None
        assert subscription.end_date > datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=2)


@pytest.mark.asyncio
async def test_clients_nested_navigation_returns_to_same_page_and_search(factory, admin_dispatcher, monkeypatch):
    import admin_content_authoring
    import automation_admin
    import handlers
    import keyboards

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(admin_content_authoring, "async_session_maker", factory)
    monkeypatch.setattr(automation_admin, "async_session_maker", factory)
    monkeypatch.setattr(keyboards, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "OWNER_IDS", [11])
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(automation_admin, "get_all_admin_ids", AsyncMock(return_value={11}))

    async with factory() as session:
        session.add(DBUser(id=11, is_admin=True, first_name="Admin"))
        session.add_all(
            [DBUser(id=100 + index, first_name=f"target-{index}") for index in range(12)]
        )
        await session.commit()

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    dispatcher = admin_dispatcher
    root_message = _admin_message(bot, user_id=11, text="/admin")
    await dispatcher.feed_update(bot, Update(update_id=700, message=root_message))
    state = dispatcher.fsm.get_context(bot=bot, chat_id=11, user_id=11)
    await state.update_data(client_search_query="target")

    await _feed_callback(dispatcher, bot, root_message, "admin_clients_page_1", 701)
    clients_page = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    assert "2/2" in clients_page.text
    client_callback = next(
        button.callback_data
        for row in clients_page.reply_markup.inline_keyboard
        for button in row
        if button.callback_data and button.callback_data.startswith("view_client_")
    )
    assert client_callback.endswith("_page_1")
    await _feed_callback(dispatcher, bot, root_message, client_callback, 702)
    profile = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    profile_callbacks = {
        button.callback_data
        for row in profile.reply_markup.inline_keyboard
        for button in row
        if button.callback_data
    }
    assert "admin_clients_page_1" in profile_callbacks
    user_id = int(client_callback.split("_")[2])
    await _feed_callback(dispatcher, bot, root_message, f"client_payment_info_{user_id}", 703)
    payment = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    nested_back = next(
        button.callback_data
        for row in payment.reply_markup.inline_keyboard
        for button in row
        if button.callback_data and button.callback_data.startswith("view_client_")
    )
    await _feed_callback(dispatcher, bot, root_message, nested_back, 704)
    profile_again = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    assert "Профиль клиента" in profile_again.text
    await _feed_callback(dispatcher, bot, root_message, "admin_clients_page_1", 705)
    returned = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    assert "2/2" in returned.text
    assert any(
        "target-" in button.text
        for row in returned.reply_markup.inline_keyboard
        for button in row
    )


@pytest.mark.asyncio
async def test_every_client_profile_action_preserves_page_context(factory, admin_dispatcher, monkeypatch):
    import admin_content_authoring
    import automation_admin
    import handlers
    import keyboards

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(admin_content_authoring, "async_session_maker", factory)
    monkeypatch.setattr(automation_admin, "async_session_maker", factory)
    monkeypatch.setattr(keyboards, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "OWNER_IDS", [11])
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(automation_admin, "get_all_admin_ids", AsyncMock(return_value={11}))

    async with factory() as session:
        session.add_all([
            DBUser(id=11, is_admin=True, first_name="Owner"),
            DBUser(id=321, first_name="Nested target", can_view_history=True),
            *[
                DBUser(id=322 + index, first_name=f"nested-{index}")
                for index in range(15)
            ],
            AIConfig(id=1),
            SubscriptionConfig(id=1),
        ])
        await session.commit()

    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    dispatcher = admin_dispatcher
    root_message = _admin_message(bot, user_id=11, text="/admin")
    await dispatcher.feed_update(bot, Update(update_id=750, message=root_message))
    state = dispatcher.fsm.get_context(bot=bot, chat_id=11, user_id=11)
    await state.update_data(client_search_query="nested", client_list_page=1)

    await _feed_callback(dispatcher, bot, root_message, "view_client_321_page_1", 751)
    profile = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    profile_callbacks = [
        button.callback_data
        for row in profile.reply_markup.inline_keyboard
        for button in row
        if button.callback_data and button.callback_data != "admin_clients_page_1"
    ]
    assert profile_callbacks

    for offset, callback_data in enumerate(profile_callbacks, start=1):
        session.calls.clear()
        context = dispatcher.fsm.get_context(bot=bot, chat_id=11, user_id=11)
        await context.update_data(client_search_query="nested", client_list_page=1, viewing_client_id=321)
        await _feed_callback(dispatcher, bot, root_message, callback_data, 751 + offset)
        methods = [
            method
            for method in session.calls
            if isinstance(method, (EditMessageText, SendMessage))
        ]
        assert methods, callback_data
        rendered = methods[-1]
        assert isinstance(rendered.text, str), callback_data
        back_callbacks = [
            button.callback_data
            for row in (rendered.reply_markup.inline_keyboard if rendered.reply_markup else [])
            for button in row
            if button.callback_data and button.callback_data.startswith("view_client_")
        ]
        if back_callbacks:
            await _feed_callback(dispatcher, bot, root_message, back_callbacks[-1], 800 + offset)
            profile_again = next(
                method for method in reversed(session.calls) if isinstance(method, EditMessageText)
            )
            assert "Профиль клиента" in profile_again.text
            list_callback = next(
                button.callback_data
                for row in profile_again.reply_markup.inline_keyboard
                for button in row
                if button.callback_data == "admin_clients_page_1"
            )
        else:
            list_callback = next(
                button.callback_data
                for row in (rendered.reply_markup.inline_keyboard if rendered.reply_markup else [])
                for button in row
                if button.callback_data == "admin_clients_page_1"
            )
        await _feed_callback(dispatcher, bot, root_message, list_callback, 900 + offset)
        returned = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
        assert "2/2" in returned.text, callback_data
        assert any(
            "nested" in button.text
            for row in returned.reply_markup.inline_keyboard
            for button in row
        )


def test_language_overview_explains_default_fallback_and_single_language_warning():
    from handlers import _language_overview_text

    config = SimpleNamespace(
        telegram_default_language="ru",
        telegram_enabled_languages='["ru"]',
        telegram_language_selection_enabled=True,
        multilingual_authoring_enabled=True,
    )
    readiness = {"locales": {"en": {"canonical": False}, "pt": {"canonical": False}}}
    text = _language_overview_text(config, readiness)
    assert "Язык по умолчанию: <b>🇷🇺 Русский</b>" in text
    assert "Выбор языка: <b>Включён</b>" in text
    assert "Мультиязычность: <b>Включена</b>" in text
    assert "сохранённый доступный язык" not in text
    assert "доступен только один язык" in text


@pytest.mark.asyncio
async def test_localized_menu_label_resolves_same_content_key(factory, monkeypatch):
    import keyboards
    import content_menu
    from content_authoring import save_content_value
    from translation_service import refresh_translation_cache

    monkeypatch.setattr(keyboards, "async_session_maker", factory)
    monkeypatch.setattr(content_menu, "async_session_maker", factory)
    async with factory() as session:
        content = Content(key="about_me", button_title="Об авторе", text_content="RU")
        session.add_all([content, DBUser(id=22, telegram_language_code="pt")])
        config = await session.get(BotGeneralConfig, 1)
        config.telegram_language_selection_enabled = True
        await session.flush()
        await save_content_value(session, "content", content, "button_title", "pt", "Sobre")
        await session.commit()
    await refresh_translation_cache(factory, force=True)

    markup = await keyboards.main_client_keyboard(22)
    labels = [button.text for row in markup.keyboard for button in row]
    assert "Sobre" in labels
    assert await content_menu.resolve_menu(22, "Sobre", "content", session_maker=factory) == "about_me"


@pytest.mark.asyncio
async def test_visibility_toggle_returns_to_same_localized_card(factory, monkeypatch):
    import admin_content_authoring as module

    monkeypatch.setattr(module, "async_session_maker", factory)
    monkeypatch.setattr(module, "allowed", AsyncMock(return_value=True))
    card = AsyncMock()
    monkeypatch.setattr(module, "resource_card", card)
    async with factory() as session:
        session.add(Content(key="about_me", button_title="Об авторе", is_visible=True))
        await session.commit()

    callback = SimpleNamespace(
        data="ca:visibility:content:about_me:pt:2",
        answer=AsyncMock(),
        message=SimpleNamespace(),
    )
    state = SimpleNamespace(clear=AsyncMock())
    await module.content_callback(callback, state)

    async with factory() as session:
        assert (await session.get(Content, "about_me")).is_visible is False
    card.assert_awaited_once_with(callback, "content", "about_me", "pt", 2)


@pytest.mark.asyncio
async def test_language_selection_warning_allows_single_configured_language(factory, monkeypatch):
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    async with factory() as session:
        config = await session.get(BotGeneralConfig, 1)
        config.telegram_enabled_languages = '["ru"]'
        config.telegram_language_selection_enabled = False
        await session.commit()
    callback = SimpleNamespace(from_user=SimpleNamespace(id=1), answer=AsyncMock())
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    notice = await handlers._toggle_language_selection_setting(callback)
    assert "только один язык" in notice
    async with factory() as session:
        assert (await session.get(BotGeneralConfig, 1)).telegram_language_selection_enabled is True


@pytest.mark.asyncio
async def test_content_save_returns_to_same_card_and_page(factory, monkeypatch):
    import handlers
    import admin_content_authoring as module

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(module, "async_session_maker", factory)
    card = AsyncMock()
    monkeypatch.setattr(module, "resource_card", card)
    async with factory() as session:
        session.add(Content(key="about_me", button_title="Об авторе", text_content="RU"))
        await session.commit()

    class State:
        async def get_data(self):
            return {
                "content_key": "about_me",
                "text_content": "PT",
                "media_files": [],
                "content_order": "media_top",
                "authoring_locale": "pt",
                "parent_kind": "content",
                "parent_page": 2,
                "media_variant_present": False,
                "media_variant_touched": False,
            }

        async def clear(self):
            return None

    callback = SimpleNamespace(message=SimpleNamespace(edit_text=AsyncMock()), answer=AsyncMock())
    await handlers.save_content(callback, State())
    card.assert_awaited_once_with(callback, "content", "about_me", "pt", 2)


@pytest.mark.asyncio
async def test_content_cancel_returns_to_same_card_without_mutation(factory, monkeypatch):
    import handlers
    import admin_content_authoring as module

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(module, "async_session_maker", factory)
    card = AsyncMock()
    monkeypatch.setattr(module, "resource_card", card)

    class State:
        async def get_data(self):
            return {"content_key": "about_me", "authoring_locale": "pt", "parent_kind": "content", "parent_page": 3}

        async def clear(self):
            return None

    callback = SimpleNamespace(
        data="cancel_content_edit_about_me",
        message=SimpleNamespace(edit_text=AsyncMock()),
        answer=AsyncMock(),
    )
    await handlers.cancel_content_edit_handler(callback, State())
    card.assert_awaited_once_with(callback, "content", "about_me", "pt", 3)


@pytest.mark.asyncio
async def test_delayed_notification_after_save_does_not_report_lost_session(factory, monkeypatch):
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    async with factory() as session:
        session.add(Content(key="about_me", button_title="Об авторе", text_content="RU"))
        await session.commit()

    class State:
        def __init__(self):
            self.data = {"content_key": "about_me", "text_content": "RU", "media_files": [], "message_id_to_edit": 5}

        async def get_data(self):
            return dict(self.data)

        async def update_data(self, **values):
            self.data.update(values)

        async def clear(self):
            self.data.clear()

    state = State()

    async def delayed_cleanup(*args, **kwargs):
        await state.clear()

    monkeypatch.setattr(handlers, "send_temp_notification", delayed_cleanup)
    message = SimpleNamespace(
        text="Новый текст",
        html_text="Новый текст",
        from_user=SimpleNamespace(id=11),
        delete=AsyncMock(),
        answer=AsyncMock(),
    )
    bot = SimpleNamespace()
    await handlers.process_content_update(message, state, bot)
    assert not any("сессии" in str(call.args[0]) for call in message.answer.await_args_list)


@pytest.mark.asyncio
async def test_duplicate_save_after_editor_close_is_safe(monkeypatch):
    import handlers

    callback = SimpleNamespace(answer=AsyncMock())

    class ClosedState:
        async def get_data(self):
            return {}

    await handlers.save_content(callback, ClosedState())
    assert "закрыт" in callback.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_content_admin_journey_round_trips_locale_storage_and_runtime(factory, monkeypatch):
    import admin_content_authoring as admin_module
    import content_menu
    import handlers
    import keyboards
    from admin_authoring_context import content_editing_locale
    from content_authoring import save_content_value
    from handlers import render_static_content_telegram
    from translation_service import refresh_translation_cache

    for module in (admin_module, handlers, keyboards, content_menu):
        monkeypatch.setattr(module, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "send_temp_notification", AsyncMock())

    async with factory() as session:
        config = await session.get(BotGeneralConfig, 1)
        config.telegram_language_selection_enabled = True
        content = Content(key="about_me", button_title="Об авторе", text_content="RU <b>текст</b>")
        session.add_all([content, DBUser(id=22, telegram_language_code="pt")])
        await session.flush()
        await save_content_value(session, "content", content, "button_title", "pt", "Sobre")
        await save_content_value(session, "content", content, "text_content", "pt", "PT <b>старый</b>")
        await session.commit()
    await refresh_translation_cache(factory, force=True)

    bot = SimpleNamespace(
        get_me=AsyncMock(return_value=SimpleNamespace(username="demo_bot")),
        edit_message_text=AsyncMock(),
        send_message=AsyncMock(),
        send_photo=AsyncMock(),
        send_video=AsyncMock(),
    )
    admin_message = SimpleNamespace(
        message_id=500,
        chat=SimpleNamespace(id=11),
        edit_text=AsyncMock(),
        answer=AsyncMock(),
    )
    admin_callback = SimpleNamespace(
        bot=bot,
        message=admin_message,
        from_user=SimpleNamespace(id=11),
        answer=AsyncMock(),
        data="ca:view:content:about_me:0",
    )

    await admin_module.resource_list(admin_callback, "content", 0)
    list_markup = admin_message.edit_text.await_args.kwargs["reply_markup"]
    assert any(
        button.callback_data == "ca:view:content:about_me:0"
        for row in list_markup.inline_keyboard
        for button in row
    )

    await admin_module.resource_card(admin_callback, "content", "about_me", "pt", 0)
    card_markup = admin_message.edit_text.await_args.kwargs["reply_markup"]
    edit_button = next(
        button for row in card_markup.inline_keyboard for button in row
        if button.text == "✏️ Изменить: контент"
    )
    assert edit_button.callback_data == "ca:edit_content:content:about_me:pt:0"

    state = _JourneyState({"authoring_locale": "pt", "parent_kind": "content", "parent_page": 0})
    edit_callback = SimpleNamespace(
        data="edit_content_about_me",
        bot=bot,
        message=admin_message,
        from_user=SimpleNamespace(id=11),
        answer=AsyncMock(),
    )
    token = content_editing_locale.set("pt")
    try:
        await handlers.start_content_edit(edit_callback, state)
    finally:
        content_editing_locale.reset(token)
    assert state.data["authoring_locale"] == "pt"
    assert "старый" in state.data["text_content"]

    incoming = SimpleNamespace(
        text="PT <b>новый</b>",
        html_text="PT <b>новый</b>",
        from_user=SimpleNamespace(id=11),
        delete=AsyncMock(),
        answer=AsyncMock(),
    )
    await handlers.process_content_update(incoming, state, bot)
    save_callback = SimpleNamespace(
        data="save_content_about_me",
        bot=bot,
        message=admin_message,
        from_user=SimpleNamespace(id=11),
        answer=AsyncMock(),
    )
    await handlers.save_content(save_callback, state)

    async with factory() as session:
        assert (await session.get(Content, "about_me")).text_content == "RU <b>текст</b>"
        pt_value = await session.scalar(
            select(BotTranslation.text).where(
                BotTranslation.locale == "pt",
                BotTranslation.translation_key == "content.about_me.text_content",
            )
        )
        assert pt_value == "PT <b>новый</b>"

    await refresh_translation_cache(factory, force=True)
    user_keyboard = await keyboards.main_client_keyboard(22)
    assert "Sobre" in [button.text for row in user_keyboard.keyboard for button in row]
    assert await content_menu.resolve_menu(22, "Sobre", "content", session_maker=factory) == "about_me"
    assert await render_static_content_telegram(bot, 22, 22, "about_me")
    assert any("PT <b>новый</b>" in call.args[1] for call in bot.send_message.await_args_list if len(call.args) > 1)

    await admin_module.resource_card(admin_callback, "content", "about_me", "pt", 0)
    await admin_module.content_callback(
        SimpleNamespace(
            data="ca:list:content:0",
            message=admin_message,
            bot=bot,
            from_user=SimpleNamespace(id=11),
            answer=AsyncMock(),
        ),
        _JourneyState(),
    )
    assert "Контент" in admin_message.edit_text.await_args.args[0]


@pytest.mark.asyncio
async def test_content_deep_link_routes_to_same_localized_resource(factory, monkeypatch):
    import handlers
    import keyboards
    from content_authoring import save_content_value
    from translation_service import refresh_translation_cache

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "refresh_commands_for_user", AsyncMock())
    monkeypatch.setattr(handlers, "_sync_user_birthdate_from_telegram", AsyncMock())
    monkeypatch.setattr(keyboards, "main_client_keyboard", AsyncMock(return_value=None))

    async with factory() as session:
        config = await session.get(BotGeneralConfig, 1)
        config.multilingual_authoring_enabled = True
        config.telegram_language_selection_enabled = True
        config.profile_collect_name = False
        config.profile_collect_gender = False
        config.profile_collect_age = False
        content = Content(key="about_me", button_title="Об авторе", text_content="RU")
        user = DBUser(id=44, telegram_language_code="pt", first_name="User")
        session.add_all([content, user])
        await session.flush()
        await save_content_value(session, "content", content, "button_title", "pt", "Sobre")
        await save_content_value(session, "content", content, "text_content", "pt", "Texto português")
        await session.commit()
    await refresh_translation_cache(factory, force=True)

    bot = SimpleNamespace(
        send_message=AsyncMock(),
        send_photo=AsyncMock(),
        send_video=AsyncMock(),
        send_media_group=AsyncMock(),
    )
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=44, username="user", full_name="User"),
        chat=SimpleNamespace(id=44),
        answer=AsyncMock(),
    )
    await handlers._run_start_business(
        message,
        _JourneyState(),
        bot,
        args="about_me",
    )

    assert any(
        "Texto português" in call.args[1]
        for call in bot.send_message.await_args_list
        if len(call.args) > 1
    )


class _JourneyState:
    def __init__(self, data=None):
        self.data = dict(data or {})
        self.state = None

    async def get_data(self):
        return dict(self.data)

    async def update_data(self, **values):
        self.data.update(values)

    async def set_state(self, state):
        self.state = state

    async def set_data(self, data):
        self.data = dict(data)

    async def clear(self):
        self.data.clear()
        self.state = None


@pytest.mark.asyncio
async def test_system_message_locale_repeat_finishes_callback_on_every_visible_page(
    factory,
    admin_dispatcher,
    monkeypatch,
):
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))

    async with factory() as db_session:
        db_session.add(DBUser(id=61, is_admin=True, first_name="Admin"))
        await db_session.commit()

    session = ValidatingTelegramSession(stateful=True)
    bot = Bot("123456:TEST", session=session)
    root_message = _admin_message(bot, user_id=61, text="/admin")
    await admin_dispatcher.feed_update(bot, Update(update_id=6100, message=root_message))
    await _feed_callback(admin_dispatcher, bot, root_message, "admin_general_settings", 6101)
    await _feed_callback(admin_dispatcher, bot, root_message, "admin_system_messages", 6102)

    def buttons(message):
        return [
            button
            for row in (message.reply_markup.inline_keyboard if message.reply_markup else [])
            for button in row
            if button.callback_data
        ]

    def locale_button(message, locale):
        return next(
            button
            for button in buttons(message)
            if parse_system_message_callback(button.callback_data)
            and parse_system_message_callback(button.callback_data)["action"] == "l"
            and parse_system_message_callback(button.callback_data)["locale"] == locale
        )

    total_pages = system_message_page_count("telegram")
    assert total_pages >= 3
    for page in range(total_pages):
        for locale in ("ru", "en", "pt"):
            current = session.current_message(root_message)
            selected = locale_button(current, locale)
            for repeat in range(2):
                current = session.current_message(root_message)
                before = current.model_dump(mode="json", exclude_none=True)
                session.calls.clear()
                await _feed_callback(
                    admin_dispatcher,
                    bot,
                    current,
                    selected.callback_data,
                    6103 + page * 20 + len(locale) * 2 + repeat,
                )
                assert any(isinstance(method, AnswerCallbackQuery) for method in session.calls)
                if repeat == 1:
                    assert not any(isinstance(method, EditMessageText) for method in session.calls)
                    after = session.current_message(root_message).model_dump(
                        mode="json", exclude_none=True
                    )
                    before.pop("date", None)
                    after.pop("date", None)
                    assert after == before
                selected = locale_button(session.current_message(root_message), locale)
        if page == total_pages - 1:
            continue
        current = session.current_message(root_message)
        next_button = next(
            button
            for button in buttons(current)
            if parse_system_message_callback(button.callback_data)
            and parse_system_message_callback(button.callback_data)["action"] == "p"
            and int(parse_system_message_callback(button.callback_data)["page"]) == page + 1
        )
        await _feed_callback(
            admin_dispatcher,
            bot,
            current,
            next_button.callback_data,
            6200 + page,
        )


@pytest.mark.asyncio
async def test_waiting_message_has_one_telegram_admin_path_and_uses_effective_text(
    factory,
    admin_dispatcher,
    monkeypatch,
):
    import handlers
    import httpx

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers.ai_integration, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))

    async with factory() as db_session:
        config = await db_session.get(BotGeneralConfig, 1)
        config.ai_processing_message_enabled = True
        config.ai_processing_message_text = DEFAULT_AI_PROCESSING_MESSAGE_TEXT
        config.telegram_language_selection_enabled = True
        db_session.add_all([
            DBUser(id=62, is_admin=True, first_name="Admin", telegram_language_code="ru", current_dialogue_id=1),
            DBUser(id=63, first_name="Português", telegram_language_code="pt", current_dialogue_id=1),
            DBUser(id=64, first_name="Русский", telegram_language_code="ru", current_dialogue_id=1),
            AIConfig(
                id=1,
                provider="OpenRouter",
                openrouter_api_key="sk-test-key",
                openrouter_model="qwen/qwen3.7-plus",
                system_prompt="Система",
                memory_mode="global",
            ),
            SubscriptionConfig(id=1, subscriptions_enabled=False),
            BotAddressingConfig(bot_scope="telegram", addressing_style="formal"),
            BotSystemMessage(
                bot_scope="telegram",
                translation_key="ui.ai.processing_default",
                locale="ru",
                addressing_style="formal",
                text="Подождите, пожалуйста...",
            ),
            BotSystemMessage(
                bot_scope="telegram",
                translation_key="ui.ai.processing_default",
                locale="pt",
                addressing_style="formal",
                text="Aguarde, por favor...",
            ),
            BotSystemMessage(
                bot_scope="telegram",
                translation_key="ui.ai.processing_default",
                locale="pt",
                addressing_style="informal",
                text="Aguarde um pouco...",
            ),
            BotSystemMessage(
                bot_scope="telegram",
                translation_key="ui.ai.processing_default",
                locale="ru",
                addressing_style="informal",
                text="Думаю по-дружески...",
            ),
        ])
        await db_session.commit()

    async def fake_provider_post(self, url, *args, **kwargs):
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "Ответ"}}]},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_provider_post)
    session = ValidatingTelegramSession(stateful=True)
    bot = Bot("123456:TEST", session=session)
    root_message = _admin_message(bot, user_id=62, text="/admin")
    await admin_dispatcher.feed_update(bot, Update(update_id=6300, message=root_message))
    await _feed_callback(admin_dispatcher, bot, root_message, "admin_general_settings", 6301)
    general = session.current_message(root_message)
    await _feed_callback(admin_dispatcher, bot, general, "admin_system_messages", 6302)
    system_list = session.current_message(root_message)
    processing_button = next(
        button
        for row in system_list.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "d"
        and parse_system_message_callback(button.callback_data)["key"] == "ui.ai.processing_default"
    )
    await _feed_callback(admin_dispatcher, bot, system_list, processing_button.callback_data, 6303)
    detail = session.current_message(root_message)
    back_to_list = next(
        button
        for row in detail.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "p"
    )
    await _feed_callback(admin_dispatcher, bot, detail, back_to_list.callback_data, 6304)
    system_list = session.current_message(root_message)
    pt_locale = next(
        button
        for row in system_list.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "l"
        and parse_system_message_callback(button.callback_data)["locale"] == "pt"
    )
    await _feed_callback(admin_dispatcher, bot, system_list, pt_locale.callback_data, 6305)
    system_list = session.current_message(root_message)
    processing_button = next(
        button
        for row in system_list.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "d"
        and parse_system_message_callback(button.callback_data)["key"] == "ui.ai.processing_default"
    )
    await _feed_callback(admin_dispatcher, bot, system_list, processing_button.callback_data, 6306)
    detail = session.current_message(root_message)
    editor_button = next(
        button
        for row in detail.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "e"
        and parse_system_message_callback(button.callback_data)["style"] == "formal"
    )
    await _feed_callback(admin_dispatcher, bot, detail, editor_button.callback_data, 6307)
    editor = session.current_message(root_message)
    assert "Aguarde, por favor..." in editor.text

    await admin_dispatcher.feed_update(
        bot,
        Update(
            update_id=6308,
            message=_admin_message(bot, user_id=62, message_id=2, text="Meu texto de espera"),
        ),
    )
    draft = session.latest_message_with_keyboard(bot, 62, User(id=62, is_bot=False, first_name="Admin"))
    save_button = next(
        button
        for row in draft.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "s"
    )
    await _feed_callback(admin_dispatcher, bot, draft, save_button.callback_data, 6309)
    detail = session.current_message(draft)
    assert "Meu texto de espera" in detail.text
    back_to_list = next(
        button
        for row in detail.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "p"
    )
    await _feed_callback(admin_dispatcher, bot, detail, back_to_list.callback_data, 6310)
    system_list = session.current_message(detail)
    general_button = next(
        button
        for row in system_list.reply_markup.inline_keyboard
        for button in row
        if button.callback_data == "admin_general_settings"
    )
    await _feed_callback(admin_dispatcher, bot, system_list, general_button.callback_data, 6311)
    general = session.current_message(system_list)
    assert "Действующий текст: Подождите, пожалуйста..." in general.text
    async with factory() as db_session:
        config = await db_session.get(BotGeneralConfig, 1)
        assert config.ai_processing_message_text == DEFAULT_AI_PROCESSING_MESSAGE_TEXT
        assert config.ai_processing_message_text_explicit is False
        assert await db_session.scalar(
            select(BotSystemMessage.text).where(
                BotSystemMessage.translation_key == "ui.ai.processing_default",
                BotSystemMessage.locale == "pt",
                BotSystemMessage.addressing_style == "formal",
            )
        ) == "Meu texto de espera"

    await _feed_callback(admin_dispatcher, bot, general, "admin_panel", 6312)
    panel = session.current_message(general)
    await _feed_callback(
        admin_dispatcher,
        bot,
        panel,
        "admin_general_settings",
        6313,
    )
    general = session.current_message(panel)
    assert "Действующий текст: Подождите, пожалуйста..." in general.text

    handlers.user_message_buffers[63] = ["Olá"]
    session.calls.clear()
    await handlers.process_buffered_messages(63, bot)
    waiting = next(
        method
        for method in session.calls
        if isinstance(method, SendMessage) and method.chat_id == 63 and method.text == "Meu texto de espera"
    )
    waiting_index = session.calls.index(waiting)
    assert any(
        isinstance(method, DeleteMessage)
        and method.chat_id == 63
        and method.message_id == 1
        and session.calls.index(method) > waiting_index
        for method in session.calls
    )

    await _feed_callback(
        admin_dispatcher,
        bot,
        general,
        "admin_general_toggle_addressing_style",
        6314,
    )
    await _feed_callback(
        admin_dispatcher,
        bot,
        general,
        "admin_system_messages",
        6315,
    )
    system_list = session.current_message(general)
    processing_button = next(
        button
        for row in system_list.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "d"
        and parse_system_message_callback(button.callback_data)["key"] == "ui.ai.processing_default"
    )
    await _feed_callback(admin_dispatcher, bot, system_list, processing_button.callback_data, 6316)
    detail = session.current_message(system_list)
    edit_informal = next(
        button
        for row in detail.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "e"
        and parse_system_message_callback(button.callback_data)["style"] == "informal"
    )
    await _feed_callback(admin_dispatcher, bot, detail, edit_informal.callback_data, 6317)
    await admin_dispatcher.feed_update(
        bot,
        Update(
            update_id=6318,
            message=_admin_message(bot, user_id=62, message_id=3, text=DEFAULT_AI_PROCESSING_MESSAGE_TEXT),
        ),
    )
    draft = session.latest_message_with_keyboard(bot, 62, User(id=62, is_bot=False, first_name="Admin"))
    save_button = next(
        button
        for row in draft.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "s"
    )
    await _feed_callback(admin_dispatcher, bot, draft, save_button.callback_data, 6319)
    async with factory() as db_session:
        config = await db_session.get(BotGeneralConfig, 1)
        assert config.ai_processing_message_text == DEFAULT_AI_PROCESSING_MESSAGE_TEXT
        assert config.ai_processing_message_text_explicit is False

    handlers.user_message_buffers[63] = ["Olá de novo"]
    session.calls.clear()
    await handlers.process_buffered_messages(63, bot)
    assert any(
        isinstance(method, SendMessage) and method.chat_id == 63 and method.text == "Aguarde um pouco..."
        for method in session.calls
    )
    handlers.user_message_buffers[64] = ["Привет"]
    session.calls.clear()
    await handlers.process_buffered_messages(64, bot)
    assert any(
        isinstance(method, SendMessage) and method.chat_id == 64 and method.text == DEFAULT_AI_PROCESSING_MESSAGE_TEXT
        for method in session.calls
    )


@pytest.mark.asyncio
async def test_waiting_message_has_a_canonical_telegram_catalog_setting(
    factory,
    admin_dispatcher,
    monkeypatch,
):
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))

    async with factory() as db_session:
        db_session.add(DBUser(id=65, is_admin=True, first_name="Admin"))
        await db_session.commit()

    session = ValidatingTelegramSession(stateful=True)
    bot = Bot("123456:TEST", session=session)
    root_message = _admin_message(bot, user_id=65, text="/admin")
    await admin_dispatcher.feed_update(bot, Update(update_id=6400, message=root_message))
    await _feed_callback(admin_dispatcher, bot, root_message, "admin_general_settings", 6401)
    await _feed_callback(admin_dispatcher, bot, root_message, "admin_system_messages", 6402)

    found_processing_message = False
    for page in range(system_message_page_count("telegram")):
        current = session.current_message(root_message)
        for row in current.reply_markup.inline_keyboard:
            for button in row:
                parsed = parse_system_message_callback(button.callback_data or "")
                if parsed and parsed["action"] == "d":
                    if parsed["key"] == "ui.ai.processing_default":
                        found_processing_message = True
                        assert button.text == "Сообщение во время ожидания ответа"
        if page == system_message_page_count("telegram") - 1:
            break
        next_button = next(
            button
            for row in current.reply_markup.inline_keyboard
            for button in row
            if parse_system_message_callback(button.callback_data or "")
            and parse_system_message_callback(button.callback_data)["action"] == "p"
            and int(parse_system_message_callback(button.callback_data)["page"]) == page + 1
        )
        await _feed_callback(admin_dispatcher, bot, current, next_button.callback_data, 6403 + page)
    assert found_processing_message


@pytest.mark.asyncio
async def test_waiting_message_general_screen_is_bounded_and_escaped(
    factory,
    admin_dispatcher,
    monkeypatch,
):
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))

    async with factory() as db_session:
        config = await db_session.get(BotGeneralConfig, 1)
        config.ai_processing_message_text = "<b>" + ("A" * 4089) + "</b>"
        config.ai_processing_message_text_explicit = True
        db_session.add(DBUser(id=66, is_admin=True, first_name="Admin"))
        await db_session.commit()

    session = ValidatingTelegramSession(stateful=True)
    bot = Bot("123456:TEST", session=session)
    root_message = _admin_message(bot, user_id=66, text="/admin")
    await admin_dispatcher.feed_update(bot, Update(update_id=6500, message=root_message))
    session.calls.clear()
    await _feed_callback(admin_dispatcher, bot, root_message, "admin_general_settings", 6501)
    rendered = [method for method in session.calls if isinstance(method, EditMessageText)]
    assert rendered
    assert len(rendered[-1].text) <= 4096
    assert "&lt;b&gt;" in rendered[-1].text


@pytest.mark.asyncio
async def test_waiting_message_canonical_editor_preserves_existing_variants(
    factory,
    admin_dispatcher,
    monkeypatch,
):
    import handlers
    import httpx

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers.ai_integration, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))

    async with factory() as db_session:
        config = await db_session.get(BotGeneralConfig, 1)
        config.ai_processing_message_enabled = True
        config.ai_processing_message_text = DEFAULT_AI_PROCESSING_MESSAGE_TEXT
        config.telegram_language_selection_enabled = True
        db_session.add_all([
            DBUser(id=67, is_admin=True, first_name="Admin", telegram_language_code="ru", current_dialogue_id=1),
            DBUser(id=68, first_name="Русский", telegram_language_code="ru", current_dialogue_id=1),
            DBUser(id=69, first_name="Português", telegram_language_code="pt", current_dialogue_id=1),
            AIConfig(
                id=1,
                provider="OpenRouter",
                openrouter_api_key="sk-test-key",
                openrouter_model="qwen/qwen3.7-plus",
                system_prompt="Система",
                memory_mode="global",
            ),
            SubscriptionConfig(id=1, subscriptions_enabled=False),
            BotAddressingConfig(bot_scope="telegram", addressing_style="formal"),
            BotSystemMessage(
                bot_scope="telegram",
                translation_key="ui.ai.processing_default",
                locale="ru",
                addressing_style="formal",
                text="Старый русский вариант",
            ),
            BotSystemMessage(
                bot_scope="telegram",
                translation_key="ui.ai.processing_default",
                locale="pt",
                addressing_style="formal",
                text="Variante portuguesa antiga",
            ),
            BotTranslation(
                locale="pt",
                translation_key="bot_general_config.1.ai_processing_message_text",
                text="Старый PT перевод общего текста",
                source_hash=source_hash(DEFAULT_AI_PROCESSING_MESSAGE_TEXT),
            ),
        ])
        await db_session.commit()

    from translation_service import refresh_translation_cache

    await refresh_translation_cache(factory, force=True)

    async def fake_provider_post(self, url, *args, **kwargs):
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "Ответ"}}]},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_provider_post)
    session = ValidatingTelegramSession(stateful=True)
    bot = Bot("123456:TEST", session=session)
    root_message = _admin_message(bot, user_id=67, text="/admin")
    await admin_dispatcher.feed_update(bot, Update(update_id=6700, message=root_message))
    panel = session.current_message(root_message)
    general_button = next(
        button
        for row in panel.reply_markup.inline_keyboard
        for button in row
        if button.callback_data == "admin_general_settings"
    )
    await _feed_callback(admin_dispatcher, bot, panel, general_button.callback_data, 6701)
    general = session.current_message(root_message)
    await _feed_callback(admin_dispatcher, bot, general, "admin_system_messages", 6702)
    system_list = session.current_message(root_message)
    processing_button = next(
        button
        for row in system_list.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "d"
        and parse_system_message_callback(button.callback_data)["key"] == "ui.ai.processing_default"
    )
    await _feed_callback(
        admin_dispatcher,
        bot,
        system_list,
        processing_button.callback_data,
        6703,
    )
    detail = session.current_message(root_message)
    editor_button = next(
        button
        for row in detail.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "e"
        and parse_system_message_callback(button.callback_data)["style"] == "formal"
    )
    await _feed_callback(
        admin_dispatcher,
        bot,
        detail,
        editor_button.callback_data,
        6704,
    )
    editor = session.current_message(root_message)
    assert "Старый русский вариант" in editor.text
    await admin_dispatcher.feed_update(
        bot,
        Update(
            update_id=6705,
            message=_admin_message(bot, user_id=67, message_id=2, text="Новый русский вариант"),
        ),
    )
    draft = session.latest_message_with_keyboard(bot, 67, User(id=67, is_bot=False, first_name="Admin"))
    save_button = next(
        button
        for row in draft.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "s"
    )
    await _feed_callback(admin_dispatcher, bot, draft, save_button.callback_data, 6706)
    detail = session.current_message(draft)
    back_to_list = next(
        button
        for row in detail.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "p"
    )
    await _feed_callback(admin_dispatcher, bot, detail, back_to_list.callback_data, 6707)
    system_list = session.current_message(detail)
    general_button = next(
        button
        for row in system_list.reply_markup.inline_keyboard
        for button in row
        if button.callback_data == "admin_general_settings"
    )
    await _feed_callback(admin_dispatcher, bot, system_list, general_button.callback_data, 6708)

    general = session.current_message(system_list)
    assert "Действующий текст: Новый русский вариант" in general.text

    async with factory() as db_session:
        config = await db_session.get(BotGeneralConfig, 1)
        assert config.ai_processing_message_text == DEFAULT_AI_PROCESSING_MESSAGE_TEXT
        assert config.ai_processing_message_text_explicit is False
        assert await db_session.scalar(
            select(BotSystemMessage.text).where(
                BotSystemMessage.bot_scope == "telegram",
                BotSystemMessage.translation_key == "ui.ai.processing_default",
                BotSystemMessage.locale == "ru",
                BotSystemMessage.addressing_style == "formal",
            )
        ) == "Новый русский вариант"
        assert await db_session.scalar(
            select(BotSystemMessage.text).where(
                BotSystemMessage.bot_scope == "telegram",
                BotSystemMessage.translation_key == "ui.ai.processing_default",
                BotSystemMessage.locale == "pt",
                BotSystemMessage.addressing_style == "formal",
            )
        ) == "Variante portuguesa antiga"

    handlers.user_message_buffers[68] = ["Привет"]
    session.calls.clear()
    await handlers.process_buffered_messages(68, bot)
    waiting = next(
        method
        for method in session.calls
        if isinstance(method, SendMessage) and method.chat_id == 68
    )
    assert waiting.text == "Новый русский вариант"
    assert any(
        isinstance(method, DeleteMessage)
        and method.chat_id == 68
        and method.message_id == 1
        and session.calls.index(method) > session.calls.index(waiting)
        for method in session.calls
    )

    handlers.user_message_buffers[69] = ["Olá"]
    session.calls.clear()
    await handlers.process_buffered_messages(69, bot)
    waiting_pt = next(
        method
        for method in session.calls
        if isinstance(method, SendMessage) and method.chat_id == 69
    )
    assert waiting_pt.text == "Variante portuguesa antiga"


@pytest.mark.asyncio
async def test_waiting_message_variants_use_canonical_ui_and_preserve_other_locales(
    factory,
    admin_dispatcher,
    monkeypatch,
):
    import handlers
    import httpx

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers.ai_integration, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))

    async with factory() as db_session:
        config = await db_session.get(BotGeneralConfig, 1)
        config.ai_processing_message_enabled = True
        config.ai_processing_message_text = "Единый текст"
        config.ai_processing_message_text_explicit = True
        config.telegram_language_selection_enabled = True
        db_session.add_all([
            DBUser(id=70, is_admin=True, first_name="Admin", telegram_language_code="ru", current_dialogue_id=1),
            DBUser(id=71, first_name="Português", telegram_language_code="pt", current_dialogue_id=1),
            DBUser(id=72, first_name="Русский", telegram_language_code="ru", current_dialogue_id=1),
            AIConfig(
                id=1,
                provider="OpenRouter",
                openrouter_api_key="sk-test-key",
                openrouter_model="qwen/qwen3.7-plus",
                system_prompt="Система",
                memory_mode="global",
            ),
            SubscriptionConfig(id=1, subscriptions_enabled=False),
            BotAddressingConfig(bot_scope="telegram", addressing_style="formal"),
            BotSystemMessage(
                bot_scope="telegram",
                translation_key="ui.ai.processing_default",
                locale="ru",
                addressing_style="formal",
                text="Старый русский Вы",
            ),
            BotSystemMessage(
                bot_scope="telegram",
                translation_key="ui.ai.processing_default",
                locale="ru",
                addressing_style="informal",
                text="Старый русский Ты",
            ),
            BotSystemMessage(
                bot_scope="telegram",
                translation_key="ui.ai.processing_default",
                locale="pt",
                addressing_style="formal",
                text="Старый португальский Вы",
            ),
            BotSystemMessage(
                bot_scope="telegram",
                translation_key="ui.ai.processing_default",
                locale="pt",
                addressing_style="informal",
                text="Старый португальский Ты",
            ),
        ])
        await db_session.commit()

    async def fake_provider_post(self, url, *args, **kwargs):
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "Ответ"}}]},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_provider_post)
    session = ValidatingTelegramSession(stateful=True)
    bot = Bot("123456:TEST", session=session)
    admin_user = User(id=70, is_bot=False, first_name="Admin")
    root_message = _admin_message(bot, user_id=70, text="/admin")
    await admin_dispatcher.feed_update(bot, Update(update_id=7000, message=root_message))
    panel = session.current_message(root_message)
    general_button = next(
        button
        for row in panel.reply_markup.inline_keyboard
        for button in row
        if button.callback_data == "admin_general_settings"
    )
    await _feed_callback(admin_dispatcher, bot, panel, general_button.callback_data, 7001)
    general = session.current_message(root_message)
    assert "единый для всех языков" in general.text
    assert not any(
        button.callback_data in {
            "admin_general_edit_ai_processing_message_text",
            "admin_general_open_ai_processing_variants",
        }
        for row in general.reply_markup.inline_keyboard
        for button in row
        if button.callback_data
    )
    system_button = next(
        button
        for row in general.reply_markup.inline_keyboard
        for button in row
        if button.callback_data == "admin_system_messages"
    )
    await _feed_callback(
        admin_dispatcher,
        bot,
        general,
        system_button.callback_data,
        7002,
    )
    system_list = session.current_message(root_message)
    processing_button = next(
        button
        for row in system_list.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "d"
        and parse_system_message_callback(button.callback_data)["key"] == "ui.ai.processing_default"
    )
    assert processing_button.text == "Сообщение во время ожидания ответа"
    await _feed_callback(admin_dispatcher, bot, system_list, processing_button.callback_data, 7003)
    detail = session.current_message(root_message)
    back_to_list = next(
        button
        for row in detail.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "p"
    )
    await _feed_callback(admin_dispatcher, bot, detail, back_to_list.callback_data, 7004)
    system_list = session.current_message(root_message)
    pt_locale = next(
        button
        for row in system_list.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "l"
        and parse_system_message_callback(button.callback_data)["locale"] == "pt"
    )
    await _feed_callback(admin_dispatcher, bot, system_list, pt_locale.callback_data, 7005)
    system_list = session.current_message(root_message)
    processing_button = next(
        button
        for row in system_list.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "d"
        and parse_system_message_callback(button.callback_data)["key"] == "ui.ai.processing_default"
    )
    await _feed_callback(admin_dispatcher, bot, system_list, processing_button.callback_data, 7006)
    detail = session.current_message(root_message)
    edit_formal = next(
        button
        for row in detail.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "e"
        and parse_system_message_callback(button.callback_data)["style"] == "formal"
    )
    await _feed_callback(admin_dispatcher, bot, detail, edit_formal.callback_data, 7007)
    editor = session.current_message(root_message)
    assert "Единый текст" in editor.text
    assert "Сейчас используется единый текст" in editor.text
    assert {button.text for button in _ai_callback_buttons(editor.reply_markup)} >= {"💾 Сохранить", "Отмена"}
    cancel_button = next(
        button
        for button in _ai_callback_buttons(editor.reply_markup)
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "c"
    )
    await _feed_callback(admin_dispatcher, bot, editor, cancel_button.callback_data, 7008)
    detail = session.current_message(root_message)
    async with factory() as db_session:
        config = await db_session.get(BotGeneralConfig, 1)
        assert config.ai_processing_message_text_explicit is True
        assert await db_session.scalar(
            select(BotSystemMessage.text).where(
                BotSystemMessage.translation_key == "ui.ai.processing_default",
                BotSystemMessage.locale == "pt",
                BotSystemMessage.addressing_style == "formal",
            )
        ) == "Старый португальский Вы"
    edit_formal = next(
        button
        for button in _ai_callback_buttons(detail.reply_markup)
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "e"
        and parse_system_message_callback(button.callback_data)["style"] == "formal"
    )
    await _feed_callback(admin_dispatcher, bot, detail, edit_formal.callback_data, 7009)
    assert "Единый текст" in session.current_message(root_message).text
    await admin_dispatcher.feed_update(
        bot,
        Update(
            update_id=7010,
            message=_admin_message(bot, user_id=70, message_id=2, text="Новый PT Вы"),
        ),
    )
    draft = session.latest_message_with_keyboard(bot, 70, admin_user)
    save_button = next(
        button
        for row in draft.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "s"
    )
    await _feed_callback(admin_dispatcher, bot, draft, save_button.callback_data, 7011)
    detail = session.current_message(draft)

    async with factory() as db_session:
        config = await db_session.get(BotGeneralConfig, 1)
        assert config.ai_processing_message_text == "Единый текст"
        assert config.ai_processing_message_text_explicit is False
        assert await db_session.scalar(
            select(BotSystemMessage.text).where(
                BotSystemMessage.translation_key == "ui.ai.processing_default",
                BotSystemMessage.locale == "pt",
                BotSystemMessage.addressing_style == "formal",
            )
        ) == "Новый PT Вы"
        assert await db_session.scalar(
            select(BotSystemMessage.text).where(
                BotSystemMessage.translation_key == "ui.ai.processing_default",
                BotSystemMessage.locale == "ru",
                BotSystemMessage.addressing_style == "formal",
            )
        ) == "Старый русский Вы"

    edit_informal = next(
        button
        for row in detail.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "e"
        and parse_system_message_callback(button.callback_data)["style"] == "informal"
    )
    await _feed_callback(admin_dispatcher, bot, detail, edit_informal.callback_data, 7012)
    await admin_dispatcher.feed_update(
        bot,
        Update(
            update_id=7013,
            message=_admin_message(bot, user_id=70, message_id=3, text="Не сохранять"),
        ),
    )
    draft = session.latest_message_with_keyboard(bot, 70, admin_user)
    cancel_button = next(
        button
        for row in draft.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "c"
    )
    await _feed_callback(admin_dispatcher, bot, draft, cancel_button.callback_data, 7014)
    detail = session.current_message(draft)

    async with factory() as db_session:
        assert await db_session.scalar(
            select(BotSystemMessage.text).where(
                BotSystemMessage.translation_key == "ui.ai.processing_default",
                BotSystemMessage.locale == "pt",
                BotSystemMessage.addressing_style == "informal",
            )
        ) == "Старый португальский Ты"

    session.calls.clear()
    handlers.user_message_buffers[71] = ["Olá"]
    await handlers.process_buffered_messages(71, bot)
    waiting = next(
        method
        for method in session.calls
        if isinstance(method, SendMessage) and method.chat_id == 71
    )
    assert waiting.text == "Новый PT Вы"
    assert any(
        isinstance(method, DeleteMessage)
        and method.chat_id == 71
        and session.calls.index(method) > session.calls.index(waiting)
        for method in session.calls
    )

    back_to_list = next(
        button
        for row in detail.reply_markup.inline_keyboard
        for button in row
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "p"
    )
    await _feed_callback(admin_dispatcher, bot, detail, back_to_list.callback_data, 7015)
    system_list = session.current_message(draft)
    general_button = next(
        button
        for row in system_list.reply_markup.inline_keyboard
        for button in row
        if button.callback_data == "admin_general_settings"
    )
    await _feed_callback(admin_dispatcher, bot, system_list, general_button.callback_data, 7016)
    general = session.current_message(draft)
    toggle_button = next(
        button
        for row in general.reply_markup.inline_keyboard
        for button in row
        if button.callback_data == "admin_general_toggle_addressing_style"
    )
    await _feed_callback(admin_dispatcher, bot, general, toggle_button.callback_data, 7017)
    general = session.current_message(draft)
    assert "по языку и обращению" in general.text

    session.calls.clear()
    handlers.user_message_buffers[71] = ["Olá de novo"]
    await handlers.process_buffered_messages(71, bot)
    assert any(
        isinstance(method, SendMessage)
        and method.chat_id == 71
        and method.text == "Старый португальский Ты"
        for method in session.calls
    )
    session.calls.clear()
    handlers.user_message_buffers[72] = ["Привет"]
    await handlers.process_buffered_messages(72, bot)
    assert any(
        isinstance(method, SendMessage)
        and method.chat_id == 72
        and method.text == "Старый русский Ты"
        for method in session.calls
    )


@pytest.mark.asyncio
async def test_waiting_message_system_variant_with_literal_html_chars_uses_plain_send(
    factory,
    admin_dispatcher,
    monkeypatch,
):
    import handlers
    import httpx

    monkeypatch.setattr(handlers, "async_session_maker", factory)
    monkeypatch.setattr(handlers.ai_integration, "async_session_maker", factory)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))

    async with factory() as db_session:
        config = await db_session.get(BotGeneralConfig, 1)
        config.ai_processing_message_enabled = True
        config.ai_processing_message_text = DEFAULT_AI_PROCESSING_MESSAGE_TEXT
        config.ai_processing_message_text_explicit = False
        config.telegram_language_selection_enabled = True
        db_session.add_all([
            DBUser(id=73, is_admin=True, first_name="Admin", telegram_language_code="ru", current_dialogue_id=1),
            DBUser(id=74, first_name="Português", telegram_language_code="pt", current_dialogue_id=1),
            AIConfig(
                id=1,
                provider="OpenRouter",
                openrouter_api_key="sk-test-key",
                openrouter_model="qwen/qwen3.7-plus",
                system_prompt="Система",
                memory_mode="global",
            ),
            SubscriptionConfig(id=1, subscriptions_enabled=False),
            BotAddressingConfig(bot_scope="telegram", addressing_style="informal"),
        ])
        await db_session.commit()

    async def fake_provider_post(self, url, *args, **kwargs):
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "Ответ"}}]},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_provider_post)
    session = ValidatingTelegramSession(stateful=True)
    bot = Bot("123456:TEST", session=session)
    root_message = _admin_message(bot, user_id=73, text="/admin")
    await admin_dispatcher.feed_update(bot, Update(update_id=7300, message=root_message))

    def visible(message):
        return [
            button
            for row in (message.reply_markup.inline_keyboard if message.reply_markup else [])
            for button in row
            if button.callback_data
        ]

    async def press(message, predicate, update_id):
        current = session.current_message(message)
        button = next(button for button in visible(current) if predicate(button))
        session.calls.clear()
        await _feed_callback(admin_dispatcher, bot, current, button.callback_data, update_id)
        assert any(isinstance(method, AnswerCallbackQuery) for method in session.calls)
        return session.current_message(current)

    panel = session.current_message(root_message)
    general = await press(panel, lambda button: button.callback_data == "admin_general_settings", 7301)
    formal = await press(general, lambda button: button.callback_data == "admin_general_toggle_addressing_style", 7302)
    variants = await press(formal, lambda button: button.callback_data == "admin_system_messages", 7303)
    processing = next(
        button
        for button in visible(variants)
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "d"
        and parse_system_message_callback(button.callback_data)["key"] == "ui.ai.processing_default"
    )
    detail = await press(variants, lambda button: button.callback_data == processing.callback_data, 7304)
    pt_locale = await press(
        detail,
        lambda button: (
            parse_system_message_callback(button.callback_data or "")
            and parse_system_message_callback(button.callback_data)["action"] == "p"
        ),
        7305,
    )
    pt_locale = await press(
        pt_locale,
        lambda button: (
            parse_system_message_callback(button.callback_data or "")
            and parse_system_message_callback(button.callback_data)["action"] == "l"
            and parse_system_message_callback(button.callback_data)["locale"] == "pt"
        ),
        7306,
    )
    processing = next(
        button
        for button in visible(pt_locale)
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "d"
        and parse_system_message_callback(button.callback_data)["key"] == "ui.ai.processing_default"
    )
    detail = await press(pt_locale, lambda button: button.callback_data == processing.callback_data, 7307)
    edit = next(
        button
        for button in visible(detail)
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "e"
        and parse_system_message_callback(button.callback_data)["style"] == "formal"
    )
    editor = await press(detail, lambda button: button.callback_data == edit.callback_data, 7308)
    assert "Язык: <b>Português</b>" in editor.text
    await admin_dispatcher.feed_update(
        bot,
        Update(
            update_id=7309,
            message=_admin_message(
                bot,
                user_id=73,
                message_id=2,
                text="Подождите <пример> & продолжайте",
            ),
        ),
    )
    draft = session.latest_message_with_keyboard(bot, 73, editor.from_user)
    save = next(
        button
        for button in visible(draft)
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "s"
    )
    detail = await press(draft, lambda button: button.callback_data == save.callback_data, 7310)

    async with factory() as db_session:
        saved_text = await db_session.scalar(
            select(BotSystemMessage.text).where(
                BotSystemMessage.bot_scope == "telegram",
                BotSystemMessage.translation_key == "ui.ai.processing_default",
                BotSystemMessage.locale == "pt",
                BotSystemMessage.addressing_style == "formal",
            )
        )
    assert saved_text == "Подождите <пример> & продолжайте"

    edit = next(
        button
        for button in visible(detail)
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "e"
        and parse_system_message_callback(button.callback_data)["style"] == "formal"
    )
    reopened_editor = await press(detail, lambda button: button.callback_data == edit.callback_data, 7311)
    assert "Подождите &lt;пример&gt; &amp; продолжайте" in reopened_editor.text
    cancel = next(
        button
        for button in visible(reopened_editor)
        if parse_system_message_callback(button.callback_data or "")
        and parse_system_message_callback(button.callback_data)["action"] == "c"
    )
    await press(reopened_editor, lambda button: button.callback_data == cancel.callback_data, 7312)

    handlers.user_message_buffers[74] = ["Olá"]
    session.calls.clear()
    await handlers.process_buffered_messages(74, bot)
    waiting = next(
        method
        for method in session.calls
        if isinstance(method, SendMessage)
        and method.chat_id == 74
        and method.text == "Подождите <пример> & продолжайте"
    )
    assert waiting.parse_mode is None
    assert not waiting.entities
    waiting_index = session.calls.index(waiting)
    assert any(
        isinstance(method, DeleteMessage)
        and method.chat_id == 74
        and session.calls.index(method) > waiting_index
        for method in session.calls
    )
