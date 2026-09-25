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
from aiogram.methods import EditMessageText, GetMe, SendMessage
from aiogram.types import CallbackQuery, Chat, Message, MessageEntity, Update, User
from aiogram.client.session.base import BaseSession
from aiogram import Bot
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database import (
    AIConfig,
    Base,
    BotGeneralConfig,
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
                from_user=User(id=999, is_bot=True, first_name="TestBot", username="testbot"),
                text=getattr(method, "text", ""),
            ).as_(bot)
        if isinstance(method, GetMe):
            return User(id=999, is_bot=True, first_name="TestBot", username="testbot").as_(bot)
        return True

    async def stream_content(self, *args, **kwargs):
        if False:
            yield b""


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
        "admin_general_edit_ai_processing_message_text",
        "admin_main_collections_page_0",
        "admin_language_settings",
        "admin_panel",
    } <= general_callbacks

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
    await screen("model_setting_reasoning_Deepseek", "Reasoning DeepSeek", 416)
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
            detail = await press("model_setting_reasoning_Deepseek", "Reasoning DeepSeek")
            reasoning_back = next(button.callback_data for button in _ai_callback_buttons(detail.reply_markup) if button.text == "⬅️ Назад")
            detail = await press(reasoning_back, "Параметры модели")
            nested_back_contracts["deepseek_reasoning"][0] += 1
            detail = await press("model_setting_reasoning_Deepseek", "Reasoning DeepSeek")
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
        fallback = await press(provider_button.callback_data, "Резерв текста")
        model_picker = await press("admin_ai_fallback_model", "Выберите модель")
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
