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
    assert "Об авторе · about_me" in labels
    assert "Записаться на сессию · btn_4d49f0df8b" in labels
    assert "menu: Меню" not in labels

    await module.resource_card(
        _callback(bot, message, "ca:view:content:menu:ru:0"),
        "content",
        "menu",
        "ru",
        0,
    )
    assert "ID: <code>menu</code>" in recording.calls[-1].text


@pytest.mark.asyncio
async def test_perplexity_model_screen_omits_unrepresented_rub_pricing(factory, monkeypatch):
    import handlers

    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    session = ValidatingTelegramSession()
    bot = Bot("123456:TEST", session=session)
    message = _admin_message(bot)
    await handlers.view_models_by_provider(_callback(bot, message, "view_models_Perplexity"))
    rendered = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    assert "Выберите режим Perplexity:" in rendered.text
    assert "Прайсинг" not in rendered.text
    assert "руб" not in rendered.text.lower()
    assert "Быстрый поиск" in rendered.text
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
    assert "toggle_deepseek_thinking" in ai_callbacks
    assert "set_max_output_tokens" in ai_callbacks
    await screen("toggle_deepseek_thinking", "Thinking", 414)
    await screen("admin_ai_settings", "Настройки ИИ", 415)
    await screen("admin_ai_keys", "Ключи, модели", 416)
    await screen("admin_ai_settings", "Настройки ИИ", 417)
    await screen("admin_panel", "Добро пожаловать в админ-панель", 418)


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
    content_view = next(
        button.callback_data
        for row in content_list.reply_markup.inline_keyboard
        for button in row
        if button.callback_data and button.callback_data.startswith("ca:view:content:about_me")
    )
    await _feed_callback(dispatcher, bot, root_message, content_view, 502)
    card = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    assert "Контент #about_me" in card.text
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
