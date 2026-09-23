import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

os.environ.setdefault("BOT_TOKEN", "123456:test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from aiogram.types import Chat, Message, User
from aiogram.client.session.base import BaseSession
from aiogram import Bot
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database import Base, BotGeneralConfig, BotTranslation, Content, User as DBUser, UserMenuBinding
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


def _callback(bot, message, data):
    return SimpleNamespace(
        bot=bot,
        message=message,
        from_user=message.from_user,
        data=data,
        answer=AsyncMock(),
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
        date=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
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
        date=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
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
        date=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
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
        date=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
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
    assert "сохранённый доступный язык" in text
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
