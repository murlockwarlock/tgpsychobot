from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import database
import handlers
import keyboards as tg_kb
from database import Base, Content, Topic, User
import max_messenger_bot.legacy as max_legacy
import max_messenger_bot.storage as max_storage
from max_messenger_bot import app as max_app
from max_messenger_bot.services import settings as max_settings
from max_messenger_bot.app import MaxBotApplication
from max_messenger_bot.keyboards import inline_keyboard, main_menu_row
from max_messenger_bot.models import IncomingCallback, IncomingMessage, Sender
from max_messenger_bot.services import common as max_common
from max_messenger_bot.services import topics as max_topics
from max_messenger_bot.services import subscriptions as max_subscriptions
from max_messenger_bot.services import admin_content as max_admin_content
from max_messenger_bot.services import admin as max_admin
from max_messenger_bot.keyboards import admin_panel_keyboard as max_admin_panel_keyboard
from max_messenger_bot.storage import MaxContentMedia, StorageBase
from response_buttons import ResponseButton, extract_response_buttons


@pytest_asyncio.fixture
async def db_session(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test-menu-buttons.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(StorageBase.metadata.create_all)

    monkeypatch.setattr(database, "async_session_maker", sessions)
    monkeypatch.setattr(handlers, "async_session_maker", sessions)
    monkeypatch.setattr(tg_kb, "async_session_maker", sessions)
    monkeypatch.setattr(max_legacy, "async_session_maker", sessions)
    monkeypatch.setattr(max_storage, "async_session_maker", sessions)
    monkeypatch.setattr(max_common, "async_session_maker", sessions)
    monkeypatch.setattr(max_topics, "async_session_maker", sessions)
    monkeypatch.setattr(max_settings, "async_session_maker", sessions)
    monkeypatch.setattr(max_app, "async_session_maker", sessions)
    monkeypatch.setattr(max_subscriptions, "async_session_maker", sessions)
    monkeypatch.setattr(max_admin, "async_session_maker", sessions)
    monkeypatch.setattr(max_admin_content, "async_session_maker", sessions)

    handlers.user_message_buffers.clear()
    handlers.user_processing_tasks.clear()
    handlers._ai_button_claims.clear()

    try:
        yield sessions
    finally:
        handlers.user_message_buffers.clear()
        handlers.user_processing_tasks.clear()
        handlers._ai_button_claims.clear()
        await engine.dispose()


# ==============================================================================
# MAX BOT TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_max_admin_absent_from_global_commands():
    from max_messenger_bot.app import build_max_global_commands

    all_enabled = build_max_global_commands(
        topics_enabled=True,
        subscriptions_enabled=True,
        referral_enabled=True,
    )
    command_names = [c["name"] for c in all_enabled]
    assert "admin" not in command_names
    assert "start" in command_names
    assert "help" in command_names
    assert "topics" in command_names
    assert "new_dialogue" in command_names
    assert "settings" in command_names
    assert "subscription" in command_names
    assert "ref" in command_names

    disabled = build_max_global_commands(
        topics_enabled=False,
        subscriptions_enabled=False,
        referral_enabled=False,
    )
    disabled_names = [c["name"] for c in disabled]
    assert "admin" not in disabled_names
    assert "topics" not in disabled_names
    assert "subscription" not in disabled_names
    assert "ref" not in disabled_names
    assert "start" in disabled_names
    assert "help" in disabled_names
    assert "new_dialogue" in disabled_names
    assert "settings" in disabled_names


@pytest.mark.asyncio
async def test_max_typed_admin_command_admin_vs_non_admin(db_session):
    async with db_session() as session:
        session.add(User(id=1001, first_name="Admin", is_admin=True, accepted_disclaimer=True))
        session.add(User(id=1002, first_name="User", is_admin=False, accepted_disclaimer=True))
        await session.commit()

    mock_client = SimpleNamespace(
        send_message=AsyncMock(),
    )
    app = MaxBotApplication(client=mock_client)

    # Non-admin typing /admin -> silently ignored
    msg_user = IncomingMessage(
        raw={},
        message_id="m1",
        chat_id=1002,
        sender=Sender(user_id=1002, username="user", first_name="User", last_name=None),
        text="/admin",
    )
    with patch("max_messenger_bot.ai.get_ai_response", new_callable=AsyncMock) as mock_ai:
        await app.handle_message(msg_user)
        mock_client.send_message.assert_not_called()
        mock_ai.assert_not_called()

    # Admin typing /admin -> admin panel displayed
    msg_admin = IncomingMessage(
        raw={},
        message_id="m2",
        chat_id=1001,
        sender=Sender(user_id=1001, username="admin", first_name="Admin", last_name=None),
        text="/admin",
    )
    await app.handle_message(msg_admin)
    mock_client.send_message.assert_called_once()
    assert "админ-панель" in mock_client.send_message.call_args.kwargs["text"].lower()


@pytest.mark.asyncio
async def test_max_start_welcome_with_parsed_buttons_no_dynamic_menu(db_session):
    async with db_session() as session:
        session.add(User(id=123, first_name="Test User", is_admin=False))
        session.add(
            Content(
                key="start_message",
                text_content="Добро пожаловать!\n[Выбрать тему](btn:svc:topics) [Подписка](btn:svc:subscription)",
                is_visible=True,
            )
        )
        await session.commit()

    mock_client = SimpleNamespace(send_message=AsyncMock())
    await max_common.show_start_screen(mock_client, chat_id=123, user_id=123)

    assert mock_client.send_message.call_count == 1
    call_kwargs = mock_client.send_message.call_args.kwargs
    assert "Добро пожаловать!" in call_kwargs["text"]
    assert "[Выбрать тему]" not in call_kwargs["text"]
    attachments = call_kwargs["attachments"]
    assert attachments is not None
    # Verify inline keyboard has the 2 parsed buttons
    buttons = attachments[0]["payload"]["buttons"]
    assert buttons[0][0]["text"] == "Выбрать тему"
    assert buttons[0][0]["payload"] == "ai_btn:svc:topics"
    assert buttons[1][0]["text"] == "Подписка"
    assert buttons[1][0]["payload"] == "ai_btn:svc:subscription"
    # start_message does NOT auto-append "⬅️ В меню"
    button_texts = [b["text"] for row in buttons for b in row]
    assert "⬅️ В меню" not in button_texts


@pytest.mark.asyncio
async def test_max_static_menu_rendering_and_placeholders(db_session):
    mock_client = SimpleNamespace(send_message=AsyncMock())

    # 1. Missing menu row -> safe placeholder
    await max_common.show_menu(mock_client, chat_id=123)
    mock_client.send_message.assert_called_with(
        chat_id=123,
        text="Раздел меню пока не настроен.",
    )

    mock_client.send_message.reset_mock()

    # 2. Empty text menu row -> safe placeholder
    async with db_session() as session:
        session.add(Content(key="menu", text_content="", is_visible=False))
        await session.commit()

    await max_common.show_menu(mock_client, chat_id=123)
    mock_client.send_message.assert_called_with(
        chat_id=123,
        text="Раздел меню пока не настроен.",
        attachments=None,
    )

    mock_client.send_message.reset_mock()

    # 3. Buttons-only menu row -> clean rendering with keyboard, no "В меню" row appended
    async with db_session() as session:
        content = await session.get(Content, "menu")
        content.text_content = "[Темы](btn:svc:topics) [Подписка](btn:svc:subscription)"
        await session.commit()

    await max_common.show_menu(mock_client, chat_id=123)
    call_kwargs = mock_client.send_message.call_args.kwargs
    assert call_kwargs["text"] == ""
    attachments = call_kwargs["attachments"]
    buttons = attachments[0]["payload"]["buttons"]
    button_texts = [b["text"] for row in buttons for b in row]
    assert "Темы" in button_texts
    assert "Подписка" in button_texts
    assert "⬅️ В меню" not in button_texts


@pytest.mark.asyncio
async def test_max_ordinary_content_custom_buttons_and_in_menu_in_single_keyboard(db_session):
    async with db_session() as session:
        session.add(
            Content(
                key="about_us",
                text_content="О нас\n[Сайт](https://example.com) [Подписка](btn:svc:subscription)",
                is_visible=True,
            )
        )
        await session.commit()

    mock_client = SimpleNamespace(send_message=AsyncMock())
    rendered = await max_common.render_static_content(mock_client, chat_id=123, user_id=123, content_key="about_us")
    assert rendered is True

    call_kwargs = mock_client.send_message.call_args.kwargs
    assert "О нас" in call_kwargs["text"]
    attachments = call_kwargs["attachments"]
    assert len(attachments) == 1
    buttons = attachments[0]["payload"]["buttons"]
    # Row 0: Site (link)
    assert buttons[0][0]["type"] == "link"
    assert buttons[0][0]["url"] == "https://example.com"
    # Row 1: Subscription
    assert buttons[1][0]["type"] == "callback"
    assert buttons[1][0]["payload"] == "ai_btn:svc:subscription"
    # Row 2: "⬅️ В меню"
    assert buttons[2][0]["type"] == "callback"
    assert buttons[2][0]["text"] == "⬅️ В меню"
    assert buttons[2][0]["payload"] == "main_menu"


@pytest.mark.asyncio
async def test_max_content_deep_link_same_renderer(db_session):
    async with db_session() as session:
        session.add(User(id=123, first_name="Test User", is_admin=False))
        session.add(
            Content(
                key="special_offer",
                text_content="Спецпредложение\n[Купить](https://offer.com)",
                is_visible=True,
            )
        )
        await session.commit()

    mock_client = SimpleNamespace(send_message=AsyncMock())
    await max_common.show_start_screen(mock_client, chat_id=123, user_id=123, start_payload="special_offer")

    call_kwargs = mock_client.send_message.call_args.kwargs
    assert "Спецпредложение" in call_kwargs["text"]
    buttons = call_kwargs["attachments"][0]["payload"]["buttons"]
    assert buttons[0][0]["url"] == "https://offer.com"
    assert buttons[1][0]["text"] == "⬅️ В меню"


@pytest.mark.asyncio
async def test_max_start_menu_canonical_route(db_session):
    async with db_session() as session:
        session.add(User(id=123, first_name="Test User", is_admin=False))
        session.add(
            Content(
                key="menu",
                text_content="Пользовательское меню\n[Темы](btn:svc:topics)",
                is_visible=False,  # Canonical menu is accessible even if is_visible is False
            )
        )
        await session.commit()

    mock_client = SimpleNamespace(send_message=AsyncMock())

    # Deep link ?start=menu
    await max_common.show_start_screen(mock_client, chat_id=123, user_id=123, start_payload="menu")

    assert mock_client.send_message.call_count == 1
    call_kwargs = mock_client.send_message.call_args.kwargs
    assert "Пользовательское меню" in call_kwargs["text"]
    assert "Главное меню:" not in call_kwargs["text"]
    buttons = call_kwargs["attachments"][0]["payload"]["buttons"]
    button_texts = [b["text"] for row in buttons for b in row]
    assert "Темы" in button_texts
    # Must NOT have automatic "⬅️ В меню" self-loop
    assert "⬅️ В меню" not in button_texts


@pytest.mark.asyncio
async def test_max_reset_topic_static_parser_behavior(db_session):
    async with db_session() as session:
        session.add(User(id=555, first_name="User", current_topic_id=1))
        session.add(
            Content(
                key="start_message",
                text_content="Приветствие\n[Темы](btn:svc:topics)",
                is_visible=True,
            )
        )
        await session.commit()

    mock_client = SimpleNamespace(send_message=AsyncMock())
    await max_topics.reset_topic(mock_client, chat_id=555, user_id=555)

    # 1. render_static_content for start_message (parsed buttons, no raw markdown)
    # 2. "✅ Тема сброшена." with "⬅️ В меню"
    assert mock_client.send_message.call_count == 2
    first_call = mock_client.send_message.call_args_list[0].kwargs
    assert "Приветствие" in first_call["text"]
    assert "[Темы]" not in first_call["text"]

    second_call = mock_client.send_message.call_args_list[1].kwargs
    assert "✅ Тема сброшена." in second_call["text"]
    assert second_call["attachments"][0]["payload"]["buttons"][0][0]["text"] == "⬅️ В меню"


@pytest.mark.asyncio
async def test_max_cancel_test_no_dynamic_menu(db_session):
    mock_client = SimpleNamespace(send_message=AsyncMock(), answer_callback=AsyncMock())
    app = MaxBotApplication(client=mock_client)

    # Cancel test callback
    callback = IncomingCallback(
        raw={},
        callback_id="cb_cancel",
        payload="cancel_test",
        chat_id=123,
        message_id="m123",
        sender=Sender(user_id=123, username="user", first_name="User", last_name=None),
    )
    await app.handle_callback(callback)

    mock_client.send_message.assert_called_once()
    call = mock_client.send_message.call_args.kwargs
    assert "❌ Тестирование прервано." in call["text"]
    buttons = call["attachments"][0]["payload"]["buttons"]
    assert buttons[0][0]["text"] == "⬅️ В меню"


@pytest.mark.asyncio
async def test_max_onboarding_completion_no_dynamic_menu(db_session):
    async with db_session() as session:
        session.add(User(id=777, first_name="Onboarding User", is_admin=False))
        await session.commit()

    mock_client = SimpleNamespace(send_message=AsyncMock(), answer_callback=AsyncMock())
    app = MaxBotApplication(client=mock_client)

    # Set user in onboarding state
    await app.states.set(user_id=777, chat_id=777, state="onboarding_gender", data={"is_onboarding": True, "name": "Onboarding User"})

    callback = IncomingCallback(
        raw={},
        callback_id="cb_gender",
        payload="gender_male",
        chat_id=777,
        message_id="m777",
        sender=Sender(user_id=777, username="user", first_name="Onboarding User", last_name=None),
    )
    await app.handle_callback(callback)

    # Verify callback answered and confirmation message sent with static "⬅️ В меню" row
    mock_client.answer_callback.assert_called_with("cb_gender", notification="Пол сохранён")
    mock_client.send_message.assert_called_once()
    call_kwargs = mock_client.send_message.call_args.kwargs
    assert "Отлично!" in call_kwargs["text"]
    buttons = call_kwargs["attachments"][0]["payload"]["buttons"]
    assert buttons[0][0]["text"] == "⬅️ В меню"
    assert buttons[0][0]["payload"] == "main_menu"


@pytest.mark.asyncio
async def test_max_callback_ack_and_unknown_svc(db_session):
    mock_client = SimpleNamespace(send_message=AsyncMock(), answer_callback=AsyncMock())
    app = MaxBotApplication(client=mock_client)

    # 1. main_menu callback answered and opens menu
    callback_main = IncomingCallback(
        raw={},
        callback_id="cb_main",
        payload="main_menu",
        chat_id=123,
        message_id="m123",
        sender=Sender(user_id=123, username="user", first_name="User", last_name=None),
    )
    with patch("max_messenger_bot.services.common.show_menu", new_callable=AsyncMock) as mock_show_menu:
        await app.handle_callback(callback_main)
        mock_client.answer_callback.assert_called_with("cb_main")
        mock_show_menu.assert_called_once_with(mock_client, 123, user_id=123)

    mock_client.answer_callback.reset_mock()

    # 2. svc:menu callback answered and opens menu
    callback_svc_menu = IncomingCallback(
        raw={},
        callback_id="cb_svc_menu",
        payload="ai_btn:svc:menu",
        chat_id=123,
        message_id="m123",
        sender=Sender(user_id=123, username="user", first_name="User", last_name=None),
    )
    with patch("max_messenger_bot.services.common.show_menu", new_callable=AsyncMock) as mock_show_menu:
        await app.handle_callback(callback_svc_menu)
        mock_client.answer_callback.assert_called_with("cb_svc_menu")
        mock_show_menu.assert_called_once_with(mock_client, 123, user_id=123)

    mock_client.answer_callback.reset_mock()

    # 3. Unknown svc action acknowledged and consumed, never reaching AI
    callback_unknown = IncomingCallback(
        raw={},
        callback_id="cb_unk",
        payload="ai_btn:svc:unknown_action",
        chat_id=123,
        message_id="m123",
        sender=Sender(user_id=123, username="user", first_name="User", last_name=None),
    )
    with patch("max_messenger_bot.ai.get_ai_response", new_callable=AsyncMock) as mock_ai:
        await app.handle_callback(callback_unknown)
        mock_client.answer_callback.assert_called_with("cb_unk")
        mock_ai.assert_not_called()


# ==============================================================================
# TELEGRAM BOT TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_telegram_start_and_ordinary_static_buttons_with_reply_kb_preserved(db_session):
    async with db_session() as session:
        session.add(User(id=2001, first_name="TG User", is_admin=False))
        session.add(
            Content(
                key="start_message",
                text_content="Привет!\n[Подписка](btn:svc:subscription)",
                is_visible=True,
            )
        )
        session.add(
            Content(
                key="info_page",
                text_content="Инфо\n[На сайт](https://example.com)",
                is_visible=True,
            )
        )
        await session.commit()

    mock_bot = SimpleNamespace(send_message=AsyncMock())

    # 1. start_message rendering
    rendered = await handlers.render_static_content_telegram(
        mock_bot,
        chat_id=2001,
        user_id=2001,
        content_key="start_message",
        is_start=True,
    )
    assert rendered is True
    # First message: text with inline keyboard
    # Second message: NAVIGATION_MENU_HINT with main_client_keyboard
    assert mock_bot.send_message.call_count == 2
    msg1_args, msg1_kwargs = mock_bot.send_message.call_args_list[0]
    msg1_text = msg1_args[1] if len(msg1_args) > 1 else msg1_kwargs.get("text")
    msg1_markup = msg1_kwargs.get("reply_markup")
    assert "Привет!" in msg1_text
    assert isinstance(msg1_markup, handlers.InlineKeyboardMarkup)

    msg2_args, msg2_kwargs = mock_bot.send_message.call_args_list[1]
    msg2_text = msg2_args[1] if len(msg2_args) > 1 else msg2_kwargs.get("text")
    assert msg2_text == handlers.NAVIGATION_MENU_HINT
    assert msg2_kwargs.get("reply_markup") is not None  # reply menu preserved

    mock_bot.send_message.reset_mock()

    # 2. Ordinary content rendering
    rendered = await handlers.render_static_content_telegram(
        mock_bot,
        chat_id=2001,
        user_id=2001,
        content_key="info_page",
    )
    assert rendered is True
    assert mock_bot.send_message.call_count == 2
    msg_info_args, msg_info_kwargs = mock_bot.send_message.call_args_list[0]
    msg_info_text = msg_info_args[1] if len(msg_info_args) > 1 else msg_info_kwargs.get("text")
    assert "Инфо" in msg_info_text
    msg_hint_args, msg_hint_kwargs = mock_bot.send_message.call_args_list[1]
    msg_hint_text = msg_hint_args[1] if len(msg_hint_args) > 1 else msg_hint_kwargs.get("text")
    assert msg_hint_text == handlers.NAVIGATION_MENU_HINT


@pytest.mark.asyncio
async def test_telegram_svc_menu_preserves_telegram_menu(db_session):
    async with db_session() as session:
        session.add(User(id=2002, first_name="TG User", is_admin=False))
        await session.commit()

    callback = SimpleNamespace(
        data="ai_btn:svc:menu",
        from_user=SimpleNamespace(id=2002),
        message=SimpleNamespace(
            message_id=101,
            chat=SimpleNamespace(id=2002),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="В меню", callback_data="ai_btn:svc:menu")]]
            ),
        ),
        answer=AsyncMock(),
    )
    mock_bot = SimpleNamespace()
    mock_state = AsyncMock()

    await handlers.process_response_button(callback, mock_state, mock_bot)

    assert callback.message.answer.call_count == 2
    hint_call = callback.message.answer.call_args_list[1]
    assert hint_call[0][0] == handlers.NAVIGATION_MENU_HINT
    # Reply keyboard is passed, not Content("menu")
    assert hint_call[1]["reply_markup"] is not None


@pytest.mark.asyncio
async def test_telegram_svc_start_pure_rendering(db_session):
    async with db_session() as session:
        session.add(User(id=2003, first_name="TG User", is_admin=False))
        session.add(Content(key="start_message", text_content="Старт текст", is_visible=True))
        await session.commit()

    callback = SimpleNamespace(
        data="ai_btn:svc:start",
        from_user=SimpleNamespace(id=2003),
        message=SimpleNamespace(
            message_id=102,
            chat=SimpleNamespace(id=2003),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="В начало", callback_data="ai_btn:svc:start")]]
            ),
        ),
        answer=AsyncMock(),
    )
    mock_bot = SimpleNamespace(send_message=AsyncMock())
    mock_state = AsyncMock()

    await handlers.process_response_button(callback, mock_state, mock_bot)

    # Renders start_message without executing referral/trial logic
    assert mock_bot.send_message.call_count >= 1
    call_args = mock_bot.send_message.call_args_list[0]
    text = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("text")
    assert "Старт текст" in text


@pytest.mark.asyncio
async def test_telegram_svc_topic_security(db_session):
    async with db_session() as session:
        session.add(User(id=2004, first_name="Non Admin", is_admin=False, current_topic_id=None))
        session.add(Topic(id=10, name="Inactive Topic", is_active=False, admin_only=False))
        session.add(Topic(id=11, name="Admin Topic", is_active=True, admin_only=True))
        session.add(Topic(id=12, name="Public Topic", is_active=True, admin_only=False))
        await session.commit()

    mock_bot = SimpleNamespace(send_message=AsyncMock())
    mock_state = AsyncMock()

    # 1. Attempt inactive topic switch
    cb_inactive = SimpleNamespace(
        data="ai_btn:svc:topic:10",
        from_user=SimpleNamespace(id=2004),
        message=SimpleNamespace(
            message_id=103,
            chat=SimpleNamespace(id=2004),
            delete=AsyncMock(),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="Тема 10", callback_data="ai_btn:svc:topic:10")]]
            ),
        ),
        answer=AsyncMock(),
        model_copy=lambda update: SimpleNamespace(
            data=update.get("data", "select_topic_10"),
            from_user=SimpleNamespace(id=2004),
            message=SimpleNamespace(
                message_id=103,
                chat=SimpleNamespace(id=2004),
                delete=AsyncMock(),
                answer=AsyncMock(),
                edit_reply_markup=AsyncMock(),
                reply_markup=handlers.InlineKeyboardMarkup(
                    inline_keyboard=[[handlers.InlineKeyboardButton(text="Тема 10", callback_data="ai_btn:svc:topic:10")]]
                ),
            ),
            answer=AsyncMock(),
        ),
    )
    await handlers.process_response_button(cb_inactive, mock_state, mock_bot)
    # User's topic must NOT be changed in database
    async with db_session() as session:
        user = await session.get(User, 2004)
        assert user.current_topic_id is None

    # 2. Attempt admin_only topic switch for non-admin
    cb_admin_only = SimpleNamespace(
        data="ai_btn:svc:topic:11",
        from_user=SimpleNamespace(id=2004),
        message=SimpleNamespace(
            message_id=104,
            chat=SimpleNamespace(id=2004),
            delete=AsyncMock(),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="Тема 11", callback_data="ai_btn:svc:topic:11")]]
            ),
        ),
        answer=AsyncMock(),
        model_copy=lambda update: SimpleNamespace(
            data=update.get("data", "select_topic_11"),
            from_user=SimpleNamespace(id=2004),
            message=SimpleNamespace(
                message_id=104,
                chat=SimpleNamespace(id=2004),
                delete=AsyncMock(),
                answer=AsyncMock(),
                edit_reply_markup=AsyncMock(),
                reply_markup=handlers.InlineKeyboardMarkup(
                    inline_keyboard=[[handlers.InlineKeyboardButton(text="Тема 11", callback_data="ai_btn:svc:topic:11")]]
                ),
            ),
            answer=AsyncMock(),
        ),
    )
    await handlers.process_response_button(cb_admin_only, mock_state, mock_bot)
    async with db_session() as session:
        user = await session.get(User, 2004)
        assert user.current_topic_id is None


@pytest.mark.asyncio
async def test_telegram_svc_content_visibility(db_session):
    async with db_session() as session:
        session.add(User(id=2005, first_name="TG User", is_admin=False))
        session.add(Content(key="hidden_section", text_content="Секретный текст", is_visible=False))
        session.add(Content(key="visible_section", text_content="Публичный текст", is_visible=True))
        await session.commit()

    mock_bot = SimpleNamespace(send_message=AsyncMock())
    mock_state = AsyncMock()

    # 1. Hidden content -> rejected
    cb_hidden = SimpleNamespace(
        data="ai_btn:svc:content:hidden_section",
        from_user=SimpleNamespace(id=2005),
        message=SimpleNamespace(
            message_id=105,
            chat=SimpleNamespace(id=2005),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="Секрет", callback_data="ai_btn:svc:content:hidden_section")]]
            ),
        ),
        answer=AsyncMock(),
    )
    await handlers.process_response_button(cb_hidden, mock_state, mock_bot)
    mock_bot.send_message.assert_not_called()

    # 2. Visible content -> rendered
    cb_visible = SimpleNamespace(
        data="ai_btn:svc:content:visible_section",
        from_user=SimpleNamespace(id=2005),
        message=SimpleNamespace(
            message_id=106,
            chat=SimpleNamespace(id=2005),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="Публично", callback_data="ai_btn:svc:content:visible_section")]]
            ),
        ),
        answer=AsyncMock(),
    )
    await handlers.process_response_button(cb_visible, mock_state, mock_bot)
    assert mock_bot.send_message.call_count >= 1
    call_args = mock_bot.send_message.call_args_list[0]
    text = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("text")
    assert "Публичный текст" in text


@pytest.mark.asyncio
async def test_telegram_unknown_svc_and_normal_ai_action(db_session):
    async with db_session() as session:
        session.add(User(id=2006, first_name="TG User", is_admin=False, current_dialogue_id=1))
        await session.commit()

    mock_bot = SimpleNamespace(send_message=AsyncMock())
    mock_state = AsyncMock()

    # 1. Unknown svc action -> consumed safely, never sent to AI
    cb_unknown = SimpleNamespace(
        data="ai_btn:svc:non_existent",
        from_user=SimpleNamespace(id=2006),
        message=SimpleNamespace(
            message_id=107,
            chat=SimpleNamespace(id=2006),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="Неизвестно", callback_data="ai_btn:svc:non_existent")]]
            ),
        ),
        answer=AsyncMock(),
    )
    with patch("handlers.process_buffered_messages", new_callable=AsyncMock) as mock_process:
        await handlers.process_response_button(cb_unknown, mock_state, mock_bot)
        mock_process.assert_not_called()
        assert 2006 not in handlers.user_message_buffers

    # 2. Normal AI action -> buffered and processed
    cb_ai = SimpleNamespace(
        data="ai_btn:custom_choice:1",
        from_user=SimpleNamespace(id=2006),
        message=SimpleNamespace(
            message_id=108,
            chat=SimpleNamespace(id=2006),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[
                    [handlers.InlineKeyboardButton(text="Мой выбор", callback_data="ai_btn:custom_choice:1")]
                ]
            ),
        ),
        answer=AsyncMock(),
    )
    with patch("handlers.process_buffered_messages", new_callable=AsyncMock) as mock_process:
        await handlers.process_response_button(cb_ai, mock_state, mock_bot)
        mock_process.assert_called_once()
        assert 2006 in handlers.user_message_buffers
        assert "Мой выбор" in handlers.user_message_buffers[2006][0]


@pytest.mark.asyncio
async def test_telegram_legacy_action_btn_fallback(db_session):
    async with db_session() as session:
        session.add(User(id=2007, first_name="TG User", is_admin=False))
        session.add(
            Content(
                key="start_message",
                text_content="Текст без разметки кнопок",
                action_btn_text="Legacy Кнопка",
                action_btn_payload="legacy_payload",
                is_visible=True,
            )
        )
        await session.commit()

    mock_bot = SimpleNamespace(send_message=AsyncMock())
    await handlers.render_static_content_telegram(
        mock_bot,
        chat_id=2007,
        user_id=2007,
        content_key="start_message",
        is_start=True,
    )

    assert mock_bot.send_message.call_count >= 1
    call1 = mock_bot.send_message.call_args_list[0]
    text = call1[0][1] if len(call1[0]) > 1 else call1[1].get("text")
    assert "Текст без разметки кнопок" in text
    inline_kb = call1[1].get("reply_markup")
    assert inline_kb is not None
    assert inline_kb.inline_keyboard[0][0].text == "Legacy Кнопка"


@pytest.mark.asyncio
async def test_telegram_long_text_split_and_inline_keyboard_on_last_chunk(db_session):
    async with db_session() as session:
        session.add(User(id=2010, first_name="TG User", is_admin=False))
        # Text with >4000 characters and buttons at the end
        long_paragraph = "Это длинный текст параграфа для проверки разбиения. " * 120
        full_text = f"{long_paragraph}\n\n[Выбрать тему](btn:svc:topics)"
        session.add(
            Content(
                key="long_page",
                text_content=full_text,
                is_visible=True,
            )
        )
        await session.commit()

    mock_bot = SimpleNamespace(send_message=AsyncMock())
    rendered = await handlers.render_static_content_telegram(
        mock_bot,
        chat_id=2010,
        user_id=2010,
        content_key="long_page",
    )
    assert rendered is True

    # 1. First chunk: text with reply_markup=None
    # 2. Last chunk: text with reply_markup=inline_kb
    # 3. Third message: NAVIGATION_MENU_HINT with main_client_keyboard
    assert mock_bot.send_message.call_count >= 3
    chunk1_call = mock_bot.send_message.call_args_list[0]
    chunk1_markup = chunk1_call[1].get("reply_markup")
    assert chunk1_markup is None

    last_chunk_call = mock_bot.send_message.call_args_list[-2]
    last_chunk_markup = last_chunk_call[1].get("reply_markup")
    assert isinstance(last_chunk_markup, handlers.InlineKeyboardMarkup)
    assert last_chunk_markup.inline_keyboard[0][0].text == "Выбрать тему"

    hint_call = mock_bot.send_message.call_args_list[-1]
    assert (hint_call[0][1] if len(hint_call[0]) > 1 else hint_call[1].get("text")) == handlers.NAVIGATION_MENU_HINT
    assert hint_call[1].get("reply_markup") is not None


@pytest.mark.asyncio
async def test_telegram_eleven_media_batches_10_and_1(db_session):
    from database import ContentMedia
    async with db_session() as session:
        session.add(User(id=2011, first_name="TG User", is_admin=False))
        content = Content(
            key="multi_media_page",
            text_content="Описание галереи",
            is_visible=True,
            content_order="media_top",
        )
        session.add(content)
        await session.flush()
        for idx in range(11):
            session.add(
                ContentMedia(
                    content_key="multi_media_page",
                    file_id=f"file_id_{idx}",
                    file_type="photo",
                )
            )
        await session.commit()

    mock_bot = SimpleNamespace(
        send_message=AsyncMock(),
        send_media_group=AsyncMock(),
        send_photo=AsyncMock(),
    )
    rendered = await handlers.render_static_content_telegram(
        mock_bot,
        chat_id=2011,
        user_id=2011,
        content_key="multi_media_page",
    )
    assert rendered is True

    # First batch (10 items) via send_media_group
    mock_bot.send_media_group.assert_called_once()
    media_arg = mock_bot.send_media_group.call_args.kwargs["media"]
    assert len(media_arg) == 10

    # 11th item sent via send_photo
    mock_bot.send_photo.assert_called_once()
    photo_arg = mock_bot.send_photo.call_args[0][1] if len(mock_bot.send_photo.call_args[0]) > 1 else mock_bot.send_photo.call_args[1].get("photo")
    assert photo_arg == "file_id_10"

    # Text message sent with main_client_keyboard
    mock_bot.send_message.assert_called_once()
    text_call = mock_bot.send_message.call_args
    assert "Описание галереи" in (text_call[0][1] if len(text_call[0]) > 1 else text_call[1].get("text"))
    assert text_call[1].get("reply_markup") is not None


@pytest.mark.asyncio
async def test_telegram_media_only_multi_media_restores_reply_keyboard(db_session):
    from database import ContentMedia
    async with db_session() as session:
        session.add(User(id=2012, first_name="TG User", is_admin=False))
        content = Content(
            key="media_only_page",
            text_content="",
            is_visible=True,
            content_order="media_top",
        )
        session.add(content)
        await session.flush()
        for idx in range(3):
            session.add(
                ContentMedia(
                    content_key="media_only_page",
                    file_id=f"photo_{idx}",
                    file_type="photo",
                )
            )
        await session.commit()

    mock_bot = SimpleNamespace(
        send_message=AsyncMock(),
        send_media_group=AsyncMock(),
        send_photo=AsyncMock(),
    )
    rendered = await handlers.render_static_content_telegram(
        mock_bot,
        chat_id=2012,
        user_id=2012,
        content_key="media_only_page",
    )
    assert rendered is True

    # Media group sent
    mock_bot.send_media_group.assert_called_once()
    assert len(mock_bot.send_media_group.call_args.kwargs["media"]) == 3

    # NAVIGATION_MENU_HINT sent with reply keyboard to restore Telegram navigation
    mock_bot.send_message.assert_called_once()
    hint_call = mock_bot.send_message.call_args
    assert (hint_call[0][1] if len(hint_call[0]) > 1 else hint_call[1].get("text")) == handlers.NAVIGATION_MENU_HINT
    assert hint_call[1].get("reply_markup") is not None


@pytest.mark.asyncio
async def test_telegram_duplicate_action_buttons_collision_and_label_resolution(db_session):
    async with db_session() as session:
        session.add(User(id=2013, first_name="TG User", is_admin=False, current_dialogue_id=1))
        session.add(
            Content(
                key="duplicate_buttons_page",
                text_content="Выберите опцию:\n[Первый](btn:same_action) | [Второй](btn:same_action)",
                is_visible=True,
            )
        )
        await session.commit()

    mock_bot = SimpleNamespace(send_message=AsyncMock())
    rendered = await handlers.render_static_content_telegram(
        mock_bot,
        chat_id=2013,
        user_id=2013,
        content_key="duplicate_buttons_page",
    )
    assert rendered is True

    from response_buttons import build_action_callback_data

    # Check rendered markup has distinct indexed callback_data
    msg1_call = mock_bot.send_message.call_args_list[0]
    markup = msg1_call[1].get("reply_markup")
    assert isinstance(markup, handlers.InlineKeyboardMarkup)
    row0 = markup.inline_keyboard[0]
    assert len(row0) == 2
    btn0 = row0[0]
    btn1 = row0[1]
    assert btn0.text == "Первый"
    assert btn0.callback_data == build_action_callback_data("same_action", 0)
    assert btn1.text == "Второй"
    assert btn1.callback_data == build_action_callback_data("same_action", 1)
    assert btn0.callback_data != btn1.callback_data

    # Verify clicking each button resolves its own visible label correctly in _prepare_ai_button_submission
    mock_state = AsyncMock()

    # Click btn0
    cb0 = SimpleNamespace(
        data=btn0.callback_data,
        from_user=SimpleNamespace(id=2013),
        message=SimpleNamespace(
            message_id=201,
            chat=SimpleNamespace(id=2013),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=markup,
        ),
        answer=AsyncMock(),
    )
    with patch("handlers.process_buffered_messages", new_callable=AsyncMock) as mock_process:
        await handlers.process_response_button(cb0, mock_state, mock_bot)
        assert 2013 in handlers.user_message_buffers
        assert '"Первый"' in handlers.user_message_buffers[2013][0]
        handlers.user_message_buffers.clear()

    # Click btn1
    cb1 = SimpleNamespace(
        data=btn1.callback_data,
        from_user=SimpleNamespace(id=2013),
        message=SimpleNamespace(
            message_id=202,
            chat=SimpleNamespace(id=2013),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=markup,
        ),
        answer=AsyncMock(),
    )
    with patch("handlers.process_buffered_messages", new_callable=AsyncMock) as mock_process:
        await handlers.process_response_button(cb1, mock_state, mock_bot)
        assert 2013 in handlers.user_message_buffers
        assert '"Второй"' in handlers.user_message_buffers[2013][0]
        handlers.user_message_buffers.clear()


# ==============================================================================
# MAX SERVICE ACTION MATRIX
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,target_patch",
    [
        ("ai_btn:svc:menu", "max_messenger_bot.services.common.show_menu"),
        ("ai_btn:svc:topics", "max_messenger_bot.services.topics.show_topics"),
        ("ai_btn:svc:subscription", "max_messenger_bot.services.subscriptions.show_subscription_info"),
        ("ai_btn:svc:referral", "max_messenger_bot.services.subscriptions.show_referral_info"),
        ("ai_btn:svc:settings", "max_messenger_bot.services.settings.show_settings"),
        ("ai_btn:svc:start", "max_messenger_bot.services.common.render_static_content"),
        ("ai_btn:svc:reset", "max_messenger_bot.services.common.reset_dialogue"),
    ],
)
async def test_max_service_action_matrix_routing(db_session, payload, target_patch):
    mock_client = SimpleNamespace(send_message=AsyncMock(), answer_callback=AsyncMock())
    app = MaxBotApplication(client=mock_client)

    callback = IncomingCallback(
        raw={},
        callback_id="cb_test_matrix",
        payload=payload,
        chat_id=100,
        message_id="m100",
        sender=Sender(user_id=100, username="user", first_name="User", last_name=None),
    )

    with patch(target_patch, new_callable=AsyncMock) as mock_target, \
         patch("max_messenger_bot.ai.get_ai_response", new_callable=AsyncMock) as mock_ai, \
         patch("max_messenger_bot.services.common.run_ai_dialogue", new_callable=AsyncMock) as mock_dialogue:

        await app.handle_callback(callback)

        assert mock_target.call_count == 1
        mock_ai.assert_not_called()
        mock_dialogue.assert_not_called()
        assert mock_client.answer_callback.call_count == 1
        mock_client.answer_callback.assert_called_with("cb_test_matrix")


@pytest.mark.asyncio
async def test_max_service_action_continue(db_session):
    mock_client = SimpleNamespace(send_message=AsyncMock(), answer_callback=AsyncMock())
    app = MaxBotApplication(client=mock_client)

    callback = IncomingCallback(
        raw={},
        callback_id="cb_continue",
        payload="ai_btn:svc:continue",
        chat_id=100,
        message_id="m100",
        sender=Sender(user_id=100, username="user", first_name="User", last_name=None),
    )
    with patch("max_messenger_bot.ai.get_ai_response", new_callable=AsyncMock) as mock_ai:
        await app.handle_callback(callback)
        assert mock_client.answer_callback.call_count == 1
        mock_client.send_message.assert_called_once()
        assert "Введите ваше сообщение" in mock_client.send_message.call_args.kwargs["text"]
        mock_ai.assert_not_called()


@pytest.mark.asyncio
async def test_max_service_action_topic_valid_routing(db_session):
    async with db_session() as session:
        session.add(User(id=100, first_name="User", is_admin=False))
        session.add(Topic(id=5, name="Active Public Topic", is_active=True, admin_only=False))
        await session.commit()

    mock_client = SimpleNamespace(send_message=AsyncMock(), answer_callback=AsyncMock())
    app = MaxBotApplication(client=mock_client)

    callback = IncomingCallback(
        raw={},
        callback_id="cb_topic",
        payload="ai_btn:svc:topic:5",
        chat_id=100,
        message_id="m100",
        sender=Sender(user_id=100, username="user", first_name="User", last_name=None),
    )
    with patch("max_messenger_bot.services.topics.select_topic", new_callable=AsyncMock) as mock_select, \
         patch("max_messenger_bot.ai.get_ai_response", new_callable=AsyncMock) as mock_ai:
        await app.handle_callback(callback)
        assert mock_client.answer_callback.call_count == 1
        mock_select.assert_called_once_with(mock_client, 100, 100, 5)
        mock_ai.assert_not_called()


@pytest.mark.asyncio
async def test_max_service_action_content_visible_and_hidden(db_session):
    async with db_session() as session:
        session.add(User(id=100, first_name="User", is_admin=False))
        session.add(Content(key="visible_sec", text_content="Видно всем", is_visible=True))
        session.add(Content(key="hidden_sec", text_content="Скрыто", is_visible=False))
        await session.commit()

    mock_client = SimpleNamespace(send_message=AsyncMock(), answer_callback=AsyncMock())
    app = MaxBotApplication(client=mock_client)

    # 1. Visible content -> rendered
    cb_vis = IncomingCallback(
        raw={},
        callback_id="cb_vis",
        payload="ai_btn:svc:content:visible_sec",
        chat_id=100,
        message_id="m100",
        sender=Sender(user_id=100, username="user", first_name="User", last_name=None),
    )
    with patch("max_messenger_bot.ai.get_ai_response", new_callable=AsyncMock) as mock_ai:
        await app.handle_callback(cb_vis)
        assert mock_client.answer_callback.call_count == 1
        mock_client.send_message.assert_called_once()
        assert "Видно всем" in mock_client.send_message.call_args.kwargs["text"]
        mock_ai.assert_not_called()

    mock_client.send_message.reset_mock()
    mock_client.answer_callback.reset_mock()

    # 2. Hidden content -> not rendered
    cb_hid = IncomingCallback(
        raw={},
        callback_id="cb_hid",
        payload="ai_btn:svc:content:hidden_sec",
        chat_id=100,
        message_id="m100",
        sender=Sender(user_id=100, username="user", first_name="User", last_name=None),
    )
    with patch("max_messenger_bot.ai.get_ai_response", new_callable=AsyncMock) as mock_ai:
        await app.handle_callback(cb_hid)
        assert mock_client.answer_callback.call_count == 1
        mock_client.send_message.assert_not_called()
        mock_ai.assert_not_called()


@pytest.mark.asyncio
async def test_max_service_action_topic_security_matrix(db_session):
    async with db_session() as session:
        session.add(User(id=100, first_name="Regular User", is_admin=False, current_topic_id=None))
        session.add(Topic(id=21, name="Inactive Topic", is_active=False, admin_only=False))
        session.add(Topic(id=22, name="Admin Topic", is_active=True, admin_only=True))
        await session.commit()

    mock_client = SimpleNamespace(send_message=AsyncMock(), answer_callback=AsyncMock())
    app = MaxBotApplication(client=mock_client)

    # Inactive topic
    cb_inactive = IncomingCallback(
        raw={},
        callback_id="cb_inact",
        payload="ai_btn:svc:topic:21",
        chat_id=100,
        message_id="m100",
        sender=Sender(user_id=100, username="user", first_name="Regular User", last_name=None),
    )
    with patch("max_messenger_bot.ai.get_ai_response", new_callable=AsyncMock) as mock_ai:
        await app.handle_callback(cb_inactive)
        assert mock_client.answer_callback.call_count == 1
        async with db_session() as session:
            user = await session.get(User, 100)
            assert user.current_topic_id is None
        mock_ai.assert_not_called()

    mock_client.answer_callback.reset_mock()

    # Admin-only topic for non-admin user
    cb_admin_only = IncomingCallback(
        raw={},
        callback_id="cb_adm",
        payload="ai_btn:svc:topic:22",
        chat_id=100,
        message_id="m100",
        sender=Sender(user_id=100, username="user", first_name="Regular User", last_name=None),
    )
    with patch("max_messenger_bot.ai.get_ai_response", new_callable=AsyncMock) as mock_ai:
        await app.handle_callback(cb_admin_only)
        assert mock_client.answer_callback.call_count == 1
        async with db_session() as session:
            user = await session.get(User, 100)
            assert user.current_topic_id is None
        mock_ai.assert_not_called()


# ==============================================================================
# TELEGRAM SERVICE ACTION MATRIX
# ==============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,target_patch",
    [
        ("ai_btn:svc:topics", "handlers.select_topic_menu"),
        ("ai_btn:svc:subscription", "handlers.show_subscription_info"),
        ("ai_btn:svc:referral", "handlers.show_referral_info"),
        ("ai_btn:svc:settings", "handlers.user_settings_menu"),
        ("ai_btn:svc:reset", "handlers.ask_delete_history"),
    ],
)
async def test_telegram_service_action_matrix_positive_routing(db_session, payload, target_patch):
    async with db_session() as session:
        session.add(User(id=2020, first_name="TG User", is_admin=False))
        await session.commit()

    callback = SimpleNamespace(
        data=payload,
        from_user=SimpleNamespace(id=2020),
        message=SimpleNamespace(
            message_id=301,
            chat=SimpleNamespace(id=2020),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="Кнопка", callback_data=payload)]]
            ),
        ),
        answer=AsyncMock(),
    )
    mock_bot = SimpleNamespace(send_message=AsyncMock())
    mock_state = AsyncMock()

    with patch(target_patch, new_callable=AsyncMock) as mock_service:
        await handlers.process_response_button(callback, mock_state, mock_bot)
        assert callback.answer.call_count == 1
        assert mock_service.call_count == 1


@pytest.mark.asyncio
async def test_telegram_service_action_continue_and_topic_positive_routing(db_session):
    async with db_session() as session:
        session.add(User(id=2021, first_name="TG User", is_admin=False, current_topic_id=None))
        session.add(Topic(id=50, name="Public Active Topic", is_active=True, admin_only=False))
        await session.commit()

    mock_bot = SimpleNamespace(send_message=AsyncMock())
    mock_state = AsyncMock()

    # 1. svc:continue
    cb_continue = SimpleNamespace(
        data="ai_btn:svc:continue",
        from_user=SimpleNamespace(id=2021),
        message=SimpleNamespace(
            message_id=302,
            chat=SimpleNamespace(id=2021),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="Продолжить", callback_data="ai_btn:svc:continue")]]
            ),
        ),
        answer=AsyncMock(),
    )
    await handlers.process_response_button(cb_continue, mock_state, mock_bot)
    assert cb_continue.answer.call_count == 1
    assert cb_continue.message.answer.call_count == 2
    assert cb_continue.message.answer.call_args_list[1].args[0] == "Введите ваше сообщение для начала/продолжения диалога:"

    # 2. svc:topic:<public active id>
    cb_topic = SimpleNamespace(
        data="ai_btn:svc:topic:50",
        from_user=SimpleNamespace(id=2021),
        message=SimpleNamespace(
            message_id=303,
            chat=SimpleNamespace(id=2021),
            delete=AsyncMock(),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="Тема 50", callback_data="ai_btn:svc:topic:50")]]
            ),
        ),
        answer=AsyncMock(),
        model_copy=lambda update: SimpleNamespace(
            data=update.get("data", "select_topic_50"),
            from_user=SimpleNamespace(id=2021),
            message=SimpleNamespace(
                message_id=303,
                chat=SimpleNamespace(id=2021),
                delete=AsyncMock(),
                answer=AsyncMock(),
                edit_reply_markup=AsyncMock(),
                reply_markup=handlers.InlineKeyboardMarkup(
                    inline_keyboard=[[handlers.InlineKeyboardButton(text="Тема 50", callback_data="ai_btn:svc:topic:50")]]
                ),
            ),
            answer=AsyncMock(),
        ),
    )
    await handlers.process_response_button(cb_topic, mock_state, mock_bot)
    async with db_session() as session:
        user = await session.get(User, 2021)
        assert user.current_topic_id == 50


@pytest.mark.asyncio
async def test_telegram_service_actions_exact_single_acknowledgement(db_session):
    async with db_session() as session:
        session.add(User(id=2022, first_name="TG User", is_admin=False))
        session.add(Content(key="start_message", text_content="Старт", is_visible=True))
        await session.commit()

    mock_bot = SimpleNamespace(send_message=AsyncMock())
    mock_state = AsyncMock()

    # 1. svc:menu -> exactly 1 answer
    cb_menu = SimpleNamespace(
        data="ai_btn:svc:menu",
        from_user=SimpleNamespace(id=2022),
        message=SimpleNamespace(
            message_id=304,
            chat=SimpleNamespace(id=2022),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="В меню", callback_data="ai_btn:svc:menu")]]
            ),
        ),
        answer=AsyncMock(),
    )
    await handlers.process_response_button(cb_menu, mock_state, mock_bot)
    assert cb_menu.answer.call_count == 1

    # 2. svc:start -> exactly 1 answer
    cb_start = SimpleNamespace(
        data="ai_btn:svc:start",
        from_user=SimpleNamespace(id=2022),
        message=SimpleNamespace(
            message_id=305,
            chat=SimpleNamespace(id=2022),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="В начало", callback_data="ai_btn:svc:start")]]
            ),
        ),
        answer=AsyncMock(),
    )
    await handlers.process_response_button(cb_start, mock_state, mock_bot)
    assert cb_start.answer.call_count == 1

    # 3. unknown svc -> exactly 1 answer
    cb_unknown = SimpleNamespace(
        data="ai_btn:svc:nonexistent_custom_action",
        from_user=SimpleNamespace(id=2022),
        message=SimpleNamespace(
            message_id=306,
            chat=SimpleNamespace(id=2022),
            answer=AsyncMock(),
            edit_reply_markup=AsyncMock(),
            reply_markup=handlers.InlineKeyboardMarkup(
                inline_keyboard=[[handlers.InlineKeyboardButton(text="Неизвестно", callback_data="ai_btn:svc:nonexistent_custom_action")]]
            ),
        ),
        answer=AsyncMock(),
    )
    await handlers.process_response_button(cb_unknown, mock_state, mock_bot)
    assert cb_unknown.answer.call_count == 1


# ==============================================================================
# MAX STATIC MEDIA RENDERER TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_max_media_renderer_ordinary_content_media_top(db_session):
    async with db_session() as session:
        session.add(
            Content(
                key="media_page_top",
                text_content="Текст раздела\n[Действие](btn:svc:topics)",
                content_order="media_top",
                is_visible=True,
            )
        )
        session.add(
            MaxContentMedia(
                content_key="media_page_top",
                media_type="photo",
                token="tok_top_123",
            )
        )
        await session.commit()

    mock_client = SimpleNamespace(send_message=AsyncMock())
    rendered = await max_common.render_static_content(mock_client, 100, 100, "media_page_top")
    assert rendered is True

    # Call 1: Media attachment first
    # Call 2: Text + inline keyboard (custom button + "⬅️ В меню") second
    assert mock_client.send_message.call_count == 2
    call1 = mock_client.send_message.call_args_list[0].kwargs
    assert call1["text"] == ""
    assert call1["attachments"] == [{"type": "image", "payload": {"token": "tok_top_123"}}]

    call2 = mock_client.send_message.call_args_list[1].kwargs
    assert "Текст раздела" in call2["text"]
    assert len(call2["attachments"]) == 1
    buttons = call2["attachments"][0]["payload"]["buttons"]
    assert len(buttons) == 2
    assert buttons[0][0]["text"] == "Действие"
    assert buttons[0][0]["payload"] == "ai_btn:svc:topics"
    assert buttons[1][0]["text"] == "⬅️ В меню"
    assert buttons[1][0]["payload"] == "main_menu"


@pytest.mark.asyncio
async def test_max_media_renderer_ordinary_content_text_top(db_session):
    async with db_session() as session:
        session.add(
            Content(
                key="media_page_bottom",
                text_content="Текст раздела\n[Действие](btn:svc:topics)",
                content_order="text_top",
                is_visible=True,
            )
        )
        session.add(
            MaxContentMedia(
                content_key="media_page_bottom",
                media_type="photo",
                token="tok_bottom_456",
            )
        )
        await session.commit()

    mock_client = SimpleNamespace(send_message=AsyncMock())
    rendered = await max_common.render_static_content(mock_client, 100, 100, "media_page_bottom")
    assert rendered is True

    # Call 1: Text + inline keyboard first
    # Call 2: Media attachment second
    assert mock_client.send_message.call_count == 2
    call1 = mock_client.send_message.call_args_list[0].kwargs
    assert "Текст раздела" in call1["text"]
    assert len(call1["attachments"]) == 1
    buttons = call1["attachments"][0]["payload"]["buttons"]
    assert buttons[0][0]["text"] == "Действие"
    assert buttons[1][0]["text"] == "⬅️ В меню"

    call2 = mock_client.send_message.call_args_list[1].kwargs
    assert call2["text"] == ""
    assert call2["attachments"] == [{"type": "image", "payload": {"token": "tok_bottom_456"}}]


@pytest.mark.asyncio
async def test_max_media_renderer_ordinary_content_media_only(db_session):
    async with db_session() as session:
        session.add(
            Content(
                key="media_only_sec",
                text_content="",
                content_order="media_top",
                is_visible=True,
            )
        )
        session.add(
            MaxContentMedia(
                content_key="media_only_sec",
                media_type="photo",
                token="tok_only_789",
            )
        )
        await session.commit()

    mock_client = SimpleNamespace(send_message=AsyncMock())
    rendered = await max_common.render_static_content(mock_client, 100, 100, "media_only_sec")
    assert rendered is True

    # Call 1: Media rendered
    # Call 2: User gets "⬅️ В меню" (single keyboard, no duplicate inline keyboards)
    assert mock_client.send_message.call_count == 2
    call1 = mock_client.send_message.call_args_list[0].kwargs
    assert call1["text"] == ""
    assert call1["attachments"] == [{"type": "image", "payload": {"token": "tok_only_789"}}]

    call2 = mock_client.send_message.call_args_list[1].kwargs
    assert call2["text"] == ""
    attachments2 = call2["attachments"]
    assert len(attachments2) == 1
    assert attachments2[0]["type"] == "inline_keyboard"
    buttons = attachments2[0]["payload"]["buttons"]
    assert buttons[0][0]["text"] == "⬅️ В меню"


@pytest.mark.asyncio
async def test_max_media_renderer_menu_with_media_and_buttons_no_self_loop(db_session):
    async with db_session() as session:
        session.add(
            Content(
                key="menu",
                text_content="[Темы](btn:svc:topics)",
                content_order="media_top",
                is_visible=True,
            )
        )
        session.add(
            MaxContentMedia(
                content_key="menu",
                media_type="photo",
                token="tok_menu_photo",
            )
        )
        await session.commit()

    mock_client = SimpleNamespace(send_message=AsyncMock())
    await max_common.show_menu(mock_client, chat_id=100)

    # Menu has custom button but NO "⬅️ В меню" self loop
    assert mock_client.send_message.call_count >= 1
    last_call = mock_client.send_message.call_args_list[-1].kwargs
    kb_atts = [att for att in (last_call.get("attachments") or []) if att.get("type") == "inline_keyboard"]
    assert len(kb_atts) == 1
    buttons = kb_atts[0]["payload"]["buttons"]
    button_texts = [b["text"] for row in buttons for b in row]
    assert "Темы" in button_texts
    assert "⬅️ В меню" not in button_texts


# ==============================================================================
# TELEGRAM STATIC EDGE CASES
# ==============================================================================

@pytest.mark.asyncio
async def test_telegram_buttons_only_static_content(db_session):
    async with db_session() as session:
        session.add(User(id=2030, first_name="TG User", is_admin=False))
        session.add(
            Content(
                key="buttons_only_page",
                text_content="[Выбрать тему](btn:svc:topics)",
                is_visible=True,
            )
        )
        await session.commit()

    mock_bot = SimpleNamespace(send_message=AsyncMock())
    rendered = await handlers.render_static_content_telegram(
        mock_bot,
        chat_id=2030,
        user_id=2030,
        content_key="buttons_only_page",
    )
    assert rendered is True

    # 1. First message: valid message with inline keyboard
    # 2. Second message: NAVIGATION_MENU_HINT with reply menu
    assert mock_bot.send_message.call_count == 2
    msg1_args, msg1_kwargs = mock_bot.send_message.call_args_list[0]
    inline_markup = msg1_kwargs.get("reply_markup")
    assert isinstance(inline_markup, handlers.InlineKeyboardMarkup)
    assert inline_markup.inline_keyboard[0][0].text == "Выбрать тему"

    msg2_args, msg2_kwargs = mock_bot.send_message.call_args_list[1]
    msg2_text = msg2_args[1] if len(msg2_args) > 1 else msg2_kwargs.get("text")
    assert msg2_text == handlers.NAVIGATION_MENU_HINT
    assert msg2_kwargs.get("reply_markup") is not None


@pytest.mark.asyncio
async def test_telegram_single_photo_short_text_and_inline_button(db_session):
    from database import ContentMedia
    async with db_session() as session:
        session.add(User(id=2031, first_name="TG User", is_admin=False))
        content = Content(
            key="photo_short_sec",
            text_content="Короткое описание\n[Действие](btn:svc:subscription)",
            content_order="media_top",
            is_visible=True,
        )
        session.add(content)
        await session.flush()
        session.add(ContentMedia(content_key="photo_short_sec", file_id="ph_1", file_type="photo"))
        await session.commit()

    mock_bot = SimpleNamespace(send_message=AsyncMock(), send_photo=AsyncMock())
    rendered = await handlers.render_static_content_telegram(
        mock_bot,
        chat_id=2031,
        user_id=2031,
        content_key="photo_short_sec",
    )
    assert rendered is True

    # Photo sent with caption and inline markup
    mock_bot.send_photo.assert_called_once()
    photo_kwargs = mock_bot.send_photo.call_args.kwargs
    assert photo_kwargs.get("caption") == "Короткое описание"
    assert isinstance(photo_kwargs.get("reply_markup"), handlers.InlineKeyboardMarkup)
    assert photo_kwargs.get("reply_markup").inline_keyboard[0][0].text == "Действие"

    # Hint sent to restore reply navigation
    mock_bot.send_message.assert_called_once()
    hint_call = mock_bot.send_message.call_args
    assert (hint_call[0][1] if len(hint_call[0]) > 1 else hint_call[1].get("text")) == handlers.NAVIGATION_MENU_HINT
    assert hint_call[1].get("reply_markup") is not None


@pytest.mark.asyncio
async def test_telegram_modern_button_declarations_win_over_legacy_fields(db_session):
    async with db_session() as session:
        session.add(User(id=2032, first_name="TG User", is_admin=False))
        session.add(
            Content(
                key="modern_vs_legacy",
                text_content="Основной текст\n[Новая кнопка](btn:svc:topics)",
                action_btn_text="Старая кнопка",
                action_btn_payload="legacy_payload",
                is_visible=True,
            )
        )
        await session.commit()

    mock_bot = SimpleNamespace(send_message=AsyncMock())
    rendered = await handlers.render_static_content_telegram(
        mock_bot,
        chat_id=2032,
        user_id=2032,
        content_key="modern_vs_legacy",
    )
    assert rendered is True

    # Check that modern parsed button is present and legacy button is NOT
    msg1_kwargs = mock_bot.send_message.call_args_list[0][1]
    markup = msg1_kwargs.get("reply_markup")
    assert isinstance(markup, handlers.InlineKeyboardMarkup)
    button_texts = [b.text for row in markup.inline_keyboard for b in row]
    assert "Новая кнопка" in button_texts
    assert "Старая кнопка" not in button_texts


# ==============================================================================
# ADMIN UX TESTS
# ==============================================================================

def test_max_admin_panel_keyboard_has_no_manage_buttons():
    markup = max_admin_panel_keyboard()
    # Extract all callback payloads from inline keyboard
    buttons = markup[0]["payload"]["buttons"]
    callbacks = [btn["payload"] for row in buttons for btn in row]
    assert "admin_manage_buttons" not in callbacks
    assert "admin_stats" in callbacks
    assert "admin_content" in callbacks


@pytest.mark.asyncio
async def test_admin_content_management_includes_menu(db_session):
    async with db_session() as session:
        session.add(Content(key="menu", button_title=None, text_content="Меню бота", is_visible=True, sort_order=0))
        await session.commit()

    # 1. MAX content list includes "menu"
    mock_client = SimpleNamespace(send_message=AsyncMock())
    await max_admin_content.show_content_list(mock_client, chat_id=999)
    mock_client.send_message.assert_called_once()
    attachments = mock_client.send_message.call_args.kwargs["attachments"]
    buttons = attachments[0]["payload"]["buttons"]
    max_button_payloads = [btn["payload"] for row in buttons for btn in row]
    assert "admin_edit_content_menu" in max_button_payloads

    # 2. Telegram content management keyboard includes "menu"
    tg_markup = await tg_kb.content_management_keyboard()
    tg_callbacks = [btn.callback_data for row in tg_markup.inline_keyboard for btn in row]
    assert "edit_content_menu" in tg_callbacks
