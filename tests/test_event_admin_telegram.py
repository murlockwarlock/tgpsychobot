import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

os.environ.setdefault("BOT_TOKEN", "test_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from aiogram import Bot, Dispatcher
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.methods import AnswerCallbackQuery, EditMessageText, SendMessage
from aiogram.types import CallbackQuery, Chat, Message, Update, User as TgUser

from database import (
    Base,
    EventDelivery,
    EventRegistration,
    EventScenario,
    EventStep,
    User,
)
import event_admin_telegram
from event_admin_telegram import router as event_admin_router
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from time_helpers import utc_now


@pytest_asyncio.fixture
async def db_session_maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield maker
    await engine.dispose()


@pytest.fixture
def mock_bot():
    bot = AsyncMock(spec=Bot)
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=999))
    return bot


def _make_tg_user(user_id=12345, first_name="Admin"):
    return TgUser(id=user_id, is_bot=False, first_name=first_name, username="admin_user")


def _make_callback(data: str, user_id=12345, message_id=100) -> CallbackQuery:
    user = _make_tg_user(user_id)
    chat = Chat(id=user_id, type="private")
    message = MagicMock(spec=Message)
    message.chat = chat
    message.message_id = message_id
    message.edit_text = AsyncMock()
    message.answer = AsyncMock()

    callback = MagicMock(spec=CallbackQuery)
    callback.id = f"cb_{data}_{message_id}"
    callback.from_user = user
    callback.data = data
    callback.message = message
    callback.answer = AsyncMock()
    return callback


def _make_message(text: str, user_id=12345, message_id=101) -> Message:
    user = _make_tg_user(user_id)
    chat = Chat(id=user_id, type="private")
    msg = MagicMock(spec=Message)
    msg.from_user = user
    msg.chat = chat
    msg.message_id = message_id
    msg.text = text
    msg.answer = AsyncMock()
    return msg


@pytest.mark.asyncio
async def test_telegram_admin_root_list_and_card(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr("event_admin_telegram.get_all_admin_ids", AsyncMock(return_value={12345}))

    # 1. Create a test scenario in DB
    future_utc = datetime.utcnow() + timedelta(days=2)
    from event_admin_service import create_scenario
    scen = await create_scenario(
        db_session_maker,
        code="demo_webinar",
        event_type="webinar",
        title="Demo Webinar",
        starts_at_utc=future_utc,
        link="https://webinar.com",
    )

    storage = MemoryStorage()
    state = FSMContext(storage=storage, key=MagicMock(bot_id=1, chat_id=12345, user_id=12345))

    # 2. Open root list
    cb_root = _make_callback("admin_events")
    await event_admin_telegram.show_events_root(cb_root, state)
    cb_root.message.edit_text.assert_called_once()
    args, kwargs = cb_root.message.edit_text.call_args
    assert "Сценарии мероприятий" in args[0]
    assert "Demo Webinar" in args[0]

    # 3. Open scenario card
    cb_open = _make_callback(f"admin_ev_open_{scen.id}")
    await event_admin_telegram.open_scenario_card(cb_open, state)
    cb_open.message.edit_text.assert_called_once()
    args, kwargs = cb_open.message.edit_text.call_args
    assert "demo_webinar" in args[0]
    assert "Карточка мероприятия" in args[0]

    # 4. Toggle active state
    cb_toggle = _make_callback(f"admin_ev_toggle_{scen.id}")
    await event_admin_telegram.toggle_active_handler(cb_toggle)
    cb_toggle.answer.assert_called_with("Регистрация включена!")

    # 5. Archive confirmation flow
    cb_ask_arch = _make_callback(f"admin_ev_archive_ask_{scen.id}")
    await event_admin_telegram.ask_archive_handler(cb_ask_arch)
    args, kwargs = cb_ask_arch.message.edit_text.call_args
    assert "Вы уверены, что хотите архивировать" in args[0]

    cb_yes_arch = _make_callback(f"admin_ev_archive_yes_{scen.id}")
    await event_admin_telegram.archive_yes_handler(cb_yes_arch)
    cb_yes_arch.answer.assert_called_with("Мероприятие архивировано.")


@pytest.mark.asyncio
async def test_telegram_webinar_creation_flow(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr("event_admin_telegram.get_all_admin_ids", AsyncMock(return_value={12345}))

    storage = MemoryStorage()
    state = FSMContext(storage=storage, key=MagicMock(bot_id=1, chat_id=12345, user_id=12345))

    # Step 0: Click "Create Webinar"
    cb = _make_callback("admin_ev_create_webinar")
    await event_admin_telegram.start_scenario_creation(cb, state)
    assert await state.get_state() == event_admin_telegram.EventAdminStates.create_code.state

    # Step 1: Input Code
    msg_code = _make_message("new_online_conf")
    await event_admin_telegram.process_create_code(msg_code, state)
    assert await state.get_state() == event_admin_telegram.EventAdminStates.create_title.state

    # Step 2: Input Title
    msg_title = _make_message("Online Conference 2026")
    await event_admin_telegram.process_create_title(msg_title, state)
    assert await state.get_state() == event_admin_telegram.EventAdminStates.create_starts_at.state

    # Step 3: Input Starts at MSK
    msg_time = _make_message("29.10.2026 18:30")
    await event_admin_telegram.process_create_starts_at(msg_time, state)
    assert await state.get_state() == event_admin_telegram.EventAdminStates.create_duration.state

    # Step 4: Input Duration
    msg_dur = _make_message("90")
    await event_admin_telegram.process_create_duration(msg_dur, state)
    assert await state.get_state() == event_admin_telegram.EventAdminStates.create_link.state

    # Step 5: Input Link
    msg_link = _make_message("https://youtube.com/live/xyz123")
    await event_admin_telegram.process_create_link(msg_link, state)
    # State cleared and created confirmation sent
    assert await state.get_state() is None
    msg_link.answer.assert_called_once()
    args, kwargs = msg_link.answer.call_args
    assert "Мероприятие успешно создано" in args[0]
    assert "new_online_conf" in args[0]


@pytest.mark.asyncio
async def test_telegram_meeting_creation_flow(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr("event_admin_telegram.get_all_admin_ids", AsyncMock(return_value={12345}))

    storage = MemoryStorage()
    state = FSMContext(storage=storage, key=MagicMock(bot_id=1, chat_id=12345, user_id=12345))

    cb = _make_callback("admin_ev_create_meeting")
    await event_admin_telegram.start_scenario_creation(cb, state)

    await event_admin_telegram.process_create_code(_make_message("moscow_meetup"), state)
    await event_admin_telegram.process_create_title(_make_message("Moscow Offline Meetup"), state)
    await event_admin_telegram.process_create_starts_at(_make_message("30.10.2026 19:00"), state)
    await event_admin_telegram.process_create_duration(_make_message("180"), state)
    assert await state.get_state() == event_admin_telegram.EventAdminStates.create_place.state

    # Enter place
    msg_place = _make_message("Москва, ул. Тверская 7, Зал 3")
    await event_admin_telegram.process_create_place(msg_place, state)
    assert await state.get_state() == event_admin_telegram.EventAdminStates.create_meeting_link.state

    # Skip meeting link
    cb_skip = _make_callback("admin_ev_skip_meeting_link")
    await event_admin_telegram.skip_meeting_link(cb_skip, state)
    assert await state.get_state() is None


@pytest.mark.asyncio
async def test_telegram_reschedule_and_copy(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr("event_admin_telegram.get_all_admin_ids", AsyncMock(return_value={12345}))

    from event_admin_service import create_scenario
    scen = await create_scenario(
        db_session_maker,
        code="resched_tg",
        event_type="webinar",
        title="Resched TG Webinar",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://zoom.us/test",
    )

    storage = MemoryStorage()
    state = FSMContext(storage=storage, key=MagicMock(bot_id=1, chat_id=12345, user_id=12345))

    # 1. Reschedule
    cb_resched = _make_callback(f"admin_ev_reschedule_{scen.id}")
    await event_admin_telegram.start_reschedule(cb_resched, state)
    assert await state.get_state() == event_admin_telegram.EventAdminStates.reschedule_starts_at.state

    msg_new_time = _make_message("05.11.2026 20:00")
    await event_admin_telegram.process_reschedule(msg_new_time, state)
    assert await state.get_state() is None
    msg_new_time.answer.assert_called_once()
    assert "Мероприятие успешно перенесено" in msg_new_time.answer.call_args[0][0]

    # 2. Copy
    cb_copy = _make_callback(f"admin_ev_copy_{scen.id}")
    await event_admin_telegram.start_copy(cb_copy, state)
    assert await state.get_state() == event_admin_telegram.EventAdminStates.copy_code.state

    msg_copy_code = _make_message("resched_tg_next")
    await event_admin_telegram.process_copy_code(msg_copy_code, state)
    assert await state.get_state() == event_admin_telegram.EventAdminStates.copy_starts_at.state

    msg_copy_time = _make_message("10.11.2026 19:00")
    await event_admin_telegram.process_copy_starts_at(msg_copy_time, state)
    assert await state.get_state() is None
    msg_copy_time.answer.assert_called_once()
    assert "Сценарий успешно скопирован" in msg_copy_time.answer.call_args[0][0]


@pytest.mark.asyncio
async def test_telegram_steps_management(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr("event_admin_telegram.get_all_admin_ids", AsyncMock(return_value={12345}))

    from event_admin_service import create_scenario
    scen = await create_scenario(
        db_session_maker,
        code="steps_tg",
        event_type="webinar",
        title="Steps TG",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://link.com",
    )

    storage = MemoryStorage()
    state = FSMContext(storage=storage, key=MagicMock(bot_id=1, chat_id=12345, user_id=12345))

    # 1. View steps
    cb_steps = _make_callback(f"admin_ev_steps_{scen.id}")
    await event_admin_telegram.show_steps_list(cb_steps, state)
    assert "Цепочка сообщений-напоминаний" in cb_steps.message.edit_text.call_args[0][0]

    # 2. Add step with preset offset
    cb_add = _make_callback(f"admin_ev_step_add_{scen.id}")
    await event_admin_telegram.start_add_step(cb_add, state)

    cb_preset = _make_callback(f"admin_ev_offset_preset_{scen.id}_-1440")
    await event_admin_telegram.process_offset_preset(cb_preset, state)
    assert await state.get_state() == event_admin_telegram.EventAdminStates.step_text.state

    msg_text = _make_message("Завтра вебинар {event_title}! Ссылка: {link}")
    await event_admin_telegram.process_step_text(msg_text, state)
    assert await state.get_state() is None
    assert "Шаг напоминания успешно добавлен" in msg_text.answer.call_args[0][0]


@pytest.mark.asyncio
async def test_telegram_monitoring_and_uncertain_resolution(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr("event_admin_telegram.get_all_admin_ids", AsyncMock(return_value={12345}))

    from event_admin_service import create_scenario
    scen = await create_scenario(
        db_session_maker,
        code="mon_tg",
        event_type="webinar",
        title="Mon TG",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://link.com",
    )

    # Add mock uncertain delivery
    async with db_session_maker() as session:
        u = User(id=555, first_name="TargetUser")
        session.add(u)
        await session.flush()
        reg = EventRegistration(event_id=scen.id, user_id=555, status="started")
        session.add(reg)
        await session.flush()
        deliv = EventDelivery(
            event_id=scen.id,
            user_id=555,
            registration_id=reg.id,
            message_kind="immediate_confirmation",
            delivery_key="mon_deliv_key",
            due_at=utc_now(),
            next_attempt_at=utc_now(),
            platform="telegram",
            status="uncertain",
            last_error="Client network timeout",
        )
        session.add(deliv)
        await session.commit()
        deliv_id = deliv.id

    storage = MemoryStorage()
    state = FSMContext(storage=storage, key=MagicMock(bot_id=1, chat_id=12345, user_id=12345))

    # 1. Show stats
    cb_stats = _make_callback(f"admin_ev_stats_{scen.id}")
    await event_admin_telegram.show_delivery_stats(cb_stats)
    assert "Мониторинг отправок" in cb_stats.message.edit_text.call_args[0][0]
    assert "(uncertain): <b>1</b>" in cb_stats.message.edit_text.call_args[0][0]

    # 2. Show problem deliveries
    cb_probs = _make_callback(f"admin_ev_problems_{scen.id}_0")
    await event_admin_telegram.show_problem_deliveries(cb_probs)
    assert "Проблемные доставки (1)" in cb_probs.message.edit_text.call_args[0][0]

    # 3. Open problem delivery card
    cb_item = _make_callback(f"admin_ev_problem_item_{deliv_id}")
    await event_admin_telegram.open_problem_delivery(cb_item)
    assert f"Детали доставки ID {deliv_id}" in cb_item.message.edit_text.call_args[0][0]

    # 4. Resolve: prompt reason for mark_sent
    cb_res = _make_callback(f"admin_ev_resolve_sent_{deliv_id}")
    await event_admin_telegram.prompt_resolution_reason(cb_res, state)
    assert await state.get_state() == event_admin_telegram.EventAdminStates.resolve_reason.state

    # 5. Process reason
    msg_reason = _make_message("User confirmed receiving the message in chat")
    await event_admin_telegram.process_resolution_reason(msg_reason, state)
    assert await state.get_state() is None
    assert "успешно разрешена" in msg_reason.answer.call_args[0][0]
    assert "resolved_sent" in msg_reason.answer.call_args[0][0]


@pytest.mark.asyncio
async def test_telegram_test_send_to_self(db_session_maker, monkeypatch, mock_bot):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr("event_admin_telegram.get_all_admin_ids", AsyncMock(return_value={12345}))

    from event_admin_service import add_step, create_scenario
    scen = await create_scenario(
        db_session_maker,
        code="test_send_tg",
        event_type="webinar",
        title="Test Send TG",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://webinar.com/live",
    )
    step = await add_step(db_session_maker, scen.id, -120, "Привет, {name}! Ссылка: {link}")

    # Test send single step
    cb = _make_callback(f"admin_ev_test_step_{step.id}")
    await event_admin_telegram.test_send_single_step(cb, mock_bot)

    mock_bot.send_message.assert_called_once()
    sent_text = mock_bot.send_message.call_args.kwargs["text"]
    assert "ТЕСТОВОЕ НАПОМИНАНИЕ" in sent_text
    assert "https://webinar.com/live" in sent_text
    assert "Привет, Admin!" in sent_text


async def _feed_callback(dp: Dispatcher, bot: Bot, data: str, user_id: int = 12345, message_id: int = 100):
    user = TgUser(id=user_id, is_bot=False, first_name="Admin", username="admin_user")
    chat = Chat(id=user_id, type="private")
    msg = Message(message_id=message_id, date=datetime.now(), chat=chat, text="Original text")
    cb = CallbackQuery(id=f"cb_{data}_{message_id}", from_user=user, chat_instance="inst", data=data, message=msg)
    update = Update(update_id=1, callback_query=cb)
    bot.reset_mock()
    await dp.feed_update(bot, update)
    return [call.args[0] for call in bot.call_args_list if call.args and hasattr(call.args[0], "__class__")]


async def _feed_message(dp: Dispatcher, bot: Bot, text: str, user_id: int = 12345, message_id: int = 101):
    user = TgUser(id=user_id, is_bot=False, first_name="Admin", username="admin_user")
    chat = Chat(id=user_id, type="private")
    msg = Message(message_id=message_id, date=datetime.now(), chat=chat, text=text, from_user=user)
    update = Update(update_id=2, message=msg)
    bot.reset_mock()
    await dp.feed_update(bot, update)
    return [call.args[0] for call in bot.call_args_list if call.args and hasattr(call.args[0], "__class__")]


@pytest.mark.asyncio
async def test_telegram_dispatcher_step_routing_and_ordering(db_session_maker, monkeypatch):
    """
    Dispatcher-level integration test verifying that:
    1. Generic step card callback 'admin_ev_step_<id>' does NOT collide with specialized step callbacks:
       - edit text: 'admin_ev_step_edit_text_<id>'
       - edit offset: 'admin_ev_step_edit_offset_<id>'
       - move up: 'admin_ev_step_up_<id>'
       - move down: 'admin_ev_step_down_<id>'
       - delete ask/yes: 'admin_ev_step_del_ask_<id>', 'admin_ev_step_del_yes_<id>'
    2. move up / down handles real state dependency without unconfigured FSMContext errors.
    """
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr("event_admin_telegram.get_all_admin_ids", AsyncMock(return_value={12345}))

    from event_admin_service import add_step, create_scenario, get_step_item, list_steps

    scen = await create_scenario(
        db_session_maker,
        code="dispatch_steps",
        event_type="webinar",
        title="Dispatch Steps",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://webinar.com/dispatch",
    )
    step1 = await add_step(db_session_maker, scen.id, -1440, "Шаг 1: За сутки {event_title}")
    step2 = await add_step(db_session_maker, scen.id, -120, "Шаг 2: За 2 часа {event_title}")

    event_admin_router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(event_admin_router)
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789

    # 1. Open step card directly
    methods = await _feed_callback(dp, bot, f"admin_ev_step_{step1.id}")
    edit_methods = [m for m in methods if isinstance(m, EditMessageText)]
    assert len(edit_methods) >= 1
    assert "Шаг напоминания" in edit_methods[0].text
    assert "Шаг 1" in edit_methods[0].text

    # 2. Click Edit Text (must NOT open step card!)
    methods = await _feed_callback(dp, bot, f"admin_ev_step_edit_text_{step1.id}")
    edit_methods = [m for m in methods if isinstance(m, EditMessageText)]
    assert len(edit_methods) >= 1
    assert "Введите новый текст сообщения" in edit_methods[0].text

    # Provide new text via Message update
    methods = await _feed_message(dp, bot, "Обновлённый текст шага 1: {link}")
    send_methods = [m for m in methods if isinstance(m, SendMessage)]
    assert len(send_methods) >= 1
    assert "Текст шага обновлен" in send_methods[0].text
    updated_step1 = await get_step_item(db_session_maker, step1.id)
    assert updated_step1.message_text == "Обновлённый текст шага 1: {link}"

    # 3. Click Edit Offset (must NOT open step card!)
    methods = await _feed_callback(dp, bot, f"admin_ev_step_edit_offset_{step1.id}")
    edit_methods = [m for m in methods if isinstance(m, EditMessageText)]
    assert len(edit_methods) >= 1
    assert "Выберите новое время отправки" in edit_methods[0].text

    # Select preset offset (-60)
    methods = await _feed_callback(dp, bot, f"admin_ev_step_edit_preset_{step1.id}_-60")
    edit_methods = [m for m in methods if isinstance(m, EditMessageText)]
    assert any("Время отправки обновлено" in m.text for m in edit_methods)
    updated_step1 = await get_step_item(db_session_maker, step1.id)
    assert updated_step1.offset_minutes == -60

    # 4. Click Move Up on step2
    # Before fix, step_move_up called open_step_card(callback, FSMContext()) without storage/key
    methods = await _feed_callback(dp, bot, f"admin_ev_step_up_{step2.id}")
    # Verify card re-rendered cleanly and step moved
    edit_methods = [m for m in methods if isinstance(m, EditMessageText)]
    assert len(edit_methods) >= 1
    assert "Шаг напоминания" in edit_methods[0].text

    # Verify order in DB: step2 is now first
    steps = await list_steps(db_session_maker, scen.id, include_archived=False)
    assert steps[0].id == step2.id
    assert steps[1].id == step1.id

    # 5. Click Move Down on step2
    methods = await _feed_callback(dp, bot, f"admin_ev_step_down_{step2.id}")
    edit_methods = [m for m in methods if isinstance(m, EditMessageText)]
    assert len(edit_methods) >= 1
    assert "Шаг напоминания" in edit_methods[0].text

    steps = await list_steps(db_session_maker, scen.id, include_archived=False)
    assert steps[0].id == step1.id
    assert steps[1].id == step2.id

    # 6. Click Delete / Archive Ask on step1
    methods = await _feed_callback(dp, bot, f"admin_ev_step_del_ask_{step1.id}")
    edit_methods = [m for m in methods if isinstance(m, EditMessageText)]
    assert len(edit_methods) >= 1
    assert "Удалить/архивировать шаг" in edit_methods[0].text

    # Confirm Delete Yes
    methods = await _feed_callback(dp, bot, f"admin_ev_step_del_yes_{step1.id}")
    answer_methods = [m for m in methods if isinstance(m, AnswerCallbackQuery)]
    assert any("Шаг архивирован" in (m.text or "") for m in answer_methods)
    steps_remaining = await list_steps(db_session_maker, scen.id, include_archived=False)
    assert len(steps_remaining) == 1
    assert steps_remaining[0].id == step2.id


@pytest.mark.asyncio
async def test_telegram_dispatcher_lifecycle_monitoring_and_test_send(db_session_maker, monkeypatch):
    """
    Dispatcher-level integration tests for:
    - archive flow (ask & yes)
    - stats monitoring
    - uncertain resolution flow (prompt & reason)
    - test send
    """
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr("event_admin_telegram.get_all_admin_ids", AsyncMock(return_value={12345}))

    from event_admin_service import add_step, create_scenario, get_scenario_summary

    scen = await create_scenario(
        db_session_maker,
        code="dispatch_lifecycle",
        event_type="webinar",
        title="Dispatch Lifecycle",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://webinar.com/life",
    )
    step = await add_step(db_session_maker, scen.id, -120, "Привет, {name}! Ссылка: {link}")

    async with db_session_maker() as session:
        u = User(id=888, first_name="UncertainUser")
        session.add(u)
        await session.flush()
        reg = EventRegistration(event_id=scen.id, user_id=888, status="started")
        session.add(reg)
        await session.flush()
        deliv = EventDelivery(
            event_id=scen.id,
            user_id=888,
            registration_id=reg.id,
            message_kind="immediate_confirmation",
            delivery_key="dispatch_deliv_key",
            due_at=utc_now(),
            next_attempt_at=utc_now(),
            platform="telegram",
            status="uncertain",
            last_error="Client network timeout",
        )
        session.add(deliv)
        await session.commit()
        deliv_id = deliv.id

    event_admin_router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(event_admin_router)
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=999))

    # 1. Stats callback
    methods = await _feed_callback(dp, bot, f"admin_ev_stats_{scen.id}")
    edit_methods = [m for m in methods if isinstance(m, EditMessageText)]
    assert len(edit_methods) >= 1
    assert "Мониторинг отправок" in edit_methods[0].text
    assert "(uncertain): <b>1</b>" in edit_methods[0].text

    # 2. Uncertain resolution: click resolve sent
    methods = await _feed_callback(dp, bot, f"admin_ev_resolve_sent_{deliv_id}")
    edit_methods = [m for m in methods if isinstance(m, EditMessageText)]
    assert len(edit_methods) >= 1
    assert "Введите причину разрешения" in edit_methods[0].text

    # Submit reason via Message
    methods = await _feed_message(dp, bot, "Пользователь подтвердил получение в чате поддержки")
    send_methods = [m for m in methods if isinstance(m, SendMessage)]
    assert len(send_methods) >= 1
    assert "успешно разрешена" in send_methods[0].text

    async with db_session_maker() as session:
        d = await session.get(EventDelivery, deliv_id)
        assert d.status == "resolved_sent"

    # 3. Test send step
    methods = await _feed_callback(dp, bot, f"admin_ev_test_step_{step.id}")
    # In test_send_single_step, it calls bot.send_message
    assert bot.send_message.called
    sent_text = bot.send_message.call_args.kwargs["text"]
    assert "ТЕСТОВОЕ НАПОМИНАНИЕ" in sent_text

    # 4. Archive flow: ask
    methods = await _feed_callback(dp, bot, f"admin_ev_archive_ask_{scen.id}")
    edit_methods = [m for m in methods if isinstance(m, EditMessageText)]
    assert len(edit_methods) >= 1
    assert "Вы уверены, что хотите архивировать" in edit_methods[0].text

    # Archive flow: yes
    methods = await _feed_callback(dp, bot, f"admin_ev_archive_yes_{scen.id}")
    answer_methods = [m for m in methods if isinstance(m, AnswerCallbackQuery)]
    assert any("Мероприятие архивировано" in (m.text or "") for m in answer_methods)
    summary = await get_scenario_summary(db_session_maker, scen.id)
    assert summary.is_archived is True


def test_scenario_card_text_with_and_without_bot_username():
    from event_admin_telegram import _format_scenario_card_text
    from event_admin_service import ScenarioSummary

    scen = ScenarioSummary(
        id=1,
        code="webinar_ai",
        event_type="webinar",
        title="Webinar AI",
        starts_at=datetime.utcnow() + timedelta(days=2),
        starts_at_msk="15.10.2026 19:00 (МСК)",
        duration_minutes=60,
        place=None,
        link="https://example.com/join",
        is_active=True,
        is_archived=False,
        schedule_revision=1,
        content_version=1,
        registrations_count=10,
    )

    # 1. Without bot username: fallback to just code
    text_without = _format_scenario_card_text(scen, bot_username=None)
    assert "• <b>Код события (/start):</b> <code>webinar_ai</code>" in text_without
    assert "• <b>Ссылка для регистрации:</b>" not in text_without

    # 2. With bot username: displays full clickable deep link and code
    text_with = _format_scenario_card_text(scen, bot_username="psycho_bot")
    assert "• <b>Ссылка для регистрации:</b> <code>https://t.me/psycho_bot?start=webinar_ai</code>" in text_with
    assert "• <b>Код события (/start):</b> <code>webinar_ai</code>" in text_with

