import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

os.environ.setdefault("BOT_TOKEN", "test_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from database import (
    Base,
    EventDelivery,
    EventRegistration,
    EventScenario,
    EventStep,
    User,
)
from max_messenger_bot.api import MaxApiClient
from max_messenger_bot.services import admin_events as max_admin_events
from max_messenger_bot.storage import StateStore
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
def mock_client():
    client = AsyncMock(spec=MaxApiClient)
    client.send_message = AsyncMock(return_value={"message_id": "max_msg_100"})
    return client


class MemoryStateStore:
    def __init__(self):
        self._store = {}

    async def get(self, user_id: int):
        val = self._store.get(user_id)
        if val is None:
            return None
        return MagicMock(state=val["state"], data=val.get("data", {}))

    async def set(self, user_id: int, chat_id: int, state: str, data: dict = None):
        self._store[user_id] = {"state": state, "data": data or {}}

    async def update(self, user_id: int, **kwargs):
        if user_id in self._store:
            self._store[user_id]["data"].update(kwargs)

    async def clear(self, user_id: int):
        self._store.pop(user_id, None)


@pytest.fixture
def states():
    return MemoryStateStore()


@pytest.mark.asyncio
async def test_max_admin_list_and_card(db_session_maker, mock_client, states, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)

    from event_admin_service import create_scenario
    scen = await create_scenario(
        db_session_maker,
        code="max_demo_webinar",
        event_type="webinar",
        title="MAX Demo Webinar",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://max.webinar.com",
    )

    # 1. Show scenarios list
    await max_admin_events.show_scenarios(mock_client, chat_id=123, page=0)
    mock_client.send_message.assert_called_once()
    text = mock_client.send_message.call_args.kwargs["text"]
    assert "MAX Demo Webinar" in text
    assert "max_demo_webinar" in text

    # 2. Show scenario card
    mock_client.send_message.reset_mock()
    await max_admin_events.show_scenario(mock_client, chat_id=123, event_id=scen.id)
    text = mock_client.send_message.call_args.kwargs["text"]
    assert "Карточка мероприятия" in text
    assert "MAX Demo Webinar" in text

    # 3. Toggle active
    mock_client.send_message.reset_mock()
    await max_admin_events.toggle_scenario(mock_client, chat_id=123, event_id=scen.id)
    # Sends feedback message and updates card
    assert mock_client.send_message.call_count == 2
    assert "Регистрация включена" in mock_client.send_message.call_args_list[0].kwargs["text"]

    # 4. Archive confirmation
    mock_client.send_message.reset_mock()
    await max_admin_events.archive_scenario_ask(mock_client, chat_id=123, event_id=scen.id)
    assert "Архивировать мероприятие" in mock_client.send_message.call_args.kwargs["text"]

    await max_admin_events.archive_scenario_confirm(mock_client, chat_id=123, event_id=scen.id)
    assert "Мероприятие архивировано" in mock_client.send_message.call_args_list[-2].kwargs["text"]


@pytest.mark.asyncio
async def test_max_webinar_creation_flow(db_session_maker, mock_client, states, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)

    # Step 0: Start create
    await max_admin_events.start_scenario_create(mock_client, states, chat_id=123, user_id=456, event_type="webinar")
    st = await states.get(456)
    assert st.state == "max_event_create_code"

    # Step 1: Code
    await max_admin_events.receive_create_code(mock_client, states, chat_id=123, user_id=456, raw_code="max_ai_conf")
    st = await states.get(456)
    assert st.state == "max_event_create_title"

    # Step 2: Title
    await max_admin_events.receive_create_title(mock_client, states, chat_id=123, user_id=456, raw_title="MAX AI Summit")
    st = await states.get(456)
    assert st.state == "max_event_create_starts_at"

    # Step 3: Starts at MSK
    await max_admin_events.receive_create_starts_at(mock_client, states, chat_id=123, user_id=456, raw_dt="05.11.2026 17:00")
    st = await states.get(456)
    assert st.state == "max_event_create_duration"

    # Step 4: Duration
    await max_admin_events.receive_create_duration(mock_client, states, chat_id=123, user_id=456, raw_dur="120")
    st = await states.get(456)
    assert st.state == "max_event_create_link"

    # Step 5: Link
    mock_client.send_message.reset_mock()
    await max_admin_events.receive_create_link(mock_client, states, chat_id=123, user_id=456, raw_link="https://max.conf.ru")
    assert await states.get(456) is None
    # Verify created
    assert any("Мероприятие успешно создано" in call.kwargs["text"] for call in mock_client.send_message.call_args_list)


@pytest.mark.asyncio
async def test_max_meeting_creation_flow(db_session_maker, mock_client, states, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)

    await max_admin_events.start_scenario_create(mock_client, states, chat_id=123, user_id=456, event_type="meeting")
    await max_admin_events.receive_create_code(mock_client, states, chat_id=123, user_id=456, raw_code="spb_offline")
    await max_admin_events.receive_create_title(mock_client, states, chat_id=123, user_id=456, raw_title="SPB Live Meetup")
    await max_admin_events.receive_create_starts_at(mock_client, states, chat_id=123, user_id=456, raw_dt="08.11.2026 19:00")
    await max_admin_events.receive_create_duration(mock_client, states, chat_id=123, user_id=456, raw_dur="180")

    st = await states.get(456)
    assert st.state == "max_event_create_place"

    await max_admin_events.receive_create_place(mock_client, states, chat_id=123, user_id=456, raw_place="Санкт-Петербург, Невский пр. 28")
    st = await states.get(456)
    assert st.state == "max_event_create_meeting_link"

    mock_client.send_message.reset_mock()
    await max_admin_events.skip_meeting_link(mock_client, states, chat_id=123, user_id=456)
    assert await states.get(456) is None
    assert any("Мероприятие успешно создано" in call.kwargs["text"] for call in mock_client.send_message.call_args_list)


@pytest.mark.asyncio
async def test_max_reschedule_and_copy_flow(db_session_maker, mock_client, states, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)

    from event_admin_service import create_scenario
    scen = await create_scenario(
        db_session_maker,
        code="max_resched_test",
        event_type="webinar",
        title="MAX Resched Test",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://webinar.com",
    )

    # 1. Reschedule
    await max_admin_events.start_reschedule(mock_client, states, chat_id=123, user_id=456, event_id=scen.id)
    assert (await states.get(456)).state == "max_event_reschedule_starts_at"

    mock_client.send_message.reset_mock()
    await max_admin_events.receive_reschedule(mock_client, states, chat_id=123, user_id=456, raw_dt="12.11.2026 20:00")
    assert await states.get(456) is None
    assert any("Мероприятие перенесено" in call.kwargs["text"] for call in mock_client.send_message.call_args_list)

    # 2. Copy
    await max_admin_events.start_copy(mock_client, states, chat_id=123, user_id=456, source_id=scen.id)
    assert (await states.get(456)).state == "max_event_copy_code"

    await max_admin_events.receive_copy_code(mock_client, states, chat_id=123, user_id=456, raw_code="max_resched_cloned")
    assert (await states.get(456)).state == "max_event_copy_starts_at"

    mock_client.send_message.reset_mock()
    await max_admin_events.receive_copy_starts_at(mock_client, states, chat_id=123, user_id=456, raw_dt="15.11.2026 19:00")
    assert await states.get(456) is None
    assert any("Сценарий успешно скопирован" in call.kwargs["text"] for call in mock_client.send_message.call_args_list)


@pytest.mark.asyncio
async def test_max_steps_management(db_session_maker, mock_client, states, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)

    from event_admin_service import create_scenario
    scen = await create_scenario(
        db_session_maker,
        code="max_steps_event",
        event_type="webinar",
        title="MAX Steps Event",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://link.com",
    )

    # 1. View steps
    await max_admin_events.show_steps(mock_client, chat_id=123, event_id=scen.id)
    assert "Цепочка напоминаний" in mock_client.send_message.call_args.kwargs["text"]

    # 2. Add step with preset
    await max_admin_events.start_step_add(mock_client, states, chat_id=123, user_id=456, event_id=scen.id)
    await max_admin_events.select_step_offset_preset(mock_client, states, chat_id=123, user_id=456, event_id=scen.id, offset_val=-1440)
    assert (await states.get(456)).state == "max_event_step_text"

    mock_client.send_message.reset_mock()
    await max_admin_events.receive_step_text(mock_client, states, chat_id=123, user_id=456, raw_text="Напоминаем: вебинар завтра в {event_time}! {link}")
    assert await states.get(456) is None
    assert any("Шаг успешно добавлен" in call.kwargs["text"] for call in mock_client.send_message.call_args_list)


@pytest.mark.asyncio
async def test_max_stats_and_resolution(db_session_maker, mock_client, states, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)

    from event_admin_service import create_scenario
    scen = await create_scenario(
        db_session_maker,
        code="max_mon_event",
        event_type="meeting",
        title="MAX Mon Event",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        place="Office 1",
    )

    async with db_session_maker() as session:
        u = User(id=777, first_name="MaxUser")
        session.add(u)
        await session.flush()
        reg = EventRegistration(event_id=scen.id, user_id=777, status="started")
        session.add(reg)
        await session.flush()
        deliv = EventDelivery(
            event_id=scen.id,
            user_id=777,
            registration_id=reg.id,
            message_kind="immediate_confirmation",
            delivery_key="max_unc_key",
            due_at=utc_now(),
            next_attempt_at=utc_now(),
            platform="max",
            status="uncertain",
            last_error="Gateway 504 Gateway Timeout",
        )
        session.add(deliv)
        await session.commit()
        deliv_id = deliv.id

    # 1. Show stats
    await max_admin_events.show_stats(mock_client, chat_id=123, event_id=scen.id)
    assert "Мониторинг доставок" in mock_client.send_message.call_args.kwargs["text"]
    assert "uncertain (неопределенно): 1" in mock_client.send_message.call_args.kwargs["text"]

    # 2. Show problems
    await max_admin_events.show_problems(mock_client, chat_id=123, event_id=scen.id, page=0)
    assert "Проблемные доставки (1)" in mock_client.send_message.call_args.kwargs["text"]

    # 3. Resolve start
    await max_admin_events.start_resolve(mock_client, states, chat_id=123, user_id=456, delivery_id=deliv_id, action="sent")
    assert (await states.get(456)).state == "max_event_resolve_reason"

    # 4. Resolve finish
    mock_client.send_message.reset_mock()
    await max_admin_events.receive_resolve_reason(mock_client, states, chat_id=123, user_id=456, raw_reason="Verified via MAX admin dashboard")
    assert await states.get(456) is None
    assert any("успешно разрешена" in call.kwargs["text"] for call in mock_client.send_message.call_args_list)


@pytest.mark.asyncio
async def test_max_test_send_to_self(db_session_maker, mock_client, states, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)

    from event_admin_service import add_step, create_scenario
    scen = await create_scenario(
        db_session_maker,
        code="max_test_send_event",
        event_type="meeting",
        title="MAX Test Send Meet",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        place="Conference Room B",
    )
    step = await add_step(db_session_maker, scen.id, 0, "Начинаем прямо сейчас: {event_title}! Ждём в {place}")

    await max_admin_events.do_test_send(mock_client, chat_id=123, user_id=456, event_id=scen.id, step_id=step.id)
    mock_client.send_message.assert_called_once()
    text = mock_client.send_message.call_args.kwargs["text"]
    assert "ТЕСТОВОЕ СООБЩЕНИЕ MAX" in text
    assert "MAX Test Send Meet" in text
    assert "Conference Room B" in text
