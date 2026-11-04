import os
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

os.environ.setdefault("BOT_TOKEN", "test_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import select

from database import (
    Base,
    EventDelivery,
    EventRegistration,
    EventScenario,
    EventScenarioMedia,
    EventStep,
    User,
)
from max_messenger_bot.api import MaxApiClient
from max_messenger_bot.app import MaxBotApplication
from max_messenger_bot.models import IncomingCallback, Sender
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
async def test_max_pagination_callback_preserves_archive_filter(mock_client, monkeypatch):
    scenarios = [
        SimpleNamespace(
            id=index,
            title=f"Event {index}",
            starts_at_msk="01.01.2030 12:00",
            code=f"event_{index}",
            event_type="webinar",
            is_archived=index == 7,
            is_active=True,
            registrations_count=0,
        )
        for index in range(1, 8)
    ]
    monkeypatch.setattr(max_admin_events, "list_scenarios", AsyncMock(return_value=scenarios))

    await max_admin_events.show_scenarios(mock_client, chat_id=123, page=0, include_archived=True)

    buttons = mock_client.send_message.call_args.kwargs["attachments"][0]["payload"]["buttons"]
    callbacks = [button["payload"] for row in buttons for button in row if "payload" in button]
    assert "admin_ev_page_1_1" in callbacks


@pytest.mark.asyncio
async def test_max_description_and_media_edit_routes(db_session_maker, mock_client, states, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    from event_admin_service import create_scenario

    scen = await create_scenario(
        db_session_maker,
        code="max_media_edit",
        event_type="webinar",
        title="MAX edit routes",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/max-edit",
    )

    await max_admin_events.start_edit_description(mock_client, states, 123, 456, scen.id)
    await max_admin_events.receive_edit_description(mock_client, states, 123, 456, "<b>MAX формат</b>")
    await max_admin_events.save_description(mock_client, states, 123, 456)

    await max_admin_events.show_media_editor(mock_client, states, 123, 456, scen.id)
    await max_admin_events.start_add_media(mock_client, states, 123, 456, scen.id)
    await max_admin_events.receive_media(
        mock_client,
        states,
        123,
        456,
        media_type="image",
        media_token="max-photo",
    )
    await max_admin_events.save_media(mock_client, states, 123, 456, scen.id)

    async with db_session_maker() as session:
        saved = await session.get(EventScenario, scen.id)
        media = (await session.execute(select(EventScenarioMedia).where(EventScenarioMedia.event_id == scen.id))).scalars().all()
        assert saved.description == "<b>MAX формат</b>"
        assert [(item.file_type, item.file_id, item.max_token) for item in media] == [("photo", "max-photo", "max-photo")]


@pytest.mark.asyncio
async def test_max_description_and_media_buttons_are_displayed(db_session_maker, mock_client, states, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    from event_admin_service import create_scenario

    scen = await create_scenario(
        db_session_maker,
        code="max_button_contract",
        event_type="webinar",
        title="MAX button contract",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/max-buttons",
    )

    await max_admin_events.start_edit_description(mock_client, states, 123, 456, scen.id)
    description_buttons = mock_client.send_message.call_args.kwargs["attachments"][0]["payload"]["buttons"]
    description_payloads = [button["payload"] for row in description_buttons for button in row]
    assert description_payloads == [
        f"admin_ev_description_save_{scen.id}",
        f"admin_ev_description_delete_{scen.id}",
        f"admin_ev_edit_menu_{scen.id}",
    ]
    assert (await states.get(456)).state == "max_event_edit_description"

    mock_client.send_message.reset_mock()
    await max_admin_events.show_media_editor(mock_client, states, 123, 456, scen.id)
    media_buttons = mock_client.send_message.call_args.kwargs["attachments"][0]["payload"]["buttons"]
    media_payloads = [button["payload"] for row in media_buttons for button in row]
    assert media_payloads == [
        f"admin_ev_media_add_{scen.id}",
        f"admin_ev_media_save_{scen.id}",
        f"admin_ev_edit_menu_{scen.id}",
    ]

    mock_client.send_message.reset_mock()
    await max_admin_events.start_add_media(mock_client, states, 123, 456, scen.id)
    assert (await states.get(456)).state == "max_event_edit_media_add"
    add_buttons = mock_client.send_message.call_args.kwargs["attachments"][0]["payload"]["buttons"]
    assert add_buttons[0][0]["payload"] == f"admin_ev_edit_media_{scen.id}"


@pytest.mark.asyncio
async def test_max_media_back_preserves_draft_and_cancel_state_is_clear(db_session_maker, mock_client, states, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    from event_admin_service import create_scenario

    scen = await create_scenario(
        db_session_maker,
        code="max_media_back",
        event_type="webinar",
        title="MAX media back",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/max-media-back",
    )

    await max_admin_events.show_media_editor(mock_client, states, 123, 456, scen.id)
    await max_admin_events.start_add_media(mock_client, states, 123, 456, scen.id)
    await max_admin_events.receive_media(
        mock_client,
        states,
        123,
        456,
        media_type="image",
        media_token="draft-max-photo",
    )
    await max_admin_events.start_add_media(mock_client, states, 123, 456, scen.id)
    await max_admin_events.receive_media(
        mock_client,
        states,
        123,
        456,
        media_type="image",
        media_token="draft-max-photo-2",
    )
    snapshot = await states.get(456)
    assert snapshot.state == "max_event_edit_media"
    assert len(snapshot.data["media_draft"]) == 2

    await max_admin_events.start_add_media(mock_client, states, 123, 456, scen.id)
    await max_admin_events.show_media_editor(mock_client, states, 123, 456, scen.id)
    media_buttons = mock_client.send_message.call_args.kwargs["attachments"][0]["payload"]["buttons"]
    media_payloads = [button["payload"] for row in media_buttons for button in row]
    assert f"admin_ev_media_remove_{scen.id}_0" in media_payloads
    assert f"admin_ev_media_remove_{scen.id}_1" in media_payloads

    await states.clear(456)
    await max_admin_events.show_media_editor(mock_client, states, 123, 456, scen.id)
    snapshot = await states.get(456)
    assert snapshot.state == "max_event_edit_media"
    assert snapshot.data["media_draft"] == []


@pytest.mark.asyncio
async def test_max_media_cancel_route_clears_unsaved_draft(states, mock_client, monkeypatch):
    app = MaxBotApplication(mock_client)
    app.states = states
    monkeypatch.setattr("max_messenger_bot.app.common.is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr("max_messenger_bot.app.admin_events_service.show_edit_menu", AsyncMock())
    await states.set(456, 123, "max_event_edit_media", {"event_id": 77, "media_draft": [{"file_type": "photo"}], "media_draft_dirty": True})

    callback = IncomingCallback(
        raw={},
        callback_id="cancel-media",
        payload="admin_ev_edit_menu_77",
        chat_id=123,
        message_id=None,
        sender=Sender(user_id=456, username="admin", first_name="Admin", last_name=None),
    )
    await app._handle_callback_impl(callback)
    assert await states.get(456) is None


@pytest.mark.asyncio
async def test_max_paginated_edit_roundtrip_and_delete_stays_draft(db_session_maker, mock_client, states, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    from event_admin_service import create_scenario

    scen = await create_scenario(
        db_session_maker,
        code="max_paginated_edit",
        event_type="webinar",
        title="MAX paginated edit",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/max-paginated-edit",
        description="x" * 5000,
    )

    await max_admin_events.show_scenario(mock_client, 123, scen.id, back="admin_ev_page_2_1")
    card_buttons = mock_client.send_message.call_args.kwargs["attachments"][0]["payload"]["buttons"]
    assert any(
        button[0]["payload"] == f"admin_ev_edit_menu_{scen.id}_2_1"
        for button in card_buttons
        if button
    )

    await max_admin_events.show_edit_menu(mock_client, 123, scen.id, back="admin_ev_page_2_1")
    menu_buttons = mock_client.send_message.call_args.kwargs["attachments"][0]["payload"]["buttons"]
    assert menu_buttons[-1][0]["payload"] == f"admin_ev_open_{scen.id}_2_1"
    await states.set(456, 123, "max_event_edit_menu", {"event_id": scen.id, "event_list_back": "admin_ev_page_2_1"})

    await max_admin_events.start_edit_description(mock_client, states, 123, 456, scen.id)
    long_preview = mock_client.send_message.call_args.kwargs["text"]
    assert len(long_preview) < 4096
    await max_admin_events.delete_description(mock_client, states, 123, 456, scen.id)
    async with db_session_maker() as session:
        unchanged = await session.get(EventScenario, scen.id)
        assert unchanged.description == "x" * 5000

    await max_admin_events.save_description(mock_client, states, 123, 456)
    async with db_session_maker() as session:
        saved = await session.get(EventScenario, scen.id)
        assert saved.description is None
    saved_buttons = mock_client.send_message.call_args.kwargs["attachments"][0]["payload"]["buttons"]
    assert saved_buttons[0][0]["payload"] == f"admin_ev_open_{scen.id}_2_1"


@pytest.mark.asyncio
async def test_max_callback_open_and_edit_preserve_page_context(mock_client, states, monkeypatch):
    app = MaxBotApplication(mock_client)
    app.states = states
    monkeypatch.setattr("max_messenger_bot.app.common.is_admin", AsyncMock(return_value=True))
    show_scenario = AsyncMock()
    show_edit_menu = AsyncMock()
    monkeypatch.setattr("max_messenger_bot.app.admin_events_service.show_scenario", show_scenario)
    monkeypatch.setattr("max_messenger_bot.app.admin_events_service.show_edit_menu", show_edit_menu)

    callback = IncomingCallback(
        raw={}, callback_id="open-page", payload="admin_ev_open_77_2_1", chat_id=123, message_id=None,
        sender=Sender(user_id=456, username="admin", first_name="Admin", last_name=None),
    )
    await app._handle_callback_impl(callback)
    show_scenario.assert_awaited_once_with(mock_client, 123, 77, back="admin_ev_page_2_1")

    callback.payload = "admin_ev_edit_menu_77_2_1"
    await app._handle_callback_impl(callback)
    show_edit_menu.assert_awaited_once_with(mock_client, 123, 77, back="admin_ev_page_2_1")
    snapshot = await states.get(456)
    assert snapshot.state == "max_event_edit_menu"
    assert snapshot.data["event_list_back"] == "admin_ev_page_2_1"

    show_edit_menu.reset_mock()
    await states.set(456, 123, "max_event_edit_media", {"event_id": 77, "event_list_back": "admin_ev_page_2_1"})
    callback.payload = "admin_ev_edit_menu_77"
    await app._handle_callback_impl(callback)
    show_edit_menu.assert_awaited_once_with(mock_client, 123, 77, back="admin_ev_page_2_1")


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


@pytest.mark.asyncio
async def test_max_scenario_card_registration_link(db_session_maker, mock_client, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    from event_admin_service import create_scenario

    scen = await create_scenario(
        db_session_maker,
        code="max_link_test",
        event_type="webinar",
        title="MAX Link Test Webinar",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.com/webinar",
    )

    # 1. Without bot_name attribute on client -> falls back to code
    mock_client.bot_name = None
    mock_client.send_message.reset_mock()
    await max_admin_events.show_scenario(mock_client, chat_id=123, event_id=scen.id)
    text = mock_client.send_message.call_args.kwargs["text"]
    assert "• <b>Код события:</b> <code>max_link_test</code>" in text
    assert "• <b>Ссылка для регистрации:</b>" not in text

    # 2. With bot_name on client -> includes registration link
    mock_client.bot_name = "test_bot"
    mock_client.send_message.reset_mock()
    await max_admin_events.show_scenario(mock_client, chat_id=123, event_id=scen.id)
    text = mock_client.send_message.call_args.kwargs["text"]
    assert "• <b>Ссылка для регистрации:</b> <code>https://max.ru/test_bot?start=max_link_test</code>" in text
    assert "• <b>Код события:</b> <code>max_link_test</code>" in text
