import io
import os
import re
import json
import asyncio
import html
from functools import wraps
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
    EventDeliveryPart,
    EventRegistration,
    EventScenario,
    EventScenarioMedia,
    EventStep,
    User,
)
from max_messenger_bot.api import MaxApiClient
from max_messenger_bot.app import MaxBotApplication
from max_messenger_bot.formatting import MAX_MESSAGE_TEXT_LEN
from max_messenger_bot.models import MAX_ID_OFFSET, IncomingCallback, IncomingMessage, Sender
from max_messenger_bot.services import admin_events as max_admin_events
from max_messenger_bot.storage import StateStore
from time_helpers import utc_now
from event_admin_service import DeliveryItem, DeliveryPartItem, DeliveryStats


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


class _RecordedMaxResponse:
    status = 200
    content_type = "application/json"

    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def text(self):
        return json.dumps(self.payload, ensure_ascii=False)

    async def json(self):
        return self.payload


class _RecordedMaxSession:
    def __init__(self):
        self.requests = []

    def request(self, method, url, **kwargs):
        body = kwargs.get("json")
        if body is not None:
            body = json.loads(json.dumps(body, ensure_ascii=False))
        self.requests.append({
            "method": method,
            "url": url,
            "params": dict(kwargs.get("params") or {}),
            "body": body,
        })
        return _RecordedMaxResponse({
            "message": {"body": {"mid": f"max-runtime-{len(self.requests)}"}},
        })


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


def _max_event_payloads(mock_client) -> list[str]:
    payloads = []
    for call in mock_client.send_message.call_args_list:
        for attachment in call.kwargs.get("attachments") or []:
            if attachment.get("type") != "inline_keyboard":
                continue
            for row in attachment["payload"]["buttons"]:
                payloads.extend(button["payload"] for button in row if "payload" in button)
    return payloads


_MAX_EVENT_BUTTON_CONTRACTS = (
    (re.compile(r"admin_events$"), "navigation", "events/list", "max.events.root", "show_scenarios", None),
    (re.compile(r"admin_panel$"), "navigation", "admin/panel", "max.events.root", "show_admin_panel", None),
    (re.compile(r"admin_ev_create_(?:webinar|meeting)$"), "navigation", "events/create", "max.events.root", "start_scenario_create", "max_event_create_code"),
    (re.compile(r"admin_ev_skip_meeting_link$"), "mutation", "events/create", "max.events.root", "skip_meeting_link", None),
    (re.compile(r"admin_ev_page_\d+_[01]$"), "navigation", "events/list", "max.events.pagination", "show_scenarios", None),
    (re.compile(r"admin_ev_toggle_archive_view_[01]$"), "navigation", "events/list", "max.events.pagination", "show_scenarios", None),
    (re.compile(r"admin_ev_open_\d+(?:_\d+_[01])?$"), "navigation", "events/card", "max.events.card", "show_scenario", "max_event_card"),
    (re.compile(r"admin_ev_toggle_media_order_\d+$"), "mutation", "events/edit", "max.description-media", "toggle_media_order", None),
    (re.compile(r"admin_ev_toggle_\d+$"), "mutation", "events/card", "max.events.card", "toggle_scenario", None),
    (re.compile(r"admin_ev_stats_\d+$"), "navigation", "events/stats", "max.delivery.lifecycle", "show_stats", None),
    (re.compile(r"admin_ev_reschedule_\d+$"), "mutation", "events/reschedule", "max.events.card", "start_reschedule", "max_event_reschedule_starts_at"),
    (re.compile(r"admin_ev_edit_menu_\d+_\d+_[01]$"), "navigation", "events/edit", "max.events.edit", "show_edit_menu", "max_event_edit_menu"),
    (re.compile(r"admin_ev_edit_menu_\d+$"), "navigation", "events/edit", "max.events.edit", "show_edit_menu", None),
    (re.compile(r"admin_ev_copy_\d+$"), "navigation", "events/copy", "max.events.card", "start_copy", "max_event_copy_code"),
    (re.compile(r"admin_ev_test_send_menu_\d+$"), "navigation", "events/test-send", "max.delivery.test-send", "show_test_send_menu", None),
    (re.compile(r"admin_ev_archive_ask_\d+$"), "navigation", "events/archive", "max.delivery.lifecycle", "archive_scenario_ask", None),
    (re.compile(r"admin_ev_archive_yes_\d+$"), "destructive", "events/archive", "max.delivery.lifecycle", "archive_scenario_confirm", None),
    (re.compile(r"admin_ev_edit_title_\d+$"), "mutation", "events/edit", "max.events.edit", "start_edit_title", "max_event_edit_title"),
    (re.compile(r"admin_ev_edit_duration_\d+$"), "mutation", "events/edit", "max.events.edit", "start_edit_duration", "max_event_edit_duration"),
    (re.compile(r"admin_ev_edit_link_\d+$"), "mutation", "events/edit", "max.events.edit", "start_edit_link", "max_event_edit_link"),
    (re.compile(r"admin_ev_edit_place_\d+$"), "mutation", "events/edit", "max.events.edit", "start_edit_place", "max_event_edit_place"),
    (re.compile(r"admin_ev_format_menu_\d+$"), "navigation", "events/edit-format", "max.events.edit", "show_format_menu", None),
    (re.compile(r"admin_ev_set_fmt_\d+_[a-z_]+$"), "mutation", "events/edit-format", "max.events.edit", "set_format_handler", None),
    (re.compile(r"admin_ev_preview_tg_\d+$"), "navigation", "events/preview-tg", "max.events.edit", "show_telegram_preview", None),
    (re.compile(r"admin_ev_preview_max_\d+$"), "navigation", "events/preview-max", "max.events.edit", "show_max_preview", None),
    (re.compile(r"admin_ev_edit_description_\d+$"), "mutation", "events/edit", "max.events.edit", "start_edit_description", "max_event_edit_description"),
    (re.compile(r"admin_ev_edit_media_\d+$"), "navigation", "events/edit-media", "max.description-media", "show_media_editor", "max_event_edit_media"),
    (re.compile(r"admin_ev_description_save_\d+$"), "mutation", "events/edit", "max.description-media", "save_description", None),
    (re.compile(r"admin_ev_cancel_draft_\d+$"), "destructive", "events/edit", "max.events.edit", "cancel_draft", None),
    (re.compile(r"admin_ev_description_delete_\d+$"), "destructive", "events/edit", "max.description-media", "delete_description", "max_event_edit_description"),
    (re.compile(r"admin_ev_media_add_\d+$"), "navigation", "events/edit-media", "max.description-media", "start_add_media", "max_event_edit_media_add"),
    (re.compile(r"admin_ev_media_save_\d+$"), "mutation", "events/edit-media", "max.description-media", "save_media", None),
    (re.compile(r"admin_ev_media_remove_\d+_\d+$"), "destructive", "events/edit-media", "max.description-media", "mutate_media_draft", "max_event_edit_media"),
    (re.compile(r"admin_ev_media_up_\d+_\d+$"), "mutation", "events/edit-media", "max.description-media", "mutate_media_draft", "max_event_edit_media"),
    (re.compile(r"admin_ev_media_down_\d+_\d+$"), "mutation", "events/edit-media", "max.description-media", "mutate_media_draft", "max_event_edit_media"),
    (re.compile(r"admin_ev_steps_\d+(?:_\d+_[01])?$"), "navigation", "events/steps", "max.steps.roundtrip", "show_steps", "max_event_steps_menu"),
    (re.compile(r"admin_ev_step_add_\d+(?:_\d+_[01])?$"), "navigation", "events/steps", "max.steps.roundtrip", "start_step_add", "max_event_step_offset"),
    (re.compile(r"admin_ev_offset_preset_\d+_-?\d+$"), "mutation", "events/steps", "max.steps.roundtrip", "select_step_offset_preset", "max_event_step_text"),
    (re.compile(r"admin_ev_offset_custom_\d+$"), "navigation", "events/steps", "max.steps.roundtrip", "start_step_offset_custom", "max_event_step_offset_custom"),
    (re.compile(r"admin_ev_step_toggle_details_\d+(?:_\d+_[01])?$"), "mutation", "events/step", "max.steps.dispatcher", "toggle_step_event_details", None),
    (re.compile(r"admin_ev_step_edit_text_\d+(?:_\d+_[01])?$"), "mutation", "events/step", "max.steps.dispatcher", "start_edit_step_text", "max_event_step_edit_text"),
    (re.compile(r"admin_ev_step_edit_offset_\d+(?:_\d+_[01])?$"), "navigation", "events/step", "max.steps.dispatcher", "start_edit_step_offset", "max_event_step_edit_offset"),
    (re.compile(r"admin_ev_step_up_\d+(?:_\d+_[01])?$"), "mutation", "events/step", "max.steps.dispatcher", "step_move_up", None),
    (re.compile(r"admin_ev_step_down_\d+(?:_\d+_[01])?$"), "mutation", "events/step", "max.steps.dispatcher", "step_move_down", None),
    (re.compile(r"admin_ev_step_del_ask_\d+(?:_\d+_[01])?$"), "navigation", "events/step", "max.steps.dispatcher", "ask_delete_step", None),
    (re.compile(r"admin_ev_step_del_yes_\d+(?:_\d+_[01])?$"), "destructive", "events/step", "max.steps.dispatcher", "confirm_delete_step", None),
    (re.compile(r"admin_ev_step_edit_preset_\d+_-?\d+(?:_\d+_[01])?$"), "mutation", "events/step", "max.steps.dispatcher", "select_edit_step_offset_preset", None),
    (re.compile(r"admin_ev_step_edit_custom_\d+(?:_\d+_[01])?$"), "navigation", "events/step", "max.steps.dispatcher", "start_edit_step_offset_custom", "max_event_step_edit_offset_custom"),
    (re.compile(r"admin_ev_step_\d+(?:_\d+_[01])?$"), "navigation", "events/step", "max.steps.roundtrip", "show_step", None),
    (re.compile(r"admin_ev_problems_\d+_\d+$"), "navigation", "events/problems", "max.delivery.lifecycle", "show_problems", None),
    (re.compile(r"admin_ev_problem_item_\d+(?:_\d+)?$"), "navigation", "events/problem", "max.delivery.lifecycle", "show_problem_detail", None),
    (re.compile(r"admin_ev_resolve_(?:force|sent|skipped)_\d+(?:_\d+)?$"), "mutation", "events/problem", "max.delivery.resolution", "start_resolve", "max_event_resolve_reason"),
    (re.compile(r"admin_ev_test_step_\d+(?:_\d+_[01])?$"), "external", "max/test-send", "max.delivery.test-send", "do_test_send", None),
    (re.compile(r"admin_ev_do_test_send_\d+_\d+$"), "external", "max/test-send", "max.delivery.test-send", "do_test_send", None),
)

_MAX_EVENT_JOURNEY_COVERAGE = {
    "max.events.root",
    "max.events.pagination",
    "max.events.card",
    "max.events.edit",
    "max.description-media",
    "max.steps.roundtrip",
    "max.steps.dispatcher",
    "max.delivery.lifecycle",
    "max.delivery.resolution",
    "max.delivery.test-send",
}

_MAX_REAL_JOURNEY_TESTS = {
    "max.events.root": (
        "test_max_admin_list_and_card",
        "test_max_webinar_creation_flow",
        "test_max_meeting_creation_flow",
    ),
    "max.events.pagination": (
        "test_max_pagination_callback_preserves_archive_filter",
        "test_max_pagination_boundaries_for_scenarios_and_problems",
    ),
    "max.events.card": (
        "test_max_admin_list_and_card",
        "test_max_reschedule_and_copy_flow",
        "test_max_callback_open_and_edit_preserve_page_context",
    ),
    "max.events.edit": (
        "test_max_description_and_media_edit_routes",
        "test_max_paginated_edit_roundtrip_and_delete_stays_draft",
        "test_max_callback_open_and_edit_preserve_page_context",
    ),
    "max.description-media": (
        "test_max_description_and_media_edit_routes",
        "test_max_media_back_route_preserves_unsaved_draft",
        "test_max_paginated_edit_roundtrip_and_delete_stays_draft",
    ),
    "max.steps.roundtrip": (
        "test_max_dispatcher_steps_roundtrip_page_context",
        "test_max_steps_management",
    ),
    "max.steps.dispatcher": (
        "test_max_dispatcher_steps_roundtrip_page_context",
        "test_max_dispatcher_delete_last_step_returns_empty_page",
    ),
    "max.delivery.lifecycle": (
        "test_max_stats_and_resolution",
        "test_max_dispatcher_delete_last_step_returns_empty_page",
    ),
    "max.delivery.resolution": (
        "test_max_stats_and_resolution",
    ),
    "max.delivery.test-send": (
        "test_max_test_send_to_self",
        "test_max_test_send_uses_runtime_media_plan_without_delivery_rows",
        "test_max_test_send_materializes_telegram_media_without_delivery_rows",
    ),
}

_MAX_EXECUTED_JOURNEYS: set[str] = set()


@pytest.fixture(autouse=True)
def _record_max_real_journey(request):
    for journey_id, test_names in _MAX_REAL_JOURNEY_TESTS.items():
        if request.node.name in test_names:
            _MAX_EXECUTED_JOURNEYS.add(journey_id)
    yield


@pytest.fixture(scope="module", autouse=True)
def _assert_max_real_journey_coverage(request):
    yield
    required_tests = {name for names in _MAX_REAL_JOURNEY_TESTS.values() for name in names}
    selected_tests = {item.name for item in request.session.items}
    if required_tests <= selected_tests:
        assert _MAX_EXECUTED_JOURNEYS == _MAX_EVENT_JOURNEY_COVERAGE


def _max_event_callback_contract(payload: str) -> dict[str, str]:
    for pattern, classification, target, journey_id, route, expected_state in _MAX_EVENT_BUTTON_CONTRACTS:
        if pattern.fullmatch(payload):
            assert journey_id in _MAX_EVENT_JOURNEY_COVERAGE
            assert classification in {"navigation", "mutation", "destructive", "external"}
            return {
                "classification": classification,
                "target": target,
                "journey_id": journey_id,
                "route": route,
                "expected_state": expected_state,
            }
    raise AssertionError(f"Uncontracted event payload: {payload}")


def _assert_max_event_callback_contract(payloads: list[str]) -> None:
    for payload in payloads:
        _max_event_callback_contract(payload)


@pytest.mark.ui_journey
@pytest.mark.asyncio
async def test_max_event_visible_button_contract_covers_changed_screens(
    db_session_maker, mock_client, states, monkeypatch
):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    from event_admin_service import add_step, create_scenario

    scen = await create_scenario(
        db_session_maker,
        code="max_visible_button_contract",
        event_type="meeting",
        title="MAX visible buttons",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        place="Room 1",
    )
    step = await add_step(db_session_maker, scen.id, -60, "Reminder", include_event_details=True)
    mock_client.send_message.reset_mock()
    await max_admin_events.show_scenarios(mock_client, chat_id=123, page=0, include_archived=True)
    await max_admin_events.show_scenario(mock_client, chat_id=123, event_id=scen.id, back="admin_ev_page_1_1")
    await max_admin_events.show_edit_menu(mock_client, chat_id=123, event_id=scen.id, back="admin_ev_page_1_1")
    await max_admin_events.show_media_editor(mock_client, states, 123, 456, scen.id)
    await max_admin_events.start_edit_description(mock_client, states, 123, 456, scen.id)
    await max_admin_events.show_steps(mock_client, chat_id=123, event_id=scen.id, back="admin_ev_page_1_1")
    await max_admin_events.show_step(mock_client, chat_id=123, step_id=step.id, back="admin_ev_page_1_1")
    await max_admin_events.show_test_send_menu(mock_client, chat_id=123, event_id=scen.id, back="admin_ev_page_1_1")
    monkeypatch.setattr(
        max_admin_events,
        "get_scenario_delivery_stats",
        AsyncMock(return_value=DeliveryStats(uncertain=1)),
    )
    await max_admin_events.show_stats(mock_client, chat_id=123, event_id=scen.id)
    problem = DeliveryItem(
        id=900,
        event_id=scen.id,
        user_id=901,
        user_display="Problem user",
        message_kind="immediate_confirmation",
        step_id=None,
        status="uncertain",
        due_at_msk="01.01.2030 12:00",
        attempt_count=1,
        last_error="timeout",
        sent_or_finished_msk=None,
        platform="max",
        retry_index=0,
        parent_delivery_id=None,
    )
    monkeypatch.setattr(max_admin_events, "list_problem_deliveries", AsyncMock(return_value=([problem], 1)))
    await max_admin_events.show_problems(mock_client, chat_id=123, event_id=scen.id)

    payloads = _max_event_payloads(mock_client)
    _assert_max_event_callback_contract(payloads)
    assert f"admin_ev_edit_description_{scen.id}" in payloads
    assert f"admin_ev_media_save_{scen.id}" in payloads
    assert f"admin_ev_step_edit_text_{step.id}_1_1" in payloads
    assert f"admin_ev_test_send_menu_{scen.id}" in payloads
    assert f"admin_ev_problems_{scen.id}_0" in payloads


@pytest.mark.ui_journey
@pytest.mark.asyncio
async def test_max_visible_event_buttons_dispatch_through_application_contract(db_session_maker, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    from event_admin_service import add_step, create_scenario

    scenario = await create_scenario(
        db_session_maker,
        code="max_dispatch_contract",
        event_type="meeting",
        title="MAX contract meeting",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        place="Room 1",
    )
    step = await add_step(db_session_maker, scenario.id, -60, "Reminder", include_event_details=True)
    delivery = DeliveryItem(
        id=30,
        event_id=scenario.id,
        user_id=40,
        user_display="User",
        message_kind="immediate_confirmation",
        step_id=None,
        status="uncertain",
        due_at_msk="01.01.2030 11:00",
        attempt_count=1,
        last_error="timeout",
        sent_or_finished_msk=None,
        platform="max",
        retry_index=0,
        parent_delivery_id=None,
    )
    part = DeliveryPartItem(
        id=31,
        part_index=0,
        part_kind="media",
        status="uncertain",
        attempt_count=1,
        last_error="timeout",
        external_message_id=None,
    )
    render_client = AsyncMock(spec=MaxApiClient)
    render_states = MemoryStateStore()
    await max_admin_events.show_scenarios(render_client, chat_id=123, page=0, include_archived=True)
    await max_admin_events.show_scenario(render_client, chat_id=123, event_id=scenario.id, back="admin_ev_page_1_1")
    await max_admin_events.show_edit_menu(render_client, chat_id=123, event_id=scenario.id, back="admin_ev_page_1_1")
    await max_admin_events.start_edit_description(render_client, render_states, 123, 456, scenario.id)
    await max_admin_events.receive_create_place(render_client, render_states, 123, 456, "Room 1")
    await render_states.set(
        456,
        123,
        "max_event_edit_media",
        {
            "event_id": scenario.id,
            "media_draft": [
                {"file_type": "photo", "file_id": "photo-1", "max_token": "photo-1", "source_platform": "max"},
                {"file_type": "video", "file_id": "video-1", "max_token": "video-1", "source_platform": "max"},
            ],
            "media_draft_dirty": True,
            "event_list_back": "admin_ev_page_1_1",
        },
    )
    await max_admin_events.show_media_editor(render_client, render_states, 123, 456, scenario.id)
    await max_admin_events.show_steps(render_client, chat_id=123, event_id=scenario.id, back="admin_ev_page_1_1")
    await max_admin_events.show_step(render_client, chat_id=123, step_id=step.id, back="admin_ev_page_1_1")
    await max_admin_events.start_step_add(render_client, render_states, 123, 456, scenario.id, back="admin_ev_page_1_1")
    await max_admin_events.start_edit_step_offset(render_client, render_states, 123, 456, step.id, back="admin_ev_page_1_1")
    await max_admin_events.show_test_send_menu(render_client, chat_id=123, event_id=scenario.id, back="admin_ev_page_1_1")
    monkeypatch.setattr(max_admin_events, "get_scenario_delivery_stats", AsyncMock(return_value=DeliveryStats(uncertain=1)))
    await max_admin_events.show_stats(render_client, chat_id=123, event_id=scenario.id)
    monkeypatch.setattr(max_admin_events, "list_problem_deliveries", AsyncMock(return_value=([delivery], 1)))
    await max_admin_events.show_problems(render_client, chat_id=123, event_id=scenario.id)
    monkeypatch.setattr(max_admin_events, "get_delivery_item", AsyncMock(return_value=delivery))
    monkeypatch.setattr(max_admin_events, "list_delivery_parts", AsyncMock(return_value=[part]))
    await max_admin_events.show_problem_detail(render_client, chat_id=123, delivery_id=delivery.id)
    payloads = _max_event_payloads(render_client)
    assert payloads
    contracts = {payload: _max_event_callback_contract(payload) for payload in payloads}

    import event_admin_service
    from max_messenger_bot import app as max_app_module

    client = AsyncMock(spec=MaxApiClient)
    states = MemoryStateStore()
    app = MaxBotApplication(client)
    app.states = states
    sender = Sender(user_id=456, username="admin", first_name="Admin", last_name=None)
    monkeypatch.setattr(max_app_module.common, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(
        event_admin_service,
        "get_step_item",
        AsyncMock(return_value=step),
    )

    service_names = {
        "show_scenarios", "show_scenario", "toggle_scenario", "toggle_media_order",
        "archive_scenario_ask", "archive_scenario_confirm", "start_scenario_create",
        "skip_meeting_link", "show_edit_menu", "start_edit_description", "save_description",
        "delete_description", "show_media_editor", "start_add_media", "mutate_media_draft",
        "save_media", "start_edit_title", "start_edit_duration", "start_edit_link",
        "start_edit_place", "start_reschedule", "start_copy", "show_steps", "start_step_add",
        "select_step_offset_preset", "start_step_offset_custom", "start_edit_step_text",
        "toggle_step_event_details", "start_edit_step_offset", "select_edit_step_offset_preset",
        "start_edit_step_offset_custom", "step_move_up", "step_move_down", "ask_delete_step",
        "confirm_delete_step", "show_step", "show_stats", "show_problems", "show_problem_detail",
        "start_resolve", "show_test_send_menu", "do_test_send",
        "show_format_menu", "set_format_handler", "show_telegram_preview", "show_max_preview",
    }
    routed: list[str] = []

    def make_probe(name: str):
        async def probe(client_arg, *args, **kwargs):
            routed.append(name)
            await client_arg.send_message(chat_id=kwargs.get("chat_id", 123), text=f"route:{name}")

        return probe

    for name in service_names:
        mock = AsyncMock(side_effect=make_probe(name))
        monkeypatch.setattr(max_admin_events, name, mock)
    admin_panel_mock = AsyncMock(side_effect=make_probe("show_admin_panel"))
    monkeypatch.setattr(max_app_module.admin_service, "show_admin_panel", admin_panel_mock)

    for payload in payloads:
        routed.clear()
        client.reset_mock()
        result = await app._handle_callback_impl(
            IncomingCallback(
                raw={},
                callback_id=f"contract-{payload}",
                payload=payload,
                chat_id=123,
                message_id=None,
                sender=sender,
            )
        )
        assert result is None
        assert routed == [contracts[payload]["route"]], f"Visible payload selected {routed}: {payload}"
        assert client.send_message.called, payload


@pytest.mark.ui_journey
@pytest.mark.asyncio
async def test_max_visible_event_buttons_execute_real_application_journeys(db_session_maker, monkeypatch):
    """Every changed visible payload executes its real MAX service route and state transition."""
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    from event_admin_service import add_step, create_scenario, get_scenario_summary, list_scenario_media, list_steps, replace_scenario_media

    scenario = await create_scenario(
        db_session_maker,
        code="max_real_contract",
        event_type="meeting",
        title="MAX real contract",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        place="Room 1",
    )
    first_step = await add_step(db_session_maker, scenario.id, -1440, "First reminder")
    second_step = await add_step(db_session_maker, scenario.id, -60, "Second reminder")
    await replace_scenario_media(
        db_session_maker,
        scenario.id,
        [
            {"file_type": "photo", "file_id": "max-photo", "max_token": "max-photo", "source_platform": "max"},
            {"file_type": "video", "file_id": "max-video", "max_token": "max-video", "source_platform": "max"},
        ],
    )
    async with db_session_maker() as session:
        user = User(id=40, first_name="Problem user")
        session.add(user)
        await session.flush()
        registration = EventRegistration(event_id=scenario.id, user_id=user.id, status="started")
        session.add(registration)
        await session.flush()
        delivery = EventDelivery(
            event_id=scenario.id,
            user_id=user.id,
            registration_id=registration.id,
            message_kind="immediate_confirmation",
            delivery_key="max_real_contract_delivery",
            due_at=utc_now(),
            next_attempt_at=utc_now(),
            platform="max",
            status="uncertain",
            attempt_count=1,
            last_error="timeout",
            content_version_snapshot=1,
        )
        session.add(delivery)
        await session.flush()
        part = EventDeliveryPart(
            delivery_id=delivery.id,
            part_index=0,
            part_kind="media",
            payload_json=json.dumps({"kind": "image", "token": "max-photo"}),
            content_version=1,
            status="uncertain",
            attempt_count=1,
            last_error="timeout",
        )
        session.add(part)
        await session.commit()
        delivery_id, part_id = delivery.id, part.id

    scenario = await get_scenario_summary(db_session_maker, scenario.id)
    assert scenario is not None
    steps = await list_steps(db_session_maker, scenario.id, include_archived=False)
    media = await list_scenario_media(db_session_maker, scenario.id)
    delivery_item = DeliveryItem(
        id=delivery_id,
        event_id=scenario.id,
        user_id=40,
        user_display="Problem user",
        message_kind="immediate_confirmation",
        step_id=None,
        status="uncertain",
        due_at_msk="01.01.2030 11:00",
        attempt_count=1,
        last_error="timeout",
        sent_or_finished_msk=None,
        platform="max",
        retry_index=0,
        parent_delivery_id=None,
    )
    part_item = DeliveryPartItem(
        id=part_id,
        part_index=0,
        part_kind="media",
        status="uncertain",
        attempt_count=1,
        last_error="timeout",
        external_message_id=None,
    )
    render_client = AsyncMock(spec=MaxApiClient)
    render_states = MemoryStateStore()
    await max_admin_events.show_scenarios(render_client, chat_id=123, page=0, include_archived=True)
    await max_admin_events.show_scenario(render_client, chat_id=123, event_id=scenario.id, back="admin_ev_page_1_1")
    await max_admin_events.show_edit_menu(render_client, chat_id=123, event_id=scenario.id, back="admin_ev_page_1_1")
    await max_admin_events.show_media_editor(render_client, render_states, 123, 456, scenario.id)
    await max_admin_events.show_steps(render_client, chat_id=123, event_id=scenario.id, back="admin_ev_page_1_1")
    await max_admin_events.show_step(render_client, chat_id=123, step_id=first_step.id, back="admin_ev_page_1_1")
    await max_admin_events.show_test_send_menu(render_client, chat_id=123, event_id=scenario.id, back="admin_ev_page_1_1")
    await max_admin_events.show_stats(render_client, chat_id=123, event_id=scenario.id)
    await max_admin_events.show_problems(render_client, chat_id=123, event_id=scenario.id)
    await max_admin_events.show_problem_detail(render_client, chat_id=123, delivery_id=delivery_id)
    payloads = list(dict.fromkeys(_max_event_payloads(render_client)))
    assert payloads
    contracts = {payload: _max_event_callback_contract(payload) for payload in payloads}

    from max_messenger_bot import app as max_app_module

    client = AsyncMock(spec=MaxApiClient)
    states = MemoryStateStore()
    app = MaxBotApplication(client)
    app.states = states
    sender = Sender(user_id=456, username="admin", first_name="Admin", last_name=None)
    monkeypatch.setattr(max_app_module.common, "is_admin", AsyncMock(return_value=True))
    monkeypatch.setattr(max_app_module, "async_session_maker", db_session_maker)
    routed: list[str] = []
    service_names = {
        "show_scenarios", "show_scenario", "toggle_scenario", "toggle_media_order",
        "archive_scenario_ask", "archive_scenario_confirm", "start_scenario_create",
        "skip_meeting_link", "show_edit_menu", "start_edit_description", "save_description",
        "delete_description", "show_media_editor", "start_add_media", "mutate_media_draft",
        "save_media", "start_edit_title", "start_edit_duration", "start_edit_link",
        "start_edit_place", "start_reschedule", "start_copy", "show_steps", "start_step_add",
        "select_step_offset_preset", "start_step_offset_custom", "start_edit_step_text",
        "toggle_step_event_details", "start_edit_step_offset", "select_edit_step_offset_preset",
        "start_edit_step_offset_custom", "step_move_up", "step_move_down", "ask_delete_step",
        "confirm_delete_step", "show_step", "show_stats", "show_problems", "show_problem_detail",
        "start_resolve", "show_test_send_menu", "do_test_send",
        "show_format_menu", "set_format_handler", "show_telegram_preview", "show_max_preview",
    }

    def wrap_service(name: str, original):
        @wraps(original)
        async def wrapped(*args, **kwargs):
            routed.append(name)
            return await original(*args, **kwargs)

        return wrapped

    for name in service_names:
        monkeypatch.setattr(max_admin_events, name, wrap_service(name, getattr(max_admin_events, name)))
    monkeypatch.setattr(
        max_app_module.admin_service,
        "show_admin_panel",
        wrap_service("show_admin_panel", max_app_module.admin_service.show_admin_panel),
    )

    async def prepare_state(payload: str) -> None:
        await states.clear(456)
        event_id = scenario.id
        if payload == "admin_ev_skip_meeting_link":
            await states.set(
                456,
                123,
                "max_event_create_meeting_link",
                {
                    "create_event_type": "meeting",
                    "create_code": "max_real_created",
                    "create_title": "Created from route",
                    "create_starts_at": (datetime.utcnow() + timedelta(days=3)).isoformat(),
                    "create_duration": 120,
                    "create_place": "Room 1",
                },
            )
        elif payload.startswith("admin_ev_description_save_"):
            await states.set(456, 123, "max_event_edit_description", {
                "event_id": event_id,
                "description_draft": "Updated description",
                "event_list_back": "admin_ev_page_1_1",
            })
        elif payload.startswith("admin_ev_media_"):
            await states.set(456, 123, "max_event_edit_media", {
                "event_id": event_id,
                "media_draft": [
                    {"file_type": "photo", "file_id": "max-photo", "max_token": "max-photo", "source_platform": "max"},
                    {"file_type": "video", "file_id": "max-video", "max_token": "max-video", "source_platform": "max"},
                ],
                "media_draft_dirty": True,
                "event_list_back": "admin_ev_page_1_1",
            })
        elif payload.startswith("admin_ev_offset_"):
            await states.set(456, 123, "max_event_step_offset", {"event_id": event_id, "event_list_back": "admin_ev_page_1_1"})
        elif payload.startswith("admin_ev_step_edit_preset_") or payload.startswith("admin_ev_step_edit_custom_"):
            await states.set(456, 123, "max_event_step_edit_offset", {"step_id": first_step.id, "event_list_back": "admin_ev_page_1_1"})

    try:
        for payload in payloads:
            await prepare_state(payload)
            routed.clear()
            client.reset_mock()
            result = await app._handle_callback_impl(
                IncomingCallback(
                    raw={},
                    callback_id=f"real-{payload}",
                    payload=payload,
                    chat_id=123,
                    message_id=None,
                    sender=sender,
                )
            )
            assert result is None
            assert routed and routed[0] == contracts[payload]["route"], f"Unexpected real service for {payload}: {routed}"
            assert client.send_message.called, payload
            expected_state = contracts[payload]["expected_state"]
            if expected_state is not None:
                snapshot = await states.get(456)
                assert snapshot is not None and snapshot.state == expected_state, payload
    finally:
        await states.clear(456)


@pytest.mark.asyncio
async def test_max_paginated_card_mutations_and_stats_return_to_original_list(
    db_session_maker, monkeypatch
):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    from event_admin_service import create_scenario
    from max_messenger_bot import app as max_app_module

    scenarios = []
    for index in range(max_admin_events.PAGE_SIZE + 1):
        scenarios.append(
            await create_scenario(
                db_session_maker,
                code=f"max_page_context_{index}",
                event_type="meeting",
                title=f"MAX page context {index}",
                starts_at_utc=datetime.utcnow() + timedelta(days=2, minutes=index),
                place="Room 1",
            )
        )
    target = scenarios[0]

    client = AsyncMock(spec=MaxApiClient)
    app = MaxBotApplication(client)
    app.states = MemoryStateStore()
    sender = Sender(user_id=456, username="admin", first_name="Admin", last_name=None)
    monkeypatch.setattr(max_app_module.common, "is_admin", AsyncMock(return_value=True))

    async def feed(payload: str):
        client.reset_mock()
        await app._handle_callback_impl(
            IncomingCallback(
                raw={},
                callback_id=f"context-{payload}",
                payload=payload,
                chat_id=123,
                message_id=None,
                sender=sender,
            )
        )

    def last_payloads() -> list[str]:
        return _max_event_payloads(client)

    try:
        await feed(f"admin_ev_open_{target.id}_1_1")
        assert f"admin_ev_page_1_1" in last_payloads()

        await feed(f"admin_ev_toggle_{target.id}")
        assert "admin_ev_page_1_1" in last_payloads()

        await feed(f"admin_ev_archive_ask_{target.id}")
        assert f"admin_ev_open_{target.id}_1_1" in last_payloads()
        assert f"admin_ev_archive_yes_{target.id}" in last_payloads()

        await feed(f"admin_ev_open_{target.id}_1_1")
        assert "admin_ev_page_1_1" in last_payloads()

        await feed(f"admin_ev_archive_ask_{target.id}")
        await feed(f"admin_ev_archive_yes_{target.id}")
        assert "admin_ev_page_1_1" in last_payloads()

        await feed("admin_ev_page_1_1")
        assert "Страница 2 из 2" in client.send_message.call_args.kwargs["text"]
        assert "admin_ev_toggle_archive_view_0" in last_payloads()

        await feed(f"admin_ev_stats_{target.id}")
        assert f"admin_ev_open_{target.id}_1_1" in last_payloads()
        await feed(f"admin_ev_open_{target.id}_1_1")
        assert "admin_ev_page_1_1" in last_payloads()
    finally:
        await app.states.clear(456)


@pytest.mark.asyncio
async def test_max_long_title_delivery_uses_serialized_runtime_request(db_session_maker):
    from event_admin_service import create_scenario
    from event_delivery_worker import build_test_delivery_plan, send_test_delivery_plan

    title = "<&>" + "Я" * 997
    scen = await create_scenario(
        db_session_maker,
        code="max_title_delivery_runtime",
        event_type="webinar",
        title=title,
        starts_at_utc=datetime.utcnow() + timedelta(days=5),
        link="https://example.test/max-title-runtime",
    )

    async with db_session_maker() as session:
        stored = await session.get(EventScenario, scen.id)
        assert stored is not None and stored.title == title

    plan = await build_test_delivery_plan(db_session_maker, scen.id, None, None)
    client = MaxApiClient(token="test-token", base_url="https://max.test")
    transport = _RecordedMaxSession()
    client._session = transport

    sent_ids = await send_test_delivery_plan(
        plan,
        platform="max",
        user_id=MAX_ID_OFFSET + 456,
        event_id=scen.id,
        max_client=client,
    )

    assert sent_ids == ["max-runtime-1"]
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request["method"] == "POST"
    assert request["url"] == "https://max.test/messages"
    assert request["params"]["user_id"] == 456
    body = request["body"]
    assert body["format"] == "html"
    assert html.escape(title, quote=False) in body["text"]
    assert title in html.unescape(body["text"])
    assert len(body["text"].encode("utf-16-le")) // 2 <= MAX_MESSAGE_TEXT_LEN
    assert json.loads(json.dumps(body, ensure_ascii=False)) == body


@pytest.mark.asyncio
async def test_max_long_title_navigation_preserves_page_and_archive_context(db_session_maker, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    from max_messenger_bot import app as max_app_module
    monkeypatch.setattr(max_app_module.common, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "is_admin", AsyncMock(return_value=True))
    from event_admin_service import archive_scenario, create_scenario

    original_title = "А" * 1000
    base_start = datetime.utcnow() + timedelta(days=5)
    target = await create_scenario(
        db_session_maker,
        code="max_title_navigation_target",
        event_type="webinar",
        title=original_title,
        starts_at_utc=base_start,
        link="https://example.test/max-navigation",
    )
    for index in range(6):
        filler = await create_scenario(
            db_session_maker,
            code=f"max_title_navigation_filler_{index}",
            event_type="webinar",
            title=f"Навигация {index}",
            starts_at_utc=base_start + timedelta(days=1),
            link=f"https://example.test/max-navigation/{index}",
        )
        if index == 0:
            await archive_scenario(db_session_maker, filler.id)

    transport = _RecordedMaxSession()
    client = MaxApiClient(token="test-token", base_url="https://max.test")
    client._session = transport
    app = MaxBotApplication(client)
    app.states = MemoryStateStore()
    sender = Sender(user_id=456, username="admin", first_name="Admin", last_name=None)

    async def press(payload):
        transport.requests.clear()
        await app.handle_callback(
            IncomingCallback(
                raw={},
                callback_id=f"navigation-{payload}",
                payload=payload,
                chat_id=123,
                message_id=None,
                sender=sender,
            )
        )
        assert transport.requests, payload
        return transport.requests[-1]["body"]

    async def send_input(text, suffix):
        transport.requests.clear()
        await app.handle_message(
            IncomingMessage(
                raw={},
                message_id=f"max-navigation-{suffix}",
                chat_id=123,
                sender=sender,
                text=text,
            )
        )
        assert transport.requests, suffix
        return transport.requests[-1]["body"]

    def button_payload(body, prefix):
        for attachment in body.get("attachments") or []:
            for row in attachment.get("payload", {}).get("buttons", []):
                for button in row:
                    payload = button.get("payload")
                    if payload and payload.startswith(prefix):
                        return payload
        raise AssertionError(f"No visible MAX callback with prefix {prefix!r}")

    try:
        listing = await press("admin_events")
        assert "Страница 1 из 1" in listing["text"]
        listing = await press(button_payload(listing, "admin_ev_toggle_archive_view_1"))
        assert "Страница 1 из 2" in listing["text"]

        listing = await press(button_payload(listing, "admin_ev_page_1_1"))
        assert "Страница 2 из 2" in listing["text"]
        target_callback = button_payload(listing, f"admin_ev_open_{target.id}_1_1")

        card = await press(target_callback)
        assert original_title in card["text"]
        edit_callback = button_payload(card, f"admin_ev_edit_menu_{target.id}_1_1")
        edit_menu = await press(edit_callback)
        title_callback = button_payload(edit_menu, f"admin_ev_edit_title_{target.id}")
        prompt = await press(title_callback)
        assert "1000 символов" in prompt["text"]
        assert (await app.states.get(456)).state == "max_event_edit_title"

        card = await press(button_payload(prompt, f"admin_ev_open_{target.id}_1_1"))
        assert original_title in card["text"]
        assert (await app.states.get(456)).state == "max_event_card"

        edit_menu = await press(button_payload(card, f"admin_ev_edit_menu_{target.id}_1_1"))
        prompt = await press(button_payload(edit_menu, f"admin_ev_edit_title_{target.id}"))
        invalid = await send_input("В" * 1001, "invalid")
        assert "1000" in invalid["text"]
        assert (await app.states.get(456)).state == "max_event_edit_title"
        async with db_session_maker() as session:
            assert (await session.get(EventScenario, target.id)).title == original_title

        updated_title = "Б" * 1000
        saved_card = await send_input(updated_title, "save")
        assert updated_title in saved_card["text"]
        assert await app.states.get(456) is None
        listing = await press(button_payload(saved_card, "admin_ev_page_1_1"))
        assert "Страница 2 из 2" in listing["text"]
        assert target.code in listing["text"]

        card = await press(button_payload(listing, f"admin_ev_open_{target.id}_1_1"))
        assert updated_title in card["text"]
        assert button_payload(card, "admin_ev_page_1_1") == "admin_ev_page_1_1"
        assert "Мероприятия и цепочки напоминаний" not in card["text"]
    finally:
        await app.states.clear(456)


@pytest.mark.ui_journey
@pytest.mark.asyncio
async def test_max_real_handle_update_paginated_format_save_cancel_journey(db_session_maker, monkeypatch):
    """MAX BotApplication ingress journey against the recorded HTTP JSON boundary."""
    from event_admin_service import create_scenario
    from max_messenger_bot import app as max_app_module

    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "is_admin", AsyncMock(return_value=True))

    base_start = datetime.utcnow() + timedelta(days=2)
    target = await create_scenario(
        db_session_maker,
        code="max_real_page_two_target",
        event_type="webinar",
        title="MAX page two target",
        starts_at_utc=base_start,
        link="https://example.test/max-page-two-target",
    )
    for index in range(max_admin_events.PAGE_SIZE):
        await create_scenario(
            db_session_maker,
            code=f"max_real_page_two_fill_{index}",
            event_type="webinar",
            title=f"MAX page two filler {index}",
            starts_at_utc=base_start + timedelta(days=index + 1),
            link=f"https://example.test/max-page-two/{index}",
        )

    transport = _RecordedMaxSession()
    client = MaxApiClient(token="test-token", base_url="https://max.test")
    client._session = transport
    app = MaxBotApplication(client)
    app.states = MemoryStateStore()
    update_counter = 0

    def message_body() -> dict:
        for request in reversed(transport.requests):
            if request["url"].endswith("/messages"):
                return request["body"]
        raise AssertionError("No MAX /messages request was recorded")

    def button_payload(body: dict, prefix: str) -> str:
        for attachment in body.get("attachments") or []:
            for row in attachment.get("payload", {}).get("buttons", []):
                for button in row:
                    payload = button.get("payload")
                    if payload and payload.startswith(prefix):
                        return payload
        raise AssertionError(f"No visible MAX callback with prefix {prefix!r}")

    async def press(payload: str) -> dict:
        nonlocal update_counter
        update_counter += 1
        transport.requests.clear()
        await app.handle_update(
            {
                "update_type": "message_callback",
                "update_id": f"max-journey-cb-{update_counter}",
                "callback": {
                    "callback_id": f"max-journey-callback-{update_counter}",
                    "payload": payload,
                    "user": {"user_id": 456, "first_name": "Admin", "username": "admin"},
                },
                "message": {
                    "mid": f"max-journey-message-{update_counter}",
                    "recipient": {"chat_id": 123},
                    "body": {"text": "admin"},
                },
            }
        )
        body = message_body()
        json.loads(json.dumps(body, ensure_ascii=False))
        return body

    async def send_text(text: str) -> dict:
        nonlocal update_counter
        update_counter += 1
        transport.requests.clear()
        await app.handle_update(
            {
                "update_type": "message_created",
                "update_id": f"max-journey-message-{update_counter}",
                "message": {
                    "mid": f"max-journey-incoming-{update_counter}",
                    "recipient": {"chat_id": 123},
                    "sender": {"user_id": 456, "first_name": "Admin", "username": "admin"},
                    "body": {"text": text},
                },
            }
        )
        body = message_body()
        json.loads(json.dumps(body, ensure_ascii=False))
        return body

    try:
        listing = await press("admin_events")
        assert "Страница 1 из 2" in listing["text"]
        listing = await press(button_payload(listing, "admin_ev_page_1_0"))
        assert "Страница 2 из 2" in listing["text"]
        target_callback = button_payload(listing, f"admin_ev_open_{target.id}_1_0")

        card = await press(target_callback)
        assert "MAX page two target" in card["text"]
        edit_menu = await press(button_payload(card, f"admin_ev_edit_menu_{target.id}_1_0"))
        format_menu = await press(button_payload(edit_menu, f"admin_ev_format_menu_{target.id}"))
        assert button_payload(format_menu, f"admin_ev_edit_menu_{target.id}_1_0")
        edit_menu = await press(button_payload(format_menu, f"admin_ev_edit_menu_{target.id}_1_0"))

        description_screen = await press(button_payload(edit_menu, f"admin_ev_edit_description_{target.id}"))
        draft_screen = await send_text("Описание для отмены")
        edit_menu = await press(button_payload(draft_screen, f"admin_ev_edit_menu_{target.id}_1_0"))
        card = await press(button_payload(edit_menu, f"admin_ev_cancel_draft_{target.id}"))
        assert "Карточка мероприятия" in card["text"]
        assert "Описание для отмены" not in card["text"]
        async with db_session_maker() as session:
            assert (await session.get(EventScenario, target.id)).description is None

        edit_menu = await press(button_payload(card, f"admin_ev_edit_menu_{target.id}_1_0"))
        description_screen = await press(button_payload(edit_menu, f"admin_ev_edit_description_{target.id}"))
        draft_screen = await send_text("Описание для сохранения")
        card = await press(button_payload(draft_screen, f"admin_ev_description_save_{target.id}"))
        assert "Описание для сохранения" in card["text"]
        async with db_session_maker() as session:
            assert (await session.get(EventScenario, target.id)).description == "Описание для сохранения"

        listing = await press(button_payload(card, "admin_ev_page_1_0"))
        assert "Страница 2 из 2" in listing["text"]
        assert "max_real_page_two_target" in listing["text"]
    finally:
        await app.states.clear(MAX_ID_OFFSET + 456)


@pytest.mark.asyncio
async def test_max_handle_update_stale_callbacks_preserve_foreign_draft_and_db(db_session_maker, monkeypatch):
    from event_admin_service import create_scenario
    from max_messenger_bot import app as max_app_module

    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "is_admin", AsyncMock(return_value=True))

    event_a = await create_scenario(
        db_session_maker,
        code="max_stale_event_a",
        event_type="webinar",
        title="MAX Event A",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/max-stale-a",
        description="A original",
    )
    event_b = await create_scenario(
        db_session_maker,
        code="max_stale_event_b",
        event_type="webinar",
        title="MAX Event B",
        starts_at_utc=datetime.utcnow() + timedelta(days=3),
        link="https://example.test/max-stale-b",
        description="B original",
    )
    client = MaxApiClient(token="test-token", base_url="https://max.test")
    transport = _RecordedMaxSession()
    client._session = transport
    app = MaxBotApplication(client)
    app.states = MemoryStateStore()
    await app.states.set(
        MAX_ID_OFFSET + 456,
        123,
        "max_event_edit_description",
        {
            "event_id": event_a.id,
            "description_draft": "A draft that must remain isolated",
            "event_list_back": "admin_ev_page_1_0",
        },
    )

    async def press(payload: str, update_number: int):
        transport.requests.clear()
        await app.handle_update(
            {
                "update_type": "message_callback",
                "update_id": f"max-stale-{update_number}",
                "callback": {
                    "callback_id": f"max-stale-callback-{update_number}",
                    "payload": payload,
                    "user": {"user_id": 456, "first_name": "Admin", "username": "admin"},
                },
                "message": {
                    "mid": f"max-stale-message-{update_number}",
                    "recipient": {"chat_id": 123},
                    "body": {"text": "admin"},
                },
            }
        )

    try:
        await press(f"admin_ev_description_save_{event_b.id}", 1)
        await press(f"admin_ev_cancel_draft_{event_b.id}", 2)
        await press(f"admin_ev_edit_menu_{event_b.id}_1_0", 3)
        snapshot = await app.states.get(MAX_ID_OFFSET + 456)
        assert snapshot is not None
        assert snapshot.data["event_id"] == event_a.id
        assert snapshot.data["description_draft"] == "A draft that must remain isolated"
        assert transport.requests
        assert any(request["url"].endswith("/answers") for request in transport.requests), transport.requests

        async with db_session_maker() as session:
            saved_a = await session.get(EventScenario, event_a.id)
            saved_b = await session.get(EventScenario, event_b.id)
            assert saved_a.description == "A original"
            assert saved_b.description == "B original"
    finally:
        await app.states.clear(MAX_ID_OFFSET + 456)


@pytest.mark.asyncio
async def test_max_concurrent_save_and_cancel_have_one_atomic_winner(db_session_maker, mock_client, monkeypatch):
    from event_admin_service import create_scenario

    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    scen = await create_scenario(
        db_session_maker,
        code="max_concurrent_draft",
        event_type="webinar",
        title="Concurrent draft",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/max-concurrent",
        description="Original",
    )
    states = MemoryStateStore()
    await states.set(
        456,
        123,
        "max_event_edit_description",
        {"event_id": scen.id, "description_draft": "Saved once", "event_list_back": "admin_events"},
    )

    await asyncio.gather(
        max_admin_events.save_description(mock_client, states, 123, 456, event_id=scen.id),
        max_admin_events.cancel_draft(mock_client, states, 123, 456, scen.id),
    )

    snapshot = await states.get(456)
    assert snapshot is None
    async with db_session_maker() as session:
        stored = await session.get(EventScenario, scen.id)
        assert stored.description in {"Original", "Saved once"}


@pytest.mark.asyncio
async def test_max_problem_delivery_page_roundtrip_uses_current_page(monkeypatch):
    selected = DeliveryItem(
        id=605,
        event_id=77,
        user_id=700,
        user_display="Problem page user",
        message_kind="immediate_confirmation",
        step_id=None,
        status="uncertain",
        due_at_msk="01.01.2030 11:00",
        attempt_count=1,
        last_error="timeout",
        sent_or_finished_msk=None,
        platform="max",
        retry_index=0,
        parent_delivery_id=None,
    )
    monkeypatch.setattr(max_admin_events, "list_problem_deliveries", AsyncMock(return_value=([selected], 6)))
    monkeypatch.setattr(max_admin_events, "get_delivery_item", AsyncMock(return_value=selected))
    monkeypatch.setattr(max_admin_events, "list_delivery_parts", AsyncMock(return_value=[]))

    from max_messenger_bot import app as max_app_module

    client = AsyncMock(spec=MaxApiClient)
    app = MaxBotApplication(client)
    app.states = MemoryStateStore()
    sender = Sender(user_id=456, username="admin", first_name="Admin", last_name=None)
    await app.states.set(456, 123, "max_event_card", {"event_id": 77, "event_list_back": "admin_ev_page_1_1"})
    monkeypatch.setattr(max_app_module.common, "is_admin", AsyncMock(return_value=True))

    async def feed(payload: str):
        client.reset_mock()
        await app._handle_callback_impl(
            IncomingCallback(
                raw={},
                callback_id=f"problem-{payload}",
                payload=payload,
                chat_id=123,
                message_id=None,
                sender=sender,
            )
        )

    try:
        await feed("admin_ev_problems_77_1")
        item_callback = next(
            payload for payload in _max_event_payloads(client)
            if payload.startswith("admin_ev_problem_item_")
        )
        assert item_callback == "admin_ev_problem_item_605_1"

        await feed(item_callback)
        assert "admin_ev_problems_77_1" in _max_event_payloads(client)

        resolve_callback = next(
            payload for payload in _max_event_payloads(client)
            if payload.startswith("admin_ev_resolve_force_605")
        )
        await feed(resolve_callback)
        assert "admin_ev_problem_item_605_1" in _max_event_payloads(client)
        snapshot = await app.states.get(456)
        assert snapshot is not None and snapshot.state == "max_event_resolve_reason"

        await feed("admin_ev_problem_item_605_1")
        assert "admin_ev_problems_77_1" in _max_event_payloads(client)
        snapshot = await app.states.get(456)
        assert snapshot is not None and snapshot.state == "max_event_card"

        await feed("admin_ev_problems_77_1")
        assert "Страница 2 из 2" in client.send_message.call_args.kwargs["text"]
    finally:
        await app.states.clear(456)


@pytest.mark.asyncio
async def test_max_successful_resolution_reopens_clamped_page_and_event_context(db_session_maker, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    from event_admin_service import create_scenario
    from max_messenger_bot import app as max_app_module

    scenarios = []
    for index in range(max_admin_events.PAGE_SIZE + 1):
        scenarios.append(
            await create_scenario(
                db_session_maker,
                code=f"max_resolution_context_{index}",
                event_type="meeting",
                title=f"MAX resolution context {index}",
                starts_at_utc=datetime.utcnow() + timedelta(days=2, minutes=index),
                place="Room 1",
            )
        )
    target = scenarios[0]
    delivery_ids = []
    async with db_session_maker() as session:
        for index in range(max_admin_events.PROBLEMS_PAGE_SIZE + 1):
            user = User(id=1200 + index, first_name=f"Problem {index}")
            session.add(user)
            await session.flush()
            registration = EventRegistration(event_id=target.id, user_id=user.id, status="started")
            session.add(registration)
            await session.flush()
            delivery = EventDelivery(
                event_id=target.id,
                user_id=user.id,
                registration_id=registration.id,
                message_kind="immediate_confirmation",
                delivery_key=f"max_resolution_delivery_{index}",
                due_at=datetime.utcnow(),
                next_attempt_at=datetime.utcnow(),
                platform="max",
                status="uncertain",
                attempt_count=1,
                last_error="timeout",
                content_version_snapshot=1,
            )
            session.add(delivery)
            await session.flush()
            delivery_ids.append(delivery.id)
        await session.commit()

    client = AsyncMock(spec=MaxApiClient)
    app = MaxBotApplication(client)
    app.states = MemoryStateStore()
    sender = Sender(user_id=456, username="admin", first_name="Admin", last_name=None)
    monkeypatch.setattr(max_app_module.common, "is_admin", AsyncMock(return_value=True))

    async def feed(payload: str):
        client.reset_mock()
        await app._handle_callback_impl(
            IncomingCallback(
                raw={},
                callback_id=f"resolution-{payload}",
                payload=payload,
                chat_id=123,
                message_id=None,
                sender=sender,
            )
        )

    try:
        await feed(f"admin_ev_open_{target.id}_1_1")
        await feed(f"admin_ev_stats_{target.id}")
        await feed(f"admin_ev_problems_{target.id}_1")
        item_callback = next(payload for payload in _max_event_payloads(client) if payload.startswith("admin_ev_problem_item_"))
        assert item_callback.endswith("_1")

        await feed(item_callback)
        resolve_callback = next(payload for payload in _max_event_payloads(client) if payload == f"admin_ev_resolve_sent_{delivery_ids[0]}")
        await feed(resolve_callback)
        client.reset_mock()
        await app._handle_message_impl(
            IncomingMessage(
                raw={},
                message_id="resolution-reason",
                chat_id=123,
                sender=sender,
                text="Confirmed delivery",
            )
        )
        assert f"admin_ev_problems_{target.id}_1" in _max_event_payloads(client)

        await feed(f"admin_ev_problems_{target.id}_1")
        assert "Проблемные доставки (5)" in client.send_message.call_args.kwargs["text"]
        assert "Страница 1 из 1" in client.send_message.call_args.kwargs["text"]
        assert "Проблемных доставок нет" not in client.send_message.call_args.kwargs["text"]

        await feed(f"admin_ev_stats_{target.id}")
        assert f"admin_ev_open_{target.id}_1_1" in _max_event_payloads(client)
        await feed(f"admin_ev_open_{target.id}_1_1")
        await feed("admin_ev_page_1_1")
        assert "Страница 2 из 2" in client.send_message.call_args.kwargs["text"]
        assert "admin_ev_toggle_archive_view_0" in _max_event_payloads(client)
    finally:
        await app.states.clear(456)


@pytest.mark.asyncio
async def test_max_pagination_boundaries_for_scenarios_and_problems(mock_client, monkeypatch):
    def scenario_rows(count: int) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                id=index + 1,
                title=f"Event {index + 1}",
                starts_at_msk="01.01.2030 12:00",
                code=f"event_{index + 1}",
                event_type="webinar",
                is_archived=False,
                is_active=True,
                registrations_count=0,
            )
            for index in range(count)
        ]

    for count in (
        0,
        1,
        max_admin_events.PAGE_SIZE,
        max_admin_events.PAGE_SIZE + 1,
        max_admin_events.PAGE_SIZE * 2 + 1,
    ):
        monkeypatch.setattr(max_admin_events, "list_scenarios", AsyncMock(return_value=scenario_rows(count)))
        mock_client.send_message.reset_mock()
        await max_admin_events.show_scenarios(mock_client, chat_id=123, page=0, include_archived=True)
        payloads = _max_event_payloads(mock_client)
        assert ("admin_ev_page_1_1" in payloads) is (count > max_admin_events.PAGE_SIZE)
        assert "admin_ev_page_0_1" not in payloads
        if count > max_admin_events.PAGE_SIZE:
            mock_client.send_message.reset_mock()
            await max_admin_events.show_scenarios(mock_client, chat_id=123, page=1, include_archived=True)
            payloads = _max_event_payloads(mock_client)
            assert "admin_ev_page_0_1" in payloads
            assert ("admin_ev_page_2_1" in payloads) is (count > max_admin_events.PAGE_SIZE * 2)
            if count > max_admin_events.PAGE_SIZE * 2:
                mock_client.send_message.reset_mock()
                await max_admin_events.show_scenarios(mock_client, chat_id=123, page=2, include_archived=True)
                payloads = _max_event_payloads(mock_client)
                assert "admin_ev_page_1_1" in payloads
                assert "admin_ev_page_3_1" not in payloads

    def problem_rows(count: int) -> list[DeliveryItem]:
        return [
            DeliveryItem(
                id=index + 1,
                event_id=10,
                user_id=100 + index,
                user_display=f"User {index + 1}",
                message_kind="immediate_confirmation",
                step_id=None,
                status="failed",
                due_at_msk="01.01.2030 12:00",
                attempt_count=1,
                last_error="failure",
                sent_or_finished_msk=None,
                platform="max",
                retry_index=0,
                parent_delivery_id=None,
            )
            for index in range(count)
        ]

    for count in (
        0,
        1,
        max_admin_events.PROBLEMS_PAGE_SIZE,
        max_admin_events.PROBLEMS_PAGE_SIZE + 1,
        max_admin_events.PROBLEMS_PAGE_SIZE * 2 + 1,
    ):
        async def fake_list_problem_deliveries(session_maker, event_id, *, limit, offset, _count=count):
            rows = problem_rows(_count)
            return rows[offset : offset + limit], _count

        monkeypatch.setattr(max_admin_events, "list_problem_deliveries", fake_list_problem_deliveries)
        mock_client.send_message.reset_mock()
        await max_admin_events.show_problems(mock_client, chat_id=123, event_id=10, page=0)
        payloads = _max_event_payloads(mock_client)
        assert ("admin_ev_problems_10_1" in payloads) is (count > max_admin_events.PROBLEMS_PAGE_SIZE)
        assert "admin_ev_problems_10_0" not in payloads
        if count > max_admin_events.PROBLEMS_PAGE_SIZE:
            mock_client.send_message.reset_mock()
            await max_admin_events.show_problems(mock_client, chat_id=123, event_id=10, page=1)
            payloads = _max_event_payloads(mock_client)
            assert "admin_ev_problems_10_0" in payloads
            assert ("admin_ev_problems_10_2" in payloads) is (count > max_admin_events.PROBLEMS_PAGE_SIZE * 2)
            if count > max_admin_events.PROBLEMS_PAGE_SIZE * 2:
                mock_client.send_message.reset_mock()
                await max_admin_events.show_problems(mock_client, chat_id=123, event_id=10, page=2)
                payloads = _max_event_payloads(mock_client)
                assert "admin_ev_problems_10_1" in payloads
                assert "admin_ev_problems_10_3" not in payloads


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
async def test_max_description_chunks_append_and_runtime_split_after_save(db_session_maker, mock_client, states, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    from event_admin_service import create_scenario
    from event_delivery_worker import build_test_delivery_plan

    scen = await create_scenario(
        db_session_maker,
        code="max_chunked_description",
        event_type="webinar",
        title="Chunked MAX description",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/max-chunks",
        media_mode="legacy_auto",
    )
    await max_admin_events.start_edit_description(mock_client, states, 123, 456, scen.id)
    await max_admin_events.receive_edit_description(mock_client, states, 123, 456, "A" * 3000)
    await max_admin_events.receive_edit_description(mock_client, states, 123, 456, "B" * 3000)
    await max_admin_events.save_description(mock_client, states, 123, 456)

    async with db_session_maker() as session:
        saved = await session.get(EventScenario, scen.id)
        assert saved.description == ("A" * 3000) + "\n" + ("B" * 3000)
    plan = await build_test_delivery_plan(db_session_maker, scen.id, None, None)
    text_parts = [part for part in plan if part["kind"] == "text"]
    assert len(text_parts) >= 2
    assert all(len(part["text"]) <= 4000 for part in text_parts)


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
async def test_max_media_back_route_preserves_unsaved_draft(states, mock_client, monkeypatch):
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
    snapshot = await states.get(456)
    assert snapshot is not None
    assert snapshot.state == "max_event_edit_media"
    assert snapshot.data["media_draft"][0]["file_type"] == "photo"


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
        media_mode="legacy_auto",
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
    saved_text = mock_client.send_message.call_args.kwargs["text"]
    saved_buttons = mock_client.send_message.call_args.kwargs["attachments"][0]["payload"]["buttons"]
    assert "Изменения мероприятия сохранены" in saved_text
    assert any(
        button.get("payload") == "admin_ev_page_2_1"
        for row in saved_buttons
        for button in row
    )


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
async def test_max_steps_roundtrip_preserves_paginated_back_callbacks(mock_client, states, monkeypatch):
    app = MaxBotApplication(mock_client)
    app.states = states
    monkeypatch.setattr("max_messenger_bot.app.common.is_admin", AsyncMock(return_value=True))
    show_steps = AsyncMock()
    show_step = AsyncMock()
    monkeypatch.setattr("max_messenger_bot.app.admin_events_service.show_steps", show_steps)
    monkeypatch.setattr("max_messenger_bot.app.admin_events_service.show_step", show_step)

    callback = IncomingCallback(
        raw={},
        callback_id="steps-page",
        payload="admin_ev_steps_77_2_1",
        chat_id=123,
        message_id=None,
        sender=Sender(user_id=456, username="admin", first_name="Admin", last_name=None),
    )
    await app._handle_callback_impl(callback)
    show_steps.assert_awaited_once_with(mock_client, 123, 77, back="admin_ev_page_2_1")
    snapshot = await states.get(456)
    assert snapshot.data["event_list_back"] == "admin_ev_page_2_1"

    callback.payload = "admin_ev_step_19"
    await app._handle_callback_impl(callback)
    show_step.assert_awaited_once_with(mock_client, 123, 19, back="admin_ev_page_2_1")


@pytest.mark.asyncio
async def test_max_dispatcher_steps_buttons_roundtrip_page_context(db_session_maker, mock_client, states, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    monkeypatch.setattr("max_messenger_bot.app.common.is_admin", AsyncMock(return_value=True))
    from event_admin_service import add_step, create_scenario

    scen = await create_scenario(
        db_session_maker,
        code="max_dispatcher_steps",
        event_type="webinar",
        title="MAX dispatcher steps",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/max-dispatcher-steps",
    )
    step = await add_step(db_session_maker, scen.id, -1440, "Initial reminder")
    app = MaxBotApplication(mock_client)
    app.states = states
    sender = Sender(user_id=456, username="admin", first_name="Admin", last_name=None)

    async def press(payload: str) -> None:
        await app._handle_callback_impl(
            IncomingCallback(
                raw={},
                callback_id=payload,
                payload=payload,
                chat_id=123,
                message_id=None,
                sender=sender,
            )
        )

    def visible_payloads() -> list[str]:
        rows = mock_client.send_message.call_args.kwargs["attachments"][0]["payload"]["buttons"]
        return [button["payload"] for row in rows for button in row if "payload" in button]

    await press(f"admin_ev_open_{scen.id}_2_1")
    steps_payload = next(payload for payload in visible_payloads() if payload.startswith(f"admin_ev_steps_{scen.id}"))
    assert steps_payload == f"admin_ev_steps_{scen.id}_2_1"

    await press(steps_payload)
    step_payload = next(payload for payload in visible_payloads() if payload.startswith(f"admin_ev_step_{step.id}"))
    assert step_payload == f"admin_ev_step_{step.id}_2_1"
    assert f"admin_ev_open_{scen.id}_2_1" in visible_payloads()

    await press(step_payload)
    edit_payload = next(payload for payload in visible_payloads() if payload.startswith(f"admin_ev_step_edit_text_{step.id}"))
    assert edit_payload == f"admin_ev_step_edit_text_{step.id}_2_1"

    await press(edit_payload)
    await app._handle_message_impl(
        IncomingMessage(
            raw={},
            message_id="max-msg-1",
            chat_id=123,
            sender=sender,
            text="Updated reminder",
        )
    )
    assert f"admin_ev_steps_{scen.id}_2_1" in visible_payloads()

    await press(edit_payload)
    cancel_payload = next(payload for payload in visible_payloads() if payload.startswith(f"admin_ev_step_{step.id}"))
    assert cancel_payload == f"admin_ev_step_{step.id}_2_1"
    await press(cancel_payload)
    assert f"admin_ev_steps_{scen.id}_2_1" in visible_payloads()
    await press(f"admin_ev_steps_{scen.id}_2_1")
    assert f"admin_ev_open_{scen.id}_2_1" in visible_payloads()


@pytest.mark.asyncio
async def test_max_dispatcher_delete_last_step_returns_empty_page(db_session_maker, mock_client, states, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    monkeypatch.setattr("max_messenger_bot.app.common.is_admin", AsyncMock(return_value=True))
    from event_admin_service import add_step, create_scenario, list_steps

    scen = await create_scenario(
        db_session_maker,
        code="max_delete_last_step",
        event_type="webinar",
        title="MAX delete last step",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/max-delete-last",
    )
    step = await add_step(db_session_maker, scen.id, -60, "Last reminder")
    app = MaxBotApplication(mock_client)
    app.states = states
    sender = Sender(user_id=456, username="admin", first_name="Admin", last_name=None)

    async def press(payload: str) -> None:
        await app._handle_callback_impl(
            IncomingCallback(
                raw={},
                callback_id=payload,
                payload=payload,
                chat_id=123,
                message_id=None,
                sender=sender,
            )
        )

    def visible_payloads() -> list[str]:
        return _max_event_payloads(mock_client)

    await press(f"admin_ev_steps_{scen.id}_2_1")
    await press(f"admin_ev_step_{step.id}_2_1")
    await press(f"admin_ev_step_del_ask_{step.id}_2_1")
    assert f"admin_ev_step_del_yes_{step.id}_2_1" in visible_payloads()
    mock_client.send_message.reset_mock()

    await press(f"admin_ev_step_del_yes_{step.id}_2_1")
    messages = [call.kwargs["text"] for call in mock_client.send_message.call_args_list if "text" in call.kwargs]
    assert any("Шагов пока нет" in text for text in messages)
    assert f"admin_ev_open_{scen.id}_2_1" in visible_payloads()
    assert not any(
        payload.startswith("admin_ev_step_") and not payload.startswith("admin_ev_step_add_")
        for payload in visible_payloads()
    )
    assert await list_steps(db_session_maker, scen.id, include_archived=False) == []


@pytest.mark.asyncio
async def test_max_dispatcher_description_chunks_save_reopen_and_runtime_split(db_session_maker, mock_client, states, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    monkeypatch.setattr("max_messenger_bot.app.common.is_admin", AsyncMock(return_value=True))
    from event_admin_service import create_scenario
    from event_delivery_worker import build_test_delivery_plan

    scen = await create_scenario(
        db_session_maker,
        code="max_dispatcher_description",
        event_type="webinar",
        title="MAX dispatcher description",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/max-dispatcher-description",
        media_mode="legacy_auto",
    )
    app = MaxBotApplication(mock_client)
    app.states = states
    sender = Sender(user_id=456, username="admin", first_name="Admin", last_name=None)

    async def press(payload: str) -> None:
        await app._handle_callback_impl(
            IncomingCallback(
                raw={},
                callback_id=payload,
                payload=payload,
                chat_id=123,
                message_id=None,
                sender=sender,
            )
        )

    def visible_payloads() -> list[str]:
        rows = mock_client.send_message.call_args.kwargs["attachments"][0]["payload"]["buttons"]
        return [button["payload"] for row in rows for button in row if "payload" in button]

    await press(f"admin_ev_open_{scen.id}_2_1")
    edit_callback = next(payload for payload in visible_payloads() if payload.startswith("admin_ev_edit_menu_"))
    await press(edit_callback)
    description_callback = next(
        payload for payload in visible_payloads() if payload.startswith(f"admin_ev_edit_description_{scen.id}")
    )
    await press(description_callback)
    await app._handle_message_impl(
        IncomingMessage(raw={}, message_id="max-description-1", chat_id=123, sender=sender, text="A" * 3000)
    )
    await app._handle_message_impl(
        IncomingMessage(raw={}, message_id="max-description-2", chat_id=123, sender=sender, text="B" * 3000)
    )
    await press(f"admin_ev_description_save_{scen.id}")

    async with db_session_maker() as session:
        saved = await session.get(EventScenario, scen.id)
        assert saved.description == ("A" * 3000) + "\n" + ("B" * 3000)
    plan = await build_test_delivery_plan(db_session_maker, scen.id, None, None)
    text_parts = [part for part in plan if part["kind"] == "text"]
    assert len(text_parts) >= 2
    assert all(len(part["text"]) <= 4000 for part in text_parts)

    await press(f"admin_ev_open_{scen.id}_2_1")
    assert any(payload.startswith("admin_ev_edit_menu_") for payload in visible_payloads())


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
    await max_admin_events.receive_create_starts_at(mock_client, states, chat_id=123, user_id=456, raw_dt=(datetime.utcnow() + timedelta(days=30)).strftime("%d.%m.%Y %H:%M"))
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
    await max_admin_events.receive_create_starts_at(mock_client, states, chat_id=123, user_id=456, raw_dt=(datetime.utcnow() + timedelta(days=30)).strftime("%d.%m.%Y %H:%M"))
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
async def test_max_test_send_uses_runtime_media_plan_without_delivery_rows(db_session_maker, mock_client, states, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)

    from event_admin_service import add_step, create_scenario
    max_user_id = 100000000456
    scen = await create_scenario(
        db_session_maker,
        code="max_test_send_media",
        event_type="webinar",
        title="MAX Test Media",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://max.example/media",
        description="MAX gallery description",
    )
    async with db_session_maker() as session:
        session.add(EventScenarioMedia(event_id=scen.id, file_type="photo", file_id="max-preview-photo", max_token="max-preview-token", sort_order=0))
        session.add(EventScenarioMedia(event_id=scen.id, file_type="video", file_id="max-preview-video", max_token="max-preview-video-token", sort_order=1))
        await session.commit()
    step = await add_step(
        db_session_maker,
        scen.id,
        -120,
        "Напоминание {event_title}",
        include_event_details=True,
    )
    mock_client.send_message.reset_mock()

    await max_admin_events.do_test_send(
        mock_client,
        chat_id=123,
        user_id=max_user_id,
        event_id=scen.id,
        step_id=step.id,
    )

    assert mock_client.send_message.await_count == 2
    media_call = mock_client.send_message.call_args_list[0].kwargs
    text_call = mock_client.send_message.call_args_list[1].kwargs
    assert any(item.get("type") == "image" for item in media_call["attachments"])
    assert "Напоминание MAX Test Media" in text_call["text"]
    async with db_session_maker() as session:
        assert (await session.execute(select(EventDelivery))).scalars().all() == []


@pytest.mark.asyncio
async def test_max_test_send_materializes_telegram_media_without_delivery_rows(
    db_session_maker, mock_client, monkeypatch
):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)

    from event_admin_service import add_step, create_scenario

    scen = await create_scenario(
        db_session_maker,
        code="max_test_send_telegram_media",
        event_type="webinar",
        title="MAX Telegram Media",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://max.example/telegram-media",
        description="Описание после Telegram-загрузки",
    )
    async with db_session_maker() as session:
        session.add_all(
            [
                EventScenarioMedia(
                    event_id=scen.id,
                    file_type="photo",
                    file_id="telegram-photo",
                    source_platform="telegram",
                    sort_order=0,
                ),
                EventScenarioMedia(
                    event_id=scen.id,
                    file_type="video",
                    file_id="telegram-video",
                    source_platform="telegram",
                    sort_order=1,
                ),
            ]
        )
        await session.commit()
    step = await add_step(
        db_session_maker,
        scen.id,
        -120,
        "Напоминание {event_title}",
        include_event_details=True,
    )

    class FakeTelegramBot:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get_file(self, file_id):
            return SimpleNamespace(file_path=f"{file_id}.bin")

        async def download_file(self, file_path):
            return io.BytesIO(file_path.encode())

    monkeypatch.setattr("telegram_client.create_telegram_bot", lambda token: FakeTelegramBot())
    mock_client.upload_file = AsyncMock(side_effect=[{"token": "max-photo-token"}, {"token": "max-video-token"}])
    mock_client.send_message.reset_mock()

    await max_admin_events.do_test_send(
        mock_client,
        chat_id=123,
        user_id=456,
        event_id=scen.id,
        step_id=step.id,
    )

    assert mock_client.upload_file.await_count == 2
    assert mock_client.send_message.await_count == 2
    media_payload = mock_client.send_message.call_args_list[0].kwargs
    assert {item["type"] for item in media_payload["attachments"]} >= {"image", "video"}
    assert {item["payload"]["token"] for item in media_payload["attachments"] if item["type"] in {"image", "video"}} == {
        "max-photo-token",
        "max-video-token",
    }
    async with db_session_maker() as session:
        assert (await session.execute(select(EventDelivery))).scalars().all() == []


@pytest.mark.asyncio
async def test_max_test_send_reports_media_materialization_error(db_session_maker, mock_client, monkeypatch):
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    from event_admin_service import create_scenario
    from event_media import EventMediaMaterializationError

    scen = await create_scenario(
        db_session_maker,
        code="max_test_send_materialization_error",
        event_type="webinar",
        title="MAX materialization error",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://max.example/materialization-error",
    )
    monkeypatch.setattr(
        "event_delivery_worker.ensure_max_test_media_materialized",
        AsyncMock(side_effect=EventMediaMaterializationError("Telegram media unavailable")),
    )
    mock_client.send_message.reset_mock()

    await max_admin_events.do_test_send(
        mock_client,
        chat_id=123,
        user_id=456,
        event_id=scen.id,
        step_id=None,
    )

    assert mock_client.send_message.await_count == 1
    assert "Не удалось подготовить медиа" in mock_client.send_message.call_args.kwargs["text"]
    async with db_session_maker() as session:
        assert (await session.execute(select(EventDelivery))).scalars().all() == []


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


@pytest.mark.asyncio
async def test_max_long_title_application_list_edit_cancel_save_reopen(db_session_maker, monkeypatch):
    """MAX incoming callback/message -> real service -> rendered keyboard -> DB -> back."""
    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    from event_admin_service import create_scenario
    from max_messenger_bot import app as max_app_module

    # Full MAX API/application ingress invokes ensure_user before FSM handlers.
    monkeypatch.setattr(max_app_module.common, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module, "async_session_maker", db_session_maker)

    original = "А" * 1000
    scen = await create_scenario(
        db_session_maker,
        code="max_long_event_title",
        event_type="webinar",
        title=original,
        starts_at_utc=datetime.utcnow() + timedelta(days=5),
        link="https://example.test/max-long-title",
    )
    client = AsyncMock(spec=MaxApiClient)
    app = MaxBotApplication(client)
    app.states = MemoryStateStore()
    sender = Sender(user_id=456, username="admin", first_name="Admin", last_name=None)
    monkeypatch.setattr(max_app_module.common, "is_admin", AsyncMock(return_value=True))

    async def press(payload):
        client.reset_mock()
        await app.handle_callback(
            IncomingCallback(
                raw={},
                callback_id=f"long-title-{payload}",
                payload=payload,
                chat_id=123,
                message_id=None,
                sender=sender,
            )
        )
        assert client.send_message.called, payload
        return client.send_message.await_args_list[-1]

    def button_payload(call, prefix):
        for attachment in call.kwargs.get("attachments") or []:
            for row in attachment.get("payload", {}).get("buttons", []):
                for button in row:
                    if button.get("payload", "").startswith(prefix):
                        return button["payload"], button.get("text", "")
        raise AssertionError(f"No visible MAX button {prefix!r}")

    try:
        listing = await press("admin_events")
        assert original not in listing.kwargs["text"]
        assert len(listing.kwargs["text"]) < 4000
        target, label = button_payload(listing, f"admin_ev_open_{scen.id}")
        assert len(label) < 100 and "…" in label

        card = await press(target)
        assert original in card.kwargs["text"]
        menu_payload, _ = button_payload(card, f"admin_ev_edit_menu_{scen.id}")
        edit_menu = await press(menu_payload)
        title_payload, _ = button_payload(edit_menu, f"admin_ev_edit_title_{scen.id}")
        prompt = await press(title_payload)
        assert "1000 символов" in prompt.kwargs["text"]
        cancel_payload, _ = button_payload(prompt, f"admin_ev_open_{scen.id}")
        card = await press(cancel_payload)
        assert original in card.kwargs["text"]

        edit_menu = await press(button_payload(card, f"admin_ev_edit_menu_{scen.id}")[0])
        prompt = await press(button_payload(edit_menu, f"admin_ev_edit_title_{scen.id}")[0])
        client.reset_mock()
        await app.handle_message(
            IncomingMessage(
                raw={},
                message_id="max-long-invalid",
                chat_id=123,
                sender=sender,
                text="В" * 1001,
            )
        )
        assert "1000" in client.send_message.await_args_list[-1].kwargs["text"]
        assert (await app.states.get(456)).state == "max_event_edit_title"
        client.reset_mock()
        await app.handle_message(
            IncomingMessage(
                raw={},
                message_id="max-long-valid",
                chat_id=123,
                sender=sender,
                text="Б" * 1000,
            )
        )
        async with db_session_maker() as session:
            assert (await session.get(EventScenario, scen.id)).title == "Б" * 1000
        card = await press(target)
        assert "Б" * 1000 in card.kwargs["text"]
        back_payload, _ = button_payload(card, "admin_ev_page_0_0")
        listing = await press(back_payload)
        assert "Мероприятия и цепочки" in listing.kwargs["text"]

        # Create another full-length title through the real MAX Application FSM.
        create_payload, _ = button_payload(listing, "admin_ev_create_webinar")
        prompt = await press(create_payload)
        assert "Шаг 1 из 5" in prompt.kwargs["text"]

        async def send_input(text, suffix):
            client.reset_mock()
            await app.handle_message(
                IncomingMessage(
                    raw={},
                    message_id=f"max-title-create-{suffix}",
                    chat_id=123,
                    sender=sender,
                    text=text,
                )
            )
            assert client.send_message.called
            return client.send_message.await_args_list[-1]

        await send_input("max_new_long_title_event", "code")
        title_prompt = await send_input("Г" * 1000, "title")
        assert "Шаг 3 из 5" in title_prompt.kwargs["text"]
        await send_input(
            (datetime.utcnow() + timedelta(days=40)).strftime("%d.%m.%Y %H:%M"),
            "date",
        )
        await send_input("120", "duration")
        await send_input("https://example.test/new-max-long-title", "link")
        async with db_session_maker() as session:
            created = await session.scalar(
                select(EventScenario).where(EventScenario.code == "max_new_long_title_event")
            )
            assert created is not None and created.title == "Г" * 1000
        listing = await press("admin_events")
        assert "Мероприятия и цепочки" in listing.kwargs["text"]
    finally:
        await app.states.clear(456)


@pytest.mark.asyncio
async def test_max_real_application_limits_error_screen_edit_and_reopen_journey(db_session_maker, monkeypatch):
    """Real MAX Application journey:

    1. Open event description editor and enter long description exceeding safe caption limit.
    2. Press the real Save button.
    3. Receive validated outgoing message explaining limit excess (platform, actual length, limit, excess, recommendation).
    4. Verify working edit buttons on error screen (edit description, edit title, format menu, save, cancel draft).
    5. Test unified draft across format change and cancel draft.
    6. Fix description text to a valid length.
    7. Press Save and verify success confirmation.
    8. Reopen event card and verify updated description in DB and card view.
    """
    from max_messenger_bot import app as max_app_module
    from max_messenger_bot.app import MaxBotApplication
    from max_messenger_bot.api import MaxApiClient
    from max_messenger_bot.models import IncomingCallback, IncomingMessage, Sender
    from event_admin_service import create_scenario, update_scenario_fields

    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "is_admin", AsyncMock(return_value=True))

    now = datetime.utcnow()
    scen = await create_scenario(
        db_session_maker,
        code="max_journey_limits",
        event_type="webinar",
        title="MAX Journey Limits Webinar",
        starts_at_utc=now + timedelta(days=2),
        link="https://example.test/max-journey",
        media_format="separate_media_top",
    )
    async with db_session_maker() as session:
        session.add(EventScenarioMedia(
            event_id=scen.id,
            file_type="photo",
            file_id="max_photo_1",
            max_token="max_photo_1",
            source_platform="max",
            sort_order=0,
        ))
        await session.commit()
    await update_scenario_fields(db_session_maker, scen.id, media_format="caption")

    client = AsyncMock(spec=MaxApiClient)
    app = MaxBotApplication(client)
    app.states = MemoryStateStore()
    sender = Sender(user_id=456, username="admin", first_name="Admin", last_name=None)

    async def press(payload):
        client.reset_mock()
        await app.handle_callback(
            IncomingCallback(
                raw={},
                callback_id=f"journey-{payload}",
                payload=payload,
                chat_id=123,
                message_id=None,
                sender=sender,
            )
        )
        assert client.send_message.called, f"No message sent for callback {payload}"
        return client.send_message.await_args_list[-1]

    async def send_msg(text):
        client.reset_mock()
        await app.handle_message(
            IncomingMessage(
                raw={},
                message_id="msg-journey",
                chat_id=123,
                sender=sender,
                text=text,
            )
        )
        assert client.send_message.called, f"No message sent for text {text[:20]}"
        return client.send_message.await_args_list[-1]

    def get_buttons(call):
        res = []
        for attachment in call.kwargs.get("attachments") or []:
            for row in attachment.get("payload", {}).get("buttons", []):
                for button in row:
                    res.append((button.get("payload", ""), button.get("text", "")))
        return res

    try:
        # Step 1: Open description editor
        call = await press(f"admin_ev_edit_description_{scen.id}")
        assert "Описание мероприятия" in call.kwargs["text"]

        # Step 2: Feed long description exceeding caption limit (950 chars > 900)
        long_desc = "Д" * 950
        call = await send_msg(long_desc)
        assert "Черновик обновлён" in call.kwargs["text"]

        # Step 3: Press real Save button
        call = await press(f"admin_ev_description_save_{scen.id}")
        error_text = call.kwargs["text"]
        assert "Невозможно сохранить изменения!" in error_text
        assert "• Платформа: <b>TELEGRAM</b>" in error_text or "• Платформа: <b>MAX</b>" in error_text
        assert "Фактическая длина:" in error_text
        assert "лимит:" in error_text
        assert "Превышение:" in error_text
        assert "Рекомендация:" in error_text

        # Step 4: Verify working edit buttons on error screen
        buttons = get_buttons(call)
        button_payloads = [b[0] for b in buttons]
        assert f"admin_ev_edit_description_{scen.id}" in button_payloads
        assert f"admin_ev_edit_title_{scen.id}" in button_payloads
        assert f"admin_ev_format_menu_{scen.id}" in button_payloads
        assert f"admin_ev_description_save_{scen.id}" in button_payloads
        assert f"admin_ev_cancel_draft_{scen.id}" in button_payloads

        # Test unified draft: edit title while draft exists
        call = await press(f"admin_ev_edit_title_{scen.id}")
        assert "новое название" in call.kwargs["text"]
        call = await send_msg("MAX Новое Название 🎯")
        assert "Название в черновике обновлено" in call.kwargs["text"]

        # Check DB is still untouched
        async with db_session_maker() as session:
            unchanged = await session.get(EventScenario, scen.id)
            assert unchanged.title == "MAX Journey Limits Webinar"
            assert unchanged.description is None

        # Test cancel draft: click cancel draft button
        call = await press(f"admin_ev_cancel_draft_{scen.id}")
        assert any("Черновик сброшен" in c.kwargs.get("text", "") for c in client.send_message.await_args_list)
        assert "Карточка мероприятия" in call.kwargs["text"]
        state_after_cancel = await app.states.get(456)
        assert state_after_cancel is None or state_after_cancel.state is None

        # Re-open description editor and feed long description again
        call = await press(f"admin_ev_edit_description_{scen.id}")
        call = await send_msg(long_desc)
        call = await press(f"admin_ev_description_save_{scen.id}")
        assert "Невозможно сохранить" in call.kwargs["text"]

        # Step 5: Click edit description button from error screen
        call = await press(f"admin_ev_edit_description_{scen.id}")
        assert "Описание мероприятия" in call.kwargs["text"]

        # Step 6: Fix text to valid length
        valid_desc = "Корректное описание мероприятия для MAX."
        call = await send_msg(valid_desc)
        assert "Черновик обновлён" in call.kwargs["text"]

        # Step 7: Click Save
        call = await press(f"admin_ev_description_save_{scen.id}")
        assert "Карточка мероприятия" in call.kwargs["text"]
        assert "Изменения мероприятия сохранены" in call.kwargs["text"]

        # Step 8: Reopen event card and check result
        call = await press(f"admin_ev_open_{scen.id}")
        assert valid_desc in call.kwargs["text"]

        async with db_session_maker() as session:
            refreshed = await session.get(EventScenario, scen.id)
            assert refreshed.description == valid_desc
            assert refreshed.media_mode == "caption"
    finally:
        await app.states.clear(456)


@pytest.mark.asyncio
async def test_max_cross_event_draft_stale_callback_rejection_journey(db_session_maker, monkeypatch):
    """MAX Application negative journey:
    Admin has active draft for Event A.
    Stale callback buttons for Event B must NOT save, reset, or transfer draft of Event A.
    DB records for both events remain completely unchanged.
    """
    from max_messenger_bot import app as max_app_module
    from max_messenger_bot.app import MaxBotApplication
    from max_messenger_bot.api import MaxApiClient
    from max_messenger_bot.models import IncomingCallback, IncomingMessage, Sender
    from event_admin_service import create_scenario

    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "is_admin", AsyncMock(return_value=True))

    now = datetime.utcnow()
    event_a = await create_scenario(
        db_session_maker,
        code="max_draft_event_a",
        event_type="webinar",
        title="MAX Event A",
        starts_at_utc=now + timedelta(days=2),
        link="https://example.test/max-a",
        description="Original MAX Event A Description",
        media_format="separate_media_top",
    )
    event_b = await create_scenario(
        db_session_maker,
        code="max_draft_event_b",
        event_type="webinar",
        title="MAX Event B",
        starts_at_utc=now + timedelta(days=3),
        link="https://example.test/max-b",
        description="Original MAX Event B Description",
        media_format="separate_media_top",
    )

    client = AsyncMock(spec=MaxApiClient)
    app = MaxBotApplication(client)
    app.states = MemoryStateStore()
    sender = Sender(user_id=456, username="admin", first_name="Admin", last_name=None)

    async def press(payload):
        client.reset_mock()
        await app.handle_callback(
            IncomingCallback(
                raw={},
                callback_id=f"cb-{payload}",
                payload=payload,
                chat_id=123,
                message_id=None,
                sender=sender,
            )
        )
        return client.send_message.await_args_list[-1] if client.send_message.called else None

    async def send_msg(text):
        client.reset_mock()
        await app.handle_message(
            IncomingMessage(
                raw={},
                message_id=f"msg-{len(text)}",
                chat_id=123,
                sender=sender,
                text=text,
            )
        )
        return client.send_message.await_args_list[-1] if client.send_message.called else None

    try:
        # Step 1: Open description editor for Event A and input draft
        await press(f"admin_ev_edit_description_{event_a.id}")
        draft_text = "Новый незафиксированный черновик для MAX Event A"
        await send_msg(draft_text)

        # Step 2: Stale Save button from Event B
        call = await press(f"admin_ev_description_save_{event_b.id}")
        assert call is not None
        assert "Черновик принадлежит другому мероприятию" in call.kwargs["text"]

        # Check DB records are untouched
        async with db_session_maker() as session:
            ea = await session.get(EventScenario, event_a.id)
            eb = await session.get(EventScenario, event_b.id)
            assert ea.description == "Original MAX Event A Description"
            assert eb.description == "Original MAX Event B Description"

        # Step 3: Stale Cancel button from Event B
        call = await press(f"admin_ev_cancel_draft_{event_b.id}")
        assert call is not None
        assert "Черновик принадлежит другому мероприятию" in call.kwargs["text"]

        # Check DB and state still untouched
        async with db_session_maker() as session:
            ea = await session.get(EventScenario, event_a.id)
            eb = await session.get(EventScenario, event_b.id)
            assert ea.description == "Original MAX Event A Description"
            assert eb.description == "Original MAX Event B Description"
        state_obj = await app.states.get(456)
        assert state_obj.data.get("event_id") == event_a.id
        assert state_obj.data.get("description_draft") == draft_text

        # Step 4: Stale Format selection from Event B
        call = await press(f"admin_ev_set_fmt_{event_b.id}_caption")
        assert call is not None
        assert "Сначала завершите или отмените" in call.kwargs["text"]

        # Step 5: Stale Edit Title from Event B
        call = await press(f"admin_ev_edit_title_{event_b.id}")
        assert call is not None
        assert "Сначала завершите или отмените" in call.kwargs["text"]

        # Other stale card mutations must be rejected before they can change
        # Event B or replace the active Event A draft.
        for stale_payload in (
            f"admin_ev_toggle_{event_b.id}",
            f"admin_ev_archive_yes_{event_b.id}",
            f"admin_ev_steps_{event_b.id}_2_1",
        ):
            await press(stale_payload)
            assert any(
                "Сначала завершите или отмените" in c.kwargs.get("notification", "")
                for c in client.answer_callback.await_args_list
            ) or any(
                "Сначала завершите или отмените" in c.kwargs.get("text", "")
                for c in client.send_message.await_args_list
            )
            state_obj = await app.states.get(456)
            assert state_obj.data.get("event_id") == event_a.id
            assert state_obj.data.get("description_draft") == draft_text

        # Step 6: Valid Save for Event A
        await press(f"admin_ev_description_save_{event_a.id}")
        assert any("сохранены" in c.kwargs.get("text", "") for c in client.send_message.await_args_list)

        # Verify Event A in DB is updated, Event B is untouched
        async with db_session_maker() as session:
            ea = await session.get(EventScenario, event_a.id)
            eb = await session.get(EventScenario, event_b.id)
            assert ea.description == draft_text
            assert eb.description == "Original MAX Event B Description"
    finally:
        await app.states.clear(456)


@pytest.mark.asyncio
async def test_max_html_length_mismatch_application_limits_journey(db_session_maker, monkeypatch):
    """MAX Application journey for HTML length limit mismatch:
    Visible text fits within limit, but raw HTML serialization exceeds 3900/4000 limit.
    Rejection explains MAX limits; DB unchanged; fixing with shorter HTML commits successfully.
    """
    from max_messenger_bot import app as max_app_module
    from max_messenger_bot.app import MaxBotApplication
    from max_messenger_bot.api import MaxApiClient
    from max_messenger_bot.models import IncomingCallback, IncomingMessage, Sender
    from event_admin_service import create_scenario

    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "is_admin", AsyncMock(return_value=True))

    now = datetime.utcnow()
    scen = await create_scenario(
        db_session_maker,
        code="max_html_mismatch_ev",
        event_type="webinar",
        title="HTML Mismatch Webinar",
        starts_at_utc=now + timedelta(days=2),
        link="https://example.test/html-mismatch",
        media_format="separate_media_top",
    )

    client = AsyncMock(spec=MaxApiClient)
    app = MaxBotApplication(client)
    app.states = MemoryStateStore()
    sender = Sender(user_id=456, username="admin", first_name="Admin", last_name=None)

    async def press(payload):
        client.reset_mock()
        await app.handle_callback(
            IncomingCallback(
                raw={},
                callback_id=f"cb-{payload}",
                payload=payload,
                chat_id=123,
                message_id=None,
                sender=sender,
            )
        )
        return client.send_message.await_args_list[-1] if client.send_message.called else None

    async def send_msg(text):
        client.reset_mock()
        await app.handle_message(
            IncomingMessage(
                raw={},
                message_id=f"msg-{len(text)}",
                chat_id=123,
                sender=sender,
                text=text,
            )
        )
        return client.send_message.await_args_list[-1] if client.send_message.called else None

    try:
        # Open description editor
        await press(f"admin_ev_edit_description_{scen.id}")

        # Construct payload with many HTML tags: visible plain text ~2000 chars,
        # but raw HTML length > 4000 characters.
        long_html = "<b>текст</b>" * 450
        assert len(long_html) > 4000
        from event_content_validator import render_plain_text_for_max
        assert len(render_plain_text_for_max(long_html)) < 3000

        await send_msg(long_html)

        # Press Save
        call = await press(f"admin_ev_description_save_{scen.id}")
        assert call is not None
        error_text = call.kwargs["text"]
        assert "Невозможно сохранить изменения!" in error_text
        assert "• Платформа: <b>MAX</b>" in error_text
        assert "Фактическая длина:" in error_text
        assert "Превышение:" in error_text

        # Verify DB is unchanged
        async with db_session_maker() as session:
            unchanged = await session.get(EventScenario, scen.id)
            assert unchanged.description is None

        # Fix description to valid HTML text
        valid_html = "<b>Короткое корректное описание вебинара</b>"
        await press(f"admin_ev_edit_description_{scen.id}")
        await press(f"admin_ev_description_delete_{scen.id}")
        await send_msg(valid_html)

        # Press Save again
        await press(f"admin_ev_description_save_{scen.id}")
        assert any("сохранены" in c.kwargs.get("text", "") for c in client.send_message.await_args_list)

        # Verify DB is updated
        async with db_session_maker() as session:
            refreshed = await session.get(EventScenario, scen.id)
            assert refreshed.description == valid_html
    finally:
        await app.states.clear(456)


@pytest.mark.asyncio
async def test_max_back_preserves_draft_across_format_title_desc_preview_journey(db_session_maker, monkeypatch):
    """MAX Application journey verifying Back buttons preserve active draft:
    Return from format menu, edit title, preview TG, preview MAX preserves StateStore draft.
    Cancel draft explicitly discards it.
    """
    from max_messenger_bot import app as max_app_module
    from max_messenger_bot.app import MaxBotApplication
    from max_messenger_bot.api import MaxApiClient
    from max_messenger_bot.models import IncomingCallback, IncomingMessage, Sender
    from event_admin_service import create_scenario

    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "is_admin", AsyncMock(return_value=True))

    now = datetime.utcnow()
    scen = await create_scenario(
        db_session_maker,
        code="max_draft_preservation_ev",
        event_type="webinar",
        title="Original Draft Webinar",
        starts_at_utc=now + timedelta(days=2),
        link="https://example.test/draft-preservation",
        description="Original description",
        media_format="separate_media_top",
    )

    client = AsyncMock(spec=MaxApiClient)
    app = MaxBotApplication(client)
    app.states = MemoryStateStore()
    sender = Sender(user_id=456, username="admin", first_name="Admin", last_name=None)

    async def press(payload):
        client.reset_mock()
        await app.handle_callback(
            IncomingCallback(
                raw={},
                callback_id=f"cb-{payload}",
                payload=payload,
                chat_id=123,
                message_id=None,
                sender=sender,
            )
        )
        return client.send_message.await_args_list[-1] if client.send_message.called else None

    async def send_msg(text):
        client.reset_mock()
        await app.handle_message(
            IncomingMessage(
                raw={},
                message_id=f"msg-{len(text)}",
                chat_id=123,
                sender=sender,
                text=text,
            )
        )
        return client.send_message.await_args_list[-1] if client.send_message.called else None

    def get_buttons(call):
        res = []
        for att in call.kwargs.get("attachments") or []:
            for row in att.get("payload", {}).get("buttons", []):
                for b in row:
                    res.append((b.get("payload"), b.get("text")))
        return res

    try:
        # Step 1: Open edit description and add draft
        await press(f"admin_ev_edit_description_{scen.id}")
        await send_msg("Несохранённый текст описания для проверки Back")

        # Step 2: Open edit title and add draft title
        await press(f"admin_ev_edit_title_{scen.id}")
        await send_msg("Новое название в черновике")

        # Step 3: Open format menu and select new format
        await press(f"admin_ev_format_menu_{scen.id}")
        call = await press(f"admin_ev_set_fmt_{scen.id}_separate_text_top")
        assert "Черновик ещё не сохранён" in call.kwargs["text"]

        # Back from format menu -> edit menu
        call = await press(f"admin_ev_edit_menu_{scen.id}")
        assert "Есть несохранённый черновик" in call.kwargs["text"]
        button_payloads = [b[0] for b in get_buttons(call)]
        assert f"admin_ev_description_save_{scen.id}" in button_payloads
        assert f"admin_ev_cancel_draft_{scen.id}" in button_payloads

        # Step 4: Open TG preview and click Back
        call = await press(f"admin_ev_preview_tg_{scen.id}")
        assert "Предпросмотр доставки (Telegram)" in call.kwargs["text"]
        call = await press(f"admin_ev_edit_menu_{scen.id}")
        assert "Есть несохранённый черновик" in call.kwargs["text"]

        # Step 5: Open MAX preview and click Back
        call = await press(f"admin_ev_preview_max_{scen.id}")
        assert "Предпросмотр доставки (MAX)" in call.kwargs["text"]
        call = await press(f"admin_ev_edit_menu_{scen.id}")
        assert "Есть несохранённый черновик" in call.kwargs["text"]

        # Step 6: Verify StateStore still retains all draft properties
        state_obj = await app.states.get(456)
        assert state_obj.data.get("event_id") == scen.id
        assert state_obj.data.get("description_draft") == "Несохранённый текст описания для проверки Back"
        assert state_obj.data.get("draft_title") == "Новое название в черновике"
        assert state_obj.data.get("draft_format") == "separate_text_top"

        # Step 7: Cancel draft -> StateStore cleared
        await press(f"admin_ev_cancel_draft_{scen.id}")
        assert any("Черновик сброшен" in c.kwargs.get("text", "") for c in client.send_message.await_args_list)

        # Verify DB is unchanged
        async with db_session_maker() as session:
            unchanged = await session.get(EventScenario, scen.id)
            assert unchanged.title == "Original Draft Webinar"
            assert unchanged.description == "Original description"
            assert unchanged.media_format == "separate_media_top"
    finally:
        await app.states.clear(456)


@pytest.mark.asyncio
async def test_max_custom_emoji_fallback_render(db_session_maker, monkeypatch):
    """MAX custom emoji render:
    Scenario title containing Telegram <tg-emoji> entity falls back to Unicode emoji on MAX.
    """
    from max_messenger_bot import app as max_app_module
    from max_messenger_bot.app import MaxBotApplication
    from max_messenger_bot.api import MaxApiClient
    from event_admin_service import create_scenario
    from event_content_validator import format_event_html_title

    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module, "async_session_maker", db_session_maker)

    raw_title = '<tg-emoji emoji-id="5368324170671202286">🔥</tg-emoji> Горячий вебинар MAX'
    scen = await create_scenario(
        db_session_maker,
        code="max_emoji_scen",
        event_type="webinar",
        title=raw_title,
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/emoji-max",
    )

    client = AsyncMock(spec=MaxApiClient)
    app = MaxBotApplication(client)

    # Test format_event_html_title for MAX platform
    max_title = format_event_html_title(scen.title, platform="max")
    assert "<tg-emoji" not in max_title
    assert "emoji-id" not in max_title
    assert "🔥 Горячий вебинар MAX" in max_title

    # Test show_scenario in MAX
    await max_admin_events.show_scenario(client, 123, scen.id)
    sent_text = client.send_message.await_args_list[-1].kwargs["text"]
    assert "<tg-emoji" not in sent_text
    assert "🔥 Горячий вебинар MAX" in sent_text

    # Test show_edit_menu in MAX
    await max_admin_events.show_edit_menu(client, 123, scen.id)
    sent_text = client.send_message.await_args_list[-1].kwargs["text"]
    assert "<tg-emoji" not in sent_text
    assert "🔥 Горячий вебинар MAX" in sent_text

    # Test show_steps in MAX
    await max_admin_events.show_steps(client, 123, scen.id)
    sent_text = client.send_message.await_args_list[-1].kwargs["text"]
    assert "<tg-emoji" not in sent_text
    assert "🔥 Горячий вебинар MAX" in sent_text

    # Test show_stats in MAX
    await max_admin_events.show_stats(client, 123, scen.id)
    sent_text = client.send_message.await_args_list[-1].kwargs["text"]
    assert "<tg-emoji" not in sent_text
    assert "🔥 Горячий вебинар MAX" in sent_text


@pytest.mark.asyncio
async def test_max_holistic_draft_media_save_journey(db_session_maker, monkeypatch):
    """MAX Application Journey: description -> title -> format -> media -> Cancel -> repeat with Save Media.
    Verifies that Save Media commits description, title, format, and media items atomically in one transaction,
    tested through real MaxBotApplication.handle_update() ingress with visible button inspection,
    MAX transport validation (text <= 4000, attachments <= 12), and delivery plan execution."""
    from max_messenger_bot import app as max_app_module
    from max_messenger_bot.app import MaxBotApplication
    from max_messenger_bot.api import MaxApiClient
    from event_admin_service import create_scenario, list_scenario_media
    from event_delivery_worker import _build_delivery_plan, _send_delivery_part
    from database import EventScenario, User
    from sqlalchemy.orm import selectinload
    from sqlalchemy import select

    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "is_admin", AsyncMock(return_value=True))

    admin_raw_id = 456
    admin_effective_id = MAX_ID_OFFSET + admin_raw_id
    chat_id = 777

    # Prepare user and scenario in DB
    async with db_session_maker() as session:
        user_row = User(
            id=admin_effective_id,
            username="admin_user",
            first_name="Admin Max User",
        )
        session.add(user_row)
        await session.commit()

    scen = await create_scenario(
        db_session_maker,
        code="max_holistic_draft",
        event_type="webinar",
        title="Initial Max Title & <Legacy>",
        starts_at_utc=datetime.utcnow() + timedelta(days=3),
        link="https://example.test/max-draft",
        description="Initial Max Description",
        media_format="separate_media_top",
    )

    transport = _RecordedMaxSession()
    client = MaxApiClient(token="test-token", base_url="https://max.test")
    client._session = transport
    app = MaxBotApplication(client)
    app.states = MemoryStateStore()
    update_counter = 0

    def message_body() -> dict:
        for request in reversed(transport.requests):
            if request["url"].endswith("/messages"):
                return request["body"]
        raise AssertionError("No MAX /messages request was recorded")

    def validate_max_message(body: dict) -> dict:
        text = body.get("text") or ""
        attachments = body.get("attachments") or []
        assert len(text) <= 4000, f"MAX text length exceeded: {len(text)} > 4000"
        assert len(attachments) <= 12, f"MAX attachments limit exceeded: {len(attachments)} > 12"
        # Validate that body is JSON serializable
        json.loads(json.dumps(body, ensure_ascii=False))
        return body

    def button_payload(body: dict, prefix: str) -> str:
        for attachment in body.get("attachments") or []:
            for row in attachment.get("payload", {}).get("buttons", []):
                for button in row:
                    payload = button.get("payload")
                    if payload and payload.startswith(prefix):
                        return payload
        raise AssertionError(f"No visible MAX callback with prefix {prefix!r} in {body.get('attachments')}")

    def all_button_texts(body: dict) -> list[str]:
        texts = []
        for attachment in body.get("attachments") or []:
            for row in attachment.get("payload", {}).get("buttons", []):
                for button in row:
                    if "text" in button:
                        texts.append(button["text"])
        return texts

    async def press(payload: str) -> dict:
        nonlocal update_counter
        update_counter += 1
        transport.requests.clear()
        await app.handle_update(
            {
                "update_type": "message_callback",
                "update_id": f"max-journey-cb-{update_counter}",
                "callback": {
                    "callback_id": f"max-journey-callback-{update_counter}",
                    "payload": payload,
                    "user": {"user_id": admin_raw_id, "first_name": "Admin", "username": "admin"},
                },
                "message": {
                    "mid": f"max-journey-message-{update_counter}",
                    "recipient": {"chat_id": chat_id},
                    "body": {"text": "admin"},
                },
            }
        )
        return validate_max_message(message_body())

    async def send_text(text: str) -> dict:
        nonlocal update_counter
        update_counter += 1
        transport.requests.clear()
        await app.handle_update(
            {
                "update_type": "message_created",
                "update_id": f"max-journey-message-{update_counter}",
                "message": {
                    "mid": f"max-journey-incoming-{update_counter}",
                    "recipient": {"chat_id": chat_id},
                    "sender": {"user_id": admin_raw_id, "first_name": "Admin", "username": "admin"},
                    "body": {"text": text},
                },
            }
        )
        return validate_max_message(message_body())

    async def send_media(media_type: str, token: str) -> dict:
        nonlocal update_counter
        update_counter += 1
        transport.requests.clear()
        await app.handle_update(
            {
                "update_type": "message_created",
                "update_id": f"max-journey-media-{update_counter}",
                "message": {
                    "mid": f"max-journey-incoming-media-{update_counter}",
                    "recipient": {"chat_id": chat_id},
                    "sender": {"user_id": admin_raw_id, "first_name": "Admin", "username": "admin"},
                    "body": {
                        "attachments": [
                            {
                                "type": media_type,
                                "payload": {"token": token},
                            }
                        ]
                    },
                },
            }
        )
        return validate_max_message(message_body())

    try:
        # =========================================================================
        # PHASE 1: Build draft (desc, title, format, media) -> Cancel -> verify DB untouched
        # =========================================================================
        card = await press(f"admin_ev_open_{scen.id}_0_1")
        assert "Initial Max Title" in card["text"]
        assert "Initial Max Description" in card["text"]
        assert "Текст отдельно, медиа выше" in card["text"]

        # Open edit menu
        edit_menu = await press(button_payload(card, f"admin_ev_edit_menu_{scen.id}"))
        assert any("Описание" in t for t in all_button_texts(edit_menu))

        # 1. Edit description
        await press(button_payload(edit_menu, f"admin_ev_edit_description_{scen.id}"))
        desc_screen = await send_text("Pending Max Description")
        assert "Черновик обновлён" in desc_screen["text"]
        edit_menu = await press(button_payload(desc_screen, f"admin_ev_edit_menu_{scen.id}"))

        # 2. Edit title
        await press(button_payload(edit_menu, f"admin_ev_edit_title_{scen.id}"))
        title_screen = await send_text("Pending Max Title & <Draft>")
        edit_menu = await press(button_payload(title_screen, f"admin_ev_edit_menu_{scen.id}"))

        # 3. Format menu
        fmt_menu = await press(button_payload(edit_menu, f"admin_ev_format_menu_{scen.id}"))
        fmt_menu = await press(button_payload(fmt_menu, f"admin_ev_set_fmt_{scen.id}_separate_text_top"))
        edit_menu = await press(button_payload(fmt_menu, f"admin_ev_edit_menu_{scen.id}"))

        # 4. Add media
        media_editor = await press(button_payload(edit_menu, f"admin_ev_edit_media_{scen.id}"))
        await press(button_payload(media_editor, f"admin_ev_media_add_{scen.id}"))
        media_editor = await send_media("image", "token_cancel_1")
        assert "В черновике: 1" in media_editor["text"]

        # 5. Cancel draft
        edit_menu = await press(button_payload(media_editor, f"admin_ev_edit_menu_{scen.id}"))
        card = await press(button_payload(edit_menu, f"admin_ev_cancel_draft_{scen.id}"))
        assert "Черновик сброшен" in card["text"]

        # Verify DB is completely untouched
        async with db_session_maker() as session:
            s_after_cancel = await session.get(EventScenario, scen.id)
            assert "Initial Max Title" in s_after_cancel.title
            assert s_after_cancel.description == "Initial Max Description"
            assert s_after_cancel.media_format == "separate_media_top"
        media_after_cancel = await list_scenario_media(db_session_maker, scen.id)
        assert len(media_after_cancel) == 0
        assert await app.states.get(admin_effective_id) is None

        # =========================================================================
        # PHASE 2: Re-enter draft -> description, title, format, 2 media items -> reorder -> SAVE MEDIA
        # =========================================================================
        card = await press(f"admin_ev_open_{scen.id}_0_1")
        edit_menu = await press(button_payload(card, f"admin_ev_edit_menu_{scen.id}"))

        # 1. Enter description
        await press(button_payload(edit_menu, f"admin_ev_edit_description_{scen.id}"))
        desc_screen = await send_text("Final Saved Max Description with <b>Bold</b> & <i>Italic</i>")
        assert "Черновик обновлён" in desc_screen["text"]
        edit_menu = await press(button_payload(desc_screen, f"admin_ev_edit_menu_{scen.id}"))

        # 2. Enter title with HTML entities
        await press(button_payload(edit_menu, f"admin_ev_edit_title_{scen.id}"))
        title_screen = await send_text("Final Saved Max Title & <Safe>")
        edit_menu = await press(button_payload(title_screen, f"admin_ev_edit_menu_{scen.id}"))

        # 3. Choose format separate_text_top
        fmt_menu = await press(button_payload(edit_menu, f"admin_ev_format_menu_{scen.id}"))
        fmt_menu = await press(button_payload(fmt_menu, f"admin_ev_set_fmt_{scen.id}_separate_text_top"))
        edit_menu = await press(button_payload(fmt_menu, f"admin_ev_edit_menu_{scen.id}"))

        # 4. Add 2 media items: photo then video
        media_editor = await press(button_payload(edit_menu, f"admin_ev_edit_media_{scen.id}"))
        await press(button_payload(media_editor, f"admin_ev_media_add_{scen.id}"))
        media_editor = await send_media("image", "max_media_photo_1")
        assert "В черновике: 1" in media_editor["text"]

        await press(button_payload(media_editor, f"admin_ev_media_add_{scen.id}"))
        media_editor = await send_media("video", "max_media_video_2")
        assert "В черновике: 2" in media_editor["text"]

        # 5. Reorder media items: move second item (video) UP
        media_editor = await press(button_payload(media_editor, f"admin_ev_media_up_{scen.id}_1"))

        # 6. Click Save Media
        save_confirmation = await press(button_payload(media_editor, f"admin_ev_media_save_{scen.id}"))
        assert "сохранены" in save_confirmation["text"]
        assert "&amp;amp;" not in save_confirmation["text"]

        # Verify DB is atomically updated
        async with db_session_maker() as session:
            s_final = await session.get(EventScenario, scen.id)
            assert "Final Saved Max Title &" in s_final.title
            assert s_final.description == "Final Saved Max Description with <b>Bold</b> & <i>Italic</i>"
            assert s_final.media_format == "separate_text_top"

        media_final = await list_scenario_media(db_session_maker, scen.id)
        assert len(media_final) == 2
        # Verify reordering: video is first, photo is second
        assert media_final[0].file_id == "max_media_video_2"
        assert media_final[0].file_type == "video"
        assert media_final[0].source_platform == "max"
        assert media_final[0].source_file_id == "max_media_video_2"

        assert media_final[1].file_id == "max_media_photo_1"
        assert media_final[1].file_type == "photo"
        assert media_final[1].source_platform == "max"
        assert media_final[1].source_file_id == "max_media_photo_1"

        # Verify state is cleared
        assert await app.states.get(admin_effective_id) is None

        # Reopen card directly via callback and verify clean presentation without double escaping
        card = await press(button_payload(save_confirmation, f"admin_ev_open_{scen.id}"))
        assert "Final Saved Max Title" in card["text"]
        assert "&amp;amp;" not in card["text"]
        assert "&amp;lt;" not in card["text"]
        assert "Final Saved Max Description with &lt;b&gt;Bold&lt;/b&gt; &amp; &lt;i&gt;Italic&lt;/i&gt;" in card["text"]
        assert "Текст отдельно, медиа ниже" in card["text"]
        assert "Медиа:</b> 2" in card["text"]

        # =========================================================================
        # PHASE 3: Delivery plan and execution matching saved state
        # =========================================================================
        async with db_session_maker() as session:
            scen_delivery = (await session.execute(
                select(EventScenario).where(EventScenario.id == scen.id).options(selectinload(EventScenario.media))
            )).scalar_one()

        plan = _build_delivery_plan(scen_delivery, scen.id, "Runtime MAX announcement", include_media=True)
        assert len(plan) >= 2
        assert plan[0]["kind"] == "text"
        assert plan[1]["kind"] == "media"
        assert len(plan[1]["items"]) == 2
        # Verify item order in plan matches reordered DB state
        assert plan[1]["items"][0]["max_token"] == "max_media_video_2"
        assert plan[1]["items"][0]["source_file_id"] == "max_media_video_2"
        assert plan[1]["items"][1]["max_token"] == "max_media_photo_1"
        assert plan[1]["items"][1]["source_file_id"] == "max_media_photo_1"

        delivery_transport = _RecordedMaxSession()
        delivery_client = MaxApiClient(token="test-token", base_url="https://max.test")
        delivery_client._session = delivery_transport

        # Send text and media delivery parts
        await _send_delivery_part(
            plan[0],
            platform="max",
            user_id=admin_effective_id,
            event_id=scen.id,
            is_last=False,
            bot=None,
            max_client=delivery_client,
        )
        await _send_delivery_part(
            plan[1],
            platform="max",
            user_id=admin_effective_id,
            event_id=scen.id,
            is_last=True,
            bot=None,
            max_client=delivery_client,
        )

        # Validate all delivered messages adhere to MAX transport limits
        assert len(delivery_transport.requests) == 2
        for req in delivery_transport.requests:
            body = req["body"]
            validate_max_message(body)

    finally:
        await app.states.clear(admin_effective_id)


@pytest.mark.asyncio
async def test_max_format_menu_draft_reflection_and_back(db_session_maker, monkeypatch):
    """Verifies show_format_menu in MAX marks active draft_format with radio button and preserves it across Back."""
    from max_messenger_bot.api import MaxApiClient
    from event_admin_service import create_scenario

    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)

    scen = await create_scenario(
        db_session_maker,
        code="max_fmt_draft",
        event_type="webinar",
        title="Format Draft Max Event",
        starts_at_utc=datetime.utcnow() + timedelta(days=3),
        link="https://example.test/fmt-max",
        description="Format Draft Max Description",
        media_format="separate_media_top",
    )

    client = AsyncMock(spec=MaxApiClient)
    states = MemoryStateStore()
    user_id = 999
    chat_id = 999

    # Open format menu initially - should reflect separate_media_top
    await max_admin_events.show_format_menu(client, chat_id, scen.id, states=states, user_id=user_id)
    sent_att = client.send_message.await_args_list[-1].kwargs["attachments"]
    button_texts = [b["text"] for att in sent_att if att.get("type") == "inline_keyboard" for row in att["payload"]["buttons"] for b in row]
    assert any("🔘 🖼️ Текст отдельно, медиа выше" in t for t in button_texts)

    # Start edit description so active draft context exists
    await max_admin_events.start_edit_description(client, states, chat_id, user_id, scen.id)
    await max_admin_events.receive_edit_description(client, states, chat_id, user_id, "Description draft")

    # Set draft format to separate_text_top
    await max_admin_events.set_format_handler(client, states, chat_id, user_id, scen.id, "separate_text_top")

    # Reopen format menu - should now reflect draft_format (separate_text_top)
    await max_admin_events.show_format_menu(client, chat_id, scen.id, states=states, user_id=user_id)
    sent_att = client.send_message.await_args_list[-1].kwargs["attachments"]
    button_texts = [b["text"] for att in sent_att if att.get("type") == "inline_keyboard" for row in att["payload"]["buttons"] for b in row]
    assert any("🔘 📝 Текст отдельно, медиа ниже" in t for t in button_texts)
    assert any("⚪️ 🖼️ Текст отдельно, медиа выше" in t for t in button_texts)

    # Navigate Back to edit menu
    await max_admin_events.show_edit_menu(client, chat_id, scen.id, states=states, user_id=user_id)
    sent_att = client.send_message.await_args_list[-1].kwargs["attachments"]
    edit_button_texts = [b["text"] for att in sent_att if att.get("type") == "inline_keyboard" for row in att["payload"]["buttons"] for b in row]
    assert any("Формат:" in t and "Текст отдельно, медиа ниже" in t for t in edit_button_texts)


@pytest.mark.asyncio
async def test_max_visiting_editor_does_not_block_foreign_events(db_session_maker, monkeypatch):
    """Merely opening edit menu, description editor or media editor without dirty changes does not block foreign events in MAX."""
    from max_messenger_bot.api import MaxApiClient
    from event_admin_service import create_scenario

    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)

    event_a = await create_scenario(
        db_session_maker,
        code="max_visit_a",
        event_type="webinar",
        title="Max Event A",
        starts_at_utc=datetime.utcnow() + timedelta(days=3),
        link="https://example.test/max-a",
    )
    event_b = await create_scenario(
        db_session_maker,
        code="max_visit_b",
        event_type="webinar",
        title="Max Event B",
        starts_at_utc=datetime.utcnow() + timedelta(days=4),
        link="https://example.test/max-b",
    )

    client = AsyncMock(spec=MaxApiClient)
    states = MemoryStateStore()
    user_id = 999
    chat_id = 999

    # 1. Visit Event A edit menu
    await max_admin_events.show_edit_menu(client, chat_id, event_a.id, states=states, user_id=user_id)

    # 2. Visit Event A description editor (without typing)
    await max_admin_events.start_edit_description(client, states, chat_id, user_id, event_a.id)

    # 3. Visit Event A media editor (without adding/removing media)
    await max_admin_events.show_media_editor(client, states, chat_id, user_id, event_a.id)

    # 4. Now open Event B edit menu
    await max_admin_events.show_edit_menu(client, chat_id, event_b.id, states=states, user_id=user_id)
    sent_text = client.send_message.await_args_list[-1].kwargs["text"]
    assert "Это старое меню другого мероприятия" not in sent_text
    assert "Max Event B" in sent_text

    # 5. Open Event B description editor
    await max_admin_events.start_edit_description(client, states, chat_id, user_id, event_b.id)
    sent_text = client.send_message.await_args_list[-1].kwargs["text"]
    assert "Это старое меню другого мероприятия" not in sent_text
    assert "Max Event B" in sent_text


@pytest.mark.asyncio
async def test_max_title_length_visible_and_storage_validation(db_session_maker, monkeypatch):
    """Verifies that MAX receive_create_title and receive_edit_title validate visible and storage limits."""
    from max_messenger_bot.api import MaxApiClient
    from event_admin_service import create_scenario

    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)

    scen = await create_scenario(
        db_session_maker,
        code="max_title_limits",
        event_type="webinar",
        title="Initial Valid Max Title",
        starts_at_utc=datetime.utcnow() + timedelta(days=3),
        link="https://example.test/max-limits",
    )

    client = AsyncMock(spec=MaxApiClient)
    states = MemoryStateStore()
    user_id = 999
    chat_id = 999

    # Test receive_create_title with visible text > 1000
    await max_admin_events.receive_create_title(client, states, chat_id, user_id, "A" * 1005)
    sent_text = client.send_message.await_args_list[-1].kwargs["text"]
    assert "1000 симв." in sent_text
    assert "1005" in sent_text

    # Test receive_create_title with storage overflow > 1000
    emojis = '<tg-emoji emoji-id="1234567890123456789">🔥</tg-emoji>' * 25
    raw_storage_overflow = ("B" * 50) + emojis
    await max_admin_events.receive_create_title(client, states, chat_id, user_id, raw_storage_overflow)
    sent_text = client.send_message.await_args_list[-1].kwargs["text"]
    assert "лимит хранилища (1000 симв.)" in sent_text

    # Test receive_edit_title with visible text > 1000
    await max_admin_events.start_edit_title(client, states, chat_id, user_id, scen.id)
    await max_admin_events.receive_edit_title(client, states, chat_id, user_id, "C" * 1005)
    sent_text = client.send_message.await_args_list[-1].kwargs["text"]
    assert "1000 симв." in sent_text
    assert "1005" in sent_text

    # Test receive_edit_title with storage overflow > 1000
    await max_admin_events.receive_edit_title(client, states, chat_id, user_id, raw_storage_overflow)
    sent_text = client.send_message.await_args_list[-1].kwargs["text"]
    assert "лимит хранилища (1000 симв.)" in sent_text


@pytest.mark.ui_journey
@pytest.mark.asyncio
async def test_max_premium_emoji_paginated_journey_a_to_ya_to_a(db_session_maker, monkeypatch):
    """
    MAX A -> Ya -> A journey:
    Paginated list (page 2) -> real button with title and Premium Emoji -> card ->
    archive / reschedule / copy / steps / monitoring -> back / cancel -> card ->
    Back -> original list page 2 with original filter.
    Every outgoing MAX HTTP payload is strictly validated (text <= 4000, attachments <= 12, valid JSON).
    """
    from event_admin_service import create_scenario
    from database import EventStep
    from max_messenger_bot import app as max_app_module

    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "is_admin", AsyncMock(return_value=True))

    now = datetime.utcnow()
    base_start = now + timedelta(days=20)
    # PAGE_SIZE is 6 in MAX, so 6 filler scenarios land on page 1, target scenario on page 2 (index 1)
    for idx in range(max_admin_events.PAGE_SIZE):
        await create_scenario(
            db_session_maker,
            code=f"max_prem_filler_{idx}",
            event_type="webinar",
            title=f"Filler Event {idx}",
            starts_at_utc=base_start + timedelta(days=idx + 1),
            link=f"https://example.test/max-filler/{idx}",
        )

    target_title = '<tg-emoji emoji-id="5368324170671202286">🔥</tg-emoji> Premium Q&amp;A &lt;Special Event&gt; with Long Title for Truncation Testing'
    target = await create_scenario(
        db_session_maker,
        code="max_prem_target",
        event_type="webinar",
        title=target_title,
        starts_at_utc=base_start - timedelta(days=2),
        link="https://example.test/max-target",
        description="Описание целевого мероприятия с <b>HTML</b>",
        media_format="separate_media_top",
    )

    async with db_session_maker() as session:
        step = EventStep(
            event_id=target.id,
            offset_minutes=60,
            message_text="Напоминание о вебинаре MAX",
        )
        session.add(step)
        await session.commit()

    transport = _RecordedMaxSession()
    client = MaxApiClient(token="test-token", base_url="https://max.test")
    client._session = transport
    app = MaxBotApplication(client)
    app.states = MemoryStateStore()
    update_counter = 0
    chat_id = 123
    admin_raw_id = 456

    def message_body() -> dict:
        for request in reversed(transport.requests):
            if request["url"].endswith("/messages"):
                return request["body"]
        raise AssertionError("No MAX /messages request was recorded")

    def validate_max_message(body: dict) -> dict:
        text = body.get("text") or ""
        attachments = body.get("attachments") or []
        assert len(text) <= 4000, f"MAX text length exceeded: {len(text)} > 4000"
        assert len(attachments) <= 12, f"MAX attachments limit exceeded: {len(attachments)} > 12"
        json.loads(json.dumps(body, ensure_ascii=False))
        return body

    def button_payload(body: dict, prefix: str) -> str:
        for attachment in body.get("attachments") or []:
            for row in attachment.get("payload", {}).get("buttons", []):
                for button in row:
                    payload = button.get("payload")
                    if payload and payload.startswith(prefix):
                        return payload
        raise AssertionError(f"No visible MAX callback with prefix {prefix!r} in {body.get('attachments')}")

    def find_button(body: dict, prefix: str) -> dict:
        for attachment in body.get("attachments") or []:
            for row in attachment.get("payload", {}).get("buttons", []):
                for button in row:
                    payload = button.get("payload")
                    if payload and payload.startswith(prefix):
                        return button
        raise AssertionError(f"No visible MAX button with prefix {prefix!r} in {body.get('attachments')}")

    async def press(payload: str) -> dict:
        nonlocal update_counter
        update_counter += 1
        transport.requests.clear()
        await app.handle_update(
            {
                "update_type": "message_callback",
                "update_id": f"max-journey-cb-{update_counter}",
                "callback": {
                    "callback_id": f"max-journey-callback-{update_counter}",
                    "payload": payload,
                    "user": {"user_id": admin_raw_id, "first_name": "Admin", "username": "admin"},
                },
                "message": {
                    "mid": f"max-journey-message-{update_counter}",
                    "recipient": {"chat_id": chat_id},
                    "body": {"text": "admin"},
                },
            }
        )
        return validate_max_message(message_body())

    try:
        # 1. Open scenarios list
        listing = await press("admin_events")
        assert "Страница 1 из 2" in listing["text"]

        # Navigate to page 2 (0-indexed page 1)
        listing = await press(button_payload(listing, "admin_ev_page_1_0"))
        assert "Страница 2 из 2" in listing["text"]

        # 2. Inspect target button on page 2: plain text, Unicode fallback emoji, no raw HTML tags
        target_btn = find_button(listing, f"admin_ev_open_{target.id}_1_0")
        btn_text = target_btn["text"]
        assert "<tg-emoji" not in btn_text
        assert "</tg-emoji>" not in btn_text
        assert "&amp;" not in btn_text
        assert "&lt;" not in btn_text
        assert "🔥" in btn_text
        assert "Premium Q&A <Special Event>" in btn_text
        assert btn_text.endswith("… (" + target.starts_at_msk[:10] + ")")

        # 3. Open scenario card
        card = await press(target_btn["payload"])
        card_text = card["text"]
        # MAX card strips custom emoji tags and renders fallback emoji in HTML
        assert "<tg-emoji" not in card_text
        assert "&lt;tg-emoji" not in card_text
        assert "&amp;amp;" not in card_text
        assert "&amp;lt;" not in card_text
        assert "🔥" in card_text
        assert "Premium Q&amp;A &lt;Special Event&gt;" in card_text

        # 4. Nested screen: Archive
        archive_btn = button_payload(card, f"admin_ev_archive_ask_{target.id}")
        archive_screen = await press(archive_btn)
        arch_text = archive_screen["text"]
        assert "<tg-emoji" not in arch_text
        assert "&lt;tg-emoji" not in arch_text
        assert "&amp;amp;" not in arch_text
        assert "🔥" in arch_text
        assert "Premium Q&amp;A &lt;Special Event&gt;" in arch_text
        # Back to card
        card_back = button_payload(archive_screen, f"admin_ev_open_{target.id}_1_0")
        card = await press(card_back)

        # 5. Nested screen: Reschedule
        resched_btn = button_payload(card, f"admin_ev_reschedule_{target.id}")
        resched_screen = await press(resched_btn)
        resched_text = resched_screen["text"]
        assert "<tg-emoji" not in resched_text
        assert "&lt;tg-emoji" not in resched_text
        assert "&amp;amp;" not in resched_text
        assert "🔥" in resched_text
        assert "Premium Q&amp;A &lt;Special Event&gt;" in resched_text
        # Back to card
        card_back = button_payload(resched_screen, f"admin_ev_open_{target.id}_1_0")
        card = await press(card_back)

        # 6. Nested screen: Copy
        copy_btn = button_payload(card, f"admin_ev_copy_{target.id}")
        copy_screen = await press(copy_btn)
        copy_text = copy_screen["text"]
        assert "<tg-emoji" not in copy_text
        assert "&lt;tg-emoji" not in copy_text
        assert "&amp;amp;" not in copy_text
        assert "🔥" in copy_text
        assert "Premium Q&amp;A &lt;Special Event&gt;" in copy_text
        # Back to card
        card_back = button_payload(copy_screen, f"admin_ev_open_{target.id}_1_0")
        card = await press(card_back)

        # 7. Nested screen: Steps
        steps_btn = button_payload(card, f"admin_ev_steps_{target.id}_1_0")
        steps_screen = await press(steps_btn)
        steps_text = steps_screen["text"]
        assert "<tg-emoji" not in steps_text
        assert "&lt;tg-emoji" not in steps_text
        assert "&amp;amp;" not in steps_text
        assert "🔥" in steps_text
        assert "Premium Q&amp;A &lt;Special Event&gt;" in steps_text
        # Back to card
        card_back = button_payload(steps_screen, f"admin_ev_open_{target.id}_1_0")
        card = await press(card_back)

        # 8. Nested screen: Monitoring / Stats
        stats_btn = button_payload(card, f"admin_ev_stats_{target.id}")
        stats_screen = await press(stats_btn)
        stats_text = stats_screen["text"]
        assert "<tg-emoji" not in stats_text
        assert "&lt;tg-emoji" not in stats_text
        assert "&amp;amp;" not in stats_text
        assert "🔥" in stats_text
        assert "Premium Q&amp;A &lt;Special Event&gt;" in stats_text
        # Back to card
        card_back = button_payload(stats_screen, f"admin_ev_open_{target.id}_1_0")
        card = await press(card_back)

        # 9. Return to list (Page 2) with original filter
        to_list_btn = button_payload(card, "admin_ev_page_1_0")
        listing = await press(to_list_btn)
        assert "Страница 2 из 2" in listing["text"]
        assert button_payload(listing, f"admin_ev_open_{target.id}_1_0") is not None
    finally:
        await app.states.clear(MAX_ID_OFFSET + admin_raw_id)


@pytest.mark.asyncio
async def test_max_scenarios_list_button_preserves_raw_legacy_and_escaped_titles(db_session_maker, monkeypatch):
    """
    Verifies that MAX scenario list buttons preserve literal angle brackets like '<Новый сезон>'
    both for raw legacy titles and HTML-escaped titles, convert custom emojis to Unicode fallback,
    and strip only allowed markup tags.
    """
    from event_admin_service import create_scenario
    from max_messenger_bot import app as max_app_module

    monkeypatch.setattr(max_admin_events, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module, "async_session_maker", db_session_maker)
    monkeypatch.setattr(max_app_module.common, "is_admin", AsyncMock(return_value=True))

    now = datetime.utcnow()
    # 1. Raw legacy scenario with literal angle brackets
    scen_raw = await create_scenario(
        db_session_maker,
        code="max_raw_legacy",
        event_type="webinar",
        title="Презентация <Новый сезон>",
        starts_at_utc=now + timedelta(days=2),
        link="https://example.test/raw",
    )
    # 2. Escaped scenario with HTML entities
    scen_escaped = await create_scenario(
        db_session_maker,
        code="max_escaped_title",
        event_type="webinar",
        title="Презентация &lt;Новый сезон&gt;",
        starts_at_utc=now + timedelta(days=3),
        link="https://example.test/escaped",
    )
    # 3. Custom emoji with literal angle brackets
    scen_emoji = await create_scenario(
        db_session_maker,
        code="max_emoji_raw",
        event_type="webinar",
        title='<tg-emoji emoji-id="5368324170671202286">🔥</tg-emoji> Презентация &lt;Новый сезон&gt;',
        starts_at_utc=now + timedelta(days=4),
        link="https://example.test/emoji",
    )

    transport = _RecordedMaxSession()
    client = MaxApiClient(token="test-token", base_url="https://max.test")
    client._session = transport
    app = MaxBotApplication(client)
    app.states = MemoryStateStore()
    chat_id = 123
    admin_raw_id = 456

    def message_body() -> dict:
        for request in reversed(transport.requests):
            if request["url"].endswith("/messages"):
                return request["body"]
        raise AssertionError("No MAX /messages request was recorded")

    def find_button(body: dict, prefix: str) -> dict:
        for attachment in body.get("attachments") or []:
            for row in attachment.get("payload", {}).get("buttons", []):
                for button in row:
                    payload = button.get("payload")
                    if payload and payload.startswith(prefix):
                        return button
        raise AssertionError(f"No visible MAX button with prefix {prefix!r} in {body.get('attachments')}")

    try:
        transport.requests.clear()
        await app.handle_update(
            {
                "update_type": "message_callback",
                "update_id": "max-legacy-title-cb",
                "callback": {
                    "callback_id": "max-legacy-title-callback",
                    "payload": "admin_events",
                    "user": {"user_id": admin_raw_id, "first_name": "Admin", "username": "admin"},
                },
                "message": {
                    "mid": "max-legacy-title-msg",
                    "recipient": {"chat_id": chat_id},
                    "body": {"text": "admin"},
                },
            }
        )
        body = message_body()

        # Raw legacy button check
        btn_raw = find_button(body, f"admin_ev_open_{scen_raw.id}")
        assert "<Новый сезон>" in btn_raw["text"]
        assert "Презентация <Новый сезон>" in btn_raw["text"]

        # Escaped button check
        btn_escaped = find_button(body, f"admin_ev_open_{scen_escaped.id}")
        assert "<Новый сезон>" in btn_escaped["text"]
        assert "Презентация <Новый сезон>" in btn_escaped["text"]
        assert "&lt;" not in btn_escaped["text"]
        assert "&gt;" not in btn_escaped["text"]

        # Emoji + literal angle brackets button check
        btn_emoji = find_button(body, f"admin_ev_open_{scen_emoji.id}")
        assert "🔥" in btn_emoji["text"]
        assert "<Новый сезон>" in btn_emoji["text"]
        assert "<tg-emoji" not in btn_emoji["text"]
    finally:
        await app.states.clear(MAX_ID_OFFSET + admin_raw_id)


