import os
import re
import json
import html
from typing import Any
from functools import wraps
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock

os.environ.setdefault("BOT_TOKEN", "test_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from aiogram import Bot, Dispatcher
from aiogram.dispatcher.event.bases import UNHANDLED
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.storage.base import StorageKey
from aiogram.methods import AnswerCallbackQuery, EditMessageText, SendMessage
from aiogram.types import CallbackQuery, Chat, Message, MessageEntity, Update, User as TgUser
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
import event_admin_telegram
from event_admin_telegram import router as event_admin_router
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from time_helpers import utc_now
from event_admin_service import DeliveryItem, DeliveryPartItem, DeliveryStats, ScenarioSummary, StepItem


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


def _future_msk_text(days: int = 30, hour: int = 19, minute: int = 0) -> str:
    value = (datetime.utcnow() + timedelta(days=days)).replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    )
    return value.strftime("%d.%m.%Y %H:%M")


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
async def test_telegram_pagination_preserves_archive_filter(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    storage = MemoryStorage()
    state = FSMContext(storage=storage, key=MagicMock(bot_id=1, chat_id=12345, user_id=12345))
    await state.update_data(include_archived=True)
    rendered = AsyncMock()
    monkeypatch.setattr(event_admin_telegram, "_show_scenarios_page", rendered)

    await event_admin_telegram.show_events_page(_make_callback("admin_ev_page_2"), state)

    rendered.assert_awaited_once_with(ANY, page=2, include_archived=True)
    assert (await state.get_data())["include_archived"] is True


def _telegram_event_callbacks(markup) -> list[str]:
    return [
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if button.callback_data
    ]


_TELEGRAM_EVENT_BUTTON_CONTRACTS = (
    (re.compile(r"admin_events$"), "navigation", "events/list", "telegram.events.root", "show_events_root", None),
    (re.compile(r"admin_panel$"), "navigation", "admin/panel", "telegram.events.root", "back_to_admin_panel", None),
    (re.compile(r"admin_ev_create_(?:webinar|meeting)$"), "navigation", "events/create", "telegram.events.root", "start_scenario_creation", "EventAdminStates.create_code"),
    (re.compile(r"admin_ev_skip_meeting_link$"), "mutation", "events/create", "telegram.events.root", "skip_meeting_link", None),
    (re.compile(r"admin_ev_page_\d+_[01]$"), "navigation", "events/list", "telegram.events.pagination", "show_events_page", None),
    (re.compile(r"admin_ev_toggle_archive_view_[01]$"), "navigation", "events/list", "telegram.events.pagination", "toggle_archive_view", None),
    (re.compile(r"admin_ev_open_\d+(?:_\d+_[01])?$"), "navigation", "events/card", "telegram.events.card", "open_scenario_card", None),
    (re.compile(r"admin_ev_toggle_media_order_\d+$"), "mutation", "events/edit", "telegram.description-media", "toggle_active_handler", None),
    (re.compile(r"admin_ev_toggle_\d+$"), "mutation", "events/card", "telegram.events.card", "toggle_active_handler", None),
    (re.compile(r"admin_ev_stats_\d+$"), "navigation", "events/stats", "telegram.delivery.lifecycle", "show_delivery_stats", None),
    (re.compile(r"admin_ev_reschedule_\d+$"), "mutation", "events/reschedule", "telegram.events.card", "start_reschedule", "EventAdminStates.reschedule_starts_at"),
    (re.compile(r"admin_ev_edit_menu_\d+(?:_\d+_[01])?$"), "navigation", "events/edit", "telegram.events.edit", "show_edit_menu", None),
    (re.compile(r"admin_ev_copy_\d+$"), "navigation", "events/copy", "telegram.events.card", "start_copy", "EventAdminStates.copy_code"),
    (re.compile(r"admin_ev_test_send_menu_\d+$"), "navigation", "events/test-send", "telegram.delivery.test-send", "show_test_send_picker", None),
    (re.compile(r"admin_ev_archive_ask_\d+$"), "navigation", "events/archive", "telegram.delivery.lifecycle", "ask_archive_handler", None),
    (re.compile(r"admin_ev_archive_yes_\d+$"), "destructive", "events/archive", "telegram.delivery.lifecycle", "archive_yes_handler", None),
    (re.compile(r"admin_ev_edit_title_\d+$"), "mutation", "events/edit", "telegram.events.edit", "start_edit_title", "EventAdminStates.edit_title"),
    (re.compile(r"admin_ev_edit_duration_\d+$"), "mutation", "events/edit", "telegram.events.edit", "start_edit_duration", "EventAdminStates.edit_duration"),
    (re.compile(r"admin_ev_edit_link_\d+$"), "mutation", "events/edit", "telegram.events.edit", "start_edit_link", "EventAdminStates.edit_link"),
    (re.compile(r"admin_ev_edit_place_\d+$"), "mutation", "events/edit", "telegram.events.edit", "start_edit_place", "EventAdminStates.edit_place"),
    (re.compile(r"admin_ev_format_menu_\d+$"), "navigation", "events/edit-format", "telegram.events.edit", "show_format_menu", None),
    (re.compile(r"admin_ev_set_fmt_\d+_[a-z_]+$"), "mutation", "events/edit-format", "telegram.events.edit", "set_format_handler", None),
    (re.compile(r"admin_ev_preview_tg_\d+$"), "navigation", "events/preview-tg", "telegram.events.edit", "show_telegram_preview", None),
    (re.compile(r"admin_ev_preview_max_\d+$"), "navigation", "events/preview-max", "telegram.events.edit", "show_max_preview", None),
    (re.compile(r"admin_ev_edit_description_\d+$"), "mutation", "events/edit", "telegram.events.edit", "start_edit_description", "EventAdminStates.edit_description"),
    (re.compile(r"admin_ev_edit_media_\d+$"), "navigation", "events/edit-media", "telegram.description-media", "start_edit_media", "EventAdminStates.edit_media"),
    (re.compile(r"admin_ev_description_save_\d+$"), "mutation", "events/edit", "telegram.description-media", "save_description", None),
    (re.compile(r"admin_ev_cancel_draft_\d+$"), "destructive", "events/edit", "telegram.events.edit", "cancel_draft_handler", None),
    (re.compile(r"admin_ev_description_delete_\d+$"), "destructive", "events/edit", "telegram.description-media", "delete_description", "EventAdminStates.edit_description"),
    (re.compile(r"admin_ev_media_add_\d+$"), "navigation", "events/edit-media", "telegram.description-media", "start_add_media", "EventAdminStates.edit_media_add"),
    (re.compile(r"admin_ev_media_save_\d+$"), "mutation", "events/edit-media", "telegram.description-media", "save_media", None),
    (re.compile(r"admin_ev_media_remove_\d+_\d+$"), "destructive", "events/edit-media", "telegram.description-media", "remove_media", "EventAdminStates.edit_media"),
    (re.compile(r"admin_ev_media_up_\d+_\d+$"), "mutation", "events/edit-media", "telegram.description-media", "move_media_up", "EventAdminStates.edit_media"),
    (re.compile(r"admin_ev_media_down_\d+_\d+$"), "mutation", "events/edit-media", "telegram.description-media", "move_media_down", "EventAdminStates.edit_media"),
    (re.compile(r"admin_ev_steps_\d+(?:_\d+_[01])?$"), "navigation", "events/steps", "telegram.steps.roundtrip", "show_steps_list", None),
    (re.compile(r"admin_ev_step_add_\d+$"), "navigation", "events/steps", "telegram.steps.roundtrip", "start_add_step", None),
    (re.compile(r"admin_ev_offset_preset_\d+_-?\d+$"), "mutation", "events/steps", "telegram.steps.roundtrip", "process_offset_preset", "EventAdminStates.step_text"),
    (re.compile(r"admin_ev_offset_custom_\d+$"), "navigation", "events/steps", "telegram.steps.roundtrip", "start_offset_custom", "EventAdminStates.step_offset_custom"),
    (re.compile(r"admin_ev_step_\d+(?:_\d+_[01])?$"), "navigation", "events/step", "telegram.steps.roundtrip", "open_step_card", None),
    (re.compile(r"admin_ev_step_toggle_details_\d+$"), "mutation", "events/step", "telegram.steps.dispatcher", "toggle_step_event_details", None),
    (re.compile(r"admin_ev_step_edit_text_\d+$"), "mutation", "events/step", "telegram.steps.dispatcher", "start_edit_step_text", "EventAdminStates.step_edit_text"),
    (re.compile(r"admin_ev_step_edit_offset_\d+$"), "navigation", "events/step", "telegram.steps.dispatcher", "start_edit_step_offset", None),
    (re.compile(r"admin_ev_step_up_\d+$"), "mutation", "events/step", "telegram.steps.dispatcher", "step_move_up", None),
    (re.compile(r"admin_ev_step_down_\d+$"), "mutation", "events/step", "telegram.steps.dispatcher", "step_move_down", None),
    (re.compile(r"admin_ev_step_del_ask_\d+$"), "navigation", "events/step", "telegram.steps.dispatcher", "ask_delete_step", None),
    (re.compile(r"admin_ev_step_del_yes_\d+$"), "destructive", "events/step", "telegram.steps.dispatcher", "delete_step_yes", None),
    (re.compile(r"admin_ev_step_edit_preset_\d+_-?\d+$"), "mutation", "events/step", "telegram.steps.dispatcher", "process_edit_offset_preset", None),
    (re.compile(r"admin_ev_step_edit_custom_\d+$"), "navigation", "events/step", "telegram.steps.dispatcher", "start_edit_offset_custom", "EventAdminStates.step_edit_offset_custom"),
    (re.compile(r"admin_ev_problems_\d+_\d+$"), "navigation", "events/problems", "telegram.delivery.lifecycle", "show_problem_deliveries", None),
    (re.compile(r"admin_ev_problem_item_\d+(?:_\d+)?$"), "navigation", "events/problem", "telegram.delivery.lifecycle", "open_problem_delivery", None),
    (re.compile(r"admin_ev_resolve_(?:force|sent|skipped)_\d+(?:_\d+)?$"), "mutation", "events/problem", "telegram.delivery.resolution", "prompt_resolution_reason", "EventAdminStates.resolve_reason"),
    (re.compile(r"admin_ev_test_step_\d+$"), "external", "telegram/test-send", "telegram.delivery.test-send", "test_send_single_step", None),
    (re.compile(r"admin_ev_do_test_send_\d+_\d+$"), "external", "telegram/test-send", "telegram.delivery.test-send", "do_test_send", None),
)

_TELEGRAM_EVENT_JOURNEY_COVERAGE = {
    "telegram.events.root",
    "telegram.events.pagination",
    "telegram.events.card",
    "telegram.events.edit",
    "telegram.description-media",
    "telegram.steps.roundtrip",
    "telegram.steps.dispatcher",
    "telegram.delivery.lifecycle",
    "telegram.delivery.resolution",
    "telegram.delivery.test-send",
}

_TELEGRAM_REAL_JOURNEY_TESTS = {
    "telegram.events.root": (
        "test_telegram_admin_root_list_and_card",
        "test_telegram_webinar_creation_flow",
        "test_telegram_meeting_creation_flow",
    ),
    "telegram.events.pagination": (
        "test_telegram_pagination_preserves_archive_filter",
        "test_telegram_pagination_boundaries_for_scenarios_and_problems",
    ),
    "telegram.events.card": (
        "test_telegram_admin_root_list_and_card",
        "test_telegram_reschedule_and_copy",
        "test_telegram_dispatcher_lifecycle_monitoring_and_test_send",
    ),
    "telegram.events.edit": (
        "test_telegram_description_and_media_edit_routes",
        "test_telegram_paginated_edit_roundtrip_and_delete_stays_draft",
        "test_telegram_dispatcher_description_chunks_save_reopen_and_runtime_split",
    ),
    "telegram.description-media": (
        "test_telegram_description_and_media_edit_routes",
        "test_telegram_media_back_preserves_draft_and_cancel_discards_it",
        "test_telegram_paginated_edit_roundtrip_and_delete_stays_draft",
    ),
    "telegram.steps.roundtrip": (
        "test_telegram_dispatcher_steps_buttons_roundtrip_page_context",
        "test_telegram_steps_management",
    ),
    "telegram.steps.dispatcher": (
        "test_telegram_dispatcher_step_routing_and_ordering",
        "test_telegram_dispatcher_delete_last_step_returns_empty_page",
    ),
    "telegram.delivery.lifecycle": (
        "test_telegram_dispatcher_lifecycle_monitoring_and_test_send",
        "test_telegram_monitoring_and_uncertain_resolution",
    ),
    "telegram.delivery.resolution": (
        "test_telegram_monitoring_and_uncertain_resolution",
        "test_telegram_uncertain_resolution_buttons_target_specific_part",
    ),
    "telegram.delivery.test-send": (
        "test_telegram_test_send_to_self",
        "test_telegram_test_send_uses_runtime_media_plan_without_delivery_rows",
    ),
}

_TELEGRAM_EXECUTED_JOURNEYS: set[str] = set()


@pytest.fixture(autouse=True)
def _record_telegram_real_journey(request):
    for journey_id, test_names in _TELEGRAM_REAL_JOURNEY_TESTS.items():
        if request.node.name in test_names:
            _TELEGRAM_EXECUTED_JOURNEYS.add(journey_id)
    yield


@pytest.fixture(scope="module", autouse=True)
def _assert_telegram_real_journey_coverage(request):
    yield
    required_tests = {name for names in _TELEGRAM_REAL_JOURNEY_TESTS.values() for name in names}
    selected_tests = {item.name for item in request.session.items}
    if required_tests <= selected_tests:
        assert _TELEGRAM_EXECUTED_JOURNEYS == _TELEGRAM_EVENT_JOURNEY_COVERAGE


def _event_callback_contract(callback: str) -> dict[str, str]:
    for pattern, classification, target, journey_id, route, expected_state in _TELEGRAM_EVENT_BUTTON_CONTRACTS:
        if pattern.fullmatch(callback):
            assert journey_id in _TELEGRAM_EVENT_JOURNEY_COVERAGE
            assert classification in {"navigation", "mutation", "destructive", "external"}
            return {
                "classification": classification,
                "target": target,
                "journey_id": journey_id,
                "route": route,
                "expected_state": expected_state,
            }
    raise AssertionError(f"Uncontracted event callback: {callback}")


def _assert_event_callback_contract(callbacks: list[str]) -> None:
    for callback in callbacks:
        _event_callback_contract(callback)


@pytest.mark.ui_journey
def test_telegram_event_visible_button_contract_covers_changed_screens():
    scenario = ScenarioSummary(
        id=10,
        code="event_contract",
        event_type="meeting",
        title="Contract meeting",
        starts_at=datetime.utcnow(),
        starts_at_msk="01.01.2030 12:00",
        duration_minutes=90,
        place="Room 1",
        link=None,
        is_active=True,
        is_archived=False,
        schedule_revision=1,
        content_version=1,
        registrations_count=2,
        description="Description",
        media_order="media_top",
        media_count=2,
    )
    step = StepItem(
        id=20,
        event_id=10,
        offset_minutes=-60,
        offset_display="1 ч. до начала",
        message_text="Reminder",
        step_version=1,
        sort_order=0,
        is_archived=False,
        include_event_details=True,
    )
    delivery = DeliveryItem(
        id=30,
        event_id=10,
        user_id=40,
        user_display="User",
        message_kind="immediate_confirmation",
        step_id=None,
        status="uncertain",
        due_at_msk="01.01.2030 11:00",
        attempt_count=1,
        last_error="timeout",
        sent_or_finished_msk=None,
        platform="telegram",
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
    keyboards = [
        event_admin_telegram._scenarios_list_keyboard([scenario], 0, 2, True),
        event_admin_telegram._scenario_card_keyboard(scenario, "admin_ev_page_1_1"),
        event_admin_telegram._scenario_edit_keyboard(scenario, list_back="admin_ev_page_1_1"),
        event_admin_telegram._steps_list_keyboard(10, [step], False, back="admin_ev_page_1_1"),
        event_admin_telegram._step_card_keyboard(10, step, False, False, False, back="admin_ev_page_1_1"),
        event_admin_telegram._step_offset_picker_keyboard(10, "admin_ev_steps_10_1_1"),
        event_admin_telegram._step_edit_offset_picker_keyboard(20),
        event_admin_telegram._media_editor_keyboard(
            10,
            [
                {"file_type": "photo", "file_id": "photo-1", "max_token": None, "source_platform": "telegram"},
                {"file_type": "video", "file_id": "video-1", "max_token": None, "source_platform": "telegram"},
            ],
        ),
        event_admin_telegram._delivery_stats_keyboard(10, DeliveryStats(uncertain=1)),
        event_admin_telegram._problem_deliveries_keyboard(10, [delivery], 0, 2),
        event_admin_telegram._delivery_resolution_keyboard(delivery, [part]),
        event_admin_telegram._test_send_picker_keyboard(10, [step], back="admin_ev_page_1_1"),
        event_admin_telegram.InlineKeyboardBuilder().button(
            text="Пропустить ссылку", callback_data="admin_ev_skip_meeting_link"
        ).as_markup(),
    ]
    _assert_event_callback_contract(
        [callback for keyboard in keyboards for callback in _telegram_event_callbacks(keyboard)]
    )


@pytest.mark.ui_journey
@pytest.mark.asyncio
async def test_telegram_visible_event_buttons_dispatch_through_router_contract(monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    scenario = ScenarioSummary(
        id=10,
        code="event_dispatch_contract",
        event_type="meeting",
        title="Contract meeting",
        starts_at=datetime.utcnow(),
        starts_at_msk="01.01.2030 12:00",
        duration_minutes=90,
        place="Room 1",
        link=None,
        is_active=True,
        is_archived=False,
        schedule_revision=1,
        content_version=1,
        registrations_count=2,
        description="Description",
        media_order="media_top",
        media_count=2,
    )
    step = StepItem(
        id=20,
        event_id=10,
        offset_minutes=-60,
        offset_display="1 ч. до начала",
        message_text="Reminder",
        step_version=1,
        sort_order=0,
        is_archived=False,
        include_event_details=True,
    )
    delivery = DeliveryItem(
        id=30,
        event_id=10,
        user_id=40,
        user_display="User",
        message_kind="immediate_confirmation",
        step_id=None,
        status="uncertain",
        due_at_msk="01.01.2030 11:00",
        attempt_count=1,
        last_error="timeout",
        sent_or_finished_msk=None,
        platform="telegram",
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
    keyboards = [
        event_admin_telegram._scenarios_list_keyboard([scenario], 0, 2, True),
        event_admin_telegram._scenario_card_keyboard(scenario, "admin_ev_page_1_1"),
        event_admin_telegram._scenario_edit_keyboard(scenario, list_back="admin_ev_page_1_1"),
        event_admin_telegram._steps_list_keyboard(10, [step], False, back="admin_ev_page_1_1"),
        event_admin_telegram._step_card_keyboard(10, step, False, False, False, back="admin_ev_page_1_1"),
        event_admin_telegram._step_offset_picker_keyboard(10, "admin_ev_steps_10_1_1"),
        event_admin_telegram._step_edit_offset_picker_keyboard(20),
        event_admin_telegram._media_editor_keyboard(
            10,
            [
                {"file_type": "photo", "file_id": "photo-1", "max_token": None, "source_platform": "telegram"},
                {"file_type": "video", "file_id": "video-1", "max_token": None, "source_platform": "telegram"},
            ],
        ),
        event_admin_telegram._delivery_stats_keyboard(10, DeliveryStats(uncertain=1)),
        event_admin_telegram._problem_deliveries_keyboard(10, [delivery], 0, 2),
        event_admin_telegram._delivery_resolution_keyboard(delivery, [part]),
        event_admin_telegram._test_send_picker_keyboard(10, [step], back="admin_ev_page_1_1"),
        event_admin_telegram.InlineKeyboardBuilder().button(
            text="Пропустить ссылку", callback_data="admin_ev_skip_meeting_link"
        ).as_markup(),
    ]
    callbacks = [callback for keyboard in keyboards for callback in _telegram_event_callbacks(keyboard)]
    assert callbacks
    contracts = {callback: _event_callback_contract(callback) for callback in callbacks}

    import handlers

    routers = (event_admin_router, handlers.router)
    original_callbacks = {
        router: [handler.callback for handler in router.callback_query.handlers]
        for router in routers
    }
    routed: list[str] = []

    def make_probe(handler_name: str):
        async def probe(*args, **kwargs):
            callback = kwargs.get("callback") or (args[0] if args else None)
            assert callback is not None
            routed.append(handler_name)
            await callback.message.edit_text(f"route:{handler_name}")

        return probe

    try:
        for router in routers:
            for handler in router.callback_query.handlers:
                handler.callback = make_probe(handler.callback.__name__)
            router._parent_router = None
        dp = Dispatcher(storage=MemoryStorage())
        for router in routers:
            dp.include_router(router)
        bot = AsyncMock(spec=Bot)
        bot.id = 123456789
        for callback in callbacks:
            routed.clear()
            bot.reset_mock()
            user = TgUser(id=12345, is_bot=False, first_name="Admin", username="admin_user")
            chat = Chat(id=12345, type="private")
            message = Message(message_id=100, date=datetime.now(), chat=chat, text="Original text")
            query = CallbackQuery(
                id=f"contract-{callback}",
                from_user=user,
                chat_instance="contract",
                data=callback,
                message=message,
            )
            result = await dp.feed_update(bot, Update(update_id=100, callback_query=query))
            assert result is not UNHANDLED, f"Visible callback was not routed: {callback}"
            assert routed == [contracts[callback]["route"]], f"Visible callback selected {routed}: {callback}"
            methods = [call.args[0] for call in bot.call_args_list if call.args]
            assert any(isinstance(method, EditMessageText) for method in methods), callback
    finally:
        for router in routers:
            for handler, original in zip(router.callback_query.handlers, original_callbacks[router]):
                handler.callback = original
            router._parent_router = None


@pytest.mark.ui_journey
@pytest.mark.asyncio
async def test_telegram_visible_event_buttons_execute_real_dispatcher_journeys(db_session_maker, monkeypatch):
    """Every changed visible callback executes its registered handler unpatched."""
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import add_step, create_scenario, get_scenario_summary, list_scenario_media, list_steps, replace_scenario_media

    scenario = await create_scenario(
        db_session_maker,
        code="telegram_real_contract",
        event_type="meeting",
        title="Telegram real contract",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        place="Room 1",
    )
    first_step = await add_step(db_session_maker, scenario.id, -1440, "First reminder")
    second_step = await add_step(db_session_maker, scenario.id, -60, "Second reminder")
    await replace_scenario_media(
        db_session_maker,
        scenario.id,
        [
            {"file_type": "photo", "file_id": "telegram-photo", "source_platform": "telegram"},
            {"file_type": "video", "file_id": "telegram-video", "source_platform": "telegram"},
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
            delivery_key="telegram_real_contract_delivery",
            due_at=utc_now(),
            next_attempt_at=utc_now(),
            platform="telegram",
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
            payload_json=json.dumps({"kind": "photo", "file_id": "telegram-photo"}),
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
        platform="telegram",
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
    keyboards = [
        event_admin_telegram._scenarios_list_keyboard([scenario], 0, 2, True),
        event_admin_telegram._scenario_card_keyboard(scenario, "admin_ev_page_1_1"),
        event_admin_telegram._scenario_edit_keyboard(scenario, list_back="admin_ev_page_1_1"),
        event_admin_telegram._steps_list_keyboard(scenario.id, steps, False, back="admin_ev_page_1_1"),
        event_admin_telegram._step_card_keyboard(scenario.id, steps[0], False, False, False, back="admin_ev_page_1_1"),
        event_admin_telegram._step_offset_picker_keyboard(scenario.id, f"admin_ev_steps_{scenario.id}_1_1"),
        event_admin_telegram._step_edit_offset_picker_keyboard(steps[0].id),
        event_admin_telegram._media_editor_keyboard(scenario.id, [
            {"file_type": item.file_type, "file_id": item.file_id, "max_token": item.max_token, "source_platform": item.source_platform}
            for item in media
        ]),
        event_admin_telegram._delivery_stats_keyboard(scenario.id, DeliveryStats(uncertain=1)),
        event_admin_telegram._problem_deliveries_keyboard(scenario.id, [delivery_item], 0, 2),
        event_admin_telegram._delivery_resolution_keyboard(delivery_item, [part_item]),
        event_admin_telegram._test_send_picker_keyboard(scenario.id, steps, back="admin_ev_page_1_1"),
        event_admin_telegram.InlineKeyboardBuilder().button(
            text="Пропустить ссылку", callback_data="admin_ev_skip_meeting_link"
        ).as_markup(),
    ]
    callbacks = list(dict.fromkeys(
        callback for keyboard in keyboards for callback in _telegram_event_callbacks(keyboard)
    ))
    contracts = {callback: _event_callback_contract(callback) for callback in callbacks}

    import handlers

    routers = (event_admin_router, handlers.router)
    original_callbacks = {
        router: [handler.callback for handler in router.callback_query.handlers]
        for router in routers
    }
    original_parents = {router: router._parent_router for router in routers}
    routed: list[str] = []

    def wrap_handler(original):
        @wraps(original)
        async def wrapped(*args, **kwargs):
            routed.append(original.__name__)
            return await original(*args, **kwargs)

        return wrapped

    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    for router in routers:
        for handler in router.callback_query.handlers:
            handler.callback = wrap_handler(handler.callback)
        router._parent_router = None
        dp.include_router(router)
    state = FSMContext(storage=storage, key=StorageKey(bot_id=bot.id, chat_id=12345, user_id=12345))

    async def prepare_state(callback: str) -> None:
        await state.clear()
        event_id = scenario.id
        if callback == "admin_ev_skip_meeting_link":
            await state.update_data(
                create_event_type="meeting",
                create_code="telegram_real_created",
                create_title="Created from route",
                create_starts_at=(datetime.utcnow() + timedelta(days=3)).isoformat(),
                create_duration=120,
                create_place="Room 1",
            )
            await state.set_state(event_admin_telegram.EventAdminStates.create_meeting_link)
        elif callback.startswith("admin_ev_description_save_"):
            await state.update_data(edit_event_id=event_id, description_draft="Updated description", event_list_back="admin_ev_page_1_1")
            await state.set_state(event_admin_telegram.EventAdminStates.edit_description)
        elif callback.startswith("admin_ev_media_"):
            await state.update_data(
                edit_event_id=event_id,
                media_draft=[
                    {"file_type": "photo", "file_id": "telegram-photo", "max_token": None, "source_platform": "telegram"},
                    {"file_type": "video", "file_id": "telegram-video", "max_token": None, "source_platform": "telegram"},
                ],
                media_draft_dirty=True,
                event_list_back="admin_ev_page_1_1",
            )
            await state.set_state(event_admin_telegram.EventAdminStates.edit_media)
        elif callback.startswith("admin_ev_offset_"):
            await state.update_data(step_event_id=event_id, event_list_back="admin_ev_page_1_1")
        elif callback.startswith("admin_ev_step_edit_preset_"):
            await state.update_data(edit_step_id=first_step.id, event_list_back="admin_ev_page_1_1")
        elif callback.startswith("admin_ev_step_edit_custom_"):
            await state.update_data(edit_step_id=first_step.id, event_list_back="admin_ev_page_1_1")

    try:
        for callback in callbacks:
            await prepare_state(callback)
            routed.clear()
            methods = await _feed_callback(dp, bot, callback)
            assert routed == [contracts[callback]["route"]], f"Unexpected real handler for {callback}: {routed}"
            assert any(isinstance(method, (EditMessageText, SendMessage, AnswerCallbackQuery)) for method in methods), callback
            expected_state = contracts[callback]["expected_state"]
            if expected_state is not None and expected_state.startswith("EventAdminStates."):
                assert await state.get_state() == getattr(event_admin_telegram.EventAdminStates, expected_state.rsplit(".", 1)[1]).state
    finally:
        for router in routers:
            for handler, original in zip(router.callback_query.handlers, original_callbacks[router]):
                handler.callback = original
            router._parent_router = original_parents[router]


@pytest.mark.asyncio
async def test_telegram_paginated_card_mutations_and_stats_return_to_original_list(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import create_scenario

    scenarios = []
    for index in range(event_admin_telegram.SCENARIOS_PAGE_SIZE + 1):
        scenarios.append(
            await create_scenario(
                db_session_maker,
                code=f"telegram_page_context_{index}",
                event_type="meeting",
                title=f"Telegram page context {index}",
                starts_at_utc=datetime.utcnow() + timedelta(days=2, minutes=index),
                place="Room 1",
            )
        )
    target = scenarios[0]
    page_back = f"admin_ev_page_1_1"

    import handlers

    routers = (event_admin_router, handlers.router)
    original_parents = {router: router._parent_router for router in routers}
    for router in routers:
        router._parent_router = None
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    for router in routers:
        dp.include_router(router)

    def callbacks(methods):
        markup = next(method for method in methods if isinstance(method, EditMessageText)).reply_markup
        return _telegram_event_callbacks(markup)

    try:
        methods = await _feed_callback(dp, bot, f"admin_ev_open_{target.id}_1_1")
        card_callbacks = callbacks(methods)
        assert page_back in card_callbacks

        methods = await _feed_callback(dp, bot, f"admin_ev_toggle_{target.id}")
        assert page_back in callbacks(methods)

        methods = await _feed_callback(dp, bot, f"admin_ev_archive_ask_{target.id}")
        confirm_callbacks = callbacks(methods)
        assert f"admin_ev_open_{target.id}_1_1" in confirm_callbacks
        assert f"admin_ev_archive_yes_{target.id}" in confirm_callbacks

        methods = await _feed_callback(dp, bot, f"admin_ev_open_{target.id}_1_1")
        assert page_back in callbacks(methods)

        await _feed_callback(dp, bot, f"admin_ev_archive_ask_{target.id}")
        methods = await _feed_callback(dp, bot, f"admin_ev_archive_yes_{target.id}")
        assert page_back in callbacks(methods)

        methods = await _feed_callback(dp, bot, f"admin_ev_page_1_1")
        list_edit = next(method for method in methods if isinstance(method, EditMessageText))
        assert "страница 2 из 2" in list_edit.text
        assert "Скрыть архив" in list_edit.text or "admin_ev_toggle_archive_view_0" in callbacks(methods)

        await _feed_callback(dp, bot, f"admin_ev_open_{target.id}_1_1")
        methods = await _feed_callback(dp, bot, f"admin_ev_stats_{target.id}")
        assert f"admin_ev_open_{target.id}_1_1" in callbacks(methods)
        methods = await _feed_callback(dp, bot, f"admin_ev_open_{target.id}_1_1")
        assert page_back in callbacks(methods)
    finally:
        for router in routers:
            router._parent_router = original_parents[router]


@pytest.mark.asyncio
async def test_telegram_problem_delivery_page_roundtrip_uses_current_page(monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
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
        platform="telegram",
        retry_index=0,
        parent_delivery_id=None,
    )
    page_items = [selected]

    async def fake_list_problem_deliveries(session_maker, event_id, *, limit, offset):
        assert event_id == 77
        assert limit == event_admin_telegram.PROBLEMS_PAGE_SIZE
        assert offset == event_admin_telegram.PROBLEMS_PAGE_SIZE
        return page_items, 6

    monkeypatch.setattr(event_admin_telegram, "list_problem_deliveries", fake_list_problem_deliveries)
    monkeypatch.setattr(event_admin_telegram, "get_delivery_item", AsyncMock(return_value=selected))
    monkeypatch.setattr(event_admin_telegram, "list_delivery_parts", AsyncMock(return_value=[]))

    import handlers

    routers = (event_admin_router, handlers.router)
    original_parents = {router: router._parent_router for router in routers}
    for router in routers:
        router._parent_router = None
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    for router in routers:
        dp.include_router(router)

    def callbacks(methods):
        markup = next(method for method in methods if isinstance(method, EditMessageText)).reply_markup
        return _telegram_event_callbacks(markup)

    try:
        methods = await _feed_callback(dp, bot, "admin_ev_problems_77_1")
        item_callback = next(callback for callback in callbacks(methods) if callback.startswith("admin_ev_problem_item_"))
        assert item_callback == "admin_ev_problem_item_605_1"

        methods = await _feed_callback(dp, bot, item_callback)
        assert "admin_ev_problems_77_1" in callbacks(methods)

        resolve_callback = next(
            callback for callback in callbacks(methods)
            if callback.startswith("admin_ev_resolve_force_605")
        )
        methods = await _feed_callback(dp, bot, resolve_callback)
        assert "admin_ev_problem_item_605_1" in callbacks(methods)
        state = FSMContext(storage=storage, key=StorageKey(bot_id=bot.id, chat_id=12345, user_id=12345))
        assert await state.get_state() == event_admin_telegram.EventAdminStates.resolve_reason.state

        methods = await _feed_callback(dp, bot, "admin_ev_problem_item_605_1")
        assert "admin_ev_problems_77_1" in callbacks(methods)
        assert await state.get_state() is None

        methods = await _feed_callback(dp, bot, "admin_ev_problems_77_1")
        detail_list = next(method for method in methods if isinstance(method, EditMessageText))
        assert "Страница 2 из 2" in detail_list.text
    finally:
        for router in routers:
            router._parent_router = original_parents[router]


@pytest.mark.asyncio
async def test_telegram_successful_resolution_reopens_clamped_page_and_event_context(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import create_scenario

    scenarios = []
    for index in range(event_admin_telegram.SCENARIOS_PAGE_SIZE + 1):
        scenarios.append(
            await create_scenario(
                db_session_maker,
                code=f"telegram_resolution_context_{index}",
                event_type="meeting",
                title=f"Telegram resolution context {index}",
                starts_at_utc=datetime.utcnow() + timedelta(days=2, minutes=index),
                place="Room 1",
            )
        )
    target = scenarios[0]
    delivery_ids = []
    async with db_session_maker() as session:
        for index in range(event_admin_telegram.PROBLEMS_PAGE_SIZE + 1):
            user = User(id=1100 + index, first_name=f"Problem {index}")
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
                delivery_key=f"telegram_resolution_delivery_{index}",
                due_at=utc_now(),
                next_attempt_at=utc_now(),
                platform="telegram",
                status="uncertain",
                attempt_count=1,
                last_error="timeout",
                content_version_snapshot=1,
            )
            session.add(delivery)
            await session.flush()
            delivery_ids.append(delivery.id)
        await session.commit()

    import handlers

    routers = (event_admin_router, handlers.router)
    original_parents = {router: router._parent_router for router in routers}
    for router in routers:
        router._parent_router = None
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    for router in routers:
        dp.include_router(router)

    def markup_callbacks(methods):
        method = next(
            method for method in methods
            if isinstance(method, (EditMessageText, SendMessage)) and method.reply_markup is not None
        )
        return _telegram_event_callbacks(method.reply_markup)

    try:
        await _feed_callback(dp, bot, f"admin_ev_open_{target.id}_1_1")
        methods = await _feed_callback(dp, bot, f"admin_ev_stats_{target.id}")
        problem_root = next(callback for callback in markup_callbacks(methods) if callback.startswith(f"admin_ev_problems_{target.id}_"))

        methods = await _feed_callback(dp, bot, f"admin_ev_problems_{target.id}_1")
        item_callback = next(callback for callback in markup_callbacks(methods) if callback.startswith("admin_ev_problem_item_"))
        assert item_callback.endswith("_1")

        methods = await _feed_callback(dp, bot, item_callback)
        resolve_callback = next(callback for callback in markup_callbacks(methods) if callback == f"admin_ev_resolve_sent_{delivery_ids[0]}")
        await _feed_callback(dp, bot, resolve_callback)
        methods = await _feed_message(dp, bot, "Confirmed delivery")
        assert f"admin_ev_problems_{target.id}_1" in markup_callbacks(methods)

        methods = await _feed_callback(dp, bot, f"admin_ev_problems_{target.id}_1")
        list_edit = next(method for method in methods if isinstance(method, EditMessageText))
        assert "Проблемные доставки (5)" in list_edit.text
        assert "Страница 1 из 1" in list_edit.text
        assert "Проблемных доставок нет" not in list_edit.text

        methods = await _feed_callback(dp, bot, f"admin_ev_stats_{target.id}")
        assert f"admin_ev_open_{target.id}_1_1" in markup_callbacks(methods)
        await _feed_callback(dp, bot, f"admin_ev_open_{target.id}_1_1")
        methods = await _feed_callback(dp, bot, f"admin_ev_page_1_1")
        list_edit = next(method for method in methods if isinstance(method, EditMessageText))
        assert "страница 2 из 2" in list_edit.text
        assert "admin_ev_toggle_archive_view_0" in markup_callbacks(methods)
    finally:
        for router in routers:
            router._parent_router = original_parents[router]


@pytest.mark.asyncio
async def test_telegram_pagination_boundaries_for_scenarios_and_problems(monkeypatch):
    def scenario_rows(count: int) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                id=index + 1,
                code=f"event_{index + 1}",
                title=f"Event {index + 1}",
                event_type="webinar",
                starts_at_msk="01.01.2030 12:00",
                is_archived=False,
                is_active=True,
                registrations_count=0,
            )
            for index in range(count)
        ]

    for count in (
        0,
        1,
        event_admin_telegram.SCENARIOS_PAGE_SIZE,
        event_admin_telegram.SCENARIOS_PAGE_SIZE + 1,
        event_admin_telegram.SCENARIOS_PAGE_SIZE * 2 + 1,
    ):
        monkeypatch.setattr(event_admin_telegram, "list_scenarios", AsyncMock(return_value=scenario_rows(count)))
        callback = _make_callback("admin_events")
        await event_admin_telegram._show_scenarios_page(callback, page=0, include_archived=True)
        callbacks = _telegram_event_callbacks(callback.message.edit_text.call_args.kwargs["reply_markup"])
        assert (f"admin_ev_page_1_1" in callbacks) is (count > event_admin_telegram.SCENARIOS_PAGE_SIZE)
        assert "admin_ev_page_0_1" not in callbacks
        if count > event_admin_telegram.SCENARIOS_PAGE_SIZE:
            callback = _make_callback("admin_ev_events_page_1")
            await event_admin_telegram._show_scenarios_page(callback, page=1, include_archived=True)
            callbacks = _telegram_event_callbacks(callback.message.edit_text.call_args.kwargs["reply_markup"])
            assert "admin_ev_page_0_1" in callbacks
            assert ("admin_ev_page_2_1" in callbacks) is (count > event_admin_telegram.SCENARIOS_PAGE_SIZE * 2)
            if count > event_admin_telegram.SCENARIOS_PAGE_SIZE * 2:
                callback = _make_callback("admin_ev_events_page_2")
                await event_admin_telegram._show_scenarios_page(callback, page=2, include_archived=True)
                callbacks = _telegram_event_callbacks(callback.message.edit_text.call_args.kwargs["reply_markup"])
                assert "admin_ev_page_1_1" in callbacks
                assert "admin_ev_page_3_1" not in callbacks

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
                platform="telegram",
                retry_index=0,
                parent_delivery_id=None,
            )
            for index in range(count)
        ]

    for count in (
        0,
        1,
        event_admin_telegram.PROBLEMS_PAGE_SIZE,
        event_admin_telegram.PROBLEMS_PAGE_SIZE + 1,
        event_admin_telegram.PROBLEMS_PAGE_SIZE * 2 + 1,
    ):
        async def fake_list_problem_deliveries(session_maker, event_id, *, limit, offset, _count=count):
            rows = problem_rows(_count)
            return rows[offset : offset + limit], _count

        monkeypatch.setattr(event_admin_telegram, "list_problem_deliveries", fake_list_problem_deliveries)
        callback = _make_callback("admin_ev_problems_10_0")
        await event_admin_telegram.show_problem_deliveries(callback)
        callbacks = _telegram_event_callbacks(callback.message.edit_text.call_args.kwargs["reply_markup"])
        assert ("admin_ev_problems_10_1" in callbacks) is (count > event_admin_telegram.PROBLEMS_PAGE_SIZE)
        assert "admin_ev_problems_10_0" not in callbacks
        if count > event_admin_telegram.PROBLEMS_PAGE_SIZE:
            callback = _make_callback("admin_ev_problems_10_1")
            await event_admin_telegram.show_problem_deliveries(callback)
            callbacks = _telegram_event_callbacks(callback.message.edit_text.call_args.kwargs["reply_markup"])
            assert "admin_ev_problems_10_0" in callbacks
            assert ("admin_ev_problems_10_2" in callbacks) is (count > event_admin_telegram.PROBLEMS_PAGE_SIZE * 2)
            if count > event_admin_telegram.PROBLEMS_PAGE_SIZE * 2:
                callback = _make_callback("admin_ev_problems_10_2")
                await event_admin_telegram.show_problem_deliveries(callback)
                callbacks = _telegram_event_callbacks(callback.message.edit_text.call_args.kwargs["reply_markup"])
                assert "admin_ev_problems_10_1" in callbacks
                assert "admin_ev_problems_10_3" not in callbacks


@pytest.mark.asyncio
async def test_telegram_steps_roundtrip_preserves_paginated_back_callbacks(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    from event_admin_service import add_step, create_scenario

    scen = await create_scenario(
        db_session_maker,
        code="telegram_steps_back",
        event_type="webinar",
        title="Telegram steps back",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/steps-back",
    )
    step = await add_step(db_session_maker, scen.id, -60, "Шаг")
    storage = MemoryStorage()
    state = FSMContext(storage=storage, key=MagicMock(bot_id=1, chat_id=12345, user_id=12345))

    await event_admin_telegram.open_scenario_card(_make_callback(f"admin_ev_open_{scen.id}_2_1"), state)
    callback = _make_callback(f"admin_ev_steps_{scen.id}_2_1")
    await event_admin_telegram.show_steps_list(callback, state)
    markup = callback.message.edit_text.call_args.kwargs["reply_markup"]
    back_button = markup.inline_keyboard[-1][0]
    assert back_button.callback_data == f"admin_ev_open_{scen.id}_2_1"

    card = _make_callback(f"admin_ev_step_{step.id}")
    await event_admin_telegram.open_step_card(card, state)
    card_markup = card.message.edit_text.call_args.kwargs["reply_markup"]
    assert card_markup.inline_keyboard[-1][0].callback_data == f"admin_ev_steps_{scen.id}_2_1"


@pytest.mark.asyncio
async def test_telegram_dispatcher_steps_buttons_roundtrip_page_context(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import add_step, create_scenario

    scen = await create_scenario(
        db_session_maker,
        code="telegram_dispatcher_steps_back",
        event_type="webinar",
        title="Telegram dispatcher steps",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/dispatcher-steps",
    )
    step = await add_step(db_session_maker, scen.id, -60, "Шаг {event_title}")
    event_admin_router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(event_admin_router)
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789

    methods = await _feed_callback(dp, bot, f"admin_ev_open_{scen.id}_2_1")
    card = next(method for method in methods if isinstance(method, EditMessageText))
    steps_callback = next(
        button.callback_data
        for row in card.reply_markup.inline_keyboard
        for button in row
        if button.callback_data and button.callback_data.startswith("admin_ev_steps_")
    )

    methods = await _feed_callback(dp, bot, steps_callback)
    steps_screen = next(method for method in methods if isinstance(method, EditMessageText))
    steps_back = steps_screen.reply_markup.inline_keyboard[-1][0].callback_data
    assert steps_back == f"admin_ev_open_{scen.id}_2_1"
    step_callback = next(
        button.callback_data
        for row in steps_screen.reply_markup.inline_keyboard
        for button in row
        if button.callback_data and button.callback_data.startswith("admin_ev_step_")
        and not button.callback_data.startswith("admin_ev_step_add_")
    )

    methods = await _feed_callback(dp, bot, step_callback)
    step_screen = next(method for method in methods if isinstance(method, EditMessageText))
    assert step_screen.reply_markup.inline_keyboard[-1][0].callback_data == f"admin_ev_steps_{scen.id}_2_1"
    edit_callback = next(
        button.callback_data
        for row in step_screen.reply_markup.inline_keyboard
        for button in row
        if button.callback_data and button.callback_data.startswith("admin_ev_step_edit_text_")
    )

    await _feed_callback(dp, bot, edit_callback)
    methods = await _feed_message(dp, bot, "Обновлённый {event_title}")
    saved = next(method for method in methods if isinstance(method, SendMessage) and "Текст шага обновлен" in method.text)
    assert saved.reply_markup.inline_keyboard[-1][0].callback_data == f"admin_ev_steps_{scen.id}_2_1"

    methods = await _feed_callback(dp, bot, f"admin_ev_step_{step.id}")
    step_screen = next(method for method in methods if isinstance(method, EditMessageText))
    edit_callback = next(
        button.callback_data
        for row in step_screen.reply_markup.inline_keyboard
        for button in row
        if button.callback_data and button.callback_data.startswith("admin_ev_step_edit_text_")
    )
    methods = await _feed_callback(dp, bot, edit_callback)
    edit_screen = next(method for method in methods if isinstance(method, EditMessageText))
    cancel_callback = next(
        button.callback_data
        for row in edit_screen.reply_markup.inline_keyboard
        for button in row
        if button.text == "⬅️ Отмена"
    )
    methods = await _feed_callback(dp, bot, cancel_callback)
    cancelled = next(method for method in methods if isinstance(method, EditMessageText))
    assert cancelled.reply_markup.inline_keyboard[-1][0].callback_data == f"admin_ev_steps_{scen.id}_2_1"


@pytest.mark.asyncio
async def test_telegram_dispatcher_delete_last_step_returns_empty_page(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import add_step, create_scenario, list_steps

    scen = await create_scenario(
        db_session_maker,
        code="telegram_delete_last_step",
        event_type="webinar",
        title="Delete last step",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/delete-last",
    )
    step = await add_step(db_session_maker, scen.id, -60, "Last reminder")
    event_admin_router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(event_admin_router)
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789

    methods = await _feed_callback(dp, bot, f"admin_ev_steps_{scen.id}_2_1")
    steps_screen = next(method for method in methods if isinstance(method, EditMessageText))
    await _feed_callback(dp, bot, f"admin_ev_step_{step.id}")
    methods = await _feed_callback(dp, bot, f"admin_ev_step_del_ask_{step.id}")
    confirmation = next(method for method in methods if isinstance(method, EditMessageText))
    assert f"admin_ev_step_del_yes_{step.id}" in _telegram_event_callbacks(confirmation.reply_markup)

    methods = await _feed_callback(dp, bot, f"admin_ev_step_del_yes_{step.id}")
    empty_page = next(method for method in methods if isinstance(method, EditMessageText))
    assert "Всего активных шагов: 0" in empty_page.text
    assert not any(
        button.callback_data and button.callback_data.startswith("admin_ev_step_")
        and not button.callback_data.startswith("admin_ev_step_add_")
        for row in empty_page.reply_markup.inline_keyboard
        for button in row
    )
    assert any(
        button.callback_data == f"admin_ev_open_{scen.id}_2_1"
        for row in empty_page.reply_markup.inline_keyboard
        for button in row
    )
    assert await list_steps(db_session_maker, scen.id, include_archived=False) == []


@pytest.mark.asyncio
async def test_telegram_dispatcher_description_chunks_save_reopen_and_runtime_split(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import create_scenario
    from event_delivery_worker import build_test_delivery_plan

    scen = await create_scenario(
        db_session_maker,
        code="telegram_dispatcher_description",
        event_type="webinar",
        title="Telegram dispatcher description",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/dispatcher-description",
        media_mode="legacy_auto",
    )
    event_admin_router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(event_admin_router)
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789

    methods = await _feed_callback(dp, bot, f"admin_ev_open_{scen.id}_2_1")
    card = next(method for method in methods if isinstance(method, EditMessageText))
    edit_callback = next(
        button.callback_data
        for row in card.reply_markup.inline_keyboard
        for button in row
        if button.callback_data and button.callback_data.startswith("admin_ev_edit_menu_")
    )
    methods = await _feed_callback(dp, bot, edit_callback)
    edit_menu = next(method for method in methods if isinstance(method, EditMessageText))
    description_callback = next(
        button.callback_data
        for row in edit_menu.reply_markup.inline_keyboard
        for button in row
        if button.callback_data and button.callback_data.startswith("admin_ev_edit_description_")
    )
    await _feed_callback(dp, bot, description_callback)
    await _feed_message(dp, bot, "A" * 3000)
    await _feed_message(dp, bot, "B" * 3000)
    await _feed_callback(dp, bot, f"admin_ev_description_save_{scen.id}")

    async with db_session_maker() as session:
        saved = await session.get(EventScenario, scen.id)
        assert saved.description == ("A" * 3000) + "\n" + ("B" * 3000)
    plan = await build_test_delivery_plan(db_session_maker, scen.id, None, None)
    text_parts = [part for part in plan if part["kind"] == "text"]
    assert len(text_parts) >= 2
    assert all(len(part["text"]) <= 4000 for part in text_parts)

    methods = await _feed_callback(dp, bot, f"admin_ev_open_{scen.id}_2_1")
    card = next(method for method in methods if isinstance(method, EditMessageText))
    assert any(
        button.callback_data and button.callback_data.startswith("admin_ev_edit_menu_")
        for row in card.reply_markup.inline_keyboard
        for button in row
    )


@pytest.mark.asyncio
async def test_telegram_description_and_media_edit_routes(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr("event_admin_telegram.get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import create_scenario

    scen = await create_scenario(
        db_session_maker,
        code="telegram_media_edit",
        event_type="webinar",
        title="Edit routes",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/edit",
    )
    storage = MemoryStorage()
    state = FSMContext(storage=storage, key=MagicMock(bot_id=1, chat_id=12345, user_id=12345))

    await event_admin_telegram.start_edit_description(_make_callback(f"admin_ev_edit_description_{scen.id}"), state)
    description_message = _make_message("ignored")
    description_message.html_text = "<b>С форматированием</b>"
    await event_admin_telegram.process_edit_description(description_message, state)
    await event_admin_telegram.save_description(_make_callback(f"admin_ev_description_save_{scen.id}"), state)

    await event_admin_telegram.start_edit_media(_make_callback(f"admin_ev_edit_media_{scen.id}"), state)
    await event_admin_telegram.start_add_media(_make_callback(f"admin_ev_media_add_{scen.id}"), state)
    media_message = _make_message("")
    media_message.text = None
    media_message.caption = None
    media_message.photo = [SimpleNamespace(file_id="telegram-photo")]
    media_message.video = None
    await event_admin_telegram.receive_media(media_message, state)
    await event_admin_telegram.save_media(_make_callback(f"admin_ev_media_save_{scen.id}"), state)

    async with db_session_maker() as session:
        saved = await session.get(EventScenario, scen.id)
        media = (await session.execute(select(EventScenarioMedia).where(EventScenarioMedia.event_id == scen.id))).scalars().all()
        assert saved.description == "<b>С форматированием</b>"
        assert [(item.file_type, item.file_id) for item in media] == [("photo", "telegram-photo")]


@pytest.mark.asyncio
async def test_telegram_description_and_media_buttons_are_displayed(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr("event_admin_telegram.get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import create_scenario

    scen = await create_scenario(
        db_session_maker,
        code="telegram_button_contract",
        event_type="webinar",
        title="Button contract",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/buttons",
    )
    storage = MemoryStorage()
    state = FSMContext(storage=storage, key=MagicMock(bot_id=1, chat_id=12345, user_id=12345))

    callback = _make_callback(f"admin_ev_edit_description_{scen.id}")
    await event_admin_telegram.start_edit_description(callback, state)
    assert await state.get_state() == event_admin_telegram.EventAdminStates.edit_description.state
    markup = callback.message.edit_text.call_args.kwargs["reply_markup"]
    callback_data = [button.callback_data for row in markup.inline_keyboard for button in row]
    assert callback_data == [
        f"admin_ev_description_save_{scen.id}",
        f"admin_ev_description_delete_{scen.id}",
        f"admin_ev_edit_menu_{scen.id}",
    ]

    media_callback = _make_callback(f"admin_ev_edit_media_{scen.id}")
    await event_admin_telegram.start_edit_media(media_callback, state)
    media_markup = media_callback.message.edit_text.call_args.kwargs["reply_markup"]
    media_data = [button.callback_data for row in media_markup.inline_keyboard for button in row]
    assert media_data == [
        f"admin_ev_media_add_{scen.id}",
        f"admin_ev_media_save_{scen.id}",
        f"admin_ev_edit_menu_{scen.id}",
    ]

    add_callback = _make_callback(f"admin_ev_media_add_{scen.id}")
    await event_admin_telegram.start_add_media(add_callback, state)
    assert await state.get_state() == event_admin_telegram.EventAdminStates.edit_media_add.state
    add_markup = add_callback.message.edit_text.call_args.kwargs["reply_markup"]
    assert add_markup.inline_keyboard[0][0].callback_data == f"admin_ev_edit_media_{scen.id}"


@pytest.mark.asyncio
async def test_telegram_media_back_preserves_draft_and_cancel_discards_it(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    from event_admin_service import create_scenario

    scen = await create_scenario(
        db_session_maker,
        code="telegram_media_back",
        event_type="webinar",
        title="Media back",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/media-back",
    )
    storage = MemoryStorage()
    state = FSMContext(storage=storage, key=MagicMock(bot_id=1, chat_id=12345, user_id=12345))

    await event_admin_telegram.start_edit_media(_make_callback(f"admin_ev_edit_media_{scen.id}"), state)
    await event_admin_telegram.start_add_media(_make_callback(f"admin_ev_media_add_{scen.id}"), state)
    media_message = _make_message("")
    media_message.text = None
    media_message.caption = None
    media_message.photo = [SimpleNamespace(file_id="draft-photo")]
    media_message.video = None
    await event_admin_telegram.receive_media(media_message, state)

    back_callback = _make_callback(f"admin_ev_edit_media_{scen.id}")
    await event_admin_telegram.start_edit_media(back_callback, state)
    back_markup = back_callback.message.edit_text.call_args.kwargs["reply_markup"]
    back_payloads = [button.callback_data for row in back_markup.inline_keyboard for button in row]
    assert f"admin_ev_media_remove_{scen.id}_0" in back_payloads
    assert await state.get_state() == event_admin_telegram.EventAdminStates.edit_media.state

    await event_admin_telegram.show_edit_menu(_make_callback(f"admin_ev_edit_menu_{scen.id}"), state)
    assert await state.get_state() == event_admin_telegram.EventAdminStates.edit_media.state
    preserved = await state.get_data()
    assert preserved["media_draft"][0]["file_id"] == "draft-photo"
    await event_admin_telegram.cancel_draft_handler(_make_callback(f"admin_ev_cancel_draft_{scen.id}"), state)
    assert await state.get_state() is None
    reopen_callback = _make_callback(f"admin_ev_edit_media_{scen.id}")
    await event_admin_telegram.start_edit_media(reopen_callback, state)
    reopened_markup = reopen_callback.message.edit_text.call_args.kwargs["reply_markup"]
    reopened_payloads = [button.callback_data for row in reopened_markup.inline_keyboard for button in row]
    assert f"admin_ev_media_remove_{scen.id}_0" not in reopened_payloads


@pytest.mark.asyncio
async def test_telegram_paginated_edit_roundtrip_and_delete_stays_draft(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    from event_admin_service import create_scenario

    scen = await create_scenario(
        db_session_maker,
        code="telegram_paginated_edit",
        event_type="webinar",
        title="Paginated edit",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/paginated-edit",
        description="Stored description",
    )
    storage = MemoryStorage()
    state = FSMContext(storage=storage, key=MagicMock(bot_id=1, chat_id=12345, user_id=12345))

    await event_admin_telegram.open_scenario_card(_make_callback(f"admin_ev_open_{scen.id}_2_1"), state)
    edit_callback = _make_callback(f"admin_ev_edit_menu_{scen.id}")
    await event_admin_telegram.show_edit_menu(edit_callback, state)
    edit_markup = edit_callback.message.edit_text.call_args.kwargs["reply_markup"]
    edit_callbacks = [button.callback_data for row in edit_markup.inline_keyboard for button in row]
    assert f"admin_ev_open_{scen.id}_2_1" in edit_callbacks
    rendered = _make_callback(f"admin_ev_edit_description_{scen.id}")
    await event_admin_telegram.start_edit_description(rendered, state)
    description_markup = rendered.message.edit_text.call_args.kwargs["reply_markup"]
    description_callbacks = [button.callback_data for row in description_markup.inline_keyboard for button in row]
    assert description_callbacks[-1] == f"admin_ev_edit_menu_{scen.id}_2_1"

    delete_callback = _make_callback(f"admin_ev_description_delete_{scen.id}")
    await event_admin_telegram.delete_description(delete_callback, state)
    async with db_session_maker() as session:
        unchanged = await session.get(EventScenario, scen.id)
        assert unchanged.description == "Stored description"

    save_callback = _make_callback(f"admin_ev_description_save_{scen.id}")
    await event_admin_telegram.save_description(save_callback, state)
    async with db_session_maker() as session:
        saved = await session.get(EventScenario, scen.id)
        assert saved.description is None
    saved_markup = save_callback.message.edit_text.call_args.kwargs["reply_markup"]
    saved_callbacks = [button.callback_data for row in saved_markup.inline_keyboard for button in row]
    assert "admin_ev_page_2_1" in saved_callbacks


@pytest.mark.ui_journey
@pytest.mark.asyncio
async def test_telegram_real_paginated_format_save_cancel_journey_preserves_page_filter(db_session_maker, monkeypatch):
    """Real Dispatcher journey: page 2 -> card -> format -> draft -> cancel/save -> page 2."""
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import create_scenario
    import handlers

    base_start = datetime.utcnow() + timedelta(days=2)
    target = await create_scenario(
        db_session_maker,
        code="telegram_real_page_two_target",
        event_type="webinar",
        title="Page two target",
        starts_at_utc=base_start,
        link="https://example.test/page-two-target",
    )
    for index in range(event_admin_telegram.SCENARIOS_PAGE_SIZE):
        await create_scenario(
            db_session_maker,
            code=f"telegram_real_page_two_fill_{index}",
            event_type="webinar",
            title=f"Page two filler {index}",
            starts_at_utc=base_start + timedelta(days=index + 1),
            link=f"https://example.test/page-two/{index}",
        )

    routers = (event_admin_router, handlers.router)
    original_parents = {router: router._parent_router for router in routers}
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    for router in routers:
        router._parent_router = None
        dp.include_router(router)

    def visible_callback(methods, prefix: str) -> str:
        for method in methods:
            if not isinstance(method, (EditMessageText, SendMessage)) or method.reply_markup is None:
                continue
            for row in method.reply_markup.inline_keyboard:
                for button in row:
                    if button.callback_data and button.callback_data.startswith(prefix):
                        return button.callback_data
        raise AssertionError(f"No visible Telegram callback with prefix {prefix!r}")

    async def feed_callback(data: str):
        methods = await _feed_callback(dp, bot, data)
        for method in methods:
            if hasattr(method, "model_dump"):
                method.__class__.model_validate(method.model_dump())
        return methods

    async def feed_message(text: str):
        methods = await _feed_message(dp, bot, text)
        for method in methods:
            if hasattr(method, "model_dump"):
                method.__class__.model_validate(method.model_dump())
        return methods

    try:
        methods = await feed_callback("admin_events")
        listing = next(method for method in methods if isinstance(method, EditMessageText))
        assert "страница 1 из 2" in listing.text
        page_two_callback = visible_callback(methods, "admin_ev_page_1_0")

        methods = await feed_callback(page_two_callback)
        listing = next(method for method in methods if isinstance(method, EditMessageText))
        assert "страница 2 из 2" in listing.text
        target_callback = visible_callback(methods, f"admin_ev_open_{target.id}_1_0")

        methods = await feed_callback(target_callback)
        edit_menu_callback = visible_callback(methods, f"admin_ev_edit_menu_{target.id}_1_0")
        methods = await feed_callback(edit_menu_callback)
        format_callback = visible_callback(methods, f"admin_ev_format_menu_{target.id}")

        methods = await feed_callback(format_callback)
        format_back = visible_callback(methods, f"admin_ev_edit_menu_{target.id}_1_0")
        methods = await feed_callback(format_back)
        description_callback = visible_callback(methods, f"admin_ev_edit_description_{target.id}")

        await feed_callback(description_callback)
        methods = await feed_message("Описание для отмены")
        draft_back = visible_callback(methods, f"admin_ev_edit_menu_{target.id}")
        assert draft_back == f"admin_ev_edit_menu_{target.id}_1_0"
        methods = await feed_callback(draft_back)
        cancel_callback = visible_callback(methods, f"admin_ev_cancel_draft_{target.id}")
        methods = await feed_callback(cancel_callback)
        card = next(method for method in methods if isinstance(method, EditMessageText))
        assert "Карточка мероприятия" in card.text
        assert "Описание для отмены" not in card.text
        async with db_session_maker() as session:
            assert (await session.get(EventScenario, target.id)).description is None

        edit_menu_callback = visible_callback(methods, f"admin_ev_edit_menu_{target.id}_1_0")
        methods = await feed_callback(edit_menu_callback)
        description_callback = visible_callback(methods, f"admin_ev_edit_description_{target.id}")
        await feed_callback(description_callback)
        methods = await feed_message("Описание для сохранения")
        save_callback = visible_callback(methods, f"admin_ev_description_save_{target.id}")
        methods = await feed_callback(save_callback)
        card = next(method for method in methods if isinstance(method, EditMessageText))
        assert "Описание для сохранения" in card.text
        async with db_session_maker() as session:
            assert (await session.get(EventScenario, target.id)).description == "Описание для сохранения"

        methods = await feed_callback(visible_callback(methods, "admin_ev_page_1_0"))
        listing = next(method for method in methods if isinstance(method, EditMessageText))
        assert "страница 2 из 2" in listing.text
        assert "telegram_real_page_two_target" in listing.text
    finally:
        for router in routers:
            router._parent_router = original_parents[router]


@pytest.mark.asyncio
async def test_telegram_stale_mutation_callback_cannot_touch_other_event_draft(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import create_scenario

    event_a = await create_scenario(
        db_session_maker,
        code="telegram_stale_event_a",
        event_type="webinar",
        title="Event A",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/stale-a",
        description="A original",
    )
    event_b = await create_scenario(
        db_session_maker,
        code="telegram_stale_event_b",
        event_type="webinar",
        title="Event B",
        starts_at_utc=datetime.utcnow() + timedelta(days=3),
        link="https://example.test/stale-b",
        description="B original",
    )

    storage = MemoryStorage()
    state = FSMContext(storage=storage, key=StorageKey(bot_id=123456789, chat_id=12345, user_id=12345))
    await state.update_data(
        edit_event_id=event_a.id,
        description_draft="A draft that must remain isolated",
        description_input_started=True,
        event_list_back="admin_ev_page_1_0",
    )
    await state.set_state(event_admin_telegram.EventAdminStates.edit_description)

    routers = (event_admin_router,)
    original_parents = {router: router._parent_router for router in routers}
    for router in routers:
        router._parent_router = None
    dp = Dispatcher(storage=storage)
    dp.include_router(event_admin_router)
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    try:
        methods = await _feed_callback(dp, bot, f"admin_ev_description_save_{event_b.id}")
        assert any(isinstance(method, AnswerCallbackQuery) and method.show_alert for method in methods)
        methods = await _feed_callback(dp, bot, f"admin_ev_cancel_draft_{event_b.id}")
        assert any(isinstance(method, AnswerCallbackQuery) and method.show_alert for method in methods)
        snapshot = await state.get_data()
        assert snapshot["edit_event_id"] == event_a.id
        assert snapshot["description_draft"] == "A draft that must remain isolated"

        async with db_session_maker() as session:
            saved_a = await session.get(EventScenario, event_a.id)
            saved_b = await session.get(EventScenario, event_b.id)
            assert saved_a.description == "A original"
            assert saved_b.description == "B original"
    finally:
        for router in routers:
            router._parent_router = original_parents[router]


@pytest.mark.asyncio
async def test_telegram_long_description_preview_is_bounded(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    from event_admin_service import create_scenario

    scen = await create_scenario(
        db_session_maker,
        code="telegram_long_preview",
        event_type="webinar",
        title="Long preview",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/long-preview",
        description="x" * 5000,
        media_mode="legacy_auto",
    )
    storage = MemoryStorage()
    state = FSMContext(storage=storage, key=MagicMock(bot_id=1, chat_id=12345, user_id=12345))
    callback = _make_callback(f"admin_ev_edit_description_{scen.id}")
    await event_admin_telegram.start_edit_description(callback, state)
    rendered = callback.message.edit_text.call_args.args[0]
    assert len(rendered) < 4096


@pytest.mark.asyncio
async def test_telegram_description_chunks_append_and_runtime_split_after_save(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    from event_admin_service import create_scenario
    from event_delivery_worker import build_test_delivery_plan

    scen = await create_scenario(
        db_session_maker,
        code="telegram_chunked_description",
        event_type="webinar",
        title="Chunked Telegram description",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/chunks",
        media_mode="legacy_auto",
    )
    storage = MemoryStorage()
    state = FSMContext(storage=storage, key=MagicMock(bot_id=1, chat_id=12345, user_id=12345))
    await event_admin_telegram.start_edit_description(_make_callback(f"admin_ev_edit_description_{scen.id}"), state)
    first = _make_message("A" * 3000)
    first.html_text = "A" * 3000
    second = _make_message("B" * 3000)
    second.html_text = "B" * 3000
    await event_admin_telegram.process_edit_description(first, state)
    await event_admin_telegram.process_edit_description(second, state)
    await event_admin_telegram.save_description(_make_callback(f"admin_ev_description_save_{scen.id}"), state)

    async with db_session_maker() as session:
        saved = await session.get(EventScenario, scen.id)
        assert saved.description == ("A" * 3000) + "\n" + ("B" * 3000)
    plan = await build_test_delivery_plan(db_session_maker, scen.id, None, None)
    text_parts = [part for part in plan if part["kind"] == "text"]
    assert len(text_parts) >= 2
    assert all(len(part["text"]) <= 4000 for part in text_parts)


def test_telegram_uncertain_resolution_buttons_target_specific_part():
    item = DeliveryItem(
        id=41,
        event_id=7,
        user_id=9,
        user_display="User",
        message_kind="immediate_confirmation",
        step_id=None,
        status="uncertain",
        due_at_msk="04.11.2026 12:00",
        attempt_count=1,
        last_error="timeout",
        sent_or_finished_msk=None,
        platform="telegram",
        retry_index=0,
        parent_delivery_id=None,
    )
    part = DeliveryPartItem(
        id=99,
        part_index=1,
        part_kind="media",
        status="uncertain",
        attempt_count=1,
        last_error="timeout",
        external_message_id=None,
    )
    markup = event_admin_telegram._delivery_resolution_keyboard(item, [part])
    callbacks = [button.callback_data for row in markup.inline_keyboard for button in row]
    assert callbacks[:3] == [
        "admin_ev_resolve_force_41_99",
        "admin_ev_resolve_sent_41_99",
        "admin_ev_resolve_skipped_41_99",
    ]


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
    msg_time = _make_message(_future_msk_text())
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
    await event_admin_telegram.process_create_starts_at(_make_message(_future_msk_text()), state)
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

    msg_new_time = _make_message(_future_msk_text(days=10, hour=20))
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

    msg_copy_time = _make_message(_future_msk_text(days=20))
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


@pytest.mark.asyncio
async def test_telegram_test_send_uses_runtime_media_plan_without_delivery_rows(db_session_maker, monkeypatch, mock_bot):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    from event_admin_service import add_step, create_scenario

    scen = await create_scenario(
        db_session_maker,
        code="test_send_tg_media",
        event_type="webinar",
        title="Test Send TG Media",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://webinar.com/media",
        description="Описание для тестовой галереи",
    )
    async with db_session_maker() as session:
        session.add(EventScenarioMedia(event_id=scen.id, file_type="photo", file_id="tg-preview-photo", sort_order=0))
        session.add(EventScenarioMedia(event_id=scen.id, file_type="video", file_id="tg-preview-video", sort_order=1))
        await session.commit()
    step = await add_step(
        db_session_maker,
        scen.id,
        -120,
        "Напоминание {event_title}",
        include_event_details=True,
    )
    mock_bot.send_media_group = AsyncMock(return_value=[MagicMock(message_id=201)])
    mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=202))

    await event_admin_telegram.test_send_single_step(_make_callback(f"admin_ev_test_step_{step.id}"), mock_bot)

    assert mock_bot.send_media_group.await_count == 1
    assert mock_bot.send_message.await_count == 1
    from aiogram.methods import SendMediaGroup, SendMessage
    SendMediaGroup(chat_id=12345, media=mock_bot.send_media_group.call_args.kwargs["media"])
    SendMessage(
        chat_id=12345,
        text=mock_bot.send_message.call_args.kwargs["text"],
        parse_mode=mock_bot.send_message.call_args.kwargs["parse_mode"],
        reply_markup=mock_bot.send_message.call_args.kwargs.get("reply_markup"),
    )
    async with db_session_maker() as session:
        assert (await session.execute(select(EventDelivery))).scalars().all() == []


async def _feed_callback(dp: Dispatcher, bot: Bot, data: str, user_id: int = 12345, message_id: int = 100):
    user = TgUser(id=user_id, is_bot=False, first_name="Admin", username="admin_user")
    chat = Chat(id=user_id, type="private")
    msg = Message(message_id=message_id, date=datetime.now(), chat=chat, text="Original text")
    cb = CallbackQuery(id=f"cb_{data}_{message_id}", from_user=user, chat_instance="inst", data=data, message=msg)
    update = Update(update_id=1, callback_query=cb)
    bot.reset_mock()
    await dp.feed_update(bot, update)
    return [call.args[0] for call in bot.call_args_list if call.args and hasattr(call.args[0], "__class__")]


async def _feed_message(
    dp: Dispatcher,
    bot: Bot,
    text: str = "",
    user_id: int = 12345,
    message_id: int = 101,
    photo: Any | None = None,
    video: Any | None = None,
):
    user = TgUser(id=user_id, is_bot=False, first_name="Admin", username="admin_user")
    chat = Chat(id=user_id, type="private")
    msg = Message(
        message_id=message_id,
        date=datetime.now(),
        chat=chat,
        text=text,
        from_user=user,
        photo=photo,
        video=video,
    )
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
        starts_at_msk=f"{_future_msk_text(days=2)} (МСК)",
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



@pytest.mark.asyncio
async def test_telegram_long_title_real_dispatcher_list_edit_cancel_save_reopen(db_session_maker, monkeypatch):
    """A->Z->A: actual keyboard callbacks and aiogram-validated Telegram methods."""
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import create_scenario

    original = "А" * 1000
    scen = await create_scenario(
        db_session_maker,
        code="telegram_long_event_title",
        event_type="webinar",
        title=original,
        starts_at_utc=datetime.utcnow() + timedelta(days=5),
        link="https://example.test/long-title",
    )

    import handlers

    routers = (event_admin_router, handlers.router)
    original_parents = {r: r._parent_router for r in routers}
    dp = Dispatcher(storage=MemoryStorage())
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    for router in routers:
        router._parent_router = None
        dp.include_router(router)

    def visible_callback(methods, prefix):
        method = next(
            m for m in methods
            if isinstance(m, (EditMessageText, SendMessage)) and m.reply_markup is not None
        )
        for row in method.reply_markup.inline_keyboard:
            for button in row:
                if button.callback_data and button.callback_data.startswith(prefix):
                    return button.callback_data
        raise AssertionError(f"No visible button with prefix {prefix!r}")

    try:
        methods = await _feed_callback(dp, bot, "admin_events")
        listing = next(m for m in methods if isinstance(m, EditMessageText))
        assert original not in listing.text
        assert len(listing.text) < 4096
        target_callback = visible_callback(methods, f"admin_ev_open_{scen.id}")
        button = next(b for row in listing.reply_markup.inline_keyboard for b in row if b.callback_data == target_callback)
        assert len(button.text) < 100 and button.text.endswith(("… (" + scen.starts_at_msk[:10] + ")"))

        methods = await _feed_callback(dp, bot, target_callback)
        card = next(m for m in methods if isinstance(m, EditMessageText))
        assert original in card.text
        edit_callback = visible_callback(methods, f"admin_ev_edit_menu_{scen.id}")
        methods = await _feed_callback(dp, bot, edit_callback)
        edit_title_callback = visible_callback(methods, f"admin_ev_edit_title_{scen.id}")
        methods = await _feed_callback(dp, bot, edit_title_callback)
        assert "1000 символов" in next(m for m in methods if isinstance(m, EditMessageText)).text
        cancel_callback = visible_callback(methods, f"admin_ev_open_{scen.id}")
        methods = await _feed_callback(dp, bot, cancel_callback)
        assert original in next(m for m in methods if isinstance(m, EditMessageText)).text

        methods = await _feed_callback(dp, bot, visible_callback(methods, f"admin_ev_edit_menu_{scen.id}"))
        methods = await _feed_callback(dp, bot, visible_callback(methods, f"admin_ev_edit_title_{scen.id}"))
        invalid = await _feed_message(dp, bot, "В" * 1001)
        assert "1000 символов" in next(m for m in invalid if isinstance(m, SendMessage)).text
        valid = await _feed_message(dp, bot, "Б" * 1000)
        assert "Б" * 1000 in next(m for m in valid if isinstance(m, SendMessage)).text
        async with db_session_maker() as session:
            assert (await session.get(EventScenario, scen.id)).title == "Б" * 1000

        methods = await _feed_callback(dp, bot, target_callback)
        assert "Б" * 1000 in next(m for m in methods if isinstance(m, EditMessageText)).text
        back_callback = visible_callback(methods, "admin_ev_page_0_0")
        methods = await _feed_callback(dp, bot, back_callback)
        assert "Сценарии мероприятий" in next(m for m in methods if isinstance(m, EditMessageText)).text

        # Creation: real visible button, wizard inputs, saved DB row and screen.
        create_callback = visible_callback(methods, "admin_ev_create_webinar")
        methods = await _feed_callback(dp, bot, create_callback)
        assert "Шаг 1 из 5" in next(m for m in methods if isinstance(m, EditMessageText)).text
        await _feed_message(dp, bot, "tg_new_long_title_event")
        methods = await _feed_message(dp, bot, "Г" * 1000)
        assert "Шаг 3 из 5" in next(m for m in methods if isinstance(m, SendMessage)).text
        await _feed_message(dp, bot, _future_msk_text(days=40))
        await _feed_message(dp, bot, "120")
        methods = await _feed_message(dp, bot, "https://example.test/new-long-title")
        created_card = next(m for m in methods if isinstance(m, SendMessage))
        assert "Мероприятие успешно создано" in created_card.text
        assert "Г" * 1000 in created_card.text
        async with db_session_maker() as session:
            created = await session.scalar(
                select(EventScenario).where(EventScenario.code == "tg_new_long_title_event")
            )
            assert created is not None and created.title == "Г" * 1000
        list_callback = visible_callback(methods, "admin_events")
        methods = await _feed_callback(dp, bot, list_callback)
        assert "Сценарии мероприятий" in next(m for m in methods if isinstance(m, EditMessageText)).text
    finally:
        for router in routers:
            router._parent_router = original_parents[router]


@pytest.mark.asyncio
async def test_telegram_long_title_delivery_uses_validated_outgoing_method(db_session_maker):
    from event_admin_service import create_scenario
    from event_delivery_worker import build_test_delivery_plan, send_test_delivery_plan

    title = "<&>" + "Я" * 997
    scen = await create_scenario(
        db_session_maker,
        code="telegram_title_delivery_runtime",
        event_type="webinar",
        title=title,
        starts_at_utc=datetime.utcnow() + timedelta(days=5),
        link="https://example.test/telegram-title-runtime",
    )

    async with db_session_maker() as session:
        stored = await session.get(EventScenario, scen.id)
        assert stored is not None and stored.title == title

    plan = await build_test_delivery_plan(db_session_maker, scen.id, None, None)

    class ValidatingTelegramTransport:
        def __init__(self):
            self.methods = []

        async def send_message(self, *, chat_id, text, parse_mode, reply_markup=None):
            method = SendMessage(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
            validated = SendMessage.model_validate(method.model_dump())
            self.methods.append(validated)
            return SimpleNamespace(message_id=f"telegram-runtime-{len(self.methods)}")

    transport = ValidatingTelegramTransport()
    sent_ids = await send_test_delivery_plan(
        plan,
        platform="telegram",
        user_id=12345,
        event_id=scen.id,
        bot=transport,
    )

    assert sent_ids == ["telegram-runtime-1"]
    assert len(transport.methods) == 1
    method = transport.methods[0]
    assert method.parse_mode == "HTML"
    assert html.escape(title, quote=False) in method.text
    assert title in html.unescape(method.text)
    assert len(method.text.encode("utf-16-le")) // 2 <= 4096
    assert SendMessage.model_validate(method.model_dump()).text == method.text


@pytest.mark.asyncio
async def test_telegram_long_title_navigation_preserves_page_and_archive_context(db_session_maker, monkeypatch):
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import archive_scenario, create_scenario
    import handlers

    original_title = "А" * 1000
    base_start = datetime.utcnow() + timedelta(days=5)
    target = await create_scenario(
        db_session_maker,
        code="telegram_title_navigation_target",
        event_type="webinar",
        title=original_title,
        starts_at_utc=base_start,
        link="https://example.test/telegram-navigation",
    )
    for index in range(6):
        filler = await create_scenario(
            db_session_maker,
            code=f"telegram_title_navigation_filler_{index}",
            event_type="webinar",
            title=f"Навигация {index}",
            starts_at_utc=base_start + timedelta(days=1),
            link=f"https://example.test/telegram-navigation/{index}",
        )
        if index == 0:
            await archive_scenario(db_session_maker, filler.id)

    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    routers = (event_admin_router, handlers.router)
    original_parents = {router: router._parent_router for router in routers}
    for router in routers:
        router._parent_router = None
        dp.include_router(router)
    fsm = FSMContext(
        storage=storage,
        key=StorageKey(bot_id=bot.id, chat_id=12345, user_id=12345),
    )

    def validate_methods(methods):
        for method in methods:
            assert hasattr(method, "model_dump")
            method.__class__.model_validate(method.model_dump())
        return methods

    def visible_callback(methods, prefix):
        for method in methods:
            if not isinstance(method, (EditMessageText, SendMessage)) or method.reply_markup is None:
                continue
            for row in method.reply_markup.inline_keyboard:
                for button in row:
                    if button.callback_data and button.callback_data.startswith(prefix):
                        return button.callback_data
        raise AssertionError(f"No visible Telegram callback with prefix {prefix!r}")

    async def feed_callback(data):
        return validate_methods(await _feed_callback(dp, bot, data))

    async def feed_message(text):
        return validate_methods(await _feed_message(dp, bot, text))

    try:
        methods = await feed_callback("admin_events")
        listing = next(method for method in methods if isinstance(method, EditMessageText))
        assert "страница 1 из 1" in listing.text
        methods = await feed_callback(visible_callback(methods, "admin_ev_toggle_archive_view_1"))
        listing = next(method for method in methods if isinstance(method, EditMessageText))
        assert "страница 1 из 2" in listing.text

        methods = await feed_callback(visible_callback(methods, "admin_ev_page_1_1"))
        listing = next(method for method in methods if isinstance(method, EditMessageText))
        assert "страница 2 из 2" in listing.text
        target_callback = visible_callback(methods, f"admin_ev_open_{target.id}_1_1")

        methods = await feed_callback(target_callback)
        card = next(method for method in methods if isinstance(method, EditMessageText))
        assert original_title in card.text
        edit_callback = visible_callback(methods, f"admin_ev_edit_menu_{target.id}_1_1")
        methods = await feed_callback(edit_callback)
        title_callback = visible_callback(methods, f"admin_ev_edit_title_{target.id}")
        methods = await feed_callback(title_callback)
        assert "1000 символов" in next(method for method in methods if isinstance(method, EditMessageText)).text
        assert (await fsm.get_state()).endswith("edit_title")

        methods = await feed_callback(visible_callback(methods, f"admin_ev_open_{target.id}_1_1"))
        card = next(method for method in methods if isinstance(method, EditMessageText))
        assert original_title in card.text
        assert await fsm.get_state() is None

        methods = await feed_callback(visible_callback(methods, f"admin_ev_edit_menu_{target.id}_1_1"))
        methods = await feed_callback(visible_callback(methods, f"admin_ev_edit_title_{target.id}"))
        invalid_methods = await feed_message("В" * 1001)
        assert "1000 символов" in next(method for method in invalid_methods if isinstance(method, SendMessage)).text
        assert (await fsm.get_state()).endswith("edit_title")
        async with db_session_maker() as session:
            assert (await session.get(EventScenario, target.id)).title == original_title

        updated_title = "Б" * 1000
        methods = await feed_message(updated_title)
        saved_card = next(method for method in methods if isinstance(method, SendMessage))
        assert updated_title in saved_card.text
        assert await fsm.get_state() is None
        back_callback = visible_callback(methods, "admin_ev_page_1_1")

        methods = await feed_callback(back_callback)
        listing = next(method for method in methods if isinstance(method, EditMessageText))
        assert "страница 2 из 2" in listing.text
        assert target.code in listing.text
        reopened_callback = visible_callback(methods, f"admin_ev_open_{target.id}_1_1")

        methods = await feed_callback(reopened_callback)
        reopened_card = next(method for method in methods if isinstance(method, EditMessageText))
        assert updated_title in reopened_card.text
        assert visible_callback(methods, "admin_ev_page_1_1") == "admin_ev_page_1_1"
        assert "Сценарии мероприятий и цепочки напоминаний" not in reopened_card.text
    finally:
        for router in routers:
            router._parent_router = original_parents[router]


@pytest.mark.asyncio
async def test_telegram_real_dispatcher_limits_error_screen_edit_and_reopen_journey(db_session_maker, monkeypatch):
    """Full A -> Z -> A real dispatcher journey:
    1. Enter description exceeding limit
    2. Click Save
    3. Get validated outgoing error screen explaining excess without AttributeError
    4. See working edit buttons
    5. Fix text
    6. Save successfully
    7. Reopen event and verify updated description in DB and card.
    """
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import create_scenario
    from database import EventScenarioMedia

    now = datetime.utcnow()
    scen = await create_scenario(
        db_session_maker,
        code="tg_journey_limits",
        event_type="webinar",
        title="Journey Limits Webinar",
        starts_at_utc=now + timedelta(days=2),
        link="https://example.test/journey",
        media_format="separate_media_top",
    )
    async with db_session_maker() as session:
        session.add(EventScenarioMedia(
            event_id=scen.id,
            file_type="photo",
            file_id="tg_photo_1",
            sort_order=0,
        ))
        await session.commit()
    from event_admin_service import update_scenario_fields
    await update_scenario_fields(db_session_maker, scen.id, media_format="caption")

    import handlers

    routers = (event_admin_router, handlers.router)
    original_parents = {r: r._parent_router for r in routers}
    dp = Dispatcher(storage=MemoryStorage())
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    for router in routers:
        router._parent_router = None
        dp.include_router(router)

    def extract_buttons(methods):
        for m in methods:
            if isinstance(m, (EditMessageText, SendMessage)) and m.reply_markup is not None:
                return [b.callback_data for row in m.reply_markup.inline_keyboard for b in row if b.callback_data]
        return []

    try:
        # Step 1: Open description editor
        methods = await _feed_callback(dp, bot, f"admin_ev_edit_description_{scen.id}")
        assert any(isinstance(m, EditMessageText) for m in methods)

        # Step 2: Feed long description exceeding caption limit (950 chars > 900)
        long_desc = "Д" * 950
        methods = await _feed_message(dp, bot, long_desc)
        assert any(isinstance(m, SendMessage) for m in methods)

        # Step 3: Click Save
        methods = await _feed_callback(dp, bot, f"admin_ev_description_save_{scen.id}")
        error_msg = next(m for m in methods if isinstance(m, EditMessageText))
        # Verify validated outgoing message explaining excess
        assert "Невозможно сохранить" in error_msg.text
        assert "TELEGRAM" in error_msg.text
        assert "Фактическая длина:" in error_msg.text
        assert "лимит:" in error_msg.text
        assert "Превышение:" in error_msg.text

        # Step 4: Verify working edit buttons on error screen
        buttons = extract_buttons([error_msg])
        assert f"admin_ev_edit_description_{scen.id}" in buttons
        assert f"admin_ev_edit_title_{scen.id}" in buttons
        assert f"admin_ev_format_menu_{scen.id}" in buttons
        assert f"admin_ev_description_save_{scen.id}" in buttons
        assert f"admin_ev_cancel_draft_{scen.id}" in buttons

        # Step 5: Click edit description button
        methods = await _feed_callback(dp, bot, f"admin_ev_edit_description_{scen.id}")
        assert any(isinstance(m, EditMessageText) for m in methods)

        # Step 6: Fix text to valid length
        valid_desc = "Корректное описание вебинара для проверки сохранения."
        methods = await _feed_message(dp, bot, valid_desc)

        # Step 7: Click Save
        methods = await _feed_callback(dp, bot, f"admin_ev_description_save_{scen.id}")
        success_msg = next(m for m in methods if isinstance(m, EditMessageText))
        assert "сохранены" in success_msg.text

        # Step 8: Reopen event card and check result
        methods = await _feed_callback(dp, bot, f"admin_ev_open_{scen.id}")
        card = next(m for m in methods if isinstance(m, EditMessageText))
        assert valid_desc in card.text

        async with db_session_maker() as session:
            refreshed = await session.get(EventScenario, scen.id)
            assert refreshed.description == valid_desc
    finally:
        for router in routers:
            router._parent_router = original_parents[router]


@pytest.mark.asyncio
async def test_telegram_real_dispatcher_custom_emoji_title_journey(db_session_maker, monkeypatch):
    """Real dispatcher journey for custom emoji support in title."""
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import create_scenario

    now = datetime.utcnow()
    scen = await create_scenario(
        db_session_maker,
        code="tg_journey_custom_emoji",
        event_type="webinar",
        title="Обычное название",
        starts_at_utc=now + timedelta(days=2),
        link="https://example.test/emoji",
    )

    import handlers

    routers = (event_admin_router, handlers.router)
    original_parents = {r: r._parent_router for r in routers}
    dp = Dispatcher(storage=MemoryStorage())
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    for router in routers:
        router._parent_router = None
        dp.include_router(router)

    try:
        # Start edit title
        methods = await _feed_callback(dp, bot, f"admin_ev_edit_title_{scen.id}")
        assert any(isinstance(m, EditMessageText) for m in methods)

        # Send a real Telegram custom-emoji entity, as Bot API delivers it.
        emoji_id = "5368324170671202286"
        emoji_title = "🔥 Горячий вебинар"
        normalized_title = '<tg-emoji emoji-id="5368324170671202286">🔥</tg-emoji> Горячий вебинар'
        user = _make_tg_user()
        chat = Chat(id=user.id, type="private")
        entity_message = Message(
            message_id=101,
            date=datetime.now(),
            chat=chat,
            from_user=user,
            text=emoji_title,
            entities=[MessageEntity(type="custom_emoji", offset=0, length=2, custom_emoji_id=emoji_id)],
        )
        bot.reset_mock()
        await dp.feed_update(bot, Update(update_id=3, message=entity_message))
        methods = [call.args[0] for call in bot.call_args_list if call.args and hasattr(call.args[0], "__class__")]
        saved_msg = next(m for m in methods if isinstance(m, SendMessage))
        # Must preserve <tg-emoji> with normalized emoji-id without escaping into &lt;tg-emoji&gt;
        assert normalized_title in saved_msg.text
        assert '&lt;tg-emoji' not in saved_msg.text

        # Reopen card
        methods = await _feed_callback(dp, bot, f"admin_ev_open_{scen.id}")
        card = next(m for m in methods if isinstance(m, EditMessageText))
        assert normalized_title in card.text

        async with db_session_maker() as session:
            db_scen = await session.get(EventScenario, scen.id)
            assert db_scen.title == normalized_title
            assert len(db_scen.title) <= 1000
    finally:
        for router in routers:
            router._parent_router = original_parents[router]


@pytest.mark.ui_journey
@pytest.mark.asyncio
async def test_telegram_pagination_page_2_edit_menu_back_roundtrip_journey(db_session_maker, monkeypatch):
    """Full journey:
    List page 2 -> card -> edit menu -> format menu -> Back -> Save/Cancel draft -> card -> To list = original page 2 with original filter.
    """
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import create_scenario

    now = datetime.utcnow()
    # Create 8 events (PAGE_SIZE=6 -> 2 pages)
    scenarios = []
    for i in range(8):
        scen = await create_scenario(
            db_session_maker,
            code=f"tg_page2_ev_{i:02d}",
            event_type="webinar",
            title=f"Вебинар {i:02d}",
            starts_at_utc=now + timedelta(days=i + 1),
            link=f"https://example.test/{i}",
            media_format="separate_media_top",
        )
        scenarios.append(scen)

    import handlers

    routers = (event_admin_router, handlers.router)
    original_parents = {r: r._parent_router for r in routers}
    dp = Dispatcher(storage=MemoryStorage())
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    for router in routers:
        router._parent_router = None
        dp.include_router(router)

    def find_callback(methods, prefix):
        for m in methods:
            if isinstance(m, (EditMessageText, SendMessage)) and m.reply_markup is not None:
                for row in m.reply_markup.inline_keyboard:
                    for b in row:
                        if b.callback_data and b.callback_data.startswith(prefix):
                            return b.callback_data
        raise AssertionError(f"Button with prefix {prefix!r} not found in methods")

    try:
        # Step 1: Open list
        methods = await _feed_callback(dp, bot, "admin_events")
        listing = next(m for m in methods if isinstance(m, EditMessageText))
        assert "страница 1 из 2" in listing.text

        # Step 2: Navigate to Page 2 (0-indexed page 1)
        page2_cb = find_callback(methods, "admin_ev_page_1_0")
        methods = await _feed_callback(dp, bot, page2_cb)
        listing = next(m for m in methods if isinstance(m, EditMessageText))
        assert "страница 2 из 2" in listing.text

        # Find target scenario from page 2 (index 0 or 1 because list is sorted by starts_at desc)
        target = scenarios[0]
        open_cb = find_callback(methods, f"admin_ev_open_{target.id}_1_0")

        # Step 3: Open card from page 2
        methods = await _feed_callback(dp, bot, open_cb)
        card = next(m for m in methods if isinstance(m, EditMessageText))
        assert target.title in card.text

        # Step 4: Open edit menu from card
        edit_cb = find_callback(methods, f"admin_ev_edit_menu_{target.id}_1_0")
        methods = await _feed_callback(dp, bot, edit_cb)
        edit_menu = next(m for m in methods if isinstance(m, EditMessageText))
        assert "Редактирование мероприятия" in edit_menu.text

        # Step 5: Open format menu
        fmt_cb = find_callback(methods, f"admin_ev_format_menu_{target.id}")
        methods = await _feed_callback(dp, bot, fmt_cb)
        fmt_menu = next(m for m in methods if isinstance(m, EditMessageText))
        assert "Формат доставки" in fmt_menu.text

        # Step 6: Back from format menu -> returns to edit menu with pagination intact
        back_to_edit = find_callback(methods, f"admin_ev_edit_menu_{target.id}_1_0")
        methods = await _feed_callback(dp, bot, back_to_edit)
        edit_menu = next(m for m in methods if isinstance(m, EditMessageText))
        assert "Редактирование мероприятия" in edit_menu.text

        # Step 7: Back from edit menu to card -> returns to card with page 2 back intact
        back_to_card = find_callback(methods, f"admin_ev_open_{target.id}_1_0")
        methods = await _feed_callback(dp, bot, back_to_card)
        card = next(m for m in methods if isinstance(m, EditMessageText))
        assert target.title in card.text

        # Step 8: Back from card to list -> returns to original Page 2
        back_to_list = find_callback(methods, "admin_ev_page_1_0")
        methods = await _feed_callback(dp, bot, back_to_list)
        listing = next(m for m in methods if isinstance(m, EditMessageText))
        assert "страница 2 из 2" in listing.text

        # Step 9: Full cycle with description + format draft change and Save
        # Re-open card and edit menu
        methods = await _feed_callback(dp, bot, open_cb)
        methods = await _feed_callback(dp, bot, edit_cb)
        # Start description draft
        methods = await _feed_callback(dp, bot, f"admin_ev_edit_description_{target.id}")
        await _feed_message(dp, bot, "Новое описание в черновике для страницы 2")
        # Now open format menu while draft exists
        methods = await _feed_callback(dp, bot, fmt_cb)
        set_fmt_cb = find_callback(methods, f"admin_ev_set_fmt_{target.id}_separate_text_top")
        methods = await _feed_callback(dp, bot, set_fmt_cb)
        draft_msg = next(m for m in methods if isinstance(m, EditMessageText))
        assert "Черновик ещё не сохранён" in draft_msg.text

        # First verify Cancel draft returns to card preserving pagination back
        cancel_draft_cb = find_callback(methods, f"admin_ev_cancel_draft_{target.id}")
        methods = await _feed_callback(dp, bot, cancel_draft_cb)
        card = next(m for m in methods if isinstance(m, EditMessageText))
        assert "Черновик сброшен" in card.text
        assert find_callback(methods, "admin_ev_page_1_0") == "admin_ev_page_1_0"

        # Now re-create draft and Save
        methods = await _feed_callback(dp, bot, edit_cb)
        methods = await _feed_callback(dp, bot, f"admin_ev_edit_description_{target.id}")
        await _feed_message(dp, bot, "Сохранённое описание для страницы 2")
        methods = await _feed_callback(dp, bot, fmt_cb)
        methods = await _feed_callback(dp, bot, set_fmt_cb)
        save_cb = find_callback(methods, f"admin_ev_description_save_{target.id}")
        methods = await _feed_callback(dp, bot, save_cb)
        card = next(m for m in methods if isinstance(m, EditMessageText))
        assert "сохранены" in card.text

        # Back to list -> returns to original Page 2
        back_to_list = find_callback(methods, "admin_ev_page_1_0")
        methods = await _feed_callback(dp, bot, back_to_list)
        listing = next(m for m in methods if isinstance(m, EditMessageText))
        assert "страница 2 из 2" in listing.text
        assert target.code in listing.text
    finally:
        for router in routers:
            router._parent_router = original_parents[router]


@pytest.mark.asyncio
async def test_telegram_cross_event_draft_stale_callback_rejection_journey(db_session_maker, monkeypatch):
    """Negative journey:
    Admin has active draft for Event A.
    Stale callback buttons for Event B must NOT save, reset, or transfer draft of Event A.
    DB records for both events remain completely unchanged.
    """
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import create_scenario

    now = datetime.utcnow()
    event_a = await create_scenario(
        db_session_maker,
        code="tg_draft_event_a",
        event_type="webinar",
        title="Event A Title",
        starts_at_utc=now + timedelta(days=2),
        link="https://example.test/a",
        description="Original Event A Description",
        media_format="separate_media_top",
    )
    event_b = await create_scenario(
        db_session_maker,
        code="tg_draft_event_b",
        event_type="webinar",
        title="Event B Title",
        starts_at_utc=now + timedelta(days=3),
        link="https://example.test/b",
        description="Original Event B Description",
        media_format="separate_media_top",
    )

    import handlers

    routers = (event_admin_router, handlers.router)
    original_parents = {r: r._parent_router for r in routers}
    dp = Dispatcher(storage=MemoryStorage())
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    for router in routers:
        router._parent_router = None
        dp.include_router(router)

    try:
        # Step 1: Open description editor for Event A and type new draft
        await _feed_callback(dp, bot, f"admin_ev_edit_description_{event_a.id}")
        draft_text = "Новый незафиксированный черновик для Мероприятия А"
        await _feed_message(dp, bot, draft_text)

        # Step 2: Send STALE Save callback from Event B
        methods = await _feed_callback(dp, bot, f"admin_ev_description_save_{event_b.id}")
        answer = next(m for m in methods if isinstance(m, AnswerCallbackQuery))
        assert answer.show_alert is True
        assert "Черновик принадлежит другому мероприятию" in answer.text

        # Verify DB for both events is UNCHANGED
        async with db_session_maker() as session:
            ea = await session.get(EventScenario, event_a.id)
            eb = await session.get(EventScenario, event_b.id)
            assert ea.description == "Original Event A Description"
            assert eb.description == "Original Event B Description"

        # Step 3: Send STALE Cancel callback from Event B
        methods = await _feed_callback(dp, bot, f"admin_ev_cancel_draft_{event_b.id}")
        answer = next(m for m in methods if isinstance(m, AnswerCallbackQuery))
        assert answer.show_alert is True
        assert "Сначала завершите или отмените" in answer.text

        # Verify DB and draft still unchanged
        async with db_session_maker() as session:
            ea = await session.get(EventScenario, event_a.id)
            eb = await session.get(EventScenario, event_b.id)
            assert ea.description == "Original Event A Description"
            assert eb.description == "Original Event B Description"

        # Step 4: Send STALE Format callback from Event B
        methods = await _feed_callback(dp, bot, f"admin_ev_set_fmt_{event_b.id}_caption")
        answer = next(m for m in methods if isinstance(m, AnswerCallbackQuery))
        assert answer.show_alert is True
        assert "Сначала завершите или отмените" in answer.text

        async with db_session_maker() as session:
            eb = await session.get(EventScenario, event_b.id)
            assert eb.media_format == "separate_media_top"

        # Step 5: Send STALE Edit Title callback from Event B
        methods = await _feed_callback(dp, bot, f"admin_ev_edit_title_{event_b.id}")
        answer = next(m for m in methods if isinstance(m, AnswerCallbackQuery))
        assert answer.show_alert is True
        assert "Сначала завершите или отмените" in answer.text

        # Step 6: Now send VALID Save callback for Event A
        methods = await _feed_callback(dp, bot, f"admin_ev_description_save_{event_a.id}")
        edit_msg = next(m for m in methods if isinstance(m, EditMessageText))
        assert "сохранены" in edit_msg.text

        # Verify Event A in DB is updated, Event B remains unchanged
        async with db_session_maker() as session:
            ea = await session.get(EventScenario, event_a.id)
            eb = await session.get(EventScenario, event_b.id)
            assert ea.description == draft_text
            assert eb.description == "Original Event B Description"
    finally:
        for router in routers:
            router._parent_router = original_parents[router]


@pytest.mark.asyncio
async def test_telegram_holistic_draft_media_save_journey(db_session_maker, monkeypatch):
    """Holistic A -> Я -> A journey:
    1. Start from scenario card callback
    2. Inspect visible keyboard to navigate to edit menu
    3. Modify description, title with entities/custom-emoji, format, add photo + video, reorder
    4. Cancel draft -> verify DB untouched, reopen card
    5. Phase 2: Repeat modifications through Dispatcher, execute Save Media
    6. Validate all outgoing aiogram method calls via Pydantic model_validate
    7. Verify DB atomically committed with media provenance (source_platform, source_file_id)
    8. Reopen card -> verify clean rendering without double escaping
    9. Execute runtime delivery plan -> verify correct message order and delivery
    """
    import handlers
    from aiogram.types import PhotoSize, Video
    from aiogram.methods import SendMessage, EditMessageText, AnswerCallbackQuery
    from database import EventScenario, EventScenarioMedia
    from event_admin_service import create_scenario, list_scenario_media
    from event_delivery_worker import _build_delivery_plan, _send_delivery_part
    from sqlalchemy.orm import selectinload

    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))

    scen = await create_scenario(
        db_session_maker,
        code="tg_holistic_draft",
        event_type="webinar",
        title="Initial TG Title",
        starts_at_utc=datetime.utcnow() + timedelta(days=3),
        link="https://example.test/tg-draft",
        description="Initial TG Description",
        media_format="separate_media_top",
    )

    routers = (event_admin_router, handlers.router)
    original_parents = {r: r._parent_router for r in routers}
    dp = Dispatcher(storage=MemoryStorage())
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    for router in routers:
        router._parent_router = None
        dp.include_router(router)

    async def feed_callback(data: str):
        methods = await _feed_callback(dp, bot, data)
        for method in methods:
            if hasattr(method, "model_dump"):
                method.__class__.model_validate(method.model_dump())
        return methods

    async def feed_message(text: str, **kwargs):
        methods = await _feed_message(dp, bot, text, **kwargs)
        for method in methods:
            if hasattr(method, "model_dump"):
                method.__class__.model_validate(method.model_dump())
        return methods

    def find_button(methods, prefix: str) -> str:
        for method in methods:
            if not isinstance(method, (EditMessageText, SendMessage)) or method.reply_markup is None:
                continue
            for row in method.reply_markup.inline_keyboard:
                for button in row:
                    if button.callback_data and button.callback_data.startswith(prefix):
                        return button.callback_data
        raise AssertionError(f"Button with prefix '{prefix}' not found in methods: {methods}")

    try:
        # Step 1: Open card
        methods = await feed_callback(f"admin_ev_open_{scen.id}_0_1")
        edit_menu_cb = find_button(methods, f"admin_ev_edit_menu_{scen.id}")

        # Phase 1: Modify description, title, format, media, then CANCEL
        # 1. Edit description
        methods = await feed_callback(edit_menu_cb)
        desc_btn_cb = find_button(methods, f"admin_ev_edit_description_{scen.id}")
        methods = await feed_callback(desc_btn_cb)
        methods = await feed_message("Pending Draft Description")
        assert any("Pending Draft Description" in (m.text or "") for m in methods if isinstance(m, (SendMessage, EditMessageText)))
        edit_menu_cb = find_button(methods, f"admin_ev_edit_menu_{scen.id}")
        methods = await feed_callback(edit_menu_cb)

        # 2. Edit title with custom emoji
        title_btn_cb = find_button(methods, f"admin_ev_edit_title_{scen.id}")
        methods = await feed_callback(title_btn_cb)
        methods = await feed_message('Draft Title & <tg-emoji emoji-id="111">⭐</tg-emoji> <Pending>')
        # Verify no double escaping in response
        assert not any("&amp;amp;" in (m.text or "") for m in methods if isinstance(m, (SendMessage, EditMessageText)))

        # 3. Change format
        fmt_btn_cb = find_button(methods, f"admin_ev_format_menu_{scen.id}")
        methods = await feed_callback(fmt_btn_cb)
        set_fmt_cb = find_button(methods, f"admin_ev_set_fmt_{scen.id}_separate_text_top")
        methods = await feed_callback(set_fmt_cb)
        assert any("Выбран формат" in (m.text or "") for m in methods if isinstance(m, (SendMessage, EditMessageText)))

        # Reopen format menu to verify radio button indicator
        fmt_btn_cb = find_button(methods, f"admin_ev_format_menu_{scen.id}")
        methods = await feed_callback(fmt_btn_cb)
        assert any("🔘" in b.text for m in methods if hasattr(m, "reply_markup") and m.reply_markup for row in m.reply_markup.inline_keyboard for b in row if "медиа ниже" in b.text.lower())
        edit_menu_cb = find_button(methods, f"admin_ev_edit_menu_{scen.id}")
        methods = await feed_callback(edit_menu_cb)

        # 4. Add media
        media_btn_cb = find_button(methods, f"admin_ev_edit_media_{scen.id}")
        methods = await feed_callback(media_btn_cb)
        add_media_cb = find_button(methods, f"admin_ev_media_add_{scen.id}")
        methods = await feed_callback(add_media_cb)
        mock_photo = [PhotoSize(file_id="tg_photo_draft_1", file_unique_id="u1", width=100, height=100)]
        methods = await feed_message("", photo=mock_photo)
        assert any("В черновике: <b>1</b>" in (m.text or "") for m in methods if isinstance(m, (SendMessage, EditMessageText)))

        # 5. Cancel draft
        edit_menu_cb = find_button(methods, f"admin_ev_edit_menu_{scen.id}")
        methods = await feed_callback(edit_menu_cb)
        cancel_cb = find_button(methods, f"admin_ev_cancel_draft_{scen.id}")
        methods = await feed_callback(cancel_cb)

        # Verify DB is completely untouched
        async with db_session_maker() as session:
            s_after_cancel = await session.get(EventScenario, scen.id)
            assert s_after_cancel.title == "Initial TG Title"
            assert s_after_cancel.description == "Initial TG Description"
            assert s_after_cancel.media_format == "separate_media_top"
        media_after_cancel = await list_scenario_media(db_session_maker, scen.id)
        assert len(media_after_cancel) == 0

        # Phase 2: Modify description, title, format, media, and SAVE MEDIA
        methods = await feed_callback(f"admin_ev_open_{scen.id}_0_1")
        edit_menu_cb = find_button(methods, f"admin_ev_edit_menu_{scen.id}")
        methods = await feed_callback(edit_menu_cb)

        # 1. Edit description
        desc_btn_cb = find_button(methods, f"admin_ev_edit_description_{scen.id}")
        methods = await feed_callback(desc_btn_cb)
        methods = await feed_message("Final Saved Description")
        edit_menu_cb = find_button(methods, f"admin_ev_edit_menu_{scen.id}")
        methods = await feed_callback(edit_menu_cb)

        # 2. Edit title with custom emoji and HTML entities
        title_btn_cb = find_button(methods, f"admin_ev_edit_title_{scen.id}")
        methods = await feed_callback(title_btn_cb)
        methods = await feed_message('Final Title & <tg-emoji emoji-id="222">🚀</tg-emoji> <Final>')

        # 3. Change format to separate_text_top
        fmt_btn_cb = find_button(methods, f"admin_ev_format_menu_{scen.id}")
        methods = await feed_callback(fmt_btn_cb)
        set_fmt_cb = find_button(methods, f"admin_ev_set_fmt_{scen.id}_separate_text_top")
        methods = await feed_callback(set_fmt_cb)
        edit_menu_cb = find_button(methods, f"admin_ev_edit_menu_{scen.id}")
        methods = await feed_callback(edit_menu_cb)

        # 4. Add media 1 (photo)
        media_btn_cb = find_button(methods, f"admin_ev_edit_media_{scen.id}")
        methods = await feed_callback(media_btn_cb)
        add_media_cb = find_button(methods, f"admin_ev_media_add_{scen.id}")
        methods = await feed_callback(add_media_cb)
        mock_photo = [PhotoSize(file_id="tg_photo_final_1", file_unique_id="u2", width=200, height=200)]
        methods = await feed_message("", photo=mock_photo)

        # Add media 2 (video)
        add_media_cb = find_button(methods, f"admin_ev_media_add_{scen.id}")
        methods = await feed_callback(add_media_cb)
        mock_video = Video(file_id="tg_video_final_2", file_unique_id="u3", width=200, height=200, duration=10)
        methods = await feed_message("", video=mock_video)
        assert any("В черновике: <b>2</b>" in (m.text or "") for m in methods if isinstance(m, (SendMessage, EditMessageText)))

        # 5. Reorder media: move item 2 up to position 0
        move_up_cb = find_button(methods, f"admin_ev_media_up_{scen.id}_1")
        methods = await feed_callback(move_up_cb)

        # 6. Click Save Media (admin_ev_media_save_{scen.id})
        save_media_cb = find_button(methods, f"admin_ev_media_save_{scen.id}")
        methods = await feed_callback(save_media_cb)
        assert any(
            isinstance(m, (SendMessage, EditMessageText)) and "сохранены" in (m.text or "")
            for m in methods
        )

        # 7. Verify DB is atomically updated with ALL changes and media provenance
        async with db_session_maker() as session:
            s_final = await session.get(EventScenario, scen.id)
            assert s_final.title == 'Final Title &amp; <tg-emoji emoji-id="222">🚀</tg-emoji> &lt;Final&gt;'
            assert s_final.description == "Final Saved Description"
            assert s_final.media_format == "separate_text_top"
        media_final = await list_scenario_media(db_session_maker, scen.id)
        assert len(media_final) == 2
        # First item was moved up: video
        assert media_final[0].file_id == "tg_video_final_2"
        assert media_final[0].file_type == "video"
        assert media_final[0].source_platform == "telegram"
        assert media_final[0].source_file_id == "tg_video_final_2"
        # Second item: photo
        assert media_final[1].file_id == "tg_photo_final_1"
        assert media_final[1].file_type == "photo"
        assert media_final[1].source_platform == "telegram"
        assert media_final[1].source_file_id == "tg_photo_final_1"

        # 8. Reopen card callback and verify clean rendered card
        methods = await feed_callback(f"admin_ev_open_{scen.id}_0_1")
        card_msg = next(m for m in methods if isinstance(m, (SendMessage, EditMessageText)))
        assert "&amp;amp;" not in card_msg.text
        assert "&amp;lt;" not in card_msg.text
        assert "Final Title &amp;" in card_msg.text
        assert "Медиа:</b> 2" in card_msg.text

        # 9. Test runtime delivery plan execution
        async with db_session_maker() as session:
            scen_delivery = (await session.execute(
                select(EventScenario).where(EventScenario.id == scen.id).options(selectinload(EventScenario.media))
            )).scalar_one()

        plan = _build_delivery_plan(scen_delivery, scen.id, "Runtime announcement", include_media=True)
        # Because format is separate_text_top: text first, media second
        assert len(plan) >= 2
        assert plan[0]["kind"] == "text"
        assert plan[1]["kind"] == "media"
        assert len(plan[1]["items"]) == 2

        # Send text part
        bot_delivery = AsyncMock(spec=Bot)
        bot_delivery.send_message = AsyncMock(return_value=MagicMock(message_id=501))
        bot_delivery.send_media_group = AsyncMock(return_value=[MagicMock(message_id=502), MagicMock(message_id=503)])

        sent_text_id = await _send_delivery_part(
            plan[0],
            platform="telegram",
            user_id=8888,
            event_id=scen.id,
            is_last=False,
            bot=bot_delivery,
            max_client=None,
        )
        assert sent_text_id == "501"

        sent_media_id = await _send_delivery_part(
            plan[1],
            platform="telegram",
            user_id=8888,
            event_id=scen.id,
            is_last=True,
            bot=bot_delivery,
            max_client=None,
        )
        assert sent_media_id == "502,503"
    finally:
        for router in routers:
            router._parent_router = original_parents[router]


@pytest.mark.asyncio
async def test_telegram_format_menu_draft_reflection_and_back(db_session_maker, monkeypatch):
    """Verifies show_format_menu marks active draft_format with radio button and preserves it across Back."""
    import handlers
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import create_scenario

    scen = await create_scenario(
        db_session_maker,
        code="tg_fmt_draft",
        event_type="webinar",
        title="Format Draft Event",
        starts_at_utc=datetime.utcnow() + timedelta(days=3),
        link="https://example.test/fmt-draft",
        description="Format Draft Description",
        media_format="separate_media_top",
    )

    routers = (event_admin_router, handlers.router)
    original_parents = {r: r._parent_router for r in routers}
    dp = Dispatcher(storage=MemoryStorage())
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    for router in routers:
        router._parent_router = None
        dp.include_router(router)

    try:
        # Open format menu initially - should reflect separate_media_top
        methods = await _feed_callback(dp, bot, f"admin_ev_format_menu_{scen.id}")
        msg = next(m for m in methods if isinstance(m, (EditMessageText, SendMessage)))
        button_texts = [b.text for row in msg.reply_markup.inline_keyboard for b in row]
        assert any("🔘 🖼️ Текст отдельно, медиа выше" in t for t in button_texts)

        # Set draft format to separate_text_top
        await _feed_callback(dp, bot, f"admin_ev_set_fmt_{scen.id}_separate_text_top")

        # Reopen format menu - should now reflect draft_format (separate_text_top)
        methods = await _feed_callback(dp, bot, f"admin_ev_format_menu_{scen.id}")
        msg = next(m for m in methods if isinstance(m, (EditMessageText, SendMessage)))
        button_texts = [b.text for row in msg.reply_markup.inline_keyboard for b in row]
        assert any("🔘 📝 Текст отдельно, медиа ниже" in t for t in button_texts)
        assert any("⚪️ 🖼️ Текст отдельно, медиа выше" in t for t in button_texts)

        # Navigate Back to edit menu
        methods = await _feed_callback(dp, bot, f"admin_ev_edit_menu_{scen.id}")
        edit_msg = next(m for m in methods if isinstance(m, (EditMessageText, SendMessage)))
        edit_button_texts = [b.text for row in edit_msg.reply_markup.inline_keyboard for b in row]
        assert any("Формат:" in t and "Текст отдельно, медиа ниже" in t for t in edit_button_texts)
    finally:
        for router in routers:
            router._parent_router = original_parents[router]


@pytest.mark.asyncio
async def test_telegram_visiting_editor_does_not_block_foreign_events(db_session_maker, monkeypatch):
    """Merely opening edit menu, description editor or media editor without dirty changes does not block foreign events."""
    import handlers
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import create_scenario

    event_a = await create_scenario(
        db_session_maker,
        code="tg_visit_a",
        event_type="webinar",
        title="Event A",
        starts_at_utc=datetime.utcnow() + timedelta(days=3),
        link="https://example.test/a",
    )
    event_b = await create_scenario(
        db_session_maker,
        code="tg_visit_b",
        event_type="webinar",
        title="Event B",
        starts_at_utc=datetime.utcnow() + timedelta(days=4),
        link="https://example.test/b",
    )

    routers = (event_admin_router, handlers.router)
    original_parents = {r: r._parent_router for r in routers}
    dp = Dispatcher(storage=MemoryStorage())
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    for router in routers:
        router._parent_router = None
        dp.include_router(router)

    try:
        # 1. Visit Event A edit menu
        await _feed_callback(dp, bot, f"admin_ev_edit_menu_{event_a.id}")

        # 2. Visit Event A description editor (do NOT enter any text)
        await _feed_callback(dp, bot, f"admin_ev_edit_description_{event_a.id}")

        # 3. Visit Event A media editor (do NOT add or remove media)
        await _feed_callback(dp, bot, f"admin_ev_edit_media_{event_a.id}")

        # 4. Now open Event B card
        methods = await _feed_callback(dp, bot, f"admin_ev_open_{event_b.id}")
        assert not any(isinstance(m, AnswerCallbackQuery) and m.show_alert for m in methods)
        card_msg = next(m for m in methods if isinstance(m, (EditMessageText, SendMessage)))
        assert "Event B" in card_msg.text

        # 5. Open Event B edit menu
        methods = await _feed_callback(dp, bot, f"admin_ev_edit_menu_{event_b.id}")
        assert not any(isinstance(m, AnswerCallbackQuery) and m.show_alert for m in methods)
        menu_msg = next(m for m in methods if isinstance(m, (EditMessageText, SendMessage)))
        assert "Event B" in menu_msg.text
    finally:
        for router in routers:
            router._parent_router = original_parents[router]


@pytest.mark.asyncio
async def test_telegram_title_length_visible_and_storage_validation(db_session_maker, monkeypatch):
    """Verifies that visible text limit (1000) and raw storage limit (1000) are both validated with informative errors."""
    import handlers
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import create_scenario

    scen = await create_scenario(
        db_session_maker,
        code="tg_title_limits",
        event_type="webinar",
        title="Initial Valid Title",
        starts_at_utc=datetime.utcnow() + timedelta(days=3),
        link="https://example.test/limits",
    )

    routers = (event_admin_router, handlers.router)
    original_parents = {r: r._parent_router for r in routers}
    dp = Dispatcher(storage=MemoryStorage())
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    for router in routers:
        router._parent_router = None
        dp.include_router(router)

    try:
        # Start editing title
        await _feed_callback(dp, bot, f"admin_ev_edit_title_{scen.id}")

        # 1. Send title with visible text > 1000 chars
        invalid_visible = "X" * 1005
        methods = await _feed_message(dp, bot, invalid_visible)
        msg = next(m for m in methods if isinstance(m, SendMessage))
        assert "1000 символов" in msg.text
        assert "1005" in msg.text

        # 2. Send title with visible text <= 1000 chars, but raw HTML with tg-emoji tags > 1000 chars
        emojis = '<tg-emoji emoji-id="1234567890123456789">🔥</tg-emoji>' * 25
        raw_storage_overflow = ("Y" * 50) + emojis
        methods = await _feed_message(dp, bot, raw_storage_overflow)
        msg = next(m for m in methods if isinstance(m, SendMessage))
        assert "предел хранения (1000 символов)" in msg.text

        # 3. Send valid title with custom emoji
        valid_with_emoji = '<tg-emoji emoji-id="1234567890123456789">🔥</tg-emoji> Valid Emoji Title'
        methods = await _feed_message(dp, bot, valid_with_emoji)
        assert any(isinstance(m, (SendMessage, EditMessageText)) and "обновлено" in (m.text or "") for m in methods)
    finally:
        for router in routers:
            router._parent_router = original_parents[router]


@pytest.mark.ui_journey
@pytest.mark.asyncio
async def test_telegram_premium_emoji_paginated_journey_a_to_ya_to_a(db_session_maker, monkeypatch):
    """Full journey A -> Я -> A:
    Paginated list (page 2) -> real button with clean title & Unicode emoji -> card with custom emoji in HTML ->
    nested screens (archive, reschedule, copy, steps, monitoring) without &amp;amp; or literal <tg-emoji> ->
    Back -> original list page 2 with original filter.
    Every outgoing Telegram method is strictly validated through its Pydantic model.
    """
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import create_scenario
    from database import EventStep
    import handlers

    now = datetime.utcnow()
    # 6 filler scenarios on page 1 so target scenario lands on page 2 (0-indexed page 1)
    base_start = now + timedelta(days=20)
    for idx in range(6):
        await create_scenario(
            db_session_maker,
            code=f"tg_prem_filler_{idx}",
            event_type="webinar",
            title=f"Filler Event {idx}",
            starts_at_utc=base_start + timedelta(days=idx + 1),
            link=f"https://example.test/filler/{idx}",
        )

    target_title = '<tg-emoji emoji-id="5368324170671202286">🔥</tg-emoji> Premium Q&amp;A &lt;Special Event&gt; with Long Title for Truncation Testing'
    target = await create_scenario(
        db_session_maker,
        code="tg_prem_target",
        event_type="webinar",
        title=target_title,
        starts_at_utc=base_start - timedelta(days=2),
        link="https://example.test/target",
        description="Описание целевого мероприятия с <b>HTML</b>",
        media_format="separate_media_top",
    )

    async with db_session_maker() as session:
        step = EventStep(
            event_id=target.id,
            offset_minutes=60,
            message_text="Напоминание о вебинаре",
        )
        session.add(step)
        await session.commit()

    routers = (event_admin_router, handlers.router)
    original_parents = {r: r._parent_router for r in routers}
    dp = Dispatcher(storage=MemoryStorage())
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    for router in routers:
        router._parent_router = None
        dp.include_router(router)

    def validate_methods(methods):
        for m in methods:
            m.__class__.model_validate(m.model_dump())
        return methods

    def find_button(methods, prefix):
        for m in methods:
            if isinstance(m, (EditMessageText, SendMessage)) and m.reply_markup is not None:
                for row in m.reply_markup.inline_keyboard:
                    for b in row:
                        if b.callback_data and b.callback_data.startswith(prefix):
                            return b
        raise AssertionError(f"Button with prefix {prefix!r} not found in methods")

    try:
        # 1. Open list (Page 1)
        methods = validate_methods(await _feed_callback(dp, bot, "admin_events"))
        # Navigate to Page 2 (0-indexed page 1)
        page2_btn = find_button(methods, "admin_ev_page_1_0")
        methods = validate_methods(await _feed_callback(dp, bot, page2_btn.callback_data))
        listing_msg = next(m for m in methods if isinstance(m, EditMessageText))
        assert "страница 2 из 2" in listing_msg.text

        # 2. Inspect target button in list: plain text, Unicode fallback emoji, no raw HTML tags
        target_btn = find_button(methods, f"admin_ev_open_{target.id}_1_0")
        assert "<tg-emoji" not in target_btn.text
        assert "</tg-emoji>" not in target_btn.text
        assert "&amp;" not in target_btn.text
        assert "&lt;" not in target_btn.text
        assert "🔥" in target_btn.text
        assert "Premium Q&A <Special Event>" in target_btn.text
        assert target_btn.text.endswith("… (" + target.starts_at_msk[:10] + ")")

        # 3. Open scenario card
        methods = validate_methods(await _feed_callback(dp, bot, target_btn.callback_data))
        card_msg = next(m for m in methods if isinstance(m, EditMessageText))
        # Card preserves custom emoji in HTML, without double-escaping &amp;amp;
        assert '<tg-emoji emoji-id="5368324170671202286">🔥</tg-emoji>' in card_msg.text
        assert "&amp;amp;" not in card_msg.text
        assert "&amp;lt;" not in card_msg.text
        assert "Premium Q&amp;A &lt;Special Event&gt;" in card_msg.text

        # 4. Nested screen: Archive
        archive_btn = find_button(methods, f"admin_ev_archive_ask_{target.id}")
        methods = validate_methods(await _feed_callback(dp, bot, archive_btn.callback_data))
        archive_msg = next(m for m in methods if isinstance(m, EditMessageText))
        assert '<tg-emoji emoji-id="5368324170671202286">🔥</tg-emoji>' in archive_msg.text
        assert "&amp;amp;" not in archive_msg.text
        assert "&lt;tg-emoji" not in archive_msg.text
        # Back to card
        card_back_btn = find_button(methods, f"admin_ev_open_{target.id}_1_0")
        methods = validate_methods(await _feed_callback(dp, bot, card_back_btn.callback_data))

        # 5. Nested screen: Reschedule
        resched_btn = find_button(methods, f"admin_ev_reschedule_{target.id}")
        methods = validate_methods(await _feed_callback(dp, bot, resched_btn.callback_data))
        resched_msg = next(m for m in methods if isinstance(m, EditMessageText))
        assert '<tg-emoji emoji-id="5368324170671202286">🔥</tg-emoji>' in resched_msg.text
        assert "&amp;amp;" not in resched_msg.text
        assert "&lt;tg-emoji" not in resched_msg.text
        # Back to card
        card_back_btn = find_button(methods, f"admin_ev_open_{target.id}_1_0")
        methods = validate_methods(await _feed_callback(dp, bot, card_back_btn.callback_data))

        # 6. Nested screen: Copy
        copy_btn = find_button(methods, f"admin_ev_copy_{target.id}")
        methods = validate_methods(await _feed_callback(dp, bot, copy_btn.callback_data))
        copy_msg = next(m for m in methods if isinstance(m, EditMessageText))
        assert '<tg-emoji emoji-id="5368324170671202286">🔥</tg-emoji>' in copy_msg.text
        assert "&amp;amp;" not in copy_msg.text
        assert "&lt;tg-emoji" not in copy_msg.text
        # Back to card
        card_back_btn = find_button(methods, f"admin_ev_open_{target.id}_1_0")
        methods = validate_methods(await _feed_callback(dp, bot, card_back_btn.callback_data))

        # 7. Nested screen: Steps
        steps_btn = find_button(methods, f"admin_ev_steps_{target.id}")
        methods = validate_methods(await _feed_callback(dp, bot, steps_btn.callback_data))
        steps_msg = next(m for m in methods if isinstance(m, EditMessageText))
        assert '<tg-emoji emoji-id="5368324170671202286">🔥</tg-emoji>' in steps_msg.text
        assert "&amp;amp;" not in steps_msg.text
        assert "&lt;tg-emoji" not in steps_msg.text
        # Back to card
        card_back_btn = find_button(methods, f"admin_ev_open_{target.id}_1_0")
        methods = validate_methods(await _feed_callback(dp, bot, card_back_btn.callback_data))

        # 8. Nested screen: Monitoring / Stats
        stats_btn = find_button(methods, f"admin_ev_stats_{target.id}")
        methods = validate_methods(await _feed_callback(dp, bot, stats_btn.callback_data))
        stats_msg = next(m for m in methods if isinstance(m, EditMessageText))
        assert '<tg-emoji emoji-id="5368324170671202286">🔥</tg-emoji>' in stats_msg.text
        assert "&amp;amp;" not in stats_msg.text
        assert "&lt;tg-emoji" not in stats_msg.text
        # Back to card
        card_back_btn = find_button(methods, f"admin_ev_open_{target.id}_1_0")
        methods = validate_methods(await _feed_callback(dp, bot, card_back_btn.callback_data))

        # 9. Return to list (Page 2) with original filter
        to_list_btn = find_button(methods, "admin_ev_page_1_0")
        methods = validate_methods(await _feed_callback(dp, bot, to_list_btn.callback_data))
        final_listing = next(m for m in methods if isinstance(m, EditMessageText))
        assert "страница 2 из 2" in final_listing.text
        assert find_button(methods, f"admin_ev_open_{target.id}_1_0") is not None
    finally:
        for router in routers:
            router._parent_router = original_parents[router]


@pytest.mark.asyncio
async def test_telegram_scenarios_list_button_preserves_raw_legacy_and_escaped_titles(db_session_maker, monkeypatch):
    """
    Verifies that Telegram scenario list buttons preserve literal angle brackets like '<Новый сезон>'
    both for raw legacy titles and HTML-escaped titles, convert custom emojis to Unicode fallback,
    and strip only allowed markup tags.
    """
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import create_scenario
    import handlers

    now = datetime.utcnow()
    # 1. Raw legacy scenario with literal angle brackets
    scen_raw = await create_scenario(
        db_session_maker,
        code="tg_raw_legacy",
        event_type="webinar",
        title="Презентация <Новый сезон>",
        starts_at_utc=now + timedelta(days=2),
        link="https://example.test/raw",
    )
    # 2. Escaped scenario with HTML entities
    scen_escaped = await create_scenario(
        db_session_maker,
        code="tg_escaped_title",
        event_type="webinar",
        title="Презентация &lt;Новый сезон&gt;",
        starts_at_utc=now + timedelta(days=3),
        link="https://example.test/escaped",
    )
    # 3. Custom emoji with literal angle brackets
    scen_emoji = await create_scenario(
        db_session_maker,
        code="tg_emoji_raw",
        event_type="webinar",
        title='<tg-emoji emoji-id="5368324170671202286">🔥</tg-emoji> Презентация &lt;Новый сезон&gt;',
        starts_at_utc=now + timedelta(days=4),
        link="https://example.test/emoji",
    )

    routers = (event_admin_router, handlers.router)
    original_parents = {r: r._parent_router for r in routers}
    dp = Dispatcher(storage=MemoryStorage())
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    for router in routers:
        router._parent_router = None
        dp.include_router(router)

    def find_button(methods, prefix):
        for m in methods:
            if isinstance(m, (EditMessageText, SendMessage)) and m.reply_markup is not None:
                for row in m.reply_markup.inline_keyboard:
                    for b in row:
                        if b.callback_data and b.callback_data.startswith(prefix):
                            return b
        raise AssertionError(f"Button with prefix {prefix!r} not found in methods")

    try:
        methods = await _feed_callback(dp, bot, "admin_events")
        for m in methods:
            m.__class__.model_validate(m.model_dump())

        # Raw legacy button check
        btn_raw = find_button(methods, f"admin_ev_open_{scen_raw.id}")
        assert "<Новый сезон>" in btn_raw.text
        assert "Презентация <Новый сезон>" in btn_raw.text

        # Escaped button check
        btn_escaped = find_button(methods, f"admin_ev_open_{scen_escaped.id}")
        assert "<Новый сезон>" in btn_escaped.text
        assert "Презентация <Новый сезон>" in btn_escaped.text
        assert "&lt;" not in btn_escaped.text
        assert "&gt;" not in btn_escaped.text

        # Emoji + raw angle brackets button check
        btn_emoji = find_button(methods, f"admin_ev_open_{scen_emoji.id}")
        assert "🔥" in btn_emoji.text
        assert "<Новый сезон>" in btn_emoji.text
        assert "<tg-emoji" not in btn_emoji.text
    finally:
        for router in routers:
            router._parent_router = original_parents[router]


def test_safe_callback_alert_text_boundaries():
    from event_admin_telegram import (
        TELEGRAM_CALLBACK_ALERT_MAX_LENGTH,
        safe_callback_alert_text,
    )

    assert safe_callback_alert_text("") == ""
    assert safe_callback_alert_text("   ") == ""
    assert safe_callback_alert_text(None) == ""

    short_msg = "Короткое сообщение"
    assert safe_callback_alert_text(short_msg) == short_msg

    exact_200 = "x" * 200
    res_200 = safe_callback_alert_text(exact_200)
    assert len(res_200) == 200
    assert res_200 == exact_200

    over_200 = "a" * 201
    res_over = safe_callback_alert_text(over_200)
    assert len(res_over) == 200
    assert res_over.endswith("…")

    huge = "О" * 500
    res_huge = safe_callback_alert_text(huge)
    assert len(res_huge) == 200
    assert res_huge.endswith("…")


def test_format_event_content_validation_alert_rules():
    from event_admin_telegram import (
        ALERT_MEDIA_CAPTION_REQUIRES_MEDIA,
        TELEGRAM_CALLBACK_ALERT_MAX_LENGTH,
        _format_event_content_validation_alert,
    )
    from event_content_validator import (
        EventContentValidationError,
        PathValidationResult,
        ScenarioValidationResult,
    )

    # 1. Missing media for caption format
    violation = PathValidationResult(
        path_id="media_mode_caption_count",
        platform="all",
        message_kind="media_validation",
        rendered_text="",
        visible_text="",
        length=0,
        limit=1,
        is_valid=False,
        excess=0,
        error_message=(
            "Режим «подпись к медиа» требует ровно 1 медиа-вложение (сейчас: 0). "
            "Прикрепите одно фото/видео или переключите формат доставки на раздельный."
        ),
        recommendation="Выберите «Текст отдельно, медиа выше/ниже» или оставьте ровно 1 медиа.",
    )
    res = ScenarioValidationResult(
        is_valid=False,
        media_format="caption",
        media_count=0,
        violations=[violation],
        all_checks=[violation],
        available_description_chars={},
    )
    err = EventContentValidationError(violation.error_message, res, actual_length=0, limit_length=1)

    alert_text = _format_event_content_validation_alert(err)
    assert alert_text == ALERT_MEDIA_CAPTION_REQUIRES_MEDIA
    assert alert_text == "Для этого формата нужно одно фото или видео. Добавьте медиа либо выберите раздельный формат"
    assert len(alert_text) <= TELEGRAM_CALLBACK_ALERT_MAX_LENGTH

    # 2. Too many media items for caption format
    violation_many = PathValidationResult(
        path_id="media_mode_caption_count",
        platform="all",
        message_kind="media_validation",
        rendered_text="",
        visible_text="",
        length=2,
        limit=1,
        is_valid=False,
        excess=1,
        error_message=(
            "Режим «подпись к медиа» требует ровно 1 медиа-вложение (сейчас: 2). "
            "Прикрепите одно фото/видео или переключите формат доставки на раздельный."
        ),
        recommendation="Выберите «Текст отдельно, медиа выше/ниже» или оставьте ровно 1 медиа.",
    )
    res_many = ScenarioValidationResult(
        is_valid=False,
        media_format="caption",
        media_count=2,
        violations=[violation_many],
        all_checks=[violation_many],
        available_description_chars={},
    )
    err_many = EventContentValidationError(violation_many.error_message, res_many, actual_length=2, limit_length=1)
    alert_many = _format_event_content_validation_alert(err_many)
    assert len(alert_many) <= TELEGRAM_CALLBACK_ALERT_MAX_LENGTH
    assert "сейчас: 2" in alert_many

    # 3. Super long message safely capped at 200
    long_msg = "Превышение лимита описания: " + ("очень длинный текст ошибки " * 15)
    err_long = EventContentValidationError(long_msg, None, actual_length=1500, limit_length=900)
    alert_long = _format_event_content_validation_alert(err_long)
    assert len(alert_long) <= TELEGRAM_CALLBACK_ALERT_MAX_LENGTH
    assert alert_long.endswith("…")


@pytest.mark.asyncio
async def test_telegram_event_format_caption_no_media_alert_and_recovery_journey(
    db_session_maker, monkeypatch
):
    """Real aiogram Dispatcher journey for format change validation alert:
    1. Open event from list.
    2. Click format selection button.
    3. Choose 'Подпись к медиа' without attachments.
    4. Verify clear, safe warning alert (not exception, len <= 200) without crashing.
    5. Verify format screen, event_id, FSM and return context preserved.
    6. Press Back, add media, save, reopen format menu, select caption (succeeds).
    7. Verify return to scenario card and initial list page.
    """
    import handlers
    from aiogram.methods import AnswerCallbackQuery, EditMessageText, SendMessage
    from aiogram.types import PhotoSize
    from database import EventScenario
    from event_admin_service import create_scenario, list_scenario_media
    from event_admin_telegram import (
        ALERT_MEDIA_CAPTION_REQUIRES_MEDIA,
        TELEGRAM_CALLBACK_ALERT_MAX_LENGTH,
    )

    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))

    scen = await create_scenario(
        db_session_maker,
        code="webinar_fmt_alert",
        event_type="webinar",
        title="Вебинар по запуску продукта",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://example.test/webinar-fmt",
        description="Подробное описание вебинара без медиа.",
        media_format="separate_media_top",
    )

    routers = (event_admin_router, handlers.router)
    original_parents = {r: r._parent_router for r in routers}
    dp = Dispatcher(storage=MemoryStorage())
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    for router in routers:
        router._parent_router = None
        dp.include_router(router)

    async def feed_callback(data: str):
        methods = await _feed_callback(dp, bot, data)
        for method in methods:
            if hasattr(method, "model_dump"):
                method.__class__.model_validate(method.model_dump())
        return methods

    async def feed_message(text: str, **kwargs):
        methods = await _feed_message(dp, bot, text, **kwargs)
        for method in methods:
            if hasattr(method, "model_dump"):
                method.__class__.model_validate(method.model_dump())
        return methods

    def find_button(methods, prefix: str):
        for method in methods:
            if not isinstance(method, (EditMessageText, SendMessage)) or method.reply_markup is None:
                continue
            for row in method.reply_markup.inline_keyboard:
                for button in row:
                    if button.callback_data and button.callback_data.startswith(prefix):
                        return button
        raise AssertionError(f"Button with prefix '{prefix}' not found in methods: {methods}")

    try:
        # Step 1: Open event list and navigate into the target event card
        methods = await feed_callback("admin_events")
        open_btn = find_button(methods, f"admin_ev_open_{scen.id}")
        methods = await feed_callback(open_btn.callback_data)

        # Step 2: In scenario card, press Edit to reach edit menu
        edit_btn = find_button(methods, f"admin_ev_edit_menu_{scen.id}")
        methods = await feed_callback(edit_btn.callback_data)

        # Step 3: Press real format selection button
        fmt_menu_btn = find_button(methods, f"admin_ev_format_menu_{scen.id}")
        methods = await feed_callback(fmt_menu_btn.callback_data)

        # Verify format screen is displayed with all options
        caption_btn = find_button(methods, f"admin_ev_set_fmt_{scen.id}_caption")
        back_edit_btn = find_button(methods, f"admin_ev_edit_menu_{scen.id}")
        assert caption_btn is not None
        assert back_edit_btn is not None

        # Step 4: Click "Подпись к медиа" without attachments
        methods = await feed_callback(caption_btn.callback_data)

        # Assert clean warning alert without exception
        answer = next((m for m in methods if isinstance(m, AnswerCallbackQuery)), None)
        assert answer is not None, f"Expected AnswerCallbackQuery in methods: {methods}"
        assert answer.show_alert is True
        assert len(answer.text) <= TELEGRAM_CALLBACK_ALERT_MAX_LENGTH
        assert answer.text == ALERT_MEDIA_CAPTION_REQUIRES_MEDIA
        assert answer.text == "Для этого формата нужно одно фото или видео. Добавьте медиа либо выберите раздельный формат"

        # Assert no message edit occurred on error: the user remains on the format menu screen
        edit_calls = [m for m in methods if isinstance(m, EditMessageText)]
        assert len(edit_calls) == 0, "Current screen must be preserved on error"

        # Verify DB is unchanged
        async with db_session_maker() as session:
            s_check = await session.get(EventScenario, scen.id)
            assert s_check.media_format == "separate_media_top"

        # Step 5: Press Back to return to edit menu
        methods = await feed_callback(back_edit_btn.callback_data)
        assert any("Редактирование мероприятия" in (m.text or "") for m in methods if isinstance(m, EditMessageText))

        # Step 6: Go to Media manager and add 1 photo
        media_btn = find_button(methods, f"admin_ev_edit_media_{scen.id}")
        methods = await feed_callback(media_btn.callback_data)

        add_media_btn = find_button(methods, f"admin_ev_media_add_{scen.id}")
        methods = await feed_callback(add_media_btn.callback_data)

        photo = [PhotoSize(file_id="tg_photo_caption_test", file_unique_id="uniq_1", width=300, height=300)]
        methods = await feed_message("", photo=photo)

        save_media_btn = find_button(methods, f"admin_ev_media_save_{scen.id}")
        methods = await feed_callback(save_media_btn.callback_data)

        # Verify photo saved in DB
        db_media = await list_scenario_media(db_session_maker, scen.id)
        assert len(db_media) == 1

        # Step 7: Now reopen edit menu and format menu again
        edit_btn = find_button(methods, f"admin_ev_edit_menu_{scen.id}")
        methods = await feed_callback(edit_btn.callback_data)

        fmt_menu_btn = find_button(methods, f"admin_ev_format_menu_{scen.id}")
        methods = await feed_callback(fmt_menu_btn.callback_data)

        caption_btn = find_button(methods, f"admin_ev_set_fmt_{scen.id}_caption")
        # With 1 media attached, choosing media_caption succeeds!
        methods = await feed_callback(caption_btn.callback_data)

        answer_ok = next((m for m in methods if isinstance(m, AnswerCallbackQuery)), None)
        assert answer_ok is not None
        assert answer_ok.show_alert is False or answer_ok.show_alert is None
        assert "Фото/видео с текстом (подпись)" in answer_ok.text

        edit_ok = next((m for m in methods if isinstance(m, EditMessageText)), None)
        assert edit_ok is not None
        assert "Формат доставки успешно обновлён" in edit_ok.text

        # Verify DB was atomically updated to caption format
        async with db_session_maker() as session:
            s_updated = await session.get(EventScenario, scen.id)
            assert s_updated.media_format == "caption"

        # Step 8: Return to scenario card and back to initial events list
        card_btn = find_button(methods, f"admin_ev_open_{scen.id}")
        methods = await feed_callback(card_btn.callback_data)
        assert any("Фото/видео с текстом (подпись)" in (m.text or "") for m in methods if isinstance(m, EditMessageText))

        list_btn = find_button(methods, "admin_ev_page_")
        methods = await feed_callback(list_btn.callback_data)
        assert any("Сценарии мероприятий" in (m.text or "") for m in methods if isinstance(m, EditMessageText))

    finally:
        for router in routers:
            router._parent_router = original_parents[router]


@pytest.mark.asyncio
async def test_telegram_event_callback_alert_long_error_safe(db_session_maker, monkeypatch):
    """Verify that any excessively long error message (300+ chars) passed to callback.answer
    is safely bounded to <= 200 chars and does not trigger MESSAGE_TOO_LONG."""
    import handlers
    from aiogram.methods import AnswerCallbackQuery
    from event_admin_service import create_scenario
    from event_admin_telegram import TELEGRAM_CALLBACK_ALERT_MAX_LENGTH

    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))

    scen = await create_scenario(
        db_session_maker,
        code="webinar_long_err",
        event_type="webinar",
        title="Тест длинных ошибок",
        starts_at_utc=datetime.utcnow() + timedelta(days=1),
        link="https://example.test/long-err",
        description="Описание",
        media_format="separate_media_top",
    )

    routers = (event_admin_router, handlers.router)
    original_parents = {r: r._parent_router for r in routers}
    dp = Dispatcher(storage=MemoryStorage())
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789
    for router in routers:
        router._parent_router = None
        dp.include_router(router)

    # Monkeypatch update_scenario_fields to throw a huge 400-char exception
    huge_msg = "Критическая ошибка хранилища базы данных: " + ("детали ошибки сценария " * 12)
    assert len(huge_msg) > 250

    async def mock_update_scenario_fields(*args, **kwargs):
        raise RuntimeError(huge_msg)

    monkeypatch.setattr(event_admin_telegram, "update_scenario_fields", mock_update_scenario_fields)

    try:
        methods = await _feed_callback(dp, bot, f"admin_ev_set_fmt_{scen.id}_separate_text_top")
        for m in methods:
            m.__class__.model_validate(m.model_dump())

        answer = next((m for m in methods if isinstance(m, AnswerCallbackQuery)), None)
        assert answer is not None
        assert answer.show_alert is True
        assert len(answer.text) <= TELEGRAM_CALLBACK_ALERT_MAX_LENGTH
        assert answer.text.endswith("…")
    finally:
        for router in routers:
            router._parent_router = original_parents[router]


@pytest.mark.asyncio
async def test_telegram_futuro_steps_html_preview_and_full_journey_a_to_ya_to_a(db_session_maker, monkeypatch):
    """Full journey A->Я->A for 'ВСЕЛЕННАЯ ФУТУРО' reminder chain:
    - HTML preview renderer preserves valid HTML tags (<code>, <b>, <a>) without raw escaped &lt;code&gt;
    - Variables ({event_title}, {link}, {name}) are preserved in preview and substituted for recipient
    - Long messages are truncated cleanly without cutting tags
    - Buttons are stripped of raw tags and validated through Pydantic/aiogram
    - Full roundtrip: card -> steps list -> open step -> edit text -> save -> reopen -> edit -> cancel -> reopen -> back to steps -> back to card
    - Raw message in DB is never altered by preview rendering
    """
    monkeypatch.setattr(event_admin_telegram, "async_session_maker", db_session_maker)
    monkeypatch.setattr(event_admin_telegram, "get_all_admin_ids", AsyncMock(return_value={12345}))
    from event_admin_service import add_step, create_scenario, get_step_item
    from event_service import render_event_template

    scen = await create_scenario(
        db_session_maker,
        code="futuro_universe_event",
        event_type="webinar",
        title="ВСЕЛЕННАЯ ФУТУРО",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://futuro.example/webinar",
        is_active=True,
    )
    step1_text = "Напоминаем: мероприятие <code>{event_title}</code> начнется через 24 ч! Ссылка: {link}"
    step1 = await add_step(db_session_maker, scen.id, -1440, step1_text)

    step2_text = (
        "Привет, {name}! <b>Важное оповещение:</b> подключайтесь к <code>{event_title}</code> "
        "по ссылке <a href=\"https://futuro.example/zoom\">трансляция</a>. "
        "Пожалуйста, проверьте микрофон и камеру заранее, чтобы не пропустить ключевую сессию в эфире!"
    )
    step2 = await add_step(db_session_maker, scen.id, -120, step2_text)

    event_admin_router._parent_router = None
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(event_admin_router)
    bot = AsyncMock(spec=Bot)
    bot.id = 123456789

    def _validate_methods(methods):
        for m in methods:
            if hasattr(m, "model_validate") and hasattr(m, "model_dump"):
                m.__class__.model_validate(m.model_dump())
        return methods

    # 1. Open event scenario card
    methods = _validate_methods(await _feed_callback(dp, bot, f"admin_ev_open_{scen.id}_0_0"))
    card = next(m for m in methods if isinstance(m, EditMessageText))
    assert "ВСЕЛЕННАЯ ФУТУРО" in card.text

    # 2. Click steps list button
    steps_callback = next(
        b.callback_data
        for row in card.reply_markup.inline_keyboard
        for b in row
        if b.callback_data and b.callback_data.startswith(f"admin_ev_steps_{scen.id}")
    )
    methods = _validate_methods(await _feed_callback(dp, bot, steps_callback))
    steps_screen = next(m for m in methods if isinstance(m, EditMessageText))

    # Verify HTML preview requirements:
    # - Raw <code> tag is NOT escaped to &lt;code&gt;
    assert "<code>{event_title}</code>" in steps_screen.text
    assert "&lt;code&gt;" not in steps_screen.text
    assert "<blockquote>" in steps_screen.text
    # - Formatted blockquote preview preserves tags and variables
    assert "Напоминаем: мероприятие <code>{event_title}</code> начнется через 24 ч! Ссылка: {link}" in steps_screen.text
    # - Step 2 is safely truncated without unclosed tags or &lt;code&gt;
    assert "&lt;code&gt;" not in steps_screen.text
    assert "&lt;b&gt;" not in steps_screen.text

    # Check button text on steps list: does not contain literal HTML tags
    step1_btn = steps_screen.reply_markup.inline_keyboard[0][0]
    assert "<" not in step1_btn.text
    assert ">" not in step1_btn.text
    assert "ВСЕЛЕННАЯ ФУТУРО" in step1_btn.text or "Напоминаем" in step1_btn.text

    # 3. Open Step 1 card
    methods = _validate_methods(await _feed_callback(dp, bot, f"admin_ev_step_{step1.id}"))
    step_screen = next(m for m in methods if isinstance(m, EditMessageText))
    assert "<code>{event_title}</code>" in step_screen.text
    assert "&lt;code&gt;" not in step_screen.text

    # 4. Click edit text
    edit_callback = next(
        b.callback_data
        for row in step_screen.reply_markup.inline_keyboard
        for b in row
        if b.callback_data and b.callback_data.startswith(f"admin_ev_step_edit_text_{step1.id}")
    )
    methods = _validate_methods(await _feed_callback(dp, bot, edit_callback))
    edit_screen = next(m for m in methods if isinstance(m, EditMessageText))
    assert "<code>{event_title}</code>" in edit_screen.text
    assert "&lt;code&gt;" not in edit_screen.text

    # 4a. HTML pre-commit validation tests:
    # 4a.1: Unclosed tag rejection
    methods_err1 = _validate_methods(await _feed_message(dp, bot, "<b>Текст без закрывающего тега {event_title}"))
    err_msg1 = next(m for m in methods_err1 if isinstance(m, SendMessage) and "❌ Ошибка" in m.text)
    assert "не закрыты теги" in err_msg1.text
    # Verify DB step intact
    db_step_mid = await get_step_item(db_session_maker, step1.id)
    assert db_step_mid.message_text == step1_text
    assert db_step_mid.step_version == 1
    # Verify FSM state preserved
    fsm_key = StorageKey(bot_id=bot.id, chat_id=12345, user_id=12345)
    assert await dp.fsm.storage.get_state(fsm_key) == event_admin_telegram.EventAdminStates.step_edit_text.state

    # 4a.2: Unsupported tag rejection
    methods_err2 = _validate_methods(await _feed_message(dp, bot, "<script>alert('xss')</script>"))
    err_msg2 = next(m for m in methods_err2 if isinstance(m, SendMessage) and "❌ Ошибка" in m.text)
    assert "неподдерживается" in err_msg2.text
    db_step_mid = await get_step_item(db_session_maker, step1.id)
    assert db_step_mid.message_text == step1_text
    assert db_step_mid.step_version == 1
    assert await dp.fsm.storage.get_state(fsm_key) == event_admin_telegram.EventAdminStates.step_edit_text.state

    # 5. Enter new valid text with HTML tags and variables
    updated_input = "Обновлено: <code>{event_title}</code> стартует! Ссылка: {link}"
    methods = _validate_methods(await _feed_message(dp, bot, updated_input))
    saved = next(m for m in methods if isinstance(m, SendMessage) and "Текст шага обновлен" in m.text)
    assert "<code>{event_title}</code>" in saved.text
    assert "&lt;code&gt;" not in saved.text

    # 6. Reopen step 1 card
    methods = _validate_methods(await _feed_callback(dp, bot, f"admin_ev_step_{step1.id}"))
    step_screen = next(m for m in methods if isinstance(m, EditMessageText))
    assert "Обновлено: <code>{event_title}</code>" in step_screen.text

    # 7. Start edit text and click cancel
    methods = _validate_methods(await _feed_callback(dp, bot, edit_callback))
    edit_screen = next(m for m in methods if isinstance(m, EditMessageText))
    cancel_btn = next(b for row in edit_screen.reply_markup.inline_keyboard for b in row if b.text == "⬅️ Отмена")
    methods = _validate_methods(await _feed_callback(dp, bot, cancel_btn.callback_data))
    reopened_step = next(m for m in methods if isinstance(m, EditMessageText))
    assert f"admin_ev_step_edit_text_{step1.id}" in _telegram_event_callbacks(reopened_step.reply_markup)

    # 8. Return back to steps list
    back_to_steps_cb = reopened_step.reply_markup.inline_keyboard[-1][0].callback_data
    methods = _validate_methods(await _feed_callback(dp, bot, back_to_steps_cb))
    back_steps_screen = next(m for m in methods if isinstance(m, EditMessageText))
    assert "Цепочка сообщений-напоминаний" in back_steps_screen.text

    # 9. Return back to scenario card
    back_to_card_cb = back_steps_screen.reply_markup.inline_keyboard[-1][0].callback_data
    methods = _validate_methods(await _feed_callback(dp, bot, back_to_card_cb))
    final_card = next(m for m in methods if isinstance(m, EditMessageText))
    assert "ВСЕЛЕННАЯ ФУТУРО" in final_card.text

    # 10. Recipient substitution verification
    recipient_user = User(id=777, first_name="Дмитрий", name="Дмитрий")
    recipient_text = render_event_template(updated_input, scen, user=recipient_user, platform="telegram")
    assert "<code>ВСЕЛЕННАЯ ФУТУРО</code>" in recipient_text
    assert "&lt;code&gt;" not in recipient_text
    assert "https://futuro.example/webinar" in recipient_text

    # 11. Database text integrity check
    db_step = await get_step_item(db_session_maker, step1.id)
    assert db_step.message_text == updated_input

    # 12. Test-send menu & buttons contract check
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=999))
    methods = _validate_methods(await _feed_callback(dp, bot, f"admin_ev_test_send_menu_{scen.id}"))
    test_send_screen = next(m for m in methods if isinstance(m, EditMessageText))
    assert "Тестовая отправка" in test_send_screen.text

    # Verify test-send keyboard buttons have tags stripped and valid callbacks
    test_send_buttons = [b for row in test_send_screen.reply_markup.inline_keyboard for b in row]
    for b in test_send_buttons:
        assert "<" not in b.text
        assert ">" not in b.text
    step1_test_btn = next(b for b in test_send_buttons if b.callback_data == f"admin_ev_do_test_send_{scen.id}_{step1.id}")
    assert "ВСЕЛЕННАЯ ФУТУРО" in step1_test_btn.text or "Обновлено" in step1_test_btn.text

    # Trigger test send through dispatcher and validate outgoing Telegram method via Pydantic
    methods = _validate_methods(await _feed_callback(dp, bot, f"admin_ev_do_test_send_{scen.id}_{step1.id}"))
    assert bot.send_message.called
    test_call_kwargs = bot.send_message.call_args.kwargs
    validated_test_send = SendMessage.model_validate({
        "chat_id": test_call_kwargs["chat_id"],
        "text": test_call_kwargs["text"],
        "parse_mode": test_call_kwargs.get("parse_mode", "HTML"),
        "reply_markup": test_call_kwargs.get("reply_markup"),
    })
    assert "🧪 <b>[ТЕСТОВОЕ СООБЩЕНИЕ]</b>" in validated_test_send.text
    assert "Обновлено: <code>ВСЕЛЕННАЯ ФУТУРО</code> стартует! Ссылка: https://futuro.example/webinar" in validated_test_send.text
    assert "&lt;code&gt;" not in validated_test_send.text

    # 13. Real event delivery worker execution with Pydantic validation of outgoing message
    from event_service import register_user_for_event
    from event_delivery_worker import run_event_delivery_worker_cycle

    recipient_user_id = 777001
    async with db_session_maker() as session:
        session.add(User(id=recipient_user_id, first_name="Дмитрий", name="Дмитрий", username="dmitry_futuro"))
        await session.commit()

    reg_now = scen.starts_at - timedelta(days=3)
    reg_res = await register_user_for_event(
        db_session_maker,
        recipient_user_id,
        scen.code,
        platform="telegram",
        now=reg_now,
    )
    assert reg_res.status == "registered"

    delivery_bot = AsyncMock(spec=Bot)
    delivery_bot.id = 123456789
    mock_deliv_msg = MagicMock()
    mock_deliv_msg.message_id = 880055
    delivery_bot.send_message = AsyncMock(return_value=mock_deliv_msg)

    # Worker executes at step 1 due time
    step1_due = scen.starts_at - timedelta(minutes=1440) + timedelta(seconds=1)
    worker_stats = await run_event_delivery_worker_cycle(
        db_session_maker,
        bot=delivery_bot,
        now=step1_due,
    )
    assert worker_stats["sent"] >= 1
    assert worker_stats["failed"] == 0

    # Intercept outgoing Telegram method and perform Pydantic validation
    step_call = next(
        call for call in delivery_bot.send_message.call_args_list
        if "Обновлено:" in call.kwargs.get("text", "")
    )
    delivery_kwargs = step_call.kwargs
    validated_worker_send = SendMessage.model_validate({
        "chat_id": delivery_kwargs["chat_id"],
        "text": delivery_kwargs["text"],
        "parse_mode": delivery_kwargs.get("parse_mode", "HTML"),
        "reply_markup": delivery_kwargs.get("reply_markup"),
    })
    assert validated_worker_send.chat_id == recipient_user_id
    assert "<code>ВСЕЛЕННАЯ ФУТУРО</code>" in validated_worker_send.text
    assert "&lt;code&gt;" not in validated_worker_send.text
    assert "https://futuro.example/webinar" in validated_worker_send.text

    # Delivery row in database marked as sent with external_message_id
    async with db_session_maker() as session:
        deliv = (
            await session.execute(
                select(EventDelivery).where(
                    EventDelivery.user_id == recipient_user_id,
                    EventDelivery.step_id == step1.id,
                )
            )
        ).scalar_one()
        assert deliv.status == "sent"
        assert deliv.external_message_id == "880055"

    # 14. Return back to Admin scenario card and list, verify FSM state is clear
    methods = _validate_methods(await _feed_callback(dp, bot, f"admin_ev_open_{scen.id}_0_0"))
    card_reopened = next(m for m in methods if isinstance(m, EditMessageText))
    assert "ВСЕЛЕННАЯ ФУТУРО" in card_reopened.text

    methods = _validate_methods(await _feed_callback(dp, bot, "admin_ev_page_0_0"))
    list_reopened = next(m for m in methods if isinstance(m, EditMessageText))
    assert "Сценарии мероприятий" in list_reopened.text

    fsm_key = StorageKey(bot_id=bot.id, chat_id=12345, user_id=12345)
    fsm_state = await dp.fsm.storage.get_state(fsm_key)
    assert fsm_state is None





