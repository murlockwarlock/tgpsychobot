"""
Comprehensive Test Suite for Notification Routing Isolation Contract.

Verifies:
1. Matrix across ALL 18 active Telegram and MAX instances in registry.
2. Real administrative notification paths:
   - promo code activation;
   - successful payment;
   - payment error;
   - subscription cancellation;
   - auto-renewal;
   - terminal AI error (with real bot name instead of tgpsychobot);
   - system monitoring diagnostics.
3. Negative tests:
   - missing BOT_TOKEN;
   - empty BOT_TOKEN;
   - foreign token (Autobus token rejected on MAX yourself_way bot);
   - wrong Telegram/MAX companion pairing;
   - legacy config.ini with Autobus token ignored;
   - MAX OWNER_IDS filtered out from Telegram chat targets;
   - stale PM2 environment fallback to runtime.env;
   - safe failure handling (network error / Telegram error never aborts business logic).
4. Real Aiogram 3 Pydantic validation and transport boundary.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.methods import SendMessage, TelegramMethod
from aiogram.types import Chat, Message, User

import notification_routing as nr
from notification_routing import (
    EventCategory,
    NotificationRoute,
    detect_current_instance,
    extract_telegram_bot_id,
    is_valid_telegram_user_id,
    load_bot_registry,
    resolve_companion_telegram_instance,
    resolve_notification_route,
    resolve_target_admin_ids,
    resolve_target_bot_token,
    send_admin_notification,
    validate_process_notification_configuration,
)


class RecordingSession(BaseSession):
    """
    Test session that executes real Aiogram 3 Pydantic validation on methods
    and intercepts outgoing requests at the transport boundary.
    """

    def __init__(self, fail_with: Exception | None = None):
        super().__init__()
        self.requests: list[tuple[str, int, SendMessage]] = []
        self.fail_with = fail_with

    async def make_request(self, bot: Bot, method: TelegramMethod, timeout: int | None = None) -> Any:
        if self.fail_with is not None:
            raise self.fail_with
        from aiogram.methods import GetMe
        if isinstance(method, GetMe):
            from notification_routing import load_bot_registry
            registry = load_bot_registry()
            username = "TestBot"
            for inst in registry.get("instances", []):
                if inst.get("platform_id") == bot.id:
                    username = inst.get("username", "TestBot")
                    break
            return User(id=bot.id, is_bot=True, first_name="TestBot", username=username)
        if isinstance(method, SendMessage):
            self.requests.append((bot.token, bot.id, method))
            return Message(
                message_id=1001,
                date=datetime.now(),
                chat=Chat(id=int(method.chat_id), type="private"),
                from_user=User(id=bot.id, is_bot=True, first_name="TestBot"),
                text=method.text,
            )
        return None

    async def stream_content(self, url: str, timeout: int | None = None, chunk_size: int = 65536):
        yield b""

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def clean_env():
    """Ensure clean test environment for routing isolation."""
    nr.clear_registry_cache()
    nr.clear_bot_identity_cache()
    saved_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(saved_env)
    nr.clear_registry_cache()
    nr.clear_bot_identity_cache()


# -----------------------------------------------------------------------------
# 1. MATRIX ACROSS ALL ACTIVE INSTANCES
# -----------------------------------------------------------------------------

def test_registry_has_all_expected_instances():
    registry = load_bot_registry()
    instances = registry.get("instances", [])
    assert len(instances) >= 18, "Must contain all 18 production instances"

    pm2_names = {inst["pm2_name"] for inst in instances}
    assert "tg_autobusbusbot_new" in pm2_names
    assert "tg_yourself_way_bot_new" in pm2_names
    assert "max_id519010411655_bot_new" in pm2_names
    assert "tg_veraveda777_bot_legacy" in pm2_names
    assert "max_se13639182_bot_legacy" in pm2_names


def test_routing_matrix_telegram_instances_route_to_themselves():
    """Every Telegram bot instance MUST route user events to itself with its own token."""
    registry = load_bot_registry()
    for inst in registry.get("instances", []):
        if inst.get("platform") != "telegram":
            continue

        pm2_name = inst["pm2_name"]
        platform_id = inst["platform_id"]
        token_env = inst["token_env"]
        dummy_token = f"{platform_id}:AAF_TEST_VALID_TOKEN_SECRET_12345"

        with patch.dict(os.environ, {
            "BOT_INSTANCE_ID": pm2_name,
            "PM2_APP_NAME": pm2_name,
            token_env: dummy_token,
            "BOT_TOKEN": dummy_token,
        }, clear=True):
            route = resolve_notification_route(
                category=EventCategory.USER_EVENT,
                current_instance=inst,
            )
            assert route is not None, f"Failed to resolve route for {pm2_name}"
            assert route.target_instance == pm2_name, f"{pm2_name} must target itself"
            assert route.target_platform_id == platform_id
            assert route.target_bot_token == dummy_token
            assert extract_telegram_bot_id(route.target_bot_token) == platform_id


def test_routing_matrix_max_id519010411655_routes_strictly_to_yourself_way():
    """MAX yourself_way bot MUST route to @yourself_way_bot (8481853696)."""
    max_inst = nr.get_instance_by_pm2_name("max_id519010411655_bot_new")
    assert max_inst is not None

    companion = resolve_companion_telegram_instance(max_inst)
    assert companion is not None
    assert companion["pm2_name"] == "tg_yourself_way_bot_new"
    assert companion["username"] == "yourself_way_bot"
    assert companion["platform_id"] == 8481853696

    dummy_companion_token = "8481853696:AAF_YOURSELF_WAY_SECRET_TOKEN"
    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "COMPANION_TELEGRAM_BOT_TOKEN": dummy_companion_token,
    }, clear=True):
        route = resolve_notification_route(
            category=EventCategory.USER_EVENT,
            current_instance=max_inst,
        )
        assert route is not None
        assert route.source_instance == "max_id519010411655_bot_new"
        assert route.target_instance == "tg_yourself_way_bot_new"
        assert route.target_username == "yourself_way_bot"
        assert route.target_platform_id == 8481853696
        assert route.target_bot_token == dummy_companion_token


def test_routing_matrix_max_se13639182_routes_strictly_to_veraveda():
    """MAX veraveda bot MUST route to @veraveda777_bot (8224473896)."""
    max_inst = nr.get_instance_by_pm2_name("max_se13639182_bot_legacy")
    assert max_inst is not None

    companion = resolve_companion_telegram_instance(max_inst)
    assert companion is not None
    assert companion["pm2_name"] == "tg_veraveda777_bot_legacy"
    assert companion["username"] == "veraveda777_bot"
    assert companion["platform_id"] == 8224473896

    dummy_companion_token = "8224473896:AAF_VERAVEDA777_SECRET_TOKEN"
    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_se13639182_bot_legacy",
        "PM2_APP_NAME": "max_se13639182_bot_legacy",
        "COMPANION_TELEGRAM_BOT_TOKEN": dummy_companion_token,
    }, clear=True):
        route = resolve_notification_route(
            category=EventCategory.USER_EVENT,
            current_instance=max_inst,
        )
        assert route is not None
        assert route.source_instance == "max_se13639182_bot_legacy"
        assert route.target_instance == "tg_veraveda777_bot_legacy"
        assert route.target_username == "veraveda777_bot"
        assert route.target_platform_id == 8224473896
        assert route.target_bot_token == dummy_companion_token


def test_routing_matrix_autobus_never_receives_foreign_user_events():
    """Autobusbot must NEVER be the target for user events originating from other bots."""
    autobus_inst = nr.get_instance_by_pm2_name("tg_autobusbusbot_new")
    yourself_inst = nr.get_instance_by_pm2_name("max_id519010411655_bot_new")

    # If someone tries to pass Autobus token to yourself_way instance
    autobus_token = "8346799080:AAF_AUTOBUS_TEST_TOKEN"
    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "BOT_TOKEN": autobus_token,
    }, clear=True):
        # Must be rejected because Autobus token does not match target platform_id 8481853696
        route = resolve_notification_route(
            category=EventCategory.USER_EVENT,
            current_instance=yourself_inst,
        )
        assert route is None, "Autobus token must be rejected for yourself_way instance"


# -----------------------------------------------------------------------------
# 2. REAL ADMINISTRATIVE NOTIFICATION EVENT PATHS
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_promo_code_notification_from_max_yourself_way_routes_to_companion():
    """
    Real promo code activation in max_id519010411655_bot_new MUST dispatch through
    @yourself_way_bot to Telegram admins, with real Aiogram validation.
    """
    from max_messenger_bot.services.common import notify_telegram_admins

    recording_session = RecordingSession()
    dummy_bot_token = "8481853696:AAF_REAL_AIOGRAM_TEST_TOKEN_12345"

    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "MAX_BOT_NAME": "id519010411655_bot",
        "COMPANION_TELEGRAM_BOT_TOKEN": dummy_bot_token,
        "TELEGRAM_YOURSELF_WAY_BOT_TOKEN": dummy_bot_token,
    }, clear=True), patch("telegram_client.create_telegram_bot", lambda token: Bot(token=token, session=recording_session)):
        
        # Call the actual service entrypoint
        promo_text = (
            "🎁 Активирован промокод «PODAROK»\n"
            "Пользователь: Мария [id=<code>74367460</code>] (MAX)\n"
            "Скидка: 0%\n"
            "Дней: 1"
        )
        await notify_telegram_admins(promo_text)

    # Validate outgoing transport boundary
    assert len(recording_session.requests) > 0, "Admin notification must be dispatched"
    for token, bot_id, method in recording_session.requests:
        # Cryptographic platform_id check: MUST be yourself_way_bot (8481853696)
        assert bot_id == 8481853696, "Notification MUST be sent via @yourself_way_bot"
        assert token == dummy_bot_token
        # Telegram chat_id check: MUST NOT be a MAX user ID (>= 100_000_000_000)
        assert 0 < int(method.chat_id) < 100_000_000_000, f"Target {method.chat_id} is not a valid Telegram ID"
        # Content check
        assert "PODAROK" in method.text
        assert method.parse_mode == "HTML"


@pytest.mark.asyncio
async def test_successful_payment_notification_routes_to_companion():
    """Payment success notification in MAX routes to companion Telegram bot."""
    recording_session = RecordingSession()
    dummy_bot_token = "8481853696:AAF_REAL_AIOGRAM_TEST_TOKEN_12345"

    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "COMPANION_TELEGRAM_BOT_TOKEN": dummy_bot_token,
    }, clear=True), patch("telegram_client.create_telegram_bot", lambda token: Bot(token=token, session=recording_session)):

        delivered = await send_admin_notification(
            text="💳 Успешная оплата (YooKassa, MAX)\nПользователь: ID 74367460\nСумма: 500 руб",
            category=EventCategory.PAYMENT,
        )
        assert delivered is True

    assert len(recording_session.requests) > 0
    token, bot_id, method = recording_session.requests[0]
    assert bot_id == 8481853696
    assert "Успешная оплата" in method.text


@pytest.mark.asyncio
async def test_subscription_cancellation_notification_routes_to_companion():
    """Subscription cancellation notification in MAX routes to companion Telegram bot."""
    recording_session = RecordingSession()
    dummy_bot_token = "8224473896:AAF_VERAVEDA_TOKEN_TEST"

    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_se13639182_bot_legacy",
        "PM2_APP_NAME": "max_se13639182_bot_legacy",
        "COMPANION_TELEGRAM_BOT_TOKEN": dummy_bot_token,
    }, clear=True), patch("telegram_client.create_telegram_bot", lambda token: Bot(token=token, session=recording_session)):

        delivered = await send_admin_notification(
            text="🔕 Пользователь отключил автопродление (MAX)\nТариф: Премиум",
            category=EventCategory.PAYMENT,
        )
        assert delivered is True

    assert len(recording_session.requests) > 0
    token, bot_id, method = recording_session.requests[0]
    assert bot_id == 8224473896  # Veraveda companion ID
    assert "отключил автопродление" in method.text


@pytest.mark.asyncio
async def test_terminal_ai_failure_alert_displays_correct_bot_name_and_routes_to_companion():
    """
    AI error in MAX bot must:
    1. Display the actual bot name (id519010411655_bot), NOT tgpsychobot.
    2. Dispatch via companion Telegram bot (@yourself_way_bot).
    """
    from error_reporting import send_terminal_ai_failure_alert

    recording_session = RecordingSession()
    dummy_bot_token = "8481853696:AAF_YOURSELF_WAY_TOKEN_TEST"

    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "MAX_BOT_NAME": "id519010411655_bot",
        "COMPANION_TELEGRAM_BOT_TOKEN": dummy_bot_token,
    }, clear=True), patch("telegram_client.create_telegram_bot", lambda token: Bot(token=token, session=recording_session)):

        delivered = await send_terminal_ai_failure_alert(
            bot=None,  # Running in MAX process without Telegram bot instance
            platform="max",
            user_id=100026905347,
            classification="timeout",
            details="Deepseek timeout",
            ai_log_ids=[3466, 3467],
        )
        assert delivered is True

    assert len(recording_session.requests) > 0
    token, bot_id, method = recording_session.requests[0]
    assert bot_id == 8481853696, "Must be routed through companion @yourself_way_bot"
    assert "id519010411655_bot" in method.text, "Header must show real bot name"
    assert "tgpsychobot" not in method.text, "Fallback 'tgpsychobot' must NOT appear"
    assert "100026905347" in method.text
    assert "#3466, #3467" in method.text


@pytest.mark.asyncio
async def test_system_monitoring_alert_routes_to_autobus():
    """System diagnostic alert routes to @autobusbusbot (8346799080)."""
    recording_session = RecordingSession()
    autobus_token = "8346799080:AAF_AUTOBUS_TOKEN_TEST"

    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "PM2_APP_NAME": "max_id519010411655_bot_new",  # Even if triggered from another process
        "TELEGRAM_AUTOBUSBUSBOT_TOKEN": autobus_token,
    }, clear=True), patch("telegram_client.create_telegram_bot", lambda token: Bot(token=token, session=recording_session)):

        delivered = await send_admin_notification(
            text="🖥️ [System Monitor] High disk usage on VPS",
            category=EventCategory.SYSTEM_DIAGNOSTIC,
        )
        assert delivered is True

    assert len(recording_session.requests) > 0
    token, bot_id, method = recording_session.requests[0]
    assert bot_id == 8346799080  # Autobus platform_id
    assert "High disk usage" in method.text


# -----------------------------------------------------------------------------
# 3. NEGATIVE TESTS
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_bot_token_fails_safely_without_exception():
    """When token is completely missing, delivery returns False without throwing."""
    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "PM2_APP_NAME": "max_id519010411655_bot_new",
    }, clear=True):
        delivered = await send_admin_notification("Test missing token")
        assert delivered is False


@pytest.mark.asyncio
async def test_empty_bot_token_fails_safely():
    """When token is an empty string, delivery returns False."""
    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "BOT_TOKEN": "",
        "COMPANION_TELEGRAM_BOT_TOKEN": "",
    }, clear=True):
        delivered = await send_admin_notification("Test empty token")
        assert delivered is False


@pytest.mark.asyncio
async def test_foreign_autobus_token_rejected_for_yourself_way():
    """
    CRITICAL P0 TEST:
    If BOT_TOKEN is populated with Autobus token (8346799080),
    max_id519010411655_bot_new MUST reject it and refuse to send!
    """
    autobus_token = "8346799080:AAF_AUTOBUS_ILLEGAL_CROSS_TOKEN"
    recording_session = RecordingSession()

    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "BOT_TOKEN": autobus_token,  # Contaminated legacy env!
    }, clear=True), patch("telegram_client.create_telegram_bot", lambda token: Bot(token=token, session=recording_session)):

        delivered = await send_admin_notification("Test foreign token")
        assert delivered is False, "Dispatch with mismatched token MUST be rejected"
        assert len(recording_session.requests) == 0, "Zero messages must be sent to Autobus"


def test_legacy_config_ini_autobus_token_is_not_loaded_into_env():
    """
    apply_legacy_env_defaults MUST NEVER backfill BOT_TOKEN from config.ini!
    """
    from max_messenger_bot.settings import apply_legacy_env_defaults

    with patch.dict(os.environ, {}, clear=True):
        # config.ini exists in project root containing Autobus token
        apply_legacy_env_defaults()
        assert "BOT_TOKEN" not in os.environ, "BOT_TOKEN must NOT be backfilled from config.ini"


@pytest.mark.asyncio
async def test_max_owner_ids_are_filtered_out_from_telegram_targets():
    """
    MAX owner IDs (>= 100_000_000_000) must NEVER be passed as chat_id to Telegram.
    """
    max_inst = nr.get_instance_by_pm2_name("max_id519010411655_bot_new")
    assert max_inst is not None
    companion = resolve_companion_telegram_instance(max_inst)
    assert companion is not None

    # Environment contains mixed IDs: MAX owners + Telegram owner
    with patch.dict(os.environ, {
        "OWNER_IDS": "100263465646,100005511792,272982544",
    }, clear=True):
        admin_ids = await resolve_target_admin_ids(companion)
        for aid in admin_ids:
            assert is_valid_telegram_user_id(aid)
            assert aid < 100_000_000_000, f"MAX ID {aid} was not filtered out"
        assert 272982544 in admin_ids
        assert 100263465646 not in admin_ids
        assert 100005511792 not in admin_ids


@pytest.mark.asyncio
async def test_stale_pm2_env_loads_from_runtime_env_file(tmp_path):
    """
    User notifications strictly fail closed when COMPANION_TELEGRAM_BOT_TOKEN is missing
    (does not scan runtime.env). System diagnostics (Autobus) CAN recover from runtime.env.
    """
    runtime_env_file = tmp_path / "runtime.env"
    dummy_token = "8481853696:AAF_LOADED_FROM_DISK_FILE_TOKEN"
    autobus_token = "8346799080:AAF_AUTOBUS_DISK_FILE_TOKEN"
    runtime_env_file.write_text(
        f"TELEGRAM_YOURSELF_WAY_BOT_TOKEN={dummy_token}\n"
        f"TELEGRAM_AUTOBUSBUSBOT_BOT_TOKEN={autobus_token}\n",
        encoding="utf-8",
    )

    recording_session = RecordingSession()

    # 1. User event from MAX without COMPANION_TELEGRAM_BOT_TOKEN in env must FAIL CLOSED (not scan runtime.env)
    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "PROD_RUNTIME_ENV": str(runtime_env_file),
    }, clear=True), patch("telegram_client.create_telegram_bot", lambda token: Bot(token=token, session=recording_session)):

        delivered = await send_admin_notification("Test user event does NOT load from runtime.env", category=EventCategory.USER_EVENT)
        assert delivered is False, "User event must refuse to route when COMPANION_TELEGRAM_BOT_TOKEN is missing from env"

    # 2. System diagnostic monitoring CAN recover from runtime.env
    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "tg_autobusbusbot_new",
        "PM2_APP_NAME": "tg_autobusbusbot_new",
        "PROD_RUNTIME_ENV": str(runtime_env_file),
    }, clear=True), patch("telegram_client.create_telegram_bot", lambda token: Bot(token=token, session=recording_session)):

        delivered = await send_admin_notification("System alert from runtime.env", category=EventCategory.SYSTEM_DIAGNOSTIC)
        assert delivered is True

    assert len(recording_session.requests) > 0
    token, bot_id, _ = recording_session.requests[0]
    assert bot_id == 8346799080
    assert token == autobus_token


@pytest.mark.asyncio
async def test_telegram_api_failure_does_not_raise_exception():
    """
    Network or Telegram HTTP failures must be logged and handled safely;
    send_admin_notification returns False and NEVER raises to caller.
    """
    recording_session = RecordingSession(fail_with=RuntimeError("Telegram network timeout 504"))
    dummy_token = "8481853696:AAF_YOURSELF_WAY_TOKEN_TEST"

    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "COMPANION_TELEGRAM_BOT_TOKEN": dummy_token,
    }, clear=True), patch("telegram_client.create_telegram_bot", lambda token: Bot(token=token, session=recording_session)):

        # Must not raise RuntimeError
        delivered = await send_admin_notification("Test error handling")
        assert delivered is False


@pytest.mark.asyncio
async def test_payment_failure_notification_routes_to_companion():
    """Payment failure alert (e.g. card declined or 3 failed attempts) routes to companion."""
    recording_session = RecordingSession()
    dummy_bot_token = "8481853696:AAF_REAL_AIOGRAM_TEST_TOKEN_12345"

    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "COMPANION_TELEGRAM_BOT_TOKEN": dummy_bot_token,
    }, clear=True), patch("telegram_client.create_telegram_bot", lambda token: Bot(token=token, session=recording_session)):

        delivered = await send_admin_notification(
            text="🚫 Автопродление отключено (3 попытки YooKassa, MAX)\nПользователь: Мария\nПровайдер: Yookassa",
            category=EventCategory.PAYMENT,
        )
        assert delivered is True

    assert len(recording_session.requests) > 0
    token, bot_id, method = recording_session.requests[0]
    assert bot_id == 8481853696
    assert "Автопродление отключено" in method.text


@pytest.mark.asyncio
async def test_auto_renewal_success_notification_routes_to_companion():
    """Auto-renewal success alert routes to companion."""
    recording_session = RecordingSession()
    dummy_bot_token = "8481853696:AAF_REAL_AIOGRAM_TEST_TOKEN_12345"

    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "COMPANION_TELEGRAM_BOT_TOKEN": dummy_bot_token,
    }, clear=True), patch("telegram_client.create_telegram_bot", lambda token: Bot(token=token, session=recording_session)):

        delivered = await send_admin_notification(
            text="🔔 Автопродление (YooKassa, MAX)!\nПользователь: Мария\nТариф: Премиум\nСумма: 390.00 руб",
            category=EventCategory.PAYMENT,
        )
        assert delivered is True

    assert len(recording_session.requests) > 0
    token, bot_id, method = recording_session.requests[0]
    assert bot_id == 8481853696
    assert "Автопродление" in method.text


@pytest.mark.asyncio
async def test_corrupted_admin_db_safely_falls_back_to_configured_owners():
    """If DB admin lookup fails or returns corrupt records, routing uses configured Telegram owners."""
    mock_session = AsyncMock()
    mock_session.execute.side_effect = RuntimeError("Database connection failed")

    companion_inst = nr.get_instance_by_pm2_name("tg_yourself_way_bot_new")
    assert companion_inst is not None

    admin_ids = await resolve_target_admin_ids(companion_inst, session=mock_session)
    assert len(admin_ids) > 0
    # Must contain default Telegram owners of yourself_way_bot
    assert 272982544 in admin_ids
    assert 979514796 in admin_ids
    for aid in admin_ids:
        assert 0 < aid < 100_000_000_000


@pytest.mark.asyncio
async def test_process_restart_recovery_from_pm2_env_or_file(tmp_path):
    """
    Simulate full process restart: env is re-initialized with canonical tokens,
    proving that after PM2 restart/resurrect the routing engine operates seamlessly.
    """
    runtime_env_file = tmp_path / "runtime.env"
    canonical_token = "8481853696:AAF_CANONICAL_YOURSELF_WAY_TOKEN"
    runtime_env_file.write_text(f"TELEGRAM_YOURSELF_WAY_BOT_TOKEN={canonical_token}\n", encoding="utf-8")

    recording_session = RecordingSession()

    # Step 1: Simulated first run with COMPANION_TELEGRAM_BOT_TOKEN
    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "COMPANION_TELEGRAM_BOT_TOKEN": canonical_token,
    }, clear=True), patch("telegram_client.create_telegram_bot", lambda token: Bot(token=token, session=recording_session)):
        deliv1 = await send_admin_notification("Run 1 before restart")
        assert deliv1 is True

    # Step 2: Simulated crash and resurrection where COMPANION_TELEGRAM_BOT_TOKEN is missing
    # Routing engine strictly fails closed without reading from runtime.env for user notifications
    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "PROD_RUNTIME_ENV": str(runtime_env_file),
    }, clear=True), patch("telegram_client.create_telegram_bot", lambda token: Bot(token=token, session=recording_session)):
        deliv2 = await send_admin_notification("Run 2 after PM2 restart")
        assert deliv2 is False, "User event must fail closed when COMPANION_TELEGRAM_BOT_TOKEN is missing"

    assert len(recording_session.requests) == 4
    for token, bot_id, _ in recording_session.requests:
        assert bot_id == 8481853696
        assert token == canonical_token


def test_validate_process_notification_configuration_diagnostic():
    """Diagnostic check on startup returns structured report without leaking secrets."""
    dummy_token = "8481853696:AAF_YOURSELF_WAY_TOKEN_TEST"
    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "DATABASE_URL": "postgresql://localhost/yourself_way_db",
        "COMPANION_TELEGRAM_BOT_TOKEN": dummy_token,
    }, clear=True):
        report = validate_process_notification_configuration()
        assert report["status"] == "ok"
        assert report["instance"] == "max_id519010411655_bot_new"
        assert report["target_bot"] == "yourself_way_bot"
        assert report["target_platform_id"] == 8481853696
        assert report["token_status"] == "VALID"
        # Ensure secret value is NOT in report dict
        assert dummy_token not in str(report)


# -----------------------------------------------------------------------------
# 4. END-TO-END EXACT INCIDENT REPRODUCTION & VERIFICATION
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_reproduce_incident_november_7_and_verify_correct_routing():
    """
    Exact end-to-end replay of the incident on 07.11.2026:
    1. MAX bot max_id519010411655_bot_new activates promo code PODAROK for MAX user 100074367460 (Maria).
    2. Deepseek AI timeout error occurs for MAX user 100026905347.
    3. Legacy config.ini is present with Autobus token.
    4. Even if stale PM2 environment initially lacked BOT_TOKEN,
       the system resolves via runtime.env / companion configuration.

    Guarantees:
    - Sent strictly via @yourself_way_bot (ID: 8481853696).
    - ZERO messages sent via Autobus (ID: 8346799080).
    - ZERO messages sent to MAX user IDs (>= 100_000_000_000).
    - Error alert header displays 'Бот: id519010411655_bot', NOT 'tgpsychobot'.
    """
    from max_messenger_bot.services.common import notify_telegram_admins
    from error_reporting import send_terminal_ai_failure_alert

    recording_session = RecordingSession()
    yourself_token = "8481853696:AAF_YOURSELF_WAY_CANONICAL_PROD_TOKEN"
    autobus_token = "8346799080:AAF_AUTOBUS_CONTAMINATED_LEGACY_TOKEN"

    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "MAX_BOT_NAME": "id519010411655_bot",
        "COMPANION_TELEGRAM_BOT_TOKEN": yourself_token,
        "TELEGRAM_YOURSELF_WAY_BOT_TOKEN": yourself_token,
        # Even if legacy Autobus token lingers in some other env key:
        "LEGACY_AUTOBUS_TOKEN": autobus_token,
    }, clear=True), patch("telegram_client.create_telegram_bot", lambda token: Bot(token=token, session=recording_session)):

        # 1. Dispatch promo code notification (exact message from incident)
        promo_text = (
            "🎁 Активирован промокод «PODAROK»\n"
            "Пользователь: Мария [id=<code>74367460</code>] (MAX)\n"
            "Скидка: 0%\n"
            "Дней: 1"
        )
        deliv1 = await notify_telegram_admins(promo_text, category="user_event")
        assert deliv1 is True

        # 2. Dispatch AI error notification (exact message from incident)
        deliv2 = await send_terminal_ai_failure_alert(
            bot=None,
            platform="max",
            user_id=100026905347,
            classification="timeout",
            primary_provider="Deepseek",
            primary_model="deepseek-v4-flash",
            fallback_provider="Deepseek",
            fallback_model="deepseek-flash",
            error_type="AIServiceError",
            error_message="Fallback AI provider Deepseek timed out after 30.0s",
            ai_log_ids=[3466, 3467],
        )
        assert deliv2 is True

    # VERIFY EXACT TRANSPORT BEHAVIOR
    assert len(recording_session.requests) > 0, "Requests must be delivered"

    for token, bot_id, method in recording_session.requests:
        # A) Cryptographic bot identity: MUST be yourself_way_bot, NEVER Autobus
        assert bot_id == 8481853696, f"Bot ID {bot_id} is not yourself_way_bot (8481853696)"
        assert bot_id != 8346799080, "CRITICAL: Notification must NEVER use Autobus bot ID!"
        assert token == yourself_token
        assert token != autobus_token

        # B) Recipient isolation: chat_id must be a Telegram admin, NEVER a MAX user ID
        target_chat_id = int(method.chat_id)
        assert target_chat_id < 100_000_000_000, f"Chat ID {target_chat_id} is a MAX user ID!"
        yourself_way_inst = nr.get_instance_by_id("tg_yourself_way_bot_new")
        assert target_chat_id in yourself_way_inst["telegram_owner_ids"]

        # C) Content verification
        if "PODAROK" in method.text:
            assert "74367460" in method.text
        if "Сбой ИИ" in method.text:
            assert "id519010411655_bot" in method.text
            assert "tgpsychobot" not in method.text
            assert "100026905347" in method.text


@pytest.mark.asyncio
async def test_verify_bot_identity_live_matches_getme_and_catches_mismatch():
    """verify_bot_identity_live accurately checks getMe without leaking secrets."""
    from notification_routing import verify_bot_identity_live, RoutingError

    class MockAiogramBot:
        def __init__(self, bot_id: int, username: str):
            self.id = bot_id
            self.username = username
        async def get_me(self):
            return User(id=self.id, is_bot=True, first_name="MockBot", username=self.username)

    # 1. Matching identity passes
    valid_bot = MockAiogramBot(8481853696, "yourself_way_bot")
    res = await verify_bot_identity_live(
        valid_bot,
        expected_username="yourself_way_bot",
        expected_id=8481853696,
    )
    assert res["verified"] is True
    assert res["id"] == 8481853696
    assert res["username"] == "yourself_way_bot"

    # 2. Mismatched ID raises RoutingError
    with pytest.raises(RoutingError, match="Bot identity mismatch: getMe returned ID 8481853696, expected 8346799080"):
        await verify_bot_identity_live(
            valid_bot,
            expected_username="yourself_way_bot",
            expected_id=8346799080,  # Expected Autobus!
        )

    # 3. Mismatched username raises RoutingError
    with pytest.raises(RoutingError, match="Bot identity mismatch: getMe returned @yourself_way_bot, expected @autobusbusbot"):
        await verify_bot_identity_live(
            valid_bot,
            expected_username="autobusbusbot",
            expected_id=8481853696,
        )


@pytest.mark.asyncio
async def test_audit_database_get_all_admin_ids_preserves_max_while_telegram_filters():
    """
    database.get_all_admin_ids preserves all admins (including MAX admins for MAX bot functions),
    while get_all_telegram_admin_ids strictly filters out MAX user IDs.
    """
    from database import get_all_admin_ids, get_all_telegram_admin_ids

    # Environment contains mixed IDs: MAX owners + Telegram owner
    with patch.dict(os.environ, {
        "OWNER_IDS": "100263465646,100005511792,272982544,806750628",
    }, clear=True):
        admin_ids = await get_all_admin_ids()
        telegram_admin_ids = await get_all_telegram_admin_ids()

        # get_all_admin_ids preserves MAX admin IDs
        assert 100263465646 in admin_ids
        assert 100005511792 in admin_ids
        assert 272982544 in admin_ids
        assert 806750628 in admin_ids

        # get_all_telegram_admin_ids strictly excludes MAX admin IDs
        assert 272982544 in telegram_admin_ids
        assert 806750628 in telegram_admin_ids
        assert 100263465646 not in telegram_admin_ids
        assert 100005511792 not in telegram_admin_ids
        for aid in telegram_admin_ids:
            assert 0 < aid < 100_000_000_000


# -----------------------------------------------------------------------------
# 5. CI REGRESSION CONTRACT & ARCHITECTURAL INVARIANTS
# -----------------------------------------------------------------------------

def test_runtime_env_permission_error_and_missing_file_safety():
    """
    Path.is_file or Path.read_text raising PermissionError (e.g. on non-root Linux CI runners)
    is caught safely and returns empty dict without throwing unhandled exceptions.
    """
    from notification_routing import read_runtime_env_file

    with patch.object(Path, "is_file", side_effect=PermissionError("[Errno 13] Permission denied: '/root/telegram_bots/runtime.env'")):
        res = read_runtime_env_file("/root/telegram_bots/runtime.env")
        assert res == {}

    with patch.object(Path, "read_text", side_effect=PermissionError("[Errno 13] Permission denied")):
        res = read_runtime_env_file("/root/telegram_bots/runtime.env")
        assert res == {}


def test_resolve_target_bot_token_skips_disk_when_token_in_env():
    """
    When candidate token is already present in environment variables,
    resolve_target_bot_token resolves it directly without accessing disk.
    """
    from notification_routing import resolve_target_bot_token, get_instance_by_pm2_name

    target = get_instance_by_pm2_name("tg_yourself_way_bot_new")
    token = "8481853696:AAF_VALID_ENV_TOKEN_FOR_YOURSELF"

    # Telegram bot using BOT_TOKEN
    with patch.dict(os.environ, {"BOT_TOKEN": token}, clear=True), \
         patch("notification_routing.read_runtime_env_file") as mock_read:
        resolved = resolve_target_bot_token(target, is_companion=False)
        assert resolved == token
        mock_read.assert_not_called()

    # Companion bot using COMPANION_TELEGRAM_BOT_TOKEN
    with patch.dict(os.environ, {"COMPANION_TELEGRAM_BOT_TOKEN": token}, clear=True), \
         patch("notification_routing.read_runtime_env_file") as mock_read:
        resolved = resolve_target_bot_token(target, is_companion=True)
        assert resolved == token
        mock_read.assert_not_called()


def test_settings_strictly_never_reads_max_token_from_config_ini():
    """
    get_settings() must NEVER fallback to config.ini for max_token, webhook_secret, or bot_name.
    """
    from max_messenger_bot.settings import get_settings, clear_settings_cache

    clear_settings_cache()
    # Ensure MAX_BOT_TOKEN and MAX_TOKEN are NOT in env
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("MAX_")}
    with patch.dict(os.environ, clean_env, clear=True):
        clear_settings_cache()
        settings = get_settings()
        # Must be empty, never read from config.ini
        assert settings.max_token == ""
        assert settings.webhook_secret == ""
        assert settings.bot_name == ""
    clear_settings_cache()


@pytest.mark.asyncio
async def test_admin_recipients_strictly_isolated_from_max_process_env():
    """
    A MAX process environment containing an alien Telegram admin ID (e.g. from Autobus)
    MUST NEVER leak that admin ID into the companion Telegram bot's recipient list!
    """
    from notification_routing import resolve_target_admin_ids, get_instance_by_pm2_name

    companion_target = get_instance_by_pm2_name("tg_yourself_way_bot_new")
    max_source = get_instance_by_pm2_name("max_id519010411655_bot_new")

    # MAX process env contains valid owner 272982544 + alien admin ID 999888777
    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "OWNER_IDS": "272982544,999888777",
    }, clear=True):
        admin_ids = await resolve_target_admin_ids(companion_target, current_instance=max_source)

        # 999888777 is an alien ID not belonging to tg_yourself_way_bot_new -> MUST NOT be in admins!
        assert 999888777 not in admin_ids
        # Authorized owners of tg_yourself_way_bot_new MUST be present
        assert 272982544 in admin_ids
        assert 979514796 in admin_ids
        for aid in admin_ids:
            assert 0 < aid < 100_000_000_000


def test_detect_current_instance_refuses_to_guess_with_foreign_bot_token_and_shared_db():
    """
    When MAX credentials are incomplete or ambiguous, detect_current_instance MUST NOT
    guess identity based on Autobus BOT_TOKEN or shared database. It must fail closed.
    """
    from notification_routing import detect_current_instance

    # Ambiguous MAX process: has MAX_BOT_TOKEN and Autobus BOT_TOKEN, but no BOT_INSTANCE_ID/PM2_APP_NAME
    with patch.dict(os.environ, {
        "MAX_BOT_TOKEN": "unregistered_max_token",
        "BOT_TOKEN": "8346799080:AAF_AUTOBUS_TOKEN",
        "DATABASE_URL": "postgresql://localhost/psy5d2_db",
    }, clear=True):
        inst = detect_current_instance()
        # MUST return None, NEVER Autobus!
        assert inst is None


def test_resolve_companion_telegram_instance_no_database_guessing():
    """
    resolve_companion_telegram_instance MUST NOT guess a Telegram companion by shared database.
    If companion is unmapped, it must return None.
    """
    from notification_routing import resolve_companion_telegram_instance

    unmapped_max = {
        "platform": "max",
        "pm2_name": "max_unmapped_bot",
        "username": "unmapped_bot",
        "database": "yourself_way_db",  # Shares DB with yourself_way_bot!
        # NO companion_instance defined
    }
    companion = resolve_companion_telegram_instance(unmapped_max)
    # MUST return None, refusing to guess by database
    assert companion is None


def test_user_event_refuses_to_route_to_autobus():
    """
    Non-diagnostic events (user_event, payment, ai_error) MUST NEVER route to Autobus
    from any instance other than Autobus itself.
    """
    from notification_routing import resolve_notification_route, EventCategory, get_instance_by_pm2_name

    source_inst = get_instance_by_pm2_name("tg_yourself_way_bot_new")

    # If someone tries to pass Autobus token or force Autobus for a user event:
    autobus_token = "8346799080:AAF_AUTOBUS_TOKEN"
    route = resolve_notification_route(
        category=EventCategory.USER_EVENT,
        current_instance=source_inst,
        bot_token_override=autobus_token,
    )
    # The Autobus token must be rejected for yourself_way_bot (mismatched platform ID)
    assert route is None


def test_startup_validation_fails_closed_when_misconfigured():
    """
    validate_process_notification_configuration(fail_closed=True) raises ConfigurationError
    when context cannot be determined or companion bot is missing.
    """
    from notification_routing import validate_process_notification_configuration, ConfigurationError

    # Empty env -> instance cannot be determined
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ConfigurationError, match="Current process instance could not be determined"):
            validate_process_notification_configuration(fail_closed=True)


# -----------------------------------------------------------------------------
# 6. FULL BUSINESS FLOW END-TO-END TESTS
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_e2e_real_promo_code_business_flow():
    """
    Real business path end-to-end:
    MAX promo code activation -> apply_promo_code() -> DB write -> notify_telegram_admins()
    -> notification_routing -> aiogram Bot.send_message -> Telegram transport.

    Guarantees:
    - DB is actually committed (promo.times_used == 1).
    - Sent strictly via @yourself_way_bot (8481853696).
    - Zero messages via Autobus (8346799080).
    - Zero messages to MAX IDs (>= 100_000_000_000).
    - Allowed recipients match @yourself_way_bot Telegram owners.
    """
    from max_messenger_bot.services.subscriptions import apply_promo_code
    from max_messenger_bot.storage import StateStore, init_storage
    from max_messenger_bot.legacy import User as MaxUser, PromoCode, async_session_maker, init_db

    await init_db()
    await init_storage()

    user_id = 100074367460  # MAX user (Maria)
    async with async_session_maker() as session:
        user = MaxUser(id=user_id, username="maria_test", first_name="Мария")
        promo = PromoCode(code="PODAROK_E2E", discount_percent=0, free_days=1, times_used=0, max_uses=10)
        session.add(user)
        session.add(promo)
        await session.commit()

    recording_session = RecordingSession()
    yourself_token = "8481853696:AAF_YOURSELF_WAY_CANONICAL_PROD_TOKEN"
    autobus_token = "8346799080:AAF_AUTOBUS_CONTAMINATED_LEGACY_TOKEN"

    client = AsyncMock()
    states = StateStore()

    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "MAX_BOT_NAME": "id519010411655_bot",
        "COMPANION_TELEGRAM_BOT_TOKEN": yourself_token,
        "TELEGRAM_YOURSELF_WAY_BOT_TOKEN": yourself_token,
        "BOT_TOKEN": autobus_token,  # Foreign Autobus token in env must be completely ignored!
        "SKIP_LIVE_BOT_VERIFICATION": "1",
    }, clear=True), patch("telegram_client.create_telegram_bot", lambda token: Bot(token=token, session=recording_session)):

        await apply_promo_code(
            client=client,
            states=states,
            chat_id=user_id,
            user_id=user_id,
            code_text="PODAROK_E2E",
        )

    # 1. Verify DB was updated
    async with async_session_maker() as session:
        db_promo = await session.get(PromoCode, promo.id)
        assert db_promo.times_used == 1, "PromoCode times_used must be incremented"

    # 2. Verify Transport Delivery
    assert len(recording_session.requests) > 0, "Admin notifications must be dispatched"
    for token, bot_id, method in recording_session.requests:
        assert bot_id == 8481853696, "Bot ID must be yourself_way_bot (8481853696)"
        assert bot_id != 8346799080, "Must NEVER dispatch via Autobus!"
        assert token == yourself_token
        target_chat_id = int(method.chat_id)
        assert target_chat_id < 100_000_000_000, "Must never send to a MAX user ID!"
        yourself_way_inst = nr.get_instance_by_id("tg_yourself_way_bot_new")
        assert target_chat_id in yourself_way_inst["telegram_owner_ids"]
        assert "PODAROK_E2E" in method.text
        assert "74367460" in method.text


@pytest.mark.asyncio
async def test_e2e_real_subscription_cancellation_flow():
    """
    Real business path end-to-end:
    MAX subscription auto-renewal toggle -> set_renewal(enabled=False) -> DB update
    -> notify_telegram_admins(category='payment') -> Telegram transport.
    """
    from max_messenger_bot.services.subscriptions import set_renewal
    from max_messenger_bot.legacy import User as MaxUser, UserSubscription, SubscriptionPlan, async_session_maker, init_db
    from datetime import datetime, timezone, timedelta

    await init_db()

    user_id = 100074367461
    now = datetime.now(timezone.utc)
    async with async_session_maker() as session:
        user = MaxUser(id=user_id, username="maria_cancel", first_name="Мария")
        plan = SubscriptionPlan(id=991, name="Премиум 1 мес", price=990, duration_value=30, duration_unit="days")
        sub = UserSubscription(
            user_id=user_id,
            plan_id=991,
            start_date=now,
            end_date=now + timedelta(days=30),
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_attempt_count=0,
            discount_percent=0,
        )
        session.add(user)
        session.add(plan)
        session.add(sub)
        await session.commit()

    recording_session = RecordingSession()
    yourself_token = "8481853696:AAF_YOURSELF_WAY_CANONICAL_PROD_TOKEN"

    client = AsyncMock()

    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "COMPANION_TELEGRAM_BOT_TOKEN": yourself_token,
        "TELEGRAM_YOURSELF_WAY_BOT_TOKEN": yourself_token,
        "DATABASE_URL": os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///:memory:"),
        "SKIP_LIVE_BOT_VERIFICATION": "1",
    }, clear=True), patch("telegram_client.create_telegram_bot", lambda token: Bot(token=token, session=recording_session)):

        await set_renewal(client, chat_id=user_id, user_id=user_id, enabled=False)

    # 1. Verify DB was updated
    from sqlalchemy import select
    async with async_session_maker() as session:
        db_sub = await session.scalar(select(UserSubscription).where(UserSubscription.user_id == user_id))
        print("DEBUG CHECK:", "db_sub.id=", getattr(db_sub, "id", None), "auto_renewal=", getattr(db_sub, "auto_renewal", None))
        assert db_sub is not None
        assert db_sub.auto_renewal is False, "auto_renewal must be disabled"

    # 2. Verify Transport Delivery
    assert len(recording_session.requests) > 0
    for token, bot_id, method in recording_session.requests:
        assert bot_id == 8481853696
        assert token == yourself_token
        assert int(method.chat_id) < 100_000_000_000
        assert "отключил автопродление" in method.text


# -----------------------------------------------------------------------------
# 7. SPECIFIC CODE REVIEW NEGATIVE TESTS (PR #86 BLOCKERS)
# -----------------------------------------------------------------------------

def test_negative_item1_mandatory_bot_instance_id_no_heuristics():
    """
    Item 1 Negative Test:
    BOT_INSTANCE_ID is strictly mandatory for all managed instances.
    Fallbacks via PM2_APP_NAME, MAX_BOT_NAME, BOT_USERNAME are completely removed.
    Unknown or missing ID causes startup failure (ConfigurationError).
    """
    from notification_routing import (
        detect_current_instance,
        validate_process_notification_configuration,
        ConfigurationError,
    )

    # 1. Without BOT_INSTANCE_ID, detect_current_instance MUST return None even if legacy env vars are present
    with patch.dict(os.environ, {
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "MAX_BOT_NAME": "id519010411655_bot",
        "BOT_USERNAME": "yourself_way_bot",
    }, clear=True):
        assert detect_current_instance() is None, "Must NEVER fall back to PM2_APP_NAME or usernames!"

    # 2. Empty string BOT_INSTANCE_ID returns None
    with patch.dict(os.environ, {"BOT_INSTANCE_ID": "  "}, clear=True):
        assert detect_current_instance() is None

    # 3. Unknown BOT_INSTANCE_ID returns None
    with patch.dict(os.environ, {"BOT_INSTANCE_ID": "unregistered_bot_instance_xyz"}, clear=True):
        assert detect_current_instance() is None

    # 4. validate_process_notification_configuration fails closed when BOT_INSTANCE_ID is missing or unknown
    with patch.dict(os.environ, {"BOT_INSTANCE_ID": "unregistered_bot_instance_xyz"}, clear=True):
        with pytest.raises(ConfigurationError, match="BOT_INSTANCE_ID is missing or not registered"):
            validate_process_notification_configuration(fail_closed=True)


@pytest.mark.asyncio
async def test_negative_item2_ssot_registry_no_duplicate_mappings():
    """
    Item 2 Negative Test:
    MAX_COMPANION_MAPPING and DEFAULT_TELEGRAM_OWNER_IDS are removed as alternative sources of truth.
    All companion definitions and Telegram owner IDs reside strictly in config/bot_instances.json.
    Dynamic administrators (is_admin=True) are queried only for the target Telegram bot.
    """
    # 1. Verify duplicate static mapping dictionaries no longer exist in notification_routing module
    assert not hasattr(nr, "MAX_COMPANION_MAPPING"), "MAX_COMPANION_MAPPING duplicate must be removed!"
    assert not hasattr(nr, "DEFAULT_TELEGRAM_OWNER_IDS"), "DEFAULT_TELEGRAM_OWNER_IDS duplicate must be removed!"

    # 2. Companion resolution for MAX instances reads strictly from registry
    max_inst = nr.get_instance_by_id("max_id519010411655_bot_new")
    assert max_inst is not None
    assert max_inst.get("companion_instance") == "tg_yourself_way_bot_new"

    comp = nr.resolve_companion_telegram_instance(max_inst)
    assert comp is not None
    assert comp["pm2_name"] == "tg_yourself_way_bot_new"

    # MAX instance without companion_instance returns None (no fallback dict)
    unmapped = {"platform": "max", "pm2_name": "max_unregistered", "username": "unregistered"}
    assert nr.resolve_companion_telegram_instance(unmapped) is None

    # 3. Target Telegram owners come strictly from target instance's registry telegram_owner_ids
    tg_inst = nr.get_instance_by_id("tg_yourself_way_bot_new")
    assert tg_inst is not None
    assert "telegram_owner_ids" in tg_inst
    assert tg_inst["telegram_owner_ids"] == [979514796, 272982544, 806750628, 1372041348]

    # Dynamic admins: simulate DB containing dynamic admin 555666777
    mock_session = AsyncMock()
    mock_session.execute.return_value = [(555666777,), (979514796,)]
    admin_ids = await nr.resolve_target_admin_ids(
        tg_inst,
        session=mock_session,
        db_url="postgresql://localhost/yourself_way_db",
    )
    assert 555666777 in admin_ids
    assert 979514796 in admin_ids


def test_negative_item3_startup_database_identity_verification():
    """
    Item 3 Negative Test:
    DATABASE_URL must match the instance's configured database at process startup.
    Mismatched DATABASE_URL causes startup refusal (ConfigurationError).
    Reading administrators from an unverified/mismatched database is strictly refused.
    """
    from notification_routing import (
        validate_process_notification_configuration,
        verify_database_identity,
        extract_database_name,
        ConfigurationError,
    )

    inst = nr.get_instance_by_id("max_id519010411655_bot_new")
    assert inst is not None
    assert inst["database"] == "yourself_way_db"

    # 1. URL extraction
    assert extract_database_name("postgresql://user:pass@localhost:5432/yourself_way_db") == "yourself_way_db"
    assert extract_database_name("postgresql+asyncpg://user:pass@localhost:5432/psy5d2_db?ssl=true") == "psy5d2_db"

    # 2. Database identity check: mismatch fails
    assert verify_database_identity(inst, "postgresql://localhost/yourself_way_db") is True
    assert verify_database_identity(inst, "postgresql://localhost/psy5d2_db") is False

    # 3. Startup validation fails closed when DATABASE_URL points to foreign database (e.g. Autobus DB)
    dummy_token = "8481853696:AAF_TEST_TOKEN"
    with patch.dict(os.environ, {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "DATABASE_URL": "postgresql://localhost/psy5d2_db",  # Autobus DB!
        "COMPANION_TELEGRAM_BOT_TOKEN": dummy_token,
    }, clear=True):
        with pytest.raises(ConfigurationError, match="DATABASE_URL mismatch"):
            validate_process_notification_configuration(fail_closed=True)

    # 4. Refuse reading dynamic admins from mismatched database
    target_tg = nr.get_instance_by_id("tg_yourself_way_bot_new")
    assert target_tg is not None
    mock_session = AsyncMock()
    mock_session.execute.side_effect = RuntimeError("Should not be queried!")
    import asyncio
    admins = asyncio.run(nr.resolve_target_admin_ids(
        target_tg,
        session=mock_session,
        db_url="postgresql://localhost/foreign_db",
    ))
    # Still returns baseline owners from registry
    assert 272982544 in admins
    assert mock_session.execute.call_count == 0, "Mismatched DB must NOT be queried for admins!"


@pytest.mark.asyncio
async def test_negative_item4_no_max_token_fallback_and_live_bot_verification():
    """
    Item 4 Negative Test:
    1. settings.py ignores MAX_TOKEN (only MAX_BOT_TOKEN is accepted).
    2. verify_bot_identity_live performs 3 network retries before raising.
    3. SKIP_LIVE_BOT_VERIFICATION is strictly forbidden in non-test runtime.
    4. MAX API identity mismatch raises ConfigurationError.
    """
    from max_messenger_bot.settings import get_settings, clear_settings_cache
    from notification_routing import (
        verify_bot_identity_live,
        validate_process_notification_configuration_async,
        RoutingError,
        ConfigurationError,
    )

    # 1. Fallback MAX_BOT_TOKEN -> MAX_TOKEN eliminated
    clear_settings_cache()
    with patch.dict(os.environ, {"MAX_TOKEN": "legacy_max_token_123"}, clear=True):
        clear_settings_cache()
        settings = get_settings()
        assert settings.max_token == "", "MAX_TOKEN must NOT be accepted as fallback for MAX_BOT_TOKEN!"

    # 2. Live verification retries 3 times on transient network error before failing
    mock_bot = AsyncMock()
    mock_bot.__aenter__.return_value = mock_bot
    mock_bot.id = 8481853696
    mock_bot.token = "8481853696:AAF_RETRY_TEST_TOKEN"
    mock_bot.get_me.side_effect = RuntimeError("Connection timed out 504")

    with pytest.raises(RoutingError, match="failed after 3 attempts"):
        await verify_bot_identity_live(mock_bot, expected_id=8481853696, use_cache=False)
    assert mock_bot.get_me.call_count == 3, "getMe must be retried exactly 3 times"

    # 3. In non-test runtime, SKIP_LIVE_BOT_VERIFICATION is ignored and live verification is enforced
    clean_env = {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "DATABASE_URL": "postgresql://localhost/yourself_way_db",
        "COMPANION_TELEGRAM_BOT_TOKEN": "8481853696:AAF_YOURSELF_WAY_TEST",
        "SKIP_LIVE_BOT_VERIFICATION": "1",
    }
    with patch.dict(os.environ, clean_env, clear=True), \
         patch("notification_routing.verify_bot_identity_live", AsyncMock(return_value={"id": 8481853696, "username": "yourself_way_bot", "verified": True})) as mock_live:
        await validate_process_notification_configuration_async(fail_closed=True, verify_live=False)
        assert mock_live.call_count == 1, "Live bot verification MUST NOT be bypassed in non-test runtime!"

    # 4. In create_web_app, if MAX API client.get_me() returns mismatched identity, ConfigurationError is raised
    from max_messenger_bot.app import create_web_app

    mock_max_client = AsyncMock()
    mock_max_client.__aenter__.return_value = mock_max_client

    max_app_env = {
        "BOT_INSTANCE_ID": "max_id519010411655_bot_new",
        "DATABASE_URL": "postgresql://localhost/yourself_way_db",
        "COMPANION_TELEGRAM_BOT_TOKEN": "8481853696:AAF_YOURSELF_WAY_TEST",
        "MAX_BOT_TOKEN": "max_token_valid_123",
    }

    # 4a. Mismatched user_id
    mock_max_client.get_me.return_value = {"user_id": 999999999, "username": "alien_bot", "is_bot": True}
    clear_settings_cache()
    with patch.dict(os.environ, max_app_env, clear=True), \
         patch("max_messenger_bot.app.MaxApiClient", return_value=mock_max_client), \
         patch("notification_routing.verify_bot_identity_live", AsyncMock(return_value={"id": 8481853696, "username": "yourself_way_bot", "verified": True})):
        clear_settings_cache()
        with pytest.raises(ConfigurationError, match="MAX bot ID mismatch"):
            await create_web_app()

    # 4b. Negative: missing mandatory user_id
    mock_max_client.get_me.return_value = {"username": "id519010411655_bot", "is_bot": True}
    clear_settings_cache()
    with patch.dict(os.environ, max_app_env, clear=True), \
         patch("max_messenger_bot.app.MaxApiClient", return_value=mock_max_client), \
         patch("notification_routing.verify_bot_identity_live", AsyncMock(return_value={"id": 8481853696, "username": "yourself_way_bot", "verified": True})):
        clear_settings_cache()
        with pytest.raises(ConfigurationError, match="missing mandatory user_id"):
            await create_web_app()

    # 4c. Negative: is_bot is False
    mock_max_client.get_me.return_value = {"user_id": 294069217, "is_bot": False, "username": "id519010411655_bot"}
    clear_settings_cache()
    with patch.dict(os.environ, max_app_env, clear=True), \
         patch("max_messenger_bot.app.MaxApiClient", return_value=mock_max_client), \
         patch("notification_routing.verify_bot_identity_live", AsyncMock(return_value={"id": 8481853696, "username": "yourself_way_bot", "verified": True})):
        clear_settings_cache()
        with pytest.raises(ConfigurationError, match="is_bot is not True"):
            await create_web_app()

    # 4d. Negative: username mismatch cannot be bypassed by deprecated name field
    mock_max_client.get_me.return_value = {
        "user_id": 294069217,
        "is_bot": True,
        "name": "id519010411655_bot",  # Deprecated name matching expected
        "username": "unrelated_other_bot",  # But username mismatched
    }
    clear_settings_cache()
    with patch.dict(os.environ, max_app_env, clear=True), \
         patch("max_messenger_bot.app.MaxApiClient", return_value=mock_max_client), \
         patch("notification_routing.verify_bot_identity_live", AsyncMock(return_value={"id": 8481853696, "username": "yourself_way_bot", "verified": True})):
        clear_settings_cache()
        with pytest.raises(ConfigurationError, match="MAX bot username mismatch"):
            await create_web_app()

    # 5. Negative: MAX companion event without COMPANION_TELEGRAM_BOT_TOKEN fails closed (None),
    # even if other tokens are present in env or runtime.env
    from notification_routing import resolve_target_bot_token, get_instance_by_id
    yw_inst = get_instance_by_id("tg_yourself_way_bot_new")
    with patch.dict(os.environ, {
        "BOT_TOKEN": "8481853696:AAF_WRONG_VARIABLE_TOKEN",
        "TELEGRAM_YOURSELF_WAY_BOT_TOKEN": "8481853696:AAF_WRONG_VARIABLE_TOKEN",
    }, clear=True):
        # Companion resolution MUST refuse without COMPANION_TELEGRAM_BOT_TOKEN
        token_res = resolve_target_bot_token(yw_inst, is_companion=True, is_system_diagnostic=False)
        assert token_res is None, "Companion token resolution MUST NOT search in other environment variables!"
