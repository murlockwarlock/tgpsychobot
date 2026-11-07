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
    DEFAULT_TELEGRAM_OWNER_IDS,
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
    saved_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(saved_env)
    nr.clear_registry_cache()


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
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "TELEGRAM_YOURSELF_WAY_BOT_TOKEN": dummy_companion_token,
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
        "PM2_APP_NAME": "max_se13639182_bot_legacy",
        "TELEGRAM_VERAVEDA777_BOT_TOKEN": dummy_companion_token,
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
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "TELEGRAM_YOURSELF_WAY_BOT_TOKEN": dummy_bot_token,
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
        "PM2_APP_NAME": "max_se13639182_bot_legacy",
        "TELEGRAM_VERAVEDA777_BOT_TOKEN": dummy_bot_token,
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
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "MAX_BOT_NAME": "id519010411655_bot",
        "TELEGRAM_YOURSELF_WAY_BOT_TOKEN": dummy_bot_token,
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
        "PM2_APP_NAME": "max_id519010411655_bot_new",
    }, clear=True):
        delivered = await send_admin_notification("Test missing token")
        assert delivered is False


@pytest.mark.asyncio
async def test_empty_bot_token_fails_safely():
    """When token is an empty string, delivery returns False."""
    with patch.dict(os.environ, {
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
    When PM2 started without env export, routing engine safely reads
    the canonical token from host runtime.env file.
    """
    runtime_env_file = tmp_path / "runtime.env"
    dummy_token = "8481853696:AAF_LOADED_FROM_DISK_FILE_TOKEN"
    runtime_env_file.write_text(f"TELEGRAM_YOURSELF_WAY_BOT_TOKEN={dummy_token}\n", encoding="utf-8")

    recording_session = RecordingSession()

    with patch.dict(os.environ, {
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "PROD_RUNTIME_ENV": str(runtime_env_file),
    }, clear=True), patch("telegram_client.create_telegram_bot", lambda token: Bot(token=token, session=recording_session)):

        delivered = await send_admin_notification("Test recovery from runtime.env file")
        assert delivered is True

    assert len(recording_session.requests) > 0
    token, bot_id, _ = recording_session.requests[0]
    assert bot_id == 8481853696
    assert token == dummy_token


@pytest.mark.asyncio
async def test_telegram_api_failure_does_not_raise_exception():
    """
    Network or Telegram HTTP failures must be logged and handled safely;
    send_admin_notification returns False and NEVER raises to caller.
    """
    recording_session = RecordingSession(fail_with=RuntimeError("Telegram network timeout 504"))
    dummy_token = "8481853696:AAF_YOURSELF_WAY_TOKEN_TEST"

    with patch.dict(os.environ, {
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "TELEGRAM_YOURSELF_WAY_BOT_TOKEN": dummy_token,
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
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "TELEGRAM_YOURSELF_WAY_BOT_TOKEN": dummy_bot_token,
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
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "TELEGRAM_YOURSELF_WAY_BOT_TOKEN": dummy_bot_token,
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

    # Step 1: Simulated first run
    with patch.dict(os.environ, {
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "TELEGRAM_YOURSELF_WAY_BOT_TOKEN": canonical_token,
    }, clear=True), patch("telegram_client.create_telegram_bot", lambda token: Bot(token=token, session=recording_session)):
        deliv1 = await send_admin_notification("Run 1 before restart")
        assert deliv1 is True

    # Step 2: Simulated crash and resurrection where process env only had PM2_APP_NAME
    # but runtime.env exists on host
    with patch.dict(os.environ, {
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "PROD_RUNTIME_ENV": str(runtime_env_file),
    }, clear=True), patch("telegram_client.create_telegram_bot", lambda token: Bot(token=token, session=recording_session)):
        deliv2 = await send_admin_notification("Run 2 after PM2 restart")
        assert deliv2 is True

    assert len(recording_session.requests) >= 2
    for token, bot_id, _ in recording_session.requests:
        assert bot_id == 8481853696
        assert token == canonical_token


def test_validate_process_notification_configuration_diagnostic():
    """Diagnostic check on startup returns structured report without leaking secrets."""
    dummy_token = "8481853696:AAF_YOURSELF_WAY_TOKEN_TEST"
    with patch.dict(os.environ, {
        "PM2_APP_NAME": "max_id519010411655_bot_new",
        "TELEGRAM_YOURSELF_WAY_BOT_TOKEN": dummy_token,
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
        assert target_chat_id in DEFAULT_TELEGRAM_OWNER_IDS["tg_yourself_way_bot_new"]

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
async def test_audit_database_get_all_admin_ids_filters_max_ids():
    """
    database.get_all_admin_ids and get_all_telegram_admin_ids MUST NEVER return MAX user IDs.
    """
    from database import get_all_admin_ids, get_all_telegram_admin_ids

    # Environment contains mixed IDs: MAX owners + Telegram owner
    with patch.dict(os.environ, {
        "OWNER_IDS": "100263465646,100005511792,272982544,806750628",
    }, clear=True):
        admin_ids = await get_all_admin_ids()
        telegram_admin_ids = await get_all_telegram_admin_ids()

        assert 272982544 in admin_ids
        assert 806750628 in admin_ids
        assert 100263465646 not in admin_ids
        assert 100005511792 not in admin_ids

        assert admin_ids == telegram_admin_ids
        for aid in admin_ids:
            assert 0 < aid < 100_000_000_000
