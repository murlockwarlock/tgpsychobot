import asyncio
from datetime import datetime, timedelta
import html
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from aiogram import Bot
from aiogram.methods import SendMessage

import alert_cooldown
import error_reporting
import monitor_psychobots as monitor
import scheduler
from database import AIConfig
from ai_incident_manager import (
    load_ai_incident_state,
    save_ai_incident_state,
    get_active_provider_incidents,
    is_provider_incident_active,
    get_provider_account_fingerprint,
    record_provider_incident_alerted,
    record_provider_incident_recovered,
    clear_all_provider_incidents,
    format_provider_balance_alert_message,
    format_provider_recovery_message,
)


class MockSessionContext:
    def __init__(self, config):
        self.config = config

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def get(self, model, id_val):
        if model == AIConfig:
            return self.config
        return None

    async def commit(self):
        pass


@pytest.mark.asyncio
async def test_scenario_1_bot_a_low_balance_alert(monkeypatch, tmp_path):
    """Scenario 1: Bot A with key KIE-A and balance 1.91 alerts Bot A admins from Bot A."""
    incident_file = str(tmp_path / "ai_incidents.json")
    monkeypatch.setenv("PSYCHOBOTS_AI_INCIDENT_STATE", incident_file)

    config_a = AIConfig(
        id=1,
        provider="Gemini",
        image_generation_provider="KIE",
        kie_api_key="kie_secret_key_a",
        kie_credit_alert_threshold=2.0,
        kie_credit_alert_sent=False,
    )
    monkeypatch.setattr(scheduler, "async_session_maker", lambda: MockSessionContext(config_a))
    monkeypatch.setattr(scheduler, "get_kie_remaining_credits", AsyncMock(return_value=1.91))
    monkeypatch.setattr(scheduler, "get_all_admin_ids", AsyncMock(return_value=[111]))

    bot_a = AsyncMock(spec=Bot)
    bot_a.token = "1111:AA"
    bot_a.send_message = AsyncMock(return_value=True)

    await scheduler.check_kie_credit_balance(bot_a)

    assert bot_a.send_message.call_count == 1
    call_args = bot_a.send_message.call_args
    assert call_args[0][0] == 111
    text = call_args[0][1]
    assert "Низкий баланс KIE" in text
    assert "1,91" in text
    assert "пополнить баланс KIE" in text
    assert "Traceback" not in text
    assert config_a.kie_credit_alert_sent is True

    # Check incident state on disk
    active = get_active_provider_incidents()
    assert len(active) == 1
    key = list(active.keys())[0]
    assert "bot_1111:KIE:" in key


@pytest.mark.asyncio
async def test_scenario_2_bot_b_sufficient_balance_no_alert(monkeypatch, tmp_path):
    """Scenario 2: Bot B with key KIE-B and balance 50.0 sends no alert and creates no incident."""
    incident_file = str(tmp_path / "ai_incidents.json")
    monkeypatch.setenv("PSYCHOBOTS_AI_INCIDENT_STATE", incident_file)

    config_b = AIConfig(
        id=1,
        provider="Gemini",
        image_generation_provider="KIE",
        kie_api_key="kie_secret_key_b",
        kie_credit_alert_threshold=2.0,
        kie_credit_alert_sent=False,
    )
    monkeypatch.setattr(scheduler, "async_session_maker", lambda: MockSessionContext(config_b))
    monkeypatch.setattr(scheduler, "get_kie_remaining_credits", AsyncMock(return_value=50.0))
    monkeypatch.setattr(scheduler, "get_all_admin_ids", AsyncMock(return_value=[222]))

    bot_b = AsyncMock(spec=Bot)
    bot_b.token = "2222:BB"
    bot_b.send_message = AsyncMock()

    await scheduler.check_kie_credit_balance(bot_b)

    assert bot_b.send_message.call_count == 0
    assert config_b.kie_credit_alert_sent is False
    assert len(get_active_provider_incidents()) == 0


@pytest.mark.asyncio
async def test_scenario_3_and_4_shared_key_admin_isolation(monkeypatch, tmp_path):
    """Scenarios 3 & 4: Bots A & C share KIE-A key; alerts route strictly to respective admins, no cross-leak."""
    incident_file = str(tmp_path / "ai_incidents.json")
    monkeypatch.setenv("PSYCHOBOTS_AI_INCIDENT_STATE", incident_file)

    shared_key = "kie_shared_secret_123"

    config_a = AIConfig(id=1, provider="Gemini", image_generation_provider="KIE", kie_api_key=shared_key)
    config_c = AIConfig(id=1, provider="Gemini", image_generation_provider="KIE", kie_api_key=shared_key)

    bot_a = AsyncMock(spec=Bot)
    bot_a.token = "1111:AA"
    bot_a.send_message = AsyncMock(return_value=True)

    bot_c = AsyncMock(spec=Bot)
    bot_c.token = "3333:CC"
    bot_c.send_message = AsyncMock(return_value=True)

    class InsufficientBalanceError(Exception):
        pass

    exc = InsufficientBalanceError("KIE task creation failed: Credits insufficient.")

    # 1. Bot A encounters error during generation
    monkeypatch.setattr(error_reporting, "async_session_maker", lambda: MockSessionContext(config_a))
    monkeypatch.setattr(error_reporting, "get_all_admin_ids", AsyncMock(return_value=[101]))

    await error_reporting.notify_admins_about_error(
        bot_a,
        title="Сбой генерации",
        provider="KIE",
        stage="generate_image",
        classification_override="insufficient_balance_quota",
        exception=exc,
    )

    # Bot A delivered to its admin 101 from bot_a
    assert bot_a.send_message.call_count == 1
    assert bot_a.send_message.call_args[0][0] == 101
    assert bot_c.send_message.call_count == 0

    # 2. Bot C encounters error during generation
    monkeypatch.setattr(error_reporting, "async_session_maker", lambda: MockSessionContext(config_c))
    monkeypatch.setattr(error_reporting, "get_all_admin_ids", AsyncMock(return_value=[303]))

    await error_reporting.notify_admins_about_error(
        bot_c,
        title="Сбой генерации",
        provider="KIE",
        stage="generate_image",
        classification_override="insufficient_balance_quota",
        exception=exc,
    )

    # Bot C delivered to its admin 303 from bot_c
    assert bot_c.send_message.call_count == 1
    assert bot_c.send_message.call_args[0][0] == 303
    # Bot A received no additional calls
    assert bot_a.send_message.call_count == 1

    # 3. Subsequent errors on both bots are suppressed by cooldown
    await error_reporting.notify_admins_about_error(bot_a, title="Сбой", provider="KIE", stage="gen", classification_override="insufficient_balance_quota", exception=exc)
    await error_reporting.notify_admins_about_error(bot_c, title="Сбой", provider="KIE", stage="gen", classification_override="insufficient_balance_quota", exception=exc)
    assert bot_a.send_message.call_count == 1
    assert bot_c.send_message.call_count == 1


@pytest.mark.asyncio
async def test_scenario_5_repeated_generation_errors_suppressed(monkeypatch, tmp_path):
    """Scenario 5: 10 repeated generation errors result in exactly 1 alert."""
    incident_file = str(tmp_path / "ai_incidents.json")
    monkeypatch.setenv("PSYCHOBOTS_AI_INCIDENT_STATE", incident_file)

    config = AIConfig(id=1, provider="Gemini", image_generation_provider="KIE", kie_api_key="kie_key_scenario_5")
    monkeypatch.setattr(error_reporting, "async_session_maker", lambda: MockSessionContext(config))
    monkeypatch.setattr(error_reporting, "get_all_admin_ids", AsyncMock(return_value=[999]))

    bot = AsyncMock(spec=Bot)
    bot.token = "5555:TT"
    bot.send_message = AsyncMock(return_value=True)

    class InsufficientBalanceError(Exception):
        pass

    exc = InsufficientBalanceError("KIE task creation failed: Credits insufficient.")

    for _ in range(10):
        await error_reporting.notify_admins_about_error(
            bot,
            title="Сбой генерации",
            provider="KIE",
            stage="generate_image",
            classification_override="insufficient_balance_quota",
            exception=exc,
        )

    assert bot.send_message.call_count == 1


@pytest.mark.asyncio
async def test_scenario_6_process_restart_persistence(monkeypatch, tmp_path):
    """Scenario 6: Bot restart preserves incident state from disk without re-alerting."""
    incident_file = str(tmp_path / "ai_incidents.json")
    monkeypatch.setenv("PSYCHOBOTS_AI_INCIDENT_STATE", incident_file)

    config = AIConfig(id=1, provider="Gemini", image_generation_provider="KIE", kie_api_key="kie_key_scenario_6")
    monkeypatch.setattr(error_reporting, "async_session_maker", lambda: MockSessionContext(config))
    monkeypatch.setattr(error_reporting, "get_all_admin_ids", AsyncMock(return_value=[999]))

    bot = AsyncMock(spec=Bot)
    bot.token = "6666:TT"
    bot.send_message = AsyncMock(return_value=True)

    class InsufficientBalanceError(Exception):
        pass

    exc = InsufficientBalanceError("KIE task creation failed: Credits insufficient.")

    # 1. Initial alert
    await error_reporting.notify_admins_about_error(bot, title="Сбой", provider="KIE", stage="gen", classification_override="insufficient_balance_quota", exception=exc)
    assert bot.send_message.call_count == 1

    # 2. Simulate process restart: reset in-memory cooldown cache
    error_reporting._insufficient_balance_cooldown.reset_all()

    # 3. Next generation failure occurs after restart
    await error_reporting.notify_admins_about_error(bot, title="Сбой", provider="KIE", stage="gen", classification_override="insufficient_balance_quota", exception=exc)
    # Disk state prevented duplicate alert!
    assert bot.send_message.call_count == 1


@pytest.mark.asyncio
async def test_scenario_7_balance_remains_low_6h_reminder(monkeypatch, tmp_path):
    """Scenario 7: Balance remains low; incident stays active and sends reminder after 6 hours."""
    incident_file = str(tmp_path / "ai_incidents.json")
    monkeypatch.setenv("PSYCHOBOTS_AI_INCIDENT_STATE", incident_file)

    now = int(datetime.now().timestamp())
    bot = AsyncMock(spec=Bot)
    bot.token = "1111:AA"
    bot.send_message = AsyncMock(return_value=True)

    config = AIConfig(id=1, provider="Gemini", image_generation_provider="KIE", kie_api_key="kie_key_a", kie_credit_alert_threshold=2.0, kie_credit_alert_sent=True)
    monkeypatch.setattr(scheduler, "async_session_maker", lambda: MockSessionContext(config))
    monkeypatch.setattr(scheduler, "get_kie_remaining_credits", AsyncMock(return_value=1.5))
    monkeypatch.setattr(scheduler, "get_all_admin_ids", AsyncMock(return_value=[111]))

    # Seed active incident alerted 1 hour ago
    fp = get_provider_account_fingerprint("kie_key_a")
    record_provider_incident_alerted("bot_1111", "KIE", fp, now=now - 3600)

    # 1. Run at +1 hour: no reminder yet
    await scheduler.check_kie_credit_balance(bot)
    assert bot.send_message.call_count == 0

    # 2. Update state to 7 hours ago (> 6h cooldown)
    record_provider_incident_alerted("bot_1111", "KIE", fp, now=now - 7 * 3600)

    # 3. Run at +7 hours: reminder sent!
    await scheduler.check_kie_credit_balance(bot)
    assert bot.send_message.call_count == 1
    assert "Низкий баланс KIE" in bot.send_message.call_args[0][1]
    assert is_provider_incident_active("bot_1111", "KIE", fp) is True


@pytest.mark.asyncio
async def test_scenario_8_balance_restoration_resolves_incident(monkeypatch, tmp_path):
    """Scenario 8: Confirmed balance check restores balance and resolves incident."""
    incident_file = str(tmp_path / "ai_incidents.json")
    monkeypatch.setenv("PSYCHOBOTS_AI_INCIDENT_STATE", incident_file)

    now = int(datetime.now().timestamp())
    bot = AsyncMock(spec=Bot)
    bot.token = "1111:AA"
    bot.send_message = AsyncMock(return_value=True)

    config = AIConfig(id=1, provider="Gemini", image_generation_provider="KIE", kie_api_key="kie_key_a", kie_credit_alert_threshold=2.0, kie_credit_alert_sent=True)
    monkeypatch.setattr(scheduler, "async_session_maker", lambda: MockSessionContext(config))
    monkeypatch.setattr(scheduler, "get_kie_remaining_credits", AsyncMock(return_value=50.0))
    monkeypatch.setattr(scheduler, "get_all_admin_ids", AsyncMock(return_value=[111]))

    fp = get_provider_account_fingerprint("kie_key_a")
    record_provider_incident_alerted("bot_1111", "KIE", fp, now=now)

    # Balance check detects 50.0 >= 2.0
    await scheduler.check_kie_credit_balance(bot)

    assert bot.send_message.call_count == 1
    text = bot.send_message.call_args[0][1]
    assert "Баланс KIE восстановлен" in text
    assert "50" in text
    assert "Генерация и редактирование изображений снова доступны." in text
    assert config.kie_credit_alert_sent is False
    assert is_provider_incident_active("bot_1111", "KIE", fp) is False


@pytest.mark.asyncio
async def test_scenario_9_balance_api_down_forbids_false_recovery(monkeypatch, tmp_path):
    """Scenario 9: If balance check API fails, false recovery is forbidden and incident stays active."""
    incident_file = str(tmp_path / "ai_incidents.json")
    monkeypatch.setenv("PSYCHOBOTS_AI_INCIDENT_STATE", incident_file)

    now = int(datetime.now().timestamp())
    bot = AsyncMock(spec=Bot)
    bot.token = "1111:AA"
    bot.send_message = AsyncMock(return_value=True)

    config = AIConfig(id=1, provider="Gemini", image_generation_provider="KIE", kie_api_key="kie_key_a", kie_credit_alert_threshold=2.0, kie_credit_alert_sent=True)
    monkeypatch.setattr(scheduler, "async_session_maker", lambda: MockSessionContext(config))

    from ai_integration import AIServiceError
    monkeypatch.setattr(scheduler, "get_kie_remaining_credits", AsyncMock(side_effect=AIServiceError("503 Service Unavailable")))
    monkeypatch.setattr(scheduler, "get_all_admin_ids", AsyncMock(return_value=[111]))

    fp = get_provider_account_fingerprint("kie_key_a")
    record_provider_incident_alerted("bot_1111", "KIE", fp, now=now)

    await scheduler.check_kie_credit_balance(bot)

    # NO false recovery alert sent!
    assert bot.send_message.call_count == 0
    assert config.kie_credit_alert_sent is True
    # Incident MUST REMAIN ACTIVE!
    assert is_provider_incident_active("bot_1111", "KIE", fp) is True


@pytest.mark.asyncio
async def test_scenario_10_telegram_delivery_failure_safe_retry(monkeypatch, tmp_path):
    """Scenario 10: If Telegram delivery fails, delivery is unconfirmed and retried safely."""
    incident_file = str(tmp_path / "ai_incidents.json")
    monkeypatch.setenv("PSYCHOBOTS_AI_INCIDENT_STATE", incident_file)

    config = AIConfig(id=1, provider="Gemini", image_generation_provider="KIE", kie_api_key="kie_key_scenario_10")
    monkeypatch.setattr(error_reporting, "async_session_maker", lambda: MockSessionContext(config))
    monkeypatch.setattr(error_reporting, "get_all_admin_ids", AsyncMock(return_value=[111]))

    bot = AsyncMock(spec=Bot)
    bot.token = "1010:AA"
    # Attempt 1: Telegram fails (e.g. 502 / network error)
    bot.send_message = AsyncMock(side_effect=Exception("Telegram 502 Bad Gateway"))

    class InsufficientBalanceError(Exception):
        pass

    exc = InsufficientBalanceError("Credits insufficient.")

    await error_reporting.notify_admins_about_error(bot, title="Сбой", provider="KIE", stage="gen", classification_override="insufficient_balance_quota", exception=exc)
    fp = get_provider_account_fingerprint("kie_key_scenario_10")
    # Incident is not confirmed delivered
    assert is_provider_incident_active("bot_1010", "KIE", fp) is False

    # Attempt 2: Telegram succeeds
    bot.send_message = AsyncMock(return_value=True)
    await error_reporting.notify_admins_about_error(bot, title="Сбой", provider="KIE", stage="gen", classification_override="insufficient_balance_quota", exception=exc)
    assert bot.send_message.call_count == 1
    assert is_provider_incident_active("bot_1010", "KIE", fp) is True


@pytest.mark.asyncio
async def test_scenario_11_autobus_isolation_no_kie_balance_alerts(monkeypatch, tmp_path):
    """Scenario 11: KIE balance errors are completely absent from central autobus monitoring."""
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    apps = [
        {"name": "tg_someonelikeyouai02_bot_legacy", "status": "online", "token": "tok_1", "owner_ids": [272982544]}
    ]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    async def fake_empty_async(*args, **kwargs):
        return []

    monkeypatch.setattr(monitor, "check_db_schema", fake_empty_async)
    monkeypatch.setattr(monitor, "check_telegram", fake_empty_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", fake_empty_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_empty_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", fake_empty_async)

    # PM2 error log contains KIE InsufficientBalanceError
    monkeypatch.setattr(
        monitor, "read_new_log_errors",
        lambda *args, **kwargs: [
            "ERROR | handlers | InsufficientBalanceError: KIE task creation failed: Credits insufficient. Your current balance isn't enough to run this request."
        ]
    )

    sent_alerts = []
    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True
    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    exit_code = await monitor.run_check(False)
    # No issues found for Autobus!
    assert exit_code == 0
    assert len(sent_alerts) == 0

    state = monitor.load_state(state_file)
    assert "ai_balance_kie" not in state.get("active_incidents", {})


@pytest.mark.asyncio
async def test_telegram_http_502_single_transient_no_alert(monkeypatch, tmp_path):
    """Telegram HTTP 502 single transient failure: suppressed without alert."""
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    apps = [
        {"name": "tg_someonewithyou01_bot_new", "status": "online", "token": "tok_1", "owner_ids": [272982544]},
        {"name": "tg_someonewithyou02_bot_new", "status": "online", "token": "tok_2", "owner_ids": [272982544]},
    ]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    async def fake_empty_async(*args, **kwargs):
        return []

    monkeypatch.setattr(monitor, "check_db_schema", fake_empty_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", fake_empty_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_empty_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", fake_empty_async)
    monkeypatch.setattr(monitor, "read_new_log_errors", lambda *args, **kwargs: [])

    # Single isolated bot has getMe HTTP 502 on first check
    async def fake_check_telegram(app, http):
        if app["name"] == "tg_someonewithyou01_bot_new":
            return ["Telegram getMe failed: HTTP 502 {'ok': False, 'error_code': 502, 'description': 'Bad Gateway'}"]
        return []

    monkeypatch.setattr(monitor, "check_telegram", fake_check_telegram)

    sent_alerts = []

    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True

    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    exit_code = await monitor.run_check(False)
    # Transient isolated 502 is suppressed on check 1
    assert exit_code == 0
    assert len(sent_alerts) == 0

    state = monitor.load_state(state_file)
    assert state.get("polling_failure_counts", {}).get("tg_someonewithyou01_bot_new") == 1
    assert "network_tg_someonewithyou01_bot_new" not in state.get("active_incidents", {})


@pytest.mark.asyncio
async def test_telegram_http_502_sustained_alerts(monkeypatch, tmp_path):
    """Telegram HTTP 502 sustained failure: alerts on repeatability."""
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    apps = [
        {"name": "tg_someonewithyou01_bot_new", "status": "online", "token": "tok_1", "owner_ids": [272982544]},
        {"name": "tg_someonewithyou02_bot_new", "status": "online", "token": "tok_2", "owner_ids": [272982544]},
    ]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    async def fake_empty_async(*args, **kwargs):
        return []

    monkeypatch.setattr(monitor, "check_db_schema", fake_empty_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", fake_empty_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_empty_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", fake_empty_async)
    monkeypatch.setattr(monitor, "read_new_log_errors", lambda *args, **kwargs: [])

    async def fake_check_telegram(app, http):
        if app["name"] == "tg_someonewithyou01_bot_new":
            return ["Telegram getMe failed: HTTP 502 {'ok': False, 'error_code': 502, 'description': 'Bad Gateway'}"]
        return []

    monkeypatch.setattr(monitor, "check_telegram", fake_check_telegram)

    sent_alerts = []

    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True

    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    # Check 1: transient -> suppressed
    exit_code_1 = await monitor.run_check(False)
    assert exit_code_1 == 0
    assert len(sent_alerts) == 0

    # Check 2: sustained -> alert sent
    exit_code_2 = await monitor.run_check(False)
    assert exit_code_2 == 1
    assert len(sent_alerts) == 1
    assert "tg_someonewithyou01_bot_new" in sent_alerts[0]
    assert "Telegram getMe failed" in sent_alerts[0]

    state = monitor.load_state(state_file)
    assert "network_tg_someonewithyou01_bot_new" in state.get("active_incidents", {})


@pytest.mark.asyncio
async def test_polling_errors_count_growth_one_continuing_incident(monkeypatch, tmp_path):
    """Polling errors 9 -> 14 -> 20: one continuing incident, not three different alerts."""
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    apps = [
        {"name": "tg_someonewithyou01_bot_new", "status": "online", "token": "tok_1", "owner_ids": [272982544]}
    ]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    async def fake_empty_async(*args, **kwargs):
        return []

    monkeypatch.setattr(monitor, "check_db_schema", fake_empty_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", fake_empty_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_empty_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", fake_empty_async)

    sent_alerts = []

    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True

    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    # Run 1: 9 polling errors + 502
    async def fake_tg(app, http):
        return ["Telegram getMe failed: HTTP 502 Bad Gateway"]
    monkeypatch.setattr(monitor, "check_telegram", fake_tg)

    # Start with sustained state (consecutive fails >= 2)
    state = monitor.load_state(state_file)
    state["polling_failure_counts"] = {"tg_someonewithyou01_bot_new": 2}
    monitor.save_state(state, state_file)

    monkeypatch.setattr(
        monitor, "read_new_log_errors",
        lambda *args, **kwargs: ["Logs: 9 polling fetch errors since last check"]
    )
    await monitor.run_check(False)
    assert len(sent_alerts) == 1
    assert "9 polling fetch errors" in sent_alerts[0]

    # Run 2: error count grows to 14 polling errors
    monkeypatch.setattr(
        monitor, "read_new_log_errors",
        lambda *args, **kwargs: ["Logs: 14 polling fetch errors since last check"]
    )
    await monitor.run_check(False)
    # NO second alert sent!
    assert len(sent_alerts) == 1

    # Run 3: error count grows to 20 polling errors
    monkeypatch.setattr(
        monitor, "read_new_log_errors",
        lambda *args, **kwargs: ["Logs: 20 polling fetch errors since last check"]
    )
    await monitor.run_check(False)
    # NO third alert sent!
    assert len(sent_alerts) == 1


@pytest.mark.asyncio
async def test_telegram_network_recovery_single_recovery_alert(monkeypatch, tmp_path):
    """Restoration of connectivity sends exactly one recovery alert and clears incident."""
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    apps = [
        {"name": "tg_someonewithyou01_bot_new", "status": "online", "token": "tok_1", "owner_ids": [272982544]}
    ]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    async def fake_empty_async(*args, **kwargs):
        return []

    monkeypatch.setattr(monitor, "check_db_schema", fake_empty_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", fake_empty_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_empty_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", fake_empty_async)

    sent_alerts = []

    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True

    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    # Active single-bot network incident in state
    state = monitor.load_state(state_file)
    state["active_incidents"]["network_tg_someonewithyou01_bot_new"] = {
        "first_seen": int(datetime.now().timestamp()) - 600,
        "last_seen": int(datetime.now().timestamp()),
        "last_alerted": int(datetime.now().timestamp()) - 600,
        "alert_count": 1,
        "affected_bots": ["tg_someonewithyou01_bot_new"],
        "summary": "Telegram HTTP 502 (Bad Gateway)",
    }
    monitor.save_state(state, state_file)

    # Check: bot is now fully recovered
    monkeypatch.setattr(monitor, "check_telegram", fake_empty_async)
    monkeypatch.setattr(monitor, "read_new_log_errors", lambda *args, **kwargs: [])

    exit_code = await monitor.run_check(False)
    assert exit_code == 0

    # Exactly one recovery alert sent
    assert len(sent_alerts) == 1
    assert "Восстановление: Telegram API / Прокси в норме" in sent_alerts[0]
    assert "tg_someonewithyou01_bot_new" in sent_alerts[0]

    state = monitor.load_state(state_file)
    assert "network_tg_someonewithyou01_bot_new" not in state.get("active_incidents", {})


@pytest.mark.asyncio
async def test_two_or_more_bots_form_group_network_incident(monkeypatch, tmp_path):
    """Two or more affected bots form a single group network incident."""
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    apps = [
        {"name": "tg_someonewithyou01_bot_new", "status": "online", "token": "tok_1", "owner_ids": [272982544]},
        {"name": "tg_someonewithyou02_bot_new", "status": "online", "token": "tok_2", "owner_ids": [272982544]},
        {"name": "tg_someonelikeyouai02_bot_legacy", "status": "online", "token": "tok_3", "owner_ids": [272982544]},
    ]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    async def fake_empty_async(*args, **kwargs):
        return []

    monkeypatch.setattr(monitor, "check_db_schema", fake_empty_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", fake_empty_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_empty_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", fake_empty_async)
    monkeypatch.setattr(monitor, "read_new_log_errors", lambda *args, **kwargs: [])

    # Two bots have HTTP 502
    async def fake_tg(app, http):
        if app["name"] in ("tg_someonewithyou01_bot_new", "tg_someonewithyou02_bot_new"):
            return ["Telegram getMe failed: HTTP 502 {'ok': False, 'error_code': 502, 'description': 'Bad Gateway'}"]
        return []

    monkeypatch.setattr(monitor, "check_telegram", fake_tg)

    sent_alerts = []

    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True

    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    exit_code = await monitor.run_check(False)
    assert exit_code == 1
    # Exactly one combined group alert
    assert len(sent_alerts) == 1
    assert "Общий сетевой инцидент Telegram API / Прокси" in sent_alerts[0]
    assert "Затронуто ботов: 2" in sent_alerts[0]

    state = monitor.load_state(state_file)
    assert monitor.INCIDENT_SHARED_NETWORK in state.get("active_incidents", {})


@pytest.mark.asyncio
async def test_pm2_stopped_and_db_errors_preserved(monkeypatch, tmp_path):
    """PM2 stopped and database errors are preserved and alerted without suppression."""
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    apps = [
        {"name": "tg_bot_stopped", "status": "stopped", "token": "tok_1", "owner_ids": [272982544]},
        {"name": "tg_bot_db_broken", "status": "online", "token": "tok_2", "owner_ids": [272982544]},
    ]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    async def fake_empty_async(*args, **kwargs):
        return []

    monkeypatch.setattr(monitor, "check_telegram", fake_empty_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", fake_empty_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_empty_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", fake_empty_async)
    monkeypatch.setattr(monitor, "read_new_log_errors", lambda *args, **kwargs: [])

    async def mock_db(app):
        if app["name"] == "tg_bot_db_broken":
            return ["DB: database connection failed: connection refused"]
        return []

    monkeypatch.setattr(monitor, "check_db_schema", mock_db)

    sent_alerts = []

    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True

    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    exit_code = await monitor.run_check(False)
    assert exit_code == 1
    assert len(sent_alerts) == 1
    assert "PM2: status is stopped" in sent_alerts[0]
    assert "DB: database connection failed: connection refused" in sent_alerts[0]


@pytest.mark.asyncio
async def test_bot_blocked_errors_completely_suppressed(monkeypatch, tmp_path):
    """Bot blocked by user errors remain completely suppressed from alerts."""
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    error_log_file = tmp_path / "test-bot-error.log"
    error_log_file.write_text(
        "2026-11-14 10:00:00 | ERROR | aiogram.dispatcher | TelegramForbiddenError: Forbidden: bot was blocked by the user\n"
        "2026-11-14 10:01:00 | ERROR | notification_transport | TelegramForbiddenError: bot was blocked by the user\n"
    )
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    apps = [
        {
            "name": "tg_bot_active",
            "status": "online",
            "token": "tok_1",
            "owner_ids": [272982544],
            "error_log": str(error_log_file),
        },
    ]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    async def fake_empty_async(*args, **kwargs):
        return []

    monkeypatch.setattr(monitor, "check_telegram", fake_empty_async)
    monkeypatch.setattr(monitor, "check_db_schema", fake_empty_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", fake_empty_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_empty_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", fake_empty_async)

    sent_alerts = []

    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True

    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    # With include_existing_log_errors=True, it reads the error log containing blocked lines
    exit_code = await monitor.run_check(True)
    assert exit_code == 0
    assert len(sent_alerts) == 0


def test_generated_telegram_methods_pass_aiogram_pydantic_validation():
    """All generated Telegram message texts pass aiogram and Pydantic model validation."""
    alert_1 = monitor.format_ai_balance_alert("KIE", ["tg_someonelikeyouai02_bot_legacy", "tg_someonewithyou01_bot_new"])
    alert_2 = monitor.format_ai_balance_recovery_alert("KIE", ["tg_someonelikeyouai02_bot_legacy"], "15 минут")
    alert_3 = monitor.format_recovery_alert(1, "45 минут", bot_names=["tg_someonewithyou01_bot_new"])
    alert_4 = monitor.format_recovery_alert(3, "1 час 20 минут")
    alert_5 = monitor.format_issues_alert(
        all_issues=[
            ("tg_someonewithyou01_bot_new", [
                "Telegram getMe failed: HTTP 502 {'ok': False, 'error_code': 502, 'description': 'Bad Gateway'}",
                "Logs: 14 polling fetch errors since last check",
            ])
        ],
        is_shared_network_active=False,
        include_network_summary=False,
        shared_network_summary="",
        affected_net_bots=["tg_someonewithyou01_bot_new"],
        total_tg_bots=17,
        proxy_status="настроен (http://127.0.0.1:3128)",
    )
    alert_6 = format_provider_balance_alert_message("KIE", 1.91)
    alert_7 = format_provider_recovery_message("KIE", 50.0)

    for text in (alert_1, alert_2, alert_3, alert_4, alert_5, alert_6, alert_7):
        # Validate that text can be instantiated into aiogram SendMessage method
        method = SendMessage(chat_id=272982544, text=text[:3900], parse_mode="HTML")
        assert method.chat_id == 272982544
        assert len(method.text) > 0
        assert len(method.text) <= 4096
