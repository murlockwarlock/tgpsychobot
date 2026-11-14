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


@pytest.mark.asyncio
async def test_kie_credits_exhausted_single_service_alert(monkeypatch, tmp_path):
    """KIE credits exhausted: single service alert, not dozens across repeated failures."""
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    apps = [
        {
            "name": "tg_someonelikeyouai02_bot_legacy",
            "status": "online",
            "token": "tok_1",
            "owner_ids": [272982544],
        }
    ]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    async def fake_empty_async(*args, **kwargs):
        return []

    monkeypatch.setattr(monitor, "check_db_schema", fake_empty_async)
    monkeypatch.setattr(monitor, "check_telegram", fake_empty_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", fake_empty_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_empty_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", fake_empty_async)

    # 1. Error reporting in handlers: 5 consecutive failed image generations
    error_reporting.reset_insufficient_balance_cooldown()
    fake_bot = AsyncMock(spec=Bot)
    fake_bot.send_message = AsyncMock()

    class InsufficientBalanceError(Exception):
        pass

    exc = InsufficientBalanceError("KIE task creation failed: Credits insufficient. Your current balance isn't enough to run this request.")

    with patch("error_reporting.get_all_admin_ids", return_value=[272982544]):
        for _ in range(5):
            await error_reporting.notify_admins_about_error(
                fake_bot,
                title="Сбой генерации изображения",
                provider="KIE",
                model="bytedance/seedream-v4-text-to-image",
                stage="generate_image",
                classification_override="insufficient_balance_quota",
                exception=exc,
            )

    # Admins received exactly ONE alert from error_reporting, not 5
    assert fake_bot.send_message.call_count == 1

    # 2. Central watchdog (monitor_psychobots):
    def mock_log_errors(app, state, include_existing):
        return [
            "ERROR | handlers | Сбой генерации изображения | provider=KIE | "
            "classification=insufficient_balance_quota | exception_class=InsufficientBalanceError | "
            "root_cause=KIE task creation failed: Credits insufficient : Your current balance isn’t enough"
        ]

    monkeypatch.setattr(monitor, "read_new_log_errors", mock_log_errors)

    sent_alerts = []

    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True

    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    exit_code = await monitor.run_check(False)
    assert exit_code == 1
    assert len(sent_alerts) == 1
    assert "Недостаточно кредитов AI-провайдера (KIE)" in sent_alerts[0]
    assert "tg_someonelikeyouai02_bot_legacy" in sent_alerts[0]

    state = monitor.load_state(state_file)
    assert "ai_balance_kie" in state.get("active_incidents", {})


@pytest.mark.asyncio
async def test_kie_repeated_failures_no_spam(monkeypatch, tmp_path):
    """Repeated identical KIE failures: no repeat spam on subsequent monitor runs."""
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

    sent_alerts = []

    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True

    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    # Run 1: first failure
    def mock_log_1(app, state, include_existing):
        return ["Logs: InsufficientBalanceError: Credits insufficient | extra=prompt_len=120"]

    monkeypatch.setattr(monitor, "read_new_log_errors", mock_log_1)
    await monitor.run_check(False)
    assert len(sent_alerts) == 1

    # Run 2: second failure with different prompt_len
    def mock_log_2(app, state, include_existing):
        return ["Logs: InsufficientBalanceError: Credits insufficient | extra=prompt_len=350"]

    monkeypatch.setattr(monitor, "read_new_log_errors", mock_log_2)
    await monitor.run_check(False)

    # NO duplicate alert sent!
    assert len(sent_alerts) == 1

    # Run 3: third failure
    def mock_log_3(app, state, include_existing):
        return ["Logs: InsufficientBalanceError: Credits insufficient | extra=prompt_len=512"]

    monkeypatch.setattr(monitor, "read_new_log_errors", mock_log_3)
    await monitor.run_check(False)
    assert len(sent_alerts) == 1


@pytest.mark.asyncio
async def test_kie_balance_restoration_resolves_incident(monkeypatch, tmp_path):
    """KIE balance restoration resolves incident and sends exactly one recovery alert."""
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

    sent_alerts = []

    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True

    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    # Run 1: KIE failure occurs
    monkeypatch.setattr(
        monitor, "read_new_log_errors",
        lambda *args, **kwargs: ["Logs: InsufficientBalanceError: KIE task creation failed: Credits insufficient"]
    )
    await monitor.run_check(False)
    assert len(sent_alerts) == 1
    assert "Недостаточно кредитов AI-провайдера (KIE)" in sent_alerts[0]

    state = monitor.load_state(state_file)
    assert "ai_balance_kie" in state["active_incidents"]

    # Run 2: Balance restored (no log errors)
    monkeypatch.setattr(monitor, "read_new_log_errors", lambda *args, **kwargs: [])
    await monitor.run_check(False)

    assert len(sent_alerts) == 2
    assert "Восстановление: Баланс KIE пополнен" in sent_alerts[1]
    assert "tg_someonelikeyouai02_bot_legacy" in sent_alerts[1]

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

    for text in (alert_1, alert_2, alert_3, alert_4, alert_5):
        # Validate that text can be instantiated into aiogram SendMessage method
        method = SendMessage(chat_id=272982544, text=text[:3900], parse_mode="HTML")
        assert method.chat_id == 272982544
        assert len(method.text) > 0
        assert len(method.text) <= 4096
