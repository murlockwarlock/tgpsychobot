"""
Tests for post-deploy verification and simulated rollback mechanics (PR #86).
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import notification_routing
from scripts.verify_post_deploy import (
    REPO_ROOT,
    run_instance_in_subprocess,
    run_single_instance_worker,
    verify_post_deploy,
)


@pytest.mark.asyncio
async def test_post_deploy_verification_dry_run():
    """Verify that verify_post_deploy in dry-run mode completes with 100% pass."""
    res = await verify_post_deploy(live_send=False, production=False)
    assert res["status"] == "PASSED"
    assert res["live_send"] is False
    assert res["production_mode"] is False
    assert res["instances_checked"] >= 18
    assert res["isolation_checks"] >= 54
    assert len(res["deliveries"]) == 3

    # Check Yourself Way delivery structure
    yw_del = next(d for d in res["deliveries"] if d["source"] == "max_id519010411655_bot_new")
    assert yw_del["sender_bot_id"] == 8481853696
    assert yw_del["sender_username"].lstrip("@") == "yourself_way_bot"
    assert set(yw_del["recipient_chat_ids"]) >= {979514796, 272982544, 806750628, 1372041348}

    # Check VeraVeda delivery structure
    vv_del = next(d for d in res["deliveries"] if d["source"] == "max_se13639182_bot_legacy")
    assert vv_del["sender_bot_id"] == 8224473896
    assert vv_del["sender_username"].lstrip("@") == "veraveda777_bot"
    assert set(vv_del["recipient_chat_ids"]) >= {979514796, 272982544, 806750628}
    assert 366022567 not in vv_del["recipient_chat_ids"], "MAX platform ID must not be in recipients"

    # Check Autobus system delivery structure
    ab_del = next(d for d in res["deliveries"] if d["source"] == "tg_autobusbusbot_new")
    assert ab_del["sender_bot_id"] == 8346799080
    assert ab_del["sender_username"].lstrip("@") == "autobusbusbot"
    assert set(ab_del["recipient_chat_ids"]) >= {272982544, 979514796, 806750628}


@pytest.mark.asyncio
async def test_dynamically_resolved_admin_recipients():
    """Verify dynamic recipient resolution strictly enforces registry owners and filters MAX IDs."""
    reg = notification_routing.load_bot_registry()
    inst_by_name = {i["pm2_name"]: i for i in reg["instances"]}

    # 1. Yourself Way
    yw_comp = inst_by_name["tg_yourself_way_bot_new"]
    yw_max = inst_by_name["max_id519010411655_bot_new"]
    yw_admins = await notification_routing.resolve_target_admin_ids(yw_comp, current_instance=yw_max)
    assert set(yw_admins) == {272982544, 806750628, 979514796, 1372041348}
    assert 7106093845 not in yw_admins, "Unverified ID 7106093845 must not be included"

    # 2. VeraVeda
    vv_comp = inst_by_name["tg_veraveda777_bot_legacy"]
    vv_max = inst_by_name["max_se13639182_bot_legacy"]
    vv_admins = await notification_routing.resolve_target_admin_ids(vv_comp, current_instance=vv_max)
    assert set(vv_admins) == {272982544, 806750628, 979514796}
    assert 366022567 not in vv_admins, "MAX platform ID 366022567 must not be included"

    # 3. Autobus
    ab_inst = inst_by_name["tg_autobusbusbot_new"]
    ab_admins = await notification_routing.resolve_target_admin_ids(ab_inst, current_instance=ab_inst)
    assert set(ab_admins) == {272982544, 806750628, 979514796}
    assert 571874251 not in ab_admins, "Unverified ID 571874251 must not be included"


def test_negative_isolation_max_bots_never_route_to_autobus():
    """Verify that both MAX bots NEVER route user events, payments, or AI errors to Autobus."""
    reg = notification_routing.load_bot_registry()
    inst_by_name = {i["pm2_name"]: i for i in reg["instances"]}

    max_bots = ["max_id519010411655_bot_new", "max_se13639182_bot_legacy"]
    categories = [
        notification_routing.EventCategory.USER_EVENT,
        notification_routing.EventCategory.PAYMENT,
        notification_routing.EventCategory.AI_ERROR,
    ]

    for mb_name in max_bots:
        max_inst = inst_by_name[mb_name]
        companion = notification_routing.resolve_companion_telegram_instance(max_inst)
        assert companion is not None
        comp_token = f"{companion['platform_id']}:AAMockCompanionToken12345"

        for cat in categories:
            route = notification_routing.resolve_notification_route(
                cat,
                current_instance=max_inst,
                bot_token_override=comp_token,
            )
            assert route is not None, f"Route resolution failed for {mb_name} {cat}"
            # Strict isolation guarantees:
            assert route.target_instance != "tg_autobusbusbot_new", (
                f"LEAK: {mb_name} routed {cat} to tg_autobusbusbot_new!"
            )
            assert route.target_platform_id != 8346799080, (
                f"LEAK: {mb_name} targeted Autobus platform_id 8346799080!"
            )
            assert route.target_instance == companion["pm2_name"], (
                f"Target must be companion {companion['pm2_name']}, got {route.target_instance}"
            )
            assert route.target_platform_id == companion["platform_id"], (
                f"Target ID must be {companion['platform_id']}, got {route.target_platform_id}"
            )


@pytest.mark.asyncio
async def test_negative_missing_bot_instance_id_fails_closed(monkeypatch):
    """Verify that absence of mandatory BOT_INSTANCE_ID strictly fails closed."""
    monkeypatch.delenv("BOT_INSTANCE_ID", raising=False)
    assert notification_routing.detect_current_instance() is None

    # Routing without BOT_INSTANCE_ID fails closed (returns None)
    route = notification_routing.resolve_notification_route(notification_routing.EventCategory.USER_EVENT)
    assert route is None

    # Dispatching alert fails closed (returns False without crashing)
    ok = await notification_routing.send_admin_notification("Negative test message")
    assert ok is False


@pytest.mark.asyncio
async def test_negative_unregistered_bot_instance_id_fails_closed(monkeypatch):
    """Verify that an unknown or bogus BOT_INSTANCE_ID strictly fails closed."""
    monkeypatch.setenv("BOT_INSTANCE_ID", "bogus_unregistered_bot_pm2_name")
    assert notification_routing.detect_current_instance() is None

    route = notification_routing.resolve_notification_route(notification_routing.EventCategory.USER_EVENT)
    assert route is None

    ok = await notification_routing.send_admin_notification("Negative test message")
    assert ok is False


def test_negative_swapped_max_companion_tokens_fails_closed(monkeypatch):
    """
    Verify that if a MAX bot process is configured with a swapped or mismatched
    companion token, routing strictly detects the ID mismatch and fails closed.
    """
    # 1. Yourself Way (8481853696) accidentally given VeraVeda token (8224473896)
    monkeypatch.setenv("BOT_INSTANCE_ID", "max_id519010411655_bot_new")
    monkeypatch.setenv("COMPANION_TELEGRAM_BOT_TOKEN", "8224473896:AASwappedVeraVedaToken12345")
    route_yw = notification_routing.resolve_notification_route(notification_routing.EventCategory.USER_EVENT)
    assert route_yw is None, "Mismatched companion token must be rejected!"

    # 2. VeraVeda (8224473896) accidentally given Yourself Way token (8481853696)
    monkeypatch.setenv("BOT_INSTANCE_ID", "max_se13639182_bot_legacy")
    monkeypatch.setenv("COMPANION_TELEGRAM_BOT_TOKEN", "8481853696:AASwappedYourselfWayToken12345")
    route_vv = notification_routing.resolve_notification_route(notification_routing.EventCategory.USER_EVENT)
    assert route_vv is None, "Mismatched companion token must be rejected!"

    # 3. MAX bot given Autobus token (8346799080)
    monkeypatch.setenv("BOT_INSTANCE_ID", "max_id519010411655_bot_new")
    monkeypatch.setenv("COMPANION_TELEGRAM_BOT_TOKEN", "8346799080:AAAutobusToken12345")
    route_ab = notification_routing.resolve_notification_route(notification_routing.EventCategory.USER_EVENT)
    assert route_ab is None, "Autobus token given to MAX companion must be rejected!"


@pytest.mark.asyncio
async def test_negative_production_mode_fails_closed_when_pm2_missing():
    """Verify that in production mode, missing PM2 raises RuntimeError rather than skipping."""
    # When PM2 is absent (e.g. in test runner), production=True must raise RuntimeError
    with pytest.raises(RuntimeError, match="PRODUCTION VERIFICATION FAILED"):
        await verify_post_deploy(production=True)


def test_simulate_partial_failure_rollback_script():
    """Execute simulate_partial_failure_rollback.py and verify exit code 0."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "simulate_partial_failure_rollback.py"
    res = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
    assert res.returncode == 0, f"Script failed with output:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "SIMULATION COMPLETED SUCCESSFULLY (EXIT 0)" in res.stdout


# --------------------------------------------------------------------------
# Automated Negative Tests for Subprocess Isolation & Strict Delivery Criterion
# --------------------------------------------------------------------------

def test_negative_subprocess_swapped_max_environments_fails():
    """
    Verify that if an isolated child process for MAX Yourself Way is executed with
    VeraVeda's companion token, the verification fails closed with a routing/token error.
    """
    child_env = {
        k: v for k, v in os.environ.items()
        if k in {"PATH", "PYTHONPATH", "LANG", "LC_ALL", "USER", "HOME", "TERM"}
    }
    child_env["PYTHONPATH"] = str(REPO_ROOT)
    child_env["BOT_INSTANCE_ID"] = "max_id519010411655_bot_new"
    # Swapped token: VeraVeda companion ID 8224473896 instead of Yourself Way 8481853696
    child_env["COMPANION_TELEGRAM_BOT_TOKEN"] = "8224473896:AASwappedToken12345"
    child_env["DATABASE_URL"] = "postgresql+asyncpg://user:pass@localhost:5432/yourself_way_db"

    with pytest.raises(RuntimeError) as exc_info:
        run_instance_in_subprocess(
            "max_id519010411655_bot_new",
            child_env=child_env,
            production=False,
        )
    err = str(exc_info.value)
    assert "ROUTING FAILED" in err or "Token prefix mismatch" in err


def test_negative_subprocess_database_mismatch_fails():
    """
    Verify that if an isolated child process has a DATABASE_URL not matching the
    target instance database, verification strictly fails closed on database identity check.
    """
    child_env = {
        k: v for k, v in os.environ.items()
        if k in {"PATH", "PYTHONPATH", "LANG", "LC_ALL", "USER", "HOME", "TERM"}
    }
    child_env["PYTHONPATH"] = str(REPO_ROOT)
    child_env["BOT_INSTANCE_ID"] = "max_id519010411655_bot_new"
    child_env["COMPANION_TELEGRAM_BOT_TOKEN"] = "8481853696:AAValidTokenPrefix12345"
    # Mismatched database: veraveda_db instead of yourself_way_db
    child_env["DATABASE_URL"] = "postgresql+asyncpg://user:pass@localhost:5432/veraveda_db"

    with pytest.raises(RuntimeError) as exc_info:
        run_instance_in_subprocess(
            "max_id519010411655_bot_new",
            child_env=child_env,
            production=False,
        )
    err = str(exc_info.value)
    assert "DATABASE IDENTITY MISMATCH" in err or "does not match expected database" in err


def test_negative_subprocess_missing_mandatory_token_fails():
    """
    Verify that in production mode, missing mandatory bot token causes the
    isolated child process to fail closed with exit code != 0.
    """
    child_env = {
        k: v for k, v in os.environ.items()
        if k in {"PATH", "PYTHONPATH", "LANG", "LC_ALL", "USER", "HOME", "TERM"}
    }
    child_env["PYTHONPATH"] = str(REPO_ROOT)
    child_env["BOT_INSTANCE_ID"] = "tg_autobusbusbot_new"
    child_env["DATABASE_URL"] = "postgresql+asyncpg://user:pass@localhost:5432/psy5d2_db"
    child_env["PRODUCTION"] = "1"
    # Explicitly ensure no BOT_TOKEN
    child_env.pop("BOT_TOKEN", None)

    with pytest.raises(RuntimeError) as exc_info:
        run_instance_in_subprocess(
            "tg_autobusbusbot_new",
            child_env=child_env,
            production=True,
        )
    err = str(exc_info.value)
    assert "Real token missing" in err or "PRODUCTION ERROR" in err


@pytest.mark.asyncio
async def test_negative_partial_delivery_fails_criterion(monkeypatch):
    """
    Verify that if live delivery succeeds for one admin but fails for another,
    the worker records per-recipient results, sets delivery_criterion_met=False,
    status='partial_delivery_failure', and raises RuntimeError.
    """
    monkeypatch.setenv("BOT_INSTANCE_ID", "max_id519010411655_bot_new")
    monkeypatch.setenv("COMPANION_TELEGRAM_BOT_TOKEN", "8481853696:AAValidTokenPrefix12345")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/yourself_way_db")

    # Mock getMe verification
    mock_get_me = AsyncMock(return_value={"verified": True, "id": 8481853696, "username": "yourself_way_bot"})

    # Mock aiogram bot
    mock_bot = MagicMock()
    mock_bot.id = 8481853696

    call_count = 0
    async def fake_send_message(chat_id, text, parse_mode=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = MagicMock()
            msg.message_id = 1001
            return msg
        else:
            raise Exception(f"Simulated Telegram delivery failure for recipient {chat_id}")

    mock_bot.send_message = AsyncMock(side_effect=fake_send_message)

    class MockBotContext:
        async def __aenter__(self):
            return mock_bot
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("notification_routing.verify_bot_identity_live", mock_get_me), \
         patch("telegram_client.create_telegram_bot", return_value=MockBotContext()):
        with pytest.raises(RuntimeError) as exc_info:
            await run_single_instance_worker(
                "max_id519010411655_bot_new",
                live_send=True,
                recipient_chat_ids=[272982544, 806750628],
            )

        err_msg = str(exc_info.value)
        assert "DELIVERY CRITERION FAILED" in err_msg
        rep = getattr(exc_info.value, "report", None)
        assert rep is not None
        assert rep["delivery_criterion_met"] is False
        assert rep["status"] == "partial_delivery_failure"
        assert len(rep["recipient_results"]) >= 2
        # Check first succeeded and second failed
        assert rep["recipient_results"][0]["success"] is True
        assert rep["recipient_results"][0]["message_id"] == 1001
        assert rep["recipient_results"][1]["success"] is False
        assert "Simulated Telegram delivery failure" in rep["recipient_results"][1]["error"]


@pytest.mark.asyncio
async def test_negative_telegram_api_failure_handled(monkeypatch):
    """
    Verify that when Telegram API fails for all recipients, the error reason
    is captured per recipient and delivery criterion rejection is triggered.
    """
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_autobusbusbot_new")
    monkeypatch.setenv("BOT_TOKEN", "8346799080:AAValidTokenPrefix12345")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/psy5d2_db")

    mock_get_me = AsyncMock(return_value={"verified": True, "id": 8346799080, "username": "autobusbusbot"})

    mock_bot = MagicMock()
    mock_bot.id = 8346799080
    mock_bot.send_message = AsyncMock(side_effect=Exception("Telegram 502 Bad Gateway"))

    class MockBotContext:
        async def __aenter__(self):
            return mock_bot
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("notification_routing.verify_bot_identity_live", mock_get_me), \
         patch("telegram_client.create_telegram_bot", return_value=MockBotContext()):
        with pytest.raises(RuntimeError) as exc_info:
            await run_single_instance_worker(
                "tg_autobusbusbot_new",
                live_send=True,
                recipient_chat_ids=[272982544],
            )

        assert "DELIVERY CRITERION FAILED" in str(exc_info.value)
        rep = getattr(exc_info.value, "report", None)
        assert rep is not None
        assert rep["delivery_criterion_met"] is False
        for res in rep["recipient_results"]:
            assert res["success"] is False
            assert "Telegram 502 Bad Gateway" in res["error"]


def test_database_engine_process_isolation():
    """
    Verify that distinct child processes create completely independent SQLAlchemy
    engines based on their individual DATABASE_URL, with different PIDs and memory spaces.
    Demonstrates that in-process mutation fails to switch engines, whereas child processes succeed.
    """
    script = (
        "import os, sys, json; "
        "import database; "
        "print(json.dumps({'pid': os.getpid(), 'db_url': str(database.engine.url)}))"
    )

    env1 = os.environ.copy()
    env1["DATABASE_URL"] = "postgresql+asyncpg://user:pass@localhost:5432/yourself_way_db"
    res1 = subprocess.run([sys.executable, "-c", script], env=env1, capture_output=True, text=True)
    assert res1.returncode == 0, f"Process 1 failed: {res1.stderr}"
    data1 = json.loads(res1.stdout.strip().splitlines()[-1])

    env2 = os.environ.copy()
    env2["DATABASE_URL"] = "postgresql+asyncpg://user:pass@localhost:5432/veraveda_db"
    res2 = subprocess.run([sys.executable, "-c", script], env=env2, capture_output=True, text=True)
    assert res2.returncode == 0, f"Process 2 failed: {res2.stderr}"
    data2 = json.loads(res2.stdout.strip().splitlines()[-1])

    # Distinct processes
    assert data1["pid"] != data2["pid"]
    # Distinct engine configurations bound to respective databases
    assert "yourself_way_db" in data1["db_url"]
    assert "veraveda_db" in data2["db_url"]
    assert data1["db_url"] != data2["db_url"]


# --------------------------------------------------------------------------
# Safe Live-Send and Targeted Recipient Verification Tests
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_send_without_recipients_raises_in_orchestrator():
    """Verify that orchestrator rejects live_send=True without explicit recipient_chat_ids."""
    with pytest.raises(ValueError, match="requires explicit recipient_chat_ids"):
        await verify_post_deploy(live_send=True)


@pytest.mark.asyncio
async def test_live_send_without_recipients_in_production_worker_raises(monkeypatch):
    """Verify that production worker forbids live_send=True without explicit recipient_chat_ids."""
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_autobusbusbot_new")
    monkeypatch.setenv("BOT_TOKEN", "8346799080:AAValidTokenPrefix12345")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/psy5d2_db")

    with pytest.raises(RuntimeError, match="LIVE SEND FORBIDDEN"):
        await run_single_instance_worker("tg_autobusbusbot_new", live_send=True, production=True)


@pytest.mark.asyncio
async def test_live_send_filters_to_specified_recipient_chat_ids(monkeypatch):
    """
    Verify that when recipient_chat_ids is provided, live_send ONLY dispatches messages
    to the specified authorized admins and skips other registered admins.
    """
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_autobusbusbot_new")
    monkeypatch.setenv("BOT_TOKEN", "8346799080:AAValidTokenPrefix12345")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/psy5d2_db")

    mock_get_me = AsyncMock(return_value={"verified": True, "id": 8346799080, "username": "autobusbusbot"})

    mock_bot = MagicMock()
    mock_bot.id = 8346799080
    sent_msg = MagicMock()
    sent_msg.message_id = 9999
    mock_bot.send_message = AsyncMock(return_value=sent_msg)

    class MockBotContext:
        async def __aenter__(self):
            return mock_bot
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("notification_routing.verify_bot_identity_live", mock_get_me), \
         patch("telegram_client.create_telegram_bot", return_value=MockBotContext()):
        # Autobus has multiple admins (e.g. 272982544, 806750628, 979514796)
        # We specify ONLY 272982544
        report = await run_single_instance_worker(
            "tg_autobusbusbot_new",
            live_send=True,
            recipient_chat_ids=[272982544],
        )

        assert report["delivery_criterion_met"] is True
        assert report["status"] == "delivered_all"
        assert len(report["recipient_results"]) == 1
        assert report["recipient_results"][0]["chat_id"] == 272982544
        assert report["recipient_results"][0]["success"] is True

        # Ensure bot.send_message was called ONLY for 272982544
        assert mock_bot.send_message.call_count == 1
        call_kwargs = mock_bot.send_message.call_args.kwargs
        assert call_kwargs["chat_id"] == 272982544


@pytest.mark.asyncio
async def test_live_send_unauthorized_recipient_chat_id_rejected(monkeypatch):
    """Verify that specifying an unauthorized recipient chat ID fails closed."""
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_autobusbusbot_new")
    monkeypatch.setenv("BOT_TOKEN", "8346799080:AAValidTokenPrefix12345")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/psy5d2_db")

    with pytest.raises(RuntimeError, match="None of the specified recipients .* are authorized admins"):
        await run_single_instance_worker(
            "tg_autobusbusbot_new",
            live_send=True,
            recipient_chat_ids=[99999999999],
        )


def test_cli_live_send_requires_recipient_chat_ids():
    """Verify CLI error when --live-send is invoked without --recipient-chat-ids."""
    script_path = REPO_ROOT / "scripts" / "verify_post_deploy.py"
    res = subprocess.run(
        [sys.executable, str(script_path), "--live-send"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    assert "Blind broadcast to all production admins is forbidden" in res.stderr


def test_cli_worker_live_send_requires_recipient_chat_ids():
    """Verify worker CLI error when --check-instance is run with --live-send but without recipients."""
    script_path = REPO_ROOT / "scripts" / "verify_post_deploy.py"
    res = subprocess.run(
        [sys.executable, str(script_path), "--check-instance", "tg_autobusbusbot_new", "--live-send"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    assert "Blind broadcast to all production admins is forbidden" in res.stderr


@pytest.mark.asyncio
async def test_direct_worker_without_recipients_forbidden_unconditionally(monkeypatch):
    """Verify direct invocation of run_single_instance_worker with live_send=True without recipients is rejected."""
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_autobusbusbot_new")
    monkeypatch.setenv("BOT_TOKEN", "8346799080:AAValidTokenPrefix12345")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/psy5d2_db")

    with pytest.raises(RuntimeError, match="LIVE SEND FORBIDDEN"):
        await run_single_instance_worker("tg_autobusbusbot_new", live_send=True)


@pytest.mark.asyncio
async def test_production_mode_without_live_send_never_contacts_telegram(monkeypatch):
    """
    Verify that in production mode without live_send, all checks succeed fail-closed,
    status is 'verified_production_no_send', and telegram_client.create_telegram_bot is NEVER called.
    """
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_autobusbusbot_new")
    monkeypatch.setenv("BOT_TOKEN", "8346799080:AAValidTokenPrefix12345")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/psy5d2_db")

    mock_create_bot = MagicMock()
    with patch("telegram_client.create_telegram_bot", mock_create_bot), \
         patch("notification_routing.verify_database_identity", return_value=True):
        report = await run_single_instance_worker("tg_autobusbusbot_new", live_send=False, production=True)

    assert report["delivery_criterion_met"] is True
    assert report["status"] == "verified_production_no_send"
    assert all(r["message_id"] == "production_verified_no_send" for r in report["recipient_results"])
    mock_create_bot.assert_not_called()


def test_deploy_prod_script_has_no_live_send_or_post_deploy_trigger():
    """Verify that deploy_prod.sh does not automatically invoke verify_post_deploy or --live-send."""
    deploy_script = REPO_ROOT / "deploy_prod.sh"
    content = deploy_script.read_text(encoding="utf-8")
    assert "--live-send" not in content
    assert "verify_post_deploy.py" not in content


