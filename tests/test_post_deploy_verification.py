"""
Tests for post-deploy verification and simulated rollback mechanics (PR #86).
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

import notification_routing
from scripts.verify_post_deploy import verify_post_deploy


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
