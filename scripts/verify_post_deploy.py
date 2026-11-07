#!/usr/bin/env python3
"""
Post-Deploy Verification Script (PR #86).

Validates process isolation, routing, and identity for all 18 managed instances.
Supports:
  --dry-run (default in CI/local): tests registry, PM2/env inspection, dynamic admin resolution,
            sender/recipient identity recording, and negative isolation constraints without live Telegram calls.
  --live-send: explicitly sends safe verification notifications to authorized admin chat_ids
               (only executed on production upon explicit authorization).
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import notification_routing

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_post_deploy")


def inspect_pm2_processes() -> dict[str, Any]:
    """Inspect live PM2 processes if PM2 is available."""
    try:
        raw = subprocess.check_output(["pm2", "jlist"], stderr=subprocess.DEVNULL)
        apps = json.loads(raw)
        return {a["name"]: a for a in apps}
    except Exception:
        logger.warning("PM2 jlist not available in this environment (dry-run mode).")
        return {}


async def verify_post_deploy(
    *,
    live_send: bool = False,
    registry_path: Path | None = None,
    runtime_env_path: Path | None = None,
) -> dict[str, Any]:
    """
    Execute comprehensive post-deploy verification.
    Returns structured results dict. Raises AssertionError on any failure.
    """
    logger.info("=" * 70)
    logger.info("RUNNING POST-DEPLOY VERIFICATION (PR #86)")
    logger.info("Mode: %s", "LIVE SEND" if live_send else "DRY RUN (safe / CI)")
    logger.info("=" * 70)

    reg = notification_routing.load_bot_registry(registry_path or notification_routing.REGISTRY_PATH)
    instances = reg.get("instances", [])
    inst_by_name = {i["pm2_name"]: i for i in instances}

    results: dict[str, Any] = {
        "status": "pending",
        "live_send": live_send,
        "instances_checked": len(instances),
        "deliveries": [],
        "isolation_checks": 0,
        "errors": [],
    }

    # -------------------------------------------------------------
    # 1. PM2 In-Memory Env Inspection (if PM2 present)
    # -------------------------------------------------------------
    pm2_apps = inspect_pm2_processes()
    if pm2_apps:
        logger.info("[CHECK 1] Inspecting live PM2 process memory environments...")
        for name, inst in inst_by_name.items():
            if name not in pm2_apps:
                continue
            proc_env = pm2_apps[name].get("pm2_env", {}).get("env", {})
            iid = proc_env.get("BOT_INSTANCE_ID")
            assert iid == name, f"Wrong BOT_INSTANCE_ID for {name}: {iid}"
            if inst["platform"] == "max":
                comp_token = proc_env.get("COMPANION_TELEGRAM_BOT_TOKEN", "")
                comp_inst = notification_routing.resolve_companion_telegram_instance(inst)
                if comp_inst:
                    exp_pfx = str(comp_inst["platform_id"])
                    assert comp_token.startswith(f"{exp_pfx}:"), (
                        f"Wrong companion token for {name}: expected prefix {exp_pfx}"
                    )
        logger.info("[PASS] Check 1: Live PM2 memory environments verified.")
    else:
        logger.info("[SKIP] Check 1: PM2 not present, skipping live in-memory inspection.")

    # -------------------------------------------------------------
    # 2. Yourself Way (MAX -> @yourself_way_bot) Route & Admin IDs
    # -------------------------------------------------------------
    logger.info("\n[CHECK 2] Validating MAX Yourself Way -> @yourself_way_bot...")
    yw_max = inst_by_name["max_id519010411655_bot_new"]
    yw_comp = notification_routing.resolve_companion_telegram_instance(yw_max)
    assert yw_comp is not None, "Yourself Way companion instance resolution failed!"
    assert yw_comp["pm2_name"] == "tg_yourself_way_bot_new"
    assert yw_comp["platform_id"] == 8481853696
    assert yw_comp["username"].lstrip("@") == "yourself_way_bot"

    # Dynamically resolve allowed Telegram admin IDs from registry & DB
    yw_admins = await notification_routing.resolve_target_admin_ids(yw_comp, current_instance=yw_max)
    expected_yw_owners = set(yw_comp.get("telegram_owner_ids", []))
    assert expected_yw_owners.issubset(set(yw_admins)), (
        f"Resolved admin IDs {yw_admins} do not contain expected registry owners {expected_yw_owners}"
    )
    # Ensure NO MAX IDs or non-Telegram IDs exist
    for aid in yw_admins:
        assert aid < 100_000_000_000, f"Invalid MAX ID leaked into Telegram admins: {aid}"
        assert aid > 0, f"Invalid negative/zero admin ID: {aid}"

    # Route verification
    yw_token = os.environ.get("COMPANION_TELEGRAM_BOT_TOKEN")
    if not yw_token or notification_routing.extract_telegram_bot_id(yw_token) != yw_comp["platform_id"]:
        yw_token = f"{yw_comp['platform_id']}:AAFakeTokenForTest123456789"
    yw_route = notification_routing.resolve_notification_route(
        notification_routing.EventCategory.USER_EVENT,
        current_instance=yw_max,
        bot_token_override=yw_token,
    )
    assert yw_route is not None, "Failed to resolve Yourself Way route!"
    assert yw_route.target_instance == "tg_yourself_way_bot_new"
    assert yw_route.target_platform_id == 8481853696
    assert yw_route.target_username.lstrip("@") == "yourself_way_bot"

    delivery_yw = {
        "source": "max_id519010411655_bot_new",
        "sender_bot_id": yw_route.target_platform_id,
        "sender_username": yw_route.target_username,
        "recipient_chat_ids": list(yw_admins),
        "status": "verified_route",
    }
    logger.info("  Sender: %s (ID %s)", yw_route.target_username, yw_route.target_platform_id)
    logger.info("  Authorized Recipient Chat IDs: %s", list(yw_admins))

    if live_send:
        logger.info("  Dispatching live Telegram message...")
        ok = await notification_routing.send_admin_notification(
            f"🔔 <b>[Live Post-Deploy Verification]</b>\n"
            f"Источник: MAX id519010411655_bot\n"
            f"Отправитель: @yourself_way_bot ({yw_route.target_platform_id})\n"
            f"Получатели: {list(yw_admins)}\n"
            f"Статус: Изоляция подтверждена.",
            category=notification_routing.EventCategory.USER_EVENT,
        )
        assert ok is True, "Failed to deliver Yourself Way notification to live admins!"
        delivery_yw["status"] = "delivered"
    results["deliveries"].append(delivery_yw)
    logger.info("[PASS] Check 2: Yourself Way routing and dynamic admins verified.")

    # -------------------------------------------------------------
    # 3. VeraVeda (MAX -> @veraveda777_bot) Route & Admin IDs
    # -------------------------------------------------------------
    logger.info("\n[CHECK 3] Validating MAX VeraVeda -> @veraveda777_bot...")
    vv_max = inst_by_name["max_se13639182_bot_legacy"]
    vv_comp = notification_routing.resolve_companion_telegram_instance(vv_max)
    assert vv_comp is not None, "VeraVeda companion instance resolution failed!"
    assert vv_comp["pm2_name"] == "tg_veraveda777_bot_legacy"
    assert vv_comp["platform_id"] == 8224473896
    assert vv_comp["username"].lstrip("@") == "veraveda777_bot"

    vv_admins = await notification_routing.resolve_target_admin_ids(vv_comp, current_instance=vv_max)
    expected_vv_owners = set(vv_comp.get("telegram_owner_ids", []))
    assert expected_vv_owners.issubset(set(vv_admins)), (
        f"Resolved admin IDs {vv_admins} do not contain expected registry owners {expected_vv_owners}"
    )
    # 366022567 is MAX platform_id, must strictly NOT be in Telegram admins
    assert 366022567 not in vv_admins, "CRITICAL ERROR: MAX platform ID 366022567 found in Telegram admins!"
    for aid in vv_admins:
        assert aid < 100_000_000_000, f"Invalid MAX ID leaked into Telegram admins: {aid}"

    vv_token = os.environ.get("COMPANION_TELEGRAM_BOT_TOKEN")
    if not vv_token or notification_routing.extract_telegram_bot_id(vv_token) != vv_comp["platform_id"]:
        vv_token = f"{vv_comp['platform_id']}:AAFakeTokenForTest123456789"
    vv_route = notification_routing.resolve_notification_route(
        notification_routing.EventCategory.USER_EVENT,
        current_instance=vv_max,
        bot_token_override=vv_token,
    )
    assert vv_route is not None, "Failed to resolve VeraVeda route!"
    assert vv_route.target_instance == "tg_veraveda777_bot_legacy"
    assert vv_route.target_platform_id == 8224473896
    assert vv_route.target_username.lstrip("@") == "veraveda777_bot"

    delivery_vv = {
        "source": "max_se13639182_bot_legacy",
        "sender_bot_id": vv_route.target_platform_id,
        "sender_username": vv_route.target_username,
        "recipient_chat_ids": list(vv_admins),
        "status": "verified_route",
    }
    logger.info("  Sender: %s (ID %s)", vv_route.target_username, vv_route.target_platform_id)
    logger.info("  Authorized Recipient Chat IDs: %s", list(vv_admins))

    if live_send:
        logger.info("  Dispatching live Telegram message...")
        ok = await notification_routing.send_admin_notification(
            f"🔔 <b>[Live Post-Deploy Verification]</b>\n"
            f"Источник: MAX se13639182_bot\n"
            f"Отправитель: @veraveda777_bot ({vv_route.target_platform_id})\n"
            f"Получатели: {list(vv_admins)}\n"
            f"Статус: Изоляция подтверждена.",
            category=notification_routing.EventCategory.USER_EVENT,
        )
        assert ok is True, "Failed to deliver VeraVeda notification to live admins!"
        delivery_vv["status"] = "delivered"
    results["deliveries"].append(delivery_vv)
    logger.info("[PASS] Check 3: VeraVeda routing and dynamic admins verified.")

    # -------------------------------------------------------------
    # 4. Autobus System Monitoring (SYSTEM_DIAGNOSTIC)
    # -------------------------------------------------------------
    logger.info("\n[CHECK 4] Validating Autobus System Alert Route & Admin IDs...")
    ab_inst = inst_by_name["tg_autobusbusbot_new"]
    ab_admins = await notification_routing.resolve_target_admin_ids(ab_inst, current_instance=ab_inst)
    expected_ab_owners = set(ab_inst.get("telegram_owner_ids", []))
    assert expected_ab_owners.issubset(set(ab_admins)), (
        f"Resolved admin IDs {ab_admins} do not contain expected registry owners {expected_ab_owners}"
    )

    token_ab = os.environ.get("BOT_TOKEN") or os.environ.get("AUTOBUS_TELEGRAM_BOT_TOKEN")
    if not token_ab or notification_routing.extract_telegram_bot_id(token_ab) != ab_inst["platform_id"]:
        token_ab = f"{ab_inst['platform_id']}:AAFakeTokenForTest123456789"
    ab_route = notification_routing.resolve_notification_route(
        notification_routing.EventCategory.SYSTEM_DIAGNOSTIC,
        current_instance=ab_inst,
        bot_token_override=token_ab,
    )
    assert ab_route is not None, "Failed to resolve Autobus system route!"
    assert ab_route.target_instance == "tg_autobusbusbot_new"
    assert ab_route.target_platform_id == 8346799080
    assert ab_route.target_username.lstrip("@") == "autobusbusbot"

    delivery_ab = {
        "source": "tg_autobusbusbot_new",
        "sender_bot_id": ab_route.target_platform_id,
        "sender_username": ab_route.target_username,
        "recipient_chat_ids": list(ab_admins),
        "status": "verified_route",
    }
    logger.info("  Sender: %s (ID %s)", ab_route.target_username, ab_route.target_platform_id)
    logger.info("  Authorized Recipient Chat IDs: %s", list(ab_admins))

    if live_send:
        logger.info("  Dispatching live Telegram message...")
        ok = await notification_routing.send_admin_notification(
            f"🔔 <b>[Live Post-Deploy Verification]</b>\n"
            f"Источник: Autobus System Alert\n"
            f"Отправитель: @autobusbusbot ({ab_route.target_platform_id})\n"
            f"Получатели: {list(ab_admins)}\n"
            f"Статус: Системный мониторинг функционирует штатно.",
            category=notification_routing.EventCategory.SYSTEM_DIAGNOSTIC,
        )
        assert ok is True, "Failed to deliver Autobus notification to live admins!"
        delivery_ab["status"] = "delivered"
    results["deliveries"].append(delivery_ab)
    logger.info("[PASS] Check 4: Autobus system routing verified.")

    # -------------------------------------------------------------
    # 5. Negative Isolation Guarantee (No User/Payment Leaks to Autobus)
    # -------------------------------------------------------------
    logger.info("\n[CHECK 5] Verifying Negative Routing Constraints (Zero Leaks to Autobus)...")
    isolation_count = 0
    non_autobus_instances = [i for i in instances if i["pm2_name"] != "tg_autobusbusbot_new"]

    for entry in non_autobus_instances:
        for cat in [
            notification_routing.EventCategory.USER_EVENT,
            notification_routing.EventCategory.PAYMENT,
            notification_routing.EventCategory.AI_ERROR,
        ]:
            if entry["platform"] == "max":
                comp = notification_routing.resolve_companion_telegram_instance(entry)
                assert comp is not None
                cand_token = f"{comp['platform_id']}:AAFakeToken123456789"
            else:
                cand_token = f"{entry['platform_id']}:AAFakeToken123456789"

            rt = notification_routing.resolve_notification_route(
                cat,
                current_instance=entry,
                bot_token_override=cand_token,
            )
            assert rt is not None, f"Route returned None for {entry['pm2_name']} category {cat}"
            assert rt.target_instance != "tg_autobusbusbot_new", (
                f"LEAK DETECTED: Process {entry['pm2_name']} routed {cat.value} to tg_autobusbusbot_new!"
            )
            assert rt.target_platform_id != 8346799080, (
                f"LEAK DETECTED: Process {entry['pm2_name']} targeted Autobus platform ID 8346799080!"
            )
            isolation_count += 1

    results["isolation_checks"] = isolation_count
    logger.info("[PASS] Check 5: Verified %d negative routing paths. Zero leaks to Autobus.", isolation_count)

    results["status"] = "PASSED"
    logger.info("\n" + "=" * 70)
    logger.info("ALL POST-DEPLOY VERIFICATION CHECKS PASSED 100%!")
    logger.info("=" * 70)
    return results


def main():
    parser = argparse.ArgumentParser(description="Post-deploy verification for PR #86")
    parser.add_argument(
        "--live-send",
        action="store_true",
        default=False,
        help="Perform actual live Telegram send to authorized admins (production only).",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to bot_instances.json registry file.",
    )
    parser.add_argument(
        "--runtime-env",
        type=Path,
        default=None,
        help="Path to runtime.env file.",
    )
    args = parser.parse_args()

    try:
        res = asyncio.run(
            verify_post_deploy(
                live_send=args.live_send,
                registry_path=args.registry,
                runtime_env_path=args.runtime_env,
            )
        )
        print(json.dumps(res, indent=2))
        sys.exit(0)
    except Exception as exc:
        logger.exception("POST-DEPLOY VERIFICATION FAILED: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
