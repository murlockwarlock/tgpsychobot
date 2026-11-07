#!/usr/bin/env python3
"""
Post-Deploy Verification Script (PR #86).

Validates process isolation, routing, and identity for all 18 managed instances.
Supports:
  --dry-run (default in CI/local): tests registry, isolated environment switching, dynamic admin resolution,
            sender/recipient identity recording, and negative isolation constraints without live Telegram calls.
  --production: strictly requires live PM2 to be running, all 18 managed processes to be online,
                and forbids any mock or placeholder tokens (FAILS instead of SKIP).
  --live-send: explicitly verifies bot identity via getMe and sends safe verification notifications
               to authorized admin chat_ids (requires production mode, only executed on production
               upon explicit operator authorization).
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from contextlib import contextmanager
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
        logger.warning("PM2 jlist not available in this environment.")
        return {}


@contextmanager
def isolated_instance_env(
    instance_name: str,
    pm2_apps: dict[str, Any],
    *,
    is_production: bool = False,
    runtime_env: dict[str, str] | None = None,
    mock_token: str | None = None,
):
    """
    Context manager to execute checks inside the strictly isolated environment of a specific instance.
    Prevents any cross-contamination of environment variables between bot checks.
    """
    saved_env = os.environ.copy()
    try:
        # 1. Clean out all bot-specific variables
        for var in [
            "BOT_INSTANCE_ID",
            "BOT_TOKEN",
            "COMPANION_TELEGRAM_BOT_TOKEN",
            "DATABASE_URL",
            "MAX_BOT_TOKEN",
            "OWNER_IDS",
            "AUTOBUS_TELEGRAM_BOT_TOKEN",
        ]:
            os.environ.pop(var, None)

        # 2. Set authoritative BOT_INSTANCE_ID
        os.environ["BOT_INSTANCE_ID"] = instance_name

        # 3. Populate from PM2 process memory if available
        if pm2_apps and instance_name in pm2_apps:
            proc_env = pm2_apps[instance_name].get("pm2_env", {}).get("env", {})
            for k in [
                "BOT_TOKEN",
                "COMPANION_TELEGRAM_BOT_TOKEN",
                "DATABASE_URL",
                "MAX_BOT_TOKEN",
                "OWNER_IDS",
                "AUTOBUS_TELEGRAM_BOT_TOKEN",
            ]:
                if proc_env.get(k):
                    os.environ[k] = str(proc_env[k]).strip()
        elif runtime_env:
            for k, v in runtime_env.items():
                if v:
                    os.environ[k] = str(v).strip()

        # 4. In dry-run / non-production mode, if token is missing, allow mock_token
        if not is_production and mock_token:
            if instance_name.startswith("max_"):
                if not os.environ.get("COMPANION_TELEGRAM_BOT_TOKEN"):
                    os.environ["COMPANION_TELEGRAM_BOT_TOKEN"] = mock_token
            else:
                if not os.environ.get("BOT_TOKEN"):
                    os.environ["BOT_TOKEN"] = mock_token

        yield
    finally:
        os.environ.clear()
        os.environ.update(saved_env)


async def verify_post_deploy(
    *,
    live_send: bool = False,
    production: bool = False,
    registry_path: Path | None = None,
    runtime_env_path: Path | None = None,
) -> dict[str, Any]:
    """
    Execute comprehensive post-deploy verification.
    Returns structured results dict. Raises AssertionError / RuntimeError on any failure.
    """
    is_production = production or live_send or (os.environ.get("PRODUCTION") == "1")

    logger.info("=" * 70)
    logger.info("RUNNING POST-DEPLOY VERIFICATION (PR #86)")
    logger.info("Production Mode: %s", "YES (Strict Fail-Closed)" if is_production else "NO (Dry-Run / CI)")
    logger.info("Live Send:       %s", "YES (Active Telegram Dispatch)" if live_send else "NO (Safe Simulation)")
    logger.info("=" * 70)

    reg = notification_routing.load_bot_registry(registry_path or notification_routing.REGISTRY_PATH)
    instances = reg.get("instances", [])
    inst_by_name = {i["pm2_name"]: i for i in instances}

    runtime_env = None
    if runtime_env_path and runtime_env_path.is_file():
        runtime_env = notification_routing.read_runtime_env_file(runtime_env_path)

    results: dict[str, Any] = {
        "status": "pending",
        "production_mode": is_production,
        "live_send": live_send,
        "instances_checked": len(instances),
        "deliveries": [],
        "isolation_checks": 0,
        "errors": [],
    }

    # -------------------------------------------------------------
    # 1. PM2 In-Memory Env Inspection
    # -------------------------------------------------------------
    pm2_apps = inspect_pm2_processes()
    if is_production:
        if not pm2_apps:
            raise RuntimeError("PRODUCTION VERIFICATION FAILED: PM2 is not running or returned no processes!")

        managed_instances = [i for i in instances if i.get("deploy_managed")]
        logger.info("[CHECK 1] Strictly verifying %d managed PM2 processes in production mode...", len(managed_instances))
        for inst in managed_instances:
            name = inst["pm2_name"]
            if name not in pm2_apps:
                raise RuntimeError(f"PRODUCTION VERIFICATION FAILED: Mandatory process {name} is missing in PM2!")
            app_info = pm2_apps[name]
            status = app_info.get("pm2_env", {}).get("status")
            if status != "online":
                raise RuntimeError(f"PRODUCTION VERIFICATION FAILED: Process {name} is not online (status={status})!")
            proc_env = app_info.get("pm2_env", {}).get("env", {})
            iid = proc_env.get("BOT_INSTANCE_ID")
            if iid != name:
                raise RuntimeError(
                    f"PRODUCTION VERIFICATION FAILED: Process {name} has invalid BOT_INSTANCE_ID: {iid!r} (expected {name!r})!"
                )

            # Check tokens strictly in PM2 memory
            if inst["platform"] == "telegram":
                tok = proc_env.get("BOT_TOKEN")
                if not tok or "AAFakeToken" in tok:
                    raise RuntimeError(f"PRODUCTION VERIFICATION FAILED: Missing or fake BOT_TOKEN for {name} in PM2!")
                exp_id = inst["platform_id"]
                if notification_routing.extract_telegram_bot_id(tok) != exp_id:
                    raise RuntimeError(
                        f"PRODUCTION VERIFICATION FAILED: BOT_TOKEN for {name} has wrong ID prefix (expected {exp_id})!"
                    )
            elif inst["platform"] == "max":
                if not proc_env.get("MAX_BOT_TOKEN"):
                    raise RuntimeError(f"PRODUCTION VERIFICATION FAILED: Missing MAX_BOT_TOKEN for {name} in PM2!")
                comp_tok = proc_env.get("COMPANION_TELEGRAM_BOT_TOKEN")
                if not comp_tok or "AAFakeToken" in comp_tok:
                    raise RuntimeError(
                        f"PRODUCTION VERIFICATION FAILED: Missing or fake COMPANION_TELEGRAM_BOT_TOKEN for {name} in PM2!"
                    )
                comp_inst = notification_routing.resolve_companion_telegram_instance(inst)
                if comp_inst:
                    exp_comp_id = comp_inst["platform_id"]
                    if notification_routing.extract_telegram_bot_id(comp_tok) != exp_comp_id:
                        raise RuntimeError(
                            f"PRODUCTION VERIFICATION FAILED: Companion token for {name} has wrong ID prefix (expected {exp_comp_id})!"
                        )
        logger.info("[PASS] Check 1: All %d production PM2 processes online with verified memory environments.", len(managed_instances))
    elif pm2_apps:
        logger.info("[PASS] Check 1: PM2 processes inspected in dry-run mode.")
    else:
        logger.info("[SKIP] Check 1: PM2 not present, skipping live in-memory inspection (dry-run mode).")

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

    mock_yw_tok = f"{yw_comp['platform_id']}:AAFakeTokenForTest123456789" if not is_production else None

    with isolated_instance_env("max_id519010411655_bot_new", pm2_apps, is_production=is_production, runtime_env=runtime_env, mock_token=mock_yw_tok):
        active_tok = os.environ.get("COMPANION_TELEGRAM_BOT_TOKEN")
        if is_production:
            if not active_tok or "AAFakeToken" in active_tok:
                raise RuntimeError("PRODUCTION VERIFICATION FAILED: Missing real COMPANION_TELEGRAM_BOT_TOKEN for Yourself Way!")
            if notification_routing.extract_telegram_bot_id(active_tok) != 8481853696:
                raise RuntimeError(f"PRODUCTION VERIFICATION FAILED: Wrong token ID prefix for Yourself Way: {active_tok[:10]}")

        # Resolve route inside the isolated environment without passing current_instance,
        # proving that detect_current_instance() correctly resolves BOT_INSTANCE_ID
        yw_route = notification_routing.resolve_notification_route(notification_routing.EventCategory.USER_EVENT)
        assert yw_route is not None, "Failed to resolve Yourself Way route from active environment!"
        assert yw_route.target_instance == "tg_yourself_way_bot_new"
        assert yw_route.target_platform_id == 8481853696
        assert yw_route.target_username.lstrip("@") == "yourself_way_bot"

        # Dynamically resolve allowed Telegram admin IDs from registry & DB
        yw_admins = await notification_routing.resolve_target_admin_ids(yw_comp, current_instance=yw_max)
        expected_yw_owners = set(yw_comp.get("telegram_owner_ids", []))
        assert expected_yw_owners.issubset(set(yw_admins)), (
            f"Resolved admin IDs {yw_admins} do not contain expected registry owners {expected_yw_owners}"
        )
        for aid in yw_admins:
            assert aid < 100_000_000_000, f"Invalid MAX ID leaked into Telegram admins: {aid}"
            assert aid > 0, f"Invalid negative/zero admin ID: {aid}"

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
            logger.info("  Confirming live Telegram getMe identity before dispatch...")
            identity = await notification_routing.verify_bot_identity_live(
                yw_route.target_bot_token,
                expected_username=yw_route.target_username,
                expected_id=yw_route.target_platform_id,
                use_cache=False,
            )
            assert identity["verified"] is True, f"Live getMe verification failed for Yourself Way: {identity}"
            assert identity["id"] == 8481853696, f"Live getMe ID mismatch: {identity['id']}"

            logger.info("  Dispatching live Telegram message to verified chat IDs: %s", list(yw_admins))
            ok = await notification_routing.send_admin_notification(
                f"🔔 <b>[Live Post-Deploy Verification]</b>\n"
                f"Источник: MAX id519010411655_bot\n"
                f"Отправитель: @yourself_way_bot ({yw_route.target_platform_id})\n"
                f"Получатели: {list(yw_admins)}\n"
                f"Статус: Изоляция подтверждена на live production.",
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

    mock_vv_tok = f"{vv_comp['platform_id']}:AAFakeTokenForTest123456789" if not is_production else None

    with isolated_instance_env("max_se13639182_bot_legacy", pm2_apps, is_production=is_production, runtime_env=runtime_env, mock_token=mock_vv_tok):
        active_tok = os.environ.get("COMPANION_TELEGRAM_BOT_TOKEN")
        if is_production:
            if not active_tok or "AAFakeToken" in active_tok:
                raise RuntimeError("PRODUCTION VERIFICATION FAILED: Missing real COMPANION_TELEGRAM_BOT_TOKEN for VeraVeda!")
            if notification_routing.extract_telegram_bot_id(active_tok) != 8224473896:
                raise RuntimeError(f"PRODUCTION VERIFICATION FAILED: Wrong token ID prefix for VeraVeda: {active_tok[:10]}")

        vv_route = notification_routing.resolve_notification_route(notification_routing.EventCategory.USER_EVENT)
        assert vv_route is not None, "Failed to resolve VeraVeda route from active environment!"
        assert vv_route.target_instance == "tg_veraveda777_bot_legacy"
        assert vv_route.target_platform_id == 8224473896
        assert vv_route.target_username.lstrip("@") == "veraveda777_bot"

        vv_admins = await notification_routing.resolve_target_admin_ids(vv_comp, current_instance=vv_max)
        expected_vv_owners = set(vv_comp.get("telegram_owner_ids", []))
        assert expected_vv_owners.issubset(set(vv_admins)), (
            f"Resolved admin IDs {vv_admins} do not contain expected registry owners {expected_vv_owners}"
        )
        assert 366022567 not in vv_admins, "CRITICAL ERROR: MAX platform ID 366022567 found in Telegram admins!"
        for aid in vv_admins:
            assert aid < 100_000_000_000, f"Invalid MAX ID leaked into Telegram admins: {aid}"
            assert aid > 0, f"Invalid negative/zero admin ID: {aid}"

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
            logger.info("  Confirming live Telegram getMe identity before dispatch...")
            identity = await notification_routing.verify_bot_identity_live(
                vv_route.target_bot_token,
                expected_username=vv_route.target_username,
                expected_id=vv_route.target_platform_id,
                use_cache=False,
            )
            assert identity["verified"] is True, f"Live getMe verification failed for VeraVeda: {identity}"
            assert identity["id"] == 8224473896, f"Live getMe ID mismatch: {identity['id']}"

            logger.info("  Dispatching live Telegram message to verified chat IDs: %s", list(vv_admins))
            ok = await notification_routing.send_admin_notification(
                f"🔔 <b>[Live Post-Deploy Verification]</b>\n"
                f"Источник: MAX se13639182_bot\n"
                f"Отправитель: @veraveda777_bot ({vv_route.target_platform_id})\n"
                f"Получатели: {list(vv_admins)}\n"
                f"Статус: Изоляция подтверждена на live production.",
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
    assert ab_inst["platform_id"] == 8346799080
    assert ab_inst["username"].lstrip("@") == "autobusbusbot"

    mock_ab_tok = f"{ab_inst['platform_id']}:AAFakeTokenForTest123456789" if not is_production else None

    with isolated_instance_env("tg_autobusbusbot_new", pm2_apps, is_production=is_production, runtime_env=runtime_env, mock_token=mock_ab_tok):
        active_tok = os.environ.get("BOT_TOKEN")
        if is_production:
            if not active_tok or "AAFakeToken" in active_tok:
                raise RuntimeError("PRODUCTION VERIFICATION FAILED: Missing real BOT_TOKEN for Autobus!")
            if notification_routing.extract_telegram_bot_id(active_tok) != 8346799080:
                raise RuntimeError(f"PRODUCTION VERIFICATION FAILED: Wrong token ID prefix for Autobus: {active_tok[:10]}")

        ab_route = notification_routing.resolve_notification_route(notification_routing.EventCategory.SYSTEM_DIAGNOSTIC)
        assert ab_route is not None, "Failed to resolve Autobus system route from active environment!"
        assert ab_route.target_instance == "tg_autobusbusbot_new"
        assert ab_route.target_platform_id == 8346799080
        assert ab_route.target_username.lstrip("@") == "autobusbusbot"

        ab_admins = await notification_routing.resolve_target_admin_ids(ab_inst, current_instance=ab_inst)
        expected_ab_owners = set(ab_inst.get("telegram_owner_ids", []))
        assert expected_ab_owners.issubset(set(ab_admins)), (
            f"Resolved admin IDs {ab_admins} do not contain expected registry owners {expected_ab_owners}"
        )

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
            logger.info("  Confirming live Telegram getMe identity before dispatch...")
            identity = await notification_routing.verify_bot_identity_live(
                ab_route.target_bot_token,
                expected_username=ab_route.target_username,
                expected_id=ab_route.target_platform_id,
                use_cache=False,
            )
            assert identity["verified"] is True, f"Live getMe verification failed for Autobus: {identity}"
            assert identity["id"] == 8346799080, f"Live getMe ID mismatch: {identity['id']}"

            logger.info("  Dispatching live Telegram message to verified chat IDs: %s", list(ab_admins))
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
        "--production",
        action="store_true",
        default=False,
        help="Strict production mode: requires live PM2, all 18 processes online, and real tokens (fails closed).",
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
                production=args.production,
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
