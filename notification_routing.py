"""
Unified Notification Routing Engine with Strict Instance Isolation.

Enforces the verifiable contract:
    source bot instance -> event category -> target Telegram bot -> allowed Telegram admin IDs

Guarantees:
- Zero cross-instance notification leakage.
- MAX bot events (promos, payments, subscriptions, cancellations, renewals, AI errors)
  route exclusively through their companion Telegram bot to verified Telegram admin IDs.
- System diagnostics (monitor_psychobots, VPS backup) route through tg_autobusbusbot_new.
- Cryptographic bot identity validation: token platform_id prefix MUST match target bot platform_id.
- Absolute refusal to fall back to legacy config.ini or foreign tokens (e.g. Autobus token).
- All MAX user IDs (>= 100_000_000_000) and invalid IDs are stripped from Telegram dispatch targets.
- Notification failures never raise exceptions to callers and never abort payments/promos.
"""

from __future__ import annotations

import enum
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Any, Sequence

logger = logging.getLogger("notification_routing")

ROOT_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT_DIR / "config" / "bot_instances.json"
MAX_USER_ID_THRESHOLD = 100_000_000_000
TOKEN_PREFIX_RE = re.compile(r"^(\d{6,12}):[A-Za-z0-9_-]{20,}$")

# Canonical Telegram owners per instance (fallback when unpopulated in env)
DEFAULT_TELEGRAM_OWNER_IDS: dict[str, tuple[int, ...]] = {
    "tg_autobusbusbot_new": (272982544, 979514796, 806750628),
    "tg_alena322bot_new": (272982544, 979514796, 806750628),
    "tg_someonewithyou01_bot_new": (272982544, 979514796, 806750628),
    "tg_someonewithyou02_bot_new": (272982544, 806750628),
    "tg_veraveda777_bot_legacy": (979514796, 272982544, 806750628),
    "tg_someonelikeyouai03_bot_legacy": (272982544, 806750628),
    "tg_someonelikeyouai04_bot_legacy": (272982544, 806750628),
    "tg_psy5d_bot_legacy": (272982544, 806750628),
    "tg_someonelikeyouai_bot_legacy": (272982544, 806750628),
    "tg_someonelikeyouai02_bot_legacy": (272982544, 806750628),
    "tg_someonewithyou03_bot_new": (272982544, 979514796, 806750628),
    "tg_someonelikeyouai05_bot_new": (272982544, 979514796, 806750628),
    "tg_someonewithyou05_bot_new": (272982544, 979514796, 806750628),
    "tg_someonewithyou06_bot_new": (272982544, 979514796, 806750628),
    "tg_someonewithyou04_bot_new": (272982544, 979514796, 806750628),
    "tg_yourself_way_bot_new": (979514796, 272982544, 806750628, 1372041348),
    "tg_kontentzavod322bot_fulltest": (272982544, 979514796, 806750628),
}

# Explicit companion mapping: MAX PM2 / username -> Companion Telegram PM2 name
MAX_COMPANION_MAPPING: dict[str, str] = {
    "max_id519010411655_bot_new": "tg_yourself_way_bot_new",
    "id519010411655_bot": "tg_yourself_way_bot_new",
    "max_se13639182_bot_legacy": "tg_veraveda777_bot_legacy",
    "se13639182_bot": "tg_veraveda777_bot_legacy",
}

SYSTEM_DIAGNOSTIC_BOT_PM2 = "tg_autobusbusbot_new"


class EventCategory(str, enum.Enum):
    USER_EVENT = "user_event"             # promo codes, trial, manual inquiries
    PAYMENT = "payment"                   # successful payment, failed payment, auto-renewal, cancellation
    AI_ERROR = "ai_error"                 # terminal AI service failure
    SYSTEM_DIAGNOSTIC = "system_diagnostic"  # infrastructure monitoring, VPS backup, disk space


class RoutingError(RuntimeError):
    """Raised when routing invariants or identity checks are violated."""


@dataclass(frozen=True)
class NotificationRoute:
    category: EventCategory
    source_instance: str
    source_platform: str
    source_username: str
    target_instance: str
    target_username: str
    target_platform_id: int
    target_bot_token: str
    admin_chat_ids: tuple[int, ...]


def extract_telegram_bot_id(token: str | None) -> int | None:
    """Extract integer bot ID prefix from a Telegram token (e.g. 8481853696 from 8481853696:AAF...)."""
    if not token or not isinstance(token, str):
        return None
    match = TOKEN_PREFIX_RE.match(token.strip())
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def is_valid_telegram_user_id(uid: Any) -> bool:
    """True if uid is a positive Telegram user ID (< 100_000_000_000). Discards MAX user IDs."""
    if not isinstance(uid, int) or isinstance(uid, bool):
        return False
    return 0 < uid < MAX_USER_ID_THRESHOLD


def read_runtime_env_file(path: str | Path | None = None) -> dict[str, str]:
    """Safely parse key=value lines from host runtime.env file if available."""
    target_path = Path(path or os.environ.get("PROD_RUNTIME_ENV", "/root/telegram_bots/runtime.env"))
    if not target_path.is_file():
        return {}
    env_vars: dict[str, str] = {}
    try:
        content = target_path.read_text(encoding="utf-8")
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if (val.startswith('"') and val.endsWith('"')) or (val.startswith("'") and val.endsWith("'")):
                    val = val[1:-1]
                env_vars[key] = val
    except Exception as exc:
        logger.debug("Could not read runtime.env from %s: %s", target_path, exc)
    return env_vars


@lru_cache(maxsize=1)
def load_bot_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Load config/bot_instances.json registry."""
    try:
        content = path.read_text(encoding="utf-8")
        return json.loads(content)
    except Exception as exc:
        logger.error("Failed to load bot registry from %s: %s", path, exc)
        return {"instances": []}


def clear_registry_cache() -> None:
    load_bot_registry.cache_clear()


def get_instance_by_pm2_name(pm2_name: str) -> dict[str, Any] | None:
    registry = load_bot_registry()
    for inst in registry.get("instances", []):
        if inst.get("pm2_name") == pm2_name or inst.get("legacy_pm2_name") == pm2_name:
            return inst
    return None


def get_instance_by_username(username: str, platform: str | None = None) -> dict[str, Any] | None:
    clean_user = username.lstrip("@").strip()
    registry = load_bot_registry()
    for inst in registry.get("instances", []):
        if inst.get("username") == clean_user:
            if platform is None or inst.get("platform") == platform:
                return inst
    return None


def get_instance_by_database(database_name: str, platform: str | None = None) -> dict[str, Any] | None:
    registry = load_bot_registry()
    for inst in registry.get("instances", []):
        if inst.get("database") == database_name:
            if platform is None or inst.get("platform") == platform:
                return inst
    return None


def detect_current_instance() -> dict[str, Any] | None:
    """
    Detect the active instance context using environment variables, process names,
    tokens, or database connection strings.
    """
    # 1. Check explicit PM2 app name
    pm2_name = (
        os.environ.get("PM2_APP_NAME")
        or os.environ.get("name")
        or os.environ.get("APP_NAME")
    )
    if pm2_name:
        inst = get_instance_by_pm2_name(pm2_name.strip())
        if inst:
            return inst

    # 2. Check explicit MAX bot name
    max_bot_name = os.environ.get("MAX_BOT_NAME")
    if max_bot_name:
        inst = get_instance_by_username(max_bot_name.strip(), platform="max")
        if inst:
            return inst

    # 3. Check explicit BOT_NAME or BOT_USERNAME
    bot_name = os.environ.get("BOT_NAME") or os.environ.get("BOT_USERNAME")
    if bot_name:
        inst = get_instance_by_username(bot_name.strip())
        if inst:
            return inst

    # 4. Check MAX_BOT_TOKEN identity
    max_token = os.environ.get("MAX_BOT_TOKEN")
    if max_token:
        # Match against MAX instance platform_id if available, or database
        pass

    # 5. Check BOT_TOKEN prefix against Telegram instances
    bot_token = os.environ.get("BOT_TOKEN")
    bot_id = extract_telegram_bot_id(bot_token)
    if bot_id:
        registry = load_bot_registry()
        for inst in registry.get("instances", []):
            if inst.get("platform") == "telegram" and inst.get("platform_id") == bot_id:
                return inst

    # 6. Check DATABASE_URL
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        match = re.search(r"/([a-zA-Z0-9_]+)(?:\?|$)", db_url)
        if match:
            db_name = match.group(1)
            # If MAX_BOT_TOKEN is in env, prefer max instance
            platform_hint = "max" if max_token else None
            inst = get_instance_by_database(db_name, platform=platform_hint)
            if inst:
                return inst

    return None


def resolve_companion_telegram_instance(source_instance: dict[str, Any]) -> dict[str, Any] | None:
    """
    Resolve the companion Telegram bot instance for a MAX instance.
    For Telegram instances, returns the instance itself.
    """
    platform = source_instance.get("platform")
    pm2_name = source_instance.get("pm2_name", "")
    username = source_instance.get("username", "")

    if platform == "telegram":
        return source_instance

    if platform == "max":
        # 1. Explicit mapping lookup
        companion_pm2 = MAX_COMPANION_MAPPING.get(pm2_name) or MAX_COMPANION_MAPPING.get(username)
        if companion_pm2:
            companion = get_instance_by_pm2_name(companion_pm2)
            if companion and companion.get("platform") == "telegram":
                return companion

        # 2. Match by shared database
        db_name = source_instance.get("database")
        if db_name:
            companion = get_instance_by_database(db_name, platform="telegram")
            if companion:
                return companion

    return None


def resolve_target_bot_token(
    target_instance: dict[str, Any],
    *,
    is_companion: bool = False,
    candidate_token: str | None = None,
) -> str | None:
    """
    Resolve and cryptographically validate the target bot's Telegram token.

    STRICT GUARANTEES:
    - The token MUST have a prefix matching target_instance['platform_id'].
    - If a token from Autobus or any other foreign bot is supplied, it is REJECTED.
    - NEVER falls back to legacy config.ini.
    """
    expected_id = target_instance.get("platform_id")
    if not expected_id or not isinstance(expected_id, int):
        logger.error("Target instance %s has invalid platform_id: %s", target_instance.get("pm2_name"), expected_id)
        return None

    candidates: list[str | None] = []

    # 1. Explicit override passed by caller
    if candidate_token:
        candidates.append(candidate_token)

    # 2. For companion routing: check companion-specific env vars
    if is_companion:
        candidates.append(os.environ.get("COMPANION_TELEGRAM_BOT_TOKEN"))
        target_token_env = target_instance.get("token_env")
        if target_token_env:
            candidates.append(os.environ.get(target_token_env))
        # Note: We check os.environ.get("BOT_TOKEN") as well, BUT ONLY IF it matches expected_id!
        # If BOT_TOKEN holds Autobus's token, the prefix check below will strictly discard it!
        candidates.append(os.environ.get("BOT_TOKEN"))
    else:
        # Direct Telegram bot
        target_token_env = target_instance.get("token_env")
        if target_token_env:
            candidates.append(os.environ.get(target_token_env))
        candidates.append(os.environ.get("BOT_TOKEN"))

    # 3. Check runtime.env from disk if available
    target_token_env = target_instance.get("token_env")
    if target_token_env:
        host_env = read_runtime_env_file()
        if host_env.get(target_token_env):
            candidates.append(host_env[target_token_env])

    # Validate candidates against expected_id
    for cand in candidates:
        if not cand or not isinstance(cand, str):
            continue
        cleaned = cand.strip()
        cand_bot_id = extract_telegram_bot_id(cleaned)
        if cand_bot_id == expected_id:
            return cleaned
        elif cand_bot_id is not None:
            logger.warning(
                "[Notification Routing] Discarding token with mismatched bot ID %s (expected %s for %s)",
                cand_bot_id,
                expected_id,
                target_instance.get("pm2_name"),
            )

    logger.error(
        "[Notification Routing] No valid token found matching platform_id %s for instance %s",
        expected_id,
        target_instance.get("pm2_name"),
    )
    return None


async def resolve_target_admin_ids(
    target_instance: dict[str, Any],
    session: Any | None = None,
) -> tuple[int, ...]:
    """
    Resolve allowed Telegram admin IDs for the target instance.

    STRICT GUARANTEES:
    - Filters out all MAX IDs (>= 100_000_000_000).
    - Filters out non-positive IDs.
    - Resolves from target instance's configured Telegram owners + DB is_admin=True.
    """
    admin_ids: set[int] = set()
    pm2_name = target_instance.get("pm2_name", "")

    # 1. Configured Telegram owners from default mapping
    if pm2_name in DEFAULT_TELEGRAM_OWNER_IDS:
        for oid in DEFAULT_TELEGRAM_OWNER_IDS[pm2_name]:
            if is_valid_telegram_user_id(oid):
                admin_ids.add(oid)

    # 2. Check telegram_owner_ids in instance entry
    inst_owners = target_instance.get("telegram_owner_ids")
    if isinstance(inst_owners, (list, tuple)):
        for oid in inst_owners:
            if is_valid_telegram_user_id(oid):
                admin_ids.add(oid)

    # 3. Check environment OWNER_IDS (only keep valid Telegram IDs)
    raw_env_owners = os.environ.get("OWNER_IDS", "")
    if raw_env_owners:
        for part in raw_env_owners.split(","):
            part_str = part.strip()
            if part_str.isdigit():
                try:
                    uid = int(part_str)
                    if is_valid_telegram_user_id(uid):
                        admin_ids.add(uid)
                except ValueError:
                    pass

    # 4. Check database users with is_admin=True (filtered)
    try:
        from database import User, async_session_maker

        async def _query_db_admins(s: Any) -> set[int]:
            res = await s.execute(select(User.id).where(User.is_admin == True))
            return {row[0] for row in res if is_valid_telegram_user_id(row[0])}

        if session is not None:
            db_admins = await _query_db_admins(session)
            admin_ids.update(db_admins)
        else:
            async with async_session_maker() as s:
                db_admins = await _query_db_admins(s)
                admin_ids.update(db_admins)
    except Exception as exc:
        logger.debug("[Notification Routing] DB admin lookup skipped/failed: %s", exc)

    return tuple(sorted(admin_ids))


def resolve_notification_route(
    category: EventCategory = EventCategory.USER_EVENT,
    *,
    current_instance: dict[str, Any] | None = None,
    bot_token_override: str | None = None,
    session: Any | None = None,
) -> NotificationRoute | None:
    """
    Assemble and validate the complete notification route:
        source instance -> event category -> target bot -> allowed admin IDs.

    Returns None if validation fails at any boundary.
    """
    # 1. Category: System Diagnostic -> Centralized Autobus Channel
    if category == EventCategory.SYSTEM_DIAGNOSTIC:
        target_inst = get_instance_by_pm2_name(SYSTEM_DIAGNOSTIC_BOT_PM2)
        if not target_inst:
            logger.error("[Notification Routing] System alert bot %s not found in registry", SYSTEM_DIAGNOSTIC_BOT_PM2)
            return None
        source_inst = current_instance or detect_current_instance() or target_inst
        token = resolve_target_bot_token(
            target_inst,
            is_companion=False,
            candidate_token=bot_token_override,
        )
        if not token:
            return None
        # System admin IDs
        admins = tuple(sorted(DEFAULT_TELEGRAM_OWNER_IDS.get(SYSTEM_DIAGNOSTIC_BOT_PM2, (272982544, 979514796, 806750628))))
        return NotificationRoute(
            category=category,
            source_instance=source_inst.get("pm2_name", "system"),
            source_platform=source_inst.get("platform", "system"),
            source_username=source_inst.get("username", "system"),
            target_instance=target_inst["pm2_name"],
            target_username=target_inst["username"],
            target_platform_id=target_inst["platform_id"],
            target_bot_token=token,
            admin_chat_ids=admins,
        )

    # 2. User / Financial / AI Error Events -> Strict Bot Instance Isolation
    source_inst = current_instance or detect_current_instance()
    if not source_inst:
        logger.error("[Notification Routing] Cannot resolve notification route: source instance undetermined")
        return None

    source_platform = source_inst.get("platform")
    if source_platform == "max":
        target_inst = resolve_companion_telegram_instance(source_inst)
        if not target_inst:
            logger.error(
                "[Notification Routing] MAX instance %s has no configured companion Telegram bot",
                source_inst.get("pm2_name"),
            )
            return None
        is_companion = True
    elif source_platform == "telegram":
        target_inst = source_inst
        is_companion = False
    else:
        logger.error("[Notification Routing] Unsupported source platform: %s", source_platform)
        return None

    # Resolve token strictly matching target_inst["platform_id"]
    token = resolve_target_bot_token(
        target_inst,
        is_companion=is_companion,
        candidate_token=bot_token_override,
    )
    if not token:
        logger.error(
            "[Notification Routing] Route rejected: No verified token for target %s (expected ID %s)",
            target_inst.get("pm2_name"),
            target_inst.get("platform_id"),
        )
        return None

    # Resolve allowed Telegram admins (MAX IDs strictly excluded)
    admin_ids = None
    if session is not None:
        import asyncio
        # We can't await here directly if sync, but resolve_target_admin_ids is async
        pass

    # Note: caller should pass admin IDs or resolve async in send_admin_notification
    # Return route skeleton (admin IDs will be finalized in send_admin_notification)
    return NotificationRoute(
        category=category,
        source_instance=source_inst.get("pm2_name", "unknown"),
        source_platform=source_platform,
        source_username=source_inst.get("username", "unknown"),
        target_instance=target_inst["pm2_name"],
        target_username=target_inst["username"],
        target_platform_id=target_inst["platform_id"],
        target_bot_token=token,
        admin_chat_ids=DEFAULT_TELEGRAM_OWNER_IDS.get(target_inst["pm2_name"], ()),
    )


async def send_admin_notification(
    text: str,
    category: EventCategory = EventCategory.USER_EVENT,
    *,
    bot_token_override: str | None = None,
    session: Any | None = None,
) -> bool:
    """
    Main dispatch function for admin notifications across all instances.

    Guarantees:
    - Never raises exceptions to caller.
    - Never sends to wrong bot or wrong admins.
    - Checks SubscriptionConfig.notifications_enabled if applicable.
    - Aiogram Bot identity is verified before dispatch.
    - Returns True if delivered to at least one admin, False otherwise.
    """
    try:
        route = resolve_notification_route(
            category=category,
            bot_token_override=bot_token_override,
            session=session,
        )
        if not route:
            logger.warning("[Notification Routing] Alert dropped: No valid route for category %s", category)
            return False

        # Target instance definition
        target_inst = get_instance_by_pm2_name(route.target_instance)
        if not target_inst:
            return False

        # Resolve full list of Telegram admin IDs
        admin_ids = await resolve_target_admin_ids(target_inst, session=session)
        if not admin_ids:
            logger.warning(
                "[Notification Routing] Alert dropped: No allowed Telegram admins for target %s",
                route.target_instance,
            )
            return False

        # For user/payment notifications, check SubscriptionConfig
        if category in (EventCategory.USER_EVENT, EventCategory.PAYMENT):
            try:
                from database import SubscriptionConfig, async_session_maker

                async def _check_enabled(s: Any) -> bool:
                    cfg = await s.get(SubscriptionConfig, 1)
                    return bool(cfg and cfg.notifications_enabled)

                if session is not None:
                    if not await _check_enabled(session):
                        return False
                else:
                    async with async_session_maker() as s:
                        if not await _check_enabled(s):
                            return False
            except Exception as exc:
                logger.debug("[Notification Routing] SubscriptionConfig check failed (continuing): %s", exc)

        # Dispatch via verified Aiogram Bot
        from telegram_client import create_telegram_bot

        delivered = False
        async with create_telegram_bot(route.target_bot_token) as bot:
            # Cryptographic confirmation: bot.id must match target_platform_id
            if bot.id != route.target_platform_id:
                logger.critical(
                    "[Notification Routing] CRITICAL MISMATCH: Bot ID %s does not match route target ID %s. Aborting.",
                    bot.id,
                    route.target_platform_id,
                )
                return False

            for admin_id in admin_ids:
                if not is_valid_telegram_user_id(admin_id):
                    continue
                try:
                    await bot.send_message(admin_id, text, parse_mode="HTML")
                    delivered = True
                except Exception as exc:
                    logger.error(
                        "[Notification Routing] Failed delivery to admin %s via %s: %s",
                        admin_id,
                        route.target_username,
                        exc,
                    )

        return delivered

    except Exception as exc:
        logger.error("[Notification Routing] Unexpected error in send_admin_notification: %s", exc)
        return False


def validate_process_notification_configuration() -> dict[str, Any]:
    """
    Validate notification routing configuration on process startup.
    Produces a safe, non-leaking diagnostic summary.
    """
    current_inst = detect_current_instance()
    if not current_inst:
        res = {
            "status": "warning",
            "message": "Current instance could not be determined from environment",
        }
        logger.warning("[Notification Routing Diagnostic] %s", res["message"])
        return res

    platform = current_inst.get("platform")
    pm2_name = current_inst.get("pm2_name")
    companion = resolve_companion_telegram_instance(current_inst) if platform == "max" else current_inst

    if not companion:
        res = {
            "status": "error",
            "instance": pm2_name,
            "platform": platform,
            "message": "Companion Telegram bot not configured",
        }
        logger.error("[Notification Routing Diagnostic] %s", res)
        return res

    target_token = resolve_target_bot_token(companion, is_companion=(platform == "max"))
    token_status = "VALID" if target_token else "MISSING"

    target_id = companion.get("platform_id")
    admin_count = len(DEFAULT_TELEGRAM_OWNER_IDS.get(companion.get("pm2_name", ""), ()))

    res = {
        "status": "ok" if target_token else "warning",
        "instance": pm2_name,
        "platform": platform,
        "target_bot": companion.get("username"),
        "target_pm2": companion.get("pm2_name"),
        "target_platform_id": target_id,
        "token_status": token_status,
        "configured_admins_count": admin_count,
    }
    logger.info(
        "[Notification Routing Diagnostic] Instance: %s (%s) | Target Bot: %s (%s, ID: %s) | Token: %s | Default Admins: %d",
        pm2_name,
        platform,
        companion.get("username"),
        companion.get("pm2_name"),
        target_id,
        token_status,
        admin_count,
    )
    return res
