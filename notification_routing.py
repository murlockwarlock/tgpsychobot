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
from urllib.parse import urlparse

logger = logging.getLogger("notification_routing")

ROOT_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT_DIR / "config" / "bot_instances.json"
MAX_USER_ID_THRESHOLD = 100_000_000_000
TOKEN_PREFIX_RE = re.compile(r"^(\d{6,12}):[A-Za-z0-9_-]{20,}$")

SYSTEM_DIAGNOSTIC_BOT_PM2 = "tg_autobusbusbot_new"


class EventCategory(str, enum.Enum):
    USER_EVENT = "user_event"             # promo codes, trial, manual inquiries
    PAYMENT = "payment"                   # successful payment, failed payment, auto-renewal, cancellation
    AI_ERROR = "ai_error"                 # terminal AI service failure
    SYSTEM_DIAGNOSTIC = "system_diagnostic"  # infrastructure monitoring, VPS backup, disk space


class RoutingError(RuntimeError):
    """Raised when routing invariants or identity checks are violated."""


class ConfigurationError(RuntimeError):
    """Raised when process notification routing configuration is invalid or missing."""


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
    try:
        raw_path = path or os.environ.get("PROD_RUNTIME_ENV", "/root/telegram_bots/runtime.env")
        target_path = Path(raw_path)
        # Handle cases where parent directory has restrictive permissions (e.g. /root on non-root Linux CI runner)
        if not target_path.is_file():
            return {}
        content = target_path.read_text(encoding="utf-8")
        env_vars: dict[str, str] = {}
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                env_vars[key] = val
        return env_vars
    except (PermissionError, OSError, Exception) as exc:
        logger.debug("Could not read runtime.env from %s: %s", path, exc)
        return {}


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


def get_instance_by_id(instance_id: str) -> dict[str, Any] | None:
    registry = load_bot_registry()
    clean_id = instance_id.strip()
    for inst in registry.get("instances", []):
        if (
            inst.get("bot_instance_id") == clean_id
            or inst.get("pm2_name") == clean_id
            or inst.get("legacy_pm2_name") == clean_id
        ):
            return inst
    return None


def get_instance_by_pm2_name(pm2_name: str) -> dict[str, Any] | None:
    return get_instance_by_id(pm2_name)


def get_instance_by_username(username: str, platform: str | None = None) -> dict[str, Any] | None:
    clean_user = username.lstrip("@").strip()
    registry = load_bot_registry()
    for inst in registry.get("instances", []):
        if inst.get("username") == clean_user:
            if platform is None or inst.get("platform") == platform:
                return inst
    return None


def detect_current_instance() -> dict[str, Any] | None:
    """
    Detect the active instance context strictly from mandatory BOT_INSTANCE_ID.
    Strictly fails closed without guessing if BOT_INSTANCE_ID is missing or not registered.
    """
    bot_instance_id = (os.environ.get("BOT_INSTANCE_ID") or "").strip()
    if not bot_instance_id:
        logger.error("[Notification Routing] Mandatory BOT_INSTANCE_ID environment variable is missing or empty")
        return None

    inst = get_instance_by_id(bot_instance_id)
    if not inst:
        logger.error("[Notification Routing] Configured BOT_INSTANCE_ID %r not found in registry", bot_instance_id)
        return None

    return inst


def resolve_companion_telegram_instance(source_instance: dict[str, Any]) -> dict[str, Any] | None:
    """
    Resolve the companion Telegram bot instance for a MAX instance directly from registry.
    For Telegram instances, returns the instance itself.
    Strictly fails closed without guessing by shared database, tokens, or heuristics.
    """
    platform = source_instance.get("platform")
    if platform == "telegram":
        return source_instance

    if platform == "max":
        comp_id = source_instance.get("companion_instance")
        if not comp_id:
            logger.error(
                "[Notification Routing] MAX instance %s has no explicit companion_instance in registry. Refusing to route.",
                source_instance.get("pm2_name"),
            )
            return None

        companion = get_instance_by_id(comp_id)
        if companion and companion.get("platform") == "telegram":
            return companion

        logger.error(
            "[Notification Routing] Configured companion %s for MAX instance %s is missing or not a Telegram instance",
            comp_id,
            source_instance.get("pm2_name"),
        )
        return None

    return None


def resolve_target_bot_token(
    target_instance: dict[str, Any],
    *,
    is_companion: bool = False,
    candidate_token: str | None = None,
) -> str | None:
    """
    Resolve and validate the target bot's Telegram token.

    STRICT GUARANTEES:
    - The token MUST have a prefix matching target_instance['platform_id'].
    - If a token from Autobus or any other foreign bot is supplied, it is REJECTED.
    - NEVER falls back to legacy config.ini.
    - Prefers environment variables and only checks runtime.env on disk if not found.
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
        candidates.append(os.environ.get("BOT_TOKEN"))
    else:
        # Direct Telegram bot
        target_token_env = target_instance.get("token_env")
        if target_token_env:
            candidates.append(os.environ.get(target_token_env))
        candidates.append(os.environ.get("BOT_TOKEN"))

    # Validate candidates from environment first
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

    # 3. Only check runtime.env from disk if no valid env candidate was found
    target_token_env = target_instance.get("token_env")
    if target_token_env:
        host_env = read_runtime_env_file()
        file_cand = host_env.get(target_token_env)
        if file_cand and isinstance(file_cand, str):
            cleaned = file_cand.strip()
            if extract_telegram_bot_id(cleaned) == expected_id:
                return cleaned

    logger.error(
        "[Notification Routing] No valid token found matching platform_id %s for instance %s",
        expected_id,
        target_instance.get("pm2_name"),
    )
    return None


def extract_database_name(db_url: str | None) -> str | None:
    """Extract database name from a connection URL."""
    if not db_url or not isinstance(db_url, str):
        return None
    cleaned = db_url.strip()
    if cleaned.startswith("sqlite"):
        if ":memory:" in cleaned:
            return ":memory:"
        parsed = urlparse(cleaned)
        path = parsed.path
        if path:
            return Path(path).stem
        return "sqlite"
    parsed = urlparse(cleaned)
    path = parsed.path.lstrip("/")
    if path:
        return path.split("?")[0].split("/")[0]
    return None


def verify_database_identity(
    instance: dict[str, Any],
    db_url: str | None = None,
    *,
    allow_sqlite_in_tests: bool = True,
) -> bool:
    """
    Verify that the DATABASE_URL matches the configured database for the instance.
    Refuses mismatched databases to prevent cross-bot data or admin contamination.
    """
    expected_db = instance.get("database")
    if not expected_db:
        logger.error("[Notification Routing] Instance %s has no configured database", instance.get("pm2_name"))
        return False

    url = db_url or os.environ.get("DATABASE_URL")
    if not url:
        logger.error("[Notification Routing] DATABASE_URL is not set for instance %s", instance.get("pm2_name"))
        return False

    actual_db = extract_database_name(url)
    if not actual_db:
        logger.error("[Notification Routing] Could not parse database name from DATABASE_URL: %s", url)
        return False

    # In test environments (pytest), allow SQLite in-memory or temporary databases
    if allow_sqlite_in_tests and (url.startswith("sqlite") or actual_db in {":memory:", "sqlite"}):
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return True

    if actual_db != expected_db:
        logger.critical(
            "[Notification Routing] CRITICAL DATABASE MISMATCH: Instance %s expects database %r, but DATABASE_URL points to %r! Refusing access.",
            instance.get("pm2_name"),
            expected_db,
            actual_db,
        )
        return False

    return True


async def resolve_target_admin_ids(
    target_instance: dict[str, Any],
    *,
    current_instance: dict[str, Any] | None = None,
    session: Any | None = None,
    db_url: str | None = None,
) -> tuple[int, ...]:
    """
    Resolve allowed Telegram admin IDs for the target instance.

    STRICT GUARANTEES:
    - Filters out all MAX IDs (>= 100_000_000_000).
    - Filters out non-positive IDs.
    - Resolves baseline strictly from target Telegram instance's configured telegram_owner_ids in registry.
    - NEVER includes owner IDs from an unrelated process (e.g. alien IDs in MAX process env).
    - Supports dynamic database admins (is_admin=True) ONLY if the database identity is verified.
    """
    admin_ids: set[int] = set()
    target_pm2 = target_instance.get("pm2_name", "")

    # 1. Authoritative base owners for the target Telegram instance from registry
    inst_owners = target_instance.get("telegram_owner_ids")
    if isinstance(inst_owners, (list, tuple)):
        for oid in inst_owners:
            if is_valid_telegram_user_id(oid):
                admin_ids.add(oid)

    # 2. Environment OWNER_IDS is ONLY used if current process IS the target Telegram instance!
    # If the current process is a MAX bot, its OWNER_IDS belongs to MAX, NOT to target Telegram bot!
    curr_pm2 = current_instance.get("pm2_name", "") if current_instance else ""
    if curr_pm2 == target_pm2 and target_instance.get("platform") == "telegram":
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

    # 3. Dynamic database admins (User.is_admin == True)
    # Allowed ONLY if the database identity is verified to match target instance's database
    db_verified = verify_database_identity(target_instance, db_url=db_url)
    if not db_verified:
        logger.warning(
            "[Notification Routing] Refusing to query dynamic admins from unverified/mismatched database for %s",
            target_pm2,
        )
    else:
        try:
            from sqlalchemy import select
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
            logger.debug("[Notification Routing] DB dynamic admin lookup skipped/failed: %s", exc)

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
        admins = tuple(sorted(target_inst.get("telegram_owner_ids", (272982544, 979514796, 806750628))))
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

    # CRITICAL INVARIANT: Non-diagnostic events MUST NEVER route to Autobus from another instance!
    if (
        category != EventCategory.SYSTEM_DIAGNOSTIC
        and target_inst.get("pm2_name") == SYSTEM_DIAGNOSTIC_BOT_PM2
        and source_inst.get("pm2_name") != SYSTEM_DIAGNOSTIC_BOT_PM2
    ):
        logger.critical(
            "[Notification Routing] VIOLATION: Non-diagnostic event %s from %s attempted to route to Autobus! Dropping.",
            category,
            source_inst.get("pm2_name"),
        )
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
        admin_chat_ids=tuple(sorted(target_inst.get("telegram_owner_ids", ()))),
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
    - Live getMe bot identity is verified before dispatch.
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

        source_inst = detect_current_instance()

        # Resolve full list of Telegram admin IDs
        admin_ids = await resolve_target_admin_ids(target_inst, current_instance=source_inst, session=session)
        if not admin_ids:
            logger.warning(
                "[Notification Routing] Alert dropped: No allowed Telegram admins for target %s",
                route.target_instance,
            )
            return False

        # Live getMe identity verification before sending (cached safely)
        try:
            await verify_bot_identity_live(
                route.target_bot_token,
                expected_username=route.target_username,
                expected_id=route.target_platform_id,
                use_cache=True,
            )
        except Exception as exc:
            logger.error("[Notification Routing] Live bot identity verification failed before dispatch: %s", exc)
            return False

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


_VERIFIED_BOT_IDENTITY_CACHE: dict[str, dict[str, Any]] = {}


def _token_cache_key(token: str) -> str:
    import hashlib
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def clear_bot_identity_cache() -> None:
    _VERIFIED_BOT_IDENTITY_CACHE.clear()


async def verify_bot_identity_live(
    bot_or_token: Any,
    *,
    expected_username: str | None = None,
    expected_id: int | None = None,
    use_cache: bool = True,
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    Safely verify live Telegram bot identity via getMe without exposing token secrets.

    Guarantees:
    - Retries getMe up to 3 times on network / transient failures.
    - Validates bot ID and username against expected values.
    - Caches verified results in-memory by token digest.
    - Raises RoutingError if identity does not match.
    - Diagnostic output contains only public username and ID, never raw secrets.
    """
    token_str = bot_or_token if isinstance(bot_or_token, str) else getattr(bot_or_token, "token", None)
    if use_cache and token_str:
        cache_key = _token_cache_key(token_str)
        cached = _VERIFIED_BOT_IDENTITY_CACHE.get(cache_key)
        if cached:
            if expected_id is not None and cached["id"] != expected_id:
                raise RoutingError(
                    f"Bot identity mismatch: cached ID {cached['id']} does not match expected {expected_id}"
                )
            if expected_username is not None:
                if cached["username"].lstrip("@").lower() != expected_username.lstrip("@").lower():
                    raise RoutingError(
                        f"Bot identity mismatch: cached @{cached['username']} does not match expected @{expected_username}"
                    )
            return cached

    from telegram_client import create_telegram_bot

    if isinstance(bot_or_token, str):
        bot_ctx = create_telegram_bot(bot_or_token)
    else:
        bot_ctx = bot_or_token

    async def _inspect(bot: Any) -> dict[str, Any]:
        last_err: Exception | None = None
        me = None
        for attempt in range(1, max_retries + 1):
            try:
                me = await bot.get_me()
                last_err = None
                break
            except (RoutingError, ValueError):
                raise
            except Exception as exc:
                last_err = exc
                if attempt < max_retries:
                    import asyncio
                    await asyncio.sleep(0.5 * attempt)

        if last_err is not None or me is None:
            raise RoutingError(f"Telegram getMe network request failed after {max_retries} attempts: {last_err}")

        clean_user = (me.username or "").lstrip("@").lower()
        if expected_id is not None and me.id != expected_id:
            raise RoutingError(
                f"Bot identity mismatch: getMe returned ID {me.id}, expected {expected_id}"
            )
        if expected_username is not None:
            clean_expected = expected_username.lstrip("@").lower()
            if clean_user != clean_expected:
                raise RoutingError(
                    f"Bot identity mismatch: getMe returned @{me.username}, expected @{expected_username}"
                )
        identity = {
            "verified": True,
            "id": me.id,
            "username": me.username,
            "is_bot": bool(me.is_bot),
        }
        if token_str:
            _VERIFIED_BOT_IDENTITY_CACHE[_token_cache_key(token_str)] = identity
        return identity

    if hasattr(bot_ctx, "__aenter__"):
        async with bot_ctx as bot:
            return await _inspect(bot)
    else:
        return await _inspect(bot_ctx)


def validate_process_notification_configuration(*, fail_closed: bool = False) -> dict[str, Any]:
    """
    Validate notification routing configuration on process startup.
    Produces a safe, non-leaking diagnostic summary.
    If fail_closed is True, raises ConfigurationError on any misconfiguration.
    """
    current_inst = detect_current_instance()
    if not current_inst:
        msg = "Current process instance could not be determined: BOT_INSTANCE_ID is missing or not registered"
        logger.error("[Notification Routing Diagnostic] %s", msg)
        if fail_closed:
            raise ConfigurationError(msg)
        return {"status": "error", "message": msg}

    # Mandatory database identity check
    if not verify_database_identity(current_inst):
        url = os.environ.get("DATABASE_URL", "<not set>")
        actual_db = extract_database_name(url)
        expected_db = current_inst.get("database")
        msg = f"DATABASE_URL mismatch for instance {current_inst.get('pm2_name')}: connected to {actual_db!r}, expected {expected_db!r}"
        logger.critical("[Notification Routing Diagnostic] %s", msg)
        if fail_closed:
            raise ConfigurationError(msg)
        return {
            "status": "error",
            "instance": current_inst.get("pm2_name"),
            "platform": current_inst.get("platform"),
            "message": msg,
        }

    platform = current_inst.get("platform")
    pm2_name = current_inst.get("pm2_name")
    companion = resolve_companion_telegram_instance(current_inst) if platform == "max" else current_inst

    if not companion:
        msg = f"Companion Telegram bot not configured for instance {pm2_name}"
        logger.error("[Notification Routing Diagnostic] %s", msg)
        if fail_closed:
            raise ConfigurationError(msg)
        return {
            "status": "error",
            "instance": pm2_name,
            "platform": platform,
            "message": msg,
        }

    target_token = resolve_target_bot_token(companion, is_companion=(platform == "max"))
    if not target_token:
        msg = f"No valid Telegram bot token found for target {companion.get('pm2_name')}"
        logger.error("[Notification Routing Diagnostic] %s", msg)
        if fail_closed:
            raise ConfigurationError(msg)
        return {
            "status": "error",
            "instance": pm2_name,
            "platform": platform,
            "message": msg,
        }

    target_id = companion.get("platform_id")
    admin_count = len(companion.get("telegram_owner_ids", []))

    if admin_count == 0:
        msg = f"No allowed Telegram admin IDs configured for target {companion.get('pm2_name')}"
        logger.error("[Notification Routing Diagnostic] %s", msg)
        if fail_closed:
            raise ConfigurationError(msg)
        return {
            "status": "error",
            "instance": pm2_name,
            "platform": platform,
            "message": msg,
        }

    res = {
        "status": "ok",
        "instance": pm2_name,
        "platform": platform,
        "target_bot": companion.get("username"),
        "target_pm2": companion.get("pm2_name"),
        "target_platform_id": target_id,
        "token_status": "VALID",
        "configured_admins_count": admin_count,
    }
    logger.info(
        "[Notification Routing Diagnostic] Instance: %s (%s) | Target Bot: %s (%s, ID: %s) | Token: VALID | Default Admins: %d",
        pm2_name,
        platform,
        companion.get("username"),
        companion.get("pm2_name"),
        target_id,
        admin_count,
    )
    return res


async def validate_process_notification_configuration_async(
    *,
    fail_closed: bool = True,
    verify_live: bool = True,
) -> dict[str, Any]:
    """
    Asynchronous process notification startup validator with live getMe verification.
    Guarantees that a process never starts up without a proven, working admin notification route.
    """
    res = validate_process_notification_configuration(fail_closed=fail_closed)
    if res.get("status") != "ok":
        return res

    is_test_env = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    skip_requested = os.environ.get("SKIP_LIVE_BOT_VERIFICATION") == "1"
    if skip_requested and not is_test_env:
        logger.warning("[Security] SKIP_LIVE_BOT_VERIFICATION is forbidden outside test environments. Enforcing live verification.")
        verify_live = True
    elif skip_requested and is_test_env:
        verify_live = False

    current_inst = detect_current_instance()
    platform = current_inst.get("platform") if current_inst else None
    companion = resolve_companion_telegram_instance(current_inst) if platform == "max" else current_inst

    if verify_live and companion:
        target_token = resolve_target_bot_token(companion, is_companion=(platform == "max"))
        if target_token:
            try:
                verified = await verify_bot_identity_live(
                    target_token,
                    expected_username=companion.get("username"),
                    expected_id=companion.get("platform_id"),
                    use_cache=True,
                )
                res["live_identity_verified"] = True
                res["live_bot_id"] = verified["id"]
                res["live_username"] = verified["username"]
            except Exception as exc:
                msg = f"Live bot identity verification failed for {companion.get('pm2_name')}: {exc}"
                logger.error("[Notification Routing Diagnostic] %s", msg)
                if fail_closed:
                    raise ConfigurationError(msg) from exc
                res["status"] = "error"
                res["message"] = msg

    return res

