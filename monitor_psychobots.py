#!/usr/bin/env python3
import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

import aiohttp
import asyncpg

from max_messenger_bot.identity import is_max_user_id, raw_max_user_id


STATE_FILE = os.environ.get("PSYCHOBOTS_MONITOR_STATE", "/tmp/psychobots_monitor_state.json")
REPEAT_ALERT_SECONDS = int(os.environ.get("PSYCHOBOTS_MONITOR_REPEAT_SECONDS", str(6 * 60 * 60)))
DEFAULT_STUCK_REPEAT_SECONDS = 0
STUCK_REPEAT_SECONDS = int(
    os.environ.get("PSYCHOBOTS_MONITOR_STUCK_REPEAT_SECONDS", str(DEFAULT_STUCK_REPEAT_SECONDS))
)
DEFAULT_STUCK_MAX_AGE_HOURS = 24
STUCK_MAX_AGE_HOURS = int(
    os.environ.get(
        "PSYCHOBOTS_MONITOR_STUCK_MAX_AGE_HOURS",
        str(DEFAULT_STUCK_MAX_AGE_HOURS),
    )
)
STUCK_GRACE_SECONDS = int(os.environ.get("PSYCHOBOTS_MONITOR_STUCK_GRACE_SECONDS", "180"))
DEFAULT_LOCK_FILE = "/tmp/psychobots_monitor.lock"


def get_lock_file_path() -> str:
    return os.environ.get("PSYCHOBOTS_MONITOR_LOCK_FILE", DEFAULT_LOCK_FILE)


LOG_READ_LIMIT = int(os.environ.get("PSYCHOBOTS_MONITOR_LOG_READ_LIMIT", str(250_000)))
ALERT_BOT_PM2_NAME = os.environ.get("PSYCHOBOTS_ALERT_BOT_PM2_NAME", "tg_autobusbusbot_new").strip()
ALERT_BOT_TOKEN = os.environ.get("PSYCHOBOTS_ALERT_BOT_TOKEN", "").strip()
NL_CHECK_ENABLED = os.environ.get("PSYCHOBOTS_CHECK_NL", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
NL_HOST = os.environ.get("PSYCHOBOTS_NL_HOST", "185.70.185.209").strip()
NL_PROXY_PORT = int(os.environ.get("PSYCHOBOTS_NL_PROXY_PORT", "3128"))
NL_PUBLIC_PORT = int(os.environ.get("PSYCHOBOTS_NL_PUBLIC_PORT", "443"))
NL_MARZBAN_PORT = int(os.environ.get("PSYCHOBOTS_NL_MARZBAN_PORT", "62050"))
NL_MTPROTO_PORT = int(os.environ.get("PSYCHOBOTS_NL_MTPROTO_PORT", "4430"))
NL_XRAY_PORTS = tuple(
    int(port.strip())
    for port in os.environ.get("PSYCHOBOTS_NL_XRAY_PORTS", "2053,2069").split(",")
    if port.strip()
)


class SingleInstanceLock:
    def __init__(self, lock_path: str | None = None):
        self.lock_path = lock_path or get_lock_file_path()
        self.fd = None

    def __enter__(self):
        try:
            self.fd = open(self.lock_path, "a+")
        except Exception:
            raise

        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            if self.fd:
                try:
                    self.fd.close()
                except Exception:
                    pass
                self.fd = None
            return False
        except OSError as e:
            import errno
            if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                if self.fd:
                    try:
                        self.fd.close()
                    except Exception:
                        pass
                    self.fd = None
                return False
            if self.fd:
                try:
                    self.fd.close()
                except Exception:
                    pass
                self.fd = None
            raise
        except Exception:
            if self.fd:
                try:
                    self.fd.close()
                except Exception:
                    pass
                self.fd = None
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                self.fd.close()
            except Exception:
                pass
            self.fd = None


SUPPORTED_POSTGRES_SCHEMES = frozenset({"postgresql", "postgres", "postgresql+asyncpg"})


def classify_and_parse_db_url(
    db_url: str,
) -> tuple[str, str | None, str | None, str | None]:
    """Classifies database URL into one of three categories:

    - ("supported", safe_db_key, normalized_connection_dsn, None)
    - ("malformed_or_unsupported_pg", None, None, static_sanitized_reason)
    - ("unsupported_backend", None, None, None)
    """
    if not db_url or not isinstance(db_url, str):
        return ("unsupported_backend", None, None, None)

    raw_lower = db_url.strip().lower()
    is_intended_pg = raw_lower.startswith(
        ("postgres:", "postgresql:", "postgresql+", "postgres+", "postgresql//", "postgres//")
    )

    try:
        parsed = urlsplit(db_url)
    except Exception:
        if is_intended_pg:
            return ("malformed_or_unsupported_pg", None, None, "invalid URL syntax")
        return ("unsupported_backend", None, None, None)

    scheme = (parsed.scheme or "").lower()
    if not scheme:
        if is_intended_pg:
            return ("malformed_or_unsupported_pg", None, None, "invalid URL syntax")
        return ("unsupported_backend", None, None, None)

    if scheme in SUPPORTED_POSTGRES_SCHEMES:
        try:
            port = parsed.port
        except ValueError:
            return ("malformed_or_unsupported_pg", None, None, "invalid port")

        if port is not None and not (1 <= port <= 65535):
            return ("malformed_or_unsupported_pg", None, None, "invalid port")

        host = parsed.hostname
        if not host:
            return ("malformed_or_unsupported_pg", None, None, "missing host")

        dbname = parsed.path.lstrip("/")
        if not dbname:
            return ("malformed_or_unsupported_pg", None, None, "missing database name")

        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        effective_port = port or 5432
        safe_key = f"{host}:{effective_port}/{dbname}"

        conn_scheme = "postgresql" if scheme in ("postgresql", "postgresql+asyncpg") else "postgres"
        normalized_dsn = urlunsplit(
            (conn_scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment)
        )
        return ("supported", safe_key, normalized_dsn, None)

    if scheme.startswith("postgresql+") or scheme.startswith("postgres+"):
        return ("malformed_or_unsupported_pg", None, None, "unsupported PostgreSQL driver scheme")

    return ("unsupported_backend", None, None, None)


def make_safe_db_key(db_url: str) -> str:
    status, safe_key, _, reason = classify_and_parse_db_url(db_url)
    if status == "supported" and safe_key:
        return safe_key
    raise ValueError(f"Cannot generate safe DB key: {reason or 'unsupported backend'}")


def group_apps_by_db(
    apps: list[dict],
    malformed_issues: list | None = None,
) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for app in apps:
        db_url = app.get("db_url")
        if not db_url:
            continue
        status, safe_key, normalized_dsn, reason = classify_and_parse_db_url(db_url)
        if status == "supported" and safe_key and normalized_dsn:
            app_copy = dict(app)
            app_copy["normalized_db_url"] = normalized_dsn
            groups[safe_key].append(app_copy)
        elif status == "malformed_or_unsupported_pg":
            if malformed_issues is not None:
                app_name = app.get("name", "unknown")
                malformed_issues.append(
                    (app_name, [f"DB: malformed PostgreSQL DATABASE_URL: {reason}"])
                )
        # status == "unsupported_backend": silently ignored
    return groups


def format_age(seconds: float | int) -> str:
    total_seconds = max(0, int(seconds))
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}д")
    if hours > 0:
        parts.append(f"{hours}ч")
    if minutes > 0 or not parts:
        parts.append(f"{minutes}м")
    return "".join(parts)


def make_incident_key(
    safe_db_key: str,
    user_id: int,
    dialogue_id: int,
    scope_kind: str,
    topic_id: int | None,
) -> str:
    if scope_kind == "topic":
        topic_str = "null" if topic_id is None else str(topic_id)
        return f"{safe_db_key}:{user_id}:{dialogue_id}:topic:{topic_str}"
    return f"{safe_db_key}:{user_id}:{dialogue_id}"


def format_stuck_alert(cand: dict, group_apps: list[dict]) -> str:
    bot_names = " / ".join(sorted({app.get("name", "unknown") for app in group_apps})) or "unknown"
    user_id = cand["user_id"]
    if is_max_user_id(user_id):
        raw_id = raw_max_user_id(user_id)
        user_str = f"{user_id} (MAX raw: {raw_id})"
    else:
        user_str = f"{user_id}"

    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    first_at = cand.get("first_unanswered_at")
    if isinstance(first_at, datetime):
        age_str = f"{format_age((now_utc - first_at).total_seconds())} назад"
    else:
        age_str = "неизвестно когда"

    count = cand["unanswered_count"]
    topic_info = ""
    if cand.get("incident_scope_kind") == "topic":
        tid = cand.get("incident_topic_id")
        topic_info = f", тема {tid}" if tid is not None else ", главная тема"

    return (
        f"🚨 <b>Завис AI-диалог</b>\n"
        f"Пользователь ID: <code>{user_str}</code>\n"
        f"Бот: <b>{bot_names}</b>\n"
        f"Диалог ID: <code>{cand['dialogue_id']}</code>{topic_info}\n"
        f"Неотвеченных сообщений: <b>{count}</b> (первое {age_str})"
    )


def sanitize_db_error(exc: Exception) -> str:
    msg = f"{type(exc).__name__}: {exc}"
    return re.sub(r":[^:@]+@", ":***@", msg)


async def get_effective_memory_mode(conn) -> str:
    has_table = await conn.fetchval(
        "SELECT to_regclass('public.ai_config')"
    )
    if not has_table:
        raise RuntimeError("Table 'ai_config' is missing")

    row = await conn.fetchrow(
        "SELECT memory_mode, preserve_topic_context FROM ai_config WHERE id = 1"
    )
    if not row:
        raise RuntimeError("Canonical row AIConfig(id=1) is missing in 'ai_config'")

    raw_mode = row["memory_mode"]
    if raw_mode in ("reset", "topic", "global"):
        return raw_mode

    preserve_topic = bool(row["preserve_topic_context"])
    return "topic" if preserve_topic else "reset"


async def get_suppressed_max_user_ids(conn) -> set[int]:
    has_table = await conn.fetchval(
        "SELECT to_regclass('public.max_bot_states')"
    )
    if not has_table:
        return set()

    rows = await conn.fetch("SELECT user_id FROM max_bot_states")
    return {int(r["user_id"]) for r in rows if r["user_id"] is not None}


async def fetch_stuck_dialogue_episodes(
    conn,
    effective_memory_mode: str,
    max_age_hours: int = STUCK_MAX_AGE_HOURS,
    grace_seconds: int = STUCK_GRACE_SECONDS,
) -> list[dict]:
    if effective_memory_mode == "topic":
        scope_condition = (
            "m.user_id = u.user_id AND m.dialogue_id = u.current_dialogue_id "
            "AND m.topic_id IS NOT DISTINCT FROM u.current_topic_id"
        )
    else:
        scope_condition = "m.user_id = u.user_id AND m.dialogue_id = u.current_dialogue_id"

    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    grace_cutoff_utc = now_utc - timedelta(seconds=grace_seconds)
    params = [grace_cutoff_utc]

    sql = f"""
    WITH active_users AS (
        SELECT id AS user_id, current_dialogue_id, current_topic_id
        FROM users
    ),
    latest_non_user AS (
        SELECT m.user_id, m.dialogue_id, MAX(m.id) AS last_boundary_id
        FROM messages m
        JOIN active_users u ON {scope_condition}
        WHERE m.role IS DISTINCT FROM 'user'
        GROUP BY m.user_id, m.dialogue_id
    ),
    unanswered_suffix AS (
        SELECT
            m.user_id,
            m.dialogue_id,
            COUNT(*) AS unanswered_count,
            MIN(m.id) AS first_unanswered_id,
            MAX(m.id) AS last_unanswered_id
        FROM messages m
        JOIN active_users u ON {scope_condition}
        LEFT JOIN latest_non_user b ON m.user_id = b.user_id AND m.dialogue_id = b.dialogue_id
        WHERE m.role = 'user'
          AND (b.last_boundary_id IS NULL OR m.id > b.last_boundary_id)
        GROUP BY m.user_id, m.dialogue_id
    ),
    active_dialogue_tails AS (
        SELECT DISTINCT ON (m.user_id, m.dialogue_id)
            m.user_id, m.dialogue_id, m.id AS tail_id, m.role AS tail_role
        FROM messages m
        JOIN active_users u ON {scope_condition}
        ORDER BY m.user_id, m.dialogue_id, m.id DESC
    )
    SELECT
        s.user_id,
        s.dialogue_id,
        u.current_topic_id,
        s.unanswered_count,
        s.first_unanswered_id,
        s.last_unanswered_id,
        m_first.timestamp AS first_unanswered_at,
        m_last.timestamp AS last_unanswered_at,
        (m_last.timestamp <= $1) AS is_grace_elapsed
    FROM unanswered_suffix s
    JOIN active_users u ON s.user_id = u.user_id AND s.dialogue_id = u.current_dialogue_id
    JOIN active_dialogue_tails t ON s.user_id = t.user_id AND s.dialogue_id = t.dialogue_id
    JOIN messages m_first ON m_first.id = s.first_unanswered_id
    JOIN messages m_last ON m_last.id = s.last_unanswered_id
    WHERE t.tail_role = 'user'
      AND s.unanswered_count >= 2
    ;
    """
    rows = await conn.fetch(sql, *params)
    results = []
    scope_kind = "topic" if effective_memory_mode == "topic" else "dialogue"
    max_age_cutoff = (
        now_utc - timedelta(hours=max_age_hours) if max_age_hours > 0 else None
    )
    for r in rows:
        topic_val = int(r["current_topic_id"]) if r["current_topic_id"] is not None else None
        last_at = r["last_unanswered_at"]
        is_within_max_age = True
        if max_age_cutoff is not None and last_at is not None:
            is_within_max_age = (last_at >= max_age_cutoff)

        results.append(
            {
                "user_id": int(r["user_id"]),
                "dialogue_id": int(r["dialogue_id"]),
                "current_topic_id": topic_val,
                "memory_mode": effective_memory_mode,
                "incident_scope_kind": scope_kind,
                "incident_topic_id": topic_val if scope_kind == "topic" else None,
                "unanswered_count": int(r["unanswered_count"]),
                "first_unanswered_id": int(r["first_unanswered_id"]),
                "last_unanswered_id": int(r["last_unanswered_id"]),
                "first_unanswered_at": r["first_unanswered_at"],
                "last_unanswered_at": last_at,
                "is_grace_elapsed": bool(r["is_grace_elapsed"]),
                "is_within_max_age": is_within_max_age,
            }
        )
    return results


async def check_stuck_dialogues(
    apps: list[dict],
    state: dict,
    all_issues: list,
) -> int:
    import asyncpg

    state.setdefault("stuck_dialogues", {})
    db_groups = group_apps_by_db(apps, malformed_issues=all_issues)
    alerts_sent = 0

    for safe_db_key, group_apps in db_groups.items():
        sample_app = group_apps[0]
        conn = None
        try:
            target_dsn = sample_app["normalized_db_url"]
            conn = await asyncpg.connect(target_dsn, timeout=10)
            effective_mode = await get_effective_memory_mode(conn)
            episodes = await fetch_stuck_dialogue_episodes(
                conn,
                effective_mode,
                max_age_hours=STUCK_MAX_AGE_HOURS,
                grace_seconds=STUCK_GRACE_SECONDS,
            )
            suppressed_user_ids = await get_suppressed_max_user_ids(conn)
        except Exception as exc:
            err_msg = f"DB: {safe_db_key} stuck dialogue check failed: {sanitize_db_error(exc)}"
            all_issues.append((sample_app["name"], [err_msg]))
            continue
        finally:
            if conn:
                try:
                    await conn.close()
                except Exception:
                    pass

        active_identities = set()
        for ep in episodes:
            ep_kind = ep.get("incident_scope_kind")
            if ep_kind in ("reset", "global") or not ep_kind:
                ep_kind = "dialogue"
            ep_identity = (
                safe_db_key,
                ep["user_id"],
                ep["dialogue_id"],
                ep_kind,
                ep["incident_topic_id"],
            )
            active_identities.add(ep_identity)

            is_alert_eligible = (
                ep["is_grace_elapsed"]
                and ep.get("is_within_max_age", True)
                and ep["user_id"] not in suppressed_user_ids
            )

            if is_alert_eligible:
                incident_key = make_incident_key(
                    safe_db_key,
                    ep["user_id"],
                    ep["dialogue_id"],
                    ep_kind,
                    ep["incident_topic_id"],
                )
                is_in_cooldown = False
                if incident_key in state["stuck_dialogues"]:
                    last_alerted = state["stuck_dialogues"][incident_key].get("last_alerted_at", 0)
                    if STUCK_REPEAT_SECONDS <= 0:
                        is_in_cooldown = True
                    elif (int(time.time()) - last_alerted) < STUCK_REPEAT_SECONDS:
                        is_in_cooldown = True

                if not is_in_cooldown:
                    alert_text = format_stuck_alert(ep, group_apps)
                    delivered = await send_alert(apps, alert_text)
                    if delivered:
                        alerts_sent += 1
                        now_ts = int(time.time())
                        existing_entry = state["stuck_dialogues"].get(incident_key, {})
                        first_alerted_at = existing_entry.get("first_alerted_at", now_ts)
                        state["stuck_dialogues"][incident_key] = {
                            "db_key": safe_db_key,
                            "user_id": ep["user_id"],
                            "dialogue_id": ep["dialogue_id"],
                            "incident_scope_kind": ep_kind,
                            "incident_topic_id": ep["incident_topic_id"],
                            "memory_mode": ep.get("memory_mode"),
                            "first_alerted_at": first_alerted_at,
                            "last_alerted_at": now_ts,
                            "last_count": ep["unanswered_count"],
                            "first_id": ep["first_unanswered_id"],
                            "last_id": ep["last_unanswered_id"],
                        }
                        print(f"Stuck dialogue alert sent: user {ep['user_id']} in {sample_app['name']}")

        keys_to_delete = []
        for k, entry in state["stuck_dialogues"].items():
            if entry.get("db_key") == safe_db_key:
                entry_kind = entry.get("incident_scope_kind")
                if entry_kind in ("reset", "global") or not entry_kind:
                    entry_kind = "dialogue"

                entry_identity = (
                    safe_db_key,
                    entry.get("user_id"),
                    entry.get("dialogue_id"),
                    entry_kind,
                    entry.get("incident_topic_id"),
                )
                if entry_identity not in active_identities:
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state["stuck_dialogues"][k]

    return alerts_sent

CRITICAL_COLUMNS = {
    "users": {
        "response_length",
        "can_view_history",
        "accepted_disclaimer",
        "current_dialogue_id",
        "current_topic_id",
        "referred_by",
        "tg_user_id",
    },
    "user_subscriptions": {
        "pending_robokassa_invoice_id",
        "last_payment_attempt",
        "payment_attempt_count",
        "discount_percent",
    },
    "subscription_config": {
        "subscriptions_enabled",
        "topics_enabled",
        "test_button_enabled",
        "change_name_button_enabled",
        "topics_btn_name",
        "topics_btn_on_top",
        "welcome_bonus_days",
        "referral_enabled",
        "referral_btn_name",
        "referral_sub_btn_name",
        "referral_bonus_days_referrer",
        "referral_bonus_days_referral",
        "referral_pay_bonus_enabled",
        "referral_pay_bonus_days",
        "referral_pay_bonus_first_only",
    },
    "ai_config": {
        "memory_mode",
        "shared_prompt_block",
        "service_prompt_block",
        "fallback_provider",
        "fallback_model",
        "kie_api_key",
        "kie_model",
        "kie_base_url",
        "kie_upload_base_url",
        "kie_transcription_model",
        "kie_credit_alert_threshold",
        "kie_credit_alert_sent",
        "image_generation_provider",
        "image_generation_model",
        "image_edit_provider",
        "image_edit_model",
    },
}

LOG_ERROR_RE = re.compile(
    r"(UndefinedColumn|ProgrammingError|OperationalError|Traceback|Error during initial|"
    r"Background task .*crashed|Failed to fetch updates|Flood control exceeded)",
    re.IGNORECASE,
)


def load_state(path: str | Path | None = None) -> dict:
    target = Path(path) if path is not None else Path(os.environ.get("PSYCHOBOTS_MONITOR_STATE", STATE_FILE))
    try:
        state = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        state = {}
    state.setdefault("log_offsets", {})
    state.setdefault("alerts", {})
    state.setdefault("stuck_dialogues", {})
    return state


def save_state(state: dict, path: str | Path | None = None) -> None:
    target = Path(path) if path is not None else Path(os.environ.get("PSYCHOBOTS_MONITOR_STATE", STATE_FILE))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, target)


def run_pm2_jlist() -> list[dict]:
    result = subprocess.run(["pm2", "jlist"], text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def get_bot_apps() -> list[dict]:
    apps = []
    for proc in run_pm2_jlist():
        env = proc.get("pm2_env", {})
        token = env.get("BOT_TOKEN")
        db_url = env.get("DATABASE_URL")
        if not token or not db_url:
            continue
        apps.append(
            {
                "name": proc.get("name", "unknown"),
                "status": env.get("status"),
                "restart_time": env.get("restart_time", 0),
                "pm_uptime": env.get("pm_uptime", 0),
                "token": token,
                "db_url": db_url,
                "telegram_proxy": env.get("TELEGRAM_PROXY") or os.environ.get("TELEGRAM_PROXY"),
                "owner_ids": parse_owner_ids(env.get("OWNER_IDS", "")),
                "delivery_mode": str(env.get("TELEGRAM_DELIVERY_MODE", "")).lower(),
                "error_log": env.get("pm_err_log_path"),
                "webhook_base_url": env.get("BASE_WEBHOOK_URL"),
                "webhook_path_prefix": env.get("WEBHOOK_PATH_PREFIX"),
            }
        )
    return apps


def parse_owner_ids(value: str) -> list[int]:
    ids = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            pass
    return ids


def make_dsn(db_url: str) -> str:
    return db_url.replace("postgresql+asyncpg://", "postgresql://", 1)


def proxy_request_kwargs(proxy_url: str | None) -> dict:
    if not proxy_url:
        return {}
    parsed = urlsplit(proxy_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("TELEGRAM_PROXY must be an HTTP or HTTPS proxy URL")

    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    if parsed.port:
        host = f"{host}:{parsed.port}"
    clean_url = urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))

    kwargs = {"proxy": clean_url}
    if parsed.username is not None:
        kwargs["proxy_auth"] = aiohttp.BasicAuth(
            unquote(parsed.username),
            unquote(parsed.password or ""),
        )
    return kwargs


async def check_db_schema(app: dict) -> list[str]:
    import asyncpg

    issues = []
    conn = None
    try:
        conn = await asyncpg.connect(make_dsn(app["db_url"]), timeout=10)
        for table, expected_columns in CRITICAL_COLUMNS.items():
            rows = await conn.fetch(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = $1
                """,
                table,
            )
            existing = {row["column_name"] for row in rows}
            if not existing:
                issues.append(f"DB: table {table} is missing")
                continue
            missing = sorted(expected_columns - existing)
            if missing:
                issues.append(f"DB: {table} missing columns: {', '.join(missing)}")
        await conn.fetchval("SELECT count(*) FROM users")
    except Exception as exc:
        issues.append(f"DB: connection/query failed: {type(exc).__name__}: {exc}")
    finally:
        if conn:
            await conn.close()
    return issues


async def check_telegram(app: dict, http: aiohttp.ClientSession) -> list[str]:
    issues = []
    base_url = f"https://api.telegram.org/bot{app['token']}"
    try:
        proxy_kwargs = proxy_request_kwargs(app.get("telegram_proxy"))
    except ValueError as exc:
        return [f"Telegram proxy configuration failed: {exc}"]
    try:
        async with http.get(f"{base_url}/getMe", **proxy_kwargs) as resp:
            data = await resp.json(content_type=None)
            if resp.status != 200 or not data.get("ok"):
                issues.append(f"Telegram getMe failed: HTTP {resp.status} {data}")
    except Exception as exc:
        issues.append(f"Telegram getMe failed: {type(exc).__name__}: {exc}")
        return issues

    try:
        async with http.get(f"{base_url}/getWebhookInfo", **proxy_kwargs) as resp:
            data = await resp.json(content_type=None)
            webhook_url = (data.get("result") or {}).get("url") if data.get("ok") else None
            if app["delivery_mode"] == "polling" and webhook_url:
                issues.append(f"Telegram: webhook is set while polling mode is enabled: {webhook_url}")
    except Exception as exc:
        issues.append(f"Telegram getWebhookInfo failed: {type(exc).__name__}: {exc}")
    return issues


async def check_payment_webhooks(app: dict, http: aiohttp.ClientSession) -> list[str]:
    base_url = str(app.get("webhook_base_url") or "").rstrip("/")
    prefix = str(app.get("webhook_path_prefix") or "").strip("/")
    if not base_url or not prefix:
        return []

    url = f"{base_url}/{prefix}/webhooks/robokassa/result"
    try:
        async with http.get(url, allow_redirects=False) as resp:
            if resp.status != 400:
                return [f"Payment webhook check returned HTTP {resp.status}: {url}"]
    except Exception as exc:
        return [f"Payment webhook is unavailable: {type(exc).__name__}: {url}"]
    return []


async def check_tcp_port(host: str, port: int, label: str) -> str | None:
    writer = None
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=8)
    except Exception as exc:
        return f"{label}: {host}:{port} is unavailable ({type(exc).__name__})"
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
    return None


async def check_nl_infrastructure(http: aiohttp.ClientSession) -> list[str]:
    if not NL_CHECK_ENABLED:
        return []

    checks = [
        check_tcp_port(NL_HOST, NL_PROXY_PORT, "Proxy"),
        check_tcp_port(NL_HOST, NL_PUBLIC_PORT, "VPN entrypoint"),
        check_tcp_port(NL_HOST, NL_MARZBAN_PORT, "Marzban Node"),
        check_tcp_port(NL_HOST, NL_MTPROTO_PORT, "MTProto"),
    ]
    checks.extend(check_tcp_port(NL_HOST, port, f"Xray {port}") for port in NL_XRAY_PORTS)
    issues = [issue for issue in await asyncio.gather(*checks) if issue]

    proxy_url = os.environ.get("TELEGRAM_PROXY")
    if not proxy_url:
        issues.append("Proxy: TELEGRAM_PROXY is not configured for the monitor")
        return issues
    try:
        proxy_kwargs = proxy_request_kwargs(proxy_url)
        async with http.get("https://api.telegram.org", **proxy_kwargs) as resp:
            if resp.status >= 500:
                issues.append(f"Proxy: Telegram check returned HTTP {resp.status}")
    except Exception as exc:
        issues.append(f"Proxy: Telegram request failed ({type(exc).__name__})")
    return issues


def read_new_log_errors(app: dict, state: dict, include_existing: bool) -> list[str]:
    log_path = app.get("error_log")
    if not log_path:
        return []
    path = Path(log_path)
    if not path.exists():
        return [f"Logs: error log not found: {log_path}"]

    key = str(path)
    size = path.stat().st_size
    offsets = state.setdefault("log_offsets", {})
    previous = int(offsets.get(key, 0))
    if previous > size:
        previous = 0

    if previous == 0 and not include_existing:
        offsets[key] = size
        return []

    start = max(previous, size - LOG_READ_LIMIT)
    with path.open("rb") as fh:
        fh.seek(start)
        chunk = fh.read().decode("utf-8", errors="replace")
    offsets[key] = size

    matches = [line.strip() for line in chunk.splitlines() if LOG_ERROR_RE.search(line)]
    if not matches:
        return []

    get_updates_errors = [line for line in matches if "Failed to fetch updates" in line]
    other_errors = [line for line in matches if "Failed to fetch updates" not in line]
    issues = []
    if len(get_updates_errors) >= 3:
        issues.append(f"Logs: {len(get_updates_errors)} polling fetch errors since last check")
    elif get_updates_errors:
        issues.append(f"Logs: polling fetch error: {get_updates_errors[-1][-220:]}")
    if other_errors:
        tail = other_errors[-3:]
        issues.append("Logs: " + " | ".join(line[-220:] for line in tail))
    return issues


def should_alert(issues: list[str], state: dict) -> bool:
    if not issues:
        return False
    now = int(time.time())
    signature = hashlib.sha256("\n".join(sorted(issues)).encode("utf-8")).hexdigest()
    alerts = state.setdefault("alerts", {})
    last = alerts.get(signature, 0)
    if now - int(last) < REPEAT_ALERT_SECONDS:
        return False
    alerts[signature] = now
    return True


def get_candidate_alert_tokens(apps: list[dict]) -> list[str]:
    preferred_name = os.environ.get("PSYCHOBOTS_ALERT_BOT_PM2_NAME", ALERT_BOT_PM2_NAME).strip()
    direct_token = os.environ.get("PSYCHOBOTS_ALERT_BOT_TOKEN", ALERT_BOT_TOKEN).strip()

    tokens: list[str] = []
    seen: set[str] = set()

    if direct_token:
        tokens.append(direct_token)
        seen.add(direct_token)

    if preferred_name:
        for app in apps:
            if app.get("name") == preferred_name:
                token = app.get("token")
                if token and token not in seen:
                    tokens.append(token)
                    seen.add(token)

    for app in apps:
        token = app.get("token")
        if token and token not in seen:
            tokens.append(token)
            seen.add(token)

    return tokens


async def send_alert(apps: list[dict], text: str) -> bool:
    recipients = sorted({owner_id for app in apps for owner_id in app["owner_ids"]})
    tokens = get_candidate_alert_tokens(apps)
    if not recipients or not tokens:
        print("No notification recipients or bot tokens found")
        return False

    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as http:
        proxy_url = next(
            (app.get("telegram_proxy") for app in apps if app.get("telegram_proxy")),
            os.environ.get("TELEGRAM_PROXY"),
        )
        try:
            proxy_kwargs = proxy_request_kwargs(proxy_url)
        except ValueError:
            proxy_kwargs = {}
        for token in tokens:
            delivered = False
            for chat_id in recipients:
                try:
                    async with http.post(
                        f"https://api.telegram.org/bot{token}/sendMessage",
                        json={
                            "chat_id": chat_id,
                            "text": text[:3900],
                            "parse_mode": "HTML",
                            "disable_web_page_preview": True,
                        },
                        **proxy_kwargs,
                    ) as resp:
                        data = await resp.json(content_type=None)
                        if resp.status == 200 and data.get("ok"):
                            delivered = True
                except Exception:
                    continue
            if delivered:
                return True
    print("Failed to deliver Telegram alert")
    return False


async def run_check(include_existing_log_errors: bool, lock_path: str | None = None) -> int:
    with SingleInstanceLock(lock_path) as acquired:
        if not acquired:
            print("Psychobots monitor already running; exiting cleanly.")
            return 0

        state = load_state()
        apps = get_bot_apps()
        all_issues = []

        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as http:
            for app in apps:
                issues = []
                if app["status"] != "online":
                    issues.append(f"PM2: status is {app['status']}")
                issues.extend(await check_db_schema(app))
                issues.extend(await check_telegram(app, http))
                issues.extend(await check_payment_webhooks(app, http))
                issues.extend(read_new_log_errors(app, state, include_existing_log_errors))
                if issues:
                    all_issues.append((app["name"], issues))

            nl_issues = await check_nl_infrastructure(http)
            if nl_issues:
                all_issues.append(("nl_infrastructure", nl_issues))

        await check_stuck_dialogues(apps, state, all_issues)

        flat_issues = [f"{name}: {issue}" for name, issues in all_issues for issue in issues]
        if should_alert(flat_issues, state):
            lines = ["🚨 <b>Psychobots monitor detected issues</b>"]
            for name, issues in all_issues:
                lines.append(f"\n<b>{name}</b>")
                for issue in issues[:8]:
                    lines.append(f"• {issue}")
            await send_alert(apps, "\n".join(lines))

        save_state(state)
        if all_issues:
            for name, issues in all_issues:
                print(f"{name}:")
                for issue in issues:
                    print(f"  - {issue}")
            return 1
        print("OK")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-existing-log-errors",
        action="store_true",
        help="Read existing PM2 error log tail on first run instead of starting from EOF.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run_check(args.include_existing_log_errors)))


if __name__ == "__main__":
    main()
