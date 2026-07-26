#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

import aiohttp


STATE_FILE = os.environ.get("PSYCHOBOTS_MONITOR_STATE", "/tmp/psychobots_monitor_state.json")
REPEAT_ALERT_SECONDS = int(os.environ.get("PSYCHOBOTS_MONITOR_REPEAT_SECONDS", str(6 * 60 * 60)))
LOG_READ_LIMIT = int(os.environ.get("PSYCHOBOTS_MONITOR_LOG_READ_LIMIT", str(250_000)))
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


def load_state() -> dict:
    try:
        return json.loads(Path(STATE_FILE).read_text(encoding="utf-8"))
    except Exception:
        return {"log_offsets": {}, "alerts": {}}


def save_state(state: dict) -> None:
    path = Path(STATE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


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


async def send_alert(apps: list[dict], text: str) -> None:
    recipients = sorted({owner_id for app in apps for owner_id in app["owner_ids"]})
    tokens = [app["token"] for app in apps]
    if not recipients or not tokens:
        print("No notification recipients or bot tokens found")
        return

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
                return
    print("Failed to deliver Telegram alert")


async def run_check(include_existing_log_errors: bool) -> int:
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
            issues.extend(read_new_log_errors(app, state, include_existing_log_errors))
            if issues:
                all_issues.append((app["name"], issues))

        nl_issues = await check_nl_infrastructure(http)
        if nl_issues:
            all_issues.append(("nl_infrastructure", nl_issues))

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
