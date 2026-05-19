import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from aiogram import Bot

from config import OWNER_IDS
from database import get_all_admin_ids


log = logging.getLogger(__name__)

APP_PORT = os.environ.get("APP_PORT", "8080")
PROJECT_DIR = Path(__file__).resolve().parent
RESTART_MARKER_PATH = PROJECT_DIR / "logs" / f"admin_restart_{APP_PORT}.json"
MAX_TELEGRAM_USER_ID = 100_000_000_000


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_user_ref(user) -> str:
    if not user:
        return "неизвестно"
    full_name = getattr(user, "full_name", None) or getattr(user, "first_name", None) or ""
    username = getattr(user, "username", None)
    user_id = getattr(user, "id", None)
    text = full_name.strip() or str(user_id or "unknown")
    if username:
        text += f" (@{username})"
    if user_id:
        text += f" [id={user_id}]"
    return text


async def _run_pm2_json() -> list[dict]:
    proc = await asyncio.create_subprocess_exec(
        "pm2",
        "jlist",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError((stderr or stdout).decode("utf-8", errors="replace").strip())
    return json.loads(stdout.decode("utf-8"))


async def get_current_pm2_identity() -> dict:
    current_pid = os.getpid()
    env_pm_id = os.environ.get("pm_id") or os.environ.get("PM_ID")
    env_name = os.environ.get("name") or os.environ.get("PM2_NAME")
    identity = {
        "pm2_id": int(env_pm_id) if str(env_pm_id or "").isdigit() else None,
        "name": env_name,
        "pid": current_pid,
        "app_port": APP_PORT,
    }

    try:
        for item in await _run_pm2_json():
            pm2_env = item.get("pm2_env") or {}
            item_pid = item.get("pid")
            item_id = pm2_env.get("pm_id")
            item_env = pm2_env.get("env") or {}
            item_port = str(item_env.get("APP_PORT") or pm2_env.get("APP_PORT") or "")

            if item_pid == current_pid or (APP_PORT and item_port == str(APP_PORT)):
                identity["pm2_id"] = item_id
                identity["name"] = item.get("name") or pm2_env.get("name")
                identity["pid"] = item_pid or current_pid
                return identity
    except Exception as exc:
        log.warning("Could not resolve PM2 identity via jlist: %s", exc)

    return identity


def write_restart_marker(identity: dict, requester) -> None:
    RESTART_MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "requested_at": _utc_now_iso(),
        "requester": _safe_user_ref(requester),
        "requester_id": getattr(requester, "id", None),
        "pm2_id": identity.get("pm2_id"),
        "process_name": identity.get("name"),
        "pid_before": identity.get("pid"),
        "app_port": identity.get("app_port"),
    }
    tmp_path = RESTART_MARKER_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(RESTART_MARKER_PATH)


async def schedule_pm2_restart(pm2_id: int, delay_sec: float = 1.5) -> None:
    async def _restart_later():
        await asyncio.sleep(delay_sec)
        try:
            proc = await asyncio.create_subprocess_exec(
                "pm2",
                "restart",
                str(pm2_id),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
        except Exception as exc:
            log.error("Failed to run pm2 restart %s: %s", pm2_id, exc, exc_info=exc)

    asyncio.create_task(_restart_later())


async def notify_admins_after_requested_restart(bot: Bot, delivery_mode: str) -> None:
    if not RESTART_MARKER_PATH.exists():
        return

    try:
        data = json.loads(RESTART_MARKER_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not read restart marker %s: %s", RESTART_MARKER_PATH, exc)
        data = {}

    identity = await get_current_pm2_identity()
    try:
        bot_info = await bot.get_me()
        bot_label = f"@{bot_info.username}" if bot_info.username else bot_info.full_name
    except Exception:
        bot_label = "бот"

    text = (
        "✅ <b>Бот полностью запустился после перезагрузки</b>\n\n"
        f"<b>Бот:</b> {bot_label}\n"
        f"<b>PM2:</b> id={identity.get('pm2_id')}, процесс={identity.get('name') or 'неизвестно'}\n"
        f"<b>Порт:</b> {identity.get('app_port')}\n"
        f"<b>Режим Telegram:</b> {delivery_mode}\n"
        f"<b>Запросил:</b> {data.get('requester') or 'неизвестно'}\n"
        f"<b>Запрошено:</b> {data.get('requested_at') or 'неизвестно'}"
    )

    try:
        admin_ids = await get_all_admin_ids()
    except Exception:
        admin_ids = set(OWNER_IDS)

    for admin_id in admin_ids:
        if admin_id >= MAX_TELEGRAM_USER_ID:
            continue
        try:
            await bot.send_message(admin_id, text)
        except Exception as exc:
            log.warning("Could not send restart-complete notification to admin %s: %s", admin_id, exc)

    try:
        RESTART_MARKER_PATH.unlink(missing_ok=True)
    except Exception as exc:
        log.warning("Could not remove restart marker %s: %s", RESTART_MARKER_PATH, exc)
