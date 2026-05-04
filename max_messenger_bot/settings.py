from __future__ import annotations

import configparser
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_CONFIG_PATH = PROJECT_ROOT / "config.ini"


def apply_legacy_env_defaults() -> None:
    """
    The legacy project keeps runtime configuration in env vars.
    For the isolated MAX app we transparently backfill them from config.ini
    so database.py can be imported without changing the old code.
    """
    parser = configparser.ConfigParser()
    parser.read(LEGACY_CONFIG_PATH, encoding="utf-8")

    if "BOT_TOKEN" not in os.environ:
        os.environ["BOT_TOKEN"] = parser.get("bot", "token", fallback="")
    if "DATABASE_URL" not in os.environ:
        os.environ["DATABASE_URL"] = parser.get("database", "url", fallback="")
    if "OWNER_IDS" not in os.environ:
        os.environ["OWNER_IDS"] = parser.get("bot", "owner_ids", fallback="")


@dataclass(frozen=True)
class Settings:
    max_token: str
    max_api_base: str
    host: str
    port: int
    webhook_path: str
    webhook_base_url: str
    webhook_secret: str
    use_polling: bool
    polling_timeout: int
    polling_limit: int
    update_types: tuple[str, ...]
    log_level: str
    log_console_level: str
    log_dir: str
    log_max_bytes: int
    log_backup_count: int


WEBHOOK_SECRET_RE = re.compile(r"^[a-zA-Z0-9_-]{5,256}$")
ALLOWED_WEBHOOK_PORTS = {80, 443, 8080, 8443}
ALLOWED_WEBHOOK_PORT_RANGE = range(16384, 32384)


def clear_settings_cache() -> None:
    get_settings.cache_clear()


def validate_webhook_runtime_settings(settings: Settings) -> None:
    if not settings.webhook_base_url or settings.use_polling:
        return

    parsed = urlparse(settings.webhook_base_url)
    if parsed.scheme != "https":
        raise RuntimeError("MAX webhook требует HTTPS URL.")
    if not parsed.netloc:
        raise RuntimeError("MAX webhook URL должен содержать host.")
    port = parsed.port or 443
    if port not in ALLOWED_WEBHOOK_PORTS and port not in ALLOWED_WEBHOOK_PORT_RANGE:
        raise RuntimeError("MAX webhook должен слушать порт 80, 8080, 443, 8443 или диапазон 16384-32383.")
    if settings.webhook_secret and not WEBHOOK_SECRET_RE.fullmatch(settings.webhook_secret):
        raise RuntimeError("MAX webhook secret должен соответствовать шаблону [a-zA-Z0-9_-]{5,256}.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    parser = configparser.ConfigParser()
    parser.read(LEGACY_CONFIG_PATH, encoding="utf-8")
    max_section = "max"

    token = (
        os.getenv("MAX_BOT_TOKEN")
        or os.getenv("MAX_TOKEN")
        or parser.get(max_section, "token", fallback="")
    ).strip()

    webhook_path = os.getenv("MAX_WEBHOOK_PATH", "/max/webhook").strip() or "/max/webhook"
    update_types_raw = os.getenv(
        "MAX_UPDATE_TYPES",
        "message_created,message_callback,bot_started",
    )
    update_types = tuple(item.strip() for item in update_types_raw.split(",") if item.strip())

    return Settings(
        max_token=token,
        max_api_base=os.getenv("MAX_API_BASE", "https://platform-api.max.ru").rstrip("/"),
        host=os.getenv("MAX_APP_HOST", "0.0.0.0"),
        port=int(os.getenv("MAX_APP_PORT", "8090")),
        webhook_path=webhook_path,
        webhook_base_url=os.getenv("MAX_WEBHOOK_BASE_URL", "").rstrip("/"),
        webhook_secret=os.getenv(
            "MAX_WEBHOOK_SECRET",
            parser.get(max_section, "webhook_secret", fallback=""),
        ).strip(),
        use_polling=os.getenv("MAX_USE_POLLING", "0").lower() in {"1", "true", "yes"},
        polling_timeout=int(os.getenv("MAX_POLL_TIMEOUT", "30")),
        polling_limit=int(os.getenv("MAX_POLL_LIMIT", "50")),
        update_types=update_types or ("message_created", "message_callback", "bot_started"),
        log_level=os.getenv("MAX_LOG_LEVEL", "INFO"),
        log_console_level=os.getenv("MAX_LOG_CONSOLE_LEVEL", "INFO"),
        log_dir=os.getenv("MAX_LOG_DIR", str(PROJECT_ROOT / "logs" / "max_messenger_bot")),
        log_max_bytes=max(1024, int(os.getenv("MAX_LOG_MAX_BYTES", str(10 * 1024 * 1024)))),
        log_backup_count=max(1, int(os.getenv("MAX_LOG_BACKUP_COUNT", "5"))),
    )
