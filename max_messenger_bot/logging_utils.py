from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .settings import get_settings


BOT_LOGGER_NAME = "max_bot.bot"
AI_LOGGER_NAME = "max_bot.ai"
PAYMENTS_LOGGER_NAME = "max_bot.payments"
MAX_LOGGER_NAME = "max_bot.max"
NAMED_LOGGERS = (BOT_LOGGER_NAME, AI_LOGGER_NAME, PAYMENTS_LOGGER_NAME, MAX_LOGGER_NAME)


def get_bot_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(BOT_LOGGER_NAME if not name else f"{BOT_LOGGER_NAME}.{name}")


def get_ai_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(AI_LOGGER_NAME if not name else f"{AI_LOGGER_NAME}.{name}")


def get_payments_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(PAYMENTS_LOGGER_NAME if not name else f"{PAYMENTS_LOGGER_NAME}.{name}")


def get_max_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(MAX_LOGGER_NAME if not name else f"{MAX_LOGGER_NAME}.{name}")


def _ensure_rotating_handler(
    log_path: Path,
    level: int,
    formatter: logging.Formatter,
    *,
    max_bytes: int,
    backup_count: int,
) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
        delay=True,
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def _configure_named_logger(
    logger_name: str,
    file_path: Path,
    level: int,
    formatter: logging.Formatter,
    *,
    max_bytes: int,
    backup_count: int,
) -> None:
    logger = logging.getLogger(logger_name)
    if getattr(logger, "_max_bot_configured", False):
        return
    logger.setLevel(level)
    logger.propagate = False
    logger.addHandler(
        _ensure_rotating_handler(
            file_path,
            level,
            formatter,
            max_bytes=max_bytes,
            backup_count=backup_count,
        )
    )
    logger._max_bot_configured = True  # type: ignore[attr-defined]


def reset_logging() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    if hasattr(root, "_max_bot_console_configured"):
        delattr(root, "_max_bot_console_configured")

    for logger_name in NAMED_LOGGERS:
        logger = logging.getLogger(logger_name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        logger.propagate = True
        if hasattr(logger, "_max_bot_configured"):
            delattr(logger, "_max_bot_configured")


def configure_logging(*, force: bool = False) -> None:
    if force:
        reset_logging()

    root = logging.getLogger()
    if not getattr(root, "_max_bot_console_configured", False):
        console = logging.StreamHandler()
        settings = get_settings()
        console_level = getattr(logging, settings.log_console_level.upper(), logging.INFO)
        console.setLevel(console_level)
        console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        root.setLevel(min(getattr(logging, settings.log_level.upper(), logging.INFO), console_level))
        root.addHandler(console)
        root._max_bot_console_configured = True  # type: ignore[attr-defined]

    settings = get_settings()
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    _configure_named_logger(
        BOT_LOGGER_NAME,
        log_dir / "bot.log",
        level,
        formatter,
        max_bytes=settings.log_max_bytes,
        backup_count=settings.log_backup_count,
    )
    _configure_named_logger(
        AI_LOGGER_NAME,
        log_dir / "ai.log",
        level,
        formatter,
        max_bytes=settings.log_max_bytes,
        backup_count=settings.log_backup_count,
    )
    _configure_named_logger(
        PAYMENTS_LOGGER_NAME,
        log_dir / "payments.log",
        level,
        formatter,
        max_bytes=settings.log_max_bytes,
        backup_count=settings.log_backup_count,
    )
    _configure_named_logger(
        MAX_LOGGER_NAME,
        log_dir / "max.log",
        level,
        formatter,
        max_bytes=settings.log_max_bytes,
        backup_count=settings.log_backup_count,
    )
