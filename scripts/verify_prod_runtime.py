#!/usr/bin/env python3
"""Read-only post-deploy checks for the production PM2 runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine


REQUIRED_GENERAL_CONFIG_COLUMNS = frozenset(
    {
        "ai_processing_message_enabled",
        "ai_processing_message_text",
    }
)
STARTUP_ERROR_RE = re.compile(
    r"Traceback \(most recent call last\)"
    r"|ModuleNotFoundError"
    r"|ImportError"
    r"|SyntaxError"
    r"|sqlalchemy\.(?:exc\.)?(?:OperationalError|ProgrammingError)",
    re.IGNORECASE,
)


def parse_names(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


def load_pm2_snapshot() -> dict[str, dict[str, Any]]:
    try:
        result = subprocess.run(
            ["pm2", "jlist"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RuntimeError("unable to read PM2 process state") from exc
    if not isinstance(payload, list):
        raise RuntimeError("PM2 returned an unexpected process list")
    return {
        str(item["name"]): item
        for item in payload
        if isinstance(item, dict) and item.get("name")
    }


def pm2_env(process: dict[str, Any]) -> dict[str, Any]:
    value = process.get("pm2_env")
    return value if isinstance(value, dict) else {}


def process_status(process: dict[str, Any]) -> str | None:
    return pm2_env(process).get("status") or process.get("status")


def process_pid(process: dict[str, Any]) -> int | None:
    value = pm2_env(process).get("pid", process.get("pid"))
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def process_restart_count(process: dict[str, Any]) -> int | None:
    value = pm2_env(process).get("restart_time")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def process_is_telegram(process: dict[str, Any]) -> bool:
    path = str(pm2_env(process).get("pm_exec_path") or process.get("pm_exec_path") or "")
    return path.endswith("/main.py")


def validate_pm2_snapshot(
    snapshot: dict[str, dict[str, Any]],
    expected_names: list[str],
) -> list[str]:
    errors: list[str] = []
    for name in expected_names:
        process = snapshot.get(name)
        if process is None:
            errors.append(f"missing:{name}")
            continue
        if process_status(process) != "online":
            errors.append(f"not_online:{name}")
        if process_pid(process) is None:
            errors.append(f"missing_pid:{name}")
    return errors


def validate_pm2_stability(
    first: dict[str, dict[str, Any]],
    second: dict[str, dict[str, Any]],
    expected_names: list[str],
) -> list[str]:
    errors: list[str] = []
    for name in expected_names:
        first_process = first.get(name)
        second_process = second.get(name)
        if first_process is None or second_process is None:
            errors.append(f"disappeared:{name}")
            continue
        if process_pid(first_process) != process_pid(second_process):
            errors.append(f"pid_changed:{name}")
        first_restart_count = process_restart_count(first_process)
        second_restart_count = process_restart_count(second_process)
        if (
            first_restart_count is not None
            and second_restart_count is not None
            and first_restart_count != second_restart_count
        ):
            errors.append(f"restart_count_changed:{name}")
    return errors


def process_database_url(process: dict[str, Any]) -> str | None:
    env = pm2_env(process).get("env")
    if isinstance(env, dict):
        value = env.get("DATABASE_URL")
        if isinstance(value, str) and value:
            return value

    pid = process_pid(process)
    if pid is None:
        return None
    try:
        raw_environment = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return None
    for item in raw_environment.split(b"\0"):
        key, separator, value = item.partition(b"=")
        if separator and key == b"DATABASE_URL" and value:
            return value.decode(errors="ignore")
    return None


def recent_startup_error(process: dict[str, Any], started_at: float | None) -> bool:
    if started_at is None:
        return False
    path_value = pm2_env(process).get("pm_err_log_path")
    if not isinstance(path_value, str) or not path_value:
        return False
    path = Path(path_value)
    try:
        stat = path.stat()
        if stat.st_mtime < started_at - 120:
            return False
        content = path.read_bytes()[-131072:].decode(errors="ignore")
    except OSError:
        return False
    return STARTUP_ERROR_RE.search(content) is not None


async def verify_general_config(database_url: str) -> bool:
    engine = create_async_engine(database_url, pool_pre_ping=True)
    try:
        async with engine.connect() as connection:
            columns = await connection.run_sync(
                lambda sync_connection: {
                    column["name"]
                    for column in inspect(sync_connection).get_columns("bot_general_config")
                }
            )
            if not REQUIRED_GENERAL_CONFIG_COLUMNS.issubset(columns):
                return False
            result = await connection.execute(
                text(
                    "SELECT ai_processing_message_enabled, ai_processing_message_text "
                    "FROM bot_general_config WHERE id = 1"
                )
            )
            row = result.first()
            return row is not None and row[0] is not None and row[1] is not None
    except Exception:
        return False
    finally:
        await engine.dispose()


async def verify_migrations(
    snapshot: dict[str, dict[str, Any]],
    expected_names: list[str],
) -> tuple[int, list[str]]:
    checked = 0
    errors: list[str] = []
    for name in expected_names:
        process = snapshot.get(name)
        if process is None or not process_is_telegram(process):
            continue
        checked += 1
        database_url = process_database_url(process)
        if not database_url or not await verify_general_config(database_url):
            errors.append(f"migration_failed:{name}")
    return checked, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--pm2-names", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--started-at", type=float)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    expected_names = parse_names(args.pm2_names)
    errors: list[str] = []

    revision_path = Path(args.root) / "REVISION"
    try:
        revision = revision_path.read_text().strip()
    except OSError:
        revision = ""
    if revision != args.revision:
        errors.append("revision_mismatch")

    try:
        first_snapshot = load_pm2_snapshot()
    except RuntimeError:
        first_snapshot = {}
        errors.append("pm2_unavailable")
    errors.extend(validate_pm2_snapshot(first_snapshot, expected_names))

    if args.settle_seconds > 0:
        time.sleep(args.settle_seconds)
    try:
        second_snapshot = load_pm2_snapshot()
    except RuntimeError:
        second_snapshot = {}
        errors.append("pm2_unavailable_after_settle")
    errors.extend(validate_pm2_snapshot(second_snapshot, expected_names))
    errors.extend(validate_pm2_stability(first_snapshot, second_snapshot, expected_names))

    log_error_names = [
        name
        for name in expected_names
        if name in second_snapshot
        and recent_startup_error(second_snapshot[name], args.started_at)
    ]
    if log_error_names:
        errors.append("startup_log_errors:" + ",".join(log_error_names))

    checked_migrations, migration_errors = asyncio.run(
        verify_migrations(second_snapshot, expected_names)
    )
    errors.extend(migration_errors)
    expected_telegram_migrations = sum(
        1
        for name in expected_names
        if name in second_snapshot and process_is_telegram(second_snapshot[name])
    )
    if checked_migrations != expected_telegram_migrations:
        errors.append("migration_not_checked")

    print(f"revision={'ok' if revision == args.revision else 'failed'}")
    print(
        "pm2={} expected={} migrations_checked={} startup_errors={}".format(
            "ok" if not validate_pm2_snapshot(second_snapshot, expected_names) else "failed",
            len(expected_names),
            checked_migrations,
            "none" if not log_error_names else "found",
        )
    )
    print(f"stability={'ok' if not validate_pm2_stability(first_snapshot, second_snapshot, expected_names) else 'failed'}")
    print(f"migration={'ok' if not migration_errors else 'failed'}")
    if errors:
        print("verification=failed")
        return 1
    print("verification=ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
