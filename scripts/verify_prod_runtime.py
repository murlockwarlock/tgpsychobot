#!/usr/bin/env python3
"""Read-only post-deploy checks for the production PM2 runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine


REQUIRED_GENERAL_CONFIG_COLUMNS = frozenset(
    {
        "ai_processing_message_enabled",
        "ai_processing_message_text",
    }
)
BASELINE_PREFIX = "tgpsychobot-deploy-log-baseline-"
DB_CHECK_TIMEOUT_SECONDS = 10.0
DB_DISPOSE_TIMEOUT_SECONDS = 2.0
DB_CHECK_CONCURRENCY = 4
MAX_LOG_SCAN_BYTES = 256 * 1024
STARTUP_ERROR_RE = re.compile(
    r"Traceback \(most recent call last\)"
    r"|ModuleNotFoundError"
    r"|ImportError"
    r"|SyntaxError"
    r"|sqlalchemy\.(?:exc\.)?(?:OperationalError|ProgrammingError)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LogCheckResult:
    status: str
    reason: str | None = None


LOG_CLEAN = "clean"
LOG_ERROR = "error"
LOG_INDETERMINATE = "indeterminate"


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
    if isinstance(value, bool):
        return None
    try:
        restart_count = int(value)
    except (TypeError, ValueError):
        return None
    return restart_count if restart_count >= 0 else None


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
        if first_restart_count is None or second_restart_count is None:
            errors.append(f"restart_count_unavailable:{name}")
        elif first_restart_count != second_restart_count:
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


def _log_baseline_entry(
    name: str,
    process: dict[str, Any],
) -> dict[str, Any]:
    log_path = pm2_env(process).get("pm_err_log_path")
    if not isinstance(log_path, str) or not log_path:
        raise RuntimeError(f"missing PM2 error log path for {name}")
    try:
        file_stat = os.stat(log_path, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(f"unable to stat PM2 error log for {name}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise RuntimeError(f"PM2 error log is not a regular file for {name}")
    return {
        "path": log_path,
        "device": file_stat.st_dev,
        "inode": file_stat.st_ino,
        "offset": file_stat.st_size,
    }


def create_log_baseline(
    snapshot: dict[str, dict[str, Any]],
    expected_names: list[str],
) -> str:
    baseline: dict[str, dict[str, Any]] = {}
    for name in expected_names:
        process = snapshot.get(name)
        if process is None:
            raise RuntimeError(f"missing PM2 process for {name}")
        baseline[name] = _log_baseline_entry(name, process)

    file_descriptor, path = tempfile.mkstemp(
        prefix=BASELINE_PREFIX,
        suffix=".json",
    )
    open_descriptor: int | None = file_descriptor
    try:
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            open_descriptor = None
            json.dump(baseline, handle, ensure_ascii=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if open_descriptor is not None:
            try:
                os.close(open_descriptor)
            except OSError:
                pass
        try:
            Path(path).unlink()
        except OSError:
            pass
        raise
    return path


def load_log_baseline(path: str) -> dict[str, dict[str, Any]]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("unable to read PM2 log baseline") from exc
    if not isinstance(value, dict):
        raise RuntimeError("PM2 log baseline has an unexpected shape")
    baseline: dict[str, dict[str, Any]] = {}
    for name, entry in value.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise RuntimeError("PM2 log baseline has an unexpected entry")
        path_value = entry.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise RuntimeError("PM2 log baseline has an invalid path")
        for field in ("device", "inode", "offset"):
            field_value = entry.get(field)
            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, int)
                or field_value < 0
            ):
                raise RuntimeError("PM2 log baseline has invalid metadata")
        baseline[name] = entry
    return baseline


def recent_startup_error(
    process: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> LogCheckResult:
    if not baseline:
        return LogCheckResult(LOG_INDETERMINATE, "baseline_missing")
    path_value = baseline.get("path")
    if not isinstance(path_value, str) or not path_value:
        return LogCheckResult(LOG_INDETERMINATE, "baseline_path_missing")
    current_path = pm2_env(process).get("pm_err_log_path")
    if current_path != path_value:
        return LogCheckResult(LOG_INDETERMINATE, "log_path_changed")
    try:
        baseline_device = int(baseline["device"])
        baseline_inode = int(baseline["inode"])
        baseline_offset = int(baseline["offset"])
    except (KeyError, TypeError, ValueError):
        return LogCheckResult(LOG_INDETERMINATE, "baseline_metadata_missing")

    file_descriptor: int | None = None
    try:
        open_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        file_descriptor = os.open(path_value, open_flags)
        initial_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(initial_stat.st_mode):
            return LogCheckResult(LOG_INDETERMINATE, "fd_not_regular")
        initial_path_stat = os.stat(path_value, follow_symlinks=False)
        if not stat.S_ISREG(initial_path_stat.st_mode):
            return LogCheckResult(LOG_INDETERMINATE, "path_not_regular")
        baseline_identity = (baseline_device, baseline_inode)
        initial_fd_identity = (initial_stat.st_dev, initial_stat.st_ino)
        initial_path_identity = (initial_path_stat.st_dev, initial_path_stat.st_ino)
        if (
            initial_fd_identity != baseline_identity
            or initial_path_identity != baseline_identity
            or initial_path_identity != initial_fd_identity
        ):
            return LogCheckResult(LOG_INDETERMINATE, "identity_changed")
        if initial_stat.st_size < baseline_offset:
            return LogCheckResult(LOG_INDETERMINATE, "truncated")
        if initial_path_stat.st_size != initial_stat.st_size:
            return LogCheckResult(LOG_INDETERMINATE, "changed_before_read")
        scan_size = initial_stat.st_size - baseline_offset
        if scan_size > MAX_LOG_SCAN_BYTES:
            return LogCheckResult(LOG_INDETERMINATE, "range_too_large")

        with os.fdopen(file_descriptor, "rb") as handle:
            file_descriptor = None
            handle.seek(baseline_offset)
            content = handle.read(scan_size)
            final_stat = os.fstat(handle.fileno())
        final_path_stat = os.stat(path_value, follow_symlinks=False)
        if not stat.S_ISREG(final_path_stat.st_mode):
            return LogCheckResult(LOG_INDETERMINATE, "path_not_regular_during_read")
        final_fd_identity = (final_stat.st_dev, final_stat.st_ino)
        final_path_identity = (final_path_stat.st_dev, final_path_stat.st_ino)
        if (
            final_fd_identity != baseline_identity
            or final_path_identity != baseline_identity
            or final_path_identity != final_fd_identity
        ):
            return LogCheckResult(LOG_INDETERMINATE, "identity_changed_during_read")
        if len(content) != scan_size:
            return LogCheckResult(LOG_INDETERMINATE, "short_read")
        if (
            final_stat.st_size < baseline_offset
            or final_path_stat.st_size < baseline_offset
        ):
            return LogCheckResult(LOG_INDETERMINATE, "truncated_during_read")
        if (
            final_stat.st_size != initial_stat.st_size
            or final_path_stat.st_size != initial_path_stat.st_size
            or final_path_stat.st_size != final_stat.st_size
        ):
            return LogCheckResult(LOG_INDETERMINATE, "changed_during_read")
    except (OSError, ValueError):
        return LogCheckResult(LOG_INDETERMINATE, "unreadable")
    finally:
        if file_descriptor is not None:
            try:
                os.close(file_descriptor)
            except OSError:
                pass
    if STARTUP_ERROR_RE.search(content.decode(errors="ignore")) is not None:
        return LogCheckResult(LOG_ERROR, "startup_error")
    return LogCheckResult(LOG_CLEAN)


async def _dispose_engine(engine) -> bool:
    try:
        await asyncio.wait_for(
            engine.dispose(),
            timeout=DB_DISPOSE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return False
    except asyncio.CancelledError:
        raise
    except Exception:
        return False
    return True


async def verify_general_config(database_url: str) -> bool:
    engine = create_async_engine(database_url, pool_pre_ping=True)
    result = False
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
            query_result = await connection.execute(
                text(
                    "SELECT ai_processing_message_enabled, ai_processing_message_text "
                    "FROM bot_general_config WHERE id = 1"
                )
            )
            row = query_result.first()
            result = row is not None and row[0] is not None and row[1] is not None
    except Exception:
        result = False
    finally:
        disposed = await _dispose_engine(engine)
    return result and disposed


async def verify_migrations(
    snapshot: dict[str, dict[str, Any]],
    expected_names: list[str],
) -> tuple[int, list[str]]:
    candidates = [
        (name, snapshot[name])
        for name in expected_names
        if name in snapshot and process_is_telegram(snapshot[name])
    ]
    semaphore = asyncio.Semaphore(DB_CHECK_CONCURRENCY)

    async def verify_one(name: str, process: dict[str, Any]) -> str | None:
        async with semaphore:
            database_url = process_database_url(process)
            if not database_url:
                return f"migration_failed:{name}"
            try:
                is_valid = await asyncio.wait_for(
                    verify_general_config(database_url),
                    timeout=DB_CHECK_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                return f"migration_timeout:{name}"
            except asyncio.CancelledError:
                raise
            except Exception:
                return f"migration_failed:{name}"
            return None if is_valid else f"migration_failed:{name}"

    results = await asyncio.gather(
        *(verify_one(name, process) for name, process in candidates)
    )
    return len(candidates), [result for result in results if result is not None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision")
    parser.add_argument("--pm2-names", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--create-log-baseline", action="store_true")
    parser.add_argument("--log-baseline")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    expected_names = parse_names(args.pm2_names)

    if args.create_log_baseline:
        try:
            snapshot = load_pm2_snapshot()
            baseline_path = create_log_baseline(snapshot, expected_names)
        except (RuntimeError, OSError):
            print("log_baseline=failed", file=sys.stderr)
            return 1
        print(baseline_path)
        return 0

    if not args.revision:
        print("verification=failed")
        return 1

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
    first_pm2_errors = validate_pm2_snapshot(first_snapshot, expected_names)
    errors.extend(first_pm2_errors)

    if args.settle_seconds > 0:
        time.sleep(args.settle_seconds)
    try:
        second_snapshot = load_pm2_snapshot()
    except RuntimeError:
        second_snapshot = {}
        errors.append("pm2_unavailable_after_settle")
    second_pm2_errors = validate_pm2_snapshot(second_snapshot, expected_names)
    stability_errors = validate_pm2_stability(
        first_snapshot,
        second_snapshot,
        expected_names,
    )
    errors.extend(second_pm2_errors)
    errors.extend(stability_errors)

    log_baseline: dict[str, dict[str, Any]] = {}
    if not args.log_baseline:
        errors.append("log_baseline_required")
    else:
        try:
            log_baseline = load_log_baseline(args.log_baseline)
        except RuntimeError:
            errors.append("log_baseline_unavailable")

    log_error_names: list[str] = []
    log_indeterminate: list[str] = []
    for name in expected_names:
        process = second_snapshot.get(name)
        if process is None:
            continue
        result = recent_startup_error(process, log_baseline.get(name))
        if result.status == LOG_ERROR:
            log_error_names.append(name)
        elif result.status == LOG_INDETERMINATE:
            log_indeterminate.append(f"{name}:{result.reason}")
    if log_error_names:
        errors.append("startup_log_errors:" + ",".join(log_error_names))
    if log_indeterminate:
        errors.append("startup_log_indeterminate:" + ",".join(log_indeterminate))

    try:
        checked_migrations, migration_errors = asyncio.run(
            verify_migrations(second_snapshot, expected_names)
        )
    except Exception:
        checked_migrations, migration_errors = 0, ["migration_verifier_failed"]
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
            "ok" if not second_pm2_errors else "failed",
            len(expected_names),
            checked_migrations,
            "found"
            if log_error_names
            else "indeterminate"
            if log_indeterminate or "log_baseline_unavailable" in errors
            else "none",
        )
    )
    print(f"stability={'ok' if not stability_errors else 'failed'}")
    print(f"migration={'ok' if not migration_errors else 'failed'}")
    if log_indeterminate:
        print("startup_log_indeterminate=" + ",".join(log_indeterminate))
    if migration_errors:
        print("migration_errors=" + ",".join(migration_errors))
    if errors:
        print("verification=failed")
        return 1
    print("verification=ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
