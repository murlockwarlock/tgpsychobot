#!/usr/bin/env python3
"""Read-only post-deploy checks for the production PM2 runtime."""

from __future__ import annotations

import argparse
import asyncio
from array import array
from bisect import bisect_right
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
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
MAX_LIVE_LOG_SCAN_BYTES = 128 * 1024 * 1024
LIVE_LOG_READ_CHUNK_BYTES = 1024 * 1024
TRACEBACK_MARKER = "Traceback (most recent call last)"
LOG_TIMESTAMP_RE = re.compile(
    r"^[ \t]*(?P<timestamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"
    r"(?:[,.]\d{1,9})?(?:[ \t]?(?:Z|[+-]\d{2}:?\d{2}))?)"
)
LOG_HEADER_TIMESTAMP_BYTES_RE = re.compile(
    rb"(?m)^[ \t]*(?P<timestamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"
    rb"(?:[,.]\d{1,9})?(?:[ \t]?(?:Z|[+-]\d{2}:?\d{2}))?)"
    rb"(?=[^\r\n]*\|[ \t]*(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)[ \t]*\|)"
)
TRACEBACK_MARKER_BYTES_RE = re.compile(re.escape(TRACEBACK_MARKER.encode()))
CANDIDATE_LOG_HINT_BYTES_RE = re.compile(
    rb"Traceback \(most recent call last\)"
    rb"|ModuleNotFoundError|ImportError|SyntaxError|NameError|AttributeError|"
    rb"IntegrityError|sqlalchemy\.(?:exc\.)?(?:OperationalError|ProgrammingError)"
    rb"|TelegramBadRequest|TelegramNetworkError"
    rb"|(?:database|migration|scheduler|handler(?:[- ]registration)?|"
    rb"translation(?:[- ]cache)?|locale|start[_ -]?intent|benefit[_ -]?grant)"
    rb"[^\r\n]{0,120}(?:error|exception|failed|failure|could not|unable)"
    rb"|(?:error|exception|failed|failure|could not|unable)[^\r\n]{0,120}"
    rb"(?:database|migration|scheduler|handler(?:[- ]registration)?|"
    rb"translation(?:[- ]cache)?|locale|start[_ -]?intent|benefit[_ -]?grant)",
    re.IGNORECASE,
)
LOG_LEVEL_RE = re.compile(r"^\s*\|\s*(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\s*\|", re.IGNORECASE)
TRACEBACK_END_RE = re.compile(
    r"^\s*(?:[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning|Interrupt|Exit|Request)"
    r"|SystemExit|KeyboardInterrupt|StopIteration):"
)
TELEGRAM_BAD_REQUEST_RE = re.compile(r"\bTelegramBadRequest\b", re.IGNORECASE)
CHAT_NOT_FOUND_RE = re.compile(
    r"(?:\bTelegramBadRequest\b|Telegram server says\s*-\s*Bad Request:)"
    r"[^\n]*\bchat not found\b",
    re.IGNORECASE,
)
TELEGRAM_NETWORK_RE = re.compile(r"\bTelegramNetworkError\b", re.IGNORECASE)
RECOVERABLE_NETWORK_EXCEPTION_RE = re.compile(
    r"^[ \t]*(?:[A-Za-z_][A-Za-z0-9_]*\.)*"
    r"(?:ConnectionResetError|ClientOSError|ClientConnectorError|"
    r"ServerDisconnectedError|TimeoutError)\b",
    re.IGNORECASE | re.MULTILINE,
)
RECOVERABLE_TELEGRAM_NETWORK_RE = re.compile(
    r"^[ \t]*(?:[A-Za-z_][A-Za-z0-9_]*\.)*TelegramNetworkError\b[^\n]*"
    r"(?:ConnectionResetError|ClientOSError|ClientConnectorError|"
    r"ServerDisconnectedError|TimeoutError)\b",
    re.IGNORECASE | re.MULTILINE,
)
STARTUP_ERROR_RE = re.compile(
    r"ModuleNotFoundError"
    r"|ImportError"
    r"|SyntaxError"
    r"|NameError"
    r"|AttributeError"
    r"|IntegrityError"
    r"|sqlalchemy\.(?:exc\.)?(?:OperationalError|ProgrammingError)",
    re.IGNORECASE,
)
STARTUP_CONTEXT_FAILURE_RE = re.compile(
    r"(?:database|migration|scheduler|handler(?:[- ]registration)?|"
    r"translation(?:[- ]cache)?|locale|start[_ -]?intent|benefit[_ -]?grant)"
    r"[^\n]{0,120}(?:error|exception|failed|failure|could not|unable)"
    r"|(?:error|exception|failed|failure|could not|unable)[^\n]{0,120}"
    r"(?:database|migration|scheduler|handler(?:[- ]registration)?|"
    r"translation(?:[- ]cache)?|locale|start[_ -]?intent|benefit[_ -]?grant)",
    re.IGNORECASE,
)
FATAL_CONTEXT_MARKER_RE = re.compile(
    r"database|migration|scheduler|translation(?:[- ]cache)?|locale|"
    r"start[_ -]?intent|benefit[_ -]?grant|handler[ -]+registration",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LogCheckResult:
    status: str
    reason: str | None = None
    log_path: str | None = None
    first_timestamp: str | None = None
    process_start_timestamp: str | None = None
    classification: str | None = None
    matched_rule: str | None = None
    excerpt: str | None = None


@dataclass(frozen=True)
class TimestampedLogBlock:
    content: str
    timestamp: float | None


LOG_CLEAN = "clean"
LOG_ERROR = "error"
LOG_INDETERMINATE = "indeterminate"


def _is_recoverable_network_log(block: str) -> bool:
    return (
        RECOVERABLE_NETWORK_EXCEPTION_RE.search(block) is not None
        or RECOVERABLE_TELEGRAM_NETWORK_RE.search(block) is not None
    )


def _is_chat_not_found_log(block: str) -> bool:
    lines = block.splitlines()
    bad_request_lines = [
        line for line in lines if TELEGRAM_BAD_REQUEST_RE.search(line)
    ]
    if bad_request_lines:
        return all(CHAT_NOT_FOUND_RE.search(line) for line in bad_request_lines)
    return any(CHAT_NOT_FOUND_RE.search(line) for line in lines)


def _is_allowed_delivery_block(block: str) -> bool:
    if STARTUP_ERROR_RE.search(block):
        return False
    context_match = STARTUP_CONTEXT_FAILURE_RE.search(block)
    if context_match and FATAL_CONTEXT_MARKER_RE.search(context_match.group()):
        return False
    return _is_chat_not_found_log(block) or _is_recoverable_network_log(block)


def _remove_allowed_delivery_context(
    parts: list[str],
    allowed_indexes: set[int],
) -> str:
    residual = list(parts)
    for index in allowed_indexes:
        context_index = index - 1
        context = residual[context_index]
        lines = context.splitlines(keepends=True)
        while lines and not lines[-1].strip():
            lines.pop()
        if lines:
            context_line = lines[-1]
            context_match = STARTUP_CONTEXT_FAILURE_RE.search(context_line)
            if context_match and not FATAL_CONTEXT_MARKER_RE.search(context_match.group()):
                residual[context_index] = "".join(lines[:-1])
        residual[index] = ""
    return "".join(residual)


def pm_process_start_timestamp(process: dict[str, Any]) -> float | None:
    value = pm2_env(process).get("pm_uptime")
    if isinstance(value, bool):
        return None
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(timestamp):
        return None
    if timestamp > 100_000_000_000:
        timestamp /= 1000
    return timestamp if timestamp >= 1_000_000_000 else None


def format_timestamp(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def _parse_timestamp_value(value: str) -> float | None:
    value = value.strip().replace(",", ".")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.timestamp()


def _timestamp_from_line(line: str) -> float | None:
    match = LOG_TIMESTAMP_RE.match(line)
    return _parse_timestamp_value(match.group("timestamp")) if match else None


def _is_timestamped_log_header(line: str) -> bool:
    match = LOG_TIMESTAMP_RE.match(line)
    return (
        match is not None
        and _timestamp_from_line(line) is not None
        and LOG_LEVEL_RE.search(line[match.end():]) is not None
    )


def _strip_log_prefix(line: str) -> str:
    match = LOG_TIMESTAMP_RE.match(line)
    value = line[match.end():] if match else line
    level_match = re.match(
        r"^\s*\|\s*(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\s*\|\s*",
        value,
        re.IGNORECASE,
    )
    return value[level_match.end():] if level_match else value


def _is_traceback_terminal_line(line: str) -> bool:
    return TRACEBACK_END_RE.match(_strip_log_prefix(line).rstrip("\r\n")) is not None


def timestamped_log_blocks(content: str) -> list[TimestampedLogBlock]:
    blocks: list[TimestampedLogBlock] = []
    current_lines: list[str] = []
    current_timestamp: float | None = None
    traceback_open = False

    def finish_block() -> None:
        nonlocal current_lines, current_timestamp, traceback_open
        if current_lines:
            blocks.append(
                TimestampedLogBlock(
                    content="".join(current_lines),
                    timestamp=current_timestamp,
                )
            )
        current_lines = []
        current_timestamp = None
        traceback_open = False

    for line in content.splitlines(keepends=True):
        line_timestamp = _timestamp_from_line(line)
        is_log_header = line_timestamp is not None and _is_timestamped_log_header(line)
        if current_lines and line_timestamp is not None:
            if not traceback_open:
                finish_block()
            elif is_log_header and not _is_traceback_terminal_line(line):
                finish_block()
            elif TRACEBACK_MARKER in line:
                finish_block()
        if not current_lines and line_timestamp is not None:
            current_timestamp = line_timestamp
        elif current_timestamp is None and line_timestamp is not None:
            current_timestamp = line_timestamp
        current_lines.append(line)
        if TRACEBACK_MARKER in line:
            traceback_open = True
        elif traceback_open and _is_traceback_terminal_line(line):
            finish_block()

    finish_block()
    return blocks


def _sanitize_excerpt(content: str, matched_rule: str | None) -> str:
    lines = content.splitlines()
    needles = [matched_rule] if matched_rule else []
    if matched_rule and ":" in matched_rule:
        prefix, suffix = matched_rule.split(":", 1)
        needles.append(suffix if prefix == "startup_context" else prefix)
    selected = next(
        (
            line
            for line in lines
            if any(needle.casefold() in line.casefold() for needle in needles)
        ),
        None,
    )
    if selected is None:
        selected = next(
            (line for line in lines if line.strip()),
            "",
        )
    selected = _strip_log_prefix(selected).strip()
    selected = re.sub(
        r"(?i)\b(?:bot)?\d{6,}:[A-Za-z0-9_-]{20,}\b",
        "[REDACTED_TOKEN]",
        selected,
    )
    selected = re.sub(
        r"(?i)\b(?:chat|user|recipient)[_ ]?id\s*[=: ]\s*\d+",
        "[REDACTED_ID]",
        selected,
    )
    selected = re.sub(r"(?<![A-Za-z])\d{8,}(?![A-Za-z])", "[REDACTED_ID]", selected)
    selected = re.sub(
        r"(?i)\b(?:bot[_ -]?)?(?:token|password|secret|api[_ -]?key)\s*[=: ]\s*\S+",
        "[REDACTED_SECRET]",
        selected,
    )
    selected = re.sub(r"(?i)\bbearer\s+\S+", "[REDACTED_AUTH]", selected)
    selected = re.sub(
        r"(?i)\b(?:postgres(?:ql)?|mysql|redis)://\S+",
        "[REDACTED_DB_URL]",
        selected,
    )
    selected = re.sub(
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
        "[REDACTED_JWT]",
        selected,
    )
    return selected[:240]


def _matched_rule(content: str) -> str:
    startup_match = STARTUP_ERROR_RE.search(content)
    if startup_match:
        return startup_match.group(0)
    context_match = STARTUP_CONTEXT_FAILURE_RE.search(content)
    if context_match:
        fatal_context = FATAL_CONTEXT_MARKER_RE.search(context_match.group(0))
        return (
            "startup_context:" + fatal_context.group(0)
            if fatal_context
            else "startup_context_failure"
        )
    if TELEGRAM_BAD_REQUEST_RE.search(content):
        return "TelegramBadRequest:not_chat_not_found"
    if TELEGRAM_NETWORK_RE.search(content):
        return "TelegramNetworkError:unclassified"
    if TRACEBACK_MARKER in content:
        return "unclassified_traceback"
    return "startup_context_failure"


def classify_log_window(content: str) -> LogCheckResult:
    if not content:
        return LogCheckResult(LOG_CLEAN)

    parts = content.split(TRACEBACK_MARKER)
    allowed_indexes = {
        index
        for index, block in enumerate(parts[1:], start=1)
        if _is_allowed_delivery_block(block)
    }
    if len(allowed_indexes) < len(parts) - 1:
        rule = _matched_rule(content)
        return LogCheckResult(
            LOG_ERROR,
            "startup_error",
            matched_rule=rule,
            excerpt=_sanitize_excerpt(content, rule),
        )

    residual = _remove_allowed_delivery_context(parts, allowed_indexes)
    if STARTUP_ERROR_RE.search(residual) or STARTUP_CONTEXT_FAILURE_RE.search(residual):
        rule = _matched_rule(residual)
        return LogCheckResult(
            LOG_ERROR,
            "startup_error",
            matched_rule=rule,
            excerpt=_sanitize_excerpt(content, rule),
        )

    if TELEGRAM_BAD_REQUEST_RE.search(content) and not _is_chat_not_found_log(content):
        rule = _matched_rule(content)
        return LogCheckResult(
            LOG_ERROR,
            "startup_error",
            matched_rule=rule,
            excerpt=_sanitize_excerpt(content, rule),
        )
    if TELEGRAM_NETWORK_RE.search(content) and not _is_recoverable_network_log(content):
        rule = _matched_rule(content)
        return LogCheckResult(
            LOG_ERROR,
            "startup_error",
            matched_rule=rule,
            excerpt=_sanitize_excerpt(content, rule),
        )
    return LogCheckResult(LOG_CLEAN)


def classify_candidate_log_window(
    content: str,
    process_start: float | None,
) -> LogCheckResult:
    if process_start is None:
        result = classify_log_window(content)
        if result.status == LOG_CLEAN:
            return LogCheckResult(LOG_INDETERMINATE, "process_start_unavailable")
        return LogCheckResult(
            LOG_INDETERMINATE,
            "process_start_unavailable",
            matched_rule=result.matched_rule,
            excerpt=result.excerpt,
        )

    for block in timestamped_log_blocks(content):
        result = classify_log_window(block.content)
        if result.status == LOG_CLEAN:
            continue
        if block.timestamp is None:
            return LogCheckResult(
                LOG_INDETERMINATE,
                "timestamp_unavailable",
                classification="unattributed_candidate_error",
                process_start_timestamp=format_timestamp(process_start),
                matched_rule=result.matched_rule,
                excerpt=result.excerpt,
            )
        if block.timestamp < process_start:
            continue
        return LogCheckResult(
            result.status,
            result.reason,
            classification="fatal_candidate_error",
            first_timestamp=format_timestamp(block.timestamp),
            process_start_timestamp=format_timestamp(process_start),
            matched_rule=result.matched_rule,
            excerpt=result.excerpt,
        )
    return LogCheckResult(
        LOG_CLEAN,
        process_start_timestamp=format_timestamp(process_start),
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
    pid = process_pid(process)
    restart_count = process_restart_count(process)
    process_start = pm_process_start_timestamp(process)
    if pid is None or restart_count is None or process_start is None:
        raise RuntimeError(f"missing PM2 identity metadata for {name}")
    return {
        "path": log_path,
        "device": file_stat.st_dev,
        "inode": file_stat.st_ino,
        "offset": file_stat.st_size,
        "pid": pid,
        "restart_count": restart_count,
        "pm_uptime": process_start,
    }


def create_log_baseline(
    snapshot: dict[str, dict[str, Any]],
    expected_names: list[str],
    source_names: list[str] | None = None,
) -> str:
    if source_names is not None and len(source_names) != len(expected_names):
        raise RuntimeError("log baseline source names do not match expected names")
    baseline: dict[str, dict[str, Any]] = {}
    captured_at = time.time()
    for index, name in enumerate(expected_names):
        process = snapshot.get(name)
        if process is None and source_names is not None:
            process = snapshot.get(source_names[index])
        if process is None:
            raise RuntimeError(f"missing PM2 process for {name}")
        entry = _log_baseline_entry(name, process)
        entry["captured_at"] = captured_at
        baseline[name] = entry

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
        for field in (
            "device",
            "inode",
            "offset",
            "pid",
            "restart_count",
        ):
            field_value = entry.get(field)
            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, int)
                or field_value < 0
            ):
                raise RuntimeError("PM2 log baseline has invalid metadata")
        for field in ("pm_uptime", "captured_at"):
            field_value = entry.get(field)
            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, (int, float))
                or field_value <= 0
            ):
                raise RuntimeError("PM2 log baseline has invalid timestamp metadata")
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
    result = classify_candidate_log_window(
        content.decode(errors="ignore"),
        pm_process_start_timestamp(process),
    )
    return replace(result, log_path=path_value)


def _find_process_start_log_boundary(
    file_descriptor: int,
    file_size: int,
    process_start: float,
) -> int | None:
    local_clock, local_fraction = _timestamp_threshold_parts(process_start, None)
    utc_clock, utc_fraction = _timestamp_threshold_parts(
        process_start,
        timezone.utc,
    )
    position = file_size
    scanned = 0
    trailing = b""
    while position > 0 and scanned < MAX_LIVE_LOG_SCAN_BYTES:
        chunk_size = min(
            LIVE_LOG_READ_CHUNK_BYTES,
            position,
            MAX_LIVE_LOG_SCAN_BYTES - scanned,
        )
        chunk_start = position - chunk_size
        os.lseek(file_descriptor, chunk_start, os.SEEK_SET)
        chunk = os.read(file_descriptor, chunk_size)
        if len(chunk) != chunk_size:
            return None
        data = chunk + trailing
        search_start = 0
        if chunk_start > 0 and os.pread(file_descriptor, 1, chunk_start - 1) != b"\n":
            line_end = data.find(b"\n")
            if line_end < 0:
                trailing = data
                search_start = len(data)
            else:
                trailing = data[:line_end + 1]
                search_start = line_end + 1
        else:
            trailing = b""

        boundary_offset: int | None = None
        for match in LOG_HEADER_TIMESTAMP_BYTES_RE.finditer(data, search_start):
            if _timestamp_bytes_precede_start(
                match.group("timestamp"),
                process_start,
                local_clock,
                local_fraction,
                utc_clock,
                utc_fraction,
            ):
                boundary_offset = chunk_start + match.start()
        if boundary_offset is not None:
            return boundary_offset

        scanned += chunk_size
        position = chunk_start

    return 0 if position == 0 else None


def _timestamp_threshold_parts(
    timestamp: float,
    tz: timezone | None,
) -> tuple[bytes, bytes]:
    whole_seconds = math.floor(timestamp)
    fraction_ns = round((timestamp - whole_seconds) * 1_000_000_000)
    if fraction_ns >= 1_000_000_000:
        whole_seconds += 1
        fraction_ns = 0
    parsed = datetime.fromtimestamp(whole_seconds, tz)
    clock = parsed.strftime("%Y-%m-%d %H:%M:%S").encode()
    return clock, f"{fraction_ns:09d}".encode()


def _timestamp_bytes_precede_start(
    raw_timestamp: bytes,
    process_start: float,
    local_clock: bytes,
    local_fraction: bytes,
    utc_clock: bytes,
    utc_fraction: bytes,
) -> bool:
    suffix = raw_timestamp[19:]
    if suffix.endswith(b"Z"):
        threshold_clock = utc_clock
        threshold_fraction = utc_fraction
    elif re.search(rb"[+-]\d{2}:?\d{2}$", suffix):
        parsed = _parse_timestamp_value(raw_timestamp.decode())
        return parsed is not None and parsed < process_start
    else:
        threshold_clock = local_clock
        threshold_fraction = local_fraction

    log_clock = raw_timestamp[:19].replace(b"T", b" ")
    if log_clock != threshold_clock:
        return log_clock < threshold_clock

    fraction_match = re.match(rb"[.,](\d+)", suffix)
    log_fraction = fraction_match.group(1)[:9].ljust(9, b"0") if fraction_match else b"0" * 9
    return log_fraction < threshold_fraction


def _timestamped_header_offsets(content: bytes) -> array:
    return array(
        "Q",
        (match.start() for match in LOG_HEADER_TIMESTAMP_BYTES_RE.finditer(content)),
    )


def _log_header_timestamp(content: bytes, offset: int) -> float | None:
    line_end = content.find(b"\n", offset)
    if line_end < 0:
        line_end = len(content)
    return _timestamp_from_line(content[offset:line_end].decode(errors="ignore"))


def _traceback_end_offset(content: bytes, marker_offset: int) -> int:
    line_end = content.find(b"\n", marker_offset)
    if line_end < 0:
        return len(content)
    cursor = line_end + 1
    while cursor < len(content):
        line_end = content.find(b"\n", cursor)
        if line_end < 0:
            line_end = len(content)
        line = content[cursor:line_end].decode(errors="ignore")
        if _is_traceback_terminal_line(line):
            return min(line_end + 1, len(content))
        cursor = line_end + 1
    return len(content)


def _candidate_log_result_for_block(
    result: LogCheckResult,
    timestamp: float | None,
    process_start: float,
) -> LogCheckResult | None:
    if result.status == LOG_CLEAN:
        return None
    if timestamp is None:
        return LogCheckResult(
            LOG_INDETERMINATE,
            "timestamp_unavailable",
            classification="unattributed_candidate_error",
            process_start_timestamp=format_timestamp(process_start),
            matched_rule=result.matched_rule,
            excerpt=result.excerpt,
        )
    if timestamp < process_start:
        return None
    return LogCheckResult(
        LOG_ERROR,
        result.reason,
        classification="fatal_candidate_error",
        first_timestamp=format_timestamp(timestamp),
        process_start_timestamp=format_timestamp(process_start),
        matched_rule=result.matched_rule,
        excerpt=result.excerpt,
    )


def classify_candidate_log_bytes(
    content: bytes,
    process_start: float,
) -> LogCheckResult:
    if not CANDIDATE_LOG_HINT_BYTES_RE.search(content):
        return LogCheckResult(
            LOG_CLEAN,
            process_start_timestamp=format_timestamp(process_start),
        )

    headers = _timestamped_header_offsets(content)
    trace_ranges: list[tuple[int, int]] = []
    for marker in TRACEBACK_MARKER_BYTES_RE.finditer(content):
        header_index = bisect_right(headers, marker.start()) - 1
        block_start = int(headers[header_index]) if header_index >= 0 else marker.start()
        block_end = _traceback_end_offset(content, marker.start())
        timestamp = (
            _log_header_timestamp(content, int(headers[header_index]))
            if header_index >= 0
            else None
        )
        result = classify_log_window(content[block_start:block_end].decode(errors="ignore"))
        candidate = _candidate_log_result_for_block(result, timestamp, process_start)
        if candidate is not None:
            return candidate
        trace_ranges.append((block_start, block_end))

    handled_blocks: set[int] = set()
    trace_index = 0
    for match in CANDIDATE_LOG_HINT_BYTES_RE.finditer(content):
        position = match.start()
        while trace_index < len(trace_ranges) and trace_ranges[trace_index][1] <= position:
            trace_index += 1
        if trace_index < len(trace_ranges):
            trace_start, trace_end = trace_ranges[trace_index]
            if trace_start <= position < trace_end:
                continue
        if match.group().lower().startswith(b"traceback"):
            continue

        line_start = content.rfind(b"\n", 0, position) + 1
        header_index = bisect_right(headers, line_start) - 1
        if header_index >= 0:
            block_start = int(headers[header_index])
            block_end = int(headers[header_index + 1]) if header_index + 1 < len(headers) else len(content)
            timestamp = _log_header_timestamp(content, block_start)
        else:
            block_start = line_start
            line_end = content.find(b"\n", position)
            block_end = len(content) if line_end < 0 else line_end + 1
            timestamp = None
        if block_start in handled_blocks:
            continue
        handled_blocks.add(block_start)
        if timestamp is not None and timestamp < process_start:
            continue

        result = classify_log_window(content[block_start:block_end].decode(errors="ignore"))
        candidate = _candidate_log_result_for_block(result, timestamp, process_start)
        if candidate is not None:
            return candidate

    return LogCheckResult(
        LOG_CLEAN,
        process_start_timestamp=format_timestamp(process_start),
    )


def recent_startup_error_since_process_start(
    process: dict[str, Any],
) -> LogCheckResult:
    process_start = pm_process_start_timestamp(process)
    log_path = pm2_env(process).get("pm_err_log_path")
    if process_start is None:
        return LogCheckResult(LOG_INDETERMINATE, "process_start_unavailable")
    if not isinstance(log_path, str) or not log_path:
        return LogCheckResult(LOG_INDETERMINATE, "log_path_missing")

    file_descriptor: int | None = None
    try:
        file_descriptor = os.open(
            log_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        initial_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(initial_stat.st_mode):
            return LogCheckResult(LOG_INDETERMINATE, "fd_not_regular", log_path=log_path)
        initial_path_stat = os.stat(log_path, follow_symlinks=False)
        identity = (initial_stat.st_dev, initial_stat.st_ino)
        if (
            not stat.S_ISREG(initial_path_stat.st_mode)
            or (initial_path_stat.st_dev, initial_path_stat.st_ino) != identity
        ):
            return LogCheckResult(LOG_INDETERMINATE, "identity_changed", log_path=log_path)

        offset = _find_process_start_log_boundary(
            file_descriptor,
            initial_stat.st_size,
            process_start,
        )
        if offset is None:
            return LogCheckResult(
                LOG_INDETERMINATE,
                "process_start_boundary_not_found",
                log_path=log_path,
                process_start_timestamp=format_timestamp(process_start),
            )
        scan_size = initial_stat.st_size - offset
        with os.fdopen(file_descriptor, "rb") as handle:
            file_descriptor = None
            handle.seek(offset)
            content = handle.read(scan_size)
            final_stat = os.fstat(handle.fileno())
        final_path_stat = os.stat(log_path, follow_symlinks=False)
        if (
            len(content) != scan_size
            or (final_stat.st_dev, final_stat.st_ino) != identity
            or (final_path_stat.st_dev, final_path_stat.st_ino) != identity
            or final_stat.st_size != initial_stat.st_size
            or final_path_stat.st_size != initial_stat.st_size
        ):
            return LogCheckResult(
                LOG_INDETERMINATE,
                "changed_during_read",
                log_path=log_path,
            )
    except OSError:
        return LogCheckResult(LOG_INDETERMINATE, "unreadable", log_path=log_path)
    finally:
        if file_descriptor is not None:
            try:
                os.close(file_descriptor)
            except OSError:
                pass

    result = classify_candidate_log_bytes(content, process_start)
    return replace(result, log_path=log_path)


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
    parser.add_argument("--startup-settle-seconds", type=float, default=0.0)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--verification-only", action="store_true")
    parser.add_argument("--create-log-baseline", action="store_true")
    parser.add_argument("--baseline-source-names")
    parser.add_argument("--log-baseline")
    return parser


def print_log_diagnostic(
    process_name: str,
    process: dict[str, Any],
    result: LogCheckResult,
) -> None:
    log_path = result.log_path or pm2_env(process).get("pm_err_log_path")
    detail = {
        "process": process_name,
        "log_file": Path(log_path).name if isinstance(log_path, str) else "unknown",
        "classification": result.classification or result.status,
        "first_timestamp": result.first_timestamp,
        "process_start_timestamp": result.process_start_timestamp
        or format_timestamp(pm_process_start_timestamp(process)),
        "matched_rule": result.matched_rule or result.reason,
        "excerpt": result.excerpt,
    }
    print("startup_log_detail=" + json.dumps(detail, ensure_ascii=True, sort_keys=True))


def main() -> int:
    args = build_parser().parse_args()
    expected_names = parse_names(args.pm2_names)

    if args.create_log_baseline:
        if args.verification_only:
            print("log_baseline=failed", file=sys.stderr)
            return 1
        try:
            snapshot = load_pm2_snapshot()
            source_names = (
                parse_names(args.baseline_source_names)
                if args.baseline_source_names
                else None
            )
            baseline_path = create_log_baseline(
                snapshot,
                expected_names,
                source_names,
            )
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

    if args.startup_settle_seconds > 0:
        time.sleep(args.startup_settle_seconds)
        try:
            first_snapshot = load_pm2_snapshot()
        except RuntimeError:
            first_snapshot = {}
            errors.append("pm2_unavailable_after_startup_settle")
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
    if not args.verification_only and not args.log_baseline:
        errors.append("log_baseline_required")
    elif args.log_baseline:
        try:
            log_baseline = load_log_baseline(args.log_baseline)
        except RuntimeError:
            errors.append("log_baseline_unavailable")

    log_error_names: list[str] = []
    log_indeterminate: list[str] = []
    log_diagnostics: list[tuple[str, dict[str, Any], LogCheckResult]] = []
    for name in expected_names:
        process = second_snapshot.get(name)
        if process is None:
            continue
        if args.verification_only:
            result = recent_startup_error_since_process_start(process)
        else:
            result = recent_startup_error(process, log_baseline.get(name))
        if result.status == LOG_ERROR:
            log_error_names.append(name)
            log_diagnostics.append((name, process, result))
        elif result.status == LOG_INDETERMINATE:
            log_indeterminate.append(f"{name}:{result.reason}")
            log_diagnostics.append((name, process, result))
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
    print(f"runtime={'ok' if not second_pm2_errors else 'failed'}")
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
    if stability_errors:
        print("stability_errors=" + ",".join(stability_errors))
    print(f"migration={'ok' if not migration_errors else 'failed'}")
    if log_indeterminate:
        print("startup_log_indeterminate=" + ",".join(log_indeterminate))
    for process_name, process, result in log_diagnostics:
        print_log_diagnostic(process_name, process, result)
    if migration_errors:
        print("migration_errors=" + ",".join(migration_errors))
    if errors:
        print("verification=failed")
        return 1
    print("verification=ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
