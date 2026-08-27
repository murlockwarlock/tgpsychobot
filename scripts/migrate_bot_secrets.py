#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile
import time
from typing import Any

try:
    from scripts import verify_bot_instances as verifier
except ModuleNotFoundError:
    import verify_bot_instances as verifier


DEFAULT_RUNTIME_ENV = "/root/telegram_bots/runtime.env"
ASSIGNMENT_RE = re.compile(
    r"^(\s*(?:export\s+)?)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*)$"
)


def load_pm2_snapshot() -> dict[str, dict[str, Any]]:
    return verifier.load_pm2_snapshot()


def snapshot_safety_errors(
    snapshot: dict[str, dict[str, Any]],
    registry: dict[str, Any],
) -> list[str]:
    errors = [
        f"unexpected_managed_process:{name}"
        for name in verifier.unexpected_managed_processes(
            snapshot,
            registry,
            migration_aware=True,
        )
    ]
    token_groups: dict[tuple[str, str], set[str]] = {}
    for name, process in snapshot.items():
        if verifier.process_status(process) != "online":
            continue
        process_environment = verifier.process_env(process)
        token = process_environment.get("BOT_TOKEN") or process_environment.get("MAX_BOT_TOKEN")
        if not token:
            continue
        platform = "telegram" if process_environment.get("BOT_TOKEN") else "max"
        token_groups.setdefault(
            (platform, verifier.fingerprint_token(token) or ""),
            set(),
        ).add(name)
    errors.extend(
        f"duplicate_active_token:{platform}:{fingerprint}:{','.join(sorted(names))}"
        for (platform, fingerprint), names in token_groups.items()
        if len(names) > 1
    )
    return sorted(errors)


def env_assignments(text: str) -> dict[str, list[tuple[int, str, str, str]]]:
    result: dict[str, list[tuple[int, str, str, str]]] = {}
    for index, raw_line in enumerate(text.splitlines(keepends=True)):
        body = raw_line.rstrip("\r\n")
        newline = raw_line[len(body):]
        match = ASSIGNMENT_RE.fullmatch(body)
        if not match:
            continue
        key = match.group(2)
        result.setdefault(key, []).append(
            (index, match.group(1), match.group(3), newline or "\n")
        )
    return result


def updated_env_text(
    text: str,
    updates: dict[str, str],
) -> str:
    assignments = env_assignments(text)
    duplicate_keys = sorted(key for key in updates if len(assignments.get(key, [])) > 1)
    if duplicate_keys:
        raise ValueError("duplicate_runtime_env_key:" + ",".join(duplicate_keys))
    lines = text.splitlines(keepends=True)
    replaced: set[str] = set()
    for key, records in assignments.items():
        if key not in updates:
            continue
        index, prefix, separator, newline = records[0]
        lines[index] = f"{prefix}{key}{separator}{shlex.quote(updates[key])}{newline}"
        replaced.add(key)
    missing = [key for key in updates if key not in replaced]
    if missing:
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines[-1] += "\n"
        lines.extend(f"{key}={shlex.quote(updates[key])}\n" for key in missing)
    return "".join(lines)


def create_backup(path: Path, content: bytes) -> Path:
    for attempt in range(10):
        candidate = path.with_name(
            f"{path.name}.bak.{time.time_ns()}.{attempt}"
        )
        try:
            descriptor = os.open(
                candidate,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            continue
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(candidate, 0o600)
            return candidate
        except BaseException:
            try:
                candidate.unlink()
            except OSError:
                pass
            raise
    raise RuntimeError("unable_to_create_runtime_env_backup")


def atomic_write(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary_path.unlink()
        except OSError:
            pass


def selected_entries(
    registry: dict[str, Any],
    names: list[str] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    return verifier.selected_instances(registry, names, allow_legacy=True)


def process_source(
    entry: dict[str, Any],
    snapshot: dict[str, dict[str, Any]],
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    canonical_name = entry["pm2_name"]
    legacy_name = entry.get("legacy_pm2_name")
    canonical = snapshot.get(canonical_name)
    legacy = snapshot.get(legacy_name) if legacy_name else None
    canonical_online = canonical is not None and verifier.process_status(canonical) == "online"
    legacy_online = legacy is not None and verifier.process_status(legacy) == "online"
    if canonical_online and legacy_online:
        return None, None, f"both_processes_active:{canonical_name}:{legacy_name}"
    if legacy_online:
        return legacy_name, legacy, None
    if canonical_online:
        return canonical_name, canonical, None
    if legacy is not None:
        return legacy_name, legacy, f"legacy_not_online:{legacy_name}"
    if canonical is not None:
        return canonical_name, canonical, f"canonical_not_online:{canonical_name}"
    return None, None, f"missing_pm2_process:{canonical_name}"


def plan_entry(
    entry: dict[str, Any],
    snapshot: dict[str, dict[str, Any]],
    values: dict[str, str],
    assignments: dict[str, list[tuple[int, str, str, str]]],
) -> tuple[dict[str, str], list[str]]:
    source_name, process, source_error = process_source(entry, snapshot)
    row = {
        "platform": str(entry["platform"]),
        "username": str(entry["username"]),
        "old_pm2": str(entry.get("legacy_pm2_name") or "-"),
        "canonical_pm2": str(entry["pm2_name"]),
        "token_env": str(entry["token_env"]),
        "token_fingerprint": "-",
        "database_env": str(entry["database_env"]),
        "database": str(entry["database"]),
        "action": source_error or "ready",
    }
    errors: list[str] = []
    if source_error:
        if source_name and process and verifier.process_status(process) == "online":
            errors.append(source_error)
        elif source_error.startswith("both_processes_active"):
            errors.append(source_error)
        else:
            return row, [source_error]
    if process is None or source_name is None:
        return row, errors or [f"missing_pm2_process:{entry['pm2_name']}"]

    process_environment = verifier.process_env(process)
    token = process_environment.get(entry["runtime_token_env"])
    database_url = process_environment.get(entry["runtime_database_env"])
    if token:
        row["token_fingerprint"] = verifier.fingerprint_token(token) or "-"
    if not token:
        errors.append(f"missing_process_secret:{source_name}:{entry['runtime_token_env']}")
    if not database_url:
        errors.append(f"missing_process_database:{source_name}")
    if database_url and verifier.database_name(database_url) != entry["database"]:
        errors.append(f"database_name_mismatch:{source_name}")

    if errors:
        row["action"] = "source_invalid"
        return row, errors

    canonical_token = values.get(entry["token_env"])
    canonical_database = values.get(entry["database_env"])
    for key in (entry["token_env"], entry["database_env"]):
        if key in assignments and key not in values:
            errors.append(f"unreadable_runtime_env_key:{key}")
    if canonical_token is not None and canonical_token != token:
        errors.append(f"canonical_value_conflict:{entry['token_env']}")
    if canonical_database is not None and canonical_database != database_url:
        errors.append(f"canonical_value_conflict:{entry['database_env']}")
    if errors:
        row["action"] = "conflict"
        return row, errors

    if source_name == entry["pm2_name"] and (
        canonical_token != token or canonical_database != database_url
    ):
        row["action"] = "legacy_source_unavailable"
        return row, [f"legacy_source_unavailable:{entry['pm2_name']}"]

    if canonical_token == token and canonical_database == database_url:
        row["action"] = "already_present" if source_name != entry["pm2_name"] else "already_migrated"
    else:
        row["action"] = "ready"
    return row, []


def print_plan(rows: list[dict[str, str]]) -> None:
    keys = (
        "platform",
        "username",
        "old_pm2",
        "canonical_pm2",
        "token_env",
        "token_fingerprint",
        "database_env",
        "database",
        "action",
    )
    print("\t".join(keys))
    for row in rows:
        print("\t".join(row[key] for key in keys))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--apply", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--pm2-name")
    selection.add_argument("--pm2-names")
    parser.add_argument("--registry", default=str(verifier.DEFAULT_REGISTRY))
    parser.add_argument("--runtime-env", default=os.environ.get("PROD_RUNTIME_ENV", DEFAULT_RUNTIME_ENV))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = verifier.load_registry(args.registry)
        runtime_path = Path(args.runtime_env)
        original_content = runtime_path.read_bytes()
        original_text = original_content.decode("utf-8")
        values = verifier.parse_env_file(runtime_path)
        assignments = env_assignments(original_text)
        names = None
        if args.pm2_name:
            names = [args.pm2_name]
        elif args.pm2_names:
            names = verifier.parse_names(args.pm2_names)
        entries, selection_errors = selected_entries(registry, names)
        snapshot = load_pm2_snapshot()
    except (OSError, UnicodeError, RuntimeError, ValueError) as exc:
        print(f"migration=failed reason={exc}")
        return 1

    rows: list[dict[str, str]] = []
    errors = list(selection_errors)
    errors.extend(snapshot_safety_errors(snapshot, registry))
    plans: list[tuple[dict[str, Any], dict[str, str]]] = []
    for entry in entries:
        row, entry_errors = plan_entry(entry, snapshot, values, assignments)
        rows.append(row)
        errors.extend(entry_errors)
        if row["action"] == "ready":
            source_name, process, source_error = process_source(entry, snapshot)
            if source_error or source_name != entry.get("legacy_pm2_name") or process is None:
                continue
            process_environment = verifier.process_env(process)
            updates = {
                entry["token_env"]: process_environment[entry["runtime_token_env"]],
                entry["database_env"]: process_environment[entry["runtime_database_env"]],
            }
            plans.append((entry, updates))

    print_plan(rows)
    if errors:
        print("migration=failed")
        print("errors=" + ",".join(errors))
        return 1
    if args.plan:
        print("migration=plan")
        return 0

    updates: dict[str, str] = {}
    for _entry, entry_updates in plans:
        updates.update(entry_updates)
    if not updates:
        print("migration=unchanged")
        return 0

    backup: Path | None = None
    try:
        new_text = updated_env_text(original_text, updates)
        backup = create_backup(runtime_path, original_content)
        atomic_write(runtime_path, new_text.encode("utf-8"))
        resulting_values = verifier.parse_env_file(runtime_path)
        for key, value in updates.items():
            if resulting_values.get(key) != value:
                raise RuntimeError(f"runtime_env_verification_failed:{key}")
        for entry, entry_updates in plans:
            database_url = entry_updates[entry["database_env"]]
            if verifier.database_name(database_url) != entry["database"]:
                raise RuntimeError(f"database_name_verification_failed:{entry['pm2_name']}")
            if verifier.fingerprint_token(entry_updates[entry["token_env"]]) is None:
                raise RuntimeError(f"token_fingerprint_verification_failed:{entry['pm2_name']}")
    except (OSError, RuntimeError, ValueError) as exc:
        if backup is not None:
            try:
                atomic_write(runtime_path, original_content)
            except OSError:
                pass
        print(f"migration=failed reason={exc}")
        return 1

    print(f"migration=applied entries={len(plans)} backup={backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
