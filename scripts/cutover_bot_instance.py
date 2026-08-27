#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

try:
    from scripts import verify_bot_instances as verifier
    from scripts import verify_prod_runtime as runtime_verifier
except ModuleNotFoundError:
    import verify_bot_instances as verifier
    import verify_prod_runtime as runtime_verifier


class CutoverError(RuntimeError):
    pass


def run_pm2(arguments: list[str]) -> None:
    subprocess.run(
        ["pm2", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def process_token(process: dict[str, Any]) -> str | None:
    environment = verifier.process_env(process)
    return environment.get("BOT_TOKEN") or environment.get("MAX_BOT_TOKEN")


def duplicate_active_tokens(snapshot: dict[str, dict[str, Any]]) -> list[str]:
    groups: dict[tuple[str, str], set[str]] = {}
    for name, process in snapshot.items():
        if verifier.process_status(process) != "online":
            continue
        token = process_token(process)
        if not token:
            continue
        platform = "max" if not verifier.process_env(process).get("BOT_TOKEN") else "telegram"
        groups.setdefault((platform, verifier.fingerprint_token(token) or ""), set()).add(name)
    return [
        f"duplicate_active_token:{platform}:{fingerprint}:{','.join(sorted(names))}"
        for (platform, fingerprint), names in groups.items()
        if len(names) > 1
    ]


def verify_legacy_mapping(
    entry: dict[str, Any],
    registry: dict[str, Any],
    snapshot: dict[str, dict[str, Any]],
    environment: dict[str, str],
) -> None:
    legacy_name = entry.get("legacy_pm2_name")
    process = snapshot.get(legacy_name) if legacy_name else None
    if process is None:
        raise CutoverError(f"missing_legacy_process:{entry['pm2_name']}")
    if verifier.process_status(process) != "online":
        raise CutoverError(f"legacy_not_online:{legacy_name}")
    process_environment = verifier.process_env(process)
    expected_token = environment.get(entry["token_env"])
    actual_token = process_environment.get(entry["runtime_token_env"])
    if not expected_token:
        raise CutoverError(f"missing_runtime_secret:{entry['token_env']}")
    if not actual_token:
        raise CutoverError(f"missing_process_secret:{legacy_name}")
    if expected_token != actual_token:
        raise CutoverError(f"token_fingerprint_mismatch:{legacy_name}")
    expected_database = environment.get(entry["database_env"])
    actual_database = process_environment.get(entry["runtime_database_env"])
    if not expected_database:
        raise CutoverError(f"missing_runtime_secret:{entry['database_env']}")
    if not actual_database:
        raise CutoverError(f"missing_process_database:{legacy_name}")
    if expected_database != actual_database:
        raise CutoverError(f"database_url_mismatch:{legacy_name}")
    if verifier.database_name(actual_database) != entry["database"]:
        raise CutoverError(f"database_name_mismatch:{legacy_name}")
    unexpected = verifier.unexpected_managed_processes(
        snapshot,
        registry,
        migration_aware=True,
    )
    if unexpected:
        raise CutoverError("unexpected_managed_process:" + ",".join(unexpected))
    duplicate_tokens = duplicate_active_tokens(snapshot)
    if duplicate_tokens:
        raise CutoverError(",".join(duplicate_tokens))


def verify_canonical_mapping(
    entry: dict[str, Any],
    registry: dict[str, Any],
    snapshot: dict[str, dict[str, Any]],
    environment: dict[str, str],
    *,
    cutover: bool,
) -> None:
    _, errors = verifier.runtime_rows(
        registry,
        snapshot,
        environment,
        names=[entry["pm2_name"]],
        migration_aware=True,
        cutover_pm2_name=entry["pm2_name"] if cutover else None,
    )
    if errors:
        raise CutoverError("canonical_verification_failed:" + ",".join(errors))


def wait_for_online(
    name: str,
    snapshot_loader: Callable[[], dict[str, dict[str, Any]]],
    *,
    attempts: int = 20,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, dict[str, Any]]:
    for attempt in range(attempts):
        snapshot = snapshot_loader()
        process = snapshot.get(name)
        if process is not None and verifier.process_status(process) == "online":
            return snapshot
        if attempt + 1 < attempts:
            sleep_fn(0.5)
    raise CutoverError(f"process_not_online:{name}")


def restart_legacy(
    entry: dict[str, Any],
    registry: dict[str, Any],
    environment: dict[str, str],
    snapshot_loader: Callable[[], dict[str, dict[str, Any]]],
    pm2_runner: Callable[[list[str]], None],
) -> list[str]:
    errors: list[str] = []
    legacy_name = entry.get("legacy_pm2_name")
    if not legacy_name:
        return [f"missing_legacy_process:{entry['pm2_name']}"]
    try:
        pm2_runner(["restart", legacy_name])
        snapshot = wait_for_online(legacy_name, snapshot_loader)
        verify_legacy_mapping(entry, registry, snapshot, environment)
    except Exception as exc:
        errors.append(f"legacy_rollback_failed:{legacy_name}:{type(exc).__name__}")
    return errors


def rollback_canonical(
    canonical_name: str,
    snapshot_loader: Callable[[], dict[str, dict[str, Any]]],
    pm2_runner: Callable[[list[str]], None],
) -> list[str]:
    errors: list[str] = []
    snapshot: dict[str, dict[str, Any]] | None
    try:
        snapshot = snapshot_loader()
    except Exception:
        snapshot = None
        errors.append(f"canonical_rollback_snapshot_unavailable:{canonical_name}")
    if snapshot is not None and canonical_name not in snapshot:
        return errors
    try:
        pm2_runner(["stop", canonical_name])
    except Exception:
        errors.append(f"canonical_rollback_stop_failed:{canonical_name}")
    try:
        pm2_runner(["delete", canonical_name])
    except Exception:
        errors.append(f"canonical_rollback_delete_failed:{canonical_name}")
    return errors


def run_runtime_check(
    root: Path,
    revision: str,
    canonical_name: str,
    baseline_path: str,
    settle_seconds: float,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "verify_prod_runtime.py"),
            "--revision",
            revision,
            "--pm2-names",
            canonical_name,
            "--root",
            str(root),
            "--settle-seconds",
            str(settle_seconds),
            "--log-baseline",
            baseline_path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CutoverError("production_runtime_verification_failed")


def cutover_instance(
    entry: dict[str, Any],
    registry: dict[str, Any],
    environment: dict[str, str],
    *,
    config_path: str,
    root: Path,
    revision: str,
    allow_rename: bool,
    settle_seconds: float = 3.0,
    snapshot_loader: Callable[[], dict[str, dict[str, Any]]] = verifier.load_pm2_snapshot,
    pm2_runner: Callable[[list[str]], None] = run_pm2,
    runtime_checker: Callable[[Path, str, str, str, float], None] = run_runtime_check,
) -> str:
    canonical_name = entry["pm2_name"]
    legacy_name = entry.get("legacy_pm2_name")
    if not legacy_name:
        verify_canonical_mapping(entry, registry, snapshot_loader(), environment, cutover=False)
        return "already_migrated"

    initial_snapshot = snapshot_loader()
    canonical_process = initial_snapshot.get(canonical_name)
    legacy_process = initial_snapshot.get(legacy_name)
    canonical_online = (
        canonical_process is not None
        and verifier.process_status(canonical_process) == "online"
    )
    legacy_online = (
        legacy_process is not None and verifier.process_status(legacy_process) == "online"
    )
    if canonical_online and legacy_online:
        raise CutoverError(f"both_migration_processes_active:{canonical_name}:{legacy_name}")
    if canonical_online and legacy_process is None:
        verify_canonical_mapping(entry, registry, initial_snapshot, environment, cutover=False)
        return "already_migrated"
    if legacy_process is None:
        raise CutoverError(f"missing_legacy_process:{canonical_name}")
    if not legacy_online and not canonical_online:
        raise CutoverError(f"legacy_not_online:{legacy_name}")
    if not allow_rename:
        raise CutoverError(f"rename_not_allowed:{legacy_name}")

    if legacy_online:
        verify_legacy_mapping(entry, registry, initial_snapshot, environment)
    baseline_path: str | None = None
    legacy_deleted = False
    try:
        baseline_path = runtime_verifier.create_log_baseline(
            initial_snapshot,
            [canonical_name],
            [legacy_name],
        )
        if legacy_online:
            pm2_runner(["stop", legacy_name])
            stopped_snapshot = snapshot_loader()
            if (
                legacy_name in stopped_snapshot
                and verifier.process_status(stopped_snapshot[legacy_name]) == "online"
            ):
                raise CutoverError(f"legacy_stop_failed:{legacy_name}")
        if not canonical_online:
            pm2_runner(
                [
                    "startOrReload",
                    config_path,
                    "--only",
                    canonical_name,
                    "--update-env",
                ]
            )
        online_snapshot = wait_for_online(canonical_name, snapshot_loader)
        verify_canonical_mapping(
            entry,
            registry,
            online_snapshot,
            environment,
            cutover=True,
        )
        runtime_checker(
            root,
            revision,
            canonical_name,
            baseline_path,
            settle_seconds,
        )
        pm2_runner(["delete", legacy_name])
        legacy_deleted = True
        final_snapshot = snapshot_loader()
        if legacy_name in final_snapshot:
            raise CutoverError(f"legacy_process_remains:{legacy_name}")
        verify_canonical_mapping(
            entry,
            registry,
            final_snapshot,
            environment,
            cutover=False,
        )
        return "migrated"
    except Exception as exc:
        if legacy_deleted:
            raise CutoverError(f"cutover_cleanup_failed:{type(exc).__name__}") from exc
        rollback_errors = rollback_canonical(canonical_name, snapshot_loader, pm2_runner)
        rollback_errors.extend(
            restart_legacy(
                entry,
                registry,
                environment,
                snapshot_loader,
                pm2_runner,
            )
        )
        if rollback_errors:
            raise CutoverError(
                f"cutover_failed:{type(exc).__name__}:" + ",".join(rollback_errors)
            ) from exc
        raise CutoverError(f"cutover_failed:{type(exc).__name__}") from exc
    finally:
        if baseline_path is not None:
            try:
                Path(baseline_path).unlink()
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pm2-name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--registry", default=str(verifier.DEFAULT_REGISTRY))
    parser.add_argument("--runtime-env", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--allow-rename", action="store_true")
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = verifier.load_registry(args.registry)
        entries, errors = verifier.selected_instances(registry, [args.pm2_name])
        if errors:
            raise CutoverError(",".join(errors))
        if not entries:
            raise CutoverError(f"unregistered_pm2_name:{args.pm2_name}")
        environment = verifier.canonical_environment(runtime_env_path=args.runtime_env)
        result = cutover_instance(
            entries[0],
            registry,
            environment,
            config_path=args.config,
            root=Path(args.root),
            revision=args.revision,
            allow_rename=args.allow_rename,
            settle_seconds=args.settle_seconds,
        )
    except (CutoverError, OSError, RuntimeError, ValueError) as exc:
        print(f"cutover=failed reason={exc}")
        return 1
    print(f"cutover=ok pm2={args.pm2_name} result={result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
