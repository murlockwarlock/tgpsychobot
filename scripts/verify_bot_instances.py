#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "config" / "bot_instances.json"
PLATFORMS = {"telegram", "max"}
MODES = {"new", "legacy", "fulltest"}
EXPECTED_DEPLOY_MANAGED_COUNT = 18
TOKEN_VALUE_RE = re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{20,}\b")
SECRET_KEY_RE = re.compile(r"(?:password|secret|private|api[_-]?key)", re.IGNORECASE)


def parse_names(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    try:
        registry = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read registry: {path}") from exc
    errors = validate_registry(registry)
    if errors:
        raise ValueError("; ".join(errors))
    return registry


def validate_registry(registry: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(registry, dict):
        return ["registry_not_object"]
    if registry.get("schema_version") != 1:
        errors.append("schema_version_invalid")
    instances = registry.get("instances")
    if not isinstance(instances, list) or not instances:
        return errors + ["instances_not_nonempty_list"]

    seen_identity: dict[tuple[str, str], str] = {}
    seen_identity_database: dict[tuple[str, str], str] = {}
    seen_ids: dict[tuple[str, int], str] = {}
    seen_pm2: dict[str, str] = {}
    seen_legacy_pm2: dict[str, str] = {}
    seen_token_env: dict[str, str] = {}
    for index, entry in enumerate(instances):
        prefix = f"instances[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{prefix}:not_object")
            continue
        required = {
            "platform",
            "username",
            "platform_id",
            "pm2_name",
            "database",
            "database_env",
            "runtime_database_env",
            "token_env",
            "runtime_token_env",
            "mode",
            "port",
            "profile",
            "deploy_managed",
            "active",
        }
        missing = sorted(required - entry.keys())
        if missing:
            errors.append(f"{prefix}:missing={','.join(missing)}")
            continue

        platform = entry["platform"]
        username = entry["username"]
        platform_id = entry["platform_id"]
        pm2_name = entry["pm2_name"]
        token_env = entry["token_env"]
        if platform not in PLATFORMS:
            errors.append(f"{prefix}:platform_invalid")
        if not isinstance(username, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,64}", username):
            errors.append(f"{prefix}:username_invalid")
        if isinstance(platform_id, bool) or not isinstance(platform_id, int) or platform_id <= 0:
            errors.append(f"{prefix}:platform_id_invalid")
        if not isinstance(pm2_name, str) or not re.fullmatch(r"(?:tg|max)_[a-z0-9_]+", pm2_name):
            errors.append(f"{prefix}:pm2_name_invalid")
        if not isinstance(entry["database"], str) or not re.fullmatch(
            r"[A-Za-z0-9_]+", entry["database"]
        ):
            errors.append(f"{prefix}:database_invalid")
        if not isinstance(entry["database_env"], str) or not re.fullmatch(
            r"[A-Z][A-Z0-9_]+", entry["database_env"]
        ):
            errors.append(f"{prefix}:database_env_invalid")
        if not isinstance(entry["runtime_database_env"], str) or not re.fullmatch(
            r"[A-Z][A-Z0-9_]+", entry["runtime_database_env"]
        ):
            errors.append(f"{prefix}:runtime_database_env_invalid")
        expected_token_prefix = "TELEGRAM_" if platform == "telegram" else "MAX_"
        canonical_token_env = isinstance(token_env, str) and re.fullmatch(
            expected_token_prefix + r"[A-Z0-9_]+_TOKEN", token_env
        )
        fulltest_token_env = (
            entry.get("mode") == "fulltest"
            and not entry.get("deploy_managed")
            and token_env == "PSY5D_FULLTEST_BOT_TOKEN"
        )
        if not canonical_token_env and not fulltest_token_env:
            errors.append(f"{prefix}:token_env_invalid")
        if not isinstance(entry["runtime_token_env"], str) or not re.fullmatch(
            r"[A-Z][A-Z0-9_]+", entry["runtime_token_env"]
        ):
            errors.append(f"{prefix}:runtime_token_env_invalid")
        if entry["mode"] not in MODES:
            errors.append(f"{prefix}:mode_invalid")
        if isinstance(entry["port"], bool) or not isinstance(entry["port"], int) or entry["port"] <= 0:
            errors.append(f"{prefix}:port_invalid")
        if not isinstance(entry["profile"], str) or not entry["profile"]:
            errors.append(f"{prefix}:profile_invalid")
        if not isinstance(entry["deploy_managed"], bool) or not isinstance(entry["active"], bool):
            errors.append(f"{prefix}:lifecycle_flags_invalid")
        identity_key = (str(platform), str(username))
        if identity_key in seen_identity:
            previous = seen_identity[identity_key]
            if entry["database"] != seen_identity_database[identity_key]:
                errors.append(f"duplicate_identity_conflicting_database:{platform}:{username}")
            else:
                errors.append(f"duplicate_identity:{platform}:{username}")
        else:
            seen_identity[identity_key] = str(pm2_name)
            seen_identity_database[identity_key] = str(entry["database"])

        id_key = (str(platform), platform_id) if isinstance(platform_id, int) else None
        if id_key and id_key in seen_ids:
            errors.append(f"duplicate_platform_id:{platform}:{platform_id}")
        elif id_key:
            seen_ids[id_key] = str(pm2_name)

        if pm2_name in seen_pm2 or pm2_name in seen_legacy_pm2:
            errors.append(f"duplicate_pm2_name:{pm2_name}")
        else:
            seen_pm2[pm2_name] = str(username)

        legacy_pm2_name = entry.get("legacy_pm2_name")
        if legacy_pm2_name is not None:
            if not isinstance(legacy_pm2_name, str) or not re.fullmatch(
                r"[A-Za-z0-9_-]+", legacy_pm2_name
            ):
                errors.append(f"{prefix}:legacy_pm2_name_invalid")
            elif legacy_pm2_name == pm2_name:
                errors.append(f"{prefix}:legacy_pm2_name_matches_pm2_name")
            elif legacy_pm2_name in seen_pm2 or legacy_pm2_name in seen_legacy_pm2:
                errors.append(f"duplicate_legacy_pm2_name:{legacy_pm2_name}")
            else:
                seen_legacy_pm2[legacy_pm2_name] = str(username)

        if isinstance(token_env, str) and token_env in seen_token_env:
            errors.append(f"duplicate_token_env:{token_env}")
        elif isinstance(token_env, str):
            seen_token_env[token_env] = str(pm2_name)

        for key, value in entry.items():
            if isinstance(value, str):
                if TOKEN_VALUE_RE.search(value) or "postgres" in value.lower() and "://" in value:
                    errors.append(f"{prefix}:secret_value_in_{key}")
                if SECRET_KEY_RE.search(key) and value and not key.endswith("_env"):
                    errors.append(f"{prefix}:secret_value_in_{key}")

    external = registry.get("external_pm2_processes", [])
    if not isinstance(external, list):
        errors.append("external_pm2_processes_not_list")
    return errors


def managed_instances(registry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        entry
        for entry in registry["instances"]
        if entry.get("deploy_managed") and entry.get("active")
    ]


def selected_instances(
    registry: dict[str, Any],
    names: list[str] | None = None,
    *,
    allow_legacy: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    if names is None:
        return managed_instances(registry), []
    by_pm2_name = {entry["pm2_name"]: entry for entry in registry["instances"]}
    if allow_legacy:
        by_pm2_name.update(
            {
                entry["legacy_pm2_name"]: entry
                for entry in registry["instances"]
                if entry.get("legacy_pm2_name")
            }
        )
    entries: list[dict[str, Any]] = []
    errors: list[str] = []
    for name in names:
        entry = by_pm2_name.get(name)
        if entry is None:
            errors.append(f"unregistered_pm2_name:{name}")
            continue
        entries.append(entry)
    return entries, errors


def registry_pm2_names(registry: dict[str, Any]) -> list[str]:
    return [entry["pm2_name"] for entry in managed_instances(registry)]


def registry_legacy_pm2_names(
    registry: dict[str, Any],
    names: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    entries, errors = selected_instances(registry, names)
    return [entry["legacy_pm2_name"] for entry in entries], errors


def validate_deploy_managed_count(
    registry: dict[str, Any],
    expected: int = EXPECTED_DEPLOY_MANAGED_COUNT,
) -> list[str]:
    actual = len(managed_instances(registry))
    return [] if actual == expected else [f"deploy_managed_count:{actual}"]


def fingerprint_token(token: str | None) -> str | None:
    if not token:
        return None
    return hashlib.sha256(token.encode()).hexdigest()[:12]


def parse_env_file(path: str | Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        result[key] = value
    return result


def load_pm2_snapshot() -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(
            subprocess.check_output(["pm2", "jlist"], text=True)
        )
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RuntimeError("unable to read PM2 process state") from exc
    if not isinstance(payload, list):
        raise RuntimeError("PM2 returned an unexpected process list")
    return {
        str(item["name"]): item
        for item in payload
        if isinstance(item, dict) and item.get("name")
    }


def process_env(process: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    pm2_env = process.get("pm2_env")
    if isinstance(pm2_env, dict) and isinstance(pm2_env.get("env"), dict):
        result.update(
            {str(key): str(value) for key, value in pm2_env["env"].items() if value is not None}
        )
    pid_value = pm2_env.get("pid") if isinstance(pm2_env, dict) else process.get("pid")
    try:
        pid = int(pid_value)
    except (TypeError, ValueError):
        pid = 0
    if pid <= 0:
        return result
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return result
    for item in raw.split(b"\0"):
        key, separator, value = item.partition(b"=")
        if not separator:
            continue
        try:
            result[key.decode()] = value.decode()
        except UnicodeDecodeError:
            continue
    return result


def process_status(process: dict[str, Any]) -> str | None:
    pm2_env = process.get("pm2_env")
    if isinstance(pm2_env, dict) and pm2_env.get("status"):
        return str(pm2_env["status"])
    status = process.get("status")
    return str(status) if status else None


def database_name(database_url: str | None) -> str | None:
    if not database_url:
        return None
    parsed = urlsplit(database_url)
    if parsed.path:
        return parsed.path.rsplit("/", 1)[-1] or None
    return None


def canonical_environment(
    environment: dict[str, str] | None = None,
    runtime_env_path: str | Path | None = None,
) -> dict[str, str]:
    result = dict(os.environ if environment is None else environment)
    if runtime_env_path:
        values = parse_env_file(runtime_env_path)
        for key, value in values.items():
            result.setdefault(key, value)
    return result


def proxy_candidates(environment: dict[str, str]) -> list[str | None]:
    result: list[str | None] = []
    for key in ("TELEGRAM_PROXY", "HTTPS_PROXY", "HTTP_PROXY"):
        value = environment.get(key)
        if value and value not in result:
            result.append(value)
    result.append(None)
    return result


def request_json(
    url: str,
    environment: dict[str, str],
    headers: dict[str, str] | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    request = Request(url, headers=headers or {})
    errors: list[Exception] = []
    for proxy in proxy_candidates(environment):
        try:
            opener = build_opener(
                ProxyHandler({"http": proxy, "https": proxy})
            ) if proxy else build_opener(ProxyHandler({}))
            with opener.open(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode())
            if not isinstance(payload, dict):
                raise ValueError("identity response is not an object")
            return payload
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(exc)
    if errors:
        raise errors[-1]
    raise URLError("identity request unavailable")


def verify_identity(
    entry: dict[str, Any],
    token: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    try:
        if entry["platform"] == "telegram":
            payload = request_json(
                f"https://api.telegram.org/bot{token}/getMe",
                environment,
            )
            if not payload.get("ok") or not isinstance(payload.get("result"), dict):
                return {"status": "unavailable"}
            result = payload["result"]
            actual_username = str(result.get("username") or "").lstrip("@").lower()
            actual_id = result.get("id")
        else:
            payload = request_json(
                "https://platform-api.max.ru/me",
                environment,
                headers={"Authorization": token},
            )
            actual_username = str(payload.get("name") or "").lstrip("@").lower()
            actual_id = payload.get("user_id", payload.get("id"))
    except (HTTPError, OSError, ValueError, json.JSONDecodeError):
        return {"status": "unavailable"}

    if actual_username != str(entry["username"]).lower() or actual_id != entry["platform_id"]:
        return {
            "status": "mismatch",
            "username": actual_username,
            "platform_id": actual_id,
        }
    return {"status": "verified", "username": actual_username, "platform_id": actual_id}


def managed_runtime_process(process: dict[str, Any]) -> bool:
    pm2_env = process.get("pm2_env")
    if not isinstance(pm2_env, dict):
        pm2_env = {}
    path = str(pm2_env.get("pm_exec_path") or process.get("pm_exec_path") or "")
    cwd = str(pm2_env.get("pm_cwd") or process.get("pm_cwd") or "")
    return "/newbots/" in path or cwd.endswith("/newbots")


def unexpected_managed_processes(
    snapshot: dict[str, dict[str, Any]],
    registry: dict[str, Any],
    *,
    allow_legacy: bool = False,
    migration_aware: bool = False,
) -> list[str]:
    known = {
        entry["pm2_name"]
        for entry in registry["instances"]
    }
    known.update(
        entry["legacy_pm2_name"]
        for entry in registry["instances"]
        if not entry.get("deploy_managed")
        and entry.get("legacy_pm2_name")
        and (
            entry["legacy_pm2_name"] not in snapshot
            or process_status(snapshot[entry["legacy_pm2_name"]]) != "online"
        )
    )
    if allow_legacy or migration_aware:
        known.update(
            entry.get("legacy_pm2_name")
            for entry in registry["instances"]
            if entry.get("legacy_pm2_name")
        )
    return sorted(
        name
        for name, process in snapshot.items()
        if managed_runtime_process(process) and name not in known
    )


def migration_process(
    entry: dict[str, Any],
    snapshot: dict[str, dict[str, Any]],
    *,
    cutover_pm2_name: str | None = None,
) -> tuple[str | None, dict[str, Any] | None, list[str]]:
    canonical_name = entry["pm2_name"]
    legacy_name = entry.get("legacy_pm2_name")
    canonical_process = snapshot.get(canonical_name)
    legacy_process = snapshot.get(legacy_name) if legacy_name else None
    canonical_online = (
        canonical_process is not None and process_status(canonical_process) == "online"
    )
    legacy_online = (
        legacy_process is not None and process_status(legacy_process) == "online"
    )
    errors: list[str] = []

    if canonical_online and legacy_online:
        errors.append(f"both_migration_processes_active:{canonical_name}:{legacy_name}")
        return canonical_name, canonical_process, errors

    if canonical_process is not None and legacy_process is not None:
        if canonical_name != cutover_pm2_name:
            errors.append(f"legacy_process_present_after_migration:{legacy_name}")
        return canonical_name, canonical_process, errors
    if canonical_process is not None:
        return canonical_name, canonical_process, errors
    if legacy_process is not None:
        return legacy_name, legacy_process, errors
    return None, None, errors


def validate_runtime_env(
    registry: dict[str, Any],
    environment: dict[str, str],
    names: list[str] | None = None,
    *,
    allow_legacy: bool = False,
) -> list[str]:
    errors: list[str] = []
    entries, selection_errors = selected_instances(
        registry,
        names,
        allow_legacy=allow_legacy,
    )
    errors.extend(selection_errors)
    for entry in entries:
        if names is not None and not entry.get("deploy_managed") and not allow_legacy:
            if entry.get("active") is not False:
                errors.append(f"optional_process_not_inactive:{entry['pm2_name']}")
        for field in ("token_env", "database_env"):
            key = entry[field]
            if not environment.get(key) and not allow_legacy:
                errors.append(f"missing_runtime_secret:{key}")
    return errors


def runtime_rows(
    registry: dict[str, Any],
    snapshot: dict[str, dict[str, Any]],
    environment: dict[str, str],
    *,
    allow_legacy: bool = False,
    identity_check: bool = False,
    names: list[str] | None = None,
    allow_missing: bool = False,
    migration_aware: bool = False,
    cutover_pm2_name: str | None = None,
    steady_state: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    token_groups: dict[tuple[str, str], set[str]] = {}
    entries, selection_errors = selected_instances(
        registry,
        names,
        allow_legacy=allow_legacy and not steady_state,
    )
    errors.extend(selection_errors)
    for entry in entries:
        name = entry["pm2_name"]
        process_name = name
        process = snapshot.get(name)
        migration_errors: list[str] = []
        if steady_state:
            legacy_name = entry.get("legacy_pm2_name")
            if legacy_name and legacy_name in snapshot:
                errors.append(f"legacy_process_present_after_migration:{legacy_name}")
        elif migration_aware:
            process_name, process, migration_errors = migration_process(
                entry,
                snapshot,
                cutover_pm2_name=cutover_pm2_name,
            )
            errors.extend(migration_errors)
        elif process is None and allow_legacy:
            legacy_name = entry.get("legacy_pm2_name")
            process = snapshot.get(legacy_name) if legacy_name else None
            process_name = legacy_name or name
        if process is None:
            if not allow_missing:
                errors.append(f"missing_pm2_process:{name}")
            rows.append(
                {
                    "platform": entry["platform"],
                    "username": entry["username"],
                    "pm2_name": name,
                    "database": entry["database"],
                    "token_env": entry["token_env"],
                    "token_fingerprint": None,
                    "identity": "missing_process",
                    "runtime_process": None,
                }
            )
            continue
        if process_status(process) != "online":
            errors.append(f"not_online:{process_name}")
        env = process_env(process)
        canonical_token = environment.get(entry["token_env"])
        actual_token = env.get(entry["runtime_token_env"])
        if not canonical_token and not allow_legacy:
            errors.append(f"missing_runtime_secret:{entry['token_env']}")
        if not actual_token:
            errors.append(f"missing_process_secret:{process_name}:{entry['runtime_token_env']}")
        if canonical_token and actual_token and canonical_token != actual_token:
            errors.append(f"token_fingerprint_mismatch:{process_name}")
        if actual_token:
            token_groups.setdefault(
                (entry["platform"], fingerprint_token(actual_token) or ""), set()
            ).add(process_name)

        canonical_database = environment.get(entry["database_env"])
        actual_database = env.get(entry["runtime_database_env"])
        if not canonical_database and not allow_legacy:
            errors.append(f"missing_runtime_secret:{entry['database_env']}")
        if not actual_database:
            errors.append(f"missing_process_database:{process_name}")
        if canonical_database and actual_database and canonical_database != actual_database:
            errors.append(f"database_url_mismatch:{process_name}")
        if actual_database and database_name(actual_database) != entry["database"]:
            errors.append(f"database_name_mismatch:{process_name}")

        identity = "not_requested"
        if identity_check and actual_token:
            identity_result = verify_identity(entry, actual_token, environment)
            identity = identity_result["status"]
            if identity == "mismatch":
                errors.append(f"identity_mismatch:{process_name}")
        rows.append(
            {
                "platform": entry["platform"],
                "username": entry["username"],
                "pm2_name": name,
                "database": entry["database"],
                "token_env": entry["token_env"],
                "token_fingerprint": fingerprint_token(actual_token),
                "identity": identity,
                "runtime_process": process_name,
            }
        )

    if not allow_legacy:
        runtime_token_mapping = {
            process_name: (entry["platform"], entry["runtime_token_env"])
            for entry in registry["instances"]
            for process_name in (entry["pm2_name"], entry.get("legacy_pm2_name"))
            if process_name
        }
        for process_name, process in snapshot.items():
            if process_status(process) != "online":
                continue
            process_environment = process_env(process)
            platform, token_key = runtime_token_mapping.get(
                process_name,
                (
                    "telegram",
                    "BOT_TOKEN",
                )
                if process_environment.get("BOT_TOKEN")
                else ("max", "MAX_BOT_TOKEN"),
            )
            token = process_environment.get(token_key)
            if token:
                token_groups.setdefault(
                    (platform, fingerprint_token(token) or ""), set()
                ).add(process_name)

    for (platform, fingerprint), process_names in token_groups.items():
        if len(process_names) > 1:
            errors.append(
                f"duplicate_active_token:{platform}:{fingerprint}:{','.join(sorted(process_names))}"
            )
    unexpected = unexpected_managed_processes(
        snapshot,
        registry,
        allow_legacy=allow_legacy,
        migration_aware=migration_aware or steady_state,
    )
    errors.extend(f"unexpected_managed_process:{name}" for name in unexpected)
    return rows, errors


def print_rows(rows: list[dict[str, Any]]) -> None:
    print("platform\tusername\tpm2_name\tdatabase\ttoken_env\ttoken_fingerprint\tidentity")
    for row in rows:
        print(
            "\t".join(
                str(row.get(key) or "-")
                for key in (
                    "platform",
                    "username",
                    "pm2_name",
                    "database",
                    "token_env",
                    "token_fingerprint",
                    "identity",
                )
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--runtime-env")
    parser.add_argument("--pm2-names")
    parser.add_argument("--print-pm2-names", action="store_true")
    parser.add_argument("--print-legacy-pm2-names", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate-runtime-env", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--identity-check", action="store_true")
    parser.add_argument("--allow-legacy", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--migration-aware", action="store_true")
    parser.add_argument("--cutover-pm2-name")
    parser.add_argument("--steady-state", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = load_registry(args.registry)
    except ValueError as exc:
        print(f"registry=failed reason={exc}")
        return 1

    count_errors = validate_deploy_managed_count(registry)
    if count_errors:
        print("registry=failed")
        print("errors=" + ",".join(count_errors))
        return 1
    if args.print_pm2_names or args.print_legacy_pm2_names:
        selected_names = parse_names(args.pm2_names) if args.pm2_names else None
        if args.print_legacy_pm2_names:
            names, errors = registry_legacy_pm2_names(registry, selected_names)
            if errors:
                print("registry=failed")
                print("errors=" + ",".join(errors))
                return 1
            print(",".join(names))
            return 0
        print(",".join(registry_pm2_names(registry)))
        return 0
    if args.validate or not (args.validate_runtime_env or args.runtime):
        print(
            f"registry=ok instances={len(registry['instances'])} "
            f"deploy_managed={len(managed_instances(registry))}"
        )
    environment = canonical_environment(runtime_env_path=args.runtime_env)
    selected_names = parse_names(args.pm2_names) if args.pm2_names else None
    if args.validate_runtime_env:
        errors = validate_runtime_env(
            registry,
            environment,
            selected_names,
            allow_legacy=args.allow_legacy,
        )
        if errors:
            print("runtime_env=failed")
            print("errors=" + ",".join(errors))
            return 1
        print("runtime_env=ok")
    if not args.runtime:
        return 0
    try:
        snapshot = load_pm2_snapshot()
    except RuntimeError as exc:
        print(f"runtime=failed reason={exc}")
        return 1
    rows, errors = runtime_rows(
        registry,
        snapshot,
        environment,
        allow_legacy=args.allow_legacy,
        identity_check=args.identity_check,
        names=selected_names,
        allow_missing=args.allow_missing,
        migration_aware=args.migration_aware,
        cutover_pm2_name=args.cutover_pm2_name,
        steady_state=args.steady_state,
    )
    print_rows(rows)
    if errors:
        print("runtime=failed")
        print("errors=" + ",".join(errors))
        return 1
    print("runtime=ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
