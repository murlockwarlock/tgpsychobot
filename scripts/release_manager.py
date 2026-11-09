#!/usr/bin/env python3
"""Release Manager for Psychobot Fleet.

Provides physical directory isolation for releases and safe, targeted,
canary deployment and rollback per PM2 bot instance.

Directory Layout:
  <BASE_DIR>/
    releases/<commit_sha_40>/     # Immutable code tree per release
      REVISION                    # 40-character commit SHA
      requirements.txt            # Dependency specification
      requirements.sha256         # Hash of normalized requirements
    deployments/<instance_name>   # Atomic symlink to releases/<commit_sha_40>
    shared/                       # Shared runtime environment, logs, backups
    newbots/                      # Legacy static code tree for unmigrated bots
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import py_compile
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = Path(os.environ.get("PROD_BASE_DIR", "/root/telegram_bots"))
DEFAULT_REGISTRY_PATH = REPO_ROOT / "config" / "bot_instances.json"
DEFAULT_ECOSYSTEM_PATH = REPO_ROOT / "ecosystem.config.js"
DEFAULT_RUNTIME_ENV_PATH = Path(os.environ.get("PROD_RUNTIME_ENV", "/root/telegram_bots/runtime.env"))
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SHORT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


# =============================================================================
# Helper Utilities & Registry
# =============================================================================

def load_registry(path: Path | str = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    """Load and validate the bot instance registry."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Bot instance registry not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "instances" not in data:
        raise ValueError(f"Invalid registry format in {path}")
    return data


def get_managed_instances(registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all deploy_managed instances from the registry."""
    return [
        inst for inst in registry.get("instances", [])
        if inst.get("deploy_managed", True) and inst.get("active", True)
    ]


def get_instance_by_name(registry: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Find instance entry by pm2_name or legacy_pm2_name."""
    for inst in registry.get("instances", []):
        if inst.get("pm2_name") == name or inst.get("legacy_pm2_name") == name:
            return inst
    return None


def resolve_git_commit_sha(repo_dir: Path = REPO_ROOT, revision: str | None = None) -> str:
    """Resolve a git revision (defaults to HEAD) to a full 40-character commit SHA."""
    rev = revision or "HEAD"
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", rev],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        sha = proc.stdout.strip()
        if not SHA_RE.match(sha):
            raise ValueError(f"Invalid SHA format resolved: {sha}")
        return sha
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Unable to resolve git revision '{rev}' in {repo_dir}: {exc.stderr.strip()}") from exc


# =============================================================================
# Dependency & Virtualenv Compatibility
# =============================================================================

def normalize_requirements(req_path: Path) -> str:
    """Normalize requirements.txt contents by stripping comments, blanks, and sorting."""
    if not req_path.is_file():
        return ""
    lines: list[str] = []
    for raw in req_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Strip inline comments
        if " #" in line:
            line = line.split(" #", 1)[0].strip()
        lines.append(line.lower())
    lines.sort()
    return "\n".join(lines)


def hash_requirements(req_path: Path) -> str:
    """Compute SHA-256 of normalized requirements."""
    normalized = normalize_requirements(req_path)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def audit_dependency_compatibility(
    base_dir: Path,
    target_req_path: Path,
    registry: dict[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    """Audit dependency compatibility between target release and active instances.

    Ensures that releasing code does not silently mutate or break the shared virtualenv
    for running or rollback-target releases.
    """
    target_hash = hash_requirements(target_req_path)
    deployments_dir = base_dir / "deployments"
    legacy_dir = base_dir / "newbots"
    
    active_hashes: dict[str, str] = {}
    managed = get_managed_instances(registry)

    for inst in managed:
        name = inst["pm2_name"]
        dep_link = deployments_dir / name
        if dep_link.is_symlink() or dep_link.is_dir():
            try:
                resolved_dir = dep_link.resolve()
                req_file = resolved_dir / "requirements.txt"
                if req_file.is_file():
                    active_hashes[name] = hash_requirements(req_file)
            except OSError:
                pass
        elif legacy_dir.is_dir():
            legacy_req = legacy_dir / "requirements.txt"
            if legacy_req.is_file():
                active_hashes[name] = hash_requirements(legacy_req)

    # Check if there are active instances with a differing requirements hash
    differing = {name: h for name, h in active_hashes.items() if h != target_hash}

    details = {
        "target_hash": target_hash,
        "active_instances_audited": len(active_hashes),
        "differing_instances": differing,
    }

    if not differing:
        return True, "dependencies_strictly_compatible", details

    # If requirements differ, check whether isolated venv exists for target release
    # (e.g. <base_dir>/venvs/<target_hash> or <target_release>/venv)
    target_venv = base_dir / "venvs" / target_hash
    if target_venv.is_dir() and (target_venv / "bin" / "python").is_file():
        details["isolated_venv"] = str(target_venv)
        return True, "isolated_virtualenv_available", details

    # Fail closed: shared venv mutation would risk breaking active releases
    return (
        False,
        f"dependency_mismatch: target requirements ({target_hash[:8]}) differ from "
        f"{len(differing)} active instance(s) using shared venv. An isolated virtualenv is required.",
        details,
    )


# =============================================================================
# Schema Compatibility & PR A Guard
# =============================================================================

def audit_schema_compatibility(repo_root: Path = REPO_ROOT) -> tuple[bool, str]:
    """Ensure PR A does not apply automatic database migrations.

    PR A must fail-closed if unapplied schema migrations or DDL mutations
    are bundled without the PR B versioned migration infrastructure.
    """
    migrations_dir = repo_root / "migrations"
    if migrations_dir.is_dir():
        untracked_migrations = list(migrations_dir.glob("*.sql"))
        if untracked_migrations:
            return (
                False,
                f"unapplied_migrations_detected: Found {len(untracked_migrations)} migration files. "
                f"PR A is release isolation only; database migrations require PR B.",
            )
    return True, "schema_guard_passed"


# =============================================================================
# Immutable Release Staging
# =============================================================================

def stage_release(
    base_dir: Path = DEFAULT_BASE_DIR,
    target_sha: str | None = None,
    source_dir: Path = REPO_ROOT,
    force: bool = False,
) -> Path:
    """Stage an immutable code release into <base_dir>/releases/<commit_sha_40>.

    Atomic staging: writes to a temporary directory first, compiles all python files,
    writes REVISION and requirements.sha256, and atomically renames the directory.
    """
    sha = target_sha or resolve_git_commit_sha(source_dir)
    if not SHA_RE.match(sha):
        raise ValueError(f"Invalid commit SHA: {sha}")

    releases_dir = base_dir / "releases"
    releases_dir.mkdir(parents=True, exist_ok=True)
    target_dir = releases_dir / sha

    # Idempotent staging: if directory exists and is valid, reuse it
    if target_dir.is_dir() and not force:
        rev_file = target_dir / "REVISION"
        req_hash_file = target_dir / "requirements.sha256"
        if rev_file.is_file() and rev_file.read_text(encoding="utf-8").strip() == sha and req_hash_file.is_file():
            return target_dir

    tmp_stage_dir = releases_dir / f".tmp_{sha}_{os.getpid()}_{int(time.time() * 1000)}"
    if tmp_stage_dir.exists():
        shutil.rmtree(tmp_stage_dir, ignore_errors=True)
    tmp_stage_dir.mkdir(parents=True)

    try:
        # Determine tracked files from git if in a git repository
        git_tracked_files: list[str] = []
        try:
            proc = subprocess.run(
                ["git", "ls-files", "-z"],
                cwd=str(source_dir),
                capture_output=True,
                check=True,
            )
            raw = proc.stdout
            git_tracked_files = [f for f in raw.decode("utf-8", errors="replace").split("\0") if f]
        except Exception:
            git_tracked_files = []

        if git_tracked_files:
            for rel_file in git_tracked_files:
                src_path = source_dir / rel_file
                if not src_path.is_file():
                    continue
                dst_path = tmp_stage_dir / rel_file
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
        else:
            # Fallback: copy tree excluding git, cache, and tmp files
            for root, dirs, files in os.walk(source_dir):
                dirs[:] = [
                    d for d in dirs
                    if not d.startswith((".", "__pycache__", ".venv", ".tmp"))
                ]
                for file in files:
                    if file.startswith((".", "._")):
                        continue
                    src_path = Path(root) / file
                    rel_path = src_path.relative_to(source_dir)
                    dst_path = tmp_stage_dir / rel_path
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)

        # Write REVISION file containing full 40-char SHA
        (tmp_stage_dir / "REVISION").write_text(f"{sha}\n", encoding="utf-8")

        # Write requirements and hash
        req_src = source_dir / "requirements.txt"
        if req_src.is_file():
            shutil.copy2(req_src, tmp_stage_dir / "requirements.txt")
            req_hash = hash_requirements(req_src)
            (tmp_stage_dir / "requirements.sha256").write_text(f"{req_hash}\n", encoding="utf-8")
        else:
            (tmp_stage_dir / "requirements.sha256").write_text("empty\n", encoding="utf-8")

        # Compile all python files to verify syntax integrity before publishing
        for py_file in tmp_stage_dir.rglob("*.py"):
            if py_file.is_file() and not py_file.name.startswith("._"):
                py_compile.compile(str(py_file), doraise=True)

        # Atomically publish the release directory
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        os.replace(tmp_stage_dir, target_dir)

        return target_dir

    except Exception:
        if tmp_stage_dir.exists():
            shutil.rmtree(tmp_stage_dir, ignore_errors=True)
        raise


# =============================================================================
# Atomic Symlink Switching
# =============================================================================

def atomic_symlink_switch(
    base_dir: Path,
    instance_name: str,
    target_release_dir: Path,
) -> Path | None:
    """Atomically update <base_dir>/deployments/<instance_name> to point to target_release_dir.

    Returns the previous target Path if one existed, or None.
    """
    deployments_dir = base_dir / "deployments"
    deployments_dir.mkdir(parents=True, exist_ok=True)
    deployment_link = deployments_dir / instance_name

    previous_target: Path | None = None
    if deployment_link.is_symlink() or deployment_link.exists():
        try:
            previous_target = deployment_link.resolve()
        except OSError:
            previous_target = None

    tmp_link = deployments_dir / f".tmp_{instance_name}_{os.getpid()}_{int(time.time() * 1000)}"
    try:
        os.symlink(target_release_dir, tmp_link)
        os.replace(tmp_link, deployment_link)
    finally:
        if tmp_link.is_symlink() or tmp_link.exists():
            try:
                tmp_link.unlink()
            except OSError:
                pass

    return previous_target


# =============================================================================
# Targeted Ghost Process Cleanup
# =============================================================================

def cleanup_instance_ghost_port(port: int | None, pm2_pids: set[int]) -> list[int]:
    """Clean up any orphan/ghost process holding ONLY the instance's specific port.

    NEVER executes broad range sweeps (like 8080-8100) and NEVER kills PM2 processes.
    """
    if not port or port <= 0:
        return []

    killed_pids: list[int] = []
    try:
        proc = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            for raw_pid in proc.stdout.splitlines():
                raw_pid = raw_pid.strip()
                if not raw_pid.isdigit():
                    continue
                pid = int(raw_pid)
                if pid not in pm2_pids and pid != os.getpid():
                    try:
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(0.2)
                        if os.path.exists(f"/proc/{pid}"):
                            os.kill(pid, signal.SIGKILL)
                        killed_pids.append(pid)
                    except OSError:
                        pass
    except (FileNotFoundError, PermissionError):
        pass

    return killed_pids


# =============================================================================
# PM2 Runtime Inspection & Control
# =============================================================================

def get_pm2_cmd() -> list[str]:
    """Return the PM2 executable command as a list (allowing test overrides via PM2_CMD)."""
    raw = os.environ.get("PM2_CMD", "pm2")
    return shlex.split(raw)


def load_pm2_snapshot() -> dict[str, dict[str, Any]]:
    """Load the current PM2 process list as a dict keyed by process name."""
    try:
        proc = subprocess.run(
            get_pm2_cmd() + ["jlist"],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        if not isinstance(data, list):
            return {}
        return {item["name"]: item for item in data if isinstance(item, dict) and "name" in item}
    except Exception:
        return {}


def inspect_runtime_process(
    process: dict[str, Any],
    base_dir: Path = DEFAULT_BASE_DIR,
) -> dict[str, Any]:
    """Inspect actual running process attributes, resolving cwd symlinks and REVISION.

    Rather than trusting APP_REVISION from env, inspects /proc/<pid>/cwd (on Linux)
    or pm_cwd/pm_exec_path, follows symlinks to the real release directory, and reads REVISION.
    """
    pm2_env = process.get("pm2_env") if isinstance(process.get("pm2_env"), dict) else {}
    pid = pm2_env.get("pid") or process.get("pid")
    status = pm2_env.get("status") or process.get("status") or "unknown"
    pm_cwd = str(pm2_env.get("pm_cwd") or process.get("pm_cwd") or "")
    pm_exec_path = str(pm2_env.get("pm_exec_path") or process.get("pm_exec_path") or "")
    env_vars = pm2_env.get("env") if isinstance(pm2_env.get("env"), dict) else {}
    configured_revision = env_vars.get("APP_REVISION") or ""

    resolved_root: Path | None = None
    proc_cwd_link: Path | None = None

    # Try Linux /proc/<pid>/cwd inspection first
    if pid and isinstance(pid, int):
        proc_cwd_path = Path(f"/proc/{pid}/cwd")
        if proc_cwd_path.exists():
            try:
                proc_cwd_link = proc_cwd_path.resolve()
                resolved_root = proc_cwd_link
            except OSError:
                pass

    # Fallback to pm_cwd / pm_exec_path resolution
    if not resolved_root:
        if pm_cwd:
            try:
                resolved_root = Path(pm_cwd).resolve()
            except OSError:
                resolved_root = Path(pm_cwd)
        elif pm_exec_path:
            try:
                resolved_root = Path(pm_exec_path).parent.resolve()
            except OSError:
                resolved_root = Path(pm_exec_path).parent

    resolved_sha = ""
    if resolved_root and resolved_root.is_dir():
        rev_file = resolved_root / "REVISION"
        if rev_file.is_file():
            try:
                resolved_sha = rev_file.read_text(encoding="utf-8").strip()
            except OSError:
                pass

    # Determine deployment isolation mode
    mode = "unknown"
    resolved_str = str(resolved_root) if resolved_root else ""
    if "/releases/" in resolved_str:
        mode = "isolated"
    elif "/newbots" in resolved_str or resolved_str.endswith("newbots"):
        mode = "legacy"
    elif "/deployments/" in pm_cwd or "/deployments/" in pm_exec_path:
        mode = "isolated"

    is_alive = False
    if pid and isinstance(pid, int) and pid > 0:
        try:
            os.kill(pid, 0)
            is_alive = True
        except OSError:
            is_alive = False

    return {
        "name": process.get("name"),
        "pm_id": process.get("pm_id"),
        "pid": pid,
        "is_alive": is_alive,
        "status": status,
        "mode": mode,
        "pm_cwd": pm_cwd,
        "pm_exec_path": pm_exec_path,
        "resolved_root": str(resolved_root) if resolved_root else "",
        "resolved_sha": resolved_sha,
        "configured_sha": configured_revision,
        "uptime": pm2_env.get("pm_uptime"),
        "restart_time": pm2_env.get("restart_time", 0),
        "created_at": pm2_env.get("created_at"),
    }


def reload_pm2_instance(
    instance_name: str,
    pm2_config: Path = DEFAULT_ECOSYSTEM_PATH,
    runtime_env: Path | None = None,
) -> tuple[bool, str]:
    """Reload exactly ONE PM2 instance with --update-env without touching any other instance."""
    cmd = get_pm2_cmd() + [
        "startOrReload",
        str(pm2_config),
        "--only",
        instance_name,
        "--update-env",
    ]
    env = os.environ.copy()
    if runtime_env and runtime_env.is_file():
        env["PROD_RUNTIME_ENV"] = str(runtime_env)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            return False, f"PM2 reload failed (exit code {proc.returncode}): {proc.stderr.strip()}"
        return True, proc.stdout
    except FileNotFoundError:
        return False, "pm2 binary not found in PATH"


def verify_instance_health(
    instance_name: str,
    expected_sha: str,
    settle_seconds: float = 3.0,
    base_dir: Path = DEFAULT_BASE_DIR,
) -> tuple[bool, list[str]]:
    """Verify that an instance is healthy, online, and running the expected SHA."""
    if settle_seconds > 0:
        time.sleep(settle_seconds)

    errors: list[str] = []
    snapshot = load_pm2_snapshot()
    proc = snapshot.get(instance_name)

    if not proc:
        errors.append(f"missing_pm2_process:{instance_name}")
        return False, errors

    info = inspect_runtime_process(proc, base_dir=base_dir)

    if info["status"] != "online":
        errors.append(f"process_not_online:{instance_name}:{info['status']}")

    if not info["is_alive"]:
        errors.append(f"process_pid_dead:{instance_name}:{info['pid']}")

    if expected_sha:
        resolved = info["resolved_sha"]
        if not resolved:
            errors.append(f"missing_runtime_revision:{instance_name}")
        elif not (resolved == expected_sha or expected_sha.startswith(resolved) or resolved.startswith(expected_sha)):
            errors.append(f"runtime_sha_mismatch:{instance_name}:expected={expected_sha[:8]},actual={resolved[:8]}")

    return len(errors) == 0, errors


# =============================================================================
# High-Level Actions: Canary Deploy & Rollback
# =============================================================================

def deploy_canary(
    instance_name: str,
    target_sha: str,
    base_dir: Path = DEFAULT_BASE_DIR,
    pm2_config: Path = DEFAULT_ECOSYSTEM_PATH,
    runtime_env: Path | None = DEFAULT_RUNTIME_ENV_PATH,
    settle_seconds: float = 3.0,
    auto_rollback: bool = True,
    registry: dict[str, Any] | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Perform an isolated canary deployment for a single bot instance.

    1. Ensures target release is staged into releases/<target_sha>.
    2. Atomically switches deployments/<instance_name> to the new release.
    3. Reloads ONLY this PM2 instance.
    4. Validates health and runtime SHA.
    5. If verification fails and auto_rollback=True, automatically rolls back to previous target.
    """
    reg = registry or load_registry()
    entry = get_instance_by_name(reg, instance_name)
    if not entry:
        return False, f"Instance '{instance_name}' not found in registry", {}

    # 1. Stage the release
    release_dir = stage_release(base_dir=base_dir, target_sha=target_sha)

    # 2. Collect current state for potential rollback
    deployment_link = base_dir / "deployments" / instance_name
    previous_target = deployment_link.resolve() if deployment_link.is_symlink() or deployment_link.is_dir() else None

    # Targeted ghost port check
    port = entry.get("port")
    snapshot_before = load_pm2_snapshot()
    pm2_pids = {
        inspect_runtime_process(p, base_dir)["pid"]
        for p in snapshot_before.values()
        if inspect_runtime_process(p, base_dir)["pid"]
    }
    cleaned_ghost_pids = cleanup_instance_ghost_port(port, pm2_pids)

    # 3. Atomically switch the deployment symlink
    atomic_symlink_switch(base_dir, instance_name, release_dir)

    # 4. Reload PM2 ONLY for this instance
    reload_ok, reload_msg = reload_pm2_instance(
        instance_name=instance_name,
        pm2_config=pm2_config,
        runtime_env=runtime_env,
    )
    if not reload_ok:
        if auto_rollback and previous_target:
            atomic_symlink_switch(base_dir, instance_name, previous_target)
            reload_pm2_instance(instance_name, pm2_config, runtime_env)
        return False, f"PM2 reload failed: {reload_msg}", {"ghost_pids_killed": cleaned_ghost_pids}

    # 5. Verify instance health
    health_ok, errors = verify_instance_health(
        instance_name=instance_name,
        expected_sha=target_sha,
        settle_seconds=settle_seconds,
        base_dir=base_dir,
    )

    if not health_ok:
        rollback_msg = ""
        if auto_rollback:
            if previous_target:
                atomic_symlink_switch(base_dir, instance_name, previous_target)
                reload_pm2_instance(instance_name, pm2_config, runtime_env)
                rollback_msg = f" (auto-rolled back to {previous_target.name})"
            else:
                # If first migration, remove symlink to revert to legacy
                if deployment_link.is_symlink():
                    deployment_link.unlink()
                reload_pm2_instance(instance_name, pm2_config, runtime_env)
                rollback_msg = " (auto-rolled back to legacy newbots)"
        return False, f"Health check failed: {', '.join(errors)}{rollback_msg}", {"errors": errors}

    # Fetch post-deploy runtime details
    snapshot_after = load_pm2_snapshot()
    post_info = inspect_runtime_process(snapshot_after.get(instance_name, {}), base_dir)

    return True, f"Canary deploy of {instance_name} succeeded with SHA {target_sha[:8]}", {
        "instance_name": instance_name,
        "runtime_sha": post_info.get("resolved_sha"),
        "pid": post_info.get("pid"),
        "mode": post_info.get("mode"),
        "ghost_pids_killed": cleaned_ghost_pids,
    }


def rollback_instance(
    instance_name: str,
    target_sha: str | None = None,
    base_dir: Path = DEFAULT_BASE_DIR,
    pm2_config: Path = DEFAULT_ECOSYSTEM_PATH,
    runtime_env: Path | None = DEFAULT_RUNTIME_ENV_PATH,
    settle_seconds: float = 3.0,
    registry: dict[str, Any] | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Roll back a single instance to a previous release without touching other instances."""
    reg = registry or load_registry()
    entry = get_instance_by_name(reg, instance_name)
    if not entry:
        return False, f"Instance '{instance_name}' not found in registry", {}

    deployment_link = base_dir / "deployments" / instance_name
    releases_dir = base_dir / "releases"

    target_release_dir: Path | None = None

    if target_sha:
        target_release_dir = releases_dir / target_sha
        if not target_release_dir.is_dir():
            # Check for prefix match
            candidates = [d for d in releases_dir.glob(f"{target_sha}*") if d.is_dir()]
            if candidates:
                target_release_dir = candidates[0]
            else:
                return False, f"Target release for SHA '{target_sha}' not found in {releases_dir}", {}
    else:
        # If no target specified, select the latest release different from current
        current_target = deployment_link.resolve() if deployment_link.is_symlink() else None
        available = [
            d for d in releases_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".") and (current_target is None or d != current_target)
        ]
        if not available:
            return False, "No previous releases available in releases/ to roll back to", {}
        # Sort by modification time descending
        available.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        target_release_dir = available[0]

    expected_sha = ""
    rev_file = target_release_dir / "REVISION"
    if rev_file.is_file():
        expected_sha = rev_file.read_text(encoding="utf-8").strip()

    # Atomically switch symlink
    atomic_symlink_switch(base_dir, instance_name, target_release_dir)

    # Reload PM2 for this instance only
    reload_ok, reload_msg = reload_pm2_instance(
        instance_name=instance_name,
        pm2_config=pm2_config,
        runtime_env=runtime_env,
    )
    if not reload_ok:
        return False, f"PM2 reload failed during rollback: {reload_msg}", {}

    # Verify health
    health_ok, errors = verify_instance_health(
        instance_name=instance_name,
        expected_sha=expected_sha,
        settle_seconds=settle_seconds,
        base_dir=base_dir,
    )
    if not health_ok:
        return False, f"Health check failed after rollback: {', '.join(errors)}", {"errors": errors}

    snapshot = load_pm2_snapshot()
    info = inspect_runtime_process(snapshot.get(instance_name, {}), base_dir)

    return True, f"Rollback of {instance_name} succeeded to {target_release_dir.name[:8]}", {
        "instance_name": instance_name,
        "runtime_sha": info.get("resolved_sha"),
        "pid": info.get("pid"),
        "mode": info.get("mode"),
    }


# =============================================================================
# Release Pruning
# =============================================================================

def prune_releases(
    base_dir: Path = DEFAULT_BASE_DIR,
    keep: int = 5,
    dry_run: bool = False,
) -> tuple[list[str], list[str]]:
    """Prune inactive releases, always protecting releases referenced by deployments/*.

    Returns (pruned_shas, retained_shas).
    """
    releases_dir = base_dir / "releases"
    deployments_dir = base_dir / "deployments"
    if not releases_dir.is_dir():
        return [], []

    # 1. Identify all active releases directly referenced by any deployment symlink
    active_release_dirs: set[Path] = set()
    if deployments_dir.is_dir():
        for link in deployments_dir.iterdir():
            if link.is_symlink() or link.is_dir():
                try:
                    active_release_dirs.add(link.resolve())
                except OSError:
                    pass

    # 2. Collect all valid release directories
    all_releases = [
        d for d in releases_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    # Sort by mtime descending (newest first)
    all_releases.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    retained: list[str] = []
    pruned: list[str] = []

    keep_count = max(keep, len(active_release_dirs))

    for idx, rel in enumerate(all_releases):
        is_active = rel in active_release_dirs
        if is_active or idx < keep:
            retained.append(rel.name)
        else:
            if not dry_run:
                shutil.rmtree(rel, ignore_errors=True)
            pruned.append(rel.name)

    return pruned, retained


# =============================================================================
# Status Reporting
# =============================================================================

def get_fleet_status(
    base_dir: Path = DEFAULT_BASE_DIR,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    expected_sha: str | None = None,
) -> dict[str, Any]:
    """Compile fleet status with runtime SHA resolution and managed/unmanaged breakdown."""
    reg = load_registry(registry_path)
    snapshot = load_pm2_snapshot()
    managed_entries = get_managed_instances(reg)
    managed_names = {entry["pm2_name"] for entry in managed_entries}

    managed_status: list[dict[str, Any]] = []
    for entry in managed_entries:
        name = entry["pm2_name"]
        proc = snapshot.get(name)
        if proc:
            info = inspect_runtime_process(proc, base_dir=base_dir)
            match_status = "UNKNOWN"
            if expected_sha:
                res_sha = info["resolved_sha"]
                if res_sha and (res_sha == expected_sha or expected_sha.startswith(res_sha) or res_sha.startswith(expected_sha)):
                    match_status = "MATCH"
                else:
                    match_status = "MISMATCH"
            managed_status.append({
                "name": name,
                "platform": entry.get("platform"),
                "port": entry.get("port"),
                "database": entry.get("database"),
                "pm2_status": info["status"],
                "pid": info["pid"],
                "is_alive": info["is_alive"],
                "mode": info["mode"],
                "resolved_sha": info["resolved_sha"],
                "expected_sha": expected_sha or "",
                "sha_match": match_status,
                "uptime": info["uptime"],
                "restarts": info["restart_time"],
            })
        else:
            managed_status.append({
                "name": name,
                "platform": entry.get("platform"),
                "port": entry.get("port"),
                "database": entry.get("database"),
                "pm2_status": "missing",
                "pid": None,
                "is_alive": False,
                "mode": "missing",
                "resolved_sha": "",
                "expected_sha": expected_sha or "",
                "sha_match": "MISSING",
                "uptime": 0,
                "restarts": 0,
            })

    unmanaged_status: list[dict[str, Any]] = []
    for name, proc in snapshot.items():
        if name not in managed_names:
            info = inspect_runtime_process(proc, base_dir=base_dir)
            unmanaged_status.append({
                "name": name,
                "pm2_status": info["status"],
                "pid": info["pid"],
                "is_alive": info["is_alive"],
                "pm_cwd": info["pm_cwd"],
                "protection": "PROTECTED_UNMANAGED",
            })

    return {
        "managed": managed_status,
        "unmanaged": unmanaged_status,
        "base_dir": str(base_dir),
        "expected_sha": expected_sha or "",
    }


def print_status_table(status_data: dict[str, Any]) -> None:
    """Print human-readable formatted status table."""
    managed = status_data.get("managed", [])
    unmanaged = status_data.get("unmanaged", [])

    print("\n=== MANAGED BOT INSTANCES ===")
    header = f"{'INSTANCE':<32} {'PLATFORM':<10} {'PORT':<6} {'STATUS':<9} {'PID':<8} {'MODE':<10} {'RUNTIME SHA':<12} {'MATCH':<8}"
    print(header)
    print("-" * len(header))
    for m in managed:
        sha_str = m["resolved_sha"][:8] if m["resolved_sha"] else "-"
        print(f"{m['name']:<32} {m['platform']:<10} {str(m['port'] or '-'):<6} {m['pm2_status']:<9} {str(m['pid'] or '-'):<8} {m['mode']:<10} {sha_str:<12} {m['sha_match']:<8}")

    if unmanaged:
        print("\n=== UNMANAGED PROCESSES (PROTECTED - NEVER TOUCHED) ===")
        u_header = f"{'NAME':<28} {'STATUS':<10} {'PID':<8} {'CWD':<40}"
        print(u_header)
        print("-" * len(u_header))
        for u in unmanaged:
            print(f"{u['name']:<28} {u['pm2_status']:<10} {str(u['pid'] or '-'):<8} {u['pm_cwd']:<40}")
    print()


# =============================================================================
# CLI Interface
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Psychobot Release Manager")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR, help="Base bots root directory")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH, help="Path to bot_instances.json")
    parser.add_argument("--pm2-config", type=Path, default=DEFAULT_ECOSYSTEM_PATH, help="Path to ecosystem.config.js")
    parser.add_argument("--runtime-env", type=Path, default=DEFAULT_RUNTIME_ENV_PATH, help="Path to runtime.env")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # preflight
    p_preflight = subparsers.add_parser("preflight", help="Run deployment preflight validation")
    p_preflight.add_argument("--target-sha", help="Target commit SHA (defaults to HEAD)")
    p_preflight.add_argument("--allow-dirty", action="store_true", help="Allow uncommitted changes (dev only)")

    # stage
    p_stage = subparsers.add_parser("stage", help="Stage immutable release into releases/<sha>")
    p_stage.add_argument("--target-sha", help="Target commit SHA (defaults to HEAD)")
    p_stage.add_argument("--source-dir", type=Path, default=REPO_ROOT, help="Source repository directory")
    p_stage.add_argument("--force", action="store_true", help="Force re-stage even if target exists")

    # deploy
    p_deploy = subparsers.add_parser("deploy", help="Deploy release to bot instance(s)")
    group = p_deploy.add_mutually_exclusive_group(required=True)
    group.add_argument("--canary", metavar="INSTANCE", help="Deploy a single instance as a canary")
    group.add_argument("--instances", help="Comma-separated instance names to deploy")
    group.add_argument("--all-managed", action="store_true", help="Deploy to all managed instances")
    p_deploy.add_argument("--target-sha", help="Target commit SHA (defaults to HEAD)")
    p_deploy.add_argument("--settle-seconds", type=float, default=3.0, help="Settle seconds before health check")
    p_deploy.add_argument("--no-auto-rollback", action="store_true", help="Disable automatic rollback on failure")
    p_deploy.add_argument("--skip-preflight", action="store_true", help="Skip preflight checks")

    # rollback
    p_rollback = subparsers.add_parser("rollback", help="Roll back an instance to a previous release")
    p_rollback.add_argument("--instance", required=True, help="Target instance name")
    p_rollback.add_argument("--target-sha", help="Specific SHA to roll back to (defaults to previous)")
    p_rollback.add_argument("--settle-seconds", type=float, default=3.0, help="Settle seconds before health check")

    # status
    p_status = subparsers.add_parser("status", help="Inspect actual running fleet status")
    p_status.add_argument("--json", action="store_true", help="Output JSON instead of table")
    p_status.add_argument("--expected-sha", help="Target SHA to verify against")

    # prune
    p_prune = subparsers.add_parser("prune", help="Prune inactive release directories")
    p_prune.add_argument("--keep", type=int, default=5, help="Number of releases to retain")
    p_prune.add_argument("--dry-run", action="store_true", help="Simulate without deleting")

    # migrate-instance
    p_migrate = subparsers.add_parser("migrate-instance", help="Migrate an instance from legacy to isolated layout")
    p_migrate.add_argument("--instance", required=True, help="Target instance name")
    p_migrate.add_argument("--target-sha", help="Target commit SHA (defaults to HEAD)")
    p_migrate.add_argument("--settle-seconds", type=float, default=3.0, help="Settle seconds")

    # verify-dependencies
    p_deps = subparsers.add_parser("verify-dependencies", help="Verify dependency compatibility")
    p_deps.add_argument("--target-req", type=Path, default=REPO_ROOT / "requirements.txt", help="Requirements file")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cmd = args.command

    if cmd == "preflight":
        target_sha = args.target_sha or resolve_git_commit_sha()
        reg = load_registry(args.registry)
        print(f"Running preflight checks for SHA: {target_sha}")

        # Git cleanliness check
        if not args.allow_dirty:
            proc = subprocess.run(["git", "status", "--porcelain"], cwd=str(REPO_ROOT), capture_output=True, text=True)
            if proc.stdout.strip():
                print("✗ Preflight failed: git working tree is dirty", file=sys.stderr)
                return 1

        # Dependency check
        deps_ok, deps_msg, _ = audit_dependency_compatibility(
            base_dir=args.base_dir,
            target_req_path=REPO_ROOT / "requirements.txt",
            registry=reg,
        )
        if not deps_ok:
            print(f"✗ Preflight failed: {deps_msg}", file=sys.stderr)
            return 1

        # Schema guard check (PR A)
        schema_ok, schema_msg = audit_schema_compatibility(REPO_ROOT)
        if not schema_ok:
            print(f"✗ Preflight failed: {schema_msg}", file=sys.stderr)
            return 1

        print("✓ All preflight checks passed successfully.")
        return 0

    elif cmd == "stage":
        target_sha = args.target_sha or resolve_git_commit_sha(args.source_dir)
        print(f"Staging release {target_sha} from {args.source_dir}...")
        rel_dir = stage_release(
            base_dir=args.base_dir,
            target_sha=target_sha,
            source_dir=args.source_dir,
            force=args.force,
        )
        print(f"✓ Release staged at: {rel_dir}")
        return 0

    elif cmd == "deploy":
        target_sha = args.target_sha or resolve_git_commit_sha()
        reg = load_registry(args.registry)

        if not args.skip_preflight:
            deps_ok, deps_msg, _ = audit_dependency_compatibility(
                base_dir=args.base_dir,
                target_req_path=REPO_ROOT / "requirements.txt",
                registry=reg,
            )
            if not deps_ok:
                print(f"✗ Preflight dependency check failed: {deps_msg}", file=sys.stderr)
                return 1
            schema_ok, schema_msg = audit_schema_compatibility(REPO_ROOT)
            if not schema_ok:
                print(f"✗ Preflight schema guard failed: {schema_msg}", file=sys.stderr)
                return 1

        instances_to_deploy: list[str] = []
        if args.canary:
            instances_to_deploy = [args.canary]
        elif args.instances:
            instances_to_deploy = [i.strip() for i in args.instances.split(",") if i.strip()]
        elif args.all_managed:
            instances_to_deploy = [i["pm2_name"] for i in get_managed_instances(reg)]

        print(f"Deploying SHA {target_sha[:8]} to {len(instances_to_deploy)} instance(s): {', '.join(instances_to_deploy)}")

        for inst in instances_to_deploy:
            print(f"\n--- Deploying {inst} ---")
            ok, msg, details = deploy_canary(
                instance_name=inst,
                target_sha=target_sha,
                base_dir=args.base_dir,
                pm2_config=args.pm2_config,
                runtime_env=args.runtime_env,
                settle_seconds=args.settle_seconds,
                auto_rollback=not args.no_auto_rollback,
                registry=reg,
            )
            if not ok:
                print(f"✗ Deploy failed for {inst}: {msg}", file=sys.stderr)
                return 1
            print(f"✓ {msg}")

        print("\n✓ Deployment completed successfully.")
        return 0

    elif cmd == "rollback":
        reg = load_registry(args.registry)
        print(f"Rolling back instance: {args.instance}")
        ok, msg, details = rollback_instance(
            instance_name=args.instance,
            target_sha=args.target_sha,
            base_dir=args.base_dir,
            pm2_config=args.pm2_config,
            runtime_env=args.runtime_env,
            settle_seconds=args.settle_seconds,
            registry=reg,
        )
        if not ok:
            print(f"✗ Rollback failed: {msg}", file=sys.stderr)
            return 1
        print(f"✓ {msg}")
        return 0

    elif cmd == "status":
        status_data = get_fleet_status(
            base_dir=args.base_dir,
            registry_path=args.registry,
            expected_sha=args.expected_sha,
        )
        if args.json:
            print(json.dumps(status_data, indent=2))
        else:
            print_status_table(status_data)
        return 0

    elif cmd == "prune":
        pruned, retained = prune_releases(
            base_dir=args.base_dir,
            keep=args.keep,
            dry_run=args.dry_run,
        )
        mode = "[DRY-RUN] " if args.dry_run else ""
        print(f"{mode}Retained {len(retained)} release(s): {', '.join(r[:8] for r in retained)}")
        if pruned:
            print(f"{mode}Pruned {len(pruned)} release(s): {', '.join(p[:8] for p in pruned)}")
        else:
            print(f"{mode}No releases pruned.")
        return 0

    elif cmd == "migrate-instance":
        target_sha = args.target_sha or resolve_git_commit_sha()
        reg = load_registry(args.registry)
        print(f"Migrating {args.instance} to isolated layout with SHA {target_sha[:8]}...")
        ok, msg, details = deploy_canary(
            instance_name=args.instance,
            target_sha=target_sha,
            base_dir=args.base_dir,
            pm2_config=args.pm2_config,
            runtime_env=args.runtime_env,
            settle_seconds=args.settle_seconds,
            auto_rollback=True,
            registry=reg,
        )
        if not ok:
            print(f"✗ Migration failed: {msg}", file=sys.stderr)
            return 1
        print(f"✓ Migration succeeded: {msg}")
        return 0

    elif cmd == "verify-dependencies":
        reg = load_registry(args.registry)
        ok, msg, details = audit_dependency_compatibility(
            base_dir=args.base_dir,
            target_req_path=args.target_req,
            registry=reg,
        )
        print(f"Dependency compatibility: {'OK' if ok else 'FAILED'}")
        print(f"Message: {msg}")
        if not ok:
            return 1
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
