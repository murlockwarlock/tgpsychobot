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
    deployments/.state/           # Instance state and rollback history files
    shared/                       # Shared runtime environment, logs, backups
    newbots/                      # Legacy static code tree for unmigrated bots
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import py_compile
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = Path(os.environ.get("PROD_BASE_DIR", "/root/telegram_bots"))
DEFAULT_REGISTRY_PATH = REPO_ROOT / "config" / "bot_instances.json"
DEFAULT_ECOSYSTEM_PATH = REPO_ROOT / "ecosystem.config.js"
DEFAULT_RUNTIME_ENV_PATH = Path(os.environ.get("PROD_RUNTIME_ENV", "/root/telegram_bots/runtime.env"))
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SHORT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
STATE_DIR_NAME = ".state"


# =============================================================================
# Helper Utilities & Registry
# =============================================================================

def safe_is_file(path: Path | str | None) -> bool:
    """Check if path is a file without raising PermissionError on restricted parents."""
    if path is None:
        return False
    try:
        p = Path(path)
        return p.is_file()
    except (OSError, PermissionError, ValueError):
        return False


def safe_is_dir(path: Path | str | None) -> bool:
    """Check if path is a directory without raising PermissionError on restricted parents."""
    if path is None:
        return False
    try:
        p = Path(path)
        return p.is_dir()
    except (OSError, PermissionError, ValueError):
        return False


def safe_exists(path: Path | str | None) -> bool:
    """Check if path exists without raising PermissionError on restricted parents."""
    if path is None:
        return False
    try:
        p = Path(path)
        return p.exists()
    except (OSError, PermissionError, ValueError):
        return False


def safe_is_symlink(path: Path | str | None) -> bool:
    """Check if path is a symlink without raising PermissionError on restricted parents."""
    if path is None:
        return False
    try:
        p = Path(path)
        return p.is_symlink()
    except (OSError, PermissionError, ValueError):
        return False


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
            ["git", "rev-parse", "--verify", f"{rev}^{{commit}}"],
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
# Instance State & Concurrency Locking
# =============================================================================

def get_state_dir(base_dir: Path) -> Path:
    """Return and create the state directory <base_dir>/deployments/.state."""
    state_dir = base_dir / "deployments" / STATE_DIR_NAME
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def get_instance_state_path(base_dir: Path, instance_name: str) -> Path:
    return get_state_dir(base_dir) / f"{instance_name}.json"


@contextmanager
def lock_instance(base_dir: Path, instance_name: str, timeout_seconds: float = 10.0):
    """File lock to serialize deploy and rollback per instance."""
    lock_path = get_state_dir(base_dir) / f"{instance_name}.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    start_time = time.time()
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except (BlockingIOError, OSError):
                if time.time() - start_time > timeout_seconds:
                    raise TimeoutError(f"Unable to acquire lock for instance '{instance_name}' after {timeout_seconds}s")
                time.sleep(0.1)
        yield
    finally:
        if acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


@contextmanager
def lock_staging(base_dir: Path, timeout_seconds: float = 30.0):
    """File lock to serialize release staging operations."""
    releases_dir = base_dir / "releases"
    releases_dir.mkdir(parents=True, exist_ok=True)
    lock_path = releases_dir / ".staging.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    start_time = time.time()
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except (BlockingIOError, OSError):
                if time.time() - start_time > timeout_seconds:
                    raise TimeoutError(f"Unable to acquire staging lock after {timeout_seconds}s")
                time.sleep(0.1)
        yield
    finally:
        if acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


@contextmanager
def lock_pruning(base_dir: Path, timeout_seconds: float = 30.0):
    """File lock to serialize release pruning operations."""
    releases_dir = base_dir / "releases"
    releases_dir.mkdir(parents=True, exist_ok=True)
    lock_path = releases_dir / ".prune.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    start_time = time.time()
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except (BlockingIOError, OSError):
                if time.time() - start_time > timeout_seconds:
                    raise TimeoutError(f"Unable to acquire pruning lock after {timeout_seconds}s")
                time.sleep(0.1)
        yield
    finally:
        if acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


def load_instance_state(base_dir: Path, instance_name: str) -> dict[str, Any]:
    """Load the state history of a single bot instance.
    
    Fails closed if the state file exists but is corrupted, preventing history loss.
    """
    path = get_instance_state_path(base_dir, instance_name)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
            raise ValueError(f"State file {path} content is not a JSON object")
        except Exception as exc:
            raise RuntimeError(
                f"Corrupted state file detected for instance '{instance_name}': {path}. Error: {exc}. "
                f"Aborting operation to prevent silent loss of rollback history. Manual operator inspection required."
            ) from exc

    # Initialize from current deployment link if no state file exists
    dep_link = base_dir / "deployments" / instance_name
    current_sha = None
    current_target = None
    if dep_link.is_symlink() or dep_link.is_dir():
        try:
            resolved = dep_link.resolve()
            current_target = str(resolved)
            rev_f = resolved / "REVISION"
            if rev_f.is_file():
                current_sha = rev_f.read_text(encoding="utf-8").strip()
        except OSError:
            pass

    return {
        "instance_name": instance_name,
        "current_release_sha": current_sha,
        "current_target_dir": current_target,
        "previous_successful_release_sha": None,
        "previous_successful_target_dir": None,
        "last_action": "init",
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "history": [],
    }


def save_instance_state(base_dir: Path, instance_name: str, state: dict[str, Any]) -> None:
    """Atomically save instance state to file."""
    path = get_instance_state_path(base_dir, instance_name)
    state["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


# =============================================================================
# Release Protection & Integrity Verification
# =============================================================================

def is_release_protected(base_dir: Path, sha: str) -> tuple[bool, str]:
    """Check if a release is actively in use or protected as a rollback target."""
    deployments_dir = base_dir / "deployments"
    if deployments_dir.is_dir():
        for link in deployments_dir.iterdir():
            if link.name == STATE_DIR_NAME or link.name.startswith("."):
                continue
            if link.is_symlink() or link.is_dir():
                try:
                    resolved = link.resolve()
                    if resolved.name == sha or resolved.name.startswith(sha):
                        return True, f"in active use by deployment symlink '{link.name}'"
                except OSError:
                    pass

    # Check instance state files for rollback protection
    state_dir = deployments_dir / STATE_DIR_NAME
    if state_dir.is_dir():
        for st_file in state_dir.glob("*.json"):
            try:
                st = json.loads(st_file.read_text(encoding="utf-8"))
                inst = st.get("instance_name", st_file.stem)
                if st.get("current_release_sha") == sha:
                    return True, f"recorded as active release for instance '{inst}'"
                if st.get("previous_successful_release_sha") == sha:
                    return True, f"protected as rollback target for instance '{inst}'"
            except Exception:
                pass
    return False, ""


def generate_release_manifest(release_dir: Path) -> dict[str, str]:
    """Generate SHA-256 checksums for all source and configuration files in a release."""
    manifest: dict[str, str] = {}
    for p in sorted(release_dir.rglob("*")):
        if p.is_file():
            if p.name in {"MANIFEST.sha256", "REVISION", ".staging.lock", ".prune.lock"}:
                continue
            if p.name.startswith("._") or p.name.startswith(".tmp") or p.name.endswith(".tmp") or "__pycache__" in p.parts:
                continue
            rel = str(p.relative_to(release_dir))
            manifest[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return manifest


def write_release_manifest(release_dir: Path) -> Path:
    """Compute and write MANIFEST.sha256 in the release directory."""
    manifest = generate_release_manifest(release_dir)
    manifest_file = release_dir / "MANIFEST.sha256"
    lines = [f"{h}  {path}\n" for path, h in manifest.items()]
    manifest_file.write_text("".join(lines), encoding="utf-8")
    return manifest_file


def verify_release_integrity(release_dir: Path, expected_sha: str) -> tuple[bool, str]:
    """Verify release directory integrity: REVISION, requirements.sha256, MANIFEST.sha256, and python bytecode compile."""
    if not release_dir.is_dir():
        return False, f"Directory does not exist: {release_dir}"
    rev_file = release_dir / "REVISION"
    if not rev_file.is_file():
        return False, "Missing REVISION file"
    actual_sha = rev_file.read_text(encoding="utf-8").strip()
    if actual_sha != expected_sha and not actual_sha.startswith(expected_sha) and not expected_sha.startswith(actual_sha):
        return False, f"REVISION mismatch: expected {expected_sha[:8]}, got {actual_sha[:8]}"
    hash_file = release_dir / "requirements.sha256"
    if not hash_file.is_file():
        return False, "Missing requirements.sha256 file"
    req_file = release_dir / "requirements.txt"
    if req_file.is_file():
        expected_hash = hash_requirements(req_file)
        actual_hash = hash_file.read_text(encoding="utf-8").strip()
        if actual_hash != expected_hash:
            return False, f"requirements.sha256 mismatch: expected {expected_hash[:8]}, got {actual_hash[:8]}"

    # Manifest verification (tampering check)
    manifest_file = release_dir / "MANIFEST.sha256"
    if manifest_file.is_file():
        recorded_manifest: dict[str, str] = {}
        for line in manifest_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("  ", 1)
            if len(parts) == 2:
                recorded_manifest[parts[1]] = parts[0]
        # Check all recorded files
        for rel_path, expected_h in recorded_manifest.items():
            f = release_dir / rel_path
            if not f.is_file():
                return False, f"Integrity failed: file '{rel_path}' recorded in MANIFEST.sha256 is missing"
            actual_h = hashlib.sha256(f.read_bytes()).hexdigest()
            if actual_h != expected_h:
                return False, f"Integrity failed: file '{rel_path}' has been tampered with (expected {expected_h[:8]}, got {actual_h[:8]})"
        # Check for unrecorded files added post-staging
        current_manifest = generate_release_manifest(release_dir)
        extra_files = set(current_manifest.keys()) - set(recorded_manifest.keys())
        if extra_files:
            return False, f"Integrity failed: unverified file(s) added post-staging: {', '.join(sorted(extra_files))}"

    # Syntax validation
    for py_file in release_dir.rglob("*.py"):
        if py_file.is_file() and not py_file.name.startswith("._") and "__pycache__" not in str(py_file):
            try:
                py_compile.compile(str(py_file), doraise=True)
            except Exception as exc:
                return False, f"Python compile failed on {py_file.name}: {exc}"
    return True, "integrity_ok"


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
    """Audit dependency compatibility between target release and active instances."""
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

    differing = {name: h for name, h in active_hashes.items() if h != target_hash}

    details = {
        "target_hash": target_hash,
        "active_instances_audited": len(active_hashes),
        "differing_instances": differing,
    }

    if not differing:
        return True, "dependencies_strictly_compatible", details

    # If requirements differ, check for isolated virtualenv
    target_venv = base_dir / "venvs" / target_hash
    if target_venv.is_dir() and (target_venv / "bin" / "python").is_file():
        details["isolated_venv"] = str(target_venv)
        return True, "isolated_virtualenv_available", details

    return (
        False,
        f"dependency_mismatch: target requirements ({target_hash[:8]}) differ from "
        f"{len(differing)} active instance(s) using shared venv. An isolated virtualenv is required.",
        details,
    )


def resolve_release_python_interpreter(release_dir: Path, base_dir: Path) -> Path:
    """Determine the appropriate Python interpreter for the release with unified priority."""
    # 1. Release-specific dedicated venv
    for sub in ["venv/bin/python", "venv/bin/python3"]:
        candidate = release_dir / sub
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    # 2. Hash-addressed shared venv (matches audit_dependency_compatibility and ecosystem.config.js)
    req_hash = ""
    hash_file = release_dir / "requirements.sha256"
    if hash_file.is_file():
        req_hash = hash_file.read_text(encoding="utf-8").strip()
    elif (release_dir / "requirements.txt").is_file():
        req_hash = hash_requirements(release_dir / "requirements.txt")

    if req_hash and req_hash != "empty":
        for sub in ["bin/python", "bin/python3"]:
            candidate = base_dir / "venvs" / req_hash / sub
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate

    # 3. Base fleet venv
    for sub in ["venv/bin/python", "venv/bin/python3"]:
        candidate = base_dir / sub
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    # 4. Environment or system python
    env_py = os.environ.get("PROD_REMOTE_PY")
    if env_py and safe_is_file(env_py):
        return Path(env_py)

    return Path(sys.executable)


def verify_python_interpreter(interpreter: Path) -> tuple[bool, str]:
    """Verify that the selected interpreter is functional."""
    try:
        proc = subprocess.run(
            [str(interpreter), "-c", "import sys; print(sys.version)"],
            capture_output=True,
            text=True,
            check=True,
        )
        return True, proc.stdout.strip()
    except Exception as exc:
        return False, f"Interpreter {interpreter} failed execution check: {exc}"


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
    archive_path: Path | None = None,
    force: bool = False,
) -> Path:
    """Stage an immutable code release into <base_dir>/releases/<commit_sha_40>.

    Atomic staging: extracts directly from the Git commit tree (git archive)
    or a verified release archive artifact, compiles all python files, writes REVISION,
    requirements.sha256, and MANIFEST.sha256, verifies full manifest integrity,
    and atomically publishes the release under an exclusive staging lock.
    """
    with lock_staging(base_dir):
        releases_dir = base_dir / "releases"
        releases_dir.mkdir(parents=True, exist_ok=True)

        # 1. Resolve commit SHA
        is_git = False
        resolved_sha = target_sha or ""
        try:
            rev = target_sha or "HEAD"
            proc = subprocess.run(
                ["git", "rev-parse", "--verify", f"{rev}^{{commit}}"],
                cwd=str(source_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            resolved_sha = proc.stdout.strip()
            is_git = True
        except Exception as exc:
            if target_sha and SHA_RE.match(target_sha):
                resolved_sha = target_sha
            elif not resolved_sha:
                raise RuntimeError(f"Unable to resolve git commit '{target_sha or 'HEAD'}' in {source_dir}: {exc}") from exc

        if not SHA_RE.match(resolved_sha):
            raise ValueError(f"Invalid full 40-character commit SHA: {resolved_sha}")

        target_dir = releases_dir / resolved_sha

        # 2. Protection check: never destroy an in-use or rollback-protected release
        is_prot, prot_reason = is_release_protected(base_dir, resolved_sha)
        if target_dir.is_dir():
            ok, int_msg = verify_release_integrity(target_dir, resolved_sha)
            if is_prot and force:
                raise RuntimeError(f"Refusing to overwrite protected release '{resolved_sha[:8]}' even with --force: {prot_reason}")
            if ok and not force:
                return target_dir
            if is_prot:
                if ok:
                    return target_dir
                raise RuntimeError(f"Cannot overwrite protected release '{resolved_sha[:8]}': {prot_reason}")

        # 3. Stage into atomic temporary directory
        tmp_stage_dir = releases_dir / f".tmp_{resolved_sha}_{os.getpid()}_{int(time.time() * 1000)}"
        if tmp_stage_dir.exists():
            shutil.rmtree(tmp_stage_dir, ignore_errors=True)
        tmp_stage_dir.mkdir(parents=True)

        try:
            if is_git:
                # Extract directly from git commit tree via git archive
                archive_proc = subprocess.run(
                    ["git", "archive", resolved_sha],
                    cwd=str(source_dir),
                    capture_output=True,
                    check=True,
                )
                with tarfile.open(fileobj=io.BytesIO(archive_proc.stdout)) as tar:
                    if hasattr(tarfile, "data_filter"):
                        tar.extractall(path=tmp_stage_dir, filter="data")
                    else:
                        tar.extractall(path=tmp_stage_dir)
            else:
                # Non-git source: require an explicit verified release archive
                cand_archives = []
                if archive_path:
                    cand_archives.append(Path(archive_path))
                cand_archives.extend([
                    source_dir / f"release_{resolved_sha}.tar.gz",
                    source_dir / f"{resolved_sha}.tar.gz",
                    source_dir / "release.tar.gz",
                ])
                chosen_archive = next((a for a in cand_archives if a.is_file()), None)
                if chosen_archive:
                    with tarfile.open(str(chosen_archive), "r:*") as tar:
                        if hasattr(tarfile, "data_filter"):
                            tar.extractall(path=tmp_stage_dir, filter="data")
                        else:
                            tar.extractall(path=tmp_stage_dir)
                else:
                    # STRICT FAIL-CLOSED: Refuse to stamp arbitrary disk files with an unverified SHA
                    raise RuntimeError(
                        f"Cannot stage release '{resolved_sha}': source '{source_dir}' is not a Git repository "
                        f"and no verified release archive was provided."
                    )

            # Write REVISION
            (tmp_stage_dir / "REVISION").write_text(f"{resolved_sha}\n", encoding="utf-8")

            # Write requirements and hash
            req_file = tmp_stage_dir / "requirements.txt"
            if req_file.is_file():
                req_hash = hash_requirements(req_file)
                (tmp_stage_dir / "requirements.sha256").write_text(f"{req_hash}\n", encoding="utf-8")
            else:
                (tmp_stage_dir / "requirements.sha256").write_text("empty\n", encoding="utf-8")

            # Compile all python files to verify syntax integrity before publishing
            for py_file in tmp_stage_dir.rglob("*.py"):
                if py_file.is_file() and not py_file.name.startswith("._") and "__pycache__" not in str(py_file):
                    py_compile.compile(str(py_file), doraise=True)

            # Generate and write full cryptographic manifest
            write_release_manifest(tmp_stage_dir)

            # Atomically publish the release directory
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            os.replace(tmp_stage_dir, target_dir)

            # Verify integrity of newly published release
            ok, int_msg = verify_release_integrity(target_dir, resolved_sha)
            if not ok:
                raise RuntimeError(f"Release integrity validation failed after staging: {int_msg}")

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
    """Atomically update <base_dir>/deployments/<instance_name> to point to target_release_dir."""
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
# Port Safety & Authoritative Process Ownership
# =============================================================================

def verify_process_ownership(
    pid: int,
    instance_name: str,
    base_dir: Path,
    port: int,
) -> bool:
    """Authoritatively prove that PID is an orphaned process of instance_name.
    
    Never assumes matching APP_PORT alone proves ownership. Requires matching entrypoint
    and working directory within the instance's deployment or legacy directory.
    """
    if pid <= 0 or pid == os.getpid():
        return False
    cmdline = ""
    cwd_str = ""
    environ_str = ""

    # Check Linux /proc/<pid>
    proc_dir = Path(f"/proc/{pid}")
    if safe_is_dir(proc_dir):
        try:
            cmdline = (proc_dir / "cmdline").read_bytes().decode("utf-8", errors="replace")
        except (OSError, PermissionError):
            pass
        try:
            cwd_str = str((proc_dir / "cwd").resolve())
        except (OSError, PermissionError):
            pass
        try:
            environ_str = (proc_dir / "environ").read_bytes().decode("utf-8", errors="replace")
        except (OSError, PermissionError):
            pass
    else:
        # macOS / BSD fallback via ps and lsof
        try:
            ps_proc = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
            if ps_proc.returncode == 0:
                cmdline = ps_proc.stdout.strip()
        except Exception:
            pass
        try:
            lsof_cwd = subprocess.run(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"], capture_output=True, text=True)
            if lsof_cwd.returncode == 0:
                for l in lsof_cwd.stdout.splitlines():
                    if l.startswith("n"):
                        cwd_str = l[1:]
        except Exception:
            pass

    # Strictly verify:
    # 1. Entrypoint matches this bot runner
    has_bot_entrypoint = ("main.py" in cmdline) or ("max_messenger_bot.app" in cmdline)
    has_instance_id = f"BOT_INSTANCE_ID={instance_name}" in environ_str or f"--name {instance_name}" in cmdline

    # 2. Cwd is strictly within this instance deployment or legacy directory
    dep_path = base_dir / "deployments" / instance_name
    legacy_path = base_dir / "newbots"
    expected_dep = ""
    if safe_exists(dep_path):
        try:
            expected_dep = str(dep_path.resolve())
        except (OSError, PermissionError):
            pass
    legacy_dir = ""
    if safe_exists(legacy_path):
        try:
            legacy_dir = str(legacy_path.resolve())
        except (OSError, PermissionError):
            pass

    in_instance_dep = bool(expected_dep and (expected_dep == cwd_str or expected_dep in cwd_str or f"deployments/{instance_name}" in cwd_str))
    in_legacy_dir = bool(legacy_dir and (legacy_dir == cwd_str or legacy_dir in cwd_str))

    # CRITICAL: Never accept matching APP_PORT alone as proof of ownership!
    if has_instance_id and (in_instance_dep or in_legacy_dir):
        return True

    if has_bot_entrypoint and in_instance_dep:
        return True

    return False


def get_listening_pids(port: int) -> tuple[bool, list[int], str]:
    """Query strictly TCP listening sockets on port.
    
    Returns (success, pids, tool_used). If no tool is available, success=False (fail-closed).
    """
    # 1. Try lsof strictly querying LISTEN sockets:
    try:
        proc = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 or (proc.returncode == 1 and not proc.stdout.strip()):
            pids = []
            for raw in proc.stdout.splitlines():
                raw = raw.strip()
                if raw.isdigit():
                    pids.append(int(raw))
            return True, pids, "lsof"
    except (FileNotFoundError, PermissionError):
        pass

    # 2. Try ss querying LISTEN sockets:
    try:
        proc = subprocess.run(
            ["ss", "-ltnp", f"sport = :{port}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            pids = []
            for raw in re.findall(r"pid=(\d+)", proc.stdout):
                pids.append(int(raw))
            return True, pids, "ss"
    except (FileNotFoundError, PermissionError):
        pass

    # 3. Try netstat querying LISTEN sockets:
    try:
        proc = subprocess.run(
            ["netstat", "-ltnp"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            pids = []
            for line in proc.stdout.splitlines():
                if f":{port} " in line and "LISTEN" in line:
                    m = re.search(r"(\d+)/", line)
                    if m:
                        pids.append(int(m.group(1)))
            return True, pids, "netstat"
    except (FileNotFoundError, PermissionError):
        pass

    # No port inspection tool succeeded: FAIL CLOSED
    return False, [], "none"


def check_and_resolve_port_conflict(
    port: int | None,
    instance_name: str,
    base_dir: Path,
    pm2_snapshot: dict[str, Any],
    registry: dict[str, Any],
) -> tuple[bool, str, list[int]]:
    """Check port availability with fail-closed semantics and authoritative process ownership."""
    if not port or port <= 0:
        return True, "no_port_configured", []

    if not pm2_snapshot:
        return False, "PM2 snapshot is unavailable; refusing to inspect ports.", []

    ok, listening_pids, tool_used = get_listening_pids(port)
    if not ok:
        # STRICT FAIL-CLOSED: Refuse to continue if port inspection tools are missing or broken
        return False, f"port_check_unavailable: No port inspection tool (lsof, ss, netstat) available. Failing closed for safety.", []

    if not listening_pids:
        return True, "port_clear", []

    managed_entries = {inst.get("pm2_name"): inst for inst in registry.get("instances", [])}
    pm2_pid_map = {}
    for name, p in pm2_snapshot.items():
        info = inspect_runtime_process(p, base_dir)
        if info.get("pid"):
            pm2_pid_map[info["pid"]] = name

    killed_pids: list[int] = []
    for pid in listening_pids:
        if pid == os.getpid():
            continue
        # If pid is in PM2:
        if pid in pm2_pid_map:
            owning_inst = pm2_pid_map[pid]
            if owning_inst == instance_name:
                # Current process of this instance; PM2 reload will handle cutover
                continue
            # Port collision with another PM2 instance!
            return False, f"Port {port} is occupied by active PM2 instance '{owning_inst}' (PID {pid}). Aborting deploy.", []

        # Check if it belongs to an unmanaged process
        for p in pm2_snapshot.values():
            if p.get("name") not in managed_entries and p.get("pid") == pid:
                return False, f"Port {port} is occupied by protected unmanaged process '{p.get('name')}' (PID {pid}). Aborting deploy.", []

        # Non-PM2 process: verify ownership
        is_proven_orphan = verify_process_ownership(pid, instance_name, base_dir, port)
        if not is_proven_orphan:
            # FAIL CLOSED: Cannot kill unknown/foreign process
            return False, f"Port {port} is bound by unknown foreign process PID {pid}. Ownership cannot be proven. Aborting deploy to prevent outage.", []

        # Terminate proven orphan
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.3)
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            killed_pids.append(pid)
        except OSError:
            pass

    return True, "port_ready", killed_pids


def cleanup_instance_ghost_port(
    port: int | None,
    pm2_pids: set[int],
    instance_name: str | None = None,
    base_dir: Path = DEFAULT_BASE_DIR,
) -> list[int]:
    """Clean up orphan process on port; strictly verifies ownership and NEVER kills foreign processes."""
    if not port or port <= 0 or not instance_name:
        return []

    killed_pids: list[int] = []
    ok, listening_pids, _ = get_listening_pids(port)
    if not ok or not listening_pids:
        return []

    for pid in listening_pids:
        if pid not in pm2_pids and pid != os.getpid():
            if not verify_process_ownership(pid, instance_name, base_dir, port):
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
                killed_pids.append(pid)
            except OSError:
                pass

    return killed_pids


# =============================================================================
# PM2 Runtime Inspection & Control
# =============================================================================

def get_pm2_cmd() -> list[str]:
    """Return the PM2 executable command as a list."""
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
    """Inspect actual running process attributes, resolving cwd symlinks and REVISION."""
    pm2_env = process.get("pm2_env") if isinstance(process.get("pm2_env"), dict) else {}
    pid = pm2_env.get("pid") or process.get("pid")
    status = pm2_env.get("status") or process.get("status") or "unknown"
    pm_cwd = str(pm2_env.get("pm_cwd") or process.get("pm_cwd") or "")
    pm_exec_path = str(pm2_env.get("pm_exec_path") or process.get("pm_exec_path") or "")
    env_vars = pm2_env.get("env") if isinstance(pm2_env.get("env"), dict) else {}
    configured_revision = env_vars.get("APP_REVISION") or ""

    resolved_root: Path | None = None
    proc_cwd_link: Path | None = None

    # Linux /proc/<pid>/cwd inspection
    if pid and isinstance(pid, int):
        proc_cwd_path = Path(f"/proc/{pid}/cwd")
        if safe_is_symlink(proc_cwd_path):
            try:
                proc_cwd_link = proc_cwd_path.resolve(strict=True)
                resolved_root = proc_cwd_link
            except (OSError, PermissionError, ValueError):
                proc_cwd_link = None

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
    restart: bool = False,
) -> tuple[bool, str]:
    """Reload exactly ONE PM2 instance with --update-env without touching any other instance.

    If restart=True (e.g. migrating between legacy and isolated layouts where cwd/script changes),
    clears the PM2 process descriptor first so PM2 re-evaluates the ecosystem config fresh.
    """
    env = os.environ.copy()
    if safe_is_file(runtime_env):
        env["PROD_RUNTIME_ENV"] = str(runtime_env)

    if restart:
        try:
            subprocess.run(
                get_pm2_cmd() + ["delete", instance_name],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
        except Exception:
            pass

    # Truncate / flush previous logs for target instance to prevent stale crash logs
    try:
        subprocess.run(
            get_pm2_cmd() + ["flush", instance_name],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        pass

    cmd = get_pm2_cmd() + [
        "startOrReload",
        str(pm2_config),
        "--only",
        instance_name,
        "--update-env",
    ]
    env = os.environ.copy()
    if safe_is_file(runtime_env):
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
    expected_sha: str = "",
    settle_seconds: float = 3.0,
    base_dir: Path = DEFAULT_BASE_DIR,
    registry: dict[str, Any] | None = None,
    check_readiness: bool = True,
) -> tuple[bool, list[str]]:
    """Verify that an instance is healthy, online, and running the expected SHA.

    Performs deep readiness verification:
    - PM2 status is 'online' and process PID is alive.
    - Process has not suffered unstable restarts.
    - Configured application port is actively listening and responsive to TCP connections.
    - PM2 error logs do not contain fatal unhandled crash tracebacks.
    """
    if settle_seconds > 0:
        time.sleep(settle_seconds)

    errors: list[str] = []
    snapshot = load_pm2_snapshot()
    proc = snapshot.get(instance_name)

    if not proc:
        errors.append(f"missing_pm2_process:{instance_name}")
        return False, errors

    info = inspect_runtime_process(proc, base_dir=base_dir)
    pm2_env = proc.get("pm2_env") if isinstance(proc.get("pm2_env"), dict) else {}

    if info["status"] != "online":
        errors.append(f"process_not_online:{instance_name}:{info['status']}")

    if not info["is_alive"]:
        errors.append(f"process_pid_dead:{instance_name}:{info['pid']}")

    if pm2_env.get("unstable_restarts", 0) > 0:
        errors.append(f"process_unstable_restarts:{instance_name}:{pm2_env.get('unstable_restarts')}")

    if expected_sha:
        resolved = info["resolved_sha"]
        if not resolved:
            errors.append(f"missing_runtime_revision:{instance_name}")
        elif not (resolved == expected_sha or expected_sha.startswith(resolved) or resolved.startswith(expected_sha)):
            errors.append(f"runtime_sha_mismatch:{instance_name}:expected={expected_sha[:8]},actual={resolved[:8]}")

    # Deep Application Readiness & Socket Check
    if check_readiness:
        # 1. Determine port from registry or process env
        reg = registry or (load_registry() if DEFAULT_REGISTRY_PATH.is_file() else {})
        entry = get_instance_by_name(reg, instance_name) if reg else None
        target_port = 0
        if entry and entry.get("port"):
            try:
                target_port = int(entry["port"])
            except (ValueError, TypeError):
                pass
        if target_port <= 0:
            env_vars = pm2_env.get("env") if isinstance(pm2_env.get("env"), dict) else {}
            for port_key in ["APP_PORT", "MAX_APP_PORT", "PORT"]:
                if env_vars.get(port_key):
                    try:
                        target_port = int(env_vars[port_key])
                        break
                    except (ValueError, TypeError):
                        pass

        # If a port is configured, verify that it accepts connections
        if target_port > 0:
            connected = False
            last_exc: Exception | None = None
            probe_deadline = time.time() + 3.0
            while time.time() < probe_deadline:
                try:
                    with socket.create_connection(("127.0.0.1", target_port), timeout=0.5):
                        connected = True
                        break
                except (ConnectionRefusedError, TimeoutError, OSError) as exc:
                    last_exc = exc
                    time.sleep(0.1)
            if not connected:
                errors.append(f"port_unresponsive:{instance_name}:port={target_port}:error={last_exc}")

        # 2. Check PM2 error log tail for fatal exceptions
        err_log_path = pm2_env.get("pm_err_log_path")
        if err_log_path and Path(err_log_path).is_file():
            try:
                log_file = Path(err_log_path)
                size = log_file.stat().st_size
                if size > 0:
                    read_bytes = min(size, 8192)
                    with log_file.open("rb") as f:
                        f.seek(size - read_bytes)
                        tail = f.read().decode("utf-8", errors="replace")
                    fatal_markers = [
                        "ModuleNotFoundError",
                        "ImportError",
                        "Address already in use",
                        "OperationalError: (psycopg2",
                    ]
                    for marker in fatal_markers:
                        if marker in tail:
                            errors.append(f"fatal_error_in_logs:{instance_name}:{marker}")
                            break
                    if "Traceback (most recent call last)" in tail:
                        tail_clean = tail.strip()
                        lines = [ln for ln in tail_clean.splitlines() if ln.strip()]
                        last_err = lines[-1] if lines else ""
                        if not any(ign in last_err for ign in ["KeyboardInterrupt", "SystemExit"]):
                            errors.append(f"fatal_error_in_logs:{instance_name}:{last_err}")
            except Exception:
                pass

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
    """Perform an isolated canary deployment for a single bot instance."""
    reg = registry or load_registry()
    entry = get_instance_by_name(reg, instance_name)
    if not entry:
        return False, f"Instance '{instance_name}' not found in registry", {}

    if not entry.get("deploy_managed", True) or not entry.get("active", True):
        return False, f"Instance '{instance_name}' is not deploy_managed or active", {}

    with lock_instance(base_dir, instance_name):
        state = load_instance_state(base_dir, instance_name)

        effective_runtime_env = runtime_env
        if effective_runtime_env == DEFAULT_RUNTIME_ENV_PATH or effective_runtime_env is not None:
            shared_cand = base_dir / "shared" / "runtime.env"
            if safe_is_file(shared_cand):
                effective_runtime_env = shared_cand
            elif not safe_is_file(effective_runtime_env):
                effective_runtime_env = None

        # 1. Stage the release from Git commit tree
        release_dir = stage_release(base_dir=base_dir, target_sha=target_sha)

        # 2. Check Python interpreter
        interpreter = resolve_release_python_interpreter(release_dir, base_dir)
        py_ok, py_msg = verify_python_interpreter(interpreter)
        if not py_ok:
            return False, f"Target python interpreter check failed: {py_msg}", {}

        # 3. Capture previous state
        deployment_link = base_dir / "deployments" / instance_name
        previous_target: Path | None = None
        is_first_migration = False
        if deployment_link.is_symlink() or deployment_link.is_dir():
            try:
                previous_target = deployment_link.resolve()
            except OSError:
                previous_target = None
        else:
            is_first_migration = True

        previous_sha = state.get("current_release_sha")
        if not previous_sha and previous_target and (previous_target / "REVISION").is_file():
            try:
                previous_sha = (previous_target / "REVISION").read_text(encoding="utf-8").strip()
            except OSError:
                pass

        # 4. Port safety check (fail-closed)
        port = entry.get("port")
        snapshot_before = load_pm2_snapshot()
        if not snapshot_before:
            return False, "PM2 snapshot unavailable; aborting deploy.", {}

        port_ok, port_msg, cleaned_ghost_pids = check_and_resolve_port_conflict(
            port=port,
            instance_name=instance_name,
            base_dir=base_dir,
            pm2_snapshot=snapshot_before,
            registry=reg,
        )
        if not port_ok:
            return False, f"Port safety check failed: {port_msg}", {}

        # 5. Atomically switch symlink
        atomic_symlink_switch(base_dir, instance_name, release_dir)

        # 6. Reload PM2
        reload_ok, reload_msg = reload_pm2_instance(
            instance_name=instance_name,
            pm2_config=pm2_config,
            runtime_env=effective_runtime_env,
            restart=is_first_migration,
        )

        deploy_failed = not reload_ok
        fail_reason = f"PM2 reload failed: {reload_msg}" if deploy_failed else ""

        if not deploy_failed:
            # 7. Health verification
            health_ok, errors = verify_instance_health(
                instance_name=instance_name,
                expected_sha=target_sha,
                settle_seconds=settle_seconds,
                base_dir=base_dir,
            )
            if not health_ok:
                deploy_failed = True
                fail_reason = f"Health check failed: {', '.join(errors)}"

        if deploy_failed:
            if not auto_rollback:
                return False, fail_reason, {"ghost_pids_killed": cleaned_ghost_pids}

            # EXECUTE STRICT AUTO-ROLLBACK
            if is_first_migration or previous_target is None:
                # Rollback to legacy
                if deployment_link.is_symlink():
                    deployment_link.unlink()
                rb_ok, rb_reload_msg = reload_pm2_instance(instance_name, pm2_config, effective_runtime_env, restart=True)
                if not rb_ok:
                    return False, f"CRITICAL_EMERGENCY: {fail_reason} AND rollback reload to legacy failed: {rb_reload_msg}", {}
                rb_health, rb_errors = verify_instance_health(instance_name, expected_sha="", settle_seconds=settle_seconds, base_dir=base_dir)
                if not rb_health:
                    return False, f"CRITICAL_EMERGENCY: {fail_reason} AND legacy health check failed after rollback: {', '.join(rb_errors)}", {}
                return False, f"{fail_reason} (auto-rolled back to legacy newbots successfully)", {"ghost_pids_killed": cleaned_ghost_pids}
            else:
                # Rollback to previous target
                atomic_symlink_switch(base_dir, instance_name, previous_target)
                rb_ok, rb_reload_msg = reload_pm2_instance(instance_name, pm2_config, effective_runtime_env)
                if not rb_ok:
                    return False, f"CRITICAL_EMERGENCY: {fail_reason} AND rollback reload to {previous_sha[:8] if previous_sha else 'prev'} failed: {rb_reload_msg}", {}
                rb_health, rb_errors = verify_instance_health(instance_name, expected_sha=previous_sha or "", settle_seconds=settle_seconds, base_dir=base_dir)
                if not rb_health:
                    return False, f"CRITICAL_EMERGENCY: {fail_reason} AND health check failed after rollback to {previous_sha[:8] if previous_sha else 'prev'}: {', '.join(rb_errors)}", {}
                return False, f"{fail_reason} (auto-rolled back to {previous_sha[:8] if previous_sha else previous_target.name[:8]} successfully)", {"ghost_pids_killed": cleaned_ghost_pids}

        # 8. Success: Update state
        state["previous_successful_release_sha"] = previous_sha
        state["previous_successful_target_dir"] = str(previous_target) if previous_target else ("legacy" if is_first_migration else None)
        state["current_release_sha"] = target_sha
        state["current_target_dir"] = str(release_dir)
        state["last_action"] = "deploy"
        state["history"].append({
            "action": "deploy",
            "target_sha": target_sha,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "success",
        })
        save_instance_state(base_dir, instance_name, state)

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

    with lock_instance(base_dir, instance_name):
        state = load_instance_state(base_dir, instance_name)
        deployment_link = base_dir / "deployments" / instance_name
        releases_dir = base_dir / "releases"

        effective_runtime_env = runtime_env
        if effective_runtime_env == DEFAULT_RUNTIME_ENV_PATH or effective_runtime_env is not None:
            shared_cand = base_dir / "shared" / "runtime.env"
            if safe_is_file(shared_cand):
                effective_runtime_env = shared_cand
            elif not safe_is_file(effective_runtime_env):
                effective_runtime_env = None

        target_release_dir: Path | None = None
        resolved_target_sha: str | None = None

        if target_sha:
            target_release_dir = releases_dir / target_sha
            if not target_release_dir.is_dir():
                candidates = [d for d in releases_dir.glob(f"{target_sha}*") if d.is_dir()]
                if candidates:
                    target_release_dir = candidates[0]
                else:
                    return False, f"Target release for SHA '{target_sha}' not found in {releases_dir}", {}
            resolved_target_sha = target_release_dir.name
        else:
            # Use previous successful release of THIS instance
            prev_sha = state.get("previous_successful_release_sha")
            if prev_sha:
                target_release_dir = releases_dir / prev_sha
                if not target_release_dir.is_dir():
                    return False, f"Recorded previous release '{prev_sha[:8]}' not found in {releases_dir}", {}
                resolved_target_sha = prev_sha
            else:
                prev_target = state.get("previous_successful_target_dir")
                if prev_target == "legacy" or (not safe_exists(deployment_link) and not state.get("current_release_sha")):
                    if safe_is_symlink(deployment_link):
                        deployment_link.unlink()
                    reload_ok, reload_msg = reload_pm2_instance(instance_name, pm2_config, effective_runtime_env, restart=True)
                    if not reload_ok:
                        return False, f"PM2 reload failed during rollback to legacy: {reload_msg}", {}
                    # Deep health and readiness check for legacy rollback
                    legacy_health, legacy_errors = verify_instance_health(
                        instance_name=instance_name,
                        expected_sha="",
                        settle_seconds=settle_seconds,
                        base_dir=base_dir,
                        registry=reg,
                    )
                    if not legacy_health:
                        return False, f"Health check failed after rollback to legacy: {', '.join(legacy_errors)}", {"errors": legacy_errors}

                    state["current_release_sha"] = None
                    state["current_target_dir"] = "legacy"
                    save_instance_state(base_dir, instance_name, state)
                    return True, f"Rollback of {instance_name} succeeded to legacy newbots", {"mode": "legacy"}
                return False, f"No previous successful release recorded for instance '{instance_name}'.", {}

        # Verify integrity of target release
        ok, int_msg = verify_release_integrity(target_release_dir, resolved_target_sha)
        if not ok:
            return False, f"Integrity check failed for rollback target {resolved_target_sha[:8]}: {int_msg}", {}

        # Verify interpreter
        interpreter = resolve_release_python_interpreter(target_release_dir, base_dir)
        py_ok, py_msg = verify_python_interpreter(interpreter)
        if not py_ok:
            return False, f"Rollback target python interpreter check failed: {py_msg}", {}

        current_sha_before = state.get("current_release_sha")
        target_dir_before = None
        if deployment_link.is_symlink() or deployment_link.is_dir():
            try:
                target_dir_before = deployment_link.resolve()
            except OSError:
                target_dir_before = None

        # Atomically switch symlink
        atomic_symlink_switch(base_dir, instance_name, target_release_dir)

        # Reload PM2
        reload_ok, reload_msg = reload_pm2_instance(
            instance_name=instance_name,
            pm2_config=pm2_config,
            runtime_env=effective_runtime_env,
        )

        health_ok, errors = (True, [])
        if reload_ok:
            health_ok, errors = verify_instance_health(
                instance_name=instance_name,
                expected_sha=resolved_target_sha,
                settle_seconds=settle_seconds,
                base_dir=base_dir,
                registry=reg,
            )

        # If reload or health check failed: REVERT SAFELY TO PRE-ROLLBACK STATE
        if not reload_ok or not health_ok:
            fail_reason = f"PM2 reload failed: {reload_msg}" if not reload_ok else f"Health check failed: {', '.join(errors)}"
            if target_dir_before:
                atomic_symlink_switch(base_dir, instance_name, target_dir_before)
                rv_ok, rv_msg = reload_pm2_instance(instance_name, pm2_config, effective_runtime_env)
                rv_health, rv_errors = verify_instance_health(
                    instance_name=instance_name,
                    expected_sha=current_sha_before or "",
                    settle_seconds=settle_seconds,
                    base_dir=base_dir,
                    registry=reg,
                )
                if rv_ok and rv_health:
                    return False, f"Rollback failed: {fail_reason} (safely reverted to pre-rollback state {current_sha_before[:8] if current_sha_before else 'prior'})", {"errors": errors}
                else:
                    state["last_action"] = "rollback_failed"
                    state["status"] = "CRITICAL_EMERGENCY"
                    save_instance_state(base_dir, instance_name, state)
                    return False, f"CRITICAL_EMERGENCY: Rollback failed: {fail_reason} AND revert to pre-rollback state also failed: {rv_msg}", {"errors": errors, "revert_errors": rv_errors}
            else:
                if deployment_link.is_symlink():
                    deployment_link.unlink()
                reload_pm2_instance(instance_name, pm2_config, effective_runtime_env, restart=True)
                return False, f"Rollback failed: {fail_reason} (reverted to legacy)", {"errors": errors}

        # Update state: record TRUE previous target
        state["previous_successful_release_sha"] = current_sha_before
        state["previous_successful_target_dir"] = str(target_dir_before) if target_dir_before else "legacy"
        state["current_release_sha"] = resolved_target_sha
        state["current_target_dir"] = str(target_release_dir)
        state["last_action"] = "rollback"
        state["history"].append({
            "action": "rollback",
            "target_sha": resolved_target_sha,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "success",
        })
        save_instance_state(base_dir, instance_name, state)

        snapshot = load_pm2_snapshot()
        info = inspect_runtime_process(snapshot.get(instance_name, {}), base_dir)

        return True, f"Rollback of {instance_name} succeeded to {resolved_target_sha[:8]}", {
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
    """Prune inactive releases, strictly protecting active deployments AND instance rollback targets."""
    with lock_pruning(base_dir):
        releases_dir = base_dir / "releases"
        deployments_dir = base_dir / "deployments"
        if not releases_dir.is_dir():
            return [], []

    protected_shas: set[str] = set()

    # 1. Active deployment symlinks
    if deployments_dir.is_dir():
        for link in deployments_dir.iterdir():
            if link.name == STATE_DIR_NAME or link.name.startswith("."):
                continue
            if link.is_symlink() or link.is_dir():
                try:
                    resolved = link.resolve()
                    protected_shas.add(resolved.name)
                except OSError:
                    pass

    # 2. Instance state rollback targets and current SHAs
    state_dir = deployments_dir / STATE_DIR_NAME
    if state_dir.is_dir():
        for st_file in state_dir.glob("*.json"):
            try:
                st = json.loads(st_file.read_text(encoding="utf-8"))
                if st.get("current_release_sha"):
                    protected_shas.add(st["current_release_sha"])
                if st.get("previous_successful_release_sha"):
                    protected_shas.add(st["previous_successful_release_sha"])
            except Exception:
                pass

    all_releases = [
        d for d in releases_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    all_releases.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    retained: list[str] = []
    pruned: list[str] = []

    for idx, rel in enumerate(all_releases):
        is_protected = (rel.name in protected_shas) or any(s.startswith(rel.name) or rel.name.startswith(s) for s in protected_shas)
        if is_protected or idx < keep:
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
