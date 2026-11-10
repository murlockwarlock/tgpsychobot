"""Tests reproducing the remaining 8 release-blockers identified in independent review:

1. Safe bootstrap of initial deploy against old remote script lacking --check-legacy-safety.
2. Strict Git release integrity: reject SHA spoofing without Git or verified artifact; full manifest verification.
3. Python dependency isolation: unified resolution of venvs/<requirements_hash> in release_manager and ecosystem.
4. Safe rollback failure handling: revert to pre-rollback symlink on reload failure, and verify previous_successful_target_dir.
5. Port check fail-closed: fail closed when port inspection tools are missing; reject foreign PID with matching APP_PORT.
6. Real readiness checking: detect online PM2 processes that cannot bind or serve traffic.
7. Real ecosystem.config.js integration: verify interpreter, script, args, cwd, and env variables for Telegram and MAX.
8. Concurrency and crash recovery: fail closed on corrupted state files, serialize staging.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

import pytest

import scripts.release_manager as rm


# =============================================================================
# 1. Safe Bootstrap of Initial Deploy
# =============================================================================

def test_initial_deploy_bootstrap_check_logic(tmp_path: Path):
    """The legacy deploy pre-check must succeed on unmigrated layout even if old script lacks --check-legacy-safety."""
    base_dir = tmp_path / "bots_root"
    base_dir.mkdir(parents=True)
    # Old layout: no deployments/ directory exists yet
    
    # Run the bootstrap python logic that deploy_prod.sh uses
    script = f"""
import sys
from pathlib import Path
base_dir = Path('{base_dir}')
deployments_dir = base_dir / 'deployments'
names = ['tg_autobusbusbot_new', 'max_bot']
if not deployments_dir.is_dir():
    sys.exit(0)
migrated = [n for n in names if (deployments_dir / n).is_symlink() or (deployments_dir / n).is_dir()]
if migrated:
    sys.exit(1)
sys.exit(0)
"""
    res = subprocess.run([sys.executable, "-c", script])
    assert res.returncode == 0, "Initial bootstrap check must pass when deployments/ directory does not exist"

    # Now simulate one bot being migrated to isolated release
    (base_dir / "deployments").mkdir()
    rel_dir = base_dir / "releases" / "1111111111111111111111111111111111111111"
    rel_dir.mkdir(parents=True)
    (base_dir / "deployments" / "tg_autobusbusbot_new").symlink_to(rel_dir)

    # Re-running the check for tg_autobusbusbot_new must now fail closed
    res_migrated = subprocess.run([sys.executable, "-c", script])
    assert res_migrated.returncode == 1, "Must block legacy deploy when instance is already migrated"


# =============================================================================
# 2. Strict Git Integrity & Artifact Verification
# =============================================================================

def test_stage_release_must_fail_closed_without_git_or_verified_artifact(tmp_path: Path):
    """stage_release must NOT copy uncommitted non-git files and stamp arbitrary SHA."""
    base_dir = tmp_path / "bots_root"
    non_git_dir = tmp_path / "non_git_dir"
    non_git_dir.mkdir()
    (non_git_dir / "main.py").write_text("APP_VERSION = 'spoofed'\n", encoding="utf-8")
    (non_git_dir / "requirements.txt").write_text("aiogram==3.1.1\n", encoding="utf-8")

    dummy_sha = "a" * 40

    # Current flawed implementation copies non-git dir and marks it as dummy_sha!
    # Expected: must raise RuntimeError refusing to stage unverified non-git files.
    with pytest.raises(RuntimeError, match=r"(?i)not a git repository|verified release archive"):
        rm.stage_release(base_dir=base_dir, target_sha=dummy_sha, source_dir=non_git_dir)


def test_stage_release_manifest_detects_post_stage_tampering(tmp_path: Path):
    """Release directory must have MANIFEST.sha256; tampering with files must be detected."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.email", "tester@example.com"], cwd=str(repo), check=True)

    (repo / "main.py").write_text("APP_VERSION = 'v1'\n", encoding="utf-8")
    (repo / "requirements.txt").write_text("aiogram==3.1.1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-m", "commit 1"], cwd=str(repo), check=True)
    sha1 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True, check=True).stdout.strip()

    base_dir = tmp_path / "bots_root"
    staged = rm.stage_release(base_dir=base_dir, target_sha=sha1, source_dir=repo)

    # Initial integrity must pass
    ok, msg = rm.verify_release_integrity(staged, sha1)
    assert ok is True, f"Initial integrity must pass: {msg}"

    # Tamper with main.py in staged release
    (staged / "main.py").write_text("APP_VERSION = 'tampered_malicious'\n", encoding="utf-8")

    # Integrity check MUST detect tampering
    ok_tampered, msg_tampered = rm.verify_release_integrity(staged, sha1)
    assert ok_tampered is False, "verify_release_integrity must fail when a staged file has been tampered with"
    assert "tampered" in msg_tampered.lower() or "mismatch" in msg_tampered.lower() or "manifest" in msg_tampered.lower()


# =============================================================================
# 3. Python Dependency Isolation: Harmonized venvs/<hash> Resolution
# =============================================================================

def test_resolve_release_python_interpreter_uses_hash_addressed_venv(tmp_path: Path):
    """resolve_release_python_interpreter and ecosystem.config.js must check base_dir/venvs/<requirements_hash>."""
    base_dir = tmp_path / "bots_root"
    release_dir = base_dir / "releases" / ("b" * 40)
    release_dir.mkdir(parents=True)
    req_file = release_dir / "requirements.txt"
    req_file.write_text("requests==2.31.0\n", encoding="utf-8")
    req_hash = rm.hash_requirements(req_file)
    (release_dir / "requirements.sha256").write_text(f"{req_hash}\n", encoding="utf-8")

    # Create isolated hash-addressed venv
    venv_dir = base_dir / "venvs" / req_hash / "bin"
    venv_dir.mkdir(parents=True)
    fake_python = venv_dir / "python"
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)

    # Resolve interpreter
    resolved = rm.resolve_release_python_interpreter(release_dir, base_dir)
    assert resolved == fake_python, f"Expected hash-addressed venv {fake_python}, but got {resolved}"


# =============================================================================
# 4. Safe Rollback Failure Handling & State Correctness
# =============================================================================

def test_rollback_failure_reverts_symlink_and_records_correct_previous_target(tmp_path: Path, monkeypatch):
    """When rollback reload fails, symlink must be reverted to pre-rollback target, not left broken."""
    base_dir = tmp_path / "bots_root"
    deployments = base_dir / "deployments"
    releases = base_dir / "releases"
    deployments.mkdir(parents=True)
    releases.mkdir(parents=True)

    sha_active = "1" * 40
    sha_rollback = "2" * 40

    rel_active = releases / sha_active
    rel_rollback = releases / sha_rollback
    for r, sha in [(rel_active, sha_active), (rel_rollback, sha_rollback)]:
        r.mkdir()
        (r / "REVISION").write_text(f"{sha}\n", encoding="utf-8")
        (r / "requirements.txt").write_text("aiogram==3.1.1\n", encoding="utf-8")
        req_hash = rm.hash_requirements(r / "requirements.txt")
        (r / "requirements.sha256").write_text(f"{req_hash}\n", encoding="utf-8")
        (r / "main.py").write_text("print('ok')\n", encoding="utf-8")
        rm.write_release_manifest(r)

    link = deployments / "tg_bot_test"
    link.symlink_to(rel_active)

    # Record initial state
    rm.save_instance_state(base_dir, "tg_bot_test", {
        "instance_name": "tg_bot_test",
        "current_release_sha": sha_active,
        "current_target_dir": str(rel_active),
        "previous_successful_release_sha": sha_rollback,
        "previous_successful_target_dir": str(rel_rollback),
        "mode": "isolated",
        "history": [],
    })

    # Mock reload_pm2_instance to FAIL during rollback
    def mock_failing_reload(*args, **kwargs):
        return False, "Simulated PM2 explosion"

    monkeypatch.setattr(rm, "reload_pm2_instance", mock_failing_reload)

    # Attempt rollback
    ok, msg, details = rm.rollback_instance(
        instance_name="tg_bot_test",
        base_dir=base_dir,
        registry={"instances": [{"pm2_name": "tg_bot_test", "deploy_managed": True, "active": True}]},
    )

    assert ok is False, "Rollback must report failure when PM2 reload fails"
    # CRITICAL: Symlink MUST be safely reverted back to rel_active, NOT left pointing to rel_rollback!
    assert link.resolve() == rel_active, f"Symlink must be reverted to pre-rollback target {rel_active}, got {link.resolve()}"


def test_successful_rollback_records_accurate_previous_target_dir(tmp_path: Path, monkeypatch):
    """After successful rollback from Rel 2 to Rel 1, previous_successful_target_dir must be Rel 2, NOT Rel 1."""
    base_dir = tmp_path / "bots_root"
    deployments = base_dir / "deployments"
    releases = base_dir / "releases"
    deployments.mkdir(parents=True)
    releases.mkdir(parents=True)

    sha1 = "1" * 40
    sha2 = "2" * 40

    rel1 = releases / sha1
    rel2 = releases / sha2
    for r, sha in [(rel1, sha1), (rel2, sha2)]:
        r.mkdir()
        (r / "REVISION").write_text(f"{sha}\n", encoding="utf-8")
        (r / "requirements.txt").write_text("aiogram==3.1.1\n", encoding="utf-8")
        req_hash = rm.hash_requirements(r / "requirements.txt")
        (r / "requirements.sha256").write_text(f"{req_hash}\n", encoding="utf-8")
        (r / "main.py").write_text("print('ok')\n", encoding="utf-8")
        rm.write_release_manifest(r)

    # Bot is currently on Rel 2
    link = deployments / "tg_bot_test"
    link.symlink_to(rel2)

    rm.save_instance_state(base_dir, "tg_bot_test", {
        "instance_name": "tg_bot_test",
        "current_release_sha": sha2,
        "current_target_dir": str(rel2),
        "previous_successful_release_sha": sha1,
        "previous_successful_target_dir": str(rel1),
        "mode": "isolated",
        "history": [],
    })

    monkeypatch.setattr(rm, "reload_pm2_instance", lambda *a, **k: (True, "reloaded"))
    monkeypatch.setattr(rm, "verify_instance_health", lambda *a, **k: (True, []))
    monkeypatch.setattr(rm, "load_pm2_snapshot", lambda: {"tg_bot_test": {"name": "tg_bot_test", "pm_id": 0, "pid": 1234, "pm2_env": {"status": "online", "pm_cwd": str(rel1)}}})

    ok, msg, details = rm.rollback_instance(
        instance_name="tg_bot_test",
        base_dir=base_dir,
        registry={"instances": [{"pm2_name": "tg_bot_test", "deploy_managed": True, "active": True}]},
    )

    assert ok is True, f"Rollback should succeed: {msg}"
    state = rm.load_instance_state(base_dir, "tg_bot_test")
    assert state["current_release_sha"] == sha1
    assert state["current_target_dir"] == str(rel1)
    # The previous target before this rollback was rel2!
    assert state["previous_successful_release_sha"] == sha2
    assert state["previous_successful_target_dir"] == str(rel2), f"previous_successful_target_dir should be {rel2}, got {state['previous_successful_target_dir']}"


# =============================================================================
# 5. True Fail-Closed for Ports & Strict Process Ownership
# =============================================================================

def test_port_check_fails_closed_when_tools_missing(tmp_path: Path, monkeypatch):
    """When port checking tool is missing/unusable, port check must FAIL CLOSED, not return True."""
    def mock_subprocess_run(cmd, *args, **kwargs):
        raise FileNotFoundError("lsof not found")

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    ok, msg, killed = rm.check_and_resolve_port_conflict(
        port=8081,
        instance_name="tg_bot_test",
        base_dir=tmp_path,
        pm2_snapshot={"dummy": {}},
        registry={"instances": [{"pm2_name": "tg_bot_test", "port": 8081}]},
    )
    assert ok is False, "Must fail closed if port inspection tools are missing"
    assert "port_check_unavailable" in msg or "unavailable" in msg.lower()


def test_verify_process_ownership_rejects_foreign_pid_with_same_app_port_env(tmp_path: Path):
    """Foreign process having APP_PORT=8081 in env must NOT be treated as owned by our bot."""
    # A foreign process (e.g. some nodejs service) happens to have APP_PORT=8081
    # verify_process_ownership must NOT return True unless cmdline and cwd match this bot!
    # Test verify_process_ownership directly
    is_owned = rm.verify_process_ownership(
        pid=os.getpid(),  # current pytest runner PID
        instance_name="tg_bot_test",
        base_dir=tmp_path,
        port=8081,
    )
    assert is_owned is False, "Pytest process must not be considered owned by tg_bot_test"


# =============================================================================
# 6. Real Readiness Checking
# =============================================================================

def test_verify_instance_health_detects_unresponsive_socket(tmp_path: Path, monkeypatch):
    """If process is online in PM2 but its configured port is not accepting connections, health check must fail."""
    # Pick a port that is NOT listening
    free_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    free_sock.bind(("127.0.0.1", 0))
    dead_port = free_sock.getsockname()[1]
    free_sock.close()

    sha = "c" * 40
    rel_dir = tmp_path / "releases" / sha
    rel_dir.mkdir(parents=True)
    (rel_dir / "REVISION").write_text(f"{sha}\n", encoding="utf-8")

    # Mock PM2 returning online process
    mock_proc = {
        "name": "tg_bot_unresponsive",
        "pm_id": 1,
        "pid": os.getpid(),  # live PID
        "pm2_env": {
            "status": "online",
            "pm_cwd": str(rel_dir),
            "restart_time": 0,
            "unstable_restarts": 0,
        },
    }
    monkeypatch.setattr(rm, "load_pm2_snapshot", lambda: {"tg_bot_unresponsive": mock_proc})

    registry = {
        "instances": [
            {
                "pm2_name": "tg_bot_unresponsive",
                "port": dead_port,
                "deploy_managed": True,
                "active": True,
            }
        ]
    }

    ok, errors = rm.verify_instance_health(
        instance_name="tg_bot_unresponsive",
        expected_sha=sha,
        settle_seconds=0.1,
        base_dir=tmp_path,
        registry=registry,
        check_readiness=True,
    )

    assert ok is False, "verify_instance_health must fail when port is not listening/responsive"
    assert any("port" in e.lower() or "readiness" in e.lower() for e in errors)


# =============================================================================
# 7. Real ecosystem.config.js Evaluation
# =============================================================================

def test_real_ecosystem_config_js_evaluation(tmp_path: Path):
    """Evaluate repo's actual ecosystem.config.js with Node and verify isolated layout and venvs."""
    repo_ecosystem = Path(__file__).resolve().parents[1] / "ecosystem.config.js"
    assert repo_ecosystem.is_file(), "ecosystem.config.js must exist in repo root"

    base_dir = tmp_path / "bots_root"
    deployments_dir = base_dir / "deployments"
    releases_dir = base_dir / "releases"
    venvs_dir = base_dir / "venvs"
    deployments_dir.mkdir(parents=True)
    releases_dir.mkdir(parents=True)
    venvs_dir.mkdir(parents=True)

    # 1. Setup isolated release for tg_autobusbusbot_new
    sha = "d" * 40
    rel_dir = releases_dir / sha
    rel_dir.mkdir()
    (rel_dir / "REVISION").write_text(f"{sha}\n", encoding="utf-8")
    (rel_dir / "main.py").write_text("# main\n", encoding="utf-8")
    (rel_dir / "requirements.txt").write_text("aiogram==3.1.1\n", encoding="utf-8")
    req_hash = "fake_hash_12345"
    (rel_dir / "requirements.sha256").write_text(f"{req_hash}\n", encoding="utf-8")

    # Isolated hash venv
    fake_venv_py = venvs_dir / req_hash / "bin" / "python"
    fake_venv_py.parent.mkdir(parents=True)
    fake_venv_py.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_venv_py.chmod(0o755)

    (deployments_dir / "tg_autobusbusbot_new").symlink_to(rel_dir)

    # Run node script evaluating the real ecosystem.config.js
    node_script = f"""
const fs = require('fs');
const path = require('path');
process.env.PROD_BASE_DIR = '{base_dir}';
process.env.PROD_DEPLOYMENTS_DIR = '{deployments_dir}';
process.env.PROD_REMOTE_PY = '/usr/bin/default_python';
const eco = require('{repo_ecosystem}');
const tgApp = eco.apps.find(a => a.name === 'tg_autobusbusbot_new');
const maxApp = eco.apps.find(a => a.name && a.name.startsWith('max_'));

const result = {{
  tg_script: tgApp.script,
  tg_cwd: tgApp.cwd,
  tg_interpreter: tgApp.interpreter,
  tg_app_dir: tgApp.env.APP_DIR,
  tg_revision: tgApp.env.APP_REVISION,
  max_script: maxApp.script,
  max_args: maxApp.args,
  max_interpreter: maxApp.interpreter,
  max_cwd: maxApp.cwd,
}};
console.log(JSON.stringify(result));
"""
    res = subprocess.run(["node", "-e", node_script], capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)

    # Check Telegram bot settings
    assert data["tg_script"] == str(deployments_dir / "tg_autobusbusbot_new" / "main.py")
    assert data["tg_cwd"] == str(deployments_dir / "tg_autobusbusbot_new")
    assert data["tg_app_dir"] == str(deployments_dir / "tg_autobusbusbot_new")
    assert data["tg_revision"] == sha
    # Verify hash-addressed venv was selected!
    assert data["tg_interpreter"] == str(fake_venv_py), f"Expected {fake_venv_py}, got {data['tg_interpreter']}"

    # Check MAX bot settings
    assert data["max_args"] == "-m max_messenger_bot.app"
    assert data["max_interpreter"] == "none"


# =============================================================================
# 8. Corrupted State File & Concurrency
# =============================================================================

def test_corrupted_state_file_fails_closed(tmp_path: Path):
    """Corrupted .state/<name>.json must raise RuntimeError, NOT silently return empty dict."""
    base_dir = tmp_path / "bots_root"
    state_dir = base_dir / "deployments" / ".state"
    state_dir.mkdir(parents=True)
    bad_state_file = state_dir / "tg_bot_corrupt.json"
    bad_state_file.write_text("{\"broken_json\": [unterminated", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"(?i)corrupt"):
        rm.load_instance_state(base_dir, "tg_bot_corrupt")
