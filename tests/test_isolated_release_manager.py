"""Integration and unit tests for Psychobot Isolated Release Manager.

Verifies:
1. Physical release isolation and immutable directory structure.
2. Symlink atomicity without broken intermediate states.
3. Dependency compatibility and virtualenv hash auditing.
4. Schema migration PR A guard (fail-closed when unapplied migrations detected).
5. Real runtime testing on an isolated harness:
   - Canary deploy updates only the targeted instance; other instances keep PID, SHA, and cwd.
   - MAX bot executes with isolated module cwd and PYTHONPATH.
   - Two bots sharing a database: updating one does not touch or restart the other.
   - Unmanaged processes (e.g. vitaliy_bot) are never touched or reloaded.
   - Rollback of single instance restores previous release without affecting others.
6. Failure injection:
   - Staging failure aborts cleanly without corrupting release tree.
   - Startup crash on deploy automatically triggers targeted rollback.
   - Ghost port cleanup targets ONLY the instance's configured port.
   - Release pruning protects actively referenced releases.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

import pytest

import scripts.release_manager as rm


REPO_ROOT = Path(__file__).resolve().parents[1]
MOCK_SHA_A = "a" * 40
MOCK_SHA_B = "b" * 40
MOCK_SHA_C = "c" * 40


# =============================================================================
# Unit Tests: Staging, Symlinks, Dependencies, and Schema Guard
# =============================================================================

def test_stage_release_creates_immutable_directory_and_hash(tmp_path: Path):
    base_dir = tmp_path / "bots_root"
    source_dir = tmp_path / "repo_source"
    source_dir.mkdir(parents=True)

    # Create dummy source files
    (source_dir / "main.py").write_text("print('hello telegram')\n", encoding="utf-8")
    (source_dir / "requirements.txt").write_text("aiogram==3.1.1\nsqlalchemy>=2.0.0\n", encoding="utf-8")

    staged_dir = rm.stage_release(
        base_dir=base_dir,
        target_sha=MOCK_SHA_A,
        source_dir=source_dir,
    )

    assert staged_dir.is_dir()
    assert staged_dir.name == MOCK_SHA_A
    assert (staged_dir / "REVISION").read_text(encoding="utf-8").strip() == MOCK_SHA_A
    assert (staged_dir / "main.py").is_file()
    assert (staged_dir / "requirements.sha256").is_file()

    # Verify idempotency
    staged_again = rm.stage_release(
        base_dir=base_dir,
        target_sha=MOCK_SHA_A,
        source_dir=source_dir,
    )
    assert staged_again == staged_dir


def test_stage_release_fails_on_syntax_error_without_leaving_directory(tmp_path: Path):
    base_dir = tmp_path / "bots_root"
    source_dir = tmp_path / "repo_source"
    source_dir.mkdir(parents=True)

    # Injected invalid syntax
    (source_dir / "broken.py").write_text("def syntax_error(:\n  pass\n", encoding="utf-8")

    with pytest.raises(Exception):
        rm.stage_release(
            base_dir=base_dir,
            target_sha=MOCK_SHA_B,
            source_dir=source_dir,
        )

    # Ensure target directory was NOT published
    assert not (base_dir / "releases" / MOCK_SHA_B).exists()


def test_atomic_symlink_switch(tmp_path: Path):
    base_dir = tmp_path / "bots_root"
    rel_a = base_dir / "releases" / MOCK_SHA_A
    rel_b = base_dir / "releases" / MOCK_SHA_B
    rel_a.mkdir(parents=True)
    rel_b.mkdir(parents=True)

    # First switch to A
    prev = rm.atomic_symlink_switch(base_dir, "tg_autobusbusbot_new", rel_a)
    assert prev is None
    link = base_dir / "deployments" / "tg_autobusbusbot_new"
    assert link.is_symlink()
    assert link.resolve() == rel_a.resolve()

    # Switch to B
    prev2 = rm.atomic_symlink_switch(base_dir, "tg_autobusbusbot_new", rel_b)
    assert prev2 == rel_a.resolve()
    assert link.resolve() == rel_b.resolve()


def test_audit_dependency_compatibility_matching_hashes(tmp_path: Path):
    base_dir = tmp_path / "bots_root"
    req_path = tmp_path / "requirements.txt"
    req_path.write_text("aiogram==3.1.1\nsqlalchemy==2.0.23\n", encoding="utf-8")

    # Mock registry
    reg = {
        "instances": [
            {"pm2_name": "tg_bot1", "deploy_managed": True, "active": True},
            {"pm2_name": "tg_bot2", "deploy_managed": True, "active": True},
        ]
    }

    # Setup deployments with matching requirements
    rel_dir = base_dir / "releases" / MOCK_SHA_A
    rel_dir.mkdir(parents=True)
    (rel_dir / "requirements.txt").write_text("aiogram==3.1.1\nsqlalchemy==2.0.23\n", encoding="utf-8")

    rm.atomic_symlink_switch(base_dir, "tg_bot1", rel_dir)
    rm.atomic_symlink_switch(base_dir, "tg_bot2", rel_dir)

    ok, msg, details = rm.audit_dependency_compatibility(base_dir, req_path, reg)
    assert ok is True
    assert msg == "dependencies_strictly_compatible"


def test_audit_dependency_compatibility_mismatch_fails_closed(tmp_path: Path):
    base_dir = tmp_path / "bots_root"
    req_path = tmp_path / "requirements.txt"
    # New requirements have a different version
    req_path.write_text("aiogram==3.2.0\nsqlalchemy==2.0.25\n", encoding="utf-8")

    reg = {
        "instances": [
            {"pm2_name": "tg_bot1", "deploy_managed": True, "active": True},
        ]
    }

    # Running release has old version
    rel_dir = base_dir / "releases" / MOCK_SHA_A
    rel_dir.mkdir(parents=True)
    (rel_dir / "requirements.txt").write_text("aiogram==3.1.0\nsqlalchemy==2.0.20\n", encoding="utf-8")

    rm.atomic_symlink_switch(base_dir, "tg_bot1", rel_dir)

    ok, msg, details = rm.audit_dependency_compatibility(base_dir, req_path, reg)
    assert ok is False
    assert "dependency_mismatch" in msg


def test_audit_dependency_compatibility_with_isolated_venv(tmp_path: Path):
    base_dir = tmp_path / "bots_root"
    req_path = tmp_path / "requirements.txt"
    req_path.write_text("aiogram==3.2.0\n", encoding="utf-8")
    target_hash = rm.hash_requirements(req_path)

    reg = {
        "instances": [
            {"pm2_name": "tg_bot1", "deploy_managed": True, "active": True},
        ]
    }

    # Running release has different requirements
    rel_dir = base_dir / "releases" / MOCK_SHA_A
    rel_dir.mkdir(parents=True)
    (rel_dir / "requirements.txt").write_text("aiogram==3.1.0\n", encoding="utf-8")
    rm.atomic_symlink_switch(base_dir, "tg_bot1", rel_dir)

    # Provide an isolated venv matching the target requirements hash
    target_venv = base_dir / "venvs" / target_hash / "bin"
    target_venv.mkdir(parents=True)
    (target_venv / "python").write_text("#!/bin/sh\n", encoding="utf-8")

    ok, msg, details = rm.audit_dependency_compatibility(base_dir, req_path, reg)
    assert ok is True
    assert msg == "isolated_virtualenv_available"


def test_audit_schema_compatibility_guard(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    # Clean repo without migrations directory passes
    ok, msg = rm.audit_schema_compatibility(repo_root)
    assert ok is True

    # Repo with unapplied SQL migrations fails closed (PR A requirement)
    migrations_dir = repo_root / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "001_new_table.sql").write_text("CREATE TABLE test (id int);\n", encoding="utf-8")

    ok2, msg2 = rm.audit_schema_compatibility(repo_root)
    assert ok2 is False
    assert "unapplied_migrations_detected" in msg2


def test_prune_releases_protects_active_releases(tmp_path: Path):
    base_dir = tmp_path / "bots_root"
    releases_dir = base_dir / "releases"
    releases_dir.mkdir(parents=True)

    # Create 4 mock releases
    shas = [f"{i:02x}" * 20 for i in range(1, 5)]
    for sha in shas:
        d = releases_dir / sha
        d.mkdir()
        time.sleep(0.01)  # ensure different mtimes

    # Link sha 01 to an active deployment
    rm.atomic_symlink_switch(base_dir, "tg_active_bot", releases_dir / shas[0])

    # Prune keeping only 1 newest release
    pruned, retained = rm.prune_releases(base_dir, keep=1)

    # shas[0] MUST be retained because it is actively referenced by tg_active_bot!
    assert shas[0] in retained
    assert (releases_dir / shas[0]).is_dir()


# =============================================================================
# Real Process Runtime Isolated Test Bench
# =============================================================================

class MockPm2Harness:
    """Manages real background Python processes representing Telegram and MAX bots.

    Emulates the PM2 CLI interface (`startOrReload`, `jlist`) while executing real OS
    subprocesses with real PIDs, real working directories, and real signal handling.
    """

    def __init__(self, base_dir: Path, ecosystem_path: Path):
        self.base_dir = base_dir
        self.ecosystem_path = ecosystem_path
        self.state_file = base_dir / "pm2_state.json"
        self.processes: dict[str, subprocess.Popen] = {}
        self.metadata: dict[str, dict[str, Any]] = {}

    def start_process(
        self,
        name: str,
        script: Path,
        cwd: Path,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        app_port: int | None = None,
    ) -> int:
        """Spawn a real Python subprocess and record its state."""
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        if app_port:
            full_env["APP_PORT"] = str(app_port)

        cmd = [sys.executable]
        if args:
            cmd.extend(args)
        else:
            cmd.append(str(script))

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.processes[name] = proc
        self.metadata[name] = {
            "pm_id": len(self.processes),
            "name": name,
            "pid": proc.pid,
            "pm_cwd": str(cwd),
            "pm_exec_path": str(script),
            "status": "online",
            "restart_time": 0,
            "pm_uptime": int(time.time()),
            "env": full_env,
        }
        self._save_state()
        return proc.pid

    def stop_process(self, name: str) -> None:
        """Terminate a specific process."""
        proc = self.processes.get(name)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        if name in self.metadata:
            self.metadata[name]["status"] = "stopped"
            self.metadata[name]["pid"] = None
        self._save_state()

    def update_process_states(self) -> None:
        """Reap terminated child processes and update metadata like PM2 daemon."""
        for name, proc in list(self.processes.items()):
            poll = proc.poll()
            if poll is not None:
                if name in self.metadata:
                    self.metadata[name]["status"] = "errored" if poll != 0 else "stopped"
                    self.metadata[name]["pid"] = None
        self._save_state()

    def reload_process(self, name: str, new_cwd: Path, new_script: Path, args: list[str] | None = None, env: dict[str, str] | None = None) -> int:
        """Reload a single process with new directory/script without touching others."""
        old_meta = self.metadata.get(name, {})
        restarts = old_meta.get("restart_time", 0) + 1
        self.stop_process(name)

        pid = self.start_process(name, new_script, new_cwd, args=args, env=env)
        self.metadata[name]["restart_time"] = restarts
        self._save_state()
        return pid

    def _save_state(self) -> None:
        out = []
        for name, meta in self.metadata.items():
            proc = self.processes.get(name)
            pid = proc.pid if proc and proc.poll() is None else None
            status = "online" if pid else "stopped"
            out.append({
                "name": name,
                "pm_id": meta.get("pm_id", 0),
                "pid": pid,
                "status": status,
                "pm2_env": {
                    "status": status,
                    "pid": pid,
                    "pm_cwd": meta.get("pm_cwd", ""),
                    "pm_exec_path": meta.get("pm_exec_path", ""),
                    "restart_time": meta.get("restart_time", 0),
                    "pm_uptime": meta.get("pm_uptime", 0),
                    "env": meta.get("env", {}),
                }
            })
        self.state_file.write_text(json.dumps(out), encoding="utf-8")

    def cleanup_all(self) -> None:
        for name in list(self.processes.keys()):
            self.stop_process(name)


@pytest.fixture
def runtime_harness(tmp_path: Path, monkeypatch):
    """Fixture providing a complete isolated runtime environment with mock PM2."""
    base_dir = tmp_path / "mock_prod"
    base_dir.mkdir(parents=True)
    releases_dir = base_dir / "releases"
    deployments_dir = base_dir / "deployments"
    newbots_dir = base_dir / "newbots"
    shared_dir = base_dir / "shared"
    releases_dir.mkdir()
    deployments_dir.mkdir()
    newbots_dir.mkdir()
    shared_dir.mkdir()

    # Create dummy worker scripts
    worker_code = """
import os, sys, time, signal

def handle_sig(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sig)
signal.signal(signal.SIGINT, handle_sig)

# Loop and stay alive
while True:
    time.sleep(0.1)
"""

    max_app_code = """
import os, sys, time, signal

def handle_sig(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sig)
signal.signal(signal.SIGINT, handle_sig)

while True:
    time.sleep(0.1)
"""

    crashing_worker_code = """
import sys
raise RuntimeError("Fatal startup crash in bad release")
"""

    # Stage Release A
    rel_a = releases_dir / MOCK_SHA_A
    rel_a.mkdir()
    (rel_a / "REVISION").write_text(f"{MOCK_SHA_A}\n")
    (rel_a / "requirements.txt").write_text("aiogram==3.1.1\n")
    (rel_a / "requirements.sha256").write_text("hash_a\n")
    (rel_a / "main.py").write_text(worker_code)
    (rel_a / "max_messenger_bot").mkdir()
    (rel_a / "max_messenger_bot" / "__init__.py").write_text("")
    (rel_a / "max_messenger_bot" / "app.py").write_text(max_app_code)

    # Stage Release B
    rel_b = releases_dir / MOCK_SHA_B
    rel_b.mkdir()
    (rel_b / "REVISION").write_text(f"{MOCK_SHA_B}\n")
    (rel_b / "requirements.txt").write_text("aiogram==3.1.1\n")
    (rel_b / "requirements.sha256").write_text("hash_a\n")
    (rel_b / "main.py").write_text(worker_code)
    (rel_b / "max_messenger_bot").mkdir()
    (rel_b / "max_messenger_bot" / "__init__.py").write_text("")
    (rel_b / "max_messenger_bot" / "app.py").write_text(max_app_code)

    # Stage Crashing Release C
    rel_c = releases_dir / MOCK_SHA_C
    rel_c.mkdir()
    (rel_c / "REVISION").write_text(f"{MOCK_SHA_C}\n")
    (rel_c / "requirements.txt").write_text("aiogram==3.1.1\n")
    (rel_c / "requirements.sha256").write_text("hash_a\n")
    (rel_c / "main.py").write_text(crashing_worker_code)

    ecosystem_path = base_dir / "ecosystem.config.js"
    ecosystem_path.write_text("// Mock ecosystem\n", encoding="utf-8")

    pm2 = MockPm2Harness(base_dir, ecosystem_path)

    # Create mock CLI script for PM2
    mock_pm2_script = base_dir / "mock_pm2.py"
    mock_pm2_script.write_text("""#!/usr/bin/env python3
import sys, json, os

state_path = os.environ.get("PM2_STATE_FILE", "")

if len(sys.argv) > 1 and sys.argv[1] == "jlist":
    if state_path and os.path.isfile(state_path):
        with open(state_path) as f:
            data = json.load(f)
        for item in data:
            pid = item.get("pid")
            if pid:
                try:
                    os.kill(pid, 0)
                except OSError:
                    item["status"] = "errored"
                    item["pid"] = None
                    if "pm2_env" in item:
                        item["pm2_env"]["status"] = "errored"
                        item["pm2_env"]["pid"] = None
        print(json.dumps(data))
    else:
        print("[]")
    sys.exit(0)

if len(sys.argv) > 1 and sys.argv[1] == "startOrReload":
    # Emulate startOrReload
    sys.exit(0)

sys.exit(0)
""", encoding="utf-8")
    mock_pm2_script.chmod(0o755)

    monkeypatch.setenv("PM2_STATE_FILE", str(pm2.state_file))
    monkeypatch.setenv("PM2_CMD", f"{sys.executable} {mock_pm2_script}")

    orig_load_snapshot = rm.load_pm2_snapshot

    def harness_load_snapshot():
        pm2.update_process_states()
        return orig_load_snapshot()

    monkeypatch.setattr(rm, "load_pm2_snapshot", harness_load_snapshot)

    yield {
        "base_dir": base_dir,
        "releases_dir": releases_dir,
        "deployments_dir": deployments_dir,
        "pm2": pm2,
        "ecosystem_path": ecosystem_path,
        "worker_code": worker_code,
        "max_app_code": max_app_code,
    }

    pm2.cleanup_all()


def test_real_runtime_canary_deploy_isolation(runtime_harness, monkeypatch):
    """Test that deploying to Bot 1 updates ONLY Bot 1 while Bot 2 keeps its PID, SHA, and cwd."""
    base_dir = runtime_harness["base_dir"]
    releases_dir = runtime_harness["releases_dir"]
    pm2 = runtime_harness["pm2"]

    # Start Bot 1 and Bot 2 on Release A
    link_1 = rm.atomic_symlink_switch(base_dir, "tg_autobusbusbot_new", releases_dir / MOCK_SHA_A)
    link_2 = rm.atomic_symlink_switch(base_dir, "tg_someonewithyou01_bot_new", releases_dir / MOCK_SHA_A)

    pid1_before = pm2.start_process(
        "tg_autobusbusbot_new",
        script=base_dir / "deployments" / "tg_autobusbusbot_new" / "main.py",
        cwd=base_dir / "deployments" / "tg_autobusbusbot_new",
        app_port=8081,
    )
    pid2_before = pm2.start_process(
        "tg_someonewithyou01_bot_new",
        script=base_dir / "deployments" / "tg_someonewithyou01_bot_new" / "main.py",
        cwd=base_dir / "deployments" / "tg_someonewithyou01_bot_new",
        app_port=8083,
    )

    # Start unmanaged process (vitaliy_bot)
    unmanaged_dir = base_dir / "vitaliy_bot_dir"
    unmanaged_dir.mkdir()
    (unmanaged_dir / "main.py").write_text(runtime_harness["worker_code"])
    pid_unmanaged_before = pm2.start_process(
        "vitaliy_bot",
        script=unmanaged_dir / "main.py",
        cwd=unmanaged_dir,
        app_port=8090,
    )

    # Intercept reload_pm2_instance to update MockPm2Harness for the target instance
    def mock_reload(instance_name, pm2_config=None, runtime_env=None):
        target_link = base_dir / "deployments" / instance_name
        pm2.reload_process(
            instance_name,
            new_cwd=target_link,
            new_script=target_link / "main.py",
        )
        return True, "reloaded"

    monkeypatch.setattr(rm, "reload_pm2_instance", mock_reload)

    # Deploy Release B to Bot 1 ONLY
    ok, msg, details = rm.deploy_canary(
        instance_name="tg_autobusbusbot_new",
        target_sha=MOCK_SHA_B,
        base_dir=base_dir,
        settle_seconds=0.1,
    )

    assert ok is True

    # 1. Verify Bot 1 was updated to Release B and has a new PID
    snapshot = pm2.metadata
    info1 = rm.inspect_runtime_process(snapshot["tg_autobusbusbot_new"], base_dir)
    assert info1["resolved_sha"] == MOCK_SHA_B
    assert info1["pid"] != pid1_before

    # 2. Verify Bot 2 was UNTOUCHED: same PID, same cwd, same SHA
    info2 = rm.inspect_runtime_process(snapshot["tg_someonewithyou01_bot_new"], base_dir)
    assert info2["resolved_sha"] == MOCK_SHA_A
    assert info2["pid"] == pid2_before

    # 3. Verify Unmanaged vitaliy_bot was UNTOUCHED
    info_u = rm.inspect_runtime_process(snapshot["vitaliy_bot"], base_dir)
    assert info_u["pid"] == pid_unmanaged_before


def test_real_runtime_max_bot_module_execution(runtime_harness, monkeypatch):
    """Test that MAX bot executes with isolated module cwd and PYTHONPATH."""
    base_dir = runtime_harness["base_dir"]
    releases_dir = runtime_harness["releases_dir"]
    pm2 = runtime_harness["pm2"]

    rm.atomic_symlink_switch(base_dir, "max_se13639182_bot_legacy", releases_dir / MOCK_SHA_A)

    max_cwd = base_dir / "deployments" / "max_se13639182_bot_legacy"
    pid_before = pm2.start_process(
        "max_se13639182_bot_legacy",
        script=Path(sys.executable),
        cwd=max_cwd,
        args=["-m", "max_messenger_bot.app"],
        env={"PYTHONPATH": str(max_cwd)},
        app_port=8099,
    )

    def mock_reload(instance_name, pm2_config=None, runtime_env=None):
        target_link = base_dir / "deployments" / instance_name
        pm2.reload_process(
            instance_name,
            new_cwd=target_link,
            new_script=Path(sys.executable),
            args=["-m", "max_messenger_bot.app"],
            env={"PYTHONPATH": str(target_link)},
        )
        return True, "reloaded"

    monkeypatch.setattr(rm, "reload_pm2_instance", mock_reload)

    # Deploy Release B to MAX bot
    ok, msg, details = rm.deploy_canary(
        instance_name="max_se13639182_bot_legacy",
        target_sha=MOCK_SHA_B,
        base_dir=base_dir,
        settle_seconds=0.1,
    )

    assert ok is True
    info = rm.inspect_runtime_process(pm2.metadata["max_se13639182_bot_legacy"], base_dir)
    assert info["resolved_sha"] == MOCK_SHA_B
    assert info["pid"] != pid_before


def test_real_runtime_shared_db_isolation(runtime_harness, monkeypatch):
    """Test that two bots sharing a database url can be upgraded independently."""
    base_dir = runtime_harness["base_dir"]
    releases_dir = runtime_harness["releases_dir"]
    pm2 = runtime_harness["pm2"]

    # Both bots point to Release A
    rm.atomic_symlink_switch(base_dir, "tg_veraveda777_bot_legacy", releases_dir / MOCK_SHA_A)
    rm.atomic_symlink_switch(base_dir, "max_se13639182_bot_legacy", releases_dir / MOCK_SHA_A)

    shared_db_env = {"DATABASE_URL": "postgresql+asyncpg://postgres:secret@localhost:5432/veraveda_db"}

    pid_tg = pm2.start_process(
        "tg_veraveda777_bot_legacy",
        script=base_dir / "deployments" / "tg_veraveda777_bot_legacy" / "main.py",
        cwd=base_dir / "deployments" / "tg_veraveda777_bot_legacy",
        env=shared_db_env,
        app_port=8085,
    )
    pid_max = pm2.start_process(
        "max_se13639182_bot_legacy",
        script=Path(sys.executable),
        cwd=base_dir / "deployments" / "max_se13639182_bot_legacy",
        args=["-m", "max_messenger_bot.app"],
        env=shared_db_env,
        app_port=8099,
    )

    def mock_reload(instance_name, pm2_config=None, runtime_env=None):
        target_link = base_dir / "deployments" / instance_name
        if instance_name.startswith("max_"):
            pm2.reload_process(
                instance_name,
                new_cwd=target_link,
                new_script=Path(sys.executable),
                args=["-m", "max_messenger_bot.app"],
                env=shared_db_env,
            )
        else:
            pm2.reload_process(
                instance_name,
                new_cwd=target_link,
                new_script=target_link / "main.py",
                env=shared_db_env,
            )
        return True, "reloaded"

    monkeypatch.setattr(rm, "reload_pm2_instance", mock_reload)

    # Deploy Release B to Telegram bot only
    ok, msg, details = rm.deploy_canary(
        instance_name="tg_veraveda777_bot_legacy",
        target_sha=MOCK_SHA_B,
        base_dir=base_dir,
        settle_seconds=0.1,
    )

    assert ok is True

    # Telegram bot was updated
    info_tg = rm.inspect_runtime_process(pm2.metadata["tg_veraveda777_bot_legacy"], base_dir)
    assert info_tg["resolved_sha"] == MOCK_SHA_B
    assert info_tg["pid"] != pid_tg

    # MAX bot sharing same DB was NOT touched!
    info_max = rm.inspect_runtime_process(pm2.metadata["max_se13639182_bot_legacy"], base_dir)
    assert info_max["resolved_sha"] == MOCK_SHA_A
    assert info_max["pid"] == pid_max


def test_real_runtime_targeted_rollback(runtime_harness, monkeypatch):
    """Test targeted rollback restores previous release without affecting other instances."""
    base_dir = runtime_harness["base_dir"]
    releases_dir = runtime_harness["releases_dir"]
    pm2 = runtime_harness["pm2"]

    # Bot 1 is running Release B, Bot 2 is running Release A
    rm.atomic_symlink_switch(base_dir, "tg_autobusbusbot_new", releases_dir / MOCK_SHA_B)
    rm.atomic_symlink_switch(base_dir, "tg_someonewithyou01_bot_new", releases_dir / MOCK_SHA_A)

    pid1_b = pm2.start_process(
        "tg_autobusbusbot_new",
        script=base_dir / "deployments" / "tg_autobusbusbot_new" / "main.py",
        cwd=base_dir / "deployments" / "tg_autobusbusbot_new",
        app_port=8081,
    )
    pid2 = pm2.start_process(
        "tg_someonewithyou01_bot_new",
        script=base_dir / "deployments" / "tg_someonewithyou01_bot_new" / "main.py",
        cwd=base_dir / "deployments" / "tg_someonewithyou01_bot_new",
        app_port=8083,
    )

    def mock_reload(instance_name, pm2_config=None, runtime_env=None):
        target_link = base_dir / "deployments" / instance_name
        pm2.reload_process(
            instance_name,
            new_cwd=target_link,
            new_script=target_link / "main.py",
        )
        return True, "reloaded"

    monkeypatch.setattr(rm, "reload_pm2_instance", mock_reload)

    # Roll back Bot 1 to Release A
    ok, msg, details = rm.rollback_instance(
        instance_name="tg_autobusbusbot_new",
        target_sha=MOCK_SHA_A,
        base_dir=base_dir,
        settle_seconds=0.1,
    )

    assert ok is True

    # Bot 1 is now running Release A
    info1 = rm.inspect_runtime_process(pm2.metadata["tg_autobusbusbot_new"], base_dir)
    assert info1["resolved_sha"] == MOCK_SHA_A
    assert info1["pid"] != pid1_b

    # Bot 2 remained untouched
    info2 = rm.inspect_runtime_process(pm2.metadata["tg_someonewithyou01_bot_new"], base_dir)
    assert info2["resolved_sha"] == MOCK_SHA_A
    assert info2["pid"] == pid2


def test_failure_injection_startup_crash_triggers_automatic_rollback(runtime_harness, monkeypatch):
    """Test that a startup crash on deploy automatically triggers targeted rollback to previous release."""
    base_dir = runtime_harness["base_dir"]
    releases_dir = runtime_harness["releases_dir"]
    pm2 = runtime_harness["pm2"]

    # Initially on Release A
    rm.atomic_symlink_switch(base_dir, "tg_autobusbusbot_new", releases_dir / MOCK_SHA_A)
    rm.atomic_symlink_switch(base_dir, "tg_someonewithyou01_bot_new", releases_dir / MOCK_SHA_A)

    pm2.start_process(
        "tg_autobusbusbot_new",
        script=base_dir / "deployments" / "tg_autobusbusbot_new" / "main.py",
        cwd=base_dir / "deployments" / "tg_autobusbusbot_new",
        app_port=8081,
    )
    pid2 = pm2.start_process(
        "tg_someonewithyou01_bot_new",
        script=base_dir / "deployments" / "tg_someonewithyou01_bot_new" / "main.py",
        cwd=base_dir / "deployments" / "tg_someonewithyou01_bot_new",
        app_port=8083,
    )

    def mock_reload(instance_name, pm2_config=None, runtime_env=None):
        target_link = base_dir / "deployments" / instance_name
        pm2.reload_process(
            instance_name,
            new_cwd=target_link,
            new_script=target_link / "main.py",
        )
        return True, "reloaded"

    monkeypatch.setattr(rm, "reload_pm2_instance", mock_reload)

    # Attempt to deploy crashing Release C
    ok, msg, details = rm.deploy_canary(
        instance_name="tg_autobusbusbot_new",
        target_sha=MOCK_SHA_C,
        base_dir=base_dir,
        settle_seconds=0.2,
        auto_rollback=True,
    )

    # Deployment should fail
    assert ok is False
    assert "Health check failed" in msg
    assert "auto-rolled back" in msg

    # Symlink must have been rolled back to Release A!
    link_1 = base_dir / "deployments" / "tg_autobusbusbot_new"
    assert link_1.resolve() == (releases_dir / MOCK_SHA_A).resolve()

    # Bot 2 remained unaffected throughout
    info2 = rm.inspect_runtime_process(pm2.metadata["tg_someonewithyou01_bot_new"], base_dir)
    assert info2["pid"] == pid2
    assert info2["resolved_sha"] == MOCK_SHA_A


def test_status_fleet_reporting_precision(runtime_harness):
    """Test that get_fleet_status accurately reports running SHA, mode, and unmanaged processes."""
    base_dir = runtime_harness["base_dir"]
    releases_dir = runtime_harness["releases_dir"]
    pm2 = runtime_harness["pm2"]

    # Bot 1 on isolated release B
    rm.atomic_symlink_switch(base_dir, "tg_autobusbusbot_new", releases_dir / MOCK_SHA_B)
    pm2.start_process(
        "tg_autobusbusbot_new",
        script=base_dir / "deployments" / "tg_autobusbusbot_new" / "main.py",
        cwd=base_dir / "deployments" / "tg_autobusbusbot_new",
        app_port=8081,
    )

    # Unmanaged process
    unmanaged_dir = base_dir / "darimiru_bot_dir"
    unmanaged_dir.mkdir()
    pm2.start_process(
        "darimiru_bot",
        script=unmanaged_dir / "main.py",
        cwd=unmanaged_dir,
        app_port=8092,
    )

    status = rm.get_fleet_status(
        base_dir=base_dir,
        expected_sha=MOCK_SHA_B,
    )

    # Find Bot 1 in managed status
    m_dict = {m["name"]: m for m in status["managed"]}
    assert "tg_autobusbusbot_new" in m_dict
    assert m_dict["tg_autobusbusbot_new"]["sha_match"] == "MATCH"
    assert m_dict["tg_autobusbusbot_new"]["mode"] == "isolated"

    # Find unmanaged process
    u_names = [u["name"] for u in status["unmanaged"]]
    assert "darimiru_bot" in u_names
    u_dict = {u["name"]: u for u in status["unmanaged"]}
    assert u_dict["darimiru_bot"]["protection"] == "PROTECTED_UNMANAGED"


def test_ghost_port_cleanup_targets_only_configured_port(monkeypatch):
    """Test that cleanup_instance_ghost_port kills ONLY orphan on the specific port."""
    killed: list[int] = []

    def mock_kill(pid, sig):
        killed.append(pid)

    monkeypatch.setattr(os, "kill", mock_kill)
    monkeypatch.setattr(os.path, "exists", lambda p: False)

    def mock_subprocess_run(cmd, *args, **kwargs):
        # cmd is ['lsof', '-t', '-i:8081']
        port_arg = cmd[2]
        if port_arg == "-i:8081":
            # Simulate PID 9999 (orphan) and PID 101 (PM2 managed)
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="9999\n101\n")
        elif port_arg == "-i:8083":
            # Simulate PID 8888 on another bot's port
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="8888\n")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="")

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    # 101 is an active PM2 PID
    pm2_pids = {101, 102}

    # Clean port 8081
    res = rm.cleanup_instance_ghost_port(8081, pm2_pids)

    # Only 9999 should be killed; 101 is managed so never killed; 8888 on port 8083 is untouched!
    assert res == [9999]
    assert 9999 in killed
    assert 101 not in killed
    assert 8888 not in killed


def test_cli_subcommands(tmp_path: Path, monkeypatch):
    """Test CLI commands: preflight, stage, status, verify-dependencies."""
    base_dir = tmp_path / "cli_test"
    base_dir.mkdir()
    (base_dir / "releases").mkdir()
    (base_dir / "deployments").mkdir()
    (base_dir / "newbots").mkdir()

    # Preflight
    test_args = ["release_manager.py", "--base-dir", str(base_dir), "preflight", "--allow-dirty"]
    monkeypatch.setattr(sys, "argv", test_args)
    assert rm.main() == 0

    # Stage
    test_args = ["release_manager.py", "--base-dir", str(base_dir), "stage", "--target-sha", MOCK_SHA_A]
    monkeypatch.setattr(sys, "argv", test_args)
    assert rm.main() == 0
    assert (base_dir / "releases" / MOCK_SHA_A).is_dir()

    # Status
    test_args = ["release_manager.py", "--base-dir", str(base_dir), "status", "--json"]
    monkeypatch.setattr(sys, "argv", test_args)
    assert rm.main() == 0

    # Verify dependencies
    test_args = ["release_manager.py", "--base-dir", str(base_dir), "verify-dependencies"]
    monkeypatch.setattr(sys, "argv", test_args)
    assert rm.main() == 0

