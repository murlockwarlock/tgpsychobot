"""Regression tests reproducing defects in PR #95 before fixes.

1. Staging from dirty git worktree (must extract from git commit tree, not disk).
2. Port cleanup killing unverified/foreign processes (must fail closed).
3. Rollback picking latest mtime among all releases instead of instance history.
4. Overwriting or destroying in-use release via stage --force.
5. PM2 ecosystem ignoring isolated virtualenv.
6. Legacy deploy_prod.sh affecting migrated instances.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

import scripts.release_manager as rm


@pytest.fixture
def git_repo_with_commits(tmp_path: Path) -> tuple[Path, str, str]:
    """Create a temporary git repository with two distinct commits and a dirty worktree."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.email", "tester@example.com"], cwd=str(repo), check=True)

    # Commit 1
    (repo / "main.py").write_text("APP_VERSION = 'v1'\n", encoding="utf-8")
    (repo / "requirements.txt").write_text("aiogram==3.1.1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-m", "commit 1"], cwd=str(repo), check=True)
    sha1 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True, check=True).stdout.strip()

    # Commit 2
    (repo / "main.py").write_text("APP_VERSION = 'v2'\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-m", "commit 2"], cwd=str(repo), check=True)
    sha2 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True, check=True).stdout.strip()

    # Dirty worktree: uncommitted change to main.py
    (repo / "main.py").write_text("APP_VERSION = 'v_dirty_uncommitted'\n", encoding="utf-8")

    return repo, sha1, sha2


def test_regression_stage_release_must_extract_exact_commit_tree(tmp_path: Path, git_repo_with_commits: tuple[Path, str, str]):
    """Regression Test 1: stage_release must extract exact git commit tree, NOT dirty disk worktree."""
    repo, sha1, sha2 = git_repo_with_commits
    base_dir = tmp_path / "bots_root"

    # Request staging of commit 1 (sha1)
    staged_dir = rm.stage_release(
        base_dir=base_dir,
        target_sha=sha1,
        source_dir=repo,
    )

    staged_main = (staged_dir / "main.py").read_text(encoding="utf-8")
    # MUST contain APP_VERSION = 'v1', NOT v2 and definitely NOT v_dirty_uncommitted!
    assert "APP_VERSION = 'v1'" in staged_main, f"Expected v1 from git commit, got:\n{staged_main}"
    assert "v_dirty_uncommitted" not in staged_main
    assert "v2" not in staged_main


def test_regression_port_cleanup_must_not_kill_foreign_process(tmp_path: Path):
    """Regression Test 2: Port cleanup must NEVER kill a foreign non-PM2 process; must fail closed."""
    # Start a dummy server on a free port
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(1)
    port = server_sock.getsockname()[1]

    # Spawn an independent background process (e.g. sleep or python server)
    proc = subprocess.Popen(
        [sys.executable, "-c", f"import time, socket; s=socket.socket(); s.bind(('127.0.0.1', {port})); s.listen(1); time.sleep(60)"],
    )
    server_sock.close()
    time.sleep(0.5)

    try:
        assert proc.poll() is None, "Dummy foreign process should be running"

        # Check port safety using the release manager check_and_resolve_port_conflict
        # It must NOT kill this foreign process and must fail closed!
        ok, msg, killed = rm.check_and_resolve_port_conflict(
            port=port,
            instance_name="tg_autobusbusbot_new",
            base_dir=tmp_path,
            pm2_snapshot={"dummy": {"name": "dummy"}},
            registry={"instances": [{"pm2_name": "tg_autobusbusbot_new", "port": port}]},
        )
        assert ok is False, "Must fail closed when foreign process is bound to port"
        assert "cannot be proven" in msg.lower() or "aborting deploy" in msg.lower()
        assert proc.pid not in killed, f"Foreign PID {proc.pid} was wrongfully killed!"
        assert proc.poll() is None, f"Foreign process {proc.pid} was terminated!"

        # Also verify cleanup_instance_ghost_port with instance_name does not kill unverified foreign process
        killed_ghost = rm.cleanup_instance_ghost_port(port=port, pm2_pids=set(), instance_name="tg_autobusbusbot_new", base_dir=tmp_path)
        assert proc.pid not in killed_ghost
        assert proc.poll() is None
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_regression_rollback_must_use_instance_history_not_mtime(tmp_path: Path):
    """Regression Test 3: Rollback must return to the instance's own previous release, not latest mtime in releases/."""
    base_dir = tmp_path / "bots_root"
    releases_dir = base_dir / "releases"
    deployments_dir = base_dir / "deployments"
    releases_dir.mkdir(parents=True)
    deployments_dir.mkdir(parents=True)

    sha_v1 = "1" * 40
    sha_v2 = "2" * 40

    rel_1 = releases_dir / sha_v1
    rel_1.mkdir()
    (rel_1 / "REVISION").write_text(sha_v1 + "\n", encoding="utf-8")
    (rel_1 / "requirements.sha256").write_text("hash1\n", encoding="utf-8")

    rel_2 = releases_dir / sha_v2
    rel_2.mkdir()
    (rel_2 / "REVISION").write_text(sha_v2 + "\n", encoding="utf-8")
    (rel_2 / "requirements.sha256").write_text("hash2\n", encoding="utf-8")

    # Instance A was upgraded to v2 recently, so rel_2 has newer mtime
    os.utime(rel_1, (time.time() - 100, time.time() - 100))
    os.utime(rel_2, (time.time(), time.time()))

    # Instance B is on v1, and has NEVER been on v2!
    inst_b_link = deployments_dir / "bot_beta"
    os.symlink(rel_1, inst_b_link)

    # If an operator runs rollback on bot_beta (without --target-sha),
    # it must NOT pick rel_2! If there is no previous release for bot_beta, it must fail safely.
    # If bot_beta previously ran v0, it should return to v0.
    registry = {
        "instances": [
            {
                "pm2_name": "bot_beta",
                "platform": "telegram",
                "port": 9999,
                "deploy_managed": True,
                "active": True,
            }
        ]
    }

    ok, msg, details = rm.rollback_instance(
        instance_name="bot_beta",
        target_sha=None,
        base_dir=base_dir,
        registry=registry,
    )

    # In old implementation, it picks rel_2 (latest mtime) and switches bot_beta to v2!
    # In fixed implementation, it must NOT switch to v2!
    assert inst_b_link.resolve() != rel_2, "Rollback wrongfully switched bot_beta to rel_2 (which has newer mtime)!"


def test_regression_force_stage_must_not_destroy_in_use_release(tmp_path: Path):
    """Regression Test 4: stage_release with force=True must NOT delete a release currently in use."""
    base_dir = tmp_path / "bots_root"
    releases_dir = base_dir / "releases"
    deployments_dir = base_dir / "deployments"
    releases_dir.mkdir(parents=True)
    deployments_dir.mkdir(parents=True)

    sha = "a" * 40
    target_rel = releases_dir / sha
    target_rel.mkdir()
    (target_rel / "REVISION").write_text(sha + "\n", encoding="utf-8")
    (target_rel / "requirements.sha256").write_text("h\n", encoding="utf-8")
    (target_rel / "canary.txt").write_text("in_use_marker\n", encoding="utf-8")

    # Instance is actively pointing to target_rel
    os.symlink(target_rel, deployments_dir / "bot_alpha")

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "main.py").write_text("print('overwrite')\n", encoding="utf-8")
    (source_dir / "requirements.txt").write_text("aiogram==3.1.1\n", encoding="utf-8")

    # Trying to force re-stage active release must raise error or preserve running release
    with pytest.raises(Exception) as exc_info:
        rm.stage_release(
            base_dir=base_dir,
            target_sha=sha,
            source_dir=source_dir,
            force=True,
        )
    assert "protected" in str(exc_info.value).lower() or "in use" in str(exc_info.value).lower()


def test_regression_ecosystem_isolated_python_interpreter(tmp_path: Path):
    """Regression Test 5: ecosystem.config.js must select isolated venv if present."""
    base_dir = tmp_path / "bots_root"
    deployments_dir = base_dir / "deployments"
    deployments_dir.mkdir(parents=True)

    # 1. Telegram bot with isolated venv
    tg_dir = deployments_dir / "tg_autobusbusbot_new"
    tg_venv_py = tg_dir / "venv" / "bin" / "python"
    tg_venv_py.parent.mkdir(parents=True)
    tg_venv_py.write_text("#!/bin/sh\n", encoding="utf-8")

    # 2. MAX bot with isolated venv
    max_dir = deployments_dir / "max_id519010411655_bot_new"
    max_venv_py = max_dir / "venv" / "bin" / "python"
    max_venv_py.parent.mkdir(parents=True)
    max_venv_py.write_text("#!/bin/sh\n", encoding="utf-8")

    # Evaluate ecosystem.config.js using node
    node_script = f"""
    process.env.PROD_BASE_DIR = '{base_dir}';
    process.env.PROD_DEPLOYMENTS_DIR = '{deployments_dir}';
    const config = require('{rm.REPO_ROOT}/ecosystem.config.js');
    const apps = config.apps;
    const tg = apps.find(a => a.name === 'tg_autobusbusbot_new');
    const max = apps.find(a => a.name === 'max_id519010411655_bot_new');
    console.log(JSON.stringify({{
        tg_interpreter: tg.interpreter,
        tg_cwd: tg.cwd,
        max_script: max.script,
        max_interpreter: max.interpreter,
        max_cwd: max.cwd,
    }}));
    """
    proc = subprocess.run(
        ["node", "-e", node_script],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(proc.stdout)
    assert result["tg_interpreter"] == str(tg_venv_py), f"Expected TG interpreter {tg_venv_py}, got {result['tg_interpreter']}"
    assert result["max_script"] == str(max_venv_py), f"Expected MAX script {max_venv_py}, got {result['max_script']}"


def test_regression_verify_bot_instances_rejects_legacy_deploy_for_migrated_instance(tmp_path: Path):
    """Regression Test 6: verify_bot_instances --check-legacy-safety must reject migrated instances."""
    base_dir = tmp_path / "bots_root"
    deployments_dir = base_dir / "deployments"
    deployments_dir.mkdir(parents=True)

    # Migrate tg_autobusbusbot_new
    os.symlink(tmp_path, deployments_dir / "tg_autobusbusbot_new")

    # Call verify_bot_instances.py with --check-legacy-safety
    cmd = [
        sys.executable,
        str(rm.REPO_ROOT / "scripts" / "verify_bot_instances.py"),
        "--check-legacy-safety",
        "--base-dir",
        str(base_dir),
        "--pm2-names",
        "tg_autobusbusbot_new",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0, f"Expected non-zero exit code when targeting migrated instance, got {proc.returncode}"
    assert "migrated to isolated release management" in proc.stdout or "legacy_deploy_safety=failed" in proc.stdout

    # Now call for an unmigrated instance: tg_someonewithyou01_bot_new
    cmd_unmigrated = [
        sys.executable,
        str(rm.REPO_ROOT / "scripts" / "verify_bot_instances.py"),
        "--check-legacy-safety",
        "--base-dir",
        str(base_dir),
        "--pm2-names",
        "tg_someonewithyou01_bot_new",
    ]
    proc_unmigrated = subprocess.run(cmd_unmigrated, capture_output=True, text=True)
    assert proc_unmigrated.returncode == 0
    assert "legacy_deploy_safety=ok" in proc_unmigrated.stdout

