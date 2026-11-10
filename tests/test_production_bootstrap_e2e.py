"""End-to-end integration test verifying production bootstrap with real PM2 daemon.

Requirements:
1. Real PM2 daemon with isolated PM2_HOME.
2. Old legacy /newbots directory layout WITHOUT .git repository.
3. Real ecosystem.config.js evaluating runtime paths and environment.
4. Packaging release artifact from Git commit (pack command with MANIFEST.sha256 and digest).
5. Delivering artifact, staging, initial canary deploy (cutover from legacy to isolated).
6. Deep readiness verification.
7. Rollback back to legacy newbots.
8. Re-deploy back to isolated release.
9. Verify neighbor managed bot and unmanaged bot are never disturbed throughout the lifecycle.
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
import urllib.request

import pytest

import scripts.release_manager as rm


def get_free_port() -> int:
    """Allocate an available local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_real_pm2_bin() -> str:
    """Find real PM2 binary; fail if missing."""
    bin_path = shutil.which("pm2") or "/opt/homebrew/bin/pm2"
    if not (bin_path and os.path.isfile(bin_path) and os.access(bin_path, os.X_OK)):
        pytest.fail("Real PM2 binary is required for this bootstrap suite, but 'pm2' was not found.")
    return str(bin_path)


def test_production_bootstrap_and_isolated_lifecycle_e2e(tmp_path: Path, monkeypatch):
    """Full production bootstrap lifecycle test with real PM2 and actual shell CLI invocations."""
    pm2_bin = get_real_pm2_bin()
    pm2_home = Path(tempfile.mkdtemp(prefix="pm2_boot_", dir="/tmp"))
    base_dir = tmp_path / "telegram_bots"
    base_dir.mkdir(parents=True, exist_ok=True)
    pm2_home.mkdir(parents=True, exist_ok=True)

    git_repo_dir = tmp_path / "git_source_repo"
    git_repo_dir.mkdir(parents=True, exist_ok=True)

    port_prod = get_free_port()
    port_neighbor = get_free_port()
    port_unmanaged = get_free_port()

    target_instance = "tg_autobusbusbot_new"
    neighbor_instance = "tg_someonewithyou01_bot_new"
    unmanaged_instance = "vitaliy_bot"

    # Set PM2 environment variables for the test process
    pm2_env = {
        **os.environ,
        "PM2_HOME": str(pm2_home),
        "PROD_BASE_DIR": str(base_dir),
        "PROD_REMOTE_DIR": str(base_dir / "newbots"),
        "PROD_REMOTE_PY": sys.executable,
        "PROD_RUNTIME_ENV": str(base_dir / "shared" / "runtime.env"),
        "PATH": f"{Path(pm2_bin).parent}:{os.environ.get('PATH', '')}",
    }
    monkeypatch.setenv("PM2_HOME", str(pm2_home))
    monkeypatch.setenv("PM2_CMD", pm2_bin)
    monkeypatch.setenv("PROD_BASE_DIR", str(base_dir))
    monkeypatch.setenv("PROD_REMOTE_DIR", str(base_dir / "newbots"))
    monkeypatch.setenv("PROD_REMOTE_PY", sys.executable)
    monkeypatch.setenv("PROD_RUNTIME_ENV", str(base_dir / "shared" / "runtime.env"))

    # Cleanup PM2 daemon at teardown
    def cleanup():
        subprocess.run([pm2_bin, "kill"], env=pm2_env, capture_output=True)
        shutil.rmtree(pm2_home, ignore_errors=True)

    # 1. Setup shared runtime environment
    shared_dir = base_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    runtime_env_file = shared_dir / "runtime.env"
    runtime_env_file.write_text(f"""
TELEGRAM_AUTOBUSBUSBOT_TOKEN=test_token_alpha
TELEGRAM_SOMEONEWITHYOU01BOT_TOKEN=test_token_neighbor
DATABASE_URL=sqlite:///{base_dir}/shared/test.db
""", encoding="utf-8")

    # 2. Setup Git Source Repository (authoritative origin of releases)
    subprocess.run(["git", "init"], cwd=str(git_repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Bootstrap Builder"], cwd=str(git_repo_dir), check=True)
    subprocess.run(["git", "config", "user.email", "builder@example.com"], cwd=str(git_repo_dir), check=True)

    worker_code_v1 = f"""import http.server, os, signal, sys, time

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'RELEASE_V1')
    def log_message(self, *args):
        pass

signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

port = int(os.environ.get('APP_PORT', '0'))
if port > 0:
    server = http.server.HTTPServer(('127.0.0.1', port), Handler)
    server.serve_forever()
else:
    while True:
        time.sleep(0.2)
"""

    (git_repo_dir / "main.py").write_text(worker_code_v1, encoding="utf-8")
    (git_repo_dir / "requirements.txt").write_text("aiogram==3.1.1\n", encoding="utf-8")
    (git_repo_dir / "max_messenger_bot").mkdir(parents=True)
    (git_repo_dir / "max_messenger_bot" / "__init__.py").write_text("", encoding="utf-8")
    (git_repo_dir / "max_messenger_bot" / "app.py").write_text("import time\nwhile True: time.sleep(0.2)\n", encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=str(git_repo_dir), check=True)
    subprocess.run(["git", "commit", "-m", "release v1"], cwd=str(git_repo_dir), check=True)
    sha1 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(git_repo_dir), capture_output=True, text=True, check=True).stdout.strip()

    # 3. Setup Legacy Production Layout: /root/telegram_bots/newbots WITHOUT .git directory!
    legacy_dir = base_dir / "newbots"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_worker_code = f"""import http.server, os, signal, sys, time

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'LEGACY_CODE')
    def log_message(self, *args):
        pass

signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

port = int(os.environ.get('APP_PORT', '0'))
if port > 0:
    server = http.server.HTTPServer(('127.0.0.1', port), Handler)
    server.serve_forever()
else:
    while True:
        time.sleep(0.2)
"""
    (legacy_dir / "main.py").write_text(legacy_worker_code, encoding="utf-8")
    (legacy_dir / "requirements.txt").write_text("aiogram==3.1.1\n", encoding="utf-8")
    (legacy_dir / "REVISION").write_text("legacy_initial_rev\n", encoding="utf-8")
    # Verify legacy_dir has NO .git
    assert not (legacy_dir / ".git").exists()

    # 4. Setup Registry and real ecosystem.config.js
    config_dir = base_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    registry_file = config_dir / "bot_instances.json"
    
    # Read real registry and adjust ports
    real_registry_src = Path(__file__).resolve().parent.parent / "config" / "bot_instances.json"
    reg_obj = json.loads(real_registry_src.read_text(encoding="utf-8"))
    for inst in reg_obj.get("instances", []):
        if inst.get("pm2_name") == target_instance:
            inst["port"] = port_prod
        elif inst.get("pm2_name") == neighbor_instance:
            inst["port"] = port_neighbor
    registry_file.write_text(json.dumps(reg_obj, indent=2), encoding="utf-8")

    # Read real ecosystem.config.js and adjust ports
    real_ecosystem_src = Path(__file__).resolve().parent.parent / "ecosystem.config.js"
    eco_content = real_ecosystem_src.read_text(encoding="utf-8")
    eco_content = eco_content.replace('"APP_PORT": 8081,', f'"APP_PORT": {port_prod},')
    eco_content = eco_content.replace('"APP_PORT": 8083,', f'"APP_PORT": {port_neighbor},')
    ecosystem_file = base_dir / "ecosystem.config.js"
    ecosystem_file.write_text(eco_content, encoding="utf-8")

    # Copy scripts to base_dir/scripts so remote commands can execute them
    scripts_src = Path(__file__).resolve().parent.parent / "scripts"
    scripts_dest = base_dir / "scripts"
    shutil.copytree(scripts_src, scripts_dest)

    # Unmanaged worker script
    unmanaged_script = base_dir / "vitaliy_bot.py"
    unmanaged_script.write_text(f"""import time, signal, sys
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
while True:
    time.sleep(0.5)
""", encoding="utf-8")

    try:
        # 5. Start PM2 fleet in legacy state
        start_cmd = [
            pm2_bin, "start", str(ecosystem_file),
            "--only", f"{target_instance},{neighbor_instance}"
        ]
        subprocess.run(start_cmd, env=pm2_env, capture_output=True, check=True)

        # Start unmanaged process
        subprocess.run(
            [pm2_bin, "start", str(unmanaged_script), "--name", unmanaged_instance, "--interpreter", sys.executable],
            env=pm2_env, capture_output=True, check=True
        )
        time.sleep(2.0)

        # Verify initial legacy state
        snap_0 = rm.load_pm2_snapshot()
        info_prod_0 = rm.inspect_runtime_process(snap_0[target_instance], base_dir)
        info_neigh_0 = rm.inspect_runtime_process(snap_0[neighbor_instance], base_dir)
        info_unman_0 = rm.inspect_runtime_process(snap_0[unmanaged_instance], base_dir)

        assert info_prod_0["mode"] == "legacy"
        assert info_neigh_0["mode"] == "legacy"
        pid_neigh_0 = info_neigh_0["pid"]
        pid_unman_0 = info_unman_0["pid"]

        # Verify initial HTTP response is LEGACY_CODE
        req = urllib.request.urlopen(f"http://127.0.0.1:{port_prod}", timeout=2.0)
        assert req.read().decode("utf-8") == "LEGACY_CODE"

        # 6. Package release artifact from Git repository using actual CLI `pack` command
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        archive_path = artifact_dir / f"release_{sha1}.tar.gz"

        pack_cmd = [
            sys.executable,
            str(scripts_dest / "release_manager.py"),
            "pack",
            "--source-dir", str(git_repo_dir),
            "--target-sha", sha1,
            "--output", str(archive_path),
        ]
        pack_res = subprocess.run(pack_cmd, env=pm2_env, capture_output=True, text=True)
        assert pack_res.returncode == 0, f"pack failed: {pack_res.stderr}"
        assert archive_path.is_file()
        assert archive_path.with_name(f"{archive_path.name}.sha256").is_file()

        # 7. Deliver and stage artifact on server (without git) using actual CLI `stage` command
        stage_cmd = [
            sys.executable,
            str(scripts_dest / "release_manager.py"),
            "--base-dir", str(base_dir),
            "stage",
            "--target-sha", sha1,
            "--source-dir", str(legacy_dir),  # Non-git source!
            "--archive-path", str(archive_path),
        ]
        stage_res = subprocess.run(stage_cmd, env=pm2_env, capture_output=True, text=True)
        assert stage_res.returncode == 0, f"stage failed: {stage_res.stderr}"

        staged_release_dir = base_dir / "releases" / sha1
        assert staged_release_dir.is_dir()
        assert (staged_release_dir / "MANIFEST.sha256").is_file()

        # 8. First Deploy: Cutover target_instance from legacy to isolated layout using CLI `deploy`
        deploy_cmd = [
            sys.executable,
            str(scripts_dest / "release_manager.py"),
            "--base-dir", str(base_dir),
            "--registry", str(registry_file),
            "--pm2-config", str(ecosystem_file),
            "--runtime-env", str(runtime_env_file),
            "deploy",
            "--canary", target_instance,
            "--target-sha", sha1,
            "--settle-seconds", "1.5",
            "--skip-preflight",
        ]
        deploy_res = subprocess.run(deploy_cmd, env=pm2_env, capture_output=True, text=True)
        assert deploy_res.returncode == 0, f"deploy failed: {deploy_res.stderr}\nstdout: {deploy_res.stdout}"

        # Verify target_instance is now running in isolated mode on sha1
        snap_1 = rm.load_pm2_snapshot()
        info_prod_1 = rm.inspect_runtime_process(snap_1[target_instance], base_dir)
        info_neigh_1 = rm.inspect_runtime_process(snap_1[neighbor_instance], base_dir)
        info_unman_1 = rm.inspect_runtime_process(snap_1[unmanaged_instance], base_dir)

        assert info_prod_1["mode"] == "isolated"
        assert info_prod_1["resolved_sha"] == sha1
        assert info_neigh_1["mode"] == "legacy", "Neighbor bot must remain in legacy mode"
        assert info_neigh_1["pid"] == pid_neigh_0, "Neighbor bot must not be restarted!"
        assert info_unman_1["pid"] == pid_unman_0, "Unmanaged bot must not be restarted!"

        # Verify traffic serving
        req_1 = urllib.request.urlopen(f"http://127.0.0.1:{port_prod}", timeout=2.0)
        assert req_1.read().decode("utf-8") == "RELEASE_V1"

        # 9. Rollback target_instance back to legacy layout using actual CLI `rollback` command
        rollback_cmd = [
            sys.executable,
            str(scripts_dest / "release_manager.py"),
            "--base-dir", str(base_dir),
            "--registry", str(registry_file),
            "--pm2-config", str(ecosystem_file),
            "--runtime-env", str(runtime_env_file),
            "rollback",
            "--instance", target_instance,
            "--settle-seconds", "1.5",
        ]
        rollback_res = subprocess.run(rollback_cmd, env=pm2_env, capture_output=True, text=True)
        assert rollback_res.returncode == 0, f"rollback failed: {rollback_res.stderr}\nstdout: {rollback_res.stdout}"

        # Verify target_instance returned to legacy mode
        snap_2 = rm.load_pm2_snapshot()
        info_prod_2 = rm.inspect_runtime_process(snap_2[target_instance], base_dir)
        assert info_prod_2["mode"] == "legacy"
        # Neighbor and unmanaged bots still untouched
        info_neigh_2 = rm.inspect_runtime_process(snap_2[neighbor_instance], base_dir)
        assert info_neigh_2["pid"] == pid_neigh_0
        info_unman_2 = rm.inspect_runtime_process(snap_2[unmanaged_instance], base_dir)
        assert info_unman_2["pid"] == pid_unman_0

        req_2 = urllib.request.urlopen(f"http://127.0.0.1:{port_prod}", timeout=2.0)
        assert req_2.read().decode("utf-8") == "LEGACY_CODE"

        # 10. Repeat deploy: re-deploy to sha1
        redeploy_res = subprocess.run(deploy_cmd, env=pm2_env, capture_output=True, text=True)
        assert redeploy_res.returncode == 0, f"re-deploy failed: {redeploy_res.stderr}\nstdout: {redeploy_res.stdout}"

        snap_3 = rm.load_pm2_snapshot()
        info_prod_3 = rm.inspect_runtime_process(snap_3[target_instance], base_dir)
        assert info_prod_3["mode"] == "isolated"
        assert info_prod_3["resolved_sha"] == sha1

        req_3 = urllib.request.urlopen(f"http://127.0.0.1:{port_prod}", timeout=2.0)
        assert req_3.read().decode("utf-8") == "RELEASE_V1"

        # Verify state file accurately recorded previous target as legacy
        state = rm.load_instance_state(base_dir, target_instance)
        assert state["previous_successful_target_dir"] == "legacy" or "newbots" in state["previous_successful_target_dir"]

    finally:
        cleanup()
