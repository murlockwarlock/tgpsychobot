"""Mandatory Integration Test Suite using REAL PM2 Process Manager.

Verifies the full lifecycle with real PM2 daemons and actual Python processes:
1. Physical release isolation and dynamic ecosystem resolution.
2. Targeted canary deploy updates ONLY the selected instance.
3. Neighbor instances retain PID, cwd, and runtime SHA.
4. Unmanaged processes (e.g. vitaliy_bot) are never touched.
5. MAX bot runs via module `-m max_messenger_bot.app`.
6. Startup crash triggers automatic rollback to previous verified release.
7. Rollback uses instance-specific history.
8. Multiple versions run concurrently on real PM2.
9. Pruning strictly protects active and rollback releases.
10. Unconditional teardown kills isolated PM2 daemon cleanly.
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


def get_free_port() -> int:
    """Allocate an available local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_real_pm2_bin() -> str:
    """Find real PM2 binary; fails if missing (no mocking)."""
    bin_path = shutil.which("pm2") or "/opt/homebrew/bin/pm2"
    if not (bin_path and os.path.isfile(bin_path) and os.access(bin_path, os.X_OK)):
        pytest.fail("Real PM2 binary is required for this integration suite, but 'pm2' was not found.")
    return str(bin_path)


@pytest.fixture
def real_pm2_harness(tmp_path: Path):
    """Set up an isolated environment with separate PM2_HOME and real Git commits."""
    pm2_bin = get_real_pm2_bin()
    base_dir = tmp_path / "bots_root"
    # Unix domain socket sun_path is limited to 104 chars on macOS; use short /tmp path
    pm2_home = Path(tempfile.mkdtemp(prefix="pm2_h_", dir="/tmp"))
    repo_dir = tmp_path / "repo"

    base_dir.mkdir(parents=True, exist_ok=True)
    pm2_home.mkdir(parents=True, exist_ok=True)
    repo_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = base_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    runtime_env_file = shared_dir / "runtime.env"
    runtime_env_file.write_text("# Test isolated runtime env\n", encoding="utf-8")

    # Free ports for 3 managed instances and 1 unmanaged instance
    port_alpha = get_free_port()
    port_beta = get_free_port()
    port_gamma = get_free_port()
    port_unmanaged = get_free_port()

    # 1. Initialize Git repository
    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "PM2 Real Tester"], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "config", "user.email", "tester@example.com"], cwd=str(repo_dir), check=True)

    # Worker script templates
    def write_main_py(version: str, crash_on_start: bool = False):
        if crash_on_start:
            return "import sys\nsys.exit(42)\n"
        return f"""import http.server, os, signal, sys, time

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'{version}')
    def log_message(self, *args):
        pass

def handle_term(signum, frame):
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_term)
signal.signal(signal.SIGINT, handle_term)

port = int(os.environ.get('APP_PORT', '0'))
if port > 0:
    server = http.server.HTTPServer(('127.0.0.1', port), Handler)
    server.serve_forever()
else:
    while True:
        time.sleep(0.2)
"""

    def write_max_app_py(version: str):
        return f"""import http.server, os, signal, sys, time

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'MAX_{version}')
    def log_message(self, *args):
        pass

def handle_term(signum, frame):
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_term)
signal.signal(signal.SIGINT, handle_term)

port = int(os.environ.get('MAX_APP_PORT', '0'))
if port > 0:
    server = http.server.HTTPServer(('127.0.0.1', port), Handler)
    server.serve_forever()
else:
    while True:
        time.sleep(0.2)
"""

    (repo_dir / "requirements.txt").write_text("aiogram==3.1.1\n", encoding="utf-8")
    (repo_dir / "max_messenger_bot").mkdir(parents=True)
    (repo_dir / "max_messenger_bot" / "__init__.py").write_text("", encoding="utf-8")

    # Commit 1: v1
    (repo_dir / "main.py").write_text(write_main_py("v1"), encoding="utf-8")
    (repo_dir / "max_messenger_bot" / "app.py").write_text(write_max_app_py("v1"), encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "commit", "-m", "commit 1"], cwd=str(repo_dir), check=True)
    sha1 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True, check=True).stdout.strip()

    # Commit 2: v2
    (repo_dir / "main.py").write_text(write_main_py("v2"), encoding="utf-8")
    (repo_dir / "max_messenger_bot" / "app.py").write_text(write_max_app_py("v2"), encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "commit", "-m", "commit 2"], cwd=str(repo_dir), check=True)
    sha2 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True, check=True).stdout.strip()

    # Commit 3: v3
    (repo_dir / "main.py").write_text(write_main_py("v3"), encoding="utf-8")
    (repo_dir / "max_messenger_bot" / "app.py").write_text(write_max_app_py("v3"), encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "commit", "-m", "commit 3"], cwd=str(repo_dir), check=True)
    sha3 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True, check=True).stdout.strip()

    # Commit 4: crashing release
    (repo_dir / "main.py").write_text(write_main_py("v4_crash", crash_on_start=True), encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "commit", "-m", "commit 4 crash"], cwd=str(repo_dir), check=True)
    sha4_crash = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True, check=True).stdout.strip()

    # Create dummy unmanaged script
    unmanaged_script = base_dir / "unmanaged_bot.py"
    unmanaged_script.write_text(f"""import time, signal, sys
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
while True:
    time.sleep(0.5)
""", encoding="utf-8")

    # Create registry
    registry = {
        "schema_version": 1,
        "instances": [
            {
                "pm2_name": "tg_bot_alpha",
                "platform": "telegram",
                "username": "bot_alpha",
                "platform_id": 1001,
                "port": port_alpha,
                "database": "shared_db",
                "database_env": "DB_URL",
                "runtime_database_env": "DB_URL",
                "token_env": "TELEGRAM_ALPHA_TOKEN",
                "runtime_token_env": "TELEGRAM_ALPHA_TOKEN",
                "mode": "new",
                "profile": "default",
                "deploy_managed": True,
                "active": True,
            },
            {
                "pm2_name": "tg_bot_beta",
                "platform": "telegram",
                "username": "bot_beta",
                "platform_id": 1002,
                "port": port_beta,
                "database": "shared_db",
                "database_env": "DB_URL",
                "runtime_database_env": "DB_URL",
                "token_env": "TELEGRAM_BETA_TOKEN",
                "runtime_token_env": "TELEGRAM_BETA_TOKEN",
                "mode": "new",
                "profile": "default",
                "deploy_managed": True,
                "active": True,
            },
            {
                "pm2_name": "max_bot_gamma",
                "platform": "max",
                "username": "bot_gamma",
                "platform_id": 1003,
                "port": port_gamma,
                "database": "shared_db",
                "database_env": "DB_URL",
                "runtime_database_env": "DB_URL",
                "token_env": "MAX_GAMMA_TOKEN",
                "runtime_token_env": "MAX_GAMMA_TOKEN",
                "mode": "new",
                "profile": "default",
                "deploy_managed": True,
                "active": True,
            },
        ]
    }

    registry_path = base_dir / "bot_instances.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    # Generate real ecosystem.config.js for test environment
    ecosystem_path = base_dir / "ecosystem.config.js"
    ecosystem_content = f"""const fs = require('fs');
const path = require('path');

const PM2_LOG_DIR = path.join(process.env.PM2_HOME, 'logs');
const BASE_BOTS_DIR = '{base_dir}';
const DEPLOYMENTS_DIR = path.join(BASE_BOTS_DIR, 'deployments');
const REMOTE_PY = '{sys.executable}';

const apps = [
  {{
    name: 'tg_bot_alpha',
    script: path.join(BASE_BOTS_DIR, 'newbots', 'main.py'),
    env: {{
      APP_PORT: {port_alpha},
      BOT_TOKEN: 'token_alpha',
    }}
  }},
  {{
    name: 'tg_bot_beta',
    script: path.join(BASE_BOTS_DIR, 'newbots', 'main.py'),
    env: {{
      APP_PORT: {port_beta},
      BOT_TOKEN: 'token_beta',
    }}
  }},
  {{
    name: 'max_bot_gamma',
    script: REMOTE_PY,
    args: '-m max_messenger_bot.app',
    interpreter: 'none',
    env: {{
      MAX_APP_PORT: {port_gamma},
      MAX_BOT_TOKEN: 'token_gamma',
    }}
  }}
];

apps.forEach(app => {{
  app.env = app.env || {{}};
  app.env.BOT_INSTANCE_ID = app.name;

  const deploymentDir = path.join(DEPLOYMENTS_DIR, app.name);
  const isIsolated = fs.existsSync(deploymentDir);
  const appRoot = isIsolated ? deploymentDir : path.join(BASE_BOTS_DIR, 'newbots');

  let pythonInterpreter = REMOTE_PY;
  const isolatedVenvPy = path.join(appRoot, 'venv', 'bin', 'python');
  if (fs.existsSync(isolatedVenvPy)) {{
    pythonInterpreter = isolatedVenvPy;
  }}

  if (app.name && app.name.startsWith('max_')) {{
    app.script = pythonInterpreter;
    app.args = '-m max_messenger_bot.app';
    app.interpreter = 'none';
    app.cwd = appRoot;
  }} else {{
    app.script = path.join(appRoot, 'main.py');
    app.cwd = appRoot;
    app.interpreter = pythonInterpreter;
  }}

  app.env.APP_DIR = appRoot;
  app.env.PYTHONPATH = appRoot;

  const revFile = path.join(appRoot, 'REVISION');
  if (fs.existsSync(revFile)) {{
    try {{
      app.env.APP_REVISION = fs.readFileSync(revFile, 'utf-8').trim();
    }} catch (_) {{}}
  }}
}});

module.exports = {{ apps: apps }};
"""
    ecosystem_path.write_text(ecosystem_content, encoding="utf-8")

    pm2_env = os.environ.copy()
    pm2_env["PM2_HOME"] = str(pm2_home)
    pm2_env["PM2_CMD"] = pm2_bin
    pm2_env["PROD_BASE_DIR"] = str(base_dir)
    pm2_env["PROD_RUNTIME_ENV"] = str(runtime_env_file)

    yield {
        "pm2_bin": pm2_bin,
        "pm2_env": pm2_env,
        "base_dir": base_dir,
        "pm2_home": pm2_home,
        "runtime_env_file": runtime_env_file,
        "repo_dir": repo_dir,
        "ecosystem_path": ecosystem_path,
        "registry": registry,
        "registry_path": registry_path,
        "sha1": sha1,
        "sha2": sha2,
        "sha3": sha3,
        "sha4_crash": sha4_crash,
        "port_alpha": port_alpha,
        "port_beta": port_beta,
        "port_gamma": port_gamma,
        "port_unmanaged": port_unmanaged,
        "unmanaged_script": unmanaged_script,
    }

    # Strict teardown: kill test PM2 daemon unconditionally
    try:
        subprocess.run([pm2_bin, "kill"], env=pm2_env, capture_output=True, check=False)
    except Exception:
        pass
    shutil.rmtree(pm2_home, ignore_errors=True)


def test_real_pm2_isolated_fleet_lifecycle(real_pm2_harness: dict[str, Any], monkeypatch):
    """Full lifecycle with real PM2: canary deploy, isolation, rollback, crash auto-rollback, and safe prune."""
    h = real_pm2_harness
    pm2_bin = h["pm2_bin"]
    pm2_env = h["pm2_env"]
    base_dir = h["base_dir"]
    ecosystem_path = h["ecosystem_path"]
    repo_dir = h["repo_dir"]
    registry = h["registry"]
    sha1 = h["sha1"]
    sha2 = h["sha2"]
    sha3 = h["sha3"]
    sha4_crash = h["sha4_crash"]

    # Point release_manager to isolated PM2 environment
    monkeypatch.setenv("PM2_HOME", str(h["pm2_home"]))
    monkeypatch.setenv("PM2_CMD", pm2_bin)
    monkeypatch.setenv("PROD_BASE_DIR", str(base_dir))
    monkeypatch.setenv("PROD_RUNTIME_ENV", str(h["runtime_env_file"]))

    # 1. Stage Release 1 from Git
    rel1_dir = rm.stage_release(base_dir=base_dir, target_sha=sha1, source_dir=repo_dir)
    assert rel1_dir.is_dir()
    assert (rel1_dir / "REVISION").read_text(encoding="utf-8").strip() == sha1

    # 2. Deploy initial v1 to all 3 bots
    rm.atomic_symlink_switch(base_dir, "tg_bot_alpha", rel1_dir)
    rm.atomic_symlink_switch(base_dir, "tg_bot_beta", rel1_dir)
    rm.atomic_symlink_switch(base_dir, "max_bot_gamma", rel1_dir)

    # Start managed apps with real PM2
    start_proc = subprocess.run(
        [pm2_bin, "start", str(ecosystem_path)],
        env=pm2_env,
        capture_output=True,
        text=True,
        check=True,
    )
    time.sleep(2.0)

    # Start unmanaged process with real PM2
    start_unmanaged = subprocess.run(
        [pm2_bin, "start", str(h["unmanaged_script"]), "--name", "vitaliy_bot"],
        env=pm2_env,
        capture_output=True,
        text=True,
        check=True,
    )
    time.sleep(1.0)

    # Verify all 4 processes are running in real PM2 snapshot
    snapshot_1 = rm.load_pm2_snapshot()
    assert "tg_bot_alpha" in snapshot_1
    assert "tg_bot_beta" in snapshot_1
    assert "max_bot_gamma" in snapshot_1
    assert "vitaliy_bot" in snapshot_1

    pid_alpha_1 = snapshot_1["tg_bot_alpha"]["pid"]
    pid_beta_1 = snapshot_1["tg_bot_beta"]["pid"]
    pid_gamma_1 = snapshot_1["max_bot_gamma"]["pid"]
    pid_vitaliy_1 = snapshot_1["vitaliy_bot"]["pid"]

    assert pid_alpha_1 and pid_alpha_1 > 0
    assert pid_beta_1 and pid_beta_1 > 0
    assert pid_gamma_1 and pid_gamma_1 > 0
    assert pid_vitaliy_1 and pid_vitaliy_1 > 0

    # 3. Stage Release 2 from Git
    rel2_dir = rm.stage_release(base_dir=base_dir, target_sha=sha2, source_dir=repo_dir)
    assert rel2_dir.is_dir()
    assert (rel2_dir / "REVISION").read_text(encoding="utf-8").strip() == sha2

    # 4. CANARY DEPLOY Release 2 to tg_bot_alpha ONLY
    ok, msg, details = rm.deploy_canary(
        instance_name="tg_bot_alpha",
        target_sha=sha2,
        base_dir=base_dir,
        pm2_config=ecosystem_path,
        settle_seconds=2.0,
        registry=registry,
    )
    assert ok is True, f"Canary deploy failed: {msg}"
    time.sleep(1.0)

    # Verify tg_bot_alpha was updated in real PM2 runtime
    snapshot_2 = rm.load_pm2_snapshot()
    alpha_info = rm.inspect_runtime_process(snapshot_2["tg_bot_alpha"], base_dir=base_dir)
    assert alpha_info["resolved_sha"] == sha2
    assert alpha_info["status"] == "online"
    assert alpha_info["pid"] != pid_alpha_1, "Alpha PID should have updated on reload"

    # CRITICAL: Verify neighbor instances were NOT touched
    beta_info = rm.inspect_runtime_process(snapshot_2["tg_bot_beta"], base_dir=base_dir)
    assert beta_info["pid"] == pid_beta_1, "Beta PID must remain identical!"
    assert beta_info["resolved_sha"] == sha1, "Beta must still run sha1!"

    vitaliy_info = rm.inspect_runtime_process(snapshot_2["vitaliy_bot"], base_dir=base_dir)
    assert vitaliy_info["pid"] == pid_vitaliy_1, "Unmanaged vitaliy_bot must NEVER be touched!"

    # 5. CANARY DEPLOY Release 2 to MAX Bot max_bot_gamma
    ok_max, msg_max, details_max = rm.deploy_canary(
        instance_name="max_bot_gamma",
        target_sha=sha2,
        base_dir=base_dir,
        pm2_config=ecosystem_path,
        settle_seconds=2.0,
        registry=registry,
    )
    assert ok_max is True, f"MAX bot canary deploy failed: {msg_max}"
    time.sleep(1.0)

    snapshot_3 = rm.load_pm2_snapshot()
    gamma_info = rm.inspect_runtime_process(snapshot_3["max_bot_gamma"], base_dir=base_dir)
    assert gamma_info["resolved_sha"] == sha2
    assert gamma_info["status"] == "online"
    assert gamma_info["mode"] == "isolated"

    # Beta and unmanaged still unchanged
    assert snapshot_3["tg_bot_beta"]["pid"] == pid_beta_1
    assert snapshot_3["vitaliy_bot"]["pid"] == pid_vitaliy_1

    # 6. Deploy Release 3 to tg_bot_alpha
    rm.stage_release(base_dir=base_dir, target_sha=sha3, source_dir=repo_dir)
    ok_v3, msg_v3, _ = rm.deploy_canary(
        instance_name="tg_bot_alpha",
        target_sha=sha3,
        base_dir=base_dir,
        pm2_config=ecosystem_path,
        settle_seconds=2.0,
        registry=registry,
    )
    assert ok_v3 is True

    snapshot_4 = rm.load_pm2_snapshot()
    assert rm.inspect_runtime_process(snapshot_4["tg_bot_alpha"], base_dir)["resolved_sha"] == sha3

    # 7. ROLLBACK tg_bot_alpha (Default rollback: must return to sha2, NOT sha1!)
    ok_rb, msg_rb, _ = rm.rollback_instance(
        instance_name="tg_bot_alpha",
        target_sha=None,  # Default to instance previous successful release
        base_dir=base_dir,
        pm2_config=ecosystem_path,
        settle_seconds=2.0,
        registry=registry,
    )
    assert ok_rb is True, f"Rollback failed: {msg_rb}"

    snapshot_5 = rm.load_pm2_snapshot()
    alpha_rb_info = rm.inspect_runtime_process(snapshot_5["tg_bot_alpha"], base_dir)
    assert alpha_rb_info["resolved_sha"] == sha2, "Alpha must have rolled back to its previous release sha2!"

    # Beta still on sha1!
    assert snapshot_5["tg_bot_beta"]["pid"] == pid_beta_1
    assert rm.inspect_runtime_process(snapshot_5["tg_bot_beta"], base_dir)["resolved_sha"] == sha1

    # 8. FAILURE INJECTION: Deploy Crashing Release 4 -> Automatic Rollback
    rm.stage_release(base_dir=base_dir, target_sha=sha4_crash, source_dir=repo_dir)
    ok_crash, msg_crash, _ = rm.deploy_canary(
        instance_name="tg_bot_alpha",
        target_sha=sha4_crash,
        base_dir=base_dir,
        pm2_config=ecosystem_path,
        settle_seconds=1.5,
        auto_rollback=True,
        registry=registry,
    )
    assert ok_crash is False, "Crashing release deploy must fail!"
    assert "auto-rolled back" in msg_crash

    # Confirm tg_bot_alpha is still online and restored to sha2
    snapshot_6 = rm.load_pm2_snapshot()
    alpha_restored_info = rm.inspect_runtime_process(snapshot_6["tg_bot_alpha"], base_dir)
    assert alpha_restored_info["status"] == "online"
    assert alpha_restored_info["resolved_sha"] == sha2

    # 9. SAFE PRUNE: prune keeping 1 release
    # Active releases (sha1 on beta, sha2 on alpha/gamma) and rollback releases MUST be protected!
    pruned, retained = rm.prune_releases(base_dir=base_dir, keep=1)
    assert sha1 in retained, "Active release sha1 must NOT be pruned!"
    assert sha2 in retained, "Active release sha2 must NOT be pruned!"

    # 10. Status reporting verification
    fleet_status = rm.get_fleet_status(base_dir=base_dir, registry_path=h["registry_path"])
    assert len(fleet_status["managed"]) == 3
    assert len(fleet_status["unmanaged"]) >= 1
    assert any(u["name"] == "vitaliy_bot" for u in fleet_status["unmanaged"])


def test_real_pm2_legacy_coexistence_and_cutover(real_pm2_harness: dict[str, Any], monkeypatch):
    """Test coexistence of unmigrated legacy instance and isolated instances with real PM2."""
    h = real_pm2_harness
    pm2_bin = h["pm2_bin"]
    pm2_env = h["pm2_env"]
    base_dir = h["base_dir"]
    ecosystem_path = h["ecosystem_path"]
    repo_dir = h["repo_dir"]
    registry = h["registry"]
    sha1 = h["sha1"]
    sha2 = h["sha2"]

    monkeypatch.setenv("PM2_HOME", str(h["pm2_home"]))
    monkeypatch.setenv("PM2_CMD", pm2_bin)
    monkeypatch.setenv("PROD_BASE_DIR", str(base_dir))
    monkeypatch.setenv("PROD_RUNTIME_ENV", str(h["runtime_env_file"]))

    # 1. Setup legacy tree
    legacy_dir = base_dir / "newbots"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    (legacy_dir / "main.py").write_text("import time\nwhile True: time.sleep(0.5)\n", encoding="utf-8")
    (legacy_dir / "REVISION").write_text(f"legacy_{sha1[:8]}\n", encoding="utf-8")

    # Bot Alpha runs from legacy directory (no deployments/tg_bot_alpha symlink exists)
    # Bot Beta is isolated on sha1
    rel1_dir = rm.stage_release(base_dir=base_dir, target_sha=sha1, source_dir=repo_dir)
    rm.atomic_symlink_switch(base_dir, "tg_bot_beta", rel1_dir)

    # Start apps
    subprocess.run([pm2_bin, "start", str(ecosystem_path)], env=pm2_env, capture_output=True, check=True)
    time.sleep(2.0)

    snap_1 = rm.load_pm2_snapshot()
    info_alpha_1 = rm.inspect_runtime_process(snap_1["tg_bot_alpha"], base_dir)
    info_beta_1 = rm.inspect_runtime_process(snap_1["tg_bot_beta"], base_dir)

    assert info_alpha_1["mode"] == "legacy", "tg_bot_alpha should be running in legacy mode"
    assert info_beta_1["mode"] == "isolated", "tg_bot_beta should be running in isolated mode"
    assert info_beta_1["resolved_sha"] == sha1

    pid_alpha_initial = info_alpha_1["pid"]
    pid_beta_initial = info_beta_1["pid"]

    # 2. Deploy Release 2 to Bot Beta only
    rm.stage_release(base_dir=base_dir, target_sha=sha2, source_dir=repo_dir)
    ok_beta, msg_beta, _ = rm.deploy_canary(
        instance_name="tg_bot_beta",
        target_sha=sha2,
        base_dir=base_dir,
        pm2_config=ecosystem_path,
        settle_seconds=1.5,
        registry=registry,
    )
    assert ok_beta is True

    snap_2 = rm.load_pm2_snapshot()
    info_alpha_2 = rm.inspect_runtime_process(snap_2["tg_bot_alpha"], base_dir)
    info_beta_2 = rm.inspect_runtime_process(snap_2["tg_bot_beta"], base_dir)

    # Legacy bot Alpha was completely untouched!
    assert info_alpha_2["pid"] == pid_alpha_initial
    assert info_alpha_2["mode"] == "legacy"

    # Beta updated
    assert info_beta_2["pid"] != pid_beta_initial
    assert info_beta_2["resolved_sha"] == sha2

    # 3. Now migrate Bot Alpha from legacy to isolated layout
    ok_mig, msg_mig, _ = rm.deploy_canary(
        instance_name="tg_bot_alpha",
        target_sha=sha2,
        base_dir=base_dir,
        pm2_config=ecosystem_path,
        settle_seconds=1.5,
        registry=registry,
    )
    assert ok_mig is True, f"Migration failed: {msg_mig}"

    snap_3 = rm.load_pm2_snapshot()
    info_alpha_3 = rm.inspect_runtime_process(snap_3["tg_bot_alpha"], base_dir)
    assert info_alpha_3["mode"] == "isolated"
    assert info_alpha_3["resolved_sha"] == sha2


def test_real_pm2_shared_database_instances_are_not_disturbed(real_pm2_harness: dict[str, Any], monkeypatch):
    """Test that two bots sharing a database are isolated; deploying Bot 1 never restarts Bot 2."""
    h = real_pm2_harness
    pm2_bin = h["pm2_bin"]
    pm2_env = h["pm2_env"]
    base_dir = h["base_dir"]
    ecosystem_path = h["ecosystem_path"]
    repo_dir = h["repo_dir"]
    registry = h["registry"]
    sha1 = h["sha1"]
    sha2 = h["sha2"]

    monkeypatch.setenv("PM2_HOME", str(h["pm2_home"]))
    monkeypatch.setenv("PM2_CMD", pm2_bin)
    monkeypatch.setenv("PROD_BASE_DIR", str(base_dir))
    monkeypatch.setenv("PROD_RUNTIME_ENV", str(h["runtime_env_file"]))

    # Deploy v1 to both
    rel1_dir = rm.stage_release(base_dir=base_dir, target_sha=sha1, source_dir=repo_dir)
    rm.atomic_symlink_switch(base_dir, "tg_bot_alpha", rel1_dir)
    rm.atomic_symlink_switch(base_dir, "tg_bot_beta", rel1_dir)

    subprocess.run([pm2_bin, "start", str(ecosystem_path)], env=pm2_env, capture_output=True, check=True)
    time.sleep(2.0)

    snap_1 = rm.load_pm2_snapshot()
    pid_beta = snap_1["tg_bot_beta"]["pid"]

    # Deploy v2 to alpha only
    rm.stage_release(base_dir=base_dir, target_sha=sha2, source_dir=repo_dir)
    ok_alpha, _, _ = rm.deploy_canary(
        instance_name="tg_bot_alpha",
        target_sha=sha2,
        base_dir=base_dir,
        pm2_config=ecosystem_path,
        settle_seconds=1.5,
        registry=registry,
    )
    assert ok_alpha is True

    snap_2 = rm.load_pm2_snapshot()
    assert snap_2["tg_bot_beta"]["pid"] == pid_beta, "Bot Beta sharing database must not be restarted!"
    info_beta = rm.inspect_runtime_process(snap_2["tg_bot_beta"], base_dir)
    assert info_beta["resolved_sha"] == sha1

