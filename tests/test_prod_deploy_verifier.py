import asyncio
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

import scripts.verify_prod_runtime as verifier
from scripts.verify_prod_runtime import (
    DB_CHECK_CONCURRENCY,
    DB_DISPOSE_TIMEOUT_SECONDS,
    DB_CHECK_TIMEOUT_SECONDS,
    LOG_CLEAN,
    LOG_ERROR,
    LOG_INDETERMINATE,
    create_log_baseline,
    load_log_baseline,
    parse_names,
    process_database_url,
    recent_startup_error,
    validate_pm2_snapshot,
    validate_pm2_stability,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _process(
    *,
    name="psy5d_new",
    status="online",
    pid=101,
    restart_time=4,
    database_url="postgresql+asyncpg://db",
    log_path=None,
    script_path="/root/telegram_bots/newbots/main.py",
):
    process = {
        "name": name,
        "pm2_env": {
            "status": status,
            "pid": pid,
            "restart_time": restart_time,
            "pm_exec_path": script_path,
            "env": {"DATABASE_URL": database_url},
        },
    }
    if log_path is not None:
        process["pm2_env"]["pm_err_log_path"] = str(log_path)
    return process


def _baseline_for(tmp_path, process):
    path = create_log_baseline(
        {process["name"]: process},
        [process["name"]],
    )
    return Path(path), load_log_baseline(path)[process["name"]]


def _remove_baseline(path):
    path.unlink(missing_ok=True)


def test_parse_names_trims_and_ignores_empty_values():
    assert parse_names(" a, ,b,, a ") == ["a", "b", "a"]


def test_pm2_snapshot_requires_online_processes_with_pids():
    assert validate_pm2_snapshot({"psy5d_new": _process()}, ["psy5d_new"]) == []
    assert validate_pm2_snapshot(
        {"psy5d_new": _process(status="stopped", pid=0)},
        ["psy5d_new"],
    ) == ["not_online:psy5d_new", "missing_pid:psy5d_new"]


def test_pm2_stability_detects_restart_or_pid_change():
    first = {"psy5d_new": _process()}
    second = {"psy5d_new": _process(pid=202, restart_time=5)}
    assert validate_pm2_stability(first, second, ["psy5d_new"]) == [
        "pid_changed:psy5d_new",
        "restart_count_changed:psy5d_new",
    ]


def test_pm2_stability_rejects_missing_restart_counter():
    first = {"psy5d_new": _process()}
    second = {"psy5d_new": _process(restart_time=None)}
    assert validate_pm2_stability(first, second, ["psy5d_new"]) == [
        "restart_count_unavailable:psy5d_new",
    ]


@pytest.mark.parametrize("restart_time", [True, -1, "not-a-number"])
def test_pm2_stability_rejects_invalid_restart_counter(restart_time):
    first = {"psy5d_new": _process()}
    second = {"psy5d_new": _process(restart_time=restart_time)}
    assert validate_pm2_stability(first, second, ["psy5d_new"]) == [
        "restart_count_unavailable:psy5d_new",
    ]


def test_database_url_is_read_from_process_environment_without_reformatting():
    process = _process(database_url="postgresql+asyncpg://user:password@host/db")
    assert process_database_url(process) == "postgresql+asyncpg://user:password@host/db"


def test_log_baselines_are_unique_restricted_and_include_identity(tmp_path):
    log_path = tmp_path / "bot-error.log"
    log_path.write_text("historical log\n", encoding="utf-8")
    process = _process(log_path=log_path)
    paths = []
    try:
        first_path = Path(create_log_baseline({"psy5d_new": process}, ["psy5d_new"]))
        second_path = Path(create_log_baseline({"psy5d_new": process}, ["psy5d_new"]))
        paths.extend((first_path, second_path))

        assert first_path != second_path
        assert stat.S_IMODE(first_path.stat().st_mode) == 0o600

        entry = load_log_baseline(str(first_path))["psy5d_new"]
        log_stat = log_path.stat()
        assert entry["device"] == log_stat.st_dev
        assert entry["inode"] == log_stat.st_ino
        assert entry["offset"] == log_stat.st_size
    finally:
        for path in paths:
            _remove_baseline(path)


def test_log_append_ignores_history_and_detects_new_error(tmp_path):
    log_path = tmp_path / "bot-error.log"
    historical = "old Traceback (most recent call last)\n"
    log_path.write_text(historical, encoding="utf-8")
    process = _process(log_path=log_path)
    baseline_path, baseline = _baseline_for(tmp_path, process)
    try:
        assert recent_startup_error(process, baseline).status == LOG_CLEAN

        with log_path.open("ab") as handle:
            handle.write(b"new ModuleNotFoundError: missing\n")
        result = recent_startup_error(process, baseline)
        assert result.status == LOG_ERROR
        assert result.reason == "startup_error"
    finally:
        _remove_baseline(baseline_path)


def test_log_identity_change_is_indeterminate_not_historical_error(tmp_path):
    log_path = tmp_path / "bot-error.log"
    log_path.write_text("old Traceback (most recent call last)\n", encoding="utf-8")
    process = _process(log_path=log_path)
    baseline_path, baseline = _baseline_for(tmp_path, process)
    try:
        rotated_path = tmp_path / "bot-error.log.1"
        log_path.rename(rotated_path)
        log_path.write_text("old Traceback (most recent call last)\n", encoding="utf-8")

        result = recent_startup_error(process, baseline)
        assert result.status == LOG_INDETERMINATE
        assert result.reason == "identity_changed"
    finally:
        _remove_baseline(baseline_path)


def test_log_rotation_after_open_is_indeterminate(monkeypatch, tmp_path):
    log_path = tmp_path / "bot-error.log"
    log_path.write_text("historical\n", encoding="utf-8")
    process = _process(log_path=log_path)
    baseline_path, baseline = _baseline_for(tmp_path, process)
    original_stat = verifier.os.stat
    stat_calls = 0

    def rotate_on_final_path_stat(path, *args, **kwargs):
        nonlocal stat_calls
        stat_calls += 1
        if stat_calls == 2:
            log_path.rename(tmp_path / "bot-error.log.1")
            log_path.write_text(
                "Traceback (most recent call last)\n",
                encoding="utf-8",
            )
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(verifier.os, "stat", rotate_on_final_path_stat)
    try:
        result = recent_startup_error(process, baseline)
    finally:
        _remove_baseline(baseline_path)

    assert stat_calls == 2
    assert result.status == LOG_INDETERMINATE
    assert result.reason == "identity_changed_during_read"


def test_log_truncation_is_indeterminate_not_historical_error(tmp_path):
    log_path = tmp_path / "bot-error.log"
    log_path.write_text(
        "old Traceback (most recent call last)\n" + "x" * 100,
        encoding="utf-8",
    )
    process = _process(log_path=log_path)
    baseline_path, baseline = _baseline_for(tmp_path, process)
    try:
        log_path.write_text("old Traceback (most recent call last)\n", encoding="utf-8")

        result = recent_startup_error(process, baseline)
        assert result.status == LOG_INDETERMINATE
        assert result.reason == "truncated"
    finally:
        _remove_baseline(baseline_path)


def test_baseline_write_failure_removes_partial_file(monkeypatch, tmp_path):
    log_path = tmp_path / "bot-error.log"
    log_path.write_text("historical\n", encoding="utf-8")
    created_path = None
    real_mkstemp = verifier.tempfile.mkstemp

    def tracking_mkstemp(*args, **kwargs):
        nonlocal created_path
        descriptor, path = real_mkstemp(*args, **kwargs)
        created_path = Path(path)
        return descriptor, path

    def fail_dump(*_args, **_kwargs):
        raise OSError("simulated baseline write failure")

    monkeypatch.setattr(verifier.tempfile, "mkstemp", tracking_mkstemp)
    monkeypatch.setattr(verifier.json, "dump", fail_dump)

    with pytest.raises(OSError, match="simulated baseline write failure"):
        create_log_baseline(
            {"psy5d_new": _process(log_path=log_path)},
            ["psy5d_new"],
        )

    assert created_path is not None
    assert not created_path.exists()


def test_baseline_cleanup_trap_preserves_failure_status(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text("baseline", encoding="utf-8")
    script = (
        "set -euo pipefail; "
        f"baseline_path={shlex.quote(str(baseline_path))}; "
        "trap 'status=$?; if [[ -n \"$baseline_path\" ]]; then rm -f -- \"$baseline_path\" 2>/dev/null || true; fi; "
        "trap - EXIT; exit \"$status\"' EXIT; "
        "exit 23"
    )

    result = subprocess.run(["bash", "-c", script], check=False)

    assert result.returncode == 23
    assert not baseline_path.exists()


@pytest.mark.asyncio
async def test_database_timeout_is_bounded_and_reported(monkeypatch):
    async def hanging_check(_database_url):
        await asyncio.sleep(1)

    monkeypatch.setattr(verifier, "DB_CHECK_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(verifier, "verify_general_config", hanging_check)
    started = time.monotonic()

    checked, errors = await verifier.verify_migrations(
        {"psy5d_new": _process()},
        ["psy5d_new"],
    )

    assert time.monotonic() - started < 0.5
    assert checked == 1
    assert errors == ["migration_timeout:psy5d_new"]


@pytest.mark.asyncio
async def test_database_timeout_disposes_engine(monkeypatch):
    class SlowConnection:
        async def __aenter__(self):
            await asyncio.sleep(1)
            return self

        async def __aexit__(self, *_args):
            return False

    class FakeEngine:
        def __init__(self):
            self.disposed = False

        def connect(self):
            return SlowConnection()

        async def dispose(self):
            self.disposed = True

    engine = FakeEngine()
    monkeypatch.setattr(verifier, "DB_CHECK_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(verifier, "create_async_engine", lambda *_args, **_kwargs: engine)

    checked, errors = await verifier.verify_migrations(
        {"psy5d_new": _process()},
        ["psy5d_new"],
    )

    assert checked == 1
    assert errors == ["migration_timeout:psy5d_new"]
    assert engine.disposed is True


@pytest.mark.asyncio
async def test_database_timeout_does_not_wait_forever_for_dispose(monkeypatch):
    class SlowConnection:
        async def __aenter__(self):
            await asyncio.sleep(1)
            return self

        async def __aexit__(self, *_args):
            return False

    class HangingDisposeEngine:
        def __init__(self):
            self.dispose_started = False

        def connect(self):
            return SlowConnection()

        async def dispose(self):
            self.dispose_started = True
            await asyncio.sleep(1)

    engine = HangingDisposeEngine()
    monkeypatch.setattr(verifier, "DB_CHECK_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(verifier, "DB_DISPOSE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(verifier, "create_async_engine", lambda *_args, **_kwargs: engine)

    started = time.monotonic()
    checked, errors = await verifier.verify_migrations(
        {"psy5d_new": _process()},
        ["psy5d_new"],
    )

    assert time.monotonic() - started < 0.5
    assert checked == 1
    assert errors == ["migration_timeout:psy5d_new"]
    assert engine.dispose_started is True


@pytest.mark.asyncio
async def test_database_checks_are_concurrent_but_bounded(monkeypatch):
    active = 0
    maximum_active = 0

    async def successful_check(_database_url):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return True

    monkeypatch.setattr(verifier, "DB_CHECK_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(verifier, "DB_CHECK_CONCURRENCY", 2)
    monkeypatch.setattr(verifier, "verify_general_config", successful_check)
    snapshot = {
        f"psy5d_{index}": _process(
            name=f"psy5d_{index}",
            database_url=f"postgresql+asyncpg://db{index}",
        )
        for index in range(5)
    }

    checked, errors = await verifier.verify_migrations(snapshot, list(snapshot))

    assert checked == 5
    assert errors == []
    assert maximum_active == 2


@pytest.mark.asyncio
async def test_general_config_migration_check_is_read_only_and_disposes_engine(monkeypatch):
    class Result:
        def first(self):
            return (True, "configured")

    class Connection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def run_sync(self, callback):
            return callback(object())

        async def execute(self, _statement):
            return Result()

    class Engine:
        def __init__(self):
            self.disposed = False

        def connect(self):
            return Connection()

        async def dispose(self):
            self.disposed = True

    engine = Engine()
    monkeypatch.setattr(
        verifier,
        "create_async_engine",
        lambda *_args, **_kwargs: engine,
    )
    monkeypatch.setattr(
        verifier,
        "inspect",
        lambda _connection: SimpleNamespace(
            get_columns=lambda _table: [
                {"name": "ai_processing_message_enabled"},
                {"name": "ai_processing_message_text"},
            ]
        ),
    )

    assert await verifier.verify_general_config("postgresql+asyncpg://db") is True
    assert engine.disposed is True


def test_invalid_pm2_names_are_rejected_before_git_or_ssh():
    environment = os.environ.copy()
    environment["PROD_PM2_NAMES"] = "psy5d_new;touch /tmp/should-not-exist"
    environment["PROD_HOST"] = "example.invalid"
    result = subprocess.run(
        ["bash", "deploy_prod.sh"],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "invalid process names" in result.stderr
    assert "ssh" not in result.stderr.lower()


def test_deploy_chains_verifier_failure_and_cleanup_trap():
    deploy_script = (REPO_ROOT / "deploy_prod.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in deploy_script
    assert "--create-log-baseline" in deploy_script
    assert "baseline_path= &&" in deploy_script
    assert "trap 'status=\\$?; if [[ -n \"\\$baseline_path\" ]]; then rm -f -- \"\\$baseline_path\"" in deploy_script
    assert "pm2 status &&" in deploy_script
    assert '--log-baseline "\\$baseline_path"' in deploy_script
    verifier_marker = "'${REMOTE_PY}' 'scripts/verify_prod_runtime.py' \\\n         --revision '${REVISION}'"
    assert verifier_marker in deploy_script
    assert verifier_marker + " \\\n         --pm2-names" in deploy_script


def test_verifier_failure_is_not_masked_by_remote_shell_chain():
    deploy_script = (REPO_ROOT / "deploy_prod.sh").read_text(encoding="utf-8")
    verifier_start = deploy_script.index(
        "     '${REMOTE_PY}' 'scripts/verify_prod_runtime.py' \\\n         --revision"
    )
    verifier_command = deploy_script[verifier_start:]

    assert "--log-baseline \"\\$baseline_path\"\"" in verifier_command
    assert "|| true" not in verifier_command


@pytest.mark.parametrize("revision", ["wrong", "expected"])
def test_main_revision_check_and_pm2_validation(
    monkeypatch,
    tmp_path,
    capsys,
    revision,
):
    log_path = tmp_path / "bot-error.log"
    log_path.write_text("historical\n", encoding="utf-8")
    process = _process(log_path=log_path)
    baseline_path, _ = _baseline_for(tmp_path, process)
    (tmp_path / "REVISION").write_text("expected\n", encoding="utf-8")
    snapshot = {"psy5d_new": process}

    monkeypatch.setattr(verifier, "load_pm2_snapshot", lambda: snapshot)
    monkeypatch.setattr(verifier.time, "sleep", lambda _seconds: None)

    async def migrations_ok(_snapshot, _expected_names):
        return 1, []

    monkeypatch.setattr(verifier, "verify_migrations", migrations_ok)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_prod_runtime.py",
            "--revision",
            revision,
            "--pm2-names",
            "psy5d_new",
            "--root",
            str(tmp_path),
            "--settle-seconds",
            "0",
            "--log-baseline",
            str(baseline_path),
        ],
    )

    try:
        result = verifier.main()
        output = capsys.readouterr().out
        if revision == "expected":
            assert result == 0
            assert "verification=ok" in output
        else:
            assert result == 1
            assert "revision=failed" in output
    finally:
        _remove_baseline(baseline_path)


def test_baseline_creation_failure_does_not_accept_missing_process():
    with pytest.raises(RuntimeError, match="missing PM2 process"):
        create_log_baseline({}, ["psy5d_new"])


def test_timeout_constants_are_explicit_and_bounded():
    assert DB_CHECK_TIMEOUT_SECONDS > 0
    assert DB_DISPOSE_TIMEOUT_SECONDS > 0
    assert DB_CHECK_CONCURRENCY > 0
