import json
from pathlib import Path
import stat

import pytest

import scripts.cutover_bot_instance as cutover
import scripts.migrate_bot_secrets as migration
import scripts.verify_bot_instances as verifier


def _entry():
    return {
        "platform": "telegram",
        "username": "demo_bot",
        "platform_id": 123456789,
        "pm2_name": "tg_demo_bot_new",
        "legacy_pm2_name": "demo_old",
        "database": "demo_db",
        "database_env": "TELEGRAM_DEMO_BOT_DATABASE_URL",
        "runtime_database_env": "DATABASE_URL",
        "token_env": "TELEGRAM_DEMO_BOT_TOKEN",
        "runtime_token_env": "BOT_TOKEN",
        "mode": "new",
        "port": 8081,
        "profile": "/demo",
        "deploy_managed": True,
        "active": True,
    }


def _registry(entry=None):
    return {"schema_version": 1, "instances": [entry or _entry()]}


def _process(name, token, database_url, status="online"):
    return {
        "name": name,
        "pm2_env": {
            "status": status,
            "pid": 0,
            "pm_exec_path": "/root/telegram_bots/newbots/main.py",
            "env": {"BOT_TOKEN": token, "DATABASE_URL": database_url},
        },
    }


def _cutover_state():
    entry = _entry()
    registry = _registry(entry)
    token = "123456:abcdefghijklmnopqrstuvwxyz"
    database_url = "postgresql://user:password@host/demo_db"
    state = {
        entry["legacy_pm2_name"]: _process(
            entry["legacy_pm2_name"], token, database_url
        )
    }
    return entry, registry, state, {entry["token_env"]: token, entry["database_env"]: database_url}


def test_cutover_verifies_before_deleting_legacy_and_uses_one_instance(tmp_path, monkeypatch):
    entry, registry, state, environment = _cutover_state()
    events = []
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}", encoding="utf-8")

    def snapshot_loader():
        return state

    def pm2_runner(arguments):
        events.append(tuple(arguments))
        command = arguments[0]
        name = arguments[1]
        if command == "stop":
            state[name]["pm2_env"]["status"] = "stopped"
        elif command == "startOrReload":
            state[entry["pm2_name"]] = _process(
                entry["pm2_name"],
                environment[entry["token_env"]],
                environment[entry["database_env"]],
            )
        elif command == "delete":
            state.pop(name, None)
        else:
            raise AssertionError(arguments)

    monkeypatch.setattr(cutover.runtime_verifier, "create_log_baseline", lambda *_args: str(baseline))
    monkeypatch.setattr(cutover, "verify_legacy_mapping", lambda *_args: None)
    monkeypatch.setattr(cutover, "verify_canonical_mapping", lambda *_args, **_kwargs: None)

    def runtime_checker(*_args):
        events.append(("verify",))
        assert entry["legacy_pm2_name"] in state
        assert state[entry["legacy_pm2_name"]]["pm2_env"]["status"] == "stopped"

    result = cutover.cutover_instance(
        entry,
        registry,
        environment,
        config_path="ecosystem.config.js",
        root=tmp_path,
        revision="abc123",
        allow_rename=True,
        snapshot_loader=snapshot_loader,
        pm2_runner=pm2_runner,
        runtime_checker=runtime_checker,
    )

    assert result == "migrated"
    assert events.index(("stop", entry["legacy_pm2_name"])) < events.index(("verify",))
    assert events.index(("verify",)) < events.index(("delete", entry["legacy_pm2_name"]))
    assert entry["pm2_name"] in state
    assert entry["legacy_pm2_name"] not in state


def test_cutover_failure_restarts_legacy_without_deleting_it_first(tmp_path, monkeypatch):
    entry, registry, state, environment = _cutover_state()
    events = []
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}", encoding="utf-8")

    def pm2_runner(arguments):
        events.append(tuple(arguments))
        command = arguments[0]
        name = arguments[1]
        if command == "stop":
            state[name]["pm2_env"]["status"] = "stopped"
        elif command == "startOrReload":
            state[entry["pm2_name"]] = _process(
                entry["pm2_name"],
                environment[entry["token_env"]],
                environment[entry["database_env"]],
            )
        elif command == "delete":
            state.pop(name, None)
        elif command == "restart":
            state[name]["pm2_env"]["status"] = "online"
        else:
            raise AssertionError(arguments)

    monkeypatch.setattr(cutover.runtime_verifier, "create_log_baseline", lambda *_args: str(baseline))
    monkeypatch.setattr(cutover, "verify_legacy_mapping", lambda *_args: None)
    monkeypatch.setattr(cutover, "verify_canonical_mapping", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cutover, "wait_for_online", lambda name, loader: loader())

    def runtime_checker(*_args):
        events.append(("verify",))
        raise RuntimeError("verification failed")

    with pytest.raises(cutover.CutoverError, match="cutover_failed"):
        cutover.cutover_instance(
            entry,
            registry,
            environment,
            config_path="ecosystem.config.js",
            root=tmp_path,
            revision="abc123",
            allow_rename=True,
            snapshot_loader=lambda: state,
            pm2_runner=pm2_runner,
            runtime_checker=runtime_checker,
        )

    assert ("verify",) in events
    assert ("delete", entry["legacy_pm2_name"]) not in events
    assert events[-1] == ("restart", entry["legacy_pm2_name"])
    assert entry["legacy_pm2_name"] in state
    assert state[entry["legacy_pm2_name"]]["pm2_env"]["status"] == "online"
    assert entry["pm2_name"] not in state


def test_secret_migration_plan_is_read_only_and_redacts_values(tmp_path, monkeypatch, capsys):
    entry = _entry()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(_registry(entry)), encoding="utf-8")
    runtime_path = tmp_path / "runtime.env"
    original = "# keep this\nUNRELATED=ok\n"
    runtime_path.write_text(original, encoding="utf-8")
    token = "123456:abcdefghijklmnopqrstuvwxyz"
    database_url = "postgresql://user:password@host/demo_db"
    snapshot = {
        entry["legacy_pm2_name"]: _process(
            entry["legacy_pm2_name"], token, database_url
        )
    }
    monkeypatch.setattr(migration, "load_pm2_snapshot", lambda: snapshot)
    before = runtime_path.stat()

    result = migration.main(
        [
            "--plan",
            "--registry",
            str(registry_path),
            "--runtime-env",
            str(runtime_path),
            "--pm2-name",
            entry["pm2_name"],
        ]
    )
    output = capsys.readouterr().out

    assert result == 0
    assert runtime_path.read_text(encoding="utf-8") == original
    assert runtime_path.stat().st_mtime_ns == before.st_mtime_ns
    assert not list(tmp_path.glob("runtime.env.bak.*"))
    assert verifier.fingerprint_token(token) in output
    assert token not in output
    assert database_url not in output


def test_secret_migration_apply_is_atomic_preserves_unrelated_entries_and_verifies(tmp_path, monkeypatch, capsys):
    entry = _entry()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(_registry(entry)), encoding="utf-8")
    runtime_path = tmp_path / "runtime.env"
    runtime_path.write_text("# keep this\nUNRELATED=ok\n", encoding="utf-8")
    token = "123456:abcdefghijklmnopqrstuvwxyz"
    database_url = "postgresql://user:password@host/demo_db"
    monkeypatch.setattr(
        migration,
        "load_pm2_snapshot",
        lambda: {entry["legacy_pm2_name"]: _process(entry["legacy_pm2_name"], token, database_url)},
    )

    result = migration.main(
        [
            "--apply",
            "--registry",
            str(registry_path),
            "--runtime-env",
            str(runtime_path),
            "--pm2-name",
            entry["pm2_name"],
        ]
    )
    output = capsys.readouterr().out
    values = verifier.parse_env_file(runtime_path)
    backups = list(tmp_path.glob("runtime.env.bak.*"))

    assert result == 0
    assert values[entry["token_env"]] == token
    assert values[entry["database_env"]] == database_url
    assert values["UNRELATED"] == "ok"
    assert len(backups) == 1
    assert stat.S_IMODE(backups[0].stat().st_mode) == 0o600
    assert token not in output
    assert database_url not in output


def test_secret_migration_refuses_different_existing_canonical_value(tmp_path, monkeypatch, capsys):
    entry = _entry()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(_registry(entry)), encoding="utf-8")
    runtime_path = tmp_path / "runtime.env"
    existing = "TELEGRAM_DEMO_BOT_TOKEN=654321:abcdefghijklmnopqrstuvwxyz\n"
    runtime_path.write_text(existing, encoding="utf-8")
    token = "123456:abcdefghijklmnopqrstuvwxyz"
    database_url = "postgresql://user:password@host/demo_db"
    monkeypatch.setattr(
        migration,
        "load_pm2_snapshot",
        lambda: {entry["legacy_pm2_name"]: _process(entry["legacy_pm2_name"], token, database_url)},
    )

    result = migration.main(
        [
            "--apply",
            "--registry",
            str(registry_path),
            "--runtime-env",
            str(runtime_path),
            "--pm2-name",
            entry["pm2_name"],
        ]
    )
    output = capsys.readouterr().out

    assert result == 1
    assert runtime_path.read_text(encoding="utf-8") == existing
    assert not list(tmp_path.glob("runtime.env.bak.*"))
    assert token not in output
    assert database_url not in output
