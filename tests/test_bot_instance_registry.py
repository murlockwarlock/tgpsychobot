import json
from pathlib import Path

import scripts.verify_bot_instances as verifier


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "config" / "bot_instances.json"


def _entry(
    *,
    username="demo_bot",
    platform_id=123456789,
    pm2_name="tg_demo_bot_new",
    database="demo_db",
    token_env="TELEGRAM_DEMO_BOT_TOKEN",
    database_env="TELEGRAM_DEMO_BOT_DATABASE_URL",
    deploy_managed=True,
    active=True,
):
    return {
        "platform": "telegram",
        "username": username,
        "platform_id": platform_id,
        "pm2_name": pm2_name,
        "database": database,
        "database_env": database_env,
        "runtime_database_env": "DATABASE_URL",
        "token_env": token_env,
        "runtime_token_env": "BOT_TOKEN",
        "mode": "new",
        "port": 8081,
        "profile": "/demo",
        "deploy_managed": deploy_managed,
        "active": active,
    }


def _registry(*entries):
    return {"schema_version": 1, "instances": list(entries)}


def _process(name, token, database_url, *, path="/root/telegram_bots/newbots/main.py"):
    return {
        "name": name,
        "pm2_env": {
            "status": "online",
            "pid": 0,
            "pm_exec_path": path,
            "env": {"BOT_TOKEN": token, "DATABASE_URL": database_url},
        },
    }


def test_production_registry_is_structurally_valid_and_unique():
    registry = verifier.load_registry(REGISTRY_PATH)

    assert verifier.validate_registry(registry) == []
    assert verifier.validate_deploy_managed_count(registry) == []
    assert len(verifier.managed_instances(registry)) == 18
    assert len({(item["platform"], item["username"]) for item in registry["instances"]}) == len(
        registry["instances"]
    )
    assert len({item["pm2_name"] for item in registry["instances"]}) == len(registry["instances"])
    assert len({item["token_env"] for item in registry["instances"]}) == len(registry["instances"])


def test_registry_contains_no_secret_values():
    raw = REGISTRY_PATH.read_text(encoding="utf-8")

    assert ":" not in raw.replace(": ", "")
    assert "postgres" not in raw.lower()
    assert "@" not in raw
    assert "password" not in raw.lower()


def test_deploy_and_ecosystem_use_every_canonical_managed_name():
    registry = verifier.load_registry(REGISTRY_PATH)
    ecosystem = (ROOT / "ecosystem.config.js").read_text(encoding="utf-8")
    deploy = (ROOT / "deploy_prod.sh").read_text(encoding="utf-8")
    configured_names = {
        line.split('"', 2)[1]
        for line in ecosystem.splitlines()
        if 'name: "' in line
    }

    assert configured_names == set(verifier.registry_pm2_names(registry))
    assert "someone01_new" not in deploy
    assert 'name: "someone01_new"' not in ecosystem
    assert "scripts/verify_bot_instances.py" in deploy
    assert "PROD_ALLOW_PM2_RENAME" in deploy
    assert 'pm2 delete "\\$legacy_name"' in deploy
    assert "--baseline-source-names" in deploy


def test_ecosystem_uses_registry_secret_and_database_keys():
    registry = verifier.load_registry(REGISTRY_PATH)
    ecosystem = (ROOT / "ecosystem.config.js").read_text(encoding="utf-8")

    for item in verifier.managed_instances(registry):
        start = ecosystem.index(f'name: "{item["pm2_name"]}"')
        end = ecosystem.find("\n  },", start)
        block = ecosystem[start:end if end >= 0 else None]
        assert f"process.env.{item['token_env']}" in block
        assert f"process.env.{item['database_env']}" in block


def test_documentation_is_a_projection_of_the_registry():
    registry = verifier.load_registry(REGISTRY_PATH)
    expected = {
        (
            item["platform"],
            item["username"],
            item["pm2_name"],
            item["database"],
            item["token_env"],
            item["mode"],
        )
        for item in registry["instances"]
    }
    rows = set()
    for line in (ROOT / "docs/bot_instances.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| ---") or "Public username" in line:
            continue
        values = [value.strip() for value in line.strip("|").split("|")]
        rows.add(tuple(values))

    assert rows == expected


def test_runtime_check_accepts_matching_canonical_secret_mapping():
    entry = _entry()
    registry = _registry(entry)
    token = "123456:abcdefghijklmnopqrstuvwxyz"
    database_url = "postgresql+asyncpg://user:password@host/demo_db"
    rows, errors = verifier.runtime_rows(
        registry,
        {entry["pm2_name"]: _process(entry["pm2_name"], token, database_url)},
        {entry["token_env"]: token, entry["database_env"]: database_url},
    )

    assert errors == []
    assert rows[0]["token_fingerprint"] == verifier.fingerprint_token(token)
    assert rows[0]["identity"] == "not_requested"


def test_runtime_check_detects_duplicate_active_token_mapping():
    first = _entry()
    second = _entry(
        username="other_bot",
        platform_id=123456790,
        pm2_name="tg_other_bot_new",
        database="other_db",
        token_env="TELEGRAM_OTHER_BOT_TOKEN",
        database_env="TELEGRAM_OTHER_BOT_DATABASE_URL",
    )
    registry = _registry(first, second)
    token = "123456:abcdefghijklmnopqrstuvwxyz"
    snapshot = {
        first["pm2_name"]: _process(first["pm2_name"], token, "postgresql://u:p@h/demo_db"),
        second["pm2_name"]: _process(second["pm2_name"], token, "postgresql://u:p@h/other_db"),
    }
    environment = {
        first["token_env"]: token,
        first["database_env"]: "postgresql://u:p@h/demo_db",
        second["token_env"]: token,
        second["database_env"]: "postgresql://u:p@h/other_db",
    }

    _, errors = verifier.runtime_rows(registry, snapshot, environment)

    assert any(error.startswith("duplicate_active_token:telegram:") for error in errors)


def test_runtime_check_detects_missing_runtime_secret():
    entry = _entry()

    errors = verifier.validate_runtime_env(_registry(entry), {})

    assert errors == [
        "missing_runtime_secret:TELEGRAM_DEMO_BOT_TOKEN",
        "missing_runtime_secret:TELEGRAM_DEMO_BOT_DATABASE_URL",
    ]


def test_runtime_check_detects_unexpected_managed_process():
    entry = _entry()
    snapshot = {
        entry["pm2_name"]: _process(
            entry["pm2_name"],
            "123456:abcdefghijklmnopqrstuvwxyz",
            "postgresql://u:p@h/demo_db",
        ),
        "tg_unregistered_bot_new": _process(
            "tg_unregistered_bot_new",
            "654321:abcdefghijklmnopqrstuvwxyz",
            "postgresql://u:p@h/other_db",
        ),
    }
    environment = {
        entry["token_env"]: "123456:abcdefghijklmnopqrstuvwxyz",
        entry["database_env"]: "postgresql://u:p@h/demo_db",
    }

    _, errors = verifier.runtime_rows(_registry(entry), snapshot, environment)

    assert "unexpected_managed_process:tg_unregistered_bot_new" in errors


def test_runtime_check_detects_registry_token_in_unregistered_process():
    entry = _entry()
    token = "123456:abcdefghijklmnopqrstuvwxyz"
    snapshot = {
        entry["pm2_name"]: _process(
            entry["pm2_name"], token, "postgresql://u:p@h/demo_db"
        ),
        "tg_unregistered_bot_new": _process(
            "tg_unregistered_bot_new", token, "postgresql://u:p@h/other_db"
        ),
    }
    environment = {
        entry["token_env"]: token,
        entry["database_env"]: "postgresql://u:p@h/demo_db",
    }

    _, errors = verifier.runtime_rows(_registry(entry), snapshot, environment)

    assert any(error.startswith("duplicate_active_token:telegram:") for error in errors)


def test_identity_mismatch_is_reported_without_printing_token(monkeypatch, capsys):
    entry = _entry()
    token = "123456:abcdefghijklmnopqrstuvwxyz"
    database_url = "postgresql://u:p@h/demo_db"
    monkeypatch.setattr(
        verifier,
        "verify_identity",
        lambda *_args: {"status": "mismatch", "username": "wrong_bot", "platform_id": 1},
    )

    _, errors = verifier.runtime_rows(
        _registry(entry),
        {entry["pm2_name"]: _process(entry["pm2_name"], token, database_url)},
        {entry["token_env"]: token, entry["database_env"]: database_url},
        identity_check=True,
    )

    assert "identity_mismatch:tg_demo_bot_new" in errors
    assert token not in capsys.readouterr().out


def test_telegram_identity_verification_uses_read_only_getme(monkeypatch):
    entry = _entry()
    calls = []

    def fake_request(url, environment, **kwargs):
        calls.append((url, environment, kwargs))
        return {"ok": True, "result": {"username": entry["username"], "id": entry["platform_id"]}}

    monkeypatch.setattr(verifier, "request_json", fake_request)

    result = verifier.verify_identity(entry, "123456:abcdefghijklmnopqrstuvwxyz", {})

    assert result["status"] == "verified"
    assert calls[0][0].endswith("/getMe")


def test_max_identity_verification_uses_read_only_me(monkeypatch):
    entry = _entry()
    entry.update(
        {
            "platform": "max",
            "username": "max_demo_bot",
            "platform_id": 987654321,
            "pm2_name": "max_max_demo_bot_new",
            "token_env": "MAX_MAX_DEMO_BOT_TOKEN",
            "database_env": "MAX_MAX_DEMO_BOT_DATABASE_URL",
        }
    )
    calls = []

    def fake_request(url, environment, **kwargs):
        calls.append((url, environment, kwargs))
        return {"name": entry["username"], "user_id": entry["platform_id"]}

    monkeypatch.setattr(verifier, "request_json", fake_request)

    result = verifier.verify_identity(entry, "max-secret", {})

    assert result["status"] == "verified"
    assert calls[0][0] == "https://platform-api.max.ru/me"
    assert calls[0][2]["headers"] == {"Authorization": "max-secret"}


def test_registry_duplicate_username_with_conflicting_database_is_rejected():
    first = _entry()
    second = _entry(
        pm2_name="tg_demo_bot_legacy",
        database="other_db",
        database_env="TELEGRAM_DEMO_BOT_LEGACY_DATABASE_URL",
        token_env="TELEGRAM_DEMO_BOT_LEGACY_TOKEN",
    )

    errors = verifier.validate_registry(_registry(first, second))

    assert "duplicate_identity_conflicting_database:telegram:demo_bot" in errors
