from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine


APP = Path("/root/telegram_bots/newbots")
WORK = Path("/root/translation_work")


def process_environment(process):
    pid = int(process["pid"])
    result = {}
    for entry in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        key, _, value = entry.partition(b"=")
        if key in {b"DATABASE_URL", b"BOT_TOKEN"}:
            result[key.decode()] = value.decode()
    if "DATABASE_URL" not in result:
        raise RuntimeError("Database environment unavailable")
    return result


def digest(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, default=str, separators=(",", ":")).encode()).hexdigest()


def normalize_language_configuration(config):
    default_language, enabled_raw, selector_enabled, revision = config
    enabled = json.loads(enabled_raw) if isinstance(enabled_raw, str) else enabled_raw
    if default_language != "ru" or set(enabled) != {"ru", "en", "pt"}:
        raise RuntimeError("Production language configuration changed")
    if not isinstance(selector_enabled, (bool, int)) or selector_enabled not in (False, True, 0, 1):
        raise RuntimeError("Invalid language selector configuration")
    return {
        "default": default_language,
        "enabled": sorted(enabled),
        "selector": bool(selector_enabled),
        "revision": revision,
    }


def language_configuration_changes(current, baseline):
    current_config = current.get("language_config") or {
        "default": current["default"],
        "enabled": sorted(current["enabled"]),
        "selector": current["selector"],
    }
    baseline_config = baseline.get("language_config") or {
        "default": baseline["default"],
        "enabled": sorted(baseline["enabled"]),
        "selector": baseline["selector"],
    }
    return [
        field
        for field in ("default", "enabled", "selector")
        if current_config[field] != baseline_config[field]
    ]


async def inspect_database(database_url, expected_database):
    url = make_url(database_url)
    if url.database != expected_database:
        raise RuntimeError("Database mapping mismatch")
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SET TRANSACTION READ ONLY"))
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from payment_index_validation import read_postgres_unresolved_index, validate_postgres_unresolved_index
            validate_postgres_unresolved_index(await connection.run_sync(read_postgres_unresolved_index))
            rows = (await connection.execute(text("SELECT locale, translation_key, text, source_hash, created_at, updated_at FROM bot_translations ORDER BY locale, translation_key"))).all()
            counts = {locale: sum(row[0] == locale for row in rows) for locale in ("en", "pt")}
            config = (await connection.execute(text("SELECT telegram_default_language, telegram_enabled_languages, telegram_language_selection_enabled, translations_revision FROM bot_general_config WHERE id=1"))).one()
            language_config = normalize_language_configuration(config)
            size = await connection.scalar(text("SELECT pg_database_size(current_database())"))
            preferences = (await connection.execute(text("SELECT telegram_language_code, count(*) FROM users GROUP BY telegram_language_code ORDER BY telegram_language_code NULLS FIRST"))).all()
            return {"database": expected_database, "counts": counts, "translation_digest": digest([tuple(row) for row in rows]), "default": language_config["default"], "enabled": language_config["enabled"], "selector": language_config["selector"], "revision": language_config["revision"], "language_config": language_config, "size_bytes": size, "preference_counts": [tuple(row) for row in preferences]}
    finally:
        await engine.dispose()


async def smoke_database(database_url):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault("DATABASE_URL", database_url)
    os.environ.setdefault("BOT_TOKEN", "123456:test")
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from sqlalchemy import select
    from content_authoring import RESOURCES, read_content_value
    from database import BotTranslation
    from translation_registry import build_translation_registry
    from translation_pack_manager import audit_translation_readiness
    engine = create_async_engine(database_url)
    count = 0
    try:
        async with async_sessionmaker(engine)() as session:
            await session.execute(text("SET TRANSACTION READ ONLY"))
            registry = await build_translation_registry(session)
            readiness = await audit_translation_readiness(session, registry, locales=("en", "pt"))
            if not all(readiness["locales"][locale]["ready"] for locale in ("en", "pt")):
                raise RuntimeError("System locale readiness failed")
            for kind in ("topic", "content", "plan", "test_question", "referral_template", "mailing"):
                spec = RESOURCES[kind]
                resources = (await session.scalars(select(spec.model).limit(3))).all()
                for resource in resources:
                    for field, _, _ in spec.fields:
                        key = f"{kind}.{getattr(resource, spec.identity_field)}.{field}"
                        for locale in ("en", "pt"):
                            expected = await session.scalar(select(BotTranslation.text).where(BotTranslation.locale == locale, BotTranslation.translation_key == key))
                            actual = await read_content_value(session, kind, resource, field, locale)
                            if actual.text != expected:
                                raise RuntimeError("Authoring view does not match stored translation")
                            count += int(bool(expected))
                        await read_content_value(session, kind, resource, field, "ru")
            for table in ("admin_content_preferences", "content_identity_counters", "user_menu_bindings"):
                exists = await session.scalar(text("SELECT to_regclass(:name) IS NOT NULL"), {"name": table})
                if not exists:
                    raise RuntimeError("Missing authoring schema")
            column = await session.scalar(text("SELECT count(*) FROM information_schema.columns WHERE table_name='test_sessions' AND column_name='question_snapshot'"))
            if not column:
                raise RuntimeError("Missing stable test session schema")
        return count
    finally:
        await engine.dispose()


def backup_database(environment, path):
    url = make_url(environment["DATABASE_URL"])
    pg_environment = os.environ.copy()
    pg_environment.update(PGHOST=url.host or "localhost", PGPORT=str(url.port or 5432), PGUSER=url.username or "postgres", PGPASSWORD=url.password or "", PGDATABASE=url.database)
    with path.open("xb") as output:
        os.chmod(path, 0o600)
        command = ["pg_dump", "--format=custom", "--no-owner", "--no-acl"]
        if url.host in {"localhost", "127.0.0.1"}:
            command = ["runuser", "-u", "postgres", "--", *command, "--dbname", url.database]
            pg_environment = {key: value for key, value in os.environ.items() if not key.startswith("PG")}
        result = subprocess.run(command, env=pg_environment, stdout=output, stderr=subprocess.PIPE)
    if result.returncode or not path.stat().st_size:
        raise RuntimeError("Database backup failed; production unchanged")
    check = subprocess.run(["pg_restore", "--list", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if check.returncode:
        raise RuntimeError("Database backup cannot be listed")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("baseline", "backup", "verify", "smoke"))
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    instances = json.loads((APP / "config/bot_instances.json").read_text())["instances"]
    managed = [item for item in instances if item.get("active") and item.get("deploy_managed")]
    processes = json.loads(subprocess.run(["pm2", "jlist"], capture_output=True, check=True, text=True).stdout)
    by_name = {item["name"]: item for item in processes}
    for item in managed:
        if by_name.get(item["pm2_name"], {}).get("pm2_env", {}).get("status") != "online":
            raise RuntimeError("Managed process is not online")
    databases = {}
    for item in managed:
        if item["platform"] == "telegram":
            databases.setdefault(item["database"], process_environment(by_name[item["pm2_name"]]))
    if len(managed) != 18 or len(databases) != 16:
        raise RuntimeError("Unexpected production inventory")
    reports = []
    for name, environment in databases.items():
        report = await inspect_database(environment["DATABASE_URL"], name)
        reports.append(report)
        print(json.dumps({"database": name, "counts": report["counts"], "config": "ok", "digest": report["translation_digest"]}), flush=True)
        if args.action == "smoke":
            samples = await smoke_database(environment["DATABASE_URL"])
            print(json.dumps({"database": name, "authoring_views": "ru/en/pt ok", "existing_translations_checked": samples, "system_readiness": "ok", "migration": "ok"}), flush=True)
    if args.action == "baseline":
        directory = Path(tempfile.mkdtemp(prefix="admin_authoring_", dir=WORK))
        path = directory / "baseline.json"
        with path.open("x") as output:
            os.chmod(path, 0o600)
            json.dump({"revision": (APP / "REVISION").read_text().strip(), "reports": reports}, output)
        print(json.dumps({"baseline": str(path), "database_count": len(reports), "managed": len(managed), "bytes": sum(item["size_bytes"] for item in reports)}))
    else:
        if not args.baseline or args.baseline.name != "baseline.json" or args.baseline.resolve().parent.parent != WORK:
            raise RuntimeError("Expected a baseline created by this verifier")
        baseline = json.loads(args.baseline.read_text())
        expected = {item["database"]: item for item in baseline["reports"]}
        for report in reports:
            baseline_report = expected[report["database"]]
            if report["translation_digest"] != baseline_report["translation_digest"]:
                raise RuntimeError("Translation rows changed since baseline: " + report["database"])
            changes = language_configuration_changes(report, baseline_report)
            if changes:
                raise RuntimeError(
                    "Language configuration changed since baseline: "
                    + report["database"]
                    + " ("
                    + ", ".join(changes)
                    + ")"
                )
        if args.action == "backup":
            required = sum(item["size_bytes"] for item in reports)
            if shutil.disk_usage(WORK).free < required * 1.5:
                raise RuntimeError("Insufficient backup space")
            for name, environment in databases.items():
                backup_database(environment, args.baseline.parent / f"{name}.dump")
                print(json.dumps({"database": name, "backup": "verified"}), flush=True)
        print(json.dumps({"translations_preserved": True, "database_count": 16, "managed": 18}))


if __name__ == "__main__":
    asyncio.run(main())
