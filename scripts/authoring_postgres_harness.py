import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import argparse

from sqlalchemy import select, text
from sqlalchemy.engine import make_url

from verify_content_authoring_production import process_environment, digest


def checked(command, stdin=None):
    result = subprocess.run(command, stdin=stdin, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError("Isolated PostgreSQL operation failed: " + command[3])


async def exercise():
    import database
    from content_authoring import create_resource, read_content_value, save_content_value
    from translation_service import refresh_translation_cache, translation_cache
    async with database.engine.connect() as connection:
        before = (await connection.execute(text("SELECT locale, translation_key, text, source_hash, created_at, updated_at FROM bot_translations ORDER BY locale, translation_key"))).all()
        from payment_index_validation import read_postgres_unresolved_index, validate_postgres_unresolved_index
        catalog_before = dict(await connection.run_sync(read_postgres_unresolved_index))
        validate_postgres_unresolved_index(catalog_before)
        counts_before = {table: await connection.scalar(text(f"SELECT count(*) FROM {table}")) for table in ("yookassa_recurring_attempts", "yookassa_payments", "robokassa_payments", "user_subscriptions", "payment_notification_outbox")}
    await database.init_db()
    await database.init_db()
    async with database.engine.connect() as connection:
        after = (await connection.execute(text("SELECT locale, translation_key, text, source_hash, created_at, updated_at FROM bot_translations ORDER BY locale, translation_key"))).all()
        catalog_after = dict(await connection.run_sync(read_postgres_unresolved_index))
        assert catalog_before == catalog_after
        counts_after = {table: await connection.scalar(text(f"SELECT count(*) FROM {table}")) for table in counts_before}
        assert counts_before == counts_after
    assert digest([tuple(row) for row in before]) == digest([tuple(row) for row in after])
    import main
    from bot_commands import build_command_sets
    from aiogram import Dispatcher
    dispatcher = Dispatcher()
    dispatcher.include_router(main.content_authoring_router)
    dispatcher.include_router(main.automation_admin_router)
    dispatcher.include_router(main.router)
    await refresh_translation_cache(database.async_session_maker, force=True)
    user_commands, admin_commands, _ = await build_command_sets()
    assert user_commands and admin_commands and dispatcher.resolve_used_update_types()
    print(json.dumps({"restore_init_startup": "ok", "catalog_unchanged": catalog_after, "payment_table_counts_unchanged": True, "external_workers_started": False}), flush=True)
    async with database.async_session_maker() as session:
        topic = await create_resource(session, "topic", "pt", {"name": "Verificação isolada " + uuid.uuid4().hex})
        identity = topic.id
        assert topic.name == ""
        await session.commit()
    await refresh_translation_cache(database.async_session_maker, force=True)
    assert translation_cache.get(f"topic.{identity}.name", "ru") is None
    assert translation_cache.get(f"topic.{identity}.name", "pt")
    async with database.async_session_maker() as session:
        topic = await session.get(database.Topic, identity)
        await save_content_value(session, "topic", topic, "name", "ru", "Изолированная проверка " + uuid.uuid4().hex)
        assert (await read_content_value(session, "topic", topic, "name", "pt")).needs_review
        await session.commit()
    print(json.dumps({"migration_twice": "ok", "existing_translation_digest": "unchanged", "pt_first": "ok", "same_id_ru": "ok", "runtime_visibility": "ok"}), flush=True)
    await database.engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup", required=True, type=Path)
    args = parser.parse_args()
    if args.backup.resolve().parent.parent != Path("/root/translation_work") or args.backup.name != "psy5d2_db.dump":
        raise RuntimeError("Unexpected backup target")
    processes = json.loads(subprocess.run(["pm2", "jlist"], capture_output=True, check=True, text=True).stdout)
    for process in processes:
        if not process.get("pid"):
            continue
        try:
            environment = process_environment(process)
            url = make_url(environment["DATABASE_URL"])
        except (RuntimeError, FileNotFoundError):
            continue
        if url.database == "psy5d2_db":
            break
    else:
        raise RuntimeError("Source database not found")
    if url.host not in {"localhost", "127.0.0.1"}:
        raise RuntimeError("Expected local PostgreSQL")
    target = "authoring_isolated_" + uuid.uuid4().hex[:16]
    checked(["runuser", "-u", "postgres", "--", "createdb", "--owner", url.username, target])
    with args.backup.open("rb") as archive:
        checked(["runuser", "-u", "postgres", "--", "pg_restore", "--no-owner", "--no-acl", "--role", url.username, "--dbname", target], stdin=archive)
    os.environ.update(DATABASE_URL=url.set(database=target).render_as_string(hide_password=False), BOT_TOKEN="123456:test", BASE_WEBHOOK_URL="https://example.invalid", SERVER_IP="127.0.0.1")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    print(json.dumps({"isolated_database": target}), flush=True)
    asyncio.run(exercise())
