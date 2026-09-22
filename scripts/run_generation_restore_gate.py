import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine


ROOT = Path("/root/telegram_bots/newbots")
WORK = Path("/root/translation_work/deepseek_generation_20260922")
BACKUP = WORK / "psy5d2_db_fresh.dump"


def db_environment(url, database):
    return {
        "PGHOST": url.host or "localhost",
        "PGPORT": str(url.port or 5432),
        "PGUSER": url.username or "postgres",
        "PGPASSWORD": url.password or "",
        "PGDATABASE": database,
    }


def process_database_url(database):
    processes = json.loads(subprocess.run(["pm2", "jlist"], capture_output=True, check=True, text=True).stdout)
    registry = json.loads((ROOT / "config/bot_instances.json").read_text())["instances"]
    names = {
        item["pm2_name"]
        for item in registry
        if item.get("active") and item.get("deploy_managed") and item.get("platform") == "telegram" and item.get("database") == database
    }
    for process in processes:
        if process.get("name") not in names or not process.get("pid"):
            continue
        values = {}
        for entry in Path(f"/proc/{int(process['pid'])}/environ").read_bytes().split(b"\0"):
            key, _, value = entry.partition(b"=")
            if key == b"DATABASE_URL":
                values["DATABASE_URL"] = value.decode()
        if values.get("DATABASE_URL"):
            return make_url(values["DATABASE_URL"])
    raise RuntimeError(f"No active process mapped to {database}")


def run(command, environment):
    subprocess.run(command, check=True, env=environment, cwd=ROOT)


async def init_and_validate(database_url):
    os.environ["DATABASE_URL"] = database_url
    os.environ.setdefault("BOT_TOKEN", "restore-gate-token")
    os.environ.setdefault("BASE_WEBHOOK_URL", "https://example.invalid")
    os.environ.setdefault("TELEGRAM_DELIVERY_MODE", "webhook")
    sys.path.insert(0, str(ROOT))
    from database import async_session_maker, engine, init_db
    from payment_index_validation import read_postgres_unresolved_index, validate_postgres_unresolved_index

    await init_db()
    async with engine.connect() as connection:
        index = await connection.run_sync(read_postgres_unresolved_index)
        validate_postgres_unresolved_index(index)
        columns = {
            row[0]
            for row in (await connection.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='ai_config'"))).all()
        }
        payment_counts = {}
        for table in ("yookassa_recurring_attempts", "user_subscriptions", "payment_notification_outbox"):
            payment_counts[table] = int((await connection.execute(text(f"SELECT count(*) FROM {table}"))).scalar_one())
    async with async_session_maker() as session:
        await session.execute(text("SELECT 1"))
    await engine.dispose()
    required = {"max_output_tokens", "deepseek_thinking_enabled"}
    if not required.issubset(columns):
        raise RuntimeError(f"Generation columns missing after init_db: {sorted(required - columns)}")
    return index, payment_counts


async def main():
    WORK.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(WORK, 0o700)
    source_url = process_database_url("psy5d2_db")
    source_env = db_environment(source_url, "veraveda_db")
    backup_env = os.environ.copy()
    backup_env.update(source_env)
    run(["pg_dump", "--format=custom", "--no-owner", "--no-acl", "--file", str(BACKUP), "psy5d2_db"], backup_env)
    os.chmod(BACKUP, 0o600)

    target = "generation_restore_gate_" + uuid.uuid4().hex[:12]
    if source_url.host not in {None, "localhost", "127.0.0.1"}:
        raise RuntimeError("Restore gate requires local PostgreSQL administration")
    admin_env = {key: value for key, value in os.environ.items() if not key.startswith("PG")}
    run(["runuser", "-u", "postgres", "--", "dropdb", "--if-exists", target], admin_env)
    run(["runuser", "-u", "postgres", "--", "createdb", "--owner", source_url.username or "postgres", target], admin_env)
    restore_env = {key: value for key, value in os.environ.items() if not key.startswith("PG")}
    with BACKUP.open("rb") as archive:
        subprocess.run(
            ["runuser", "-u", "postgres", "--", "pg_restore", "--no-owner", "--no-acl", "--exit-on-error", "--role", source_url.username or "postgres", "--dbname", target],
            check=True,
            env=restore_env,
            cwd=ROOT,
            stdin=archive,
        )

    target_url = source_url.set(database=target).render_as_string(hide_password=False)
    index, payment_counts = await init_and_validate(target_url)

    startup_env = os.environ.copy()
    startup_env.update({
        "DATABASE_URL": target_url,
        "BOT_TOKEN": "restore-gate-token",
        "BASE_WEBHOOK_URL": "https://example.invalid",
        "TELEGRAM_DELIVERY_MODE": "webhook",
    })
    run([
        "/root/telegram_bots/venv/bin/python",
        "-c",
        "import asyncio; import main; from database import init_db; asyncio.run(init_db())",
    ], startup_env)
    run(["runuser", "-u", "postgres", "--", "dropdb", "--if-exists", target], admin_env)
    print(json.dumps({
        "backup": str(BACKUP),
        "target": target,
        "validator": "ok",
        "index": dict(index),
        "payment_counts": payment_counts,
        "init_db": "ok",
        "startup_import": "ok",
        "cleanup": "ok",
    }, default=str, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
