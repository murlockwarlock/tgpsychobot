import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine


APP = Path("/root/telegram_bots/newbots")
WORK = Path("/root/translation_work/openrouter_perplexity_deepgram")
BASELINE_NAME = "ai-baseline.json"


def stable_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def process_environment(process):
    values = {}
    for entry in Path(f"/proc/{int(process['pid'])}/environ").read_bytes().split(b"\0"):
        key, _, value = entry.partition(b"=")
        if key == b"DATABASE_URL":
            values["DATABASE_URL"] = value.decode()
    if "DATABASE_URL" not in values:
        raise RuntimeError("Database environment unavailable")
    return values


async def inspect_database(database_url, expected_database):
    url = make_url(database_url)
    if url.database != expected_database:
        raise RuntimeError("Database mapping mismatch")
    engine = create_async_engine(database_url, pool_pre_ping=True)
    try:
        async with engine.connect() as connection:
            table_names = {
                row[0]
                for row in (
                    await connection.execute(
                        text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
                    )
                ).all()
            }
            columns = {
                row[0]
                for row in (
                    await connection.execute(
                        text("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='ai_config'")
                    )
                ).all()
            }
            ai_fields = (
                "provider",
                "gemini_model",
                "claude_model",
                "deepseek_model",
                "openai_model",
                "kie_model",
                "openrouter_model",
                "perplexity_model",
                "deepgram_model",
                "vision_provider",
                "vision_model",
                "transcription_provider",
                "max_output_tokens",
                "deepseek_thinking_enabled",
                "fallback_provider",
                "fallback_model",
                "allow_fallback",
            )
            available_ai_fields = [field for field in ai_fields if field in columns]
            ai_row = {}
            if available_ai_fields:
                ai_row = (
                    await connection.execute(
                        text("SELECT " + ", ".join(available_ai_fields) + " FROM ai_config WHERE id=1")
                    )
                ).mappings().first() or {}
            key_fields = ("openrouter_api_key", "perplexity_api_key", "deepgram_api_key")
            available_key_fields = [field for field in key_fields if field in columns]
            key_row = {}
            if available_key_fields:
                key_row = (
                    await connection.execute(
                        text("SELECT " + ", ".join(available_key_fields) + " FROM ai_config WHERE id=1")
                    )
                ).mappings().first() or {}
            language = {}
            if "bot_general_config" in table_names:
                language = dict(
                    (
                        await connection.execute(
                            text(
                                "SELECT telegram_default_language, telegram_enabled_languages, "
                                "telegram_language_selection_enabled FROM bot_general_config WHERE id=1"
                            )
                        )
                    ).mappings().first()
                    or {}
                )
            user_digest = None
            user_count = 0
            if "users" in table_names:
                user_rows = (
                    await connection.execute(
                        text("SELECT id, telegram_language_code FROM users ORDER BY id")
                    )
                ).all()
                user_count = len(user_rows)
                user_digest = stable_digest([(row[0], row[1]) for row in user_rows])
            content_digest = None
            content_counts = {}
            content_rows = {}
            for table_name in ("topics", "content", "test_questions", "subscription_plans"):
                if table_name not in table_names:
                    continue
                count = (
                    await connection.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                ).scalar_one()
                content_counts[table_name] = int(count)
                rows = (
                    await connection.execute(text(f"SELECT * FROM {table_name} ORDER BY 1"))
                ).all()
                content_rows[table_name] = [tuple(row) for row in rows]
            if content_rows:
                content_digest = stable_digest(content_rows)
            return {
                "database": expected_database,
                "ai": dict(ai_row or {}),
                "credential_configured": {
                    field: bool((key_row or {}).get(field)) for field in available_key_fields
                },
                "language": language,
                "user_preferences": {"count": user_count, "digest": user_digest},
                "content": {"counts": content_counts, "digest": content_digest},
                "columns": sorted(columns),
            }
    finally:
        await engine.dispose()


async def collect():
    registry = json.loads((APP / "config/bot_instances.json").read_text())["instances"]
    managed = [item for item in registry if item.get("active") and item.get("deploy_managed")]
    processes = json.loads(
        subprocess.run(["pm2", "jlist"], capture_output=True, check=True, text=True).stdout
    )
    process_by_name = {item.get("name"): item for item in processes}
    statuses = {}
    database_inputs = {}
    process_names = {}
    for item in managed:
        process = process_by_name.get(item["pm2_name"])
        if not process or not process.get("pid"):
            raise RuntimeError(f"Process unavailable: {item['pm2_name']}")
        statuses[item["pm2_name"]] = process.get("pm2_env", {}).get("status")
        database_inputs[item["database"]] = process_environment(process)["DATABASE_URL"]
        process_names.setdefault(item["database"], []).append(item["pm2_name"])
    databases = {}
    for database, database_url in database_inputs.items():
        report = await inspect_database(database_url, database)
        report["process_names"] = sorted(process_names[database])
        databases[database] = report
    return {"processes": statuses, "databases": databases}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("baseline", "verify"))
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    expected_path = WORK / BASELINE_NAME
    if args.baseline != expected_path:
        raise RuntimeError("Unexpected baseline path")
    current = await collect()
    if args.action == "baseline":
        args.baseline.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with args.baseline.open("x") as output:
            os.chmod(args.baseline, 0o600)
            json.dump(current, output, ensure_ascii=False, indent=2, sort_keys=True)
        print(json.dumps({"processes": len(current["processes"]), "databases": len(current["databases"])}, ensure_ascii=False))
        return
    baseline = json.loads(args.baseline.read_text())
    changed = []
    for name, expected in baseline.get("processes", {}).items():
        if current.get("processes", {}).get(name) != expected:
            changed.append(f"process:{name}")
    for database, expected in baseline.get("databases", {}).items():
        actual = current.get("databases", {}).get(database)
        if actual is None:
            changed.append(f"database:{database}:missing")
            continue
        for key in ("language", "user_preferences", "content", "process_names"):
            if actual.get(key) != expected.get(key):
                changed.append(f"database:{database}:{key}")
        for field, value in expected.get("ai", {}).items():
            if actual.get("ai", {}).get(field) != value:
                changed.append(f"database:{database}:ai:{field}")
        for field, value in expected.get("credential_configured", {}).items():
            if actual.get("credential_configured", {}).get(field) != value:
                changed.append(f"database:{database}:credential:{field}")
    if set(current.get("databases", {})) != set(baseline.get("databases", {})):
        changed.append("database-set")
    if changed:
        raise RuntimeError("Production state changed: " + ",".join(changed))
    print(json.dumps({"processes": len(current["processes"]), "databases": len(current["databases"]), "preserved": True}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
