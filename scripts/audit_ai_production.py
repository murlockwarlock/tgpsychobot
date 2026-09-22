import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine


APP = Path("/root/telegram_bots/newbots")
WORK = Path("/root/translation_work")


def process_environment(process):
    values = {}
    for entry in Path(f"/proc/{int(process['pid'])}/environ").read_bytes().split(b"\0"):
        key, _, value = entry.partition(b"=")
        if key == b"DATABASE_URL":
            values["DATABASE_URL"] = value.decode()
    if "DATABASE_URL" not in values:
        raise RuntimeError("Database environment unavailable")
    return values


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


async def inspect_database(database_url, expected_database):
    url = make_url(database_url)
    if url.database != expected_database:
        raise RuntimeError("Database mapping mismatch")
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            columns = {
                row[0]
                for row in (await connection.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='ai_config'"))).all()
            }
            selected = [
                "provider", "gemini_model", "claude_model", "deepseek_model", "openai_model", "kie_model",
                "temperature", "fallback_provider", "fallback_model", "allow_fallback", "fallback_timeout",
                "max_output_tokens", "deepseek_thinking_enabled",
            ]
            available = [field for field in selected if field in columns]
            row = (await connection.execute(text("SELECT " + ", ".join(available) + " FROM ai_config WHERE id=1"))).mappings().first()
            general = (await connection.execute(text("SELECT telegram_default_language, telegram_enabled_languages, telegram_language_selection_enabled FROM bot_general_config WHERE id=1"))).mappings().first()
            return {
                "database": expected_database,
                "ai_config_columns": sorted(columns),
                "ai": dict(row or {}),
                "language": dict(general or {}),
            }
    finally:
        await engine.dispose()


async def collect():
    registry = json.loads((APP / "config/bot_instances.json").read_text())["instances"]
    expected = {
        item["pm2_name"]: item
        for item in registry
        if item.get("active") and item.get("deploy_managed") and item["platform"] == "telegram"
    }
    processes = json.loads(subprocess.run(["pm2", "jlist"], capture_output=True, check=True, text=True).stdout)
    reports = {}
    process_status = {}
    for name, item in expected.items():
        process = next((value for value in processes if value.get("name") == name), None)
        if not process or not process.get("pid"):
            raise RuntimeError(f"Process unavailable: {name}")
        process_status[name] = process.get("pm2_env", {}).get("status")
        report = await inspect_database(process_environment(process)["DATABASE_URL"], item["database"])
        report["process_names"] = sorted(name_value for name_value, item_value in expected.items() if item_value["database"] == item["database"])
        reports[item["database"]] = report
    return {"processes": process_status, "databases": reports}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("baseline", "verify"))
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    if args.baseline.name != "ai-generation-baseline.json" or args.baseline.resolve().parent.parent != WORK:
        raise RuntimeError("Unexpected baseline path")
    current = await collect()
    if args.action == "baseline":
        args.baseline.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with args.baseline.open("x") as output:
            os.chmod(args.baseline, 0o600)
            json.dump(current, output, ensure_ascii=False, indent=2, sort_keys=True)
        deepseek = {
            database: report["ai"]
            for database, report in current["databases"].items()
            if str(report["ai"].get("provider", "")).lower() == "deepseek"
        }
        print(json.dumps({"databases": len(current["databases"]), "processes": len(current["processes"]), "deepseek_databases": deepseek}, ensure_ascii=False, default=str))
        return
    baseline = json.loads(args.baseline.read_text())
    changed = []
    for database, expected in baseline["databases"].items():
        actual = current["databases"].get(database)
        if not actual:
            changed.append(database)
            continue
        expected_ai = expected.get("ai", {})
        actual_ai = actual.get("ai", {})
        ai_changed = any(actual_ai.get(key) != value for key, value in expected_ai.items())
        if (
            ai_changed
            or actual.get("language") != expected.get("language")
            or actual.get("process_names") != expected.get("process_names")
        ):
            changed.append(database)
    if set(current["databases"]) != set(baseline["databases"]):
        changed.extend(sorted(set(current["databases"]) - set(baseline["databases"])))
    if changed:
        raise RuntimeError("AI configuration changed: " + ",".join(changed))
    print(json.dumps({"databases": len(current["databases"]), "provider_model_configuration_preserved": True, "language_configuration_preserved": True}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
