import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
import subprocess

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine


APP = Path("/root/telegram_bots/newbots")
HOTFIX_AT = "2026-09-13 16:45:00"
WINDOW_END = "2026-09-14 12:00:00"


def process_environment(process):
    values = {}
    for entry in Path(f"/proc/{int(process['pid'])}/environ").read_bytes().split(b"\0"):
        key, _, value = entry.partition(b"=")
        if key == b"DATABASE_URL":
            values["DATABASE_URL"] = value.decode()
    return values


async def inspect(database_url, database):
    url = make_url(database_url)
    if url.database != database:
        raise RuntimeError("Database mapping mismatch")
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            rows = (await connection.execute(text("""
                SELECT created_at, request_payload, diagnostics_json, finish_reason, model
                FROM ai_logs
                WHERE lower(provider) = 'deepseek'
                  AND created_at >= :start AND created_at < :finish
                ORDER BY created_at
            """), {"start": datetime.fromisoformat("2026-09-13 00:00:00"), "finish": datetime.fromisoformat(WINDOW_END)})).all()
            before = []
            after = []
            for created_at, payload_text, diagnostics_text, finish_reason, model in rows:
                try:
                    payload = json.loads(payload_text or "{}")
                except (TypeError, ValueError):
                    payload = {}
                try:
                    diagnostics = json.loads(diagnostics_text or "{}")
                except (TypeError, ValueError):
                    diagnostics = {}
                item = {
                    "created_at": str(created_at),
                    "model": model,
                    "max_tokens": payload.get("payload", {}).get("max_tokens"),
                    "thinking": payload.get("payload", {}).get("extra_body", {}).get("thinking", {}).get("type"),
                    "reasoning_present": bool(diagnostics.get("reasoning_content_present")),
                    "reasoning_length": int(diagnostics.get("reasoning_content_length") or 0),
                    "finish_reason": finish_reason,
                }
                (before if str(created_at) < HOTFIX_AT else after).append(item)
            def summarize(items):
                return {
                    "rows": len(items),
                    "thinking_values": sorted({item["thinking"] or "absent" for item in items}),
                    "max_tokens": sorted({item["max_tokens"] for item in items}),
                    "reasoning_rows": sum(item["reasoning_present"] for item in items),
                    "reasoning_length_total": sum(item["reasoning_length"] for item in items),
                    "finish_reasons": sorted({item["finish_reason"] for item in items}),
                }
            return {"database": database, "before_hotfix": summarize(before), "after_hotfix": summarize(after)}
    finally:
        await engine.dispose()


async def main():
    instances = json.loads((APP / "config/bot_instances.json").read_text())["instances"]
    expected = {item["pm2_name"]: item for item in instances if item.get("active") and item.get("deploy_managed") and item["platform"] == "telegram"}
    processes = json.loads(subprocess.run(["pm2", "jlist"], capture_output=True, check=True, text=True).stdout)
    reports = {}
    for name, item in expected.items():
        process = next((value for value in processes if value.get("name") == name and value.get("pid")), None)
        if process is None:
            raise RuntimeError(f"Process unavailable: {name}")
        database = item["database"]
        if database not in reports:
            reports[database] = await inspect(process_environment(process)["DATABASE_URL"], database)
    print(json.dumps(reports, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
