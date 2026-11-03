from __future__ import annotations

import os
import tempfile
from pathlib import Path

from sqlalchemy import select

from database import EventScenarioMedia
from max_messenger_bot.api import MaxApiClient


class EventMediaMaterializationError(RuntimeError):
    pass


async def materialize_event_scenario_media(
    event_id: int,
    telegram_bot,
    max_client: MaxApiClient,
    *,
    session_factory,
) -> int:
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(EventScenarioMedia)
                .where(EventScenarioMedia.event_id == event_id)
                .order_by(EventScenarioMedia.sort_order.asc(), EventScenarioMedia.id.asc())
            )
        ).scalars().all()
        if not rows:
            return 0

        for index, row in enumerate(rows, start=1):
            if row.max_token:
                continue
            if row.source_platform == "max":
                row.max_token = row.file_id
                continue
            try:
                file_info = await telegram_bot.get_file(row.file_id)
                downloaded = await telegram_bot.download_file(file_info.file_path)
                content = downloaded.read()
                if not content:
                    raise ValueError("empty download")
                media_type = "image" if row.file_type == "photo" else "video"
                suffix = Path(file_info.file_path or "").suffix or (".jpg" if media_type == "image" else ".mp4")
                with tempfile.NamedTemporaryFile(prefix="event-media-", suffix=suffix) as temporary:
                    temporary.write(content)
                    temporary.flush()
                    response = await max_client.upload_file(media_type, temporary.name)
                token = response.get("token") if isinstance(response, dict) else None
                if not isinstance(token, str) or not token:
                    raise ValueError("MAX upload returned no token")
                row.max_token = token
                row.source_file_id = row.source_file_id or row.file_id
            except Exception as exc:
                raise EventMediaMaterializationError(
                    f"event={event_id} media_index={index} materialization failed: {type(exc).__name__}"
                ) from exc
        await session.commit()
        return len(rows)


async def materialize_event_scenario_media_from_telegram(
    event_id: int,
    max_client: MaxApiClient,
    *,
    session_factory,
) -> int:
    telegram_token = os.getenv("BOT_TOKEN", "").strip()
    if not telegram_token:
        raise EventMediaMaterializationError("Telegram credentials are not configured")
    from telegram_client import create_telegram_bot

    async with create_telegram_bot(telegram_token) as telegram_bot:
        return await materialize_event_scenario_media(
            event_id,
            telegram_bot,
            max_client,
            session_factory=session_factory,
        )
