from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from sqlalchemy import delete, select

from .api import MaxApiClient
from .legacy import ContentMedia, async_session_maker
from .settings import get_settings
from .storage import MaxContentMedia


_materialization_locks: dict[str, asyncio.Lock] = {}


class ContentMediaMaterializationError(RuntimeError):
    pass


def _lock_for(content_key: str) -> asyncio.Lock:
    lock = _materialization_locks.get(content_key)
    if lock is None:
        lock = asyncio.Lock()
        _materialization_locks[content_key] = lock
    return lock


def _max_media_type(file_type: str) -> str:
    if file_type in {"photo", "image"}:
        return "image"
    if file_type in {"video", "audio", "file"}:
        return file_type
    raise ContentMediaMaterializationError(f"unsupported media type: {file_type}")


async def _load_media_rows(session_factory, content_key: str):
    async with session_factory() as session:
        canonical = (
            await session.execute(
                select(ContentMedia)
                .where(ContentMedia.content_key == content_key)
                .order_by(ContentMedia.id.asc())
            )
        ).scalars().all()
        max_rows = (
            await session.execute(
                select(MaxContentMedia)
                .where(MaxContentMedia.content_key == content_key)
                .order_by(MaxContentMedia.id.asc())
            )
        ).scalars().all()
    return canonical, max_rows


def _legacy_rows_match(canonical, max_rows) -> bool:
    if not max_rows:
        return not canonical
    if not canonical:
        return True
    if len(canonical) != len(max_rows):
        return False
    for source, row in zip(canonical, max_rows):
        existing_type = "image" if row.media_type == "photo" else row.media_type
        if not row.token or _max_media_type(source.file_type) != existing_type:
            return False
        if row.source_media_id is not None and row.source_media_id != source.id:
            return False
        if row.source_file_id is not None and row.source_file_id != source.file_id:
            return False
    return True


async def materialize_content_media(
    content_key: str,
    telegram_bot,
    max_client: MaxApiClient,
    *,
    session_factory=None,
    force_refresh: bool = False,
) -> int:
    session_factory = session_factory or async_session_maker
    async with _lock_for(content_key):
        canonical, existing = await _load_media_rows(session_factory, content_key)
        if not force_refresh and _legacy_rows_match(canonical, existing):
            return len(existing)
        if not canonical:
            async with session_factory() as session:
                await session.execute(delete(MaxContentMedia).where(MaxContentMedia.content_key == content_key))
                await session.commit()
            return 0

        uploaded: list[tuple[str, str, int, str]] = []
        for index, media in enumerate(canonical, start=1):
            try:
                media_type = _max_media_type(media.file_type)
                file_info = await telegram_bot.get_file(media.file_id)
                downloaded = await telegram_bot.download_file(file_info.file_path)
                content = downloaded.read()
                if not content:
                    raise ValueError("empty download")
                suffix = Path(file_info.file_path or "").suffix or (".jpg" if media_type == "image" else f".{media_type}")
                with tempfile.NamedTemporaryFile(prefix="max-content-", suffix=suffix) as temporary:
                    temporary.write(content)
                    temporary.flush()
                    result = await max_client.upload_file(media_type, temporary.name)
                token = result.get("token") if isinstance(result, dict) else None
                if not isinstance(token, str) or not token:
                    raise ValueError("MAX upload returned no token")
                uploaded.append((media_type, token, media.id, media.file_id))
            except Exception as exc:
                raise ContentMediaMaterializationError(
                    f"content={content_key} media_index={index} materialization failed: {type(exc).__name__}"
                ) from exc

        async with session_factory() as session:
            await session.execute(delete(MaxContentMedia).where(MaxContentMedia.content_key == content_key))
            session.add_all(
                [
                    MaxContentMedia(
                        content_key=content_key,
                        media_type=media_type,
                        token=token,
                        source_media_id=source_media_id,
                        source_file_id=source_file_id,
                    )
                    for media_type, token, source_media_id, source_file_id in uploaded
                ]
            )
            await session.commit()
        return len(uploaded)


async def bind_legacy_content_media(content_key: str, *, session_factory=None) -> bool:
    session_factory = session_factory or async_session_maker
    async with _lock_for(content_key):
        canonical, existing = await _load_media_rows(session_factory, content_key)
        if not canonical or not existing or any(
            row.source_media_id is not None or row.source_file_id is not None for row in existing
        ):
            return False
        if not _legacy_rows_match(canonical, existing):
            return False
        async with session_factory() as session:
            rows = (
                await session.execute(
                    select(MaxContentMedia)
                    .where(MaxContentMedia.content_key == content_key)
                    .order_by(MaxContentMedia.id.asc())
                )
            ).scalars().all()
            for source, row in zip(canonical, rows):
                row.source_media_id = source.id
                row.source_file_id = source.file_id
            await session.commit()
        return True


async def materialize_content_media_from_telegram(
    content_key: str,
    telegram_bot,
    *,
    session_factory=None,
) -> int:
    settings = get_settings()
    if not settings.max_token:
        raise ContentMediaMaterializationError("MAX credentials are not configured")
    async with MaxApiClient(settings.max_token, settings.max_api_base) as max_client:
        return await materialize_content_media(
            content_key,
            telegram_bot,
            max_client,
            session_factory=session_factory,
            force_refresh=True,
        )


async def materialize_missing_content_media(content_key: str, *, session_factory=None) -> int:
    settings = get_settings()
    telegram_token = os.getenv("BOT_TOKEN", "").strip()
    if not settings.max_token or not telegram_token:
        raise ContentMediaMaterializationError("Telegram or MAX credentials are not configured")
    from telegram_client import create_telegram_bot

    async with create_telegram_bot(telegram_token) as telegram_bot:
        async with MaxApiClient(settings.max_token, settings.max_api_base) as max_client:
            return await materialize_content_media(
                content_key,
                telegram_bot,
                max_client,
                session_factory=session_factory,
            )
