from __future__ import annotations

import os
import re
from dataclasses import dataclass

from aiogram.exceptions import TelegramBadRequest

from sqlalchemy import select
from sqlalchemy.orm import undefer

from database import AIConfig, MediaLibrary, User
from media_scope import PHOTO_MEDIA_TYPES, load_media_scope


class MediaSourceError(ValueError):
    pass


@dataclass(frozen=True)
class MediaEditDirective:
    media_id: int
    prompt: str


def extract_media_edit_directive(text: str) -> tuple[MediaEditDirective | None, str]:
    pattern = re.compile(r"\[EDIT_IMG_MEDIA\s*:[^\]]*(?:\]|$)", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    directive = None
    if len(matches) == 1:
        match = re.fullmatch(
            r"\[EDIT_IMG_MEDIA\s*:\s*([0-9]{1,18})\s*\|\s*([^\[\]]+?)\s*\]",
            matches[0].group(), re.IGNORECASE | re.DOTALL,
        )
        if match and 0 < int(match[1]) <= 2147483647 and match[2].strip():
            directive = MediaEditDirective(int(match[1]), match[2].strip())
    clean = pattern.sub("", text)
    clean = re.sub(r"(?im)^\s*EDIT_IMG_MEDIA\s*:.*$", "", clean)
    return directive, clean.strip()


async def resolve_editable_media_source(session, *, media_id: int, topic_id: int | None):
    if type(media_id) is not int or not 0 < media_id <= 2147483647:
        raise MediaSourceError("Изображение недоступно в текущей теме.")
    scope = await load_media_scope(session, topic_id, include_media_ids=False)
    media = await session.scalar(select(MediaLibrary).options(undefer(MediaLibrary.source_image_bytes)).where(
        MediaLibrary.id == media_id,
        scope.predicate(),
        MediaLibrary.media_type.in_(PHOTO_MEDIA_TYPES),
    ))
    if media is None:
        raise MediaSourceError("Изображение недоступно в текущей теме.")
    return media


async def media_edit_context_is_current(session_factory, user_id, dialogue_id, topic_id):
    async with session_factory() as session:
        user = await session.get(User, user_id)
        return bool(user and user.current_dialogue_id == dialogue_id and user.current_topic_id == topic_id)


async def load_media_image_bytes(media, *, telegram_bot=None, max_client=None) -> bytes:
    async def load_max():
        async def download(client):
            if media.source_url:
                return await client.download_attachment_url(media.source_url)
            return await client.download_attachment(media.file_id)

        if max_client is not None:
            return await download(max_client)
        from max_messenger_bot.api import MaxApiClient
        from max_messenger_bot.settings import get_settings
        settings = get_settings()
        if not settings.max_token:
            raise MediaSourceError("Источник изображения недоступен.")
        async with MaxApiClient(settings.max_token, settings.max_api_base) as client:
            return await download(client)

    async def load_telegram(bot):
        try:
            file = await bot.get_file(media.file_id)
        except TelegramBadRequest as exc:
            invalid_id = any(value in str(exc).lower() for value in ('wrong file identifier', 'wrong remote file identifier', 'invalid file_id'))
            if media.source_platform is None and invalid_id:
                return await load_max()
            raise
        downloaded = await bot.download_file(file.file_path)
        return downloaded.read()

    try:
        platform = media.source_platform
        if platform == "max":
            if getattr(media, 'source_image_bytes', None):
                content = media.source_image_bytes
            else:
                content = await load_max()
        elif platform in (None, "telegram"):
            if telegram_bot is None:
                from telegram_client import create_telegram_bot
                token = os.getenv("BOT_TOKEN", "").strip()
                if not token:
                    raise MediaSourceError("Источник изображения недоступен.")
                async with create_telegram_bot(token) as bot:
                    content = await load_telegram(bot)
            else:
                content = await load_telegram(telegram_bot)
        else:
            raise MediaSourceError("Источник изображения недоступен.")
        if not isinstance(content, bytes) or not content:
            raise MediaSourceError("Не удалось загрузить исходное изображение.")
        return content
    except MediaSourceError:
        raise
    except Exception as exc:
        raise MediaSourceError("Не удалось загрузить исходное изображение. Попробуйте позже.") from exc


async def edit_media_image(directive, *, session_factory, user_id, dialogue_id, topic_id, edit_image, telegram_bot=None, max_client=None):
    async def current():
        return await media_edit_context_is_current(session_factory, user_id, dialogue_id, topic_id)

    async def ensure_enabled(session):
        config = await session.get(AIConfig, 1)
        if not config or not config.allow_image_edit:
            raise MediaSourceError("Редактирование изображений отключено.")

    if not await current():
        return None
    async with session_factory() as session:
        await ensure_enabled(session)
        media = await resolve_editable_media_source(session, media_id=directive.media_id, topic_id=topic_id)
    image_bytes = await load_media_image_bytes(media, telegram_bot=telegram_bot, max_client=max_client)
    if not await current():
        return None
    async with session_factory() as session:
        await ensure_enabled(session)
        current_media = await resolve_editable_media_source(session, media_id=directive.media_id, topic_id=topic_id)
        if current_media.file_id != media.file_id:
            raise MediaSourceError("Исходное изображение изменилось. Повторите запрос.")
    edited = await edit_image(directive.prompt, image_bytes)
    if not await current():
        return None
    async with session_factory() as session:
        current_media = await resolve_editable_media_source(session, media_id=directive.media_id, topic_id=topic_id)
        if current_media.file_id != media.file_id:
            raise MediaSourceError("Исходное изображение изменилось. Повторите запрос.")
    return edited
