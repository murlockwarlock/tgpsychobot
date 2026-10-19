from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass

from aiogram.exceptions import TelegramBadRequest

from sqlalchemy import select
from sqlalchemy.orm import undefer

from database import AIConfig, MediaLibrary, User
from media_scope import PHOTO_MEDIA_TYPES, load_media_scope
from response_buttons import extract_test_start_directive


class MediaSourceError(ValueError):
    pass


@dataclass(frozen=True)
class MediaEditDirective:
    media_id: int
    prompt: str


@dataclass(frozen=True)
class EditedMediaImage:
    image_bytes: bytes
    source_identity: tuple[str, str | None, str | None, str]


def media_source_identity(media, *, include_content_digest=False):
    identity = (media.file_id, media.source_platform, media.source_url)
    if not include_content_digest:
        return identity
    content = getattr(media, "source_image_bytes", None)
    digest = hashlib.sha256(content).hexdigest() if content is not None else "<missing>"
    return (*identity, digest)


def source_identity_matches(media, source_identity):
    if len(source_identity) > 3:
        return media_source_identity(media, include_content_digest=True) == source_identity
    return media_source_identity(media) == source_identity[:3]


def extract_media_edit_directive(text: str) -> tuple[MediaEditDirective | None, str]:
    pattern = re.compile(r"\[EDIT_IMG_MEDIA\b[^\]]*(?:\]|$)", re.IGNORECASE)
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
    clean = re.sub(r"(?im)^\s*EDIT_IMG_MEDIA\b.*$", "", clean)
    inline_marker = re.search(r"\bEDIT_IMG_MEDIA\b", clean, re.IGNORECASE)
    if inline_marker:
        clean = re.sub(r"\bEDIT_IMG_MEDIA\b[^\n]*", "", clean, count=1, flags=re.IGNORECASE)
        directive = None
    if matches or inline_marker:
        starts_test, without_test = extract_test_start_directive(clean)
        other_commands = re.compile(
            r"\[(?:SHOW_IMG|RANDOM_IMG|CHOICE_IMG(?:_HIDDEN)?|SEND_AUDIO|GEN_IMG|IMG|EDIT_IMG)\s*:[^\]]*(?:\]|$)"
            r"|\b(?:GEN_IMG|EDIT_IMG)\s*:[^\n]*",
            re.IGNORECASE,
        )
        if starts_test or other_commands.search(without_test):
            directive = None
            clean = other_commands.sub('', without_test)
    return directive, clean.strip()


async def resolve_editable_media_source(session, *, media_id: int, topic_id: int | None, include_source_bytes: bool = False):
    if type(media_id) is not int or not 0 < media_id <= 2147483647:
        raise MediaSourceError("Изображение недоступно в текущей теме.")
    scope = await load_media_scope(session, topic_id, include_media_ids=False)
    query = select(MediaLibrary).where(
        MediaLibrary.id == media_id,
        scope.predicate(),
        MediaLibrary.media_type.in_(PHOTO_MEDIA_TYPES),
    )
    if include_source_bytes:
        query = query.options(undefer(MediaLibrary.source_image_bytes))
    media = await session.scalar(query)
    if media is None:
        raise MediaSourceError("Изображение недоступно в текущей теме.")
    return media


async def media_edit_context_is_current(session_factory, user_id, dialogue_id, topic_id):
    async with session_factory() as session:
        user = await session.get(User, user_id)
        return bool(user and user.current_dialogue_id == dialogue_id and user.current_topic_id == topic_id)


async def validate_edited_media_source(session_factory, *, media_id, topic_id, source_identity):
    async with session_factory() as session:
        include_source_bytes = len(source_identity) > 3
        media = await resolve_editable_media_source(session, media_id=media_id, topic_id=topic_id, include_source_bytes=include_source_bytes)
        if not source_identity_matches(media, source_identity):
            raise MediaSourceError("Исходное изображение изменилось. Повторите запрос.")


async def download_max_image_source(client, file_id, source_url=None):
    def validate(content):
        if not isinstance(content, bytes) or not content or len(content) > 20 * 1024 * 1024:
            raise MediaSourceError("Не удалось загрузить исходное изображение.")
        return content

    if source_url:
        try:
            return validate(await client.download_attachment_url(source_url))
        except Exception:
            pass
    return validate(await client.download_attachment(file_id))


async def load_media_image_bytes(media, *, telegram_bot=None, max_client=None) -> bytes:
    async def load_max():
        async def download(client):
            return await download_max_image_source(client, media.file_id, media.source_url)

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
        media = await resolve_editable_media_source(session, media_id=directive.media_id, topic_id=topic_id, include_source_bytes=True)
    source_identity = media_source_identity(media, include_content_digest=True)
    image_bytes = await load_media_image_bytes(media, telegram_bot=telegram_bot, max_client=max_client)
    if not await current():
        return None
    async with session_factory() as session:
        await ensure_enabled(session)
        current_media = await resolve_editable_media_source(session, media_id=directive.media_id, topic_id=topic_id)
        if media_source_identity(current_media) != source_identity[:3]:
            raise MediaSourceError("Исходное изображение изменилось. Повторите запрос.")
    edited = await edit_image(directive.prompt, image_bytes)
    if not await current():
        return None
    async with session_factory() as session:
        current_media = await resolve_editable_media_source(session, media_id=directive.media_id, topic_id=topic_id)
        if media_source_identity(current_media) != source_identity[:3]:
            raise MediaSourceError("Исходное изображение изменилось. Повторите запрос.")
    return EditedMediaImage(edited, source_identity)
