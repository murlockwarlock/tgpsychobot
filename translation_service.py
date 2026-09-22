from __future__ import annotations

import hashlib
import json
import string
import time
from collections import Counter
from dataclasses import dataclass
import asyncio
from typing import Any
from content_locales import is_admin_content_key


SUPPORTED_TELEGRAM_LOCALES = ("ru", "en", "pt")
LOCALE_LABELS = {
    "ru": "🇷🇺 Русский",
    "en": "🇬🇧 English",
    "pt": "🇵🇹 Português",
}


class TranslationValidationError(ValueError):
    pass


@dataclass(frozen=True)
class TranslationSnapshot:
    revision: int
    sources: dict[str, str]
    translations: dict[tuple[str, str], str]

    def get(self, translation_key: str, locale: Any, default_locale: Any = "ru") -> str | None:
        source = self.sources.get(translation_key)
        if source is None:
            return None
        requested_locale = normalize_locale(locale) or "ru"
        default = normalize_default_language(default_locale)
        if requested_locale == "ru":
            return source or None
        exact = self.translations.get((requested_locale, translation_key))
        if exact:
            return exact
        if is_admin_content_key(translation_key):
            return source or None
        if default not in {"ru", requested_locale}:
            default_text = self.translations.get((default, translation_key))
            if default_text:
                return default_text
        return source


class TranslationCache:
    def __init__(self) -> None:
        self._snapshot = TranslationSnapshot(0, {}, {})

    @property
    def revision(self) -> int:
        return self._snapshot.revision

    @property
    def snapshot(self) -> TranslationSnapshot:
        return self._snapshot

    def install(
        self,
        revision: int,
        sources: dict[str, str],
        translations: dict[tuple[str, str], str],
    ) -> None:
        self._snapshot = TranslationSnapshot(
            revision=revision,
            sources=dict(sources),
            translations=dict(translations),
        )

    def get(self, translation_key: str, locale: Any, default_locale: Any = "ru") -> str | None:
        return self._snapshot.get(translation_key, locale, default_locale)


translation_cache = TranslationCache()
_cache_refresh_lock = asyncio.Lock()
_cache_last_refresh_monotonic = 0.0
_CACHE_REFRESH_INTERVAL_SECONDS = 1.0


async def ensure_translation_cache_fresh(
    session_maker,
    sources: dict[str, str],
) -> TranslationSnapshot:
    global _cache_last_refresh_monotonic
    now = time.monotonic()
    if (
        translation_cache.snapshot.sources == sources
        and now - _cache_last_refresh_monotonic < _CACHE_REFRESH_INTERVAL_SECONDS
    ):
        return translation_cache.snapshot

    async with _cache_refresh_lock:
        now = time.monotonic()
        if (
            translation_cache.snapshot.sources == sources
            and now - _cache_last_refresh_monotonic < _CACHE_REFRESH_INTERVAL_SECONDS
        ):
            return translation_cache.snapshot
        async with session_maker() as session:
            from translation_pack_manager import translation_coordination_lock

            async with translation_coordination_lock(session):
                return await _install_translation_snapshot(session, sources, now)


async def _install_translation_snapshot(
    session,
    sources: dict[str, str],
    now: float,
) -> TranslationSnapshot:
    from sqlalchemy import select
    from database import BotGeneralConfig, BotTranslation

    global _cache_last_refresh_monotonic
    config = await session.get(BotGeneralConfig, 1)
    revision = int(getattr(config, "translations_revision", 0) or 0)
    if translation_cache.revision == revision and translation_cache.snapshot.sources == sources:
        _cache_last_refresh_monotonic = now
        return translation_cache.snapshot
    rows = (
        await session.execute(
            select(
                BotTranslation.locale,
                BotTranslation.translation_key,
                BotTranslation.text,
                BotTranslation.source_hash,
            )
        )
    ).all()
    translations = {
        (locale, key): text
        for locale, key, text, stored_hash in rows
        if (
            locale in SUPPORTED_TELEGRAM_LOCALES
            and locale != "ru"
            and key in sources
            and isinstance(text, str)
            and text != ""
            and (is_admin_content_key(key) or stored_hash == source_hash(sources[key]))
            and (not is_admin_content_key(key) or dynamic_translation_safe(sources[key], text))
        )
    }
    translation_cache.install(revision, sources, translations)
    _cache_last_refresh_monotonic = now
    return translation_cache.snapshot


def dynamic_translation_safe(source: str, translated: str) -> bool:
    from translation_registry import _validate_telegram_html_value, embedded_target_signature
    try:
        _validate_telegram_html_value(translated)
        if source:
            validate_translation_format(source, translated)
            if embedded_target_signature(source) != embedded_target_signature(translated):
                return False
        return True
    except (ValueError, TypeError):
        return False


def translate(
    translation_key: str,
    locale: Any,
    *,
    default_locale: Any = "ru",
    fallback: str | None = None,
    source: str | None = None,
) -> str | None:
    if source is not None and translation_cache.snapshot.sources.get(translation_key) != source:
        return fallback
    value = translation_cache.get(translation_key, locale, default_locale)
    return value if value is not None else fallback


async def refresh_translation_cache(session_maker, *, force: bool = False) -> TranslationSnapshot:
    from translation_registry import build_translation_registry
    from translation_pack_manager import translation_coordination_lock

    global _cache_last_refresh_monotonic
    now = time.monotonic()
    if (
        not force and translation_cache.snapshot.sources
        and now - _cache_last_refresh_monotonic < _CACHE_REFRESH_INTERVAL_SECONDS
    ):
        return translation_cache.snapshot

    async with _cache_refresh_lock:
        now = time.monotonic()
        if (
            not force and translation_cache.snapshot.sources
            and now - _cache_last_refresh_monotonic < _CACHE_REFRESH_INTERVAL_SECONDS
        ):
            return translation_cache.snapshot
        async with session_maker() as session:
            async with translation_coordination_lock(session):
                registry = await build_translation_registry(session)
                sources = {
                    key: source.source for key, source in registry.snapshot().items()
                }
                return await _install_translation_snapshot(session, sources, now)


async def resolve_user_effective_locale(
    session,
    user_or_id: Any,
    *,
    platform: str = "telegram",
    admin: bool = False,
) -> str:
    from sqlalchemy.exc import OperationalError
    from database import BotGeneralConfig, User

    if platform.lower() == "max" or admin:
        return "ru"
    user = user_or_id
    try:
        if not isinstance(user_or_id, User):
            user = await session.get(User, user_or_id)
        config = await session.get(BotGeneralConfig, 1)
    except OperationalError:
        user = None
        config = None
    return resolve_effective_locale(
        getattr(user, "telegram_language_code", None) if user else None,
        getattr(config, "telegram_default_language", "ru") if config else "ru",
        bool(getattr(config, "telegram_language_selection_enabled", False)) if config else False,
        getattr(config, "telegram_enabled_languages", '["ru"]') if config else '["ru"]',
        platform=platform,
    )


def normalize_locale(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in SUPPORTED_TELEGRAM_LOCALES else None


def normalize_enabled_languages(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            raw = []
    if not isinstance(raw, (list, tuple, set)):
        raw = []
    enabled = {normalize_locale(value) for value in raw}
    enabled.discard(None)
    enabled.add("ru")
    return tuple(locale for locale in SUPPORTED_TELEGRAM_LOCALES if locale in enabled)


def normalize_default_language(value: Any) -> str:
    return normalize_locale(value) or "ru"


def resolve_effective_locale(
    requested: Any,
    default: Any,
    selector_enabled: bool,
    enabled: Any,
    *,
    platform: str = "telegram",
    admin: bool = False,
) -> str:
    if platform.lower() == "max" or admin:
        return "ru"
    enabled_locales = normalize_enabled_languages(enabled)
    default_locale = normalize_default_language(default)
    if default_locale not in enabled_locales:
        default_locale = "ru"
    if not selector_enabled:
        return default_locale
    requested_locale = normalize_locale(requested)
    if requested_locale in enabled_locales:
        return requested_locale
    return default_locale


def source_hash(text: str) -> str:
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def format_signature(text: str) -> Counter[tuple[str, str | None, str]]:
    signature: Counter[tuple[str, str | None, str]] = Counter()
    try:
        parsed = string.Formatter().parse(text)
        for _, field_name, format_spec, conversion in parsed:
            if field_name is not None:
                signature[(field_name, conversion, format_spec)] += 1
    except (ValueError, TypeError) as exc:
        raise TranslationValidationError(f"Invalid format string: {exc}") from exc
    return signature


def validate_translation_format(source: str, translated: str) -> None:
    source_signature = format_signature(source)
    translated_signature = format_signature(translated)
    if source_signature != translated_signature:
        raise TranslationValidationError(
            "Translation placeholders do not match the source"
        )
