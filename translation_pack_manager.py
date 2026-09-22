from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from collections import Counter
import json
from content_locales import is_admin_content_key
from typing import Any

from sqlalchemy import select, text, inspect as sqlalchemy_inspect

from database import BotGeneralConfig, BotTranslation
from translation_registry import TranslationRegistry, validate_translation_value
from translation_service import (
    SUPPORTED_TELEGRAM_LOCALES,
    normalize_enabled_languages,
    translation_cache,
    source_hash,
    dynamic_translation_safe,
)


class TranslationPackValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(errors))


class TranslationResourceNotReady(ValueError):
    pass


_translation_coordination_lock = asyncio.Lock()
TRANSLATION_PACK_SCHEMA_VERSION = 2
TRANSLATION_PACK_LOCALES = tuple(locale for locale in SUPPORTED_TELEGRAM_LOCALES if locale != "ru")


def get_locale_readiness(readiness: dict[str, Any], locale: str) -> dict[str, Any]:
    if locale == "ru":
        return {
            "ready": True,
            "canonical": True,
            "translated": 0,
            "required": 0,
            "missing": [],
            "stale": [],
            "invalid": [],
        }
    return readiness.get("locales", {}).get(
        locale,
        {
            "ready": False,
            "canonical": False,
            "translated": 0,
            "required": 0,
            "missing": [],
            "stale": [],
            "invalid": [],
        },
    )


def humanize_translation_pack_errors(
    errors: list[str] | tuple[str, ...],
    *,
    expected_locale: str | None = None,
) -> tuple[str, ...]:
    combined = "\n".join(str(error) for error in errors).lower()
    messages: list[str] = []
    if "pack target" in combined:
        messages.append(
            "Этот файл перевода создан для другой версии или другого бота. "
            "Экспортируйте пакет на экране этого бота."
        )
    if "pack locale" in combined or "entry locale" in combined or "unsupported locale" in combined:
        if expected_locale in TRANSLATION_PACK_LOCALES:
            messages.append(
                f"В файле указан другой язык. Здесь ожидается пакет {expected_locale.upper()}."
            )
        else:
            messages.append("В файле указан неподдерживаемый язык.")
    if "schema_version" in combined:
        messages.append("Формат файла устарел. Экспортируйте новый шаблон из этого меню.")
    if "stale source hash" in combined:
        messages.append(
            "Перевод устарел после изменения русского текста. "
            "Экспортируйте актуальный шаблон и перенесите переводы."
        )
    missing_count = sum("missing required translation" in str(error).lower() for error in errors)
    if missing_count:
        messages.append(f"Не хватает обязательных переводов: {missing_count}.")
    if "invalid translation" in combined or "replykeyboard collision" in combined:
        messages.append(
            "Проверьте плейсхолдеры, HTML-разметку, кнопки, ссылки и ограничения Telegram."
        )
    if "duplicate translation entry" in combined:
        messages.append("В файле есть повторяющиеся записи. Удалите дубликаты и загрузите файл снова.")
    if "unknown translation key" in combined:
        messages.append("В файле есть записи, которых нет в текущем наборе переводов этого бота.")
    if "empty required translation" in combined:
        messages.append("Заполните все обязательные переводы.")
    if "translations must be a list" in combined or "pack must be an object" in combined:
        messages.append("Структура файла не распознана. Экспортируйте актуальный шаблон.")
    if "russian is the canonical source" in combined:
        messages.append("Русский — исходный язык; импортировать можно только EN или PT.")
    if not messages:
        messages.append("Файл не прошёл проверку. Переводы не изменены.")
    return tuple(dict.fromkeys(messages))


def validate_translation_pack(
    pack: dict[str, Any],
    registry: TranslationRegistry,
    *,
    required_locales: tuple[str, ...] = (),
    expected_locale: str | None = None,
    expected_target: dict[str, Any] | None = None,
) -> None:
    registry = registry.system()
    errors: list[str] = []
    if not isinstance(pack, dict):
        raise TranslationPackValidationError(["pack must be an object"])
    if pack.get("schema_version") != TRANSLATION_PACK_SCHEMA_VERSION:
        errors.append(f"schema_version must be {TRANSLATION_PACK_SCHEMA_VERSION}")
    if expected_target is not None:
        target = pack.get("target")
        if not isinstance(target, dict):
            errors.append("pack target is missing")
        else:
            for field in ("telegram_bot_id", "database"):
                if target.get(field) != expected_target.get(field):
                    errors.append(f"pack target {field} does not match")
    pack_locale = pack.get("locale")
    if pack_locale not in TRANSLATION_PACK_LOCALES:
        errors.append(f"unsupported pack locale: {pack_locale}")
    if expected_locale is not None and expected_locale not in TRANSLATION_PACK_LOCALES:
        errors.append(f"unsupported expected locale: {expected_locale}")
    elif expected_locale is not None and pack_locale != expected_locale:
        errors.append(
            f"pack locale {pack_locale} does not match expected locale {expected_locale}"
        )
    entries = pack.get("translations")
    if not isinstance(entries, list):
        errors.append("translations must be a list")
        raise TranslationPackValidationError(errors)

    target_locale = expected_locale or pack_locale
    effective_required_locales = (
        (target_locale,)
        if target_locale in TRANSLATION_PACK_LOCALES
        else tuple(required_locales)
    )

    seen: Counter[tuple[Any, Any]] = Counter()
    present: set[tuple[str, str]] = set()
    reply_button_values: dict[str, dict[str, str]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"translations[{index}] must be an object")
            continue
        locale = entry.get("locale")
        key = entry.get("translation_key")
        text = entry.get("text")
        entry_id = (locale, key)
        seen[entry_id] += 1
        if seen[entry_id] > 1:
            errors.append(f"duplicate translation entry: {locale}/{key}")
        if locale != pack_locale:
            errors.append(f"entry locale does not match pack locale: {locale}/{key}")
        if locale not in TRANSLATION_PACK_LOCALES:
            errors.append(f"unsupported locale: {locale}")
            continue
        if locale == "ru":
            errors.append("Russian is the canonical source and cannot be imported")
            continue
        if is_admin_content_key(key):
            continue
        source = registry.get(key)
        if source is None:
            errors.append(f"unknown translation key: {key}")
            continue
        if not isinstance(text, str):
            errors.append(f"translation text must be a string: {key}")
            continue
        if locale in required_locales and source.required and not text:
            errors.append(f"empty required translation: {locale}/{key}")
        if entry.get("source_hash") != source_hash(source.source):
            errors.append(f"stale source hash: {key}")
        try:
            validate_translation_value(source, text)
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid translation {locale}/{key}: {exc}")
        if source.kind == "reply_button":
            prior_key = reply_button_values.setdefault(locale, {}).get(text)
            if prior_key and prior_key != key:
                errors.append(
                    f"ReplyKeyboard collision for {locale}: {prior_key} and {key}"
                )
            else:
                reply_button_values.setdefault(locale, {})[text] = key
        present.add((locale, key))

    for locale in effective_required_locales:
        if locale not in TRANSLATION_PACK_LOCALES:
            errors.append(f"unsupported required locale: {locale}")
            continue
        for key in registry.required_keys():
            if (locale, key) not in present:
                errors.append(f"missing required translation: {locale}/{key}")

    if errors:
        raise TranslationPackValidationError(errors)


async def export_translation_pack(
    session,
    registry: TranslationRegistry,
    *,
    locale: str,
    target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry = registry.system()
    if locale not in TRANSLATION_PACK_LOCALES:
        raise ValueError(f"unsupported translation pack locale: {locale}")
    rows = (
        await session.execute(
            select(BotTranslation.locale, BotTranslation.translation_key, BotTranslation.text)
            .where(BotTranslation.locale == locale)
            .order_by(BotTranslation.locale, BotTranslation.translation_key)
        )
    ).all()
    stored = {(locale, key): value for locale, key, value in rows}
    entries = []
    for key in registry.keys():
        source = registry.get(key)
        if source is None:
            continue
        entries.append(
            {
                "locale": locale,
                "translation_key": key,
                "source_hash": source.source_hash,
                "text": stored.get((locale, key), ""),
            }
        )
    pack = {
        "schema_version": TRANSLATION_PACK_SCHEMA_VERSION,
        "scope": "system",
        "locale": locale,
        "translations": entries,
    }
    if target is not None:
        pack["target"] = target
    return pack


async def _acquire_translation_lock(session) -> None:
    bind = session.get_bind()
    dialect_name = getattr(getattr(bind, "dialect", None), "name", "")
    if dialect_name == "postgresql":
        await session.execute(text("SELECT pg_advisory_xact_lock(:lock_key)"), {"lock_key": 0x54474C4E})
        return
    await _translation_coordination_lock.acquire()


def _release_translation_lock(session) -> None:
    bind = session.get_bind()
    dialect_name = getattr(getattr(bind, "dialect", None), "name", "")
    if dialect_name != "postgresql" and _translation_coordination_lock.locked():
        _translation_coordination_lock.release()


@asynccontextmanager
async def translation_coordination_lock(session):
    if session.info.get("translation_lock_depth", 0):
        session.info["translation_lock_depth"] += 1
        try:
            yield
        finally:
            session.info["translation_lock_depth"] -= 1
        return
    dialect_name = getattr(getattr(session.get_bind(), "dialect", None), "name", "")
    acquired = False
    try:
        await _acquire_translation_lock(session)
        acquired = True
        session.info["translation_lock_depth"] = 1
        yield
    finally:
        session.info.pop("translation_lock_depth", None)
        if acquired and dialect_name != "postgresql":
            _release_translation_lock(session)


readiness_critical_resource_lock = translation_coordination_lock


_RESOURCE_TRANSLATION_FIELDS = {
    "BotGeneralConfig": {"ai_processing_message_text"},
    "Content": {"button_title", "text_content", "action_btn_text"},
    "Topic": {"name", "description", "start_message", "start_button_text"},
    "SubscriptionPlan": {"name", "description"},
    "SubscriptionConfig": {"topics_btn_name", "referral_btn_name", "referral_sub_btn_name"},
    "ReferralTemplate": {"text"},
    "FollowupStep": {"message_text"},
    "Mailing": {"text"},
    "AutomationAction": {"message_template"},
    "TestQuestion": {"text", "comment", "answer_options_json"},
    "SecretTestQuestion": {"text"},
}

_RESOURCE_ACTIVE_FIELDS = {
    "BotGeneralConfig": ("ai_processing_message_enabled",),
    "Content": ("is_visible",),
    "Topic": ("is_active", "show_in_list", "show_in_main_menu"),
    "SubscriptionPlan": ("is_active",),
    "ReferralTemplate": ("is_enabled",),
    "Mailing": ("is_enabled",),
    "FollowupCampaign": ("is_active",),
    "TestConfig": ("is_enabled", "secret_test_enabled"),
    "SubscriptionConfig": ("subscriptions_enabled", "topics_enabled", "referral_enabled"),
    "AutomationHandler": ("is_active",),
}


def _resource_has_translation_change(resource) -> bool:
    resource_name = type(resource).__name__
    state = sqlalchemy_inspect(resource)
    active_fields = _RESOURCE_ACTIVE_FIELDS.get(resource_name, ())
    if any(
        field in state.attrs and state.attrs[field].history.has_changes()
        for field in active_fields
    ):
        return True
    if resource_name not in _RESOURCE_TRANSLATION_FIELDS:
        return resource_name in _RESOURCE_ACTIVE_FIELDS
    if state.transient or state.pending:
        return True
    return any(
        state.attrs[field].history.has_changes()
        for field in _RESOURCE_TRANSLATION_FIELDS[resource_name]
        if field in state.attrs
    )


def _resource_is_new(session, resource) -> bool:
    return resource in session.new


def _resource_translation_prefixes(resource) -> tuple[str, ...]:
    resource_name = type(resource).__name__
    resource_id = getattr(resource, "id", None)
    if resource_name == "Content":
        return (f"content.{getattr(resource, 'key', '')}.",)
    if resource_name == "BotGeneralConfig":
        return (f"bot_general_config.{resource_id}.ai_processing_message_text",)
    if resource_name == "Topic":
        return (f"topic.{resource_id}.",)
    if resource_name == "SubscriptionPlan":
        return (f"plan.{resource_id}.",)
    if resource_name == "SubscriptionConfig":
        return (f"subscription_config.{resource_id}.",)
    if resource_name == "ReferralTemplate":
        return (f"referral_template.{resource_id}.",)
    if resource_name == "FollowupStep":
        return (f"followup_step.{resource_id}.",)
    if resource_name == "Mailing":
        return (f"mailing.{resource_id}.",)
    if resource_name == "AutomationAction":
        return (f"automation_action.{resource_id}.",)
    if resource_name == "TestQuestion":
        return (f"test_question.{resource_id}.",)
    if resource_name == "SecretTestQuestion":
        return (f"secret_test_question.{resource_id}.",)
    return ()


def _resource_has_required_sources(registry, resource) -> bool:
    prefixes = _resource_translation_prefixes(resource)
    if not prefixes:
        return type(resource).__name__ in {
            "FollowupCampaign",
            "TestConfig",
            "SubscriptionConfig",
            "AutomationHandler",
        }
    return any(
        source.required
        for key, source in registry.snapshot().items()
        if any(key.startswith(prefix) for prefix in prefixes)
    )


async def _deactivate_untranslated_new_resource(session, resource) -> None:
    resource_name = type(resource).__name__
    if resource_name == "Content" and getattr(resource, "key", None) == "start_message":
        raise TranslationResourceNotReady("start_message requires translations before activation")
    if resource_name == "Mailing" and getattr(resource, "recurring_type", None) is None:
        return
    active_fields = _RESOURCE_ACTIVE_FIELDS.get(resource_name, ())
    if active_fields:
        changed = False
        for field in active_fields:
            if getattr(resource, field, None) is not False:
                setattr(resource, field, False)
                changed = True
        if changed:
            return
    if resource_name in {"TestQuestion", "SecretTestQuestion"}:
        from database import TestConfig

        config = await session.get(TestConfig, 1)
        if config and config.is_enabled:
            config.is_enabled = False
        return
    if resource_name == "FollowupStep":
        from database import FollowupCampaign

        campaign = await session.get(FollowupCampaign, resource.campaign_id)
        if campaign and campaign.is_active:
            campaign.is_active = False
        return
    if resource_name == "AutomationAction":
        from database import AutomationHandler

        handler = await session.get(AutomationHandler, resource.handler_id)
        if handler and handler.is_active:
            handler.is_active = False


async def commit_readiness_critical_mutation(session) -> None:
    from content_authoring import bump_revision

    async with translation_coordination_lock(session):
        await session.flush()
        await bump_revision(session)
        await session.commit()


async def import_translation_pack(
    session_maker,
    pack: dict[str, Any],
    *,
    registry: TranslationRegistry | None = None,
    required_locales: tuple[str, ...] | None = None,
    expected_locale: str | None = None,
    expected_target: dict[str, Any] | None = None,
) -> int:
    from translation_registry import build_translation_registry

    async with session_maker() as session:
        async with translation_coordination_lock(session):
            active_registry = registry or await build_translation_registry(session)
            config = await session.get(BotGeneralConfig, 1)
            pack_locale = pack.get("locale") if isinstance(pack, dict) else None
            target_locale = expected_locale or pack_locale
            if required_locales is None:
                required_locales = (
                    (target_locale,)
                    if target_locale in TRANSLATION_PACK_LOCALES
                    else ()
                )
            elif target_locale in TRANSLATION_PACK_LOCALES and target_locale not in required_locales:
                required_locales = (*required_locales, target_locale)
            validate_translation_pack(
                pack,
                active_registry,
                required_locales=required_locales,
                expected_locale=expected_locale,
                expected_target=expected_target,
            )
            entries = [entry for entry in pack["translations"] if not is_admin_content_key(entry.get("translation_key"))]
            bind = session.get_bind()
            dialect_name = getattr(getattr(bind, "dialect", None), "name", "")
            if dialect_name == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as dialect_insert
            else:
                from sqlalchemy.dialects.sqlite import insert as dialect_insert

            if entries:
                values = [
                    {
                        "locale": entry["locale"],
                        "translation_key": entry["translation_key"],
                        "text": entry["text"],
                        "source_hash": entry["source_hash"],
                    }
                    for entry in entries
                ]
                statement = dialect_insert(BotTranslation).values(values)
                statement = statement.on_conflict_do_update(
                    index_elements=["locale", "translation_key"],
                    set_={
                        "text": statement.excluded.text,
                        "source_hash": statement.excluded.source_hash,
                        "updated_at": text("CURRENT_TIMESTAMP"),
                    },
                )
                await session.execute(statement)

            if config is None:
                config = BotGeneralConfig(id=1)
                session.add(config)
                await session.flush()
            config.translations_revision = int(config.translations_revision or 0) + 1
            await session.commit()

            source_snapshot = {
                key: source.source for key, source in active_registry.snapshot().items()
            }
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
            translation_snapshot = {
                (locale, key): value
                for locale, key, value, stored_hash in rows
                if (
                    locale in SUPPORTED_TELEGRAM_LOCALES
                    and locale != "ru"
                    and key in source_snapshot
                    and isinstance(value, str)
                    and value != ""
                    and (is_admin_content_key(key) or stored_hash == source_hash(source_snapshot[key]))
                    and (not is_admin_content_key(key) or dynamic_translation_safe(source_snapshot[key], value))
                )
            }
            translation_cache.install(
                config.translations_revision,
                source_snapshot,
                translation_snapshot,
            )
            return len(entries)


async def audit_translation_readiness(
    session,
    registry: TranslationRegistry,
    *,
    locales: tuple[str, ...],
) -> dict[str, Any]:
    full_registry = registry
    registry = registry.system()
    rows = (
        await session.execute(
            select(BotTranslation.locale, BotTranslation.translation_key, BotTranslation.text, BotTranslation.source_hash)
        )
    ).all()
    by_key = {(locale, key): (value, stored_hash) for locale, key, value, stored_hash in rows}
    locale_reports: dict[str, dict[str, Any]] = {}
    for locale in locales:
        if locale == "ru":
            locale_reports[locale] = get_locale_readiness({}, locale)
            continue
        missing: list[str] = []
        stale: list[str] = []
        invalid: list[str] = []
        translated = 0
        for key in registry.required_keys():
            source = registry.get(key)
            row = by_key.get((locale, key))
            if row is None or row[0] == "":
                missing.append(key)
                continue
            if not isinstance(row[0], str):
                invalid.append(key)
                continue
            if row[1] != source.source_hash:
                stale.append(key)
                continue
            try:
                validate_translation_value(source, row[0])
            except (TypeError, ValueError):
                invalid.append(key)
                continue
            translated += 1
        locale_reports[locale] = {
            "ready": not missing and not stale and not invalid,
            "canonical": False,
            "translated": translated,
            "required": len(registry.required_keys()),
            "missing": missing,
            "stale": stale,
            "invalid": invalid,
        }
    orphaned = sorted(
        f"{locale}/{key}"
        for locale, key in by_key
        if locale not in SUPPORTED_TELEGRAM_LOCALES or full_registry.get(key) is None
    )
    return {
        "ready": all(report["ready"] for report in locale_reports.values()),
        "locales": locale_reports,
        "orphaned": orphaned,
        "content": content_completeness(full_registry, by_key, locales),
    }


def content_completeness(registry, by_key, locales):
    primary = {"topic": "name", "plan": "name", "content": "text_content", "test_question": "text", "secret_test_question": "text", "mailing": "text", "automation_action": "message_template", "followup_step": "message_text", "referral_template": "text", "media_library": "description", "bot_general_config": "ai_processing_message_text", "case_study": "text"}
    groups = {}
    for key, source in registry.snapshot().items():
        if source.domain != "admin_content":
            continue
        parts = key.split(".")
        groups.setdefault(parts[0], {}).setdefault(parts[1], []).append(source)
    result = {}
    for locale in locales:
        report = {}
        for kind, resources in groups.items():
            complete = review = 0
            for sources in resources.values():
                required = [source for source in sources if source.source or source.translation_key.rsplit(".", 1)[-1] == primary.get(kind)]
                values = [(source, by_key.get((locale, source.translation_key))) for source in required]
                if values and all(row and row[0] for _, row in values):
                    complete += 1
                if any(row and row[0] and row[1] != source.source_hash for source, row in values):
                    review += 1
            report[kind] = {"complete": complete, "total": len(resources), "review": review}
        result[locale] = report
    return result
