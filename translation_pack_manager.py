from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from collections import Counter
import json
from typing import Any

from sqlalchemy import select, text, inspect as sqlalchemy_inspect

from database import BotGeneralConfig, BotTranslation
from translation_registry import TranslationRegistry, validate_translation_value
from translation_service import (
    SUPPORTED_TELEGRAM_LOCALES,
    normalize_enabled_languages,
    translation_cache,
    source_hash,
)


class TranslationPackValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(errors))


class TranslationResourceNotReady(ValueError):
    pass


_translation_coordination_lock = asyncio.Lock()


def validate_translation_pack(
    pack: dict[str, Any],
    registry: TranslationRegistry,
    *,
    required_locales: tuple[str, ...] = (),
) -> None:
    errors: list[str] = []
    if not isinstance(pack, dict):
        raise TranslationPackValidationError(["pack must be an object"])
    if pack.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    entries = pack.get("translations")
    if not isinstance(entries, list):
        errors.append("translations must be a list")
        raise TranslationPackValidationError(errors)

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
        if locale not in SUPPORTED_TELEGRAM_LOCALES:
            errors.append(f"unsupported locale: {locale}")
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

    for locale in required_locales:
        if locale not in SUPPORTED_TELEGRAM_LOCALES:
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
    locales: tuple[str, ...] = ("en", "pt"),
) -> dict[str, Any]:
    rows = (
        await session.execute(
            select(BotTranslation.locale, BotTranslation.translation_key, BotTranslation.text)
            .where(BotTranslation.locale.in_(locales))
            .order_by(BotTranslation.locale, BotTranslation.translation_key)
        )
    ).all()
    stored = {(locale, key): value for locale, key, value in rows}
    entries = []
    for locale in locales:
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
    return {"schema_version": 1, "translations": entries}


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
    dialect_name = getattr(getattr(session.get_bind(), "dialect", None), "name", "")
    acquired = False
    try:
        await _acquire_translation_lock(session)
        acquired = True
        yield
    finally:
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
    from translation_registry import build_translation_registry

    async with translation_coordination_lock(session):
        config = await session.get(BotGeneralConfig, 1)
        required_locales = tuple(
            locale
            for locale in normalize_enabled_languages(
                getattr(config, "telegram_enabled_languages", '["ru"]') if config else '["ru"]'
            )
            if locale != "ru"
        )
        changed_resources = [
            resource
            for resource in tuple(session.new) + tuple(session.dirty)
            if _resource_has_translation_change(resource)
        ]
        if required_locales and changed_resources:
            resource_flags = {
                id(resource): (
                    _resource_is_new(session, resource),
                    type(resource).__name__ in _RESOURCE_TRANSLATION_FIELDS
                    and (
                        _resource_is_new(session, resource)
                        or any(
                            sqlalchemy_inspect(resource).attrs[field].history.has_changes()
                            for field in _RESOURCE_TRANSLATION_FIELDS[type(resource).__name__]
                            if field in sqlalchemy_inspect(resource).attrs
                        )
                    ),
                    any(
                        field in sqlalchemy_inspect(resource).attrs
                        and sqlalchemy_inspect(resource).attrs[field].history.has_changes()
                        and bool(getattr(resource, field, False))
                        for field in _RESOURCE_ACTIVE_FIELDS.get(type(resource).__name__, ())
                    ),
                )
                for resource in changed_resources
            }
            for resource in tuple(changed_resources):
                if _resource_is_new(session, resource):
                    await _deactivate_untranslated_new_resource(session, resource)
            await session.flush()
            registry = await build_translation_registry(session)
            readiness = await audit_translation_readiness(
                session,
                registry,
                locales=required_locales,
            )
            if not readiness["ready"]:
                persistent_source_change = any(
                    not resource_flags[id(resource)][0]
                    and resource_flags[id(resource)][1]
                    and _resource_has_required_sources(registry, resource)
                    for resource in changed_resources
                )
                active_resource_change = any(
                    not resource_flags[id(resource)][0]
                    and resource_flags[id(resource)][2]
                    and _resource_has_required_sources(registry, resource)
                    for resource in changed_resources
                )
                if persistent_source_change or active_resource_change:
                    await session.rollback()
                    details = []
                    for locale in required_locales:
                        report = readiness.get(locale, {})
                        details.extend(report.get("missing", [])[:3])
                        details.extend(report.get("stale", [])[:3])
                    raise TranslationResourceNotReady(
                        "Translations are required before activation: " + ", ".join(details)
                    )
        await session.commit()


async def import_translation_pack(
    session_maker,
    pack: dict[str, Any],
    *,
    registry: TranslationRegistry | None = None,
    required_locales: tuple[str, ...] | None = None,
) -> int:
    from translation_registry import build_translation_registry

    async with session_maker() as session:
        async with translation_coordination_lock(session):
            active_registry = registry or await build_translation_registry(session)
            config = await session.get(BotGeneralConfig, 1)
            if required_locales is None:
                required_locales = tuple(
                    locale
                    for locale in normalize_enabled_languages(
                        getattr(config, "telegram_enabled_languages", '["ru"]')
                        if config
                        else '["ru"]'
                    )
                    if locale != "ru"
                )
            validate_translation_pack(
                pack,
                active_registry,
                required_locales=required_locales,
            )
            entries = pack["translations"]
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
                    and stored_hash == source_hash(source_snapshot[key])
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
    rows = (
        await session.execute(
            select(BotTranslation.locale, BotTranslation.translation_key, BotTranslation.text, BotTranslation.source_hash)
        )
    ).all()
    by_key = {(locale, key): (value, stored_hash) for locale, key, value, stored_hash in rows}
    missing: dict[str, list[str]] = {}
    stale: dict[str, list[str]] = {}
    for locale in locales:
        if locale == "ru":
            continue
        for key in registry.required_keys():
            row = by_key.get((locale, key))
            if row is None:
                missing.setdefault(locale, []).append(key)
            elif row[1] != registry.get(key).source_hash:
                stale.setdefault(locale, []).append(key)
    orphaned = sorted(
        f"{locale}/{key}"
        for locale, key in by_key
        if locale not in SUPPORTED_TELEGRAM_LOCALES or registry.get(key) is None
    )
    return {
        "ready": not missing and not stale,
        "missing": missing,
        "stale": stale,
        "orphaned": orphaned,
    }
