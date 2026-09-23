from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import func, select

from content_locales import ContentValue
from database import (
    AdminContentPreference, AutomationAction, BotGeneralConfig, BotTranslation,
    Content, ContentIdentityCounter, FollowupStep, Mailing, MediaLibrary, CaseStudy,
    ReferralTemplate, SecretTestQuestion, SubscriptionConfig, SubscriptionPlan,
    TestQuestion, Topic,
)
from translation_registry import TranslationSource, _decode_ai_processing_text, validate_translation_value
from translation_service import SUPPORTED_TELEGRAM_LOCALES, source_hash
from universal_tests import get_answer_options, json_dumps


@dataclass(frozen=True)
class AuthoringResource:
    model: type
    title: str
    fields: tuple[tuple[str, str, str], ...]
    identity_field: str = "id"


RESOURCES = {
    "topic": AuthoringResource(Topic, "Темы", (("name", "Название", "reply_button"), ("description", "Описание", "text"), ("start_message", "Стартовое сообщение", "html"), ("start_button_text", "Кнопка старта", "inline_button"))),
    "content": AuthoringResource(Content, "Контент", (("button_title", "Название кнопки", "reply_button"), ("text_content", "Текст", "html"), ("action_btn_text", "Кнопка действия", "inline_button")), "key"),
    "plan": AuthoringResource(SubscriptionPlan, "Тарифы", (("name", "Название", "text"), ("description", "Описание", "text"))),
    "test_question": AuthoringResource(TestQuestion, "Вопросы теста", (("text", "Вопрос", "html"), ("comment", "Пояснение", "html"))),
    "secret_test_question": AuthoringResource(SecretTestQuestion, "Секретные вопросы", (("text", "Вопрос", "html"),)),
    "mailing": AuthoringResource(Mailing, "Рассылки", (("text", "Текст", "html"),)),
    "automation_action": AuthoringResource(AutomationAction, "Сообщения автоматизаций", (("message_template", "Сообщение", "html"),)),
    "followup_step": AuthoringResource(FollowupStep, "Сообщения follow-up", (("message_text", "Сообщение", "html"),)),
    "referral_template": AuthoringResource(ReferralTemplate, "Реферальные сообщения", (("text", "Текст", "html"),)),
    "subscription_config": AuthoringResource(SubscriptionConfig, "Названия кнопок", (("topics_btn_name", "Темы", "reply_button"), ("referral_btn_name", "Пригласить друзей", "reply_button"), ("referral_sub_btn_name", "Бонус за приглашение", "reply_button"))),
    "bot_general_config": AuthoringResource(BotGeneralConfig, "Сообщение ожидания", (("ai_processing_message_text", "Думаю…", "text"),)),
    "media_library": AuthoringResource(MediaLibrary, "Подписи медиа", (("description", "Подпись", "caption"),)),
    "case_study": AuthoringResource(CaseStudy, "Истории и кейсы", (("text", "Текст истории", "text"),)),
}

AUTHORING_RESOURCES = {
    "topic": AuthoringResource(
        Topic,
        "Темы",
        (("name", "Название", "reply_button"), ("description", "Описание", "text"),
         ("start_message", "Стартовое сообщение", "html"),
         ("start_button_text", "Кнопка старта", "inline_button")),
    ),
    "content": AuthoringResource(
        Content,
        "Контент",
        (("button_title", "Название кнопки", "reply_button"),
         ("text_content", "Текст", "html"),
         ("action_btn_text", "Кнопка действия", "inline_button")),
        "key",
    ),
    "plan": AuthoringResource(
        SubscriptionPlan,
        "Тарифы",
        (("name", "Название", "text"), ("description", "Описание", "text")),
    ),
    "referral_template": AuthoringResource(
        ReferralTemplate,
        "Реферальные сообщения",
        (("text", "Текст", "text"),),
    ),
    "subscription_config": AuthoringResource(
        SubscriptionConfig,
        "Кнопки меню",
        (("topics_btn_name", "Кнопка тем", "reply_button"),
         ("referral_btn_name", "Кнопка рефералов", "reply_button"),
         ("referral_sub_btn_name", "Кнопка бонуса", "reply_button")),
    ),
    "media_library": AuthoringResource(
        MediaLibrary,
        "Медиаматериалы",
        (("description", "Подпись", "caption"),),
    ),
}


async def multilingual_authoring_enabled(session) -> bool:
    config = await session.get(BotGeneralConfig, 1)
    return bool(getattr(config, "multilingual_authoring_enabled", False))


async def authoring_locales(session) -> tuple[str, ...]:
    if not await multilingual_authoring_enabled(session):
        return ("ru",)
    config = await session.get(BotGeneralConfig, 1)
    try:
        enabled = tuple(
            item
            for item in json.loads(getattr(config, "telegram_enabled_languages", None) or "[\"ru\"]")
            if isinstance(item, str)
        )
    except (TypeError, ValueError):
        enabled = ("ru",)
    locales = tuple(locale for locale in SUPPORTED_TELEGRAM_LOCALES if locale in enabled)
    return locales or ("ru",)


async def editing_locale(session, bot_id: int, admin_id: int) -> str:
    preference = await session.get(AdminContentPreference, (bot_id, admin_id))
    locale = preference.content_locale if preference else "ru"
    return locale if locale in SUPPORTED_TELEGRAM_LOCALES else "ru"


async def set_editing_locale(session, bot_id: int, admin_id: int, locale: str) -> None:
    if locale not in SUPPORTED_TELEGRAM_LOCALES:
        raise ValueError("Язык не поддерживается.")
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    insert = pg_insert if session.get_bind().dialect.name == "postgresql" else sqlite_insert
    statement = insert(AdminContentPreference).values(bot_id=bot_id, admin_id=admin_id, content_locale=locale)
    await session.execute(statement.on_conflict_do_update(
        index_elements=["bot_id", "admin_id"], set_={"content_locale": locale},
    ))


def resource_key(kind: str, resource, field: str) -> str:
    return f"{kind}.{getattr(resource, RESOURCES[kind].identity_field)}.{field}"


def content_media_value(resource) -> str:
    return json.dumps(
        [
            {"type": media.file_type, "file_id": media.file_id}
            for media in (getattr(resource, "media", None) or ())
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_content_media_value(value: str | None) -> list[dict[str, str]] | None:
    if not value:
        return None
    try:
        items = json.loads(value)
    except (TypeError, ValueError):
        return None
    if not isinstance(items, list):
        return None
    result = []
    for item in items:
        if not isinstance(item, dict) or item.get("type") not in {"photo", "video"} or not item.get("file_id"):
            return None
        result.append({"type": str(item["type"]), "file_id": str(item["file_id"])})
    return result


def field_source(kind: str, resource, field: str) -> TranslationSource:
    if kind == "automation_action" and (resource.action_type != "send_message" or "admin" in (resource.recipient_type or "")):
        raise ValueError("Сообщение администраторам редактируется по-русски в настройках автоматизации.")
    if kind == "followup_step" and resource.message_type != "static":
        raise ValueError("AI-инструкция является общей настройкой.")
    if kind == "content" and field == "media":
        from sqlalchemy import inspect
        if "media" in inspect(resource).unloaded:
            media_source = ""
        else:
            media_source = content_media_value(resource)
        return TranslationSource(resource_key(kind, resource, field), media_source, kind="text")
    for name, _, value_kind in RESOURCES[kind].fields:
        if field == name:
            value = getattr(resource, field) or ""
            if kind == "bot_general_config":
                value = _decode_ai_processing_text(value)
            return TranslationSource(resource_key(kind, resource, field), value, kind=value_kind)
    if kind == "test_question" and field.startswith("option."):
        _, slot, name = field.split(".")
        if name not in {"text", "button_text"}:
            raise ValueError("Неизвестное поле ответа.")
        for option in get_answer_options(resource):
            if option.translation_slot == slot:
                return TranslationSource(resource_key(kind, resource, field), getattr(option, name) or "", kind="inline_button" if name == "button_text" else "text")
    raise ValueError("Поле не поддерживается.")


async def read_content_value(session, kind: str, resource, field: str, locale: str) -> ContentValue:
    source = field_source(kind, resource, field)
    if locale == "ru":
        return ContentValue(source.source or None, source.source or None)
    row = await session.scalar(select(BotTranslation).where(
        BotTranslation.locale == locale, BotTranslation.translation_key == source.translation_key,
    ))
    return ContentValue(row.text if row else None, source.source or None, bool(row and row.text and row.source_hash != source.source_hash))


async def admin_value(session, kind: str, resource, field: str, *, label=False, locale=None):
    from admin_authoring_context import content_editing_locale
    locale = locale or content_editing_locale.get() or "ru"
    value = await read_content_value(session, kind, resource, field, locale)
    return value.admin_label() if label else value.text or ""


async def admin_projection(session, kind, resource, *, locale=None):
    from types import SimpleNamespace
    values = {key: value for key, value in vars(resource).items() if not key.startswith("_")}
    for field, _, _ in RESOURCES[kind].fields:
        values[field] = await admin_value(session, kind, resource, field, label=True, locale=locale)
    return SimpleNamespace(**values)


async def bump_revision(session) -> None:
    session.info["authoring_revision_bumped"] = True
    from sqlalchemy import update
    config = await session.get(BotGeneralConfig, 1)
    if config is None:
        session.add(BotGeneralConfig(id=1, translations_revision=1))
        await session.flush()
    else:
        await session.execute(update(BotGeneralConfig).where(BotGeneralConfig.id == 1).values(
            translations_revision=BotGeneralConfig.translations_revision + 1,
        ))


async def save_content_value(session, kind: str, resource, field: str, locale: str, value: str) -> None:
    if not session.info.get("authoring_flush"):
        session.info["explicit_authoring"] = True
    if locale not in SUPPORTED_TELEGRAM_LOCALES:
        raise ValueError("Язык не поддерживается.")
    source = field_source(kind, resource, field)
    value = value.strip()
    stored_value = value
    if kind == "bot_general_config" and locale == "ru":
        value = _decode_ai_processing_text(value)
    validation_source = source if source.source and locale != "ru" else TranslationSource(source.translation_key, value, kind=source.kind)
    if not source.source:
        existing = await session.scalar(select(BotTranslation.text).where(BotTranslation.translation_key == source.translation_key, BotTranslation.text != "").order_by(BotTranslation.created_at, BotTranslation.locale).limit(1))
        if existing:
            validation_source = TranslationSource(source.translation_key, existing, kind=source.kind)
    if value and not (kind == "content" and field == "media"):
        try:
            validate_translation_value(validation_source, value)
        except (ValueError, TypeError) as exc:
            raise ValueError("Текст не сохранён: проверьте длину, форматирование, переменные и адреса кнопок.") from exc
    if kind == "bot_general_config" and len(value) > 200:
        raise ValueError("Максимум 200 символов.")
    if kind == "content" and field == "media" and value and parse_content_media_value(value) is None:
        raise ValueError("Медиафайлы имеют неверный формат.")
    if source.kind == "reply_button" and value:
        await validate_button_label(session, source.translation_key, locale, value)
    if locale == "ru":
        if field.startswith("option."):
            await ensure_answer_identities(session, resource)
            _, slot, name = field.split(".")
            items = json.loads(resource.answer_options_json)
            for item in items:
                if item["translation_slot"] == slot:
                    item[name] = value
            resource.answer_options_json = json_dumps(items)
        elif kind == "content" and field == "media":
            from database import ContentMedia
            resource.media.clear()
            for media in parse_content_media_value(value) or []:
                resource.media.append(ContentMedia(file_type=media["type"], file_id=media["file_id"]))
        else:
            setattr(resource, field, stored_value)
    else:
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        insert = pg_insert if session.get_bind().dialect.name == "postgresql" else sqlite_insert
        statement = insert(BotTranslation).values(locale=locale, translation_key=source.translation_key, text=value, source_hash=source.source_hash)
        await session.execute(statement.on_conflict_do_update(
            index_elements=["locale", "translation_key"],
            set_={"text": value, "source_hash": source.source_hash, "updated_at": func.current_timestamp()},
        ))
    await bump_revision(session)


async def validate_button_label(session, key: str, locale: str, value: str) -> None:
    from database import UserMenuBinding
    from translation_registry import build_translation_registry
    registry = await build_translation_registry(session)
    translations = dict((await session.execute(select(BotTranslation.translation_key, BotTranslation.text).where(BotTranslation.locale == locale))).all())
    for other_key, source in registry.snapshot().items():
        if other_key != key and source.kind == "reply_button" and (translations.get(other_key) or source.source) == value:
            raise ValueError("Такое название уже используется другой кнопкой. Выберите другое название.")
    kind, identity, _ = key.split(".", 2)
    if kind in {"topic", "content"}:
        old_targets = (await session.scalars(select(UserMenuBinding).where(UserMenuBinding.label == value))).all()
        if any(row.resource_kind != kind or row.resource_id != identity for row in old_targets):
            raise ValueError("Это название было выдано пользователям для другого материала. Выберите новое название.")


async def allocate_identity(session, namespace: str, minimum: int = 0) -> int:
    counter = await session.get(ContentIdentityCounter, namespace)
    if counter is None:
        counter = ContentIdentityCounter(namespace=namespace, next_id=minimum)
        session.add(counter)
    result = max(counter.next_id, minimum)
    counter.next_id = result + 1
    await session.flush()
    return result


async def ensure_answer_identities(session, question) -> list[dict]:
    options = get_answer_options(question)
    items = []
    for index, option in enumerate(options):
        items.append({
            "text": option.text, "button_text": option.button_text, "value": option.value,
            "identity": option.identity or uuid4().hex,
            "translation_slot": option.translation_slot or str(index),
            "callback_id": option.callback_id if option.callback_id is not None else index,
        })
    question.answer_options_json = json_dumps(items)
    return items


async def insert_answer(session, question, position: int, *, value: float | None = None) -> dict:
    from test_content_identity import preserve_active_test_definitions
    await preserve_active_test_definitions(session)
    items = await ensure_answer_identities(session, question)
    minimum = max((item["callback_id"] for item in items), default=-1) + 1
    callback_id = await allocate_identity(session, f"test_question.{question.id}.answers", minimum)
    item = {"identity": uuid4().hex, "translation_slot": "a" + uuid4().hex, "callback_id": callback_id, "text": "", "button_text": None, "value": value}
    items.insert(position, item)
    question.answer_options_json = json_dumps(items)
    return item


async def reorder_answers(session, question, identities: list[str]) -> None:
    from test_content_identity import preserve_active_test_definitions
    await preserve_active_test_definitions(session)
    items = await ensure_answer_identities(session, question)
    by_id = {item["identity"]: item for item in items}
    if len(identities) != len(set(identities)) or set(identities) != set(by_id):
        raise ValueError("Список ответов изменился. Откройте вопрос заново.")
    question.answer_options_json = json_dumps([by_id[identity] for identity in identities])


async def delete_answer(session, question, identity: str) -> None:
    from test_content_identity import preserve_active_test_definitions
    await preserve_active_test_definitions(session)
    items = await ensure_answer_identities(session, question)
    minimum = max((item["callback_id"] for item in items), default=-1) + 1
    await allocate_identity(session, f"test_question.{question.id}.answers", minimum)
    question.answer_options_json = json_dumps([item for item in items if item["identity"] != identity])


async def create_resource(session, kind: str, locale: str, values: dict, shared: dict | None = None):
    session.info["explicit_authoring"] = True
    if kind == "test_question":
        from test_content_identity import preserve_active_test_definitions
        await preserve_active_test_definitions(session)
    spec = RESOURCES[kind]
    defaults = {
        "topic": {"name": ""},
        "content": {"key": "material_" + uuid4().hex},
        "plan": {"name": "", "price": 0, "duration_value": 1, "duration_unit": "days", "is_active": False},
        "test_question": {"text": "", "category": "custom", "variable_name": "answer_" + uuid4().hex},
        "secret_test_question": {"text": ""},
        "referral_template": {"text": ""},
        "case_study": {"text": ""},
        "mailing": {"target_audience": "all", "status": "draft", "is_enabled": False},
    }
    if kind not in defaults:
        raise ValueError("Добавьте объект в соответствующем разделе настроек.")
    kwargs = defaults[kind] | (shared or {})
    if spec.identity_field == "id" and session.get_bind().dialect.name != "postgresql":
        maximum = await session.scalar(select(func.max(spec.model.id))) or 0
        keys = (await session.scalars(select(BotTranslation.translation_key).where(BotTranslation.translation_key.like(kind + ".%")))).all()
        maximum = max([maximum] + [int(key.split(".")[1]) for key in keys if key.split(".")[1].isdigit()])
        kwargs["id"] = await allocate_identity(session, kind, maximum + 1)
    resource = spec.model(**kwargs)
    session.add(resource)
    await session.flush()
    for field, value in values.items():
        await save_content_value(session, kind, resource, field, locale, value)
    return resource


async def localize_legacy_mutations(session) -> None:
    from admin_authoring_context import content_editing_locale
    from sqlalchemy import inspect
    locale = content_editing_locale.get()
    if not locale or locale == "ru" or session.info.get("explicit_authoring"):
        return
    pending = []
    with session.no_autoflush:
        for resource in tuple(session.new) + tuple(session.dirty):
            kind = next((kind for kind, spec in RESOURCES.items() if isinstance(resource, spec.model)), None)
            if kind is None:
                continue
            if kind == "automation_action" and (resource.action_type != "send_message" or "admin" in (resource.recipient_type or "")):
                continue
            if kind == "followup_step" and resource.message_type != "static":
                continue
            for field, _, _ in RESOURCES[kind].fields:
                history = inspect(resource).attrs[field].history
                if not history.has_changes():
                    continue
                value = getattr(resource, field)
                if value is None:
                    value = ""
                if kind == "bot_general_config":
                    value = _decode_ai_processing_text(value)
                original = history.deleted[0] if history.deleted else ""
                setattr(resource, field, original)
                pending.append((kind, resource, field, value))
        await session.flush()
        for kind, resource, field, value in pending:
            await save_content_value(session, kind, resource, field, locale, value)
