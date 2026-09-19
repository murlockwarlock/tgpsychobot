import ast
import os
from pathlib import Path

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database import Base, BotGeneralConfig, Mailing, TestConfig as DBTestConfig, Topic
from response_buttons import extract_response_buttons
from translation_registry import (
    STATIC_TRANSLATION_SOURCES,
    TranslationRegistry,
    TranslationSource,
    build_translation_registry,
)
from translation_pack_manager import TranslationPackValidationError, validate_translation_pack
from translation_service import source_hash


def test_registry_exposes_stable_sources_and_hashes():
    registry = TranslationRegistry([
        TranslationSource("ui.greeting", "Hello {name}"),
        TranslationSource("ui.buttons", "[Open](btn:open)"),
    ])

    snapshot = registry.snapshot()

    assert snapshot["ui.greeting"].source_hash == source_hash("Hello {name}")
    assert tuple(registry.keys()) == ("ui.buttons", "ui.greeting")


def test_user_action_error_is_registered_in_the_static_namespace():
    assert "ui.action.unassigned" in STATIC_TRANSLATION_SOURCES


def test_literal_translation_keys_use_the_canonical_namespace():
    literal_keys = set()
    project_root = Path(__file__).resolve().parents[1]
    for path in project_root.rglob("*.py"):
        if "tests" in path.parts or ".git" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"translate", "tr"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                literal_keys.add(node.args[0].value)

    dynamic_prefixes = (
        "bot_general_config.",
        "content.",
        "topic.",
        "plan.",
        "subscription_config.",
        "referral_template.",
        "followup_step.",
        "mailing.",
        "automation_action.",
        "test_question.",
        "secret_test_question.",
    )
    assert sorted(
        key
        for key in literal_keys
        if key not in STATIC_TRANSLATION_SOURCES
        and not key.startswith(dynamic_prefixes)
    ) == []


def test_pack_validation_allows_label_changes_but_preserves_targets():
    registry = TranslationRegistry([
        TranslationSource("ui.buttons", "[Open](btn:open) | [Docs](https://example.com/docs)"),
    ])
    pack = {
        "schema_version": 2,
        "locale": "en",
        "translations": [{
            "locale": "en",
            "translation_key": "ui.buttons",
            "text": "[Open now](btn:open) | [Documentation](https://example.com/docs)",
            "source_hash": source_hash("[Open](btn:open) | [Docs](https://example.com/docs)"),
        }],
    }

    validate_translation_pack(pack, registry, required_locales=("en",))


def test_pack_validation_rejects_target_changes_without_writes():
    registry = TranslationRegistry([
        TranslationSource("ui.buttons", "[Open](btn:open)"),
    ])
    pack = {
        "schema_version": 2,
        "locale": "en",
        "translations": [{
            "locale": "en",
            "translation_key": "ui.buttons",
            "text": "[Open](btn:closed)",
            "source_hash": source_hash("[Open](btn:open)"),
        }],
    }

    with pytest.raises(TranslationPackValidationError):
        validate_translation_pack(pack, registry, required_locales=("en",))


def test_pack_validation_rejects_incomplete_enabled_locale():
    registry = TranslationRegistry([
        TranslationSource("ui.one", "One"),
        TranslationSource("ui.two", "Two"),
    ])
    pack = {
        "schema_version": 2,
        "locale": "en",
        "translations": [{
            "locale": "en",
            "translation_key": "ui.one",
            "text": "Uno",
            "source_hash": source_hash("One"),
        }],
    }

    with pytest.raises(TranslationPackValidationError) as exc_info:
        validate_translation_pack(pack, registry, required_locales=("en",))

    assert "ui.two" in str(exc_info.value)


def test_pack_validation_rejects_empty_required_translation():
    registry = TranslationRegistry([TranslationSource("ui.one", "One")])
    pack = {
        "schema_version": 2,
        "locale": "en",
        "translations": [{
            "locale": "en",
            "translation_key": "ui.one",
            "text": "",
            "source_hash": source_hash("One"),
        }],
    }

    with pytest.raises(TranslationPackValidationError) as exc_info:
        validate_translation_pack(pack, registry, required_locales=("en",))

    assert "empty required translation" in str(exc_info.value)


def test_pack_validation_rejects_html_link_target_changes():
    registry = TranslationRegistry([
        TranslationSource("ui.link", '<a href="https://example.com">Открыть</a>')
    ])
    pack = {
        "schema_version": 2,
        "locale": "en",
        "translations": [{
            "locale": "en",
            "translation_key": "ui.link",
            "text": '<a href="https://example.org">Open</a>',
            "source_hash": source_hash('<a href="https://example.com">Открыть</a>'),
        }],
    }

    with pytest.raises(TranslationPackValidationError):
        validate_translation_pack(pack, registry, required_locales=("en",))


def test_pack_validation_enforces_telegram_text_and_caption_limits():
    message_registry = TranslationRegistry([TranslationSource("ui.message", "Текст")])
    message_pack = {
        "schema_version": 2,
        "locale": "en",
        "translations": [{
            "locale": "en",
            "translation_key": "ui.message",
            "text": "x" * 4097,
            "source_hash": source_hash("Текст"),
        }],
    }
    with pytest.raises(TranslationPackValidationError):
        validate_translation_pack(message_pack, message_registry, required_locales=("en",))

    caption_registry = TranslationRegistry([
        TranslationSource("ui.caption", "Подпись", kind="caption"),
    ])
    caption_pack = {
        "schema_version": 2,
        "locale": "en",
        "translations": [{
            "locale": "en",
            "translation_key": "ui.caption",
            "text": "x" * 1025,
            "source_hash": source_hash("Подпись"),
        }],
    }
    with pytest.raises(TranslationPackValidationError):
        validate_translation_pack(caption_pack, caption_registry, required_locales=("en",))


@pytest.mark.asyncio
async def test_one_off_mailings_use_the_explicit_russian_fallback_policy(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'mailings.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with sessions() as session:
            session.add(BotGeneralConfig(id=1, telegram_enabled_languages='["ru", "en"]'))
            one_off = Mailing(
                text="Одноразовое сообщение",
                target_audience="all",
                recurring_type=None,
                is_enabled=True,
                status="pending",
            )
            birthday = Mailing(
                text="Поздравление",
                target_audience="birthday_today",
                recurring_type="birthday",
                is_enabled=True,
                status="active",
            )
            session.add_all([one_off, birthday])
            await session.flush()
            registry = await build_translation_registry(session)

            assert registry.get(f"mailing.{one_off.id}.text").required is False
            assert registry.get(f"mailing.{birthday.id}.text").required is True
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_internal_prompts_are_not_translation_sources(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'internal-prompts.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with sessions() as session:
            session.add_all([
                DBTestConfig(
                    id=1,
                    is_enabled=True,
                    test_system_prompt="internal test prompt",
                    result_system_prompt="internal result prompt",
                ),
                Topic(
                    name="Тема",
                    is_active=True,
                    system_prompt="internal topic prompt",
                ),
            ])
            await session.flush()
            registry = await build_translation_registry(session)

            assert not any("system_prompt" in key for key in registry.keys())
            assert all(
                source.source not in {
                    "internal test prompt",
                    "internal result prompt",
                    "internal topic prompt",
                }
                for source in registry.snapshot().values()
            )
    finally:
        await engine.dispose()
