import os

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database import Base, BotGeneralConfig, BotTranslation
from translation_pack_manager import (
    TranslationPackValidationError,
    audit_translation_readiness,
    export_translation_pack,
    import_translation_pack,
    validate_translation_pack,
)
from translation_registry import TranslationRegistry, TranslationSource
from translation_service import source_hash, translation_cache


def _registry():
    return TranslationRegistry([
        TranslationSource("ui.one", "One"),
        TranslationSource("ui.format", "Value: {value:>4}"),
    ])


def _valid_entry(registry, key, text):
    source = registry.get(key).source
    return {
        "locale": "en",
        "translation_key": key,
        "source_hash": source_hash(source),
        "text": text,
    }


def _single_locale_pack(registry, locale="en", entries=None):
    return {
        "schema_version": 2,
        "locale": locale,
        "translations": entries if entries is not None else [
            {
                **_valid_entry(registry, key, text),
                "locale": locale,
            }
            for key, text in (("ui.one", "One"), ("ui.format", "Value: {value:>4}"))
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", ("en", "pt"))
async def test_export_translation_pack_contains_exactly_one_locale(tmp_path, locale):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / f'export-{locale}.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    registry = _registry()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with sessions() as session:
            pack = await export_translation_pack(session, registry, locale=locale)

        assert pack["schema_version"] == 2
        assert pack["locale"] == locale
        assert {entry["locale"] for entry in pack["translations"]} == {locale}
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_export_translation_pack_can_bind_file_to_bot_and_database(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'bound-export.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with sessions() as session:
            pack = await export_translation_pack(
                session,
                _registry(),
                locale="en",
                target={"telegram_bot_id": 123, "database": "someone01"},
            )

        assert pack["target"] == {"telegram_bot_id": 123, "database": "someone01"}
    finally:
        await engine.dispose()


def test_pack_target_must_match_the_selected_bot_and_database():
    registry = _registry()
    pack = _single_locale_pack(registry)
    pack["target"] = {"telegram_bot_id": 123, "database": "someone01"}
    expected_target = {"telegram_bot_id": 123, "database": "someone01"}

    validate_translation_pack(
        pack,
        registry,
        required_locales=("en",),
        expected_locale="en",
        expected_target=expected_target,
    )

    for target in (
        {"telegram_bot_id": 456, "database": "someone01"},
        {"telegram_bot_id": 123, "database": "another_bot"},
        None,
    ):
        invalid = {**pack}
        if target is None:
            invalid.pop("target")
        else:
            invalid["target"] = target
        with pytest.raises(TranslationPackValidationError):
            validate_translation_pack(
                invalid,
                registry,
                required_locales=("en",),
                expected_locale="en",
                expected_target=expected_target,
            )


def test_translation_pack_errors_are_explained_in_russian():
    from translation_pack_manager import humanize_translation_pack_errors

    messages = humanize_translation_pack_errors(
        [
            "pack target database does not match",
            "stale source hash: ui.one",
            "invalid translation en/ui.format: placeholders differ",
            "missing required translation: en/ui.one",
        ],
        expected_locale="en",
    )

    assert any("другого бота" in message for message in messages)
    assert any("изменения русского текста" in message.lower() for message in messages)
    assert any("плейсхолдеры" in message.lower() for message in messages)
    assert any("обязательных переводов" in message.lower() for message in messages)


def test_translation_pack_rejects_mixed_ru_or_unexpected_locales():
    registry = _registry()
    valid_en = _valid_entry(registry, "ui.one", "One")
    valid_en["locale"] = "en"

    mixed = _single_locale_pack(registry, entries=[valid_en, {**valid_en, "translation_key": "ui.format", "locale": "pt", "text": "Valor: {value:>4}", "source_hash": source_hash("Value: {value:>4}")}])
    with pytest.raises(TranslationPackValidationError):
        validate_translation_pack(mixed, registry, required_locales=("en",), expected_locale="en")

    russian = _single_locale_pack(registry, locale="ru", entries=[{**valid_en, "locale": "ru"}])
    with pytest.raises(TranslationPackValidationError):
        validate_translation_pack(russian, registry, required_locales=("ru",), expected_locale="ru")

    unexpected = _single_locale_pack(registry, locale="fr", entries=[{**valid_en, "locale": "fr"}])
    with pytest.raises(TranslationPackValidationError):
        validate_translation_pack(unexpected, registry, required_locales=("fr",), expected_locale="fr")


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", ("en", "pt"))
async def test_import_translation_pack_isolated_to_expected_locale(tmp_path, locale):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / f'import-{locale}.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    registry = _registry()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with sessions() as session:
            session.add(
                BotTranslation(
                    locale="pt" if locale == "en" else "en",
                    translation_key="ui.one",
                    text="Texto existente" if locale == "en" else "Existing text",
                    source_hash=source_hash("One"),
                )
            )
            await session.commit()

        await import_translation_pack(
            sessions,
            _single_locale_pack(
                registry,
                locale=locale,
                entries=[
                    {**_valid_entry(registry, "ui.one", "One"), "locale": locale},
                    {**_valid_entry(registry, "ui.format", "Value: {value:>4}"), "locale": locale},
                ],
            ),
            registry=registry,
            expected_locale=locale,
        )

        async with sessions() as session:
            rows = (await session.execute(select(BotTranslation))).scalars().all()
            expected_existing_locale = "pt" if locale == "en" else "en"
            assert {(row.locale, row.translation_key, row.text) for row in rows} == {
                (locale, "ui.one", "One"),
                (locale, "ui.format", "Value: {value:>4}"),
                (
                    expected_existing_locale,
                    "ui.one",
                    "Texto existente" if locale == "en" else "Existing text",
                ),
            }
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_import_rejects_unexpected_locale_without_writes(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'import-mismatch.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    registry = _registry()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        with pytest.raises(TranslationPackValidationError):
            await import_translation_pack(
                sessions,
                _single_locale_pack(
                    registry,
                    locale="pt",
                    entries=[
                        {**_valid_entry(registry, "ui.one", "One"), "locale": "pt"},
                        {**_valid_entry(registry, "ui.format", "Value: {value:>4}"), "locale": "pt"},
                    ],
                ),
                registry=registry,
                expected_locale="en",
            )

        async with sessions() as session:
            assert await session.scalar(select(func.count()).select_from(BotTranslation)) == 0
            assert await session.get(BotGeneralConfig, 1) is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_import_rejects_wrong_bot_target_without_writes_or_revision_change(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'wrong-target.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    registry = _registry()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        pack = _single_locale_pack(registry)
        pack["target"] = {"telegram_bot_id": 123, "database": "someone01"}

        with pytest.raises(TranslationPackValidationError):
            await import_translation_pack(
                sessions,
                pack,
                registry=registry,
                expected_locale="en",
                expected_target={"telegram_bot_id": 456, "database": "someone01"},
            )

        async with sessions() as session:
            assert await session.scalar(select(func.count()).select_from(BotTranslation)) == 0
            assert await session.get(BotGeneralConfig, 1) is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pack_locale", "entry_locales"),
    (("ru", ("ru", "ru")), ("fr", ("fr", "fr")), ("en", ("en", "pt"))),
)
async def test_import_rejects_forbidden_or_mixed_pack_atomically(
    tmp_path,
    pack_locale,
    entry_locales,
):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / f'forbidden-{pack_locale}.db'}"
    )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    registry = _registry()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        entries = [
            {
                **_valid_entry(registry, key, text),
                "locale": entry_locale,
            }
            for (key, text), entry_locale in zip(
                (("ui.one", "One"), ("ui.format", "Value: {value:>4}")),
                entry_locales,
            )
        ]
        pack = {
            "schema_version": 2,
            "locale": pack_locale,
            "translations": entries,
        }
        with pytest.raises(TranslationPackValidationError):
            await import_translation_pack(
                sessions,
                pack,
                registry=registry,
                expected_locale=pack_locale if pack_locale in {"en", "pt"} else None,
            )

        async with sessions() as session:
            assert await session.scalar(select(func.count()).select_from(BotTranslation)) == 0
            assert await session.get(BotGeneralConfig, 1) is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_readiness_reports_each_locale_and_detects_invalid_rows(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'readiness.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    registry = _registry()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with sessions() as session:
            session.add_all([
                BotTranslation(
                    locale="en",
                    translation_key="ui.one",
                    text="One",
                    source_hash=source_hash("One"),
                ),
                BotTranslation(
                    locale="en",
                    translation_key="ui.format",
                    text="Value: {value}",
                    source_hash=source_hash("Value: {value:>4}"),
                ),
            ])
            await session.commit()

        async with sessions() as session:
            readiness = await audit_translation_readiness(
                session,
                registry,
                locales=("en", "pt"),
            )

        assert readiness["locales"]["en"]["ready"] is False
        assert readiness["locales"]["en"]["translated"] == 1
        assert readiness["locales"]["en"]["required"] == 2
        assert readiness["locales"]["en"]["invalid"] == ["ui.format"]
        assert readiness["locales"]["pt"]["ready"] is False
        assert readiness["locales"]["pt"]["translated"] == 0
        assert len(readiness["locales"]["pt"]["missing"]) == 2
        assert readiness["ready"] is False
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_empty_translation_table_marks_en_and_pt_not_ready(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'empty-readiness.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    registry = _registry()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with sessions() as session:
            readiness = await audit_translation_readiness(
                session,
                registry,
                locales=("en", "pt"),
            )

        assert readiness["locales"]["en"]["ready"] is False
        assert readiness["locales"]["pt"]["ready"] is False
        assert readiness["locales"]["en"]["translated"] == 0
        assert readiness["locales"]["pt"]["translated"] == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", ("en", "pt"))
async def test_complete_single_locale_readiness_is_independent(tmp_path, locale):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / f'ready-{locale}.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    registry = _registry()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with sessions() as session:
            session.add_all([
                BotTranslation(
                    locale=locale,
                    translation_key="ui.one",
                    text="One" if locale == "en" else "Um",
                    source_hash=source_hash("One"),
                ),
                BotTranslation(
                    locale=locale,
                    translation_key="ui.format",
                    text="Value: {value:>4}" if locale == "en" else "Valor: {value:>4}",
                    source_hash=source_hash("Value: {value:>4}"),
                ),
            ])
            await session.commit()

        async with sessions() as session:
            readiness = await audit_translation_readiness(
                session,
                registry,
                locales=("en", "pt"),
            )

        assert readiness["locales"][locale]["ready"] is True
        other = "pt" if locale == "en" else "en"
        assert readiness["locales"][other]["ready"] is False
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_invalid_pack_writes_zero_rows_and_does_not_advance_revision(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'packs.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    registry = _registry()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        pack = _single_locale_pack(
            registry,
            entries=[
                _valid_entry(registry, "ui.one", "Uno"),
                _valid_entry(registry, "ui.one", "Duplicate"),
            ],
        )
        with pytest.raises(TranslationPackValidationError):
            await import_translation_pack(sessions, pack, registry=registry, required_locales=("en",))

        async with sessions() as session:
            assert await session.scalar(select(func.count()).select_from(BotTranslation)) == 0
            assert await session.get(BotGeneralConfig, 1) is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_invalid_source_hash_writes_zero_rows_and_does_not_advance_revision(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'packs-hash.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    registry = _registry()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        invalid_entry = _valid_entry(registry, "ui.format", "Valor: {value:>4}")
        invalid_entry["source_hash"] = "invalid"
        pack = _single_locale_pack(
            registry,
            entries=[_valid_entry(registry, "ui.one", "Uno"), invalid_entry],
        )
        with pytest.raises(TranslationPackValidationError):
            await import_translation_pack(sessions, pack, registry=registry, required_locales=("en",))

        async with sessions() as session:
            assert await session.scalar(select(func.count()).select_from(BotTranslation)) == 0
            assert await session.get(BotGeneralConfig, 1) is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_valid_pack_upserts_and_advances_revision_once(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'packs-valid.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    registry = _registry()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        pack = _single_locale_pack(
            registry,
            entries=[
                _valid_entry(registry, "ui.one", "Uno"),
                _valid_entry(registry, "ui.format", "Valor: {value:>4}"),
            ],
        )
        assert await import_translation_pack(sessions, pack, registry=registry, required_locales=("en",)) == 2

        async with sessions() as session:
            config = await session.get(BotGeneralConfig, 1)
            assert config.translations_revision == 1
            rows = (await session.execute(select(BotTranslation))).scalars().all()
            assert {(row.locale, row.translation_key, row.text) for row in rows} == {
                ("en", "ui.one", "Uno"),
                ("en", "ui.format", "Valor: {value:>4}"),
            }
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_valid_import_does_not_install_stale_rows_in_cache(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'packs-cache.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    registry = _registry()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with sessions() as session:
            session.add(
                BotTranslation(
                    locale="pt",
                    translation_key="ui.one",
                    text="Texto antigo",
                    source_hash="stale",
                )
            )
            await session.commit()

        pack = _single_locale_pack(
            registry,
            entries=[
                _valid_entry(registry, "ui.one", "One pt"),
                _valid_entry(registry, "ui.format", "Value: {value:>4}"),
            ],
        )
        await import_translation_pack(sessions, pack, registry=registry, required_locales=("en",))

        assert translation_cache.get("ui.one", "pt") == "One"
        assert translation_cache.get("ui.one", "en") == "One pt"
    finally:
        await engine.dispose()
