import os

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database import Base, BotGeneralConfig, BotTranslation
from translation_pack_manager import TranslationPackValidationError, import_translation_pack
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


@pytest.mark.asyncio
async def test_invalid_pack_writes_zero_rows_and_does_not_advance_revision(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'packs.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    registry = _registry()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        pack = {
            "schema_version": 1,
            "translations": [
                _valid_entry(registry, "ui.one", "Uno"),
                _valid_entry(registry, "ui.one", "Duplicate"),
            ],
        }
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
        pack = {
            "schema_version": 1,
            "translations": [
                _valid_entry(registry, "ui.one", "Uno"),
                invalid_entry,
            ],
        }
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

        pack = {
            "schema_version": 1,
            "translations": [
                _valid_entry(registry, "ui.one", "Uno"),
                _valid_entry(registry, "ui.format", "Valor: {value:>4}"),
            ],
        }
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

        pack = {
            "schema_version": 1,
            "translations": [
                _valid_entry(registry, "ui.one", "One pt"),
                _valid_entry(registry, "ui.format", "Value: {value:>4}"),
            ],
        }
        await import_translation_pack(sessions, pack, registry=registry, required_locales=("en",))

        assert translation_cache.get("ui.one", "pt") == "One"
        assert translation_cache.get("ui.one", "en") == "One pt"
    finally:
        await engine.dispose()
