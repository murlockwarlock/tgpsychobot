import io
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import handlers
import translation_registry
from database import Base, BotGeneralConfig, BotTranslation, User
from translation_registry import TranslationRegistry, TranslationSource
from translation_service import source_hash, translation_cache


def _registry(source="Main menu"):
    return TranslationRegistry([TranslationSource("ui.menu", source)])


def _pack(registry, target, *, locale="en", text="Main menu", source=None):
    source_text = source or registry.get("ui.menu").source
    return {
        "schema_version": 2,
        "locale": locale,
        "target": target,
        "translations": [
            {
                "locale": locale,
                "translation_key": "ui.menu",
                "source_hash": source_hash(source_text),
                "text": text,
            }
        ],
    }


class _State:
    def __init__(self):
        self.data = {}
        self.current = None

    async def set_state(self, value):
        self.current = value

    async def update_data(self, **values):
        self.data.update(values)

    async def get_data(self):
        return dict(self.data)

    async def get_state(self):
        return getattr(self.current, "state", self.current)

    async def clear(self):
        self.current = None
        self.data.clear()


class _Bot:
    def __init__(self, pack, *, bot_id=12345, username="example_bot"):
        self.pack = pack
        self.bot_info = SimpleNamespace(
            id=bot_id,
            username=username,
            first_name="Example Bot",
        )

    async def get_me(self):
        return self.bot_info

    async def get_file(self, file_id):
        return SimpleNamespace(file_path="translation.json")

    async def download_file(self, file_path):
        return io.BytesIO(json.dumps(self.pack).encode())


class _Message:
    def __init__(self, bot, *, text=None, document=True):
        self.from_user = SimpleNamespace(id=9001)
        self.bot = bot
        self.text = text
        self.document = (
            SimpleNamespace(file_id="file-id", file_name="pack.json", file_size=1024)
            if document
            else None
        )
        self.answer = AsyncMock()
        self.answer_document = AsyncMock()
        self.edit_text = AsyncMock()


def _callback(bot, message, data):
    return SimpleNamespace(
        from_user=SimpleNamespace(id=9001),
        data=data,
        bot=bot,
        message=message,
        answer=AsyncMock(),
    )


async def _test_db(tmp_path, monkeypatch, database="admin-language-flow"):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / f'{database}.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(handlers, "async_session_maker", sessions)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(return_value=True))
    return engine, sessions


def _patch_registry(monkeypatch, registry):
    async def build(session):
        return registry

    monkeypatch.setattr(handlers, "build_translation_registry", build)
    monkeypatch.setattr(translation_registry, "build_translation_registry", build)


@pytest.mark.asyncio
async def test_admin_can_open_language_screen_and_readiness_audit(tmp_path, monkeypatch):
    registry = _registry()
    _patch_registry(monkeypatch, registry)
    engine, sessions = await _test_db(tmp_path, monkeypatch)
    bot = _Bot({})
    message = _Message(bot, document=False)
    try:
        async with sessions() as session:
            session.add(
                BotGeneralConfig(
                    id=1,
                    telegram_default_language="ru",
                    telegram_language_selection_enabled=True,
                    telegram_enabled_languages='["ru", "en", "pt"]',
                )
            )
            await session.commit()

        await handlers.admin_language_settings(_callback(bot, message, "admin_language_settings"))
        screen = message.edit_text.await_args.args[0]
        assert "Язык по умолчанию" in screen
        assert "Выбор языка пользователем: <b>Включён</b>" in screen
        assert "@example_bot" in screen
        assert "🇬🇧 English — ❌ не готов" in screen
        assert "🇵🇹 Português — ❌ не готов" in screen
        assert "🇬🇧 English — ✅ готов" not in screen
        assert "🇵🇹 Português — ✅ готов" not in screen

        await handlers.admin_translation_audit(_callback(bot, message, "admin_translation_audit"))
        audit = message.edit_text.await_args.args[0]
        assert "Проверка готовности переводов" in audit
        assert "🇬🇧 English — ❌ не готов" in audit
        assert "🇵🇹 Português — ❌ не готов" in audit
        assert "Отсутствует: 1" in audit
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", ("en", "pt"))
async def test_admin_export_is_locale_and_bot_database_specific(tmp_path, monkeypatch, locale):
    registry = _registry()
    _patch_registry(monkeypatch, registry)
    engine, _ = await _test_db(tmp_path, monkeypatch, database="someone01")
    bot = _Bot({})
    message = _Message(bot, document=False)
    callback = _callback(bot, message, f"admin_translation_export_{locale}")
    try:
        await handlers.admin_translation_export(callback)
        document = message.answer_document.await_args.args[0]
        payload = json.loads(bytes(document.data).decode())
        assert document.filename == f"telegram_translations_someone01_example_bot_{locale}.json"
        assert payload["locale"] == locale
        assert payload["target"] == {
            "telegram_bot_id": 12345,
            "telegram_username": "example_bot",
            "database": "someone01",
        }
        assert {entry["locale"] for entry in payload["translations"]} == {locale}
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_upload_shows_confirmation_then_real_import_updates_readiness(tmp_path, monkeypatch):
    prior_snapshot = translation_cache.snapshot
    registry = _registry()
    _patch_registry(monkeypatch, registry)
    engine, sessions = await _test_db(tmp_path, monkeypatch, database="someone01")
    bot = _Bot({})
    target = handlers._translation_pack_target(bot.bot_info, "someone01")
    bot.pack = _pack(registry, target)
    state = _State()
    try:
        async with sessions() as session:
            session.add(
                BotGeneralConfig(
                    id=1,
                    telegram_default_language="ru",
                    telegram_language_selection_enabled=False,
                    telegram_enabled_languages='["ru"]',
                    translations_revision=0,
                )
            )
            await session.commit()

        await state.set_state(handlers.AdminStates.upload_translation_pack)
        await state.update_data(translation_locale="en")
        upload_message = _Message(bot)
        await handlers.admin_translation_import_file(upload_message, state, bot)

        preview = upload_message.answer.await_args.args[0]
        assert "Язык: 🇬🇧 English" in preview
        assert "Записей в файле: 1" in preview
        assert "Совпадает с этим ботом и базой: ✅" in preview
        assert "Отсутствует: 0" in preview
        assert "Устарело: 0" in preview
        assert "Ошибок: 0" in preview
        assert "Проверка пакета: ✅ готов к импорту" in preview
        assert state.current == handlers.AdminStates.confirm_translation_pack

        async with sessions() as session:
            assert await session.scalar(select(func.count()).select_from(BotTranslation)) == 0
            assert (await session.get(BotGeneralConfig, 1)).translations_revision == 0

        preview_message = _Message(bot, document=False)
        await handlers.admin_translation_import_confirm(
            _callback(bot, preview_message, "admin_translation_import_confirm"),
            state,
            bot,
        )

        async with sessions() as session:
            assert await session.scalar(select(func.count()).select_from(BotTranslation)) == 1
            assert (await session.get(BotGeneralConfig, 1)).translations_revision == 1
        assert state.current is None
        assert "Статус перевода: ✅ готов" in preview_message.edit_text.await_args.args[0]
    finally:
        translation_cache.install(
            prior_snapshot.revision,
            prior_snapshot.sources,
            prior_snapshot.translations,
        )
        await engine.dispose()


@pytest.mark.asyncio
async def test_wrong_bot_and_stale_or_invalid_packs_are_rejected_without_writes(tmp_path, monkeypatch):
    registry = _registry("Value: {value:>4}")
    _patch_registry(monkeypatch, registry)
    engine, sessions = await _test_db(tmp_path, monkeypatch, database="someone01")
    bot = _Bot({})
    target = handlers._translation_pack_target(bot.bot_info, "someone01")
    cases = [
        (_pack(registry, {**target, "telegram_bot_id": 99}, text="Value:    1"), "другого бота"),
        (_pack(registry, {**target, "database": "another_bot"}, text="Value:    1"), "другого бота"),
        (_pack(registry, target, text="Value:    1", source="Old source"), "устарел"),
        (_pack(registry, target, text="Value: {other}"), "плейсхолдеры"),
        (_pack(registry, target, locale="pt", text="Valor: {value:>4}"), "сейчас ожидается пакет en"),
    ]
    try:
        for pack, expected_message in cases:
            bot.pack = pack
            state = _State()
            await state.set_state(handlers.AdminStates.upload_translation_pack)
            await state.update_data(translation_locale="en")
            message = _Message(bot)
            await handlers.admin_translation_import_file(message, state, bot)
            response = message.answer.await_args.args[0].lower()
            assert expected_message in response
            assert "traceback" not in response
            assert state.current == handlers.AdminStates.upload_translation_pack

        async with sessions() as session:
            assert await session.scalar(select(func.count()).select_from(BotTranslation)) == 0
            config = await session.get(BotGeneralConfig, 1)
            assert config is None or config.translations_revision == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_ready_locale_can_be_enabled_and_selector_can_be_toggled(tmp_path, monkeypatch):
    registry = _registry()
    _patch_registry(monkeypatch, registry)
    engine, sessions = await _test_db(tmp_path, monkeypatch)
    bot = _Bot({})
    message = _Message(bot, document=False)
    monkeypatch.setattr(
        handlers,
        "build_command_sets",
        AsyncMock(return_value=([], [], (False, True, True))),
    )
    monkeypatch.setattr(handlers, "refresh_default_commands", AsyncMock())
    try:
        async with sessions() as session:
            session.add(
                BotGeneralConfig(
                    id=1,
                    telegram_default_language="ru",
                    telegram_language_selection_enabled=False,
                    telegram_enabled_languages='["ru"]',
                )
            )
            session.add(
                BotTranslation(
                    locale="en",
                    translation_key="ui.menu",
                    text="Main menu",
                    source_hash=source_hash("Main menu"),
                )
            )
            await session.commit()

        await handlers.admin_language_toggle_locale(
            _callback(bot, message, "admin_language_toggle_en")
        )
        async with sessions() as session:
            config = await session.get(BotGeneralConfig, 1)
            assert config.telegram_enabled_languages == '["ru", "en"]'
            assert config.telegram_default_language == "ru"

        await handlers.admin_language_toggle_selector(
            _callback(bot, message, "admin_language_toggle_selector")
        )
        async with sessions() as session:
            config = await session.get(BotGeneralConfig, 1)
            assert config.telegram_language_selection_enabled is True

        await handlers.admin_language_toggle_selector(
            _callback(bot, message, "admin_language_toggle_selector")
        )
        async with sessions() as session:
            config = await session.get(BotGeneralConfig, 1)
            assert config.telegram_language_selection_enabled is False
            assert config.telegram_enabled_languages == '["ru", "en"]'
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_not_ready_locale_and_selector_activation_are_blocked(tmp_path, monkeypatch):
    registry = _registry()
    _patch_registry(monkeypatch, registry)
    engine, sessions = await _test_db(tmp_path, monkeypatch)
    bot = _Bot({})
    message = _Message(bot, document=False)
    try:
        async with sessions() as session:
            session.add(
                BotGeneralConfig(
                    id=1,
                    telegram_default_language="ru",
                    telegram_language_selection_enabled=False,
                    telegram_enabled_languages='["ru"]',
                )
            )
            await session.commit()

        enable = _callback(bot, message, "admin_language_toggle_en")
        await handlers.admin_language_toggle_locale(enable)
        assert enable.answer.await_args.kwargs.get("show_alert") is True
        assert "не готов" in enable.answer.await_args.args[0]

        async with sessions() as session:
            config = await session.get(BotGeneralConfig, 1)
            config.telegram_enabled_languages = '["ru", "en"]'
            await session.commit()

        selector = _callback(bot, message, "admin_language_toggle_selector")
        await handlers.admin_language_toggle_selector(selector)
        assert selector.answer.await_args.kwargs.get("show_alert") is True
        assert "не готов" in selector.answer.await_args.args[0]
        async with sessions() as session:
            assert (await session.get(BotGeneralConfig, 1)).telegram_language_selection_enabled is False
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_ru_only_confirmation_preserves_translation_rows_and_config_is_telegram_scoped(tmp_path, monkeypatch):
    registry = _registry()
    _patch_registry(monkeypatch, registry)
    engine, sessions = await _test_db(tmp_path, monkeypatch)
    bot = _Bot({})
    preview_message = _Message(bot, document=False)
    try:
        async with sessions() as session:
            session.add(
                BotGeneralConfig(
                    id=1,
                    telegram_default_language="ru",
                    telegram_language_selection_enabled=True,
                    telegram_enabled_languages='["ru", "en", "pt"]',
                    translations_revision=2,
                )
            )
            session.add(
                BotTranslation(
                    locale="en",
                    translation_key="ui.menu",
                    text="Main menu",
                    source_hash=source_hash("Main menu"),
                )
            )
            session.add(User(id=9002, first_name="Test", telegram_language_code="pt"))
            await session.commit()

        await handlers.admin_language_ru_only(
            _callback(bot, preview_message, "admin_language_ru_only")
        )
        assert "Переводы сохранятся" in preview_message.edit_text.await_args.args[0]
        async with sessions() as session:
            config = await session.get(BotGeneralConfig, 1)
            assert config.telegram_language_selection_enabled is True
            assert config.telegram_enabled_languages == '["ru", "en", "pt"]'

        monkeypatch.setattr(handlers, "build_command_sets", AsyncMock(return_value=([], [], (False, True, True))))
        monkeypatch.setattr(handlers, "refresh_default_commands", AsyncMock())
        await handlers.admin_language_ru_only_confirm(
            _callback(bot, preview_message, "admin_language_ru_only_confirm")
        )

        async with sessions() as session:
            config = await session.get(BotGeneralConfig, 1)
            assert config.telegram_default_language == "ru"
            assert config.telegram_language_selection_enabled is False
            assert config.telegram_enabled_languages == '["ru"]'
            assert config.translations_revision == 2
            assert await session.scalar(select(func.count()).select_from(BotTranslation)) == 1
            assert (await session.get(User, 9002)).telegram_language_code == "pt"
        assert "Включён режим «только русский»" in preview_message.edit_text.await_args.args[0]
    finally:
        await engine.dispose()
