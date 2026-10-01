import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aiogram.types import MessageEntity

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import handlers
from database import DEFAULT_AI_PROCESSING_MESSAGE_TEXT


class _Session:
    def __init__(self, config):
        self.config = config
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, model, key):
        return self.config

    async def execute(self, statement, *args, **kwargs):
        return SimpleNamespace(
            scalars=lambda: SimpleNamespace(all=lambda: []),
            scalar_one_or_none=lambda: None,
            scalar=lambda: None,
        )

    async def scalar(self, statement, *args, **kwargs):
        return None

    def add(self, value):
        return None

    async def commit(self):
        self.commits += 1


class _Bot:
    def __init__(self):
        self.send_message = AsyncMock(return_value=SimpleNamespace())


def _custom(offset: int, length: int, emoji_id: str) -> MessageEntity:
    return MessageEntity(
        type="custom_emoji",
        offset=offset,
        length=length,
        custom_emoji_id=emoji_id,
    )


@pytest.mark.asyncio
async def test_plain_text_and_normal_unicode_emoji_remain_legacy_values():
    assert handlers.serialize_ai_processing_message_text("Думаю...") == "Думаю..."
    assert handlers.serialize_ai_processing_message_text("🙂 Жду") == "🙂 Жду"

    bot = _Bot()
    config = SimpleNamespace(
        ai_processing_message_enabled=True,
        ai_processing_message_text="🙂 Жду",
    )
    await handlers._send_ai_processing_message(bot, 42, config)
    kwargs = bot.send_message.await_args.kwargs
    assert kwargs["text"] == "🙂 Жду"
    assert kwargs["parse_mode"] is None
    assert "entities" not in kwargs


@pytest.mark.asyncio
async def test_admin_save_preserves_exact_custom_emoji_id(monkeypatch):
    config = SimpleNamespace(
        ai_processing_message_enabled=True,
        ai_processing_message_text="Думаю...",
    )
    session = _Session(config)
    message = SimpleNamespace(
        text="Жду 🙂",
        entities=[_custom(5, 2, "987654321")],
        chat=SimpleNamespace(id=42),
        delete=AsyncMock(),
        answer=AsyncMock(),
    )
    state = SimpleNamespace(get_data=AsyncMock(return_value={}), clear=AsyncMock())
    bot = SimpleNamespace(edit_message_text=AsyncMock())

    monkeypatch.setattr(handlers, "async_session_maker", lambda: session)
    monkeypatch.setattr(handlers, "admin_general_settings", AsyncMock())

    await handlers.admin_save_ai_processing_message_text(message, state, bot)

    assert config.ai_processing_message_text.startswith(handlers._AI_PROCESSING_ENTITIES_PREFIX)
    decoded_text, entities, encoded = handlers._decode_ai_processing_message_text(
        config.ai_processing_message_text
    )
    assert encoded is True
    assert decoded_text == "Жду 🙂"
    assert [entity.custom_emoji_id for entity in entities] == ["987654321"]


@pytest.mark.asyncio
async def test_multiple_mixed_custom_emoji_are_sent_as_aiogram_entities():
    text = "🙂 A 😀"
    entities = [
        _custom(0, 2, "first-id"),
        MessageEntity(type="bold", offset=3, length=1),
        _custom(5, 2, "second-id"),
    ]
    stored = handlers.serialize_ai_processing_message_text(text, entities)
    bot = _Bot()
    config = SimpleNamespace(ai_processing_message_enabled=True, ai_processing_message_text=stored)

    await handlers._send_ai_processing_message(bot, 42, config)

    kwargs = bot.send_message.await_args.kwargs
    assert kwargs["text"] == text
    assert [entity.custom_emoji_id for entity in kwargs["entities"] if entity.custom_emoji_id] == [
        "first-id",
        "second-id",
    ]
    assert [entity.type for entity in kwargs["entities"]] == [
        "custom_emoji",
        "bold",
        "custom_emoji",
    ]
    assert kwargs["parse_mode"] is None


def test_preview_escapes_arbitrary_legacy_html_text():
    assert handlers._ai_processing_message_html("<b>& unsafe") == "&lt;b&gt;&amp; unsafe"


@pytest.mark.asyncio
async def test_legacy_plain_text_value_is_sent_unchanged():
    bot = _Bot()
    config = SimpleNamespace(
        ai_processing_message_enabled=True,
        ai_processing_message_text="legacy <text> & value",
    )

    await handlers._send_ai_processing_message(bot, 42, config)

    kwargs = bot.send_message.await_args.kwargs
    assert kwargs["text"] == "legacy <text> & value"
    assert kwargs["parse_mode"] is None
    assert "entities" not in kwargs


@pytest.mark.asyncio
async def test_malformed_serialized_value_uses_safe_default_and_does_not_raise():
    bot = _Bot()
    config = SimpleNamespace(
        ai_processing_message_enabled=True,
        ai_processing_message_text=handlers._AI_PROCESSING_ENTITIES_PREFIX + "not-valid",
    )

    await handlers._send_ai_processing_message(bot, 42, config)

    kwargs = bot.send_message.await_args.kwargs
    assert kwargs["text"] == DEFAULT_AI_PROCESSING_MESSAGE_TEXT
    assert kwargs["parse_mode"] is None
    assert "entities" not in kwargs


@pytest.mark.asyncio
async def test_exact_regression_montage_emoji_custom_entity_roundtrip(monkeypatch):
    """Exact user scenario: '🎬 Монтирую ответ...' with Telegram Premium custom emoji.

    In the old implementation, this produced 203 characters which failed the old 200 limit.
    With the new limits (visible 4096, storage 16000), it succeeds, saves, and restores cleanly.
    """
    text = "🎬 Монтирую ответ..."
    entities = [_custom(0, 2, "5373111497228854483")]

    serialized = handlers.serialize_ai_processing_message_text(text, entities)
    assert len(serialized) > 200
    assert len(serialized) == 203
    assert serialized.startswith(handlers._AI_PROCESSING_ENTITIES_PREFIX)

    config = SimpleNamespace(
        ai_processing_message_enabled=True,
        ai_processing_message_text="Думаю...",
    )
    session = _Session(config)
    message = SimpleNamespace(
        text=text,
        entities=entities,
        chat=SimpleNamespace(id=42),
        delete=AsyncMock(),
        answer=AsyncMock(),
    )
    state = SimpleNamespace(get_data=AsyncMock(return_value={}), clear=AsyncMock())
    bot = SimpleNamespace(edit_message_text=AsyncMock())

    monkeypatch.setattr(handlers, "async_session_maker", lambda: session)
    monkeypatch.setattr(handlers, "admin_general_settings", AsyncMock())

    await handlers.admin_save_ai_processing_message_text(message, state, bot)
    assert config.ai_processing_message_text == serialized

    decoded_text, decoded_entities, encoded = handlers._decode_ai_processing_message_text(
        config.ai_processing_message_text
    )
    assert encoded is True
    assert decoded_text == text
    assert len(decoded_entities) == 1
    assert decoded_entities[0].type == "custom_emoji"
    assert decoded_entities[0].offset == 0
    assert decoded_entities[0].length == 2
    assert decoded_entities[0].custom_emoji_id == "5373111497228854483"

    mock_bot = _Bot()
    await handlers._send_ai_processing_message(mock_bot, 42, config)
    kwargs = mock_bot.send_message.await_args.kwargs
    assert kwargs["text"] == text
    assert len(kwargs["entities"]) == 1
    assert kwargs["entities"][0].type == "custom_emoji"
    assert kwargs["entities"][0].custom_emoji_id == "5373111497228854483"
    assert kwargs["entities"][0].offset == 0
    assert kwargs["entities"][0].length == 2
    assert kwargs["parse_mode"] is None


def test_visible_text_limits_and_boundaries():
    # 1 char: OK
    assert handlers.normalize_ai_processing_message_text("x") == "x"
    assert handlers.serialize_ai_processing_message_text("x") == "x"

    # 4096 chars: OK
    text_4096 = "a" * 4096
    assert handlers.normalize_ai_processing_message_text(text_4096) == text_4096
    assert handlers.serialize_ai_processing_message_text(text_4096) == text_4096

    # 4097 chars: ValueError
    text_4097 = "a" * 4097
    with pytest.raises(ValueError, match="Текст слишком длинный"):
        handlers.normalize_ai_processing_message_text(text_4097)
    with pytest.raises(ValueError, match="Текст слишком длинный"):
        handlers.serialize_ai_processing_message_text(text_4097)

    # Empty / whitespace-only: ValueError
    with pytest.raises(ValueError, match="Текст не может быть пустым"):
        handlers.normalize_ai_processing_message_text("   \n\t  ")
    with pytest.raises(ValueError, match="Текст не может быть пустым"):
        handlers.serialize_ai_processing_message_text("   \n\t  ")


def test_storage_overflow_at_16000_limit():
    import hashlib

    # Deterministically generate entities that exceed the 16000 serialized storage limit
    entities = [
        MessageEntity(
            type="custom_emoji",
            offset=i,
            length=1,
            custom_emoji_id=hashlib.sha256(str(i).encode()).hexdigest(),
        )
        for i in range(400)
    ]
    text = "a" * 400
    with pytest.raises(
        ValueError,
        match=r"Текст с форматированием слишком длинный для сохранения\. Максимум — 16000 символов в закодированном виде\.",
    ):
        handlers.serialize_ai_processing_message_text(text, entities)

    # Fits comfortably within 16000 limit
    valid_entities = [
        MessageEntity(
            type="custom_emoji",
            offset=i,
            length=1,
            custom_emoji_id=f"emoji_{i}",
        )
        for i in range(30)
    ]
    valid_text = "a" * 30
    encoded = handlers.serialize_ai_processing_message_text(valid_text, valid_entities)
    assert len(encoded) < 16000
    decoded_text, decoded_entities, is_encoded = handlers._decode_ai_processing_message_text(encoded)
    assert is_encoded is True
    assert decoded_text == valid_text
    assert len(decoded_entities) == 30


@pytest.mark.asyncio
async def test_admin_prompt_displays_max_4096_symbols(monkeypatch):
    config = SimpleNamespace(ai_processing_message_text="Думаю...")
    session = _Session(config)
    monkeypatch.setattr(handlers, "async_session_maker", lambda: session)

    callback = SimpleNamespace(
        message=SimpleNamespace(
            message_id=123,
            edit_text=AsyncMock(),
        ),
        answer=AsyncMock(),
    )
    state = SimpleNamespace(
        set_state=AsyncMock(),
        update_data=AsyncMock(),
    )

    await handlers.admin_edit_ai_processing_message_text(callback, state)
    callback.message.edit_text.assert_awaited_once()
    prompt_text = callback.message.edit_text.await_args[0][0]
    assert "После очистки текст должен содержать от 1 до 4096 символов." in prompt_text


def test_database_model_column_is_text():
    from sqlalchemy.types import Text
    from database import (
        BotGeneralConfig,
        AI_PROCESSING_MESSAGE_MAX_LENGTH,
        AI_PROCESSING_MESSAGE_STORAGE_MAX_LENGTH,
    )

    assert AI_PROCESSING_MESSAGE_MAX_LENGTH == 4096
    assert AI_PROCESSING_MESSAGE_STORAGE_MAX_LENGTH == 16000
    assert isinstance(BotGeneralConfig.ai_processing_message_text.type, Text)
    assert BotGeneralConfig.ai_processing_message_text.default.arg == "Думаю..."
    assert BotGeneralConfig.ai_processing_message_text.nullable is False


@pytest.mark.asyncio
async def test_init_db_postgresql_widens_existing_column_to_text(monkeypatch):
    import database
    from database import init_db

    executed_statements = []

    class MockInspector:
        def has_table(self, table_name):
            return True

        def get_columns(self, table_name):
            if table_name == "bot_general_config":
                return [{"name": "ai_processing_message_text"}]
            return [
                {"name": "response_length"},
                {"name": "birth_day"},
                {"name": "stage_mode"},
                {"name": "platform"},
                {"name": "memory_mode"},
                {"name": "id"},
            ]

        def get_indexes(self, table_name):
            return []

        def get_unique_constraints(self, table_name):
            return []

    class MockSyncConn:
        dialect = SimpleNamespace(name="postgresql")

        def execute(self, stmt, *args, **kwargs):
            executed_statements.append(str(stmt))

        def _run_ddl_visitor(self, *args, **kwargs):
            pass

    class MockConn:
        dialect = SimpleNamespace(name="postgresql")

        async def execute(self, stmt, *args, **kwargs):
            pass

        async def run_sync(self, fn):
            if getattr(fn, "__name__", "") == "_check_and_migrate":
                fn(MockSyncConn())

    class MockEngine:
        def begin(self):
            class MockContext:
                async def __aenter__(self):
                    return MockConn()

                async def __aexit__(self, *args):
                    return False

            return MockContext()

    monkeypatch.setattr(database, "engine", MockEngine())
    monkeypatch.setattr("sqlalchemy.inspect", lambda conn: MockInspector())
    monkeypatch.setattr(database, "_acquire_database_init_lock", AsyncMock())
    monkeypatch.setattr(database, "_migrate_legacy_media_ownership", lambda conn: None)
    monkeypatch.setattr(database, "_migrate_ai_config_models", lambda conn: None)
    monkeypatch.setattr(database, "verify_yookassa_recurring_safety_schema", lambda conn: None)
    monkeypatch.setattr(database, "verify_payment_notification_outbox_schema", lambda conn: None)
    monkeypatch.setattr(database, "async_session_maker", lambda: _Session(None))

    await init_db()

    assert any(
        "ALTER TABLE bot_general_config ALTER COLUMN ai_processing_message_text TYPE TEXT" in stmt
        for stmt in executed_statements
    )


@pytest.mark.asyncio
async def test_init_db_adds_column_as_text_when_missing(monkeypatch):
    import database
    from database import init_db

    executed_statements = []

    class MockInspector:
        def has_table(self, table_name):
            return True

        def get_columns(self, table_name):
            if table_name == "bot_general_config":
                return []
            return [
                {"name": "response_length"},
                {"name": "birth_day"},
                {"name": "stage_mode"},
                {"name": "platform"},
                {"name": "memory_mode"},
                {"name": "id"},
            ]

        def get_indexes(self, table_name):
            return []

        def get_unique_constraints(self, table_name):
            return []

    class MockSyncConn:
        dialect = SimpleNamespace(name="postgresql")

        def execute(self, stmt, *args, **kwargs):
            executed_statements.append(str(stmt))

        def _run_ddl_visitor(self, *args, **kwargs):
            pass

    class MockConn:
        dialect = SimpleNamespace(name="postgresql")

        async def execute(self, stmt, *args, **kwargs):
            pass

        async def run_sync(self, fn):
            if getattr(fn, "__name__", "") == "_check_and_migrate":
                fn(MockSyncConn())

    class MockEngine:
        def begin(self):
            class MockContext:
                async def __aenter__(self):
                    return MockConn()

                async def __aexit__(self, *args):
                    return False

            return MockContext()

    monkeypatch.setattr(database, "engine", MockEngine())
    monkeypatch.setattr("sqlalchemy.inspect", lambda conn: MockInspector())
    monkeypatch.setattr(database, "_acquire_database_init_lock", AsyncMock())
    monkeypatch.setattr(database, "_migrate_legacy_media_ownership", lambda conn: None)
    monkeypatch.setattr(database, "_migrate_ai_config_models", lambda conn: None)
    monkeypatch.setattr(database, "verify_yookassa_recurring_safety_schema", lambda conn: None)
    monkeypatch.setattr(database, "verify_payment_notification_outbox_schema", lambda conn: None)
    monkeypatch.setattr(database, "async_session_maker", lambda: _Session(None))

    await init_db()

    assert any(
        "ALTER TABLE bot_general_config ADD COLUMN ai_processing_message_text TEXT DEFAULT 'Думаю...' NOT NULL" in stmt
        for stmt in executed_statements
    )


@pytest.mark.asyncio
async def test_init_db_sqlite_does_not_alter_column_type(monkeypatch):
    import database
    from database import init_db

    executed_statements = []

    class MockInspector:
        def has_table(self, table_name):
            return True

        def get_columns(self, table_name):
            if table_name == "bot_general_config":
                return [{"name": "ai_processing_message_text"}]
            return [
                {"name": "response_length"},
                {"name": "birth_day"},
                {"name": "stage_mode"},
                {"name": "platform"},
                {"name": "memory_mode"},
                {"name": "id"},
            ]

        def get_indexes(self, table_name):
            return []

        def get_unique_constraints(self, table_name):
            return []

    class MockSyncConn:
        dialect = SimpleNamespace(name="sqlite")

        def execute(self, stmt, *args, **kwargs):
            executed_statements.append(str(stmt))

        def _run_ddl_visitor(self, *args, **kwargs):
            pass

    class MockConn:
        dialect = SimpleNamespace(name="sqlite")

        async def execute(self, stmt, *args, **kwargs):
            pass

        async def run_sync(self, fn):
            if getattr(fn, "__name__", "") == "_check_and_migrate":
                fn(MockSyncConn())

    class MockEngine:
        def begin(self):
            class MockContext:
                async def __aenter__(self):
                    return MockConn()

                async def __aexit__(self, *args):
                    return False

            return MockContext()

    monkeypatch.setattr(database, "engine", MockEngine())
    monkeypatch.setattr("sqlalchemy.inspect", lambda conn: MockInspector())
    monkeypatch.setattr(database, "_acquire_database_init_lock", AsyncMock())
    monkeypatch.setattr(database, "_migrate_legacy_media_ownership", lambda conn: None)
    monkeypatch.setattr(database, "_migrate_ai_config_models", lambda conn: None)
    monkeypatch.setattr(database, "verify_yookassa_recurring_safety_schema", lambda conn: None)
    monkeypatch.setattr(database, "verify_payment_notification_outbox_schema", lambda conn: None)
    monkeypatch.setattr(database, "async_session_maker", lambda: _Session(None))

    await init_db()

    assert not any(
        "ALTER TABLE bot_general_config ALTER COLUMN ai_processing_message_text" in stmt
        for stmt in executed_statements
    )
