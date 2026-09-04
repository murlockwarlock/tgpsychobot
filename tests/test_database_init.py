from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import database
from database import Base, TelegramPendingAIReply


class _FakeConnection:
    def __init__(self, dialect_name: str, events: list):
        self.dialect = SimpleNamespace(name=dialect_name)
        self.events = events

    async def execute(self, statement, parameters=None):
        self.events.append(("execute", str(statement), parameters))

    async def run_sync(self, operation):
        self.events.append(("run_sync", operation))


class _BeginContext:
    def __init__(self, connection: _FakeConnection, events: list):
        self.connection = connection
        self.events = events

    async def __aenter__(self):
        self.events.append("begin")
        return self.connection

    async def __aexit__(self, *_args):
        self.events.append("commit")
        return False


class _FakeEngine:
    def __init__(self, connection: _FakeConnection, events: list):
        self.connection = connection
        self.events = events

    def begin(self):
        return _BeginContext(self.connection, self.events)


class _FakeSession:
    def __init__(self, events: list):
        self.events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def get(self, _model, _key):
        return None

    async def execute(self, _statement, _parameters=None):
        return SimpleNamespace(
            scalars=lambda: SimpleNamespace(all=lambda: []),
        )

    def add(self, value):
        self.events.append(("add", value))

    async def commit(self):
        self.events.append("session_commit")


def _install_fake_init_dependencies(monkeypatch, dialect_name: str):
    events = []
    connection = _FakeConnection(dialect_name, events)
    monkeypatch.setattr(database, "engine", _FakeEngine(connection, events))
    monkeypatch.setattr(
        database,
        "async_session_maker",
        lambda: _FakeSession(events),
    )
    return events


@pytest.mark.asyncio
async def test_postgresql_init_locks_before_create_all_and_migrations(monkeypatch):
    events = _install_fake_init_dependencies(monkeypatch, "postgresql")

    await database.init_db()

    assert [event[0] if isinstance(event, tuple) else event for event in events[:5]] == [
        "begin",
        "execute",
        "run_sync",
        "run_sync",
        "commit",
    ]
    lock_event = events[1]
    assert "pg_advisory_xact_lock" in lock_event[1]
    assert lock_event[2] == {"lock_key": database.DATABASE_INIT_ADVISORY_LOCK_KEY}


@pytest.mark.asyncio
async def test_sqlite_init_does_not_execute_postgresql_advisory_lock(monkeypatch):
    events = _install_fake_init_dependencies(monkeypatch, "sqlite")

    await database.init_db()

    assert not any(
        isinstance(event, tuple) and event[0] == "execute"
        for event in events
    )
    assert [event[0] if isinstance(event, tuple) else event for event in events[:4]] == [
        "begin",
        "run_sync",
        "run_sync",
        "commit",
    ]


@pytest.mark.asyncio
async def test_init_db_still_succeeds_on_sqlite(tmp_path, monkeypatch):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'database-init.db'}"
    )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "async_session_maker", sessions)

    try:
        await database.init_db()
        async with engine.connect() as connection:
            has_pending_table = await connection.run_sync(
                lambda sync_connection: inspect(sync_connection).has_table(
                    TelegramPendingAIReply.__tablename__
                )
            )
        assert has_pending_table is True
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_init_db_migrates_existing_followup_columns(tmp_path, monkeypatch):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'followup-migration.db'}"
    )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "async_session_maker", sessions)

    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            await connection.execute(text(
                "ALTER TABLE followup_campaigns DROP COLUMN stage_include_unset"
            ))
            await connection.execute(text(
                "ALTER TABLE followup_delivery_attempts DROP COLUMN attempt_count"
            ))
            await connection.execute(text(
                "INSERT INTO followup_campaigns ("
                "name, is_active, all_topics, include_main_dialogue, stage_mode, stage_values, "
                "metadata_field_path, metadata_operator, metadata_expected_value, stop_events, "
                "timezone, quiet_start_minute, quiet_end_minute, jitter_min_seconds, jitter_max_seconds, "
                "created_at, updated_at"
                ") VALUES ("
                "'legacy', 1, 0, 1, 'selected', 'guide_choice', NULL, NULL, NULL, '', "
                "'Europe/Moscow', 1320, 540, 0, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP"
                ")"
            ))
            for name, mode, stage_values in (
                ("legacy all", "all", ""),
                ("legacy all except", "all_except", "blocked"),
                ("legacy not set", "not_set", ""),
                ("canonical all except unset", "all_except", "blocked, [не задан]"),
            ):
                await connection.execute(text(
                    "INSERT INTO followup_campaigns ("
                    "name, is_active, all_topics, include_main_dialogue, stage_mode, stage_values, "
                    "metadata_field_path, metadata_operator, metadata_expected_value, stop_events, "
                    "timezone, quiet_start_minute, quiet_end_minute, jitter_min_seconds, jitter_max_seconds, "
                    "created_at, updated_at"
                    ") VALUES ("
                    ":name, 1, 0, 1, :stage_mode, :stage_values, NULL, NULL, NULL, '', "
                    "'Europe/Moscow', 1320, 540, 0, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP"
                    ")"
                ), {"name": name, "stage_mode": mode, "stage_values": stage_values})
        await database.init_db()
        async with engine.connect() as connection:
            campaign_columns = await connection.run_sync(
                lambda sync_connection: inspect(sync_connection).get_columns("followup_campaigns")
            )
            attempt_columns = await connection.run_sync(
                lambda sync_connection: inspect(sync_connection).get_columns("followup_delivery_attempts")
            )
            migrated_value = await connection.scalar(text(
                "SELECT stage_include_unset FROM followup_campaigns WHERE name = 'legacy'"
            ))
            migrated_modes = dict((await connection.execute(text(
                "SELECT name, stage_mode FROM followup_campaigns "
                "WHERE name IN ('legacy all', 'legacy all except', 'legacy not set', "
                "'canonical all except unset')"
            ))).all())
        campaign_column = next(column for column in campaign_columns if column["name"] == "stage_include_unset")
        attempt_column = next(column for column in attempt_columns if column["name"] == "attempt_count")
        assert campaign_column["nullable"] is False
        assert attempt_column["nullable"] is False
        assert migrated_value == 0
        assert migrated_modes == {
            "legacy all": "all_legacy",
            "legacy all except": "all_except_legacy",
            "legacy not set": "not_set",
            "canonical all except unset": "all_except",
        }
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_init_db_seeds_menu_content_and_preserves_custom_text(tmp_path, monkeypatch):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'menu-content-init.db'}"
    )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "async_session_maker", sessions)

    try:
        # First run: fresh DB -> Content("menu") is seeded with neutral placeholder
        await database.init_db()

        async with sessions() as session:
            menu_content = await session.get(database.Content, "menu")
            assert menu_content is not None
            assert menu_content.text_content == "Раздел меню пока не настроен."

            # Update Content("menu") with custom text
            menu_content.text_content = "Пользовательское меню\n[Темы](btn:svc:topics)"
            await session.commit()

        # Second run: re-running init_db must NEVER overwrite custom text
        await database.init_db()

        async with sessions() as session:
            reloaded_menu = await session.get(database.Content, "menu")
            assert reloaded_menu is not None
            assert reloaded_menu.text_content == "Пользовательское меню\n[Темы](btn:svc:topics)"
    finally:
        await engine.dispose()
