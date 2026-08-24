from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import database
from database import TelegramPendingAIReply


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
