from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import background_worker
import birthday_mailings
import database
from database import Base, Mailing, MailingDeliveryLog, User
from max_messenger_bot.services import admin_mailing as max_admin_mailing
from mailing_utils import get_mailing_platform_label, resolve_mailing_platform


@pytest.mark.parametrize(
    ("mailing", "platform", "label"),
    [
        (SimpleNamespace(platform="telegram", creator_id=None), "telegram", "Telegram"),
        (SimpleNamespace(platform="max", creator_id=None), "max", "MAX"),
        (SimpleNamespace(platform=None, creator_id=42), "telegram", "Telegram"),
        (SimpleNamespace(platform=None, creator_id=100_000_000_042), "max", "MAX"),
        (SimpleNamespace(platform=None, creator_id=None), None, "Не определена"),
    ],
)
def test_mailing_platform_uses_explicit_value_then_justified_legacy_source(mailing, platform, label):
    assert resolve_mailing_platform(mailing) == platform
    assert get_mailing_platform_label(mailing) == label


@pytest.mark.asyncio
async def test_telegram_worker_does_not_claim_max_mailings_or_recipients(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'mailing-platform.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(background_worker, "async_session_maker", sessions)
    monkeypatch.setattr(background_worker, "refresh_translation_cache", AsyncMock())
    monkeypatch.setattr(database, "get_all_admin_ids", AsyncMock(return_value=set()))

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with sessions() as session:
        session.add_all(
            [
                User(id=11, first_name="Telegram"),
                User(id=100_000_000_011, first_name="MAX"),
                Mailing(
                    text="Telegram only",
                    target_audience="all",
                    creator_id=11,
                    platform="telegram",
                    status="pending",
                ),
                Mailing(
                    text="MAX only",
                    target_audience="all",
                    creator_id=100_000_000_099,
                    platform="max",
                    status="pending",
                ),
            ]
        )
        await session.commit()

    sent = AsyncMock()
    sleep_calls = 0

    async def sleep(delay):
        nonlocal sleep_calls
        if delay == 15 and sleep_calls:
            raise asyncio.CancelledError
        sleep_calls += 1

    monkeypatch.setattr(background_worker.asyncio, "sleep", sleep)

    with pytest.raises(asyncio.CancelledError):
        await background_worker.process_mailings(SimpleNamespace(send_message=sent))

    recipient_ids = [call.args[0] for call in sent.await_args_list if call.args]
    assert recipient_ids == [11]
    async with sessions() as session:
        mailings = (await session.scalars(select(Mailing).order_by(Mailing.id))).all()
        assert [mailing.status for mailing in mailings] == ["completed", "pending"]
        assert mailings[0].success_count == 1
        assert mailings[0].failure_count == 0

    await engine.dispose()


@pytest.mark.asyncio
async def test_birthday_worker_stays_on_telegram_platform(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'birthday-platform.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(birthday_mailings, "async_session_maker", sessions)
    monkeypatch.setattr(database, "get_all_admin_ids", AsyncMock(return_value=set()))

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with sessions() as session:
        session.add_all(
            [
                User(id=11, first_name="Telegram", birth_day=2, birth_month=11),
                User(id=100_000_000_011, first_name="MAX", birth_day=2, birth_month=11),
                Mailing(
                    text="Telegram birthday",
                    target_audience="birthday_today",
                    creator_id=11,
                    platform="telegram",
                    recurring_type="birthday",
                    is_enabled=True,
                    status="active",
                ),
                Mailing(
                    text="MAX birthday",
                    target_audience="birthday_today",
                    creator_id=100_000_000_099,
                    platform="max",
                    recurring_type="birthday",
                    is_enabled=True,
                    status="active",
                ),
            ]
        )
        await session.commit()

    sent = AsyncMock()
    await birthday_mailings.process_birthday_mailings(
        SimpleNamespace(send_message=sent),
        now=datetime(2026, 11, 2, tzinfo=timezone.utc),
    )

    assert [call.args[0] for call in sent.await_args_list if call.args] == [11]
    async with sessions() as session:
        logs = (await session.scalars(select(MailingDeliveryLog))).all()
        assert [(log.user_id, log.status) for log in logs] == [(11, "sent")]

    await engine.dispose()


@pytest.mark.asyncio
async def test_max_recipient_selection_excludes_telegram_users(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'max-recipient-platform.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(max_admin_mailing, "async_session_maker", sessions)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with sessions() as session:
        session.add_all([
            User(id=11, first_name="Telegram"),
            User(id=100_000_000_011, first_name="MAX"),
        ])
        await session.commit()

    async with sessions() as session:
        recipient_ids = await max_admin_mailing._get_recipient_ids(
            session,
            "all",
            100_000_000_099,
        )

    assert recipient_ids == [100_000_000_011]
    await engine.dispose()
