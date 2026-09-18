import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from datetime import datetime, timedelta

import handlers
from database import Base, SubscriptionBenefitGrant, TelegramStartIntent, TestSession as DBTestSession, User, UserSubscription
from handlers import UserStates
from telegram_start_service import (
    grant_subscription_days,
    language_selection_enabled_for_user,
    parse_referral_payload,
    record_start_intent,
)


def test_referral_payload_parser_preserves_current_start_semantics():
    assert parse_referral_payload("ref_123", 99) == 123
    assert parse_referral_payload("ref_123_extra", 99) is None
    assert parse_referral_payload("ref_99", 99) is None
    assert parse_referral_payload("ref", 99) is None


@pytest.mark.asyncio
async def test_start_intent_keeps_referral_acquisition_when_navigation_changes(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'intent.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with sessions() as session:
            session.add_all([
                User(id=10, first_name="New"),
                User(id=123, first_name="Referrer"),
            ])
            await session.commit()

        async with sessions() as session:
            intent = await record_start_intent(
                session,
                user_id=10,
                args="ref_123",
                new_user_eligible=True,
            )
            await session.commit()
            await record_start_intent(
                session,
                user_id=10,
                args="topic_5",
                new_user_eligible=True,
            )
            await session.commit()

        async with sessions() as session:
            reloaded = await session.get(TelegramStartIntent, 10)
            assert reloaded.navigation_payload == "topic_5"
            assert reloaded.acquisition_referrer_id == 123
            assert reloaded.new_user_eligible is True

        async with sessions() as session:
            await record_start_intent(
                session,
                user_id=10,
                args=None,
                new_user_eligible=True,
            )
            await session.commit()

        async with sessions() as session:
            reloaded = await session.get(TelegramStartIntent, 10)
            assert reloaded.navigation_payload is None
            assert reloaded.acquisition_referrer_id == 123
    finally:
        await engine.dispose()


def test_language_gate_requires_new_user_intent_not_null_preference_alone():
    assert language_selection_enabled_for_user(
        selector_enabled=True,
        enabled_languages=("ru", "en"),
        user_language=None,
        intent_new_user_eligible=False,
    ) is False
    assert language_selection_enabled_for_user(
        selector_enabled=True,
        enabled_languages=("ru", "en"),
        user_language=None,
        intent_new_user_eligible=True,
    ) is True


@pytest.mark.asyncio
async def test_subscription_benefit_grant_is_idempotent_and_preserves_one_to_one_subscription(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'grant.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with sessions() as session:
            session.add(User(id=10, first_name="New"))
            await session.commit()

        now = datetime.utcnow()
        async with sessions() as session:
            first = await grant_subscription_days(
                session,
                grant_key="welcome:10",
                grant_type="welcome",
                beneficiary_user_id=10,
                days=3,
                payment_provider="Trial Welcome",
                now=now,
            )
            await session.commit()
        async with sessions() as session:
            second = await grant_subscription_days(
                session,
                grant_key="welcome:10",
                grant_type="welcome",
                beneficiary_user_id=10,
                days=3,
                payment_provider="Trial Welcome",
                now=now + timedelta(days=1),
            )
            await session.commit()
            subscription = await session.scalar(
                __import__("sqlalchemy").select(UserSubscription).where(UserSubscription.user_id == 10)
            )
            grant_count = await session.scalar(
                __import__("sqlalchemy").select(__import__("sqlalchemy").func.count()).select_from(SubscriptionBenefitGrant)
            )

        assert first is True
        assert second is False
        assert subscription.end_date == now + timedelta(days=3)
        assert grant_count == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_test_session_reconstructs_fsm_after_process_restart(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test-recovery.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(handlers, "async_session_maker", sessions)
    bot = object()
    message = SimpleNamespace(from_user=SimpleNamespace(id=10))
    state = SimpleNamespace(
        get_state=AsyncMock(return_value=None),
        set_state=AsyncMock(),
    )
    process_answer = AsyncMock()
    monkeypatch.setattr(handlers, "process_test_text_answer", process_answer)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with sessions() as session:
            session.add_all([
                User(id=10, first_name="Tester"),
                DBTestSession(
                    user_id=10,
                    current_question_index=2,
                    answers='[{"value": 1}]',
                    is_finished=False,
                ),
            ])
            await session.commit()

        assert await handlers._restore_test_state_from_db(message, state, bot) is True
        state.set_state.assert_awaited_once_with(UserStates.in_test)
        process_answer.assert_awaited_once_with(message, state, bot)
    finally:
        await engine.dispose()
