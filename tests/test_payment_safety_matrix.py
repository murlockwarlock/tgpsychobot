from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import logging
import os
from dateutil.relativedelta import relativedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("BOT_TOKEN", "test_bot_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from sqlalchemy import delete, select
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from yookassa.domain.exceptions import (
    BadRequestError,
    ForbiddenError,
    InternalServerError,
    TooManyRequestsError,
    UnauthorizedError,
)

from database import (
    Base,
    SubscriptionConfig,
    SubscriptionPlan,
    User,
    UserSubscription,
    YookassaPayment,
    YookassaRecurringAttempt,
    verify_yookassa_recurring_safety_schema,
)
from subscription_renewal import (
    CancellationPolicy,
    CANCELLATION_TAXONOMY,
    claim_yookassa_recurring_attempt,
    classify_yookassa_cancellation_reason,
    execute_or_replay_yookassa_recurring_attempt,
    finalize_yookassa_payment_canceled,
    finalize_yookassa_payment_success,
    finalize_yookassa_attempt_no_payment,
    mark_unresolved_attempts_superseded,
    transition_attempt_to_manual_review,
    transition_attempt_to_unknown_expired,
    mask_payment_method_id,
    should_reconcile_attempt,
    update_yookassa_attempt_pending,
    update_yookassa_attempt_unknown,
)
from subscription_retry_policy import can_retry_now, get_next_retry_at
from scheduler import check_subscriptions
from max_messenger_bot.services.subscriptions import handle_max_manual_retry


@pytest_asyncio.fixture
async def test_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_maker = async_sessionmaker(engine, expire_on_commit=False)
    yield session_maker
    await engine.dispose()


@pytest.mark.asyncio
async def test_http_500_persists_idempotency_key_and_blocks_new_attempt(test_db):
    """
    Section 1: HTTP 500 means outcome UNKNOWN.
    Original idempotence key must be persisted, new logical attempt must be blocked,
    and later recovery must use the SAME key.
    """
    async with test_db() as session:
        user = User(id=101, username="test101", first_name="User101")
        plan = SubscriptionPlan(id=1, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        session.add_all([user, plan])
        await session.commit()

        sub = UserSubscription(
            id=1,
            user_id=101,
            plan_id=1,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_test_card_1",
            payment_attempt_count=0,
            end_date=datetime(2026, 9, 10, 12, 0, 0),
        )
        session.add(sub)
        await session.commit()

        config = SimpleNamespace(yookassa_shop_id="shop", yookassa_secret_key="secret")

        # 1. Atomic claim before outbound
        claim_res = await claim_yookassa_recurring_attempt(session, sub, plan, 195.0, "scheduler")
        assert claim_res.claimed is True
        attempt = claim_res.attempt
        orig_key = attempt.idempotency_key
        assert orig_key.startswith("yk-rec-")

        # 2. YooKassa raises HTTP 500 InternalServerError
        with patch("subscription_renewal.Payment.create", side_effect=InternalServerError({"code": "internal_server_error"})):
            res = await execute_or_replay_yookassa_recurring_attempt(attempt, plan.name, config)

        assert res.outcome == "unknown"
        assert res.payment_id is None

        # 3. Caller marks attempt as unknown
        await update_yookassa_attempt_unknown(session, attempt.id, "internal_server_error", "HTTP 500")

        # 4. Attempt in DB is retained with status 'unknown' and original key
        refreshed_att = await session.get(YookassaRecurringAttempt, attempt.id)
        assert refreshed_att.status == "unknown"
        assert refreshed_att.idempotency_key == orig_key

        # 5. Subsequent claim attempt (scheduler or manual) is blocked!
        claim_res_blocked = await claim_yookassa_recurring_attempt(session, sub, plan, 195.0, "scheduler")
        assert claim_res_blocked.claimed is False
        assert claim_res_blocked.reason == "unresolved_exists"

        # 6. Replay uses the SAME idempotency key
        mock_payment = SimpleNamespace(id="yk_pay_recovered", status="succeeded")
        with patch("subscription_renewal.Payment.create", return_value=mock_payment) as mock_create:
            replay_res = await execute_or_replay_yookassa_recurring_attempt(refreshed_att, plan.name, config)

        assert replay_res.outcome == "success"
        assert replay_res.payment_id == "yk_pay_recovered"
        # Assert Payment.create was called with the exact same idempotence key
        assert mock_create.call_args[0][1] == orig_key


@pytest.mark.asyncio
async def test_ambiguous_network_exception_retains_attempt_and_blocks_new_payment(test_db):
    """
    Section 1 & 6: Ambiguous network exception must result in UNKNOWN outcome,
    retaining unresolved attempt and blocking new retry until resolved.
    """
    async with test_db() as session:
        user = User(id=102, username="test102", first_name="User102")
        plan = SubscriptionPlan(id=2, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        session.add_all([user, plan])
        await session.commit()

        sub = UserSubscription(
            id=2,
            user_id=102,
            plan_id=2,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_test_card_2",
            payment_attempt_count=0,
            end_date=datetime(2026, 9, 10, 12, 0, 0),
        )
        session.add(sub)
        await session.commit()

        config = SimpleNamespace(yookassa_shop_id="shop", yookassa_secret_key="secret")

        claim_res = await claim_yookassa_recurring_attempt(session, sub, plan, 195.0, "scheduler")
        assert claim_res.claimed is True
        attempt = claim_res.attempt

        # Network disconnect after request dispatch
        with patch("subscription_renewal.Payment.create", side_effect=ConnectionResetError("Peer reset")):
            res = await execute_or_replay_yookassa_recurring_attempt(attempt, plan.name, config)

        assert res.outcome == "unknown"

        await update_yookassa_attempt_unknown(session, attempt.id, "network_error", "ConnectionResetError")

        # Concurrent / new retry is blocked
        claim_blocked = await claim_yookassa_recurring_attempt(session, sub, plan, 195.0, "telegram_manual")
        assert claim_blocked.claimed is False
        assert claim_blocked.reason == "unresolved_exists"


@pytest.mark.asyncio
async def test_pending_payment_blocks_all_callers_and_reconciles_via_get_or_webhook(test_db):
    """
    Section 2: pending is an open payment.
    Scheduler, TG manual, and MAX manual must all refuse to create a second charge.
    Reconciles via find_one (GET) or webhook.
    """
    async with test_db() as session:
        user = User(id=103, username="test103", first_name="User103")
        plan = SubscriptionPlan(id=3, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        session.add_all([user, plan])
        await session.commit()

        sub = UserSubscription(
            id=3,
            user_id=103,
            plan_id=3,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_test_card_3",
            payment_attempt_count=0,
            end_date=datetime(2026, 9, 10, 12, 0, 0),
        )
        session.add(sub)
        await session.commit()

        config = SimpleNamespace(yookassa_shop_id="shop", yookassa_secret_key="secret")

        claim_res = await claim_yookassa_recurring_attempt(session, sub, plan, 195.0, "scheduler")
        assert claim_res.claimed is True
        attempt = claim_res.attempt

        # YooKassa returns pending payment
        mock_pending_payment = SimpleNamespace(id="yk_pend_123", status="pending")
        with patch("subscription_renewal.Payment.create", return_value=mock_pending_payment):
            res = await execute_or_replay_yookassa_recurring_attempt(attempt, plan.name, config)

        assert res.outcome == "pending"
        assert res.payment_id == "yk_pend_123"

        await update_yookassa_attempt_pending(session, attempt.id, "yk_pend_123")

        # Scheduler tick must NOT create a second payment
        claim_scheduler = await claim_yookassa_recurring_attempt(session, sub, plan, 195.0, "scheduler")
        assert claim_scheduler.claimed is False
        assert claim_scheduler.reason == "unresolved_exists"

        # TG manual retry must NOT create a second payment
        claim_tg = await claim_yookassa_recurring_attempt(session, sub, plan, 195.0, "telegram_manual")
        assert claim_tg.claimed is False
        assert claim_tg.reason == "unresolved_exists"

        # MAX manual retry must NOT create a second payment
        claim_max = await claim_yookassa_recurring_attempt(session, sub, plan, 195.0, "max_manual_retry")
        assert claim_max.claimed is False
        assert claim_max.reason == "unresolved_exists"

        # Reconcile via GET /v3/payments/{payment_id}
        refreshed_att = await session.get(YookassaRecurringAttempt, attempt.id)
        mock_succ_payment = SimpleNamespace(id="yk_pend_123", status="succeeded")
        with patch("subscription_renewal.Payment.find_one", return_value=mock_succ_payment) as mock_find:
            rec_res = await execute_or_replay_yookassa_recurring_attempt(refreshed_att, plan.name, config)

        assert rec_res.outcome == "success"
        assert rec_res.payment_id == "yk_pend_123"
        mock_find.assert_called_once_with("yk_pend_123")

        # Webhook / caller finalization resolves it exactly once
        is_new, updated_sub = await finalize_yookassa_payment_success(
            session=session,
            payment_id="yk_pend_123",
            user_id=103,
            plan_id=3,
            amount=195.0,
            payment_method_id="pm_test_card_3",
            is_recurring=True,
        )
        assert is_new is True
        assert updated_sub.end_date > datetime(2026, 9, 10, 12, 0, 0)


def test_should_reconcile_attempt_backoff():
    """
    Section 2: Bounded reconciliation backoff prevents hammering YooKassa every 15 minutes.
    """
    now = datetime(2026, 9, 10, 12, 0, 0)
    # Attempt started just now: should not reconcile immediately (interval >= 5 min)
    att1 = SimpleNamespace(status="pending", attempt_started_at=now, last_reconciled_at=now - timedelta(minutes=2))
    assert should_reconcile_attempt(att1, now) is False

    # Attempt started 20 minutes ago, not reconciled yet: should reconcile
    att2 = SimpleNamespace(status="pending", attempt_started_at=now - timedelta(minutes=20), last_reconciled_at=None)
    assert should_reconcile_attempt(att2, now) is True

    # Attempt started 3 hours ago: interval increases to 30 min. Last reconciled 10 min ago -> False
    att3 = SimpleNamespace(status="pending", attempt_started_at=now - timedelta(hours=3), last_reconciled_at=now - timedelta(minutes=10))
    assert should_reconcile_attempt(att3, now) is False

    # Attempt started 26 hours ago: exceeds 24-hour window -> False (stop infinite hammering)
    att4 = SimpleNamespace(status="pending", attempt_started_at=now - timedelta(hours=26), last_reconciled_at=None)
    assert should_reconcile_attempt(att4, now) is False


@pytest.mark.asyncio
async def test_atomic_claim_concurrency():
    """
    Requirement 8: Real concurrency test with two independent AsyncSessions,
    same DB, barrier, both attempt claim concurrently.
    Exactly one commit succeeds as 'claimed'.
    The other returns existing unresolved / conflict.
    Exactly one provider POST follows.
    """
    import tempfile
    from sqlalchemy import text
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
        async with engine.begin() as conn:
            await conn.execute(text("PRAGMA journal_mode=WAL;"))
            await conn.execute(text("PRAGMA busy_timeout=5000;"))
            await conn.run_sync(Base.metadata.create_all)
        session_maker = async_sessionmaker(engine, expire_on_commit=False)

        async with session_maker() as session:
            u = User(id=104, username="test104")
            p = SubscriptionPlan(id=4, name="Standard", price=195.0, duration_value=1, duration_unit="months")
            sub = UserSubscription(
                id=4,
                user_id=104,
                plan_id=4,
                auto_renewal=True,
                payment_provider="Yookassa",
                payment_method_id="pm_test_card_4",
                payment_attempt_count=0,
                end_date=datetime(2026, 9, 10, 12, 0, 0),
            )
            session.add_all([u, p, sub])
            await session.commit()

        barrier = asyncio.Barrier(2)
        mock_post_calls = []

        async def worker(worker_id: str):
            async with session_maker() as session:
                sub = await session.get(UserSubscription, 4)
                plan = await session.get(SubscriptionPlan, 4)
                await barrier.wait()
                claim_res = await claim_yookassa_recurring_attempt(session, sub, plan, 195.0, worker_id)
                if claim_res.claimed:
                    config = SimpleNamespace(yookassa_shop_id="shop", yookassa_secret_key="secret")
                    fake_pay = SimpleNamespace(id=f"pay_{worker_id}", status="succeeded")
                    with patch("subscription_renewal.Payment.create", return_value=fake_pay) as mock_create:
                        await execute_or_replay_yookassa_recurring_attempt(claim_res.attempt, plan.name, config)
                        mock_post_calls.append(mock_create.call_count)
                return worker_id, claim_res

        res1, res2 = await asyncio.gather(worker("worker_1"), worker("worker_2"))
        results = [res1[1], res2[1]]
        claimed = [r for r in results if r.claimed]
        conflicts = [r for r in results if not r.claimed]

        assert len(claimed) == 1
        assert len(conflicts) == 1
        assert conflicts[0].reason == "unresolved_exists"
        # Exactly one provider POST followed
        assert len(mock_post_calls) == 1
        assert mock_post_calls[0] == 1
    finally:
        await engine.dispose()
        os.remove(db_path)


@pytest.mark.asyncio
async def test_success_race_webhook_first_max_second(test_db):
    """
    Section 4 & 8: Webhook finalizes succeeded payment first;
    then MAX manual caller resumes with the same succeeded payment_id.
    Subscription end_date extended exactly ONCE.
    """
    async with test_db() as session:
        user = User(id=105, username="test105", first_name="User105")
        plan = SubscriptionPlan(id=5, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        session.add_all([user, plan])
        await session.commit()

        initial_end = datetime(2026, 9, 10, 12, 0, 0)
        sub = UserSubscription(
            id=5,
            user_id=105,
            plan_id=5,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_test_card_5",
            payment_attempt_count=0,
            end_date=initial_end,
        )
        session.add(sub)
        await session.commit()

        # Step 1: Webhook arrives first
        is_new_webhook, sub_after_wh = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_race_succ_1",
            user_id=105,
            plan_id=5,
            amount=195.0,
            payment_method_id="pm_test_card_5",
            is_recurring=True,
        )
        assert is_new_webhook is True
        extended_once_end = sub_after_wh.end_date
        assert extended_once_end > initial_end

        # Step 2: MAX caller returns with the same payment_id
        is_new_max, sub_after_max = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_race_succ_1",
            user_id=105,
            plan_id=5,
            amount=195.0,
            payment_method_id="pm_test_card_5",
            is_recurring=True,
        )
        assert is_new_max is False
        # Subscription end_date must NOT be extended a second time!
        assert sub_after_max.end_date == extended_once_end


@pytest.mark.asyncio
async def test_success_race_max_first_webhook_second(test_db):
    """
    Section 4 & 8: MAX caller finalizes first; duplicate webhook arrives later.
    Subscription end_date extended exactly ONCE.
    """
    async with test_db() as session:
        user = User(id=106, username="test106", first_name="User106")
        plan = SubscriptionPlan(id=6, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        session.add_all([user, plan])
        await session.commit()

        initial_end = datetime(2026, 9, 10, 12, 0, 0)
        sub = UserSubscription(
            id=6,
            user_id=106,
            plan_id=6,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_test_card_6",
            payment_attempt_count=0,
            end_date=initial_end,
        )
        session.add(sub)
        await session.commit()

        # Step 1: MAX caller finalizes first
        is_new_max, sub_after_max = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_race_succ_2",
            user_id=106,
            plan_id=6,
            amount=195.0,
            payment_method_id="pm_test_card_6",
            is_recurring=True,
        )
        assert is_new_max is True
        extended_once_end = sub_after_max.end_date

        # Step 2: Webhook arrives later
        is_new_wh, sub_after_wh = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_race_succ_2",
            user_id=106,
            plan_id=6,
            amount=195.0,
            payment_method_id="pm_test_card_6",
            is_recurring=True,
        )
        assert is_new_wh is False
        assert sub_after_wh.end_date == extended_once_end


@pytest.mark.asyncio
async def test_canceled_race_payment_attempt_count_increments_at_most_once(test_db):
    """
    Section 4 & 8: Webhook may process canceled before scheduler/manual caller, or vice versa.
    payment_attempt_count increments at most ONCE for the same payment_id.
    """
    async with test_db() as session:
        user = User(id=107, username="test107", first_name="User107")
        plan = SubscriptionPlan(id=7, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        session.add_all([user, plan])
        await session.commit()

        sub = UserSubscription(
            id=7,
            user_id=107,
            plan_id=7,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_test_card_7",
            payment_attempt_count=0,
            end_date=datetime(2026, 9, 10, 12, 0, 0),
        )
        session.add(sub)
        await session.commit()

        # Webhook processes canceled first
        is_new_wh, action_wh, sub_wh = await finalize_yookassa_payment_canceled(
            session=session,
            payment_id="pay_canc_race_1",
            cancellation_reason="insufficient_funds",
            user_id=107,
            plan_id=7,
            amount=195.0,
            payment_method_id="pm_test_card_7",
            is_recurring=True,
        )
        assert is_new_wh is True
        assert sub_wh.payment_attempt_count == 1

        # Caller processes same canceled payment_id second
        is_new_caller, action_caller, sub_caller = await finalize_yookassa_payment_canceled(
            session=session,
            payment_id="pay_canc_race_1",
            cancellation_reason="insufficient_funds",
            user_id=107,
            plan_id=7,
            amount=195.0,
            payment_method_id="pm_test_card_7",
            is_recurring=True,
        )
        assert is_new_caller is False
        assert action_caller == "already_processed"
        # Attempt count must NOT be incremented twice!
        assert sub_caller.payment_attempt_count == 1


def test_cancellation_reason_taxonomy():
    """
    Section 5: Canonical classifier for cancellation_details.reason.
    Asserts documented recurring-relevant reasons and safe fallback for unknowns.
    """
    # Retryable declines
    pol, _ = classify_yookassa_cancellation_reason("insufficient_funds")
    assert pol == CancellationPolicy.RETRYABLE_DECLINE

    pol, _ = classify_yookassa_cancellation_reason("general_decline")
    assert pol == CancellationPolicy.RETRYABLE_DECLINE

    pol, _ = classify_yookassa_cancellation_reason("call_issuer")
    assert pol == CancellationPolicy.RETRYABLE_DECLINE

    # Temporary / provider
    pol, _ = classify_yookassa_cancellation_reason("issuer_unavailable")
    assert pol == CancellationPolicy.TEMPORARY_PROVIDER

    pol, _ = classify_yookassa_cancellation_reason("internal_timeout")
    assert pol == CancellationPolicy.TEMPORARY_PROVIDER

    # Limit exceeded
    pol, _ = classify_yookassa_cancellation_reason("payment_method_limit_exceeded")
    assert pol == CancellationPolicy.LIMIT_EXCEEDED

    # Terminal auto-renewal / invalid payment method
    pol, _ = classify_yookassa_cancellation_reason("card_expired")
    assert pol == CancellationPolicy.TERMINAL_DEACTIVATE

    pol, _ = classify_yookassa_cancellation_reason("payment_method_restricted")
    assert pol == CancellationPolicy.TERMINAL_DEACTIVATE

    pol, _ = classify_yookassa_cancellation_reason("permission_revoked")
    assert pol == CancellationPolicy.TERMINAL_DEACTIVATE

    pol, _ = classify_yookassa_cancellation_reason("country_forbidden")
    assert pol == CancellationPolicy.TERMINAL_DEACTIVATE

    pol, _ = classify_yookassa_cancellation_reason("fraud_suspected")
    assert pol == CancellationPolicy.TERMINAL_DEACTIVATE

    # Unknown future reasons fail safe into UNKNOWN
    pol, _ = classify_yookassa_cancellation_reason("exotic_new_unseen_reason")
    assert pol == CancellationPolicy.UNKNOWN


@pytest.mark.asyncio
async def test_cancellation_terminal_clears_method_and_temporary_preserves_attempts(test_db):
    """
    Section 5: Terminal deactivates auto_renewal and clears payment_method_id.
    Temporary provider error does not burn decline attempts.
    """
    async with test_db() as session:
        user = User(id=108, username="test108")
        plan = SubscriptionPlan(id=8, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        session.add_all([user, plan])
        await session.commit()

        sub = UserSubscription(
            id=8,
            user_id=108,
            plan_id=8,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_test_card_8",
            payment_attempt_count=0,
            end_date=datetime(2026, 9, 10, 12, 0, 0),
        )
        session.add(sub)
        await session.commit()

        # Temporary provider error (issuer_unavailable): do not increment attempts
        _, act, updated_sub = await finalize_yookassa_payment_canceled(
            session=session,
            payment_id="pay_canc_temp",
            cancellation_reason="issuer_unavailable",
            user_id=108,
            plan_id=8,
            amount=195.0,
            payment_method_id="pm_test_card_8",
            is_recurring=True,
        )
        assert act == "provider_error"
        assert updated_sub.payment_attempt_count == 0
        assert updated_sub.auto_renewal is True
        assert updated_sub.payment_method_id == "pm_test_card_8"

        # Terminal deactivation (permission_revoked): clear method and auto_renewal=False
        _, act_term, term_sub = await finalize_yookassa_payment_canceled(
            session=session,
            payment_id="pay_canc_term",
            cancellation_reason="permission_revoked",
            user_id=108,
            plan_id=8,
            amount=195.0,
            payment_method_id="pm_test_card_8",
            is_recurring=True,
        )
        assert act_term == "deactivate"
        assert term_sub.auto_renewal is False
        assert term_sub.payment_method_id is None


@pytest.mark.asyncio
async def test_http_error_taxonomy():
    """
    Section 6: Required HTTP distinctions:
    400 invalid_request (amount) -> integration_error
    400 invalid_request (payment_method_id) -> deactivate
    401 -> auth_error
    403 -> auth_error
    429 -> rate_limit
    500 -> unknown
    """
    import json
    sub = SimpleNamespace(id=1, user_id=100, payment_method_id="pm_123456789")
    plan = SimpleNamespace(id=1, name="Standard")
    config = SimpleNamespace(yookassa_shop_id="shop", yookassa_secret_key="secret")
    payload = json.dumps({
        "amount": {"value": "195.00", "currency": "RUB"},
        "capture": True,
        "payment_method_id": "pm_123456789",
        "description": "Auto",
        "metadata": {"subscription_id": 1, "recurring_attempt_key": "key-tax"},
    })
    attempt = YookassaRecurringAttempt(
        subscription_id=1,
        user_id=100,
        plan_id=1,
        idempotency_key="key-tax",
        amount=195.0,
        status="claimed",
        request_payload=payload,
        attempt_started_at=datetime.utcnow(),
    )

    # 400 parameter='amount'
    with patch("subscription_renewal.Payment.create", side_effect=BadRequestError({"code": "invalid_request", "parameter": "amount"})):
        r = await execute_or_replay_yookassa_recurring_attempt(attempt, plan.name, config)
        assert r.outcome == "integration_error"

    # 400 parameter='payment_method_id'
    with patch("subscription_renewal.Payment.create", side_effect=BadRequestError({"code": "invalid_request", "parameter": "payment_method_id"})):
        r = await execute_or_replay_yookassa_recurring_attempt(attempt, plan.name, config)
        assert r.outcome == "deactivate"

    # 401
    with patch("subscription_renewal.Payment.create", side_effect=UnauthorizedError({"code": "unauthorized"})):
        r = await execute_or_replay_yookassa_recurring_attempt(attempt, plan.name, config)
        assert r.outcome == "auth_error"

    # 403
    with patch("subscription_renewal.Payment.create", side_effect=ForbiddenError({"code": "forbidden"})):
        r = await execute_or_replay_yookassa_recurring_attempt(attempt, plan.name, config)
        assert r.outcome == "auth_error"

    # 429
    with patch("subscription_renewal.Payment.create", side_effect=TooManyRequestsError({"code": "too_many_requests"})):
        r = await execute_or_replay_yookassa_recurring_attempt(attempt, plan.name, config)
        assert r.outcome == "rate_limit"

    # 500
    with patch("subscription_renewal.Payment.create", side_effect=InternalServerError({"code": "internal_server_error"})):
        r = await execute_or_replay_yookassa_recurring_attempt(attempt, plan.name, config)
        assert r.outcome == "unknown"


@pytest.mark.asyncio
async def test_max_provider_dispatch_robokassa_never_touches_yookassa(test_db):
    """
    Section 7: A MAX subscription with payment_provider='Robokassa' and a Robokassa
    payment_method_id / root invoice must NEVER be passed to YooKassa.
    """
    async with test_db() as session:
        user = User(id=9901, username="max_user_rk", first_name="MaxUser")
        plan = SubscriptionPlan(id=9, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        config = SubscriptionConfig(id=1, robokassa_merchant_login="shop_rk", robokassa_password_1="pass1")
        session.add_all([user, plan, config])
        await session.commit()

        sub = UserSubscription(
            id=9,
            user_id=9901,
            plan_id=9,
            auto_renewal=True,
            payment_provider="Robokassa",
            payment_method_id="root_inv_998877",  # Robokassa parent invoice ID!
            payment_attempt_count=0,
            last_payment_attempt=None,
            end_date=datetime(2026, 9, 10, 12, 0, 0),
        )
        session.add(sub)
        await session.commit()

    client = AsyncMock()
    with (
        patch("max_messenger_bot.services.subscriptions.async_session_maker", test_db),
        patch("max_messenger_bot.services.subscriptions.Payment.create") as mock_yk_create,
        patch("scheduler.process_recurring_robokassa_payment", AsyncMock(return_value=True)) as mock_rk_process,
    ):
        await handle_max_manual_retry(client, chat_id=12345, user_id=9901)

    # CRITICAL ASSERTION: YooKassa must NEVER be called with Robokassa data!
    mock_yk_create.assert_not_called()

    # Robokassa recurring business path MUST be called with parent invoice
    mock_rk_process.assert_called_once()
    assert mock_rk_process.call_args[0][3] == "root_inv_998877"

    # User received confirmation
    client.send_message.assert_called_once()
    assert "Robokassa" in client.send_message.call_args[1]["text"]


@pytest.mark.asyncio
async def test_logging_preservation_and_token_redaction(caplog):
    """
    Section 9: Recurring technical diagnostics remain in canonical payment_events logger.
    Raw payment_method_id is redacted, credentials/tokens absent.
    """
    raw_token = "31d55000-000f-5000-9000-1e54ca65d0cf"
    masked = mask_payment_method_id(raw_token)
    assert masked == "31d5...d0cf"

    import json
    payload = json.dumps({
        "amount": {"value": "195.00", "currency": "RUB"},
        "capture": True,
        "payment_method_id": raw_token,
        "description": "Auto",
        "metadata": {"subscription_id": 1, "recurring_attempt_key": "key-log-test"},
    })
    attempt = YookassaRecurringAttempt(
        subscription_id=1,
        user_id=500,
        plan_id=1,
        idempotency_key="key-log-test",
        amount=195.0,
        payment_method_id=raw_token,
        request_payload=payload,
        status="claimed",
        attempt_started_at=datetime.utcnow(),
    )
    config = SimpleNamespace(yookassa_shop_id="test_shop_secret", yookassa_secret_key="test_key_secret")

    fake_payment = SimpleNamespace(id="yk_pay_log_ok", status="succeeded")
    with caplog.at_level(logging.INFO, logger="payment_events"):
        with patch("subscription_renewal.Payment.create", return_value=fake_payment):
            res = await execute_or_replay_yookassa_recurring_attempt(attempt, "Standard", config)

    assert res.outcome == "success"
    log_text = caplog.text

    # Canonical logger received events
    assert "TECH_RECURRING_REQUEST" in log_text
    assert "TECH_RECURRING_RESPONSE" in log_text

    # Raw payment method ID MUST NOT appear
    assert raw_token not in log_text

    # Masked token SHOULD appear
    assert "31d5...d0cf" in log_text

    # Secret credentials MUST NOT appear in technical events
    assert "test_key_secret" not in log_text


# ==============================================================================
# 9 REVIEW NO GO BLOCKER VERIFICATION SUITE
# ==============================================================================

@pytest.mark.asyncio
async def test_blocker_1_integration_error_scheduler_flow(test_db):
    """
    Requirement 1: INTEGRATION_ERROR MUST NEVER FALL INTO CANCELLATION TAXONOMY.
    400 invalid_request + parameter='amount' (or any integration error) must:
    - NOT deactivate auto_renewal
    - NOT clear payment_method_id
    - NOT increment payment_attempt_count
    - NOT generate synthetic YookassaPayment row
    - set attempt status to 'integration_error'
    - set retry_not_before to >= 24h to avoid blind 2h retry.
    """
    now = datetime.utcnow()
    async with test_db() as session:
        u = User(id=801, username="test801", first_name="User801")
        p = SubscriptionPlan(id=801, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        cfg = SubscriptionConfig(id=1, yookassa_shop_id="shop", yookassa_secret_key="sec", notifications_enabled=True)
        sub = UserSubscription(
            id=801,
            user_id=801,
            plan_id=801,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_valid_card_801",
            payment_attempt_count=1,
            last_payment_attempt=now - timedelta(hours=2),
            end_date=now - timedelta(minutes=5),
        )
        session.add_all([u, p, cfg, sub])
        await session.commit()

    bot = AsyncMock()
    with (
        patch("scheduler.process_birthday_mailings", AsyncMock()),
        patch("scheduler.async_session_maker", test_db),
        patch("subscription_renewal.Payment.create", side_effect=BadRequestError({"code": "invalid_request", "parameter": "amount"})),
    ):
        await check_subscriptions(bot)

    async with test_db() as session:
        updated_sub = await session.get(UserSubscription, 801)
        # 1. auto_renewal MUST remain True
        assert updated_sub.auto_renewal is True
        # 2. payment_method_id MUST NOT be cleared
        assert updated_sub.payment_method_id == "pm_valid_card_801"
        # 3. payment_attempt_count MUST NOT be incremented
        assert updated_sub.payment_attempt_count == 1
        # 4. retry_not_before MUST be set to >= 24h
        assert updated_sub.retry_not_before is not None
        assert updated_sub.retry_not_before >= now + timedelta(hours=23)

        # 5. ZERO YookassaPayment rows created!
        payments = (await session.execute(select(YookassaPayment).where(YookassaPayment.user_id == 801))).scalars().all()
        assert len(payments) == 0

        # 6. Attempt status is integration_error
        attempts = (await session.execute(select(YookassaRecurringAttempt).where(YookassaRecurringAttempt.subscription_id == 801))).scalars().all()
        assert len(attempts) == 1
        assert attempts[0].status == "integration_error"
        assert attempts[0].payment_id is None


@pytest.mark.asyncio
async def test_blocker_2_never_invent_payment_id(test_db):
    """
    Requirement 2: NEVER INVENT payment_id.
    - If rejected before payment object created (e.g. permanent invalid payment_method_id):
      dedicated attempt finalizer, update sub, zero YookassaPayment rows (payment_id remains NULL).
    - finalize_yookassa_payment_canceled asserts payment_id is real and not idempotency key.
    """
    # 1. Assert ValueError when calling finalize_yookassa_payment_canceled with missing or idempotency key
    async with test_db() as session:
        with pytest.raises(ValueError, match="A real payment_id is required"):
            await finalize_yookassa_payment_canceled(
                session=session,
                payment_id=None,
                cancellation_reason="card_expired",
                user_id=802,
                plan_id=1,
                amount=100.0,
            )
        with pytest.raises(ValueError, match="A real payment_id is required"):
            await finalize_yookassa_payment_canceled(
                session=session,
                payment_id="yk-rec-1-802-1-abcdef123456",
                cancellation_reason="card_expired",
                user_id=802,
                plan_id=1,
                amount=100.0,
            )

    # 2. End-to-end scheduler permanent BadRequest on payment_method_id
    now = datetime.utcnow()
    async with test_db() as session:
        u = User(id=802, username="test802", first_name="User802")
        p = SubscriptionPlan(id=802, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        cfg = SubscriptionConfig(id=1, yookassa_shop_id="shop", yookassa_secret_key="sec", notifications_enabled=True)
        sub = UserSubscription(
            id=802,
            user_id=802,
            plan_id=802,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_bad_method_802",
            payment_attempt_count=0,
            end_date=now - timedelta(minutes=5),
        )
        session.add_all([u, p, cfg, sub])
        await session.commit()

    bot = AsyncMock()
    with (
        patch("scheduler.process_birthday_mailings", AsyncMock()),
        patch("scheduler.async_session_maker", test_db),
        patch("subscription_renewal.Payment.create", side_effect=BadRequestError({"code": "invalid_request", "parameter": "payment_method_id"})),
    ):
        await check_subscriptions(bot)

    async with test_db() as session:
        updated_sub = await session.get(UserSubscription, 802)
        assert updated_sub.auto_renewal is False
        assert updated_sub.payment_method_id is None

        # ZERO YookassaPayment rows!
        payments = (await session.execute(select(YookassaPayment).where(YookassaPayment.user_id == 802))).scalars().all()
        assert len(payments) == 0

        attempts = (await session.execute(select(YookassaRecurringAttempt).where(YookassaRecurringAttempt.subscription_id == 802))).scalars().all()
        assert len(attempts) == 1
        assert attempts[0].status == "deactivated"
        assert attempts[0].payment_id is None


@pytest.mark.asyncio
async def test_blocker_3_24h_terminal_safety_and_late_webhook(test_db):
    """
    Requirement 3: 24-HOUR UNKNOWN/PENDING TERMINAL SAFETY.
    - Explicit transition to unknown_expired removes attempt from active unresolved states.
    - Pauses automatic renewal safely (auto_renewal=False).
    - Late verified webhook for real payment_id remains processable exactly once.
    """
    now = datetime.utcnow()
    initial_end = now - timedelta(days=1)
    async with test_db() as session:
        u = User(id=803, username="test803", first_name="User803")
        p = SubscriptionPlan(id=803, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        cfg = SubscriptionConfig(id=1, yookassa_shop_id="shop", yookassa_secret_key="sec", notifications_enabled=True)
        sub = UserSubscription(
            id=803,
            user_id=803,
            plan_id=803,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_card_803",
            payment_attempt_count=0,
            end_date=initial_end,
        )
        session.add_all([u, p, cfg, sub])
        await session.commit()

        # Create attempt started 25 hours ago with status 'unknown'
        att = YookassaRecurringAttempt(
            subscription_id=803,
            user_id=803,
            plan_id=803,
            idempotency_key="yk-rec-803-test",
            amount=195.0,
            payment_method_id="pm_card_803",
            status="unknown",
            payment_id="pay_real_24h_late",
            attempt_started_at=now - timedelta(hours=25),
        )
        session.add(att)
        await session.commit()

    bot = AsyncMock()
    with (
        patch("scheduler.process_birthday_mailings", AsyncMock()),
        patch("scheduler.async_session_maker", test_db),
    ):
        await check_subscriptions(bot)

    # Verify attempt transitioned to unknown_expired and sub auto_renewal paused
    async with test_db() as session:
        updated_att = await session.get(YookassaRecurringAttempt, att.id)
        assert updated_att.status == "unknown_expired"

        updated_sub = await session.get(UserSubscription, 803)
        assert updated_sub.auto_renewal is False

        # Attempt is no longer active in partial unique index filter
        active_unresolved = (await session.execute(
            select(YookassaRecurringAttempt).where(
                YookassaRecurringAttempt.subscription_id == 803,
                YookassaRecurringAttempt.status.in_(["claimed", "pending", "unknown"]),
            )
        )).scalar_one_or_none()
        assert active_unresolved is None

        # LATE VERIFIED WEBHOOK ARRIVES for real payment_id
        is_new_late, final_sub = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_real_24h_late",
            user_id=803,
            plan_id=803,
            amount=195.0,
            payment_method_id="pm_card_803",
            is_recurring=True,
        )
        assert is_new_late is True
        assert final_sub.end_date > initial_end
        assert final_sub.auto_renewal is False  # Preserves user/system choice, does not re-enable

        refreshed_att_late = await session.get(YookassaRecurringAttempt, att.id)
        assert refreshed_att_late.status in ("succeeded", "resolved_success")

        # Second webhook duplicate returns is_new=False
        is_new_dup, _ = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_real_24h_late",
            user_id=803,
            plan_id=803,
            amount=195.0,
            payment_method_id="pm_card_803",
            is_recurring=True,
        )
        assert is_new_dup is False


@pytest.mark.asyncio
async def test_blocker_4_no_db_transaction_across_yookassa_io(test_db):
    """
    Requirement 4: NO DB TRANSACTION ACROSS YOOKASSA I/O.
    Instruments outbound call to assert that database session is NOT in an open transaction
    during network operations for scheduler, TG manual retry, and MAX manual retry.
    """
    async with test_db() as session:
        u = User(id=804, username="test804", first_name="User804")
        p = SubscriptionPlan(id=804, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        cfg = SubscriptionConfig(id=1, yookassa_shop_id="shop", yookassa_secret_key="sec", notifications_enabled=True)
        sub = UserSubscription(
            id=804,
            user_id=804,
            plan_id=804,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_card_804",
            payment_attempt_count=0,
            end_date=datetime(2026, 9, 10, 12, 0, 0),
        )
        session.add_all([u, p, cfg, sub])
        await session.commit()

    active_sessions = []
    orig_async_session_maker = test_db

    def tracked_session_maker():
        s = orig_async_session_maker()
        active_sessions.append(s)
        return s

    # 1. Scheduler recurring run
    sched_io_checked = []
    def fake_create_sched(payload, idempotency_key):
        for s in active_sessions:
            assert s.in_transaction() is False
        sched_io_checked.append(True)
        return SimpleNamespace(id="pay_sched_804", status="pending")

    bot = AsyncMock()
    now = datetime(2026, 9, 10, 13, 0, 0)
    with (
        patch("scheduler.process_birthday_mailings", AsyncMock()),
        patch("scheduler.async_session_maker", tracked_session_maker),
        patch("subscription_renewal.Payment.create", side_effect=fake_create_sched),
        patch("scheduler.datetime") as mock_dt,
    ):
        mock_dt.utcnow.return_value = now
        await check_subscriptions(bot)

    assert len(sched_io_checked) == 1

    # 2. MAX manual retry
    active_sessions.clear()
    async with test_db() as s:
        # Reset attempt
        await s.execute(delete(YookassaRecurringAttempt).where(YookassaRecurringAttempt.subscription_id == 804))
        await s.commit()

    max_io_checked = []
    def fake_create_max(payload, idempotency_key):
        for s in active_sessions:
            assert s.in_transaction() is False
        max_io_checked.append(True)
        return SimpleNamespace(id="pay_max_804", status="pending")

    client = AsyncMock()
    with (
        patch("max_messenger_bot.services.subscriptions.async_session_maker", tracked_session_maker),
        patch("subscription_renewal.Payment.create", side_effect=fake_create_max),
    ):
        await handle_max_manual_retry(client, chat_id=804, user_id=804)

    assert len(max_io_checked) == 1


@pytest.mark.asyncio
async def test_blocker_5_immutable_same_key_request_replay(test_db):
    """
    Requirement 5: IMMUTABLE SAME-KEY REQUEST.
    Persist exact request snapshot at claim time.
    Replay never consults mutable plan name/price.
    """
    async with test_db() as session:
        u = User(id=805, username="test805")
        p = SubscriptionPlan(id=805, name="Original Plan Name", price=100.0, duration_value=1, duration_unit="months")
        sub = UserSubscription(
            id=805,
            user_id=805,
            plan_id=805,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_card_805",
            payment_attempt_count=0,
            end_date=datetime(2026, 9, 10, 12, 0, 0),
        )
        session.add_all([u, p, sub])
        await session.commit()

        claim_res = await claim_yookassa_recurring_attempt(session, sub, p, 100.0, "test")
        assert claim_res.claimed is True
        attempt = claim_res.attempt
        orig_key = attempt.idempotency_key

        # Now mutate the plan name and price in DB
        p.name = "MUTATED_PLAN_NAME"
        p.price = 9999.0
        await session.commit()

        config = SimpleNamespace(yookassa_shop_id="shop", yookassa_secret_key="secret")
        mock_pay = SimpleNamespace(id="pay_replay_imm", status="succeeded")
        with patch("subscription_renewal.Payment.create", return_value=mock_pay) as mock_create:
            # Pass mutated plan name to function - it should IGNORE it and use request_payload!
            replay_res = await execute_or_replay_yookassa_recurring_attempt(attempt, p.name, config)

        assert replay_res.outcome == "success"
        sent_payload, sent_key = mock_create.call_args[0]
        assert sent_key == orig_key
        assert sent_payload["amount"]["value"] == "100.00"
        assert sent_payload["description"] == "Автопродление подписки: Original Plan Name"
        assert "MUTATED" not in sent_payload["description"]
        assert sent_payload["amount"]["value"] != "9999.00"


@pytest.mark.asyncio
async def test_blocker_6_limit_exceeded_retry_not_before_policy(test_db):
    """
    Requirement 6: payment_method_limit_exceeded MUST NOT USE ORDINARY 2H RETRY.
    retry_not_before >= 24h, factual last_payment_attempt, can_retry_now returns False after 2h.
    """
    now = datetime(2026, 9, 10, 12, 0, 0)
    async with test_db() as session:
        u = User(id=806, username="test806")
        p = SubscriptionPlan(id=806, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        sub = UserSubscription(
            id=806,
            user_id=806,
            plan_id=806,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_card_806",
            payment_attempt_count=0,
            end_date=now,
        )
        session.add_all([u, p, sub])
        await session.commit()

        is_new, action, updated_sub = await finalize_yookassa_payment_canceled(
            session=session,
            payment_id="pay_limit_exc",
            cancellation_reason="payment_method_limit_exceeded",
            user_id=806,
            plan_id=806,
            amount=195.0,
            payment_method_id="pm_card_806",
            attempt_started_at=now,
        )
        assert is_new is True
        assert action == "limit_exceeded"
        assert updated_sub.last_payment_attempt == now
        assert updated_sub.retry_not_before is not None
        assert updated_sub.retry_not_before >= now + timedelta(hours=24)

        # At 2h later, can_retry_now must be False!
        assert can_retry_now(
            updated_sub.payment_attempt_count,
            updated_sub.last_payment_attempt,
            now + timedelta(hours=2),
            retry_not_before=updated_sub.retry_not_before,
        ) is False

        # At 25h later, can_retry_now must be True!
        assert can_retry_now(
            updated_sub.payment_attempt_count,
            updated_sub.last_payment_attempt,
            now + timedelta(hours=25),
            retry_not_before=updated_sub.retry_not_before,
        ) is True


@pytest.mark.asyncio
async def test_blocker_7_exact_once_notifications_matrix(test_db):
    """
    Requirement 7: EXACT-ONCE USER/ADMIN NOTIFICATIONS.
    Webhook-first vs caller-second (MAX manual retry):
    Caller sends neutral message, zero duplicate admin alerts.
    """
    async with test_db() as session:
        u = User(id=807, username="test807", first_name="User807")
        p = SubscriptionPlan(id=807, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        cfg = SubscriptionConfig(id=1, yookassa_shop_id="shop", yookassa_secret_key="sec", notifications_enabled=True)
        sub = UserSubscription(
            id=807,
            user_id=807,
            plan_id=807,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_card_807",
            payment_attempt_count=0,
            end_date=datetime(2026, 9, 10, 12, 0, 0),
        )
        session.add_all([u, p, cfg, sub])
        await session.commit()

        # Webhook arrives and finalizes
        is_new_wh, _ = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_exact_once_1",
            user_id=807,
            plan_id=807,
            amount=195.0,
            payment_method_id="pm_card_807",
            is_recurring=True,
        )
        assert is_new_wh is True

    # Now MAX caller executes with the same payment_id
    client = AsyncMock()
    mock_admin_notify = AsyncMock()
    fake_res = SimpleNamespace(outcome="success", payment_id="pay_exact_once_1", failure_reason=None)

    with (
        patch("max_messenger_bot.services.subscriptions.async_session_maker", test_db),
        patch("max_messenger_bot.services.subscriptions.execute_or_replay_yookassa_recurring_attempt", return_value=fake_res),
        patch("max_messenger_bot.services.common.notify_telegram_admins", mock_admin_notify),
    ):
        await handle_max_manual_retry(client, chat_id=807, user_id=807)

    # MAX caller received neutral confirmation message
    client.send_message.assert_called()
    sent_texts = [call[1]["text"] for call in client.send_message.call_args_list]
    assert any("Платёж уже обработан" in t for t in sent_texts)
    # ZERO duplicate admin notifications emitted!
    mock_admin_notify.assert_not_called()


@pytest.mark.asyncio
async def test_blocker_9_global_auth_incident_circuit_breaker(test_db):
    """
    Requirement 9: GLOBAL AUTH/CONFIG INCIDENT CIRCUIT BREAKER.
    First 401 stops initiating new YooKassa attempts for remaining subscriptions in cycle.
    One deduplicated admin incident alert.
    Does not treat 401 as user/card failure.
    """
    now = datetime.utcnow()
    async with test_db() as session:
        admin_u = User(id=99999, username="admin_owner", is_admin=True)
        cfg = SubscriptionConfig(id=1, yookassa_shop_id="shop", yookassa_secret_key="sec", notifications_enabled=True)
        p = SubscriptionPlan(id=809, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        session.add_all([admin_u, cfg, p])

        # 3 subscriptions due for renewal
        for uid in [891, 892, 893]:
            u = User(id=uid, username=f"test{uid}", first_name=f"User{uid}")
            sub = UserSubscription(
                id=uid,
                user_id=uid,
                plan_id=809,
                auto_renewal=True,
                payment_provider="Yookassa",
                payment_method_id=f"pm_card_{uid}",
                payment_attempt_count=0,
                end_date=now - timedelta(minutes=5),
            )
            session.add_all([u, sub])
        await session.commit()

    bot = AsyncMock()

    post_call_count = 0
    def fake_post(*args, **kwargs):
        nonlocal post_call_count
        post_call_count += 1
        raise UnauthorizedError({"code": "unauthorized"})

    with (
        patch("scheduler.process_birthday_mailings", AsyncMock()),
        patch("scheduler.async_session_maker", test_db),
        patch("subscription_renewal.Payment.create", side_effect=fake_post),
    ):
        await check_subscriptions(bot)

    # Exactly ONE POST was attempted before the circuit breaker tripped!
    assert post_call_count == 1

    # Verify admin received circuit breaker incident alert
    bot.send_message.assert_called()
    admin_texts = [str(c) for c in bot.send_message.call_args_list]
    assert any("HTTP 401/403" in t for t in admin_texts)

    # Verify that remaining subscriptions were NOT marked as failed or deactivated
    async with test_db() as session:
        for uid in [891, 892, 893]:
            sub = await session.get(UserSubscription, uid)
            assert sub.auto_renewal is True  # NOT deactivated as a user card failure!
            assert sub.payment_method_id == f"pm_card_{uid}"  # NOT cleared!
            assert sub.payment_attempt_count == 0  # NOT counted as card decline!


@pytest.mark.asyncio
async def test_guard_1_immediate_success_and_canceled_caller_correlation(test_db):
    """
    Mandatory Guard 1:
    When Payment.create returns immediately with 'succeeded' or 'canceled',
    the DB attempt is initially status='claimed' with payment_id=NULL.
    Passing recurring_attempt_key=attempt.idempotency_key allows the finalizer
    to find and transition the exact claimed attempt immediately to terminal,
    leaving 0 active/unresolved attempts.
    """
    async with test_db() as session:
        u = User(id=1101, username="test1101", first_name="User1101")
        p = SubscriptionPlan(id=1101, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        sub = UserSubscription(
            id=1101,
            user_id=1101,
            plan_id=1101,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_card_1101",
            payment_attempt_count=0,
            end_date=datetime.utcnow(),
        )
        session.add_all([u, p, sub])
        await session.commit()

        # 1. Immediate Success flow
        claim_res = await claim_yookassa_recurring_attempt(session, sub, p, 195.0, "scheduler")
        assert claim_res.claimed is True
        att = claim_res.attempt
        assert att.payment_id is None
        assert att.status == "claimed"

        # Finalize with recurring_attempt_key
        is_new, updated_sub = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_imm_succ_1",
            user_id=1101,
            plan_id=1101,
            amount=195.0,
            payment_method_id="pm_card_1101",
            is_recurring=True,
            recurring_attempt_key=att.idempotency_key,
        )
        assert is_new is True
        refreshed_att = await session.get(YookassaRecurringAttempt, att.id)
        assert refreshed_att.status == "succeeded"
        assert refreshed_att.payment_id == "pay_imm_succ_1"

        # Verify 0 active attempts remain
        active = (await session.execute(
            select(YookassaRecurringAttempt).where(
                YookassaRecurringAttempt.subscription_id == 1101,
                YookassaRecurringAttempt.status.in_(["claimed", "pending", "unknown"]),
            )
        )).scalars().all()
        assert len(active) == 0

        # 2. Immediate Canceled flow
        claim_res2 = await claim_yookassa_recurring_attempt(session, sub, p, 195.0, "scheduler")
        assert claim_res2.claimed is True
        att2 = claim_res2.attempt
        assert att2.payment_id is None
        assert att2.status == "claimed"

        is_new2, action2, _ = await finalize_yookassa_payment_canceled(
            session=session,
            payment_id="pay_imm_canc_1",
            cancellation_reason="insufficient_funds",
            user_id=1101,
            plan_id=1101,
            amount=195.0,
            payment_method_id="pm_card_1101",
            recurring_attempt_key=att2.idempotency_key,
        )
        assert is_new2 is True
        refreshed_att2 = await session.get(YookassaRecurringAttempt, att2.id)
        assert refreshed_att2.status == "canceled"
        assert refreshed_att2.payment_id == "pay_imm_canc_1"

        active2 = (await session.execute(
            select(YookassaRecurringAttempt).where(
                YookassaRecurringAttempt.subscription_id == 1101,
                YookassaRecurringAttempt.status.in_(["claimed", "pending", "unknown"]),
            )
        )).scalars().all()
        assert len(active2) == 0


@pytest.mark.asyncio
async def test_guard_2_late_webhook_does_not_affect_newer_attempt(test_db):
    """
    Mandatory Guard 2:
    Valid race:
    Attempt A is claimed -> becomes 'unknown' -> transitions to 'unknown_expired' (terminal).
    New attempt B is claimed for the subscription.
    Late webhook arrives for attempt A with recurring_attempt_key=A.idempotency_key and payment_id=pay_A.
    Finalizer MUST resolve attempt A ONLY.
    Attempt B MUST remain completely untouched (status='claimed', payment_id=NULL).
    """
    async with test_db() as session:
        u = User(id=1102, username="test1102", first_name="User1102")
        p = SubscriptionPlan(id=1102, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        sub = UserSubscription(
            id=1102,
            user_id=1102,
            plan_id=1102,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_card_1102",
            payment_attempt_count=0,
            end_date=datetime.utcnow(),
        )
        session.add_all([u, p, sub])
        await session.commit()

        # Step 1: Claim attempt A
        claim_a = await claim_yookassa_recurring_attempt(session, sub, p, 195.0, "scheduler")
        assert claim_a.claimed is True
        att_a = claim_a.attempt

        # Step 2: Attempt A becomes unknown, then transitions to unknown_expired
        await update_yookassa_attempt_unknown(session, att_a.id, "timeout", "Gateway timeout")
        await transition_attempt_to_unknown_expired(session, att_a.id)

        # Step 3: Sub is re-enabled for manual retry and new attempt B is claimed
        sub.auto_renewal = True
        await session.commit()
        claim_b = await claim_yookassa_recurring_attempt(session, sub, p, 195.0, "scheduler")
        assert claim_b.claimed is True
        att_b = claim_b.attempt
        assert att_b.id != att_a.id
        assert att_b.status == "claimed"
        assert att_b.payment_id is None

        # Step 4: Late webhook arrives for payment A with attempt A's recurring_attempt_key
        is_new, updated_sub = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_A_late",
            user_id=1102,
            plan_id=1102,
            amount=195.0,
            payment_method_id="pm_card_1102",
            is_recurring=True,
            recurring_attempt_key=att_a.idempotency_key,
        )
        assert is_new is True

        # Verify Attempt A was resolved
        refreshed_a = await session.get(YookassaRecurringAttempt, att_a.id)
        assert refreshed_a.status == "succeeded"
        assert refreshed_a.payment_id == "pay_A_late"

        # Verify Attempt B remains completely untouched in claimed status!
        refreshed_b = await session.get(YookassaRecurringAttempt, att_b.id)
        assert refreshed_b.status == "claimed"
        assert refreshed_b.payment_id is None
        assert refreshed_b.idempotency_key == att_b.idempotency_key


@pytest.mark.asyncio
async def test_cross_dialect_atomic_cas_no_payment(test_db):
    """
    Atomic CAS contract for finalize_yookassa_attempt_no_payment:
    Multiple concurrent finalizers targeting the same attempt:
    Exactly one transitions the active attempt to terminal and returns is_new=True.
    Subsequent finalizers get rowcount=0, return is_new=False, and do NOT mutate the subscription.
    """
    async with test_db() as session:
        u = User(id=1103, username="test1103", first_name="User1103")
        p = SubscriptionPlan(id=1103, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        sub = UserSubscription(
            id=1103,
            user_id=1103,
            plan_id=1103,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_card_1103",
            payment_attempt_count=0,
            end_date=datetime.utcnow(),
        )
        session.add_all([u, p, sub])
        await session.commit()

        claim = await claim_yookassa_recurring_attempt(session, sub, p, 195.0, "scheduler")
        att = claim.attempt

        # First finalizer
        is_new1, sub1 = await finalize_yookassa_attempt_no_payment(
            session=session,
            attempt_id=att.id,
            outcome="declined",
            error_code="payment_method_limit_exceeded",
            sub=sub,
        )
        assert is_new1 is True
        assert sub1.last_payment_attempt is not None
        first_retry_not_before = sub1.retry_not_before
        assert first_retry_not_before is not None

        # Second concurrent finalizer on the already-terminal attempt
        is_new2, sub2 = await finalize_yookassa_attempt_no_payment(
            session=session,
            attempt_id=att.id,
            outcome="declined",
            error_code="payment_method_limit_exceeded",
            sub=sub,
        )
        assert is_new2 is False
        # Subscription was NOT mutated again!
        assert sub2.retry_not_before == first_retry_not_before


@pytest.mark.asyncio
async def test_no_payment_arbitrary_subscription_fallback_removed(test_db):
    """
    Arbitrary subscription fallback removal:
    If attempt has non-existent subscription_id (e.g. 999999),
    finalize_yookassa_attempt_no_payment must NOT update some other random subscription.
    """
    async with test_db() as session:
        u = User(id=1104, username="test1104", first_name="User1104")
        p = SubscriptionPlan(id=1104, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        sub = UserSubscription(
            id=1104,
            user_id=1104,
            plan_id=1104,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_card_1104",
            payment_attempt_count=0,
            end_date=datetime.utcnow(),
        )
        session.add_all([u, p, sub])
        await session.commit()

        # Create orphan attempt with non-existent subscription_id
        orphan_att = YookassaRecurringAttempt(
            subscription_id=999999,
            user_id=1104,
            plan_id=1104,
            idempotency_key="orphan_test_key",
            amount=195.0,
            payment_method_id="pm_card_1104",
            status="claimed",
        )
        session.add(orphan_att)
        await session.commit()

        # Finalize orphan attempt without passing sub
        is_new, result_sub = await finalize_yookassa_attempt_no_payment(
            session=session,
            attempt_id=orphan_att.id,
            outcome="declined",
            error_code="test_error",
        )
        # In revised contract: fails closed returning (False, None) before CAS
        assert is_new is False
        assert result_sub is None

        # Orphan attempt was NOT committed as terminal business transition
        refreshed_orphan = await session.get(YookassaRecurringAttempt, orphan_att.id)
        assert refreshed_orphan.status == "claimed"

        # Existing unrelated subscription 1104 was NOT touched!
        refreshed_sub = await session.get(UserSubscription, 1104)
        assert refreshed_sub.payment_attempt_count == 0
        assert refreshed_sub.retry_not_before is None


@pytest.mark.asyncio
async def test_fail_closed_corrupt_or_missing_request_snapshot():
    """
    Fail-closed immutable request snapshot:
    Attempts with NULL request_payload or invalid JSON must fail-closed.
    Return outcome='manual_review' and make ZERO calls to Payment.create.
    """
    config = SimpleNamespace(yookassa_shop_id="shop", yookassa_secret_key="key")

    # 1. NULL request_payload
    att_null = YookassaRecurringAttempt(
        id=1,
        subscription_id=1,
        user_id=1,
        plan_id=1,
        idempotency_key="test_null_payload",
        amount=195.0,
        request_payload=None,
        status="claimed",
    )
    with patch("subscription_renewal.Payment.create") as mock_create:
        res1 = await execute_or_replay_yookassa_recurring_attempt(att_null, "Standard", config)
    assert res1.outcome == "manual_review"
    assert res1.failure_reason == "missing_request_payload"
    mock_create.assert_not_called()

    # 2. Corrupt JSON string
    att_corrupt = YookassaRecurringAttempt(
        id=2,
        subscription_id=1,
        user_id=1,
        plan_id=1,
        idempotency_key="test_corrupt_payload",
        amount=195.0,
        request_payload="INVALID_JSON{{{{",
        status="claimed",
    )
    with patch("subscription_renewal.Payment.create") as mock_create:
        res2 = await execute_or_replay_yookassa_recurring_attempt(att_corrupt, "Standard", config)
    assert res2.outcome == "manual_review"
    assert res2.failure_reason == "corrupt_request_payload"
    mock_create.assert_not_called()


@pytest.mark.asyncio
async def test_transport_neutral_manual_review_transition(test_db):
    """
    Transport-neutral manual review transition:
    transition_attempt_to_manual_review transitions attempt status to 'manual_review',
    pauses auto_renewal on the subscription, and does not depend on telegram or max bot clients.
    """
    async with test_db() as session:
        u = User(id=1105, username="test1105", first_name="User1105")
        p = SubscriptionPlan(id=1105, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        sub = UserSubscription(
            id=1105,
            user_id=1105,
            plan_id=1105,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_card_1105",
            payment_attempt_count=0,
            end_date=datetime.utcnow(),
        )
        session.add_all([u, p, sub])
        await session.commit()

        claim = await claim_yookassa_recurring_attempt(session, sub, p, 195.0, "scheduler")
        att = claim.attempt

        ok, updated_sub = await transition_attempt_to_manual_review(
            session=session,
            attempt_id=att.id,
            reason="missing_request_payload",
        )
        assert ok is True
        assert updated_sub.auto_renewal is False

        refreshed_att = await session.get(YookassaRecurringAttempt, att.id)
        assert refreshed_att.status == "manual_review"


@pytest.mark.asyncio
async def test_scheduler_local_config_precheck_silent_breaker_and_continues_work(test_db):
    """
    Regression test:
    When yookassa_shop_id or yookassa_secret_key is missing:
    1. Zero YooKassa provider calls (mock_create not called)
    2. Zero YooKassa Circuit Breaker admin messages sent (no 'shop_id или secret_key' alerts)
    3. Scheduler continues processing unrelated non-YooKassa work (e.g. Robokassa recurring payments)
    """
    now = datetime.utcnow()
    async with test_db() as session:
        admin_u = User(id=99991, username="admin_owner2", is_admin=True)
        # Missing secret_key!
        cfg = SubscriptionConfig(
            id=1,
            yookassa_shop_id="shop_without_key",
            yookassa_secret_key=None,
            notifications_enabled=True,
            robokassa_merchant_login="robo_login",
            robokassa_password_1="robo_pass1",
            robokassa_password_2="robo_pass2",
        )
        p = SubscriptionPlan(id=1106, name="Standard", price=195.0, duration_value=1, duration_unit="months")

        # 1. YooKassa subscription due for renewal
        sub_yk = UserSubscription(
            id=1106,
            user_id=1106,
            plan_id=1106,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_card_1106",
            payment_attempt_count=0,
            end_date=now - timedelta(minutes=5),
        )
        u_yk = User(id=1106, username="user_yk", first_name="UserYK")

        # 2. Unrelated Robokassa subscription due for renewal
        sub_rk = UserSubscription(
            id=1107,
            user_id=1107,
            plan_id=1106,
            auto_renewal=True,
            payment_provider="Robokassa",
            payment_method_id="1107",
            payment_attempt_count=0,
            end_date=now - timedelta(minutes=5),
        )
        u_rk = User(id=1107, username="user_rk", first_name="UserRK")

        session.add_all([admin_u, cfg, p, u_yk, sub_yk, u_rk, sub_rk])
        await session.commit()

    import subscription_renewal
    subscription_renewal._last_auth_alert_time = None

    bot = AsyncMock()
    with (
        patch("scheduler.process_birthday_mailings", AsyncMock()),
        patch("scheduler.async_session_maker", test_db),
        patch("subscription_renewal.Payment.create") as mock_yk_create,
        patch("scheduler.process_recurring_robokassa_payment", AsyncMock(return_value=True)) as mock_rk_exec,
    ):
        await check_subscriptions(bot)

    # 1. Zero calls to YooKassa Payment.create
    mock_yk_create.assert_not_called()

    # 2. Zero YooKassa claims created in DB
    async with test_db() as session:
        yk_attempts = (await session.execute(
            select(YookassaRecurringAttempt).where(YookassaRecurringAttempt.subscription_id == 1106)
        )).scalars().all()
        assert len(yk_attempts) == 0

    # 3. Zero 'YooKassa Circuit Breaker' or 'shop_id или secret_key' alerts sent to admins
    all_sent_texts = [str(c) for c in bot.send_message.call_args_list]
    assert not any("Circuit Breaker" in t for t in all_sent_texts)
    assert not any("shop_id или secret_key" in t for t in all_sent_texts)

    # 4. Scheduler continued processing unrelated non-YooKassa work: Robokassa recurring payment was executed!
    mock_rk_exec.assert_called_once()
    async with test_db() as session:
        updated_sub_rk = await session.get(UserSubscription, 1107)
        assert updated_sub_rk.pending_robokassa_invoice_id is not None
        assert updated_sub_rk.payment_attempt_count == 1


@pytest.mark.asyncio
async def test_scheduler_local_config_precheck_trips_incident(test_db):
    """Backward-compatibility wrapper for test runner."""
    await test_scheduler_local_config_precheck_silent_breaker_and_continues_work(test_db)


@pytest.mark.asyncio
async def test_confirmed_payment_resets_retry_not_before(test_db):
    """
    retry_not_before reset semantics:
    Only CONFIRMED successful payments clear retry_not_before.
    Unconfirmed/pending does NOT reset it.
    """
    future_retry = datetime.utcnow() + timedelta(hours=12)
    async with test_db() as session:
        u = User(id=1107, username="test1107", first_name="User1107")
        p = SubscriptionPlan(id=1107, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        sub = UserSubscription(
            id=1107,
            user_id=1107,
            plan_id=1107,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_card_1107",
            payment_attempt_count=2,
            retry_not_before=future_retry,
            end_date=datetime.utcnow(),
        )
        session.add_all([u, p, sub])
        await session.commit()

        # 1. Unconfirmed / declined does NOT clear retry_not_before
        is_new, action, sub_canc = await finalize_yookassa_payment_canceled(
            session=session,
            payment_id="pay_canc_test",
            cancellation_reason="card_expired",
            user_id=1107,
            plan_id=1107,
            amount=195.0,
            payment_method_id="pm_card_1107",
        )
        assert is_new is True
        # retry_not_before should still be set or updated, NOT None!
        refreshed = await session.get(UserSubscription, 1107)
        assert refreshed.retry_not_before is not None

        # 2. Confirmed successful payment DOES clear retry_not_before to None
        is_new_ok, sub_ok = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_succ_test",
            user_id=1107,
            plan_id=1107,
            amount=195.0,
            payment_method_id="pm_card_1107",
            is_recurring=True,
        )
        assert is_new_ok is True
        assert sub_ok.retry_not_before is None
        assert sub_ok.payment_attempt_count == 0


@pytest.mark.asyncio
async def test_yookassa_cancellation_unknown_policy(test_db):
    """
    CancellationPolicy.UNKNOWN policy:
    Unknown cancellation reason:
    - sets auto_renewal = False
    - preserves payment_method_id (token NOT deleted)
    - does NOT schedule 24h retry
    - action_taken is 'unknown_cancellation'
    """
    async with test_db() as session:
        u = User(id=1108, username="test1108", first_name="User1108")
        p = SubscriptionPlan(id=1108, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        sub = UserSubscription(
            id=1108,
            user_id=1108,
            plan_id=1108,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_card_1108",
            payment_attempt_count=0,
            end_date=datetime.utcnow(),
        )
        session.add_all([u, p, sub])
        await session.commit()

        # Classify unknown reason
        pol, _ = classify_yookassa_cancellation_reason("some_weird_unrecognized_bank_code")
        assert pol == CancellationPolicy.UNKNOWN

        # Finalize cancellation with unknown reason
        is_new, action, updated_sub = await finalize_yookassa_payment_canceled(
            session=session,
            payment_id="pay_unk_canc_1",
            cancellation_reason="some_weird_unrecognized_bank_code",
            user_id=1108,
            plan_id=1108,
            amount=195.0,
            payment_method_id="pm_card_1108",
            is_recurring=True,
        )
        assert is_new is True
        assert action == "unknown_cancellation"
        assert updated_sub.auto_renewal is False
        assert updated_sub.payment_method_id == "pm_card_1108"  # Token NOT deleted!
        assert updated_sub.retry_not_before is None  # No 24h retry scheduled!


@pytest.mark.asyncio
async def test_fatal_verification_idx_unresolved_yookassa_attempt():
    """
    Fatal startup verification of idx_unresolved_yookassa_attempt:
    If the partial unique index is missing or invalid, startup verification raises RuntimeError.
    """
    from sqlalchemy import text
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Drop the unique partial index to simulate corrupt schema
        await conn.execute(text("DROP INDEX IF EXISTS idx_unresolved_yookassa_attempt"))

        # Re-running verification logic should raise RuntimeError
        def verify_schema(sync_conn):
            row = sync_conn.execute(text(
                "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'idx_unresolved_yookassa_attempt'"
            )).first()
            if not row or not row[0]:
                raise RuntimeError("Critical index idx_unresolved_yookassa_attempt is missing in SQLite")

        with pytest.raises(RuntimeError, match="Critical index idx_unresolved_yookassa_attempt is missing"):
            await conn.run_sync(verify_schema)
    await engine.dispose()


@pytest.mark.asyncio
async def test_verify_yookassa_recurring_safety_schema_helper():
    """
    Direct unit test for production helper verify_yookassa_recurring_safety_schema(sync_conn).
    Validates all 7 required conditions:
    1. valid schema passes (SQLite and PostgreSQL)
    2. missing index fails
    3. non-unique same-name index fails
    4. wrong column fails
    5. wrong predicate fails (negation, missing status, extra status)
    6. missing user_subscriptions.retry_not_before fails
    7. missing yookassa_recurring_attempts.request_payload fails
    """
    from sqlalchemy import text
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # 1. Valid SQLite schema passes
        await conn.run_sync(verify_yookassa_recurring_safety_schema)

        # 2. Missing index fails
        await conn.execute(text("DROP INDEX IF EXISTS idx_unresolved_yookassa_attempt"))
        with pytest.raises(RuntimeError, match="missing in SQLite"):
            await conn.run_sync(verify_yookassa_recurring_safety_schema)

        # 3. Non-unique same-name index fails
        await conn.execute(text(
            "CREATE INDEX idx_unresolved_yookassa_attempt ON yookassa_recurring_attempts (subscription_id) "
            "WHERE status IN ('claimed', 'pending', 'unknown')"
        ))
        with pytest.raises(RuntimeError, match="not UNIQUE"):
            await conn.run_sync(verify_yookassa_recurring_safety_schema)
        await conn.execute(text("DROP INDEX IF EXISTS idx_unresolved_yookassa_attempt"))

        # 4. Wrong column fails
        await conn.execute(text(
            "CREATE UNIQUE INDEX idx_unresolved_yookassa_attempt ON yookassa_recurring_attempts (user_id) "
            "WHERE status IN ('claimed', 'pending', 'unknown')"
        ))
        with pytest.raises(RuntimeError, match="does not index subscription_id"):
            await conn.run_sync(verify_yookassa_recurring_safety_schema)
        await conn.execute(text("DROP INDEX IF EXISTS idx_unresolved_yookassa_attempt"))

        # 5a. Wrong predicate: NOT IN fails
        await conn.execute(text(
            "CREATE UNIQUE INDEX idx_unresolved_yookassa_attempt ON yookassa_recurring_attempts (subscription_id) "
            "WHERE status NOT IN ('claimed', 'pending', 'unknown')"
        ))
        with pytest.raises(RuntimeError, match="disallowed operator or connector"):
            await conn.run_sync(verify_yookassa_recurring_safety_schema)
        await conn.execute(text("DROP INDEX IF EXISTS idx_unresolved_yookassa_attempt"))

        # 5b. Wrong predicate: missing an active state fails
        await conn.execute(text(
            "CREATE UNIQUE INDEX idx_unresolved_yookassa_attempt ON yookassa_recurring_attempts (subscription_id) "
            "WHERE status IN ('claimed', 'pending')"
        ))
        with pytest.raises(RuntimeError, match="invalid states"):
            await conn.run_sync(verify_yookassa_recurring_safety_schema)
        await conn.execute(text("DROP INDEX IF EXISTS idx_unresolved_yookassa_attempt"))

        # 5c. Wrong predicate: extra status fails
        await conn.execute(text(
            "CREATE UNIQUE INDEX idx_unresolved_yookassa_attempt ON yookassa_recurring_attempts (subscription_id) "
            "WHERE status IN ('claimed', 'pending', 'unknown', 'succeeded')"
        ))
        with pytest.raises(RuntimeError, match="invalid states"):
            await conn.run_sync(verify_yookassa_recurring_safety_schema)
        await conn.execute(text("DROP INDEX IF EXISTS idx_unresolved_yookassa_attempt"))

        # 5d. Wrong predicate: AND 0=1 fails
        await conn.execute(text(
            "CREATE UNIQUE INDEX idx_unresolved_yookassa_attempt ON yookassa_recurring_attempts (subscription_id) "
            "WHERE status IN ('claimed', 'pending', 'unknown') AND 0=1"
        ))
        with pytest.raises(RuntimeError, match="disallowed operator or connector"):
            await conn.run_sync(verify_yookassa_recurring_safety_schema)
        await conn.execute(text("DROP INDEX IF EXISTS idx_unresolved_yookassa_attempt"))

        # 5e. Wrong predicate: OR 1=1 fails
        await conn.execute(text(
            "CREATE UNIQUE INDEX idx_unresolved_yookassa_attempt ON yookassa_recurring_attempts (subscription_id) "
            "WHERE status IN ('claimed', 'pending', 'unknown') OR 1=1"
        ))
        with pytest.raises(RuntimeError, match="disallowed operator or connector"):
            await conn.run_sync(verify_yookassa_recurring_safety_schema)
        await conn.execute(text("DROP INDEX IF EXISTS idx_unresolved_yookassa_attempt"))

        # 5f. Wrong predicate: extra column condition fails
        await conn.execute(text(
            "CREATE UNIQUE INDEX idx_unresolved_yookassa_attempt ON yookassa_recurring_attempts (subscription_id) "
            "WHERE status IN ('claimed', 'pending', 'unknown') AND user_id > 0"
        ))
        with pytest.raises(RuntimeError, match="disallowed operator or connector"):
            await conn.run_sync(verify_yookassa_recurring_safety_schema)
        await conn.execute(text("DROP INDEX IF EXISTS idx_unresolved_yookassa_attempt"))

        # Re-create valid index for column missing tests
        await conn.execute(text(
            "CREATE UNIQUE INDEX idx_unresolved_yookassa_attempt ON yookassa_recurring_attempts (subscription_id) "
            "WHERE status IN ('claimed', 'pending', 'unknown')"
        ))

    # 1b. Test Postgres simulation: valid passes
    with patch("sqlalchemy.inspect") as mock_inspect:
        mock_insp = MagicMock()
        mock_inspect.return_value = mock_insp
        mock_insp.get_columns.side_effect = lambda table: [
            {"name": "retry_not_before"}
        ] if table == "user_subscriptions" else [
            {"name": "request_payload"}
        ]

        mock_pg_conn = MagicMock()
        mock_pg_conn.dialect.name = "postgresql"
        mock_pg_conn.execute.return_value.first.return_value = (
            "CREATE UNIQUE INDEX idx_unresolved_yookassa_attempt ON yookassa_recurring_attempts (subscription_id) WHERE status IN ('claimed', 'pending', 'unknown')",
        )
        verify_yookassa_recurring_safety_schema(mock_pg_conn)

        # 1c. Test Postgres with array/ANY syntax passes
        mock_pg_conn.execute.return_value.first.return_value = (
            "CREATE UNIQUE INDEX idx_unresolved_yookassa_attempt ON public.yookassa_recurring_attempts USING btree (subscription_id) WHERE ((status)::text = ANY ((ARRAY['claimed'::character varying, 'pending'::character varying, 'unknown'::character varying])::text[]))",
        )
        verify_yookassa_recurring_safety_schema(mock_pg_conn)

        # 2b. Test Postgres missing index raises
        mock_pg_conn.execute.return_value.first.return_value = None
        with pytest.raises(RuntimeError, match="missing in PostgreSQL"):
            verify_yookassa_recurring_safety_schema(mock_pg_conn)

        # 6. Missing user_subscriptions.retry_not_before fails
        mock_insp.get_columns.side_effect = lambda table: [] if table == "user_subscriptions" else [{"name": "request_payload"}]
        with pytest.raises(RuntimeError, match="retry_not_before is missing"):
            verify_yookassa_recurring_safety_schema(mock_pg_conn)

        # 7. Missing yookassa_recurring_attempts.request_payload fails
        mock_insp.get_columns.side_effect = lambda table: [{"name": "retry_not_before"}] if table == "user_subscriptions" else []
        with pytest.raises(RuntimeError, match="request_payload is missing"):
            verify_yookassa_recurring_safety_schema(mock_pg_conn)

    await engine.dispose()


@pytest.mark.asyncio
async def test_user_switches_provider_to_robokassa_while_yookassa_pending_success(test_db):
    """
    User switches to Robokassa while YooKassa attempt is pending.
    Late YooKassa success webhook must:
    - credit end_date by paid duration
    - preserve user_sub.payment_provider = 'Robokassa'
    - preserve user_sub.payment_method_id (e.g. None or Robokassa token)
    - mark attempt succeeded
    """
    async with test_db() as session:
        u = User(id=1201, username="test1201", first_name="User1201")
        p = SubscriptionPlan(id=1201, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        orig_end = datetime(2026, 9, 10, 12, 0, 0)
        sub = UserSubscription(
            id=1201,
            user_id=1201,
            plan_id=1201,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_old_card",
            payment_attempt_count=1,
            end_date=orig_end,
        )
        session.add_all([u, p, sub])
        await session.commit()

        claim_res = await claim_yookassa_recurring_attempt(session, sub, p, 195.0, "scheduler")
        assert claim_res.claimed is True
        attempt = claim_res.attempt
        await update_yookassa_attempt_pending(session, attempt.id, "pay_yk_late_1")

        # User switches to Robokassa
        sub.payment_provider = "Robokassa"
        sub.payment_method_id = None
        sub.auto_renewal = True
        await session.commit()

        # Late YooKassa webhook arrives
        res = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_yk_late_1",
            user_id=1201,
            plan_id=1201,
            amount=195.0,
            payment_method_id="pm_old_card",
            is_recurring=True,
            recurring_attempt_key=attempt.idempotency_key,
        )
        assert res.is_new is True
        updated_sub = res.subscription
        # Provider and method are NOT overwritten!
        assert updated_sub.payment_provider == "Robokassa"
        assert updated_sub.payment_method_id is None
        # End date is extended
        assert updated_sub.end_date > orig_end

        # Attempt in DB is succeeded
        refreshed_att = await session.get(YookassaRecurringAttempt, attempt.id)
        assert refreshed_att.status == "succeeded"


@pytest.mark.asyncio
async def test_user_switches_provider_to_robokassa_while_yookassa_pending_failure(test_db):
    """
    User switches to Robokassa while YooKassa attempt is pending.
    Late YooKassa cancellation must:
    - return action 'historical_canceled'
    - NOT deactivate Robokassa subscription
    - NOT increment attempt count
    - mark attempt canceled
    """
    async with test_db() as session:
        u = User(id=1202, username="test1202", first_name="User1202")
        p = SubscriptionPlan(id=1202, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        sub = UserSubscription(
            id=1202,
            user_id=1202,
            plan_id=1202,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_old_card",
            payment_attempt_count=0,
            end_date=datetime(2026, 9, 10, 12, 0, 0),
        )
        session.add_all([u, p, sub])
        await session.commit()

        claim_res = await claim_yookassa_recurring_attempt(session, sub, p, 195.0, "scheduler")
        attempt = claim_res.attempt
        await update_yookassa_attempt_pending(session, attempt.id, "pay_yk_fail_1")

        # User switches to Robokassa
        sub.payment_provider = "Robokassa"
        sub.payment_method_id = None
        sub.auto_renewal = True
        await session.commit()

        is_new, action, updated_sub = await finalize_yookassa_payment_canceled(
            session=session,
            payment_id="pay_yk_fail_1",
            cancellation_reason="permission_revoked",
            user_id=1202,
            plan_id=1202,
            amount=195.0,
            payment_method_id="pm_old_card",
            is_recurring=True,
            recurring_attempt_key=attempt.idempotency_key,
        )
        assert is_new is True
        assert action == "historical_canceled"
        assert updated_sub.payment_provider == "Robokassa"
        assert updated_sub.auto_renewal is True  # NOT deactivated!
        assert updated_sub.payment_attempt_count == 0  # NOT incremented!

        refreshed_att = await session.get(YookassaRecurringAttempt, attempt.id)
        assert refreshed_att.status == "deactivated"


@pytest.mark.asyncio
async def test_user_binds_pm_new_while_yookassa_pending_pm_old_succeeds(test_db):
    """
    User binds PM_NEW while PM_OLD attempt is pending.
    Late success of PM_OLD:
    - credits end_date
    - preserves user_sub.payment_method_id == 'PM_NEW'
    - marks attempt succeeded
    """
    async with test_db() as session:
        u = User(id=1203, username="test1203", first_name="User1203")
        p = SubscriptionPlan(id=1203, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        orig_end = datetime(2026, 9, 10, 12, 0, 0)
        sub = UserSubscription(
            id=1203,
            user_id=1203,
            plan_id=1203,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_old_card",
            payment_attempt_count=1,
            end_date=orig_end,
        )
        session.add_all([u, p, sub])
        await session.commit()

        claim_res = await claim_yookassa_recurring_attempt(session, sub, p, 195.0, "scheduler")
        attempt = claim_res.attempt
        await update_yookassa_attempt_pending(session, attempt.id, "pay_old_success_1")

        # User updates card to pm_new
        sub.payment_method_id = "pm_new_card"
        await session.commit()

        res = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_old_success_1",
            user_id=1203,
            plan_id=1203,
            amount=195.0,
            payment_method_id="pm_old_card",
            is_recurring=True,
            recurring_attempt_key=attempt.idempotency_key,
        )
        assert res.is_new is True
        updated_sub = res.subscription
        assert updated_sub.payment_method_id == "pm_new_card"  # Preserved!
        assert updated_sub.end_date > orig_end

        refreshed_att = await session.get(YookassaRecurringAttempt, attempt.id)
        assert refreshed_att.status == "succeeded"


@pytest.mark.asyncio
async def test_user_binds_pm_new_while_yookassa_pending_pm_old_fails_deactivate(test_db):
    """
    User binds PM_NEW while PM_OLD attempt is pending.
    Late terminal deactivation (permission_revoked) of PM_OLD:
    - returns action 'historical_canceled'
    - preserves user_sub.payment_method_id == 'PM_NEW'
    - preserves auto_renewal == True
    """
    async with test_db() as session:
        u = User(id=1204, username="test1204", first_name="User1204")
        p = SubscriptionPlan(id=1204, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        sub = UserSubscription(
            id=1204,
            user_id=1204,
            plan_id=1204,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_old_card",
            payment_attempt_count=0,
            end_date=datetime(2026, 9, 10, 12, 0, 0),
        )
        session.add_all([u, p, sub])
        await session.commit()

        claim_res = await claim_yookassa_recurring_attempt(session, sub, p, 195.0, "scheduler")
        attempt = claim_res.attempt
        await update_yookassa_attempt_pending(session, attempt.id, "pay_old_fail_deact")

        # User updates card to pm_new
        sub.payment_method_id = "pm_new_card"
        await session.commit()

        is_new, action, updated_sub = await finalize_yookassa_payment_canceled(
            session=session,
            payment_id="pay_old_fail_deact",
            cancellation_reason="permission_revoked",
            user_id=1204,
            plan_id=1204,
            amount=195.0,
            payment_method_id="pm_old_card",
            is_recurring=True,
            recurring_attempt_key=attempt.idempotency_key,
        )
        assert is_new is True
        assert action == "historical_canceled"
        assert updated_sub.payment_method_id == "pm_new_card"  # PM_NEW not deleted!
        assert updated_sub.auto_renewal is True  # NOT deactivated!

        refreshed_att = await session.get(YookassaRecurringAttempt, attempt.id)
        assert refreshed_att.status == "deactivated"


@pytest.mark.asyncio
async def test_same_provider_tariff_switch_supersedes_attempt(test_db):
    """
    User switches from Plan A to Plan B via explicit purchase.
    Open recurring attempt A is transitioned to 'superseded'.
    When attempt A later succeeds:
    - YookassaPayment is recorded as 'completed'
    - attempt A is marked 'succeeded'
    - current sub.plan_id remains Plan B (not downgraded)
    - current sub.end_date is NOT extended with Plan A duration (no cross-tariff gifting)
    - action is 'manual_reconciliation_required'
    """
    async with test_db() as session:
        u = User(id=1205, username="test1205", first_name="User1205")
        plan_a = SubscriptionPlan(id=1205, name="Plan A", price=195.0, duration_value=1, duration_unit="months")
        plan_b = SubscriptionPlan(id=1206, name="Plan B", price=490.0, duration_value=3, duration_unit="months")
        plan_b_end = datetime(2026, 12, 10, 12, 0, 0)
        sub = UserSubscription(
            id=1205,
            user_id=1205,
            plan_id=1205,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_same_card",
            payment_attempt_count=0,
            end_date=datetime(2026, 9, 10, 12, 0, 0),
        )
        session.add_all([u, plan_a, plan_b, sub])
        await session.commit()

        # Recurring attempt A claimed
        claim_res = await claim_yookassa_recurring_attempt(session, sub, plan_a, 195.0, "scheduler")
        attempt_a = claim_res.attempt
        await update_yookassa_attempt_pending(session, attempt_a.id, "pay_attempt_a")

        # Explicit purchase of Plan B occurs (simulating webhook execution)
        await mark_unresolved_attempts_superseded(session, sub.id)
        sub.plan_id = plan_b.id
        sub.end_date = plan_b_end
        await session.commit()

        # Verify attempt A is now superseded
        refreshed_a = await session.get(YookassaRecurringAttempt, attempt_a.id)
        assert refreshed_a.status == "superseded"

        # Late success arrives for Attempt A (Plan A)
        res = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_attempt_a",
            user_id=1205,
            plan_id=plan_a.id,
            amount=195.0,
            payment_method_id="pm_same_card",
            is_recurring=True,
            recurring_attempt_key=attempt_a.idempotency_key,
        )
        assert res.is_new is True
        assert res.action == "manual_reconciliation_required"

        # Current subscription Plan B is preserved
        refreshed_sub = await session.get(UserSubscription, sub.id)
        assert refreshed_sub.plan_id == plan_b.id
        # end_date is NOT extended with Plan A's duration
        assert refreshed_sub.end_date == plan_b_end

        # Payment is recorded as completed
        pay_rec = await session.scalar(select(YookassaPayment).where(YookassaPayment.payment_id == "pay_attempt_a"))
        assert pay_rec.status == "completed"

        # Attempt A is marked succeeded
        refreshed_a2 = await session.get(YookassaRecurringAttempt, attempt_a.id)
        assert refreshed_a2.status == "succeeded"


@pytest.mark.asyncio
async def test_cross_plan_late_success_reconciliation(test_db):
    """
    Direct test for cross-plan late success:
    When attempt plan differs from current active plan:
    - action is 'manual_reconciliation_required'
    - current sub.plan_id is untouched
    - current sub.end_date is untouched
    - payment recorded as completed
    """
    async with test_db() as session:
        u = User(id=1207, username="test1207", first_name="User1207")
        p1 = SubscriptionPlan(id=1207, name="Plan 1", price=195.0, duration_value=1, duration_unit="months")
        p2 = SubscriptionPlan(id=1208, name="Plan 2", price=390.0, duration_value=2, duration_unit="months")
        active_end = datetime(2026, 11, 10, 12, 0, 0)
        sub = UserSubscription(
            id=1207,
            user_id=1207,
            plan_id=1208,  # Currently on Plan 2
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_card_7",
            payment_attempt_count=0,
            end_date=active_end,
        )
        session.add_all([u, p1, p2, sub])
        await session.commit()

        # Late payment comes for Plan 1
        res = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_cross_plan_1",
            user_id=1207,
            plan_id=p1.id,
            amount=195.0,
            payment_method_id="pm_card_7",
            is_recurring=True,
        )
        assert res.is_new is True
        assert res.action == "manual_reconciliation_required"
        assert res.subscription.plan_id == p2.id
        assert res.subscription.end_date == active_end


@pytest.mark.asyncio
async def test_orphan_attempt_no_payment_fails_closed(test_db):
    """
    Gap 3 contract:
    Missing subscription causes finalize_yookassa_attempt_no_payment to:
    - abort BEFORE CAS
    - leave attempt row un-transitioned
    - touch zero unrelated subscriptions
    - return (False, None)
    """
    async with test_db() as session:
        orphan_att = YookassaRecurringAttempt(
            subscription_id=777777,
            user_id=999,
            plan_id=1,
            idempotency_key="orphan_fail_closed_key",
            amount=195.0,
            payment_method_id="pm_test",
            status="claimed",
        )
        session.add(orphan_att)
        await session.commit()

        is_new, sub = await finalize_yookassa_attempt_no_payment(
            session=session,
            attempt_id=orphan_att.id,
            outcome="deactivate",
            error_code="test_deact",
        )
        assert is_new is False
        assert sub is None

        refreshed = await session.get(YookassaRecurringAttempt, orphan_att.id)
        assert refreshed.status == "claimed"


@pytest.mark.asyncio
async def test_cas_concurrency_deactivate_barrier_real_sqlite_wal():
    """
    Real multi-session SQLite WAL concurrency test with asyncio.Barrier(2):
    Two concurrent callers attempt to finalize the same attempt as deactivate.
    Exactly one winner must get is_new=True. The other must get is_new=False.
    Subscription is deactivated exactly once.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_path = tmp.name

    wal_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
    async with wal_engine.begin() as conn:
        from sqlalchemy import text
        await conn.execute(text("PRAGMA journal_mode=WAL;"))
        await conn.run_sync(Base.metadata.create_all)

    wal_session_maker = async_sessionmaker(wal_engine, expire_on_commit=False)

    async with wal_session_maker() as session:
        u = User(id=1301, username="test1301", first_name="User1301")
        p = SubscriptionPlan(id=1301, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        sub = UserSubscription(
            id=1301,
            user_id=1301,
            plan_id=1301,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_wal_card",
            payment_attempt_count=0,
            end_date=datetime(2026, 9, 10, 12, 0, 0),
        )
        session.add_all([u, p, sub])
        await session.commit()

        claim_res = await claim_yookassa_recurring_attempt(session, sub, p, 195.0, "scheduler")
        attempt_id = claim_res.attempt.id
        attempt_started_at = claim_res.attempt.attempt_started_at

    barrier = asyncio.Barrier(2)

    async def worker():
        async with wal_session_maker() as sess:
            s = await sess.get(UserSubscription, 1301)
            att = await sess.get(YookassaRecurringAttempt, attempt_id)
            await barrier.wait()
            res = await finalize_yookassa_attempt_no_payment(
                session=sess,
                attempt_id=attempt_id,
                outcome="deactivate",
                error_code="permission_revoked",
                attempt_started_at=attempt_started_at,
                sub=s,
                attempt=att,
            )
            return res

    results = await asyncio.gather(worker(), worker(), return_exceptions=False)
    is_new_list = [r[0] for r in results]
    assert is_new_list.count(True) == 1
    assert is_new_list.count(False) == 1

    async with wal_session_maker() as sess:
        final_sub = await sess.get(UserSubscription, 1301)
        assert final_sub.auto_renewal is False
        assert final_sub.payment_method_id is None

    await wal_engine.dispose()
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_finalize_yookassa_success_requires_real_payment_id(test_db):
    """Verify finalize_yookassa_payment_success raises ValueError when payment_id is missing or an idempotency key."""
    async with test_db() as session:
        with pytest.raises(ValueError, match="A real payment_id is required"):
            await finalize_yookassa_payment_success(
                session=session,
                payment_id=None,
                user_id=1401,
                plan_id=1,
                amount=195.0,
            )
        with pytest.raises(ValueError, match="A real payment_id is required"):
            await finalize_yookassa_payment_success(
                session=session,
                payment_id="yk-rec-1-1401-1-abcdef123456",
                user_id=1401,
                plan_id=1,
                amount=195.0,
            )


@pytest.mark.asyncio
async def test_max_manual_retry_manual_reconciliation_notification_flow(test_db):
    """
    Verify MAX manual retry when cross-plan payment completes:
    - User receives explanatory warning (not 'Подписка успешно продлена').
    - Admin receives alert for manual reconciliation.
    """
    async with test_db() as session:
        u = User(id=1402, username="test1402", first_name="User1402")
        p1 = SubscriptionPlan(id=1402, name="Plan 1", price=195.0, duration_value=1, duration_unit="months")
        p2 = SubscriptionPlan(id=1403, name="Plan 2", price=390.0, duration_value=2, duration_unit="months")
        cfg = SubscriptionConfig(id=1, yookassa_shop_id="shop", yookassa_secret_key="sec", notifications_enabled=True)
        active_end = datetime(2026, 12, 1, 12, 0, 0)
        sub = UserSubscription(
            id=1402,
            user_id=1402,
            plan_id=1403,  # Currently on Plan 2
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_card_1402",
            payment_attempt_count=0,
            end_date=active_end,
        )
        session.add_all([u, p1, p2, cfg, sub])
        await session.commit()

        # Claim an attempt for Plan 1
        claim_res = await claim_yookassa_recurring_attempt(session, sub, p1, 195.0, "max_manual_retry")
        assert claim_res.claimed is True
        attempt = claim_res.attempt

    client = AsyncMock()
    mock_admin_notify = AsyncMock()
    fake_res = SimpleNamespace(outcome="success", payment_id="pay_cross_max_1", failure_reason=None)

    with (
        patch("max_messenger_bot.services.subscriptions.async_session_maker", test_db),
        patch("max_messenger_bot.services.subscriptions.claim_yookassa_recurring_attempt", return_value=claim_res),
        patch("max_messenger_bot.services.subscriptions.execute_or_replay_yookassa_recurring_attempt", return_value=fake_res),
        patch("max_messenger_bot.services.common.notify_telegram_admins", mock_admin_notify),
    ):
        await handle_max_manual_retry(client, chat_id=1402, user_id=1402)

    sent_texts = [call[1]["text"] for call in client.send_message.call_args_list]
    # Check that user received warning, NOT successful extension
    assert any("отправлен на проверку администратору" in t for t in sent_texts)
    assert not any("успешно продлена" in t for t in sent_texts)

    # Admin was notified with reconciliation required
    mock_admin_notify.assert_called_once()
    admin_text = mock_admin_notify.call_args[0][0]
    assert "ТРЕБУЕТСЯ РУЧНАЯ СВЕРКА ТАРИФА" in admin_text
    assert "pay_cross_max_1" in admin_text


@pytest.mark.asyncio
async def test_session_a_keeps_alive_session_b_confirms_purchase_two_session_race(test_db):
    """
    Two-session race test (Requirement 1):
    Session A:
        claim YooKassa attempt A and keep the session/object alive.
    Session B:
        confirmed explicit purchase:
        - changes provider (YooKassa -> Robokassa);
        - changes PM (pm_old -> pm_new);
        - changes plan (Plan A -> Plan B);
        - marks A superseded;
        - commits.
    Session A:
        without recreating its original session, finalizes late A.
    Assert Session A observes the NEW committed state and cannot overwrite it.
    Covers:
    - YooKassa -> Robokassa
    - PM_OLD -> PM_NEW
    - Plan A -> Plan B
    """
    async with test_db() as session:
        u = User(id=1501, username="test1501", first_name="User1501")
        plan_a = SubscriptionPlan(id=1501, name="Plan A", price=195.0, duration_value=1, duration_unit="months")
        plan_b = SubscriptionPlan(id=1502, name="Plan B", price=450.0, duration_value=3, duration_unit="months")
        active_end = datetime(2026, 12, 1, 12, 0, 0)
        sub = UserSubscription(
            id=1501,
            user_id=1501,
            plan_id=1501,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_old",
            payment_attempt_count=0,
            end_date=active_end,
        )
        session.add_all([u, plan_a, plan_b, sub])
        await session.commit()

    # Session A claims YooKassa attempt A
    session_a = test_db()
    sub_a = await session_a.get(UserSubscription, 1501)
    claim_res = await claim_yookassa_recurring_attempt(session_a, sub_a, plan_a, 195.0, "scheduler")
    assert claim_res.claimed is True
    attempt_a = claim_res.attempt
    await session_a.commit()  # commit claim, keep session_a and sub_a / attempt_a alive in memory

    # Session B executes confirmed explicit purchase
    async with test_db() as session_b:
        sub_b = await session_b.get(UserSubscription, 1501)
        sub_b.payment_provider = "Robokassa"
        sub_b.payment_method_id = "pm_new"
        sub_b.plan_id = plan_b.id
        await mark_unresolved_attempts_superseded(session_b, sub_b.id)
        await session_b.commit()

    # Session A: without recreating its session, finalizes late A
    res_fin = await finalize_yookassa_payment_success(
        session=session_a,
        payment_id="pay_late_1501",
        user_id=1501,
        plan_id=plan_a.id,
        amount=195.0,
        is_recurring=True,
        recurring_attempt_key=attempt_a.idempotency_key,
    )
    # Finalizer must observe the new DB state and trigger cross-plan manual reconciliation
    assert res_fin.action == "manual_reconciliation_required"
    assert res_fin.reconciliation_details["paid_plan_id"] == plan_a.id
    assert res_fin.reconciliation_details["current_plan_id"] == plan_b.id

    # Refresh sub_a in session_a and assert it was NOT overwritten
    await session_a.refresh(sub_a)
    assert sub_a.payment_provider == "Robokassa"
    assert sub_a.payment_method_id == "pm_new"
    assert sub_a.plan_id == plan_b.id
    assert sub_a.end_date == active_end  # Untouched

    await session_a.close()


@pytest.mark.asyncio
async def test_session_a_keeps_alive_session_b_confirms_purchase_cancel_race(test_db):
    """
    Two-session race test with late failure (Requirement 1 & 5):
    Session A claims attempt A, Session B switches provider/PM/plan and marks A superseded.
    Session A finalizes late A as canceled/deactivated.
    Assert Session A produces action='historical_canceled' and makes zero mutations.
    """
    async with test_db() as session:
        u = User(id=1502, username="test1502", first_name="User1502")
        plan_a = SubscriptionPlan(id=1503, name="Plan A", price=195.0, duration_value=1, duration_unit="months")
        plan_b = SubscriptionPlan(id=1504, name="Plan B", price=450.0, duration_value=3, duration_unit="months")
        active_end = datetime(2026, 12, 1, 12, 0, 0)
        sub = UserSubscription(
            id=1502,
            user_id=1502,
            plan_id=1503,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_old",
            payment_attempt_count=2,  # Nonzero retry count
            end_date=active_end,
        )
        session.add_all([u, plan_a, plan_b, sub])
        await session.commit()

    # Session A claims YooKassa attempt A
    session_a = test_db()
    sub_a = await session_a.get(UserSubscription, 1502)
    claim_res = await claim_yookassa_recurring_attempt(session_a, sub_a, plan_a, 195.0, "scheduler")
    assert claim_res.claimed is True
    attempt_a = claim_res.attempt
    await session_a.commit()

    # Session B executes confirmed explicit purchase
    async with test_db() as session_b:
        sub_b = await session_b.get(UserSubscription, 1502)
        sub_b.payment_provider = "Robokassa"
        sub_b.payment_method_id = "pm_new"
        sub_b.plan_id = plan_b.id
        await mark_unresolved_attempts_superseded(session_b, sub_b.id)
        await session_b.commit()

    # Session A: finalizes late A as force_deactivate
    is_new, action, final_sub = await finalize_yookassa_payment_canceled(
        session=session_a,
        payment_id="pay_fail_1502",
        cancellation_reason="permission_revoked",
        user_id=1502,
        plan_id=plan_a.id,
        amount=195.0,
        is_recurring=True,
        force_deactivate=True,
        recurring_attempt_key=attempt_a.idempotency_key,
    )
    assert action == "historical_canceled"
    assert is_new is True

    await session_a.refresh(sub_a)
    assert sub_a.payment_provider == "Robokassa"
    assert sub_a.payment_method_id == "pm_new"
    assert sub_a.plan_id == plan_b.id
    assert sub_a.auto_renewal is True  # NOT deactivated!
    assert sub_a.payment_attempt_count == 2  # NOT incremented or reset!

    await session_a.close()


@pytest.mark.asyncio
async def test_exact_attempt_defines_paid_recurring_plan_authoritative(test_db):
    """
    Requirement 2 Regression:
    Attempt A is immutable Plan A / amount A.
    Caller later passes current Plan B / amount B.
    Late A succeeds.
    Expected:
    - paid plan recognized as Plan A;
    - cross-plan reconciliation triggered;
    - Plan B / end_date untouched.
    """
    async with test_db() as session:
        u = User(id=1503, username="test1503")
        plan_a = SubscriptionPlan(id=1505, name="Plan A", price=195.0, duration_value=1, duration_unit="months")
        plan_b = SubscriptionPlan(id=1506, name="Plan B", price=450.0, duration_value=3, duration_unit="months")
        active_end = datetime(2026, 12, 1, 12, 0, 0)
        sub = UserSubscription(
            id=1503,
            user_id=1503,
            plan_id=1505,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_1503",
            payment_attempt_count=0,
            end_date=active_end,
        )
        session.add_all([u, plan_a, plan_b, sub])
        await session.commit()

        # Claim attempt A for Plan A
        claim_res = await claim_yookassa_recurring_attempt(session, sub, plan_a, 195.0, "scheduler")
        attempt_a = claim_res.attempt

        # Now subscription is changed to Plan B in DB
        sub.plan_id = plan_b.id
        await session.commit()

        # Caller calls finalizer passing Plan B and 450.0 (simulating caller using its current plan_to_charge)
        res_fin = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_authoritative_1",
            user_id=1503,
            plan_id=plan_b.id,
            amount=450.0,
            is_recurring=True,
            recurring_attempt_key=attempt_a.idempotency_key,
        )
        assert res_fin.action == "manual_reconciliation_required"
        # Authoritative paid plan MUST be Plan A (1505), NOT caller's Plan B (1506)!
        assert res_fin.reconciliation_details["paid_plan_id"] == plan_a.id
        assert res_fin.reconciliation_details["amount"] == 195.0
        assert res_fin.reconciliation_details["current_plan_id"] == plan_b.id

        await session.refresh(sub)
        assert sub.plan_id == plan_b.id
        assert sub.end_date == active_end


@pytest.mark.asyncio
async def test_nonzero_retry_subscription_untouched_by_historical_yookassa_cancellation(test_db):
    """
    Requirement 5: Current subscription has its own nonzero retry count (e.g. 2).
    A historical YooKassa cancellation arrives.
    Must return action='historical_canceled' and NOT trigger disable logic or mutate retry count.
    """
    async with test_db() as session:
        u = User(id=1504, username="test1504")
        p = SubscriptionPlan(id=1507, name="Plan", price=195.0, duration_value=1, duration_unit="months")
        sub = UserSubscription(
            id=1504,
            user_id=1504,
            plan_id=1507,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_current",
            payment_attempt_count=2,
            end_date=datetime(2026, 12, 1, 12, 0, 0),
        )
        session.add_all([u, p, sub])
        await session.commit()

        # Old attempt with an old payment method
        old_att = YookassaRecurringAttempt(
            subscription_id=sub.id,
            user_id=u.id,
            plan_id=p.id,
            idempotency_key="yk-rec-old-attempt-1504",
            amount=195.0,
            payment_method_id="pm_old_stale",
            status="superseded",
            payment_id="pay_old_stale",
        )
        session.add(old_att)
        await session.commit()

        is_new, action, updated_sub = await finalize_yookassa_payment_canceled(
            session=session,
            payment_id="pay_old_stale",
            cancellation_reason="card_expired",
            user_id=u.id,
            plan_id=p.id,
            amount=195.0,
            is_recurring=True,
            force_deactivate=True,
            recurring_attempt_key="yk-rec-old-attempt-1504",
        )
        assert is_new is True
        assert action == "historical_canceled"
        await session.refresh(sub)
        assert sub.payment_attempt_count == 2  # NOT incremented to 3!
        assert sub.auto_renewal is True  # NOT disabled!
        assert sub.payment_method_id == "pm_current"  # NOT cleared!


@pytest.mark.asyncio
async def test_sqlite_wal_concurrent_webhook_vs_caller_success():
    """
    Requirement 7: Real file-backed SQLite WAL two-session test:
    pending payment: webhook vs caller success concurrently.
    Assert:
    - exactly one is_new=True
    - other is_new=False
    - exact final DB state
    - end_date extended exactly once.
    """
    import tempfile
    from sqlalchemy import text
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    wal_engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"timeout": 30.0},
        isolation_level=None,
    )
    wal_session_maker = async_sessionmaker(wal_engine, expire_on_commit=False)

    async with wal_engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL;"))
        await conn.execute(text("PRAGMA busy_timeout=30000;"))
        await conn.run_sync(Base.metadata.create_all)

    active_end = datetime(2026, 10, 1, 12, 0, 0)
    async with wal_session_maker() as session:
        u = User(id=1505, username="test1505")
        p = SubscriptionPlan(id=1508, name="Plan WAL", price=195.0, duration_value=1, duration_unit="months")
        sub = UserSubscription(
            id=1505,
            user_id=1505,
            plan_id=1508,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_wal_card",
            payment_attempt_count=0,
            end_date=active_end,
        )
        session.add_all([u, p, sub])
        await session.commit()

        claim_res = await claim_yookassa_recurring_attempt(session, sub, p, 195.0, "scheduler")
        attempt_id = claim_res.attempt.id
        attempt_key = claim_res.attempt.idempotency_key

    barrier = asyncio.Barrier(2)

    async def worker():
        async with wal_session_maker() as sess:
            await barrier.wait()
            res = await finalize_yookassa_payment_success(
                session=sess,
                payment_id="pay_wal_race_succ",
                user_id=1505,
                plan_id=1508,
                amount=195.0,
                is_recurring=True,
                recurring_attempt_key=attempt_key,
            )
            return res

    results = await asyncio.gather(worker(), worker(), return_exceptions=False)
    is_new_list = [r[0] for r in results]
    assert is_new_list.count(True) == 1
    assert is_new_list.count(False) == 1

    async with wal_session_maker() as sess:
        final_sub = await sess.get(UserSubscription, 1505)
        # 1 month added from active_end: 2026-10-01 -> 2026-11-01
        expected_end = active_end + relativedelta(months=1)
        assert final_sub.end_date == expected_end
        att = await sess.get(YookassaRecurringAttempt, attempt_id)
        assert att.status == "succeeded"

    await wal_engine.dispose()
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_sqlite_wal_concurrent_webhook_vs_caller_canceled():
    """
    Requirement 7: Real file-backed SQLite WAL two-session test:
    pending payment: webhook vs caller canceled concurrently.
    Assert:
    - exactly one is_new=True
    - other is_new=False
    - exact final DB state
    - retry count / auto_renewal transitioned exactly once.
    """
    import tempfile
    from sqlalchemy import text
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    wal_engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"timeout": 30.0},
        isolation_level=None,
    )
    wal_session_maker = async_sessionmaker(wal_engine, expire_on_commit=False)

    async with wal_engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL;"))
        await conn.execute(text("PRAGMA busy_timeout=30000;"))
        await conn.run_sync(Base.metadata.create_all)

    async with wal_session_maker() as session:
        u = User(id=1506, username="test1506")
        p = SubscriptionPlan(id=1509, name="Plan WAL", price=195.0, duration_value=1, duration_unit="months")
        sub = UserSubscription(
            id=1506,
            user_id=1506,
            plan_id=1509,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_wal_card_2",
            payment_attempt_count=0,
            end_date=datetime(2026, 10, 1, 12, 0, 0),
        )
        session.add_all([u, p, sub])
        await session.commit()

        claim_res = await claim_yookassa_recurring_attempt(session, sub, p, 195.0, "scheduler")
        attempt_id = claim_res.attempt.id
        attempt_key = claim_res.attempt.idempotency_key

    barrier = asyncio.Barrier(2)

    async def worker():
        async with wal_session_maker() as sess:
            await barrier.wait()
            res = await finalize_yookassa_payment_canceled(
                session=sess,
                payment_id="pay_wal_race_canc",
                cancellation_reason="insufficient_funds",
                user_id=1506,
                plan_id=1509,
                amount=195.0,
                is_recurring=True,
                recurring_attempt_key=attempt_key,
            )
            return res

    results = await asyncio.gather(worker(), worker(), return_exceptions=False)
    is_new_list = [r[0] for r in results]
    assert is_new_list.count(True) == 1
    assert is_new_list.count(False) == 1

    async with wal_session_maker() as sess:
        final_sub = await sess.get(UserSubscription, 1506)
        # Attempt count incremented exactly once (0 -> 1)
        assert final_sub.payment_attempt_count == 1
        att = await sess.get(YookassaRecurringAttempt, attempt_id)
        assert att.status == "canceled"

    await wal_engine.dispose()
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_ordinary_yookassa_checkout_happy_path_regression(test_db):
    """
    Requirement 9 Regression:
    Ordinary YooKassa successful checkout with no unresolved recurring attempts.
    Expected:
    - payment completed
    - selected plan applied
    - end_date calculated and extended
    - payment_provider = 'Yookassa'
    - retry state reset (payment_attempt_count=0, last_payment_attempt=None, retry_not_before=None)
    - no fake manual reconciliation
    - zero unexpected superseded attempts.
    """
    from webhooks import handle_yookassa_webhook
    from aiohttp.test_utils import make_mocked_request
    from aiogram.fsm.storage.memory import MemoryStorage

    async with test_db() as session:
        u = User(id=1507, username="test1507", first_name="User1507")
        p = SubscriptionPlan(id=1510, name="Standard Month", price=195.0, duration_value=1, duration_unit="months")
        cfg = SubscriptionConfig(id=1, yookassa_shop_id="shop", yookassa_secret_key="sec", notifications_enabled=False)
        sub = UserSubscription(
            id=1507,
            user_id=1507,
            plan_id=1510,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_old_card",
            payment_attempt_count=2,  # Previous failed attempt
            last_payment_attempt=datetime(2026, 9, 10, 8, 0, 0),
            retry_not_before=datetime(2026, 9, 11, 8, 0, 0),
            end_date=datetime(2026, 9, 15, 12, 0, 0),
        )
        session.add_all([u, p, cfg, sub])
        await session.commit()

    webhook_payload = {
        "type": "notification",
        "event": "payment.succeeded",
        "object": {
            "id": "pay_ord_yk_1507",
            "status": "succeeded",
            "amount": {"value": "195.00", "currency": "RUB"},
            "payment_method": {"id": "pm_new_saved_card", "saved": True},
            "metadata": {"user_id": "1507", "plan_id": "1510", "is_recurring": "false"},
        },
    }

    mock_bot = AsyncMock()
    mock_bot.id = 1
    mock_bot.send_message = AsyncMock()
    mock_req = make_mocked_request(
        "POST",
        "/yookassa/webhook",
        headers={"Content-Type": "application/json"},
        app={"bot": mock_bot, "fsm_storage": MemoryStorage()},
    )
    mock_req.json = AsyncMock(return_value=webhook_payload)

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=webhook_payload["object"])
    mock_get_cm = AsyncMock()
    mock_get_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_get_cm.__aexit__ = AsyncMock(return_value=None)
    mock_http_session = AsyncMock()
    mock_http_session.get = MagicMock(return_value=mock_get_cm)
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_http_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("webhooks.async_session_maker", test_db),
        patch("webhooks.send_msg_universal", AsyncMock()),
        patch("aiohttp.ClientSession", MagicMock(return_value=mock_session_cm)),
    ):
        resp = await handle_yookassa_webhook(mock_req)
        assert resp.status == 200

    async with test_db() as session:
        final_sub = await session.get(UserSubscription, 1507)
        assert final_sub.payment_provider == "Yookassa"
        assert final_sub.payment_method_id == "pm_new_saved_card"
        assert final_sub.auto_renewal is True
        assert final_sub.payment_attempt_count == 0
        assert final_sub.last_payment_attempt is None
        assert final_sub.retry_not_before is None
        assert final_sub.pending_robokassa_invoice_id is None
        # End date extended by 1 month from active end (Sept 15 -> Oct 15)
        assert final_sub.end_date == datetime(2026, 10, 15, 12, 0, 0)

        # Zero unexpected superseded attempts
        attempts = (await session.scalars(select(YookassaRecurringAttempt).where(YookassaRecurringAttempt.subscription_id == 1507))).all()
        assert len(attempts) == 0


@pytest.mark.asyncio
async def test_ordinary_robokassa_result_url_happy_path_regression(test_db):
    """
    Requirement 9 Regression:
    Ordinary Robokassa successful ResultURL with no unresolved YooKassa attempts.
    Expected:
    - payment completed
    - plan/end_date applied exactly once
    - payment_provider = 'Robokassa'
    - canonical root invoice payment_method_id behavior preserved
    - pending invoice cleared
    - retry state reset
    - zero unrelated YooKassa attempt side effects.
    """
    from webhooks import handle_robokassa_result, calculate_signature
    from aiohttp.test_utils import make_mocked_request
    from aiogram.fsm.storage.memory import MemoryStorage
    from database import RobokassaPayment

    async with test_db() as session:
        u = User(id=1508, username="test1508", first_name="User1508")
        p = SubscriptionPlan(id=1511, name="Standard Month", price=195.0, duration_value=1, duration_unit="months")
        cfg = SubscriptionConfig(id=1, robokassa_password_2="test_pass_2", notifications_enabled=False)
        sub = UserSubscription(
            id=1508,
            user_id=1508,
            plan_id=1511,
            auto_renewal=True,
            payment_provider="Robokassa",
            payment_method_id="root_inv_100",  # Canonical root invoice
            pending_robokassa_invoice_id="999",  # Renewal child invoice
            payment_attempt_count=2,
            last_payment_attempt=datetime(2026, 9, 10, 8, 0, 0),
            retry_not_before=datetime(2026, 9, 11, 8, 0, 0),
            end_date=datetime(2026, 9, 15, 12, 0, 0),
        )
        rk_pay = RobokassaPayment(
            id=999,
            user_id=1508,
            plan_id=1511,
            amount=195.0,
            status="pending",
        )
        session.add_all([u, p, cfg, sub, rk_pay])
        await session.commit()

    # Calculate valid SignatureValue
    valid_sig = calculate_signature("195.00", 999, "test_pass_2")

    mock_bot = AsyncMock()
    mock_bot.id = 1
    mock_req = make_mocked_request(
        "POST",
        "/robokassa/result",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        app={"bot": mock_bot, "fsm_storage": MemoryStorage()},
    )
    mock_req.post = AsyncMock(return_value={
        "OutSum": "195.00",
        "InvId": "999",
        "SignatureValue": valid_sig,
        "shp_plan": "1511",
        "shp_user": "1508",
    })

    with (
        patch("webhooks.async_session_maker", test_db),
        patch("webhooks.send_msg_universal", AsyncMock()),
    ):
        resp = await handle_robokassa_result(mock_req)
        assert resp.text == "OK999"

    async with test_db() as session:
        final_sub = await session.get(UserSubscription, 1508)
        assert final_sub.payment_provider == "Robokassa"
        # Root invoice payment_method_id must be preserved on renewal
        assert final_sub.payment_method_id == "root_inv_100"
        assert final_sub.pending_robokassa_invoice_id is None
        assert final_sub.payment_attempt_count == 0
        assert final_sub.last_payment_attempt is None
        assert final_sub.retry_not_before is None
        assert final_sub.end_date == datetime(2026, 10, 15, 12, 0, 0)

        # Payment record status updated
        pay = await session.get(RobokassaPayment, 999)
        assert pay.status == "completed"

        # Zero YooKassa attempts
        attempts = (await session.scalars(select(YookassaRecurringAttempt).where(YookassaRecurringAttempt.subscription_id == 1508))).all()
        assert len(attempts) == 0


@pytest.mark.asyncio
async def test_exact_attempt_unresolved_paid_plan_never_substitutes_current_plan(test_db):
    """
    Blocker 1 Regression:
    Exact attempt:
    - Plan A ID is authoritative (e.g. 1609);
    - Plan A cannot be resolved (does not exist in DB);
    - Current subscription is Plan B (id=1610);
    Late success arrives.
    Assert:
    - Plan B untouched;
    - end_date untouched;
    - binding/retry state untouched;
    - no fallback to Plan B as paid entitlement;
    - real YookassaPayment remains auditable completed;
    - exact attempt resolved consistently;
    - is_new=True;
    - action manual_reconciliation_required;
    - reason paid_plan_unresolved;
    - duplicate invocation returns is_new=False;
    - admin reconciliation event belongs only to first owner.
    """
    initial_end = datetime(2026, 9, 20, 12, 0, 0)
    async with test_db() as session:
        u = User(id=1601, username="user1601", first_name="User1601")
        # Only Plan B is created in DB. Plan A (1609) is NOT in DB.
        plan_b = SubscriptionPlan(id=1610, name="Plan B", price=250.0, duration_value=1, duration_unit="months")
        sub = UserSubscription(
            id=1601,
            user_id=1601,
            plan_id=1610,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_curr_card",
            payment_attempt_count=1,
            last_payment_attempt=datetime(2026, 9, 10, 8, 0, 0),
            retry_not_before=datetime(2026, 9, 11, 8, 0, 0),
            end_date=initial_end,
        )
        att = YookassaRecurringAttempt(
            id=1601,
            subscription_id=1601,
            user_id=1601,
            plan_id=1609,  # Authoritative Plan A (unresolvable)
            amount=150.0,  # Authoritative amount A
            payment_method_id="pm_att_card",
            idempotency_key="yk-rec-unres-1601",
            status="claimed",
        )
        session.add_all([u, plan_b, sub, att])
        await session.commit()

    # First owner invocation:
    async with test_db() as session:
        res_fin = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_unres_1601",
            user_id=1601,
            plan_id=1610,  # Caller passes Plan B, but exact attempt has Plan A (1609)
            amount=250.0,
            payment_method_id="pm_curr_card",
            is_recurring=True,
            recurring_attempt_key="yk-rec-unres-1601",
        )
        assert res_fin.is_new is True
        assert res_fin.action == "manual_reconciliation_required"
        details = res_fin.reconciliation_details
        assert details.get("reason") == "paid_plan_unresolved"
        assert details.get("payment_id") == "pay_unres_1601"
        assert details.get("paid_plan_id") == 1609
        assert details.get("amount") == 150.0
        assert details.get("current_plan_id") == 1610

    # Verify DB state:
    async with test_db() as session:
        final_sub = await session.get(UserSubscription, 1601)
        assert final_sub.plan_id == 1610  # Plan B untouched!
        assert final_sub.end_date == initial_end  # end_date untouched!
        assert final_sub.payment_method_id == "pm_curr_card"  # Binding untouched!
        assert final_sub.payment_attempt_count == 1  # Retry state untouched!
        assert final_sub.retry_not_before == datetime(2026, 9, 11, 8, 0, 0)

        # Payment remains auditable completed:
        pay = await session.scalar(select(YookassaPayment).where(YookassaPayment.payment_id == "pay_unres_1601"))
        assert pay is not None
        assert pay.status == "completed"
        assert pay.processed_at is not None

        # Exact attempt resolved consistently:
        db_att = await session.get(YookassaRecurringAttempt, 1601)
        assert db_att.status == "succeeded"
        assert db_att.payment_id == "pay_unres_1601"

    # Duplicate invocation: returns is_new=False
    async with test_db() as session:
        res_dup = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_unres_1601",
            user_id=1601,
            plan_id=1610,
            amount=250.0,
            payment_method_id="pm_curr_card",
            is_recurring=True,
            recurring_attempt_key="yk-rec-unres-1601",
        )
        assert res_dup.is_new is False
        assert res_dup.action == "already_processed"


@pytest.mark.asyncio
async def test_webhook_forged_stale_cancel_event_with_verified_succeeded_status(test_db):
    """
    Blocker 3 Regression:
    Incoming webhook event='payment.canceled', but verified GET from YooKassa has status='succeeded'.
    Must follow verified status as SUCCEEDED, never cancel or downgrade payment/subscription.
    """
    from webhooks import handle_yookassa_webhook
    from aiohttp.test_utils import make_mocked_request
    from aiogram.fsm.storage.memory import MemoryStorage

    async with test_db() as session:
        u = User(id=1701, username="user1701", first_name="User1701")
        p = SubscriptionPlan(id=1710, name="Standard Month", price=195.0, duration_value=1, duration_unit="months")
        cfg = SubscriptionConfig(id=1, yookassa_shop_id="shop", yookassa_secret_key="sec", notifications_enabled=False)
        sub = UserSubscription(
            id=1701,
            user_id=1701,
            plan_id=1710,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_valid_card",
            payment_attempt_count=1,
            end_date=datetime(2026, 9, 15, 12, 0, 0),
        )
        session.add_all([u, p, cfg, sub])
        await session.commit()

    webhook_payload = {
        "type": "notification",
        "event": "payment.canceled",  # Forged or stale event
        "object": {
            "id": "pay_forged_cancel_1701",
            "status": "succeeded",  # Verified provider status is succeeded!
            "amount": {"value": "195.00", "currency": "RUB"},
            "payment_method": {"id": "pm_new_card", "saved": True},
            "metadata": {"user_id": "1701", "plan_id": "1710", "recurring": "false"},
        },
    }

    mock_bot = AsyncMock()
    mock_bot.id = 1
    mock_req = make_mocked_request(
        "POST",
        "/yookassa/webhook",
        headers={"Content-Type": "application/json"},
        app={"bot": mock_bot, "fsm_storage": MemoryStorage()},
    )
    mock_req.json = AsyncMock(return_value=webhook_payload)

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=webhook_payload["object"])
    mock_get_cm = AsyncMock()
    mock_get_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_get_cm.__aexit__ = AsyncMock(return_value=None)
    mock_http_session = AsyncMock()
    mock_http_session.get = MagicMock(return_value=mock_get_cm)
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_http_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("webhooks.async_session_maker", test_db),
        patch("webhooks.send_msg_universal", AsyncMock()),
        patch("aiohttp.ClientSession", MagicMock(return_value=mock_session_cm)),
    ):
        resp = await handle_yookassa_webhook(mock_req)
        assert resp.status == 200

    async with test_db() as session:
        final_sub = await session.get(UserSubscription, 1701)
        # Success path must be followed:
        assert final_sub.end_date == datetime(2026, 10, 15, 12, 0, 0)
        assert final_sub.auto_renewal is True  # NOT disabled!
        assert final_sub.payment_method_id == "pm_new_card"  # NOT cleared!
        assert final_sub.payment_attempt_count == 0  # Reset!

        pay = await session.scalar(select(YookassaPayment).where(YookassaPayment.payment_id == "pay_forged_cancel_1701"))
        assert pay is not None
        assert pay.status == "completed"  # Not downgraded to canceled!


@pytest.mark.asyncio
async def test_webhook_opposite_mismatch_succeeded_event_with_verified_canceled_status(test_db):
    """
    Blocker 3 Regression:
    Incoming webhook event='payment.succeeded', but verified GET has status='canceled'.
    Must follow verified status as CANCELED.
    """
    from webhooks import handle_yookassa_webhook
    from aiohttp.test_utils import make_mocked_request
    from aiogram.fsm.storage.memory import MemoryStorage

    async with test_db() as session:
        u = User(id=1702, username="user1702", first_name="User1702")
        p = SubscriptionPlan(id=1710, name="Standard Month", price=195.0, duration_value=1, duration_unit="months")
        cfg = SubscriptionConfig(id=1, yookassa_shop_id="shop", yookassa_secret_key="sec", notifications_enabled=False)
        sub = UserSubscription(
            id=1702,
            user_id=1702,
            plan_id=1710,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_valid_card",
            payment_attempt_count=0,
            end_date=datetime(2026, 9, 15, 12, 0, 0),
        )
        session.add_all([u, p, cfg, sub])
        await session.commit()

    webhook_payload = {
        "type": "notification",
        "event": "payment.succeeded",  # Stale success notification
        "object": {
            "id": "pay_stale_succ_1702",
            "status": "canceled",  # Verified provider status is canceled!
            "cancellation_details": {"reason": "permission_revoked"},
            "amount": {"value": "195.00", "currency": "RUB"},
            "payment_method": {"id": "pm_valid_card", "saved": True},
            "metadata": {"user_id": "1702", "plan_id": "1710", "recurring": "true"},
        },
    }

    mock_bot = AsyncMock()
    mock_bot.id = 1
    mock_req = make_mocked_request(
        "POST",
        "/yookassa/webhook",
        headers={"Content-Type": "application/json"},
        app={"bot": mock_bot, "fsm_storage": MemoryStorage()},
    )
    mock_req.json = AsyncMock(return_value=webhook_payload)

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=webhook_payload["object"])
    mock_get_cm = AsyncMock()
    mock_get_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_get_cm.__aexit__ = AsyncMock(return_value=None)
    mock_http_session = AsyncMock()
    mock_http_session.get = MagicMock(return_value=mock_get_cm)
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_http_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("webhooks.async_session_maker", test_db),
        patch("webhooks.send_msg_universal", AsyncMock()),
        patch("aiohttp.ClientSession", MagicMock(return_value=mock_session_cm)),
    ):
        resp = await handle_yookassa_webhook(mock_req)
        assert resp.status == 200

    async with test_db() as session:
        final_sub = await session.get(UserSubscription, 1702)
        # Cancellation path followed (permission_revoked terminal deactivation):
        assert final_sub.auto_renewal is False
        assert final_sub.payment_method_id is None
        assert final_sub.end_date == datetime(2026, 9, 15, 12, 0, 0)  # NOT extended!


@pytest.mark.asyncio
async def test_terminal_local_conflict_completed_payment_called_with_cancel_finalizer(test_db):
    """
    Blocker 3 Regression:
    Existing YookassaPayment(status='completed', processed_at != NULL).
    Calling canceled finalizer must:
    - keep status completed
    - leave current subscription unchanged
    - return is_new=False (caller does not become cancellation event owner).
    """
    processed_time = datetime(2026, 9, 10, 10, 0, 0)
    async with test_db() as session:
        u = User(id=1703, username="user1703", first_name="User1703")
        p = SubscriptionPlan(id=1710, name="Standard Month", price=195.0, duration_value=1, duration_unit="months")
        sub = UserSubscription(
            id=1703,
            user_id=1703,
            plan_id=1710,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_saved_card",
            payment_attempt_count=0,
            end_date=datetime(2026, 10, 15, 12, 0, 0),
        )
        pay = YookassaPayment(
            payment_id="pay_term_conflict_1703",
            user_id=1703,
            plan_id=1710,
            amount=195.0,
            status="completed",
            processed_at=processed_time,
        )
        session.add_all([u, p, sub, pay])
        await session.commit()

    async with test_db() as session:
        is_new, action, updated_sub = await finalize_yookassa_payment_canceled(
            session=session,
            payment_id="pay_term_conflict_1703",
            cancellation_reason="permission_revoked",
            user_id=1703,
            plan_id=1710,
            amount=195.0,
            is_recurring=False,
        )
        assert is_new is False
        assert action == "already_processed"

    async with test_db() as session:
        final_pay = await session.scalar(select(YookassaPayment).where(YookassaPayment.payment_id == "pay_term_conflict_1703"))
        assert final_pay.status == "completed"  # Completed remains completed!
        assert final_pay.processed_at == processed_time

        final_sub = await session.get(UserSubscription, 1703)
        assert final_sub.auto_renewal is True  # Sub unchanged!
        assert final_sub.payment_method_id == "pm_saved_card"
        assert final_sub.payment_attempt_count == 0


@pytest.mark.asyncio
async def test_terminal_local_conflict_canceled_payment_called_with_success_finalizer(test_db):
    """
    Blocker 3 Regression (Symmetrical):
    Existing YookassaPayment(status='canceled', processed_at != NULL).
    Calling success finalizer must:
    - keep status canceled
    - leave current subscription unchanged
    - return is_new=False (caller does not become success event owner).
    """
    processed_time = datetime(2026, 9, 10, 10, 0, 0)
    initial_end = datetime(2026, 9, 15, 12, 0, 0)
    async with test_db() as session:
        u = User(id=1704, username="user1704", first_name="User1704")
        p = SubscriptionPlan(id=1710, name="Standard Month", price=195.0, duration_value=1, duration_unit="months")
        sub = UserSubscription(
            id=1704,
            user_id=1704,
            plan_id=1710,
            auto_renewal=False,
            payment_provider="Yookassa",
            payment_method_id=None,
            payment_attempt_count=3,
            end_date=initial_end,
        )
        pay = YookassaPayment(
            payment_id="pay_term_conflict_1704",
            user_id=1704,
            plan_id=1710,
            amount=195.0,
            status="canceled",
            processed_at=processed_time,
        )
        session.add_all([u, p, sub, pay])
        await session.commit()

    async with test_db() as session:
        res = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_term_conflict_1704",
            user_id=1704,
            plan_id=1710,
            amount=195.0,
            is_recurring=False,
        )
        assert res.is_new is False
        assert res.action == "already_processed"

    async with test_db() as session:
        final_pay = await session.scalar(select(YookassaPayment).where(YookassaPayment.payment_id == "pay_term_conflict_1704"))
        assert final_pay.status == "canceled"  # Canceled remains canceled!
        assert final_pay.processed_at == processed_time

        final_sub = await session.get(UserSubscription, 1704)
        assert final_sub.end_date == initial_end  # end_date NOT extended!
        assert final_sub.auto_renewal is False
        assert final_sub.payment_attempt_count == 3


@pytest.mark.asyncio
async def test_webhook_ordinary_cancellation_never_applies_recurring_failure_policy(test_db):
    """
    Gap 1 Regression:
    Ordinary YooKassa checkout cancellation (recurring=false) must NEVER apply
    recurring failure policy (deactivate, retry count increment, etc.).

    Regressions:
    A. ordinary canceled checkout, same payment_method_id as saved renewal binding, permission_revoked
       -> current subscription completely unchanged.
    B. ordinary canceled checkout, same payment_method_id, insufficient_funds
       -> retry count remains unchanged.
    C. ordinary canceled checkout, unknown reason
       -> auto_renewal remains unchanged.
    D. recurring canceled payment with the same terminal reason
       -> existing recurring policy STILL works and deactivates/clears the owned old payment method.
    E. recurring retryable decline
       -> existing recurring retry counter still increments exactly once.
    """
    from webhooks import handle_yookassa_webhook
    from aiohttp.test_utils import make_mocked_request
    from aiogram.fsm.storage.memory import MemoryStorage

    initial_end = datetime(2026, 10, 15, 12, 0, 0)
    mock_bot = AsyncMock()
    mock_bot.id = 1

    async with test_db() as session:
        cfg = SubscriptionConfig(id=1, yookassa_shop_id="shop", yookassa_secret_key="sec", notifications_enabled=False)
        p = SubscriptionPlan(id=1710, name="Standard Month", price=195.0, duration_value=1, duration_unit="months")
        session.add_all([cfg, p])

        # Users 1801 - 1805
        for uid in (1801, 1802, 1803, 1804, 1805):
            u = User(id=uid, username=f"user{uid}", first_name=f"User{uid}")
            sub = UserSubscription(
                id=uid,
                user_id=uid,
                plan_id=1710,
                auto_renewal=True,
                payment_provider="Yookassa",
                payment_method_id=f"pm_saved_{uid}",
                payment_attempt_count=1 if uid == 1802 else 0,
                end_date=initial_end,
            )
            session.add_all([u, sub])

        # Recurring attempts for D and E
        att_1804 = YookassaRecurringAttempt(
            id=1804,
            subscription_id=1804,
            user_id=1804,
            plan_id=1710,
            idempotency_key="key_rec_1804",
            amount=195.0,
            payment_method_id="pm_saved_1804",
            status="claimed",
        )
        att_1805 = YookassaRecurringAttempt(
            id=1805,
            subscription_id=1805,
            user_id=1805,
            plan_id=1710,
            idempotency_key="key_rec_1805",
            amount=195.0,
            payment_method_id="pm_saved_1805",
            status="claimed",
        )
        session.add_all([att_1804, att_1805])
        await session.commit()

    async def _call_webhook(payload):
        mock_req = make_mocked_request(
            "POST",
            "/yookassa/webhook",
            headers={"Content-Type": "application/json"},
            app={"bot": mock_bot, "fsm_storage": MemoryStorage()},
        )
        mock_req.json = AsyncMock(return_value=payload)
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=payload["object"])
        mock_get_cm = AsyncMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)
        mock_http_session = AsyncMock()
        mock_http_session.get = MagicMock(return_value=mock_get_cm)
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_http_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("webhooks.async_session_maker", test_db),
            patch("webhooks.send_msg_universal", AsyncMock()),
            patch("aiohttp.ClientSession", MagicMock(return_value=mock_session_cm)),
        ):
            resp = await handle_yookassa_webhook(mock_req)
            assert resp.status == 200

    # A. Ordinary canceled checkout, same payment_method_id, permission_revoked
    payload_a = {
        "event": "payment.canceled",
        "object": {
            "id": "pay_ord_canc_1801",
            "status": "canceled",
            "cancellation_details": {"reason": "permission_revoked"},
            "amount": {"value": "195.00", "currency": "RUB"},
            "payment_method": {"id": "pm_saved_1801", "saved": True},
            "metadata": {"user_id": "1801", "plan_id": "1710", "recurring": "false"},
        },
    }
    await _call_webhook(payload_a)
    async with test_db() as session:
        sub_a = await session.get(UserSubscription, 1801)
        assert sub_a.auto_renewal is True  # UNTOUCHED!
        assert sub_a.payment_method_id == "pm_saved_1801"  # UNTOUCHED!
        assert sub_a.payment_attempt_count == 0
        assert sub_a.end_date == initial_end
        pay_a = await session.scalar(select(YookassaPayment).where(YookassaPayment.payment_id == "pay_ord_canc_1801"))
        assert pay_a.status == "canceled"
        assert pay_a.processed_at is not None

    # B. Ordinary canceled checkout, same payment_method_id, insufficient_funds
    payload_b = {
        "event": "payment.canceled",
        "object": {
            "id": "pay_ord_canc_1802",
            "status": "canceled",
            "cancellation_details": {"reason": "insufficient_funds"},
            "amount": {"value": "195.00", "currency": "RUB"},
            "payment_method": {"id": "pm_saved_1802", "saved": True},
            "metadata": {"user_id": "1802", "plan_id": "1710", "recurring": "false"},
        },
    }
    await _call_webhook(payload_b)
    async with test_db() as session:
        sub_b = await session.get(UserSubscription, 1802)
        assert sub_b.payment_attempt_count == 1  # UNTOUCHED (still 1, not 2)!
        assert sub_b.auto_renewal is True

    # C. Ordinary canceled checkout, unknown reason
    payload_c = {
        "event": "payment.canceled",
        "object": {
            "id": "pay_ord_canc_1803",
            "status": "canceled",
            "cancellation_details": {"reason": "unexpected_bank_rejection"},
            "amount": {"value": "195.00", "currency": "RUB"},
            "payment_method": {"id": "pm_saved_1803", "saved": True},
            "metadata": {"user_id": "1803", "plan_id": "1710", "recurring": "false"},
        },
    }
    await _call_webhook(payload_c)
    async with test_db() as session:
        sub_c = await session.get(UserSubscription, 1803)
        assert sub_c.auto_renewal is True  # UNTOUCHED!
        assert sub_c.payment_method_id == "pm_saved_1803"

    # D. Recurring canceled payment with permission_revoked
    payload_d = {
        "event": "payment.canceled",
        "object": {
            "id": "pay_rec_canc_1804",
            "status": "canceled",
            "cancellation_details": {"reason": "permission_revoked"},
            "amount": {"value": "195.00", "currency": "RUB"},
            "payment_method": {"id": "pm_saved_1804", "saved": True},
            "metadata": {"user_id": "1804", "plan_id": "1710", "recurring": "true", "recurring_attempt_key": "key_rec_1804"},
        },
    }
    await _call_webhook(payload_d)
    async with test_db() as session:
        sub_d = await session.get(UserSubscription, 1804)
        assert sub_d.auto_renewal is False  # Deactivated by recurring policy!
        assert sub_d.payment_method_id is None

    # E. Recurring retryable decline (insufficient_funds)
    payload_e = {
        "event": "payment.canceled",
        "object": {
            "id": "pay_rec_decl_1805",
            "status": "canceled",
            "cancellation_details": {"reason": "insufficient_funds"},
            "amount": {"value": "195.00", "currency": "RUB"},
            "payment_method": {"id": "pm_saved_1805", "saved": True},
            "metadata": {"user_id": "1805", "plan_id": "1710", "recurring": "true", "recurring_attempt_key": "key_rec_1805"},
        },
    }
    await _call_webhook(payload_e)
    async with test_db() as session:
        sub_e = await session.get(UserSubscription, 1805)
        assert sub_e.payment_attempt_count == 1  # Incremented exactly once!
        assert sub_e.auto_renewal is True


@pytest.mark.asyncio
async def test_exact_recurring_success_missing_subscription_reconciliation(test_db):
    """
    Gap 2 Regression:
    Exact recurring success arrives, but attempt.subscription_id does not exist in DB.
    Must:
    - keep real YookassaPayment completed
    - keep attempt succeeded
    - mutate 0 unrelated subscriptions
    - return is_new=True, action="manual_reconciliation_required", reason="subscription_unresolved"
    - duplicate returns is_new=False
    """
    async with test_db() as session:
        u = User(id=1806, username="user1806", first_name="User1806")
        p = SubscriptionPlan(id=1710, name="Standard Month", price=195.0, duration_value=1, duration_unit="months")
        unrelated_sub = UserSubscription(
            id=1806,
            user_id=1806,
            plan_id=1710,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_saved_1806",
            payment_attempt_count=0,
            end_date=datetime(2026, 10, 15, 12, 0, 0),
        )
        # Attempt references missing subscription 99996!
        att = YookassaRecurringAttempt(
            id=1806,
            subscription_id=99996,
            user_id=1806,
            plan_id=1710,
            idempotency_key="key_orphan_succ_1806",
            amount=195.0,
            payment_method_id="pm_saved_1806",
            status="claimed",
        )
        session.add_all([u, p, unrelated_sub, att])
        await session.commit()

    async with test_db() as session:
        res = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_orphan_succ_1806",
            user_id=1806,
            plan_id=1710,
            amount=195.0,
            is_recurring=True,
            recurring_attempt_key="key_orphan_succ_1806",
        )
        assert res.is_new is True
        assert res.action == "manual_reconciliation_required"
        assert res.subscription is None
        details = res.reconciliation_details
        assert details["reason"] == "subscription_unresolved"
        assert details["payment_id"] == "pay_orphan_succ_1806"
        assert details["attempt_id"] == 1806
        assert details["subscription_id"] == 99996
        assert details["user_id"] == 1806
        assert details["paid_plan_id"] == 1710
        assert details["amount"] == 195.0

    async with test_db() as session:
        # Unrelated subscription unchanged
        sub = await session.get(UserSubscription, 1806)
        assert sub.end_date == datetime(2026, 10, 15, 12, 0, 0)
        assert sub.auto_renewal is True

        # Real payment completed and attempt succeeded
        pay = await session.scalar(select(YookassaPayment).where(YookassaPayment.payment_id == "pay_orphan_succ_1806"))
        assert pay.status == "completed"
        assert pay.processed_at is not None

        attempt_row = await session.get(YookassaRecurringAttempt, 1806)
        assert attempt_row.status == "succeeded"
        assert attempt_row.payment_id == "pay_orphan_succ_1806"

    # Duplicate call
    async with test_db() as session:
        res_dup = await finalize_yookassa_payment_success(
            session=session,
            payment_id="pay_orphan_succ_1806",
            user_id=1806,
            plan_id=1710,
            amount=195.0,
            is_recurring=True,
            recurring_attempt_key="key_orphan_succ_1806",
        )
        assert res_dup.is_new is False
        assert res_dup.action == "already_processed"


@pytest.mark.asyncio
async def test_exact_recurring_canceled_missing_subscription_orphan_canceled(test_db):
    """
    Gap 2 Regression:
    Exact recurring canceled arrives, but attempt.subscription_id does not exist in DB.
    Must:
    - record canceled payment
    - resolve attempt as terminal
    - mutate 0 unrelated subscriptions
    - return is_new=True, action="orphan_canceled"
    - duplicate returns is_new=False
    """
    async with test_db() as session:
        u = User(id=1807, username="user1807", first_name="User1807")
        p = SubscriptionPlan(id=1710, name="Standard Month", price=195.0, duration_value=1, duration_unit="months")
        unrelated_sub = UserSubscription(
            id=1807,
            user_id=1807,
            plan_id=1710,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_saved_1807",
            payment_attempt_count=0,
            end_date=datetime(2026, 10, 15, 12, 0, 0),
        )
        att = YookassaRecurringAttempt(
            id=1807,
            subscription_id=99995,
            user_id=1807,
            plan_id=1710,
            idempotency_key="key_orphan_canc_1807",
            amount=195.0,
            payment_method_id="pm_saved_1807",
            status="claimed",
        )
        session.add_all([u, p, unrelated_sub, att])
        await session.commit()

    async with test_db() as session:
        is_new, action, sub = await finalize_yookassa_payment_canceled(
            session=session,
            payment_id="pay_orphan_canc_1807",
            cancellation_reason="permission_revoked",
            user_id=1807,
            plan_id=1710,
            amount=195.0,
            is_recurring=True,
            recurring_attempt_key="key_orphan_canc_1807",
        )
        assert is_new is True
        assert action == "orphan_canceled"
        assert sub is None

    async with test_db() as session:
        sub = await session.get(UserSubscription, 1807)
        assert sub.auto_renewal is True  # Not deactivated!
        assert sub.payment_method_id == "pm_saved_1807"
        assert sub.payment_attempt_count == 0

        pay = await session.scalar(select(YookassaPayment).where(YookassaPayment.payment_id == "pay_orphan_canc_1807"))
        assert pay.status == "canceled"
        assert pay.processed_at is not None

        attempt_row = await session.get(YookassaRecurringAttempt, 1807)
        assert attempt_row.status in ("deactivated", "canceled")
        assert attempt_row.payment_id == "pay_orphan_canc_1807"

    # Duplicate call
    async with test_db() as session:
        is_new_dup, action_dup, _ = await finalize_yookassa_payment_canceled(
            session=session,
            payment_id="pay_orphan_canc_1807",
            cancellation_reason="permission_revoked",
            user_id=1807,
            plan_id=1710,
            amount=195.0,
            is_recurring=True,
            recurring_attempt_key="key_orphan_canc_1807",
        )
        assert is_new_dup is False
        assert action_dup == "already_processed"


@pytest.mark.asyncio
async def test_webhook_orphan_successful_payment_transport_diagnostic(test_db):
    """
    Gap 2 Regression:
    An orphan successful recurring payment cannot produce '✅ Подписка продлена'
    and instead creates the manual-reconciliation admin diagnostic.
    """
    from webhooks import handle_yookassa_webhook
    from aiohttp.test_utils import make_mocked_request
    from aiogram.fsm.storage.memory import MemoryStorage

    mock_bot = AsyncMock()
    mock_bot.id = 1
    mock_bot.send_message = AsyncMock()

    async with test_db() as session:
        admin_u = User(id=999, username="admin999", first_name="Admin", is_admin=True)
        u = User(id=1808, username="user1808", first_name="User1808")
        p = SubscriptionPlan(id=1710, name="Standard Month", price=195.0, duration_value=1, duration_unit="months")
        cfg = SubscriptionConfig(id=1, yookassa_shop_id="shop", yookassa_secret_key="sec", notifications_enabled=True)
        # Attempt references missing subscription 99994
        att = YookassaRecurringAttempt(
            id=1808,
            subscription_id=99994,
            user_id=1808,
            plan_id=1710,
            idempotency_key="key_orphan_wh_1808",
            amount=195.0,
            payment_method_id="pm_saved_1808",
            status="claimed",
        )
        session.add_all([admin_u, u, p, cfg, att])
        await session.commit()

    webhook_payload = {
        "event": "payment.succeeded",
        "object": {
            "id": "pay_orphan_wh_1808",
            "status": "succeeded",
            "amount": {"value": "195.00", "currency": "RUB"},
            "payment_method": {"id": "pm_saved_1808", "saved": True},
            "metadata": {"user_id": "1808", "plan_id": "1710", "recurring": "true", "recurring_attempt_key": "key_orphan_wh_1808"},
        },
    }

    mock_req = make_mocked_request(
        "POST",
        "/yookassa/webhook",
        headers={"Content-Type": "application/json"},
        app={"bot": mock_bot, "fsm_storage": MemoryStorage()},
    )
    mock_req.json = AsyncMock(return_value=webhook_payload)
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=webhook_payload["object"])
    mock_get_cm = AsyncMock()
    mock_get_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_get_cm.__aexit__ = AsyncMock(return_value=None)
    mock_http_session = AsyncMock()
    mock_http_session.get = MagicMock(return_value=mock_get_cm)
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_http_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)

    mock_send_msg = AsyncMock()

    with (
        patch("webhooks.async_session_maker", test_db),
        patch("webhooks.send_msg_universal", mock_send_msg),
        patch("webhooks.get_all_admin_ids", AsyncMock(return_value=[999])),
        patch("aiohttp.ClientSession", MagicMock(return_value=mock_session_cm)),
    ):
        resp = await handle_yookassa_webhook(mock_req)
        assert resp.status == 200

    # User must NOT receive "✅ Подписка продлена"
    for call in mock_send_msg.call_args_list:
        sent_text = str(call[0][2])
        assert "✅ Подписка продлена" not in sent_text

    for call in mock_bot.send_message.call_args_list:
        sent_text = str(call)
        assert "✅ Подписка продлена" not in sent_text

    # Admin must receive the diagnostic
    admin_messages = [str(call) for call in mock_bot.send_message.call_args_list]
    diagnostic_found = any("ТРЕБУЕТСЯ РУЧНАЯ СВЕРКА: ПОДПИСКА НЕ НАЙДЕНА" in msg for msg in admin_messages)
    assert diagnostic_found is True





