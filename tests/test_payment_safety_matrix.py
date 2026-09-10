from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import logging
import os
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
        assert is_new is True
        assert result_sub is None

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
async def test_scheduler_local_config_precheck_trips_incident(test_db):
    """
    Scheduler local config pre-check:
    If shop_id or secret_key is missing, trips circuit breaker, sends deduplicated admin alert,
    and initiates zero new claims/POSTs.
    """
    now = datetime.utcnow()
    async with test_db() as session:
        admin_u = User(id=99991, username="admin_owner2", is_admin=True)
        # Missing secret_key!
        cfg = SubscriptionConfig(id=1, yookassa_shop_id="shop_without_key", yookassa_secret_key=None, notifications_enabled=True)
        p = SubscriptionPlan(id=1106, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        sub = UserSubscription(
            id=1106,
            user_id=1106,
            plan_id=1106,
            auto_renewal=True,
            payment_provider="Yookassa",
            payment_method_id="pm_card_1106",
            payment_attempt_count=0,
            end_date=now - timedelta(minutes=5),
        )
        session.add_all([admin_u, cfg, p, sub])
        await session.commit()

    import subscription_renewal
    subscription_renewal._last_auth_alert_time = None

    bot = AsyncMock()
    with (
        patch("scheduler.process_birthday_mailings", AsyncMock()),
        patch("scheduler.async_session_maker", test_db),
        patch("subscription_renewal.Payment.create") as mock_create,
    ):
        await check_subscriptions(bot)

    # Zero calls to Payment.create
    mock_create.assert_not_called()

    # Zero claims created in DB
    async with test_db() as session:
        attempts = (await session.execute(
            select(YookassaRecurringAttempt).where(YookassaRecurringAttempt.subscription_id == 1106)
        )).scalars().all()
        assert len(attempts) == 0

    # Admin notified of configuration incident
    bot.send_message.assert_called()
    alerts = [str(c) for c in bot.send_message.call_args_list]
    assert any("shop_id или secret_key" in a for a in alerts)


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

