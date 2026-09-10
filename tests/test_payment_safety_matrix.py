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
    mask_payment_method_id,
    should_reconcile_attempt,
    update_yookassa_attempt_pending,
    update_yookassa_attempt_unknown,
)
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
async def test_atomic_claim_concurrency(test_db):
    """
    Section 3: Atomic claim semantics BEFORE Payment.create.
    Two concurrent callers -> exactly one claimed attempt, second caller gets claimed=False.
    Also verifies DB-level partial index conflict handling.
    """
    async with test_db() as session:
        user = User(id=104, username="test104", first_name="User104")
        plan = SubscriptionPlan(id=4, name="Standard", price=195.0, duration_value=1, duration_unit="months")
        session.add_all([user, plan])
        await session.commit()

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
        session.add(sub)
        await session.commit()

        # 1. Caller 1 claims successfully
        res1 = await claim_yookassa_recurring_attempt(session, sub, plan, 195.0, "scheduler")
        assert res1.claimed is True
        assert res1.attempt is not None
        assert res1.attempt.status == "claimed"

        # 2. Caller 2 attempts to claim while an unresolved attempt exists -> BLOCKED
        res2 = await claim_yookassa_recurring_attempt(session, sub, plan, 195.0, "max_manual_retry")
        assert res2.claimed is False
        assert res2.reason == "unresolved_exists"
        assert res2.attempt.id == res1.attempt.id

        # 3. Simulate DB IntegrityError race (e.g. concurrent commit collision)
        from sqlalchemy.exc import IntegrityError
        with patch.object(session, "commit", side_effect=IntegrityError("stmt", "params", "orig")):
            res_race = await claim_yookassa_recurring_attempt(session, sub, plan, 195.0, "telegram_manual")
            assert res_race.claimed is False
            assert res_race.reason == "unresolved_exists"


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
    sub = SimpleNamespace(id=1, user_id=100, payment_method_id="pm_123456789")
    plan = SimpleNamespace(id=1, name="Standard")
    config = SimpleNamespace(yookassa_shop_id="shop", yookassa_secret_key="secret")
    attempt = YookassaRecurringAttempt(
        subscription_id=1,
        user_id=100,
        plan_id=1,
        idempotency_key="key-tax",
        amount=195.0,
        status="claimed",
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

    attempt = YookassaRecurringAttempt(
        subscription_id=1,
        user_id=500,
        plan_id=1,
        idempotency_key="key-log-test",
        amount=195.0,
        payment_method_id=raw_token,
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
