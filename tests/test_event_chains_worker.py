import asyncio
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("BOT_TOKEN", "test_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("BASE_WEBHOOK_URL", "https://example.com")

import pytest
import pytest_asyncio
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
from sqlalchemy import select, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import database
from database import (
    Base,
    EventDelivery,
    EventRegistration,
    EventScenario,
    EventStep,
    User,
    create_event_delivery_force_retry,
)
from event_delivery_worker import (
    PreTransmissionError,
    claim_pending_deliveries,
    finalize_delivery_outcome,
    process_single_delivery,
    recover_expired_leases,
    reset_cooldowns,
    run_event_delivery_worker_cycle,
)
from event_service import (
    cancel_event_registration,
    prepare_and_lock_delivery_for_sending,
    register_user_for_event,
    reschedule_event,
)
from max_messenger_bot.api import MaxApiError

from max_messenger_bot.models import MAX_ID_OFFSET


@pytest.fixture(autouse=True)
def clean_cooldowns():
    reset_cooldowns()
    yield
    reset_cooldowns()



@pytest_asyncio.fixture
async def sqlite_session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        echo=False,
    )
    async with engine.connect() as conn:
        await conn.execute(text("PRAGMA foreign_keys = ON;"))
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    yield session_factory
    await engine.dispose()


async def _create_test_user(session_factory, user_id=1001, name="Иван"):
    async with session_factory() as session:
        user = User(
            id=user_id,
            first_name=name,
            name=name,
            username=f"user_{user_id}",
            metadata_json=None,
        )
        session.add(user)
        await session.commit()
    return user_id


async def _create_test_scenario(
    session_factory,
    code="worker_ev",
    starts_at=None,
    duration=120,
    is_active=True,
    link="https://example.com/live",
    steps=None,
):
    if starts_at is None:
        starts_at = datetime.utcnow() + timedelta(days=2)
    if steps is None:
        steps = [
            (-1440, "Завтра эфир {event_title}!"),
            (-60, "Через час эфир {event_title}!"),
        ]

    async with session_factory() as session:
        scen = EventScenario(
            code=code,
            event_type="webinar",
            title=f"Вебинар {code}",
            starts_at=starts_at,
            duration_minutes=duration,
            link=link,
            place=None,
            is_active=is_active,
            is_archived=False,
            schedule_revision=1,
            content_version=1,
        )
        session.add(scen)
        await session.flush()

        for idx, (offset, msg) in enumerate(steps):
            step = EventStep(
                event_id=scen.id,
                offset_minutes=offset,
                message_text=msg,
                step_version=1,
                sort_order=idx,
                is_archived=False,
            )
            session.add(step)
        await session.commit()
        return scen.id


# ==============================================================================
# Worker Claim & Lease Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_worker_claim_filters_due_at_and_active_scenario(sqlite_session_factory):
    """Worker claims only pending deliveries where due_at <= now and scenario is active and non-archived."""
    user_id = await _create_test_user(sqlite_session_factory, 1001)
    scen_id = await _create_test_scenario(sqlite_session_factory, "claim_ev")

    # Register user -> creates 1 immediate confirmation (due now) and 2 future steps (due in 1d and 2d)
    await register_user_for_event(sqlite_session_factory, user_id, "claim_ev", platform="telegram")

    claimed = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)

    # Exactly 1 immediate confirmation should be claimed
    assert len(claimed) == 1
    d_id, token = claimed[0]

    async with sqlite_session_factory() as session:
        deliv = await session.get(EventDelivery, d_id)
        assert deliv.status == "claimed"
        assert deliv.claim_token == token
        assert deliv.lease_until is not None
        assert deliv.claimed_at is not None

        # The other 2 steps remain pending because due_at > now
        pending_steps = (
            await session.execute(
                select(EventDelivery).where(
                    EventDelivery.status == "pending",
                    EventDelivery.message_kind == "scheduled_step",
                )
            )
        ).scalars().all()
        assert len(pending_steps) == 2


@pytest.mark.asyncio
async def test_expired_claim_lease_recovery(sqlite_session_factory):
    """Requirement F: Claimed deliveries with expired lease and send_started_at IS NULL return to pending."""
    user_id = await _create_test_user(sqlite_session_factory, 1002)
    scen_id = await _create_test_scenario(sqlite_session_factory, "lease_ev")
    reg_res = await register_user_for_event(sqlite_session_factory, user_id, "lease_ev", platform="telegram")

    claimed = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    assert len(claimed) == 1
    d_id, token = claimed[0]

    # Artificially expire the lease
    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, d_id)
        d.lease_until = datetime.utcnow() - timedelta(seconds=10)
        d.send_started_at = None
        await session.commit()

    rec_claimed, rec_sending = await recover_expired_leases(sqlite_session_factory)
    assert rec_claimed == 1
    assert rec_sending == 0

    async with sqlite_session_factory() as session:
        d_after = await session.get(EventDelivery, d_id)
        assert d_after.status == "pending"
        assert d_after.claim_token is None
        assert d_after.lease_until is None


@pytest.mark.asyncio
async def test_expired_sending_lease_transitions_to_uncertain(sqlite_session_factory):
    """Requirement F: Sending deliveries with expired lease (>180s) transition to uncertain, not pending."""
    user_id = await _create_test_user(sqlite_session_factory, 1003)
    scen_id = await _create_test_scenario(sqlite_session_factory, "sending_ev")
    reg_res = await register_user_for_event(sqlite_session_factory, user_id, "sending_ev", platform="telegram")

    # Manually put delivery in 'sending' with expired lease
    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, reg_res.immediate_delivery.id)
        d.status = "sending"
        d.send_started_at = datetime.utcnow() - timedelta(seconds=200)
        d.lease_until = datetime.utcnow() - timedelta(seconds=20)
        await session.commit()

    rec_claimed, rec_sending = await recover_expired_leases(sqlite_session_factory)
    assert rec_claimed == 0
    assert rec_sending == 1

    async with sqlite_session_factory() as session:
        d_after = await session.get(EventDelivery, reg_res.immediate_delivery.id)
        assert d_after.status == "uncertain"
        assert d_after.last_error == "Lease expired in sending state (>180s)"
        assert d_after.claim_token is None
        assert d_after.lease_until is None


# ==============================================================================
# Pre-Send CAS & Invariant Validation Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_pre_send_generation_mismatch_skips_cancelled_registration(sqlite_session_factory):
    """Requirement D: If registration was cancelled before pre-send, pre-send transitions delivery to skipped('cancelled')."""
    user_id = await _create_test_user(sqlite_session_factory, 1004)
    scen_id = await _create_test_scenario(sqlite_session_factory, "gen_mismatch_ev")
    await register_user_for_event(sqlite_session_factory, user_id, "gen_mismatch_ev", platform="telegram")

    claimed = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    d_id, token = claimed[0]

    # User cancels registration while delivery is claimed!
    async with sqlite_session_factory() as session:
        reg = (await session.execute(select(EventRegistration))).scalars().first()
        reg.status = "cancelled"
        reg.generation = 1
        await session.commit()

    ok, d, reason = await prepare_and_lock_delivery_for_sending(
        sqlite_session_factory,
        delivery_id=d_id,
        claim_token=token,
        expected_content_version=1,
        expected_step_version=None,
        rendered_text="test",
        template_text="test",
    )
    assert ok is False
    assert reason == "registration_cancelled"

    async with sqlite_session_factory() as session:
        d_after = await session.get(EventDelivery, d_id)
        assert d_after.status == "skipped"
        assert d_after.skip_reason == "cancelled"


@pytest.mark.asyncio
async def test_pre_send_schedule_revision_mismatch_aborts(sqlite_session_factory):
    """Requirement D: If scenario was rescheduled before pre-send CAS, delivery is aborted without send."""
    user_id = await _create_test_user(sqlite_session_factory, 1005)
    scen_id = await _create_test_scenario(sqlite_session_factory, "rev_mismatch_ev")
    await register_user_for_event(sqlite_session_factory, user_id, "rev_mismatch_ev", platform="telegram")

    claimed = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    d_id, token = claimed[0]

    # Scenario revision increments
    async with sqlite_session_factory() as session:
        scen = await session.get(EventScenario, scen_id)
        scen.schedule_revision = 2
        await session.commit()

    ok, d, reason = await prepare_and_lock_delivery_for_sending(
        sqlite_session_factory,
        delivery_id=d_id,
        claim_token=token,
        expected_content_version=1,
        expected_step_version=None,
        rendered_text="test",
        template_text="test",
    )
    assert ok is False
    assert reason == "schedule_rescheduled"


@pytest.mark.asyncio
async def test_pre_send_inactive_scenario_handling(sqlite_session_factory):
    """Requirement D: If scenario was deactivated, past due deliveries become skipped('disabled_past')."""
    user_id = await _create_test_user(sqlite_session_factory, 1006)
    scen_id = await _create_test_scenario(sqlite_session_factory, "inact_ev")
    await register_user_for_event(sqlite_session_factory, user_id, "inact_ev", platform="telegram")

    claimed = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    d_id, token = claimed[0]

    async with sqlite_session_factory() as session:
        scen = await session.get(EventScenario, scen_id)
        scen.is_active = False
        await session.commit()

    ok, d, reason = await prepare_and_lock_delivery_for_sending(
        sqlite_session_factory,
        delivery_id=d_id,
        claim_token=token,
        expected_content_version=1,
        expected_step_version=None,
        rendered_text="test",
        template_text="test",
    )
    assert ok is False
    assert reason == "scenario_inactive"

    async with sqlite_session_factory() as session:
        d_after = await session.get(EventDelivery, d_id)
        assert d_after.status == "skipped"
        assert d_after.skip_reason == "disabled_past"


@pytest.mark.asyncio
async def test_pre_send_archived_step_becomes_skipped(sqlite_session_factory):
    """Requirement D: If scheduled step was archived, delivery becomes skipped('step_archived')."""
    user_id = await _create_test_user(sqlite_session_factory, 1007)
    scen_id = await _create_test_scenario(sqlite_session_factory, "step_arch_ev")
    await register_user_for_event(sqlite_session_factory, user_id, "step_arch_ev", platform="telegram")

    # Manually make step 1 due now and claim it
    async with sqlite_session_factory() as session:
        step_deliv = (
            await session.execute(
                select(EventDelivery).where(EventDelivery.message_kind == "scheduled_step")
            )
        ).scalars().first()
        step_deliv.due_at = datetime.utcnow() - timedelta(minutes=1)
        step_deliv.next_attempt_at = step_deliv.due_at
        step_deliv.status = "claimed"
        step_deliv.claim_token = "tok_step"
        step_deliv.lease_until = datetime.utcnow() + timedelta(minutes=1)

        # Archive the step
        step = await session.get(EventStep, step_deliv.step_id)
        step.is_archived = True
        await session.commit()
        d_id = step_deliv.id

    ok, d, reason = await prepare_and_lock_delivery_for_sending(
        sqlite_session_factory,
        delivery_id=d_id,
        claim_token="tok_step",
        expected_content_version=1,
        expected_step_version=1,
        rendered_text="test",
        template_text="test",
    )
    assert ok is False
    assert reason == "step_archived"

    async with sqlite_session_factory() as session:
        d_after = await session.get(EventDelivery, d_id)
        assert d_after.status == "skipped"
        assert d_after.skip_reason == "step_archived"


@pytest.mark.asyncio
async def test_pre_send_snapshots_saved_accurately(sqlite_session_factory):
    """Requirement D: Rendered text snapshot and template text snapshot saved upon CAS transition to sending."""
    user_id = await _create_test_user(sqlite_session_factory, 1008)
    scen_id = await _create_test_scenario(sqlite_session_factory, "snap_ev")
    await register_user_for_event(sqlite_session_factory, user_id, "snap_ev", platform="telegram")

    claimed = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    d_id, token = claimed[0]

    ok, d, reason = await prepare_and_lock_delivery_for_sending(
        sqlite_session_factory,
        delivery_id=d_id,
        claim_token=token,
        expected_content_version=1,
        expected_step_version=None,
        rendered_text="Rendered text snapshot",
        template_text="Template text snapshot {title}",
    )
    assert ok is True

    async with sqlite_session_factory() as session:
        d_after = await session.get(EventDelivery, d_id)
        assert d_after.status == "sending"
        assert d_after.rendered_text_snapshot == "Rendered text snapshot"
        assert d_after.template_text_snapshot == "Template text snapshot {title}"
        assert d_after.attempt_count == 1
        assert d_after.send_started_at is not None


# ==============================================================================
# Transport Sends & Error Classification Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_successful_telegram_delivery(sqlite_session_factory):
    """Requirement E: Successful Telegram send transitions from sending to sent with external_message_id."""
    user_id = await _create_test_user(sqlite_session_factory, 1009)
    scen_id = await _create_test_scenario(sqlite_session_factory, "tg_succ_ev")
    await register_user_for_event(sqlite_session_factory, user_id, "tg_succ_ev", platform="telegram")

    claimed = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    d_id, token = claimed[0]

    mock_bot = MagicMock()
    mock_msg = MagicMock()
    mock_msg.message_id = 424242
    mock_bot.send_message = AsyncMock(return_value=mock_msg)

    outcome = await process_single_delivery(
        sqlite_session_factory, d_id, token, bot=mock_bot
    )
    assert outcome == "sent"

    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, d_id)
        assert d.status == "sent"
        assert d.external_message_id == "424242"
        assert d.sent_at is not None
        assert d.finished_at is not None
        assert d.claim_token is None


@pytest.mark.asyncio
async def test_successful_max_delivery(sqlite_session_factory):
    """Requirement E: Successful MAX send transitions from sending to sent with external_message_id."""
    user_id = MAX_ID_OFFSET + 1010
    await _create_test_user(sqlite_session_factory, user_id)
    scen_id = await _create_test_scenario(sqlite_session_factory, "max_succ_ev")
    await register_user_for_event(sqlite_session_factory, user_id, "max_succ_ev", platform="max")

    claimed = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    d_id, token = claimed[0]

    mock_max = MagicMock()
    mock_max.send_message = AsyncMock(return_value={"message_id": "max_msg_999"})

    outcome = await process_single_delivery(
        sqlite_session_factory, d_id, token, max_client=mock_max
    )
    assert outcome == "sent"

    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, d_id)
        assert d.status == "sent"
        assert d.external_message_id == "max_msg_999"
        assert d.sent_at is not None
        assert d.finished_at is not None


@pytest.mark.asyncio
async def test_telegram_blocked_user(sqlite_session_factory):
    """Requirement E: Telegram user blocked bot -> transitions to 'blocked'."""
    user_id = await _create_test_user(sqlite_session_factory, 1011)
    scen_id = await _create_test_scenario(sqlite_session_factory, "tg_block_ev")
    await register_user_for_event(sqlite_session_factory, user_id, "tg_block_ev", platform="telegram")

    claimed = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    d_id, token = claimed[0]

    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock(
        side_effect=TelegramForbiddenError(method=MagicMock(), message="Forbidden: bot was blocked by the user")
    )

    outcome = await process_single_delivery(
        sqlite_session_factory, d_id, token, bot=mock_bot
    )
    assert outcome == "blocked"

    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, d_id)
        assert d.status == "blocked"
        assert "blocked by the user" in d.last_error


@pytest.mark.asyncio
async def test_telegram_rate_limited_retry(sqlite_session_factory):
    """Requirement E: Telegram 429 rate limit -> transitions to pending with retry_after backoff."""
    user_id = await _create_test_user(sqlite_session_factory, 1012)
    scen_id = await _create_test_scenario(sqlite_session_factory, "tg_rate_ev")
    await register_user_for_event(sqlite_session_factory, user_id, "tg_rate_ev", platform="telegram")

    claimed = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    d_id, token = claimed[0]

    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock(
        side_effect=TelegramRetryAfter(method=MagicMock(), message="Too Many Requests: retry after 15", retry_after=15)
    )

    outcome = await process_single_delivery(
        sqlite_session_factory, d_id, token, bot=mock_bot
    )
    assert outcome == "rate_limited"

    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, d_id)
        assert d.status == "pending"
        assert d.next_attempt_at is not None


@pytest.mark.asyncio
async def test_telegram_fatal_bad_request(sqlite_session_factory):
    """Requirement E: Telegram 400 Bad Request -> transitions to terminal 'failed'."""
    user_id = await _create_test_user(sqlite_session_factory, 1013)
    scen_id = await _create_test_scenario(sqlite_session_factory, "tg_fail_ev")
    await register_user_for_event(sqlite_session_factory, user_id, "tg_fail_ev", platform="telegram")

    claimed = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    d_id, token = claimed[0]

    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock(
        side_effect=TelegramBadRequest(method=MagicMock(), message="Bad Request: chat not found")
    )

    outcome = await process_single_delivery(
        sqlite_session_factory, d_id, token, bot=mock_bot
    )
    assert outcome == "failed"

    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, d_id)
        assert d.status == "failed"
        assert d.claim_token is None


@pytest.mark.asyncio
async def test_timeout_post_request_transitions_to_uncertain(sqlite_session_factory):
    """Requirement E: Timeout after request was potentially sent -> transitions to 'uncertain' (NOT retry)."""
    user_id = await _create_test_user(sqlite_session_factory, 1014)
    scen_id = await _create_test_scenario(sqlite_session_factory, "timeout_ev")
    await register_user_for_event(sqlite_session_factory, user_id, "timeout_ev", platform="telegram")

    claimed = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    d_id, token = claimed[0]

    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock(side_effect=asyncio.TimeoutError())

    outcome = await process_single_delivery(
        sqlite_session_factory, d_id, token, bot=mock_bot
    )
    assert outcome == "uncertain"

    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, d_id)
        assert d.status == "uncertain"
        assert "Timeout" in d.last_error


@pytest.mark.asyncio
async def test_connection_reset_after_send_transitions_to_uncertain(sqlite_session_factory):
    """Requirement 3: ConnectionResetError during transport call transitions to 'uncertain'."""
    user_id = await _create_test_user(sqlite_session_factory, 1015)
    await _create_test_scenario(sqlite_session_factory, "reset_ev")
    await register_user_for_event(sqlite_session_factory, user_id, "reset_ev", platform="telegram")

    claimed = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    d_id, token = claimed[0]

    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock(side_effect=ConnectionResetError("Connection reset by peer"))

    outcome = await process_single_delivery(
        sqlite_session_factory, d_id, token, bot=mock_bot
    )
    assert outcome == "uncertain"

    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, d_id)
        assert d.status == "uncertain"
        assert "Ambiguous" in d.last_error


@pytest.mark.asyncio
async def test_generic_network_exception_after_send_transitions_to_uncertain(sqlite_session_factory):
    """Requirement 3: Generic network exception during transport call transitions to 'uncertain'."""
    user_id = await _create_test_user(sqlite_session_factory, 1016)
    await _create_test_scenario(sqlite_session_factory, "generic_net_ev")
    await register_user_for_event(sqlite_session_factory, user_id, "generic_net_ev", platform="telegram")

    claimed = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    d_id, token = claimed[0]

    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock(side_effect=RuntimeError("ClientConnectorError: Cannot connect to host"))

    outcome = await process_single_delivery(
        sqlite_session_factory, d_id, token, bot=mock_bot
    )
    assert outcome == "uncertain"

    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, d_id)
        assert d.status == "uncertain"


@pytest.mark.asyncio
async def test_provable_pre_transmission_error_is_retryable(sqlite_session_factory):
    """Requirement 3: Provable PreTransmissionError transitions safely to retryable 'pending'."""
    user_id = await _create_test_user(sqlite_session_factory, 1017)
    await _create_test_scenario(sqlite_session_factory, "pre_tx_ev")
    await register_user_for_event(sqlite_session_factory, user_id, "pre_tx_ev", platform="telegram")

    claimed = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    d_id, token = claimed[0]

    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock(side_effect=PreTransmissionError("Local credential loading failed before network"))

    outcome = await process_single_delivery(
        sqlite_session_factory, d_id, token, bot=mock_bot
    )
    assert outcome == "pre_transmission_retry"

    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, d_id)
        assert d.status == "pending"
        assert d.next_attempt_at is not None
        assert d.send_started_at is None
        assert d.claim_token is None


@pytest.mark.asyncio
async def test_attempt_local_state_reset_after_rate_limit_retry(sqlite_session_factory):
    """Requirement 4: 429 clears send_started_at so subsequent claim crash is recoverable by sweeper."""
    user_id = await _create_test_user(sqlite_session_factory, 1018)
    await _create_test_scenario(sqlite_session_factory, "rl_reset_ev")
    await register_user_for_event(sqlite_session_factory, user_id, "rl_reset_ev", platform="telegram")

    # 1. First claim and 429 send
    claimed = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    d_id, token = claimed[0]

    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock(
        side_effect=TelegramRetryAfter(method=MagicMock(), message="Rate limit", retry_after=1)
    )
    outcome = await process_single_delivery(sqlite_session_factory, d_id, token, bot=mock_bot)
    assert outcome == "rate_limited"

    # Verify attempt-local state is cleared
    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, d_id)
        assert d.status == "pending"
        assert d.send_started_at is None
        assert d.claim_token is None
        assert d.lease_until is None

    # Clear rate limit cooldown and due time for re-claim
    reset_cooldowns()
    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, d_id)
        d.due_at = datetime.utcnow() - timedelta(seconds=5)
        d.next_attempt_at = datetime.utcnow() - timedelta(seconds=5)
        await session.commit()

    # 2. Re-claim delivery
    claimed2 = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    assert len(claimed2) == 1
    d_id2, token2 = claimed2[0]
    assert d_id2 == d_id

    # 3. Simulate worker crash before PoNR (send_started_at remains None, lease expires)
    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, d_id)
        assert d.status == "claimed"
        assert d.send_started_at is None
        d.lease_until = datetime.utcnow() - timedelta(seconds=10)
        await session.commit()

    # 4. Sweeper runs: must recover delivery to pending!
    rec_c, rec_s = await recover_expired_leases(sqlite_session_factory)
    assert rec_c == 1

    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, d_id)
        assert d.status == "pending"
        assert d.claim_token is None



@pytest.mark.asyncio
async def test_uncertain_delivery_force_retry(sqlite_session_factory):
    """Requirement C: Manual force_retry on uncertain delivery creates new child delivery with incremented retry_index."""
    user_id = await _create_test_user(sqlite_session_factory, 1016)
    scen_id = await _create_test_scenario(sqlite_session_factory, "retry_audit_ev")
    reg_res = await register_user_for_event(sqlite_session_factory, user_id, "retry_audit_ev", platform="telegram")

    d_id = reg_res.immediate_delivery.id

    # Put delivery into uncertain
    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, d_id)
        d.status = "uncertain"
        await session.commit()

        child, created = await create_event_delivery_force_retry(
            session, delivery_id=d_id, admin_id=777, resolution_reason="Admin confirmed message not received"
        )
        assert created is True
        assert child.retry_index == 1
        assert child.parent_delivery_id == d_id
        assert child.status == "pending"
        await session.commit()


# ==============================================================================
# Full Cycle Test
# ==============================================================================

@pytest.mark.asyncio
async def test_run_event_delivery_worker_cycle(sqlite_session_factory):
    """Runs a full worker cycle including sweeper, claim, and processing."""
    user_id = await _create_test_user(sqlite_session_factory, 1017)
    scen_id = await _create_test_scenario(sqlite_session_factory, "cycle_ev")
    await register_user_for_event(sqlite_session_factory, user_id, "cycle_ev", platform="telegram")

    mock_bot = MagicMock()
    mock_msg = MagicMock()
    mock_msg.message_id = 55555
    mock_bot.send_message = AsyncMock(return_value=mock_msg)

    stats = await run_event_delivery_worker_cycle(
        sqlite_session_factory, bot=mock_bot, batch_size=10
    )
    assert stats["claimed"] == 1
    assert stats["sent"] == 1
    assert stats["recovered_claimed"] == 0


# ==============================================================================
# Real PostgreSQL Concurrency & Lock Ordering Tests
# ==============================================================================

async def _get_safe_postgres_engine():
    from tests.test_event_chains_db import _get_safe_postgres_engine as _db_pg_engine
    return await _db_pg_engine()


@pytest.mark.asyncio
async def test_real_postgresql_pre_send_lock_hierarchy_and_cas():
    """
    Real PostgreSQL Integration Test:
    Executes prepare_and_lock_delivery_for_sending with real PostgreSQL locks:
    Scenario (Level 3 FOR SHARE) -> Registration (Level 4 FOR SHARE) ->
    Step (Level 5 FOR SHARE) -> Delivery (Level 6 FOR UPDATE).
    Verifies that real PostgreSQL locks and atomic CAS update succeed without deadlocks.
    """
    pg_engine, skip_reason = await _get_safe_postgres_engine()
    if pg_engine is None:
        pytest.skip(skip_reason)

    try:
        async with pg_engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA public CASCADE;"))
            await conn.execute(text("CREATE SCHEMA public;"))

        sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(database, "engine", pg_engine)
            mp.setattr(database, "async_session_maker", sessions)
            await database.init_db()

        # Create user and scenario on real PostgreSQL
        user_id = await _create_test_user(sessions, 3001)
        scen_id = await _create_test_scenario(sessions, "pg_worker_ev")

        # Register user on real PostgreSQL
        reg_res = await register_user_for_event(sessions, user_id, "pg_worker_ev", platform="telegram")
        d_id = reg_res.immediate_delivery.id

        # Claim delivery on real PostgreSQL
        claimed = await claim_pending_deliveries(sessions, batch_size=10)
        assert len(claimed) == 1
        claimed_id, token = claimed[0]
        assert claimed_id == d_id

        # Execute prepare_and_lock_delivery_for_sending with real PostgreSQL FOR SHARE / FOR UPDATE
        ok, locked_deliv, err = await prepare_and_lock_delivery_for_sending(
            sessions,
            delivery_id=d_id,
            claim_token=token,
            expected_content_version=1,
            expected_step_version=None,
            rendered_text="PG rendered text",
            template_text="PG template text",
        )
        assert ok is True
        assert err is None

        async with sessions() as session:
            deliv_check = await session.get(EventDelivery, d_id)
            assert deliv_check.status == "sending"
            assert deliv_check.rendered_text_snapshot == "PG rendered text"
            assert deliv_check.attempt_count == 1
            assert deliv_check.lease_until is not None

        # Finalize on PostgreSQL
        finalized = await finalize_delivery_outcome(
            sessions,
            delivery_id=d_id,
            claim_token=token,
            status="sent",
            external_message_id="pg_msg_100",
        )
        assert finalized is True

        async with sessions() as session:
            final_check = await session.get(EventDelivery, d_id)
            assert final_check.status == "sent"
            assert final_check.external_message_id == "pg_msg_100"

    finally:
        await pg_engine.dispose()


@pytest.mark.asyncio
async def test_scheduler_registration_and_scheduled_delivery_runtime_path(sqlite_session_factory):
    """Item 1: Worker is registered in AsyncIOScheduler with max_instances=1, coalesce=True,
    and processes scheduled delivery through the production runtime job path.
    Also verifies:
    - Telegram delivery succeeds even when MAX client is None.
    - Shutdown properly stops scheduler and exits MAX client.
    """
    from main import setup_scheduler, on_shutdown
    from aiogram import Dispatcher

    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=12345))
    mock_bot.session = MagicMock()
    mock_bot.session.close = AsyncMock()

    mock_max = MagicMock()
    mock_max.__aenter__ = AsyncMock(return_value=mock_max)
    mock_max.__aexit__ = AsyncMock()

    dispatcher = Dispatcher()
    dispatcher["_followup_max_client"] = mock_max

    scheduler = setup_scheduler(
        bot=mock_bot,
        session_maker=sqlite_session_factory,
        dispatcher=dispatcher,
        max_followup_client=mock_max,
        start=False,
    )

    job = scheduler.get_job("event_delivery_worker")
    assert job is not None
    assert job.max_instances == 1
    assert job.coalesce is True
    assert job.name == "event_delivery_worker"
    assert job.kwargs.get("batch_size") == 20
    assert job.kwargs.get("bot") is mock_bot
    assert job.kwargs.get("max_client") is mock_max

    # 1. Create a user and scenario with a scheduled step that is due
    user_id = await _create_test_user(sqlite_session_factory, 5001)
    now = datetime.utcnow()
    starts_at = now + timedelta(minutes=10)
    scen_id = await _create_test_scenario(
        sqlite_session_factory,
        code="sched_runtime_ev",
        starts_at=starts_at,
        steps=[(-15, "Reminder!")],
    )
    reg_res = await register_user_for_event(
        sqlite_session_factory, user_id, "sched_runtime_ev", platform="telegram", now=now
    )

    # 2. Trigger job execution directly through job runtime function
    job_coro = job.func(*job.args, **job.kwargs)
    stats = await job_coro
    assert stats["sent"] >= 1

    # 3. Test that missing MAX client (max_followup_client=None) does not break Telegram deliveries
    user_id_tg2 = await _create_test_user(sqlite_session_factory, 5002)
    reg_res2 = await register_user_for_event(
        sqlite_session_factory, user_id_tg2, "sched_runtime_ev", platform="telegram", now=now
    )
    stats_tg_only = await job.func(
        sqlite_session_factory, bot=mock_bot, max_client=None, batch_size=20
    )
    assert stats_tg_only["sent"] >= 1

    # 4. Verify shutdown closes scheduler and MAX client
    await on_shutdown(mock_bot, dispatcher)
    assert scheduler.running is False
    mock_max.__aexit__.assert_awaited_once()
    mock_bot.session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_max_transport_error_status_none_transitions_to_uncertain(sqlite_session_factory):
    """Item 3: MaxApiError with status=None (ClientError / transport error) transitions to 'uncertain'."""
    user_id = await _create_test_user(sqlite_session_factory, MAX_ID_OFFSET + 801)
    await _create_test_scenario(sqlite_session_factory, "max_transport_ev")
    await register_user_for_event(
        sqlite_session_factory, user_id, "max_transport_ev", platform="max"
    )

    claimed = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    assert len(claimed) == 1
    d_id, token = claimed[0]

    mock_max = MagicMock()
    mock_max.send_message = AsyncMock(
        side_effect=MaxApiError(
            status=None,
            message="aiohttp transport error: ClientOSError: [Errno 54] Connection reset by peer",
        )
    )

    outcome = await process_single_delivery(
        sqlite_session_factory, d_id, token, max_client=mock_max
    )
    assert outcome == "uncertain"

    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, d_id)
        assert d.status == "uncertain"
        assert d.finished_at is not None
        assert d.claim_token is None
        assert "transport error" in d.last_error.lower()


@pytest.mark.asyncio
async def test_dynamic_now_clock_rejects_expired_delivery_during_cycle(sqlite_session_factory):
    """Item 4: In production runtime (now=None), time is evaluated dynamically right before
    pre-send checks, rejecting delivery if expires_at or lease passed rather than using worker start time."""
    user_id = await _create_test_user(sqlite_session_factory, 5010)
    await _create_test_scenario(sqlite_session_factory, "dyn_clock_ev")
    reg_res = await register_user_for_event(sqlite_session_factory, user_id, "dyn_clock_ev")
    d_id = reg_res.immediate_delivery.id

    # Claim delivery
    claimed = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    assert len(claimed) == 1
    d_id_claimed, token = claimed[0]

    # Set expires_at to slightly in future (60ms)
    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, d_id)
        d.expires_at = datetime.utcnow() + timedelta(milliseconds=60)
        await session.commit()

    # Advance real time past expires_at
    await asyncio.sleep(0.1)

    # Calling prepare_and_lock_delivery_for_sending with now=None must evaluate dynamic clock
    ok, deliv, reason = await prepare_and_lock_delivery_for_sending(
        sqlite_session_factory,
        delivery_id=d_id,
        claim_token=token,
        expected_content_version=1,
        expected_step_version=None,
        rendered_text="text",
        template_text="text",
        now=None,
    )
    assert ok is False
    assert reason == "expired"

    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, d_id)
        assert d.status == "skipped"
        assert d.skip_reason == "expired"
        assert d.finished_at is not None


@pytest.mark.asyncio
async def test_pending_delivery_expired_before_claim_swept_to_skipped(sqlite_session_factory):
    """Item 5: Pending delivery that expired before claim is swept to skipped with skip_reason='expired'.
    Reschedule does not resurrect expired scheduled steps."""
    user_id = await _create_test_user(sqlite_session_factory, 5020)
    now = datetime.utcnow()
    scen_id = await _create_test_scenario(
        sqlite_session_factory,
        code="sweep_exp_ev",
        starts_at=now + timedelta(days=2),
        steps=[(-60, "Step 1")],
    )
    reg_res = await register_user_for_event(sqlite_session_factory, user_id, "sweep_exp_ev", now=now)
    imm_id = reg_res.immediate_delivery.id

    # Artificially set immediate delivery expires_at in the past (e.g. expired while worker was down)
    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, imm_id)
        d.expires_at = now - timedelta(minutes=10)
        await session.commit()

    # Run worker cycle
    stats = await run_event_delivery_worker_cycle(sqlite_session_factory)
    assert stats["expired_swept"] >= 1

    async with sqlite_session_factory() as session:
        d = await session.get(EventDelivery, imm_id)
        assert d.status == "skipped"
        assert d.skip_reason == "expired"
        assert d.finished_at is not None
        assert d.claim_token is None

    # Now verify that reschedule does NOT resurrect expired scheduled step deliveries
    async with sqlite_session_factory() as session:
        sched_d = await session.scalar(
            select(EventDelivery).where(
                EventDelivery.registration_id == reg_res.registration.id,
                EventDelivery.message_kind == "scheduled_step",
            )
        )
        assert sched_d is not None
        sched_id = sched_d.id
        sched_d.status = "skipped"
        sched_d.skip_reason = "expired"
        sched_d.finished_at = now
        await session.commit()

    # Reschedule scenario
    await reschedule_event(
        sqlite_session_factory, scen_id, now + timedelta(days=5), now=now
    )

    # Verify that scheduled step delivery was NOT resurrected (only immediate_confirmation is resurrected)
    async with sqlite_session_factory() as session:
        sched_check = await session.get(EventDelivery, sched_id)
        assert sched_check.status == "skipped"
        assert sched_check.skip_reason == "expired"


@pytest.mark.asyncio
async def test_cancel_vs_worker_ponr_concurrency_sqlite(sqlite_session_factory):
    """Requirement 2 & 6: Cancel vs Worker Point of No Return real barrier concurrency on SQLite.
    Tests both commit orders with concurrent tasks:
    Order 1: Worker commits PoNR before Cancel commits -> Delivery is sent, Registration is cancelled.
    Order 2: Cancel commits before Worker PoNR -> Delivery is skipped, Registration is cancelled.
    """
    user_id = await _create_test_user(sqlite_session_factory, 4001)
    scen_id = await _create_test_scenario(sqlite_session_factory, "ponr_sq_ev")

    # --- Scenario A: Worker won PoNR race ---
    reg_res = await register_user_for_event(sqlite_session_factory, user_id, "ponr_sq_ev")
    d_id = reg_res.immediate_delivery.id

    claimed = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    d_id_claimed, token = claimed[0]
    assert d_id_claimed == d_id

    worker_ready = asyncio.Event()
    cancel_started = asyncio.Event()

    async def _worker_task_a():
        # Pre-send commits PoNR (status -> sending)
        ok, _, _ = await prepare_and_lock_delivery_for_sending(
            sqlite_session_factory,
            delivery_id=d_id,
            claim_token=token,
            expected_content_version=1,
            expected_step_version=None,
            rendered_text="text",
            template_text="text",
        )
        assert ok is True
        worker_ready.set()
        await cancel_started.wait()
        fin = await finalize_delivery_outcome(
            sqlite_session_factory, d_id, token, status="sent", external_message_id="msg_sq_1"
        )
        assert fin is True

    async def _cancel_task_a():
        await worker_ready.wait()
        cancel_started.set()
        cancel_res = await cancel_event_registration(sqlite_session_factory, user_id, scen_id)
        assert cancel_res.cancelled is True

    await asyncio.gather(_worker_task_a(), _cancel_task_a())

    async with sqlite_session_factory() as session:
        reg = await session.get(EventRegistration, reg_res.registration.id)
        d = await session.get(EventDelivery, d_id)
        assert reg.status == "cancelled"
        assert d.status == "sent"

    # --- Scenario B: Cancel won before PoNR ---
    user_id2 = await _create_test_user(sqlite_session_factory, 4002)
    reg_res2 = await register_user_for_event(sqlite_session_factory, user_id2, "ponr_sq_ev")
    d_id2 = reg_res2.immediate_delivery.id

    claimed2 = await claim_pending_deliveries(sqlite_session_factory, batch_size=10)
    d_id2_claimed, token2 = claimed2[0]
    assert d_id2_claimed == d_id2

    cancel_done = asyncio.Event()

    async def _cancel_task_b():
        cancel_res = await cancel_event_registration(sqlite_session_factory, user_id2, scen_id)
        assert cancel_res.cancelled is True
        cancel_done.set()

    async def _worker_task_b():
        await cancel_done.wait()
        ok2, _, reason2 = await prepare_and_lock_delivery_for_sending(
            sqlite_session_factory,
            delivery_id=d_id2,
            claim_token=token2,
            expected_content_version=1,
            expected_step_version=None,
            rendered_text="text",
            template_text="text",
        )
        assert ok2 is False
        assert "cancelled" in str(reason2) or "mismatch" in str(reason2) or "stale_or_invalid_claim" in str(reason2)

    await asyncio.gather(_cancel_task_b(), _worker_task_b())


@pytest.mark.asyncio
async def test_cancel_vs_worker_ponr_concurrency_postgresql():
    """Requirement 2 & 6: Cancel vs Worker Point of No Return barrier concurrency on real PostgreSQL."""
    pg_engine, skip_reason = await _get_safe_postgres_engine()
    if pg_engine is None:
        pytest.skip(skip_reason)

    try:
        async with pg_engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA public CASCADE;"))
            await conn.execute(text("CREATE SCHEMA public;"))

        sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(database, "engine", pg_engine)
            mp.setattr(database, "async_session_maker", sessions)
            await database.init_db()

        user_id = await _create_test_user(sessions, 4011)
        scen_id = await _create_test_scenario(sessions, "ponr_pg_ev")

        # --- Worker won PoNR on PostgreSQL ---
        reg_res = await register_user_for_event(sessions, user_id, "ponr_pg_ev")
        d_id = reg_res.immediate_delivery.id

        claimed = await claim_pending_deliveries(sessions, batch_size=10)
        assert len(claimed) == 1
        d_id_claimed, token = claimed[0]

        worker_ready = asyncio.Event()
        cancel_started = asyncio.Event()

        async def _worker_task_pg():
            ok, _, _ = await prepare_and_lock_delivery_for_sending(
                sessions,
                delivery_id=d_id,
                claim_token=token,
                expected_content_version=1,
                expected_step_version=None,
                rendered_text="PG text",
                template_text="PG text",
            )
            assert ok is True
            worker_ready.set()
            await cancel_started.wait()
            fin = await finalize_delivery_outcome(
                sessions, d_id, token, status="sent", external_message_id="pg_ponr_msg"
            )
            assert fin is True

        async def _cancel_task_pg():
            await worker_ready.wait()
            cancel_started.set()
            cancel_res = await cancel_event_registration(sessions, user_id, scen_id)
            assert cancel_res.cancelled is True

        await asyncio.wait_for(asyncio.gather(_worker_task_pg(), _cancel_task_pg()), timeout=10.0)

        async with sessions() as session:
            reg = await session.get(EventRegistration, reg_res.registration.id)
            d = await session.get(EventDelivery, d_id)
            assert reg.status == "cancelled"
            assert d.status == "sent"

    finally:
        await pg_engine.dispose()


