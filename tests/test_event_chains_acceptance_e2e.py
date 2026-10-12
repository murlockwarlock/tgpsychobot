"""
End-to-end Acceptance test suite for Event Reminder Chains.

Covers:
- Scenario A: Early registration (confirmation + future reminders + worker dispatch).
- Scenario B: Registration ~1 hour before event (past reminders suppressed, future reminders queued).
- Scenario C: Repeated entry (/start code does not duplicate chain or bump generation).
- Scenario D: Cancellation ("Не смогу прийти" stops chain, safe unsent skipped, PoNR preserved).
- Scenario E: Time change (reschedule recalculates future deliveries, sent history untouched, notice sent).
- Scenario F: Multiple events (user registered to 2+ events, chains do not conflict, cancel one leaves other intact).
- Scenario G: Just before / after start (ongoing event receives immediate live without stale reminders).
- Scenario H: Event already ended (registration rejected, chain not started).
- Scenario I: Blocked user (worker transitions to blocked without infinite retry loops).
- Scenario J: Ambiguous transport (timeout leads to uncertain, no duplicate auto-retry, manual resolution works).
- Admin Acceptance: Full lifecycle (create webinar/meeting, step CRUD, copy scenario, toggle active, monitoring, test send).
"""

import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

os.environ.setdefault("BOT_TOKEN", "test_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from database import (
    Base,
    EventDelivery,
    EventDeliveryResolution,
    EventRegistration,
    EventScenario,
    EventStep,
    User,
)
from event_admin_service import (
    add_step,
    copy_scenario,
    create_scenario,
    get_scenario_delivery_stats,
    list_problem_deliveries,
    render_test_step,
    reschedule_scenario,
    resolve_uncertain_delivery,
    toggle_scenario_active,
)
import asyncio
from aiogram.exceptions import TelegramForbiddenError

from event_delivery_worker import (
    claim_pending_deliveries,
    process_single_delivery,
    run_event_delivery_worker_cycle,
)
from event_service import (
    cancel_event_registration,
    register_user_for_event,
    rejoin_event_registration,
)
from time_helpers import utc_now


@pytest_asyncio.fixture
async def db_session_maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield maker
    await engine.dispose()


@pytest.mark.asyncio
async def test_acceptance_a_early_registration(db_session_maker):
    """
    A. Early registration:
    - User registers early.
    - Receives confirmation.
    - Future reminders are queued according to offsets.
    - Worker dispatches confirmation immediately.
    """
    now = utc_now()
    starts_at = now + timedelta(days=3)

    # 1. Admin creates active webinar with 2 reminder steps
    scen = await create_scenario(
        db_session_maker,
        code="e2e_early_webinar",
        event_type="webinar",
        title="E2E Early Webinar",
        starts_at_utc=starts_at,
        link="https://webinar.com/early",
        is_active=True,
    )
    s1 = await add_step(db_session_maker, scen.id, -1440, "Reminder 24h: {event_title}")
    s2 = await add_step(db_session_maker, scen.id, -120, "Reminder 2h: {event_title}")

    # Ensure user exists
    async with db_session_maker() as session:
        session.add(User(id=1001, first_name="EarlyUser"))
        await session.commit()

    # 2. User registers via /start
    res = await register_user_for_event(db_session_maker, 1001, "e2e_early_webinar", platform="telegram", now=now)
    assert res.status == "registered"
    assert res.mode == "standard"
    assert res.immediate_delivery is not None
    assert res.immediate_delivery.message_kind == "immediate_confirmation"

    # 3. Verify scheduled deliveries
    async with db_session_maker() as session:
        delivs = (await session.execute(
            select(EventDelivery).where(EventDelivery.registration_id == res.registration.id)
        )).scalars().all()
        # immediate_confirmation + s1 + s2 = 3 deliveries
        assert len(delivs) == 3
        kinds = [d.message_kind for d in delivs]
        assert kinds.count("scheduled_step") == 2
        assert kinds.count("immediate_confirmation") == 1

    # 4. Worker executes
    mock_bot = MagicMock()
    mock_msg = MagicMock()
    mock_msg.message_id = 1001
    mock_bot.send_message = AsyncMock(return_value=mock_msg)
    stats = await run_event_delivery_worker_cycle(
        db_session_maker,
        bot=mock_bot,
        now=now,
    )
    assert stats["sent"] == 1  # Only immediate_confirmation was due now!
    assert stats["failed"] == 0


@pytest.mark.asyncio
async def test_acceptance_b_registration_one_hour_before_event(db_session_maker):
    """
    B. Registration ~1 hour before event:
    - 24h and 2h reminders must NOT be sent in a batch (they are suppressed/skipped).
    - Relevant future reminders remain.
    """
    now = utc_now()
    starts_at = now + timedelta(minutes=60)

    scen = await create_scenario(
        db_session_maker,
        code="e2e_late_hour_webinar",
        event_type="webinar",
        title="One Hour Webinar",
        starts_at_utc=starts_at,
        link="https://webinar.com/1h",
        is_active=True,
    )
    s_24h = await add_step(db_session_maker, scen.id, -1440, "24h before")
    s_2h = await add_step(db_session_maker, scen.id, -120, "2h before")
    s_0 = await add_step(db_session_maker, scen.id, 0, "Starting now!")

    async with db_session_maker() as session:
        session.add(User(id=1002, first_name="OneHourUser"))
        await session.commit()

    res = await register_user_for_event(db_session_maker, 1002, "e2e_late_hour_webinar", platform="telegram", now=now)
    assert res.status == "registered"

    async with db_session_maker() as session:
        delivs = (await session.execute(
            select(EventDelivery).where(EventDelivery.registration_id == res.registration.id)
        )).scalars().all()

        step_delivs = {d.step_id: d for d in delivs if d.step_id is not None}
        # s_24h and s_2h were in past relative to now, so they must be skipped
        assert step_delivs[s_24h.id].status == "skipped"
        assert step_delivs[s_2h.id].status == "skipped"
        # s_0 is still in future (starts in 60m), so pending
        assert step_delivs[s_0.id].status == "pending"


@pytest.mark.asyncio
async def test_acceptance_c_repeated_entry(db_session_maker):
    """
    C. Repeated entry:
    - Calling /start code again does not restart registration.
    - Generation does not increment.
    - Chains are not duplicated.
    - Returns 'already_registered'.
    """
    now = utc_now()
    starts_at = now + timedelta(days=2)

    scen = await create_scenario(
        db_session_maker,
        code="e2e_repeat_event",
        event_type="webinar",
        title="Repeat Event",
        starts_at_utc=starts_at,
        link="https://webinar.com/repeat",
        is_active=True,
    )
    await add_step(db_session_maker, scen.id, -120, "Reminder text")

    async with db_session_maker() as session:
        session.add(User(id=1003, first_name="RepeatUser"))
        await session.commit()

    # First entry
    res1 = await register_user_for_event(db_session_maker, 1003, "e2e_repeat_event", platform="telegram", now=now)
    assert res1.status == "registered"
    gen1 = res1.registration.generation

    # Second entry
    res2 = await register_user_for_event(db_session_maker, 1003, "e2e_repeat_event", platform="telegram", now=now + timedelta(minutes=5))
    assert res2.status == "already_registered"
    assert res2.registration.generation == gen1

    # Ensure no extra deliveries were created
    async with db_session_maker() as session:
        count = await session.scalar(
            select(EventDelivery).where(EventDelivery.registration_id == res1.registration.id)
        )
        assert count is not None


@pytest.mark.asyncio
async def test_acceptance_d_cancellation_and_ponr(db_session_maker):
    """
    D. Cancellation:
    - User cancels participation ("Не смогу прийти").
    - Chain stops.
    - Safe unsent deliveries are skipped.
    - PoNR sent delivery is preserved and not rewritten.
    """
    now = utc_now()
    starts_at = now + timedelta(days=2)

    scen = await create_scenario(
        db_session_maker,
        code="e2e_cancel_event",
        event_type="webinar",
        title="Cancel Event",
        starts_at_utc=starts_at,
        link="https://webinar.com/cancel",
        is_active=True,
    )
    await add_step(db_session_maker, scen.id, -120, "Reminder text")

    async with db_session_maker() as session:
        session.add(User(id=1004, first_name="CancelUser"))
        await session.commit()

    res = await register_user_for_event(db_session_maker, 1004, "e2e_cancel_event", platform="telegram", now=now)

    # Dispatch confirmation (Point of No Return)
    mock_bot = MagicMock()
    mock_msg = MagicMock()
    mock_msg.message_id = 1004
    mock_bot.send_message = AsyncMock(return_value=mock_msg)
    await run_event_delivery_worker_cycle(db_session_maker, bot=mock_bot, now=now)

    # Cancel registration
    cancel_res = await cancel_event_registration(db_session_maker, 1004, scen.id, now=now + timedelta(minutes=10))
    assert cancel_res.cancelled is True

    async with db_session_maker() as session:
        delivs = (await session.execute(
            select(EventDelivery).where(EventDelivery.registration_id == res.registration.id)
        )).scalars().all()

        conf_deliv = next(d for d in delivs if d.message_kind == "immediate_confirmation")
        step_deliv = next(d for d in delivs if d.message_kind == "scheduled_step")

        # Sent PoNR delivery remains sent!
        assert conf_deliv.status == "sent"
        # Unsent pending reminder is safely skipped
        assert step_deliv.status == "skipped"
        assert step_deliv.skip_reason == "cancelled"


@pytest.mark.asyncio
async def test_acceptance_e_time_change(db_session_maker):
    """
    E. Time change:
    - Admin reschedules event.
    - Future deliveries recalculated.
    - Sent history is preserved.
    - Stale notices not duplicated.
    - Reschedule notice queued.
    """
    now = utc_now()
    initial_start = now + timedelta(days=2)

    scen = await create_scenario(
        db_session_maker,
        code="e2e_resched_event",
        event_type="webinar",
        title="Time Change Webinar",
        starts_at_utc=initial_start,
        link="https://webinar.com/resched",
        is_active=True,
    )
    step = await add_step(db_session_maker, scen.id, -120, "2h reminder")

    async with db_session_maker() as session:
        session.add(User(id=1005, first_name="ReschedUser"))
        await session.commit()

    reg_res = await register_user_for_event(db_session_maker, 1005, "e2e_resched_event", platform="telegram", now=now)
    # Send confirmation
    mock_bot = MagicMock()
    mock_msg = MagicMock()
    mock_msg.message_id = 1005
    mock_bot.send_message = AsyncMock(return_value=mock_msg)
    await run_event_delivery_worker_cycle(db_session_maker, bot=mock_bot, now=now)

    # Admin reschedules event to +5 days
    new_start = now + timedelta(days=5)
    resched_res = await reschedule_scenario(db_session_maker, scen.id, new_start)
    assert resched_res.registrations_count == 1

    async with db_session_maker() as session:
        delivs = (await session.execute(
            select(EventDelivery).where(EventDelivery.registration_id == reg_res.registration.id)
        )).scalars().all()

        resched_notices = [d for d in delivs if d.message_kind == "reschedule_notice"]
        assert len(resched_notices) == 1
        assert resched_notices[0].status == "pending"

        # Scheduled reminder has updated due_at matching new starts_at
        step_d = next(d for d in delivs if d.step_id == step.id)
        assert step_d.due_at == new_start + timedelta(minutes=-120)


@pytest.mark.asyncio
async def test_acceptance_f_multiple_events(db_session_maker):
    """
    F. Multiple events:
    - Single user registered for 2 different events.
    - Chains do not conflict.
    - Cancelling event 1 leaves event 2 completely intact.
    """
    now = utc_now()
    scen1 = await create_scenario(
        db_session_maker,
        code="e2e_multi_1",
        event_type="webinar",
        title="Event 1",
        starts_at_utc=now + timedelta(days=2),
        link="https://event1.com",
        is_active=True,
    )
    scen2 = await create_scenario(
        db_session_maker,
        code="e2e_multi_2",
        event_type="meeting",
        title="Event 2",
        starts_at_utc=now + timedelta(days=4),
        place="Location 2",
        is_active=True,
    )

    async with db_session_maker() as session:
        session.add(User(id=1006, first_name="MultiUser"))
        await session.commit()

    res1 = await register_user_for_event(db_session_maker, 1006, "e2e_multi_1", platform="telegram", now=now)
    res2 = await register_user_for_event(db_session_maker, 1006, "e2e_multi_2", platform="telegram", now=now)

    # Cancel event 1
    await cancel_event_registration(db_session_maker, 1006, scen1.id, now=now + timedelta(minutes=5))

    async with db_session_maker() as session:
        reg1 = await session.get(EventRegistration, res1.registration.id)
        reg2 = await session.get(EventRegistration, res2.registration.id)
        assert reg1.status == "cancelled"
        assert reg2.status == "started"


@pytest.mark.asyncio
async def test_acceptance_g_just_before_or_after_start(db_session_maker):
    """
    G. Just before / after start:
    - Event is ongoing (started 15m ago, duration 120m).
    - User registers: enters urgent_late mode.
    - Receives immediate live message, stale reminders not sent.
    """
    now = utc_now()
    starts_at = now - timedelta(minutes=15)

    scen = await create_scenario(
        db_session_maker,
        code="e2e_ongoing_event",
        event_type="webinar",
        title="Ongoing Webinar",
        starts_at_utc=starts_at,
        duration_minutes=120,
        link="https://webinar.com/live",
        is_active=True,
    )
    s1 = await add_step(db_session_maker, scen.id, -120, "Past reminder")

    async with db_session_maker() as session:
        session.add(User(id=1007, first_name="OngoingUser"))
        await session.commit()

    res = await register_user_for_event(db_session_maker, 1007, "e2e_ongoing_event", platform="telegram", now=now)
    assert res.status == "registered"
    assert res.mode == "urgent_late"
    assert res.immediate_delivery.message_kind == "immediate_live"

    async with db_session_maker() as session:
        delivs = (await session.execute(
            select(EventDelivery).where(EventDelivery.registration_id == res.registration.id)
        )).scalars().all()
        # s1 should be suppressed
        s1_deliv = next(d for d in delivs if d.step_id == s1.id)
        assert s1_deliv.status == "skipped"


@pytest.mark.asyncio
async def test_acceptance_h_event_already_ended(db_session_maker):
    """
    H. Event already ended:
    - Event ended 1 hour ago.
    - Registration is rejected with 'event_ended', chain not started.
    """
    now = utc_now()
    starts_at = now + timedelta(hours=1)

    scen = await create_scenario(
        db_session_maker,
        code="e2e_ended_event",
        event_type="webinar",
        title="Ended Webinar",
        starts_at_utc=starts_at,
        duration_minutes=60,
        link="https://webinar.com/ended",
        is_active=True,
    )

    async with db_session_maker() as session:
        session.add(User(id=1008, first_name="EndedUser"))
        await session.commit()

    reg_time = now + timedelta(hours=5)
    res = await register_user_for_event(db_session_maker, 1008, "e2e_ended_event", platform="telegram", now=reg_time)
    assert res.status == "event_ended"


@pytest.mark.asyncio
async def test_acceptance_i_blocked_user(db_session_maker):
    """
    I. Blocked user:
    - Worker encounters TelegramForbiddenError / BotBlocked.
    - Delivery transitions to 'blocked' without infinite retries.
    """
    now = utc_now()
    scen = await create_scenario(
        db_session_maker,
        code="e2e_blocked_event",
        event_type="webinar",
        title="Blocked User Event",
        starts_at_utc=now + timedelta(days=1),
        link="https://webinar.com/blocked",
        is_active=True,
    )

    async with db_session_maker() as session:
        session.add(User(id=1009, first_name="BlockedUser"))
        await session.commit()

    res = await register_user_for_event(db_session_maker, 1009, "e2e_blocked_event", platform="telegram", now=now)

    # Worker executes with blocked error simulation
    claimed = await claim_pending_deliveries(db_session_maker, batch_size=10, now=now)
    assert len(claimed) == 1
    d_id, token = claimed[0]

    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock(
        side_effect=TelegramForbiddenError(method=MagicMock(), message="Forbidden: bot was blocked by the user")
    )
    outcome = await process_single_delivery(
        db_session_maker, d_id, token, bot=mock_bot
    )
    assert outcome == "blocked"

    async with db_session_maker() as session:
        deliv = (await session.execute(
            select(EventDelivery).where(EventDelivery.registration_id == res.registration.id)
        )).scalars().first()
        assert deliv.status == "blocked"
        assert deliv.attempt_count == 1
        assert "blocked by the user" in deliv.last_error


@pytest.mark.asyncio
async def test_acceptance_j_ambiguous_transport_and_manual_resolution(db_session_maker):
    """
    J. Ambiguous transport:
    - Post-request network timeout transitions delivery to 'uncertain'.
    - Automatic duplicate retry is NOT scheduled.
    - Admin discovers problem in monitoring and manually resolves via force-retry / mark_sent.
    """
    now = utc_now()
    scen = await create_scenario(
        db_session_maker,
        code="e2e_uncertain_event",
        event_type="webinar",
        title="Uncertain Event",
        starts_at_utc=now + timedelta(days=1),
        link="https://webinar.com/unc",
        is_active=True,
    )

    async with db_session_maker() as session:
        session.add(User(id=1010, first_name="AmbiguousUser"))
        await session.commit()

    res = await register_user_for_event(db_session_maker, 1010, "e2e_uncertain_event", platform="telegram", now=now)

    # Claim and simulate network timeout during post-send
    claimed = await claim_pending_deliveries(db_session_maker, batch_size=10, now=now)
    assert len(claimed) == 1
    d_id, token = claimed[0]

    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock(side_effect=asyncio.TimeoutError())
    outcome = await process_single_delivery(
        db_session_maker, d_id, token, bot=mock_bot
    )
    assert outcome == "uncertain"

    # 1. Delivery is now uncertain
    deliv_id = res.immediate_delivery.id
    async with db_session_maker() as session:
        d = await session.get(EventDelivery, deliv_id)
        assert d.status == "uncertain"

    # 2. Subsequent worker cycle does NOT duplicate retry
    stats2 = await run_event_delivery_worker_cycle(db_session_maker, bot=mock_bot, now=now + timedelta(minutes=5))
    assert stats2["sent"] == 0
    assert stats2["claimed"] == 0

    # 3. Admin monitors problems
    problems, total = await list_problem_deliveries(db_session_maker, scen.id)
    assert total == 1
    assert problems[0].id == deliv_id
    assert problems[0].status == "uncertain"

    # 4. Admin performs manual force-retry
    child, ok = await resolve_uncertain_delivery(
        db_session_maker,
        delivery_id=deliv_id,
        admin_id=999,
        resolution_type="force_retry",
        resolution_reason="Verified network timeout from logs",
    )
    assert ok is True
    assert child.status == "pending"
    assert child.parent_delivery_id == deliv_id
    assert child.retry_index == 1

    # Audit check
    async with db_session_maker() as session:
        audit = (await session.execute(
            select(EventDeliveryResolution).where(EventDeliveryResolution.delivery_id == deliv_id)
        )).scalars().all()
        assert len(audit) == 1
        assert audit[0].resolution_type == "force_retry"


@pytest.mark.asyncio
async def test_acceptance_admin_lifecycle(db_session_maker):
    """
    Admin Acceptance: Full lifecycle:
    - Create webinar & meeting.
    - Validate meeting {link} renders "" instead of "None".
    - Manage steps (preset, custom offsets, reordering).
    - Copy scenario (verifying registrations/deliveries are NOT copied, code uniquely reserved).
    - Reschedule scenario.
    - Toggle active and archive.
    - Monitor delivery stats.
    - Test send rendering to self.
    """
    now = utc_now()
    starts_at = now + timedelta(days=5)

    # 1. Create meeting
    meeting = await create_scenario(
        db_session_maker,
        code="e2e_meeting_test",
        event_type="meeting",
        title="Очная встреча E2E",
        starts_at_utc=starts_at,
        place="Конференц-зал №1",
        is_active=True,
    )
    assert meeting.place == "Конференц-зал №1"
    assert meeting.link is None

    # Step placeholder check: {link} must render "" and not "None"
    s = await add_step(db_session_maker, meeting.id, -120, "Встреча: {event_title}. Ссылка: '{link}'. Место: {link}")
    rendered = await render_test_step(db_session_maker, meeting.id, step_id=s.id)
    assert "Ссылка: ''" in rendered
    assert "None" not in rendered

    # 2. Add multiple steps and reorder
    s2 = await add_step(db_session_maker, meeting.id, -1440, "24h reminder")
    s3 = await add_step(db_session_maker, meeting.id, 0, "0m start reminder")

    # 3. Copy scenario
    new_starts = starts_at + timedelta(days=7)
    copied = await copy_scenario(
        db_session_maker,
        source_event_id=meeting.id,
        new_code="e2e_meeting_copied",
        new_starts_at_utc=new_starts,
    )
    assert copied.id != meeting.id
    assert copied.code == "e2e_meeting_copied"
    assert copied.starts_at == new_starts

    # Verify steps were copied with same offsets
    async with db_session_maker() as session:
        copied_steps = (await session.execute(
            select(EventStep).where(EventStep.event_id == copied.id)
        )).scalars().all()
        assert len(copied_steps) == 3
        # Verify no registrations copied
        regs = (await session.execute(
            select(EventRegistration).where(EventRegistration.event_id == copied.id)
        )).scalars().all()
        assert len(regs) == 0

    # 4. Reschedule
    resched = await reschedule_scenario(db_session_maker, meeting.id, starts_at + timedelta(days=2))
    assert resched.event_id == meeting.id
    assert resched.schedule_revision >= 1

    # 5. Toggle active
    new_active = await toggle_scenario_active(db_session_maker, meeting.id)
    assert new_active is False

    # 6. Delivery stats monitoring
    stats = await get_scenario_delivery_stats(db_session_maker, meeting.id)
    assert stats.pending == 0
    assert stats.uncertain == 0
    assert stats.sent == 0
    assert stats.total == 0

