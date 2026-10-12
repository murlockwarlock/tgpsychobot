import asyncio
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

os.environ.setdefault("BOT_TOKEN", "test_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from event_delivery_worker import run_event_delivery_worker_cycle

from database import (
    Base,
    DeepLinkNamespace,
    EventDelivery,
    EventDeliveryResolution,
    EventRegistration,
    EventScenario,
    EventStep,
    User,
)
from event_admin_service import (
    add_step,
    archive_scenario,
    archive_step,
    copy_scenario,
    create_scenario,
    format_msk_datetime,
    format_offset_minutes,
    get_delivery_item,
    get_scenario_delivery_stats,
    get_scenario_summary,
    list_problem_deliveries,
    list_scenarios,
    list_steps,
    parse_msk_datetime,
    parse_offset_minutes,
    render_test_step,
    reorder_steps,
    reschedule_scenario,
    resolve_uncertain_delivery,
    toggle_scenario_active,
    update_scenario_fields,
    update_step,
    validate_event_code,
    validate_scenario_input,
)
from time_helpers import MSK, utc_now


@pytest_asyncio.fixture
async def test_session_maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield maker
    await engine.dispose()


def test_parse_and_format_msk_datetime():
    # 25.10.2026 19:00 MSK -> 16:00 UTC
    dt_utc = parse_msk_datetime("25.10.2026 19:00")
    assert dt_utc.year == 2026
    assert dt_utc.month == 10
    assert dt_utc.day == 25
    assert dt_utc.hour == 16
    assert dt_utc.minute == 0

    # Format back to MSK
    formatted = format_msk_datetime(dt_utc)
    assert formatted == "25.10.2026 19:00 (МСК)"

    # Invalid string raises ValueError
    with pytest.raises(ValueError, match="Неверный формат"):
        parse_msk_datetime("invalid-date")


def test_parse_and_format_offset_minutes():
    assert parse_offset_minutes("0") == 0
    assert parse_offset_minutes("в момент начала") == 0
    assert parse_offset_minutes("-24h") == -1440
    assert parse_offset_minutes("-24 ч") == -1440
    assert parse_offset_minutes("-2h") == -120
    assert parse_offset_minutes("+2h") == 120
    assert parse_offset_minutes("2h") == 120
    assert parse_offset_minutes("-30m") == -30
    assert parse_offset_minutes("30") == 30
    assert parse_offset_minutes("-1d") == -1440

    assert format_offset_minutes(0) == "В момент начала"
    assert format_offset_minutes(-1440) == "За 1 день до начала"
    assert format_offset_minutes(-120) == "За 2 ч. до начала"
    assert format_offset_minutes(120) == "Через 2 ч. после начала"
    assert format_offset_minutes(-30) == "За 30 мин. до начала"


def test_validation_rules():
    # Code validation
    assert validate_event_code("webinar_2026") == "webinar_2026"
    with pytest.raises(ValueError, match="зарезервирован"):
        validate_event_code("admin")
    with pytest.raises(ValueError, match="системного префикса"):
        validate_event_code("ev_test")
    with pytest.raises(ValueError, match="от 3 до 64"):
        validate_event_code("a")

    # Scenario input validation
    future_utc = datetime.utcnow() + timedelta(days=2)
    # Webinar without link
    with pytest.raises(ValueError, match="обязательно укажите ссылку"):
        validate_scenario_input(
            code="test_webinar",
            event_type="webinar",
            title="Webinar",
            starts_at_utc=future_utc,
            duration_minutes=60,
            place=None,
            link=None,
        )

    # Webinar with place
    with pytest.raises(ValueError, match="должно быть пустым"):
        validate_scenario_input(
            code="test_webinar",
            event_type="webinar",
            title="Webinar",
            starts_at_utc=future_utc,
            duration_minutes=60,
            place="Room 101",
            link="https://zoom.us/j/123",
        )

    # Meeting without place
    with pytest.raises(ValueError, match="обязательно укажите место"):
        validate_scenario_input(
            code="test_meeting",
            event_type="meeting",
            title="Meeting",
            starts_at_utc=future_utc,
            duration_minutes=60,
            place=None,
            link=None,
        )


@pytest.mark.asyncio
async def test_scenario_crud_lifecycle(test_session_maker):
    future_utc = datetime.utcnow() + timedelta(days=5)

    # 1. Create webinar
    webinar = await create_scenario(
        test_session_maker,
        code="ai_webinar_2026",
        event_type="webinar",
        title="AI Masterclass",
        starts_at_utc=future_utc,
        duration_minutes=90,
        place=None,
        link="https://zoom.us/j/999",
        is_active=False,
    )
    assert webinar.id > 0
    assert webinar.code == "ai_webinar_2026"
    assert webinar.link == "https://zoom.us/j/999"
    assert webinar.place is None
    assert webinar.is_active is False

    # Check DeepLinkNamespace was reserved
    async with test_session_maker() as session:
        ns = await session.get(DeepLinkNamespace, "ai_webinar_2026")
        assert ns is not None
        assert ns.kind == "event"
        assert ns.reference_id == str(webinar.id)

    # 2. Toggle active
    active = await toggle_scenario_active(test_session_maker, webinar.id)
    assert active is True
    summary = await get_scenario_summary(test_session_maker, webinar.id)
    assert summary.is_active is True

    # 3. Update fields
    updated = await update_scenario_fields(
        test_session_maker,
        webinar.id,
        title="AI Masterclass Updated",
        duration_minutes=120,
    )
    assert updated.title == "AI Masterclass Updated"
    assert updated.duration_minutes == 120
    assert updated.content_version == 2

    # 4. List scenarios
    scenarios = await list_scenarios(test_session_maker)
    assert len(scenarios) == 1
    assert scenarios[0].id == webinar.id

    # 5. Archive scenario
    await archive_scenario(test_session_maker, webinar.id)
    archived_summary = await get_scenario_summary(test_session_maker, webinar.id)
    assert archived_summary.is_archived is True
    assert archived_summary.is_active is False

    # Activating archived scenario must fail
    with pytest.raises(ValueError, match="архивный"):
        await toggle_scenario_active(test_session_maker, webinar.id)


@pytest.mark.asyncio
async def test_step_crud_and_versioning(test_session_maker):
    future_utc = datetime.utcnow() + timedelta(days=3)
    scen = await create_scenario(
        test_session_maker,
        code="steps_test_event",
        event_type="meeting",
        title="Live Workshop",
        starts_at_utc=future_utc,
        duration_minutes=180,
        place="Main Hall, Office 5",
        link=None,
    )

    # 1. Add steps
    s1 = await add_step(test_session_maker, scen.id, -1440, "Reminder 24h: {event_title}")
    assert s1.offset_minutes == -1440
    assert s1.step_version == 1
    assert s1.sort_order == 0

    s2 = await add_step(test_session_maker, scen.id, -120, "Reminder 2h: {event_title}")
    assert s2.sort_order == 1

    steps = await list_steps(test_session_maker, scen.id)
    assert len(steps) == 2

    # 2. Update step
    updated_s1 = await update_step(
        test_session_maker,
        s1.id,
        offset_minutes=-720,
        message_text="Updated 12h: {event_title}",
    )
    assert updated_s1.offset_minutes == -720
    assert updated_s1.step_version == 2
    assert updated_s1.message_text == "Updated 12h: {event_title}"

    # 3. Archive step
    await archive_step(test_session_maker, s2.id)
    steps_active = await list_steps(test_session_maker, scen.id, include_archived=False)
    assert len(steps_active) == 1
    assert steps_active[0].id == s1.id

    # 4. Reorder
    await reorder_steps(test_session_maker, scen.id, [s1.id])


@pytest.mark.asyncio
async def test_copy_scenario_without_polluting_deliveries(test_session_maker):
    future_utc = datetime.utcnow() + timedelta(days=2)
    source = await create_scenario(
        test_session_maker,
        code="original_webinar",
        event_type="webinar",
        title="Original Webinar",
        starts_at_utc=future_utc,
        duration_minutes=60,
        link="https://meet.google.com/abc",
    )
    await add_step(test_session_maker, source.id, -1440, "Step 1 text")
    await add_step(test_session_maker, source.id, -120, "Step 2 text")

    # Add mock registration and delivery to source
    async with test_session_maker() as session:
        user = User(id=101, username="testuser", first_name="John")
        session.add(user)
        await session.flush()
        reg = EventRegistration(
            event_id=source.id,
            user_id=101,
            status="started",
            registration_mode="standard",
        )
        session.add(reg)
        await session.flush()
        deliv = EventDelivery(
            event_id=source.id,
            user_id=101,
            registration_id=reg.id,
            message_kind="immediate_confirmation",
            delivery_key="test_deliv_1",
            due_at=utc_now(),
            next_attempt_at=utc_now(),
            platform="telegram",
            status="sent",
        )
        session.add(deliv)
        await session.commit()

    # Copy scenario
    next_week = future_utc + timedelta(days=7)
    copied = await copy_scenario(
        test_session_maker,
        source_event_id=source.id,
        new_code="copied_webinar",
        new_starts_at_utc=next_week,
        new_title="Copied Webinar Title",
    )

    assert copied.id != source.id
    assert copied.code == "copied_webinar"
    assert copied.title == "Copied Webinar Title"
    assert copied.event_type == "webinar"
    assert copied.link == "https://meet.google.com/abc"
    assert copied.is_active is False
    assert copied.registrations_count == 0

    # Verify steps were copied
    copied_steps = await list_steps(test_session_maker, copied.id)
    assert len(copied_steps) == 2
    assert copied_steps[0].offset_minutes == -1440
    assert copied_steps[1].offset_minutes == -120

    # Verify NO registrations or deliveries were copied
    async with test_session_maker() as session:
        regs = (await session.execute(
            select(EventRegistration).where(EventRegistration.event_id == copied.id)
        )).scalars().all()
        assert len(regs) == 0

        delivs = (await session.execute(
            select(EventDelivery).where(EventDelivery.event_id == copied.id)
        )).scalars().all()
        assert len(delivs) == 0


@pytest.mark.asyncio
async def test_reschedule_scenario_flow(test_session_maker):
    now = utc_now()
    initial_utc = now + timedelta(days=2)
    scen = await create_scenario(
        test_session_maker,
        code="resched_event",
        event_type="webinar",
        title="Reschedulable Event",
        starts_at_utc=initial_utc,
        duration_minutes=60,
        link="https://zoom.us/j/777",
        is_active=True,
    )

    # Register a user
    async with test_session_maker() as session:
        user = User(id=202, username="alice", first_name="Alice")
        session.add(user)
        reg = EventRegistration(
            event_id=scen.id,
            user_id=202,
            status="started",
            registration_mode="standard",
        )
        session.add(reg)
        await session.commit()

    # Reschedule to 5 days ahead
    new_time = now + timedelta(days=5)
    result = await reschedule_scenario(test_session_maker, scen.id, new_time)
    assert result.event_id == scen.id
    assert result.registrations_count == 1
    assert result.schedule_revision == 2

    # Reschedule into past must fail
    past_time = now - timedelta(days=1)
    with pytest.raises(ValueError, match="завершилось"):
        await reschedule_scenario(test_session_maker, scen.id, past_time)


@pytest.mark.asyncio
async def test_delivery_stats_and_problem_reporting(test_session_maker):
    future_utc = datetime.utcnow() + timedelta(days=1)
    scen = await create_scenario(
        test_session_maker,
        code="stats_event",
        event_type="meeting",
        title="Stats Meeting",
        starts_at_utc=future_utc,
        place="Room 404",
    )

    async with test_session_maker() as session:
        user1 = User(id=301, first_name="Bob")
        user2 = User(id=302, first_name="Charlie")
        session.add_all([user1, user2])
        await session.flush()
        reg1 = EventRegistration(event_id=scen.id, user_id=301, status="started")
        reg2 = EventRegistration(event_id=scen.id, user_id=302, status="started")
        session.add_all([reg1, reg2])
        await session.flush()

        d1 = EventDelivery(
            event_id=scen.id,
            user_id=301,
            registration_id=reg1.id,
            message_kind="immediate_confirmation",
            delivery_key="k1",
            due_at=utc_now(),
            next_attempt_at=utc_now(),
            platform="telegram",
            status="sent",
        )
        d2 = EventDelivery(
            event_id=scen.id,
            user_id=302,
            registration_id=reg2.id,
            message_kind="immediate_confirmation",
            delivery_key="k2",
            due_at=utc_now(),
            next_attempt_at=utc_now(),
            platform="telegram",
            status="uncertain",
            last_error="Connection timed out",
            claim_token="secret_token_123",
        )
        session.add_all([d1, d2])
        await session.commit()

    stats = await get_scenario_delivery_stats(test_session_maker, scen.id)
    assert stats.sent == 1
    assert stats.uncertain == 1
    assert stats.total == 2
    assert stats.problematic_total == 1

    problems, total = await list_problem_deliveries(test_session_maker, scen.id)
    assert total == 1
    assert len(problems) == 1
    p = problems[0]
    assert p.status == "uncertain"
    assert p.user_id == 302
    assert "Charlie" in p.user_display
    assert p.last_error == "Connection timed out"
    # Ensure claim_token is NOT exposed on DeliveryItem
    assert not hasattr(p, "claim_token")


@pytest.mark.asyncio
async def test_manual_resolution_flow(test_session_maker):
    scen = await create_scenario(
        test_session_maker,
        code="resolv_event",
        event_type="webinar",
        title="Resolv Webinar",
        starts_at_utc=datetime.utcnow() + timedelta(days=1),
        link="https://webinar.com",
    )

    async with test_session_maker() as session:
        user = User(id=401, first_name="Dan")
        session.add(user)
        await session.flush()
        reg = EventRegistration(event_id=scen.id, user_id=401, status="started")
        session.add(reg)
        await session.flush()

        d_uncertain = EventDelivery(
            event_id=scen.id,
            user_id=401,
            registration_id=reg.id,
            message_kind="immediate_confirmation",
            delivery_key="k_unc",
            due_at=utc_now(),
            next_attempt_at=utc_now(),
            platform="telegram",
            status="uncertain",
            last_error="Read timeout",
        )
        d_failed = EventDelivery(
            event_id=scen.id,
            user_id=401,
            registration_id=reg.id,
            message_kind="immediate_confirmation",
            delivery_key="k_fail",
            due_at=utc_now(),
            next_attempt_at=utc_now(),
            platform="telegram",
            status="failed",
            last_error="Blocked by user",
        )
        session.add_all([d_uncertain, d_failed])
        await session.commit()
        d_unc_id = d_uncertain.id
        d_fail_id = d_failed.id

    # Resolving non-uncertain delivery must fail
    with pytest.raises(ValueError, match="ожидается статус 'uncertain'"):
        await resolve_uncertain_delivery(
            test_session_maker,
            delivery_id=d_fail_id,
            admin_id=999,
            resolution_type="mark_sent",
            resolution_reason="Manual verification",
        )

    # 1. Mark sent
    deliv, ok = await resolve_uncertain_delivery(
        test_session_maker,
        delivery_id=d_unc_id,
        admin_id=999,
        resolution_type="mark_sent",
        resolution_reason="Confirmed delivery in Telegram client logs",
    )
    assert ok is True
    assert deliv.status == "resolved_sent"

    # Audit resolution check
    async with test_session_maker() as session:
        res = (await session.execute(
            select(EventDeliveryResolution).where(EventDeliveryResolution.delivery_id == d_unc_id)
        )).scalars().all()
        assert len(res) == 1
        assert res[0].resolution_type == "mark_sent"
        assert res[0].admin_id == 999


@pytest.mark.asyncio
async def test_test_send_to_self_rendering(test_session_maker):
    scen = await create_scenario(
        test_session_maker,
        code="test_send_event",
        event_type="webinar",
        title="Test Send Webinar",
        starts_at_utc=datetime.utcnow() + timedelta(days=1),
        link="https://live.link",
    )
    step = await add_step(
        test_session_maker,
        scen.id,
        -120,
        "Привет, {name}! Напоминаем о «{event_title}». Ссылка: {link}",
    )

    admin_user = User(id=888, first_name="SuperAdmin")
    rendered = await render_test_step(test_session_maker, scen.id, step.id, admin_user)

    assert "Привет, SuperAdmin!" in rendered
    assert "Test Send Webinar" in rendered
    assert "https://live.link" in rendered

    # Test send must NOT create EventRegistration or EventDelivery
    async with test_session_maker() as session:
        regs = (await session.execute(select(EventRegistration))).scalars().all()
        delivs = (await session.execute(select(EventDelivery))).scalars().all()
        assert len(regs) == 0
        assert len(delivs) == 0


@pytest.mark.asyncio
async def test_update_step_offset_to_past_marks_pending_skipped(test_session_maker):
    """
    Defect 4 regression test:
    - active registration with pending reminder due in the future
    - admin edits step offset such that new due_at <= now
    - delivery becomes skipped with skip_reason='step_rescheduled_past'
    - worker cycle runs and never invokes transport for this delivery
    """
    base_now = datetime(2026, 10, 25, 12, 0, 0)
    starts_at = base_now + timedelta(hours=2)  # 14:00

    scen = await create_scenario(
        test_session_maker,
        code="offset_past_test",
        event_type="webinar",
        title="Offset Past Webinar",
        starts_at_utc=starts_at,
        link="https://live.link",
        now_utc=base_now,
    )
    # Step originally at -60 min (due at 13:00, which is > 12:00 base_now)
    step = await add_step(test_session_maker, scen.id, -60, "Напоминание за час", now_utc=base_now)

    async with test_session_maker() as session:
        u = User(id=701, first_name="OffsetUser")
        session.add(u)
        await session.flush()
        reg = EventRegistration(
            event_id=scen.id,
            user_id=701,
            status="started",
            registration_mode="standard",
            registered_at=base_now,
        )
        session.add(reg)
        await session.flush()
        deliv = EventDelivery(
            event_id=scen.id,
            user_id=701,
            registration_id=reg.id,
            step_id=step.id,
            message_kind="scheduled_step",
            delivery_key=f"ev:{scen.id}:reg:{reg.id}:step:{step.id}:r0",
            due_at=starts_at - timedelta(minutes=60),  # 13:00
            next_attempt_at=starts_at - timedelta(minutes=60),
            schedule_revision=1,
            registration_generation=1,
            platform="telegram",
            status="pending",
        )
        session.add(deliv)
        await session.commit()
        deliv_id = deliv.id

    # Admin edits step offset to -180 min (due at 11:00, which is < base_now 12:00)
    await update_step(test_session_maker, step.id, offset_minutes=-180, now_utc=base_now)

    async with test_session_maker() as session:
        updated_deliv = await session.get(EventDelivery, deliv_id)
        assert updated_deliv.status == "skipped"
        assert updated_deliv.skip_reason == "step_rescheduled_past"
        assert updated_deliv.finished_at == base_now

    # Run worker cycle to ensure no transport is triggered
    mock_bot = AsyncMock()
    mock_bot.send_message = AsyncMock()
    stats = await run_event_delivery_worker_cycle(test_session_maker, bot=mock_bot, now=base_now + timedelta(hours=3))
    assert stats["sent"] == 0
    assert not mock_bot.send_message.called


@pytest.mark.asyncio
async def test_concurrent_force_retry_idempotency(tmp_path):
    """
    Defect 5 concurrency test:
    - delivery in 'uncertain' status
    - two concurrent force_retry requests execute simultaneously across separate DB connections
    - exactly one retry child is created
    - exactly one force_retry audit resolution is recorded
    - one call returns created=True, the other created=False
    - no unique constraint violation / IntegrityError is raised to callers
    """
    db_file = tmp_path / "concurrent_retry.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_file}?timeout=30", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    try:
        scen = await create_scenario(
            session_maker,
            code="force_retry_race",
            event_type="webinar",
            title="Retry Race Webinar",
            starts_at_utc=datetime.utcnow() + timedelta(days=2),
            link="https://live.link",
        )

        async with session_maker() as session:
            u = User(id=702, first_name="RaceUser")
            session.add(u)
            await session.flush()
            reg = EventRegistration(event_id=scen.id, user_id=702, status="started")
            session.add(reg)
            await session.flush()
            deliv = EventDelivery(
                event_id=scen.id,
                user_id=702,
                registration_id=reg.id,
                message_kind="immediate_confirmation",
                delivery_key="race_deliv_key",
                due_at=utc_now(),
                next_attempt_at=utc_now(),
                platform="telegram",
                status="uncertain",
                last_error="Network timeout",
            )
            session.add(deliv)
            await session.commit()
            deliv_id = deliv.id

        task1 = resolve_uncertain_delivery(
            session_maker,
            deliv_id,
            admin_id=1,
            resolution_type="force_retry",
            resolution_reason="Manual retry call 1",
        )
        task2 = resolve_uncertain_delivery(
            session_maker,
            deliv_id,
            admin_id=2,
            resolution_type="force_retry",
            resolution_reason="Manual retry call 2",
        )

        (res1, created1), (res2, created2) = await asyncio.gather(task1, task2)

        assert (created1 is True and created2 is False) or (created1 is False and created2 is True)
        assert res1.id == res2.id

        async with session_maker() as session:
            children = (
                await session.execute(
                    select(EventDelivery).where(EventDelivery.parent_delivery_id == deliv_id)
                )
            ).scalars().all()
            assert len(children) == 1

            resolutions = (
                await session.execute(
                    select(EventDeliveryResolution).where(
                        EventDeliveryResolution.delivery_id == deliv_id,
                        EventDeliveryResolution.resolution_type == "force_retry",
                    )
                )
            ).scalars().all()
            assert len(resolutions) == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_deactivate_reactivate_lifecycle_skips_missed_reminders(test_session_maker):
    """
    Defect 6 acceptance test:
    - register -> disable scenario -> reminder due passes -> enable scenario -> worker
    - transport for missed reminder is NOT called
    - delivery becomes skipped with skip_reason='disabled_past'
    """
    base_now = datetime(2026, 10, 25, 10, 0, 0)
    starts_at = base_now + timedelta(hours=2)  # 12:00

    scen = await create_scenario(
        test_session_maker,
        code="deact_react_test",
        event_type="webinar",
        title="Deact React Webinar",
        starts_at_utc=starts_at,
        link="https://live.link",
        now_utc=base_now,
    )
    # Reminder due at 11:30 (-30 min)
    step = await add_step(test_session_maker, scen.id, -30, "Скоро начало!", now_utc=base_now)

    async with test_session_maker() as session:
        u = User(id=703, first_name="LifecycleUser")
        session.add(u)
        await session.flush()
        reg = EventRegistration(
            event_id=scen.id,
            user_id=703,
            status="started",
            registration_mode="standard",
            registered_at=base_now,
        )
        session.add(reg)
        await session.flush()
        deliv = EventDelivery(
            event_id=scen.id,
            user_id=703,
            registration_id=reg.id,
            step_id=step.id,
            message_kind="scheduled_step",
            delivery_key=f"ev:{scen.id}:reg:{reg.id}:step:{step.id}:r0",
            due_at=starts_at - timedelta(minutes=30),  # 11:30
            next_attempt_at=starts_at - timedelta(minutes=30),
            schedule_revision=1,
            registration_generation=1,
            platform="telegram",
            status="pending",
        )
        session.add(deliv)
        await session.commit()
        deliv_id = deliv.id

    # 1. Admin disables scenario at 10:30
    time_disable = base_now + timedelta(minutes=30)
    await toggle_scenario_active(test_session_maker, scen.id, now_utc=time_disable)

    # 2. Due passes while disabled: advance time to 11:45
    time_after_due = base_now + timedelta(minutes=105)

    # 3. Admin re-enables scenario at 11:45
    await toggle_scenario_active(test_session_maker, scen.id, now_utc=time_after_due)

    # 4. Check delivery status in DB
    async with test_session_maker() as session:
        deliv_check = await session.get(EventDelivery, deliv_id)
        assert deliv_check.status == "skipped"
        assert deliv_check.skip_reason == "disabled_past"

    # 5. Worker runs at 11:46 -> transport is NEVER called
    mock_bot = AsyncMock()
    mock_bot.send_message = AsyncMock()
    stats = await run_event_delivery_worker_cycle(test_session_maker, bot=mock_bot, now=time_after_due + timedelta(minutes=1))
    assert stats["sent"] == 0
    assert not mock_bot.send_message.called


@pytest.mark.asyncio
async def test_archive_scenario_lifecycle_terminates_pending_deliveries(test_session_maker):
    """
    Defect 7 lifecycle test:
    - scenario with pending future deliveries, plus past sent and failed history
    - admin archives scenario
    - pending deliveries become skipped with skip_reason='scenario_archived'
    - monitoring shows 0 pending deliveries
    - historical sent and failed deliveries are preserved
    - worker sends nothing
    """
    base_now = datetime(2026, 10, 25, 10, 0, 0)
    starts_at = base_now + timedelta(days=2)

    scen = await create_scenario(
        test_session_maker,
        code="archive_test",
        event_type="webinar",
        title="Archive Webinar",
        starts_at_utc=starts_at,
        link="https://live.link",
        now_utc=base_now,
    )
    step = await add_step(test_session_maker, scen.id, -120, "За 2 часа", now_utc=base_now)

    async with test_session_maker() as session:
        u = User(id=704, first_name="ArchiveUser")
        session.add(u)
        await session.flush()
        reg = EventRegistration(
            event_id=scen.id,
            user_id=704,
            status="started",
            registration_mode="standard",
            registered_at=base_now,
        )
        session.add(reg)
        await session.flush()

        # 1. Pending future reminder
        d_pending = EventDelivery(
            event_id=scen.id,
            user_id=704,
            registration_id=reg.id,
            step_id=step.id,
            message_kind="scheduled_step",
            delivery_key="arch:pending",
            due_at=starts_at - timedelta(hours=2),
            next_attempt_at=starts_at - timedelta(hours=2),
            schedule_revision=1,
            registration_generation=1,
            platform="telegram",
            status="pending",
        )
        # 2. Past sent immediate confirmation
        d_sent = EventDelivery(
            event_id=scen.id,
            user_id=704,
            registration_id=reg.id,
            message_kind="immediate_confirmation",
            delivery_key="arch:sent",
            due_at=base_now,
            next_attempt_at=base_now,
            platform="telegram",
            status="sent",
            sent_at=base_now,
            finished_at=base_now,
        )
        # 3. Past failed delivery
        d_failed = EventDelivery(
            event_id=scen.id,
            user_id=704,
            registration_id=reg.id,
            message_kind="immediate_live",
            delivery_key="arch:failed",
            due_at=base_now,
            next_attempt_at=base_now,
            platform="telegram",
            status="failed",
            last_error="Permanent block",
            finished_at=base_now,
        )
        session.add_all([d_pending, d_sent, d_failed])
        await session.commit()
        pending_id = d_pending.id
        sent_id = d_sent.id
        failed_id = d_failed.id

    # Admin archives scenario
    await archive_scenario(test_session_maker, scen.id, now_utc=base_now)

    # Verify monitoring shows 0 pending
    stats = await get_scenario_delivery_stats(test_session_maker, scen.id)
    assert stats.pending == 0
    assert stats.sent == 1
    assert stats.failed == 1
    assert stats.skipped >= 1

    # Verify delivery records
    async with test_session_maker() as session:
        d_p = await session.get(EventDelivery, pending_id)
        assert d_p.status == "skipped"
        assert d_p.skip_reason == "scenario_archived"

        d_s = await session.get(EventDelivery, sent_id)
        assert d_s.status == "sent"

        d_f = await session.get(EventDelivery, failed_id)
        assert d_f.status == "failed"

    # Worker sends nothing
    mock_bot = AsyncMock()
    mock_bot.send_message = AsyncMock()
    worker_stats = await run_event_delivery_worker_cycle(test_session_maker, bot=mock_bot, now=starts_at)
    assert worker_stats["sent"] == 0
    assert not mock_bot.send_message.called


@pytest.mark.asyncio
async def test_add_step_for_active_registrations(test_session_maker):
    """
    Defect 8 regression test:
    - when admin calls add_step(), scheduled deliveries are created for active registrations:
      - standard active registration with future due -> pending
      - standard active registration with past due -> skipped (late_registration)
      - urgent_late active registration -> skipped (late_mode_suppressed)
      - cancelled registration -> no delivery created
    """
    base_now = datetime(2026, 10, 25, 12, 0, 0)
    starts_at = base_now + timedelta(hours=3)  # 15:00

    scen = await create_scenario(
        test_session_maker,
        code="add_step_active_regs",
        event_type="webinar",
        title="Add Step Active Regs",
        starts_at_utc=starts_at,
        link="https://live.link",
        now_utc=base_now,
    )

    async with test_session_maker() as session:
        u1 = User(id=711, first_name="StandardUser")
        u2 = User(id=712, first_name="UrgentUser")
        u3 = User(id=713, first_name="CancelledUser")
        session.add_all([u1, u2, u3])
        await session.flush()

        reg_std = EventRegistration(
            event_id=scen.id,
            user_id=711,
            status="started",
            registration_mode="standard",
            registered_at=base_now,
        )
        reg_urg = EventRegistration(
            event_id=scen.id,
            user_id=712,
            status="started",
            registration_mode="urgent_late",
            registered_at=base_now,
        )
        reg_canc = EventRegistration(
            event_id=scen.id,
            user_id=713,
            status="cancelled",
            registration_mode="standard",
            registered_at=base_now,
        )
        session.add_all([reg_std, reg_urg, reg_canc])
        await session.commit()
        std_id = reg_std.id
        urg_id = reg_urg.id
        canc_id = reg_canc.id

    # 1. Admin adds future step (-60 min -> due at 14:00, future relative to base_now 12:00)
    future_step = await add_step(test_session_maker, scen.id, -60, "Напоминание за час", now_utc=base_now)

    async with test_session_maker() as session:
        d_std = (await session.execute(
            select(EventDelivery).where(EventDelivery.registration_id == std_id, EventDelivery.step_id == future_step.id)
        )).scalar_one()
        assert d_std.status == "pending"
        assert d_std.due_at == starts_at - timedelta(minutes=60)

        d_urg = (await session.execute(
            select(EventDelivery).where(EventDelivery.registration_id == urg_id, EventDelivery.step_id == future_step.id)
        )).scalar_one()
        assert d_urg.status == "skipped"
        assert d_urg.skip_reason == "late_mode_suppressed"

        d_canc = (await session.execute(
            select(EventDelivery).where(EventDelivery.registration_id == canc_id, EventDelivery.step_id == future_step.id)
        )).scalar_one_or_none()
        assert d_canc is None

    # 2. Admin adds past step (-240 min -> due at 11:00, past relative to base_now 12:00)
    past_step = await add_step(test_session_maker, scen.id, -240, "Напоминание за 4 часа", now_utc=base_now)

    async with test_session_maker() as session:
        d_std_past = (await session.execute(
            select(EventDelivery).where(EventDelivery.registration_id == std_id, EventDelivery.step_id == past_step.id)
        )).scalar_one()
        assert d_std_past.status == "skipped"
        assert d_std_past.skip_reason == "late_registration"


@pytest.mark.asyncio
async def test_duration_edit_updates_immediate_deliveries_expiry(test_session_maker):
    """
    Defect 9 regression test:
    - update_scenario_fields(duration_minutes=...) updates business expiry for unsent immediate deliveries:
      - duration decrease where new expiry <= now: delivery becomes skipped with skip_reason='expired'
      - duration increase: delivery's expires_at is pushed out to starts_at + new_duration
    """
    base_now = datetime(2026, 10, 25, 12, 0, 0)
    starts_at = datetime(2026, 10, 25, 11, 0, 0)  # Event started 1 hr ago

    scen = await create_scenario(
        test_session_maker,
        code="dur_edit_test",
        event_type="webinar",
        title="Duration Edit Webinar",
        starts_at_utc=starts_at,
        duration_minutes=120,  # Originally expires at 13:00
        link="https://live.link",
        now_utc=base_now,
    )

    async with test_session_maker() as session:
        u = User(id=721, first_name="DurUser")
        session.add(u)
        await session.flush()
        reg = EventRegistration(event_id=scen.id, user_id=721, status="started")
        session.add(reg)
        await session.flush()
        # Unsent immediate live delivery
        deliv = EventDelivery(
            event_id=scen.id,
            user_id=721,
            registration_id=reg.id,
            message_kind="immediate_live",
            delivery_key="dur:live",
            due_at=starts_at,
            next_attempt_at=base_now,
            expires_at=starts_at + timedelta(minutes=120),  # 13:00
            schedule_revision=1,
            registration_generation=1,
            platform="telegram",
            status="pending",
        )
        session.add(deliv)
        await session.commit()
        deliv_id = deliv.id

    # 1. Duration decrease: change duration to 45 mins -> new expiry is 11:45 (< base_now 12:00)
    await update_scenario_fields(test_session_maker, scen.id, duration_minutes=45, now_utc=base_now)

    async with test_session_maker() as session:
        d = await session.get(EventDelivery, deliv_id)
        assert d.status == "skipped"
        assert d.skip_reason == "expired"
        assert d.finished_at == base_now

    # 2. Duration increase test with a fresh pending delivery
    async with test_session_maker() as session:
        deliv2 = EventDelivery(
            event_id=scen.id,
            user_id=721,
            registration_id=reg.id,
            message_kind="immediate_live",
            delivery_key="dur:live2",
            due_at=base_now,
            next_attempt_at=base_now,
            expires_at=starts_at + timedelta(minutes=45),
            schedule_revision=1,
            registration_generation=1,
            platform="telegram",
            status="pending",
        )
        session.add(deliv2)
        await session.commit()
        deliv2_id = deliv2.id

    # Change duration to 180 mins -> new expiry is 11:00 + 180 min = 14:00 (> base_now 12:00)
    await update_scenario_fields(test_session_maker, scen.id, duration_minutes=180, now_utc=base_now)

    async with test_session_maker() as session:
        d2 = await session.get(EventDelivery, deliv2_id)
        assert d2.status == "pending"
        assert d2.expires_at == starts_at + timedelta(minutes=180)


@pytest.mark.asyncio
async def test_reorder_steps_validation_and_atomicity(test_session_maker):
    """
    Defect 10 test:
    - reorder_steps() validates that step IDs match exactly active steps of scenario
    - rejects duplicates or mismatched IDs
    - atomically updates sort_order
    """
    scen = await create_scenario(
        test_session_maker,
        code="reorder_test",
        event_type="webinar",
        title="Reorder Webinar",
        starts_at_utc=datetime.utcnow() + timedelta(days=2),
        link="https://live.link",
    )
    s1 = await add_step(test_session_maker, scen.id, -1440, "Шаг 1")
    s2 = await add_step(test_session_maker, scen.id, -120, "Шаг 2")
    s3 = await add_step(test_session_maker, scen.id, -15, "Шаг 3")

    # Reject duplicate IDs
    with pytest.raises(ValueError, match="[Дд]убликаты"):
        await reorder_steps(test_session_maker, scen.id, [s1.id, s1.id, s3.id])

    # Reject missing or foreign IDs
    with pytest.raises(ValueError, match="неполный|некорректный"):
        await reorder_steps(test_session_maker, scen.id, [s1.id, s2.id, 99999])

    # Valid reorder [s3, s1, s2]
    await reorder_steps(test_session_maker, scen.id, [s3.id, s1.id, s2.id])

    steps = await list_steps(test_session_maker, scen.id, include_archived=False)
    assert [s.id for s in steps] == [s3.id, s1.id, s2.id]
    assert [s.sort_order for s in steps] == [0, 1, 2]

