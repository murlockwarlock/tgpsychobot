import os
from datetime import datetime, timedelta, timezone

os.environ.setdefault("BOT_TOKEN", "test_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

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
