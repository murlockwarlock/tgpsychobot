import asyncio
import json
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

os.environ.setdefault("BOT_TOKEN", "test_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
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
)
from event_service import (
    cancel_event_registration,
    handle_event_callback_max,
    handle_event_callback_telegram,
    handle_event_start_max,
    handle_event_start_telegram,
    register_user_for_event,
    rejoin_event_registration,
    render_event_template,
    reschedule_event,
    run_user_transaction,
    sync_and_append_user_metadata,
)
from time_helpers import format_msk
from user_metadata import load_metadata_records


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
    code="webinar_101",
    starts_at=None,
    duration=120,
    is_active=True,
    link="https://example.com/webinar",
    steps=None,
):
    if starts_at is None:
        starts_at = datetime.utcnow() + timedelta(days=2)
    if steps is None:
        steps = [
            (-1440, "Завтра вебинар {event_title}!"),
            (-60, "Через час начинаем {event_title}!"),
        ]

    async with session_factory() as session:
        scen = EventScenario(
            code=code,
            event_type="webinar",
            title=f"Тестовый Вебинар {code}",
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
# Service Tests: Registration Variations
# ==============================================================================

@pytest.mark.asyncio
async def test_early_standard_registration(sqlite_session_factory):
    """User registers well in advance: gets standard mode, confirmation delivery, and future scheduled steps."""
    user_id = await _create_test_user(sqlite_session_factory, 1001)
    event_start = datetime.utcnow() + timedelta(days=2)
    scen_id = await _create_test_scenario(sqlite_session_factory, "early_ev", starts_at=event_start)

    res = await register_user_for_event(sqlite_session_factory, user_id, "early_ev", platform="telegram")

    assert res.status == "registered"
    assert res.mode == "standard"
    assert res.registration is not None
    assert res.registration.generation == 1
    assert res.registration.registration_mode == "standard"
    assert res.immediate_delivery is not None
    assert res.immediate_delivery.message_kind == "immediate_confirmation"
    assert res.immediate_delivery.status == "pending"

    async with sqlite_session_factory() as session:
        deliveries = (
            await session.execute(
                select(EventDelivery).where(EventDelivery.registration_id == res.registration.id)
            )
        ).scalars().all()
        # 1 immediate confirmation + 2 scheduled steps
        assert len(deliveries) == 3
        kinds = {d.message_kind for d in deliveries}
        assert "immediate_confirmation" in kinds
        assert "scheduled_step" in kinds
        for d in deliveries:
            if d.message_kind == "scheduled_step":
                assert d.status == "pending"
                assert d.skip_reason is None

        # Check metadata
        u = await session.get(User, user_id)
        records = load_metadata_records(u.metadata_json)
        assert len(records) == 1
        assert records[0]["data"]["status"] == "started"
        assert records[0]["data"]["code"] == "early_ev"


@pytest.mark.asyncio
async def test_registration_about_one_hour_before_event(sqlite_session_factory):
    """User registers ~1h before start: 24h step is skipped as late_registration, 1h step and confirmation are queued."""
    user_id = await _create_test_user(sqlite_session_factory, 1002)
    event_start = datetime.utcnow() + timedelta(minutes=65)
    scen_id = await _create_test_scenario(sqlite_session_factory, "hour_ev", starts_at=event_start)

    res = await register_user_for_event(sqlite_session_factory, user_id, "hour_ev", platform="telegram")

    assert res.status == "registered"
    assert res.mode == "standard"

    async with sqlite_session_factory() as session:
        deliveries = (
            await session.execute(
                select(EventDelivery)
                .where(EventDelivery.registration_id == res.registration.id)
                .order_by(EventDelivery.due_at.asc())
            )
        ).scalars().all()

        step_deliveries = [d for d in deliveries if d.message_kind == "scheduled_step"]
        # Step -1440m (24h before) was in the past: skipped
        assert step_deliveries[0].status == "skipped"
        assert step_deliveries[0].skip_reason == "late_registration"
        # Step -60m (1h before) is in the future (~5 mins from now): pending
        assert step_deliveries[1].status == "pending"
        assert step_deliveries[1].skip_reason is None


@pytest.mark.asyncio
async def test_late_registration_in_urgent_window(sqlite_session_factory):
    """User registers in urgent cutoff window (now >= starts_at - 15m): gets urgent_late mode and immediate_live."""
    user_id = await _create_test_user(sqlite_session_factory, 1003)
    event_start = datetime.utcnow() + timedelta(minutes=10)  # within 15 min cutoff
    scen_id = await _create_test_scenario(sqlite_session_factory, "urgent_ev", starts_at=event_start)

    res = await register_user_for_event(sqlite_session_factory, user_id, "urgent_ev", platform="telegram")

    assert res.status == "registered"
    assert res.mode == "urgent_late"
    assert res.registration.registration_mode == "urgent_late"
    assert res.immediate_delivery.message_kind == "immediate_live"
    assert res.immediate_delivery.status == "pending"

    async with sqlite_session_factory() as session:
        deliveries = (
            await session.execute(
                select(EventDelivery).where(EventDelivery.registration_id == res.registration.id)
            )
        ).scalars().all()
        # immediate_live + steps suppressed
        conf_deliveries = [d for d in deliveries if d.message_kind == "immediate_confirmation"]
        assert len(conf_deliveries) == 0  # No immediate_confirmation!

        for d in deliveries:
            if d.message_kind == "scheduled_step":
                assert d.status == "skipped"
                assert d.skip_reason == "late_mode_suppressed"


@pytest.mark.asyncio
async def test_registration_after_start_while_ongoing(sqlite_session_factory):
    """User registers after start while event is ongoing: gets urgent_late and immediate_live."""
    user_id = await _create_test_user(sqlite_session_factory, 1004)
    event_start = datetime.utcnow() - timedelta(minutes=30)  # started 30 mins ago
    scen_id = await _create_test_scenario(
        sqlite_session_factory, "ongoing_ev", starts_at=event_start, duration=120
    )

    res = await register_user_for_event(sqlite_session_factory, user_id, "ongoing_ev", platform="telegram")

    assert res.status == "registered"
    assert res.mode == "urgent_late"
    assert res.immediate_delivery.message_kind == "immediate_live"


@pytest.mark.asyncio
async def test_registration_after_event_ended_rejected(sqlite_session_factory):
    """User attempts to register after event has ended: returns event_ended, no registration created."""
    user_id = await _create_test_user(sqlite_session_factory, 1005)
    event_start = datetime.utcnow() - timedelta(hours=3)
    scen_id = await _create_test_scenario(
        sqlite_session_factory, "ended_ev", starts_at=event_start, duration=120
    )

    res = await register_user_for_event(sqlite_session_factory, user_id, "ended_ev", platform="telegram")

    assert res.status == "event_ended"
    assert res.registration is None

    async with sqlite_session_factory() as session:
        regs = (await session.execute(select(EventRegistration))).scalars().all()
        assert len(regs) == 0


@pytest.mark.asyncio
async def test_duplicate_start_handling(sqlite_session_factory):
    """Requirement K: Duplicate /start with active registration returns already_registered, no duplicate delivery."""
    user_id = await _create_test_user(sqlite_session_factory, 1006)
    event_start = datetime.utcnow() + timedelta(days=1)
    await _create_test_scenario(sqlite_session_factory, "dup_ev", starts_at=event_start)

    res1 = await register_user_for_event(sqlite_session_factory, user_id, "dup_ev", platform="telegram")
    assert res1.status == "registered"

    res2 = await register_user_for_event(sqlite_session_factory, user_id, "dup_ev", platform="telegram")
    assert res2.status == "already_registered"
    assert res2.registration.id == res1.registration.id
    assert res2.registration.generation == 1
    assert res2.immediate_delivery is None

    async with sqlite_session_factory() as session:
        regs = (await session.execute(select(EventRegistration))).scalars().all()
        assert len(regs) == 1
        deliveries = (await session.execute(select(EventDelivery))).scalars().all()
        assert len(deliveries) == 3


@pytest.mark.asyncio
async def test_cancel_and_repeated_cancel(sqlite_session_factory):
    """Requirement H: Cancel sets cancelled status, skips unsent deliveries, repeated cancel is idempotent."""
    user_id = await _create_test_user(sqlite_session_factory, 1007)
    event_start = datetime.utcnow() + timedelta(days=2)
    scen_id = await _create_test_scenario(sqlite_session_factory, "cancel_ev", starts_at=event_start)

    await register_user_for_event(sqlite_session_factory, user_id, "cancel_ev", platform="telegram")

    c_res1 = await cancel_event_registration(sqlite_session_factory, user_id, scen_id)
    assert c_res1.cancelled is True
    assert c_res1.already_cancelled is False

    async with sqlite_session_factory() as session:
        reg = await session.get(EventRegistration, c_res1.registration_id)
        assert reg.status == "cancelled"
        assert reg.cancelled_at is not None

        deliveries = (await session.execute(select(EventDelivery))).scalars().all()
        for d in deliveries:
            assert d.status == "skipped"
            assert d.skip_reason == "cancelled"

        # Check metadata
        u = await session.get(User, user_id)
        records = load_metadata_records(u.metadata_json)
        assert len(records) == 2
        assert records[1]["data"]["status"] == "cancelled"

    # Repeated cancel
    c_res2 = await cancel_event_registration(sqlite_session_factory, user_id, scen_id)
    assert c_res2.cancelled is False
    assert c_res2.already_cancelled is True


@pytest.mark.asyncio
async def test_rejoin_early(sqlite_session_factory):
    """Requirement I: Rejoin while in early window increments generation and restores future scheduled steps."""
    user_id = await _create_test_user(sqlite_session_factory, 1008)
    event_start = datetime.utcnow() + timedelta(days=3)
    scen_id = await _create_test_scenario(sqlite_session_factory, "rejoin_ev", starts_at=event_start)

    await register_user_for_event(sqlite_session_factory, user_id, "rejoin_ev", platform="telegram")
    await cancel_event_registration(sqlite_session_factory, user_id, scen_id)

    rej_res = await rejoin_event_registration(sqlite_session_factory, user_id, scen_id, platform="telegram")

    assert rej_res.generation == 2
    assert rej_res.mode == "standard"
    assert rej_res.immediate_delivery.message_kind == "immediate_confirmation"

    async with sqlite_session_factory() as session:
        reg = await session.get(EventRegistration, rej_res.registration_id)
        assert reg.status == "started"
        assert reg.generation == 2
        assert reg.registration_mode == "standard"

        # Scheduled steps restored to pending with generation=2
        deliveries = (await session.execute(select(EventDelivery))).scalars().all()
        steps = [d for d in deliveries if d.message_kind == "scheduled_step"]
        for s in steps:
            assert s.status == "pending"
            assert s.registration_generation == 2


@pytest.mark.asyncio
async def test_rejoin_never_restores_old_immediate_deliveries(sqlite_session_factory):
    """Requirement B Regression: Old immediate_confirmation / immediate_live must NEVER be restored upon rejoin."""
    user_id = await _create_test_user(sqlite_session_factory, 1009)
    event_start = datetime.utcnow() + timedelta(days=2)
    scen_id = await _create_test_scenario(sqlite_session_factory, "no_old_imm", starts_at=event_start)

    reg_res = await register_user_for_event(sqlite_session_factory, user_id, "no_old_imm", platform="telegram")
    old_imm_id = reg_res.immediate_delivery.id

    await cancel_event_registration(sqlite_session_factory, user_id, scen_id)

    rej_res = await rejoin_event_registration(sqlite_session_factory, user_id, scen_id, platform="telegram")
    new_imm_id = rej_res.immediate_delivery.id

    assert old_imm_id != new_imm_id

    async with sqlite_session_factory() as session:
        old_imm = await session.get(EventDelivery, old_imm_id)
        # Old immediate delivery remains skipped with skip_reason='cancelled'
        assert old_imm.status == "skipped"
        assert old_imm.skip_reason == "cancelled"
        assert old_imm.registration_generation == 1

        new_imm = await session.get(EventDelivery, new_imm_id)
        assert new_imm.status == "pending"
        assert new_imm.registration_generation == 2

        # Verify there are NOT two pending confirmations!
        pending_confs = (
            await session.execute(
                select(EventDelivery).where(
                    EventDelivery.message_kind == "immediate_confirmation",
                    EventDelivery.status == "pending",
                )
            )
        ).scalars().all()
        assert len(pending_confs) == 1


@pytest.mark.asyncio
async def test_standard_to_cancel_to_urgent_rejoin_to_reschedule(sqlite_session_factory):
    """
    Requirement A: Complete flow test:
    standard -> cancel -> urgent rejoin -> reschedule.
    The new generation must track urgent_late, and subsequent reschedule must respect active urgent_late mode.
    """
    user_id = await _create_test_user(sqlite_session_factory, 1010)
    event_start = datetime.utcnow() + timedelta(days=2)
    scen_id = await _create_test_scenario(sqlite_session_factory, "a_flow_ev", starts_at=event_start)

    # 1. Register standard
    r1 = await register_user_for_event(sqlite_session_factory, user_id, "a_flow_ev", platform="telegram")
    assert r1.mode == "standard"
    assert r1.registration.registration_mode == "standard"

    # 2. Cancel
    c1 = await cancel_event_registration(sqlite_session_factory, user_id, scen_id)
    assert c1.cancelled is True

    # 3. Simulate urgent rejoin by using effective 'now' within 15 min cutoff
    urgent_now = event_start - timedelta(minutes=10)
    rej = await rejoin_event_registration(
        sqlite_session_factory, user_id, scen_id, platform="telegram", now=urgent_now
    )
    assert rej.generation == 2
    assert rej.mode == "urgent_late"

    async with sqlite_session_factory() as session:
        reg_check = await session.get(EventRegistration, rej.registration_id)
        assert reg_check.registration_mode == "urgent_late"
        assert reg_check.generation == 2

    # 4. Reschedule event into future
    new_start = event_start + timedelta(days=3)
    resch = await reschedule_event(sqlite_session_factory, scen_id, new_start, now=urgent_now)
    assert resch.schedule_revision == 2

    # 5. Verify deliveries: active generation was urgent_late!
    async with sqlite_session_factory() as session:
        reg_final = await session.get(EventRegistration, rej.registration_id)
        assert reg_final.registration_mode == "urgent_late"

        deliveries = (
            await session.execute(
                select(EventDelivery).where(EventDelivery.registration_id == reg_final.id)
            )
        ).scalars().all()

        # Steps must remain late_mode_suppressed (NOT revived into pending!)
        steps = [d for d in deliveries if d.message_kind == "scheduled_step"]
        for s in steps:
            assert s.status == "skipped"
            assert s.skip_reason == "late_mode_suppressed"

        # Exactly one reschedule_notice created
        notices = [d for d in deliveries if d.message_kind == "reschedule_notice"]
        assert len(notices) == 1
        assert notices[0].status == "pending"
        assert notices[0].schedule_revision == 2

        # A new immediate_live created for the new start
        lives = [d for d in deliveries if d.message_kind == "immediate_live" and d.schedule_revision == 2]
        assert len(lives) == 1
        assert lives[0].status == "pending"


@pytest.mark.asyncio
async def test_reschedule_invalidates_stale_queued_reschedule_notices(sqlite_session_factory):
    """Requirement G: Repeated reschedule invalidates previous unsent reschedule_notices without rewriting history."""
    user_id = await _create_test_user(sqlite_session_factory, 1011)
    event_start = datetime.utcnow() + timedelta(days=1)
    scen_id = await _create_test_scenario(sqlite_session_factory, "double_resch", starts_at=event_start)

    await register_user_for_event(sqlite_session_factory, user_id, "double_resch", platform="telegram")

    # Reschedule 1
    new_start_1 = event_start + timedelta(days=2)
    await reschedule_event(sqlite_session_factory, scen_id, new_start_1)

    # Reschedule 2 immediately after
    new_start_2 = event_start + timedelta(days=4)
    await reschedule_event(sqlite_session_factory, scen_id, new_start_2)

    async with sqlite_session_factory() as session:
        notices = (
            await session.execute(
                select(EventDelivery)
                .where(EventDelivery.message_kind == "reschedule_notice")
                .order_by(EventDelivery.id.asc())
            )
        ).scalars().all()

        assert len(notices) == 2
        # First notice (rev 2) was invalidated as rescheduled_past, retaining its rev 2
        assert notices[0].status == "skipped"
        assert notices[0].skip_reason == "rescheduled_past"
        assert notices[0].schedule_revision == 2

        # Second notice (rev 3) is active pending with rev 3
        assert notices[1].status == "pending"
        assert notices[1].schedule_revision == 3


@pytest.mark.asyncio
async def test_reschedule_preserves_sent_history(sqlite_session_factory):
    """Requirement G: Already sent deliveries are completely untouched by reschedule."""
    user_id = await _create_test_user(sqlite_session_factory, 1012)
    event_start = datetime.utcnow() + timedelta(days=2)
    scen_id = await _create_test_scenario(sqlite_session_factory, "sent_hist_ev", starts_at=event_start)

    reg_res = await register_user_for_event(sqlite_session_factory, user_id, "sent_hist_ev", platform="telegram")

    # Manually mark the confirmation and first step as sent
    async with sqlite_session_factory() as session:
        conf = await session.get(EventDelivery, reg_res.immediate_delivery.id)
        conf.status = "sent"
        conf.sent_at = datetime.utcnow()
        await session.commit()

    new_start = event_start + timedelta(days=3)
    await reschedule_event(sqlite_session_factory, scen_id, new_start)

    async with sqlite_session_factory() as session:
        conf_after = await session.get(EventDelivery, reg_res.immediate_delivery.id)
        assert conf_after.status == "sent"
        assert conf_after.sent_at is not None
        assert conf_after.schedule_revision == 1  # historical revision retained


@pytest.mark.asyncio
async def test_reschedule_into_past_raises_value_error(sqlite_session_factory):
    """Requirement G: Reschedule into an interval that already fully ended is rejected with ValueError."""
    user_id = await _create_test_user(sqlite_session_factory, 1013)
    event_start = datetime.utcnow() + timedelta(days=1)
    scen_id = await _create_test_scenario(sqlite_session_factory, "past_resch", starts_at=event_start)

    with pytest.raises(ValueError, match="Нельзя перенести мероприятие на время, которое уже полностью завершилось"):
        await reschedule_event(sqlite_session_factory, scen_id, datetime.utcnow() - timedelta(days=1))


@pytest.mark.asyncio
async def test_multiple_events_for_one_user(sqlite_session_factory):
    """Requirement: One user can be registered for multiple distinct event scenarios concurrently."""
    user_id = await _create_test_user(sqlite_session_factory, 1014)
    scen1 = await _create_test_scenario(sqlite_session_factory, "multi_ev_1")
    scen2 = await _create_test_scenario(sqlite_session_factory, "multi_ev_2")

    r1 = await register_user_for_event(sqlite_session_factory, user_id, "multi_ev_1", platform="telegram")
    r2 = await register_user_for_event(sqlite_session_factory, user_id, "multi_ev_2", platform="telegram")

    assert r1.status == "registered"
    assert r2.status == "registered"

    # Cancel one without affecting the other
    await cancel_event_registration(sqlite_session_factory, user_id, scen1)

    async with sqlite_session_factory() as session:
        reg1 = await session.get(EventRegistration, r1.registration.id)
        reg2 = await session.get(EventRegistration, r2.registration.id)
        assert reg1.status == "cancelled"
        assert reg2.status == "started"


# ==============================================================================
# Metadata Concurrency & Multiset Merge Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_sync_and_append_user_metadata_multiset_merge(sqlite_session_factory):
    """Requirement L (C33): Two consecutive sync_and_append_user_metadata calls in same transaction before commit."""
    user_id = await _create_test_user(sqlite_session_factory, 1015)

    async with sqlite_session_factory() as session:
        user = await session.get(User, user_id)
        # First append
        await sync_and_append_user_metadata(
            session, user, [{"data": {"action": "step_A"}}]
        )
        # Second append before commit
        await sync_and_append_user_metadata(
            session, user, [{"data": {"action": "step_B"}}]
        )
        await session.commit()

    async with sqlite_session_factory() as session:
        user_check = await session.get(User, user_id)
        records = load_metadata_records(user_check.metadata_json)
        assert len(records) == 2
        actions = [r["data"]["action"] for r in records]
        assert actions == ["step_A", "step_B"]


# ==============================================================================
# Template Formatting Tests
# ==============================================================================

def test_template_rendering_msk_timezone():
    """Requirement N: Template rendering with MSK timezone and user substitutions."""
    # 25 Oct 2026 at 16:00 UTC = 19:00 MSK
    start_utc = datetime(2026, 10, 25, 16, 0, 0)
    scen = EventScenario(
        id=1,
        code="tpl_test",
        event_type="webinar",
        title="Психологический интенсив",
        starts_at=start_utc,
        duration_minutes=120,
        link="https://example.com/live",
        place=None,
    )
    user = User(id=1, first_name="Алексей", name="Алексей")

    template = (
        "Привет, {name}!\n"
        "Мероприятие «{event_title}» состоится {event_date} в {event_time} (МСК).\n"
        "Ссылка: {link}"
    )

    rendered = render_event_template(template, scen, user)

    assert "Привет, Алексей!" in rendered
    assert "Мероприятие «Психологический интенсив»" in rendered
    assert "25.10.2026" in rendered
    assert "19:00" in rendered
    assert "https://example.com/live" in rendered


# ==============================================================================
# Telegram and MAX Flow Integration Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_telegram_start_and_callback_flow(sqlite_session_factory):
    """Test Telegram start flow and cancel/rejoin callbacks."""
    user_id = await _create_test_user(sqlite_session_factory, 2001)
    scen_id = await _create_test_scenario(sqlite_session_factory, "tg_ev")

    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()

    # 1. /start tg_ev
    handled = await handle_event_start_telegram(
        mock_bot, chat_id=2001, user_id=user_id, start_arg="tg_ev", session_maker=sqlite_session_factory
    )
    assert handled is True
    assert mock_bot.send_message.called
    sent_text = mock_bot.send_message.call_args.kwargs["text"]
    assert "Вы успешно зарегистрированы" in sent_text

    # 2. Callback ev_cancel:<scen_id>
    cb_cancel = MagicMock()
    cb_cancel.data = f"ev_cancel:{scen_id}"
    cb_cancel.from_user.id = user_id
    cb_cancel.answer = AsyncMock()
    cb_cancel.message = MagicMock()
    cb_cancel.message.edit_text = AsyncMock()

    res = await handle_event_callback_telegram(cb_cancel, session_maker=sqlite_session_factory)
    assert res is True
    assert cb_cancel.message.edit_text.called
    edit_text = cb_cancel.message.edit_text.call_args[0][0]
    assert "Регистрация отменена" in edit_text

    # 3. Callback ev_rejoin:<scen_id>
    cb_rejoin = MagicMock()
    cb_rejoin.data = f"ev_rejoin:{scen_id}"
    cb_rejoin.from_user.id = user_id
    cb_rejoin.answer = AsyncMock()
    cb_rejoin.message = MagicMock()
    cb_rejoin.message.edit_text = AsyncMock()

    res2 = await handle_event_callback_telegram(cb_rejoin, session_maker=sqlite_session_factory)
    assert res2 is True
    assert cb_rejoin.message.edit_text.called
    rejoin_text = cb_rejoin.message.edit_text.call_args[0][0]
    assert "снова зарегистрированы" in rejoin_text


@pytest.mark.asyncio
async def test_max_start_and_callback_flow(sqlite_session_factory):
    """Test MAX start flow and cancel/rejoin callbacks."""
    from max_messenger_bot.models import MAX_ID_OFFSET
    user_id = MAX_ID_OFFSET + 3001
    await _create_test_user(sqlite_session_factory, user_id)
    scen_id = await _create_test_scenario(sqlite_session_factory, "max_ev")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()
    mock_client.answer_callback = AsyncMock()

    # 1. Start payload
    handled = await handle_event_start_max(
        mock_client, chat_id=100, user_id=user_id, start_payload="max_ev", session_maker=sqlite_session_factory
    )
    assert handled is True
    assert mock_client.send_message.called
    assert "Вы успешно зарегистрированы" in mock_client.send_message.call_args.kwargs["text"]

    # 2. Callback ev_cancel:<scen_id>
    cb_cancel = MagicMock()
    cb_cancel.payload = f"ev_cancel:{scen_id}"
    cb_cancel.sender.user_id = user_id
    cb_cancel.chat_id = 100
    cb_cancel.callback_id = "cb_1"

    res = await handle_event_callback_max(mock_client, cb_cancel, session_maker=sqlite_session_factory)
    assert res is True
    assert mock_client.send_message.called
    assert "Регистрация отменена" in mock_client.send_message.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_e2e_start_then_worker_cycle_exactly_one_message_telegram(sqlite_session_factory):
    """Requirement 1: /start tg_ev -> worker cycle results in EXACTLY ONE message."""
    from event_delivery_worker import run_event_delivery_worker_cycle
    user_id = await _create_test_user(sqlite_session_factory, 2011)
    scen_id = await _create_test_scenario(sqlite_session_factory, "tg_single_ev")

    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()

    # 1. User starts Telegram flow
    handled = await handle_event_start_telegram(
        mock_bot, chat_id=2011, user_id=user_id, start_arg="tg_single_ev", session_maker=sqlite_session_factory
    )
    assert handled is True
    assert mock_bot.send_message.call_count == 1

    # 2. Worker cycle runs afterwards
    cycle_stats = await run_event_delivery_worker_cycle(sqlite_session_factory, bot=mock_bot)
    assert cycle_stats["claimed"] == 0
    # No duplicate message! Total calls remain 1!
    assert mock_bot.send_message.call_count == 1


@pytest.mark.asyncio
async def test_e2e_start_then_worker_cycle_exactly_one_message_max(sqlite_session_factory):
    """Requirement 1: /start max_ev -> worker cycle results in EXACTLY ONE message."""
    from event_delivery_worker import run_event_delivery_worker_cycle
    from max_messenger_bot.models import MAX_ID_OFFSET
    user_id = MAX_ID_OFFSET + 3011
    await _create_test_user(sqlite_session_factory, user_id)
    scen_id = await _create_test_scenario(sqlite_session_factory, "max_single_ev")

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()

    # 1. User starts MAX flow
    handled = await handle_event_start_max(
        mock_client, chat_id=101, user_id=user_id, start_payload="max_single_ev", session_maker=sqlite_session_factory
    )
    assert handled is True
    assert mock_client.send_message.call_count == 1

    # 2. Worker cycle runs afterwards
    cycle_stats = await run_event_delivery_worker_cycle(sqlite_session_factory, max_client=mock_client)
    assert cycle_stats["claimed"] == 0
    # No duplicate message! Total calls remain 1!
    assert mock_client.send_message.call_count == 1


@pytest.mark.asyncio
async def test_started_cancelled_rejoin_metadata_contract(sqlite_session_factory):
    """Requirement 7: started -> cancelled -> rejoin sets status='started' and action='rejoined'."""
    user_id = await _create_test_user(sqlite_session_factory, 2020)
    scen_id = await _create_test_scenario(sqlite_session_factory, "meta_contract_ev")

    # 1. Register -> status='started'
    await register_user_for_event(sqlite_session_factory, user_id, "meta_contract_ev")
    async with sqlite_session_factory() as session:
        user = await session.get(User, user_id)
        records = load_metadata_records(user.metadata_json)
        assert len(records) == 1
        assert records[-1]["data"]["status"] == "started"

    # 2. Cancel -> status='cancelled'
    await cancel_event_registration(sqlite_session_factory, user_id, scen_id)
    async with sqlite_session_factory() as session:
        user = await session.get(User, user_id)
        records = load_metadata_records(user.metadata_json)
        assert len(records) == 2
        assert records[-1]["data"]["status"] == "cancelled"

    # 3. Rejoin -> status='started', action='rejoined'
    await rejoin_event_registration(sqlite_session_factory, user_id, scen_id)
    async with sqlite_session_factory() as session:
        user = await session.get(User, user_id)
        records = load_metadata_records(user.metadata_json)
        assert len(records) == 3
        last_rec = records[-1]["data"]
        assert last_rec["status"] == "started"
        assert last_rec.get("action") == "rejoined"


@pytest.mark.asyncio
async def test_rejoin_after_reschedule_while_cancelled_recalculates_steps(sqlite_session_factory):
    """Requirement 5: Rejoin after reschedule while cancelled recalculates steps with new starts_at and schedule_revision."""
    user_id = await _create_test_user(sqlite_session_factory, 2030)
    now = datetime.utcnow()
    starts_at = now + timedelta(days=2)
    scen_id = await _create_test_scenario(sqlite_session_factory, "rejoin_resched_ev", starts_at=starts_at)

    # 1. Standard registration
    await register_user_for_event(sqlite_session_factory, user_id, "rejoin_resched_ev", now=now)

    # 2. Cancel
    await cancel_event_registration(sqlite_session_factory, user_id, scen_id, now=now)

    # 3. Reschedule while cancelled (to +4 days)
    new_starts_at = now + timedelta(days=4)
    resched_res = await reschedule_event(sqlite_session_factory, scen_id, new_starts_at, now=now)
    assert resched_res.schedule_revision == 2

    # 4. Early rejoin
    rejoin_res = await rejoin_event_registration(sqlite_session_factory, user_id, scen_id, now=now)
    assert rejoin_res.generation == 2
    assert rejoin_res.mode == "standard"

    # Verify that scheduled steps have recalculated due_at and updated schedule_revision
    async with sqlite_session_factory() as session:
        deliveries = (
            await session.execute(
                select(EventDelivery).where(
                    EventDelivery.registration_id == rejoin_res.registration_id,
                    EventDelivery.message_kind == "scheduled_step",
                )
            )
        ).scalars().all()

        for d in deliveries:
            assert d.status == "pending"
            assert d.schedule_revision == 2
            assert d.registration_generation == 2
            # due_at recalculated from new_starts_at (+4 days), not old starts_at (+2 days)
            assert d.due_at >= now + timedelta(days=3)


@pytest.mark.asyncio
async def test_rejoin_after_reschedule_with_archived_step(sqlite_session_factory):
    """Requirement 5: Step archived while registration was cancelled remains skipped."""
    user_id = await _create_test_user(sqlite_session_factory, 2031)
    now = datetime.utcnow()
    starts_at = now + timedelta(days=2)
    scen_id = await _create_test_scenario(sqlite_session_factory, "arch_step_ev", starts_at=starts_at)

    await register_user_for_event(sqlite_session_factory, user_id, "arch_step_ev", now=now)
    await cancel_event_registration(sqlite_session_factory, user_id, scen_id, now=now)

    # Archive step 1 of scenario
    async with sqlite_session_factory() as session:
        step1 = await session.scalar(
            select(EventStep).where(EventStep.event_id == scen_id, EventStep.sort_order == 1)
        )
        assert step1 is not None
        step1.is_archived = True
        await session.commit()

    # Early rejoin
    rejoin_res = await rejoin_event_registration(sqlite_session_factory, user_id, scen_id, now=now)

    async with sqlite_session_factory() as session:
        d1 = await session.scalar(
            select(EventDelivery).where(
                EventDelivery.registration_id == rejoin_res.registration_id,
                EventDelivery.step_id == step1.id,
            )
        )
        assert d1.status == "skipped"
        assert d1.skip_reason == "step_archived"


@pytest.mark.asyncio
async def test_sqlite_concurrency_duplicate_registration(sqlite_session_factory):
    """Requirement 8: Two concurrent registrations result in exactly 1 registration and 1 immediate delivery."""
    user_id = await _create_test_user(sqlite_session_factory, 2040)
    await _create_test_scenario(sqlite_session_factory, "conc_reg_ev")

    results = await asyncio.gather(
        register_user_for_event(sqlite_session_factory, user_id, "conc_reg_ev"),
        register_user_for_event(sqlite_session_factory, user_id, "conc_reg_ev"),
        return_exceptions=False,
    )

    statuses = [r.status for r in results]
    assert "registered" in statuses
    assert "already_registered" in statuses

    async with sqlite_session_factory() as session:
        regs = (await session.execute(select(EventRegistration).where(EventRegistration.user_id == user_id))).scalars().all()
        assert len(regs) == 1
        delivs = (await session.execute(
            select(EventDelivery).where(
                EventDelivery.user_id == user_id,
                EventDelivery.message_kind == "immediate_confirmation",
            )
        )).scalars().all()
        assert len(delivs) == 1


@pytest.mark.asyncio
async def test_sqlite_concurrency_two_cancels(sqlite_session_factory):
    """Requirement 8: Two concurrent cancels handle locks and sequence safely."""
    user_id = await _create_test_user(sqlite_session_factory, 2041)
    scen_id = await _create_test_scenario(sqlite_session_factory, "conc_canc_ev")

    # Start registered
    await register_user_for_event(sqlite_session_factory, user_id, "conc_canc_ev")

    # Run cancel and another cancel concurrently
    c1, c2 = await asyncio.gather(
        cancel_event_registration(sqlite_session_factory, user_id, scen_id),
        cancel_event_registration(sqlite_session_factory, user_id, scen_id),
    )
    assert (c1.cancelled and c2.already_cancelled) or (c2.cancelled and c1.already_cancelled)

    async with sqlite_session_factory() as session:
        reg = await session.scalar(select(EventRegistration).where(EventRegistration.user_id == user_id))
        assert reg.status == "cancelled"


@pytest.mark.asyncio
async def test_sqlite_concurrency_cancel_vs_rejoin(sqlite_session_factory):
    """Requirement 8: True concurrent Cancel vs Rejoin race safely serializes."""
    user_id = await _create_test_user(sqlite_session_factory, 2042)
    scen_id = await _create_test_scenario(sqlite_session_factory, "conc_canc_rej_real")

    # Start registered, then cancel
    await register_user_for_event(sqlite_session_factory, user_id, "conc_canc_rej_real")
    await cancel_event_registration(sqlite_session_factory, user_id, scen_id)

    # Concurrently launch rejoin and cancel
    rejoin_task = rejoin_event_registration(sqlite_session_factory, user_id, scen_id)
    cancel_task = cancel_event_registration(sqlite_session_factory, user_id, scen_id)

    res_rejoin, res_cancel = await asyncio.gather(rejoin_task, cancel_task)

    assert res_rejoin.registration_id is not None
    assert res_cancel.registration_id is not None
    assert (res_cancel.cancelled or res_cancel.already_cancelled) is True

    async with sqlite_session_factory() as session:
        reg = await session.scalar(select(EventRegistration).where(EventRegistration.user_id == user_id))
        assert reg.status in ("started", "cancelled")
        assert reg.generation >= 2


@pytest.mark.asyncio
async def test_sqlite_concurrency_registration_vs_ai_service_data(sqlite_session_factory):
    """Requirement 7: Concurrent event registration and AI persist_service_data_blocks
    preserve both metadata records without lost updates or deadlocks."""
    from automation_engine import persist_service_data_blocks
    from user_metadata import ServiceDataBlock

    user_id = await _create_test_user(sqlite_session_factory, 2046)
    scen_id = await _create_test_scenario(sqlite_session_factory, "conc_reg_ai_ev")

    ai_block = ServiceDataBlock(
        current_state={},
        events=[],
        metadata={"topic": "psychology", "interest_level": "high"},
        save_mode="merge",
        raw_json='{"metadata":{"topic":"psychology"}}',
        legacy=True,
    )

    reg_task = register_user_for_event(sqlite_session_factory, user_id, "conc_reg_ai_ev")
    ai_task = persist_service_data_blocks(
        sqlite_session_factory,
        user_id=user_id,
        dialogue_id=1,
        topic_id=None,
        blocks=[ai_block],
    )

    reg_res, ai_res = await asyncio.gather(reg_task, ai_task)
    assert reg_res.status == "registered"

    async with sqlite_session_factory() as session:
        user = await session.get(User, user_id)
        records = load_metadata_records(user.metadata_json)
        # Verify event metadata was preserved
        assert any(
            (r.get("type") == "event_scenario" or r.get("data", {}).get("type") == "event_scenario")
            for r in records
        ), f"Missing event metadata in {records}"
        # Verify AI metadata was preserved
        assert any(
            (r.get("data", {}).get("topic") == "psychology")
            for r in records
        ), f"Missing AI metadata in {records}"


@pytest.mark.asyncio
async def test_real_postgresql_rejoin_vs_reschedule_deadlock_free():
    """Item 2 & Requirement 8: Real PostgreSQL concurrency test for rejoin vs reschedule.
    Verifies that with strict lock hierarchy (User -> Scenario -> Registration -> Step -> Delivery),
    concurrent rejoin and reschedule do not deadlock and complete within timeout.
    """
    from tests.test_event_chains_db import _get_safe_postgres_engine
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

        user_id = await _create_test_user(sessions, 9001)
        scen_id = await _create_test_scenario(sessions, "pg_rejoin_resched_ev")

        # Start registered, then cancel
        await register_user_for_event(sessions, user_id, "pg_rejoin_resched_ev")
        await cancel_event_registration(sessions, user_id, scen_id)

        # Run concurrent rejoin and reschedule with strict timeout
        now = datetime.utcnow()
        new_starts = now + timedelta(days=5)

        rejoin_task = rejoin_event_registration(sessions, user_id, scen_id, now=now)
        resched_task = reschedule_event(sessions, scen_id, new_starts, now=now)

        rejoin_res, resched_res = await asyncio.wait_for(
            asyncio.gather(rejoin_task, resched_task),
            timeout=10.0,
        )

        assert rejoin_res.registration_id is not None
        assert resched_res.schedule_revision >= 2

        async with sessions() as session:
            reg = await session.scalar(select(EventRegistration).where(EventRegistration.user_id == user_id))
            assert reg.status in ("started", "cancelled")
            assert reg.generation >= 2
    finally:
        await pg_engine.dispose()



@pytest.mark.asyncio
async def test_sqlite_concurrency_two_reschedules(sqlite_session_factory):
    """Requirement 8: Two concurrent reschedules serialized properly on SQLite."""
    scen_id = await _create_test_scenario(sqlite_session_factory, "conc_resched_ev")
    now = datetime.utcnow()

    res1, res2 = await asyncio.gather(
        reschedule_event(sqlite_session_factory, scen_id, now + timedelta(days=3), now=now),
        reschedule_event(sqlite_session_factory, scen_id, now + timedelta(days=5), now=now),
    )
    # Both reschedules must complete and increment schedule_revision cleanly
    revisions = {res1.schedule_revision, res2.schedule_revision}
    assert revisions == {2, 3}



@pytest.mark.asyncio
async def test_sqlite_concurrency_metadata_append(sqlite_session_factory):
    """Requirement 8: Multiple concurrent metadata appends preserve all records without loss."""
    user_id = await _create_test_user(sqlite_session_factory, 2045)

    async def _append_item(idx: int):
        payload = {"type": "test_concurrency", "index": idx}
        async def _op(session, user):
            await sync_and_append_user_metadata(
                session,
                user,
                [{"data": payload, "raw_json": json.dumps(payload)}],
            )
        await run_user_transaction(sqlite_session_factory, user_id, _op)

    await asyncio.gather(*[_append_item(i) for i in range(10)])

    async with sqlite_session_factory() as session:
        user = await session.get(User, user_id)
        records = load_metadata_records(user.metadata_json)
        assert len(records) == 10
        indices = {r["data"]["index"] for r in records}
        assert indices == set(range(10))


@pytest.mark.asyncio
async def test_automation_events_sqlite_busy_retry(sqlite_session_factory):
    """Requirement 6: automation_events handles SQLite busy retry correctly with OperationalError."""
    from unittest.mock import patch
    from sqlalchemy.exc import OperationalError
    from automation_events import process_pending_events
    from database import AutomationCondition, AutomationEvent, AutomationHandler, AutomationAction

    user_id = await _create_test_user(sqlite_session_factory, 2050)

    # Create handler and action for save_metadata
    async with sqlite_session_factory() as session:
        handler = AutomationHandler(
            name="Test Handler",
            is_active=True,
            include_main_dialogue=True,
        )
        handler.conditions.append(AutomationCondition(
            condition_type="event",
            expected_value="test_event",
            operator="equals",
        ))
        handler.actions.append(AutomationAction(
            action_type="save_metadata",
            recipient_type="current_user",
            metadata_json=json.dumps({"key": "test_val"}),
        ))
        session.add(handler)
        await session.flush()

        ev = AutomationEvent(
            user_id=user_id,
            dialogue_id=1,
            topic_id=0,
            name="test_event",
            state_json="{}",
            metadata_json="{}",
            created_at=datetime.utcnow(),
        )
        session.add(ev)
        await session.commit()

    # Simulate 1 transient SQLite busy OperationalError on sync_and_append_user_metadata
    orig_append = sync_and_append_user_metadata
    call_count = 0

    async def flaky_append(session, user, blocks, *, saved_at=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OperationalError("database is locked", {}, Exception("sqlite3.OperationalError: database is locked"))
        return await orig_append(session, user, blocks, saved_at=saved_at)

    with patch("automation_events.async_session_maker", sqlite_session_factory), \
         patch("automation_events.sync_and_append_user_metadata", side_effect=flaky_append):
        processed = await process_pending_events(bot=None)
        assert processed == 1
        assert call_count >= 2


@pytest.mark.asyncio
async def test_regression_multi_generation_rejoin_urgent_to_standard(sqlite_session_factory):
    """
    Regression Test A:
    urgent registration -> cancel -> reschedule far future -> standard rejoin.
    Verifies:
    - Future scheduled steps are pending again.
    - registration_generation == current registration.generation.
    - schedule_revision == current scenario.schedule_revision.
    - Worker really sends them and does NOT encounter generation mismatch.
    """
    from event_delivery_worker import run_event_delivery_worker_cycle
    user_id = await _create_test_user(sqlite_session_factory, 2060)
    now = datetime.utcnow()
    # Event in 10 minutes (urgent threshold is 15 min -> urgent_late)
    scen_id = await _create_test_scenario(
        sqlite_session_factory,
        code="urgent_to_std_rejoin_ev",
        starts_at=now + timedelta(minutes=10),
        steps=[(-5, "Напоминание за 5 минут")],
    )

    # 1. Urgent registration
    reg_res = await register_user_for_event(
        sqlite_session_factory, user_id, "urgent_to_std_rejoin_ev", platform="telegram", now=now
    )
    assert reg_res.status == "registered"
    assert reg_res.mode == "urgent_late"

    # Verify scheduled step suppressed
    async with sqlite_session_factory() as session:
        sched_d = await session.scalar(
            select(EventDelivery).where(
                EventDelivery.registration_id == reg_res.registration.id,
                EventDelivery.message_kind == "scheduled_step",
            )
        )
        assert sched_d is not None
        assert sched_d.status == "skipped"
        assert sched_d.skip_reason == "late_mode_suppressed"
        gen_1 = reg_res.registration.generation

    # 2. Cancel registration
    cancel_res = await cancel_event_registration(sqlite_session_factory, user_id, scen_id, now=now)
    assert cancel_res.cancelled is True

    # 3. Reschedule event far into the future (e.g. 5 days)
    far_future = now + timedelta(days=5)
    await reschedule_event(sqlite_session_factory, scen_id, far_future, now=now)

    # 4. Standard rejoin
    rejoin_res = await rejoin_event_registration(sqlite_session_factory, user_id, scen_id, now=now)
    assert rejoin_res.generation > gen_1
    current_gen = rejoin_res.generation

    async with sqlite_session_factory() as session:
        scen = await session.get(EventScenario, scen_id)
        current_rev = scen.schedule_revision
        sched_d_rejoined = await session.get(EventDelivery, sched_d.id)
        assert sched_d_rejoined.status == "pending"
        assert sched_d_rejoined.skip_reason is None
        assert sched_d_rejoined.registration_generation == current_gen
        assert sched_d_rejoined.schedule_revision == current_rev
        assert sched_d_rejoined.due_at == far_future - timedelta(minutes=5)
        assert sched_d_rejoined.claim_token is None
        assert sched_d_rejoined.lease_until is None
        assert sched_d_rejoined.claimed_at is None
        assert sched_d_rejoined.send_started_at is None

    # 5. Worker cycle actually sends scheduled step without generation mismatch
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=99991))
    send_time = far_future - timedelta(minutes=4)
    stats = await run_event_delivery_worker_cycle(sqlite_session_factory, bot=mock_bot, now=send_time)
    assert stats["sent"] >= 1
    assert stats["rejected"] == 0

    async with sqlite_session_factory() as session:
        sched_final = await session.get(EventDelivery, sched_d.id)
        assert sched_final.status == "sent"
        assert sched_final.external_message_id == "99991"


@pytest.mark.asyncio
async def test_regression_multi_generation_rejoin_late_step_rescheduled_future(sqlite_session_factory):
    """
    Regression Test B:
    standard -> cancel -> standard rejoin, one step already became late_registration ->
    reschedule step back to future.
    Verifies:
    - Revived delivery gets current generation.
    - Worker accepts and sends it without generation mismatch.
    """
    from event_delivery_worker import run_event_delivery_worker_cycle
    user_id = await _create_test_user(sqlite_session_factory, 2061)
    now = datetime.utcnow()
    # Event in 40 minutes: step at -50 min is already in past (late_registration),
    # step at -20 min is in future (pending)
    scen_id = await _create_test_scenario(
        sqlite_session_factory,
        code="std_cancel_resched_ev",
        starts_at=now + timedelta(minutes=40),
        steps=[
            (-50, "50m reminder (past)"),
            (-20, "20m reminder (future)"),
        ],
    )

    # 1. Standard registration
    reg_res = await register_user_for_event(
        sqlite_session_factory, user_id, "std_cancel_resched_ev", platform="telegram", now=now
    )
    assert reg_res.status == "registered"

    async with sqlite_session_factory() as session:
        delivs = (
            await session.scalars(
                select(EventDelivery)
                .where(EventDelivery.registration_id == reg_res.registration.id)
                .order_by(EventDelivery.id)
            )
        ).all()
        # Find the -50m delivery
        late_deliv = next(d for d in delivs if d.step_id and d.status == "skipped" and d.skip_reason == "late_registration")
        assert late_deliv is not None

    # 2. Cancel registration
    await cancel_event_registration(sqlite_session_factory, user_id, scen_id, now=now)

    # 3. Standard rejoin
    rejoin_res = await rejoin_event_registration(sqlite_session_factory, user_id, scen_id, now=now)
    current_gen = rejoin_res.generation

    # 4. Reschedule scenario into far future so that -50m step becomes future!
    new_starts_at = now + timedelta(days=2)
    await reschedule_event(sqlite_session_factory, scen_id, new_starts_at, now=now)

    # Verify that revived delivery gets current generation
    async with sqlite_session_factory() as session:
        scen = await session.get(EventScenario, scen_id)
        revived = await session.get(EventDelivery, late_deliv.id)
        assert revived.status == "pending"
        assert revived.skip_reason is None
        assert revived.registration_generation == current_gen
        assert revived.schedule_revision == scen.schedule_revision

    # 5. Worker accepts and sends it
    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=99992))
    send_time = new_starts_at - timedelta(minutes=49)
    stats = await run_event_delivery_worker_cycle(sqlite_session_factory, bot=mock_bot, now=send_time)
    assert stats["sent"] >= 1
    assert stats["rejected"] == 0

    async with sqlite_session_factory() as session:
        check = await session.get(EventDelivery, late_deliv.id)
        assert check.status == "sent"
        assert check.external_message_id == "99992"


@pytest.mark.asyncio
async def test_regression_immediate_start_dynamic_clock_expired_telegram(sqlite_session_factory):
    """
    Regression Test (Blocker 2):
    Event ends almost immediately after registration.
    Registration is still valid when handle_event_start_telegram starts,
    but before immediate pre-send execution real time passes past expires_at (now=None).
    Verifies:
    - HTTP send is NOT called.
    - Delivery is marked skipped / expired.
    """
    import event_delivery_worker
    user_id = await _create_test_user(sqlite_session_factory, 2062)
    # Event duration 120 min, started 119m 59.95s ago -> expires in 50ms
    starts_at = datetime.utcnow() - timedelta(minutes=120) + timedelta(milliseconds=60)
    scen_id = await _create_test_scenario(
        sqlite_session_factory,
        code="dyn_clock_start_tg_ev",
        starts_at=starts_at,
        duration=120,
    )

    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()

    orig_prepare = event_delivery_worker.prepare_and_lock_delivery_for_sending

    async def delayed_prepare(*args, **kwargs):
        # Simulate processing/network delay so that dynamic clock passes expires_at
        await asyncio.sleep(0.08)
        return await orig_prepare(*args, **kwargs)

    from unittest.mock import patch
    with patch("event_delivery_worker.prepare_and_lock_delivery_for_sending", side_effect=delayed_prepare):
        handled = await handle_event_start_telegram(
            mock_bot,
            chat_id=2062,
            user_id=user_id,
            start_arg="dyn_clock_start_tg_ev",
            session_maker=sqlite_session_factory,
            now=None,  # dynamic clock
        )
        assert handled is True
        # HTTP send_message must NOT be called because delivery expired before pre-send
        assert mock_bot.send_message.call_count == 0

    async with sqlite_session_factory() as session:
        deliv = await session.scalar(
            select(EventDelivery).where(
                EventDelivery.user_id == user_id,
                EventDelivery.event_id == scen_id,
            )
        )
        assert deliv is not None
        assert deliv.status == "skipped"
        assert deliv.skip_reason == "expired"


@pytest.mark.asyncio
async def test_regression_immediate_start_dynamic_clock_expired_max(sqlite_session_factory):
    """
    Regression Test (Blocker 2 for MAX):
    Same check for handle_event_start_max with dynamic clock (now=None).
    """
    import event_delivery_worker
    from max_messenger_bot.models import MAX_ID_OFFSET
    user_id = MAX_ID_OFFSET + 3063
    await _create_test_user(sqlite_session_factory, user_id)
    starts_at = datetime.utcnow() - timedelta(minutes=120) + timedelta(milliseconds=60)
    scen_id = await _create_test_scenario(
        sqlite_session_factory,
        code="dyn_clock_start_max_ev",
        starts_at=starts_at,
        duration=120,
    )

    mock_client = MagicMock()
    mock_client.send_message = AsyncMock()

    orig_prepare = event_delivery_worker.prepare_and_lock_delivery_for_sending

    async def delayed_prepare(*args, **kwargs):
        await asyncio.sleep(0.08)
        return await orig_prepare(*args, **kwargs)

    from unittest.mock import patch
    with patch("event_delivery_worker.prepare_and_lock_delivery_for_sending", side_effect=delayed_prepare):
        handled = await handle_event_start_max(
            mock_client,
            chat_id=100,
            user_id=user_id,
            start_payload="dyn_clock_start_max_ev",
            session_maker=sqlite_session_factory,
            now=None,  # dynamic clock
        )
        assert handled is True
        assert mock_client.send_message.call_count == 0

    async with sqlite_session_factory() as session:
        deliv = await session.scalar(
            select(EventDelivery).where(
                EventDelivery.user_id == user_id,
                EventDelivery.event_id == scen_id,
            )
        )
        assert deliv is not None
        assert deliv.status == "skipped"
        assert deliv.skip_reason == "expired"


@pytest.mark.asyncio
async def test_rejoin_inactive_scenario_rejected_and_callback_ux(sqlite_session_factory):
    """
    Regression Test 1:
    - active registration -> cancel -> scenario.is_active = False -> rejoin
    - rejoin rejected with ValueError
    - registration remains cancelled
    - generation does not change
    - no new EventDelivery created
    - no new metadata started/rejoined
    - Telegram and MAX callback UX: user does not receive success message.
    """
    now = datetime.utcnow()
    user_id = await _create_test_user(sqlite_session_factory, 7001)
    scen_id = await _create_test_scenario(
        sqlite_session_factory,
        code="rejoin_inactive_ev",
        starts_at=now + timedelta(days=2),
    )

    # 1. Register user
    reg_res = await register_user_for_event(
        sqlite_session_factory, user_id, "rejoin_inactive_ev", platform="telegram", now=now
    )
    assert reg_res.status == "registered"
    reg_id = reg_res.registration.id

    # 2. Cancel registration
    cancel_res = await cancel_event_registration(sqlite_session_factory, user_id, scen_id, now=now)
    assert cancel_res.cancelled is True

    # 3. Disable scenario
    async with sqlite_session_factory() as session:
        scen = await session.get(EventScenario, scen_id)
        scen.is_active = False
        await session.commit()

    # Snapshot before failed rejoin
    async with sqlite_session_factory() as session:
        delivs_before = (
            await session.scalars(
                select(EventDelivery).where(EventDelivery.registration_id == reg_id)
            )
        ).all()
        deliv_count_before = len(delivs_before)
        deliv_ids_before = {d.id for d in delivs_before}

        u = await session.get(User, user_id)
        records_before = load_metadata_records(u.metadata_json)
        reg_before = await session.get(EventRegistration, reg_id)
        gen_before = reg_before.generation
        assert reg_before.status == "cancelled"

    # 4. Attempt rejoin -> must fail closed
    with pytest.raises(ValueError, match="не активен"):
        await rejoin_event_registration(sqlite_session_factory, user_id, scen_id, now=now)

    # Verify state remains completely unchanged
    async with sqlite_session_factory() as session:
        reg_after = await session.get(EventRegistration, reg_id)
        assert reg_after.status == "cancelled"
        assert reg_after.generation == gen_before

        delivs_after = (
            await session.scalars(
                select(EventDelivery).where(EventDelivery.registration_id == reg_id)
            )
        ).all()
        assert len(delivs_after) == deliv_count_before
        assert {d.id for d in delivs_after} == deliv_ids_before

        u_after = await session.get(User, user_id)
        records_after = load_metadata_records(u_after.metadata_json)
        assert len(records_after) == len(records_before)
        assert not any(
            r.get("data", {}).get("action") == "rejoined" or r.get("data", {}).get("status") == "started"
            for r in records_after[len(records_before):]
        )

    # 5. Telegram Callback UX test
    cb_tg = MagicMock()
    cb_tg.data = f"ev_rejoin:{scen_id}"
    cb_tg.from_user.id = user_id
    cb_tg.answer = AsyncMock()
    cb_tg.message = MagicMock()
    cb_tg.message.edit_text = AsyncMock()
    cb_tg.message.answer = AsyncMock()

    handled_tg = await handle_event_callback_telegram(cb_tg, session_maker=sqlite_session_factory, now=now)
    assert handled_tg is True
    # User does NOT receive success edit_text
    assert cb_tg.message.edit_text.call_count == 0
    assert cb_tg.message.answer.call_count == 0
    # User receives error alert
    assert cb_tg.answer.called
    alert_text = cb_tg.answer.call_args[0][0]
    assert "не активен" in alert_text.lower() or "не удалось" in alert_text.lower()

    # 6. MAX Callback UX test
    from max_messenger_bot.models import MAX_ID_OFFSET
    user_max_id = MAX_ID_OFFSET + 7002
    await _create_test_user(sqlite_session_factory, user_max_id)
    # Enable scenario temporarily to register and cancel MAX user
    async with sqlite_session_factory() as session:
        scen = await session.get(EventScenario, scen_id)
        scen.is_active = True
        await session.commit()

    reg_max = await register_user_for_event(
        sqlite_session_factory, user_max_id, "rejoin_inactive_ev", platform="max", now=now
    )
    await cancel_event_registration(sqlite_session_factory, user_max_id, scen_id, now=now)

    # Disable scenario again
    async with sqlite_session_factory() as session:
        scen = await session.get(EventScenario, scen_id)
        scen.is_active = False
        await session.commit()

    mock_max_client = MagicMock()
    mock_max_client.send_message = AsyncMock()
    mock_max_client.answer_callback = AsyncMock()

    cb_max = MagicMock()
    cb_max.payload = f"ev_rejoin:{scen_id}"
    cb_max.sender.user_id = user_max_id
    cb_max.chat_id = 999
    cb_max.callback_id = "cb_max_inactive_1"

    handled_max = await handle_event_callback_max(
        mock_max_client, cb_max, session_maker=sqlite_session_factory, now=now
    )
    assert handled_max is True
    # Verify MAX client was called with error message, NOT success message
    assert mock_max_client.send_message.called
    sent_text = mock_max_client.send_message.call_args.kwargs["text"]
    assert "снова зарегистрированы" not in sent_text
    assert "не активен" in sent_text.lower() or "не удалось" in sent_text.lower()


@pytest.mark.asyncio
async def test_rejoin_with_terminal_skipped_delivery_reconciliation(sqlite_session_factory):
    """
    Regression Test 2:
    - create scheduled r0
    - set to skipped/step_archived and skipped/disabled_past
    - registration cancelled
    - restore scenario/step to active future state where rejoin is possible
    - perform rejoin
    - does not fail on UNIQUE constraint
    - terminal delivery does NOT resurrect
    - second r0 delivery is NOT created
    """
    now = datetime.utcnow()
    user_id = await _create_test_user(sqlite_session_factory, 7003)
    scen_id = await _create_test_scenario(
        sqlite_session_factory,
        code="term_skip_rejoin_ev",
        starts_at=now + timedelta(days=2),
        steps=[
            (-60, "Напоминание за 1 час"),
            (-30, "Напоминание за 30 минут"),
        ],
    )

    reg_res = await register_user_for_event(
        sqlite_session_factory, user_id, "term_skip_rejoin_ev", platform="telegram", now=now
    )
    assert reg_res.status == "registered"
    reg_id = reg_res.registration.id

    async with sqlite_session_factory() as session:
        delivs = (
            await session.scalars(
                select(EventDelivery)
                .where(EventDelivery.registration_id == reg_id, EventDelivery.message_kind == "scheduled_step")
                .order_by(EventDelivery.id)
            )
        ).all()
        assert len(delivs) == 2
        d_archived = delivs[0]
        d_disabled = delivs[1]

        # Simulate non-recoverable skip reasons
        d_archived.status = "skipped"
        d_archived.skip_reason = "step_archived"
        d_archived.finished_at = now

        d_disabled.status = "skipped"
        d_disabled.skip_reason = "disabled_past"
        d_disabled.finished_at = now
        await session.commit()

    # Cancel registration
    await cancel_event_registration(sqlite_session_factory, user_id, scen_id, now=now)

    # Rejoin registration: scenario and steps are in future and active
    rejoin_res = await rejoin_event_registration(
        sqlite_session_factory, user_id, scen_id, now=now
    )
    assert rejoin_res.generation == 2

    # Verify deliveries in DB
    async with sqlite_session_factory() as session:
        all_delivs = (
            await session.scalars(
                select(EventDelivery)
                .where(EventDelivery.registration_id == reg_id)
                .order_by(EventDelivery.id)
            )
        ).all()

        step_delivs = [d for d in all_delivs if d.message_kind == "scheduled_step"]
        # Exactly 2 scheduled step deliveries exist - no second r0 was created!
        assert len(step_delivs) == 2

        check_archived = next(d for d in step_delivs if d.id == d_archived.id)
        assert check_archived.status == "skipped"
        assert check_archived.skip_reason == "step_archived"

        check_disabled = next(d for d in step_delivs if d.id == d_disabled.id)
        assert check_disabled.status == "skipped"
        assert check_disabled.skip_reason == "disabled_past"

        # Unique r0 check
        step_ids = {d.step_id for d in step_delivs}
        for sid in step_ids:
            r0s = [d for d in step_delivs if d.step_id == sid and d.retry_index == 0]
            assert len(r0s) == 1




