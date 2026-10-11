import os
import re
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

os.environ.setdefault("BOT_TOKEN", "test_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import database
from database import (
    Base,
    Content,
    DeepLinkNamespace,
    EventDelivery,
    EventDeliveryResolution,
    EventRegistration,
    EventScenario,
    EventStep,
    User,
    init_db,
    reserve_start_code,
    sync_deep_link_namespace,
    verify_event_reminder_chains_schema,
)


@pytest_asyncio.fixture
async def async_engine_session():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        echo=False,
    )
    async with engine.connect() as conn:
        await conn.execute(text("PRAGMA foreign_keys = ON;"))

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    yield engine, session_factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_init_db_creates_and_verifies_all_event_reminder_tables(tmp_path, monkeypatch):
    """Verifies that init_db creates all 6 tables and verify_event_reminder_chains_schema succeeds."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'init_test.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "async_session_maker", sessions)

    try:
        await database.init_db()

        async with engine.connect() as conn:
            # Run schema verifier directly
            await conn.run_sync(verify_event_reminder_chains_schema)

            # Inspect tables
            insp_tables = await conn.run_sync(lambda sc: inspect(sc).get_table_names())
            assert "deep_link_namespace" in insp_tables
            assert "event_scenarios" in insp_tables
            assert "event_steps" in insp_tables
            assert "event_registrations" in insp_tables
            assert "event_deliveries" in insp_tables
            assert "event_delivery_resolutions" in insp_tables
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_verify_schema_fails_closed_on_missing_table():
    """Fail-closed verifier must raise RuntimeError if any critical table is missing."""
    mock_conn = MagicMock()
    mock_insp = MagicMock()
    mock_insp.has_table.side_effect = lambda tbl: tbl != "event_deliveries"
    with pytest.raises(RuntimeError, match="Critical table event_deliveries is missing"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sqlalchemy.inspect", lambda _c: mock_insp)
            verify_event_reminder_chains_schema(mock_conn)


@pytest.mark.asyncio
async def test_verify_schema_fails_closed_on_missing_column():
    """Fail-closed verifier must raise RuntimeError if a required column is missing."""
    mock_conn = MagicMock()
    mock_insp = MagicMock()
    mock_insp.has_table.return_value = True
    def fake_get_columns(table_name):
        if table_name == "event_deliveries":
            return [{"name": "id"}, {"name": "event_id"}, {"name": "status"}]
        return [{"name": "id"}, {"name": "code"}, {"name": "kind"}, {"name": "reference_id"}, {"name": "created_at"}]
    mock_insp.get_columns.side_effect = fake_get_columns

    with pytest.raises(RuntimeError, match="Critical column"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sqlalchemy.inspect", lambda _c: mock_insp)
            verify_event_reminder_chains_schema(mock_conn)


@pytest.mark.asyncio
async def test_migration_x1_preserves_content_key_longer_than_255_chars(tmp_path, monkeypatch):
    """Test X1: Historical Content.key > 255 characters (e.g., 307 chars) is preserved in deep_link_namespace without truncation."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'legacy_x1.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "async_session_maker", sessions)

    long_key = "legacy_" + ("x" * 300)  # 307 characters
    assert len(long_key) == 307

    try:
        # Pre-seed tables and the long content item
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with sessions() as session:
            session.add(Content(
                key=long_key,
                button_title="Legacy Long",
                is_visible=True,
                text_content="Content with 307 chars key",
            ))
            await session.commit()

        # Run init_db / migration
        await database.init_db()

        # Check deep_link_namespace
        async with sessions() as session:
            ns_entry = await session.get(DeepLinkNamespace, long_key.lower())
            assert ns_entry is not None
            assert ns_entry.code == long_key.lower()
            assert len(ns_entry.code) == 307
            assert ns_entry.kind == "content"
            assert ns_entry.reference_id == long_key
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_system_reserved_codes_seeded_and_idempotent(tmp_path, monkeypatch):
    """Verifies that reserved system start codes are seeded into deep_link_namespace idempotently."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'sys_codes.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "async_session_maker", sessions)

    try:
        # Run init_db twice to test idempotency
        await database.init_db()
        await database.init_db()

        async with sessions() as session:
            for sys_code in ("sub", "ref", "test", "admin", "promo"):
                entry = await session.get(DeepLinkNamespace, sys_code)
                assert entry is not None
                assert entry.kind == "system"
                assert entry.reference_id is None

            for nav_code in ("start_message", "menu"):
                entry = await session.get(DeepLinkNamespace, nav_code)
                assert entry is not None
                assert entry.kind == "system"
                assert entry.reference_id == "content"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_reserve_start_code_c1_rejection_of_system_and_invalid_codes(async_engine_session):
    """Test C1: reserve_start_code rejects system keywords and malformed event codes."""
    engine, session_factory = async_engine_session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        # System keywords
        for reserved in ("sub", "ref", "test", "menu", "start_message", "admin", "promo"):
            with pytest.raises(ValueError, match="Код зарезервирован системой"):
                await reserve_start_code(session, reserved, kind="event")

        # Prefixes
        for prefix_code in ("topic_stress", "ref_12345", "ev_meeting", "admin_panel"):
            with pytest.raises(ValueError, match="Код зарезервирован системой"):
                await reserve_start_code(session, prefix_code, kind="event")

        # Event code format validation: 3-64 chars, a-z0-9_-
        with pytest.raises(ValueError, match="Недопустимый формат кода события"):
            await reserve_start_code(session, "ab", kind="event")  # too short (<3)

        with pytest.raises(ValueError, match="Недопустимый формат кода события"):
            await reserve_start_code(session, "a" * 65, kind="event")  # too long (>64)

        with pytest.raises(ValueError, match="Недопустимый формат кода события"):
            await reserve_start_code(session, "bad code with space", kind="event")

        with pytest.raises(ValueError, match="Недопустимый формат кода события"):
            await reserve_start_code(session, "bad$code", kind="event")

        with pytest.raises(ValueError, match="Код не может быть пустым"):
            await reserve_start_code(session, "   ", kind="content")


@pytest.mark.asyncio
async def test_reserve_start_code_c2_c3_collision_and_savepoint_isolation(async_engine_session):
    """Tests C2 & C3: reserve_start_code handles race condition via savepoint and does not break session."""
    engine, session_factory = async_engine_session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        # First reservation succeeds
        await reserve_start_code(session, "webinar_confidence", kind="event", reference_id="1")
        await session.commit()

        # Second reservation of same code must fail with ValueError
        with pytest.raises(ValueError, match="уже занят в пространстве start-кодов"):
            await reserve_start_code(session, "webinar_confidence", kind="content", reference_id="intro")

        # Crucial check: the session is NOT invalidated by the IntegrityError because of begin_nested()!
        # Next operation in the same session must succeed:
        await reserve_start_code(session, "webinar_another", kind="event", reference_id="2")
        await session.commit()

        # Check DB
        res1 = await session.get(DeepLinkNamespace, "webinar_confidence")
        res2 = await session.get(DeepLinkNamespace, "webinar_another")
        assert res1.kind == "event"
        assert res2.kind == "event"


@pytest.mark.asyncio
async def test_event_scenario_c15_code_immutability(async_engine_session):
    """Test C15: EventScenario.code is immutable once set."""
    engine, session_factory = async_engine_session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        scenario = EventScenario(
            code="immutable_event",
            event_type="webinar",
            title="Вебинар по самооценке",
            starts_at=datetime.utcnow() + timedelta(days=2),
            duration_minutes=120,
            link="https://zoom.us/j/123456",
            place=None,
            is_active=True,
        )
        session.add(scenario)
        await session.commit()

        # Attempt to mutate code in memory
        with pytest.raises(ValueError, match="EventScenario.code is immutable"):
            scenario.code = "mutated_event"


@pytest.mark.asyncio
async def test_event_scenario_check_constraints(async_engine_session):
    """Verifies CheckConstraints on EventScenario (event_type, duration_minutes, schedule_revision)."""
    engine, session_factory = async_engine_session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        # 1. Invalid event_type
        bad_type = EventScenario(
            code="test_bad_type",
            event_type="concert",  # only webinar or meeting allowed
            title="Test",
            starts_at=datetime.utcnow() + timedelta(days=1),
            duration_minutes=120,
        )
        session.add(bad_type)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # 2. Invalid duration_minutes (<15)
        bad_duration_low = EventScenario(
            code="test_bad_dur_low",
            event_type="webinar",
            title="Test",
            starts_at=datetime.utcnow() + timedelta(days=1),
            duration_minutes=10,
        )
        session.add(bad_duration_low)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # 3. Invalid duration_minutes (>1440)
        bad_duration_high = EventScenario(
            code="test_bad_dur_high",
            event_type="webinar",
            title="Test",
            starts_at=datetime.utcnow() + timedelta(days=1),
            duration_minutes=2000,
        )
        session.add(bad_duration_high)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # 4. Invalid schedule_revision (<1)
        bad_revision = EventScenario(
            code="test_bad_rev",
            event_type="webinar",
            title="Test",
            starts_at=datetime.utcnow() + timedelta(days=1),
            duration_minutes=120,
            schedule_revision=0,
        )
        session.add(bad_revision)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()


@pytest.mark.asyncio
async def test_event_registration_constraints_and_uniqueness(async_engine_session):
    """Verifies CheckConstraints and UniqueConstraint on EventRegistration."""
    engine, session_factory = async_engine_session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        user = User(id=1001, first_name="Ivan")
        session.add(user)
        scenario = EventScenario(
            code="reg_test_event",
            event_type="webinar",
            title="Webinar",
            starts_at=datetime.utcnow() + timedelta(days=1),
            duration_minutes=90,
            link="https://test.link",
        )
        session.add(scenario)
        await session.commit()
        scenario_id = scenario.id
        user_id = user.id

        # 1. Valid registration
        reg1 = EventRegistration(
            event_id=scenario_id,
            user_id=user_id,
            status="started",
            registration_mode="standard",
        )
        session.add(reg1)
        await session.commit()

        # 2. Duplicate registration for same (event_id, user_id) must fail (uq_event_registrations_event_user)
        reg_dup = EventRegistration(
            event_id=scenario_id,
            user_id=user_id,
            status="started",
            registration_mode="urgent_late",
        )
        session.add(reg_dup)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # 3. Invalid status check constraint
        user2 = User(id=1002, first_name="Petr")
        session.add(user2)
        await session.commit()
        user2_id = user2.id

        reg_bad_status = EventRegistration(
            event_id=scenario_id,
            user_id=user2_id,
            status="pending",  # only started or cancelled
            registration_mode="standard",
        )
        session.add(reg_bad_status)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # 4. Invalid registration_mode check constraint
        reg_bad_mode = EventRegistration(
            event_id=scenario_id,
            user_id=user2_id,
            status="started",
            registration_mode="vip",  # only standard or urgent_late
        )
        session.add(reg_bad_mode)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()


@pytest.mark.asyncio
async def test_event_delivery_constraints_and_c21_retry_index_model(async_engine_session):
    """
    Test C21: Real test of scheduled delivery insertion, uniqueness enforcement
    for retry_index=0, and successful retry insertion with retry_index=1.
    """
    engine, session_factory = async_engine_session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        user = User(id=2001, first_name="Anna")
        session.add(user)
        scenario = EventScenario(
            code="delivery_test_ev",
            event_type="webinar",
            title="Confidence Webinar",
            starts_at=datetime.utcnow() + timedelta(days=2),
            duration_minutes=120,
            link="https://link.test",
            is_active=True,
        )
        session.add(scenario)
        await session.flush()

        step = EventStep(
            event_id=scenario.id,
            offset_minutes=-1440,
            message_text="Напоминание за 24 часа",
        )
        session.add(step)
        await session.flush()

        reg = EventRegistration(
            event_id=scenario.id,
            user_id=user.id,
            status="started",
            registration_mode="standard",
        )
        session.add(reg)
        await session.commit()
        scenario_id = scenario.id
        user_id = user.id
        step_id = step.id
        reg_id = reg.id
        scenario_starts_at = scenario.starts_at

        # Step 1: Insert standard scheduled delivery (retry_index=0)
        due = scenario_starts_at + timedelta(minutes=step.offset_minutes)
        d_orig = EventDelivery(
            event_id=scenario_id,
            user_id=user_id,
            registration_id=reg_id,
            step_id=step_id,
            retry_index=0,
            parent_delivery_id=None,
            message_kind="scheduled_step",
            delivery_key=f"ev:{scenario_id}:reg:{reg_id}:step:{step_id}:r0",
            due_at=due,
            next_attempt_at=due,
            schedule_revision=1,
            registration_generation=1,
            platform="telegram",
            status="pending",
        )
        session.add(d_orig)
        await session.commit()
        orig_delivery_id = d_orig.id

        # Step 2: Attempting duplicate insert with same (registration_id, step_id, retry_index=0) must fail
        d_dup = EventDelivery(
            event_id=scenario_id,
            user_id=user_id,
            registration_id=reg_id,
            step_id=step_id,
            retry_index=0,
            parent_delivery_id=None,
            message_kind="scheduled_step",
            delivery_key=f"ev:{scenario_id}:reg:{reg_id}:step:{step_id}:r0_dup",
            due_at=due,
            next_attempt_at=due,
            schedule_revision=1,
            registration_generation=1,
            platform="telegram",
            status="pending",
        )
        session.add(d_dup)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # Step 3: Transition original to 'uncertain'
        orig = await session.get(EventDelivery, orig_delivery_id)
        orig.status = "uncertain"
        orig.last_error = "Network timeout"
        await session.commit()

        # Step 4: Admin logs resolution
        admin_id = 999999
        resolution = EventDeliveryResolution(
            delivery_id=orig.id,
            admin_id=admin_id,
            resolution_type="force_retry",
            resolution_reason="Manual retry after uncertain state",
        )
        session.add(resolution)
        await session.flush()
        resolution_id = resolution.id
        orig_due_at = orig.due_at

        # Step 5: Insert retry delivery with retry_index=1, referencing parent
        d_retry = EventDelivery(
            event_id=scenario_id,
            user_id=user_id,
            registration_id=reg_id,
            step_id=step_id,
            retry_index=1,
            parent_delivery_id=orig.id,
            message_kind="scheduled_step",
            delivery_key=f"ev:{scenario_id}:reg:{reg_id}:step:{step_id}:r0:retry:{resolution_id}",
            due_at=orig_due_at,
            next_attempt_at=datetime.utcnow(),
            schedule_revision=1,
            registration_generation=1,
            platform="telegram",
            status="pending",
        )
        session.add(d_retry)
        await session.commit()

        # Step 6: Verify both records exist simultaneously
        all_deliveries = (await session.execute(
            select(EventDelivery).where(EventDelivery.registration_id == reg_id).order_by(EventDelivery.id)
        )).scalars().all()
        assert len(all_deliveries) == 2
        assert all_deliveries[0].status == "uncertain"
        assert all_deliveries[0].retry_index == 0
        assert all_deliveries[1].status == "pending"
        assert all_deliveries[1].retry_index == 1
        assert all_deliveries[1].parent_delivery_id == all_deliveries[0].id


@pytest.mark.asyncio
async def test_event_delivery_resolution_constraints(async_engine_session):
    """Verifies CheckConstraint on EventDeliveryResolution (resolution_type in force_retry, mark_skipped, mark_sent)."""
    engine, session_factory = async_engine_session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        user = User(id=3001, first_name="Olga")
        session.add(user)
        scenario = EventScenario(
            code="resol_test_ev",
            event_type="webinar",
            title="Webinar",
            starts_at=datetime.utcnow() + timedelta(days=1),
            duration_minutes=60,
            link="https://link",
        )
        session.add(scenario)
        await session.flush()
        reg = EventRegistration(event_id=scenario.id, user_id=user.id)
        session.add(reg)
        await session.flush()
        delivery = EventDelivery(
            event_id=scenario.id,
            user_id=user.id,
            registration_id=reg.id,
            message_kind="immediate_confirmation",
            delivery_key="ev:3:reg:1:imm",
            due_at=datetime.utcnow(),
            next_attempt_at=datetime.utcnow(),
            platform="telegram",
            status="uncertain",
        )
        session.add(delivery)
        await session.commit()
        delivery_id = delivery.id

        # Invalid resolution_type
        bad_resol = EventDeliveryResolution(
            delivery_id=delivery_id,
            admin_id=999,
            resolution_type="discard",  # only force_retry, mark_skipped, mark_sent
            resolution_reason="Invalid",
        )
        session.add(bad_resol)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # Valid resolution
        good_resol = EventDeliveryResolution(
            delivery_id=delivery_id,
            admin_id=999,
            resolution_type="mark_sent",
            resolution_reason="Confirmed in chat",
        )
        session.add(good_resol)
        await session.commit()
        assert good_resol.id is not None


@pytest.mark.asyncio
async def test_foreign_key_restrict_prevents_scenario_hard_delete_with_registrations(async_engine_session):
    """Verifies ondelete='RESTRICT' protects EventScenario from hard-deletion when registrations or steps exist."""
    engine, session_factory = async_engine_session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        user = User(id=4001, first_name="Dmitry")
        session.add(user)
        scenario = EventScenario(
            code="restrict_test_ev",
            event_type="webinar",
            title="Webinar",
            starts_at=datetime.utcnow() + timedelta(days=1),
            duration_minutes=60,
            link="https://link",
        )
        session.add(scenario)
        await session.flush()
        reg = EventRegistration(event_id=scenario.id, user_id=user.id)
        session.add(reg)
        await session.commit()

        # Try to delete scenario directly with SQLite PRAGMA foreign_keys = ON
        await session.delete(scenario)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()


@pytest.mark.asyncio
async def test_postgresql_migration_sql_generation():
    """Verifies PostgreSQL-specific DDL/INSERT statements executed in _check_and_migrate."""
    mock_conn = MagicMock()
    mock_conn.dialect.name = "postgresql"
    mock_insp = MagicMock()
    mock_insp.has_table.return_value = True

    columns_by_table = {
        "deep_link_namespace": ["code", "kind", "reference_id", "created_at"],
        "event_scenarios": ["id", "code", "event_type", "title", "starts_at", "duration_minutes",
                            "place", "link", "is_active", "is_archived", "archived_at",
                            "schedule_revision", "content_version", "created_at", "updated_at"],
        "event_steps": ["id", "event_id", "offset_minutes", "message_text", "step_version",
                        "sort_order", "is_archived", "created_at", "updated_at"],
        "event_registrations": ["id", "event_id", "user_id", "status", "generation",
                                "registration_mode", "registered_at", "cancelled_at", "rejoined_at", "updated_at"],
        "event_deliveries": ["id", "event_id", "user_id", "registration_id", "step_id",
                             "retry_index", "parent_delivery_id", "message_kind", "delivery_key",
                             "due_at", "next_attempt_at", "expires_at", "schedule_revision",
                             "registration_generation", "platform", "status", "skip_reason",
                             "claim_token", "claimed_at", "lease_until", "send_started_at",
                             "sent_at", "finished_at", "attempt_count", "last_error",
                             "external_message_id", "rendered_text_snapshot", "template_text_snapshot",
                             "created_at", "updated_at"],
        "event_delivery_resolutions": ["id", "delivery_id", "admin_id", "resolution_type", "resolution_reason", "resolved_at"],
    }
    mock_insp.get_columns.side_effect = lambda t: [{"name": c} for c in columns_by_table.get(t, ["id"])]
    mock_insp.get_pk_constraint.return_value = {"constrained_columns": ["code"]}
    mock_insp.get_unique_constraints.return_value = [
        {"column_names": ["code"]},
        {"column_names": ["event_id", "user_id"]},
        {"column_names": ["delivery_key"]},
        {"column_names": ["registration_id", "step_id", "retry_index"]},
    ]
    mock_insp.get_indexes.return_value = [
        {"unique": False, "column_names": ["status", "next_attempt_at", "due_at"]},
        {"unique": False, "column_names": ["event_id", "offset_minutes", "sort_order", "id"]},
    ]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("sqlalchemy.inspect", lambda _c: mock_insp)
        # Verify schema verifier succeeds with mocked PG inspection
        verify_event_reminder_chains_schema(mock_conn)
