import os
import re
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import inspect, select, text
from sqlalchemy.engine.url import make_url
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
    DeliveryRetryError,
    EventDelivery,
    EventDeliveryResolution,
    EventRegistration,
    EventScenario,
    EventStep,
    User,
    create_event_delivery_force_retry,
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

        # Validation: kind must be one of 'event', 'content', 'system'
        for invalid_kind in ("unknown", "payment", "custom", "event_wrong"):
            with pytest.raises(ValueError, match="Недопустимый kind"):
                await reserve_start_code(session, "valid_code_123", kind=invalid_kind)


@pytest.mark.asyncio
async def test_deep_link_namespace_kind_db_check_constraint(async_engine_session):
    """DB check constraint ck_deep_link_namespace_kind must reject direct insertion of unknown kind."""
    engine, session_factory = async_engine_session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        bad_entry = DeepLinkNamespace(code="test_bad_kind", kind="hacked")
        session.add(bad_entry)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # Valid kinds pass
        for valid_kind in ("event", "content", "system"):
            entry = DeepLinkNamespace(code=f"code_{valid_kind}", kind=valid_kind)
            session.add(entry)
        await session.commit()


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
            link="https://zoom.us",
            place=None,
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
            link="https://zoom.us",
            place=None,
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
            link="https://zoom.us",
            place=None,
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
            link="https://zoom.us",
            place=None,
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
def _build_valid_mock_insp():
    mock_insp = MagicMock()
    mock_insp.has_table.return_value = True

    columns_specs = {
        "deep_link_namespace": [("code", False), ("kind", False), ("reference_id", True), ("created_at", False)],
        "event_scenarios": [
            ("id", False), ("code", False), ("event_type", False), ("title", False),
            ("starts_at", False), ("duration_minutes", False), ("place", True),
            ("link", True), ("is_active", False), ("is_archived", False),
            ("archived_at", True), ("schedule_revision", False), ("content_version", False),
            ("created_at", False), ("updated_at", False),
        ],
        "event_steps": [
            ("id", False), ("event_id", False), ("offset_minutes", False),
            ("message_text", False), ("step_version", False), ("sort_order", False),
            ("is_archived", False), ("created_at", False), ("updated_at", False),
        ],
        "event_registrations": [
            ("id", False), ("event_id", False), ("user_id", False), ("status", False),
            ("generation", False), ("registration_mode", False), ("registered_at", False),
            ("cancelled_at", True), ("rejoined_at", True), ("updated_at", False),
        ],
        "event_deliveries": [
            ("id", False), ("event_id", False), ("user_id", False), ("registration_id", False),
            ("step_id", True), ("retry_index", False), ("parent_delivery_id", True),
            ("message_kind", False), ("delivery_key", False), ("due_at", False),
            ("next_attempt_at", False), ("expires_at", True), ("schedule_revision", False),
            ("registration_generation", False), ("platform", False), ("status", False),
            ("skip_reason", True), ("claim_token", True), ("claimed_at", True),
            ("lease_until", True), ("send_started_at", True), ("sent_at", True),
            ("finished_at", True), ("attempt_count", False), ("last_error", True),
            ("external_message_id", True), ("rendered_text_snapshot", True),
            ("template_text_snapshot", True), ("created_at", False), ("updated_at", False),
        ],
        "event_delivery_resolutions": [
            ("id", False), ("delivery_id", False), ("admin_id", False),
            ("resolution_type", False), ("resolution_reason", False), ("resolved_at", False),
        ],
    }
    mock_insp.get_columns.side_effect = lambda tbl: [
        {"name": name, "nullable": nullable}
        for name, nullable in columns_specs.get(tbl, [("id", False)])
    ]
    mock_insp.get_foreign_keys.side_effect = lambda tbl: {
        "event_steps": [
            {"constrained_columns": ["event_id"], "referred_table": "event_scenarios", "referred_columns": ["id"], "options": {"ondelete": "RESTRICT"}},
        ],
        "event_registrations": [
            {"constrained_columns": ["event_id"], "referred_table": "event_scenarios", "referred_columns": ["id"], "options": {"ondelete": "RESTRICT"}},
            {"constrained_columns": ["user_id"], "referred_table": "users", "referred_columns": ["id"], "options": {"ondelete": "RESTRICT"}},
        ],
        "event_deliveries": [
            {"constrained_columns": ["event_id"], "referred_table": "event_scenarios", "referred_columns": ["id"], "options": {"ondelete": "RESTRICT"}},
            {"constrained_columns": ["user_id"], "referred_table": "users", "referred_columns": ["id"], "options": {"ondelete": "RESTRICT"}},
            {"constrained_columns": ["registration_id"], "referred_table": "event_registrations", "referred_columns": ["id"], "options": {"ondelete": "RESTRICT"}},
            {"constrained_columns": ["step_id"], "referred_table": "event_steps", "referred_columns": ["id"], "options": {"ondelete": "RESTRICT"}},
            {"constrained_columns": ["parent_delivery_id"], "referred_table": "event_deliveries", "referred_columns": ["id"], "options": {"ondelete": "RESTRICT"}},
        ],
        "event_delivery_resolutions": [
            {"constrained_columns": ["delivery_id"], "referred_table": "event_deliveries", "referred_columns": ["id"], "options": {"ondelete": "RESTRICT"}},
        ],
    }.get(tbl, [])
    mock_insp.get_check_constraints.side_effect = lambda tbl: {
        "deep_link_namespace": [
            {"name": "ck_deep_link_namespace_kind"},
        ],
        "event_scenarios": [
            {"name": "ck_event_scenarios_type"},
            {"name": "ck_event_scenarios_duration"},
            {"name": "ck_event_scenarios_revision"},
            {"name": "ck_event_scenarios_content_version"},
            {"name": "ck_event_scenarios_location"},
        ],
        "event_steps": [
            {"name": "ck_event_steps_step_version"},
        ],
        "event_registrations": [
            {"name": "ck_event_registrations_status"},
            {"name": "ck_event_registrations_mode"},
            {"name": "ck_event_registrations_generation"},
        ],
        "event_deliveries": [
            {"name": "ck_event_deliveries_kind"},
            {"name": "ck_event_deliveries_status"},
            {"name": "ck_event_deliveries_step_kind"},
            {"name": "ck_event_deliveries_platform"},
            {"name": "ck_event_deliveries_retry_index"},
            {"name": "ck_event_deliveries_reg_gen"},
            {"name": "ck_event_deliveries_attempt_count"},
        ],
        "event_delivery_resolutions": [
            {"name": "ck_event_delivery_resolution_type"},
        ],
    }.get(tbl, [])
    mock_insp.get_pk_constraint.side_effect = lambda tbl: (
        {"constrained_columns": ["code"]} if tbl == "deep_link_namespace" else {"constrained_columns": ["id"]}
    )
    mock_insp.get_unique_constraints.side_effect = lambda tbl: {
        "event_scenarios": [{"column_names": ["code"]}],
        "event_registrations": [{"column_names": ["event_id", "user_id"]}],
        "event_deliveries": [
            {"column_names": ["delivery_key"]},
            {"column_names": ["registration_id", "step_id", "retry_index"]},
        ],
    }.get(tbl, [])
    mock_insp.get_indexes.side_effect = lambda tbl: {
        "event_scenarios": [{"name": "ix_event_scenarios_code", "unique": True, "column_names": ["code"]}],
        "event_steps": [
            {"name": "ix_event_steps_execution_order", "unique": False, "column_names": ["event_id", "offset_minutes", "sort_order", "id"]},
        ],
        "event_deliveries": [
            {"name": "ix_event_deliveries_poll", "unique": False, "column_names": ["status", "next_attempt_at", "due_at"]},
            {"name": "uq_event_deliveries_parent_delivery_id", "unique": True, "column_names": ["parent_delivery_id"]},
        ],
    }.get(tbl, [])
    return mock_insp


@pytest.mark.asyncio
async def test_verify_schema_mock_inspection_pass():
    """Unit test: verifier succeeds when inspection matches all fail-closed criteria."""
    mock_conn = MagicMock()
    mock_insp = _build_valid_mock_insp()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("sqlalchemy.inspect", lambda _c: mock_insp)
        verify_event_reminder_chains_schema(mock_conn)


@pytest.mark.asyncio
async def test_verify_schema_fails_closed_on_nullable_column():
    """Fail-closed verifier must raise RuntimeError if any critical NOT NULL column is nullable."""
    mock_conn = MagicMock()
    mock_insp = _build_valid_mock_insp()
    def bad_get_columns(tbl):
        cols = _build_valid_mock_insp().get_columns(tbl)
        if tbl == "event_scenarios":
            for c in cols:
                if c["name"] == "starts_at":
                    c["nullable"] = True
        return cols
    mock_insp.get_columns.side_effect = bad_get_columns

    with pytest.raises(RuntimeError, match="Critical column starts_at in event_scenarios must be NOT NULL"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sqlalchemy.inspect", lambda _c: mock_insp)
            verify_event_reminder_chains_schema(mock_conn)


@pytest.mark.asyncio
async def test_verify_schema_fails_closed_on_missing_or_bad_foreign_key():
    """Fail-closed verifier must raise RuntimeError if FK is missing or has invalid ondelete action."""
    mock_conn = MagicMock()
    mock_insp = _build_valid_mock_insp()
    # 1. Missing FK
    mock_insp.get_foreign_keys.side_effect = lambda tbl: [] if tbl == "event_steps" else _build_valid_mock_insp().get_foreign_keys(tbl)
    with pytest.raises(RuntimeError, match="Critical foreign key missing: event_steps.*event_id"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sqlalchemy.inspect", lambda _c: mock_insp)
            verify_event_reminder_chains_schema(mock_conn)

    # 2. Bad ondelete (CASCADE instead of RESTRICT)
    mock_insp2 = _build_valid_mock_insp()
    def cascade_fks(tbl):
        fks = _build_valid_mock_insp().get_foreign_keys(tbl)
        if tbl == "event_steps":
            for fk in fks:
                fk["options"]["ondelete"] = "CASCADE"
        return fks
    mock_insp2.get_foreign_keys.side_effect = cascade_fks
    with pytest.raises(RuntimeError, match="has ondelete='CASCADE', expected 'RESTRICT'"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sqlalchemy.inspect", lambda _c: mock_insp2)
            verify_event_reminder_chains_schema(mock_conn)


@pytest.mark.asyncio
async def test_verify_schema_fails_closed_on_missing_check_constraint():
    """Fail-closed verifier must raise RuntimeError if any critical CheckConstraint is missing."""
    mock_conn = MagicMock()
    # 1. Missing ck_event_scenarios_location
    mock_insp = _build_valid_mock_insp()
    mock_insp.get_check_constraints.side_effect = lambda tbl: [
        c for c in _build_valid_mock_insp().get_check_constraints(tbl)
        if c["name"] != "ck_event_scenarios_location"
    ]
    with pytest.raises(RuntimeError, match="Critical check constraint ck_event_scenarios_location missing on event_scenarios"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sqlalchemy.inspect", lambda _c: mock_insp)
            verify_event_reminder_chains_schema(mock_conn)

    # 2. Missing ck_deep_link_namespace_kind
    mock_insp2 = _build_valid_mock_insp()
    mock_insp2.get_check_constraints.side_effect = lambda tbl: [
        c for c in _build_valid_mock_insp().get_check_constraints(tbl)
        if c["name"] != "ck_deep_link_namespace_kind"
    ]
    with pytest.raises(RuntimeError, match="Critical check constraint ck_deep_link_namespace_kind missing on deep_link_namespace"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sqlalchemy.inspect", lambda _c: mock_insp2)
            verify_event_reminder_chains_schema(mock_conn)


@pytest.mark.asyncio
async def test_verify_schema_fails_closed_on_unordered_composite_index():
    """Fail-closed verifier must raise RuntimeError if composite index column order is scrambled."""
    mock_conn = MagicMock()

    # 1. Scrambled poll index columns: (due_at, status, next_attempt_at) instead of (status, next_attempt_at, due_at)
    mock_insp = _build_valid_mock_insp()
    def scrambled_poll_indexes(tbl):
        idxs = _build_valid_mock_insp().get_indexes(tbl)
        if tbl == "event_deliveries":
            for idx in idxs:
                if idx["name"] == "ix_event_deliveries_poll":
                    idx["column_names"] = ["due_at", "status", "next_attempt_at"]
        return idxs
    mock_insp.get_indexes.side_effect = scrambled_poll_indexes

    with pytest.raises(RuntimeError, match="Critical index ix_event_deliveries_poll with ordered columns"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sqlalchemy.inspect", lambda _c: mock_insp)
            verify_event_reminder_chains_schema(mock_conn)

    # 2. Scrambled execution order index: (id, sort_order, offset_minutes, event_id)
    mock_insp2 = _build_valid_mock_insp()
    def scrambled_exec_indexes(tbl):
        idxs = _build_valid_mock_insp().get_indexes(tbl)
        if tbl == "event_steps":
            for idx in idxs:
                if idx["name"] == "ix_event_steps_execution_order":
                    idx["column_names"] = ["id", "sort_order", "offset_minutes", "event_id"]
        return idxs
    mock_insp2.get_indexes.side_effect = scrambled_exec_indexes

    with pytest.raises(RuntimeError, match="Critical index ix_event_steps_execution_order with ordered columns"):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sqlalchemy.inspect", lambda _c: mock_insp2)
            verify_event_reminder_chains_schema(mock_conn)


@pytest.mark.asyncio
async def test_event_scenario_and_version_invariants(async_engine_session):
    """Verifies webinar/meeting location invariants and content_version/step_version/generation invariants."""
    engine, session_factory = async_engine_session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        user_id = 5001
        user = User(id=user_id, first_name="Sergey")
        session.add(user)
        await session.commit()

        # 1. Webinar without link -> FAILS
        webinar_no_link = EventScenario(
            code="wb_no_link",
            event_type="webinar",
            title="Webinar without link",
            starts_at=datetime.utcnow() + timedelta(days=1),
            link=None,
            place=None,
        )
        session.add(webinar_no_link)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # 2. Webinar with empty/whitespace link -> FAILS
        webinar_empty_link = EventScenario(
            code="wb_empty_link",
            event_type="webinar",
            title="Webinar with empty link",
            starts_at=datetime.utcnow() + timedelta(days=1),
            link="   ",
            place=None,
        )
        session.add(webinar_empty_link)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # 3. Webinar with place -> FAILS
        webinar_with_place = EventScenario(
            code="wb_with_place",
            event_type="webinar",
            title="Webinar with place",
            starts_at=datetime.utcnow() + timedelta(days=1),
            link="https://zoom.us",
            place="Room 101",
        )
        session.add(webinar_with_place)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # 4. Valid webinar -> SUCCEEDS
        webinar_ok = EventScenario(
            code="wb_ok",
            event_type="webinar",
            title="Valid Webinar",
            starts_at=datetime.utcnow() + timedelta(days=1),
            link="https://zoom.us/valid",
            place=None,
        )
        session.add(webinar_ok)
        await session.flush()
        webinar_ok_id = webinar_ok.id
        await session.commit()

        # 5. Meeting without place -> FAILS
        meeting_no_place = EventScenario(
            code="mt_no_place",
            event_type="meeting",
            title="Meeting without place",
            starts_at=datetime.utcnow() + timedelta(days=1),
            place=None,
            link=None,
        )
        session.add(meeting_no_place)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # 6. Meeting with empty/whitespace place -> FAILS
        meeting_empty_place = EventScenario(
            code="mt_empty_place",
            event_type="meeting",
            title="Meeting with empty place",
            starts_at=datetime.utcnow() + timedelta(days=1),
            place="   ",
            link=None,
        )
        session.add(meeting_empty_place)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # 7. Meeting with valid place and link=None -> SUCCEEDS
        meeting_ok_no_link = EventScenario(
            code="mt_ok_no_link",
            event_type="meeting",
            title="Valid Meeting No Link",
            starts_at=datetime.utcnow() + timedelta(days=1),
            place="г. Москва, ул. Арбат, 10",
            link=None,
        )
        session.add(meeting_ok_no_link)
        await session.commit()

        # 8. Meeting with valid place and link -> SUCCEEDS
        meeting_ok_with_link = EventScenario(
            code="mt_ok_with_link",
            event_type="meeting",
            title="Valid Meeting With Link",
            starts_at=datetime.utcnow() + timedelta(days=1),
            place="г. Москва, ул. Арбат, 10",
            link="https://yandex.ru/maps/1",
        )
        session.add(meeting_ok_with_link)
        await session.commit()

        # 9. content_version < 1 -> FAILS
        bad_cv = EventScenario(
            code="ev_bad_cv",
            event_type="webinar",
            title="Bad content version",
            starts_at=datetime.utcnow() + timedelta(days=1),
            link="https://zoom.us",
            content_version=0,
        )
        session.add(bad_cv)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # 10. step_version < 1 -> FAILS
        bad_step = EventStep(
            event_id=webinar_ok_id,
            offset_minutes=-60,
            message_text="Step text",
            step_version=0,
        )
        session.add(bad_step)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # 11. generation < 1 -> FAILS
        bad_gen = EventRegistration(
            event_id=webinar_ok_id,
            user_id=user_id,
            generation=0,
        )
        session.add(bad_gen)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()


@pytest.mark.asyncio
async def test_event_delivery_structural_invariants(async_engine_session):
    """Verifies EventDelivery structural CHECK invariants at DB level."""
    engine, session_factory = async_engine_session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        user = User(id=6001, first_name="Elena")
        session.add(user)
        scenario = EventScenario(
            code="delivery_inv_scenario",
            event_type="webinar",
            title="Webinar",
            starts_at=datetime.utcnow() + timedelta(days=2),
            link="https://zoom.us",
        )
        session.add(scenario)
        await session.flush()

        step = EventStep(
            event_id=scenario.id,
            offset_minutes=-60,
            message_text="Reminder",
        )
        session.add(step)
        reg = EventRegistration(
            event_id=scenario.id,
            user_id=user.id,
        )
        session.add(reg)
        await session.flush()
        scenario_id = scenario.id
        user_id = user.id
        step_id = step.id
        reg_id = reg.id
        await session.commit()

        # 1. message_kind='scheduled_step' without step_id -> FAILS
        d_bad_sched = EventDelivery(
            event_id=scenario_id,
            user_id=user_id,
            registration_id=reg_id,
            step_id=None,
            message_kind="scheduled_step",
            delivery_key="bad:sched:no:step",
            due_at=datetime.utcnow(),
            next_attempt_at=datetime.utcnow(),
            platform="telegram",
        )
        session.add(d_bad_sched)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # 2. immediate kinds with step_id NOT NULL -> FAILS
        for imm_kind in ("immediate_confirmation", "immediate_live", "reschedule_notice"):
            d_bad_imm = EventDelivery(
                event_id=scenario_id,
                user_id=user_id,
                registration_id=reg_id,
                step_id=step_id,  # MUST be NULL!
                message_kind=imm_kind,
                delivery_key=f"bad:imm:{imm_kind}:has:step",
                due_at=datetime.utcnow(),
                next_attempt_at=datetime.utcnow(),
                platform="telegram",
            )
            session.add(d_bad_imm)
            with pytest.raises(IntegrityError):
                await session.commit()
            await session.rollback()

        # 3. Invalid platform -> FAILS
        d_bad_platform = EventDelivery(
            event_id=scenario_id,
            user_id=user_id,
            registration_id=reg_id,
            step_id=step_id,
            message_kind="scheduled_step",
            delivery_key="bad:platform",
            due_at=datetime.utcnow(),
            next_attempt_at=datetime.utcnow(),
            platform="vk",  # only telegram or max
        )
        session.add(d_bad_platform)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # 4. retry_index < 0 -> FAILS
        d_bad_retry = EventDelivery(
            event_id=scenario_id,
            user_id=user_id,
            registration_id=reg_id,
            step_id=step_id,
            retry_index=-1,
            message_kind="scheduled_step",
            delivery_key="bad:retry_idx",
            due_at=datetime.utcnow(),
            next_attempt_at=datetime.utcnow(),
            platform="telegram",
        )
        session.add(d_bad_retry)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # 5. registration_generation < 1 -> FAILS
        d_bad_gen = EventDelivery(
            event_id=scenario_id,
            user_id=user_id,
            registration_id=reg_id,
            step_id=step_id,
            registration_generation=0,
            message_kind="scheduled_step",
            delivery_key="bad:reg_gen",
            due_at=datetime.utcnow(),
            next_attempt_at=datetime.utcnow(),
            platform="telegram",
        )
        session.add(d_bad_gen)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        # 6. attempt_count < 0 -> FAILS
        d_bad_att = EventDelivery(
            event_id=scenario_id,
            user_id=user_id,
            registration_id=reg_id,
            step_id=step_id,
            attempt_count=-1,
            message_kind="scheduled_step",
            delivery_key="bad:attempt_count",
            due_at=datetime.utcnow(),
            next_attempt_at=datetime.utcnow(),
            platform="telegram",
        )
        session.add(d_bad_att)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()


@pytest.mark.asyncio
async def test_manual_force_retry_scheduled_parent_idempotent(async_engine_session):
    """Verifies manual force_retry on scheduled delivery is idempotent and enforces single retry child."""
    engine, session_factory = async_engine_session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        user = User(id=7001, first_name="Maxim")
        session.add(user)
        scenario = EventScenario(
            code="retry_sched_test",
            event_type="webinar",
            title="Webinar",
            starts_at=datetime.utcnow() + timedelta(days=2),
            link="https://zoom.us",
        )
        session.add(scenario)
        await session.flush()
        step = EventStep(
            event_id=scenario.id,
            offset_minutes=-120,
            message_text="2h reminder",
        )
        session.add(step)
        reg = EventRegistration(event_id=scenario.id, user_id=user.id)
        session.add(reg)
        await session.flush()

        # Scheduled delivery (retry_index=0) that ended up uncertain
        parent_delivery = EventDelivery(
            event_id=scenario.id,
            user_id=user.id,
            registration_id=reg.id,
            step_id=step.id,
            retry_index=0,
            parent_delivery_id=None,
            message_kind="scheduled_step",
            delivery_key=f"ev:{scenario.id}:reg:{reg.id}:step:{step.id}:r0",
            due_at=datetime.utcnow(),
            next_attempt_at=datetime.utcnow(),
            platform="telegram",
            status="uncertain",
        )
        session.add(parent_delivery)
        await session.commit()
        parent_id = parent_delivery.id

        # 1. First force_retry creates child (created=True, retry_index=1, parent_delivery_id=parent.id)
        child1, created1 = await create_event_delivery_force_retry(
            session, parent_id, admin_id=888, resolution_reason="Admin manual retry"
        )
        await session.commit()
        assert created1 is True
        assert child1.id is not None
        assert child1.retry_index == 1
        assert child1.parent_delivery_id == parent_id
        assert child1.step_id == step.id
        assert child1.status == "pending"

        # 2. Repeated force_retry on the same parent is idempotently rejected (returns child1, created=False)
        child2, created2 = await create_event_delivery_force_retry(
            session, parent_id, admin_id=888, resolution_reason="Repeated retry request"
        )
        assert created2 is False
        assert child2.id == child1.id

        # 3. Repeated force_retry with raise_on_duplicate=True raises DeliveryRetryError
        with pytest.raises(DeliveryRetryError, match="already has a retry-child"):
            await create_event_delivery_force_retry(
                session, parent_id, admin_id=888, resolution_reason="Repeated retry with raise",
                raise_on_duplicate=True
            )

        # 4. DB guarantee: direct insert of duplicate child with same parent_delivery_id fails via partial unique index
        child_dup = EventDelivery(
            event_id=scenario.id,
            user_id=user.id,
            registration_id=reg.id,
            step_id=step.id,
            retry_index=2,
            parent_delivery_id=parent_id,  # duplicate parent reference!
            message_kind="scheduled_step",
            delivery_key=f"ev:{scenario.id}:reg:{reg.id}:step:{step.id}:manual_dup",
            due_at=datetime.utcnow(),
            next_attempt_at=datetime.utcnow(),
            platform="telegram",
            status="pending",
        )
        session.add(child_dup)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()


@pytest.mark.asyncio
async def test_manual_force_retry_immediate_parent_idempotent(async_engine_session):
    """Verifies manual force_retry on immediate delivery (step_id IS NULL) is idempotent and DB-enforced."""
    engine, session_factory = async_engine_session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        user = User(id=8001, first_name="Tatyana")
        session.add(user)
        scenario = EventScenario(
            code="retry_imm_test",
            event_type="webinar",
            title="Webinar",
            starts_at=datetime.utcnow() + timedelta(days=2),
            link="https://zoom.us",
        )
        session.add(scenario)
        await session.flush()
        reg = EventRegistration(event_id=scenario.id, user_id=user.id)
        session.add(reg)
        await session.flush()

        # Immediate confirmation delivery (step_id IS NULL) that ended up uncertain
        parent_delivery = EventDelivery(
            event_id=scenario.id,
            user_id=user.id,
            registration_id=reg.id,
            step_id=None,
            retry_index=0,
            parent_delivery_id=None,
            message_kind="immediate_confirmation",
            delivery_key=f"ev:{scenario.id}:reg:{reg.id}:kind:imm_conf:r0",
            due_at=datetime.utcnow(),
            next_attempt_at=datetime.utcnow(),
            platform="telegram",
            status="uncertain",
        )
        session.add(parent_delivery)
        await session.commit()
        parent_id = parent_delivery.id

        # 1. First force_retry creates immediate child
        child1, created1 = await create_event_delivery_force_retry(
            session, parent_id, admin_id=999, resolution_reason="Manual retry immediate"
        )
        await session.commit()
        assert created1 is True
        assert child1.retry_index == 1
        assert child1.parent_delivery_id == parent_id
        assert child1.step_id is None
        assert child1.message_kind == "immediate_confirmation"

        # 2. Repeated force_retry returns existing child idempotently
        child2, created2 = await create_event_delivery_force_retry(
            session, parent_id, admin_id=999, resolution_reason="Repeated immediate retry"
        )
        assert created2 is False
        assert child2.id == child1.id

        # 3. DB guarantee: direct insert of duplicate child with same parent_delivery_id fails even with step_id IS NULL
        child_dup = EventDelivery(
            event_id=scenario.id,
            user_id=user.id,
            registration_id=reg.id,
            step_id=None,
            retry_index=2,
            parent_delivery_id=parent_id,
            message_kind="immediate_confirmation",
            delivery_key=f"ev:{scenario.id}:reg:{reg.id}:kind:imm_conf:manual_dup",
            due_at=datetime.utcnow(),
            next_attempt_at=datetime.utcnow(),
            platform="telegram",
            status="pending",
        )
        session.add(child_dup)
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()


@pytest.mark.asyncio
async def test_manual_force_retry_chained_uncertain_child(async_engine_session):
    """Verifies that if a retry-child itself becomes uncertain, subsequent retry is created from child and chains."""
    engine, session_factory = async_engine_session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        user = User(id=9001, first_name="Konstantin")
        session.add(user)
        scenario = EventScenario(
            code="retry_chain_test",
            event_type="webinar",
            title="Webinar",
            starts_at=datetime.utcnow() + timedelta(days=2),
            link="https://zoom.us",
        )
        session.add(scenario)
        await session.flush()
        step = EventStep(event_id=scenario.id, offset_minutes=-15, message_text="15m reminder")
        session.add(step)
        reg = EventRegistration(event_id=scenario.id, user_id=user.id)
        session.add(reg)
        await session.flush()

        # Step 1: Root delivery fails and becomes uncertain (retry_index=0)
        root = EventDelivery(
            event_id=scenario.id,
            user_id=user.id,
            registration_id=reg.id,
            step_id=step.id,
            retry_index=0,
            parent_delivery_id=None,
            message_kind="scheduled_step",
            delivery_key=f"ev:{scenario.id}:reg:{reg.id}:step:{step.id}:r0",
            due_at=datetime.utcnow(),
            next_attempt_at=datetime.utcnow(),
            platform="telegram",
            status="uncertain",
        )
        session.add(root)
        await session.commit()

        # Step 2: Admin retries root -> creates Child (retry_index=1, parent=root.id)
        child, c_created = await create_event_delivery_force_retry(
            session, root.id, admin_id=111, resolution_reason="First manual retry"
        )
        await session.commit()
        assert c_created is True
        assert child.retry_index == 1
        assert child.parent_delivery_id == root.id

        # Step 3: Child delivery also encounters uncertain network failure
        child.status = "uncertain"
        child.last_error = "Second network failure"
        await session.commit()

        # Step 4: Admin retries from the child -> creates Grandchild (retry_index=2, parent=child.id)
        grandchild, gc_created = await create_event_delivery_force_retry(
            session, child.id, admin_id=111, resolution_reason="Second manual retry after child uncertain"
        )
        await session.commit()
        assert gc_created is True
        assert grandchild.retry_index == 2
        assert grandchild.parent_delivery_id == child.id
        assert grandchild.step_id == step.id
        assert grandchild.status == "pending"

        # Step 5: Verify the full chain in DB
        chain = (await session.execute(
            select(EventDelivery).where(EventDelivery.registration_id == reg.id).order_by(EventDelivery.retry_index)
        )).scalars().all()
        assert len(chain) == 3
        # Root (r0)
        assert chain[0].retry_index == 0
        assert chain[0].parent_delivery_id is None
        assert chain[0].status == "uncertain"
        # Child (r1)
        assert chain[1].retry_index == 1
        assert chain[1].parent_delivery_id == chain[0].id
        assert chain[1].status == "uncertain"
        # Grandchild (r2)
        assert chain[2].retry_index == 2
        assert chain[2].parent_delivery_id == chain[1].id
        assert chain[2].status == "pending"


# --- Manual Retry Contract Tests (Revision 11) ---

@pytest.mark.asyncio
async def test_manual_force_retry_preserves_expires_at(async_engine_session):
    """
    Verifies that create_event_delivery_force_retry strictly preserves parent.expires_at.
    Essential for immediate_live, immediate_confirmation, and reschedule_notice so that
    retried deliveries do not lose their business expiration window.
    """
    engine, session_factory = async_engine_session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        user = User(id=9101, first_name="Dmitry")
        session.add(user)
        scenario = EventScenario(
            code="retry_expires_test",
            event_type="webinar",
            title="Webinar",
            starts_at=datetime.utcnow() + timedelta(days=1),
            link="https://zoom.us",
        )
        session.add(scenario)
        await session.flush()
        reg = EventRegistration(event_id=scenario.id, user_id=user.id)
        session.add(reg)
        await session.flush()

        # 1. immediate_live with non-null expires_at
        expected_expiry_live = datetime.utcnow() + timedelta(hours=2)
        live_del = EventDelivery(
            event_id=scenario.id,
            user_id=user.id,
            registration_id=reg.id,
            step_id=None,
            message_kind="immediate_live",
            delivery_key=f"ev:{scenario.id}:reg:{reg.id}:live:r0",
            due_at=datetime.utcnow(),
            next_attempt_at=datetime.utcnow(),
            expires_at=expected_expiry_live,
            platform="telegram",
            status="uncertain",
        )
        session.add(live_del)
        await session.commit()

        live_child, created_live = await create_event_delivery_force_retry(
            session, live_del.id, admin_id=10, resolution_reason="Retry live broadcast reminder"
        )
        await session.commit()
        assert created_live is True
        assert live_child.expires_at is not None
        assert live_child.expires_at == expected_expiry_live

        # 2. immediate_confirmation with non-null expires_at
        expected_expiry_conf = datetime.utcnow() + timedelta(hours=1)
        conf_del = EventDelivery(
            event_id=scenario.id,
            user_id=user.id,
            registration_id=reg.id,
            step_id=None,
            message_kind="immediate_confirmation",
            delivery_key=f"ev:{scenario.id}:reg:{reg.id}:conf:r0",
            due_at=datetime.utcnow(),
            next_attempt_at=datetime.utcnow(),
            expires_at=expected_expiry_conf,
            platform="telegram",
            status="uncertain",
        )
        session.add(conf_del)
        await session.commit()

        conf_child, created_conf = await create_event_delivery_force_retry(
            session, conf_del.id, admin_id=10, resolution_reason="Retry confirmation"
        )
        await session.commit()
        assert created_conf is True
        assert conf_child.expires_at is not None
        assert conf_child.expires_at == expected_expiry_conf


@pytest.mark.asyncio
async def test_manual_force_retry_rejects_failed_status(async_engine_session):
    """
    Per Revision 11, 'failed' is strictly a terminal status.
    Manual force_retry is exclusively permitted for 'uncertain'.
    Attempting force_retry on a 'failed' delivery must be rejected and must NOT
    create a child delivery or an EventDeliveryResolution record.
    """
    engine, session_factory = async_engine_session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        user = User(id=9201, first_name="Anna")
        session.add(user)
        scenario = EventScenario(
            code="retry_failed_rejection_test",
            event_type="webinar",
            title="Webinar",
            starts_at=datetime.utcnow() + timedelta(days=1),
            link="https://zoom.us",
        )
        session.add(scenario)
        await session.flush()
        reg = EventRegistration(event_id=scenario.id, user_id=user.id)
        session.add(reg)
        await session.flush()

        failed_del = EventDelivery(
            event_id=scenario.id,
            user_id=user.id,
            registration_id=reg.id,
            step_id=None,
            message_kind="immediate_confirmation",
            delivery_key=f"ev:{scenario.id}:reg:{reg.id}:failed:r0",
            due_at=datetime.utcnow(),
            next_attempt_at=datetime.utcnow(),
            platform="telegram",
            status="failed",
            last_error="Permanent rejection: user blocked bot",
        )
        session.add(failed_del)
        await session.commit()
        failed_id = failed_del.id

        # Attempting force_retry must raise ValueError
        with pytest.raises(ValueError, match="cannot be force-retried.*expected 'uncertain'"):
            await create_event_delivery_force_retry(
                session, failed_id, admin_id=10, resolution_reason="Invalid retry of failed delivery"
            )

        # Assert no child delivery was created
        child = (await session.execute(
            select(EventDelivery).where(EventDelivery.parent_delivery_id == failed_id)
        )).scalars().first()
        assert child is None

        # Assert no EventDeliveryResolution was created
        resolution = (await session.execute(
            select(EventDeliveryResolution).where(EventDeliveryResolution.delivery_id == failed_id)
        )).scalars().first()
        assert resolution is None


# --- Real PostgreSQL Integration Tests (Protected Destructive Suite) ---

def validate_safe_test_db_name(db_name: str | None) -> str:
    """
    Validates that a database name is explicitly disposable/test.
    Rejects any database that does not start with 'test_' or contains prod/staging/live markers.
    """
    if not db_name or not db_name.strip():
        raise ValueError("SAFETY ABORT: Database name is empty or not provided.")
    lowered = db_name.strip().lower()
    if not lowered.startswith("test_"):
        raise ValueError(
            f"SAFETY ABORT: Refusing destructive test on database '{db_name}'. "
            "Database name must strictly start with 'test_' prefix (e.g. 'test_tgpsychobot_events')."
        )
    for forbidden in ("prod", "stage", "staging", "live", "main", "master"):
        if forbidden in lowered:
            raise ValueError(
                f"SAFETY ABORT: Refusing destructive test on database '{db_name}'. "
                f"Name contains forbidden keyword '{forbidden}'."
            )
    return lowered


def test_validate_safe_test_db_name_safety_guard():
    """Unit test: verifies that disposable database validator rejects dangerous/unintended DB names."""
    assert validate_safe_test_db_name("test_tgpsychobot_events") == "test_tgpsychobot_events"
    assert validate_safe_test_db_name("test_db") == "test_db"
    assert validate_safe_test_db_name("test_temp_ci_123") == "test_temp_ci_123"

    for unsafe in (
        None,
        "",
        "   ",
        "psychobot",
        "tgpsychobot",
        "postgres",
        "production",
        "test_prod",
        "test_production",
        "test_staging",
        "test_stage",
        "test_live",
        "test_master",
        "test_main",
    ):
        with pytest.raises(ValueError, match="SAFETY ABORT"):
            validate_safe_test_db_name(unsafe)


async def _get_safe_postgres_engine():
    """
    Returns (engine, None) ONLY if POSTGRES_TEST_DATABASE_URL is explicitly set and points
    to a strictly validated disposable test database.
    Otherwise returns (None, skip_reason).
    """
    postgres_test_url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not postgres_test_url or not postgres_test_url.strip():
        return None, (
            "POSTGRES_TEST_DATABASE_URL environment variable is not explicitly set; "
            "skipping real destructive PostgreSQL integration tests. "
            "Destructive PostgreSQL tests require an explicit POSTGRES_TEST_DATABASE_URL."
        )

    url_str = postgres_test_url.strip()
    try:
        parsed_url = make_url(url_str)
        validate_safe_test_db_name(parsed_url.database)
    except ValueError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"SAFETY ABORT: Invalid POSTGRES_TEST_DATABASE_URL: {exc}"

    try:
        engine = create_async_engine(url_str)
        async with engine.connect() as conn:
            actual_db = (await conn.execute(text("SELECT current_database()"))).scalar()
            validate_safe_test_db_name(actual_db)
        return engine, None
    except Exception as exc:
        return None, f"Failed to connect to validated PostgreSQL test database: {exc}"


@pytest.mark.asyncio
async def test_real_postgresql_init_and_schema_verification():
    """Real PostgreSQL integration test: verifies init_db and fail-closed schema verification on PostgreSQL."""
    pg_engine, skip_reason = await _get_safe_postgres_engine()
    if pg_engine is None:
        pytest.skip(skip_reason)

    try:
        # SAFETY WARNING: DROP SCHEMA public CASCADE is strictly destructive and is allowed
        # ONLY because we verified that POSTGRES_TEST_DATABASE_URL was explicitly set,
        # current_database() strictly starts with 'test_', and contains no production/staging keywords.
        async with pg_engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA public CASCADE;"))
            await conn.execute(text("CREATE SCHEMA public;"))

        sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(database, "engine", pg_engine)
            mp.setattr(database, "async_session_maker", sessions)

            await database.init_db()

            async with pg_engine.connect() as conn:
                await conn.run_sync(verify_event_reminder_chains_schema)
    finally:
        await pg_engine.dispose()


@pytest.mark.asyncio
async def test_real_postgresql_structural_invariants_and_retry_model():
    """Real PostgreSQL integration test: verifies CHECK constraints, partial unique index, and retry chaining on PostgreSQL."""
    pg_engine, skip_reason = await _get_safe_postgres_engine()
    if pg_engine is None:
        pytest.skip(skip_reason)

    try:
        # SAFETY WARNING: DROP SCHEMA public CASCADE is strictly destructive and is allowed
        # ONLY because we verified that POSTGRES_TEST_DATABASE_URL was explicitly set,
        # current_database() strictly starts with 'test_', and contains no production/staging keywords.
        async with pg_engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA public CASCADE;"))
            await conn.execute(text("CREATE SCHEMA public;"))
            await conn.run_sync(Base.metadata.create_all)

        sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
        async with sessions() as session:
            user = User(id=9901, first_name="PG User")
            session.add(user)

            # 1. Webinar location check on real PostgreSQL: webinar without link FAILS
            bad_wb = EventScenario(
                code="pg_bad_wb",
                event_type="webinar",
                title="PG Bad Webinar",
                starts_at=datetime.utcnow() + timedelta(days=1),
                link=None,
                place=None,
            )
            session.add(bad_wb)
            with pytest.raises(IntegrityError):
                await session.commit()
            await session.rollback()

            # 2. Valid webinar and meeting on real PostgreSQL
            wb = EventScenario(
                code="pg_good_wb",
                event_type="webinar",
                title="PG Good Webinar",
                starts_at=datetime.utcnow() + timedelta(days=1),
                link="https://zoom.us/pg",
                place=None,
            )
            session.add(wb)
            mt = EventScenario(
                code="pg_good_mt",
                event_type="meeting",
                title="PG Good Meeting",
                starts_at=datetime.utcnow() + timedelta(days=1),
                place="Москва, Офис",
                link=None,
            )
            session.add(mt)
            user_fresh = User(id=9902, first_name="PG User 2")
            session.add(user_fresh)
            await session.flush()
            wb_id = wb.id
            user_fresh_id = user_fresh.id
            await session.commit()

            # 3. Delivery invariants on PostgreSQL: scheduled without step_id FAILS
            reg = EventRegistration(event_id=wb_id, user_id=user_fresh_id)
            session.add(reg)
            await session.commit()
            reg_id = reg.id

            bad_del = EventDelivery(
                event_id=wb_id,
                user_id=user_fresh_id,
                registration_id=reg_id,
                step_id=None,
                message_kind="scheduled_step",
                delivery_key="pg:bad:sched",
                due_at=datetime.utcnow(),
                next_attempt_at=datetime.utcnow(),
                platform="telegram",
            )
            session.add(bad_del)
            with pytest.raises(IntegrityError):
                await session.commit()
            await session.rollback()

            # 4. Immediate delivery and retry model on real PostgreSQL
            imm_del = EventDelivery(
                event_id=wb_id,
                user_id=user_fresh_id,
                registration_id=reg_id,
                step_id=None,
                message_kind="immediate_confirmation",
                delivery_key="pg:imm:r0",
                due_at=datetime.utcnow(),
                next_attempt_at=datetime.utcnow(),
                platform="telegram",
                status="uncertain",
            )
            session.add(imm_del)
            await session.commit()
            imm_del_id = imm_del.id

            # First retry
            child, created = await create_event_delivery_force_retry(
                session, imm_del_id, admin_id=1, resolution_reason="PG retry"
            )
            await session.commit()
            assert created is True
            assert child.retry_index == 1
            assert child.parent_delivery_id == imm_del_id
            child_id = child.id

            # Idempotent rejection of second retry
            child2, created2 = await create_event_delivery_force_retry(
                session, imm_del_id, admin_id=1, resolution_reason="PG second retry"
            )
            assert created2 is False
            assert child2.id == child_id

            # Child itself becomes uncertain -> retry created from child
            child_db = await session.get(EventDelivery, child_id)
            child_db.status = "uncertain"
            child_db.last_error = "PG network timeout"
            await session.commit()

            grandchild, gc_created = await create_event_delivery_force_retry(
                session, child_id, admin_id=1, resolution_reason="PG retry grandchild"
            )
            await session.commit()
            assert gc_created is True
            assert grandchild.retry_index == 2
            assert grandchild.parent_delivery_id == child_id

            # PostgreSQL partial unique index enforcement:
            # direct insert with same parent_delivery_id fails
            child_dup = EventDelivery(
                event_id=wb_id,
                user_id=user_fresh_id,
                registration_id=reg_id,
                step_id=None,
                retry_index=3,
                parent_delivery_id=imm_del_id,
                message_kind="immediate_confirmation",
                delivery_key="pg:imm:manual_dup",
                due_at=datetime.utcnow(),
                next_attempt_at=datetime.utcnow(),
                platform="telegram",
                status="pending",
            )
            session.add(child_dup)
            with pytest.raises(IntegrityError):
                await session.commit()
            await session.rollback()
    finally:
        await pg_engine.dispose()


@pytest.mark.asyncio
async def test_real_postgresql_update_step_vs_reschedule_concurrency():
    """
    Defect 3 concurrency test on real PostgreSQL (disposable test DB guard):
    Verifies that update_step and reschedule_scenario respect lock hierarchy:
    Scenario -> Registration -> Step -> Delivery
    and do not deadlock under concurrent execution.
    """
    import asyncio
    from sqlalchemy.ext.asyncio import AsyncSession
    from event_admin_service import add_step, create_scenario, reschedule_scenario, update_step

    pg_engine, skip_reason = await _get_safe_postgres_engine()
    if pg_engine is None:
        pytest.skip(skip_reason)

    try:
        async with pg_engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
            await conn.run_sync(Base.metadata.create_all)

        session_maker = async_sessionmaker(pg_engine, expire_on_commit=False, class_=AsyncSession)

        base_now = datetime.utcnow()
        starts_at = base_now + timedelta(days=2)
        scen = await create_scenario(
            session_maker,
            code="pg_lock_test",
            event_type="webinar",
            title="PG Lock Order Webinar",
            starts_at_utc=starts_at,
            link="https://live.link",
        )
        step = await add_step(session_maker, scen.id, -120, "PG Reminder step")

        async with session_maker() as session:
            u = User(id=9991, first_name="PGUser")
            session.add(u)
            await session.flush()
            reg = EventRegistration(
                event_id=scen.id,
                user_id=9991,
                status="started",
                registration_mode="standard",
            )
            session.add(reg)
            await session.flush()
            deliv = EventDelivery(
                event_id=scen.id,
                user_id=9991,
                registration_id=reg.id,
                step_id=step.id,
                message_kind="scheduled_step",
                delivery_key=f"ev:{scen.id}:reg:{reg.id}:step:{step.id}:r0",
                due_at=starts_at - timedelta(hours=2),
                next_attempt_at=starts_at - timedelta(hours=2),
                schedule_revision=1,
                registration_generation=1,
                platform="telegram",
                status="pending",
            )
            session.add(deliv)
            await session.commit()

        # Concurrent execution of update_step and reschedule_scenario
        task_update = update_step(session_maker, step.id, offset_minutes=-60)
        task_resched = reschedule_scenario(
            session_maker,
            scen.id,
            new_starts_at_utc=starts_at + timedelta(days=1),
        )

        res_update, res_resched = await asyncio.wait_for(
            asyncio.gather(task_update, task_resched),
            timeout=10.0,
        )

        assert res_update is not None
        assert res_resched is not None
    finally:
        await pg_engine.dispose()


@pytest.mark.asyncio
async def test_real_postgresql_claimed_update_step_vs_presend_barrier():
    """
    Concurrency barrier test on real PostgreSQL (disposable test DB guard):
    - poller claim of due delivery
    - concurrent race: admin update_step (new future offset) vs worker prepare_and_lock_delivery_for_sending (pre-send / PoNR)
    - Valid outcomes:
      1) Admin wins before PoNR:
         worker pre-send rejected (ok=False), delivery in DB is rescheduled to 'pending' with new due_at and claim cleared
      2) Worker wins PoNR:
         pre-send succeeds (ok=True, status='sending'), admin update_step does not mutate sending delivery
    - Invariant: delivery is NEVER left in an inconsistent state or sent with stale schedule after admin update_step commit.
    """
    import asyncio
    from sqlalchemy.ext.asyncio import AsyncSession
    from event_admin_service import add_step, create_scenario, toggle_scenario_active, update_step
    from event_delivery_worker import claim_pending_deliveries
    from event_service import prepare_and_lock_delivery_for_sending

    pg_engine, skip_reason = await _get_safe_postgres_engine()
    if pg_engine is None:
        pytest.skip(skip_reason)

    try:
        async with pg_engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public;"))
            await conn.run_sync(Base.metadata.create_all)

        session_maker = async_sessionmaker(pg_engine, expire_on_commit=False, class_=AsyncSession)

        base_now = datetime.utcnow()
        starts_at = base_now + timedelta(hours=2)
        scen = await create_scenario(
            session_maker,
            code="pg_barrier_test",
            event_type="webinar",
            title="PG Concurrency Barrier Webinar",
            starts_at_utc=starts_at,
            link="https://live.link",
            now_utc=base_now,
        )
        await toggle_scenario_active(session_maker, scen.id, now_utc=base_now)

        # Step originally at -120 min (due right at base_now)
        step = await add_step(session_maker, scen.id, -120, "Barrier reminder step", now_utc=base_now - timedelta(minutes=5))

        async with session_maker() as session:
            u = User(id=9992, first_name="PGUserBarrier")
            session.add(u)
            await session.flush()
            reg = EventRegistration(
                event_id=scen.id,
                user_id=9992,
                status="started",
                registration_mode="standard",
                registered_at=base_now - timedelta(hours=1),
            )
            session.add(reg)
            await session.flush()
            deliv = EventDelivery(
                event_id=scen.id,
                user_id=9992,
                registration_id=reg.id,
                step_id=step.id,
                message_kind="scheduled_step",
                delivery_key=f"ev:{scen.id}:reg:{reg.id}:step:{step.id}:r0",
                due_at=base_now,
                next_attempt_at=base_now,
                schedule_revision=1,
                registration_generation=1,
                platform="telegram",
                status="pending",
            )
            session.add(deliv)
            await session.commit()
            deliv_id = deliv.id

        # 1. Poller claim
        claimed = await claim_pending_deliveries(session_maker, batch_size=10, now=base_now)
        assert len(claimed) == 1
        assert claimed[0][0] == deliv_id
        token = claimed[0][1]

        # 2. Race: admin update_step (-60 min -> new due at starts_at - 1h = 13:00) vs worker pre-send
        new_due = starts_at - timedelta(minutes=60)
        async def run_admin():
            return await update_step(session_maker, step.id, offset_minutes=-60, now_utc=base_now)

        async def run_worker():
            return await prepare_and_lock_delivery_for_sending(
                session_maker,
                delivery_id=deliv_id,
                claim_token=token,
                expected_content_version=1,
                expected_step_version=1,
                rendered_text="Rendered reminder",
                template_text="Barrier reminder step",
                now=base_now,
            )

        admin_res, worker_res = await asyncio.gather(run_admin(), run_worker())
        ok, locked_deliv, reject_reason = worker_res

        async with session_maker() as session:
            final_deliv = await session.get(EventDelivery, deliv_id)

            if ok:
                # Worker won PoNR before admin update_step
                # Delivery was committed to 'sending' before admin lock, so admin did not mutate it
                assert final_deliv.status == "sending"
                assert final_deliv.attempt_count >= 1
            else:
                # Admin won before PoNR: worker pre-send was rejected
                # Delivery was safely rescheduled to 'pending' with new due_at and cleared claim fields
                assert final_deliv.status == "pending"
                assert final_deliv.due_at == new_due
                assert final_deliv.claim_token is None
                assert final_deliv.claimed_at is None
                assert final_deliv.lease_until is None
                assert final_deliv.send_started_at is None
                assert reject_reason in ("step_version_mismatch", "content_version_mismatch", "stale_or_invalid_claim")

            # Invariant: delivery is NEVER in an inconsistent claimed state with stale schedule
            assert final_deliv.status in ("pending", "sending")
    finally:
        await pg_engine.dispose()

