"""Comprehensive integration tests for unified database-backed AI incident management,
outbox delivery, companion routing, and Autobus infrastructure isolation.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from aiogram import Bot
from aiogram.methods import SendMessage
from aiogram.types import User as TgUser
from sqlalchemy import create_engine, select, text
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.schema import CreateIndex, CreateTable

import monitor_psychobots as monitor
import scheduler
from ai_incident_manager import (
    DEFAULT_REPEAT_ALERT_SECONDS,
    _format_incident_message_text,
    check_incident_repeat_reminders,
    get_provider_account_fingerprint,
    handle_key_rotation,
    process_ai_incident_notifications,
    record_ai_operation_success,
    record_ai_provider_failure,
)
from database import (
    AIConfig,
    AIIncidentNotificationModel,
    AIProviderIncidentModel,
    Base,
    User,
)
from ai_integration import AIServiceError, InsufficientBalanceError
from provider_failures import ProviderFailure


@pytest_asyncio.fixture
async def test_db_session_maker(tmp_path):
    """Provide an isolated SQLite database session maker for integration tests."""
    db_file = tmp_path / "test_incident.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"
    engine = create_async_engine(db_url, echo=False)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed AIConfig
    async with session_maker() as session:
        ai_cfg = AIConfig(
            id=1,
            provider="Gemini",
            gemini_api_key="gem_test_key_1",
            image_generation_provider="KIE",
            kie_api_key="kie_secret_key_1",
            kie_credit_alert_threshold=2.0,
            kie_credit_alert_sent=False,
            allow_fallback=True,
            fallback_provider="OpenAI",
            openai_api_key="openai_test_key_1",
        )
        session.add(ai_cfg)
        await session.commit()

    yield session_maker
    await engine.dispose()


def test_schema_ddl_compilation_sqlite_and_postgresql():
    """Verify DDL compilation and partial index definitions for SQLite and PostgreSQL."""
    sq_engine = create_engine("sqlite:///:memory:")
    pg_dialect = postgresql.dialect()

    # AIProviderIncidentModel DDL
    inc_table = AIProviderIncidentModel.__table__
    inc_ddl_sqlite = str(CreateTable(inc_table).compile(sq_engine))
    inc_ddl_pg = str(CreateTable(inc_table).compile(dialect=pg_dialect))
    assert "ai_provider_incidents" in inc_ddl_sqlite
    assert "ai_provider_incidents" in inc_ddl_pg

    # Check partial unique index on PostgreSQL
    for idx in inc_table.indexes:
        if idx.name == "uq_active_ai_incident":
            compiled_pg = str(CreateIndex(idx).compile(dialect=pg_dialect))
            assert "WHERE status = 'active'" in compiled_pg
            assert "UNIQUE" in compiled_pg

    # AIIncidentNotificationModel DDL
    notif_table = AIIncidentNotificationModel.__table__
    notif_ddl_sqlite = str(CreateTable(notif_table).compile(sq_engine))
    notif_ddl_pg = str(CreateTable(notif_table).compile(dialect=pg_dialect))
    assert "ai_incident_notifications" in notif_ddl_sqlite
    assert "ai_incident_notifications" in notif_ddl_pg


@pytest.mark.asyncio
async def test_scenario_1_scheduler_soft_warning_low_balance(test_db_session_maker, monkeypatch):
    """Scenario 1: Scheduler balance check triggers soft warning via durable outbox and resolves on replenishment."""
    from aiogram.types import User
    inst_def = {"pm2_name": "tg_someonewithyou01_bot_new", "platform": "telegram", "platform_id": 8281390498, "admins": [1001]}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[1001]))

    bot = AsyncMock(spec=Bot)
    bot.token = "8281390498:AAFtest_token"
    bot.get_me = AsyncMock(return_value=User(id=8281390498, is_bot=True, first_name="bot", username="someonewithyou01_bot"))
    bot.send_message = AsyncMock(return_value=True)

    monkeypatch.setattr(scheduler, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr(scheduler, "get_kie_remaining_credits", AsyncMock(return_value=1.50))
    monkeypatch.setattr(scheduler, "get_all_admin_ids", AsyncMock(return_value=[1001]))

    # Seed AIConfig with threshold=2.0
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="KIE", kie_api_key="test_kie_key_1", kie_credit_alert_threshold=2.0, kie_credit_alert_sent=False)
            session.add(cfg)
        else:
            cfg.kie_credit_alert_threshold = 2.0
            cfg.kie_credit_alert_sent = False
        await session.commit()

    # 1. Run check: balance 1.50 < threshold 2.0
    await scheduler.check_kie_credit_balance(bot)

    # Soft warning was sent to admin 1001
    assert bot.send_message.call_count == 1
    call_kwargs = bot.send_message.call_args.kwargs
    call_args = bot.send_message.call_args.args
    recipient = call_kwargs.get("chat_id") or (call_args[0] if call_args else None)
    text_sent = call_kwargs.get("text") or (call_args[1] if len(call_args) > 1 else "")
    assert recipient == 1001
    assert "Предупреждение KIE" in text_sent
    assert "1.50" in text_sent

    # Verify low_credit_warning incident opened in DB and notification delivered
    async with test_db_session_maker() as session:
        incidents = (await session.execute(select(AIProviderIncidentModel))).scalars().all()
        assert len(incidents) == 1
        assert incidents[0].incident_type == "low_credit_warning"
        assert incidents[0].status == "active"

        notifs = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        assert len(notifs) == 1
        assert notifs[0].status == "delivered"
        assert notifs[0].event_type == "opened"

        cfg_after = await session.get(AIConfig, 1)
        assert cfg_after.kie_credit_alert_sent is True

    # 2. Second check with same low balance does not re-send
    bot.send_message.reset_mock()
    await scheduler.check_kie_credit_balance(bot)
    assert bot.send_message.call_count == 0

    # 3. Third check with replenished balance (5.00 >= 2.00) resolves incident and sends replenishment notice
    monkeypatch.setattr(scheduler, "get_kie_remaining_credits", AsyncMock(return_value=5.00))
    await scheduler.check_kie_credit_balance(bot)
    assert bot.send_message.call_count == 1
    call_kwargs = bot.send_message.call_args.kwargs
    call_args = bot.send_message.call_args.args
    text_sent_2 = call_kwargs.get("text") or (call_args[1] if len(call_args) > 1 else "")
    assert "Баланс KIE пополнен" in text_sent_2
    assert "5.00" in text_sent_2

    async with test_db_session_maker() as session:
        incidents = (await session.execute(select(AIProviderIncidentModel))).scalars().all()
        assert len(incidents) == 1
        assert incidents[0].status == "resolved"

        cfg_final = await session.get(AIConfig, 1)
        assert cfg_final.kie_credit_alert_sent is False


@pytest.mark.asyncio
async def test_scenario_2_provider_failure_opens_db_incident_and_enqueues_outbox(test_db_session_maker, monkeypatch):
    """Scenario 2: Provider failure opens active incident in DB and creates outbox notification rows."""
    inst_def = {"pm2_name": "tg_someonewithyou01_bot_new", "platform": "telegram", "admins": [272982544]}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda i: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[272982544]))

    err = InsufficientBalanceError("KIE credits exhausted: payment required")
    err.provider = "KIE"
    err.http_status = 402
    err.failure = ProviderFailure(
        provider="KIE",
        operation="image_generation",
        classification="insufficient_balance_quota",
        detail_code="credits_exhausted",
        http_status=402,
        operator_action_required=True,
    )

    inc = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=err,
        operation="image_generation",
        model="seedream-v4",
        session_maker=test_db_session_maker,
    )

    assert inc is not None
    assert inc.status == "active"
    assert inc.occurrence_count == 1
    assert inc.provider == "KIE"
    assert inc.operation == "image_generation"

    # Verify outbox row created
    async with test_db_session_maker() as session:
        notifs = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        assert len(notifs) == 1
        n = notifs[0]
        assert n.recipient_id == 272982544
        assert n.status == "pending"
        assert n.event_type == "opened"
        assert n.authorized_instance_id == "tg_someonewithyou01_bot_new"
        assert "Сбой AI-провайдера" in n.message_text
        assert "KIE" in n.message_text
        assert "Требуется вмешательство оператора" in n.message_text


@pytest.mark.asyncio
async def test_scenario_3_shared_db_max_companion_isolation(test_db_session_maker, monkeypatch):
    """Scenario 3: Shared DB: MAX registers incident, Telegram companion delivers notification."""
    tg_inst = {"pm2_name": "tg_veraveda777_bot_legacy", "platform": "telegram", "admins": [777]}
    max_inst = {"pm2_name": "max_se13639182_bot_legacy", "platform": "max", "companion_pm2_name": "tg_veraveda777_bot_legacy"}

    def mock_get_inst(name):
        if name == "tg_veraveda777_bot_legacy":
            return tg_inst
        if name == "max_se13639182_bot_legacy":
            return max_inst
        return None

    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", mock_get_inst)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: tg_inst if s["platform"] == "max" else s)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[777]))

    # Failure occurs in MAX bot
    err = InsufficientBalanceError("KIE credits insufficient")
    err.provider = "KIE"
    err.failure = ProviderFailure(
        provider="KIE",
        operation="chat",
        classification="insufficient_balance_quota",
        detail_code="credits_exhausted",
        operator_action_required=True,
    )

    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: max_inst)
    inc = await record_ai_provider_failure(
        origin_instance_id="max_se13639182_bot_legacy",
        provider="KIE",
        error=err,
        operation="chat",
        model="gpt-4o",
        session_maker=test_db_session_maker,
    )
    assert inc is not None
    assert inc.origin_instance_id == "max_se13639182_bot_legacy"
    assert inc.companion_instance_id == "tg_veraveda777_bot_legacy"

    # Outbox notification was assigned to authorized_instance_id == 'tg_veraveda777_bot_legacy'
    async with test_db_session_maker() as session:
        notif = (await session.execute(select(AIIncidentNotificationModel))).scalar_one()
        assert notif.authorized_instance_id == "tg_veraveda777_bot_legacy"

    # 1. MAX worker runs -> claims 0 because it's not authorized
    bot = AsyncMock(spec=Bot)
    bot.send_message = AsyncMock(return_value=True)
    delivered_max = await process_ai_incident_notifications(bot=bot, session_maker=test_db_session_maker)
    assert delivered_max == 0
    assert bot.send_message.call_count == 0

    # 2. Telegram companion worker runs -> claims and delivers
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: tg_inst)
    delivered_tg = await process_ai_incident_notifications(bot=bot, session_maker=test_db_session_maker)
    assert delivered_tg == 1
    assert bot.send_message.call_count == 1
    recipient_called = bot.send_message.call_args.kwargs.get("chat_id") if bot.send_message.call_args.kwargs else bot.send_message.call_args.args[0]
    assert recipient_called == 777

    # Status updated to delivered
    async with test_db_session_maker() as session:
        notif_after = (await session.execute(select(AIIncidentNotificationModel))).scalar_one()
        assert notif_after.status == "delivered"


@pytest.mark.asyncio
async def test_scenario_4_concurrent_failure_recording_atomic(test_db_session_maker, monkeypatch):
    """Scenario 4: Concurrent failure recording atomic UPSERT and counter increment."""
    inst_def = {"pm2_name": "tg_bot_conc", "platform": "telegram", "admins": [888]}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[888]))

    err = InsufficientBalanceError("Quota exceeded")
    err.provider = "Gemini"
    err.failure = ProviderFailure(
        provider="Gemini",
        operation="chat",
        classification="insufficient_balance_quota",
        detail_code="quota_exceeded",
        operator_action_required=True,
    )

    # 5 concurrent tasks
    async def record():
        return await record_ai_provider_failure(
            origin_instance_id="tg_bot_conc",
            provider="Gemini",
            error=err,
            operation="chat",
            model="gemini-2.5-pro",
            session_maker=test_db_session_maker,
        )

    results = await asyncio.gather(*(record() for _ in range(5)))
    assert all(r is not None for r in results)

    # Verify exactly 1 active incident row exists
    async with test_db_session_maker() as session:
        incidents = (await session.execute(select(AIProviderIncidentModel))).scalars().all()
        assert len(incidents) == 1
        assert incidents[0].occurrence_count == 5

        # Only 1 notification enqueued for cycle 0
        notifs = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        assert len(notifs) == 1


@pytest.mark.asyncio
async def test_scenario_5_dynamic_admin_permissions_revocation(test_db_session_maker, monkeypatch):
    """Scenario 5: Pre-send dynamic permission check drops revoked admins as failed_terminal."""
    inst_def = {"pm2_name": "tg_bot_perm", "platform": "telegram", "admins": [999]}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[999]))

    err = InsufficientBalanceError("Out of balance")
    err.provider = "KIE"
    err.failure = ProviderFailure(provider="KIE", operation="chat", classification="insufficient_balance_quota", detail_code="credits_exhausted", operator_action_required=True)

    await record_ai_provider_failure(
        origin_instance_id="tg_bot_perm",
        provider="KIE",
        error=err,
        operation="chat",
        model="*",
        session_maker=test_db_session_maker,
    )

    # Now admin 999 is revoked before dispatch
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[]))

    bot = AsyncMock(spec=Bot)
    delivered = await process_ai_incident_notifications(bot=bot, session_maker=test_db_session_maker)
    assert delivered == 0
    assert bot.send_message.call_count == 0

    async with test_db_session_maker() as session:
        notif = (await session.execute(select(AIIncidentNotificationModel))).scalar_one()
        assert notif.status == "skipped_unauthorized"
        assert notif.last_error == "recipient_permission_revoked_or_invalid"


@pytest.mark.asyncio
async def test_scenario_6_cas_lease_backoff_and_terminal_error(test_db_session_maker, monkeypatch):
    """Scenario 6: CAS lease retry backoff on transient error, failed_terminal on blocked bot."""
    from aiogram.exceptions import TelegramForbiddenError

    inst_def = {"pm2_name": "tg_bot_lease", "platform": "telegram", "admins": [555]}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[555]))

    err = InsufficientBalanceError("Balance zero")
    err.provider = "KIE"
    err.failure = ProviderFailure(provider="KIE", operation="chat", classification="insufficient_balance_quota", detail_code="credits_exhausted", operator_action_required=True)

    await record_ai_provider_failure(
        origin_instance_id="tg_bot_lease",
        provider="KIE",
        error=err,
        operation="chat",
        session_maker=test_db_session_maker,
    )

    bot = AsyncMock(spec=Bot)
    # 1. Transient error on send
    bot.send_message = AsyncMock(side_effect=Exception("Connection reset by peer"))
    delivered = await process_ai_incident_notifications(bot=bot, session_maker=test_db_session_maker)
    assert delivered == 0

    async with test_db_session_maker() as session:
        notif = (await session.execute(select(AIIncidentNotificationModel))).scalar_one()
        assert notif.status == "pending"
        assert notif.attempts == 1
        assert notif.next_retry_at > datetime.utcnow()

        # Simulate time passing to retry
        notif.next_retry_at = datetime.utcnow() - timedelta(seconds=1)
        await session.commit()

    # 2. Terminal error (bot blocked)
    bot.send_message = AsyncMock(side_effect=TelegramForbiddenError(method=SendMessage(chat_id=555, text=""), message="bot was blocked by the user"))
    delivered = await process_ai_incident_notifications(bot=bot, session_maker=test_db_session_maker)
    assert delivered == 0

    async with test_db_session_maker() as session:
        notif = (await session.execute(select(AIIncidentNotificationModel))).scalar_one()
        assert notif.status == "failed_terminal"
        assert notif.attempts == 2


@pytest.mark.asyncio
async def test_scenario_7_key_rotation_supersedes_incidents_without_false_alert(test_db_session_maker, monkeypatch):
    """Scenario 7: Key rotation supersedes old incidents and cancels outbox without false replenishment alerts."""
    inst_def = {"pm2_name": "tg_bot_rot", "platform": "telegram", "admins": [333]}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[333]))

    err = InsufficientBalanceError("Balance zero")
    err.provider = "KIE"
    err.failure = ProviderFailure(provider="KIE", operation="chat", classification="insufficient_balance_quota", detail_code="credits_exhausted", operator_action_required=True)

    await record_ai_provider_failure(
        origin_instance_id="tg_bot_rot",
        provider="KIE",
        error=err,
        operation="chat",
        credential_scope="old_fingerprint",
        session_maker=test_db_session_maker,
    )

    # Admin rotates key to "new_fresh_key"
    superseded = await handle_key_rotation(
        origin_instance_id="tg_bot_rot",
        provider="KIE",
        new_key="new_fresh_key",
        session_maker=test_db_session_maker,
    )
    assert superseded == 1

    async with test_db_session_maker() as session:
        inc = (await session.execute(select(AIProviderIncidentModel))).scalar_one()
        assert inc.status == "superseded"

        notif = (await session.execute(select(AIIncidentNotificationModel))).scalar_one()
        assert notif.status == "cancelled"
        assert notif.last_error == "superseded_by_key_rotation"

        # No 'resolved' alerts were enqueued!
        all_notifs = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        assert all(n.event_type != "resolved" for n in all_notifs)


@pytest.mark.asyncio
async def test_scenario_8_operational_recovery_scope_isolation(test_db_session_maker, monkeypatch):
    """Scenario 8: Chat operational success does NOT close image_generation incident."""
    inst_def = {"pm2_name": "tg_bot_scope", "platform": "telegram", "admins": [444]}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[444]))

    fp = get_provider_account_fingerprint("kie_secret_key_1")

    # Image gen fails with quota error
    err = InsufficientBalanceError("KIE credits insufficient")
    err.provider = "KIE"
    err.failure = ProviderFailure(provider="KIE", operation="image_generation", classification="insufficient_balance_quota", detail_code="credits_exhausted", operator_action_required=True)
    await record_ai_provider_failure(
        origin_instance_id="tg_bot_scope",
        provider="KIE",
        error=err,
        operation="image_generation",
        model="seedream-v4",
        credential_scope=fp,
        session_maker=test_db_session_maker,
    )

    # Operational call succeeds on 'chat'
    resolved_chat = await record_ai_operation_success(
        origin_instance_id="tg_bot_scope",
        provider="KIE",
        credential_scope=fp,
        operation="chat",
        model="gpt-4o",
        session_maker=test_db_session_maker,
    )
    assert resolved_chat is False  # Did NOT resolve image_generation!

    async with test_db_session_maker() as session:
        inc = (await session.execute(select(AIProviderIncidentModel))).scalar_one()
        assert inc.status == "active"

    # Operational call succeeds on 'image_generation'
    resolved_img = await record_ai_operation_success(
        origin_instance_id="tg_bot_scope",
        provider="KIE",
        credential_scope=fp,
        operation="image_generation",
        model="seedream-v4",
        session_maker=test_db_session_maker,
    )
    assert resolved_img is True  # Successfully resolved!

    async with test_db_session_maker() as session:
        inc = (await session.execute(select(AIProviderIncidentModel))).scalar_one()
        assert inc.status == "resolved"

        # Resolved notification was enqueued
        resolved_notif = (
            await session.execute(
                select(AIIncidentNotificationModel).where(AIIncidentNotificationModel.event_type == "resolved")
            )
        ).scalar_one_or_none()
        assert resolved_notif is not None
        assert "AI-провайдер восстановил работу" in resolved_notif.message_text


@pytest.mark.asyncio
async def test_scenario_9_transcription_fallback_isolation(test_db_session_maker, monkeypatch):
    """Scenario 9: Primary KIE transcription failure stays active when OpenAI fallback succeeds."""
    inst_def = {"pm2_name": "tg_bot_stt", "platform": "telegram", "admins": [123]}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[123]))

    kie_fp = get_provider_account_fingerprint("kie_secret_key_1")
    oai_fp = get_provider_account_fingerprint("openai_test_key_1")

    # KIE fails on transcription
    kie_err = InsufficientBalanceError("KIE credits insufficient")
    kie_err.provider = "KIE"
    kie_err.failure = ProviderFailure(provider="KIE", operation="transcription", classification="insufficient_balance_quota", detail_code="credits_exhausted", operator_action_required=True)

    await record_ai_provider_failure(
        origin_instance_id="tg_bot_stt",
        provider="KIE",
        error=kie_err,
        operation="transcription",
        model="elevenlabs/speech-to-text",
        fallback_provider="OpenAI",
        fallback_model="whisper-1",
        fallback_succeeded=True,
        credential_scope=kie_fp,
        session_maker=test_db_session_maker,
    )

    # Fallback OpenAI Whisper succeeds
    await record_ai_operation_success(
        origin_instance_id="tg_bot_stt",
        provider="OpenAI",
        credential_scope=oai_fp,
        operation="transcription",
        model="whisper-1",
        session_maker=test_db_session_maker,
    )

    # Verify KIE transcription incident is STILL ACTIVE!
    async with test_db_session_maker() as session:
        inc = (
            await session.execute(
                select(AIProviderIncidentModel).where(AIProviderIncidentModel.provider == "KIE")
            )
        ).scalar_one()
        assert inc.status == "active"
        assert inc.operation == "transcription"


@pytest.mark.asyncio
async def test_scenario_10_repeat_reminder_cycle(test_db_session_maker, monkeypatch):
    """Scenario 10: Repeat reminder cycle CAS check and notification enqueue."""
    inst_def = {"pm2_name": "tg_bot_rem", "platform": "telegram", "admins": [666]}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[666]))

    err = InsufficientBalanceError("Out of credits")
    err.provider = "KIE"
    err.failure = ProviderFailure(provider="KIE", operation="chat", classification="insufficient_balance_quota", detail_code="credits_exhausted", operator_action_required=True)

    inc = await record_ai_provider_failure(
        origin_instance_id="tg_bot_rem",
        provider="KIE",
        error=err,
        operation="chat",
        session_maker=test_db_session_maker,
    )

    # Set last_alerted_at to 7 hours ago
    async with test_db_session_maker() as session:
        db_inc = await session.get(AIProviderIncidentModel, inc.id)
        db_inc.last_alerted_at = datetime.utcnow() - timedelta(hours=7)
        await session.commit()

    # Check repeat reminders
    reminders = await check_incident_repeat_reminders(
        repeat_seconds=DEFAULT_REPEAT_ALERT_SECONDS,
        session_maker=test_db_session_maker,
    )
    assert reminders == 1

    # Verify cycle_index=1 reminder notification was enqueued
    async with test_db_session_maker() as session:
        rem_notif = (
            await session.execute(
                select(AIIncidentNotificationModel).where(
                    AIIncidentNotificationModel.event_type == "repeat_reminder",
                    AIIncidentNotificationModel.cycle_index == 1,
                )
            )
        ).scalar_one_or_none()
        assert rem_notif is not None
        assert "Напоминание: Сбой AI-провайдера сохраняется" in rem_notif.message_text

    # Immediate second check should do nothing
    reminders_2 = await check_incident_repeat_reminders(
        repeat_seconds=DEFAULT_REPEAT_ALERT_SECONDS,
        session_maker=test_db_session_maker,
    )
    assert reminders_2 == 0


@pytest.mark.asyncio
async def test_scenario_11_autobus_isolation_handled_ai_warning_no_alert(monkeypatch, tmp_path):
    """Scenario 11: Handled AI warnings without stack trace produce 0 alerts for Autobus (real file check)."""
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    error_log_file = tmp_path / "test-bot-error.log"
    error_log_file.write_text(
        "2026-11-14 10:00:00 | WARNING | Handled AI notice: KIE credit balance check | stage=check_credit_balance | provider=KIE | failure_type=insufficient_balance_quota\n"
        "2026-11-14 10:01:00 | WARNING | [AI Incident Worker] Worker running on tg_bot_online\n"
        "2026-11-14 10:02:00 | WARNING | KIE transcription failed, falling back to OpenAI Whisper: model=whisper-large-v3-turbo\n"
    )

    apps = [
        {
            "name": "tg_bot_online",
            "status": "online",
            "token": "tok_1",
            "owner_ids": [272982544],
            "error_log": str(error_log_file),
        }
    ]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    async def fake_empty_async(*args, **kwargs):
        return []

    monkeypatch.setattr(monitor, "check_db_schema", fake_empty_async)
    monkeypatch.setattr(monitor, "check_telegram", fake_empty_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", fake_empty_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_empty_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", fake_empty_async)

    # Note: monitor.read_new_log_errors is NOT mocked! It reads error_log_file directly.
    sent_alerts = []
    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True
    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    exit_code = await monitor.run_check(False)
    assert exit_code == 0
    assert len(sent_alerts) == 0

    # Now append an unhandled fatal error and verify Autobus DOES alert!
    with open(error_log_file, "a", encoding="utf-8") as f:
        f.write("2026-11-14 10:05:00 | ERROR | Traceback (most recent call last):\n  File \"main.py\", line 12\nRuntimeError: unhandled crash\n")

    exit_code_2 = await monitor.run_check(False)
    assert exit_code_2 == 1
    assert len(sent_alerts) == 1
    assert "RuntimeError: unhandled crash" in sent_alerts[0]


@pytest.mark.asyncio
async def test_telegram_http_502_single_transient_no_alert(monkeypatch, tmp_path):
    """Telegram HTTP 502 single transient failure: suppressed without alert."""
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    apps = [
        {"name": "tg_someonewithyou01_bot_new", "status": "online", "token": "tok_1", "owner_ids": [272982544]},
        {"name": "tg_someonewithyou02_bot_new", "status": "online", "token": "tok_2", "owner_ids": [272982544]},
    ]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    async def fake_empty_async(*args, **kwargs):
        return []

    monkeypatch.setattr(monitor, "check_db_schema", fake_empty_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", fake_empty_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_empty_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", fake_empty_async)
    monkeypatch.setattr(monitor, "read_new_log_errors", lambda *args, **kwargs: [])

    # Single isolated bot has getMe HTTP 502 on first check
    async def fake_check_telegram(app, http):
        if app["name"] == "tg_someonewithyou01_bot_new":
            return ["Telegram getMe failed: HTTP 502 {'ok': False, 'error_code': 502, 'description': 'Bad Gateway'}"]
        return []

    monkeypatch.setattr(monitor, "check_telegram", fake_check_telegram)

    sent_alerts = []
    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True

    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    exit_code = await monitor.run_check(False)
    assert exit_code == 0
    assert len(sent_alerts) == 0

    state = monitor.load_state(state_file)
    assert state.get("polling_failure_counts", {}).get("tg_someonewithyou01_bot_new") == 1
    assert "network_tg_someonewithyou01_bot_new" not in state.get("active_incidents", {})


@pytest.mark.asyncio
async def test_telegram_http_502_sustained_alerts(monkeypatch, tmp_path):
    """Telegram HTTP 502 sustained failure: alerts on repeatability."""
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    apps = [
        {"name": "tg_someonewithyou01_bot_new", "status": "online", "token": "tok_1", "owner_ids": [272982544]},
        {"name": "tg_someonewithyou02_bot_new", "status": "online", "token": "tok_2", "owner_ids": [272982544]},
    ]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    async def fake_empty_async(*args, **kwargs):
        return []

    monkeypatch.setattr(monitor, "check_db_schema", fake_empty_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", fake_empty_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_empty_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", fake_empty_async)
    monkeypatch.setattr(monitor, "read_new_log_errors", lambda *args, **kwargs: [])

    async def fake_check_telegram(app, http):
        if app["name"] == "tg_someonewithyou01_bot_new":
            return ["Telegram getMe failed: HTTP 502 {'ok': False, 'error_code': 502, 'description': 'Bad Gateway'}"]
        return []

    monkeypatch.setattr(monitor, "check_telegram", fake_check_telegram)

    sent_alerts = []
    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True

    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    # Check 1: transient -> suppressed
    exit_code_1 = await monitor.run_check(False)
    assert exit_code_1 == 0
    assert len(sent_alerts) == 0

    # Check 2: sustained -> alert sent
    exit_code_2 = await monitor.run_check(False)
    assert exit_code_2 == 1
    assert len(sent_alerts) == 1
    assert "tg_someonewithyou01_bot_new" in sent_alerts[0]
    assert "Telegram getMe failed" in sent_alerts[0]

    state = monitor.load_state(state_file)
    assert "network_tg_someonewithyou01_bot_new" in state.get("active_incidents", {})


@pytest.mark.asyncio
async def test_polling_errors_count_growth_one_continuing_incident(monkeypatch, tmp_path):
    """Polling errors 9 -> 14 -> 20: one continuing incident, not three different alerts."""
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    apps = [
        {"name": "tg_someonewithyou01_bot_new", "status": "online", "token": "tok_1", "owner_ids": [272982544]}
    ]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    async def fake_empty_async(*args, **kwargs):
        return []

    monkeypatch.setattr(monitor, "check_db_schema", fake_empty_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", fake_empty_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_empty_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", fake_empty_async)

    sent_alerts = []
    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True

    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    # Start with sustained state (consecutive fails >= 2)
    state = monitor.load_state(state_file)
    state["polling_failure_counts"] = {"tg_someonewithyou01_bot_new": 2}
    monitor.save_state(state, state_file)

    monkeypatch.setattr(
        monitor, "read_new_log_errors",
        lambda *args, **kwargs: ["Logs: 9 polling fetch errors since last check"]
    )
    await monitor.run_check(False)
    assert len(sent_alerts) == 1
    assert "9 polling fetch errors" in sent_alerts[0]

    # Run 2: error count grows to 14 polling errors -> NO second alert sent
    monkeypatch.setattr(
        monitor, "read_new_log_errors",
        lambda *args, **kwargs: ["Logs: 14 polling fetch errors since last check"]
    )
    await monitor.run_check(False)
    assert len(sent_alerts) == 1

    # Run 3: error count grows to 20 polling errors -> NO third alert sent
    monkeypatch.setattr(
        monitor, "read_new_log_errors",
        lambda *args, **kwargs: ["Logs: 20 polling fetch errors since last check"]
    )
    await monitor.run_check(False)
    assert len(sent_alerts) == 1


@pytest.mark.asyncio
async def test_telegram_network_recovery_single_recovery_alert(monkeypatch, tmp_path):
    """Restoration of connectivity sends exactly one recovery alert and clears incident."""
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    apps = [
        {"name": "tg_someonewithyou01_bot_new", "status": "online", "token": "tok_1", "owner_ids": [272982544]}
    ]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    async def fake_empty_async(*args, **kwargs):
        return []

    monkeypatch.setattr(monitor, "check_db_schema", fake_empty_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", fake_empty_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_empty_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", fake_empty_async)

    sent_alerts = []
    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True

    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    # Active single-bot network incident in state
    state = monitor.load_state(state_file)
    state["active_incidents"]["network_tg_someonewithyou01_bot_new"] = {
        "first_seen": int(datetime.now().timestamp()) - 600,
        "last_seen": int(datetime.now().timestamp()),
        "last_alerted": int(datetime.now().timestamp()) - 600,
        "alert_count": 1,
        "affected_bots": ["tg_someonewithyou01_bot_new"],
        "summary": "Telegram HTTP 502 (Bad Gateway)",
    }
    monitor.save_state(state, state_file)

    # Check: bot is now fully recovered
    monkeypatch.setattr(monitor, "check_telegram", fake_empty_async)
    monkeypatch.setattr(monitor, "read_new_log_errors", lambda *args, **kwargs: [])

    exit_code = await monitor.run_check(False)
    assert exit_code == 0

    # Exactly one recovery alert sent
    assert len(sent_alerts) == 1
    assert "Восстановление: Telegram API / Прокси в норме" in sent_alerts[0]
    assert "tg_someonewithyou01_bot_new" in sent_alerts[0]

    state = monitor.load_state(state_file)
    assert "network_tg_someonewithyou01_bot_new" not in state.get("active_incidents", {})


@pytest.mark.asyncio
async def test_two_or_more_bots_form_group_network_incident(monkeypatch, tmp_path):
    """Two or more affected bots form a single group network incident."""
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    apps = [
        {"name": "tg_someonewithyou01_bot_new", "status": "online", "token": "tok_1", "owner_ids": [272982544]},
        {"name": "tg_someonewithyou02_bot_new", "status": "online", "token": "tok_2", "owner_ids": [272982544]},
        {"name": "tg_someonelikeyouai02_bot_legacy", "status": "online", "token": "tok_3", "owner_ids": [272982544]},
    ]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    async def fake_empty_async(*args, **kwargs):
        return []

    monkeypatch.setattr(monitor, "check_db_schema", fake_empty_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", fake_empty_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_empty_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", fake_empty_async)
    monkeypatch.setattr(monitor, "read_new_log_errors", lambda *args, **kwargs: [])

    # Two bots have HTTP 502
    async def fake_tg(app, http):
        if app["name"] in ("tg_someonewithyou01_bot_new", "tg_someonewithyou02_bot_new"):
            return ["Telegram getMe failed: HTTP 502 {'ok': False, 'error_code': 502, 'description': 'Bad Gateway'}"]
        return []

    monkeypatch.setattr(monitor, "check_telegram", fake_tg)

    sent_alerts = []
    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True

    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    exit_code = await monitor.run_check(False)
    assert exit_code == 1
    # Exactly one combined group alert
    assert len(sent_alerts) == 1
    assert "Общий сетевой инцидент Telegram API / Прокси" in sent_alerts[0]
    assert "Затронуто ботов: 2" in sent_alerts[0]

    state = monitor.load_state(state_file)
    assert monitor.INCIDENT_SHARED_NETWORK in state.get("active_incidents", {})


@pytest.mark.asyncio
async def test_pm2_stopped_and_db_errors_preserved(monkeypatch, tmp_path):
    """PM2 stopped and database errors are preserved and alerted without suppression."""
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    apps = [
        {"name": "tg_bot_stopped", "status": "stopped", "token": "tok_1", "owner_ids": [272982544]},
        {"name": "tg_bot_db_broken", "status": "online", "token": "tok_2", "owner_ids": [272982544]},
    ]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    async def fake_empty_async(*args, **kwargs):
        return []

    monkeypatch.setattr(monitor, "check_telegram", fake_empty_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", fake_empty_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_empty_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", fake_empty_async)
    monkeypatch.setattr(monitor, "read_new_log_errors", lambda *args, **kwargs: [])

    async def mock_db(app):
        if app["name"] == "tg_bot_db_broken":
            return ["DB: database connection failed: connection refused"]
        return []

    monkeypatch.setattr(monitor, "check_db_schema", mock_db)

    sent_alerts = []
    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True

    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    exit_code = await monitor.run_check(False)
    assert exit_code == 1
    assert len(sent_alerts) == 1
    assert "PM2: status is stopped" in sent_alerts[0]
    assert "DB: database connection failed: connection refused" in sent_alerts[0]


@pytest.mark.asyncio
async def test_bot_blocked_errors_completely_suppressed(monkeypatch, tmp_path):
    """Bot blocked by user errors remain completely suppressed from alerts."""
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    error_log_file = tmp_path / "test-bot-error.log"
    error_log_file.write_text(
        "2026-11-14 10:00:00 | ERROR | aiogram.dispatcher | TelegramForbiddenError: Forbidden: bot was blocked by the user\n"
        "2026-11-14 10:01:00 | ERROR | notification_transport | TelegramForbiddenError: bot was blocked by the user\n"
    )
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    apps = [
        {
            "name": "tg_bot_active",
            "status": "online",
            "token": "tok_1",
            "owner_ids": [272982544],
            "error_log": str(error_log_file),
        },
    ]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    async def fake_empty_async(*args, **kwargs):
        return []

    monkeypatch.setattr(monitor, "check_telegram", fake_empty_async)
    monkeypatch.setattr(monitor, "check_db_schema", fake_empty_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", fake_empty_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_empty_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", fake_empty_async)

    sent_alerts = []
    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True

    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    exit_code = await monitor.run_check(True)
    assert exit_code == 0
    assert len(sent_alerts) == 0


def test_generated_telegram_methods_pass_aiogram_pydantic_validation():
    """All generated Telegram message texts pass aiogram and Pydantic model validation."""
    alert_1 = monitor.format_recovery_alert(1, "45 минут", bot_names=["tg_someonewithyou01_bot_new"])
    alert_2 = monitor.format_recovery_alert(3, "1 час 20 минут")
    alert_3 = monitor.format_issues_alert(
        all_issues=[
            ("tg_someonewithyou01_bot_new", [
                "Telegram getMe failed: HTTP 502 {'ok': False, 'error_code': 502, 'description': 'Bad Gateway'}",
                "Logs: 14 polling fetch errors since last check",
            ])
        ],
        is_shared_network_active=False,
        include_network_summary=False,
        shared_network_summary="",
        affected_net_bots=["tg_someonewithyou01_bot_new"],
        total_tg_bots=17,
        proxy_status="настроен (http://127.0.0.1:3128)",
    )
    alert_4 = _format_incident_message_text(
        provider="KIE",
        operation="image_generation",
        model_scope="seedream-v4",
        incident_type="insufficient_balance_quota",
        detail_code="credits_exhausted",
        evidence_tier="CONFIRMED_STRUCTURED",
        event_type="opened",
    )
    alert_5 = _format_incident_message_text(
        provider="KIE",
        operation="image_generation",
        model_scope="seedream-v4",
        incident_type="insufficient_balance_quota",
        detail_code="credits_exhausted",
        evidence_tier="CONFIRMED_STRUCTURED",
        event_type="resolved",
    )

    for text in (alert_1, alert_2, alert_3, alert_4, alert_5):
        method = SendMessage(chat_id=272982544, text=text[:3900], parse_mode="HTML")
        assert method.chat_id == 272982544
        assert len(method.text) > 0
        assert len(method.text) <= 4096


# ==============================================================================
# Additional Architectural Invariant Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_observability_ai_success_not_broken_by_incident_db_error(test_db_session_maker, monkeypatch):
    """When AI operation succeeds, failure to record incident state does NOT fail the operation or break user result."""
    import ai_integration
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", "8281390498:AAFtest_telegram_token_123456789012")
    monkeypatch.setattr(ai_integration, "async_session_maker", test_db_session_maker)

    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, image_generation_provider="Gemini", image_generation_model="gemini-3.1-flash-image")
            session.add(cfg)
        else:
            cfg.image_generation_provider = "Gemini"
            cfg.image_generation_model = "gemini-3.1-flash-image"
        await session.commit()

    async def mock_call_gemini(*args, **kwargs):
        return b"fake_png_data_123"

    monkeypatch.setattr(ai_integration, "_call_gemini_image_generation", mock_call_gemini)

    # Mock record_ai_operation_success to simulate a database crash / connection loss
    async def mock_fail_obs(*args, **kwargs):
        raise RuntimeError("Incident DB completely broken!")

    import ai_incident_manager
    monkeypatch.setattr(ai_incident_manager, "record_ai_operation_success", mock_fail_obs)

    # Calling generate_image MUST succeed and return the generated image bytes to user!
    result = await ai_integration.generate_image("A beautiful sunset")
    assert result == b"fake_png_data_123"


@pytest.mark.asyncio
async def test_observability_incident_db_error_does_not_mask_ai_failure(test_db_session_maker, monkeypatch):
    """When AI operation fails, failure to record incident in DB does NOT mask or replace the original AI exception."""
    import ai_integration
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", "8281390498:AAFtest_telegram_token_123456789012")
    monkeypatch.setattr(ai_integration, "async_session_maker", test_db_session_maker)

    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, image_generation_provider="KIE", image_generation_model="seedream/4.5-text-to-image")
            session.add(cfg)
        else:
            cfg.image_generation_provider = "KIE"
            cfg.image_generation_model = "seedream/4.5-text-to-image"
        await session.commit()

    orig_err = InsufficientBalanceError("KIE credits empty")
    orig_err.provider = "KIE"
    async def mock_call_kie(*args, **kwargs):
        raise orig_err

    monkeypatch.setattr(ai_integration, "_call_kie_image_generation", mock_call_kie)

    # Mock record_ai_provider_failure to simulate DB crash
    async def mock_fail_rec(*args, **kwargs):
        raise RuntimeError("Incident DB failure recording down!")

    import ai_incident_manager
    monkeypatch.setattr(ai_incident_manager, "record_ai_provider_failure", mock_fail_rec)

    # Must raise the original InsufficientBalanceError, NOT the DB RuntimeError!
    with pytest.raises(InsufficientBalanceError) as exc_info:
        await ai_integration.generate_image("A futuristic city")

    assert "KIE credits empty" in str(exc_info.value)


@pytest.mark.asyncio
async def test_chat_fallback_single_canonical_incident_and_notification(test_db_session_maker, monkeypatch):
    """When primary provider fails and fallback succeeds: exactly ONE canonical incident and one notification per admin."""
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    bot_token = "8281390498:AAFtest_telegram_token_123456789012"
    monkeypatch.setenv("BOT_TOKEN", bot_token)

    primary_err = InsufficientBalanceError("KIE credits 0")
    primary_err.provider = "KIE"

    # 1. Primary failure recorded with fallback_succeeded=True
    inc = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=primary_err,
        operation="chat",
        model="kie-chat-v1",
        fallback_provider="OpenAI",
        fallback_model="gpt-5.6-terra",
        fallback_succeeded=True,
        session_maker=test_db_session_maker,
    )
    assert inc is not None
    assert inc.status == "active"

    # 2. Operational success recorded for OpenAI fallback
    fp_oai = get_provider_account_fingerprint("openai_test_key_1")
    await record_ai_operation_success(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="OpenAI",
        credential_scope=fp_oai,
        operation="chat",
        model="gpt-5.6-terra",
        session_maker=test_db_session_maker,
    )

    # 3. Simulate calling notify_admins_about_error with stage="ai_provider_fallback"
    import error_reporting
    mock_bot = AsyncMock(spec=Bot)
    mock_bot.token = bot_token
    await error_reporting.notify_admins_about_error(
        mock_bot,
        title="Основной AI-провайдер не сработал — использован резервный",
        provider="KIE",
        model="kie-chat-v1",
        stage="ai_provider_fallback",
        exception=primary_err,
        classification="insufficient_balance_quota",
    )

    async with test_db_session_maker() as session:
        # Verify: Exactly ONE incident in DB, with operation='chat', NOT 'ai_provider_fallback'
        all_incidents = (await session.execute(select(AIProviderIncidentModel))).scalars().all()
        assert len(all_incidents) == 1
        assert all_incidents[0].operation == "chat"
        assert all_incidents[0].provider == "KIE"

        # Verify: Outbox has fallback_used notifications for authorized admins
        notifs = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        assert len(notifs) > 0
        for n in notifs:
            assert n.event_type == "fallback_used"
            assert "Резервный:" in n.message_text
            assert "OPENAI" in n.message_text


@pytest.mark.asyncio
async def test_admin_key_change_triggers_handle_key_rotation(test_db_session_maker, monkeypatch):
    """Rotating an API key supersedes active incidents across Telegram and MAX companion bots without false alert."""
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda_telegram")

    err = InsufficientBalanceError("KIE credits empty")
    err.provider = "KIE"

    # Open active incident for Telegram bot
    inc_tg = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        error=err,
        operation="chat",
        model="kie-v1",
        credential_scope="old_kie_fp_1",
        session_maker=test_db_session_maker,
    )
    assert inc_tg is not None

    # Open active incident for MAX companion bot in same database
    inc_max = await record_ai_provider_failure(
        origin_instance_id="max_se13639182_bot_legacy",
        provider="KIE",
        error=err,
        operation="vision",
        model="kie-vision-1",
        credential_scope="old_kie_fp_1",
        session_maker=test_db_session_maker,
    )
    assert inc_max is not None

    # Now simulate key rotation to new key
    superseded = await handle_key_rotation(
        origin_instance_id=None,
        provider="KIE",
        new_key="new_super_secret_kie_key_2026",
        base_url="https://api.kie.ai",
        session_maker=test_db_session_maker,
    )
    assert superseded == 2

    async with test_db_session_maker() as session:
        incidents = (await session.execute(select(AIProviderIncidentModel))).scalars().all()
        for inc in incidents:
            assert inc.status == "superseded"
            assert inc.resolved_at is not None

        # Verify pending notifications are cancelled
        notifs = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        for notif in notifs:
            assert notif.status == "cancelled"
            assert notif.last_error == "superseded_by_key_rotation"


@pytest.mark.asyncio
async def test_foreign_bot_token_worker_fail_closed(test_db_session_maker, monkeypatch):
    """Outbox worker rejects Bot with mismatched/foreign token and refuses delivery (fail-closed)."""
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", "8281390498:AAFcorrect_bot_token")

    # Open an incident
    err = InsufficientBalanceError("Low balance")
    err.provider = "KIE"
    err.failure = ProviderFailure(
        provider="KIE",
        operation="image_generation",
        classification="insufficient_balance_quota",
        detail_code="credits_exhausted",
        operator_action_required=True,
    )
    await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=err,
        operation="image_generation",
        session_maker=test_db_session_maker,
    )

    # Pass a Bot with foreign platform_id (e.g. Autobus token 7858169956)
    foreign_bot = AsyncMock(spec=Bot)
    foreign_bot.token = "7858169956:AAFforeign_autobus_token_wrong"

    delivered = await process_ai_incident_notifications(
        foreign_bot,
        session_maker=test_db_session_maker,
    )
    assert delivered == 0
    foreign_bot.send_message.assert_not_called()

    # Verify rows in DB remain pending and were not delivered
    async with test_db_session_maker() as session:
        notifs = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        assert len(notifs) > 0
        for n in notifs:
            assert n.status == "pending"


# ==============================================================================
# Dedicated Regression Tests for Independent Code Review Defect Items (1 - 6)
# ==============================================================================

@pytest.mark.asyncio
async def test_item_1_autobus_logger_to_pm2_error_log_full_flow(monkeypatch, tmp_path):
    """Item 1: End-to-end Autobus log flow:
    - Handled AI notice in error log does NOT trigger Autobus alert (0 alerts).
    - Genuine error (operational error, DB failure, unhandled worker traceback) DOES trigger Autobus alert (1 alert).
    """
    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    error_log_file = tmp_path / "bot-error.log"
    # Write handled notice with words like 'Error:' or 'failure' inside it
    error_log_file.write_text(
        "2026-11-14 12:00:00 | WARNING | Handled AI notice: KIE credit balance check failed with Error: low credits\n"
        "2026-11-14 12:01:00 | INFO | Handled AI notice: KIE transcription fallback to OpenAI Whisper\n"
    )

    apps = [
        {
            "name": "tg_someonewithyou01_bot_new",
            "status": "online",
            "token": "tok_1",
            "owner_ids": [272982544],
            "error_log": str(error_log_file),
        }
    ]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    async def fake_empty_async(*args, **kwargs):
        return []

    monkeypatch.setattr(monitor, "check_db_schema", fake_empty_async)
    monkeypatch.setattr(monitor, "check_telegram", fake_empty_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", fake_empty_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", fake_empty_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", fake_empty_async)

    sent_alerts = []
    async def fake_send_alert(a, text, **kwargs):
        sent_alerts.append(text)
        return True
    monkeypatch.setattr(monitor, "send_alert", fake_send_alert)

    # 1. Run monitor: handled notices produce 0 Autobus alerts
    exit_code_1 = await monitor.run_check(False)
    assert exit_code_1 == 0
    assert len(sent_alerts) == 0

    # 2. Append genuine incident worker database crash
    with open(error_log_file, "a", encoding="utf-8") as f:
        f.write(
            "2026-11-14 12:05:00 | ERROR | [AI Incident Worker] OperationalError: could not connect to server: Connection refused\n"
            "Traceback (most recent call last):\n"
            "  File \"ai_incident_manager.py\", line 990, in process_ai_incident_notifications\n"
            "sqlalchemy.exc.OperationalError: could not connect to server\n"
        )

    exit_code_2 = await monitor.run_check(False)
    assert exit_code_2 == 1
    assert len(sent_alerts) == 1
    assert "OperationalError" in sent_alerts[0] or "Traceback" in sent_alerts[0]


@pytest.mark.asyncio
async def test_item_2_telegram_live_identity_verification_fail_closed(test_db_session_maker, monkeypatch):
    """Item 2: Live Telegram identity verification via verify_bot_identity_live:
    - Valid numeric token prefix, but get_me raises TelegramUnauthorizedError -> fail-closed (0 delivered).
    - Valid numeric token prefix, but get_me returns wrong user ID -> fail-closed (0 delivered).
    - Valid numeric token prefix and get_me matches expected identity -> delivers successfully.
    """
    from aiogram.exceptions import TelegramUnauthorizedError
    inst_def = {
        "pm2_name": "tg_someonewithyou01_bot_new",
        "platform": "telegram",
        "platform_id": 8281390498,
        "username": "someonewithyou01_bot",
        "admins": [272982544],
    }
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[272982544]))
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", "8281390498:AAFtest_token_live_verification")

    err = InsufficientBalanceError("KIE credits 0")
    err.provider = "KIE"
    err.failure = ProviderFailure(
        provider="KIE",
        operation="chat",
        classification="insufficient_balance_quota",
        detail_code="credits_exhausted",
        operator_action_required=True,
    )
    inc = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=err,
        operation="chat",
        session_maker=test_db_session_maker,
    )
    assert inc is not None

    # Case A: Valid numeric prefix, but live get_me raises TelegramUnauthorizedError
    bot_unauth = AsyncMock(spec=Bot)
    bot_unauth.token = "8281390498:AAFrevoked_or_invalid_secret"
    bot_unauth.get_me = AsyncMock(side_effect=TelegramUnauthorizedError(method=SendMessage(chat_id=1, text=""), message="Unauthorized"))
    bot_unauth.send_message = AsyncMock(return_value=True)

    delivered_a = await process_ai_incident_notifications(bot_unauth, session_maker=test_db_session_maker)
    assert delivered_a == 0
    bot_unauth.send_message.assert_not_called()

    # Case B: Valid numeric prefix, but live get_me returns wrong user ID (mismatched identity)
    bot_mismatch = AsyncMock(spec=Bot)
    bot_mismatch.token = "8281390498:AAFtoken_wrong_id"
    bot_mismatch.get_me = AsyncMock(return_value=TgUser(id=9999999999, is_bot=True, first_name="other_bot", username="other_bot"))
    bot_mismatch.send_message = AsyncMock(return_value=True)

    delivered_b = await process_ai_incident_notifications(bot_mismatch, session_maker=test_db_session_maker)
    assert delivered_b == 0
    bot_mismatch.send_message.assert_not_called()

    # Case C: Correct live identity -> delivery succeeds
    bot_ok = AsyncMock(spec=Bot)
    bot_ok.token = "8281390498:AAFtoken_valid"
    bot_ok.get_me = AsyncMock(return_value=TgUser(id=8281390498, is_bot=True, first_name="bot", username="someonewithyou01_bot"))
    bot_ok.send_message = AsyncMock(return_value=True)

    delivered_c = await process_ai_incident_notifications(bot_ok, session_maker=test_db_session_maker)
    assert delivered_c == 1
    assert bot_ok.send_message.call_count == 1


@pytest.mark.asyncio
async def test_item_3_kie_soft_warning_durable_delivery_and_restart_retry(test_db_session_maker, monkeypatch):
    """Item 3: KIE soft warning durable outbox delivery:
    - Enqueues durable outbox record per recipient without unmapped ORM attributes.
    - Sets config.kie_credit_alert_sent = True in AIConfig.
    - Failed recipient (network error) is retried across restarts without re-delivering to successful recipient.
    - Replenishment enqueues resolution and resets config.kie_credit_alert_sent = False.
    """
    from aiogram.exceptions import TelegramNetworkError
    inst_def = {"pm2_name": "tg_someonewithyou01_bot_new", "platform": "telegram", "platform_id": 8281390498, "admins": [1001, 1002]}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[1001, 1002]))

    bot = AsyncMock(spec=Bot)
    bot.token = "8281390498:AAFtest_token"
    bot.get_me = AsyncMock(return_value=TgUser(id=8281390498, is_bot=True, first_name="bot", username="someonewithyou01_bot"))

    monkeypatch.setattr(scheduler, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr(scheduler, "get_kie_remaining_credits", AsyncMock(return_value=1.20))
    monkeypatch.setattr(scheduler, "get_all_admin_ids", AsyncMock(return_value=[1001, 1002]))

    # Seed AIConfig
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="KIE", kie_api_key="test_key_soft", kie_credit_alert_threshold=2.0, kie_credit_alert_sent=False)
            session.add(cfg)
        else:
            cfg.kie_credit_alert_threshold = 2.0
            cfg.kie_credit_alert_sent = False
        await session.commit()

    # Admin 1001 succeeds, admin 1002 fails with network error
    async def selective_send(chat_id, text, **kwargs):
        if chat_id == 1001:
            return True
        elif chat_id == 1002:
            raise TelegramNetworkError(method=SendMessage(chat_id=chat_id, text=text), message="Connection reset")
    bot.send_message.side_effect = selective_send

    # 1. Run check
    await scheduler.check_kie_credit_balance(bot)

    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        assert cfg.kie_credit_alert_sent is True
        # Verify no unmapped attribute crash
        assert not hasattr(cfg, "kie_credit_last_alert_at")

        notifs = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        by_adm = {n.recipient_id: n for n in notifs}
        assert by_adm[1001].status == "delivered"
        assert by_adm[1002].status == "pending"
        assert by_adm[1002].last_error == "telegram_network_error"

        # Advance next_retry_at for 1002
        by_adm[1002].next_retry_at = datetime.utcnow() - timedelta(seconds=1)
        await session.commit()

    # 2. Restart / second sweep: now admin 1002 succeeds
    bot.send_message.side_effect = None
    bot.send_message.return_value = True
    bot.send_message.reset_mock()

    delivered_retry = await process_ai_incident_notifications(bot, session_maker=test_db_session_maker)
    assert delivered_retry == 1
    # Only admin 1002 was sent to! Admin 1001 was not re-sent!
    assert bot.send_message.call_count == 1
    call_chat_id = bot.send_message.call_args.kwargs.get("chat_id") or bot.send_message.call_args.args[0]
    assert call_chat_id == 1002

    async with test_db_session_maker() as session:
        notifs = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        for n in notifs:
            assert n.status == "delivered"

    # 3. Replenishment: balance now 10.00 >= threshold 2.00
    monkeypatch.setattr(scheduler, "get_kie_remaining_credits", AsyncMock(return_value=10.00))
    bot.send_message.reset_mock()

    await scheduler.check_kie_credit_balance(bot)

    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        assert cfg.kie_credit_alert_sent is False

        # Resolution notifications created and delivered
        res_notifs = (await session.execute(
            select(AIIncidentNotificationModel).where(AIIncidentNotificationModel.event_type == "resolved")
        )).scalars().all()
        assert len(res_notifs) == 2
        for r in res_notifs:
            assert r.status == "delivered"


@pytest.mark.asyncio
async def test_item_4_chat_fallback_e2e_single_notification_on_success_and_failure(test_db_session_maker, monkeypatch):
    """Item 4: In chat fallback, eliminate premature failure recording:
    - Primary fails, fallback succeeds: exactly ONE notification per admin ('fallback_used'), ZERO 'opened'.
    - Primary fails, fallback fails: terminal failure, exactly ONE 'opened' notification per admin, ZERO 'fallback_used'.
    """
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", "8281390498:AAFtest_telegram_token_123456789012")
    inst_def = {"pm2_name": "tg_someonewithyou01_bot_new", "platform": "telegram", "platform_id": 8281390498, "admins": [7001]}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[7001]))

    # PART A: Primary fails, fallback succeeds
    primary_err = InsufficientBalanceError("KIE credits empty")
    primary_err.provider = "KIE"

    # In chat/vision: record_ai_provider_failure is NOT called before fallback attempt!
    # On fallback success, it is called with fallback_succeeded=True
    inc_success = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=primary_err,
        operation="chat",
        model="kie-chat-v1",
        fallback_provider="OpenAI",
        fallback_model="gpt-4o",
        fallback_succeeded=True,
        session_maker=test_db_session_maker,
    )
    assert inc_success is not None

    async with test_db_session_maker() as session:
        notifs_a = (await session.execute(
            select(AIIncidentNotificationModel).where(AIIncidentNotificationModel.incident_id == inc_success.id)
        )).scalars().all()
        assert len(notifs_a) == 1
        assert notifs_a[0].event_type == "fallback_used"
        assert notifs_a[0].event_type != "opened"

    # PART B: Primary fails, and fallback also fails (terminal failure)
    # On fallback failure, primary and fallback are recorded with fallback_succeeded=False
    inc_terminal = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="OpenAI",
        error=InsufficientBalanceError("OpenAI quota exceeded"),
        operation="chat",
        model="gpt-4o",
        fallback_succeeded=False,
        session_maker=test_db_session_maker,
    )
    assert inc_terminal is not None

    async with test_db_session_maker() as session:
        notifs_b = (await session.execute(
            select(AIIncidentNotificationModel).where(AIIncidentNotificationModel.incident_id == inc_terminal.id)
        )).scalars().all()
        assert len(notifs_b) == 1
        assert notifs_b[0].event_type == "opened"
        assert notifs_b[0].event_type != "fallback_used"


@pytest.mark.asyncio
async def test_item_5_max_incident_repeat_reminders_delivered_by_telegram_companion(test_db_session_maker, monkeypatch):
    """Item 5: MAX incident repeat reminders:
    - MAX bot registers failure in shared DB (companion_instance_id = tg_companion).
    - Telegram companion worker checks repeat reminders (companion_instance_id == current_bot_instance_id).
    - Repeat reminder outbox row enqueued with cycle_index=1 for Telegram companion admins.
    - Duplicate reminder cycles strictly prevented.
    - Telegram companion worker delivers reminder to Telegram admins.
    """
    tg_inst = {
        "pm2_name": "tg_veraveda777_bot_legacy",
        "platform": "telegram",
        "platform_id": 8224473896,
        "username": "veraveda777_bot",
        "admins": [777],
    }
    max_inst = {
        "pm2_name": "max_se13639182_bot_legacy",
        "platform": "max",
        "companion_pm2_name": "tg_veraveda777_bot_legacy",
    }

    def mock_get_inst(name):
        if name == "tg_veraveda777_bot_legacy":
            return tg_inst
        if name == "max_se13639182_bot_legacy":
            return max_inst
        return None

    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", mock_get_inst)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: tg_inst if s.get("platform") == "max" else s)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[777]))

    # 1. MAX bot registers failure in shared DB
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: max_inst)
    err = InsufficientBalanceError("KIE credits exhausted")
    err.provider = "KIE"
    err.failure = ProviderFailure(
        provider="KIE",
        operation="chat",
        classification="insufficient_balance_quota",
        detail_code="credits_exhausted",
        operator_action_required=True,
    )
    inc = await record_ai_provider_failure(
        origin_instance_id="max_se13639182_bot_legacy",
        provider="KIE",
        error=err,
        operation="chat",
        session_maker=test_db_session_maker,
    )
    assert inc is not None
    assert inc.origin_instance_id == "max_se13639182_bot_legacy"
    assert inc.companion_instance_id == "tg_veraveda777_bot_legacy"

    # Age incident by 7 hours
    async with test_db_session_maker() as session:
        db_inc = await session.get(AIProviderIncidentModel, inc.id)
        db_inc.last_alerted_at = datetime.utcnow() - timedelta(hours=7)
        await session.commit()

    # 2. Telegram companion worker runs check_incident_repeat_reminders
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: tg_inst)
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")
    reminders = await check_incident_repeat_reminders(
        repeat_seconds=DEFAULT_REPEAT_ALERT_SECONDS,
        session_maker=test_db_session_maker,
    )
    assert reminders == 1

    # Verify repeat reminder was created for Telegram companion
    async with test_db_session_maker() as session:
        rem_notif = (
            await session.execute(
                select(AIIncidentNotificationModel).where(
                    AIIncidentNotificationModel.incident_id == inc.id,
                    AIIncidentNotificationModel.event_type == "repeat_reminder",
                    AIIncidentNotificationModel.cycle_index == 1,
                )
            )
        ).scalar_one_or_none()
        assert rem_notif is not None
        assert rem_notif.authorized_instance_id == "tg_veraveda777_bot_legacy"
        assert rem_notif.recipient_id == 777
        assert "Напоминание: Сбой AI-провайдера сохраняется" in rem_notif.message_text

    # 3. Immediate second check creates 0 duplicate reminders
    reminders_2 = await check_incident_repeat_reminders(
        repeat_seconds=DEFAULT_REPEAT_ALERT_SECONDS,
        session_maker=test_db_session_maker,
    )
    assert reminders_2 == 0

    # 4. Telegram companion delivers reminder
    bot = AsyncMock(spec=Bot)
    bot.token = "8224473896:AAFtest_token"
    bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))
    bot.send_message = AsyncMock(return_value=True)

    delivered = await process_ai_incident_notifications(bot, session_maker=test_db_session_maker)
    assert delivered == 2  # initial opened + repeat reminder
    assert bot.send_message.call_count == 2


@pytest.mark.asyncio
async def test_item_6_admin_key_rotation_via_process_api_input_e2e(test_db_session_maker, monkeypatch):
    """Item 6: Real Admin key rotation flow via process_api_input:
    - Input validation: empty / whitespace key is rejected and clears state without modifying AIConfig.
    - Cancellation handler: admin_ai_keys callback clears state and leaves AIConfig untouched.
    - Valid key: updates AIConfig, triggers handle_key_rotation, transitions Telegram and MAX incidents
      in shared DB to 'superseded' WITHOUT false 'восстановлено' alert, cancels outbox, and re-renders settings.
    """
    from aiogram.fsm.context import FSMContext
    from aiogram.fsm.storage.memory import MemoryStorage, StorageKey
    from aiogram.types import CallbackQuery, Chat, Message
    import handlers

    monkeypatch.setattr(handlers, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")

    # Seed initial AIConfig with old key
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="KIE", kie_api_key="old_kie_key_111", kie_model="kie-v1")
            session.add(cfg)
        else:
            cfg.kie_api_key = "old_kie_key_111"
        await session.commit()

    storage = MemoryStorage()
    storage_key = StorageKey(bot_id=8224473896, chat_id=777, user_id=777)
    state = FSMContext(storage=storage, key=storage_key)

    bot = AsyncMock(spec=Bot)
    bot.id = 8224473896
    bot.token = "8224473896:AAFtest_token"
    bot.send_message = AsyncMock(return_value=True)

    user = TgUser(id=777, is_bot=False, first_name="Admin", username="admin777")
    chat = Chat(id=777, type="private")

    # --- Case 1: Empty / whitespace input is rejected ---
    await state.set_state(handlers.AdminStates.set_api_key)
    await state.update_data(provider="KIE")

    msg_empty = AsyncMock(spec=Message)
    msg_empty.text = "   "
    msg_empty.from_user = user
    msg_empty.chat = chat
    msg_empty.answer = AsyncMock()
    msg_empty.delete = AsyncMock()

    await handlers.process_api_input(msg_empty, state, bot)

    msg_empty.answer.assert_called_once_with("API ключ не может быть пустым. Настройки не изменены.")
    assert await state.get_state() is None

    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        assert cfg.kie_api_key == "old_kie_key_111"

    # --- Case 2: Cancellation via admin_ai_keys callback ---
    await state.set_state(handlers.AdminStates.set_api_key)
    await state.update_data(provider="KIE")

    cb_cancel = AsyncMock(spec=CallbackQuery)
    cb_cancel.data = "admin_ai_keys"
    cb_cancel.from_user = user
    cb_cancel.message = AsyncMock()

    await handlers.admin_ai_keys_models(cb_cancel, state=state)
    assert await state.get_state() is None

    # --- Case 3: Valid key input saves, rotates incidents, and re-renders settings ---
    # Create active incidents for Telegram and MAX companion bots
    err = InsufficientBalanceError("Credits depleted")
    err.provider = "KIE"

    inc_tg = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        error=err,
        operation="chat",
        credential_scope=get_provider_account_fingerprint("old_kie_key_111"),
        session_maker=test_db_session_maker,
    )
    inc_max = await record_ai_provider_failure(
        origin_instance_id="max_se13639182_bot_legacy",
        provider="KIE",
        error=err,
        operation="vision",
        credential_scope=get_provider_account_fingerprint("old_kie_key_111"),
        session_maker=test_db_session_maker,
    )
    assert inc_tg is not None and inc_max is not None

    await state.set_state(handlers.AdminStates.set_api_key)
    await state.update_data(
        provider="KIE",
        settings_message_id=999,
        settings_chat_id=777,
        settings_provider="KIE",
    )

    msg_valid = AsyncMock(spec=Message)
    msg_valid.text = "new_valid_kie_key_2026"
    msg_valid.from_user = user
    msg_valid.chat = chat
    msg_valid.answer = AsyncMock()
    msg_valid.delete = AsyncMock()

    mock_render = AsyncMock()
    monkeypatch.setattr(handlers, "_edit_provider_model_settings", mock_render)
    monkeypatch.setattr(handlers, "send_temp_notification", AsyncMock())

    await handlers.process_api_input(msg_valid, state, bot)

    assert await state.get_state() is None

    # 1. AIConfig updated
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        assert cfg.kie_api_key == "new_valid_kie_key_2026"

        # 2. Both Telegram and MAX incidents superseded (NOT resolved, so no false recovery alert!)
        db_inc_tg = await session.get(AIProviderIncidentModel, inc_tg.id)
        db_inc_max = await session.get(AIProviderIncidentModel, inc_max.id)
        assert db_inc_tg.status == "superseded"
        assert db_inc_max.status == "superseded"
        assert db_inc_tg.resolved_at is not None
        assert db_inc_max.resolved_at is not None

        # 3. Pending notifications cancelled
        notifs = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        for n in notifs:
            assert n.status == "cancelled"
            assert n.last_error == "superseded_by_key_rotation"

        # 4. Settings view re-rendered with new key
        mock_render.assert_called_once_with(bot, 777, 999, "KIE")


# ==============================================================================
# Real PostgreSQL Integration Tests
# ==============================================================================

@pytest_asyncio.fixture
async def pg_test_session_maker():
    """Provide a real PostgreSQL session maker on validated disposable test database."""
    pg_url = os.environ.get("POSTGRES_TEST_DATABASE_URL") or "postgresql+asyncpg://localhost/test_tgpsychobot_ai_incidents"
    try:
        engine = create_async_engine(pg_url, echo=False)
        async with engine.begin() as conn:
            curr_db = (await conn.execute(text("SELECT current_database()"))).scalar()
            if not curr_db or not curr_db.startswith("test_"):
                await engine.dispose()
                pytest.skip(f"SAFETY ABORT: PostgreSQL database {curr_db} does not start with 'test_'")
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:
        pytest.skip(f"PostgreSQL test database not available ({pg_url}): {exc}")

    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    # Seed AIConfig
    async with session_maker() as session:
        ai_cfg = AIConfig(
            id=1,
            provider="Gemini",
            gemini_api_key="gem_pg_key_1",
            image_generation_provider="KIE",
            kie_api_key="kie_pg_key_1",
            kie_credit_alert_threshold=2.0,
            kie_credit_alert_sent=False,
            allow_fallback=True,
            fallback_provider="OpenAI",
            openai_api_key="openai_pg_key_1",
        )
        session.add(ai_cfg)
        await session.commit()

    yield session_maker
    await engine.dispose()


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_incident_lifecycle_and_cas_outbox(pg_test_session_maker, monkeypatch):
    """Real PostgreSQL: full incident lifecycle, outbox CAS dispatch, and recovery on actual PostgreSQL."""
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    bot_token = "8281390498:AAFtest_telegram_token_123456789012"
    monkeypatch.setenv("BOT_TOKEN", bot_token)

    err = InsufficientBalanceError("KIE credits empty")
    err.provider = "KIE"

    # 1. Record failure on PostgreSQL
    inc = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=err,
        operation="image_generation",
        model="seedream-v4",
        session_maker=pg_test_session_maker,
    )
    assert inc is not None
    assert inc.status == "active"

    # Verify row in real PostgreSQL
    async with pg_test_session_maker() as session:
        db_inc = await session.get(AIProviderIncidentModel, inc.id)
        assert db_inc is not None
        assert db_inc.provider == "KIE"

        outbox_rows = (await session.execute(
            select(AIIncidentNotificationModel).where(AIIncidentNotificationModel.incident_id == inc.id)
        )).scalars().all()
        assert len(outbox_rows) > 0
        assert outbox_rows[0].status == "pending"

    # 2. Worker processes notifications on PostgreSQL
    mock_bot = AsyncMock(spec=Bot)
    mock_bot.token = bot_token
    mock_bot.get_me = AsyncMock(return_value=TgUser(id=8281390498, is_bot=True, first_name="bot", username="someonewithyou01_bot"))
    mock_bot.send_message.return_value = True

    delivered = await process_ai_incident_notifications(
        mock_bot,
        session_maker=pg_test_session_maker,
    )
    assert delivered > 0

    async with pg_test_session_maker() as session:
        outbox_rows = (await session.execute(
            select(AIIncidentNotificationModel).where(AIIncidentNotificationModel.incident_id == inc.id)
        )).scalars().all()
        for r in outbox_rows:
            assert r.status == "delivered"
            assert r.delivered_at is not None

    # 3. Operational success resolves incident on PostgreSQL
    fp = inc.credential_scope
    resolved = await record_ai_operation_success(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        credential_scope=fp,
        operation="image_generation",
        model="seedream-v4",
        session_maker=pg_test_session_maker,
    )
    assert resolved is True

    async with pg_test_session_maker() as session:
        db_inc = await session.get(AIProviderIncidentModel, inc.id)
        assert db_inc.status == "resolved"
        assert db_inc.resolved_at is not None


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_concurrent_failure_atomic_upsert(pg_test_session_maker, monkeypatch):
    """Real PostgreSQL: 10 concurrent tasks calling record_ai_provider_failure produce exactly 1 row with occurrence_count=10."""
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", "8281390498:AAFtest_telegram_token_123456789012")

    err = InsufficientBalanceError("KIE credits empty")
    err.provider = "KIE"

    # Launch 10 concurrent operations against real PostgreSQL
    async def worker_task():
        return await record_ai_provider_failure(
            origin_instance_id="tg_someonewithyou01_bot_new",
            provider="KIE",
            error=err,
            operation="image_generation",
            model="seedream-v4",
            session_maker=pg_test_session_maker,
        )

    results = await asyncio.gather(*(worker_task() for _ in range(10)))
    assert all(r is not None for r in results)

    async with pg_test_session_maker() as session:
        incidents = (await session.execute(select(AIProviderIncidentModel))).scalars().all()
        assert len(incidents) == 1
        assert incidents[0].occurrence_count == 10


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_cas_lease_expiration_and_recovery(pg_test_session_maker, monkeypatch):
    """Real PostgreSQL: worker lease expires, allowing recovery worker to safely claim and deliver."""
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    bot_token = "8281390498:AAFtest_telegram_token_123456789012"
    monkeypatch.setenv("BOT_TOKEN", bot_token)

    err = InsufficientBalanceError("KIE credits empty")
    err.provider = "KIE"
    inc = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=err,
        operation="image_generation",
        model="seedream-v4",
        session_maker=pg_test_session_maker,
    )

    # Manually simulate a crashed worker holding an expired processing lease on PostgreSQL
    async with pg_test_session_maker() as session:
        notif = (await session.execute(
            select(AIIncidentNotificationModel).where(AIIncidentNotificationModel.incident_id == inc.id)
        )).scalars().first()
        notif.status = "processing"
        notif.claim_token = "dead_worker_token"
        notif.lease_until = datetime.utcnow() - timedelta(seconds=10)
        await session.commit()

    mock_bot = AsyncMock(spec=Bot)
    mock_bot.token = bot_token
    mock_bot.get_me = AsyncMock(return_value=TgUser(id=8281390498, is_bot=True, first_name="bot", username="someonewithyou01_bot"))
    mock_bot.send_message.return_value = True

    # Recovery worker runs
    delivered = await process_ai_incident_notifications(
        mock_bot,
        session_maker=pg_test_session_maker,
    )
    assert delivered > 0

    async with pg_test_session_maker() as session:
        updated_notif = await session.get(AIIncidentNotificationModel, notif.id)
        assert updated_notif.status == "delivered"
        assert updated_notif.claim_token != "dead_worker_token"


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_partial_delivery_and_restart_retry(pg_test_session_maker, monkeypatch):
    """Real PostgreSQL: 3 admins (1 success, 1 temporary fail, 1 blocked). Retry after restart delivers only to remaining."""
    from aiogram.exceptions import TelegramForbiddenError, TelegramNetworkError

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    bot_token = "8281390498:AAFtest_telegram_token_123456789012"
    monkeypatch.setenv("BOT_TOKEN", bot_token)

    err = InsufficientBalanceError("KIE credits empty")
    err.provider = "KIE"
    inc = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=err,
        operation="image_generation",
        model="seedream-v4",
        session_maker=pg_test_session_maker,
    )

    # Re-seed exactly 3 recipient notifications in PostgreSQL
    admin_success = 272982544
    admin_retry = 979514796
    admin_blocked = 806750628

    async with pg_test_session_maker() as session:
        # Clear existing notifications
        await session.execute(text("DELETE FROM ai_incident_notifications"))
        now = datetime.utcnow()
        for idx, adm in enumerate([admin_success, admin_retry, admin_blocked]):
            session.add(AIIncidentNotificationModel(
                incident_id=inc.id,
                event_type="opened",
                cycle_index=0,
                recipient_id=adm,
                authorized_instance_id="tg_someonewithyou01_bot_new",
                message_text="Alert text",
                status="pending",
                attempts=0,
                max_attempts=5,
                next_retry_at=now,
            ))
        await session.commit()

    # Dispatch: admin_success succeeds, admin_retry gets network error, admin_blocked gets forbidden
    mock_bot = AsyncMock(spec=Bot)
    mock_bot.token = bot_token
    mock_bot.get_me = AsyncMock(return_value=TgUser(id=8281390498, is_bot=True, first_name="bot", username="someonewithyou01_bot"))

    async def fake_send(chat_id, text, **kwargs):
        if chat_id == admin_success:
            return True
        elif chat_id == admin_retry:
            raise TelegramNetworkError(method=SendMessage(chat_id=chat_id, text=text), message="Connection reset")
        elif chat_id == admin_blocked:
            raise TelegramForbiddenError(method=SendMessage(chat_id=chat_id, text=text), message="Forbidden: bot was blocked by the user")

    mock_bot.send_message.side_effect = fake_send

    delivered_1 = await process_ai_incident_notifications(
        mock_bot,
        session_maker=pg_test_session_maker,
    )
    assert delivered_1 == 1

    async with pg_test_session_maker() as session:
        rows = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        by_adm = {r.recipient_id: r for r in rows}
        assert by_adm[admin_success].status == "delivered"
        assert by_adm[admin_retry].status == "pending"
        assert by_adm[admin_retry].last_error == "telegram_network_error"
        assert by_adm[admin_blocked].status == "failed_terminal"
        assert by_adm[admin_blocked].last_error == "telegram_bot_blocked"

        # Advance retry timestamp for admin_retry so next sweep picks it up
        by_adm[admin_retry].next_retry_at = datetime.utcnow() - timedelta(seconds=1)
        await session.commit()

    # Restart sweep: mock_bot now succeeds for admin_retry
    mock_bot.send_message.side_effect = None
    mock_bot.send_message.return_value = True

    delivered_2 = await process_ai_incident_notifications(
        mock_bot,
        session_maker=pg_test_session_maker,
    )
    assert delivered_2 == 1

    async with pg_test_session_maker() as session:
        rows = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        by_adm = {r.recipient_id: r for r in rows}
        assert by_adm[admin_success].status == "delivered"
        assert by_adm[admin_retry].status == "delivered"
        assert by_adm[admin_blocked].status == "failed_terminal"


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_key_rotation_superseded_no_false_alert(pg_test_session_maker, monkeypatch):
    """Real PostgreSQL: rotating API key transitions active PostgreSQL incidents to superseded without false alert."""
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", "8281390498:AAFtest_telegram_token_123456789012")

    err = InsufficientBalanceError("KIE credits empty")
    err.provider = "KIE"
    inc = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=err,
        operation="image_generation",
        model="seedream-v4",
        credential_scope="old_pg_scope_1",
        session_maker=pg_test_session_maker,
    )
    assert inc is not None

    superseded = await handle_key_rotation(
        origin_instance_id=None,
        provider="KIE",
        new_key="new_pg_secret_kie_key_2026",
        base_url="https://api.kie.ai",
        session_maker=pg_test_session_maker,
    )
    assert superseded == 1

    async with pg_test_session_maker() as session:
        db_inc = await session.get(AIProviderIncidentModel, inc.id)
        assert db_inc.status == "superseded"
        assert db_inc.resolved_at is not None

        notifs = (await session.execute(
            select(AIIncidentNotificationModel).where(AIIncidentNotificationModel.incident_id == inc.id)
        )).scalars().all()
        for n in notifs:
            assert n.status == "cancelled"
            assert n.last_error == "superseded_by_key_rotation"
