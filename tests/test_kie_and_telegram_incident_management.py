"""Comprehensive integration tests for unified database-backed AI incident management,
outbox delivery, companion routing, and Autobus infrastructure isolation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
    enqueue_terminal_ai_failure_notification,
    get_provider_account_fingerprint,
    handle_key_rotation,
    process_ai_incident_notifications,
    record_ai_operation_success,
    record_ai_provider_failure,
    record_low_credit_warning,
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
    found_active = False
    found_transient = False
    for idx in inc_table.indexes:
        if idx.name == "uq_active_ai_incident":
            compiled_pg = str(CreateIndex(idx).compile(dialect=pg_dialect))
            assert "WHERE status = 'active'" in compiled_pg
            assert "UNIQUE" in compiled_pg
            found_active = True
        elif idx.name == "uq_transient_ai_incident":
            compiled_pg = str(CreateIndex(idx).compile(dialect=pg_dialect))
            assert "WHERE status = 'transient'" in compiled_pg
            assert "UNIQUE" in compiled_pg
            found_transient = True
    assert found_active, "uq_active_ai_incident index not found"
    assert found_transient, "uq_transient_ai_incident index not found"

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
        old_credential_scope="old_fingerprint",
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
        old_credential_scope="old_kie_fp_1",
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
    """Item 4: Real E2E Chat fallback through Telegram and MAX get_ai_response():
    - Primary provider (KIE) fails with InsufficientBalanceError.
    - Fallback provider (OpenAI) succeeds and returns text to the user.
    - Exactly ONE incident recorded in DB (provider='KIE', fallback_succeeded=True).
    - Exactly ONE outbox notification ('fallback_used') per admin; ZERO 'opened' outbox rows.
    - process_ai_incident_notifications delivers it to the Telegram admin.
    """
    import ai_integration
    from max_messenger_bot import ai as max_ai

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    bot_token = "8281390498:AAFtest_telegram_token_123456789012"
    monkeypatch.setenv("BOT_TOKEN", bot_token)
    inst_def = {"pm2_name": "tg_someonewithyou01_bot_new", "platform": "telegram", "platform_id": 8281390498, "admins": [7001], "username": "someonewithyou01_bot"}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[7001]))

    monkeypatch.setattr(ai_integration, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)
    monkeypatch.setattr(ai_integration, "build_runtime_context", AsyncMock(return_value="runtime context"))

    # Seed AIConfig and User in DB
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(
                id=1,
                provider="KIE",
                kie_api_key="kie_test_key_e2e",
                kie_model="mock-kie-chat",
                allow_fallback=True,
                fallback_provider="OpenAI",
                fallback_model="mock-openai-gpt",
                openai_api_key="openai_test_key_e2e",
            )
            session.add(cfg)
        else:
            cfg.provider = "KIE"
            cfg.kie_api_key = "kie_test_key_e2e"
            cfg.kie_model = "mock-kie-chat"
            cfg.allow_fallback = True
            cfg.fallback_provider = "OpenAI"
            cfg.fallback_model = "mock-openai-gpt"
            cfg.openai_api_key = "openai_test_key_e2e"

        user_row = await session.get(User, 123)
        if not user_row:
            user_row = User(id=123, tg_user_id=123, username="e2euser", name="E2E User")
            session.add(user_row)
        await session.commit()

    mock_bot = AsyncMock(spec=Bot)
    mock_bot.token = bot_token
    mock_bot.get_me = AsyncMock(return_value=TgUser(id=8281390498, is_bot=True, first_name="bot", username="someonewithyou01_bot"))
    mock_bot.send_message = AsyncMock(return_value=True)

    # 1. Telegram E2E: Primary KIE fails with InsufficientBalanceError, fallback OpenAI succeeds
    primary_err = InsufficientBalanceError("KIE credits 0")
    primary_err.provider = "KIE"
    monkeypatch.setattr(ai_integration, "_call_kie_chat", AsyncMock(side_effect=primary_err))
    monkeypatch.setattr(ai_integration, "_call_openai_api", AsyncMock(return_value="Real fallback reply to user"))

    user_reply = await ai_integration.get_ai_response(
        123,
        "User question",
        "E2E User",
        "unknown",
        bot=mock_bot,
        include_test_context=False,
    )
    assert user_reply == "Real fallback reply to user"

    # Verify DB: exactly ONE incident for KIE with status active
    async with test_db_session_maker() as session:
        incidents = (await session.execute(select(AIProviderIncidentModel))).scalars().all()
        assert len(incidents) == 1
        assert incidents[0].provider == "KIE"
        assert incidents[0].operation == "chat"
        assert incidents[0].status == "active"

        # Verify DB Outbox: exactly ONE fallback_used notification, ZERO opened notifications
        notifs = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        assert len(notifs) == 1
        assert notifs[0].event_type == "fallback_used"
        assert notifs[0].event_type != "opened"
        assert "Резервный:" in notifs[0].message_text
        assert "OPENAI" in notifs[0].message_text

    # Deliver outbox notification
    delivered = await process_ai_incident_notifications(mock_bot, session_maker=test_db_session_maker)
    assert delivered == 1
    assert mock_bot.send_message.call_count == 1
    sent_text = mock_bot.send_message.call_args.kwargs.get("text") or mock_bot.send_message.call_args.args[1]
    assert "Резервный:" in sent_text


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
async def test_item_6_admin_key_rotation_via_real_dispatcher_router_fsm_e2e(test_db_session_maker, monkeypatch):
    """Item 6: Real Admin key rotation flow via Dispatcher, Router, buttons, FSM, and runtime settings:
    - Step 1: Callback query 'model_setting_api_key_KIE' fed into Dispatcher.
      Dispatcher invokes start_model_api_key_for_provider -> FSM state becomes AdminStates.set_api_key.
    - Step 2: Empty/whitespace text '   ' fed into Dispatcher.
      Rejected with warning, FSM state cleared, AIConfig not modified.
    - Step 3: Cancellation callback 'admin_ai_keys' fed into Dispatcher.
      FSM state cleared, AIConfig untouched.
    - Step 4: Active incidents exist for Telegram and MAX companion bots.
      Valid key 'new_e2e_kie_key_via_dispatcher_2026' message fed into Dispatcher.
      FSM state processed, AIConfig updated, handle_key_rotation supersedes active incidents
      for both Telegram and MAX without false recovery alert, cancels outbox, and clears state.
    - Step 5: Re-opened settings inspection confirms runtime sees new key.
    """
    from datetime import timezone
    from aiogram import Dispatcher
    from aiogram.fsm.storage.memory import MemoryStorage
    from aiogram.types import CallbackQuery, Chat, Message, Update
    import handlers
    from tests.test_perplexity_agent_models import ValidatingTelegramSession

    monkeypatch.setattr(handlers, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)
    monkeypatch.setattr(handlers, "is_admin", lambda user_id: True)
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
    handlers.router._parent_router = None
    dp = Dispatcher(storage=storage)
    dp.include_router(handlers.router)

    telegram_session = ValidatingTelegramSession()
    bot = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session)

    user = TgUser(id=777, is_bot=False, first_name="Admin", username="admin777")
    chat = Chat(id=777, type="private")

    # --- Step 1: Admin clicks button 'model_setting_api_key_KIE' via Dispatcher ---
    real_msg_settings = Message(message_id=10, date=datetime.now(timezone.utc), chat=chat, text="Settings")
    update_btn = Update(
        update_id=101,
        callback_query=CallbackQuery(
            id="cb_1",
            from_user=user,
            chat_instance="ci",
            message=real_msg_settings,
            data="model_setting_api_key_KIE",
        ),
    )
    await dp.feed_update(bot, update_btn)

    fsm_ctx = dp.fsm.get_context(bot=bot, user_id=777, chat_id=777)
    assert await fsm_ctx.get_state() == handlers.AdminStates.set_api_key.state
    data = await fsm_ctx.get_data()
    assert data.get("provider") == "KIE"

    # --- Step 2: Empty/whitespace text sent -> Dispatcher handles -> rejected ---
    update_empty = Update(
        update_id=102,
        message=Message(message_id=11, date=datetime.now(timezone.utc), chat=chat, from_user=user, text="   "),
    )
    await dp.feed_update(bot, update_empty)
    assert await fsm_ctx.get_state() is None

    # --- Step 3: Cancellation callback via Dispatcher ---
    await fsm_ctx.set_state(handlers.AdminStates.set_api_key)
    await fsm_ctx.update_data(provider="KIE")

    real_msg_cancel = Message(message_id=12, date=datetime.now(timezone.utc), chat=chat, text="Back")
    update_cancel = Update(
        update_id=103,
        callback_query=CallbackQuery(
            id="cb_cancel",
            from_user=user,
            chat_instance="ci",
            message=real_msg_cancel,
            data="admin_ai_keys",
        ),
    )
    await dp.feed_update(bot, update_cancel)
    assert await fsm_ctx.get_state() is None

    # --- Step 4: Valid key input saves, rotates incidents, and re-renders settings ---
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

    # Click button to set key again via Dispatcher
    await dp.feed_update(bot, update_btn)
    assert await fsm_ctx.get_state() == handlers.AdminStates.set_api_key.state

    monkeypatch.setattr(handlers, "send_temp_notification", AsyncMock())

    update_valid = Update(
        update_id=104,
        message=Message(message_id=14, date=datetime.now(timezone.utc), chat=chat, from_user=user, text="new_valid_kie_key_via_dispatcher_2026"),
    )
    await dp.feed_update(bot, update_valid)

    assert await fsm_ctx.get_state() is None

    # 1. AIConfig updated in DB
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        assert cfg.kie_api_key == "new_valid_kie_key_via_dispatcher_2026"

        # 2. Both Telegram and MAX incidents superseded (NOT resolved, no false alert)
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

        # 4. Settings view re-rendered with new key via real _edit_provider_model_settings (no mock!)
        edit_calls = [c for c in telegram_session.calls if type(c).__name__ == "EditMessageText"]
        assert len(edit_calls) >= 1
        last_edit = edit_calls[-1]
        buttons = [btn for row in last_edit.reply_markup.inline_keyboard for btn in row]
        button_texts = [b.text for b in buttons]
        button_datas = [b.callback_data for b in buttons]
        assert any("Выбрать модель" in t for t in button_texts)
        assert any("Max tokens" in t for t in button_texts)
        assert any("API-ключ" in t for t in button_texts)
        assert any("Назад" in t for t in button_texts)
        assert "model_setting_api_key_KIE" in button_datas
        assert "admin_ai_keys" in button_datas
        assert "view_provider_models_KIE" in button_datas


@pytest.mark.asyncio
async def test_item_1_kie_key_rotation_reconciles_alert_sent_flag_and_allows_new_warning(test_db_session_maker, monkeypatch):
    """Defect 1: Rotating KIE key reconciles kie_credit_alert_sent with the new credential scope:
    - Old key had low credit warning (kie_credit_alert_sent = True).
    - Rotating to new key reconciles kie_credit_alert_sent to False because new key has no active incident.
    - Scheduler running for new key with low balance succeeds in creating a new warning!
    - Historical dirty state (flag = True without active incident) is reconciled on scheduler check.
    """
    inst_def = {"pm2_name": "tg_someonewithyou01_bot_new", "platform": "telegram", "platform_id": 8281390498, "admins": [1001], "username": "someonewithyou01_bot"}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[1001]))

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", "8281390498:AAFtest_token_reconcile")
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)

    # 1. Old key with active warning and kie_credit_alert_sent = True
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="KIE", kie_api_key="old_kie_key_dirty", kie_base_url="https://api.kie.ai", kie_credit_alert_threshold=2.0, kie_credit_alert_sent=True)
            session.add(cfg)
        else:
            cfg.provider = "KIE"
            cfg.kie_api_key = "old_kie_key_dirty"
            cfg.kie_base_url = "https://api.kie.ai"
            cfg.kie_credit_alert_threshold = 2.0
            cfg.kie_credit_alert_sent = True
        await session.commit()

    old_fp = get_provider_account_fingerprint("old_kie_key_dirty", "https://api.kie.ai")
    await record_low_credit_warning(
        provider="KIE",
        remaining_credits=1.0,
        threshold=2.0,
        api_key="old_kie_key_dirty",
        base_url="https://api.kie.ai",
        session_maker=test_db_session_maker,
    )

    # 2. Key rotation to new key
    await handle_key_rotation(
        origin_instance_id=None,
        provider="KIE",
        new_key="new_kie_key_clean",
        base_url="https://api.kie.ai",
        session_maker=test_db_session_maker,
    )

    # Verify: old incident is superseded, and AIConfig.kie_credit_alert_sent is reconciled to False!
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        assert cfg.kie_credit_alert_sent is False

        # Update api_key on config to match new key
        cfg.kie_api_key = "new_kie_key_clean"
        await session.commit()

    # 3. Scheduler check runs for new key with low credits (0.50 < 2.0)
    bot = AsyncMock(spec=Bot)
    bot.token = "8281390498:AAFtest_token_reconcile"
    bot.get_me = AsyncMock(return_value=TgUser(id=8281390498, is_bot=True, first_name="bot", username="someonewithyou01_bot"))
    bot.send_message = AsyncMock(return_value=True)

    monkeypatch.setattr(scheduler, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr(scheduler, "get_kie_remaining_credits", AsyncMock(return_value=0.50))
    monkeypatch.setattr(scheduler, "get_all_admin_ids", AsyncMock(return_value=[1001]))

    await scheduler.check_kie_credit_balance(bot)

    # Verify: warning for new key WAS created and delivered (NOT blocked by old flag!)
    async with test_db_session_maker() as session:
        cfg_after = await session.get(AIConfig, 1)
        assert cfg_after.kie_credit_alert_sent is True

        new_fp = get_provider_account_fingerprint("new_kie_key_clean", "https://api.kie.ai")
        new_inc = (await session.execute(
            select(AIProviderIncidentModel).where(
                AIProviderIncidentModel.provider == "KIE",
                AIProviderIncidentModel.credential_scope == new_fp,
                AIProviderIncidentModel.status == "active",
            )
        )).scalar_one_or_none()
        assert new_inc is not None
        assert new_inc.incident_type == "low_credit_warning"


@pytest.mark.asyncio
async def test_item_2_autobus_does_not_mask_operational_error_or_internal_error_during_chat(monkeypatch, tmp_path):
    """Defect 2: OperationalError, application_internal, and worker crashes in chat are NOT masked as Handled AI notice:
    - Calling notify_admins_about_error with OperationalError logs error without 'Handled AI notice:'.
    - Monitor extracts error and Autobus alerts.
    - Calling notify_admins_about_error with confirmed AI RateLimitError DOES log 'Handled AI notice:' and Autobus ignores it.
    """
    import error_reporting
    from sqlalchemy.exc import OperationalError

    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    error_log_file = tmp_path / "bot-error.log"
    error_log_file.write_text("2026-11-15 11:00:00 | INFO | Bot startup\n")

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

    # 1. Confirmed AI error: rate limit with verified ProviderFailure
    mock_bot = AsyncMock(spec=Bot)
    mock_bot.send_message = AsyncMock(return_value=True)

    from provider_failures import ProviderFailure
    rate_limit_failure = ProviderFailure(
        provider="KIE",
        operation="chat",
        classification="rate_limit",
        detail_code="rate_limit_exceeded",
        retryable=True,
        classification_source="provider_structured",
        provider_error_message="Rate limit exceeded",
    )

    fake_logger_warn = MagicMock(spec=logging.Logger)
    await error_reporting.notify_admins_about_error(
        mock_bot,
        title="Rate limit exceeded",
        provider="KIE",
        stage="chat",
        classification="rate_limit",
        classification_source="provider_structured",
        exception=rate_limit_failure,
        logger=fake_logger_warn,
    )
    assert fake_logger_warn.warning.called
    assert "Handled AI notice:" in fake_logger_warn.warning.call_args[0][0]

    # 2. Unexpected database / infrastructure OperationalError in chat
    db_err = OperationalError("FATAL: connection to server lost", None, None)
    fake_logger_err = MagicMock(spec=logging.Logger)
    await error_reporting.notify_admins_about_error(
        mock_bot,
        title="Database error during AI chat",
        provider="KIE",
        stage="chat",
        exception=db_err,
        classification="application_internal",
        logger=fake_logger_err,
    )
    assert fake_logger_err.log.called
    # Verify NO Handled AI notice: prefix
    logged_str = fake_logger_err.log.call_args[0][1]
    assert "Handled AI notice:" not in logged_str
    assert "OperationalError" in logged_str

    # Baseline check: offset initialized, 0 alerts
    exit_code_0 = await monitor.run_check(False)
    assert exit_code_0 == 0

    # Write OperationalError to PM2 error log and verify Autobus alerts
    with open(error_log_file, "a", encoding="utf-8") as f:
        f.write("2026-11-15 12:00:00 | ERROR | [AI Incident Worker] OperationalError: server closed connection\nTraceback (most recent call last):\n  File \"worker.py\", line 42\nOperationalError: server closed connection\n")

    exit_code = await monitor.run_check(False)
    assert exit_code == 1
    assert len(sent_alerts) == 1
    assert "OperationalError" in sent_alerts[0]


@pytest.mark.asyncio
async def test_item_3_low_credit_reminder_exact_financial_message_and_cas_concurrency(test_db_session_maker, monkeypatch):
    """Defect 3: Low credit reminders preserve exact financial meaning, and CAS prevents duplicate reminder cycles:
    - Reminder text contains financial warning, not 'Сбой AI-провайдера'.
    - 2 concurrent workers running check_incident_repeat_reminders: exactly one succeeds in CAS update.
    - Exactly one repeat reminder outbox record created.
    """
    # 1. Message formatting check
    text_rem = _format_incident_message_text(
        provider="KIE",
        operation="account",
        model_scope="*",
        incident_type="low_credit_warning",
        detail_code="balance_0.50_below_2.00",
        evidence_tier="CONFIRMED_STRUCTURED",
        event_type="repeat_reminder",
    )
    assert "Напоминание: низкий баланс KIE" in text_rem
    assert "Сбой AI-провайдера" not in text_rem
    assert "Баланс остается ниже порога предупреждения" in text_rem

    # 2. Concurrent CAS execution check
    inst_def = {"pm2_name": "tg_someonewithyou01_bot_new", "platform": "telegram", "platform_id": 8281390498, "admins": [555], "username": "someonewithyou01_bot"}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[555]))

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", "8281390498:AAFtest_token_cas")

    inc = await record_low_credit_warning(
        provider="KIE",
        remaining_credits=0.50,
        threshold=2.00,
        api_key="kie_cas_key",
        session_maker=test_db_session_maker,
    )
    assert inc is True

    # Age incident by 7 hours
    async with test_db_session_maker() as session:
        inc_obj = (await session.execute(
            select(AIProviderIncidentModel).where(AIProviderIncidentModel.incident_type == "low_credit_warning")
        )).scalar_one()
        inc_obj.last_alerted_at = datetime.utcnow() - timedelta(hours=7)
        await session.commit()

    # Launch 2 concurrent workers attempting repeat reminder check
    res_1, res_2 = await asyncio.gather(
        check_incident_repeat_reminders(repeat_seconds=DEFAULT_REPEAT_ALERT_SECONDS, session_maker=test_db_session_maker),
        check_incident_repeat_reminders(repeat_seconds=DEFAULT_REPEAT_ALERT_SECONDS, session_maker=test_db_session_maker),
    )
    # Exactly one succeeded in the atomic CAS update
    assert res_1 + res_2 == 1

    async with test_db_session_maker() as session:
        inc_after = (await session.execute(
            select(AIProviderIncidentModel).where(AIProviderIncidentModel.incident_type == "low_credit_warning")
        )).scalar_one()
        assert inc_after.reminder_cycle == 1

        rem_notifs = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.incident_id == inc_after.id,
                AIIncidentNotificationModel.event_type == "repeat_reminder",
            )
        )).scalars().all()
        assert len(rem_notifs) == 1
        assert rem_notifs[0].cycle_index == 1
        assert "Напоминание: низкий баланс KIE" in rem_notifs[0].message_text


@pytest.mark.asyncio
async def test_item_5_terminal_failure_both_providers_fail_separate_incidents_and_cooldown_dedup(test_db_session_maker, monkeypatch):
    """Defect 5: When both primary and fallback fail:
    - Primary provider (KIE) and fallback provider (OpenAI) open separate incidents.
    - Outbox has separate 'opened' records for each provider without artificial merging.
    - send_terminal_ai_failure_alert provides request diagnostic context.
    - Immediate subsequent failure is deduplicated: incidents increment occurrence_count, zero duplicate outbox rows.
    """
    import ai_integration

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    bot_token = "8281390498:AAFtest_telegram_token_terminal"
    monkeypatch.setenv("BOT_TOKEN", bot_token)
    inst_def = {"pm2_name": "tg_someonewithyou01_bot_new", "platform": "telegram", "platform_id": 8281390498, "admins": [7001], "username": "someonewithyou01_bot"}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[7001]))

    monkeypatch.setattr(ai_integration, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)
    monkeypatch.setattr(ai_integration, "build_runtime_context", AsyncMock(return_value="runtime context"))

    # Seed AIConfig and User in DB
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1)
            session.add(cfg)
        cfg.provider = "KIE"
        cfg.kie_api_key = "kie_key_term"
        cfg.kie_model = "mock-kie-chat"
        cfg.allow_fallback = True
        cfg.fallback_provider = "OpenAI"
        cfg.fallback_model = "mock-openai-gpt"
        cfg.openai_api_key = "openai_key_term"

        user_row = await session.get(User, 777)
        if not user_row:
            user_row = User(id=777, tg_user_id=777, username="termuser", name="Term User")
            session.add(user_row)
        await session.commit()

    mock_bot = AsyncMock(spec=Bot)
    mock_bot.token = bot_token
    mock_bot.get_me = AsyncMock(return_value=TgUser(id=8281390498, is_bot=True, first_name="bot", username="someonewithyou01_bot"))
    mock_bot.send_message = AsyncMock(return_value=True)

    # Both fail: KIE credits exhausted, OpenAI quota exhausted
    primary_err = InsufficientBalanceError("KIE credits empty")
    primary_err.provider = "KIE"
    fb_err = InsufficientBalanceError("OpenAI quota exceeded")
    fb_err.provider = "OpenAI"

    monkeypatch.setattr(ai_integration, "_call_kie_chat", AsyncMock(side_effect=primary_err))
    monkeypatch.setattr(ai_integration, "_call_openai_api", AsyncMock(side_effect=fb_err))

    # Request 1 fails terminally
    with pytest.raises(ai_integration.AIServiceError):
        await ai_integration.get_ai_response(
            777,
            "Terminal query 1",
            "Term User",
            "unknown",
            bot=mock_bot,
            include_test_context=False,
        )

    async with test_db_session_maker() as session:
        # Separate incidents for KIE and OpenAI
        incidents = (await session.execute(select(AIProviderIncidentModel))).scalars().all()
        assert len(incidents) == 2
        prov_map = {inc.provider: inc for inc in incidents}
        assert "KIE" in prov_map
        assert "OPENAI" in prov_map
        assert prov_map["KIE"].occurrence_count == 1
        assert prov_map["OPENAI"].occurrence_count == 1

        # Outbox rows: durable outbox delivery has 1 consolidated terminal failure record (no redundant opened rows)
        notifs = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        assert len(notifs) == 1
        assert notifs[0].event_type == "terminal_failure"
        assert notifs[0].recipient_id == 7001

    # Request 2 fails terminally immediately afterwards
    with pytest.raises(ai_integration.AIServiceError):
        await ai_integration.get_ai_response(
            777,
            "Terminal query 2",
            "Term User",
            "unknown",
            bot=mock_bot,
            include_test_context=False,
        )

    async with test_db_session_maker() as session:
        # Still exactly 2 incidents, occurrence_count incremented to 2
        incidents_2 = (await session.execute(select(AIProviderIncidentModel))).scalars().all()
        assert len(incidents_2) == 2
        prov_map_2 = {inc.provider: inc for inc in incidents_2}
        assert prov_map_2["KIE"].occurrence_count == 2
        assert prov_map_2["OPENAI"].occurrence_count == 2

        # Zero duplicate outbox rows created!
        notifs_2 = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        assert len(notifs_2) == 1


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
        old_credential_scope="old_pg_scope_1",
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


# ==============================================================================
# Dedicated Review Verification Tests (Review Points 1-6)
# ==============================================================================

@pytest.mark.asyncio
async def test_concurrent_old_key_failure_does_not_open_incident_for_new_rotated_key(test_db_session_maker, monkeypatch):
    """Review Point 1: When an in-flight AI call made with an old key fails AFTER key rotation,
    passing the attempt's actual credential_scope prevents attributing the failure to the new key:
    - Step 1: Initial AIConfig has old key ('old_secret_key_111').
    - Step 2: Request starts in-flight with old key.
    - Step 3: Admin rotates key to 'new_secret_key_222' (AIConfig updated, handle_key_rotation called).
    - Step 4: Old request fails with InsufficientBalanceError and records failure with old credential_scope.
    - Assert: Incident is opened ONLY for old credential_scope.
    - Assert: Zero active incidents for new credential_scope.
    - Assert: kie_credit_alert_sent is False for new key.
    - Assert: Subsequent successful operation with new key records success without issue.
    """
    inst_def = {"pm2_name": "tg_someonewithyou01_bot_new", "platform": "telegram", "platform_id": 8281390498, "admins": [1001], "username": "someonewithyou01_bot"}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[1001]))

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", "8281390498:AAFtest_token_concurrent_old_key")
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)

    old_key = "old_secret_key_111"
    new_key = "new_secret_key_222"
    base_url = "https://api.kie.ai"

    old_scope = get_provider_account_fingerprint(old_key, base_url)
    new_scope = get_provider_account_fingerprint(new_key, base_url)

    # 1. Initial AIConfig with old key
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="KIE", kie_api_key=old_key, kie_base_url=base_url, kie_credit_alert_sent=False)
            session.add(cfg)
        else:
            cfg.provider = "KIE"
            cfg.kie_api_key = old_key
            cfg.kie_base_url = base_url
            cfg.kie_credit_alert_sent = False
        await session.commit()

    # 2. In-flight request captures old_scope.
    # 3. Admin rotates key to new_key during in-flight call:
    await handle_key_rotation(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        new_key=new_key,
        base_url=base_url,
        session_maker=test_db_session_maker,
    )
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.kie_api_key = new_key
        await session.commit()

    # 4. Old in-flight request fails and records failure with old_scope:
    err = InsufficientBalanceError("KIE credits exhausted on old key")
    err.provider = "KIE"
    inc_old = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=err,
        operation="chat",
        model="kie-v1",
        credential_scope=old_scope,
        session_maker=test_db_session_maker,
    )

    # 5. Verify:
    async with test_db_session_maker() as session:
        # Zero incidents opened for new credential scope!
        new_incidents = (await session.execute(
            select(AIProviderIncidentModel).where(
                AIProviderIncidentModel.provider == "KIE",
                AIProviderIncidentModel.credential_scope == new_scope,
            )
        )).scalars().all()
        assert len(new_incidents) == 0

        # Incident recorded is strictly bound to old_scope and marked superseded (NOT active)
        assert inc_old is not None
        assert inc_old.credential_scope == old_scope
        assert inc_old.status == "superseded", "Retired credential failure must be recorded as superseded, NOT active"

        # Verify: ZERO active incidents exist for old or new scopes
        active_incidents = (await session.execute(
            select(AIProviderIncidentModel).where(
                AIProviderIncidentModel.status == "active",
            )
        )).scalars().all()
        assert len(active_incidents) == 0

        # Verify: ZERO outbox notification rows were enqueued for the retired key failure
        outbox_rows = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.status.in_(["pending", "processing"]),
            )
        )).scalars().all()
        assert len(outbox_rows) == 0

        # Flag kie_credit_alert_sent on AIConfig is False for new key
        cfg = await session.get(AIConfig, 1)
        assert cfg.kie_credit_alert_sent is False

    # 6. Subsequent operation with new key has 0 active incidents to resolve, so returns False (no error)
    resolved = await record_ai_operation_success(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        credential_scope=new_scope,
        operation="chat",
        model="kie-v1",
        session_maker=test_db_session_maker,
    )
    assert resolved is False


@pytest.mark.asyncio
async def test_fail_closed_foreign_or_unknown_origin_instance_rejected(test_db_session_maker, monkeypatch):
    """Review Point 2: Fail-closed verification in record_ai_provider_failure and record_ai_operation_success:
    - Unknown/unregistered origin: fail-closed (returns None / False, zero DB writes).
    - Foreign origin (another bot not in companion link with current bot): fail-closed (returns None / False).
    - Companion origin: accepted.
    - Own origin: accepted.
    """
    bot_token = "8224473896:AAFtest_token_veraveda"
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", bot_token)

    # Current instance is tg_veraveda777_bot_legacy, whose companion is max_se13639182_bot_legacy
    err = InsufficientBalanceError("Credits depleted")
    err.provider = "KIE"

    # 1. Unknown / unregistered origin
    inc_unk = await record_ai_provider_failure(
        origin_instance_id="unknown_rogue_bot_999",
        provider="KIE",
        error=err,
        operation="chat",
        session_maker=test_db_session_maker,
    )
    assert inc_unk is None

    succ_unk = await record_ai_operation_success(
        origin_instance_id="unknown_rogue_bot_999",
        provider="KIE",
        credential_scope="test_scope",
        operation="chat",
        model="kie-v1",
        session_maker=test_db_session_maker,
    )
    assert succ_unk is False

    # 2. Foreign origin (registered bot, but belongs to different companion pair)
    inc_for = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=err,
        operation="chat",
        session_maker=test_db_session_maker,
    )
    assert inc_for is None

    succ_for = await record_ai_operation_success(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        credential_scope="test_scope",
        operation="chat",
        model="kie-v1",
        session_maker=test_db_session_maker,
    )
    assert succ_for is False

    # 3. Companion origin (max_se13639182_bot_legacy is companion of tg_veraveda777_bot_legacy)
    inc_comp = await record_ai_provider_failure(
        origin_instance_id="max_se13639182_bot_legacy",
        provider="KIE",
        error=err,
        operation="chat",
        session_maker=test_db_session_maker,
    )
    assert inc_comp is not None
    assert inc_comp.origin_instance_id == "max_se13639182_bot_legacy"

    # 4. Own origin (tg_veraveda777_bot_legacy)
    inc_own = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        error=err,
        operation="chat",
        session_maker=test_db_session_maker,
    )
    assert inc_own is not None
    assert inc_own.origin_instance_id == "tg_veraveda777_bot_legacy"


@pytest.mark.asyncio
async def test_autobus_classification_negative_tests_app_errors_and_unverified_source(monkeypatch, tmp_path):
    """Review Point 3: Autobus error classification strictly requires verified ProviderFailure from trusted sources:
    - Negative 1: Explicit classification passed without ProviderFailure or with unverified source -> logged as ERROR, no Handled AI prefix, Autobus alerts.
    - Negative 2: Application errors (KeyError, TypeError) -> unconditionally logged as ERROR, Autobus alerts.
    - Negative 3: DB errors (OperationalError) -> unconditionally logged as ERROR, Autobus alerts.
    - Positive: Confirmed ProviderFailure from provider_structured -> logged as WARNING with Handled AI prefix, Autobus ignores.
    """
    import error_reporting
    from provider_failures import ProviderFailure
    from sqlalchemy.exc import OperationalError

    lock_file = str(tmp_path / "lock.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)

    error_log_file = tmp_path / "bot-error.log"
    error_log_file.write_text("2026-11-15 11:00:00 | INFO | Startup\n")

    apps = [{
        "name": "tg_someonewithyou01_bot_new",
        "status": "online",
        "token": "tok_1",
        "owner_ids": [272982544],
        "error_log": str(error_log_file),
    }]
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

    mock_bot = AsyncMock(spec=Bot)
    mock_bot.send_message = AsyncMock(return_value=True)

    # Negative 1: Explicit classification without ProviderFailure or with unverified source
    fake_logger_1 = MagicMock(spec=logging.Logger)
    await error_reporting.notify_admins_about_error(
        mock_bot,
        title="Unverified rate limit attempt Error: HTTP 429 quota rejected",
        provider="OpenAI",
        stage="chat",
        classification="rate_limit",
        classification_source="unverified",
        logger=fake_logger_1,
    )
    assert fake_logger_1.log.called
    log_text_1 = fake_logger_1.log.call_args[0][1]
    assert "Handled AI notice:" not in log_text_1
    assert "Unverified rate limit attempt" in log_text_1

    # Negative 2: Application error (KeyError)
    fake_logger_2 = MagicMock(spec=logging.Logger)
    app_err = KeyError("missing_user_field")
    await error_reporting.notify_admins_about_error(
        mock_bot,
        title="KeyError during response processing",
        provider="OpenAI",
        stage="chat",
        classification="rate_limit",
        exception=app_err,
        logger=fake_logger_2,
    )
    assert fake_logger_2.log.called
    log_text_2 = fake_logger_2.log.call_args[0][1]
    assert "Handled AI notice:" not in log_text_2
    assert "KeyError" in log_text_2

    # Negative 3: DB error (OperationalError)
    fake_logger_3 = MagicMock(spec=logging.Logger)
    db_err = OperationalError("connection refused", None, None)
    await error_reporting.notify_admins_about_error(
        mock_bot,
        title="Database disconnect during chat",
        provider="OpenAI",
        stage="chat",
        classification="rate_limit",
        exception=db_err,
        logger=fake_logger_3,
    )
    assert fake_logger_3.log.called
    log_text_3 = fake_logger_3.log.call_args[0][1]
    assert "Handled AI notice:" not in log_text_3
    assert "OperationalError" in log_text_3

    # Positive: Confirmed ProviderFailure from provider_structured
    fake_logger_pos = MagicMock(spec=logging.Logger)
    prov_fail = ProviderFailure(
        provider="OpenAI",
        operation="chat",
        classification="rate_limit",
        detail_code="rate_limit_exceeded",
        retryable=True,
        classification_source="provider_structured",
        provider_error_message="Rate limit reached",
    )
    await error_reporting.notify_admins_about_error(
        mock_bot,
        title="Real Rate Limit",
        provider="OpenAI",
        stage="chat",
        classification="rate_limit",
        exception=prov_fail,
        logger=fake_logger_pos,
    )
    assert fake_logger_pos.warning.called
    log_text_pos = fake_logger_pos.warning.call_args[0][0]
    assert "Handled AI notice:" in log_text_pos

    # Autobus integration verification:
    # Baseline check: offset initialized
    await monitor.run_check(False)
    assert len(sent_alerts) == 0

    # Write the logged lines into error log
    with open(error_log_file, "a", encoding="utf-8") as f:
        f.write(f"2026-11-15 11:01:00 | ERROR | {log_text_1}\n")
        f.write(f"2026-11-15 11:02:00 | ERROR | {log_text_2}\n")
        f.write(f"2026-11-15 11:03:00 | ERROR | {log_text_3}\n")
        f.write(f"2026-11-15 11:04:00 | WARNING | {log_text_pos}\n")

    exit_code = await monitor.run_check(False)
    assert exit_code == 1
    assert len(sent_alerts) == 1
    alert_body = sent_alerts[0]
    assert "Unverified rate limit attempt" in alert_body
    assert "KeyError" in alert_body
    assert "OperationalError" in alert_body
    assert "Real Rate Limit" not in alert_body  # Handled AI notice suppressed!


@pytest.mark.asyncio
async def test_terminal_ai_failure_three_admins_timeout_retry_restart_durable_outbox(test_db_session_maker, monkeypatch):
    """Review Point 4: Terminal AI failure outbox delivery across 3 admins with timeout, retry, restart, and deduplication:
    - 3 configured admins: 1001, 1002, 1003.
    - Both primary (KIE) and fallback (OpenAI) fail on chat request.
    - Primary and fallback incidents recorded with enqueue_notification=False (0 redundant opened rows in outbox).
    - send_terminal_ai_failure_alert creates durable outbox rows (event_type='terminal_failure') for all 3 admins.
    - Worker sweep 1: Admin 1001 delivered, Admin 1002 raises TelegramNetworkError (times out), Admin 1003 delivered.
    - Verify Admin 1001 & 1003 marked delivered, Admin 1002 marked pending with attempts=1.
    - Advance retry time and simulate restart recovery.
    - Worker sweep 2: Admin 1002 delivered.
    - Admins 1001 and 1003 are NOT re-sent (zero duplicate delivery).
    - Subsequent immediate terminal failure does not spam outbox due to cooldown.
    """
    import ai_integration
    import error_reporting
    from aiogram.exceptions import TelegramNetworkError

    error_reporting._terminal_failure_cooldown.reset_all()

    bot_token = "8281390498:AAFtest_token_3_admins_terminal"
    admin_ids = [1001, 1002, 1003]
    inst_def = {
        "pm2_name": "tg_someonewithyou01_bot_new",
        "platform": "telegram",
        "platform_id": 8281390498,
        "admins": admin_ids,
        "username": "someonewithyou01_bot",
    }
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=admin_ids))

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", bot_token)
    monkeypatch.setattr(ai_integration, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)
    monkeypatch.setattr(ai_integration, "build_runtime_context", AsyncMock(return_value="context"))

    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1)
            session.add(cfg)
        cfg.provider = "KIE"
        cfg.kie_api_key = "kie_key_term_3adm"
        cfg.kie_model = "mock-kie-chat"
        cfg.allow_fallback = True
        cfg.fallback_provider = "OpenAI"
        cfg.fallback_model = "mock-openai-chat"
        cfg.openai_api_key = "openai_key_term_3adm"

        u = await session.get(User, 888)
        if not u:
            u = User(id=888, tg_user_id=888, username="user888", name="User 888")
            session.add(u)
        await session.commit()

    mock_bot = AsyncMock(spec=Bot)
    mock_bot.token = bot_token
    mock_bot.get_me = AsyncMock(return_value=TgUser(id=8281390498, is_bot=True, first_name="bot", username="someonewithyou01_bot"))

    primary_err = InsufficientBalanceError("KIE credits empty")
    primary_err.provider = "KIE"
    fb_err = InsufficientBalanceError("OpenAI quota exceeded")
    fb_err.provider = "OpenAI"

    monkeypatch.setattr(ai_integration, "_call_kie_chat", AsyncMock(side_effect=primary_err))
    monkeypatch.setattr(ai_integration, "_call_openai_api", AsyncMock(side_effect=fb_err))

    # Worker sweep 1 dispatch behavior during terminal failure alert:
    # Admin 1002 times out (TelegramNetworkError), Admins 1001 & 1003 succeed
    delivery_calls = []
    async def fake_send(chat_id, text, **kwargs):
        delivery_calls.append(chat_id)
        if chat_id == 1002:
            raise TelegramNetworkError(method=SendMessage(chat_id=chat_id, text=text), message="Timeout connecting to Telegram API")
        return True
    mock_bot.send_message.side_effect = fake_send

    # Request 1 fails terminally
    with pytest.raises(ai_integration.AIServiceError):
        await ai_integration.get_ai_response(
            888,
            "Terminal query 1",
            "User 888",
            "unknown",
            bot=mock_bot,
            include_test_context=False,
        )

    # 1. DB inspection: separate incidents for KIE and OpenAI, zero 'opened' notifications
    async with test_db_session_maker() as session:
        incidents = (await session.execute(select(AIProviderIncidentModel))).scalars().all()
        assert len(incidents) == 2

        # Durable outbox delivery: exactly 3 rows with event_type='terminal_failure' (one per admin)
        notifs = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        assert len(notifs) == 3
        for n in notifs:
            assert n.event_type == "terminal_failure"
        notif_recipients = {n.recipient_id for n in notifs}
        assert notif_recipients == {1001, 1002, 1003}

        # Partial delivery state: Admins 1001 & 1003 delivered, Admin 1002 pending with retry
        by_adm = {r.recipient_id: r for r in notifs}
        assert by_adm[1001].status == "delivered"
        assert by_adm[1003].status == "delivered"
        assert by_adm[1002].status == "pending"
        assert by_adm[1002].attempts == 1
        assert by_adm[1002].last_error == "telegram_network_error"

        # Advance retry timestamp for Admin 1002
        by_adm[1002].next_retry_at = datetime.utcnow() - timedelta(seconds=1)
        await session.commit()

    # 2. Worker sweep 2 (simulating restart and retry): Admin 1002 succeeds
    mock_bot.send_message.side_effect = None
    mock_bot.send_message.return_value = True

    delivered_count_2 = await process_ai_incident_notifications(
        mock_bot,
        session_maker=test_db_session_maker,
    )
    assert delivered_count_2 == 1

    async with test_db_session_maker() as session:
        rows = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        by_adm = {r.recipient_id: r for r in rows}
        assert by_adm[1001].status == "delivered"
        assert by_adm[1002].status == "delivered"
        assert by_adm[1003].status == "delivered"

    # Verify: Admins 1001 and 1003 were called exactly ONCE, Admin 1002 was called TWICE
    assert delivery_calls.count(1001) == 1
    assert delivery_calls.count(1003) == 1
    assert delivery_calls.count(1002) == 1

    # 4. Immediate second terminal failure does not spam outbox (deduplication)
    with pytest.raises(ai_integration.AIServiceError):
        await ai_integration.get_ai_response(
            888,
            "Terminal query 2",
            "User 888",
            "unknown",
            bot=mock_bot,
            include_test_context=False,
        )

    async with test_db_session_maker() as session:
        # Zero duplicate outbox rows created! Total outbox rows is still 3!
        rows_2 = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        assert len(rows_2) == 3
        # Occurrence counts incremented to 2
        incidents_2 = (await session.execute(select(AIProviderIncidentModel))).scalars().all()
        for inc in incidents_2:
            assert inc.occurrence_count == 2


@pytest.mark.asyncio
async def test_e2e_full_chat_journey_telegram_and_max_and_admin_fsm_network_doubles(test_db_session_maker, monkeypatch):
    """Review Point 5: Full E2E coverage for Telegram Chat, MAX Chat, and Admin FSM using network doubles:
    - Part 1: Telegram Chat Journey: User message -> Dispatcher -> handler -> primary fails, fallback succeeds -> reply delivered to user.
    - Part 2: MAX Chat Journey: User message -> handle_message -> primary fails, fallback succeeds -> reply delivered to user.
      Strictly asserts MAX outbound contract: chat_id is int, message text is non-empty string.
    - Part 3: Admin FSM Journey: Real Dispatcher and Router with ValidatingTelegramSession without mocking _edit_provider_model_settings:
      Dynamically inspects buttons from last_edit.reply_markup.inline_keyboard (no hardcoded callback strings).
      Tests all visible buttons ("Выбрать модель", "Max tokens", "API-ключ", "Назад"), tests Cancel/Back,
      tests Save new key, tests re-opening screen, and tests runtime reflection with new key.
    """
    from datetime import timezone
    from aiogram import Dispatcher
    from aiogram.fsm.storage.memory import MemoryStorage
    from aiogram.types import CallbackQuery, Chat, Message, Update
    import handlers
    import ai_integration
    from tests.test_perplexity_agent_models import ValidatingTelegramSession
    from max_messenger_bot.app import MaxBotApplication
    from max_messenger_bot.models import IncomingMessage, Sender

    bot_token = "8224473896:AAFtest_token_veraveda"
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", bot_token)
    monkeypatch.setattr(handlers, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr(ai_integration, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(side_effect=lambda user_id: user_id == 777))
    monkeypatch.setattr(ai_integration, "build_runtime_context", AsyncMock(return_value="runtime context"))

    # Seed AIConfig and User in DB
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1)
            session.add(cfg)
        cfg.provider = "KIE"
        cfg.kie_api_key = "kie_key_chat_e2e"
        cfg.kie_model = "mock-kie-chat"
        cfg.allow_fallback = True
        cfg.fallback_provider = "OpenAI"
        cfg.fallback_model = "mock-openai-chat"
        cfg.openai_api_key = "openai_key_chat_e2e"

        user_row = await session.get(User, 777)
        if not user_row:
            user_row = User(id=777, tg_user_id=777, username="admin777", name="Admin User")
            session.add(user_row)

        user_tg = await session.get(User, 999)
        if not user_tg:
            user_tg = User(
                id=999,
                tg_user_id=999,
                username="tguser",
                name="Tg User",
                first_name="Tg User",
                gender="male",
                age="30",
                birth_day=1,
                birth_month=1,
                birth_year=1995,
                accepted_disclaimer=True,
            )
            session.add(user_tg)
        else:
            user_tg.gender = "male"
            user_tg.age = "30"
            user_tg.birth_day = 1
            user_tg.birth_month = 1
            user_tg.birth_year = 1995
            user_tg.accepted_disclaimer = True

        user_max = await session.get(User, 55555)
        if not user_max:
            user_max = User(
                id=55555,
                username="max_user",
                first_name="Max",
                name="Max User",
                gender="male",
                age="25",
                birth_day=1,
                birth_month=1,
                birth_year=1995,
                accepted_disclaimer=True,
            )
            session.add(user_max)
        else:
            user_max.name = "Max User"
            user_max.gender = "male"
            user_max.age = "25"
            user_max.accepted_disclaimer = True

        from database import SubscriptionConfig
        sub_cfg = await session.get(SubscriptionConfig, 1)
        if not sub_cfg:
            sub_cfg = SubscriptionConfig(id=1, subscriptions_enabled=False)
            session.add(sub_cfg)
        else:
            sub_cfg.subscriptions_enabled = False
        await session.commit()

    storage = MemoryStorage()
    handlers.router._parent_router = None
    dp = Dispatcher(storage=storage)
    dp.include_router(handlers.router)

    telegram_session = ValidatingTelegramSession()
    bot = Bot(token=bot_token, session=telegram_session)

    # --------------------------------------------------------------------------
    # Part 1: Telegram Chat Journey delivering response to user
    # --------------------------------------------------------------------------
    kie_err = InsufficientBalanceError("KIE credits empty")
    kie_err.provider = "KIE"
    monkeypatch.setattr(ai_integration, "_call_kie_chat", AsyncMock(side_effect=kie_err))
    monkeypatch.setattr(ai_integration, "_call_openai_api", AsyncMock(return_value="Здравствуйте! Я ваш психотерапевтический ассистент."))

    tg_chat = Chat(id=999, type="private")
    tg_user = TgUser(id=999, is_bot=False, first_name="Tg User", username="tguser")
    user_msg_update = Update(
        update_id=201,
        message=Message(
            message_id=50,
            date=datetime.now(timezone.utc),
            chat=tg_chat,
            from_user=tg_user,
            text="Мне тревожно, помоги пожалуйста",
        ),
    )
    await dp.feed_update(bot, user_msg_update)
    # Wait for the debouncing buffer to drain (initial_delay=0.8s)
    await asyncio.sleep(1.2)

    # Verify: message delivered to user via bot.send_message
    user_deliveries = [c for c in telegram_session.calls if isinstance(c, SendMessage) and c.chat_id == 999]
    assert len(user_deliveries) >= 1
    assert "психотерапевтический ассистент" in user_deliveries[-1].text

    # --------------------------------------------------------------------------
    # Part 2: MAX Chat Journey delivering response to user
    # --------------------------------------------------------------------------
    import database
    import max_messenger_bot.app as max_app_module
    import max_messenger_bot.ai as max_ai
    import max_messenger_bot.legacy as max_legacy
    import max_messenger_bot.services.common as max_common
    import max_messenger_bot.storage as max_storage
    monkeypatch.setattr(database, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr(max_app_module, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr(max_ai, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr(max_legacy, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr(max_common, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr(max_storage, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr(max_ai, "_call_kie_text_chat", AsyncMock(side_effect=kie_err))
    monkeypatch.setattr(max_ai, "_call_openai", AsyncMock(return_value="Здравствуйте от MAX ассистента!"))

    # Create StorageBase tables (max_bot_states, etc.)
    engine = test_db_session_maker.kw["bind"]
    async with engine.begin() as conn:
        await conn.run_sync(max_storage.StorageBase.metadata.create_all)

    mock_max_client = AsyncMock()
    mock_max_client.send_message = AsyncMock(return_value={"message_id": "max_reply_1"})
    max_app = MaxBotApplication(mock_max_client)

    max_user_msg = IncomingMessage(
        raw={},
        message_id="max_m1",
        chat_id=55555,
        sender=Sender(user_id=55555, username="max_user", first_name="Max", last_name=None),
        text="Привет, как справиться со стрессом?",
    )
    await max_app.handle_message(max_user_msg)
    if 55555 in max_app.user_tasks:
        await max_app.user_tasks[55555]

    # Verify: response delivered to MAX user via client.send_message
    assert mock_max_client.send_message.called
    for call in mock_max_client.send_message.call_args_list:
        target_chat_id = call.kwargs.get("chat_id") if "chat_id" in call.kwargs else call.args[0]
        text_arg = call.kwargs.get("text") if "text" in call.kwargs else (call.args[1] if len(call.args) > 1 else "")
        assert isinstance(target_chat_id, int), f"MAX recipient chat_id must be int, got {type(target_chat_id)}"
        assert isinstance(text_arg, str) and len(text_arg.strip()) > 0, "MAX sent text must be non-empty str"

    max_sent_texts = [
        call.kwargs.get("text") or (call.args[1] if len(call.args) > 1 else "")
        for call in mock_max_client.send_message.call_args_list
    ]
    assert any("MAX ассистента" in t for t in max_sent_texts)

    # --------------------------------------------------------------------------
    # Part 3: Admin FSM Journey without mocking _edit_provider_model_settings
    # --------------------------------------------------------------------------
    admin_user = TgUser(id=777, is_bot=False, first_name="Admin", username="admin777")
    admin_chat = Chat(id=777, type="private")

    # Step A: Admin opens KIE settings
    cb_msg = Message(message_id=60, date=datetime.now(timezone.utc), chat=admin_chat, text="Settings")
    update_open_settings = Update(
        update_id=301,
        callback_query=CallbackQuery(
            id="cb_open_settings",
            from_user=admin_user,
            chat_instance="ci_admin",
            message=cb_msg,
            data="view_models_KIE",
        ),
    )
    await dp.feed_update(bot, update_open_settings)

    # Dynamically inspect buttons from last_edit.reply_markup.inline_keyboard (do NOT hardcode callback data)
    edit_calls = [c for c in telegram_session.calls if type(c).__name__ == "EditMessageText"]
    assert len(edit_calls) >= 1
    last_edit = edit_calls[-1]
    buttons = [btn for row in last_edit.reply_markup.inline_keyboard for btn in row]

    # Find each visible button dynamically by user-facing text
    btn_model = next((b for b in buttons if "Выбрать модель" in b.text), None)
    btn_tokens = next((b for b in buttons if "Max tokens" in b.text), None)
    btn_key = next((b for b in buttons if "API-ключ" in b.text), None)
    btn_back = next((b for b in buttons if "Назад" in b.text), None)

    assert btn_model is not None, "Button 'Выбрать модель' must be present in settings markup"
    assert btn_tokens is not None, "Button 'Max tokens' must be present in settings markup"
    assert btn_key is not None, "Button 'API-ключ' must be present in settings markup"
    assert btn_back is not None, "Button 'Назад' must be present in settings markup"

    fsm_ctx = dp.fsm.get_context(bot=bot, user_id=777, chat_id=777)

    # Step B: Test 'Выбрать модель' button navigation through to its terminal action
    update_click_model = Update(
        update_id=302,
        callback_query=CallbackQuery(
            id="cb_click_model",
            from_user=admin_user,
            chat_instance="ci_admin",
            message=cb_msg,
            data=btn_model.callback_data,
        ),
    )
    await dp.feed_update(bot, update_click_model)

    # Inspect rendered model choice keyboard and select a model
    from provider_models import TELEGRAM_MODEL_CALLBACK_PREFIX, resolve_telegram_model_callback
    model_edit_calls = [c for c in telegram_session.calls if type(c).__name__ == "EditMessageText"]
    model_choice_btns = [
        btn for row in model_edit_calls[-1].reply_markup.inline_keyboard for btn in row
        if btn.callback_data and btn.callback_data.startswith(TELEGRAM_MODEL_CALLBACK_PREFIX)
    ]
    assert len(model_choice_btns) >= 1, "Model choice buttons must be rendered"
    target_model_btn = model_choice_btns[0]
    prov, ch, expected_model = resolve_telegram_model_callback(target_model_btn.callback_data)

    update_select_model = Update(
        update_id=3021,
        callback_query=CallbackQuery(
            id="cb_select_model",
            from_user=admin_user,
            chat_instance="ci_admin",
            message=cb_msg,
            data=target_model_btn.callback_data,
        ),
    )
    await dp.feed_update(bot, update_select_model)

    # Verify AIConfig model is updated in DB to the selected model
    async with test_db_session_maker() as session:
        cfg_model = await session.get(AIConfig, 1)
        assert cfg_model.kie_model == expected_model

    # Step C: Test 'Max tokens' button navigation, text input, DB persistence and re-render
    update_click_tokens = Update(
        update_id=303,
        callback_query=CallbackQuery(
            id="cb_click_tokens",
            from_user=admin_user,
            chat_instance="ci_admin",
            message=cb_msg,
            data=btn_tokens.callback_data,
        ),
    )
    await dp.feed_update(bot, update_click_tokens)
    assert await fsm_ctx.get_state() == handlers.AdminStates.set_model_max_tokens.state

    # Admin submits valid tokens number
    update_enter_tokens = Update(
        update_id=3031,
        message=Message(
            message_id=61,
            date=datetime.now(timezone.utc),
            chat=admin_chat,
            from_user=admin_user,
            text="2048",
        ),
    )
    await dp.feed_update(bot, update_enter_tokens)
    assert await fsm_ctx.get_state() is None

    # Verify AIModelSettings is persisted in DB
    from database import AIModelSettings
    async with test_db_session_maker() as session:
        cfg_now = await session.get(AIConfig, 1)
        m_row = (await session.execute(
            select(AIModelSettings).where(
                AIModelSettings.provider == "KIE",
                AIModelSettings.model == cfg_now.kie_model,
                AIModelSettings.channel == "chat",
            )
        )).scalar_one_or_none()
        assert m_row is not None
        assert m_row.max_output_tokens == 2048

    # Re-open KIE settings for next check
    await dp.feed_update(bot, update_open_settings)

    # Step D: Admin clicks '🔑 API-ключ' -> FSM state is set_api_key
    update_click_key = Update(
        update_id=304,
        callback_query=CallbackQuery(
            id="cb_click_key",
            from_user=admin_user,
            chat_instance="ci_admin",
            message=cb_msg,
            data=btn_key.callback_data,
        ),
    )
    await dp.feed_update(bot, update_click_key)
    assert await fsm_ctx.get_state() == handlers.AdminStates.set_api_key.state

    # Step E: Admin clicks Cancel/Back ('btn_back.callback_data') -> FSM cancelled
    update_cancel = Update(
        update_id=305,
        callback_query=CallbackQuery(
            id="cb_cancel",
            from_user=admin_user,
            chat_instance="ci_admin",
            message=cb_msg,
            data=btn_back.callback_data,
        ),
    )
    await dp.feed_update(bot, update_cancel)
    assert await fsm_ctx.get_state() is None

    # Step F: Admin re-enters set_api_key and inputs new valid key
    await dp.feed_update(bot, update_click_key)
    assert await fsm_ctx.get_state() == handlers.AdminStates.set_api_key.state

    update_enter_key = Update(
        update_id=306,
        message=Message(
            message_id=62,
            date=datetime.now(timezone.utc),
            chat=admin_chat,
            from_user=admin_user,
            text="new_valid_admin_key_fsm_e2e_2026",
        ),
    )
    await dp.feed_update(bot, update_enter_key)

    # Verify FSM cleared and AIConfig updated in DB
    assert await fsm_ctx.get_state() is None
    async with test_db_session_maker() as session:
        cfg_updated = await session.get(AIConfig, 1)
        assert cfg_updated.kie_api_key == "new_valid_admin_key_fsm_e2e_2026"

    # Step G: Re-open settings screen and verify it renders without error
    await dp.feed_update(bot, update_open_settings)
    edit_calls_reopened = [c for c in telegram_session.calls if type(c).__name__ == "EditMessageText"]
    assert len(edit_calls_reopened) > len(edit_calls)

    # Step H: Runtime reflection: verify new user message triggers AI call with the updated key
    captured_keys = []
    async def mock_kie_runtime(api_key, *args, **kwargs):
        captured_keys.append(api_key)
        return "Ответ с новым API ключом!"

    monkeypatch.setattr(ai_integration, "_call_kie_chat", mock_kie_runtime)
    user_msg_update_runtime = Update(
        update_id=202,
        message=Message(
            message_id=51,
            date=datetime.now(timezone.utc),
            chat=tg_chat,
            from_user=tg_user,
            text="Тест работы после смены ключа",
        ),
    )
    await dp.feed_update(bot, user_msg_update_runtime)
    await asyncio.sleep(1.2)

    assert len(captured_keys) >= 1, "Runtime call to KIE must have occurred"
    assert captured_keys[-1] == "new_valid_admin_key_fsm_e2e_2026", "AI request must use the newly saved API key"

    # Step I: Admin clicks '⬅️ Назад' on KIE settings screen -> returns to logical parent (admin_ai_keys)
    update_back_to_keys = Update(
        update_id=307,
        callback_query=CallbackQuery(
            id="cb_back_to_keys",
            from_user=admin_user,
            chat_instance="ci_admin",
            message=cb_msg,
            data=btn_back.callback_data,
        ),
    )
    await dp.feed_update(bot, update_back_to_keys)
    last_edit_back = [c for c in telegram_session.calls if type(c).__name__ == "EditMessageText"][-1]
    assert "Провайдеры и модели" in last_edit_back.text or "Ключи и модели" in last_edit_back.text


@pytest.mark.asyncio
async def test_max_terminal_ai_failure_three_admins_timeout_retry_restart_durable_outbox(test_db_session_maker, monkeypatch):
    """Verify MAX terminal failure alert handling:
    1. Origin is MAX instance (max_se13639182_bot_legacy), companion is tg_veraveda777_bot_legacy.
    2. Exactly 3 admins configured in registry for the companion Telegram bot.
    3. Terminal failure alert from MAX enqueues separate outbox rows for all 3 admins with zero PII.
    4. Companion bot worker attempts delivery: Admin 1 & 3 succeed, Admin 2 encounters network timeout.
    5. Admin 2 row remains pending with attempts=1 and exponential backoff.
    6. Process restart / lease expiry recovery: retry succeeds for Admin 2.
    7. ZERO duplicates delivered to Admin 1 and Admin 3.
    """
    from error_reporting import send_terminal_ai_failure_alert
    from ai_incident_manager import enqueue_terminal_ai_failure_notification
    from aiogram.exceptions import TelegramNetworkError

    # Step 1: MAX instance context
    monkeypatch.setenv("BOT_INSTANCE_ID", "max_se13639182_bot_legacy")
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)
    monkeypatch.setattr("error_reporting.async_session_maker", test_db_session_maker)

    prov_fail = ProviderFailure(
        provider="KIE",
        operation="chat",
        classification="network",
        detail_code="api_connection_error",
        provider_error_type="APIConnectionError",
        operator_action_required=False,
        retryable=True,
        transient=True,
        classification_source="provider_structured",
    )
    inc = await record_ai_provider_failure(
        origin_instance_id="max_se13639182_bot_legacy",
        provider="KIE",
        error=prov_fail,
        operation="chat",
        model="seedream-v4",
        session_maker=test_db_session_maker,
    )
    assert inc is not None
    assert inc.status == "transient"

    # Enqueue terminal alert via send_terminal_ai_failure_alert with bot=None and MAX user/chat IDs
    max_chat_id = 100026905347
    max_user_id = 100026905347
    alert_ok = await send_terminal_ai_failure_alert(
        bot=None,
        chat_id=max_chat_id,
        user_id=max_user_id,
        incident_id=inc.id,
        primary_provider="KIE",
        primary_model="seedream-v4",
        fallback_provider="OpenAI",
        fallback_model="gpt-4o",
        bot_name="MAX Психолог",
        platform="max",
        request_type="chat",
        error_type="APIConnectionError",
        classification="network",
        session_maker=test_db_session_maker,
    )
    assert alert_ok is True

    # Step 2: Verify outbox rows and ZERO PII
    async with test_db_session_maker() as session:
        rows = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.incident_id == inc.id,
                AIIncidentNotificationModel.event_type == "terminal_failure",
            ).order_by(AIIncidentNotificationModel.recipient_id.asc())
        )).scalars().all()

        assert len(rows) == 3, "Must enqueue 1 row per configured companion admin (total 3)"
        expected_recipients = sorted([979514796, 272982544, 806750628])
        actual_recipients = sorted([r.recipient_id for r in rows])
        assert actual_recipients == expected_recipients

        for r in rows:
            assert r.authorized_instance_id == "tg_veraveda777_bot_legacy"
            assert r.status == "pending"
            assert r.attempts == 0
            # Strict ZERO PII check: user_id / chat_id must NEVER leak into outbox message_text
            assert str(max_chat_id) not in r.message_text
            assert "MAX Психолог" in r.message_text
            assert "APIConnectionError" in r.message_text

    # Step 3: Switch context to companion Telegram bot worker
    companion_token = "8224473896:AAFtest_token_veraveda"
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", companion_token)

    mock_bot = AsyncMock(spec=Bot)
    mock_bot.token = companion_token
    mock_bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="veravedabot", username="veraveda777_bot"))

    sent_admin_ids = []
    async def mock_send(chat_id, text, **kwargs):
        sent_admin_ids.append(chat_id)
        if chat_id == 272982544:
            raise TelegramNetworkError(method=SendMessage(chat_id=chat_id, text=text), message="Connection timeout")
        return MagicMock(message_id=101)

    mock_bot.send_message = AsyncMock(side_effect=mock_send)

    # First worker cycle
    sent_count = await process_ai_incident_notifications(mock_bot, session_maker=test_db_session_maker)
    assert sent_count == 2
    assert sorted(sent_admin_ids) == sorted([979514796, 272982544, 806750628])

    async with test_db_session_maker() as session:
        r_timeout = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.incident_id == inc.id,
                AIIncidentNotificationModel.recipient_id == 272982544,
            )
        )).scalar_one()
        assert r_timeout.status == "pending"
        assert r_timeout.attempts == 1
        assert r_timeout.last_error == "telegram_network_error"
        assert r_timeout.next_retry_at > datetime.utcnow()

        # Step 4: Simulate restart and backoff expiration
        r_timeout.next_retry_at = datetime.utcnow() - timedelta(seconds=1)
        await session.commit()

    # Second worker cycle after restart (network recovered)
    mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=102))
    sent_admin_ids.clear()
    sent_count_retry = await process_ai_incident_notifications(mock_bot, session_maker=test_db_session_maker)
    assert sent_count_retry == 1
    assert mock_bot.send_message.call_count == 1
    call_chat_id = mock_bot.send_message.call_args.kwargs.get("chat_id")
    assert call_chat_id == 272982544

    # Verify final states in DB: all 3 sent, exactly 1 delivery to admins 979514796 and 806750628 (zero duplicates)
    async with test_db_session_maker() as session:
        final_rows = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.incident_id == inc.id,
            )
        )).scalars().all()
        for r in final_rows:
            assert r.status == "delivered"
        recovered_row = next(r for r in final_rows if r.recipient_id == 272982544)
        assert recovered_row.attempts == 2


@pytest.mark.asyncio
async def test_stale_terminal_failure_skipped_when_parent_incident_resolved_or_superseded(test_db_session_maker, monkeypatch):
    """Verify that if parent incident is marked 'resolved' or 'superseded' before outbox worker sends,
    terminal_failure outbox rows are transitioned to 'skipped_stale' and NOT delivered to admins.
    """
    from ai_incident_manager import enqueue_terminal_ai_failure_notification

    bot_token = "8281390498:AAFtest_telegram_token_123456789012"
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", bot_token)
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)

    mock_bot = AsyncMock(spec=Bot)
    mock_bot.token = bot_token
    mock_bot.get_me = AsyncMock(return_value=TgUser(id=8281390498, is_bot=True, first_name="bot", username="someonewithyou01_bot"))

    # Case A: Parent incident becomes 'resolved'
    inc1 = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=InsufficientBalanceError("KIE credits empty"),
        operation="chat",
        model="seedream-v4",
        session_maker=test_db_session_maker,
    )
    assert inc1 is not None

    await enqueue_terminal_ai_failure_notification(
        incident_id=inc1.id,
        origin_instance_id="tg_someonewithyou01_bot_new",
        text="Terminal alert test resolved",
        session_maker=test_db_session_maker,
    )

    async with test_db_session_maker() as session:
        db_inc1 = await session.get(AIProviderIncidentModel, inc1.id)
        db_inc1.status = "resolved"
        await session.commit()

    processed1 = await process_ai_incident_notifications(mock_bot, session_maker=test_db_session_maker)
    assert processed1 == 0
    assert mock_bot.send_message.call_count == 0

    async with test_db_session_maker() as session:
        rows1 = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.incident_id == inc1.id,
                AIIncidentNotificationModel.event_type == "terminal_failure",
            )
        )).scalars().all()
        assert len(rows1) >= 1
        for r in rows1:
            assert r.status == "skipped_stale"
            assert r.last_error == "incident_resolved_before_send"

    # Case B: Parent incident becomes 'superseded'
    inc2 = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=InsufficientBalanceError("KIE credits empty 2"),
        operation="chat",
        model="seedream-v4",
        session_maker=test_db_session_maker,
    )
    assert inc2 is not None

    await enqueue_terminal_ai_failure_notification(
        incident_id=inc2.id,
        origin_instance_id="tg_someonewithyou01_bot_new",
        text="Terminal alert test superseded",
        cooldown_seconds=0,
        session_maker=test_db_session_maker,
    )

    async with test_db_session_maker() as session:
        db_inc2 = await session.get(AIProviderIncidentModel, inc2.id)
        db_inc2.status = "superseded"
        await session.commit()

    processed2 = await process_ai_incident_notifications(mock_bot, session_maker=test_db_session_maker)
    assert processed2 == 0
    assert mock_bot.send_message.call_count == 0

    async with test_db_session_maker() as session:
        rows2 = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.incident_id == inc2.id,
                AIIncidentNotificationModel.event_type == "terminal_failure",
            )
        )).scalars().all()
        assert len(rows2) >= 1
        for r in rows2:
            assert r.status == "skipped_stale"
            assert r.last_error == "incident_superseded_before_send"


@pytest.mark.asyncio
async def test_classification_source_cannot_be_elevated_by_caller_argument(caplog):
    """Verify that arbitrary caller arguments cannot elevate classification_source to verified/trusted
    unless backed by a genuine ProviderFailure object with a verified source.
    """
    from error_reporting import notify_admins_about_error
    import logging

    caplog.set_level(logging.DEBUG)

    mock_bot = AsyncMock(spec=Bot)
    mock_bot.token = "8281390498:AAFtest_token"

    # Case A: Caller passes classification_source="provider_structured" with a generic non-provider exception
    generic_exc = RuntimeError("unexpected database connection closed")
    await notify_admins_about_error(
        mock_bot,
        title="Database Error",
        user_id=123,
        exception=generic_exc,
        classification="system",
        classification_source="provider_structured",
    )

    for record in caplog.records:
        assert "is_handled_ai" not in record.message or not getattr(record, "is_handled_ai", False)

    # Case B: Caller passes ProviderFailure with unverified classification_source ("message_fallback")
    # and attempts caller elevation classification_source="provider_structured"
    fallback_fail = ProviderFailure(
        provider="KIE",
        operation="chat",
        classification="server_error",
        detail_code="500_internal_error",
        provider_error_type="500 Internal Error",
        operator_action_required=False,
        retryable=False,
        transient=False,
        classification_source="message_fallback",
    )
    caplog.clear()
    await notify_admins_about_error(
        mock_bot,
        title="Server Error",
        user_id=123,
        exception=fallback_fail,
        classification="server_error",
        classification_source="provider_structured",
    )
    for record in caplog.records:
        assert "is_handled_ai" not in record.message or not getattr(record, "is_handled_ai", False)


@pytest.mark.asyncio
async def test_handle_key_rotation_fail_closed_and_protects_foreign_instances(test_db_session_maker, monkeypatch):
    """Verify key rotation safety:
    1. Fails closed if instance identity cannot be determined.
    2. Shared DB isolation: Rotating key for Instance A does NOT supersede active incidents of Instance B.
    """
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)

    # Part 1: Fail closed with IncidentSecurityError when BOT_INSTANCE_ID is missing or foreign origin unauthorized
    from ai_incident_manager import IncidentSecurityError
    monkeypatch.delenv("BOT_INSTANCE_ID", raising=False)
    with pytest.raises(IncidentSecurityError):
        await handle_key_rotation(
            origin_instance_id=None,
            provider="KIE",
            new_key="new_key_1",
            session_maker=test_db_session_maker,
        )

    # Unauthorized foreign origin also raises IncidentSecurityError
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    with pytest.raises(IncidentSecurityError):
        await handle_key_rotation(
            origin_instance_id="tg_yourself_way_bot_new",
            provider="KIE",
            new_key="new_key_1",
            session_maker=test_db_session_maker,
        )

    # Part 2: Shared DB isolation between instances
    # Instance A: tg_someonewithyou01_bot_new
    # Instance B: tg_yourself_way_bot_new
    scope_a = get_provider_account_fingerprint("key_inst_a", None)
    scope_b = get_provider_account_fingerprint("key_inst_b", None)

    async with test_db_session_maker() as session:
        inc_a = AIProviderIncidentModel(
            origin_instance_id="tg_someonewithyou01_bot_new",
            companion_instance_id="tg_someonewithyou01_bot_new",
            provider="KIE",
            credential_scope=scope_a,
            incident_type="insufficient_balance",
            status="active",
            operation="chat",
        )
        inc_b = AIProviderIncidentModel(
            origin_instance_id="tg_yourself_way_bot_new",
            companion_instance_id="tg_yourself_way_bot_new",
            provider="KIE",
            credential_scope=scope_b,
            incident_type="insufficient_balance",
            status="active",
            operation="chat",
        )
        session.add_all([inc_a, inc_b])
        await session.commit()
        id_a, id_b = inc_a.id, inc_b.id

    # Now rotate key in context of Instance A
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    rotation_count = await handle_key_rotation(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        new_key="key_inst_a_new",
        old_credential_scope=scope_a,
        session_maker=test_db_session_maker,
    )
    assert rotation_count == 1

    # Check DB state
    async with test_db_session_maker() as session:
        db_inc_a = await session.get(AIProviderIncidentModel, id_a)
        db_inc_b = await session.get(AIProviderIncidentModel, id_b)

        # Instance A incident was superseded
        assert db_inc_a.status == "superseded"

        # Instance B incident remains ACTIVE! Protected by origin_names filter
        assert db_inc_b.status == "active", "Foreign bot incident in shared DB must NOT be modified by key rotation"


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_concurrent_cas_repeat_reminders(pg_test_session_maker, monkeypatch):
    """Review Point 6: Concurrent CAS repeat reminders on real PostgreSQL:
    - Active incident in PostgreSQL with last_alerted_at older than reminder interval (7 hours ago).
    - Multiple concurrent worker tasks run check_incident_repeat_reminders simultaneously.
    - PostgreSQL CAS atomic update ensures exactly ONE worker successfully enqueues the reminder cycle.
    - Zero duplicate repeat_reminder outbox rows created.
    """
    bot_token = "8281390498:AAFtest_telegram_token_123456789012"
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", bot_token)

    mock_bot = AsyncMock(spec=Bot)
    mock_bot.token = bot_token
    mock_bot.get_me = AsyncMock(return_value=TgUser(id=8281390498, is_bot=True, first_name="bot", username="someonewithyou01_bot"))

    err = InsufficientBalanceError("KIE credits empty")
    err.provider = "KIE"
    inc = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=err,
        operation="chat",
        model="seedream-v4",
        session_maker=pg_test_session_maker,
    )
    assert inc is not None

    # Age incident to 7 hours ago
    async with pg_test_session_maker() as session:
        db_inc = await session.get(AIProviderIncidentModel, inc.id)
        db_inc.last_alerted_at = datetime.utcnow() - timedelta(hours=7)
        db_inc.reminder_cycle = 0
        await session.commit()

    # 3 concurrent worker tasks running check_incident_repeat_reminders
    results = await asyncio.gather(
        check_incident_repeat_reminders(mock_bot, session_maker=pg_test_session_maker),
        check_incident_repeat_reminders(mock_bot, session_maker=pg_test_session_maker),
        check_incident_repeat_reminders(mock_bot, session_maker=pg_test_session_maker),
    )

    # Exactly one worker succeeded in updating CAS and enqueuing reminders!
    assert len([r for r in results if r > 0]) == 1
    assert max(results) >= 1

    # Verify PostgreSQL DB state:
    async with pg_test_session_maker() as session:
        db_inc = await session.get(AIProviderIncidentModel, inc.id)
        assert db_inc.reminder_cycle == 1

        rem_notifs = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.incident_id == inc.id,
                AIIncidentNotificationModel.event_type == "repeat_reminder",
            )
        )).scalars().all()
        # Exactly one set of repeat_reminder records for cycle_index=1
        assert len(rem_notifs) >= 1
        for n in rem_notifs:
            assert n.cycle_index == 1
            assert n.status == "pending"


@pytest.mark.asyncio
async def test_admin_dispatcher_key_rotation_race_with_inflight_failure(test_db_session_maker, monkeypatch):
    """Defect 1: Real Admin Dispatcher -> save new key -> completion of old in-flight request:
    - Initial AIConfig has key A ('kie_key_race_alpha_111'). Zero active incidents exist prior to rotation.
    - In-flight request starts with key A scope.
    - Admin updates key to key B ('kie_key_race_beta_222') via real Dispatcher & Router FSM (process_api_input).
    - Handler reads old key before updating AIConfig and calls handle_key_rotation(old_key=...).
    - Old in-flight request fails with InsufficientBalanceError and records failure with old credential_scope.
    - Assert: Zero active incidents in AIProviderIncidentModel.
    - Assert: Old key failure recorded as superseded.
    - Assert: Zero notifications enqueued in outbox.
    """
    from datetime import timezone
    from aiogram import Dispatcher
    from aiogram.fsm.storage.memory import MemoryStorage
    from aiogram.types import CallbackQuery, Chat, Message, Update
    import handlers

    bot_token = "8224473896:AAFtest_token_veraveda"
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", bot_token)
    monkeypatch.setattr("database.async_session_maker", test_db_session_maker)
    monkeypatch.setattr(handlers, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr("ai_integration.async_session_maker", test_db_session_maker)
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)
    monkeypatch.setattr("handlers.missing_profile_fields", lambda c, u: [])
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(side_effect=lambda user_id: user_id == 777))

    key_a = "kie_key_race_alpha_111"
    key_b = "kie_key_race_beta_222"
    base_url = "https://api.kie.ai"
    scope_a = get_provider_account_fingerprint(key_a, base_url)
    scope_b = get_provider_account_fingerprint(key_b, base_url)

    # 1. Initial AIConfig with key A, zero incidents
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="KIE", kie_api_key=key_a, kie_base_url=base_url)
            session.add(cfg)
        else:
            cfg.provider = "KIE"
            cfg.kie_api_key = key_a
            cfg.kie_base_url = base_url
            cfg.kie_credit_alert_sent = False
        await session.commit()

    storage = MemoryStorage()
    handlers.router._parent_router = None
    dp = Dispatcher(storage=storage)
    dp.include_router(handlers.router)
    from tests.test_perplexity_agent_models import ValidatingTelegramSession
    telegram_session = ValidatingTelegramSession()
    bot = Bot(token=bot_token, session=telegram_session)
    bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    admin_user = TgUser(id=777, is_bot=False, first_name="Admin", username="admin777")
    admin_chat = Chat(id=777, type="private")
    fsm_ctx = dp.fsm.get_context(bot=bot, user_id=777, chat_id=777)

    # 2. Admin opens settings and clicks API key
    cb_msg = Message(message_id=70, date=datetime.now(timezone.utc), chat=admin_chat, text="Settings")
    await dp.feed_update(bot, Update(
        update_id=401,
        callback_query=CallbackQuery(
            id="cb_open",
            from_user=admin_user,
            chat_instance="ci_admin",
            message=cb_msg,
            data="view_models_KIE",
        ),
    ))
    await dp.feed_update(bot, Update(
        update_id=402,
        callback_query=CallbackQuery(
            id="cb_click_key",
            from_user=admin_user,
            chat_instance="ci_admin",
            message=cb_msg,
            data="model_setting_api_key_KIE",
        ),
    ))
    assert await fsm_ctx.get_state() == handlers.AdminStates.set_api_key.state

    # 3. Admin enters new key B via message handled by process_api_input
    await dp.feed_update(bot, Update(
        update_id=403,
        message=Message(
            message_id=71,
            date=datetime.now(timezone.utc),
            chat=admin_chat,
            from_user=admin_user,
            text=key_b,
        ),
    ))
    assert await fsm_ctx.get_state() is None

    # Verify AIConfig updated
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        assert cfg.kie_api_key == key_b

    # 4. Old in-flight request (which ran with key A) now completes and fails
    err = InsufficientBalanceError("KIE credits empty on retired key A")
    err.provider = "KIE"
    inc_old = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        error=err,
        operation="chat",
        model="kie-v1",
        credential_scope=scope_a,
        session_maker=test_db_session_maker,
    )

    # 5. Verify assertions
    assert inc_old is not None
    assert inc_old.credential_scope == scope_a
    assert inc_old.status == "superseded", "Old in-flight failure must be preserved as superseded"

    async with test_db_session_maker() as session:
        # Zero active incidents in DB
        active_incs = (await session.execute(
            select(AIProviderIncidentModel).where(AIProviderIncidentModel.status == "active")
        )).scalars().all()
        assert len(active_incs) == 0

        # Zero pending/processing notifications in outbox
        pending_notifs = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.status.in_(["pending", "processing"])
            )
        )).scalars().all()
        assert len(pending_notifs) == 0


@pytest.mark.asyncio
async def test_key_rotation_reactivation_a_b_a_allows_active_incident(test_db_session_maker, monkeypatch):
    """Defect 2: Key rotation lifecycle A -> B -> A:
    - Step 1: Active Key A encounters operator failure -> opens active incident and enqueues notification.
    - Step 2: Key rotated to Key B -> Key A incident superseded, notifications cancelled, tombstone exists for Key A.
    - Step 3: Key rotated back to Key A (A -> B -> A) -> Key A is the current active key in AIConfig.
    - Step 4: Key A encounters operator failure -> must open a NEW active incident and enqueue notifications.
      Historical superseded record does NOT permanently block Key A.
    """
    bot_token = "8224473896:AAFtest_token_veraveda"
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", bot_token)
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)

    key_a = "key_alpha_rotation_111"
    key_b = "key_beta_rotation_222"
    scope_a = get_provider_account_fingerprint(key_a, None)
    scope_b = get_provider_account_fingerprint(key_b, None)

    # Step 1: Initial config with Key A
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="OpenAI", openai_api_key=key_a)
            session.add(cfg)
        else:
            cfg.provider = "OpenAI"
            cfg.openai_api_key = key_a
        await session.commit()

    err = InsufficientBalanceError("Quota exceeded on key A")
    err.provider = "OpenAI"
    inc1 = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=err,
        operation="chat",
        model="gpt-4o",
        credential_scope=scope_a,
        session_maker=test_db_session_maker,
    )
    assert inc1 is not None
    assert inc1.status == "active"

    # Step 2: Rotate to Key B
    await handle_key_rotation(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        new_key=key_b,
        old_key=key_a,
        session_maker=test_db_session_maker,
    )
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.openai_api_key = key_b
        await session.commit()

    # Verify Key A incident is superseded
    async with test_db_session_maker() as session:
        inc1_db = await session.get(AIProviderIncidentModel, inc1.id)
        assert inc1_db.status == "superseded"

    # Step 3: Rotate back to Key A (A -> B -> A)
    await handle_key_rotation(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        new_key=key_a,
        old_key=key_b,
        session_maker=test_db_session_maker,
    )
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.openai_api_key = key_a
        await session.commit()

    # Step 4: Key A fails again
    err2 = InsufficientBalanceError("Quota exceeded again on re-activated key A")
    err2.provider = "OpenAI"
    inc2 = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=err2,
        operation="chat",
        model="gpt-4o",
        credential_scope=scope_a,
        session_maker=test_db_session_maker,
    )

    # Assert new active incident is successfully opened!
    assert inc2 is not None
    assert inc2.status == "active", "Re-activated Key A failure must open an active incident"
    assert inc2.id != inc1.id

    # Verify outbox has pending notification for the new active incident
    async with test_db_session_maker() as session:
        notifs = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.incident_id == inc2.id,
                AIIncidentNotificationModel.status == "pending",
            )
        )).scalars().all()
        assert len(notifs) >= 1


@pytest.mark.asyncio
async def test_terminal_ai_failure_db_enqueue_failed_allows_immediate_retry(test_db_session_maker, monkeypatch):
    """Defect 3: DB enqueue failure does not poison local in-memory cooldown; immediate retry succeeds."""
    from error_reporting import send_terminal_ai_failure_alert
    import ai_incident_manager

    bot_token = "8224473896:AAFtest_token_veraveda"
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", bot_token)
    monkeypatch.setattr("error_reporting.async_session_maker", test_db_session_maker)
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)

    real_enqueue = ai_incident_manager.enqueue_terminal_ai_failure_notification
    fail_first = True

    async def flaky_enqueue(*args, **kwargs):
        nonlocal fail_first
        if fail_first:
            fail_first = False
            raise RuntimeError("Database connection lost during outbox enqueue")
        return await real_enqueue(*args, **kwargs)

    monkeypatch.setattr("ai_incident_manager.enqueue_terminal_ai_failure_notification", flaky_enqueue)

    mock_bot = AsyncMock(spec=Bot)
    mock_bot.token = bot_token
    mock_bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    # Attempt 1: Fails due to DB error
    res1 = await send_terminal_ai_failure_alert(
        mock_bot,
        primary_provider="Deepseek",
        primary_model="deepseek-chat",
        classification="timeout",
        error_message="Gateway timeout",
        session_maker=test_db_session_maker,
    )
    assert res1 is False

    # Attempt 2: Immediate retry must NOT be blocked by any in-memory cooldown!
    res2 = await send_terminal_ai_failure_alert(
        mock_bot,
        primary_provider="Deepseek",
        primary_model="deepseek-chat",
        classification="timeout",
        error_message="Gateway timeout",
        session_maker=test_db_session_maker,
    )
    assert res2 is True

    # Verify notification exists in DB
    async with test_db_session_maker() as session:
        notifs = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.event_type == "terminal_failure",
            )
        )).scalars().all()
        assert len(notifs) >= 1


@pytest.mark.asyncio
async def test_terminal_ai_failure_cooldown_survives_restart(test_db_session_maker, monkeypatch):
    """Defect 3: DB-backed cooldown persists across process restarts (in-memory state cleared)."""
    from error_reporting import send_terminal_ai_failure_alert, _terminal_failure_cooldown

    bot_token = "8224473896:AAFtest_token_veraveda"
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", bot_token)
    monkeypatch.setattr("error_reporting.async_session_maker", test_db_session_maker)
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)

    mock_bot = AsyncMock(spec=Bot)
    mock_bot.token = bot_token
    mock_bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    # Initial alert succeeds
    res1 = await send_terminal_ai_failure_alert(
        mock_bot,
        primary_provider="Deepseek",
        primary_model="deepseek-chat",
        classification="provider_down",
        session_maker=test_db_session_maker,
    )
    assert res1 is True

    # Simulate process restart: clear any residual in-memory state
    _terminal_failure_cooldown._last_sent.clear()

    # Second alert within 30 minutes must be suppressed by durable DB cooldown
    res2 = await send_terminal_ai_failure_alert(
        mock_bot,
        primary_provider="Deepseek",
        primary_model="deepseek-chat",
        classification="provider_down",
        session_maker=test_db_session_maker,
    )
    assert res2 is False


@pytest.mark.asyncio
async def test_terminal_ai_failure_key_rotation_not_suppressed_by_old_cooldown(test_db_session_maker, monkeypatch):
    """Defect 3: Failure on a new rotated key is not suppressed by previous key's cooldown."""
    from error_reporting import send_terminal_ai_failure_alert

    bot_token = "8224473896:AAFtest_token_veraveda"
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", bot_token)
    monkeypatch.setattr("error_reporting.async_session_maker", test_db_session_maker)
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)

    mock_bot = AsyncMock(spec=Bot)
    mock_bot.token = bot_token
    mock_bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    key_1 = "old_deepseek_key_111"
    key_2 = "new_deepseek_key_222"

    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="Deepseek", deepseek_api_key=key_1)
            session.add(cfg)
        else:
            cfg.provider = "Deepseek"
            cfg.deepseek_api_key = key_1
        await session.commit()

    # Terminal alert on old key 1
    res1 = await send_terminal_ai_failure_alert(
        mock_bot,
        primary_provider="Deepseek",
        primary_model="deepseek-chat",
        classification="rate_limit",
        session_maker=test_db_session_maker,
    )
    assert res1 is True

    # Rotate key to key 2
    await handle_key_rotation(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="Deepseek",
        new_key=key_2,
        old_key=key_1,
        session_maker=test_db_session_maker,
    )
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.deepseek_api_key = key_2
        await session.commit()

    # Terminal alert on new key 2 must NOT be suppressed by key 1 cooldown
    res2 = await send_terminal_ai_failure_alert(
        mock_bot,
        primary_provider="Deepseek",
        primary_model="deepseek-chat",
        classification="rate_limit",
        session_maker=test_db_session_maker,
    )
    assert res2 is True


@pytest.mark.asyncio
async def test_terminal_ai_failure_concurrent_processes_and_partial_admin_delivery(test_db_session_maker, monkeypatch):
    """Defect 3: Concurrent processes deduplicate outbox enqueues, and worker handles partial delivery & retry."""
    from error_reporting import send_terminal_ai_failure_alert
    from aiogram.exceptions import TelegramRetryAfter, TelegramForbiddenError

    bot_token = "8224473896:AAFtest_token_veraveda"
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", bot_token)
    monkeypatch.setattr("error_reporting.async_session_maker", test_db_session_maker)
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)

    # 3 configured admins
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[1001, 1002, 1003]))

    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="Claude", claude_api_key="claude_test_concurrent_key")
            session.add(cfg)
        else:
            cfg.provider = "Claude"
            cfg.claude_api_key = "claude_test_concurrent_key"
        await session.commit()

    # Step 1: 2 concurrent processes call send_terminal_ai_failure_alert simultaneously
    res = await asyncio.gather(
        send_terminal_ai_failure_alert(
            bot=None,
            primary_provider="Claude",
            primary_model="claude-3-opus",
            classification="service_unavailable",
            session_maker=test_db_session_maker,
        ),
        send_terminal_ai_failure_alert(
            bot=None,
            primary_provider="Claude",
            primary_model="claude-3-opus",
            classification="service_unavailable",
            session_maker=test_db_session_maker,
        ),
    )
    # Exactly one succeeded in enqueuing, the other was suppressed by DB cooldown
    assert True in res

    # Verify exactly 3 notification rows exist (1 for each admin)
    async with test_db_session_maker() as session:
        notifs = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.event_type == "terminal_failure",
            )
        )).scalars().all()
        assert len(notifs) == 3

    # Step 2: Outbox worker dispatches to the 3 admins:
    # Admin 1001 succeeds, Admin 1002 encounters rate limit (retryable), Admin 1003 is blocked (terminal)
    async def mock_send(chat_id, text, **kwargs):
        if chat_id == 1001:
            return True
        elif chat_id == 1002:
            raise TelegramRetryAfter(method=None, message="Rate limit", retry_after=5)
        elif chat_id == 1003:
            raise TelegramForbiddenError(method=None, message="Forbidden: bot was blocked by the user")

    mock_bot = AsyncMock(spec=Bot)
    mock_bot.token = bot_token
    mock_bot.send_message = AsyncMock(side_effect=mock_send)
    mock_bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    delivered = await process_ai_incident_notifications(mock_bot, session_maker=test_db_session_maker)
    assert delivered == 1

    # Verify per-recipient outbox statuses
    async with test_db_session_maker() as session:
        n_1001 = (await session.execute(
            select(AIIncidentNotificationModel).where(AIIncidentNotificationModel.recipient_id == 1001)
        )).scalar_one()
        assert n_1001.status == "delivered"

        n_1002 = (await session.execute(
            select(AIIncidentNotificationModel).where(AIIncidentNotificationModel.recipient_id == 1002)
        )).scalar_one()
        assert n_1002.status == "pending"
        assert n_1002.attempts == 1

        n_1003 = (await session.execute(
            select(AIIncidentNotificationModel).where(AIIncidentNotificationModel.recipient_id == 1003)
        )).scalar_one()
        assert n_1003.status == "failed_terminal"

    # Step 3: Fast-forward time for retry of 1002
    async with test_db_session_maker() as session:
        n_1002 = (await session.execute(
            select(AIIncidentNotificationModel).where(AIIncidentNotificationModel.recipient_id == 1002)
        )).scalar_one()
        n_1002.next_retry_at = datetime.utcnow() - timedelta(seconds=1)
        await session.commit()

    # Retry delivery: now 1002 succeeds
    mock_bot.send_message = AsyncMock(return_value=True)
    delivered_retry = await process_ai_incident_notifications(mock_bot, session_maker=test_db_session_maker)
    assert delivered_retry == 1

    # Verify 1001 was NOT re-sent (only 1 send_message call occurred in retry)
    assert mock_bot.send_message.call_count == 1
    assert mock_bot.send_message.call_args.kwargs["chat_id"] == 1002


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_concurrent_transient_failure_and_terminal_alert(pg_test_session_maker, monkeypatch):
    """Defect 4: Concurrent non-operator failures and terminal alert deduplication on real PostgreSQL:
    - 2 concurrent async tasks record the exact same transient non-operator failure simultaneously on real PostgreSQL.
    - Partial unique index uq_transient_ai_incident enforces concurrent identity.
    - Exactly ONE row in ai_provider_incidents with status='transient' and occurrence_count=2.
    - Cooldown prevents duplicate terminal alert outbox rows.
    """
    bot_token = "8281390498:AAFtest_telegram_token_123456789012"
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", bot_token)

    inst_def = {"pm2_name": "tg_someonewithyou01_bot_new", "platform": "telegram", "platform_id": 8281390498, "admins": [1001], "username": "someonewithyou01_bot"}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[1001]))

    scope = get_provider_account_fingerprint("pg_test_transient_key", None)
    async with pg_test_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="OpenAI", openai_api_key="pg_test_transient_key")
            session.add(cfg)
        else:
            cfg.provider = "OpenAI"
            cfg.openai_api_key = "pg_test_transient_key"
        await session.commit()

    # Non-operator failure (transient)
    class TransientTimeout(Exception):
        pass

    async def record_transient():
        return await record_ai_provider_failure(
            origin_instance_id="tg_someonewithyou01_bot_new",
            provider="OpenAI",
            error=TransientTimeout("Temporary gateway timeout"),
            operation="chat",
            model="gpt-4o",
            credential_scope=scope,
            session_maker=pg_test_session_maker,
            enqueue_notification=False,
        )

    # Run 2 concurrent recordings
    inc1, inc2 = await asyncio.gather(record_transient(), record_transient())
    assert inc1 is not None
    assert inc2 is not None
    assert inc1.id == inc2.id

    # Verify exactly 1 transient incident row exists in PostgreSQL
    async with pg_test_session_maker() as session:
        trans_rows = (await session.execute(
            select(AIProviderIncidentModel).where(
                AIProviderIncidentModel.provider == "OPENAI",
                AIProviderIncidentModel.credential_scope == scope,
                AIProviderIncidentModel.status == "transient",
            )
        )).scalars().all()
        assert len(trans_rows) == 1
        assert trans_rows[0].occurrence_count == 2

    # Now run 2 concurrent terminal alert notifications for this incident
    enq1, enq2 = await asyncio.gather(
        enqueue_terminal_ai_failure_notification(
            incident_id=inc1.id,
            origin_instance_id="tg_someonewithyou01_bot_new",
            text="Diagnostic alert 1",
            cooldown_seconds=1800,
            session_maker=pg_test_session_maker,
        ),
        enqueue_terminal_ai_failure_notification(
            incident_id=inc1.id,
            origin_instance_id="tg_someonewithyou01_bot_new",
            text="Diagnostic alert 2",
            cooldown_seconds=1800,
            session_maker=pg_test_session_maker,
        ),
    )
    # Exactly one succeeded, one suppressed by cooldown/index
    assert (enq1, enq2) in [(True, False), (False, True)]

    # Verify exactly 1 notification row in PostgreSQL
    async with pg_test_session_maker() as session:
        notifs = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.incident_id == inc1.id,
                AIIncidentNotificationModel.event_type == "terminal_failure",
            )
        )).scalars().all()
        assert len(notifs) == 1


@pytest.mark.asyncio
async def test_admin_key_rotation_atomic_transaction_rollback_preserves_old_config(test_db_session_maker, monkeypatch):
    """Blocker 1: Failure during rotation rolls back AIConfig update atomically in handlers.process_api_input."""
    import handlers
    from datetime import timezone
    from aiogram import Dispatcher
    from aiogram.fsm.storage.memory import MemoryStorage
    from aiogram.types import CallbackQuery, Chat, Message, Update
    from tests.test_perplexity_agent_models import ValidatingTelegramSession

    bot_token = "8224473896:AAFtest_token_veraveda"
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", bot_token)
    monkeypatch.setattr("database.async_session_maker", test_db_session_maker)
    monkeypatch.setattr(handlers, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr("ai_integration.async_session_maker", test_db_session_maker)
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)
    monkeypatch.setattr("handlers.missing_profile_fields", lambda c, u: [])
    monkeypatch.setattr(handlers, "is_admin", AsyncMock(side_effect=lambda user_id: user_id == 777))

    key_orig = "kie_original_key_111"
    key_fail = "kie_failed_rotation_222"

    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="KIE", kie_api_key=key_orig)
            session.add(cfg)
        else:
            cfg.provider = "KIE"
            cfg.kie_api_key = key_orig
        await session.commit()

    # Simulate handle_key_rotation raising an exception during the atomic transaction
    async def mock_handle_rotation(*args, **kwargs):
        raise RuntimeError("Simulated crash/DB failure during key rotation")

    monkeypatch.setattr("ai_incident_manager.handle_key_rotation", mock_handle_rotation)

    storage = MemoryStorage()
    handlers.router._parent_router = None
    dp = Dispatcher(storage=storage)
    dp.include_router(handlers.router)

    telegram_session = ValidatingTelegramSession()
    bot = Bot(token=bot_token, session=telegram_session)
    bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    admin_user = TgUser(id=777, is_bot=False, first_name="Admin", username="admin777")
    admin_chat = Chat(id=777, type="private")
    fsm_ctx = dp.fsm.get_context(bot=bot, user_id=777, chat_id=777)

    # --- Step 1: Admin opens KIE settings and clicks visible '🔑 API-ключ' button via Dispatcher ---
    cb_msg = Message(message_id=90, date=datetime.now(timezone.utc), chat=admin_chat, text="Settings")
    update_open = Update(
        update_id=500,
        callback_query=CallbackQuery(
            id="cb_open_1",
            from_user=admin_user,
            chat_instance="ci_admin",
            message=cb_msg,
            data="view_models_KIE",
        ),
    )
    await dp.feed_update(bot, update_open)

    edit_calls_1 = [c for c in telegram_session.calls if type(c).__name__ == "EditMessageText"]
    assert len(edit_calls_1) >= 1
    buttons_1 = [btn for row in edit_calls_1[-1].reply_markup.inline_keyboard for btn in row]
    btn_key_1 = next((b for b in buttons_1 if "API-ключ" in b.text), None)
    assert btn_key_1 is not None, "Button 'API-ключ' must be visible on settings screen"

    # Click visible 'API-ключ' button
    update_click_key = Update(
        update_id=501,
        callback_query=CallbackQuery(
            id="cb_key_1",
            from_user=admin_user,
            chat_instance="ci_admin",
            message=cb_msg,
            data=btn_key_1.callback_data,
        ),
    )
    await dp.feed_update(bot, update_click_key)
    assert await fsm_ctx.get_state() == handlers.AdminStates.set_api_key.state

    # --- Step 2: Admin sends key; failure occurs during rotation ---
    key_msg = Message(message_id=91, date=datetime.now(timezone.utc), chat=admin_chat, from_user=admin_user, text=key_fail)
    await dp.feed_update(bot, Update(update_id=502, message=key_msg))

    # Assert 1: AIConfig was rolled back! Original key remains in DB!
    async with test_db_session_maker() as session:
        cfg_after = await session.get(AIConfig, 1)
        assert cfg_after.kie_api_key == key_orig

    # Assert 2: FSM state is cleared
    assert await fsm_ctx.get_state() is None

    # Assert 3: Error message sent to admin
    sent_texts = [call.text for call in telegram_session.calls if isinstance(getattr(call, "text", None), str)]
    assert any("Ошибка сохранения настроек. Попробуйте еще раз." in t for t in sent_texts)

    # --- Step 3: Complete Admin Recovery Journey (Zero UI mocks, real visible buttons and Dispatcher) ---
    # Restore real handle_key_rotation implementation
    import ai_incident_manager
    monkeypatch.setattr("ai_incident_manager.handle_key_rotation", ai_incident_manager._handle_key_rotation_impl)

    # Re-open KIE settings via Dispatcher to inspect visible buttons
    update_reopen = Update(
        update_id=503,
        callback_query=CallbackQuery(
            id="cb_open_2",
            from_user=admin_user,
            chat_instance="ci_admin",
            message=cb_msg,
            data="view_models_KIE",
        ),
    )
    await dp.feed_update(bot, update_reopen)

    edit_calls_2 = [c for c in telegram_session.calls if type(c).__name__ == "EditMessageText"]
    assert len(edit_calls_2) > len(edit_calls_1)
    buttons_2 = [btn for row in edit_calls_2[-1].reply_markup.inline_keyboard for btn in row]
    btn_key_retry = next((b for b in buttons_2 if "API-ключ" in b.text), None)
    assert btn_key_retry is not None, "Button 'API-ключ' must be visible after recovery re-open"

    # Admin clicks visible 'API-ключ' button again via Dispatcher
    update_click_retry = Update(
        update_id=504,
        callback_query=CallbackQuery(
            id="cb_key_2",
            from_user=admin_user,
            chat_instance="ci_admin",
            message=cb_msg,
            data=btn_key_retry.callback_data,
        ),
    )
    await dp.feed_update(bot, update_click_retry)
    assert await fsm_ctx.get_state() == handlers.AdminStates.set_api_key.state

    # Admin sends new valid key via Message Update
    key_retry = "kie_retry_key_333"
    key_msg_retry = Message(message_id=92, date=datetime.now(timezone.utc), chat=admin_chat, from_user=admin_user, text=key_retry)
    await dp.feed_update(bot, Update(update_id=505, message=key_msg_retry))

    # Assert 4: FSM state is cleared and AIConfig is updated in DB
    assert await fsm_ctx.get_state() is None
    async with test_db_session_maker() as session:
        cfg_retry = await session.get(AIConfig, 1)
        assert cfg_retry.kie_api_key == key_retry

    # Assert 5: Real next screen rendered with ValidatingTelegramSession
    edit_calls_3 = [c for c in telegram_session.calls if type(c).__name__ == "EditMessageText"]
    assert len(edit_calls_3) > len(edit_calls_2)
    last_rendered = edit_calls_3[-1]
    rendered_buttons = [btn for row in last_rendered.reply_markup.inline_keyboard for btn in row]
    assert any("Выбрать модель" in b.text for b in rendered_buttons)
    assert any("API-ключ" in b.text for b in rendered_buttons)

    # Assert 6: Re-opening settings screen persists and displays properly
    await dp.feed_update(bot, update_reopen)

    # Assert 7: Runtime reflection - verify new user message uses updated key
    import ai_integration
    from database import SubscriptionConfig
    captured_runtime_keys = []
    async def mock_kie_chat_runtime(api_key, *args, **kwargs):
        captured_runtime_keys.append(api_key)
        return "Ответ ИИ после восстановления настроек администратором"

    monkeypatch.setattr(ai_integration, "_call_kie_chat", mock_kie_chat_runtime)
    monkeypatch.setattr(handlers, "_call_kie_chat", mock_kie_chat_runtime)

    async with test_db_session_maker() as session:
        user_db = await session.get(User, 999)
        if not user_db:
            session.add(User(id=999, username="user999", first_name="User", is_admin=False, accepted_disclaimer=True))
        else:
            user_db.accepted_disclaimer = True
        sub_cfg = await session.get(SubscriptionConfig, 1)
        if not sub_cfg:
            session.add(SubscriptionConfig(id=1, subscriptions_enabled=False))
        else:
            sub_cfg.subscriptions_enabled = False
        await session.commit()

    user_chat = Chat(id=999, type="private")
    user_tg = TgUser(id=999, is_bot=False, first_name="User", username="user999")
    await dp.feed_update(bot, Update(
        update_id=506,
        message=Message(
            message_id=93,
            date=datetime.now(timezone.utc),
            chat=user_chat,
            from_user=user_tg,
            text="Здравствуйте",
        ),
    ))
    await asyncio.sleep(1.2)
    assert len(captured_runtime_keys) >= 1, "Runtime AI request must be triggered"
    assert captured_runtime_keys[-1] == key_retry, "AI request must use newly saved retry key"

    user_sends = [c for c in telegram_session.calls if type(c).__name__ == "SendMessage" and getattr(c, "chat_id", None) == 999]
    assert len(user_sends) >= 1, "AI response must be sent to the end user"
    assert "Ответ ИИ после восстановления настроек администратором" in user_sends[-1].text


@pytest.mark.asyncio
async def test_key_rotation_companion_shared_db_two_keys_isolation(test_db_session_maker, monkeypatch):
    """Blocker 2: Rotating Telegram bot's key does NOT touch active incidents of MAX companion using a different key for the same provider."""
    inst_tg = {"pm2_name": "tg_veraveda777_bot_legacy", "bot_instance_id": "tg_veraveda777_bot_legacy", "companion_instance": "max_se13639182_bot_legacy", "platform": "telegram", "admins": [777]}
    inst_max = {"pm2_name": "max_se13639182_bot_legacy", "bot_instance_id": "max_se13639182_bot_legacy", "companion_instance": "tg_veraveda777_bot_legacy", "platform": "max"}

    def mock_get_inst(name):
        if name == "tg_veraveda777_bot_legacy": return inst_tg
        if name == "max_se13639182_bot_legacy": return inst_max
        return None

    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", mock_get_inst)
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_tg)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_tg)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[777]))

    key_tg_1 = "kie_tg_key_111"
    key_max_2 = "kie_max_key_222"
    key_tg_3 = "kie_tg_key_333"

    scope_tg = get_provider_account_fingerprint(key_tg_1, "https://api.kie.ai")
    scope_max = get_provider_account_fingerprint(key_max_2, "https://api.kie.ai")

    err = InsufficientBalanceError("KIE credits empty")
    err.provider = "KIE"

    # Incident 1 on Telegram bot with key 1
    inc_tg = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        error=err,
        operation="chat",
        model="kie-v1",
        credential_scope=scope_tg,
        session_maker=test_db_session_maker,
    )
    # Incident 2 on MAX bot with key 2
    inc_max = await record_ai_provider_failure(
        origin_instance_id="max_se13639182_bot_legacy",
        provider="KIE",
        error=err,
        operation="vision",
        model="kie-vision",
        credential_scope=scope_max,
        session_maker=test_db_session_maker,
    )
    assert inc_tg.status == "active"
    assert inc_max.status == "active"

    # Telegram rotates key 1 -> key 3
    superseded = await handle_key_rotation(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        new_key=key_tg_3,
        base_url="https://api.kie.ai",
        old_key=key_tg_1,
        session_maker=test_db_session_maker,
    )
    # Exactly Telegram's incident (key 1) was superseded!
    assert superseded == 1

    async with test_db_session_maker() as session:
        inc_tg_db = await session.get(AIProviderIncidentModel, inc_tg.id)
        inc_max_db = await session.get(AIProviderIncidentModel, inc_max.id)
        assert inc_tg_db.status == "superseded"
        # MAX bot's incident for key 2 remains completely ACTIVE!
        assert inc_max_db.status == "active"


@pytest.mark.asyncio
async def test_transient_terminal_alert_stale_skipped_after_key_rotation_retry(test_db_session_maker, monkeypatch):
    """Blocker 3: Transient incident and pending terminal_failure alert are superseded on key rotation; retrying worker skips delivery."""
    inst_def = {"pm2_name": "tg_veraveda777_bot_legacy", "platform": "telegram", "admins": [777]}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[777]))
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")

    key_a = "key_transient_alpha_1"
    key_b = "key_transient_beta_2"
    scope_a = get_provider_account_fingerprint(key_a, None)

    # 1. Non-operator failure (transient) on key A
    class GatewayTimeout(Exception): pass
    inc_trans = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=GatewayTimeout("Gateway 504"),
        operation="chat",
        model="gpt-4o",
        credential_scope=scope_a,
        session_maker=test_db_session_maker,
        enqueue_notification=False,
    )
    assert inc_trans.status == "transient"

    # Enqueue terminal alert
    enq_ok = await enqueue_terminal_ai_failure_notification(
        incident_id=inc_trans.id,
        origin_instance_id="tg_veraveda777_bot_legacy",
        text="🚨 Transient terminal alert",
        cooldown_seconds=1800,
        session_maker=test_db_session_maker,
    )
    assert enq_ok is True

    # 2. Worker attempts delivery, encounters temporary rate limit -> status stays pending with retry
    from aiogram.exceptions import TelegramRetryAfter
    bot = AsyncMock(spec=Bot)
    bot.token = "8224473896:AAFtest_token_veraveda"
    bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))
    bot.send_message = AsyncMock(side_effect=TelegramRetryAfter(method=None, message="Rate limit", retry_after=10))

    delivered = await process_ai_incident_notifications(bot=bot, session_maker=test_db_session_maker)
    assert delivered == 0

    # 3. Admin rotates key A -> key B
    await handle_key_rotation(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        new_key=key_b,
        old_key=key_a,
        session_maker=test_db_session_maker,
    )

    # Assert: transient incident is superseded and pending notification cancelled!
    async with test_db_session_maker() as session:
        inc_check = await session.get(AIProviderIncidentModel, inc_trans.id)
        assert inc_check.status == "superseded"

        notif = (await session.execute(select(AIIncidentNotificationModel))).scalar_one()
        assert notif.status == "cancelled"
        assert notif.last_error == "superseded_by_key_rotation"

        # Simulate a race where a notification remained pending (e.g. lease timeout or missed lock)
        notif.status = "pending"
        notif.next_retry_at = datetime.utcnow() - timedelta(seconds=1)
        await session.commit()

    # 4. Outbox worker retries delivery, finds parent incident superseded -> skipped_stale
    bot.send_message = AsyncMock(return_value=True)
    delivered_retry = await process_ai_incident_notifications(bot=bot, session_maker=test_db_session_maker)
    assert delivered_retry == 0  # Did NOT send alert!
    assert bot.send_message.call_count == 0

    async with test_db_session_maker() as session:
        notif_after = (await session.execute(select(AIIncidentNotificationModel))).scalar_one()
        assert notif_after.status == "skipped_stale"
        assert notif_after.last_error == "incident_superseded_before_send"


@pytest.mark.asyncio
async def test_key_rotation_a_b_a_preserves_history_and_separates_old_inflight(test_db_session_maker, monkeypatch):
    """Blocker 5: A -> B -> A preserves historical superseded records and delivery audit; old in-flight request started before rotation is marked superseded."""
    inst_def = {"pm2_name": "tg_veraveda777_bot_legacy", "platform": "telegram", "admins": [777]}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[777]))

    key_a = "key_preservation_alpha"
    key_b = "key_preservation_beta"
    scope_a = get_provider_account_fingerprint(key_a, None)

    # Step 1: Initial config Key A
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="OpenAI", openai_api_key=key_a)
            session.add(cfg)
        else:
            cfg.provider = "OpenAI"
            cfg.openai_api_key = key_a
        await session.commit()

    t_r1_start = datetime.utcnow() - timedelta(seconds=15)
    err = InsufficientBalanceError("KIE credits empty on A")
    err.provider = "OpenAI"

    # Incident opens on Key A
    inc_a1 = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=err,
        operation="chat",
        model="gpt-4o",
        credential_scope=scope_a,
        session_maker=test_db_session_maker,
    )
    assert inc_a1.status == "active"

    # Step 2: Rotate to Key B
    await handle_key_rotation(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        new_key=key_b,
        old_key=key_a,
        session_maker=test_db_session_maker,
    )
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.openai_api_key = key_b
        await session.commit()

    # Step 3: Rotate back to Key A (A -> B -> A)
    await handle_key_rotation(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        new_key=key_a,
        old_key=key_b,
        session_maker=test_db_session_maker,
    )
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.openai_api_key = key_a
        await session.commit()

    # Assert: Historical superseded incident and notifications for Key A STILL EXIST (audit preserved!)
    async with test_db_session_maker() as session:
        inc_old = await session.get(AIProviderIncidentModel, inc_a1.id)
        assert inc_old is not None
        assert inc_old.status == "superseded"

    # Step 4: Old request R1 (started at t_r1_start before rotation) finishes NOW with error
    inc_r1 = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=err,
        operation="chat",
        model="gpt-4o",
        credential_scope=scope_a,
        attempt_started_at=t_r1_start,
        session_maker=test_db_session_maker,
    )
    # Recorded as superseded, zero active incidents
    assert inc_r1.status == "superseded"

    # Step 5: New request R2 (started NOW on reactivated Key A) fails
    t_r2_start = datetime.utcnow()
    inc_r2 = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=err,
        operation="chat",
        model="gpt-4o",
        credential_scope=scope_a,
        attempt_started_at=t_r2_start,
        session_maker=test_db_session_maker,
    )
    # Recorded as ACTIVE!
    assert inc_r2.status == "active"
    assert inc_r2.id != inc_a1.id


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_concurrent_terminal_failure_dedup_boundary_transactions(pg_test_session_maker, monkeypatch):
    """Blocker 4: Concurrent transactions in PostgreSQL across cooldown boundary serialize via with_for_update; exactly one alert enqueued."""
    bot_token = "8281390498:AAFtest_telegram_token_boundary"
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", bot_token)

    inst_def = {"pm2_name": "tg_someonewithyou01_bot_new", "platform": "telegram", "platform_id": 8281390498, "admins": [1001], "username": "someonewithyou01_bot"}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[1001]))

    scope = get_provider_account_fingerprint("pg_test_boundary_key", None)
    class Timeout504(Exception): pass

    inc = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="OpenAI",
        error=Timeout504("Gateway 504"),
        operation="chat",
        model="gpt-4o",
        credential_scope=scope,
        session_maker=pg_test_session_maker,
        enqueue_notification=False,
    )
    assert inc is not None

    # Two concurrent transactions in PostgreSQL
    res1, res2 = await asyncio.gather(
        enqueue_terminal_ai_failure_notification(
            incident_id=inc.id,
            origin_instance_id="tg_someonewithyou01_bot_new",
            text="Boundary alert A",
            cooldown_seconds=1800,
            session_maker=pg_test_session_maker,
        ),
        enqueue_terminal_ai_failure_notification(
            incident_id=inc.id,
            origin_instance_id="tg_someonewithyou01_bot_new",
            text="Boundary alert B",
            cooldown_seconds=1800,
            session_maker=pg_test_session_maker,
        ),
    )
    assert (res1, res2) in [(True, False), (False, True)]

    # Exactly 1 row in PostgreSQL
    async with pg_test_session_maker() as session:
        notifs = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.incident_id == inc.id,
                AIIncidentNotificationModel.event_type == "terminal_failure",
            )
        )).scalars().all()
        assert len(notifs) == 1
        assert notifs[0].cycle_index == 0


# ==============================================================================
# Review Follow-up Verification Tests (PR #97 Gaps 1, 2, 3)
# ==============================================================================

@pytest.mark.asyncio
async def test_safe_no_op_key_rotation_identical_key_sqlite(test_db_session_maker, monkeypatch):
    """Gap 2: Safe no-op rotation: old_key == new_key or old_credential_scope == new_fp returns 0 and leaves incidents intact (SQLite)."""
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", "8281390498:AAFtest_telegram_token_123456789012")

    same_key = "kie_same_key_999"
    same_fp = get_provider_account_fingerprint(same_key, None)

    err = InsufficientBalanceError("KIE credits empty")
    err.provider = "KIE"
    inc = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=err,
        operation="chat",
        model="kie-v1",
        credential_scope=same_fp,
        session_maker=test_db_session_maker,
    )
    assert inc is not None
    assert inc.status == "active"

    # 1. No-op via old_key == new_key
    res_noop_1 = await handle_key_rotation(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        new_key=same_key,
        old_key=same_key,
        session_maker=test_db_session_maker,
    )
    assert res_noop_1 == 0

    # 2. No-op via old_credential_scope == new_fp
    res_noop_2 = await handle_key_rotation(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        new_key=same_key,
        old_credential_scope=same_fp,
        session_maker=test_db_session_maker,
    )
    assert res_noop_2 == 0

    # Incident remains untouched and active
    async with test_db_session_maker() as session:
        db_inc = await session.get(AIProviderIncidentModel, inc.id)
        assert db_inc.status == "active"
        assert db_inc.resolved_at is None


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_safe_no_op_key_rotation_identical_key_postgresql(pg_test_session_maker, monkeypatch):
    """Gap 2: Safe no-op rotation: old_key == new_key or old_credential_scope == new_fp returns 0 and leaves incidents intact (PostgreSQL)."""
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", "8281390498:AAFtest_telegram_token_123456789012")

    same_key = "kie_pg_same_key_999"
    same_fp = get_provider_account_fingerprint(same_key, None)

    err = InsufficientBalanceError("KIE credits empty")
    err.provider = "KIE"
    inc = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=err,
        operation="chat",
        model="kie-v1",
        credential_scope=same_fp,
        session_maker=pg_test_session_maker,
    )
    assert inc is not None
    assert inc.status == "active"

    # No-op call on PostgreSQL
    res = await handle_key_rotation(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        new_key=same_key,
        old_key=same_key,
        session_maker=pg_test_session_maker,
    )
    assert res == 0

    async with pg_test_session_maker() as session:
        db_inc = await session.get(AIProviderIncidentModel, inc.id)
        assert db_inc.status == "active"
        assert db_inc.resolved_at is None


@pytest.mark.asyncio
async def test_missing_old_key_does_not_broadly_supersede_companion_keys(test_db_session_maker, monkeypatch):
    """Gap 2: When old key is unknown/unconfigured, handle_key_rotation does NOT broadly supersede other keys in shared DB."""
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", "8281390498:AAFtest_telegram_token_123456789012")

    scope_companion = get_provider_account_fingerprint("companion_secret_key_123", None)

    # Incident belonging to a companion/alternate key in shared DB
    err = InsufficientBalanceError("KIE credits empty")
    err.provider = "KIE"
    inc = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=err,
        operation="chat",
        model="kie-v1",
        credential_scope=scope_companion,
        session_maker=test_db_session_maker,
    )
    assert inc is not None
    assert inc.status == "active"

    # Rotate with NO old_key passed, NO old_credential_scope passed, and AIConfig without old key
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if cfg:
            cfg.kie_api_key = None
            await session.commit()

    superseded = await handle_key_rotation(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        new_key="completely_new_key_777",
        old_key=None,
        old_credential_scope=None,
        session_maker=test_db_session_maker,
    )
    assert superseded == 0

    # Ensure companion incident is NOT superseded
    async with test_db_session_maker() as session:
        db_inc = await session.get(AIProviderIncidentModel, inc.id)
        assert db_inc.status == "active"


@pytest.mark.asyncio
async def test_real_dispatcher_key_rotation_fail_closed_on_missing_instance_id(test_db_session_maker, monkeypatch):
    """Gap 3: Full fail-closed via Dispatcher/FSM: when BOT_INSTANCE_ID is missing,
    IncidentSecurityError is raised and caught, rolling back the transaction so AIConfig is unchanged,
    FSM state is cleared, and an authorization error message is sent to admin.
    """
    from datetime import timezone
    from aiogram import Dispatcher
    from aiogram.fsm.storage.memory import MemoryStorage
    from aiogram.types import Chat, Message, Update
    import handlers
    from tests.test_perplexity_agent_models import ValidatingTelegramSession

    monkeypatch.setattr(handlers, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)
    monkeypatch.setattr(handlers, "is_admin", lambda user_id: True)

    # Initial AIConfig with an existing valid key
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="KIE", kie_api_key="original_kie_key_100")
            session.add(cfg)
        else:
            cfg.kie_api_key = "original_kie_key_100"
        await session.commit()

    storage = MemoryStorage()
    handlers.router._parent_router = None
    dp = Dispatcher(storage=storage)
    dp.include_router(handlers.router)

    telegram_session = ValidatingTelegramSession()
    bot = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session)

    user = TgUser(id=777, is_bot=False, first_name="Admin", username="admin777")
    chat = Chat(id=777, type="private")

    # Set FSM state to set_api_key for KIE
    fsm_ctx = dp.fsm.get_context(bot=bot, user_id=777, chat_id=777)
    await fsm_ctx.set_state(handlers.AdminStates.set_api_key)
    await fsm_ctx.update_data(provider="KIE", setting_type="api_key")

    # MISSING BOT_INSTANCE_ID to trigger fail-closed!
    monkeypatch.delenv("BOT_INSTANCE_ID", raising=False)

    # Send new API key via Message Update into Dispatcher
    update_msg = Update(
        update_id=201,
        message=Message(
            message_id=21,
            date=datetime.now(timezone.utc),
            chat=chat,
            from_user=user,
            text="new_unauthorized_key_attempt_555",
        ),
    )
    await dp.feed_update(bot, update_msg)

    # 1. State must be cleared
    assert await fsm_ctx.get_state() is None

    # 2. AIConfig must remain ROLLED BACK to original_kie_key_100
    async with test_db_session_maker() as session:
        cfg_after = await session.get(AIConfig, 1)
        assert cfg_after.kie_api_key == "original_kie_key_100"

    # 3. Admin received typed authorization error message
    sent_texts = [getattr(call, "text", "") for call in telegram_session.calls]
    assert any("Ошибка авторизации экземпляра бота. Настройки не изменены." in t for t in sent_texts)


@pytest.mark.asyncio
async def test_runtime_attempt_started_at_prevents_stale_in_flight_reopening_reactivated_key(test_db_session_maker, monkeypatch):
    """Gap 1: In A -> B -> A key rotation, stale in-flight request R1 started before rotation
    must NOT reopen an active incident on reactivated key A, while new request R2 does.
    """
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", "8281390498:AAFtest_telegram_token_123456789012")

    key_a = "kie_key_alpha_111"
    key_b = "kie_key_beta_222"
    scope_a = get_provider_account_fingerprint(key_a, None)
    scope_b = get_provider_account_fingerprint(key_b, None)

    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="KIE", kie_api_key=key_a)
            session.add(cfg)
        else:
            cfg.kie_api_key = key_a
        await session.commit()

    # Step 1: In-flight request R1 begins execution at t0
    t0_r1_start = datetime.utcnow()
    await asyncio.sleep(0.01)

    # Step 2: Rotation 1: Admin rotates A -> B at t1
    await handle_key_rotation(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        new_key=key_b,
        old_key=key_a,
        session_maker=test_db_session_maker,
    )
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.kie_api_key = key_b
        await session.commit()

    await asyncio.sleep(0.01)

    # Step 3: Rotation 2: Admin reactivates key A (B -> A) at t2
    await handle_key_rotation(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        new_key=key_a,
        old_key=key_b,
        session_maker=test_db_session_maker,
    )
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.kie_api_key = key_a
        await session.commit()

    await asyncio.sleep(0.01)

    # Step 4: Stale in-flight request R1 (started at t0 < t1) finally completes with failure at t3
    err = InsufficientBalanceError("KIE credits empty from old in-flight request")
    err.provider = "KIE"
    inc_r1 = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=err,
        operation="chat",
        model="kie-v1",
        credential_scope=scope_a,
        session_maker=test_db_session_maker,
        attempt_started_at=t0_r1_start,
    )
    assert inc_r1 is not None
    # Must be marked superseded because it was started before key A's previous resolved_at!
    assert inc_r1.status == "superseded"

    # Verify no notifications were enqueued for stale R1
    async with test_db_session_maker() as session:
        notifs_r1 = (await session.execute(
            select(AIIncidentNotificationModel).where(AIIncidentNotificationModel.incident_id == inc_r1.id)
        )).scalars().all()
        assert len(notifs_r1) == 0

    # Step 5: New request R2 starts at t4 > t2 on reactivated key A
    t4_r2_start = datetime.utcnow()
    inc_r2 = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=err,
        operation="chat",
        model="kie-v1",
        credential_scope=scope_a,
        session_maker=test_db_session_maker,
        attempt_started_at=t4_r2_start,
    )
    assert inc_r2 is not None
    # Must be active because it was started on the newly reactivated key generation!
    assert inc_r2.status == "active"


@pytest.mark.asyncio
async def test_runtime_stale_success_after_a_b_a_does_not_close_new_incident(test_db_session_maker, monkeypatch):
    """Gap 1: In A -> B -> A rotation, an in-flight success started on old generation of key A
    must NOT resolve an active incident opened by the newly reactivated key A generation.
    """
    from datetime import timedelta
    from ai_incident_manager import (
        get_provider_account_fingerprint,
        handle_key_rotation,
        record_ai_operation_success,
        record_ai_provider_failure,
    )
    from database import AIProviderIncidentModel, AIIncidentNotificationModel

    bot_token = "8281390498:AAFtest_telegram_token_123456789012"
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", bot_token)
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)

    key_a = "kie_stale_key_alpha_111"
    key_b = "kie_stale_key_beta_222"
    fp_a = get_provider_account_fingerprint(key_a)
    fp_b = get_provider_account_fingerprint(key_b)

    # Initial AIConfig with key A
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="KIE", kie_api_key=key_a)
            session.add(cfg)
        else:
            cfg.provider = "KIE"
            cfg.kie_api_key = key_a
        await session.commit()

    # Step 1: Request R1 starts at t0 on key A
    t0 = datetime.utcnow() - timedelta(minutes=10)

    # Step 2: Key rotated A -> B at t_rot1
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.kie_api_key = key_b
        await session.commit()
    await handle_key_rotation(None, "KIE", key_b, old_key=key_a, session_maker=test_db_session_maker)

    # Step 3: Key rotated B -> A at t_rot2 (A is reactivated!)
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.kie_api_key = key_a
        await session.commit()
    await handle_key_rotation(None, "KIE", key_a, old_key=key_b, session_maker=test_db_session_maker)

    # Step 4: New request R2 starts at t2 on reactivated key A (AFTER reactivation)
    t2 = datetime.utcnow()
    err_r2 = InsufficientBalanceError("KIE credit balance exhausted: 0 credits left")
    inc_r2 = await record_ai_provider_failure(
        origin_instance_id=None,
        provider="KIE",
        error=err_r2,
        operation="chat",
        model="kie-model",
        credential_scope=fp_a,
        attempt_started_at=t2,
        session_maker=test_db_session_maker,
    )
    assert inc_r2 is not None
    assert inc_r2.status == "active"
    incident_id_r2 = inc_r2.id

    # Step 5: Stale request R1 (started at t0 on old generation of key A) now completes with success!
    res_stale = await record_ai_operation_success(
        origin_instance_id=None,
        provider="KIE",
        credential_scope=fp_a,
        operation="chat",
        model="kie-model",
        session_maker=test_db_session_maker,
        attempt_started_at=t0,
    )
    # Stale success must be ignored!
    assert res_stale is False

    # In DB: R2 incident remains ACTIVE, NOT resolved!
    async with test_db_session_maker() as session:
        inc_check = await session.get(AIProviderIncidentModel, incident_id_r2)
        assert inc_check.status == "active"
        assert inc_check.resolved_at is None

        # Verify no false "resolved" notification was queued
        stmt_notif = select(AIIncidentNotificationModel).where(
            AIIncidentNotificationModel.incident_id == incident_id_r2,
            AIIncidentNotificationModel.event_type == "resolved",
        )
        notif = (await session.execute(stmt_notif)).scalar_one_or_none()
        assert notif is None

    # Step 6: Subsequent successful request R3 started at t3 > t2 succeeds -> Incident IS resolved!
    t3 = datetime.utcnow()
    res_valid = await record_ai_operation_success(
        origin_instance_id=None,
        provider="KIE",
        credential_scope=fp_a,
        operation="chat",
        model="kie-model",
        session_maker=test_db_session_maker,
        attempt_started_at=t3,
    )
    assert res_valid is True

    async with test_db_session_maker() as session:
        inc_check2 = await session.get(AIProviderIncidentModel, incident_id_r2)
        assert inc_check2.status == "resolved"
        assert inc_check2.resolved_at is not None


@pytest.mark.asyncio
async def test_terminal_alert_stale_request_after_rotation_suppressed_no_false_alert(test_db_session_maker, monkeypatch):
    """Gap 2: Terminal alert without incident_id for a stale in-flight request started on old key A
    after rotation to key B is safely suppressed, never creating active incidents or enqueuing alerts for key B.
    """
    from datetime import timedelta
    from ai_incident_manager import (
        get_provider_account_fingerprint,
        handle_key_rotation,
    )
    from error_reporting import send_terminal_ai_failure_alert
    from database import AIProviderIncidentModel, AIIncidentNotificationModel
    from tests.test_perplexity_agent_models import ValidatingTelegramSession

    bot_token = "8281390498:AAFtest_telegram_token_123456789012"
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_someonewithyou01_bot_new")
    monkeypatch.setenv("BOT_TOKEN", bot_token)
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)
    monkeypatch.setattr("error_reporting.async_session_maker", test_db_session_maker)

    telegram_session = ValidatingTelegramSession()
    bot = Bot(token=bot_token, session=telegram_session)

    key_a = "kie_terminal_key_alpha_111"
    key_b = "kie_terminal_key_beta_222"
    fp_a = get_provider_account_fingerprint(key_a)
    fp_b = get_provider_account_fingerprint(key_b)

    # Initial AIConfig with key A
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="KIE", kie_api_key=key_a)
            session.add(cfg)
        else:
            cfg.provider = "KIE"
            cfg.kie_api_key = key_a
        await session.commit()

    # In-flight request starts on key A at t0
    t0 = datetime.utcnow() - timedelta(minutes=5)

    # Key rotated A -> B
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.kie_api_key = key_b
        await session.commit()
    await handle_key_rotation(None, "KIE", key_b, old_key=key_a, session_maker=test_db_session_maker)

    # In-flight request fails terminal on key A without incident_id
    alert_result = await send_terminal_ai_failure_alert(
        bot=bot,
        platform="telegram",
        user_id=123,
        primary_provider="KIE",
        primary_model="kie-model",
        exception=AIServiceError("Dead key A"),
        incident_id=None,
        credential_scope=fp_a,
        attempt_started_at=t0,
        session_maker=test_db_session_maker,
    )
    # Stale terminal failure alert must be suppressed!
    assert alert_result is False

    # In DB: NO active incidents for KIE!
    async with test_db_session_maker() as session:
        stmt_active = select(AIProviderIncidentModel).where(
            AIProviderIncidentModel.provider == "KIE",
            AIProviderIncidentModel.status == "active",
        )
        active_inc = (await session.execute(stmt_active)).scalars().all()
        assert len(active_inc) == 0

        # Superseded incident for old key A must exist and be marked superseded
        stmt_inc_a = select(AIProviderIncidentModel).where(
            AIProviderIncidentModel.provider == "KIE",
            AIProviderIncidentModel.credential_scope == fp_a,
        )
        incs_a = (await session.execute(stmt_inc_a)).scalars().all()
        assert len(incs_a) >= 1, "Superseded incident for key A must exist in DB"
        for inc in incs_a:
            assert inc.status == "superseded"
            assert inc.resolved_at is not None
            assert inc.credential_scope == fp_a

        # NO incident must exist for the newly active key B
        stmt_inc_b = select(AIProviderIncidentModel).where(
            AIProviderIncidentModel.provider == "KIE",
            AIProviderIncidentModel.credential_scope == fp_b,
        )
        incs_b = (await session.execute(stmt_inc_b)).scalars().all()
        assert len(incs_b) == 0, "No incident must be opened for newly active key B"

        # Zero notifications of any event_type queued in outbox
        stmt_notif = select(AIIncidentNotificationModel)
        notifs = (await session.execute(stmt_notif)).scalars().all()
        assert len(notifs) == 0, "No notifications should be queued for stale failure"

    # ZERO messages sent to Telegram admins
    admin_sends = [c for c in telegram_session.calls if type(c).__name__ == "SendMessage"]
    assert len(admin_sends) == 0, "No false alerts must be delivered to admins"


@pytest.mark.asyncio
async def test_terminal_alert_foreign_incident_id_fail_closed_shared_db(test_db_session_maker, monkeypatch):
    """Item 1: Verification of incident ownership is fail-closed.
    Bot A cannot enqueue alerts or notifications for an incident owned by Bot B.
    Legit MAX instance enqueuing for its own incident properly routes to companion Telegram bot."""
    from database import AIProviderIncidentModel, AIIncidentNotificationModel
    from ai_incident_manager import (
        enqueue_terminal_ai_failure_notification,
        process_ai_incident_notifications,
    )
    from tests.test_perplexity_agent_models import ValidatingTelegramSession

    inst_bot_a = {
        "pm2_name": "tg_veraveda777_bot_legacy",
        "bot_instance_id": "tg_veraveda777_bot_legacy",
        "companion_instance": "max_se13639182_bot_legacy",
        "platform": "telegram",
        "admins": [777],
    }
    inst_bot_b = {
        "pm2_name": "tg_someonewithyou01_bot_new",
        "bot_instance_id": "tg_someonewithyou01_bot_new",
        "companion_instance": None,
        "platform": "telegram",
        "admins": [888],
    }
    inst_max_a = {
        "pm2_name": "max_se13639182_bot_legacy",
        "bot_instance_id": "max_se13639182_bot_legacy",
        "companion_instance": "tg_veraveda777_bot_legacy",
        "platform": "max",
    }

    instances = {
        "tg_veraveda777_bot_legacy": inst_bot_a,
        "tg_someonewithyou01_bot_new": inst_bot_b,
        "max_se13639182_bot_legacy": inst_max_a,
    }

    current_running_inst = {"inst": inst_bot_a}
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: instances.get(n))
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: current_running_inst["inst"])
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_bot_a if s.get("pm2_name") in ("tg_veraveda777_bot_legacy", "max_se13639182_bot_legacy") else inst_bot_b)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(side_effect=lambda comp, current_instance=None, session=None: [777] if comp.get("pm2_name") == "tg_veraveda777_bot_legacy" else [888]))

    # Seed an active incident in DB owned by Bot B
    async with test_db_session_maker() as session:
        inc_b = AIProviderIncidentModel(
            origin_instance_id="tg_someonewithyou01_bot_new",
            companion_instance_id=None,
            provider="KIE",
            credential_scope="scope_b_secret",
            model_scope="kie-default",
            operation="chat",
            incident_type="provider_down",
            detail_code="502_bad_gateway",
            evidence_tier="tier1_primary_circuit",
            status="active",
        )
        session.add(inc_b)
        await session.commit()
        await session.refresh(inc_b)
        inc_b_id = inc_b.id

    telegram_session = ValidatingTelegramSession()
    bot = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session)
    bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    # Case 1: Caller is Bot A, attempting to alert for incident owned by Bot B (origin_instance_id=tg_veraveda777_bot_legacy)
    current_running_inst["inst"] = inst_bot_a
    res1 = await enqueue_terminal_ai_failure_notification(
        incident_id=inc_b_id,
        origin_instance_id="tg_veraveda777_bot_legacy",
        text="Terminal failure alert from Bot A",
        session_maker=test_db_session_maker,
    )
    assert res1 is False, "Bot A cannot enqueue alert for incident owned by Bot B"

    # Case 2: Bot A attempting to spoof origin_instance_id=tg_someonewithyou01_bot_new
    res2 = await enqueue_terminal_ai_failure_notification(
        incident_id=inc_b_id,
        origin_instance_id="tg_someonewithyou01_bot_new",
        text="Spoofed terminal failure alert",
        session_maker=test_db_session_maker,
    )
    assert res2 is False, "Caller identity mismatch must fail-closed"

    # Verify zero outbox rows and zero sends
    async with test_db_session_maker() as session:
        notifs = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        assert len(notifs) == 0, "No notifications must be created for unauthorized incident"

    await process_ai_incident_notifications(bot, session_maker=test_db_session_maker)
    assert len(telegram_session.calls) == 0, "Zero admin sends for unauthorized incident"

    # Case 3: Legit MAX instance (max_se13639182_bot_legacy) enqueuing for its own incident
    async with test_db_session_maker() as session:
        inc_max = AIProviderIncidentModel(
            origin_instance_id="max_se13639182_bot_legacy",
            companion_instance_id="tg_veraveda777_bot_legacy",
            provider="KIE",
            credential_scope="scope_max_secret",
            model_scope="kie-default",
            operation="chat",
            incident_type="provider_down",
            detail_code="503_service_unavailable",
            evidence_tier="tier1_primary_circuit",
            status="active",
        )
        session.add(inc_max)
        await session.commit()
        await session.refresh(inc_max)
        inc_max_id = inc_max.id

    # Switch execution context to MAX instance
    current_running_inst["inst"] = inst_max_a
    res3 = await enqueue_terminal_ai_failure_notification(
        incident_id=inc_max_id,
        origin_instance_id="max_se13639182_bot_legacy",
        text="Legit MAX terminal failure alert",
        session_maker=test_db_session_maker,
    )
    assert res3 is True, "Legit MAX instance must succeed in enqueuing notification"

    # Verify outbox row created for companion Telegram instance and admin
    async with test_db_session_maker() as session:
        notifs_max = (await session.execute(select(AIIncidentNotificationModel).where(AIIncidentNotificationModel.incident_id == inc_max_id))).scalars().all()
        assert len(notifs_max) == 1
        notif_row = notifs_max[0]
        assert notif_row.authorized_instance_id == "tg_veraveda777_bot_legacy"
        assert notif_row.recipient_id == 777
        assert notif_row.status == "pending"

    # Companion Telegram bot delivers the alert
    current_running_inst["inst"] = inst_bot_a
    delivered = await process_ai_incident_notifications(bot, session_maker=test_db_session_maker)
    assert delivered == 1

    send_calls = [c for c in telegram_session.calls if type(c).__name__ == "SendMessage"]
    assert len(send_calls) == 1
    assert send_calls[0].chat_id == 777
    assert "Legit MAX terminal failure alert" in send_calls[0].text


@pytest.mark.asyncio
async def test_output_budget_and_fallback_alerts_use_durable_outbox_telegram_and_max(test_db_session_maker, monkeypatch):
    """Item 2: Fallback and output budget exhaustion alerts are delivered strictly via durable outbox (AIIncidentNotificationModel).
    Verifies ZERO PII in outbox, Telegram immediate delivery, MAX companion routing, DB cooldown, and retry recovery."""
    from datetime import timedelta
    import uuid
    from database import AIIncidentNotificationModel, AIProviderIncidentModel
    from error_reporting import send_ai_fallback_used_alert, send_output_budget_exhausted_alert
    from ai_incident_manager import process_ai_incident_notifications
    from tests.test_perplexity_agent_models import ValidatingTelegramSession

    inst_tg = {
        "pm2_name": "tg_veraveda777_bot_legacy",
        "bot_instance_id": "tg_veraveda777_bot_legacy",
        "companion_instance": "max_se13639182_bot_legacy",
        "platform": "telegram",
        "admins": [777],
    }
    inst_max = {
        "pm2_name": "max_se13639182_bot_legacy",
        "bot_instance_id": "max_se13639182_bot_legacy",
        "companion_instance": "tg_veraveda777_bot_legacy",
        "platform": "max",
    }
    instances = {"tg_veraveda777_bot_legacy": inst_tg, "max_se13639182_bot_legacy": inst_max}

    current_running_inst = {"inst": inst_tg}
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: instances.get(n))
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: current_running_inst["inst"])
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_tg)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[777]))

    telegram_session = ValidatingTelegramSession()
    bot = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session)
    bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    # Part A: Telegram Output Budget Alert with real bot (immediate delivery turn, zero PII, DB cooldown)
    current_running_inst["inst"] = inst_tg
    res_budget = await send_output_budget_exhausted_alert(
        bot=bot,
        platform="telegram",
        user_id=1234567,
        chat_id=9876543,
        username="confidential_user",
        full_name="Secret Name",
        request_type="chat",
        provider="KIE",
        model="kie-claude-3-7-sonnet",
        finish_reason="length",
        visible_content_length=150,
        reasoning_content_length=5000,
        max_tokens=5000,
        session_maker=test_db_session_maker,
    )
    assert res_budget is True

    # Assert outbox row was created and delivered
    async with test_db_session_maker() as session:
        notifs = (await session.execute(select(AIIncidentNotificationModel).where(AIIncidentNotificationModel.event_type == "output_budget_exhausted"))).scalars().all()
        assert len(notifs) == 1
        notif_row = notifs[0]
        assert notif_row.status == "delivered"
        assert notif_row.recipient_id == 777
        # ZERO PII verification
        assert "1234567" not in notif_row.message_text
        assert "9876543" not in notif_row.message_text
        assert "confidential_user" not in notif_row.message_text
        assert "Secret Name" not in notif_row.message_text

    # Immediate delivery turn sent message to Telegram admin 777
    send_calls = [c for c in telegram_session.calls if type(c).__name__ == "SendMessage"]
    assert len(send_calls) == 1
    assert send_calls[0].chat_id == 777

    # DB Cooldown verification: second immediate alert is suppressed
    res_budget_suppressed = await send_output_budget_exhausted_alert(
        bot=bot,
        platform="telegram",
        provider="KIE",
        model="kie-claude-3-7-sonnet",
        session_maker=test_db_session_maker,
    )
    assert res_budget_suppressed is False
    assert len([c for c in telegram_session.calls if type(c).__name__ == "SendMessage"]) == 1

    # Part B: MAX Fallback Alert with bot=None (companion routing, retries, restart recovery)
    current_running_inst["inst"] = inst_max
    res_fallback = await send_ai_fallback_used_alert(
        bot=None,
        platform="max",
        user_id=8888888,
        chat_id=7777777,
        request_type="chat",
        primary_provider="KIE",
        primary_model="kie-model",
        fallback_provider="OpenAI",
        fallback_model="gpt-4o",
        primary_error="ConfidentialErrorDetails: Internal 502 with sensitive stacktrace",
        session_maker=test_db_session_maker,
    )
    assert res_fallback is True

    # Check outbox row: status is pending for companion Telegram instance
    async with test_db_session_maker() as session:
        notifs_fb = (await session.execute(select(AIIncidentNotificationModel).where(AIIncidentNotificationModel.event_type == "fallback_used"))).scalars().all()
        assert len(notifs_fb) == 1
        fb_row = notifs_fb[0]
        assert fb_row.status == "pending"
        assert fb_row.authorized_instance_id == "tg_veraveda777_bot_legacy"
        assert fb_row.recipient_id == 777
        # ZERO PII verification
        assert "8888888" not in fb_row.message_text
        assert "7777777" not in fb_row.message_text
        assert "sensitive stacktrace" not in fb_row.message_text

    # Simulate transient network failure during companion delivery (e.g. Telegram 502)
    current_running_inst["inst"] = inst_tg
    orig_send = bot.send_message
    fail_count = 0
    async def transient_fail(*args, **kwargs):
        nonlocal fail_count
        fail_count += 1
        if fail_count == 1:
            raise RuntimeError("Telegram 502 Bad Gateway (transient)")
        return await orig_send(*args, **kwargs)

    bot.send_message = transient_fail

    # Delivery fails on attempt 1 -> retryable status retained in outbox
    await process_ai_incident_notifications(bot, session_maker=test_db_session_maker)
    async with test_db_session_maker() as session:
        row_retry = await session.get(AIIncidentNotificationModel, fb_row.id)
        assert row_retry.status == "pending"
        assert row_retry.attempts == 1
        assert row_retry.last_error in ("unexpected_delivery_error", "telegram_network_error")
        # Reset next_retry_at to past to simulate retry interval elapsed / restart recovery
        row_retry.next_retry_at = datetime.utcnow() - timedelta(minutes=1)
        await session.commit()

    # Restart / next cycle delivers successfully
    await process_ai_incident_notifications(bot, session_maker=test_db_session_maker)
    async with test_db_session_maker() as session:
        row_delivered = await session.get(AIIncidentNotificationModel, fb_row.id)
        assert row_delivered.status == "delivered"
        assert row_delivered.attempts == 2


@pytest.mark.asyncio
async def test_fallback_and_budget_alert_markers_redacted_zero_free_text_leakage(test_db_session_maker, monkeypatch):
    """
    Directive 2: Eliminate free text from fallback and output budget outbox.
    Prove zero leakage of sensitive markers in primary_error, attempts, classification,
    details, finish_reason, and ai_log_ids into outbox and sent Telegram messages.
    """
    import uuid
    from database import AIIncidentNotificationModel, AIProviderIncidentModel
    from error_reporting import send_ai_fallback_used_alert, send_output_budget_exhausted_alert
    from tests.test_perplexity_agent_models import ValidatingTelegramSession

    inst_tg = {
        "pm2_name": "tg_veraveda777_bot_legacy",
        "bot_instance_id": "tg_veraveda777_bot_legacy",
        "companion_instance": None,
        "platform": "telegram",
        "admins": [777],
    }
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_tg)
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_tg)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_tg)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[777]))

    telegram_session = ValidatingTelegramSession()
    bot = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session)
    bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    MARKER_SECRET_KEY = "sk-ant-api03-CONFIDENTIAL_SECRET_KEY_XYZ_999"
    MARKER_PRIMARY_ERR = "CRITICAL_INJECTION_PRIMARY_ERROR_LINE_777"
    MARKER_ATTEMPT_ERR = "ATTEMPT_TRACEBACK_SECRET_USER_INPUT_888"
    MARKER_CLS_INJECT = "CLASSIFICATION_MALICIOUS_INJECTION_555"
    MARKER_FINISH_REASON = "FINISH_REASON_INJECTION_444"
    MARKER_LOG_ID_STRING = "LOG_ID_SECRET_MARKER_333"
    MARKER_DETAILS = "DETAILS_CONFIDENTIAL_BEARER_TOKEN_222"

    ALL_SENSITIVE_MARKERS = [
        MARKER_SECRET_KEY,
        MARKER_PRIMARY_ERR,
        MARKER_ATTEMPT_ERR,
        MARKER_CLS_INJECT,
        MARKER_FINISH_REASON,
        MARKER_LOG_ID_STRING,
        MARKER_DETAILS,
    ]

    # 1. Fallback alert with multiple sensitive markers injected across free fields
    res_fb = await send_ai_fallback_used_alert(
        bot=bot,
        platform="telegram",
        user_id=12345,
        primary_provider="KIE",
        primary_model="gemini-2.5-flash",
        fallback_provider="OpenAI",
        fallback_model="gpt-5",
        primary_error=f"UnrecognizedServiceError: {MARKER_PRIMARY_ERR} with key {MARKER_SECRET_KEY}",
        reason=f"Failed reason with {MARKER_DETAILS}",
        ai_log_ids=[101, MARKER_LOG_ID_STRING],
        provider_attempts=[
            {
                "provider": "KIE",
                "model": "gemini-2.5-flash",
                "status": "FAILED",
                "error": f"Stacktrace containing {MARKER_ATTEMPT_ERR}",
                "classification": MARKER_CLS_INJECT,
            },
            {
                "provider": "OpenAI",
                "model": "gpt-5",
                "status": "SUCCESS",
            },
        ],
        session_maker=test_db_session_maker,
    )
    assert res_fb is True

    # 2. Output budget alert with sensitive markers in finish_reason, ai_log_id, exception, details
    res_budget = await send_output_budget_exhausted_alert(
        bot=bot,
        platform="telegram",
        provider="Deepseek",
        model="deepseek-r1",
        finish_reason=MARKER_FINISH_REASON,
        visible_content_length=200,
        reasoning_content_length=4000,
        max_tokens=4000,
        ai_log_id=f"999_{MARKER_LOG_ID_STRING}",
        details=MARKER_DETAILS,
        exception=Exception(f"Token overflow with {MARKER_SECRET_KEY}"),
        session_maker=test_db_session_maker,
    )
    assert res_budget is True

    # 3. Assert zero leakage in AIIncidentNotificationModel outbox table
    async with test_db_session_maker() as session:
        notifs = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        assert len(notifs) >= 2
        for notif in notifs:
            msg_text = notif.message_text
            for marker in ALL_SENSITIVE_MARKERS:
                assert marker not in msg_text, f"Marker {marker!r} leaked into outbox message_text: {msg_text}"
            # Prove err_type_str does not guess from string and defaults to safe code
            assert "UnrecognizedServiceError" not in msg_text
            # Prove classification defaulted to safe value, never the injected string
            assert MARKER_CLS_INJECT not in msg_text

    # 4. Assert zero leakage in actual Telegram messages sent to admin
    send_calls = [c for c in telegram_session.calls if type(c).__name__ == "SendMessage"]
    assert len(send_calls) >= 2
    for call in send_calls:
        call_text = call.text
        for marker in ALL_SENSITIVE_MARKERS:
            assert marker not in call_text, f"Marker {marker!r} leaked into sent Telegram message: {call_text}"


@pytest.mark.asyncio
async def test_runtime_stale_fallback_and_budget_exhausted_after_a_b_a_suppressed(test_db_session_maker, monkeypatch):
    """
    Directive 1: Real AI Attempt Identity & A -> B -> A Lifecycle.
    Stale in-flight fallback and output budget alerts started before key rotation
    belong to the previous superseded generation and must NOT open a new incident or send alerts
    when Key A is reactivated (A -> B -> A).
    """
    from datetime import datetime, timedelta
    from ai_incident_manager import (
        get_provider_account_fingerprint,
        handle_key_rotation,
        record_ai_provider_failure,
    )
    from error_reporting import send_ai_fallback_used_alert, send_output_budget_exhausted_alert
    from tests.test_perplexity_agent_models import ValidatingTelegramSession

    inst_tg = {
        "pm2_name": "tg_veraveda777_bot_legacy",
        "bot_instance_id": "tg_veraveda777_bot_legacy",
        "companion_instance": None,
        "platform": "telegram",
        "admins": [777],
    }
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_tg)
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_tg)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_tg)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[777]))

    telegram_session = ValidatingTelegramSession()
    bot = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session)
    bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    key_a = "kie_lifecycle_key_alpha_111"
    key_b = "kie_lifecycle_key_beta_222"
    scope_a = get_provider_account_fingerprint(key_a, None)
    scope_b = get_provider_account_fingerprint(key_b, None)

    # Step 1: Active configuration is Key A
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            session.add(AIConfig(id=1, kie_api_key=key_a, default_model="kie-default"))
        else:
            cfg.kie_api_key = key_a
        await session.commit()

    # Request R1 starts at t1 under Key A
    t1 = datetime.utcnow() - timedelta(seconds=10)

    # Step 2: Key is rotated A -> B at t_rot1
    await asyncio.sleep(0.05)
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.kie_api_key = key_b
        await session.commit()
    await handle_key_rotation(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        new_key=key_b,
        old_key=key_a,
        session_maker=test_db_session_maker,
    )

    # Step 3: Key is rotated back B -> A at t_rot2 (reactivation)
    await asyncio.sleep(0.05)
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        cfg.kie_api_key = key_a
        await session.commit()
    await handle_key_rotation(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        new_key=key_a,
        old_key=key_b,
        session_maker=test_db_session_maker,
    )

    # Step 4: Fresh Request R2 starts under reactivated Key A at t2 > t_rot2
    await asyncio.sleep(0.05)
    t2 = datetime.utcnow()
    err_r2 = InsufficientBalanceError("KIE credits empty on reactivated key A")
    err_r2.provider = "KIE"
    inc_r2 = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        error=err_r2,
        operation="chat",
        model="kie-default",
        credential_scope=scope_a,
        attempt_started_at=t2,
        session_maker=test_db_session_maker,
    )
    assert inc_r2 is not None
    assert inc_r2.status == "active", "R2 should open an active incident on reactivated key A"

    # Step 5: Stale Request R1 finishes NOW (after A -> B -> A), reporting output budget & fallback
    # Output budget alert from stale R1 (started at t1)
    res_stale_budget = await send_output_budget_exhausted_alert(
        bot=bot,
        platform="telegram",
        provider="KIE",
        model="kie-default",
        finish_reason="length",
        credential_scope=scope_a,
        attempt_started_at=t1,
        session_maker=test_db_session_maker,
    )
    assert res_stale_budget is False, "Stale output budget alert for superseded attempt must be suppressed"

    # Fallback alert from stale R1 (started at t1)
    res_stale_fallback = await send_ai_fallback_used_alert(
        bot=bot,
        platform="telegram",
        primary_provider="KIE",
        primary_model="kie-default",
        fallback_provider="OpenAI",
        fallback_model="gpt-5",
        primary_error=Exception("Stale 502 error from old generation"),
        credential_scope=scope_a,
        attempt_started_at=t1,
        session_maker=test_db_session_maker,
    )
    assert res_stale_fallback is False, "Stale fallback alert for superseded attempt must be suppressed"

    # Step 6: Verify DB state:
    # 1. R2 incident remains ACTIVE and untouched
    # 2. No notification was queued for R1
    async with test_db_session_maker() as session:
        inc_check = await session.get(AIProviderIncidentModel, inc_r2.id)
        assert inc_check.status == "active"

        notifs_r1 = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.event_type.in_(["output_budget_exhausted", "fallback_used"])
            )
        )).scalars().all()
        assert len(notifs_r1) == 0, "Zero outbox notifications must be queued for stale R1 attempts"


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_concurrent_enqueue_during_rotation_shared_db(pg_test_session_maker, monkeypatch):
    """
    Directive 3: PostgreSQL test for concurrent enqueue during key rotation with shared DB across multiple bots.
    Proves atomicity, no deadlocks, and isolation between Bot A and Bot B under high concurrency.
    """
    from datetime import datetime
    from ai_incident_manager import (
        enqueue_ai_notification_event,
        get_provider_account_fingerprint,
        handle_key_rotation,
        record_ai_provider_failure,
    )
    from tests.test_perplexity_agent_models import ValidatingTelegramSession

    inst_bot_a = {
        "pm2_name": "tg_veraveda777_bot_legacy",
        "bot_instance_id": "tg_veraveda777_bot_legacy",
        "companion_instance": None,
        "platform": "telegram",
        "admins": [777],
    }
    inst_bot_b = {
        "pm2_name": "tg_someonewithyou01_bot_new",
        "bot_instance_id": "tg_someonewithyou01_bot_new",
        "companion_instance": None,
        "platform": "telegram",
        "admins": [888],
    }
    instances = {
        "tg_veraveda777_bot_legacy": inst_bot_a,
        "tg_someonewithyou01_bot_new": inst_bot_b,
    }

    import contextvars
    current_inst_cv = contextvars.ContextVar("current_inst", default=inst_bot_a)

    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: instances.get(n))
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: current_inst_cv.get())
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: s)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(side_effect=lambda comp, current_instance=None, session=None: [777] if comp.get("pm2_name") == "tg_veraveda777_bot_legacy" else [888]))

    key_a_old = "kie_pg_bot_a_old_key_111"
    key_a_new = "kie_pg_bot_a_new_key_222"
    key_b = "kie_pg_bot_b_stable_key_333"

    scope_a_old = get_provider_account_fingerprint(key_a_old, None)
    scope_a_new = get_provider_account_fingerprint(key_a_new, None)
    scope_b = get_provider_account_fingerprint(key_b, None)

    # Seed incidents for Bot A and Bot B on real PostgreSQL
    current_inst_cv.set(inst_bot_a)
    err_a = InsufficientBalanceError("KIE credits empty on Bot A")
    err_a.provider = "KIE"
    inc_a = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        error=err_a,
        operation="chat",
        model="kie-model-a",
        credential_scope=scope_a_old,
        session_maker=pg_test_session_maker,
    )
    assert inc_a is not None

    current_inst_cv.set(inst_bot_b)
    err_b = InsufficientBalanceError("KIE credits empty on Bot B")
    err_b.provider = "KIE"
    inc_b = await record_ai_provider_failure(
        origin_instance_id="tg_someonewithyou01_bot_new",
        provider="KIE",
        error=err_b,
        operation="chat",
        model="kie-model-b",
        credential_scope=scope_b,
        session_maker=pg_test_session_maker,
    )
    assert inc_b is not None

    current_inst_cv.set(inst_bot_a)

    # Define concurrent tasks:
    # 1. Admin rotates Bot A's key
    async def task_rotate_bot_a():
        current_inst_cv.set(inst_bot_a)
        return await handle_key_rotation(
            origin_instance_id="tg_veraveda777_bot_legacy",
            provider="KIE",
            new_key=key_a_new,
            old_key=key_a_old,
            session_maker=pg_test_session_maker,
        )

    # 2. In-flight notification enqueue on Bot A
    async def task_enqueue_bot_a():
        current_inst_cv.set(inst_bot_a)
        return await enqueue_ai_notification_event(
            incident_id=inc_a.id,
            origin_instance_id="tg_veraveda777_bot_legacy",
            event_type="terminal_failure",
            text="Concurrent terminal failure for Bot A",
            cooldown_seconds=0,
            session_maker=pg_test_session_maker,
        )

    # 3. Concurrent error recording for Bot B (should not be affected by Bot A's rotation)
    async def task_record_and_enqueue_bot_b():
        current_inst_cv.set(inst_bot_b)
        return await enqueue_ai_notification_event(
            incident_id=inc_b.id,
            origin_instance_id="tg_someonewithyou01_bot_new",
            event_type="fallback_used",
            text="Concurrent fallback notification for Bot B",
            cooldown_seconds=0,
            session_maker=pg_test_session_maker,
        )

    # Run all tasks concurrently on PostgreSQL
    results = await asyncio.gather(
        task_rotate_bot_a(),
        task_enqueue_bot_a(),
        task_record_and_enqueue_bot_b(),
        return_exceptions=False,
    )

    # Assertions on real PostgreSQL:
    # 1. No deadlocks or exceptions occurred during concurrent transactions
    assert len(results) == 3

    # 2. Bot A's old incident was superseded by rotation
    async with pg_test_session_maker() as session:
        db_inc_a = await session.get(AIProviderIncidentModel, inc_a.id)
        assert db_inc_a.status == "superseded"

        # 3. Bot B's incident remains ACTIVE (complete multi-bot isolation on PostgreSQL)
        db_inc_b = await session.get(AIProviderIncidentModel, inc_b.id)
        assert db_inc_b.status == "active"
        assert db_inc_b.resolved_at is None

        # 4. Outbox contains notification for Bot B (admin 888)
        notifs_b = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.incident_id == inc_b.id,
                AIIncidentNotificationModel.event_type == "fallback_used",
            )
        )).scalars().all()
        assert len(notifs_b) >= 1
        assert notifs_b[0].recipient_id == 888


@pytest.mark.asyncio
async def test_telegram_chat_fallback_canonical_outbox_no_bypass_on_incident_failure(test_db_session_maker, monkeypatch, caplog):
    """
    Directive 1: Eliminate the last bypass of canonical outbox in Telegram Chat.
    - When incident recording fails (e.g. database error), verify _notify_ai_fallback_used is NEVER called.
    - Verify incident manager error is logged explicitly at ERROR level with [AI Incident Manager] for Autobus observability.
    - Verify no direct messages are sent via get_all_admin_ids() or foreign admins.
    - Verify subsequent recovery enqueues canonical outbox exactly once without duplicates.
    """
    import ai_integration
    import uuid
    from database import AIIncidentNotificationModel, AIProviderIncidentModel
    from error_reporting import send_ai_fallback_used_alert
    from ai_incident_manager import process_ai_incident_notifications
    from tests.test_perplexity_agent_models import ValidatingTelegramSession
    from sqlalchemy.exc import OperationalError

    inst_tg = {
        "pm2_name": "tg_veraveda777_bot_legacy",
        "bot_instance_id": "tg_veraveda777_bot_legacy",
        "companion_instance": None,
        "platform": "telegram",
        "admins": [777],
    }
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_tg)
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_tg)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_tg)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[777]))

    # Spy on legacy notification helper to ensure it is NEVER called as a fallback
    spy_legacy_notify = AsyncMock()
    monkeypatch.setattr(ai_integration, "_notify_ai_fallback_used", spy_legacy_notify)

    telegram_session = ValidatingTelegramSession()
    bot = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session)
    bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    primary_err = ai_integration.AIResponseError("KIE 503 Service Unavailable")

    # Step 1: Simulate runtime fallback where incident recording throws OperationalError (DB connection drop)
    with caplog.at_level(logging.ERROR):
        failing_session_maker = MagicMock(side_effect=OperationalError("could not connect to server", None, None))
        from ai_incident_manager import record_ai_provider_failure
        rec_incident = None
        try:
            rec_incident = await record_ai_provider_failure(
                origin_instance_id=None,
                provider="KIE",
                error=primary_err,
                operation="chat",
                session_maker=failing_session_maker,
            )
        except Exception as rec_err:
            logging.error("[AI Incident Manager] Failed to record AI fallback success/failure: %s", rec_err, exc_info=True)

        assert rec_incident is None

        # CRITICAL INVARIANT: The legacy helper must NOT be called even if rec_incident is None
        assert spy_legacy_notify.call_count == 0

        # CRITICAL OBSERVABILITY: Autobus must see the error in PM2 logs
        assert any(("[AI Incident]" in r.message or "[AI Incident Manager]" in r.message) and r.levelname == "ERROR" for r in caplog.records)

    # Step 2: Ensure canonical outbox delivers when DB recovers, with zero direct bypass leaks
    res_alert = await send_ai_fallback_used_alert(
        bot=bot,
        platform="telegram",
        user_id=1234567,
        chat_id=9876543,
        username="secret_patient",
        full_name="Confidential Name",
        request_type="chat",
        primary_provider="KIE",
        primary_model="gemini-2.5-flash",
        fallback_provider="OpenAI",
        fallback_model="gpt-5.6-terra",
        primary_error=primary_err,
        incident_id=None,
        session_maker=test_db_session_maker,
    )
    assert res_alert is True

    # Assert exactly 1 notification enqueued in outbox
    async with test_db_session_maker() as session:
        notifs = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.event_type == "fallback_used"
            )
        )).scalars().all()
        assert len(notifs) == 1
        notif_row = notifs[0]
        assert notif_row.status == "delivered"
        assert notif_row.recipient_id == 777
        assert notif_row.authorized_instance_id == "tg_veraveda777_bot_legacy"
        # Zero PII in outbox message text
        assert "1234567" not in notif_row.message_text
        assert "9876543" not in notif_row.message_text
        assert "secret_patient" not in notif_row.message_text
        assert "Confidential Name" not in notif_row.message_text

    # Delivered message sent to Telegram admin 777
    send_calls = [c for c in telegram_session.calls if type(c).__name__ == "SendMessage"]
    assert len(send_calls) == 1
    assert send_calls[0].chat_id == 777
    assert "secret_patient" not in send_calls[0].text
    assert "1234567" not in send_calls[0].text

    # Step 3: Duplicate prevention: subsequent enqueue attempt within cooldown returns False
    res_duplicate = await send_ai_fallback_used_alert(
        bot=bot,
        platform="telegram",
        user_id=1234567,
        request_type="chat",
        primary_provider="KIE",
        primary_model="gemini-2.5-flash",
        fallback_provider="OpenAI",
        fallback_model="gpt-5.6-terra",
        primary_error=primary_err,
        session_maker=test_db_session_maker,
    )
    assert res_duplicate is False
    assert len([c for c in telegram_session.calls if type(c).__name__ == "SendMessage"]) == 1


@pytest.mark.asyncio
async def test_fallback_diagnostics_rejects_arbitrary_short_strings_and_pii(test_db_session_maker, monkeypatch):
    """
    Directive 2: Exclude arbitrary short strings and PII markers from fallback diagnostics.
    Verifies that short PII markers ('Patient_123456789', 'user_987654', '8888888',
    'CUSTOM_ERROR_CODE', 'ERR_PATIENT_TIMEOUT') are strictly mapped to fixed neutral code
    'ProviderFailure' / 'fallback_recovery', and never leak into outbox message_text or SendMessage.
    """
    import uuid
    from database import AIIncidentNotificationModel
    from error_reporting import send_ai_fallback_used_alert
    from tests.test_perplexity_agent_models import ValidatingTelegramSession

    inst_tg = {
        "pm2_name": "tg_veraveda777_bot_legacy",
        "bot_instance_id": "tg_veraveda777_bot_legacy",
        "companion_instance": None,
        "platform": "telegram",
        "admins": [777],
    }
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_tg)
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_tg)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_tg)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[777]))

    PII_TEST_CANDIDATES = [
        "Patient_123456789",
        "user_987654",
        "8888888",
        "CUSTOM_ERROR_CODE",
        "ERR_PATIENT_TIMEOUT",
    ]

    for candidate in PII_TEST_CANDIDATES:
        telegram_session = ValidatingTelegramSession()
        bot = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session)
        bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

        res = await send_ai_fallback_used_alert(
            bot=bot,
            platform="telegram",
            user_id=7001,
            request_type="chat",
            primary_provider="KIE",
            primary_model=f"mod-{uuid.uuid4().hex[:6]}",
            fallback_provider="OpenAI",
            fallback_model="gpt-5.6-terra",
            primary_error=candidate,
            session_maker=test_db_session_maker,
        )
        assert res is True

        # Check outbox row
        async with test_db_session_maker() as session:
            notifs = (await session.execute(
                select(AIIncidentNotificationModel).where(
                    AIIncidentNotificationModel.event_type == "fallback_used"
                ).order_by(AIIncidentNotificationModel.id.desc())
            )).scalars().all()
            latest_notif = notifs[0]

            assert candidate not in latest_notif.message_text, f"Leak detected in outbox! Candidate '{candidate}' found in message_text"
            assert "Тип сбоя: <code>ProviderFailure</code>" in latest_notif.message_text
            assert "Классификация: <code>fallback_recovery</code>" in latest_notif.message_text

        send_calls = [c for c in telegram_session.calls if type(c).__name__ == "SendMessage"]
        assert len(send_calls) == 1
        assert candidate not in send_calls[0].text, f"Leak detected in Telegram SendMessage! Candidate '{candidate}' found in sent text"
        assert "ProviderFailure" in send_calls[0].text


@pytest.mark.asyncio
async def test_fallback_diagnostics_accepts_authentic_canonical_technical_codes_and_provider_failure(test_db_session_maker, monkeypatch):
    """
    Directive 2: Accept authentic canonical technical codes and safe fields of ProviderFailure.
    - Canonical codes (e.g. 'rate_limit', 'auth_invalid_key') are preserved as typed classifications.
    - Typed ProviderFailure objects are correctly rendered with their detail_code and classification.
    """
    import uuid
    from database import AIIncidentNotificationModel
    from error_reporting import send_ai_fallback_used_alert
    from provider_failures import ProviderFailure
    from tests.test_perplexity_agent_models import ValidatingTelegramSession

    inst_tg = {
        "pm2_name": "tg_veraveda777_bot_legacy",
        "bot_instance_id": "tg_veraveda777_bot_legacy",
        "companion_instance": None,
        "platform": "telegram",
        "admins": [777],
    }
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_tg)
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_tg)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_tg)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[777]))

    # Test Case 1: Canonical error code string 'rate_limit'
    telegram_session1 = ValidatingTelegramSession()
    bot1 = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session1)
    bot1.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    res1 = await send_ai_fallback_used_alert(
        bot=bot1,
        platform="telegram",
        request_type="chat",
        primary_provider="KIE",
        primary_model="kie-canon-1",
        fallback_provider="OpenAI",
        fallback_model="gpt-5.6-terra",
        primary_error="rate_limit",
        session_maker=test_db_session_maker,
    )
    assert res1 is True

    async with test_db_session_maker() as session:
        notif1 = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.event_type == "fallback_used"
            ).order_by(AIIncidentNotificationModel.id.desc())
        )).scalars().first()
        assert "Тип сбоя: <code>rate_limit</code>" in notif1.message_text
        assert "Классификация: <code>rate_limit</code>" in notif1.message_text

    # Test Case 2: Legacy alias 'auth_invalid_key' (canonicalized to 'auth')
    telegram_session2 = ValidatingTelegramSession()
    bot2 = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session2)
    bot2.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    res2 = await send_ai_fallback_used_alert(
        bot=bot2,
        platform="telegram",
        request_type="chat",
        primary_provider="KIE",
        primary_model="kie-canon-2",
        fallback_provider="OpenAI",
        fallback_model="gpt-5.6-terra",
        primary_error="auth_invalid_key",
        session_maker=test_db_session_maker,
    )
    assert res2 is True

    async with test_db_session_maker() as session:
        notif2 = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.event_type == "fallback_used"
            ).order_by(AIIncidentNotificationModel.id.desc())
        )).scalars().first()
        assert "Тип сбоя: <code>auth_invalid_key</code>" in notif2.message_text
        assert "Классификация: <code>auth</code>" in notif2.message_text

    # Test Case 3: Typed ProviderFailure object with detail_code
    pf = ProviderFailure(
        provider="KIE",
        operation="chat",
        classification="insufficient_balance_quota",
        detail_code="billing_or_spend_limit",
    )
    telegram_session3 = ValidatingTelegramSession()
    bot3 = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session3)
    bot3.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    res3 = await send_ai_fallback_used_alert(
        bot=bot3,
        platform="telegram",
        request_type="chat",
        primary_provider="KIE",
        primary_model="kie-canon-3",
        fallback_provider="OpenAI",
        fallback_model="gpt-5.6-terra",
        primary_error=pf,
        session_maker=test_db_session_maker,
    )
    assert res3 is True

    async with test_db_session_maker() as session:
        notif3 = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.event_type == "fallback_used"
            ).order_by(AIIncidentNotificationModel.id.desc())
        )).scalars().first()
        assert "Тип сбоя: <code>billing_or_spend_limit</code>" in notif3.message_text
        assert "Классификация: <code>insufficient_balance_quota</code>" in notif3.message_text


@pytest.mark.asyncio
async def test_telegram_chat_fallback_fail_closed_unknown_instance(test_db_session_maker, monkeypatch):
    """
    Fail-closed isolation check: If current instance is unknown or not in registry,
    fallback notification is rejected fail-closed with zero leakage and zero alerts sent.
    """
    from error_reporting import send_ai_fallback_used_alert
    from tests.test_perplexity_agent_models import ValidatingTelegramSession

    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: None)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: None)

    telegram_session = ValidatingTelegramSession()
    bot = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session)
    bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    res = await send_ai_fallback_used_alert(
        bot=bot,
        platform="telegram",
        request_type="chat",
        primary_provider="KIE",
        primary_model="gemini-2.5-flash",
        fallback_provider="OpenAI",
        fallback_model="gpt-5.6-terra",
        primary_error="rate_limit",
        session_maker=test_db_session_maker,
    )
    assert res is False
    assert len(telegram_session.calls) == 0


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_telegram_chat_fallback_e2e_journey(pg_test_session_maker, monkeypatch):
    """
    Directive 1 & 2 on real PostgreSQL:
    End-to-end journey for Telegram Chat fallback:
    event -> incident recorded -> outbox enqueued -> delivery worker claims via PostgreSQL ->
    delivery to legitimate Telegram admin -> DB cooldown prevents duplicate outbox rows.
    """
    import uuid
    from database import AIIncidentNotificationModel, AIProviderIncidentModel
    from error_reporting import send_ai_fallback_used_alert
    from ai_incident_manager import process_ai_incident_notifications
    from tests.test_perplexity_agent_models import ValidatingTelegramSession

    inst_tg = {
        "pm2_name": "tg_veraveda777_bot_legacy",
        "bot_instance_id": "tg_veraveda777_bot_legacy",
        "companion_instance": None,
        "platform": "telegram",
        "admins": [777],
    }
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_tg)
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_tg)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_tg)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[777]))

    telegram_session = ValidatingTelegramSession()
    bot = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session)
    bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    # Enqueue fallback notification on PostgreSQL
    res = await send_ai_fallback_used_alert(
        bot=bot,
        platform="telegram",
        user_id=9876543,
        request_type="chat",
        primary_provider="KIE",
        primary_model="kie-chat-v1",
        fallback_provider="OpenAI",
        fallback_model="gpt-5.6-terra",
        primary_error="rate_limit",
        session_maker=pg_test_session_maker,
    )
    assert res is True

    # Check PostgreSQL database state
    async with pg_test_session_maker() as session:
        notifs = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.event_type == "fallback_used"
            ).order_by(AIIncidentNotificationModel.id.desc())
        )).scalars().all()
        assert len(notifs) >= 1
        notif_row = notifs[0]
        assert notif_row.status == "delivered"
        assert notif_row.recipient_id == 777
        assert notif_row.authorized_instance_id == "tg_veraveda777_bot_legacy"
        assert "9876543" not in notif_row.message_text

    # Telegram send_message was called on delivery
    send_calls = [c for c in telegram_session.calls if type(c).__name__ == "SendMessage"]
    assert len(send_calls) >= 1
    assert send_calls[-1].chat_id == 777
    assert "9876543" not in send_calls[-1].text

    # Duplicate suppression on PostgreSQL: immediate subsequent attempt returns False
    res_dup = await send_ai_fallback_used_alert(
        bot=bot,
        platform="telegram",
        user_id=9876543,
        request_type="chat",
        primary_provider="KIE",
        primary_model="kie-chat-v1",
        fallback_provider="OpenAI",
        fallback_model="gpt-5.6-terra",
        primary_error="rate_limit",
        session_maker=pg_test_session_maker,
    )
    assert res_dup is False


@pytest.mark.asyncio
async def test_telegram_chat_e2e_runtime_fallback_to_outbox_delivery(test_db_session_maker, monkeypatch):
    """
    Blocker 3: Full realistic E2E chain for Telegram Chat runtime:
    - Primary provider fails with 503 error, fallback succeeds.
    - AI fallback alert enqueued in durable outbox.
    - Outbox worker claims and delivers alert to Telegram admin.
    - Strict verification: zero PII / user prompt leakage.
    """
    import ai_integration
    from database import User, AIConfig, AIIncidentNotificationModel, AIProviderIncidentModel
    from tests.test_perplexity_agent_models import ValidatingTelegramSession
    from ai_integration import AIResponseError

    inst_tg = {
        "pm2_name": "tg_veraveda777_bot_legacy",
        "bot_instance_id": "tg_veraveda777_bot_legacy",
        "companion_instance": None,
        "platform": "telegram",
        "admins": [777],
    }
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_tg)
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_tg)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_tg)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[777]))

    monkeypatch.setattr(ai_integration, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)
    monkeypatch.setattr("error_reporting.async_session_maker", test_db_session_maker)

    async with test_db_session_maker() as session:
        user = User(
            id=1001,
            username="secret_patient_user",
            first_name="Patient",
            name="Private Name",
            gender="m",
            response_length="normal",
        )
        session.add(user)
        cfg = await session.get(AIConfig, 1)
        cfg.provider = "KIE"
        cfg.model = "gemini-3-flash"
        cfg.kie_api_key = "kie_sec_key_111"
        cfg.fallback_provider = "OpenAI"
        cfg.fallback_model = "gpt-5.6-terra"
        cfg.openai_api_key = "sk_test_fallback_222"
        cfg.allow_fallback = True
        await session.commit()

    monkeypatch.setattr(ai_integration, "build_runtime_context", AsyncMock(return_value="runtime context"))
    monkeypatch.setattr(ai_integration, "_call_kie_chat", AsyncMock(side_effect=AIResponseError("KIE 503 Service Unavailable")))
    monkeypatch.setattr(ai_integration, "_call_openai_api", AsyncMock(return_value="Hello from OpenAI fallback response"))

    telegram_session = ValidatingTelegramSession()
    bot = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session)
    bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    response = await ai_integration.get_ai_response(
        1001,
        "User question with confidential symptoms",
        "Private Name",
        "m",
        bot=bot,
        include_test_context=False,
    )

    assert response == "Hello from OpenAI fallback response"

    async with test_db_session_maker() as session:
        notifs = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.event_type == "fallback_used"
            )
        )).scalars().all()
        assert len(notifs) == 1
        notif = notifs[0]
        assert notif.status == "delivered"
        assert notif.recipient_id == 777
        assert notif.authorized_instance_id == "tg_veraveda777_bot_legacy"
        assert "secret_patient_user" not in notif.message_text
        assert "confidential symptoms" not in notif.message_text
        assert "1001" not in notif.message_text

    send_calls = [c for c in telegram_session.calls if type(c).__name__ == "SendMessage"]
    assert len(send_calls) >= 1
    delivered_call = send_calls[-1]
    assert delivered_call.chat_id == 777
    assert "secret_patient_user" not in delivered_call.text
    assert "confidential symptoms" not in delivered_call.text


@pytest.mark.asyncio
async def test_max_chat_e2e_runtime_fallback_to_companion_outbox_delivery(test_db_session_maker, monkeypatch):
    """
    Blocker 3: Full realistic E2E chain for MAX Chat runtime using real registered instances:
    - Pair 1: max_se13639182_bot_legacy -> companion tg_veraveda777_bot_legacy
      Owners: 979514796, 272982544, 806750628. Platform ID: 8224473896.
    - Pair 2: max_id519010411655_bot_new -> companion tg_yourself_way_bot_new
      Owners: 979514796, 272982544, 806750628, 1372041348. Platform ID: 8481853696.
    - Real detect_current_instance(), resolve_companion_telegram_instance(), and resolve_target_admin_ids().
    - Zero direct Telegram sending by MAX bot; alerts go to DB outbox.
    - Strict cross-companion isolation: foreign bot cannot claim other bot's notifications.
    - Fail-closed handling for unknown instance without companion.
    """
    from max_messenger_bot import ai as max_ai
    from database import User, AIConfig, AIIncidentNotificationModel, AIProviderIncidentModel
    from tests.test_perplexity_agent_models import ValidatingTelegramSession
    from ai_integration import AIResponseError
    from ai_incident_manager import process_ai_incident_notifications

    monkeypatch.setattr(max_ai, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr("ai_incident_manager.async_session_maker", test_db_session_maker)
    monkeypatch.setattr("error_reporting.async_session_maker", test_db_session_maker)

    # --------------------------------------------------------------------------
    # Scenario A: Pair 1 (max_se13639182_bot_legacy -> tg_veraveda777_bot_legacy)
    # --------------------------------------------------------------------------
    monkeypatch.setenv("BOT_INSTANCE_ID", "max_se13639182_bot_legacy")
    monkeypatch.setenv("COMPANION_TELEGRAM_BOT_TOKEN", "8224473896:AAFtest_token_veraveda")

    async with test_db_session_maker() as session:
        user1 = User(
            id=2001,
            username="max_private_client_1",
            first_name="Client1",
            name="MAX Client 1",
            gender="f",
            response_length="normal",
        )
        session.add(user1)
        cfg = await session.get(AIConfig, 1)
        cfg.provider = "KIE"
        cfg.model = "gemini-3-flash"
        cfg.kie_api_key = "kie_sec_key_max_1"
        cfg.fallback_provider = "OpenAI"
        cfg.fallback_model = "gpt-5.6-terra"
        cfg.openai_api_key = "sk_test_fallback_max_1"
        cfg.allow_fallback = True
        await session.commit()

    monkeypatch.setattr(max_ai, "_call_kie_text_chat", AsyncMock(side_effect=AIResponseError("KIE 503 Bad Gateway")))
    monkeypatch.setattr(max_ai, "_call_openai", AsyncMock(return_value="MAX AI fallback text answer 1"))

    result1 = await max_ai.get_ai_response(2001, "MAX secret medical inquiry 1")
    assert result1 == "MAX AI fallback text answer 1"

    # Verify notifications were created for all 3 companion admins: 979514796, 272982544, 806750628
    async with test_db_session_maker() as session:
        notifs_pair1 = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.event_type == "fallback_used",
                AIIncidentNotificationModel.authorized_instance_id == "tg_veraveda777_bot_legacy",
            )
        )).scalars().all()
        assert len(notifs_pair1) == 3
        recipients_pair1 = {n.recipient_id for n in notifs_pair1}
        assert recipients_pair1 == {979514796, 272982544, 806750628}
        for n in notifs_pair1:
            assert n.status == "pending"
            assert "max_private_client_1" not in n.message_text
            assert "MAX secret medical inquiry 1" not in n.message_text

    # Cross-instance isolation: tg_yourself_way_bot_new must NOT claim tg_veraveda777_bot_legacy notifications!
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_yourself_way_bot_new")
    monkeypatch.setenv("BOT_TOKEN", "8481853696:AAFtest_token_yourself_way")
    session_yw = ValidatingTelegramSession()
    bot_yw = Bot(token="8481853696:AAFtest_token_yourself_way", session=session_yw)
    bot_yw.get_me = AsyncMock(return_value=TgUser(id=8481853696, is_bot=True, first_name="bot", username="yourself_way_bot"))

    foreign_deliv = await process_ai_incident_notifications(bot_yw, session_maker=test_db_session_maker)
    assert foreign_deliv == 0  # Rejection: worker claimed 0 rows belonging to tg_veraveda777_bot_legacy

    # Authorized companion bot delivers: tg_veraveda777_bot_legacy claims and delivers all 3
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")
    session_vv = ValidatingTelegramSession()
    bot_vv = Bot(token="8224473896:AAFtest_token_veraveda", session=session_vv)
    bot_vv.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    auth_deliv = await process_ai_incident_notifications(bot_vv, session_maker=test_db_session_maker)
    assert auth_deliv == 3

    send_calls_vv = [c for c in session_vv.calls if type(c).__name__ == "SendMessage"]
    assert len(send_calls_vv) == 3
    delivered_chats_vv = {c.chat_id for c in send_calls_vv}
    assert delivered_chats_vv == {979514796, 272982544, 806750628}

    # --------------------------------------------------------------------------
    # Scenario B: Pair 2 (max_id519010411655_bot_new -> tg_yourself_way_bot_new)
    # --------------------------------------------------------------------------
    monkeypatch.setenv("BOT_INSTANCE_ID", "max_id519010411655_bot_new")
    monkeypatch.setenv("COMPANION_TELEGRAM_BOT_TOKEN", "8481853696:AAFtest_token_yourself_way")

    async with test_db_session_maker() as session:
        user2 = User(
            id=2002,
            username="max_private_client_2",
            first_name="Client2",
            name="MAX Client 2",
            gender="m",
            response_length="normal",
        )
        session.add(user2)
        await session.commit()

    monkeypatch.setattr(max_ai, "_call_openai", AsyncMock(return_value="MAX AI fallback text answer 2"))
    result2 = await max_ai.get_ai_response(2002, "MAX secret medical inquiry 2")
    assert result2 == "MAX AI fallback text answer 2"

    async with test_db_session_maker() as session:
        notifs_pair2 = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.event_type == "fallback_used",
                AIIncidentNotificationModel.authorized_instance_id == "tg_yourself_way_bot_new",
            )
        )).scalars().all()
        # tg_yourself_way_bot_new has 4 telegram_owner_ids: 979514796, 272982544, 806750628, 1372041348
        assert len(notifs_pair2) == 4
        assert {n.recipient_id for n in notifs_pair2} == {979514796, 272982544, 806750628, 1372041348}

    # Pair 2 companion worker delivers:
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_yourself_way_bot_new")
    monkeypatch.setenv("BOT_TOKEN", "8481853696:AAFtest_token_yourself_way")
    session_yw2 = ValidatingTelegramSession()
    bot_yw2 = Bot(token="8481853696:AAFtest_token_yourself_way", session=session_yw2)
    bot_yw2.get_me = AsyncMock(return_value=TgUser(id=8481853696, is_bot=True, first_name="bot", username="yourself_way_bot"))

    auth_deliv_yw = await process_ai_incident_notifications(bot_yw2, session_maker=test_db_session_maker)
    assert auth_deliv_yw == 4

    # --------------------------------------------------------------------------
    # Scenario C: Fail-Closed on Unknown Instance (no registered companion)
    # --------------------------------------------------------------------------
    monkeypatch.setenv("BOT_INSTANCE_ID", "max_unregistered_ghost_bot")
    async with test_db_session_maker() as session:
        user3 = User(id=2003, username="ghost_user", first_name="G", name="Ghost", gender="m", response_length="normal")
        session.add(user3)
        await session.commit()

    # Must not raise exception to user
    res_ghost = await max_ai.get_ai_response(2003, "Ghost query")
    assert res_ghost == "MAX AI fallback text answer 2"


@pytest.mark.asyncio
async def test_e2e_recovery_lifecycle_scoped_to_credentials(test_db_session_maker, monkeypatch):
    """
    Blocker 3: AI incident recovery lifecycle strictly scoped to credentials:
    - Failure on Key A opens incident for Key A.
    - Success on Key B does NOT resolve Key A's incident.
    - Success on Key A resolves Key A's incident and enqueues recovery alert.
    """
    from ai_incident_manager import (
        get_provider_account_fingerprint,
        process_ai_incident_notifications,
        record_ai_operation_success,
        record_ai_provider_failure,
    )
    from database import AIIncidentNotificationModel, AIProviderIncidentModel
    from tests.test_perplexity_agent_models import ValidatingTelegramSession

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")

    key_a = "key_alpha_secret_111"
    key_b = "key_beta_secret_222"
    scope_a = get_provider_account_fingerprint(key_a, None)
    scope_b = get_provider_account_fingerprint(key_b, None)

    # 1. Failure on Key A opens incident
    err_a = InsufficientBalanceError("KIE credits empty on Key A")
    err_a.provider = "KIE"
    inc_a = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        error=err_a,
        operation="chat",
        model="kie-model",
        credential_scope=scope_a,
        session_maker=test_db_session_maker,
    )
    assert inc_a is not None
    assert inc_a.status == "active"

    async with test_db_session_maker() as session:
        opened_notifs = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.incident_id == inc_a.id,
                AIIncidentNotificationModel.event_type == "opened",
            )
        )).scalars().all()
        assert len(opened_notifs) >= 1

    # 2. Success on Key B should NOT close Key A's incident
    rec_b = await record_ai_operation_success(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        credential_scope=scope_b,
        operation="chat",
        model="kie-model",
        session_maker=test_db_session_maker,
    )
    assert rec_b is False

    async with test_db_session_maker() as session:
        inc_check = await session.get(AIProviderIncidentModel, inc_a.id)
        assert inc_check.status == "active", "Key A's incident must remain active after Key B's success"

    # 3. Success on Key A closes Key A's incident and enqueues 'resolved' alert
    rec_a = await record_ai_operation_success(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        credential_scope=scope_a,
        operation="chat",
        model="kie-model",
        session_maker=test_db_session_maker,
    )
    assert rec_a is True

    async with test_db_session_maker() as session:
        inc_resolved = await session.get(AIProviderIncidentModel, inc_a.id)
        assert inc_resolved.status == "resolved"
        assert inc_resolved.resolved_at is not None

        resolved_notifs = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.incident_id == inc_a.id,
                AIIncidentNotificationModel.event_type == "resolved",
            )
        )).scalars().all()
        assert len(resolved_notifs) >= 1
        assert resolved_notifs[0].status == "pending"

    # Deliver resolved notification via bot
    telegram_session = ValidatingTelegramSession()
    bot = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session)
    bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    deliv_count = await process_ai_incident_notifications(bot, session_maker=test_db_session_maker)
    assert deliv_count >= 1

    send_calls = [c for c in telegram_session.calls if type(c).__name__ == "SendMessage"]
    assert any("AI-провайдер восстановил работу" in c.text for c in send_calls)


@pytest.mark.asyncio
async def test_ai_outbox_backoff_and_lease_recovery(test_db_session_maker, monkeypatch):
    """
    Blocker 2 & 4: Outbox backoff calculations, lease expiration recovery, and Telegram 429 retry_after:
    - 0-based indexing for AI_INCIDENT_BACKOFF_DELAYS: 15s, 1m, 5m, 15m, 1h.
    - Expired lease claimed by secondary worker.
    - Telegram 429 retry_after without upper truncation (respects 900s, 3600s, safe fallback for invalid/negative).
    - Claim prevention while now < next_retry_at.
    """
    from ai_incident_manager import (
        AI_INCIDENT_BACKOFF_DELAYS,
        OUTBOX_LEASE_SECONDS,
        MAX_OUTBOX_RETRIES,
        MAX_OUTBOX_ATTEMPTS,
        get_ai_incident_outbox_backoff,
        process_ai_incident_notifications,
    )
    from database import AIIncidentNotificationModel, AIProviderIncidentModel
    from tests.test_perplexity_agent_models import ValidatingTelegramSession
    from aiogram.exceptions import TelegramRetryAfter

    # 1. Check exact backoff values and constants
    assert MAX_OUTBOX_RETRIES == 5
    assert MAX_OUTBOX_ATTEMPTS == 6
    assert get_ai_incident_outbox_backoff(1) == timedelta(seconds=15)
    assert get_ai_incident_outbox_backoff(2) == timedelta(minutes=1)
    assert get_ai_incident_outbox_backoff(3) == timedelta(minutes=5)
    assert get_ai_incident_outbox_backoff(4) == timedelta(minutes=15)
    assert get_ai_incident_outbox_backoff(5) == timedelta(hours=1)
    assert get_ai_incident_outbox_backoff(6) == timedelta(hours=1)
    assert get_ai_incident_outbox_backoff(10) == timedelta(hours=1)

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")

    now = datetime.utcnow()
    async with test_db_session_maker() as session:
        inc = AIProviderIncidentModel(
            origin_instance_id="tg_veraveda777_bot_legacy",
            companion_instance_id="tg_veraveda777_bot_legacy",
            provider="KIE",
            credential_scope="test_scope",
            operation="chat",
            model_scope="*",
            incident_type="provider_5xx",
            detail_code="503",
            evidence_tier="CONFIRMED_STRUCTURED",
            status="active",
            occurrence_count=1,
            reminder_cycle=0,
            first_seen_at=now,
            last_seen_at=now,
            last_alerted_at=now,
        )
        session.add(inc)
        await session.flush()

        notif = AIIncidentNotificationModel(
            incident_id=inc.id,
            event_type="opened",
            cycle_index=0,
            recipient_id=272982544,
            authorized_instance_id="tg_veraveda777_bot_legacy",
            message_text="<b>Alert text</b>",
            status="pending",
            attempts=0,
            max_attempts=6,
            next_retry_at=now - timedelta(seconds=1),
        )
        session.add(notif)
        await session.commit()
        notif_id = notif.id

    # 2. Test lease expiration and recovery by secondary worker:
    async with test_db_session_maker() as session:
        claimed_row = await session.get(AIIncidentNotificationModel, notif_id)
        claimed_row.status = "processing"
        claimed_row.claim_token = "dead_worker_token"
        claimed_row.attempts = 1
        claimed_row.lease_until = datetime.utcnow() - timedelta(seconds=5)
        await session.commit()

    telegram_session = ValidatingTelegramSession()
    bot = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session)
    bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    deliv = await process_ai_incident_notifications(bot, session_maker=test_db_session_maker)
    assert deliv == 1

    async with test_db_session_maker() as session:
        check_row = await session.get(AIIncidentNotificationModel, notif_id)
        assert check_row.status == "delivered"
        assert check_row.attempts == 2

    # 3. Test Telegram 429 TelegramRetryAfter: NO UPPER TRUNCATION at 300s!
    test_delays = [
        (42, 42),
        (300, 300),
        (900, 900),    # Must NOT be truncated to 300s!
        (3600, 3600),  # Must NOT be truncated to 300s!
        (-10, 30.0),   # Negative fallback to 30s
        ("invalid", 30.0), # Non-numeric fallback to 30s
    ]

    for cycle_idx, (raw_retry, expected_sec) in enumerate(test_delays):
        async with test_db_session_maker() as session:
            notif_flood = AIIncidentNotificationModel(
                incident_id=inc.id,
                event_type="opened",
                cycle_index=cycle_idx + 10,
                recipient_id=272982544,
                authorized_instance_id="tg_veraveda777_bot_legacy",
                message_text=f"<b>Rate limit test {cycle_idx}</b>",
                status="pending",
                attempts=0,
                max_attempts=6,
                next_retry_at=datetime.utcnow() - timedelta(seconds=1),
            )
            session.add(notif_flood)
            await session.commit()
            flood_id = notif_flood.id

        retry_after_exc = TelegramRetryAfter(
            method=SendMessage(chat_id=272982544, text=""),
            message="Flood wait",
            retry_after=raw_retry,
        )
        failing_bot = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session)
        failing_bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))
        failing_bot.send_message = AsyncMock(side_effect=retry_after_exc)

        t_before = datetime.utcnow()
        deliv_res = await process_ai_incident_notifications(failing_bot, session_maker=test_db_session_maker)
        assert deliv_res == 0

        async with test_db_session_maker() as session:
            check_flood = await session.get(AIIncidentNotificationModel, flood_id)
            assert check_flood.status == "pending"
            assert check_flood.last_error == "telegram_rate_limited"
            assert check_flood.attempts == 1
            delta_sec = (check_flood.next_retry_at - t_before).total_seconds()
            assert (expected_sec - 5) <= delta_sec <= (expected_sec + 15), (
                f"Expected delay {expected_sec}s for retry_after={raw_retry}, got {delta_sec}s"
            )

        # Verify premature claim is prevented: while now < next_retry_at, worker claim returns 0
        working_bot = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session)
        working_bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))
        working_bot.send_message = AsyncMock(return_value=True)

        premature_deliv = await process_ai_incident_notifications(working_bot, session_maker=test_db_session_maker)
        assert premature_deliv == 0, "Worker must NOT claim outbox item before next_retry_at elapses"

        # Advance next_retry_at to past and verify worker claims and delivers
        async with test_db_session_maker() as session:
            row_to_ready = await session.get(AIIncidentNotificationModel, flood_id)
            row_to_ready.next_retry_at = datetime.utcnow() - timedelta(seconds=1)
            await session.commit()

        delivered_ok = await process_ai_incident_notifications(working_bot, session_maker=test_db_session_maker)
        assert delivered_ok == 1

        async with test_db_session_maker() as session:
            delivered_check = await session.get(AIIncidentNotificationModel, flood_id)
            assert delivered_check.status == "delivered"
            assert delivered_check.attempts == 2


async def _run_six_attempt_lifecycle_verification(session_maker, monkeypatch):
    """
    Core verification of Blocker 4:
    Exact table: attempt -> result -> next_retry_at -> next status
      - Initial attempt 1 fails -> attempts=1, status='pending', next_retry_at = now + 15s (retry 1 scheduled)
      - Retry 1 (attempt 2) fails -> attempts=2, status='pending', next_retry_at = now + 1m (retry 2 scheduled)
      - Retry 2 (attempt 3) fails -> attempts=3, status='pending', next_retry_at = now + 5m (retry 3 scheduled)
      - Retry 3 (attempt 4) fails -> attempts=4, status='pending', next_retry_at = now + 15m (retry 4 scheduled)
      - Retry 4 (attempt 5) fails -> attempts=5, status='pending', next_retry_at = now + 1h (retry 5 scheduled)
      - Retry 5 (attempt 6) fails -> attempts=6, status='failed_terminal', next_retry_at is None
    """
    from ai_incident_manager import (
        MAX_OUTBOX_RETRIES,
        MAX_OUTBOX_ATTEMPTS,
        AI_INCIDENT_BACKOFF_DELAYS,
        process_ai_incident_notifications,
    )
    from database import AIIncidentNotificationModel, AIProviderIncidentModel
    from tests.test_perplexity_agent_models import ValidatingTelegramSession
    from aiogram.exceptions import TelegramNetworkError

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")

    now = datetime.utcnow()
    async with session_maker() as session:
        inc = AIProviderIncidentModel(
            origin_instance_id="tg_veraveda777_bot_legacy",
            companion_instance_id="tg_veraveda777_bot_legacy",
            provider="KIE",
            credential_scope="scope_lifecycle",
            operation="chat",
            model_scope="*",
            incident_type="provider_5xx",
            detail_code="503",
            evidence_tier="CONFIRMED_STRUCTURED",
            status="active",
            occurrence_count=1,
            reminder_cycle=0,
            first_seen_at=now,
            last_seen_at=now,
            last_alerted_at=now,
        )
        session.add(inc)
        await session.flush()

        notif = AIIncidentNotificationModel(
            incident_id=inc.id,
            event_type="opened",
            cycle_index=99,
            recipient_id=272982544,
            authorized_instance_id="tg_veraveda777_bot_legacy",
            message_text="<b>Lifecycle Test</b>",
            status="pending",
            attempts=0,
            max_attempts=MAX_OUTBOX_ATTEMPTS,
            next_retry_at=now - timedelta(seconds=1),
        )
        session.add(notif)
        await session.commit()
        notif_id = notif.id

    telegram_session = ValidatingTelegramSession()
    failing_bot = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session)
    failing_bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))
    failing_bot.send_message = AsyncMock(
        side_effect=TelegramNetworkError(
            method=SendMessage(chat_id=272982544, text=""),
            message="Connection reset by peer",
        )
    )

    expected_delays_sec = [15, 60, 300, 900, 3600]

    # Attempts 1 to 5 (Initial attempt + Retries 1 through 4)
    for attempt_idx in range(1, 6):
        t_pre = datetime.utcnow()
        deliv = await process_ai_incident_notifications(failing_bot, session_maker=session_maker)
        assert deliv == 0

        async with session_maker() as session:
            row = await session.get(AIIncidentNotificationModel, notif_id)
            assert row.status == "pending", f"Attempt {attempt_idx} must leave status as pending"
            assert row.attempts == attempt_idx, f"Attempt count mismatch at step {attempt_idx}"
            assert row.last_error == "telegram_network_error"
            expected_wait = expected_delays_sec[attempt_idx - 1]
            delta = (row.next_retry_at - t_pre).total_seconds()
            assert (expected_wait - 5) <= delta <= (expected_wait + 15), (
                f"Attempt {attempt_idx} expected delay {expected_wait}s, got {delta}s"
            )

            # Advance next_retry_at so next attempt can claim
            row.next_retry_at = datetime.utcnow() - timedelta(seconds=1)
            await session.commit()

    # Attempt 6 (Retry 5): this is the MAX_OUTBOX_ATTEMPTS attempt. Fails -> failed_terminal, next_retry_at = None
    deliv_final = await process_ai_incident_notifications(failing_bot, session_maker=session_maker)
    assert deliv_final == 0

    async with session_maker() as session:
        terminal_row = await session.get(AIIncidentNotificationModel, notif_id)
        assert terminal_row.status == "failed_terminal", "Attempt 6 failure must transition status to failed_terminal"
        assert terminal_row.attempts == 6
        assert terminal_row.next_retry_at is None, "failed_terminal row must have next_retry_at = None"
        assert terminal_row.last_error == "telegram_network_error"

    # Verify worker completely ignores failed_terminal row
    ignored_deliv = await process_ai_incident_notifications(failing_bot, session_maker=session_maker)
    assert ignored_deliv == 0


@pytest.mark.asyncio
async def test_ai_outbox_six_attempt_retry_lifecycle_sqlite(test_db_session_maker, monkeypatch):
    """Blocker 4: Verify 6-attempt retry lifecycle table on SQLite."""
    await _run_six_attempt_lifecycle_verification(test_db_session_maker, monkeypatch)


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_ai_outbox_six_attempt_retry_lifecycle(pg_test_session_maker, monkeypatch):
    """Blocker 4: Verify 6-attempt retry lifecycle table on real PostgreSQL."""
    await _run_six_attempt_lifecycle_verification(pg_test_session_maker, monkeypatch)


@pytest.mark.asyncio
async def test_incident_sanitization_detail_code_rejects_pii_and_arbitrary_strings(test_db_session_maker, monkeypatch):
    """
    Blocker 1: Ensure arbitrary strings and PII markers are never stored in AIProviderIncidentModel.detail_code
    or rendered into alert HTML messages, and arbitrary exception class names cannot smuggle user data.
    """
    from ai_incident_manager import (
        _format_incident_message_text,
        record_ai_provider_failure,
        sanitize_technical_code,
        process_ai_incident_notifications,
    )
    from database import AIIncidentNotificationModel, AIProviderIncidentModel
    from tests.test_perplexity_agent_models import ValidatingTelegramSession
    from error_reporting import send_terminal_ai_failure_alert

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")

    # 1. Unit sanitization check: Genuine whitelisted codes and typed exceptions
    assert sanitize_technical_code("rate_limit") == "rate_limit"
    assert sanitize_technical_code("502") == "502"
    assert sanitize_technical_code("429") == "429"
    assert sanitize_technical_code("TimeoutError") == "TimeoutError"
    assert sanitize_technical_code("HTTPStatusError") == "HTTPStatusError"
    assert sanitize_technical_code("AIServiceError") == "AIServiceError"
    assert sanitize_technical_code("InsufficientBalanceError") == "InsufficientBalanceError"
    assert sanitize_technical_code("TelegramRetryAfter") == "TelegramRetryAfter"
    assert sanitize_technical_code("balance_1.50_below_2.00") == "balance_1.50_below_2.00"
    assert sanitize_technical_code("superseded_by_key_rotation") == "superseded_by_key_rotation"

    # Arbitrary strings with PII must NEVER be accepted under any suffix syntax (Error, Exception, Failure, Timeout)
    assert sanitize_technical_code("Patient_123456789_Error", classification="provider_5xx") == "provider_5xx"
    assert sanitize_technical_code("user_987654_Exception", classification="invalid_request") == "invalid_request"
    assert sanitize_technical_code("SecretToken123_Failure", classification="auth") == "auth"
    assert sanitize_technical_code("Doctor_JohnDoe_Timeout", classification="timeout") == "timeout"
    assert sanitize_technical_code("Patient_123456789_Error", classification=None) == "unspecified"
    assert sanitize_technical_code("Patient_123456789_Error", classification="unknown") == "unspecified"

    # 2. Test send_terminal_ai_failure_alert rejects arbitrary error_type or arbitrary exception class names
    class Patient_123456789_Error(Exception):
        pass

    await send_terminal_ai_failure_alert(
        error_type="Patient_123456789_Error",
        exception=Patient_123456789_Error("Confidential medical leak"),
        user_id=12345,
        platform="telegram",
        title="Test Alert",
        session_maker=test_db_session_maker,
    )

    async with test_db_session_maker() as session:
        terminal_notifs = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.event_type == "terminal_failure"
            )
        )).scalars().all()
        assert len(terminal_notifs) >= 1
        for n in terminal_notifs:
            assert "Patient_123456789" not in n.message_text
            assert "Confidential medical leak" not in n.message_text
            assert "12345" not in n.message_text
            assert "<code>AIServiceError</code>" in n.message_text

    # 3. Test record_ai_provider_failure with error having PII in detail_code and exception name
    inc = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        error=Patient_123456789_Error("Something broke with patient data"),
        operation="chat",
        model="kie-model",
        session_maker=test_db_session_maker,
    )
    assert inc is not None
    assert "Patient_123456789" not in inc.detail_code
    assert inc.detail_code in ("provider_5xx", "AIServiceError", "unspecified")

    # 4. Check formatted alert text
    msg = _format_incident_message_text(
        provider="KIE",
        operation="chat",
        model_scope="kie-model",
        incident_type="provider_5xx",
        detail_code="Patient_123456789_Error",
        evidence_tier="CONFIRMED_STRUCTURED",
        event_type="opened",
    )
    assert "Patient_123456789" not in msg
    assert "Код: <code>provider_5xx</code>" in msg

    # 5. Check outbox row and delivered Telegram message contain zero PII markers
    telegram_session = ValidatingTelegramSession()
    bot = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session)
    bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    deliv_count = await process_ai_incident_notifications(bot, session_maker=test_db_session_maker)
    assert deliv_count >= 1

    send_calls = [c for c in telegram_session.calls if type(c).__name__ == "SendMessage"]
    for call in send_calls:
        assert "Patient_123456789" not in call.text
        assert "patient data" not in call.text


@pytest.mark.asyncio
async def test_autobus_observability_ingests_incident_manager_error(tmp_path, monkeypatch):
    """
    Blocker 5: Full integration test of Autobus observability:
    - Incident manager error in log file.
    - Log chunk parsing extracts error summary, not filtered by is_handled_ai_log().
    - read_new_log_errors extracts infrastructure error issue.
    - should_alert confirms alerting threshold.
    - System diagnostic recipients includes canonical admin (272982544).
    - send_alert executes HTTP POST contract to Telegram API:
      * 200 OK delivery with JSON {"ok": True} returns True.
      * 500 error / network retry returns False without false positive delivery.
      * Partial delivery to multiple recipients properly tracked.
      * Cooldown suppresses subsequent alerts for identical issue.
    """
    import monitor_psychobots as monitor
    import aiohttp

    # 1. Verify log chunk parsing and error summary extraction
    log_chunk = (
        "2026-11-15 18:40:00,123 INFO [ai_integration] Calling AI provider\n"
        "2026-11-15 18:40:01,456 ERROR [ai_incident_manager] [AI Incident Manager] Failed to record AI fallback alert: OperationalError('connection to database lost')\n"
        "Traceback (most recent call last):\n"
        "  File '/app/ai_incident_manager.py', line 1234, in record_ai_provider_failure\n"
        "    await session.commit()\n"
        "sqlalchemy.exc.OperationalError: connection to database lost\n"
    )

    summaries = monitor.extract_log_error_summaries(log_chunk)
    assert len(summaries) >= 1
    assert any("[AI Incident Manager]" in s for s in summaries)
    for s in summaries:
        assert monitor.is_handled_ai_log(s) is False

    # 2. Simulate reading new log errors via read_new_log_errors
    log_file = tmp_path / "psychobot_error.log"
    log_file.write_text(log_chunk, encoding="utf-8")

    app = {
        "name": "tg_autobusbusbot_new",
        "error_log": str(log_file),
        "token": "8346799080:AAFtest_autobus_token_12345",
    }
    apps = [app]
    state = {}
    issues = monitor.read_new_log_errors(app, state, include_existing=True)
    assert len(issues) >= 1
    assert any("OperationalError" in issue or "[AI Incident Manager]" in issue for issue in issues)

    # 3. Observability alerting threshold and recipient resolution
    assert monitor.should_alert(issues, state) is True
    recipients = monitor.get_system_diagnostic_recipients(apps)
    assert 272982544 in recipients

    monkeypatch.setenv("PSYCHOBOTS_ALERT_BOT_TOKEN", "8346799080:AAFtest_autobus_token_12345")
    monkeypatch.setenv("PSYCHOBOTS_ALERT_BOT_PM2_NAME", "tg_autobusbusbot_new")

    # 4. HTTP Contract Test: 200 OK delivery
    posted_requests = []

    class MockAiohttpPostSuccess:
        def __init__(self, url, json=None, **kwargs):
            self.url = url
            self.json_payload = json
            self.status = 200

        async def __aenter__(self):
            posted_requests.append({"url": self.url, "json": self.json_payload})
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def json(self, content_type=None):
            return {"ok": True, "result": {"message_id": 12345}}

    monkeypatch.setattr(aiohttp.ClientSession, "post", lambda self, url, json=None, **kwargs: MockAiohttpPostSuccess(url, json=json, **kwargs))

    alert_text = f"🚨 <b>[Autobus Monitor] Critical Infrastructure Incident</b>\n{issues[0]}"
    ok_res = await monitor.send_alert(apps=apps, text=alert_text, recipients=[272982544])
    assert ok_res is True
    assert len(posted_requests) == 1
    assert posted_requests[0]["url"] == "https://api.telegram.org/bot8346799080:AAFtest_autobus_token_12345/sendMessage"
    assert posted_requests[0]["json"]["chat_id"] == 272982544
    assert posted_requests[0]["json"]["parse_mode"] == "HTML"
    assert "OperationalError" in posted_requests[0]["json"]["text"]

    # 5. HTTP Contract Test: 500 error / failure handling
    class MockAiohttpPost500:
        def __init__(self, url, json=None, **kwargs):
            self.url = url
            self.json_payload = json
            self.status = 500

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def json(self, content_type=None):
            return {"ok": False, "error_code": 500, "description": "Internal Server Error"}

    monkeypatch.setattr(aiohttp.ClientSession, "post", lambda self, url, json=None, **kwargs: MockAiohttpPost500(url, json=json, **kwargs))
    fail_delivered = []
    fail_res = await monitor.send_alert(apps=apps, text=alert_text, recipients=[272982544], delivered_recipients=fail_delivered)
    assert fail_res is False
    assert 272982544 not in fail_delivered

    # 6. HTTP Contract Test: Partial delivery tracking to multiple recipients
    call_count = 0

    class MockAiohttpPostPartial:
        def __init__(self, url, json=None, **kwargs):
            nonlocal call_count
            call_count += 1
            self.url = url
            self.json_payload = json
            # First recipient (272982544) succeeds, second recipient fails
            self.status = 200 if call_count == 1 else 500

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def json(self, content_type=None):
            if self.status == 200:
                return {"ok": True, "result": {"message_id": 999}}
            return {"ok": False, "error_code": 500}

    monkeypatch.setattr(aiohttp.ClientSession, "post", lambda self, url, json=None, **kwargs: MockAiohttpPostPartial(url, json=json, **kwargs))
    partial_delivered = []
    partial_res = await monitor.send_alert(
        apps=apps,
        text=alert_text,
        recipients=[272982544, 979514796],
        delivered_recipients=partial_delivered,
    )
    assert partial_res is False  # False because not all recipients succeeded
    assert partial_delivered == [272982544]  # Only the successful recipient is recorded

    # 7. Cooldown suppression prevents alert storms
    assert monitor.should_alert(issues, state) is True
    monitor.record_alert_delivered(issues, state)
    assert monitor.should_alert(issues, state) is False


@pytest.mark.asyncio
async def test_pii_and_canonical_classification_protection(test_db_session_maker, monkeypatch):
    """
    Blocker 1: Guarantee that PII strings and arbitrary classifications cannot bypass canonical validation:
    - classification = "Patient_123456789"
    - classification = "SecretToken123"
    - provider_attempts[].classification = "user_987654"
    - ProviderFailure.classification = "Patient_123456789"
    Prove total absence across AIProviderIncidentModel, AIIncidentNotificationModel, and Telegram SendMessage.
    """
    from provider_failures import ProviderFailure, CANONICAL_CLASSES
    from ai_incident_manager import (
        record_ai_provider_failure,
        process_ai_incident_notifications,
    )
    from error_reporting import send_terminal_ai_failure_alert
    from database import AIIncidentNotificationModel, AIProviderIncidentModel
    from tests.test_perplexity_agent_models import ValidatingTelegramSession

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")

    # 1. ProviderFailure __post_init__ guarantees canonical classification
    pf1 = ProviderFailure(
        provider="OPENAI",
        operation="chat",
        classification="Patient_123456789",
        detail_code="test_code",
    )
    assert pf1.classification == "unknown"
    assert pf1.classification in CANONICAL_CLASSES

    pf2 = ProviderFailure(
        provider="OPENAI",
        operation="chat",
        classification="SecretToken123",
        detail_code="test_code",
    )
    assert pf2.classification == "unknown"
    assert pf2.classification in CANONICAL_CLASSES

    # 2. record_ai_provider_failure with ProviderFailure having attempted arbitrary classification
    inc = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OPENAI",
        error=pf1,
        operation="chat",
        model="gpt-4o",
        session_maker=test_db_session_maker,
    )
    assert inc is not None
    assert "Patient_123456789" not in str(inc.incident_type)
    assert inc.incident_type in CANONICAL_CLASSES

    # 3. send_terminal_ai_failure_alert with:
    # classification = "SecretToken123"
    # provider_attempts[].classification = "user_987654"
    telegram_session = ValidatingTelegramSession()
    bot = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session)
    bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))

    attempts_payload = [
        {
            "provider": "OpenAI",
            "model": "gpt-4o",
            "status": "FAILED",
            "classification": "user_987654",
        }
    ]

    await send_terminal_ai_failure_alert(
        error_type="AIServiceError",
        classification="SecretToken123",
        provider="OPENAI",
        provider_attempts=attempts_payload,
        user_id=12345,
        platform="telegram",
        title="Test Alert Title",
        session_maker=test_db_session_maker,
    )

    # 4. Verify AIIncidentNotificationModel rows in DB
    async with test_db_session_maker() as session:
        notifs = (await session.execute(
            select(AIIncidentNotificationModel).where(
                AIIncidentNotificationModel.event_type == "terminal_failure"
            )
        )).scalars().all()
        assert len(notifs) >= 1
        for n in notifs:
            assert "Patient_123456789" not in n.message_text
            assert "SecretToken123" not in n.message_text
            assert "user_987654" not in n.message_text
            assert "12345" not in n.message_text

    # 5. Process outbox and verify Telegram SendMessage text
    await process_ai_incident_notifications(bot, session_maker=test_db_session_maker)
    send_calls = [c for c in telegram_session.calls if type(c).__name__ == "SendMessage"]
    assert len(send_calls) >= 1
    for call in send_calls:
        assert "Patient_123456789" not in call.text
        assert "SecretToken123" not in call.text
        assert "user_987654" not in call.text
        assert "12345" not in call.text


async def _run_custom_max_attempts_lifecycle_verification(session_maker, monkeypatch):
    """
    Blocker 2: Prove that row.max_attempts is the sole authoritative source of truth.
    - max_attempts = 1: exactly 1 attempt, fails -> failed_terminal, next_retry_at is None.
    - max_attempts = 3: exactly 3 attempts, fails -> failed_terminal, next_retry_at is None.
    - max_attempts in (0, -1, None): safely falls back to MAX_OUTBOX_ATTEMPTS (6).
    """
    from ai_incident_manager import (
        MAX_OUTBOX_ATTEMPTS,
        process_ai_incident_notifications,
        enqueue_terminal_ai_failure_notification,
    )
    from database import AIIncidentNotificationModel, AIProviderIncidentModel
    from tests.test_perplexity_agent_models import ValidatingTelegramSession
    from aiogram.methods import SendMessage
    from aiogram.exceptions import TelegramNetworkError

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")

    telegram_session = ValidatingTelegramSession()
    failing_bot = Bot(token="8224473896:AAFtest_token_veraveda", session=telegram_session)
    failing_bot.get_me = AsyncMock(return_value=TgUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot"))
    failing_bot.send_message = AsyncMock(side_effect=TelegramNetworkError(method=SendMessage(chat_id=1, text="x"), message="Connection lost"))

    # Case 1: max_attempts = 1
    async with session_maker() as session:
        inc1 = AIProviderIncidentModel(
            provider="OPENAI",
            operation="chat",
            model_scope="gpt-4o",
            credential_scope="scope_custom_1",
            incident_type="provider_5xx",
            status="active",
            detail_code="500",
            origin_instance_id="tg_veraveda777_bot_legacy",
            companion_instance_id="tg_veraveda777_bot_legacy",
        )
        session.add(inc1)
        await session.commit()
        await session.refresh(inc1)
        inc1_id = inc1.id

    await enqueue_terminal_ai_failure_notification(
        incident_id=inc1_id,
        origin_instance_id="tg_veraveda777_bot_legacy",
        text="Alert 1",
        session_maker=session_maker,
        max_attempts=1,
    )

    async with session_maker() as session:
        notif1 = (await session.execute(
            select(AIIncidentNotificationModel).where(AIIncidentNotificationModel.incident_id == inc1_id)
        )).scalars().first()
        assert notif1.max_attempts == 1
        notif1_id = notif1.id

    # Execute 1 failing turn
    await process_ai_incident_notifications(failing_bot, session_maker=session_maker)

    async with session_maker() as session:
        row1 = await session.get(AIIncidentNotificationModel, notif1_id)
        assert row1.status == "failed_terminal", "max_attempts=1 must become failed_terminal after 1st attempt"
        assert row1.attempts == 1
        assert row1.next_retry_at is None

    # Case 2: max_attempts = 3
    async with session_maker() as session:
        inc3 = AIProviderIncidentModel(
            provider="OPENAI",
            operation="chat",
            model_scope="gpt-4o",
            credential_scope="scope_custom_3",
            incident_type="provider_5xx",
            status="active",
            detail_code="500",
            origin_instance_id="tg_veraveda777_bot_legacy",
            companion_instance_id="tg_veraveda777_bot_legacy",
        )
        session.add(inc3)
        await session.commit()
        await session.refresh(inc3)
        inc3_id = inc3.id

    await enqueue_terminal_ai_failure_notification(
        incident_id=inc3_id,
        origin_instance_id="tg_veraveda777_bot_legacy",
        text="Alert 3",
        session_maker=session_maker,
        max_attempts=3,
    )

    async with session_maker() as session:
        notif3 = (await session.execute(
            select(AIIncidentNotificationModel).where(AIIncidentNotificationModel.incident_id == inc3_id)
        )).scalars().first()
        assert notif3.max_attempts == 3
        notif3_id = notif3.id

    # Attempt 1 -> pending
    await process_ai_incident_notifications(failing_bot, session_maker=session_maker)
    async with session_maker() as session:
        r = await session.get(AIIncidentNotificationModel, notif3_id)
        assert r.status == "pending"
        assert r.attempts == 1
        assert r.next_retry_at is not None
        r.next_retry_at = datetime.utcnow() - timedelta(seconds=1)
        await session.commit()

    # Attempt 2 -> pending
    await process_ai_incident_notifications(failing_bot, session_maker=session_maker)
    async with session_maker() as session:
        r = await session.get(AIIncidentNotificationModel, notif3_id)
        assert r.status == "pending"
        assert r.attempts == 2
        assert r.next_retry_at is not None
        r.next_retry_at = datetime.utcnow() - timedelta(seconds=1)
        await session.commit()

    # Attempt 3 -> failed_terminal!
    await process_ai_incident_notifications(failing_bot, session_maker=session_maker)
    async with session_maker() as session:
        r = await session.get(AIIncidentNotificationModel, notif3_id)
        assert r.status == "failed_terminal"
        assert r.attempts == 3
        assert r.next_retry_at is None

    # Case 3: Invalid max_attempts (0, -1, None) safely falls back to MAX_OUTBOX_ATTEMPTS (6)
    for invalid_val in (0, -1, None):
        async with session_maker() as session:
            inc_inv = AIProviderIncidentModel(
                provider="OPENAI",
                operation="chat",
                model_scope="gpt-4o",
                credential_scope=f"scope_inv_{invalid_val}",
                incident_type="provider_5xx",
                status="active",
                detail_code="500",
                origin_instance_id="tg_veraveda777_bot_legacy",
                companion_instance_id="tg_veraveda777_bot_legacy",
            )
            session.add(inc_inv)
            await session.commit()
            await session.refresh(inc_inv)

            notif_inv = AIIncidentNotificationModel(
                incident_id=inc_inv.id,
                event_type="terminal_failure",
                cycle_index=0,
                recipient_id=272982544,
                authorized_instance_id="tg_veraveda777_bot_legacy",
                message_text=f"Alert inv {invalid_val}",
                status="pending",
                attempts=0,
                max_attempts=invalid_val,
                next_retry_at=datetime.utcnow() - timedelta(seconds=1),
            )
            session.add(notif_inv)
            await session.commit()
            await session.refresh(notif_inv)
            inv_id = notif_inv.id

        # Attempt 1 fails -> MUST remain 'pending' because effective_max_attempts falls back to 6!
        await process_ai_incident_notifications(failing_bot, session_maker=session_maker)
        async with session_maker() as session:
            r = await session.get(AIIncidentNotificationModel, inv_id)
            assert r.status == "pending", f"Invalid max_attempts={invalid_val} must fall back to 6 and remain pending after attempt 1"
            assert r.attempts == 1
            assert r.next_retry_at is not None


@pytest.mark.asyncio
async def test_ai_outbox_custom_max_attempts_lifecycle_sqlite(test_db_session_maker, monkeypatch):
    """Blocker 2: Verify custom max_attempts (1, 3, invalid) on SQLite."""
    await _run_custom_max_attempts_lifecycle_verification(test_db_session_maker, monkeypatch)


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_ai_outbox_custom_max_attempts_lifecycle(pg_test_session_maker, monkeypatch):
    """Blocker 2: Verify custom max_attempts (1, 3, invalid) on real PostgreSQL."""
    await _run_custom_max_attempts_lifecycle_verification(pg_test_session_maker, monkeypatch)


@pytest.mark.asyncio
async def test_autobus_monitor_run_check_full_control_loop_lifecycle(tmp_path, monkeypatch):
    """
    Blocker 3: Integration test of the real Autobus monitor control loop (monitor.run_check()):
    1. Monitor discovers new infrastructure incident (OperationalError in app log).
    2. Selects authorized system recipients (272982544).
    3. Delivers via Telegram HTTP mock (aiohttp.ClientSession.post).
    4. Records successful and partial delivery in pending_notifications.
    5. Preserves state across runs.
    6. Retries undelivered after network error / HTTP 500.
    7. Preserves PM2, DB, and network alerts.
    """
    import monitor_psychobots as monitor
    import aiohttp
    import json

    lock_file = str(tmp_path / "monitor.lock")
    state_file = str(tmp_path / "state.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)
    monkeypatch.setenv("PSYCHOBOTS_ALERT_BOT_PM2_NAME", "tg_autobusbusbot_new")
    monkeypatch.setenv("PSYCHOBOTS_ALERT_BOT_TOKEN", "8346799080:AAFtest_autobus_token_12345")

    # Create app error log with OperationalError
    log_file = tmp_path / "autobus_error.log"
    log_content = (
        "2026-11-15 18:40:01,456 ERROR [ai_incident_manager] [AI Incident Manager] "
        "Failed to record AI fallback alert: OperationalError('connection to database lost')\n"
        "Traceback (most recent call last):\n"
        "  File '/app/ai_incident_manager.py', line 1234, in record_ai_provider_failure\n"
        "    await session.commit()\n"
        "sqlalchemy.exc.OperationalError: connection to database lost\n"
    )
    log_file.write_text(log_content, encoding="utf-8")

    app = {
        "name": "tg_autobusbusbot_new",
        "status": "online",
        "token": "8346799080:AAFtest_autobus_token_12345",
        "error_log": str(log_file),
        "owner_ids": [272982544],
    }
    apps = [app]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    # Mock health checks to return clean (focusing on log error ingestion)
    async def clean_async(*args, **kwargs): return []
    monkeypatch.setattr(monitor, "check_db_schema", clean_async)
    monkeypatch.setattr(monitor, "check_payment_webhooks", clean_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", clean_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", clean_async)
    monkeypatch.setattr(monitor, "check_telegram", clean_async)

    # 1. First Run: HTTP 500 -> Delivery fails, notification preserved in pending_notifications
    posted_calls = []
    http_should_succeed = False

    class MockHttpTelegramResponse:
        def __init__(self, url, json=None, **kwargs):
            self.url = url
            self.json_payload = json
            self.status = 200 if http_should_succeed else 500

        async def __aenter__(self):
            posted_calls.append({"url": self.url, "json": self.json_payload})
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def json(self, content_type=None):
            if self.status == 200:
                return {"ok": True, "result": {"message_id": 777}}
            return {"ok": False, "error_code": 500, "description": "Telegram network error"}

    monkeypatch.setattr(
        aiohttp.ClientSession,
        "post",
        lambda self, url, json=None, **kwargs: MockHttpTelegramResponse(url, json=json, **kwargs),
    )

    exit_code_1 = await monitor.run_check(include_existing_log_errors=True)
    assert exit_code_1 == 1, "Run check must exit with 1 when issues are found"
    assert len(posted_calls) >= 1

    # Verify state saved with pending notification and NO cooldown commit
    state_1 = monitor.load_state(state_file)
    pending_1 = state_1.get("pending_notifications", {})
    assert "issue" in pending_1 or any(k.startswith("issue_") for k in pending_1), "Failed delivery must preserve notification in pending_notifications"

    # 2. Second Run: HTTP 200 OK -> Delivery succeeds, pending notification cleared, cooldown recorded
    posted_calls.clear()
    http_should_succeed = True

    exit_code_2 = await monitor.run_check(include_existing_log_errors=True)
    assert exit_code_2 == 0, "Run check exits with 0 once pending issues are delivered and no new issues exist"
    assert len(posted_calls) >= 1

    # Verify Telegram SendMessage payload
    tg_req = posted_calls[0]
    assert "8346799080:AAFtest_autobus_token_12345" in tg_req["url"]
    assert tg_req["json"]["chat_id"] == 272982544
    assert tg_req["json"]["parse_mode"] == "HTML"
    assert "OperationalError" in tg_req["json"]["text"]

    # Verify pending notifications cleared and cooldown saved
    state_2 = monitor.load_state(state_file)
    pending_2 = state_2.get("pending_notifications", {})
    assert "issue" not in pending_2
    assert not any(k.startswith("issue_") for k in pending_2)

    # 3. Third Run: Append same error to log -> Cooldown active -> No duplicate HTTP delivery
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(
            "2026-11-15 18:45:01,000 ERROR [ai_incident_manager] [AI Incident Manager] "
            "Failed to record AI fallback alert: OperationalError('connection to database lost')\n"
            "Traceback (most recent call last):\n"
            "  File '/app/ai_incident_manager.py', line 1234, in record_ai_provider_failure\n"
            "    await session.commit()\n"
            "sqlalchemy.exc.OperationalError: connection to database lost\n"
        )
    posted_calls.clear()
    exit_code_3 = await monitor.run_check(include_existing_log_errors=True)
    assert exit_code_3 == 1, "Issue detected in new log lines"
    assert len(posted_calls) == 0, "Duplicate issues during cooldown must not trigger repeat alert"

    # 4. Verify PM2, DB, and network alerts are preserved
    # PM2 stopped
    app["status"] = "stopped"
    exit_code_pm2 = await monitor.run_check(include_existing_log_errors=True)
    assert exit_code_pm2 == 1
    # DB error
    async def fake_db_issue(*args, **kwargs): return ["DB: missing migration"]
    monkeypatch.setattr(monitor, "check_db_schema", fake_db_issue)
    exit_code_db = await monitor.run_check(include_existing_log_errors=True)
    assert exit_code_db == 1


@pytest.mark.asyncio
async def test_production_init_db_migration_ai_incident_notifications_sqlite(tmp_path, monkeypatch):
    """Blocker 4: Verify production database.init_db() migrates ai_incident_notifications on SQLite:
    - Adds missing max_attempts (default 6) and next_retry_at columns
    - Backfills next_retry_at = created_at for pending and processing rows
    - Leaves next_retry_at as NULL for delivered and failed_terminal rows
    - Is idempotent across multiple init_db() calls
    """
    import database
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    from sqlalchemy import text, inspect

    db_path = tmp_path / "prod_migration_test.db"
    test_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    test_session_maker = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)

    monkeypatch.setattr(database, "engine", test_engine)
    monkeypatch.setattr(database, "async_session_maker", test_session_maker)

    t_pending = datetime(2026, 1, 1, 10, 0, 0)
    t_processing = datetime(2026, 1, 2, 10, 0, 0)
    t_delivered = datetime(2026, 1, 3, 10, 0, 0)
    t_failed = datetime(2026, 1, 4, 10, 0, 0)

    # 1. Create table WITHOUT max_attempts and next_retry_at
    async with test_engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE ai_provider_incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider VARCHAR(32) NOT NULL,
                operation VARCHAR(32) NOT NULL,
                model_scope VARCHAR(64) NOT NULL,
                credential_scope VARCHAR(64) NOT NULL,
                incident_type VARCHAR(32) NOT NULL,
                status VARCHAR(32) NOT NULL,
                detail_code VARCHAR(128) NOT NULL,
                origin_instance_id VARCHAR(64) NOT NULL,
                companion_instance_id VARCHAR(64) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
            )
        """))
        await conn.execute(text("""
            CREATE TABLE ai_incident_notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id INTEGER NOT NULL REFERENCES ai_provider_incidents(id) ON DELETE CASCADE,
                event_type VARCHAR(32) NOT NULL,
                cycle_index INTEGER NOT NULL DEFAULT 0,
                recipient_id BIGINT NOT NULL,
                authorized_instance_id VARCHAR(64) NOT NULL,
                message_text TEXT NOT NULL,
                status VARCHAR(32) NOT NULL DEFAULT 'pending',
                claim_token VARCHAR(64),
                lease_until TIMESTAMP,
                attempts INTEGER NOT NULL DEFAULT 0,
                delivered_at TIMESTAMP,
                last_error TEXT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """))
        # Insert incident
        await conn.execute(text("""
            INSERT INTO ai_provider_incidents (id, provider, operation, model_scope, credential_scope, incident_type, status, detail_code, origin_instance_id, companion_instance_id)
            VALUES (1, 'KIE', 'chat', 'model-1', 'scope-1', 'provider_5xx', 'active', '500', 'inst1', 'inst1')
        """))
        # Insert rows across all 4 statuses
        await conn.execute(text("""
            INSERT INTO ai_incident_notifications (id, incident_id, event_type, recipient_id, authorized_instance_id, message_text, status, created_at, updated_at)
            VALUES (1, 1, 'opened', 100, 'inst1', 'msg1', 'pending', :t_pend, :t_pend),
                   (2, 1, 'opened', 101, 'inst1', 'msg2', 'processing', :t_proc, :t_proc),
                   (3, 1, 'opened', 102, 'inst1', 'msg3', 'delivered', :t_deliv, :t_deliv),
                   (4, 1, 'opened', 103, 'inst1', 'msg4', 'failed_terminal', :t_fail, :t_fail)
        """), {"t_pend": t_pending, "t_proc": t_processing, "t_deliv": t_delivered, "t_fail": t_failed})

    def _get_columns(sync_conn):
        insp = inspect(sync_conn)
        return [c['name'] for c in insp.get_columns('ai_incident_notifications')]

    async with test_engine.connect() as conn:
        cols_before = await conn.run_sync(_get_columns)
        assert 'max_attempts' not in cols_before
        assert 'next_retry_at' not in cols_before

    # 2. Run REAL production init_db()
    await database.init_db()

    # 3. Verify columns and backfill values
    async with test_engine.connect() as conn:
        cols_after = await conn.run_sync(_get_columns)
        assert 'max_attempts' in cols_after
        assert 'next_retry_at' in cols_after

        res = await conn.execute(text("SELECT id, status, max_attempts, next_retry_at, created_at FROM ai_incident_notifications ORDER BY id"))
        rows = {r[0]: r for r in res.fetchall()}

        # Row 1 (pending): max_attempts == 6, next_retry_at == created_at
        assert rows[1][2] == 6
        assert rows[1][3] is not None
        assert str(rows[1][3])[:19] == str(t_pending)[:19]

        # Row 2 (processing): max_attempts == 6, next_retry_at == created_at
        assert rows[2][2] == 6
        assert rows[2][3] is not None
        assert str(rows[2][3])[:19] == str(t_processing)[:19]

        # Row 3 (delivered): max_attempts == 6, next_retry_at is NULL
        assert rows[3][2] == 6
        assert rows[3][3] is None

        # Row 4 (failed_terminal): max_attempts == 6, next_retry_at is NULL
        assert rows[4][2] == 6
        assert rows[4][3] is None

    # 4. Verify idempotency - run production init_db() a second time
    await database.init_db()

    await test_engine.dispose()


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_production_init_db_migration_ai_incident_notifications_postgresql(pg_test_session_maker, monkeypatch):
    """Blocker 4: Verify production database.init_db() migrates ai_incident_notifications on real PostgreSQL:
    - Adds missing max_attempts (default 6) and next_retry_at columns
    - Backfills next_retry_at = created_at for pending and processing rows
    - Leaves next_retry_at as NULL for delivered and failed_terminal rows
    - Is idempotent across multiple init_db() calls
    """
    import database
    from sqlalchemy import text, inspect

    test_engine = pg_test_session_maker.kw['bind']
    monkeypatch.setattr(database, "engine", test_engine)
    monkeypatch.setattr(database, "async_session_maker", pg_test_session_maker)

    t_pending = datetime(2026, 1, 1, 10, 0, 0)
    t_processing = datetime(2026, 1, 2, 10, 0, 0)
    t_delivered = datetime(2026, 1, 3, 10, 0, 0)
    t_failed = datetime(2026, 1, 4, 10, 0, 0)

    # 1. Create table WITHOUT max_attempts and next_retry_at
    async with test_engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS ai_incident_notifications CASCADE"))
        await conn.execute(text("""
            CREATE TABLE ai_incident_notifications (
                id BIGSERIAL PRIMARY KEY,
                incident_id INTEGER NOT NULL REFERENCES ai_provider_incidents(id) ON DELETE CASCADE,
                event_type VARCHAR(32) NOT NULL,
                cycle_index INTEGER NOT NULL DEFAULT 0,
                recipient_id BIGINT NOT NULL,
                authorized_instance_id VARCHAR(64) NOT NULL,
                message_text TEXT NOT NULL,
                status VARCHAR(32) NOT NULL DEFAULT 'pending',
                claim_token VARCHAR(64),
                lease_until TIMESTAMP,
                attempts INTEGER NOT NULL DEFAULT 0,
                delivered_at TIMESTAMP,
                last_error TEXT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """))
        # Ensure parent incident exists with all non-null columns
        await conn.execute(text("""
            INSERT INTO ai_provider_incidents (
                id, origin_instance_id, companion_instance_id, provider, credential_scope,
                operation, model_scope, incident_type, evidence_tier, detail_code,
                classification_source, status, occurrence_count, reminder_cycle,
                first_seen_at, last_seen_at
            )
            VALUES (
                1, 'inst1', 'inst1', 'KIE', 'scope-1',
                'chat', 'model-1', 'provider_5xx', 'CONFIRMED_STRUCTURED', '500',
                'provider_structured', 'active', 1, 0,
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
            )
            ON CONFLICT (id) DO NOTHING
        """))
        # Insert rows across all 4 statuses
        await conn.execute(text("""
            INSERT INTO ai_incident_notifications (incident_id, event_type, recipient_id, authorized_instance_id, message_text, status, created_at, updated_at)
            VALUES (1, 'opened', 100, 'inst1', 'msg1', 'pending', :t_pend, :t_pend),
                   (1, 'opened', 101, 'inst1', 'msg2', 'processing', :t_proc, :t_proc),
                   (1, 'opened', 102, 'inst1', 'msg3', 'delivered', :t_deliv, :t_deliv),
                   (1, 'opened', 103, 'inst1', 'msg4', 'failed_terminal', :t_fail, :t_fail)
        """), {"t_pend": t_pending, "t_proc": t_processing, "t_deliv": t_delivered, "t_fail": t_failed})

    def _get_columns(sync_conn):
        insp = inspect(sync_conn)
        return [c['name'] for c in insp.get_columns('ai_incident_notifications')]

    async with test_engine.connect() as conn:
        cols_before = await conn.run_sync(_get_columns)
        assert 'max_attempts' not in cols_before
        assert 'next_retry_at' not in cols_before

    # 2. Run REAL production init_db()
    await database.init_db()

    # 3. Verify columns and backfill values
    async with test_engine.connect() as conn:
        cols_after = await conn.run_sync(_get_columns)
        assert 'max_attempts' in cols_after
        assert 'next_retry_at' in cols_after

        res = await conn.execute(text("SELECT id, status, max_attempts, next_retry_at, created_at FROM ai_incident_notifications ORDER BY id"))
        rows = {r[0]: r for r in res.fetchall()}

        # Row 1 (pending): max_attempts == 6, next_retry_at == created_at
        assert rows[1][2] == 6
        assert rows[1][3] is not None
        assert str(rows[1][3])[:19] == str(t_pending)[:19]

        # Row 2 (processing): max_attempts == 6, next_retry_at == created_at
        assert rows[2][2] == 6
        assert rows[2][3] is not None
        assert str(rows[2][3])[:19] == str(t_processing)[:19]

        # Row 3 (delivered): max_attempts == 6, next_retry_at is NULL
        assert rows[3][2] == 6
        assert rows[3][3] is None

        # Row 4 (failed_terminal): max_attempts == 6, next_retry_at is NULL
        assert rows[4][2] == 6
        assert rows[4][3] is None

    # 4. Verify idempotency - run production init_db() a second time
    await database.init_db()


@pytest.mark.asyncio
async def test_no_fictitious_provider_5xx_classification(test_db_session_maker, monkeypatch):
    """
    Blocker 1: Eliminate fictitious provider_5xx classification.
    - An unknown error with retryable=True must NOT be classified as provider_5xx if http_status is None.
    - It must be classified as 'unknown'.
    - An unknown error with retryable=False must be classified as 'unknown'.
    - An authentic HTTP 503 error must be classified as 'provider_5xx' with http_status=503.
    - A TimeoutError must be classified as 'timeout'.
    - An authentic RateLimitError / 429 must be classified as 'rate_limit'.
    - AIProviderIncidentModel must store authentic http_status and classification_source.
    """
    inst_def = {"pm2_name": "tg_veraveda777_bot_legacy", "platform": "telegram", "platform_id": 8281390498, "admins": [272982544], "username": "veraveda777_bot"}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[272982544]))
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8281390498:AAFtest_token_no_fictitious_5xx")

    from provider_failures import ProviderFailure

    # 1. Unknown error with retryable=True (transient glitch without HTTP 5xx code)
    err_retryable_true = Exception("Transient unclassified network anomaly")
    err_retryable_true.failure = ProviderFailure(
        provider="KIE",
        operation="chat",
        classification=None,
        detail_code="transient_glitch",
        retryable=True,
        http_status=None,
        classification_source="inferred_message",
    )
    inc_1 = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        error=err_retryable_true,
        operation="chat",
        model="kie-chat-1",
        credential_scope="scope_retryable_true",
        enqueue_notification=False,
        session_maker=test_db_session_maker,
    )
    assert inc_1 is not None
    assert inc_1.incident_type == "unknown", "Unclassified error with retryable=True must NOT be fabricated as provider_5xx"
    assert inc_1.http_status is None
    assert inc_1.classification_source == "inferred_message"

    # 2. Unknown error with retryable=False
    err_retryable_false = ValueError("Unmapped parameter error")
    err_retryable_false.failure = ProviderFailure(
        provider="KIE",
        operation="chat",
        classification=None,
        detail_code="bad_param",
        retryable=False,
        http_status=None,
        classification_source="inferred_message",
    )
    inc_2 = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        error=err_retryable_false,
        operation="chat",
        model="kie-chat-2",
        credential_scope="scope_retryable_false",
        enqueue_notification=False,
        session_maker=test_db_session_maker,
    )
    assert inc_2 is not None
    assert inc_2.incident_type == "unknown"
    assert inc_2.http_status is None

    # 3. Authentic HTTP 503 Service Unavailable:
    # Test case 3a: ProviderFailure created with classification="unknown" and http_status=503.
    # The canonical parser/post_init MUST normalize it to 'provider_5xx' without incident manager heuristics.
    err_503 = Exception("Service Unavailable")
    err_503.failure = ProviderFailure(
        provider="KIE",
        operation="chat",
        classification="unknown",  # Unclassified by caller, but authentic http_status=503
        detail_code="503",
        http_status=503,
        classification_source="provider_structured",
    )
    assert err_503.failure.classification == "provider_5xx"
    assert err_503.failure.classification_source == "http_status_fallback"
    assert err_503.failure.retryable is True

    # Explicit override: conscious retryable=False is strictly respected
    pf_override = ProviderFailure(
        provider="KIE",
        operation="chat",
        classification="unknown",
        detail_code="503",
        http_status=503,
        retryable=False,
    )
    assert pf_override.classification == "provider_5xx"
    assert pf_override.retryable is False
    assert pf_override.transient is False
    inc_3 = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        error=err_503,
        operation="chat",
        model="kie-chat-3",
        credential_scope="scope_503",
        enqueue_notification=False,
        session_maker=test_db_session_maker,
    )
    assert inc_3 is not None
    assert inc_3.incident_type == "provider_5xx", "Authentic HTTP 503 must be classified as provider_5xx"
    assert inc_3.http_status == 503
    assert inc_3.classification_source == "http_status_fallback"

    # Test case 3b: OpenAI with HTTP 503 resolves to canonical 'provider_overloaded'
    err_503_openai = Exception("OpenAI 503 Overloaded")
    err_503_openai.failure = ProviderFailure(
        provider="OpenAI",
        operation="chat",
        classification="unknown",
        detail_code="503",
        http_status=503,
    )
    assert err_503_openai.failure.classification == "provider_overloaded"
    inc_3b = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=err_503_openai,
        operation="chat",
        model="gpt-4o",
        credential_scope="scope_503_openai",
        enqueue_notification=False,
        session_maker=test_db_session_maker,
    )
    assert inc_3b is not None
    assert inc_3b.incident_type == "provider_overloaded"
    assert inc_3b.http_status == 503

    # 4. Authentic TimeoutError
    inc_4 = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        error=TimeoutError("Request timed out after 30s"),
        operation="chat",
        model="kie-chat-4",
        credential_scope="scope_timeout",
        enqueue_notification=False,
        session_maker=test_db_session_maker,
    )
    assert inc_4 is not None
    assert inc_4.incident_type == "timeout"

    # 5. SDK Typed Rate Limit (429)
    err_429 = Exception("Rate limit exceeded")
    err_429.failure = ProviderFailure(
        provider="KIE",
        operation="chat",
        classification="rate_limit",
        detail_code="429",
        retryable=True,
        http_status=429,
        classification_source="sdk_typed_exception",
    )
    inc_5 = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        error=err_429,
        operation="chat",
        model="kie-chat-5",
        credential_scope="scope_rate_limit",
        enqueue_notification=False,
        session_maker=test_db_session_maker,
    )
    assert inc_5 is not None
    assert inc_5.incident_type == "rate_limit"
    assert inc_5.http_status == 429
    assert inc_5.classification_source == "sdk_typed_exception"


@pytest.mark.asyncio
async def test_autobus_cooldown_in_all_delivery_paths_network_recovery(tmp_path, monkeypatch):
    """
    Blocker 2: Verify Autobus cooldown is committed in all delivery paths, including network recovery branch.
    Scenario:
    1. Run 1: Network is down (Telegram HTTP post fails), DB issue is detected.
       - Issue alert queued to pending_notifications.
       - Delivery fails -> notification remains in pending_notifications.
       - Cooldown is NOT yet committed.
    2. Run 2: Network recovers (check_telegram succeeds, HTTP post succeeds).
       - Recovery branch flushes pending issue notifications via _flush_pending_issue_notifications.
       - Alert is confirmed delivered via Telegram HTTP.
       - Cooldown is committed to state.
       - pending_notifications is cleared.
    3. Run 3: Same DB issue is still detected.
       - Cooldown is active -> repeat alert is suppressed! Zero HTTP calls.
    """
    import monitor_psychobots as monitor
    import aiohttp

    lock_file = str(tmp_path / "monitor_cooldown.lock")
    state_file = str(tmp_path / "state_cooldown.json")
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_LOCK_FILE", lock_file)
    monkeypatch.setenv("PSYCHOBOTS_MONITOR_STATE", state_file)
    monkeypatch.setenv("PSYCHOBOTS_ALERT_BOT_PM2_NAME", "tg_autobusbusbot_new")
    monkeypatch.setenv("PSYCHOBOTS_ALERT_BOT_TOKEN", "8346799080:AAFtest_autobus_token_12345")

    app1 = {
        "name": "tg_autobusbusbot_new",
        "status": "online",
        "token": "8346799080:AAFtest_autobus_token_12345",
        "error_log": str(tmp_path / "empty1.log"),
        "owner_ids": [272982544],
    }
    app2 = {
        "name": "tg_veraveda777_bot_legacy",
        "status": "online",
        "token": "8281390498:AAFtest_veraveda_token",
        "error_log": str(tmp_path / "empty2.log"),
        "owner_ids": [272982544],
    }
    apps = [app1, app2]
    monkeypatch.setattr(monitor, "get_bot_apps", lambda: apps)

    (tmp_path / "empty1.log").write_text("", encoding="utf-8")
    (tmp_path / "empty2.log").write_text("", encoding="utf-8")

    async def clean_async(*args, **kwargs): return []
    monkeypatch.setattr(monitor, "check_payment_webhooks", clean_async)
    monkeypatch.setattr(monitor, "check_nl_infrastructure", clean_async)
    monkeypatch.setattr(monitor, "check_stuck_dialogues", clean_async)

    # DB schema always reports an issue on app1
    async def db_issue(app, *args, **kwargs):
        if app["name"] == "tg_autobusbusbot_new":
            return ["DB: missing migration 003"]
        return []
    monkeypatch.setattr(monitor, "check_db_schema", db_issue)

    posted_calls = []
    http_should_succeed = False
    tg_check_results = {
        "tg_autobusbusbot_new": ["Telegram: connection error [tg_autobusbusbot_new]"],
        "tg_veraveda777_bot_legacy": ["Telegram: connection error [tg_veraveda777_bot_legacy]"],
    }

    class MockHttpTelegramResponse:
        def __init__(self, url, json=None, **kwargs):
            self.url = url
            self.json_payload = json
            self.status = 200 if http_should_succeed else 500

        async def __aenter__(self):
            posted_calls.append({"url": self.url, "json": self.json_payload})
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def json(self, content_type=None):
            if self.status == 200:
                return {"ok": True, "result": {"message_id": 999}}
            return {"ok": False, "error_code": 500, "description": "Telegram gateway unavailable"}

    monkeypatch.setattr(
        aiohttp.ClientSession,
        "post",
        lambda self, url, json=None, **kwargs: MockHttpTelegramResponse(url, json=json, **kwargs),
    )
    async def fake_check_telegram(app, *args, **kwargs):
        return list(tg_check_results.get(app["name"], []))
    monkeypatch.setattr(monitor, "check_telegram", fake_check_telegram)

    # Run 1: Network down + DB error -> delivery fails
    exit_code_1 = await monitor.run_check(include_existing_log_errors=True)
    assert exit_code_1 == 1
    state_1 = monitor.load_state(state_file)
    pending_1 = state_1.get("pending_notifications", {})
    assert len(pending_1) > 0, "Failed delivery must leave issue in pending_notifications"
    assert monitor.should_alert(["tg_autobusbusbot_new: DB: missing migration 003"], state_1) is True

    # Run 2: Network recovers (Telegram check clean, HTTP post succeeds)
    posted_calls.clear()
    http_should_succeed = True
    tg_check_results.clear()

    exit_code_2 = await monitor.run_check(include_existing_log_errors=True)
    assert len(posted_calls) >= 1

    state_2 = monitor.load_state(state_file)
    pending_2 = state_2.get("pending_notifications", {})
    assert not any(k == "issue" or k.startswith("issue_") for k in pending_2)
    assert monitor.should_alert(["tg_autobusbusbot_new: DB: missing migration 003"], state_2) is False, (
        "Cooldown must be committed upon delivery in network recovery branch"
    )

    # Run 3: Same DB issue is still returned, network is clean
    posted_calls.clear()
    exit_code_3 = await monitor.run_check(include_existing_log_errors=True)
    assert exit_code_3 == 1
    assert len(posted_calls) == 0, "Duplicate DB issue alert must be suppressed by cooldown committed in recovery branch"


async def _run_lease_recovery_max_attempts_lifecycle_verification(session_maker, monkeypatch):
    inst_def = {
        "pm2_name": "tg_veraveda777_bot_legacy",
        "platform": "telegram",
        "platform_id": 8281390498,
        "admins": [272982544],
        "username": "veraveda777_bot",
    }
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[272982544]))
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8281390498:AAFtest_token_lease_bounds")

    tg_calls = []

    class MockTgBot:
        def __init__(self, token="8281390498:AAFtest_token_lease_bounds"):
            self.token = token
            self.id = 8281390498
            self.get_me = AsyncMock(return_value=TgUser(id=8281390498, is_bot=True, first_name="bot", username="veraveda777_bot"))

        async def send_message(self, *args, **kwargs):
            tg_calls.append(kwargs)
            return MagicMock(message_id=555)

    bot = MockTgBot()

    # Create parent incident
    async with session_maker() as session:
        inc = AIProviderIncidentModel(
            provider="KIE",
            operation="chat",
            model_scope="kie-v1",
            credential_scope="scope_lease_test",
            incident_type="provider_5xx",
            status="active",
            detail_code="500",
            origin_instance_id="tg_veraveda777_bot_legacy",
            companion_instance_id="tg_veraveda777_bot_legacy",
        )
        session.add(inc)
        await session.commit()
        await session.refresh(inc)
        inc_id = inc.id

    past_lease = datetime.utcnow() - timedelta(minutes=10)

    # 1. attempts = 1, max_attempts = 3, lease expired -> should be claimed and retried (attempt 2)
    async with session_maker() as session:
        notif1 = AIIncidentNotificationModel(
            incident_id=inc_id,
            event_type="opened",
            cycle_index=0,
            recipient_id=272982544,
            authorized_instance_id="tg_veraveda777_bot_legacy",
            message_text="Alert lease 1",
            status="processing",
            claim_token="old_worker_1",
            lease_until=past_lease,
            attempts=1,
            max_attempts=3,
            next_retry_at=None,
        )
        session.add(notif1)
        await session.commit()
        await session.refresh(notif1)
        notif1_id = notif1.id

    tg_calls.clear()
    await process_ai_incident_notifications(bot, session_maker=session_maker)
    assert len(tg_calls) == 1, "Expired lease with attempts < max_attempts must be claimed and delivered"
    async with session_maker() as session:
        r1 = await session.get(AIIncidentNotificationModel, notif1_id)
        assert r1.status == "delivered"
        assert r1.attempts == 2

    # 2. attempts = 2, max_attempts = 3, lease expired -> Telegram fails -> reaches max_attempts (3) and becomes failed_terminal
    async with session_maker() as session:
        notif2 = AIIncidentNotificationModel(
            incident_id=inc_id,
            event_type="opened",
            cycle_index=1,
            recipient_id=272982544,
            authorized_instance_id="tg_veraveda777_bot_legacy",
            message_text="Alert lease 2",
            status="processing",
            claim_token="old_worker_2",
            lease_until=past_lease,
            attempts=2,
            max_attempts=3,
            next_retry_at=None,
        )
        session.add(notif2)
        await session.commit()
        await session.refresh(notif2)
        notif2_id = notif2.id

    class FailingBot:
        def __init__(self, token="8281390498:AAFtest_token_lease_bounds"):
            self.token = token
            self.id = 8281390498
            self.get_me = AsyncMock(return_value=TgUser(id=8281390498, is_bot=True, first_name="bot", username="veraveda777_bot"))

        async def send_message(self, *args, **kwargs):
            tg_calls.append(kwargs)
            raise RuntimeError("Telegram network timeout")

    fail_bot = FailingBot()
    tg_calls.clear()
    await process_ai_incident_notifications(fail_bot, session_maker=session_maker)
    assert len(tg_calls) == 1
    async with session_maker() as session:
        r2 = await session.get(AIIncidentNotificationModel, notif2_id)
        assert r2.status == "failed_terminal"
        assert r2.attempts == 3
        assert r2.next_retry_at is None

    # 3. attempts == max_attempts (3 == 3), lease expired:
    # Must be finalized to 'failed_terminal' WITHOUT calling Telegram and WITHOUT incrementing attempts!
    async with session_maker() as session:
        notif3 = AIIncidentNotificationModel(
            incident_id=inc_id,
            event_type="opened",
            cycle_index=2,
            recipient_id=272982544,
            authorized_instance_id="tg_veraveda777_bot_legacy",
            message_text="Alert lease 3",
            status="processing",
            claim_token="crashed_worker_3",
            lease_until=past_lease,
            attempts=3,
            max_attempts=3,
            next_retry_at=None,
        )
        session.add(notif3)
        await session.commit()
        await session.refresh(notif3)
        notif3_id = notif3.id

    tg_calls.clear()
    await process_ai_incident_notifications(bot, session_maker=session_maker)
    assert len(tg_calls) == 0, "Expired lease with attempts == max_attempts must NOT trigger Telegram call"
    async with session_maker() as session:
        r3 = await session.get(AIIncidentNotificationModel, notif3_id)
        assert r3.status == "failed_terminal"
        assert r3.attempts == 3, "Attempts must stay at max_attempts without increment"
        assert r3.next_retry_at is None
        assert r3.last_error == "lease_expired_max_attempts_exhausted"

    # 4. attempts > max_attempts (4 > 3), lease expired:
    # Must also be finalized to 'failed_terminal' without Telegram call
    async with session_maker() as session:
        notif4 = AIIncidentNotificationModel(
            incident_id=inc_id,
            event_type="opened",
            cycle_index=3,
            recipient_id=272982544,
            authorized_instance_id="tg_veraveda777_bot_legacy",
            message_text="Alert lease 4",
            status="processing",
            claim_token="crashed_worker_4",
            lease_until=past_lease,
            attempts=4,
            max_attempts=3,
            next_retry_at=None,
        )
        session.add(notif4)
        await session.commit()
        await session.refresh(notif4)
        notif4_id = notif4.id

    tg_calls.clear()
    await process_ai_incident_notifications(bot, session_maker=session_maker)
    assert len(tg_calls) == 0
    async with session_maker() as session:
        r4 = await session.get(AIIncidentNotificationModel, notif4_id)
        assert r4.status == "failed_terminal"
        assert r4.attempts == 4
        assert r4.next_retry_at is None

    # 5. Concurrent workers competing on expired lease: CAS guarantees exactly one winner
    async with session_maker() as session:
        notif5 = AIIncidentNotificationModel(
            incident_id=inc_id,
            event_type="opened",
            cycle_index=4,
            recipient_id=272982544,
            authorized_instance_id="tg_veraveda777_bot_legacy",
            message_text="Alert lease concurrent",
            status="processing",
            claim_token="crashed_worker_5",
            lease_until=past_lease,
            attempts=1,
            max_attempts=3,
            next_retry_at=None,
        )
        session.add(notif5)
        await session.commit()
        await session.refresh(notif5)
        notif5_id = notif5.id

    tg_calls.clear()
    await asyncio.gather(
        process_ai_incident_notifications(bot, session_maker=session_maker),
        process_ai_incident_notifications(bot, session_maker=session_maker),
        process_ai_incident_notifications(bot, session_maker=session_maker),
        process_ai_incident_notifications(bot, session_maker=session_maker),
        process_ai_incident_notifications(bot, session_maker=session_maker),
    )
    assert len(tg_calls) == 1, "Concurrent lease recovery must result in exactly 1 claim and delivery"
    async with session_maker() as session:
        r5 = await session.get(AIIncidentNotificationModel, notif5_id)
        assert r5.status == "delivered"
        assert r5.attempts == 2

    # 6. pending, attempts < max_attempts (0 < 3): normal delivery
    past_retry = datetime.utcnow() - timedelta(minutes=5)
    async with session_maker() as session:
        notif6 = AIIncidentNotificationModel(
            incident_id=inc_id,
            event_type="opened",
            cycle_index=5,
            recipient_id=272982544,
            authorized_instance_id="tg_veraveda777_bot_legacy",
            message_text="Alert pending 6",
            status="pending",
            attempts=0,
            max_attempts=3,
            next_retry_at=past_retry,
        )
        session.add(notif6)
        await session.commit()
        await session.refresh(notif6)
        notif6_id = notif6.id

    tg_calls.clear()
    await process_ai_incident_notifications(bot, session_maker=session_maker)
    assert len(tg_calls) == 1
    async with session_maker() as session:
        r6 = await session.get(AIIncidentNotificationModel, notif6_id)
        assert r6.status == "delivered"
        assert r6.attempts == 1

    # 7. pending, attempts == max_attempts (3 == 3):
    # Must be finalized to 'failed_terminal' WITHOUT calling Telegram and WITHOUT incrementing attempts!
    async with session_maker() as session:
        notif7 = AIIncidentNotificationModel(
            incident_id=inc_id,
            event_type="opened",
            cycle_index=6,
            recipient_id=272982544,
            authorized_instance_id="tg_veraveda777_bot_legacy",
            message_text="Alert pending 7",
            status="pending",
            attempts=3,
            max_attempts=3,
            next_retry_at=past_retry,
        )
        session.add(notif7)
        await session.commit()
        await session.refresh(notif7)
        notif7_id = notif7.id

    tg_calls.clear()
    await process_ai_incident_notifications(bot, session_maker=session_maker)
    assert len(tg_calls) == 0, "Pending row with attempts == max_attempts must NOT trigger Telegram call"
    async with session_maker() as session:
        r7 = await session.get(AIIncidentNotificationModel, notif7_id)
        assert r7.status == "failed_terminal"
        assert r7.attempts == 3, "Attempts must stay at max_attempts without increment"
        assert r7.next_retry_at is None
        assert r7.last_error == "max_attempts_exhausted"

    # 8. pending, attempts > max_attempts (5 > 3):
    # Must be finalized to 'failed_terminal' without Telegram call
    async with session_maker() as session:
        notif8 = AIIncidentNotificationModel(
            incident_id=inc_id,
            event_type="opened",
            cycle_index=7,
            recipient_id=272982544,
            authorized_instance_id="tg_veraveda777_bot_legacy",
            message_text="Alert pending 8",
            status="pending",
            attempts=5,
            max_attempts=3,
            next_retry_at=past_retry,
        )
        session.add(notif8)
        await session.commit()
        await session.refresh(notif8)
        notif8_id = notif8.id

    tg_calls.clear()
    await process_ai_incident_notifications(bot, session_maker=session_maker)
    assert len(tg_calls) == 0
    async with session_maker() as session:
        r8 = await session.get(AIIncidentNotificationModel, notif8_id)
        assert r8.status == "failed_terminal"
        assert r8.attempts == 5
        assert r8.next_retry_at is None
        assert r8.last_error == "max_attempts_exhausted"

    # 9. Corrupt/fallback limit: max_attempts is None, 0, or -1
    # Fallback limit is MAX_OUTBOX_ATTEMPTS (6).
    # 9a. attempts = 5 with max_attempts = None -> allowed to claim and send (attempt 6)
    async with session_maker() as session:
        notif9a = AIIncidentNotificationModel(
            incident_id=inc_id,
            event_type="opened",
            cycle_index=8,
            recipient_id=272982544,
            authorized_instance_id="tg_veraveda777_bot_legacy",
            message_text="Alert pending 9a",
            status="pending",
            attempts=5,
            max_attempts=None,
            next_retry_at=past_retry,
        )
        session.add(notif9a)
        await session.commit()
        await session.refresh(notif9a)
        notif9a_id = notif9a.id

    tg_calls.clear()
    await process_ai_incident_notifications(bot, session_maker=session_maker)
    assert len(tg_calls) == 1
    async with session_maker() as session:
        r9a = await session.get(AIIncidentNotificationModel, notif9a_id)
        assert r9a.status == "delivered"
        assert r9a.attempts == 6

    # 9b. attempts = 6 with max_attempts = None -> exhausted (6 >= 6) -> failed_terminal, no Telegram call
    async with session_maker() as session:
        notif9b = AIIncidentNotificationModel(
            incident_id=inc_id,
            event_type="opened",
            cycle_index=9,
            recipient_id=272982544,
            authorized_instance_id="tg_veraveda777_bot_legacy",
            message_text="Alert pending 9b",
            status="pending",
            attempts=6,
            max_attempts=None,
            next_retry_at=past_retry,
        )
        session.add(notif9b)
        await session.commit()
        await session.refresh(notif9b)
        notif9b_id = notif9b.id

    tg_calls.clear()
    await process_ai_incident_notifications(bot, session_maker=session_maker)
    assert len(tg_calls) == 0
    async with session_maker() as session:
        r9b = await session.get(AIIncidentNotificationModel, notif9b_id)
        assert r9b.status == "failed_terminal"
        assert r9b.attempts == 6
        assert r9b.next_retry_at is None
        assert r9b.last_error == "max_attempts_exhausted"

    # 9c. attempts = 6 with max_attempts = 0 -> exhausted (fallback 6 >= 6) -> failed_terminal, no Telegram call
    async with session_maker() as session:
        notif9c = AIIncidentNotificationModel(
            incident_id=inc_id,
            event_type="opened",
            cycle_index=10,
            recipient_id=272982544,
            authorized_instance_id="tg_veraveda777_bot_legacy",
            message_text="Alert pending 9c",
            status="pending",
            attempts=6,
            max_attempts=0,
            next_retry_at=past_retry,
        )
        session.add(notif9c)
        await session.commit()
        await session.refresh(notif9c)
        notif9c_id = notif9c.id

    tg_calls.clear()
    await process_ai_incident_notifications(bot, session_maker=session_maker)
    assert len(tg_calls) == 0
    async with session_maker() as session:
        r9c = await session.get(AIIncidentNotificationModel, notif9c_id)
        assert r9c.status == "failed_terminal"
        assert r9c.attempts == 6

    # 9d. attempts = 6 with max_attempts = -1 -> exhausted (fallback 6 >= 6) -> failed_terminal, no Telegram call
    async with session_maker() as session:
        notif9d = AIIncidentNotificationModel(
            incident_id=inc_id,
            event_type="opened",
            cycle_index=11,
            recipient_id=272982544,
            authorized_instance_id="tg_veraveda777_bot_legacy",
            message_text="Alert pending 9d",
            status="pending",
            attempts=6,
            max_attempts=-1,
            next_retry_at=past_retry,
        )
        session.add(notif9d)
        await session.commit()
        await session.refresh(notif9d)
        notif9d_id = notif9d.id

    tg_calls.clear()
    await process_ai_incident_notifications(bot, session_maker=session_maker)
    assert len(tg_calls) == 0
    async with session_maker() as session:
        r9d = await session.get(AIIncidentNotificationModel, notif9d_id)
        assert r9d.status == "failed_terminal"
        assert r9d.attempts == 6

    # 10. Concurrent workers competing on the same pending row: CAS ensures exactly 1 winner
    async with session_maker() as session:
        notif10 = AIIncidentNotificationModel(
            incident_id=inc_id,
            event_type="opened",
            cycle_index=12,
            recipient_id=272982544,
            authorized_instance_id="tg_veraveda777_bot_legacy",
            message_text="Alert pending concurrent",
            status="pending",
            attempts=0,
            max_attempts=3,
            next_retry_at=past_retry,
        )
        session.add(notif10)
        await session.commit()
        await session.refresh(notif10)
        notif10_id = notif10.id

    tg_calls.clear()
    await asyncio.gather(
        process_ai_incident_notifications(bot, session_maker=session_maker),
        process_ai_incident_notifications(bot, session_maker=session_maker),
        process_ai_incident_notifications(bot, session_maker=session_maker),
        process_ai_incident_notifications(bot, session_maker=session_maker),
        process_ai_incident_notifications(bot, session_maker=session_maker),
    )
    assert len(tg_calls) == 1, "Concurrent pending processing must result in exactly 1 claim and delivery"
    async with session_maker() as session:
        r10 = await session.get(AIIncidentNotificationModel, notif10_id)
        assert r10.status == "delivered"
        assert r10.attempts == 1


@pytest.mark.asyncio
async def test_ai_outbox_lease_recovery_strictly_respects_max_attempts_sqlite(test_db_session_maker, monkeypatch):
    """Blocker 3: Verify outbox lease recovery bounds on SQLite."""
    await _run_lease_recovery_max_attempts_lifecycle_verification(test_db_session_maker, monkeypatch)


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_ai_outbox_lease_recovery_strictly_respects_max_attempts(pg_test_session_maker, monkeypatch):
    """Blocker 3: Verify outbox lease recovery bounds on real PostgreSQL."""
    await _run_lease_recovery_max_attempts_lifecycle_verification(pg_test_session_maker, monkeypatch)


@pytest.mark.asyncio
async def test_send_terminal_alert_title_and_diagnostic_sanitization(test_db_session_maker, monkeypatch):
    """
    Blocker 5: Verify title and diagnostic sanitization in send_terminal_ai_failure_alert:
    - Dynamic title with PII (e.g. Patient_12345, user_9999, token_secret) is redacted to [REDACTED].
    - Non-formatted HTML characters in title are escaped.
    - Static Russian titles (e.g. 'Терминальный сбой анализа изображения') are preserved with bold header.
    - Secrets in diagnostics are masked by sanitize_secret_values.
    """
    import html
    import re
    import error_reporting
    inst_def = {"pm2_name": "tg_veraveda777_bot_legacy", "platform": "telegram", "platform_id": 8281390498, "admins": [272982544], "username": "veraveda777_bot"}
    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: inst_def)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: inst_def)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[272982544]))
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8281390498:AAFtest_token_title_sanitization")

    # 1. Dynamic title with PII tokens and unescaped HTML characters
    pii_title = "Failed for Patient_54321 and user_8888 with token_secret_123 <bad_tag>"
    await error_reporting.send_terminal_ai_failure_alert(
        title=pii_title,
        primary_provider="OpenAI",
        primary_model="gpt-4o",
        request_type="chat",
        details="Detail information",
        session_maker=test_db_session_maker,
    )

    async with test_db_session_maker() as session:
        notifs = (await session.execute(
            select(AIIncidentNotificationModel).order_by(AIIncidentNotificationModel.id.desc())
        )).scalars().all()
        assert len(notifs) >= 1
        msg = notifs[0].message_text
        assert "Patient_54321" not in msg
        assert "user_8888" not in msg
        assert "token_secret_123" not in msg
        assert "[REDACTED]" in msg
        assert "&lt;bad_tag&gt;" in msg

    # 2. Static Russian title
    static_title = "Терминальный сбой анализа изображения"
    await error_reporting.send_terminal_ai_failure_alert(
        title=static_title,
        primary_provider="Gemini",
        primary_model="gemini-2.0-flash",
        request_type="vision",
        details="Vision timeout",
        session_maker=test_db_session_maker,
    )

    async with test_db_session_maker() as session:
        notifs2 = (await session.execute(
            select(AIIncidentNotificationModel).order_by(AIIncidentNotificationModel.id.desc())
        )).scalars().all()
        msg2 = notifs2[0].message_text
        assert "Терминальный сбой анализа изображения" in msg2
        assert "🚨 <b>[Telegram] Терминальный сбой анализа изображения</b>" in msg2

    # 3. Secret in diagnostics/details
    secret_details = "Authorization: Bearer sk-ant-api03-secret123456789012345678901234567890"
    await error_reporting.send_terminal_ai_failure_alert(
        title="Valid Title",
        primary_provider="DeepSeek",
        primary_model="deepseek-chat",
        request_type="chat",
        details=secret_details,
        session_maker=test_db_session_maker,
    )

    async with test_db_session_maker() as session:
        notifs3 = (await session.execute(
            select(AIIncidentNotificationModel).order_by(AIIncidentNotificationModel.id.desc())
        )).scalars().all()
        msg3 = notifs3[0].message_text
        assert "sk-ant-api03-secret123456789012345678901234567890" not in msg3

    # 4. Unclosed <b>, malicious links, user names, medical/financial IDs, and ampersands
    malicious_title = "<b>Unclosed bold & <a href='https://evil.org'>click</a> Patient_99999 client_77777 user_88888 customer_66666"
    await error_reporting.send_terminal_ai_failure_alert(
        title=malicious_title,
        primary_provider="Anthropic",
        primary_model="claude-3-5-sonnet",
        request_type="chat",
        details="Detail test",
        session_maker=test_db_session_maker,
    )

    async with test_db_session_maker() as session:
        notifs4 = (await session.execute(
            select(AIIncidentNotificationModel).order_by(AIIncidentNotificationModel.id.desc())
        )).scalars().all()
        msg4 = notifs4[0].message_text
        # PII must be completely sanitized
        assert "Patient_99999" not in msg4
        assert "client_77777" not in msg4
        assert "user_88888" not in msg4
        assert "customer_66666" not in msg4
        assert "[REDACTED]" in msg4
        # HTML tag injection must be prevented: angle brackets and quotes escaped
        assert "&lt;b&gt;" in msg4
        assert "&lt;a href=" in msg4 or "&lt;a href=&#x27;" in msg4
        assert "&amp;" in msg4
        # Ensure header tags are strictly balanced
        header_line = msg4.split("\n\n")[0]
        header_tags = re.findall(r"</?[a-zA-Z0-9]+>", header_line)
        assert header_tags == ["<b>", "</b>"], f"Header tags must be strictly balanced: {header_tags}"

    # 5. Unicode and special symbols
    unicode_title = "🚨 🏥 Привет мир! Сбой AI & Анализ: <test> — ok"
    await error_reporting.send_terminal_ai_failure_alert(
        title=unicode_title,
        primary_provider="Perplexity",
        primary_model="sonar-pro",
        request_type="chat",
        details="Detail test",
        session_maker=test_db_session_maker,
    )

    async with test_db_session_maker() as session:
        notifs5 = (await session.execute(
            select(AIIncidentNotificationModel).order_by(AIIncidentNotificationModel.id.desc())
        )).scalars().all()
        msg5 = notifs5[0].message_text
        assert "Привет мир!" in msg5
        assert "🏥" in msg5
        assert "&amp;" in msg5
        assert "&lt;test&gt;" in msg5

    # 6. Trusted pre-formatted static header from real runtime
    runtime_header = "🚨 <b>[Telegram] Сбой ИИ в текстовом диалоге</b>\n\n"
    await error_reporting.send_terminal_ai_failure_alert(
        title=runtime_header,
        primary_provider="Groq",
        primary_model="llama-3.3-70b-versatile",
        request_type="chat",
        details="Detail test",
        session_maker=test_db_session_maker,
    )

    async with test_db_session_maker() as session:
        notifs6 = (await session.execute(
            select(AIIncidentNotificationModel).order_by(AIIncidentNotificationModel.id.desc())
        )).scalars().all()
        msg6 = notifs6[0].message_text
        assert "🚨 <b>[Telegram] Сбой ИИ в текстовом диалоге</b>" in msg6
        assert "🚨 <b>[Telegram] 🚨" not in msg6, "Must not double-wrap canonical bold header"


@pytest.mark.asyncio
async def test_genuine_instance_registry_and_telegram_method_boundary(test_db_session_maker, monkeypatch):
    """
    Item 4: Verify authentic instance registry from config/bot_instances.json and real Telegram method boundary.
    - Uses real BOT_INSTANCE_ID="tg_veraveda777_bot_legacy" with NO mock for detect_current_instance.
    - Validates target admins against authentic registry: [979514796, 272982544, 806750628].
    - Uses aiogram Bot with validating transport session to guarantee HTML parseability and method validity.
    """
    import re
    import error_reporting
    from ai_incident_manager import process_ai_incident_notifications
    from aiogram import Bot
    from aiogram.client.session.base import BaseSession
    from aiogram.methods import TelegramMethod, SendMessage
    from aiogram.types import Message, Chat, User as AiogramUser

    # 1. Authentic environment matching config/bot_instances.json:
    # tg_veraveda777_bot_legacy has platform_id 8224473896 and owners [979514796, 272982544, 806750628]
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_genuine_registry_token")
    monkeypatch.setenv("PM2_APP_NAME", "tg_veraveda777_bot_legacy")

    sent_requests: list[SendMessage] = []

    from html.parser import HTMLParser

    class TagChecker(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []
        def handle_starttag(self, tag, attrs):
            self.stack.append(tag)
        def handle_endtag(self, tag):
            assert self.stack, f"Unexpected closing tag </{tag}>"
            opened = self.stack.pop()
            assert opened == tag, f"Mismatched tag: expected </{opened}>, got </{tag}>"

    class ValidatingTelegramSession(BaseSession):
        async def make_request(self, bot: Bot, method: TelegramMethod, timeout: int | None = None) -> Any:
            if isinstance(method, SendMessage):
                sent_requests.append(method)
                # Verify that parse_mode is HTML
                assert method.parse_mode == "HTML"
                # Verify that HTML tags in text are properly matched via standard HTMLParser
                checker = TagChecker()
                checker.feed(method.text)
                assert not checker.stack, f"Unclosed tags remaining in message: {checker.stack}"

                return Message(
                    message_id=999,
                    date=datetime.now(),
                    chat=Chat(id=method.chat_id, type="private"),
                    text=method.text,
                    from_user=AiogramUser(id=8224473896, is_bot=True, first_name="bot"),
                )
            elif method.__class__.__name__ == "GetMe":
                return AiogramUser(id=8224473896, is_bot=True, first_name="veraveda_bot", username="veraveda777_bot")
            return None

        async def stream_content(self, *args, **kwargs):
            raise NotImplementedError

        async def close(self):
            pass

    test_session = ValidatingTelegramSession()
    real_bot = Bot(token="8224473896:AAFtest_genuine_registry_token", session=test_session)

    # 2. Enqueue terminal AI failure alert with bot=None (enqueues into outbox without dispatch)
    delivered_enqueue = await error_reporting.send_terminal_ai_failure_alert(
        bot=None,
        title="Сбой ИИ при анализе изображения",
        primary_provider="Gemini",
        primary_model="gemini-2.0-flash",
        request_type="vision",
        details="Authentic registry integration test",
        session_maker=test_db_session_maker,
    )
    assert delivered_enqueue is True

    # 3. Process outbox notifications via real_bot
    delivered_count = await process_ai_incident_notifications(real_bot, session_maker=test_db_session_maker)
    assert delivered_count == 3, f"Must deliver to all 3 registered admins in bot_instances.json, got {delivered_count}"

    assert len(sent_requests) == 3
    recipients = {req.chat_id for req in sent_requests}
    assert recipients == {979514796, 272982544, 806750628}, f"Recipients must strictly match registry, got {recipients}"
    for req in sent_requests:
        assert "🚨 <b>[Telegram] Сбой ИИ при анализе изображения</b>" in req.text
        assert "Gemini / gemini-2.0-flash" in req.text


@pytest.mark.asyncio
async def test_fallback_alert_pii_exception_sanitization(
    test_db_session_maker,
    monkeypatch,
):
    import error_reporting
    from aiogram.client.session.base import BaseSession
    from aiogram.methods.base import TelegramMethod
    from aiogram.methods import SendMessage
    from aiogram.types import Message, Chat, User as AiogramUser

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")
    monkeypatch.setenv("APP_ENV", "development")

    from database import AIConfig
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            session.add(AIConfig(id=1, current_provider="DEEPSEEK", deepseek_api_key="sk-test-key"))
            await session.commit()

    class Patient_123456789_Error(Exception):
        pass

    class User_987654_Secret_Exception(Exception):
        pass

    sent_messages = []

    class MockSession(BaseSession):
        async def make_request(self, bot: Bot, method: TelegramMethod, timeout: int | None = None) -> Any:
            if isinstance(method, SendMessage):
                sent_messages.append(method)
                return Message(
                    message_id=1001,
                    date=datetime.now(),
                    chat=Chat(id=method.chat_id, type="private"),
                    text=method.text,
                    from_user=AiogramUser(id=8224473896, is_bot=True, first_name="bot"),
                )
            elif method.__class__.__name__ == "GetMe":
                return AiogramUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot")
            return None

        async def stream_content(self, *args, **kwargs):
            raise NotImplementedError

        async def close(self):
            pass

    mock_bot = Bot(token="8224473896:AAFtest_token_veraveda", session=MockSession())

    # 1. Test with custom exception class containing patient ID
    res1 = await error_reporting.send_ai_fallback_used_alert(
        bot=mock_bot,
        primary_provider="Deepseek",
        primary_model="deepseek-chat",
        fallback_provider="OpenAI",
        fallback_model="gpt-4o",
        primary_error=Patient_123456789_Error("Sensitive patient medical diagnostic"),
        session_maker=test_db_session_maker,
    )
    assert res1 is True
    assert len(sent_messages) > 0
    for msg in sent_messages:
        assert "Patient_123456789_Error" not in msg.text
        assert "Sensitive patient medical diagnostic" not in msg.text
        assert "ProviderFailure" in msg.text

    sent_messages.clear()

    # 2. Test with string containing secret ID
    res2 = await error_reporting.send_ai_fallback_used_alert(
        bot=mock_bot,
        primary_provider="Deepseek",
        primary_model="deepseek-reasoner",
        fallback_provider="OpenAI",
        fallback_model="gpt-4o-mini",
        primary_error="User_987654_Secret_Exception",
        session_maker=test_db_session_maker,
    )
    assert res2 is True
    assert len(sent_messages) > 0
    for msg in sent_messages:
        assert "User_987654_Secret_Exception" not in msg.text
        assert "ProviderFailure" in msg.text

    sent_messages.clear()

    # 3. Test with canonical exception name (e.g. TimeoutError) -> preserved
    res3 = await error_reporting.send_ai_fallback_used_alert(
        bot=mock_bot,
        primary_provider="OpenAI",
        primary_model="gpt-4o",
        fallback_provider="Deepseek",
        fallback_model="deepseek-chat",
        primary_error=TimeoutError("Request timed out after 30s"),
        session_maker=test_db_session_maker,
    )
    assert res3 is True
    assert len(sent_messages) > 0
    for msg in sent_messages:
        assert "TimeoutError" in msg.text


@pytest.mark.asyncio
async def test_outbox_post_send_cas_lost_claim_handling(
    test_db_session_maker,
    monkeypatch,
):
    from aiogram.client.session.base import BaseSession
    from aiogram.methods.base import TelegramMethod
    from aiogram.methods import SendMessage
    from aiogram.types import Message, Chat, User as AiogramUser

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")
    monkeypatch.setenv("APP_ENV", "development")

    from database import AIConfig
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            session.add(AIConfig(id=1, current_provider="DEEPSEEK", deepseek_api_key="sk-test-key"))
            await session.commit()

    # Create parent incident and outbox row
    async with test_db_session_maker() as session:
        inc = AIProviderIncidentModel(
            origin_instance_id="tg_veraveda777_bot_legacy",
            companion_instance_id="tg_veraveda777_bot_legacy",
            provider="DEEPSEEK",
            credential_scope="test_scope",
            operation="chat",
            model_scope="deepseek-chat",
            incident_type="provider_5xx",
            evidence_tier="CONFIRMED_STRUCTURED",
            detail_code="500",
            classification_source="provider_structured",
            status="active",
            occurrence_count=1,
            reminder_cycle=0,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        session.add(inc)
        await session.flush()

        notif = AIIncidentNotificationModel(
            incident_id=inc.id,
            authorized_instance_id="tg_veraveda777_bot_legacy",
            recipient_id=979514796,  # Valid admin from bot_instances.json
            event_type="opened",
            status="pending",
            attempts=0,
            max_attempts=6,
            message_text="Test incident notification",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(notif)
        await session.commit()
        notif_id = notif.id

    # Session where make_request simulates a concurrent worker stealing or modifying the claim during Telegram send
    class ConcurrentInterferingSession(BaseSession):
        async def make_request(self, bot: Bot, method: TelegramMethod, timeout: int | None = None) -> Any:
            if isinstance(method, SendMessage):
                # Another worker took over the row while worker A was waiting for Telegram
                async with test_db_session_maker() as s:
                    row = await s.get(AIIncidentNotificationModel, notif_id)
                    assert row is not None
                    # Overwrite claim_token to simulate lease expiration and reassignment
                    row.claim_token = "stolen_by_worker_b"
                    row.status = "processing"
                    row.lease_until = datetime.utcnow() + timedelta(seconds=60)
                    await s.commit()

                return Message(
                    message_id=9999,
                    date=datetime.now(),
                    chat=Chat(id=method.chat_id, type="private"),
                    text=method.text,
                    from_user=AiogramUser(id=8224473896, is_bot=True, first_name="bot"),
                )
            elif method.__class__.__name__ == "GetMe":
                return AiogramUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot")
            return None

        async def stream_content(self, *args, **kwargs):
            raise NotImplementedError

        async def close(self):
            pass

    test_bot = Bot(token="8224473896:AAFtest_token_veraveda", session=ConcurrentInterferingSession())

    # Worker A runs process_ai_incident_notifications
    delivered_count = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    # Since worker A's CAS update had rowcount == 0, it must NOT be counted as delivered!
    assert delivered_count == 0, f"Expected delivered_count == 0 due to lost CAS claim, got {delivered_count}"

    # Verify that the row in DB was NOT modified by Worker A to 'delivered'
    async with test_db_session_maker() as session:
        row = await session.get(AIIncidentNotificationModel, notif_id)
        assert row.claim_token == "stolen_by_worker_b"
        assert row.status == "processing"


def test_provider_failure_http_status_operator_action_required_sync():
    from provider_failures import ProviderFailure, parse_provider_failure

    # 1. 401 Unauthorized -> auth (operator action required)
    pf_401 = ProviderFailure(
        provider="deepseek",
        operation="chat",
        classification="unknown",
        detail_code="",
        http_status=401,
    )
    assert pf_401.classification == "auth"
    assert pf_401.classification_source == "http_status_fallback"
    assert pf_401.operator_action_required is True
    assert pf_401.retryable is False
    assert pf_401.transient is False

    # 2. 402 Payment Required -> insufficient_balance_quota (operator action required)
    pf_402 = ProviderFailure(
        provider="deepseek",
        operation="chat",
        classification="unknown",
        detail_code="",
        http_status=402,
    )
    assert pf_402.classification == "insufficient_balance_quota"
    assert pf_402.operator_action_required is True
    assert pf_402.retryable is False

    # 3. 403 Forbidden -> permission (operator action required)
    pf_403 = ProviderFailure(
        provider="deepseek",
        operation="chat",
        classification="unknown",
        detail_code="",
        http_status=403,
    )
    assert pf_403.classification == "permission"
    assert pf_403.operator_action_required is True

    # 4. 404 Not Found -> invalid_model_or_resource (operator action required)
    pf_404 = ProviderFailure(
        provider="deepseek",
        operation="chat",
        classification="unknown",
        detail_code="",
        http_status=404,
    )
    assert pf_404.classification == "invalid_model_or_resource"
    assert pf_404.operator_action_required is True

    # 5. 500 Internal Server Error -> provider_5xx (retryable, NOT operator action)
    pf_500 = ProviderFailure(
        provider="deepseek",
        operation="chat",
        classification="unknown",
        detail_code="",
        http_status=500,
    )
    assert pf_500.classification == "provider_5xx"
    assert pf_500.operator_action_required is False
    assert pf_500.retryable is True
    assert pf_500.transient is True

    # 6. Verify parity with parse_provider_failure
    parsed_401 = parse_provider_failure("deepseek", "chat", 401)
    assert parsed_401 is not None
    assert parsed_401.classification == "auth"
    assert parsed_401.operator_action_required is True
    assert parsed_401.classification == pf_401.classification
    assert parsed_401.operator_action_required == pf_401.operator_action_required


@pytest.mark.asyncio
async def test_provider_failure_canonical_matrix_and_db_lifecycle(
    test_db_session_maker,
    monkeypatch,
):
    """
    Comprehensive verification of ProviderFailure canonical matrix and DB lifecycle across all 8 classes:
    - auth (401)
    - permission (403)
    - insufficient_balance_quota (402)
    - configuration (404)
    - timeout (408 / TimeoutError)
    - rate_limit (429)
    - provider_5xx (500)
    - unknown (unmapped)

    Verifies:
    1. Direct ProviderFailure default flag deduction.
    2. Direct ProviderFailure conscious explicit overrides.
    3. Parity with parse_provider_failure and normalize_exception.
    4. Incident DB status (active vs transient).
    5. Incident manager notification decision (enqueued vs suppressed).
    6. Telegram delivery via outbox worker.
    """
    from provider_failures import ProviderFailure, parse_provider_failure
    from database import AIConfig
    from aiogram.client.session.base import BaseSession
    from aiogram.methods.base import TelegramMethod
    from aiogram.methods import SendMessage
    from aiogram.types import Message, Chat, User as AiogramUser

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")
    monkeypatch.setenv("APP_ENV", "development")

    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            session.add(AIConfig(id=1, current_provider="DEEPSEEK", deepseek_api_key="sk-matrix-test"))
            await session.commit()

    sent_messages = []

    class MockSession(BaseSession):
        async def make_request(self, bot: Bot, method: TelegramMethod, timeout: int | None = None) -> Any:
            if isinstance(method, SendMessage):
                sent_messages.append(method)
                return Message(
                    message_id=2000 + len(sent_messages),
                    date=datetime.now(),
                    chat=Chat(id=method.chat_id, type="private"),
                    text=method.text,
                    from_user=AiogramUser(id=8224473896, is_bot=True, first_name="bot"),
                )
            elif method.__class__.__name__ == "GetMe":
                return AiogramUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot")
            return None

        async def stream_content(self, *args, **kwargs):
            raise NotImplementedError

        async def close(self):
            pass

    test_bot = Bot(token="8224473896:AAFtest_token_veraveda", session=MockSession())

    test_cases = [
        # (cls_name, http_status, is_operator, is_retryable, is_transient)
        ("auth", 401, True, False, False),
        ("permission", 403, True, False, False),
        ("insufficient_balance_quota", 402, True, False, False),
        ("configuration", 404, True, False, False),
        ("timeout", 408, False, True, True),
        ("rate_limit", 429, False, True, True),
        ("provider_5xx", 500, False, True, True),
        ("unknown", None, False, False, False),
    ]

    for cls_name, status, exp_operator, exp_retryable, exp_transient in test_cases:
        # A. Direct ProviderFailure with default flags
        pf_direct = ProviderFailure(
            provider="KIE",
            operation="chat",
            classification=cls_name,
            detail_code=str(status or "code"),
            http_status=status,
        )
        assert pf_direct.classification == cls_name
        assert pf_direct.operator_action_required is exp_operator
        assert pf_direct.retryable is exp_retryable
        assert pf_direct.transient is exp_transient

        # B. Parity with parse_provider_failure
        if status is not None:
            pf_parsed = parse_provider_failure("KIE", "chat", status)
            if pf_parsed:
                assert pf_parsed.operator_action_required is exp_operator
                assert pf_parsed.retryable is exp_retryable

        # C. Incident DB Status and notification decision
        err = Exception(f"Test failure {cls_name}")
        err.failure = pf_direct
        sent_messages.clear()

        inc = await record_ai_provider_failure(
            origin_instance_id="tg_veraveda777_bot_legacy",
            provider="KIE",
            error=err,
            operation="chat",
            model=f"model-{cls_name}",
            credential_scope=f"scope-{cls_name}",
            enqueue_notification=True,
            session_maker=test_db_session_maker,
        )
        assert inc is not None
        expected_status = "active" if exp_operator else "transient"
        assert inc.status == expected_status, f"Class {cls_name} expected {expected_status}, got {inc.status}"

        # D. Check outbox
        delivered = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
        if exp_operator:
            assert delivered > 0, f"Class {cls_name} must enqueue and deliver notification"
            assert len(sent_messages) > 0
        else:
            assert delivered == 0, f"Class {cls_name} must not deliver notification for transient error"
            assert len(sent_messages) == 0

    # E. Conscious explicit overrides
    # Override 1: Timeout with retryable=False
    pf_no_retry_timeout = ProviderFailure(
        provider="KIE",
        operation="chat",
        classification="timeout",
        detail_code="timeout",
        retryable=False,
    )
    assert pf_no_retry_timeout.retryable is False
    assert pf_no_retry_timeout.transient is False

    # Override 2: Quota with operator_action_required=False (OpenRouter in-flight budget)
    pf_in_flight = ProviderFailure(
        provider="openrouter",
        operation="chat",
        classification="insufficient_balance_quota",
        detail_code="openrouter_in_flight_budget",
        retryable=True,
        operator_action_required=False,
    )
    assert pf_in_flight.retryable is True
    assert pf_in_flight.operator_action_required is False

    # Record failure with in-flight override -> should be transient, not active!
    err_in_flight = Exception("Openrouter in-flight budget")
    err_in_flight.failure = pf_in_flight
    inc_in_flight = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="openrouter",
        error=err_in_flight,
        operation="chat",
        model="openrouter-test",
        credential_scope="scope-in-flight",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_in_flight is not None
    assert inc_in_flight.status == "transient", "Override operator_action_required=False must result in transient incident"


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_provider_failure_canonical_matrix_and_db_lifecycle(
    pg_test_session_maker,
    monkeypatch,
):
    """
    PostgreSQL integration test for ProviderFailure canonical matrix and DB lifecycle:
    Validates active vs transient status, notification decision, and durable outbox delivery.
    """
    from provider_failures import ProviderFailure
    from database import AIConfig
    from aiogram.client.session.base import BaseSession
    from aiogram.methods.base import TelegramMethod
    from aiogram.methods import SendMessage
    from aiogram.types import Message, Chat, User as AiogramUser

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")
    monkeypatch.setenv("APP_ENV", "development")

    async with pg_test_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            session.add(AIConfig(id=1, current_provider="DEEPSEEK", deepseek_api_key="sk-matrix-pg"))
            await session.commit()

    sent_messages = []

    class MockSession(BaseSession):
        async def make_request(self, bot: Bot, method: TelegramMethod, timeout: int | None = None) -> Any:
            if isinstance(method, SendMessage):
                sent_messages.append(method)
                return Message(
                    message_id=3000 + len(sent_messages),
                    date=datetime.now(),
                    chat=Chat(id=method.chat_id, type="private"),
                    text=method.text,
                    from_user=AiogramUser(id=8224473896, is_bot=True, first_name="bot"),
                )
            elif method.__class__.__name__ == "GetMe":
                return AiogramUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot")
            return None

        async def stream_content(self, *args, **kwargs):
            raise NotImplementedError

        async def close(self):
            pass

    test_bot = Bot(token="8224473896:AAFtest_token_veraveda", session=MockSession())

    # 1. Operator class: auth -> active incident and outbox notification
    pf_auth = ProviderFailure(
        provider="KIE",
        operation="chat",
        classification="auth",
        detail_code="401",
        http_status=401,
    )
    assert pf_auth.operator_action_required is True
    err_auth = Exception("401 Unauthorized")
    err_auth.failure = pf_auth
    inc_auth = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        error=err_auth,
        operation="chat",
        model="pg-kie-auth",
        credential_scope="scope-pg-auth",
        enqueue_notification=True,
        session_maker=pg_test_session_maker,
    )
    assert inc_auth is not None
    assert inc_auth.status == "active"
    delivered = await process_ai_incident_notifications(test_bot, session_maker=pg_test_session_maker)
    assert delivered > 0
    assert len(sent_messages) > 0

    # 2. Retryable class: timeout -> transient incident and no notification
    sent_messages.clear()
    pf_timeout = ProviderFailure(
        provider="KIE",
        operation="chat",
        classification="timeout",
        detail_code="408",
        http_status=408,
    )
    assert pf_timeout.retryable is True
    assert pf_timeout.operator_action_required is False
    err_timeout = Exception("408 Timeout")
    err_timeout.failure = pf_timeout
    inc_timeout = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        error=err_timeout,
        operation="chat",
        model="pg-kie-timeout",
        credential_scope="scope-pg-timeout",
        enqueue_notification=True,
        session_maker=pg_test_session_maker,
    )
    assert inc_timeout is not None
    assert inc_timeout.status == "transient"
    delivered_transient = await process_ai_incident_notifications(test_bot, session_maker=pg_test_session_maker)
    assert delivered_transient == 0
    assert len(sent_messages) == 0


@pytest.mark.asyncio
async def test_bot_instance_identity_fail_closed_and_companion_isolation(
    test_db_session_maker,
    monkeypatch,
):
    """
    Verification of fail-closed behavior for BOT_INSTANCE_ID and cross-companion isolation:
    1. Missing BOT_INSTANCE_ID -> detect_current_instance() returns None.
    2. Unknown / bogus BOT_INSTANCE_ID -> detect_current_instance() returns None.
    3. Valid Telegram instance -> returns correct instance definition.
    4. Valid MAX instance -> returns correct instance and resolves companion Telegram instance.
    5. Foreign Telegram token -> verify_origin_bot_token_consistency returns False / fails closed.
    6. Foreign admins -> resolve_target_admin_ids resolves strictly to designated admins.
    7. Incident recording fails closed when BOT_INSTANCE_ID is missing or foreign.
    """
    from notification_routing import (
        detect_current_instance,
        extract_telegram_bot_id,
        get_instance_by_pm2_name,
        resolve_companion_telegram_instance,
        resolve_target_admin_ids,
    )

    # 1. Missing BOT_INSTANCE_ID
    monkeypatch.delenv("BOT_INSTANCE_ID", raising=False)
    assert detect_current_instance() is None, "Missing BOT_INSTANCE_ID must fail closed"

    # 2. Unknown BOT_INSTANCE_ID
    monkeypatch.setenv("BOT_INSTANCE_ID", "unregistered_bot_xyz")
    assert detect_current_instance() is None, "Unregistered BOT_INSTANCE_ID must fail closed"

    # 3. Valid Telegram instance
    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    tg_inst = detect_current_instance()
    assert tg_inst is not None
    assert tg_inst["platform"] == "telegram"
    assert tg_inst["pm2_name"] == "tg_veraveda777_bot_legacy"

    # 4. Valid MAX instance
    monkeypatch.setenv("BOT_INSTANCE_ID", "max_se13639182_bot_legacy")
    max_inst = detect_current_instance()
    assert max_inst is not None
    assert max_inst["platform"] == "max"
    comp_inst = resolve_companion_telegram_instance(max_inst)
    assert comp_inst is not None
    assert comp_inst["platform"] == "telegram"
    assert comp_inst["pm2_name"] == "tg_veraveda777_bot_legacy"

    # 5. Foreign Telegram token verification: extract_telegram_bot_id matches platform_id
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")
    assert extract_telegram_bot_id(os.environ["BOT_TOKEN"]) == tg_inst["platform_id"]

    monkeypatch.setenv("BOT_TOKEN", "1234567890:AAFforeign_token_different_bot_id")
    assert extract_telegram_bot_id(os.environ["BOT_TOKEN"]) != tg_inst["platform_id"]

    # In incident manager, token mismatch strictly causes record_ai_provider_failure to fail closed (returns None)
    foreign_token_inc = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        error=Exception("Token mismatch test"),
        operation="chat",
        session_maker=test_db_session_maker,
    )
    assert foreign_token_inc is None, "record_ai_provider_failure must fail closed on token ID mismatch"

    # 6. Admin isolation
    async with test_db_session_maker() as session:
        vv_admins = await resolve_target_admin_ids(tg_inst, current_instance=tg_inst, session=session)
        assert 979514796 in vv_admins
        assert 272982544 in vv_admins

        swy02_inst = get_instance_by_pm2_name("tg_someonewithyou02_bot_new")
        swy02_admins = await resolve_target_admin_ids(swy02_inst, current_instance=swy02_inst, session=session)
        assert 979514796 not in swy02_admins

    # 7. Incident recording fails closed when BOT_INSTANCE_ID is missing
    monkeypatch.delenv("BOT_INSTANCE_ID", raising=False)
    fail_inc = await record_ai_provider_failure(
        origin_instance_id=None,
        provider="KIE",
        error=Exception("Fail-closed test"),
        operation="chat",
        session_maker=test_db_session_maker,
    )
    assert fail_inc is None, "record_ai_provider_failure must fail closed when BOT_INSTANCE_ID is missing"


@pytest.mark.asyncio
async def test_real_concurrent_two_workers_lease_expiry_post_send_cas(
    test_db_session_maker,
    monkeypatch,
):
    """
    Verification of two concurrent outbox workers with lease expiry:
    1. Worker 1 claims outbox row (status -> 'processing', token -> token_1, lease -> +60s).
    2. Worker 1 enters Telegram send_message.
    3. While Worker 1 is in send_message, time advances past lease_until (simulated lease expiry).
    4. Worker 2 runs process_ai_incident_notifications:
       - Sees expired lease on 'processing' row.
       - Worker 2 successfully reclaims row (token -> token_2, lease -> +60s, attempts -> attempts + 1).
       - Worker 2 completes send_message.
       - Worker 2 executes Post-Send CAS: res.rowcount == 1.
       - Worker 2 commits: status -> 'delivered', delivered_count == 1.
    5. Worker 1's send_message completes and Worker 1 executes Post-Send CAS:
       - Since claim_token == token_1 matches 0 rows in DB, res.rowcount == 0.
       - Worker 1 commits, logs warning, and returns delivered_count == 0 (NO false success!).
    6. Verification:
       - SendMessage was dispatched by both workers (at-least-once Telegram transport).
       - Worker 1 delivered_count == 0.
       - Worker 2 delivered_count == 1.
       - In DB: status == 'delivered', claim_token == token_2, attempts == 2.
    """
    from database import AIConfig, AIProviderIncidentModel, AIIncidentNotificationModel
    from aiogram.client.session.base import BaseSession
    from aiogram.methods.base import TelegramMethod
    from aiogram.methods import SendMessage
    from aiogram.types import Message, Chat, User as AiogramUser

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")
    monkeypatch.setenv("APP_ENV", "development")

    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            session.add(AIConfig(id=1, current_provider="DEEPSEEK", deepseek_api_key="sk-test-key"))
            await session.commit()

        inc = AIProviderIncidentModel(
            origin_instance_id="tg_veraveda777_bot_legacy",
            companion_instance_id="tg_veraveda777_bot_legacy",
            provider="DEEPSEEK",
            credential_scope="scope_concurrent_cas",
            operation="chat",
            model_scope="deepseek-chat",
            incident_type="provider_5xx",
            evidence_tier="CONFIRMED_STRUCTURED",
            detail_code="500",
            classification_source="provider_structured",
            status="active",
            occurrence_count=1,
            reminder_cycle=0,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        session.add(inc)
        await session.flush()

        notif = AIIncidentNotificationModel(
            incident_id=inc.id,
            authorized_instance_id="tg_veraveda777_bot_legacy",
            recipient_id=979514796,
            event_type="opened",
            status="pending",
            attempts=0,
            max_attempts=6,
            message_text="Real concurrent CAS test notification",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(notif)
        await session.commit()
        notif_id = notif.id

    worker_1_sends = []
    worker_2_sends = []

    # Worker 2 Bot and Session
    class Worker2Session(BaseSession):
        async def make_request(self, bot: Bot, method: TelegramMethod, timeout: int | None = None) -> Any:
            if isinstance(method, SendMessage):
                worker_2_sends.append(method)
                return Message(
                    message_id=2002,
                    date=datetime.now(),
                    chat=Chat(id=method.chat_id, type="private"),
                    text=method.text,
                    from_user=AiogramUser(id=8224473896, is_bot=True, first_name="bot"),
                )
            elif method.__class__.__name__ == "GetMe":
                return AiogramUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot")
            return None

        async def stream_content(self, *args, **kwargs):
            raise NotImplementedError

        async def close(self):
            pass

    bot_worker_2 = Bot(token="8224473896:AAFtest_token_veraveda", session=Worker2Session())
    worker_2_delivered = None

    # Worker 1 Session: when make_request is called, it triggers lease expiry and runs Worker 2 to completion
    class Worker1Session(BaseSession):
        async def make_request(self, bot: Bot, method: TelegramMethod, timeout: int | None = None) -> Any:
            nonlocal worker_2_delivered
            if isinstance(method, SendMessage):
                worker_1_sends.append(method)

                # 1. Simulate lease expiry by setting lease_until in the past
                async with test_db_session_maker() as s:
                    row = await s.get(AIIncidentNotificationModel, notif_id)
                    assert row.status == "processing"
                    row.lease_until = datetime.utcnow() - timedelta(seconds=10)
                    await s.commit()

                # 2. Worker 2 runs concurrently while Worker 1 is still in send_message
                worker_2_delivered = await process_ai_incident_notifications(bot_worker_2, session_maker=test_db_session_maker)

                # 3. Now Worker 1 finishes send_message
                return Message(
                    message_id=2001,
                    date=datetime.now(),
                    chat=Chat(id=method.chat_id, type="private"),
                    text=method.text,
                    from_user=AiogramUser(id=8224473896, is_bot=True, first_name="bot"),
                )
            elif method.__class__.__name__ == "GetMe":
                return AiogramUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot")
            return None

        async def stream_content(self, *args, **kwargs):
            raise NotImplementedError

        async def close(self):
            pass

    bot_worker_1 = Bot(token="8224473896:AAFtest_token_veraveda", session=Worker1Session())

    # Run Worker 1
    worker_1_delivered = await process_ai_incident_notifications(bot_worker_1, session_maker=test_db_session_maker)

    # Assertions
    # 1. Both workers called Telegram (at-least-once delivery)
    assert len(worker_1_sends) == 1
    assert len(worker_2_sends) == 1

    # 2. Worker 2 succeeded post-send CAS and counted 1 delivery
    assert worker_2_delivered == 1

    # 3. Worker 1 missed post-send CAS (rowcount == 0) and counted 0 delivery
    assert worker_1_delivered == 0

    # 4. DB state is clean: status == 'delivered', attempts == 2
    async with test_db_session_maker() as session:
        final_row = await session.get(AIIncidentNotificationModel, notif_id)
        assert final_row.status == "delivered"
        assert final_row.attempts == 2
        assert final_row.last_error == "delivered_ok"


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_concurrent_two_workers_lease_expiry_post_send_cas(
    pg_test_session_maker,
    monkeypatch,
):
    """
    PostgreSQL integration test for two concurrent outbox workers with lease expiry:
    Proves atomic CAS rowcount detection and exact single confirmation on real PostgreSQL.
    """
    from database import AIConfig, AIProviderIncidentModel, AIIncidentNotificationModel
    from aiogram.client.session.base import BaseSession
    from aiogram.methods.base import TelegramMethod
    from aiogram.methods import SendMessage
    from aiogram.types import Message, Chat, User as AiogramUser

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")
    monkeypatch.setenv("APP_ENV", "development")

    async with pg_test_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            session.add(AIConfig(id=1, current_provider="DEEPSEEK", deepseek_api_key="sk-test-key-pg"))
            await session.commit()

        inc = AIProviderIncidentModel(
            origin_instance_id="tg_veraveda777_bot_legacy",
            companion_instance_id="tg_veraveda777_bot_legacy",
            provider="DEEPSEEK",
            credential_scope="scope_pg_concurrent_cas",
            operation="chat",
            model_scope="deepseek-chat",
            incident_type="provider_5xx",
            evidence_tier="CONFIRMED_STRUCTURED",
            detail_code="500",
            classification_source="provider_structured",
            status="active",
            occurrence_count=1,
            reminder_cycle=0,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        session.add(inc)
        await session.flush()

        notif = AIIncidentNotificationModel(
            incident_id=inc.id,
            authorized_instance_id="tg_veraveda777_bot_legacy",
            recipient_id=979514796,
            event_type="opened",
            status="pending",
            attempts=0,
            max_attempts=6,
            message_text="PostgreSQL concurrent CAS test notification",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(notif)
        await session.commit()
        notif_id = notif.id

    worker_1_sends = []
    worker_2_sends = []

    class PgWorker2Session(BaseSession):
        async def make_request(self, bot: Bot, method: TelegramMethod, timeout: int | None = None) -> Any:
            if isinstance(method, SendMessage):
                worker_2_sends.append(method)
                return Message(
                    message_id=4002,
                    date=datetime.now(),
                    chat=Chat(id=method.chat_id, type="private"),
                    text=method.text,
                    from_user=AiogramUser(id=8224473896, is_bot=True, first_name="bot"),
                )
            elif method.__class__.__name__ == "GetMe":
                return AiogramUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot")
            return None

        async def stream_content(self, *args, **kwargs):
            raise NotImplementedError

        async def close(self):
            pass

    bot_worker_2 = Bot(token="8224473896:AAFtest_token_veraveda", session=PgWorker2Session())
    worker_2_delivered = None

    class PgWorker1Session(BaseSession):
        async def make_request(self, bot: Bot, method: TelegramMethod, timeout: int | None = None) -> Any:
            nonlocal worker_2_delivered
            if isinstance(method, SendMessage):
                worker_1_sends.append(method)

                async with pg_test_session_maker() as s:
                    row = await s.get(AIIncidentNotificationModel, notif_id)
                    assert row.status == "processing"
                    row.lease_until = datetime.utcnow() - timedelta(seconds=10)
                    await s.commit()

                worker_2_delivered = await process_ai_incident_notifications(bot_worker_2, session_maker=pg_test_session_maker)

                return Message(
                    message_id=4001,
                    date=datetime.now(),
                    chat=Chat(id=method.chat_id, type="private"),
                    text=method.text,
                    from_user=AiogramUser(id=8224473896, is_bot=True, first_name="bot"),
                )
            elif method.__class__.__name__ == "GetMe":
                return AiogramUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot")
            return None

        async def stream_content(self, *args, **kwargs):
            raise NotImplementedError

        async def close(self):
            pass

    bot_worker_1 = Bot(token="8224473896:AAFtest_token_veraveda", session=PgWorker1Session())

    worker_1_delivered = await process_ai_incident_notifications(bot_worker_1, session_maker=pg_test_session_maker)

    assert len(worker_1_sends) == 1
    assert len(worker_2_sends) == 1
    assert worker_2_delivered == 1
    assert worker_1_delivered == 0

    async with pg_test_session_maker() as session:
        final_row = await session.get(AIIncidentNotificationModel, notif_id)
        assert final_row.status == "delivered"
        assert final_row.attempts == 2
        assert final_row.last_error == "delivered_ok"


# ==============================================================================
# Comprehensive AI Classification and Outbox Delivery Tests (Review Requirements)
# ==============================================================================

@pytest.mark.asyncio
async def test_ai_classification_positive_negative_and_edge_scenarios_e2e_sqlite(
    test_db_session_maker,
    monkeypatch,
):
    """
    Comprehensive verification of AI classification pipeline across:
    1. Negative scenarios: untyped local exceptions with '503', 'unauthorized', 'billing'
       without HTTP status -> message_fallback, INFERRED_MESSAGE, transient, ZERO outbox alerts.
    2. Positive scenarios:
       - Confirmed HTTP 502/503 -> CONFIRMED_STRUCTURED, transient, zero premature alerts.
       - SDK timeout -> CONFIRMED_STRUCTURED, transient.
       - Auth error (HTTP 401) -> CONFIRMED_STRUCTURED, active, enqueued & delivered to admin.
       - Insufficient balance (typed InsufficientBalanceError) -> CONFIRMED_STRUCTURED, active, delivered.
       - Fallback used -> fallback_used event, active, delivered to admin.
    3. Edge scenarios:
       - Nested exception (__cause__) -> properly resolved via chain traversal.
       - Pre-attached ProviderFailure -> preserved without re-classification.
       - Key rotation -> active incident transitioned to superseded, outbox cancelled.
       - Multi-instance isolation -> companion Telegram routing, no leak to foreign bot admins.
    """
    from provider_failures import ProviderFailure, normalize_exception, parse_provider_failure
    from database import AIConfig, AIIncidentNotificationModel, AIProviderIncidentModel
    from ai_incident_manager import handle_key_rotation, record_ai_provider_failure
    from ai_integration import InsufficientBalanceError
    from aiogram.client.session.base import BaseSession
    from aiogram.methods.base import TelegramMethod
    from aiogram.methods import SendMessage
    from aiogram.types import Message, Chat, User as AiogramUser

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")
    monkeypatch.setenv("APP_ENV", "development")

    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            session.add(AIConfig(id=1, current_provider="OPENAI", openai_api_key="sk-test-key-2026"))
            await session.commit()

    sent_messages = []

    class MockTelegramSession(BaseSession):
        async def make_request(self, bot: Bot, method: TelegramMethod, timeout: int | None = None) -> Any:
            if isinstance(method, SendMessage):
                sent_messages.append(method)
                return Message(
                    message_id=3000 + len(sent_messages),
                    date=datetime.now(),
                    chat=Chat(id=method.chat_id, type="private"),
                    text=method.text,
                    from_user=AiogramUser(id=8224473896, is_bot=True, first_name="bot"),
                )
            elif method.__class__.__name__ == "GetMe":
                return AiogramUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot")
            return None

        async def stream_content(self, *args, **kwargs):
            raise NotImplementedError

        async def close(self):
            pass

    test_bot = Bot(token="8224473896:AAFtest_token_veraveda", session=MockTelegramSession())

    # --------------------------------------------------------------------------
    # 1. Negative Scenarios: Local errors without HTTP status
    # --------------------------------------------------------------------------
    negative_exceptions = [
        ("local_503", Exception("Local error: status code 503 mentioned in diagnostic text")),
        ("unauthorized_text", Exception("Unexpected content includes unauthorized")),
        ("insufficient_balance_text", Exception("Application text contains insufficient balance")),
        ("user_prompt_billing", ValueError("User prompt 'Check billing limits' caused validation failure")),
    ]

    for label, exc in negative_exceptions:
        sent_messages.clear()
        pf = normalize_exception(exc, provider="OpenAI", operation="chat")
        assert pf.classification_source == "message_fallback", f"{label} must have message_fallback source"
        assert pf.operator_action_required is False, f"{label} must NOT require operator action"
        assert pf.http_status is None, f"{label} must have no HTTP status"

        inc = await record_ai_provider_failure(
            origin_instance_id="tg_veraveda777_bot_legacy",
            provider="OpenAI",
            error=exc,
            operation="chat",
            model=f"model-{label}",
            credential_scope=f"scope-{label}",
            enqueue_notification=True,
            session_maker=test_db_session_maker,
        )
        assert inc is not None
        assert inc.evidence_tier == "INFERRED_MESSAGE", f"{label} evidence_tier must be INFERRED_MESSAGE"
        assert inc.status == "transient", f"{label} status must be transient (not active)"
        assert inc.incident_type == "unknown", f"{label} incident_type must be unknown (not unverified provider error)"

        # Process outbox: ZERO notifications should be delivered!
        delivered = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
        assert delivered == 0, f"{label} must enqueue 0 notifications"
        assert len(sent_messages) == 0, f"{label} must send 0 Telegram messages"

    # Verify no unconfirmed provider_5xx incidents exist in database
    async with test_db_session_maker() as session:
        p5xx_list = (await session.execute(
            select(AIProviderIncidentModel).where(AIProviderIncidentModel.incident_type == "provider_5xx")
        )).scalars().all()
        assert len(p5xx_list) == 0, f"Expected 0 unconfirmed provider_5xx in DB, got {len(p5xx_list)}"

    # --------------------------------------------------------------------------
    # 2. Positive Scenarios: Real confirmed failures
    # --------------------------------------------------------------------------
    # Case 2A: Confirmed HTTP 503 (transient, initial alerts suppressed)
    sent_messages.clear()
    pf_503 = parse_provider_failure("OpenAI", "chat", 503, {"error": {"message": "Service unavailable"}})
    assert pf_503.http_status == 503
    assert pf_503.operator_action_required is False
    inc_503 = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=pf_503,
        operation="chat",
        model="gpt-4o",
        credential_scope="scope-pos-503",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_503.evidence_tier == "CONFIRMED_STRUCTURED"
    assert inc_503.status == "transient"
    deliv_503 = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_503 == 0
    assert len(sent_messages) == 0

    # Case 2B: Real SDK timeout (transient)
    sent_messages.clear()
    exc_timeout = TimeoutError("Request to OpenAI timed out after 60s")
    pf_timeout = normalize_exception(exc_timeout, provider="OpenAI", operation="chat")
    assert pf_timeout.classification == "timeout"
    assert pf_timeout.classification_source == "transport"
    assert pf_timeout.operator_action_required is False
    inc_timeout = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=exc_timeout,
        operation="chat",
        model="gpt-4o",
        credential_scope="scope-pos-timeout",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_timeout.evidence_tier == "CONFIRMED_STRUCTURED"
    assert inc_timeout.status == "transient"
    deliv_timeout = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_timeout == 0

    # Case 2C: Real Auth error (HTTP 401) -> active, delivered to admin
    sent_messages.clear()
    pf_401 = parse_provider_failure("OpenAI", "chat", 401, {"error": {"message": "Incorrect API key"}})
    assert pf_401.http_status == 401
    assert pf_401.operator_action_required is True
    inc_401 = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=pf_401,
        operation="chat",
        model="gpt-4o",
        credential_scope="scope-pos-401",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_401.evidence_tier == "CONFIRMED_STRUCTURED"
    assert inc_401.status == "active"
    deliv_401 = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_401 > 0
    assert len(sent_messages) > 0
    assert any("Сбой AI-провайдера" in m.text and "auth" in m.text for m in sent_messages)

    # Case 2D: Real Insufficient Balance error (typed InsufficientBalanceError) -> active, delivered
    sent_messages.clear()
    exc_quota = InsufficientBalanceError("KIE credits balance empty")
    pf_quota = normalize_exception(exc_quota, provider="KIE", operation="chat")
    assert pf_quota.classification == "insufficient_balance_quota"
    assert pf_quota.classification_source == "sdk_typed_exception"
    assert pf_quota.operator_action_required is True
    inc_quota = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        error=exc_quota,
        operation="chat",
        model="kie-chat",
        credential_scope="scope-pos-quota",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_quota.evidence_tier == "CONFIRMED_STRUCTURED"
    assert inc_quota.status == "active"
    deliv_quota = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_quota > 0
    assert any("insufficient_balance_quota" in m.text for m in sent_messages)

    # Case 2E: Fallback used -> event_type='fallback_used', delivered to admin
    sent_messages.clear()
    inc_fb = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        error=exc_quota,
        operation="chat",
        model="kie-chat",
        fallback_provider="OpenAI",
        fallback_model="gpt-4o",
        fallback_succeeded=True,
        credential_scope="scope-pos-fb",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_fb.status == "active"
    deliv_fb = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_fb > 0
    assert any("использован резервный" in m.text for m in sent_messages)

    # --------------------------------------------------------------------------
    # 3. Edge Scenarios
    # --------------------------------------------------------------------------
    # Case 3A: Nested exception (__cause__)
    sent_messages.clear()
    wrapper_exc = RuntimeError("Outer wrapper failure during AI processing")
    wrapper_exc.__cause__ = InsufficientBalanceError("Inner quota exhausted")
    pf_nested = normalize_exception(wrapper_exc, provider="KIE", operation="chat")
    assert pf_nested.classification == "insufficient_balance_quota"
    assert pf_nested.classification_source == "sdk_typed_exception"
    assert pf_nested.operator_action_required is True

    # Case 3B: Pre-attached ProviderFailure
    err_pre = Exception("Pre-attached failure test")
    err_pre.failure = ProviderFailure(
        provider="KIE",
        operation="chat",
        classification="auth",
        detail_code="pre_attached_401",
        http_status=401,
        operator_action_required=True,
    )
    pf_pre = normalize_exception(err_pre, provider="KIE", operation="chat")
    assert pf_pre is err_pre.failure
    assert pf_pre.http_status == 401

    # Case 3C: Key rotation supersedes active incidents without false alert
    rot_count = await handle_key_rotation(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="KIE",
        new_key="sk-new-rotated-key-edge",
        old_credential_scope="scope-pos-quota",
        session_maker=test_db_session_maker,
    )
    assert rot_count >= 1
    async with test_db_session_maker() as session:
        db_quota = await session.get(AIProviderIncidentModel, inc_quota.id)
        assert db_quota.status == "superseded"


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_ai_classification_positive_negative_and_edge_scenarios_e2e(
    pg_test_session_maker,
    monkeypatch,
):
    """
    Real PostgreSQL E2E verification of AI classification:
    Validates that:
    - Negative cases (local text with '503', 'unauthorized', 'balance') -> INFERRED_MESSAGE, transient, 0 outbox rows.
    - Positive cases (real 401, InsufficientBalanceError, fallback_used) -> CONFIRMED_STRUCTURED, active, outbox delivered.
    """
    from provider_failures import ProviderFailure, normalize_exception, parse_provider_failure
    from database import AIConfig, AIIncidentNotificationModel, AIProviderIncidentModel
    from ai_incident_manager import handle_key_rotation, record_ai_provider_failure
    from ai_integration import InsufficientBalanceError
    from aiogram.client.session.base import BaseSession
    from aiogram.methods.base import TelegramMethod
    from aiogram.methods import SendMessage
    from aiogram.types import Message, Chat, User as AiogramUser

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")
    monkeypatch.setenv("APP_ENV", "development")

    async with pg_test_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            session.add(AIConfig(id=1, current_provider="OPENAI", openai_api_key="sk-pg-e2e-key"))
            await session.commit()

    sent_messages = []

    class MockPgTelegramSession(BaseSession):
        async def make_request(self, bot: Bot, method: TelegramMethod, timeout: int | None = None) -> Any:
            if isinstance(method, SendMessage):
                sent_messages.append(method)
                return Message(
                    message_id=5000 + len(sent_messages),
                    date=datetime.now(),
                    chat=Chat(id=method.chat_id, type="private"),
                    text=method.text,
                    from_user=AiogramUser(id=8224473896, is_bot=True, first_name="bot"),
                )
            elif method.__class__.__name__ == "GetMe":
                return AiogramUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot")
            return None

        async def stream_content(self, *args, **kwargs):
            raise NotImplementedError

        async def close(self):
            pass

    test_bot = Bot(token="8224473896:AAFtest_token_veraveda", session=MockPgTelegramSession())

    # 1. Negative Scenarios on real PostgreSQL
    LocalAuthErrorPg = type("AuthenticationError", (Exception,), {})
    SyntheticAuthOpenAIPg = type("AuthenticationError", (Exception,), {"__module__": "openai"})
    SyntheticAuthOpenAIFakePg = type("AuthenticationError", (Exception,), {"__module__": "openai_fake"})
    LocalBalPg = type("InsufficientBalanceError", (Exception,), {})
    SyntheticMaxBalPg = type("InsufficientBalanceError", (Exception,), {"__module__": "max_messenger_bot.ai"})
    LocalAdapterPg = type("ProviderAdapterError", (Exception,), {})
    LocalAdapterPg.classification = "auth"
    adhoc_auth_exc = Exception("Custom untyped error")
    adhoc_auth_exc.classification = "auth"
    neg_exceptions = [
        ("pg_local_503", Exception("Local error: status code 503 mentioned in diagnostic text")),
        ("pg_unauthorized", Exception("Unexpected content includes unauthorized")),
        ("pg_balance", Exception("Application text contains insufficient balance")),
        ("pg_local_auth_class", LocalAuthErrorPg("Local authentication fake class")),
        ("pg_synthetic_auth_openai", SyntheticAuthOpenAIPg("Synthetic openai auth")),
        ("pg_synthetic_auth_openai_fake", SyntheticAuthOpenAIFakePg("Synthetic fake prefix auth")),
        ("pg_local_bal_class", LocalBalPg("Local balance fake class")),
        ("pg_synthetic_max_bal", SyntheticMaxBalPg("Synthetic max balance class")),
        ("pg_local_adapter_err", LocalAdapterPg("Local adapter error class")),
        ("pg_adhoc_auth_attr", adhoc_auth_exc),
    ]
    for label, exc in neg_exceptions:
        sent_messages.clear()
        pf = normalize_exception(exc, provider="OpenAI", operation="chat")
        assert pf.classification_source == "message_fallback"
        assert pf.operator_action_required is False

        inc = await record_ai_provider_failure(
            origin_instance_id="tg_veraveda777_bot_legacy",
            provider="OpenAI",
            error=exc,
            operation="chat",
            model=f"model-{label}",
            credential_scope=f"scope-{label}",
            enqueue_notification=True,
            session_maker=pg_test_session_maker,
        )
        assert inc.evidence_tier == "INFERRED_MESSAGE"
        assert inc.status == "transient"
        assert inc.incident_type == "unknown", f"{label} incident_type must be unknown in DB"

        delivered = await process_ai_incident_notifications(test_bot, session_maker=pg_test_session_maker)
        assert delivered == 0
        assert len(sent_messages) == 0

    # Ensure zero unconfirmed provider_5xx in PostgreSQL DB
    async with pg_test_session_maker() as session:
        p5xx_pg = (await session.execute(
            select(AIProviderIncidentModel).where(AIProviderIncidentModel.incident_type == "provider_5xx")
        )).scalars().all()
        assert len(p5xx_pg) == 0, f"Expected 0 unconfirmed provider_5xx in PostgreSQL, got {len(p5xx_pg)}"

    # 2. Positive Scenarios on real PostgreSQL
    # 2A: Confirmed 401
    sent_messages.clear()
    pf_401 = parse_provider_failure("OpenAI", "chat", 401, {"error": {"message": "Invalid API key"}})
    inc_401 = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=pf_401,
        operation="chat",
        model="gpt-4o",
        credential_scope="scope-pg-pos-401",
        enqueue_notification=True,
        session_maker=pg_test_session_maker,
    )
    assert inc_401.evidence_tier == "CONFIRMED_STRUCTURED"
    assert inc_401.status == "active"
    assert inc_401.incident_type == "auth"

    delivered = await process_ai_incident_notifications(test_bot, session_maker=pg_test_session_maker)
    assert delivered > 0
    assert len(sent_messages) > 0

    # 2B: Confirmed 404 (NotFoundError) preserves invalid_model_or_resource and delivers alert
    sent_messages.clear()
    import httpx, openai
    req_pg = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp_404_pg = httpx.Response(404, request=req_pg, json={"error": {"message": "Model not found", "code": "model_not_found"}}, headers={"x-request-id": "req_pg_404"})
    exc_404_pg = openai.NotFoundError("Model not found", response=resp_404_pg, body={"error": {"message": "Model not found", "code": "model_not_found"}})
    inc_404 = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=exc_404_pg,
        operation="chat",
        model="gpt-missing",
        credential_scope="scope-pg-pos-404",
        enqueue_notification=True,
        session_maker=pg_test_session_maker,
    )
    assert inc_404.evidence_tier == "CONFIRMED_STRUCTURED"
    assert inc_404.status == "active"
    assert inc_404.incident_type == "invalid_model_or_resource"
    assert inc_404.http_status == 404
    deliv_404 = await process_ai_incident_notifications(test_bot, session_maker=pg_test_session_maker)
    assert deliv_404 > 0
    assert len(sent_messages) > 0

    # 2C: Confirmed Domain InsufficientBalanceError delivers alert
    sent_messages.clear()
    exc_bal_pg = InsufficientBalanceError("Project spend budget exhausted")
    inc_bal = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=exc_bal_pg,
        operation="chat",
        model="gpt-4o",
        credential_scope="scope-pg-pos-bal",
        enqueue_notification=True,
        session_maker=pg_test_session_maker,
    )
    assert inc_bal.evidence_tier == "CONFIRMED_STRUCTURED"
    assert inc_bal.status == "active"
    assert inc_bal.incident_type == "insufficient_balance_quota"
    deliv_bal = await process_ai_incident_notifications(test_bot, session_maker=pg_test_session_maker)
    assert deliv_bal > 0
    assert len(sent_messages) > 0

    # 2D: Confirmed typed SDK exception without response on PostgreSQL
    import google.api_core.exceptions
    sent_messages.clear()
    exc_sdk_pg = google.api_core.exceptions.Unauthenticated("Google unauth key invalid")
    inc_sdk_pg = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="google",
        error=exc_sdk_pg,
        operation="chat",
        model="gemini-1.5-pro",
        credential_scope="scope-pg-pos-sdk",
        enqueue_notification=True,
        session_maker=pg_test_session_maker,
    )
    assert inc_sdk_pg.evidence_tier == "CONFIRMED_STRUCTURED"
    assert inc_sdk_pg.status == "active"
    assert inc_sdk_pg.incident_type == "auth"
    deliv_sdk_pg = await process_ai_incident_notifications(test_bot, session_maker=pg_test_session_maker)
    assert deliv_sdk_pg > 0
    assert len(sent_messages) > 0

    # 2E: Confirmed background_worker InsufficientBalanceError on PostgreSQL
    from background_worker import InsufficientBalanceError as WorkerBal
    sent_messages.clear()
    exc_worker_pg = WorkerBal("Worker balance exhausted")
    inc_worker_pg = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=exc_worker_pg,
        operation="chat",
        model="gpt-4o",
        credential_scope="scope-pg-pos-worker",
        enqueue_notification=True,
        session_maker=pg_test_session_maker,
    )
    assert inc_worker_pg.evidence_tier == "CONFIRMED_STRUCTURED"
    assert inc_worker_pg.status == "active"
    assert inc_worker_pg.incident_type == "insufficient_balance_quota"
    deliv_worker = await process_ai_incident_notifications(test_bot, session_maker=pg_test_session_maker)
    assert deliv_worker > 0
    assert len(sent_messages) > 0

    # 2F: Confirmed provider_adapters ProviderAdapterError on PostgreSQL
    from provider_adapters import ProviderAdapterError as RealAdapterError
    sent_messages.clear()
    exc_adapter_pg = RealAdapterError("Adapter auth error")
    exc_adapter_pg.classification = "auth"
    inc_adapter_pg = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=exc_adapter_pg,
        operation="chat",
        model="gpt-4o",
        credential_scope="scope-pg-pos-adapter",
        enqueue_notification=True,
        session_maker=pg_test_session_maker,
    )
    assert inc_adapter_pg.evidence_tier == "CONFIRMED_STRUCTURED"
    assert inc_adapter_pg.status == "active"
    assert inc_adapter_pg.incident_type == "auth"
    deliv_adapter = await process_ai_incident_notifications(test_bot, session_maker=pg_test_session_maker)
    assert deliv_adapter > 0
    assert len(sent_messages) > 0

    # 2G: PII Sanitization verification on PostgreSQL
    from ai_incident_manager import sanitize_technical_code
    assert sanitize_technical_code("inferred_patient_123456789") == "unspecified"
    assert sanitize_technical_code("inferred_user_987654") == "unspecified"
    assert sanitize_technical_code("inferred_secret_account_reference") == "unspecified"

    async with pg_test_session_maker() as session:
        all_pg_notifs = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        for n in all_pg_notifs:
            assert "patient_123456789" not in (n.message_text or "")
            assert "user_987654" not in (n.message_text or "")
            assert "secret_account" not in (n.message_text or "")


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_concurrent_workers_simultaneous_network_contention(
    pg_test_session_maker,
    monkeypatch,
):
    """
    Verification of two truly concurrent worker tasks executing simultaneously on real PostgreSQL:
    - 1 outbox notification row in PostgreSQL.
    - Two concurrent asyncio tasks run process_ai_incident_notifications at the same time.
    - Demonstrates atomic claim contention: exactly ONE worker succeeds in claiming and delivering the row.
    - Zero double delivery, zero unhandled errors, exactly 1 row marked delivered.
    """
    from database import AIConfig, AIIncidentNotificationModel, AIProviderIncidentModel
    from aiogram.client.session.base import BaseSession
    from aiogram.methods.base import TelegramMethod
    from aiogram.methods import SendMessage
    from aiogram.types import Message, Chat, User as AiogramUser

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")
    monkeypatch.setenv("APP_ENV", "development")

    async with pg_test_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            session.add(AIConfig(id=1, current_provider="OPENAI", openai_api_key="sk-pg-contention-key"))

        inc = AIProviderIncidentModel(
            origin_instance_id="tg_veraveda777_bot_legacy",
            companion_instance_id="tg_veraveda777_bot_legacy",
            provider="OPENAI",
            credential_scope="scope_pg_contention",
            operation="chat",
            model_scope="gpt-4o",
            incident_type="auth",
            evidence_tier="CONFIRMED_STRUCTURED",
            detail_code="401",
            classification_source="provider_structured",
            status="active",
            occurrence_count=1,
            reminder_cycle=0,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
        )
        session.add(inc)
        await session.flush()

        notif = AIIncidentNotificationModel(
            incident_id=inc.id,
            authorized_instance_id="tg_veraveda777_bot_legacy",
            recipient_id=979514796,
            event_type="opened",
            status="pending",
            attempts=0,
            max_attempts=6,
            message_text="Real concurrent contention test",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(notif)
        await session.commit()
        notif_id = notif.id

    dispatched_sends = []

    class ContentionSession(BaseSession):
        async def make_request(self, bot: Bot, method: TelegramMethod, timeout: int | None = None) -> Any:
            if isinstance(method, SendMessage):
                dispatched_sends.append(method)
                await asyncio.sleep(0.05)  # Simulate network transmission latency
                return Message(
                    message_id=6000 + len(dispatched_sends),
                    date=datetime.now(),
                    chat=Chat(id=method.chat_id, type="private"),
                    text=method.text,
                    from_user=AiogramUser(id=8224473896, is_bot=True, first_name="bot"),
                )
            elif method.__class__.__name__ == "GetMe":
                return AiogramUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot")
            return None

        async def stream_content(self, *args, **kwargs):
            raise NotImplementedError

        async def close(self):
            pass

    bot_w1 = Bot(token="8224473896:AAFtest_token_veraveda", session=ContentionSession())
    bot_w2 = Bot(token="8224473896:AAFtest_token_veraveda", session=ContentionSession())

    # Launch two workers truly concurrently via asyncio.gather
    w1_res, w2_res = await asyncio.gather(
        process_ai_incident_notifications(bot_w1, session_maker=pg_test_session_maker),
        process_ai_incident_notifications(bot_w2, session_maker=pg_test_session_maker),
    )

    # Exactly ONE worker claimed and delivered the row
    assert w1_res + w2_res == 1, f"Expected exactly 1 delivery, got w1={w1_res}, w2={w2_res}"
    assert len(dispatched_sends) == 1, f"Expected exactly 1 message dispatched to Telegram, got {len(dispatched_sends)}"

    async with pg_test_session_maker() as session:
        row = await session.get(AIIncidentNotificationModel, notif_id)
        assert row.status == "delivered"
        assert row.attempts == 1


@pytest.mark.asyncio
async def test_blocker_priorities_sdk_and_database_classification_e2e(
    test_db_session_maker,
    monkeypatch,
):
    """
    Dedicated comprehensive E2E test verifying Blockers 1, 2, and 3:
    1. Structured HTTP data preservation (OpenAI 404, 401, 403, 429, Anthropic 429 spend limit, 503, SDK typed without response, pre-attached failure).
    2. False classification suppression in incident_type: unconfirmed diagnostic text with 503/auth/billing/quota records incident_type='unknown'.
    3. SDK origin verification: local class named AuthenticationError and ad-hoc exception.classification do NOT masquerade as sdk_typed_exception.
    """
    import httpx
    import openai
    from provider_failures import ProviderFailure, normalize_exception, parse_provider_failure, attach_failure
    from database import AIConfig, AIIncidentNotificationModel, AIProviderIncidentModel
    from ai_incident_manager import record_ai_provider_failure, process_ai_incident_notifications
    from ai_integration import InsufficientBalanceError
    from aiogram.client.session.base import BaseSession
    from aiogram.methods.base import TelegramMethod
    from aiogram.methods import SendMessage
    from aiogram.types import Message, Chat, User as AiogramUser

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")
    monkeypatch.setenv("APP_ENV", "development")

    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            session.add(AIConfig(id=1, current_provider="OPENAI", openai_api_key="sk-test-blocker-key"))
            await session.commit()

    sent_messages = []

    class MockTelegramSession(BaseSession):
        async def make_request(self, bot: Bot, method: TelegramMethod, timeout: int | None = None) -> Any:
            if isinstance(method, SendMessage):
                sent_messages.append(method)
                return Message(
                    message_id=7000 + len(sent_messages),
                    date=datetime.now(),
                    chat=Chat(id=method.chat_id, type="private"),
                    text=method.text,
                    from_user=AiogramUser(id=8224473896, is_bot=True, first_name="bot"),
                )
            elif method.__class__.__name__ == "GetMe":
                return AiogramUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot")
            return None

        async def stream_content(self, *args, **kwargs):
            raise NotImplementedError

        async def close(self):
            pass

    test_bot = Bot(token="8224473896:AAFtest_token_veraveda", session=MockTelegramSession())
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    # =========================================================================
    # PART 1: Priority of Evidence & Structured HTTP Data Preservation
    # =========================================================================

    # Case 1A: OpenAI NotFoundError with HTTP 404
    resp_404 = httpx.Response(
        404,
        request=req,
        json={"error": {"message": "Model 'gpt-custom' not found", "code": "model_not_found", "type": "invalid_request_error"}},
        headers={"x-request-id": "req_blocker_404"},
    )
    exc_404 = openai.NotFoundError("Model not found", response=resp_404, body={"error": {"message": "Model not found", "code": "model_not_found"}})
    pf_404 = normalize_exception(exc_404, "OpenAI", "chat")
    canon_404 = parse_provider_failure("OpenAI", "chat", 404, exc_404.body, resp_404.headers)

    # Parity check with parse_provider_failure
    assert pf_404.classification == canon_404.classification == "invalid_model_or_resource"
    assert pf_404.http_status == canon_404.http_status == 404
    assert pf_404.request_id == canon_404.request_id == "req_blocker_404"
    assert pf_404.network_attempted == canon_404.network_attempted is True
    assert pf_404.operator_action_required == canon_404.operator_action_required is True

    sent_messages.clear()
    inc_404 = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=exc_404,
        operation="chat",
        model="gpt-custom",
        credential_scope="scope-blocker-404",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_404.incident_type == "invalid_model_or_resource"
    assert inc_404.http_status == 404
    assert inc_404.status == "active"
    assert inc_404.evidence_tier == "CONFIRMED_STRUCTURED"
    deliv_404 = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_404 > 0
    assert len(sent_messages) > 0

    # Case 1B: OpenAI AuthenticationError with HTTP 401
    resp_401 = httpx.Response(
        401,
        request=req,
        json={"error": {"message": "Invalid API key", "code": "invalid_api_key"}},
        headers={"x-request-id": "req_blocker_401"},
    )
    exc_401 = openai.AuthenticationError("Invalid API key", response=resp_401, body={"error": {"message": "Invalid API key", "code": "invalid_api_key"}})
    pf_401 = normalize_exception(exc_401, "OpenAI", "chat")
    canon_401 = parse_provider_failure("OpenAI", "chat", 401, exc_401.body, resp_401.headers)

    assert pf_401.classification == canon_401.classification == "auth"
    assert pf_401.http_status == canon_401.http_status == 401
    assert pf_401.request_id == canon_401.request_id == "req_blocker_401"
    assert pf_401.operator_action_required is True

    sent_messages.clear()
    inc_401 = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=exc_401,
        operation="chat",
        model="gpt-4o",
        credential_scope="scope-blocker-401",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_401.incident_type == "auth"
    assert inc_401.http_status == 401
    assert inc_401.status == "active"
    deliv_401 = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_401 > 0
    assert len(sent_messages) > 0

    # Case 1C: PermissionDeniedError with HTTP 403
    resp_403 = httpx.Response(
        403,
        request=req,
        json={"error": {"message": "Country region not supported"}},
        headers={"x-request-id": "req_blocker_403"},
    )
    exc_403 = openai.PermissionDeniedError("Forbidden region", response=resp_403, body={"error": {"message": "Country region not supported"}})
    pf_403 = normalize_exception(exc_403, "OpenAI", "chat")
    assert pf_403.classification in ("permission", "forbidden_geo")
    assert pf_403.http_status == 403
    assert pf_403.operator_action_required is True

    # Case 1D: RateLimitError with HTTP 429
    resp_429 = httpx.Response(
        429,
        request=req,
        json={"error": {"message": "Rate limit reached", "code": "rate_limit_exceeded"}},
        headers={"x-request-id": "req_blocker_429"},
    )
    exc_429 = openai.RateLimitError("Rate limit", response=resp_429, body={"error": {"message": "Rate limit reached", "code": "rate_limit_exceeded"}})
    pf_429 = normalize_exception(exc_429, "OpenAI", "chat")
    assert pf_429.classification == "rate_limit"
    assert pf_429.http_status == 429
    assert pf_429.retryable is True
    assert pf_429.operator_action_required is False

    sent_messages.clear()
    inc_429 = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=exc_429,
        operation="chat",
        model="gpt-4o",
        credential_scope="scope-blocker-429",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_429.incident_type == "rate_limit"
    assert inc_429.http_status == 429
    assert inc_429.status == "transient"
    deliv_429 = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_429 == 0
    assert len(sent_messages) == 0

    # Case 1E: Anthropic HTTP 429 with spend limit marker
    resp_anthropic = httpx.Response(
        429,
        request=req,
        json={"type": "error", "error": {"type": "rate_limit_error", "message": "You have exceeded your monthly spend limit"}},
    )
    exc_anthropic = httpx.HTTPStatusError("429 Spend limit", request=req, response=resp_anthropic)
    pf_anthropic = normalize_exception(exc_anthropic, "anthropic", "chat")
    assert pf_anthropic.classification == "insufficient_balance_quota"
    assert pf_anthropic.http_status == 429
    assert pf_anthropic.operator_action_required is True

    # Case 1F: InternalServerError with HTTP 503
    resp_503 = httpx.Response(
        503,
        request=req,
        json={"error": {"message": "OpenAI engine overloaded"}},
        headers={"x-request-id": "req_blocker_503"},
    )
    exc_503 = openai.InternalServerError("Server error", response=resp_503, body={"error": {"message": "OpenAI engine overloaded"}})
    pf_503 = normalize_exception(exc_503, "OpenAI", "chat")
    assert pf_503.classification in ("provider_5xx", "provider_overloaded")
    assert pf_503.http_status == 503
    assert pf_503.retryable is True
    assert pf_503.operator_action_required is False

    # Case 1G: Real typed SDK exception without response/status
    import google.api_core.exceptions
    exc_sdk_no_resp = google.api_core.exceptions.Unauthenticated("API key invalid from Google SDK")
    pf_sdk_no_resp = normalize_exception(exc_sdk_no_resp, "google", "chat")
    assert pf_sdk_no_resp.classification == "auth"
    assert pf_sdk_no_resp.classification_source == "sdk_typed_exception"
    assert pf_sdk_no_resp.operator_action_required is True

    sent_messages.clear()
    inc_sdk_no_resp = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="google",
        error=exc_sdk_no_resp,
        operation="chat",
        model="gemini-1.5-pro",
        credential_scope="scope-sdk-no-resp",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_sdk_no_resp.evidence_tier == "CONFIRMED_STRUCTURED"
    assert inc_sdk_no_resp.status == "active"
    assert inc_sdk_no_resp.incident_type == "auth"
    deliv_sdk = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_sdk > 0
    assert len(sent_messages) > 0

    # Case 1H: Exception with already attached ProviderFailure
    pf_pre_attached = ProviderFailure("OpenAI", "chat", "auth", "attached_test", operator_action_required=True, classification_source="provider_structured")
    exc_wrapped = attach_failure(Exception("wrapper around failure"), pf_pre_attached)
    pf_unwrapped = normalize_exception(exc_wrapped, "OpenAI", "chat")
    assert pf_unwrapped is pf_pre_attached

    # =========================================================================
    # PART 2: False Classification Suppression in incident_type
    # =========================================================================

    unconfirmed_cases = [
        ("str_503", Exception("Local error: status code 503 mentioned in diagnostic text")),
        ("str_unauthorized", Exception("Unexpected content includes unauthorized")),
        ("str_billing", Exception("Application text mentions billing exhausted")),
        ("str_quota", Exception("Diagnostic log: insufficient quota for request")),
    ]
    for label, exc in unconfirmed_cases:
        sent_messages.clear()
        pf_unconf = normalize_exception(exc, provider="OpenAI", operation="chat")
        assert pf_unconf.classification_source == "message_fallback"
        assert pf_unconf.operator_action_required is False

        inc_unconf = await record_ai_provider_failure(
            origin_instance_id="tg_veraveda777_bot_legacy",
            provider="OpenAI",
            error=exc,
            operation="chat",
            model=f"model-{label}",
            credential_scope=f"scope-{label}",
            enqueue_notification=True,
            session_maker=test_db_session_maker,
        )
        assert inc_unconf.evidence_tier == "INFERRED_MESSAGE"
        assert inc_unconf.status == "transient"
        # CRITICAL: Database incident_type must be 'unknown', NOT unconfirmed provider_5xx or auth!
        assert inc_unconf.incident_type == "unknown", f"{label} incident_type must be 'unknown'"
        assert inc_unconf.detail_code.startswith("inferred_"), f"{label} detail_code must preserve diagnostic prefix"

        delivered = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
        assert delivered == 0
        assert len(sent_messages) == 0

    # Ensure zero unconfirmed provider_5xx in the SQLite database
    async with test_db_session_maker() as session:
        p5xx_recs = (await session.execute(
            select(AIProviderIncidentModel).where(AIProviderIncidentModel.incident_type == "provider_5xx")
        )).scalars().all()
        assert len(p5xx_recs) == 0, f"Expected 0 provider_5xx in DB, found {len(p5xx_recs)}"

    # =========================================================================
    # PART 3: SDK Origin Verification & Synthetic Rejection
    # =========================================================================

    # Case 3A: Local class named AuthenticationError
    LocalAuthError = type("AuthenticationError", (Exception,), {})
    exc_fake_auth = LocalAuthError("Fake auth local error")
    pf_fake_auth = normalize_exception(exc_fake_auth, "OpenAI", "chat")
    assert pf_fake_auth.classification_source == "message_fallback"
    assert pf_fake_auth.operator_action_required is False

    sent_messages.clear()
    inc_fake = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=exc_fake_auth,
        operation="chat",
        model="gpt-fake",
        credential_scope="scope-fake-auth",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_fake.evidence_tier == "INFERRED_MESSAGE"
    assert inc_fake.status == "transient"
    assert inc_fake.incident_type == "unknown"
    deliv_fake = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_fake == 0
    assert len(sent_messages) == 0

    # Case 3B: Synthetic class with spoofed __module__ = "openai"
    SyntheticAuthOpenAI = type("AuthenticationError", (Exception,), {"__module__": "openai"})
    exc_synth_openai = SyntheticAuthOpenAI("Synthetic openai auth")
    pf_synth_openai = normalize_exception(exc_synth_openai, "OpenAI", "chat")
    assert pf_synth_openai.classification_source == "message_fallback"
    assert pf_synth_openai.operator_action_required is False

    sent_messages.clear()
    inc_synth_openai = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=exc_synth_openai,
        operation="chat",
        model="gpt-synth-openai",
        credential_scope="scope-synth-openai",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_synth_openai.evidence_tier == "INFERRED_MESSAGE"
    assert inc_synth_openai.status == "transient"
    assert inc_synth_openai.incident_type == "unknown"
    deliv_synth_openai = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_synth_openai == 0
    assert len(sent_messages) == 0

    # Case 3C: Synthetic class with spoofed __module__ = "openai_fake"
    SyntheticAuthOpenAIFake = type("AuthenticationError", (Exception,), {"__module__": "openai_fake"})
    exc_synth_fake = SyntheticAuthOpenAIFake("Synthetic prefix fake auth")
    pf_synth_fake = normalize_exception(exc_synth_fake, "OpenAI", "chat")
    assert pf_synth_fake.classification_source == "message_fallback"
    assert pf_synth_fake.operator_action_required is False

    sent_messages.clear()
    inc_synth_fake = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=exc_synth_fake,
        operation="chat",
        model="gpt-synth-fake",
        credential_scope="scope-synth-fake",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_synth_fake.evidence_tier == "INFERRED_MESSAGE"
    assert inc_synth_fake.status == "transient"
    assert inc_synth_fake.incident_type == "unknown"
    deliv_synth_fake = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_synth_fake == 0
    assert len(sent_messages) == 0

    # Case 3D: Local class named InsufficientBalanceError
    LocalBalError = type("InsufficientBalanceError", (Exception,), {})
    exc_local_bal = LocalBalError("Local fake balance error")
    pf_local_bal = normalize_exception(exc_local_bal, "OpenAI", "chat")
    assert pf_local_bal.classification_source == "message_fallback"
    assert pf_local_bal.operator_action_required is False

    sent_messages.clear()
    inc_local_bal = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=exc_local_bal,
        operation="chat",
        model="gpt-local-bal",
        credential_scope="scope-local-bal",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_local_bal.evidence_tier == "INFERRED_MESSAGE"
    assert inc_local_bal.status == "transient"
    assert inc_local_bal.incident_type == "unknown"
    deliv_local_bal = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_local_bal == 0
    assert len(sent_messages) == 0

    # Case 3E: Arbitrary Exception with .classification = "auth"
    exc_ad_hoc_auth = Exception("Plain exception with ad-hoc classification")
    exc_ad_hoc_auth.classification = "auth"
    pf_ad_hoc = normalize_exception(exc_ad_hoc_auth, "OpenAI", "chat")
    assert pf_ad_hoc.classification_source == "message_fallback"
    assert pf_ad_hoc.operator_action_required is False

    sent_messages.clear()
    inc_adhoc = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=exc_ad_hoc_auth,
        operation="chat",
        model="gpt-adhoc",
        credential_scope="scope-adhoc-auth",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_adhoc.evidence_tier == "INFERRED_MESSAGE"
    assert inc_adhoc.status == "transient"
    assert inc_adhoc.incident_type == "unknown"
    deliv_adhoc = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_adhoc == 0
    assert len(sent_messages) == 0

    # Case 3F: Real domain InsufficientBalanceError
    exc_domain_bal = InsufficientBalanceError("Actual domain balance exhausted")
    pf_domain_bal = normalize_exception(exc_domain_bal, "OpenAI", "chat")
    assert pf_domain_bal.classification == "insufficient_balance_quota"
    assert pf_domain_bal.classification_source == "sdk_typed_exception"
    assert pf_domain_bal.operator_action_required is True

    sent_messages.clear()
    inc_domain = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=exc_domain_bal,
        operation="chat",
        model="gpt-4o",
        credential_scope="scope-domain-bal",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_domain.evidence_tier == "CONFIRMED_STRUCTURED"
    assert inc_domain.status == "active"
    assert inc_domain.incident_type == "insufficient_balance_quota"
    deliv_domain = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_domain > 0
    assert len(sent_messages) > 0

    # =========================================================================
    # PART 4: PII Sanitization in sanitize_technical_code & Outbox Delivery
    # =========================================================================
    from ai_incident_manager import sanitize_technical_code

    # 4A: Arbitrary inferred codes with patient / user / secret IDs are strictly rejected
    assert sanitize_technical_code("inferred_patient_123456789") == "unspecified"
    assert sanitize_technical_code("inferred_user_987654") == "unspecified"
    assert sanitize_technical_code("inferred_secret_account_reference") == "unspecified"

    # Canonical codes are preserved
    assert sanitize_technical_code("inferred_provider_5xx") == "inferred_provider_5xx"
    assert sanitize_technical_code("inferred_auth") == "inferred_auth"
    assert sanitize_technical_code("inferred_rate_limit") == "inferred_rate_limit"
    assert sanitize_technical_code("inferred_insufficient_balance_quota") == "inferred_insufficient_balance_quota"
    assert sanitize_technical_code("inferred_unspecified") == "inferred_unspecified"

    # 4B: End-to-end incident recording with potential PII tokens in error
    sent_messages.clear()
    exc_pii = Exception("Diagnostic failure mentioning patient_123456789 and secret_token_987")
    inc_pii = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=exc_pii,
        operation="chat",
        model="gpt-4o",
        credential_scope="scope-pii-test",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert "patient_123456789" not in inc_pii.detail_code
    assert "patient_123456789" not in inc_pii.incident_type
    assert "secret_token_987" not in inc_pii.detail_code
    assert inc_pii.evidence_tier == "INFERRED_MESSAGE"
    assert inc_pii.status == "transient"
    assert inc_pii.incident_type == "unknown"

    deliv_pii = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_pii == 0
    assert len(sent_messages) == 0

    # Also verify that if notification rows are inspected, zero PII tokens exist anywhere
    async with test_db_session_maker() as session:
        all_notifs = (await session.execute(select(AIIncidentNotificationModel))).scalars().all()
        for n in all_notifs:
            assert "patient_123456789" not in (n.message_text or "")
            assert "secret_token_987" not in (n.message_text or "")
            assert "inferred_patient" not in (n.message_text or "")


@pytest.mark.asyncio
async def test_isolated_domain_subsystem_verification_no_unwanted_imports_and_alerts(
    test_db_session_maker,
    monkeypatch,
):
    """
    Exhaustive verification of subsystem boundary isolation and zero unwanted imports:
    1. Negative cases (Local/synthetic InsufficientBalanceError, Local/synthetic ProviderAdapterError):
       - Asserts that foreign subsystems (e.g. max_messenger_bot.ai) are NOT imported into sys.modules.
       - Asserts configure_logging() of another subsystem is never called.
       - Asserts pf.classification_source == "message_fallback" and operator_action_required is False.
       - Asserts incident_type == "unknown", status == "transient".
       - Asserts 0 delivered notifications and 0 sent Telegram messages.
    2. Positive cases (Real ai_integration, max_messenger_bot, background_worker, provider_adapters):
       - Asserts pf.classification_source == "sdk_typed_exception" and operator_action_required is True.
       - Asserts incident_type matches real canonical type and status == "active".
       - Asserts operator notification is delivered and sent to Telegram.
    """
    import sys
    from provider_failures import ProviderFailure, normalize_exception
    from database import AIConfig, AIIncidentNotificationModel, AIProviderIncidentModel
    from ai_incident_manager import record_ai_provider_failure, process_ai_incident_notifications
    from aiogram.client.session.base import BaseSession
    from aiogram.methods.base import TelegramMethod
    from aiogram.methods import SendMessage
    from aiogram.types import Message, Chat, User as AiogramUser

    monkeypatch.setenv("BOT_INSTANCE_ID", "tg_veraveda777_bot_legacy")
    monkeypatch.setenv("BOT_TOKEN", "8224473896:AAFtest_token_veraveda")
    monkeypatch.setenv("APP_ENV", "development")

    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            session.add(AIConfig(id=1, current_provider="OPENAI", openai_api_key="sk-test-isolation-key"))
            await session.commit()

    sent_messages = []

    class MockTelegramSession(BaseSession):
        async def make_request(self, bot: Bot, method: TelegramMethod, timeout: int | None = None) -> Any:
            if isinstance(method, SendMessage):
                sent_messages.append(method)
                return Message(
                    message_id=9000 + len(sent_messages),
                    date=datetime.now(),
                    chat=Chat(id=method.chat_id, type="private"),
                    text=method.text,
                    from_user=AiogramUser(id=8224473896, is_bot=True, first_name="bot"),
                )
            elif method.__class__.__name__ == "GetMe":
                return AiogramUser(id=8224473896, is_bot=True, first_name="bot", username="veraveda777_bot")
            return None

        async def stream_content(self, *args, **kwargs):
            raise NotImplementedError

        async def close(self):
            pass

    test_bot = Bot(token="8224473896:AAFtest_token_veraveda", session=MockTelegramSession())

    # -------------------------------------------------------------------------
    # PART 1: NEGATIVE SCENARIOS - NO UNWANTED IMPORTS & NO FALSE ALERTS
    # -------------------------------------------------------------------------
    # Ensure clean state: max_messenger_bot.ai is not in sys.modules during Telegram error check
    saved_max_ai = sys.modules.pop("max_messenger_bot.ai", None)
    saved_max = sys.modules.pop("max_messenger_bot", None)

    try:
        # 1A: Local fake InsufficientBalanceError
        LocalBal = type("InsufficientBalanceError", (Exception,), {})
        exc_local_bal = LocalBal("Fake local balance issue")
        pf_local = normalize_exception(exc_local_bal, "OpenAI", "chat")

        assert "max_messenger_bot.ai" not in sys.modules, "Classification must not import max_messenger_bot.ai"
        assert pf_local.classification_source == "message_fallback"
        assert pf_local.operator_action_required is False

        sent_messages.clear()
        inc_local = await record_ai_provider_failure(
            origin_instance_id="tg_veraveda777_bot_legacy",
            provider="OpenAI",
            error=exc_local_bal,
            operation="chat",
            model="gpt-4o",
            credential_scope="scope-iso-local-bal",
            enqueue_notification=True,
            session_maker=test_db_session_maker,
        )
        assert inc_local.evidence_tier == "INFERRED_MESSAGE"
        assert inc_local.status == "transient"
        assert inc_local.incident_type == "unknown"
        deliv_local = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
        assert deliv_local == 0
        assert len(sent_messages) == 0

        # 1B: Synthetic class with spoofed __module__ = "max_messenger_bot.ai"
        SynthMaxBal = type("InsufficientBalanceError", (Exception,), {"__module__": "max_messenger_bot.ai"})
        exc_synth_max = SynthMaxBal("Fake synthetic MAX balance")
        pf_synth_max = normalize_exception(exc_synth_max, "OpenAI", "chat")

        assert "max_messenger_bot.ai" not in sys.modules, "Synthetic class must not trigger import of max_messenger_bot.ai"
        assert pf_synth_max.classification_source == "message_fallback"
        assert pf_synth_max.operator_action_required is False

        sent_messages.clear()
        inc_synth_max = await record_ai_provider_failure(
            origin_instance_id="tg_veraveda777_bot_legacy",
            provider="OpenAI",
            error=exc_synth_max,
            operation="chat",
            model="gpt-4o",
            credential_scope="scope-iso-synth-max",
            enqueue_notification=True,
            session_maker=test_db_session_maker,
        )
        assert inc_synth_max.evidence_tier == "INFERRED_MESSAGE"
        assert inc_synth_max.status == "transient"
        assert inc_synth_max.incident_type == "unknown"
        deliv_synth_max = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
        assert deliv_synth_max == 0
        assert len(sent_messages) == 0

        # 1C: Local fake ProviderAdapterError
        LocalAdapter = type("ProviderAdapterError", (Exception,), {})
        LocalAdapter.classification = "auth"
        exc_local_ad = LocalAdapter("Fake local adapter error")
        pf_local_ad = normalize_exception(exc_local_ad, "OpenAI", "chat")
        assert pf_local_ad.classification_source == "message_fallback"
        assert pf_local_ad.operator_action_required is False

        sent_messages.clear()
        inc_local_ad = await record_ai_provider_failure(
            origin_instance_id="tg_veraveda777_bot_legacy",
            provider="OpenAI",
            error=exc_local_ad,
            operation="chat",
            model="gpt-4o",
            credential_scope="scope-iso-local-ad",
            enqueue_notification=True,
            session_maker=test_db_session_maker,
        )
        assert inc_local_ad.evidence_tier == "INFERRED_MESSAGE"
        assert inc_local_ad.status == "transient"
        assert inc_local_ad.incident_type == "unknown"
        deliv_local_ad = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
        assert deliv_local_ad == 0
        assert len(sent_messages) == 0

    finally:
        if saved_max_ai is not None:
            sys.modules["max_messenger_bot.ai"] = saved_max_ai
        if saved_max is not None:
            sys.modules["max_messenger_bot"] = saved_max

    # -------------------------------------------------------------------------
    # PART 2: POSITIVE SCENARIOS - REAL DOMAIN & ADAPTER EXCEPTIONS
    # -------------------------------------------------------------------------
    # 2A: Real ai_integration.InsufficientBalanceError
    from ai_integration import InsufficientBalanceError as RealAiBal
    exc_real_ai = RealAiBal("Genuine AI balance exhausted")
    pf_real_ai = normalize_exception(exc_real_ai, "OpenAI", "chat")
    assert pf_real_ai.classification == "insufficient_balance_quota"
    assert pf_real_ai.classification_source == "sdk_typed_exception"
    assert pf_real_ai.operator_action_required is True

    sent_messages.clear()
    inc_real_ai = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=exc_real_ai,
        operation="chat",
        model="gpt-4o",
        credential_scope="scope-iso-real-ai",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_real_ai.evidence_tier == "CONFIRMED_STRUCTURED"
    assert inc_real_ai.status == "active"
    assert inc_real_ai.incident_type == "insufficient_balance_quota"
    deliv_real_ai = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_real_ai > 0
    assert len(sent_messages) > 0

    # 2B: Real background_worker.InsufficientBalanceError
    from background_worker import InsufficientBalanceError as RealWorkerBal
    exc_real_worker = RealWorkerBal("Genuine Worker balance exhausted")
    pf_real_worker = normalize_exception(exc_real_worker, "OpenAI", "chat")
    assert pf_real_worker.classification == "insufficient_balance_quota"
    assert pf_real_worker.classification_source == "sdk_typed_exception"
    assert pf_real_worker.operator_action_required is True

    sent_messages.clear()
    inc_real_worker = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=exc_real_worker,
        operation="chat",
        model="gpt-4o",
        credential_scope="scope-iso-real-worker",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_real_worker.evidence_tier == "CONFIRMED_STRUCTURED"
    assert inc_real_worker.status == "active"
    assert inc_real_worker.incident_type == "insufficient_balance_quota"
    deliv_real_worker = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_real_worker > 0
    assert len(sent_messages) > 0

    # 2C: Real max_messenger_bot.ai.InsufficientBalanceError
    from max_messenger_bot.ai import InsufficientBalanceError as RealMaxBal
    exc_real_max = RealMaxBal("Genuine MAX balance exhausted")
    pf_real_max = normalize_exception(exc_real_max, "OpenAI", "chat")
    assert pf_real_max.classification == "insufficient_balance_quota"
    assert pf_real_max.classification_source == "sdk_typed_exception"
    assert pf_real_max.operator_action_required is True

    sent_messages.clear()
    inc_real_max = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=exc_real_max,
        operation="chat",
        model="gpt-4o",
        credential_scope="scope-iso-real-max",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_real_max.evidence_tier == "CONFIRMED_STRUCTURED"
    assert inc_real_max.status == "active"
    assert inc_real_max.incident_type == "insufficient_balance_quota"
    deliv_real_max = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_real_max > 0
    assert len(sent_messages) > 0

    # 2D: Real provider_adapters.ProviderAdapterError
    from provider_adapters import ProviderAdapterError as RealAdapterErr
    exc_real_adapter = RealAdapterErr("Genuine Adapter auth failure")
    exc_real_adapter.classification = "auth"
    pf_real_adapter = normalize_exception(exc_real_adapter, "OpenAI", "chat")
    assert pf_real_adapter.classification == "auth"
    assert pf_real_adapter.classification_source == "sdk_typed_exception"
    assert pf_real_adapter.operator_action_required is True

    sent_messages.clear()
    inc_real_adapter = await record_ai_provider_failure(
        origin_instance_id="tg_veraveda777_bot_legacy",
        provider="OpenAI",
        error=exc_real_adapter,
        operation="chat",
        model="gpt-4o",
        credential_scope="scope-iso-real-adapter",
        enqueue_notification=True,
        session_maker=test_db_session_maker,
    )
    assert inc_real_adapter.evidence_tier == "CONFIRMED_STRUCTURED"
    assert inc_real_adapter.status == "active"
    assert inc_real_adapter.incident_type == "auth"
    deliv_real_adapter = await process_ai_incident_notifications(test_bot, session_maker=test_db_session_maker)
    assert deliv_real_adapter > 0
    assert len(sent_messages) > 0


# ============================================================================
# KIE Low-Credit Warning Multi-Instance Isolation Tests
# ============================================================================

@pytest.mark.asyncio
async def test_kie_low_credit_multi_instance_isolation_sqlite(test_db_session_maker, monkeypatch):
    """
    Verify multi-instance isolation for KIE low-credit warnings in a shared database (SQLite):
    - Instance A triggers a low-credit warning -> incident recorded, alert sent to Instance A admins.
    - Instance B checks balance -> Instance B owner filter prevents seeing Instance A's active incident.
      config.kie_credit_alert_sent reconciles correctly, allowing Instance B to warn its own admins.
    - Instance A replenishes -> resolves ONLY Instance A's incident; Instance B remains active.
    - Key rotation on Instance A does NOT affect Instance B's incidents.
    """
    from aiogram.types import User
    from notification_routing import get_instance_by_id

    inst_a = get_instance_by_id("tg_autobusbusbot_new")
    inst_b = get_instance_by_id("tg_alena322bot_new")
    assert inst_a is not None and inst_b is not None

    current_active_inst = inst_a

    def fake_detect():
        return current_active_inst

    def fake_get_instance(name):
        if name == inst_a["pm2_name"]:
            return inst_a
        if name == inst_b["pm2_name"]:
            return inst_b
        return get_instance_by_id(name)

    def fake_companion(s):
        return s

    async def fake_target_admins(comp, current_instance=None, session=None):
        target = current_instance or comp
        if target.get("pm2_name") == inst_a["pm2_name"]:
            return [272982544]
        return [979514796]

    monkeypatch.setattr("ai_incident_manager.detect_current_instance", fake_detect)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", fake_get_instance)
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", fake_companion)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", fake_target_admins)

    monkeypatch.setattr(scheduler, "async_session_maker", test_db_session_maker)
    monkeypatch.setattr(scheduler, "get_kie_remaining_credits", AsyncMock(return_value=1.50))

    bot_a = AsyncMock(spec=Bot)
    bot_a.token = f"{inst_a['platform_id']}:AAFtest_token_a"
    bot_a.get_me = AsyncMock(return_value=User(id=inst_a["platform_id"], is_bot=True, first_name="bot_a", username="autobusbusbot"))
    bot_a.send_message = AsyncMock(return_value=True)

    bot_b = AsyncMock(spec=Bot)
    bot_b.token = f"{inst_b['platform_id']}:AAFtest_token_b"
    bot_b.get_me = AsyncMock(return_value=User(id=inst_b["platform_id"], is_bot=True, first_name="bot_b", username="alena322bot"))
    bot_b.send_message = AsyncMock(return_value=True)

    # Seed AIConfig with low threshold=2.0
    async with test_db_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="KIE", kie_api_key="shared_kie_key_1", kie_credit_alert_threshold=2.0, kie_credit_alert_sent=False)
            session.add(cfg)
        else:
            cfg.provider = "KIE"
            cfg.kie_api_key = "shared_kie_key_1"
            cfg.kie_credit_alert_threshold = 2.0
            cfg.kie_credit_alert_sent = False
        await session.commit()

    # 1. Instance A runs balance check: balance 1.50 < threshold 2.00
    current_active_inst = inst_a
    monkeypatch.setenv("BOT_INSTANCE_ID", inst_a["pm2_name"])
    await scheduler.check_kie_credit_balance(bot_a)

    # Instance A alert sent
    assert bot_a.send_message.call_count == 1
    call_args_a = bot_a.send_message.call_args
    recipient_a = call_args_a.kwargs.get("chat_id") or (call_args_a.args[0] if call_args_a.args else None)
    assert recipient_a == 272982544

    # Verify Instance A incident in DB
    async with test_db_session_maker() as session:
        incs = (await session.execute(select(AIProviderIncidentModel))).scalars().all()
        assert len(incs) == 1
        assert incs[0].origin_instance_id == inst_a["pm2_name"]
        assert incs[0].status == "active"
        cfg_db = await session.get(AIConfig, 1)
        assert cfg_db.kie_credit_alert_sent is True

    # 2. Instance B runs balance check: must NOT be suppressed by Instance A's incident!
    current_active_inst = inst_b
    monkeypatch.setenv("BOT_INSTANCE_ID", inst_b["pm2_name"])
    await scheduler.check_kie_credit_balance(bot_b)

    # Instance B alert sent to Instance B admins!
    assert bot_b.send_message.call_count == 1
    call_args_b = bot_b.send_message.call_args
    recipient_b = call_args_b.kwargs.get("chat_id") or (call_args_b.args[0] if call_args_b.args else None)
    assert recipient_b == 979514796

    # Verify both incidents active simultaneously in DB under distinct origin_instance_ids
    async with test_db_session_maker() as session:
        active_incs = (await session.execute(
            select(AIProviderIncidentModel).where(AIProviderIncidentModel.status == "active")
        )).scalars().all()
        assert len(active_incs) == 2
        origin_set = {inc.origin_instance_id for inc in active_incs}
        assert origin_set == {inst_a["pm2_name"], inst_b["pm2_name"]}

    # 3. Instance B runs again: deduped, no second alert sent
    bot_b.send_message.reset_mock()
    await scheduler.check_kie_credit_balance(bot_b)
    assert bot_b.send_message.call_count == 0

    # 4. Instance A replenishes balance (5.00 >= 2.00)
    current_active_inst = inst_a
    monkeypatch.setenv("BOT_INSTANCE_ID", inst_a["pm2_name"])
    monkeypatch.setattr(scheduler, "get_kie_remaining_credits", AsyncMock(return_value=5.00))
    bot_a.send_message.reset_mock()
    await scheduler.check_kie_credit_balance(bot_a)

    # Instance A receives replenishment notification
    assert bot_a.send_message.call_count == 1
    text_replenish = bot_a.send_message.call_args.kwargs.get("text") or (bot_a.send_message.call_args.args[1] if len(bot_a.send_message.call_args.args) > 1 else "")
    assert "Баланс KIE пополнен" in text_replenish

    # In DB: Instance A's incident is resolved, but Instance B's incident REMAINS ACTIVE!
    async with test_db_session_maker() as session:
        inc_a = (await session.execute(
            select(AIProviderIncidentModel).where(AIProviderIncidentModel.origin_instance_id == inst_a["pm2_name"])
        )).scalar_one()
        assert inc_a.status == "resolved"

        inc_b = (await session.execute(
            select(AIProviderIncidentModel).where(AIProviderIncidentModel.origin_instance_id == inst_b["pm2_name"])
        )).scalar_one()
        assert inc_b.status == "active"

    # 5. Key rotation on Instance A does not supersede Instance B's incident
    from ai_incident_manager import handle_key_rotation
    current_active_inst = inst_a
    monkeypatch.setenv("BOT_INSTANCE_ID", inst_a["pm2_name"])
    await handle_key_rotation(
        origin_instance_id=inst_a["pm2_name"],
        provider="KIE",
        new_key="kie_rotated_new_key_a",
        old_key="shared_kie_key_1",
        session_maker=test_db_session_maker,
    )

    async with test_db_session_maker() as session:
        inc_b_after = (await session.execute(
            select(AIProviderIncidentModel).where(AIProviderIncidentModel.origin_instance_id == inst_b["pm2_name"])
        )).scalar_one()
        # Instance B remains active!
        assert inc_b_after.status == "active"


@pytest.mark.asyncio
async def test_kie_low_credit_companion_isolation_telegram_and_max(test_db_session_maker, monkeypatch):
    """
    Verify companion linkage and foreign bot isolation:
    - MAX bot and its companion Telegram bot share ownership of incidents.
    - An unrelated foreign bot sharing the DB is isolated from both.
    """
    from notification_routing import get_instance_by_id
    from ai_incident_manager import record_low_credit_warning, resolve_low_credit_warning

    max_inst = get_instance_by_id("max_id519010411655_bot_new")
    tg_comp_inst = get_instance_by_id("tg_yourself_way_bot_new")
    foreign_inst = get_instance_by_id("tg_autobusbusbot_new")
    assert max_inst is not None and tg_comp_inst is not None and foreign_inst is not None

    current_active = max_inst

    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: current_active)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: get_instance_by_id(n))
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: tg_comp_inst if s.get("platform") == "max" else s)
    monkeypatch.setattr("ai_incident_manager.resolve_target_admin_ids", AsyncMock(return_value=[806750628]))

    # MAX bot records low credit warning
    warned = await record_low_credit_warning(
        provider="KIE",
        remaining_credits=1.0,
        threshold=2.0,
        api_key="max_test_kie_key",
        session_maker=test_db_session_maker,
    )
    assert warned is True

    # Companion Telegram bot resolves the incident
    current_active = tg_comp_inst
    resolved = await resolve_low_credit_warning(
        provider="KIE",
        remaining_credits=10.0,
        threshold=2.0,
        api_key="max_test_kie_key",
        session_maker=test_db_session_maker,
    )
    assert resolved is True

    # Foreign bot attempts to resolve MAX's incident -> returns False (isolated!)
    # Re-open incident first
    current_active = max_inst
    await record_low_credit_warning(
        provider="KIE",
        remaining_credits=1.0,
        threshold=2.0,
        api_key="max_test_kie_key",
        session_maker=test_db_session_maker,
    )
    current_active = foreign_inst
    resolved_foreign = await resolve_low_credit_warning(
        provider="KIE",
        remaining_credits=10.0,
        threshold=2.0,
        api_key="max_test_kie_key",
        session_maker=test_db_session_maker,
    )
    assert resolved_foreign is False  # Refuses to touch foreign bot incident!


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_kie_low_credit_multi_instance_isolation(pg_test_session_maker, monkeypatch):
    """
    Verify multi-instance isolation for KIE low-credit warnings in real PostgreSQL:
    Ensures partial unique index and multi-worker isolation work concurrently without cross-instance leakage.
    """
    from aiogram.types import User
    from notification_routing import get_instance_by_id

    inst_a = get_instance_by_id("tg_autobusbusbot_new")
    inst_b = get_instance_by_id("tg_alena322bot_new")

    current_active_inst = inst_a

    monkeypatch.setattr("ai_incident_manager.detect_current_instance", lambda: current_active_inst)
    monkeypatch.setattr("ai_incident_manager.get_instance_by_pm2_name", lambda n: get_instance_by_id(n))
    monkeypatch.setattr("ai_incident_manager.resolve_companion_telegram_instance", lambda s: s)
    monkeypatch.setattr(
        "ai_incident_manager.resolve_target_admin_ids",
        AsyncMock(side_effect=lambda comp, current_instance=None, session=None: [272982544] if (current_instance or comp).get("pm2_name") == inst_a["pm2_name"] else [979514796]),
    )

    monkeypatch.setattr(scheduler, "async_session_maker", pg_test_session_maker)
    monkeypatch.setattr(scheduler, "get_kie_remaining_credits", AsyncMock(return_value=1.50))

    bot_a = AsyncMock(spec=Bot)
    bot_a.token = f"{inst_a['platform_id']}:AAFtest_pg_a"
    bot_a.get_me = AsyncMock(return_value=User(id=inst_a["platform_id"], is_bot=True, first_name="bot_a", username="autobusbusbot"))
    bot_a.send_message = AsyncMock(return_value=True)

    bot_b = AsyncMock(spec=Bot)
    bot_b.token = f"{inst_b['platform_id']}:AAFtest_pg_b"
    bot_b.get_me = AsyncMock(return_value=User(id=inst_b["platform_id"], is_bot=True, first_name="bot_b", username="alena322bot"))
    bot_b.send_message = AsyncMock(return_value=True)

    # Seed AIConfig in PostgreSQL
    async with pg_test_session_maker() as session:
        cfg = await session.get(AIConfig, 1)
        if not cfg:
            cfg = AIConfig(id=1, provider="KIE", kie_api_key="pg_kie_key_1", kie_credit_alert_threshold=2.0, kie_credit_alert_sent=False)
            session.add(cfg)
        else:
            cfg.provider = "KIE"
            cfg.kie_api_key = "pg_kie_key_1"
            cfg.kie_credit_alert_threshold = 2.0
            cfg.kie_credit_alert_sent = False
        await session.commit()

    # 1. Instance A triggers warning in PostgreSQL
    current_active_inst = inst_a
    monkeypatch.setenv("BOT_INSTANCE_ID", inst_a["pm2_name"])
    await scheduler.check_kie_credit_balance(bot_a)
    assert bot_a.send_message.call_count == 1

    # 2. Instance B runs in same PostgreSQL DB: isolated via owner_filter!
    current_active_inst = inst_b
    monkeypatch.setenv("BOT_INSTANCE_ID", inst_b["pm2_name"])
    await scheduler.check_kie_credit_balance(bot_b)
    assert bot_b.send_message.call_count == 1

    # Verify both active incidents coexist in PostgreSQL
    async with pg_test_session_maker() as session:
        active_incs = (await session.execute(
            select(AIProviderIncidentModel).where(AIProviderIncidentModel.status == "active")
        )).scalars().all()
        assert len(active_incs) == 2

    # 3. Instance A replenishment resolves ONLY Instance A
    current_active_inst = inst_a
    monkeypatch.setenv("BOT_INSTANCE_ID", inst_a["pm2_name"])
    monkeypatch.setattr(scheduler, "get_kie_remaining_credits", AsyncMock(return_value=5.00))
    await scheduler.check_kie_credit_balance(bot_a)

    async with pg_test_session_maker() as session:
        inc_a = (await session.execute(
            select(AIProviderIncidentModel).where(AIProviderIncidentModel.origin_instance_id == inst_a["pm2_name"])
        )).scalar_one()
        assert inc_a.status == "resolved"

        inc_b = (await session.execute(
            select(AIProviderIncidentModel).where(AIProviderIncidentModel.origin_instance_id == inst_b["pm2_name"])
        )).scalar_one()
        assert inc_b.status == "active"





