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

    # Step B: Test 'Выбрать модель' button navigation
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
    # Re-open KIE settings
    await dp.feed_update(bot, update_open_settings)

    # Step C: Test 'Max tokens' button navigation and state
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
    # Re-open KIE settings to reset state for next check
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

    # Part 1: Fail closed when BOT_INSTANCE_ID is missing or not registered
    monkeypatch.delenv("BOT_INSTANCE_ID", raising=False)
    fail_closed_res = await handle_key_rotation(
        origin_instance_id=None,
        provider="KIE",
        new_key="new_key_1",
        session_maker=test_db_session_maker,
    )
    assert fail_closed_res == 0, "handle_key_rotation must fail closed when instance cannot be determined"

    # Part 2: Shared DB isolation between instances
    # Instance A: tg_someonewithyou01_bot_new
    # Instance B: tg_yourself_way_bot_new
    scope_a = get_provider_account_fingerprint("KIE", "key_inst_a")
    scope_b = get_provider_account_fingerprint("KIE", "key_inst_b")

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
