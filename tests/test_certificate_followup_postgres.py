"""
Real PostgreSQL integration tests for Certificate Followup Scenario invariants:
1. Concurrent certificate issuance: strictly ONE document on the wire and ONE receipt in PostgreSQL.
2. Concurrent button clicks: strictly ONE transition and ONE response message.
3. Selection transaction failure: full atomic rollback, followed by clean repeat click.
4. <DATA> + completed: single atomic transaction under user transaction lock.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
import json
import os
from unittest.mock import AsyncMock

from aiogram import Bot, Dispatcher
from aiogram.types import (
    Update,
    CallbackQuery,
    User as TelegramUser,
    Chat,
    Message,
)
from aiogram.methods import SendDocument, SendMessage, AnswerCallbackQuery
import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import database
from database import (
    Base,
    User,
    Topic,
    MediaCollection,
    MediaLibrary,
    topic_collection_association,
    main_dialogue_collection_association,
    media_collection_items,
    AutomationConversationState,
    AutomationStepTransition,
    AutomationEvent,
    AIConfig,
)
from max_messenger_bot.storage import StorageBase
import handlers
import ai_integration
import automation_admin
import automation_events
from certificate_followup_service import (
    CertificateFollowupConfig,
    CertificateIssuanceReceipt,
    get_or_create_followup_config,
    get_or_create_receipt,
    complete_followup_delivery,
    handle_cert_meet_click,
    make_issuance_key,
)
import certificate_followup_service
from max_messenger_bot.services import common
from tests.test_event_chains_db import _get_safe_postgres_engine
from tests.test_certificate_followup_telegram import ValidatingCertTelegramTransport
from tests.test_media_templates import seed_template, VALUES


def _patch_postgres_sessions(monkeypatch, sessions, pg_engine):
    monkeypatch.setattr(database, "engine", pg_engine)
    for mod in (database, handlers, ai_integration, automation_admin, automation_events, common):
        if hasattr(mod, "async_session_maker"):
            monkeypatch.setattr(mod, "async_session_maker", sessions)


async def _setup_postgres_clean_schema(pg_engine):
    """Safely drops and re-creates clean schema on validated disposable PostgreSQL database."""
    async with pg_engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE;"))
        await conn.execute(text("CREATE SCHEMA public;"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(StorageBase.metadata.create_all)


async def _seed_postgres_fixtures(session_maker, user_id=7701):
    """Seeds baseline Topic, MediaCollection, AIConfig, MediaLibrary, and User."""
    async with session_maker() as session:
        session.add_all([
            Topic(id=1, name="Default Topic"),
            MediaCollection(id=1, name="Default Collection"),
            AIConfig(id=1, allow_image_edit=True),
            MediaLibrary(id=10, file_id="telegram-cert-10", file_name="cert_10", media_type="photo"),
            MediaLibrary(id=20, file_id="telegram-cert-20", file_name="cert_20", media_type="photo"),
            MediaLibrary(id=30, file_id="telegram-cert-30", file_name="cert_30", media_type="photo"),
            MediaLibrary(id=40, file_id="max-cert-40", file_name="cert_40", media_type="photo", source_platform="max"),
            User(
                id=user_id,
                name="PGAlice",
                first_name="PGAlice",
                gender="female",
                age="30",
                accepted_disclaimer=True,
                current_dialogue_id=1,
                current_topic_id=1,
            ),
        ])
        await session.flush()
        await session.execute(topic_collection_association.insert().values(topic_id=1, collection_id=1))
        await session.execute(main_dialogue_collection_association.insert().values(collection_id=1))
        await session.execute(media_collection_items.insert(), [
            {"collection_id": 1, "media_id": 10},
            {"collection_id": 1, "media_id": 20},
            {"collection_id": 1, "media_id": 30},
            {"collection_id": 1, "media_id": 40},
        ])
        await session.commit()


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_concurrent_certificate_issuance_single_document(monkeypatch):
    """
    Invariant 1 on real PostgreSQL:
    Concurrent certificate issuance requests for the same recipient.
    Atomic CAS in mark_receipt_sending_cert ensures strictly ONE SendDocument on the wire,
    and strictly ONE completed receipt in PostgreSQL.
    """
    pg_engine, skip_reason = await _get_safe_postgres_engine()
    if pg_engine is None:
        pytest.skip(skip_reason)

    try:
        await _setup_postgres_clean_schema(pg_engine)
        sessions = async_sessionmaker(pg_engine, expire_on_commit=False, class_=AsyncSession)
        _patch_postgres_sessions(monkeypatch, sessions, pg_engine)

        user_id = 7701
        await _seed_postgres_fixtures(sessions, user_id=user_id)
        source, config = await seed_template(sessions)

        async with sessions() as session:
            cfg = await get_or_create_followup_config(session, 3, topic_id=1)
            cfg.is_active = True
            cfg.caption_template = "🎓 Поздравляем, <b>{name}</b>! Ваш сертификат готов."
            cfg.invitation_text = "Приглашаю вас на консультацию."
            cfg.target_stage = "OFFER_MEETING"
            await session.commit()

        transport = ValidatingCertTelegramTransport()
        async def stream(*args, **kwargs): yield source
        monkeypatch.setattr(transport, "stream_content", stream)
        bot = Bot("123456:TEST_BOT_TOKEN", session=transport)

        values = dict(VALUES, recipient="PGAlice")
        ai_response = f"[RENDER_IMG_MEDIA: panda_certificate | {json.dumps(values, ensure_ascii=False)}]"

        # Run two concurrent delivery requests on real PostgreSQL
        res1, res2 = await asyncio.gather(
            handlers._deliver_media_template(bot, user_id, ai_response, dialogue_id=1, topic_id=1),
            handlers._deliver_media_template(bot, user_id, ai_response, dialogue_id=1, topic_id=1),
        )

        # On the wire: strictly ONE SendDocument method call
        docs = [c for c in transport.calls if isinstance(c, SendDocument)]
        assert len(docs) == 1, f"Expected strictly 1 SendDocument on PostgreSQL, got {len(docs)}"

        # On the wire: strictly ONE invitation SendMessage
        invites = [c for c in transport.calls if isinstance(c, SendMessage)]
        assert len(invites) == 1, f"Expected strictly 1 invitation SendMessage on PostgreSQL, got {len(invites)}"

        # In PostgreSQL: strictly ONE receipt, completed
        async with sessions() as session:
            receipts = (await session.scalars(
                select(CertificateIssuanceReceipt).where(CertificateIssuanceReceipt.user_id == user_id)
            )).all()
            assert len(receipts) == 1, f"Expected strictly 1 receipt in PostgreSQL, got {len(receipts)}"
            assert receipts[0].status == "completed"
            assert receipts[0].certificate_message_id is not None
            assert receipts[0].invitation_message_id is not None

            conv = await session.scalar(
                select(AutomationConversationState).where(AutomationConversationState.user_id == user_id)
            )
            assert conv is not None
            assert conv.current_step == "OFFER_MEETING"
    finally:
        await pg_engine.dispose()


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_concurrent_button_clicks_single_transition(monkeypatch):
    """
    Invariant 2 on real PostgreSQL:
    Concurrent button clicks by user on the invitation message.
    Atomic CAS in handle_cert_meet_click ensures strictly ONE transition is recorded
    in AutomationStepTransition, and the loser receives a repeat selection alert.
    """
    pg_engine, skip_reason = await _get_safe_postgres_engine()
    if pg_engine is None:
        pytest.skip(skip_reason)

    try:
        await _setup_postgres_clean_schema(pg_engine)
        sessions = async_sessionmaker(pg_engine, expire_on_commit=False, class_=AsyncSession)
        _patch_postgres_sessions(monkeypatch, sessions, pg_engine)

        user_id = 7702
        await _seed_postgres_fixtures(sessions, user_id=user_id)

        async with sessions() as session:
            cfg = await get_or_create_followup_config(session, 10, topic_id=1)
            cfg.is_active = True
            cfg.target_stage = "OFFER_MEETING"
            cfg.buttons_config_json = json.dumps([
                {"id": "accept", "text": "Да, записаться", "target_step": "MEETING_BOOKING", "response_text": "Отлично! Записываю."},
                {"id": "later", "text": "Позже", "target_step": "CERT_COMPLETED", "response_text": "Хорошо, пишите позже."},
            ], ensure_ascii=False)
            receipt, _ = await get_or_create_receipt(session, user_id, 1, 1, "key_pg_click_race", config_id=cfg.id)
            receipt.status = "completed"
            receipt.invitation_message_id = "555"
            await session.commit()
            config_id = cfg.id
            receipt_id = receipt.id

        transport = ValidatingCertTelegramTransport()
        bot = Bot("123456:TEST_BOT_TOKEN", session=transport)
        handlers.router._parent_router = None
        dispatcher = Dispatcher()
        dispatcher.include_router(handlers.router)

        update_accept = Update(
            update_id=30001,
            callback_query=CallbackQuery(
                id="cb_acc",
                from_user=TelegramUser(id=user_id, is_bot=False, first_name="PGAlice"),
                chat_instance="ci_pg",
                data=f"cert_meet:accept:{config_id}:{receipt_id}",
                message=Message(message_id=555, date=datetime.now(timezone.utc), chat=Chat(id=user_id, type="private"), text="invite"),
            ),
        )
        update_later = Update(
            update_id=30002,
            callback_query=CallbackQuery(
                id="cb_lat",
                from_user=TelegramUser(id=user_id, is_bot=False, first_name="PGAlice"),
                chat_instance="ci_pg",
                data=f"cert_meet:later:{config_id}:{receipt_id}",
                message=Message(message_id=555, date=datetime.now(timezone.utc), chat=Chat(id=user_id, type="private"), text="invite"),
            ),
        )

        # Execute concurrent button clicks through real aiogram Dispatcher
        await asyncio.gather(
            dispatcher.feed_update(bot, update_accept),
            dispatcher.feed_update(bot, update_later),
        )

        # Wire: strictly ONE response message to the user
        sent_msgs = [c for c in transport.calls if isinstance(c, SendMessage)]
        assert len(sent_msgs) == 1, f"Expected strictly 1 response SendMessage on wire, got {len(sent_msgs)}"

        # Wire: two callback answers: one normal, one with repeat alert
        answers = [c for c in transport.calls if isinstance(c, AnswerCallbackQuery)]
        assert len(answers) == 2, f"Expected 2 callback answers, got {len(answers)}"
        alert_answers = [a for a in answers if a.show_alert is True and "Повторный выбор не требуется" in (a.text or "")]
        assert len(alert_answers) == 1, "Exactly one click must receive idempotency alert"

        # In PostgreSQL: strictly ONE transition recorded
        async with sessions() as session:
            transitions = (await session.scalars(
                select(AutomationStepTransition).where(AutomationStepTransition.user_id == user_id)
            )).all()
            assert len(transitions) == 1, f"Expected strictly 1 transition in PostgreSQL, got {len(transitions)}"

            r = await session.get(CertificateIssuanceReceipt, receipt_id)
            assert r.selected_button_id in ("accept", "later")
            assert r.selected_at is not None

            conv = await session.scalar(
                select(AutomationConversationState).where(AutomationConversationState.user_id == user_id)
            )
            assert conv.current_step in ("MEETING_BOOKING", "CERT_COMPLETED")
            assert conv.current_step == transitions[0].current_step
    finally:
        await pg_engine.dispose()


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_button_selection_transaction_failure_full_rollback(monkeypatch):
    """
    Invariant 3 on real PostgreSQL:
    Forced crash after CAS selection but before transaction commit.
    Verifies full rollback on PostgreSQL (selected_button_id remains NULL, 0 transitions),
    followed by clean repeated click resulting in strictly ONE transition and ONE response.
    """
    pg_engine, skip_reason = await _get_safe_postgres_engine()
    if pg_engine is None:
        pytest.skip(skip_reason)

    try:
        await _setup_postgres_clean_schema(pg_engine)
        sessions = async_sessionmaker(pg_engine, expire_on_commit=False, class_=AsyncSession)
        _patch_postgres_sessions(monkeypatch, sessions, pg_engine)

        user_id = 7703
        await _seed_postgres_fixtures(sessions, user_id=user_id)

        async with sessions() as session:
            cfg = await get_or_create_followup_config(session, 20, topic_id=1)
            cfg.is_active = True
            cfg.target_stage = "OFFER_MEETING"
            cfg.buttons_config_json = json.dumps([
                {"id": "accept", "text": "Записаться", "target_step": "MEETING_BOOKING", "response_text": "Отлично! Запись подтверждена."},
            ], ensure_ascii=False)
            receipt, _ = await get_or_create_receipt(session, user_id, 1, 1, "key_pg_rollback", config_id=cfg.id)
            receipt.status = "completed"
            receipt.invitation_message_id = "666"
            await session.commit()
            config_id = cfg.id
            receipt_id = receipt.id

        transport = ValidatingCertTelegramTransport()
        bot = Bot("123456:TEST_BOT_TOKEN", session=transport)
        handlers.router._parent_router = None
        dispatcher = Dispatcher()
        dispatcher.include_router(handlers.router)

        callback_data = f"cert_meet:accept:{config_id}:{receipt_id}"

        # 1. Force failure during transition_user_to_stage (after CAS succeeded)
        real_transition = certificate_followup_service.transition_user_to_stage
        crash_injected = False

        async def crashing_transition(*args, **kwargs):
            nonlocal crash_injected
            crash_injected = True
            raise RuntimeError("PostgreSQL simulated network drop before transition commit")

        monkeypatch.setattr(certificate_followup_service, "transition_user_to_stage", crashing_transition)

        # First attempt: failure injected
        with pytest.raises(RuntimeError, match="PostgreSQL simulated network drop"):
            await handle_cert_meet_click(
                sessions,
                user_id=user_id,
                topic_id=1,
                dialogue_id=1,
                callback_data=callback_data,
            )

        assert crash_injected is True

        # Verify atomic rollback on real PostgreSQL:
        # selected_button_id must remain NULL, 0 transitions, user step unchanged
        async with sessions() as session:
            r = await session.get(CertificateIssuanceReceipt, receipt_id)
            assert r.selected_button_id is None, "selected_button_id must remain NULL on rollback in PostgreSQL"
            assert r.selected_at is None

            transitions = (await session.scalars(
                select(AutomationStepTransition).where(AutomationStepTransition.user_id == user_id)
            )).all()
            assert len(transitions) == 0, "No transitions must exist in PostgreSQL after rollback"

            conv = await session.scalar(
                select(AutomationConversationState).where(AutomationConversationState.user_id == user_id)
            )
            if conv:
                assert conv.current_step != "MEETING_BOOKING"

        # 2. Recovery: repeat click through real Telegram Dispatcher with normal transition execution
        monkeypatch.setattr(certificate_followup_service, "transition_user_to_stage", real_transition)

        update = Update(
            update_id=30003,
            callback_query=CallbackQuery(
                id="cb_retry",
                from_user=TelegramUser(id=user_id, is_bot=False, first_name="PGAlice"),
                chat_instance="ci_pg_retry",
                data=callback_data,
                message=Message(message_id=666, date=datetime.now(timezone.utc), chat=Chat(id=user_id, type="private"), text="invite"),
            ),
        )
        await dispatcher.feed_update(bot, update)

        # Wire: strictly ONE response message to the user
        sent_msgs = [c for c in transport.calls if isinstance(c, SendMessage)]
        assert len(sent_msgs) == 1, f"Expected strictly 1 response SendMessage, got {len(sent_msgs)}"
        assert "Запись подтверждена" in sent_msgs[0].text

        # In PostgreSQL: strictly ONE transition recorded and selected_button_id set
        async with sessions() as session:
            r_final = await session.get(CertificateIssuanceReceipt, receipt_id)
            assert r_final.selected_button_id == "accept"
            assert r_final.selected_at is not None

            transitions_final = (await session.scalars(
                select(AutomationStepTransition).where(AutomationStepTransition.user_id == user_id)
            )).all()
            assert len(transitions_final) == 1, f"Expected strictly 1 transition in PostgreSQL, got {len(transitions_final)}"
            assert transitions_final[0].current_step == "MEETING_BOOKING"

            conv_final = await session.scalar(
                select(AutomationConversationState).where(AutomationConversationState.user_id == user_id)
            )
            assert conv_final.current_step == "MEETING_BOOKING"
    finally:
        await pg_engine.dispose()


@pytest.mark.postgres_integration
@pytest.mark.asyncio
async def test_real_postgresql_data_plus_completed_single_atomic_transaction(monkeypatch):
    """
    Invariant 4 on real PostgreSQL:
    complete_followup_delivery applies deferred <DATA>, transitions stage, and marks completed
    in a single atomic transaction under run_user_transaction.
    Simulates crash during transaction: verifies complete rollback on PostgreSQL, followed
    by successful recovery with exactly ONE event, ONE transition, and completed status.
    """
    pg_engine, skip_reason = await _get_safe_postgres_engine()
    if pg_engine is None:
        pytest.skip(skip_reason)

    try:
        await _setup_postgres_clean_schema(pg_engine)
        sessions = async_sessionmaker(pg_engine, expire_on_commit=False, class_=AsyncSession)
        _patch_postgres_sessions(monkeypatch, sessions, pg_engine)

        user_id = 7704
        await _seed_postgres_fixtures(sessions, user_id=user_id)

        deferred_payload = json.dumps([
            {
                "current_state": {"current_step": "OFFER_MEETING", "lead_qualified": True},
                "events": ["cert_awarded"],
                "metadata": {"pg_atomicity_check": "v1"},
            }
        ], ensure_ascii=False)

        async with sessions() as session:
            cfg = await get_or_create_followup_config(session, 30, topic_id=1)
            cfg.is_active = True
            cfg.target_stage = "OFFER_MEETING"
            receipt, _ = await get_or_create_receipt(
                session,
                user_id,
                1,
                1,
                "key_pg_data_atomicity",
                config_id=cfg.id,
                deferred_data_json=deferred_payload,
            )
            receipt.status = "cert_sent"
            await session.commit()
            receipt_id = receipt.id

        # 1. First attempt: simulate crash during complete_followup_delivery transaction
        real_transition = certificate_followup_service.transition_user_to_stage
        crash_injected = False

        async def crashing_transition(*args, **kwargs):
            nonlocal crash_injected
            crash_injected = True
            raise RuntimeError("PostgreSQL simulated failure before completing receipt")

        monkeypatch.setattr(certificate_followup_service, "transition_user_to_stage", crashing_transition)

        with pytest.raises(RuntimeError, match="PostgreSQL simulated failure"):
            await complete_followup_delivery(
                sessions,
                receipt_id=receipt_id,
                invitation_message_id="msg_pg_777",
                scen_cfg=cfg,
                user_id=user_id,
                topic_id=1,
                dialogue_id=1,
            )

        assert crash_injected is True

        # Verify atomic rollback on real PostgreSQL:
        # NO events, NO transitions, User metadata untouched, receipt remains cert_sent with deferred_data_json intact
        async with sessions() as session:
            events = (await session.scalars(
                select(AutomationEvent).where(AutomationEvent.user_id == user_id)
            )).all()
            assert len(events) == 0, "DATA events must have rolled back completely in PostgreSQL"

            transitions = (await session.scalars(
                select(AutomationStepTransition).where(AutomationStepTransition.user_id == user_id)
            )).all()
            assert len(transitions) == 0, "Transitions must have rolled back completely in PostgreSQL"

            r_crash = await session.get(CertificateIssuanceReceipt, receipt_id)
            assert r_crash.status == "cert_sent", "Receipt status must remain cert_sent in PostgreSQL"
            assert r_crash.deferred_data_json is not None, "deferred_data_json must remain preserved for recovery"

            u_crash = await session.get(User, user_id)
            assert "pg_atomicity_check" not in (u_crash.metadata_json or "")

        # 2. Recovery: run complete_followup_delivery with normal execution
        monkeypatch.setattr(certificate_followup_service, "transition_user_to_stage", real_transition)

        await complete_followup_delivery(
            sessions,
            receipt_id=receipt_id,
            invitation_message_id="msg_pg_recovered_999",
            scen_cfg=cfg,
            user_id=user_id,
            topic_id=1,
            dialogue_id=1,
        )

        # Verify recovery state on real PostgreSQL: strictly ONE event, ONE transition, receipt completed
        async with sessions() as session:
            events = (await session.scalars(
                select(AutomationEvent).where(AutomationEvent.user_id == user_id)
            )).all()
            assert len(events) == 1, f"Expected strictly 1 event in PostgreSQL, got {len(events)}"
            assert events[0].name == "cert_awarded"

            transitions = (await session.scalars(
                select(AutomationStepTransition).where(AutomationStepTransition.user_id == user_id)
            )).all()
            assert len(transitions) == 1, f"Expected strictly 1 transition in PostgreSQL, got {len(transitions)}"
            assert transitions[0].current_step == "OFFER_MEETING"

            r_final = await session.get(CertificateIssuanceReceipt, receipt_id)
            assert r_final.status == "completed"
            assert r_final.deferred_data_json is None
            assert r_final.invitation_message_id == "msg_pg_recovered_999"

            u_final = await session.get(User, user_id)
            assert "pg_atomicity_check" in (u_final.metadata_json or "")
    finally:
        await pg_engine.dispose()
