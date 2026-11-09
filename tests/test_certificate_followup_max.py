import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import select

from database import (
    AIConfig,
    AutomationConversationState,
    AutomationStepTransition,
    CertificateFollowupConfig,
    CertificateIssuanceReceipt,
    MediaLibrary,
    Topic,
    User,
    async_session_maker as default_session_maker,
)
from certificate_followup_service import (
    get_or_create_followup_config,
    save_followup_caption,
    save_followup_invitation,
    save_followup_stage,
    handle_cert_callback_max,
    make_issuance_key,
    get_or_create_receipt,
    mark_receipt_sending_cert,
    mark_receipt_uncertain,
    resume_pending_certificate_followup,
)
import database
import handlers
import ai_integration
import automation_admin
import automation_events
from max_messenger_bot import ai as max_ai, app as max_app, storage as max_storage
from max_messenger_bot.services import common
from max_messenger_bot.api import MaxApiClient

from tests.test_media_image_edit import async_engine, session_factory, clean_runtime, setup_tg_user, media_setup, MaxTransport
from tests.test_media_templates import seed_template, source_png, VALUES


@pytest_asyncio.fixture(autouse=True)
async def patch_session_makers(session_factory, monkeypatch):
    for mod in (database, handlers, ai_integration, automation_admin, automation_events, common, max_ai, max_app, max_storage):
        if hasattr(mod, 'async_session_maker'):
            monkeypatch.setattr(mod, 'async_session_maker', session_factory)


async def seed_max_template(session_factory):
    source, config = await seed_template(session_factory)
    async with session_factory() as session:
        # Rename template 3 so it doesn't conflict with template 40
        tg_tpl = await session.get(MediaLibrary, 3)
        if tg_tpl:
            tg_tpl.file_name = 'telegram-template'

        # Template 40 is MAX platform template in media_setup fixture
        maximum = await session.get(MediaLibrary, 40)
        maximum.render_template_enabled = True
        maximum.render_template_config = config
        maximum.file_name = 'panda_certificate'

        # Set user 100000000007 topic to 1
        user = await session.get(User, 100000000007)
        if user:
            user.current_topic_id = 1
            user.current_dialogue_id = 1
        await session.commit()
    return source, config


@pytest.mark.asyncio
async def test_max_runtime_two_phase_delivery_success(session_factory, media_setup, monkeypatch):
    """
    MAX runtime two-phase delivery:
    1. Message 1: File attachment with admin caption delivered
    2. Message 2: Immediately following, invitation text with MAX inline keyboard
    3. Receipt: status marked 'completed' with message IDs
    4. Deferred <DATA>: service block applied only after successful delivery
    5. Conversation state: transitioned to target_stage
    """
    source, config = await seed_max_template(session_factory)

    async with session_factory() as session:
        scen_cfg = await get_or_create_followup_config(session, 40, topic_id=1)
        scen_cfg.is_active = True
        scen_cfg.caption_template = '🎉 Поздравляем, <b>{name}</b>! Ваш сертификат готов.'
        scen_cfg.invitation_text = 'Хотите записаться на консультацию?'
        scen_cfg.target_stage = 'OFFER_MEETING'
        await session.commit()

    client = MaxTransport()
    monkeypatch.setattr(client, 'download_attachment_url', AsyncMock(return_value=source))

    ai_response = (
        'Вот ваш сертификат:\n'
        f'[RENDER_IMG_MEDIA: panda_certificate | {json.dumps(VALUES, ensure_ascii=False)}]\n'
        '<DATA>{"current_state": {"current_step": "OFFER_MEETING", "lead_qualified": true}}</DATA>'
    )
    monkeypatch.setattr(max_ai, '_dispatch_provider', AsyncMock(return_value=ai_response))

    application = max_app.MaxBotApplication(client)
    await application.handle_update({
        'update_type': 'message_created',
        'update_id': 'max_two_phase_1',
        'message': {
            'sender': {'user_id': 7, 'name': 'MaxUser'},
            'recipient': {'chat_id': 100},
            'body': {'mid': 'm-in-1', 'text': 'Выдайте сертификат'}
        }
    })
    if application.background_tasks:
        await asyncio.gather(*list(application.background_tasks))

    # Verify upload occurred
    assert len(client.uploads) >= 1

    # Verify wire messages sent to MAX (excluding thinking placeholder)
    message_calls = [
        kwargs for method, path, kwargs in client.wire
        if path == '/messages' and method == 'POST' and kwargs.get('json_data', {}).get('text') != '🤖 Думаю...'
    ]
    assert len(message_calls) >= 2, f"Expected at least 2 messages, got {len(message_calls)}: {message_calls}"

    # Message 1: Certificate attachment with caption
    msg1_json = message_calls[0].get('json_data', {})
    attachments1 = msg1_json.get('attachments', [])
    assert any(a.get('type') == 'file' for a in attachments1)
    caption1 = msg1_json.get('text', '')
    assert 'Поздравляем, <b>Max</b>!' in caption1

    # Message 2: Invitation text with inline keyboard
    msg2_json = message_calls[1].get('json_data', {})
    text2 = msg2_json.get('text', '')
    assert 'Хотите записаться на консультацию?' in text2
    attachments2 = msg2_json.get('attachments', [])
    assert len(attachments2) >= 1
    kb_attachment = next((a for a in attachments2 if a.get('type') == 'inline_keyboard'), None)
    assert kb_attachment is not None
    buttons = kb_attachment.get('payload', {}).get('buttons', [])
    flat_buttons = [b for row in buttons for b in row]
    payloads = [b.get('payload', '') for b in flat_buttons]
    assert any(p.startswith('cert_meet:accept') for p in payloads)
    assert any(p.startswith('cert_meet:details') for p in payloads)
    assert any(p.startswith('cert_meet:later') for p in payloads)

    # Verify DB receipt is completed
    async with session_factory() as session:
        receipt = await session.scalar(
            select(CertificateIssuanceReceipt).where(CertificateIssuanceReceipt.user_id == 100000000007)
        )
        assert receipt is not None
        assert receipt.status == 'completed'
        assert receipt.certificate_message_id is not None
        assert receipt.invitation_message_id is not None

        # Verify conversation state has target step and service data
        conv_state = await session.scalar(
            select(AutomationConversationState).where(AutomationConversationState.user_id == 100000000007)
        )
        assert conv_state is not None
        assert conv_state.current_step == 'OFFER_MEETING'
        state_data = json.loads(conv_state.current_state_json or '{}')
        assert state_data.get('lead_qualified') is True


@pytest.mark.asyncio
async def test_max_deterministic_button_clicks(session_factory, media_setup):
    """
    Deterministic click handling on MAX:
    1. cert_meet:accept -> transitions step to target_step, sends confirmation message, no AI call
    2. cert_meet:details -> sends details explanation, no AI call
    3. cert_meet:later -> sends acknowledgement, no AI call
    """
    async with session_factory() as session:
        scen_cfg = await get_or_create_followup_config(session, 40, topic_id=1)
        scen_cfg.is_active = True
        await session.commit()
        config_id = scen_cfg.id

        # Setup confirmed issuance receipt for validation
        receipt = CertificateIssuanceReceipt(
            user_id=100000000007,
            topic_id=1,
            dialogue_id=1,
            config_id=config_id,
            issuance_key="seed_key_max_click",
            status="completed",
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )
        session.add(receipt)

        # Setup initial conversation state
        conv_state = AutomationConversationState(
            user_id=100000000007,
            topic_id=1,
            dialogue_id=1,
            current_step='OFFER_MEETING',
            current_state_json=json.dumps({"current_step": "OFFER_MEETING"}),
        )
        session.add(conv_state)
        await session.commit()

    client = MaxTransport()
    app = max_app.MaxBotApplication(client)

    sender = SimpleNamespace(user_id=100000000007, name='MaxUser', username='maxuser', full_name='MaxUser', public_name='MaxUser')

    # 1. Click Accept
    cb_accept = SimpleNamespace(
        callback_id='cb_max_1',
        payload=f'cert_meet:accept:{config_id}:{receipt.id}',
        chat_id=100,
        user_id=100000000007,
        sender=sender,
    )
    await app.handle_callback(cb_accept)

    # Verify callback answered
    assert any(path == '/answers' for method, path, kwargs in client.wire)

    # Verify message sent to user
    sent_texts = [kwargs.get('json_data', {}).get('text', '') for method, path, kwargs in client.wire if path == '/messages']
    assert any('Отлично! Запланируем встречу' in t for t in sent_texts)

    # Repeat click on Accept (Defect 3): Informs user and does NOT re-transition
    wire_before_repeat = len(client.wire)
    cb_repeat = SimpleNamespace(
        callback_id='cb_max_repeat',
        payload=f'cert_meet:accept:{config_id}:{receipt.id}',
        chat_id=100,
        user_id=100000000007,
        sender=sender,
    )
    await app.handle_callback(cb_repeat)
    repeat_messages = [kwargs for method, path, kwargs in client.wire[wire_before_repeat:] if path == '/messages']
    assert len(repeat_messages) == 1
    assert 'Повторный выбор не требуется' in repeat_messages[0].get('json_data', {}).get('text', '')

    # Unknown action click (Defect 3): Informs user and is rejected, never default to accept
    wire_before_unknown = len(client.wire)
    cb_unknown = SimpleNamespace(
        callback_id='cb_max_unknown',
        payload=f'cert_meet:invalid_action:{config_id}:{receipt.id}',
        chat_id=100,
        user_id=100000000007,
        sender=sender,
    )
    await app.handle_callback(cb_unknown)
    unknown_messages = [kwargs for method, path, kwargs in client.wire[wire_before_unknown:] if path == '/messages']
    assert len(unknown_messages) == 1
    assert 'Неизвестное действие' in unknown_messages[0].get('json_data', {}).get('text', '')

    # Verify conversation state updated
    async with session_factory() as session:
        state = await session.scalar(
            select(AutomationConversationState).where(AutomationConversationState.user_id == 100000000007)
        )
        assert state is not None
        assert state.current_step == 'MEETING_BOOKING'

        # Verify transition record created
        transitions = (await session.scalars(
            select(AutomationStepTransition).where(AutomationStepTransition.user_id == 100000000007)
        )).all()
        assert len(transitions) >= 1
        assert transitions[-1].previous_step == 'OFFER_MEETING'
        assert transitions[-1].current_step == 'MEETING_BOOKING'

    # 2. Click Details on fresh invitation receipt
    async with session_factory() as session:
        receipt_details = CertificateIssuanceReceipt(
            user_id=100000000007,
            topic_id=1,
            dialogue_id=1,
            config_id=config_id,
            issuance_key="seed_key_max_details",
            status="completed",
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )
        session.add(receipt_details)
        await session.commit()

    wire_len = len(client.wire)
    cb_details = SimpleNamespace(
        callback_id='cb_max_2',
        payload=f'cert_meet:details:{config_id}:{receipt_details.id}',
        chat_id=100,
        user_id=100000000007,
        sender=sender,
    )
    await app.handle_callback(cb_details)
    new_texts = [kwargs.get('json_data', {}).get('text', '') for method, path, kwargs in client.wire[wire_len:] if path == '/messages']
    assert any('Формат встречи' in t for t in new_texts)

    # 3. Click Later on fresh invitation receipt
    async with session_factory() as session:
        receipt_later = CertificateIssuanceReceipt(
            user_id=100000000007,
            topic_id=1,
            dialogue_id=1,
            config_id=config_id,
            issuance_key="seed_key_max_later",
            status="completed",
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )
        session.add(receipt_later)
        await session.commit()

    wire_len = len(client.wire)
    cb_later = SimpleNamespace(
        callback_id='cb_max_3',
        payload=f'cert_meet:later:{config_id}:{receipt_later.id}',
        chat_id=100,
        user_id=100000000007,
        sender=sender,
    )
    await app.handle_callback(cb_later)
    later_texts = [kwargs.get('json_data', {}).get('text', '') for method, path, kwargs in client.wire[wire_len:] if path == '/messages']
    assert any('Хорошо! Вы сможете вернуться' in t for t in later_texts)


@pytest.mark.asyncio
async def test_max_runtime_phase1_failure_prevents_message2_and_data_commit(session_factory, media_setup, monkeypatch):
    """
    Failure safety on MAX:
    If upload fails, Message 2 is NOT sent, deferred <DATA> is NOT committed,
    and receipt is marked 'failed'.
    """
    source, config = await seed_max_template(session_factory)

    async with session_factory() as session:
        scen_cfg = await get_or_create_followup_config(session, 40, topic_id=1)
        scen_cfg.is_active = True
        await session.commit()

    client = MaxTransport()
    monkeypatch.setattr(client, 'download_attachment_url', AsyncMock(return_value=source))

    # Simulate upload failure
    async def failing_upload(*args, **kwargs):
        raise RuntimeError("MAX upload transport timeout")
    monkeypatch.setattr(client, 'upload_file', failing_upload)

    ai_response = (
        'Вот сертификат:\n'
        f'[RENDER_IMG_MEDIA: panda_certificate | {json.dumps(VALUES, ensure_ascii=False)}]\n'
        '<DATA>{"current_state": {"current_step": "PREMATURE_STAGE"}}</DATA>'
    )
    monkeypatch.setattr(max_ai, '_dispatch_provider', AsyncMock(return_value=ai_response))

    application = max_app.MaxBotApplication(client)
    await application.handle_update({
        'update_type': 'message_created',
        'update_id': 'max_fail_1',
        'message': {
            'sender': {'user_id': 7, 'name': 'MaxUser'},
            'recipient': {'chat_id': 100},
            'body': {'mid': 'm-fail-1', 'text': 'Выдайте'}
        }
    })
    if application.background_tasks:
        await asyncio.gather(*list(application.background_tasks))

    # Verify no invitation message was sent
    message_calls = [kwargs for method, path, kwargs in client.wire if path == '/messages' and method == 'POST']
    # Message 2 has inline_keyboard attachment; make sure none was sent
    for call in message_calls:
        attachments = call.get('json_data', {}).get('attachments', [])
        assert not any(a.get('type') == 'inline_keyboard' for a in attachments)

    # Verify DB: receipt is failed, and DATA was NOT committed
    async with session_factory() as session:
        receipt = await session.scalar(
            select(CertificateIssuanceReceipt).where(CertificateIssuanceReceipt.user_id == 100000000007)
        )
        assert receipt is not None
        assert receipt.status in ('failed', 'uncertain')

        conv_state = await session.scalar(
            select(AutomationConversationState).where(AutomationConversationState.user_id == 100000000007)
        )
        if conv_state is not None:
            assert conv_state.current_step != 'PREMATURE_STAGE'


@pytest.mark.asyncio
async def test_max_runtime_resumption_without_duplicate_certificate(session_factory, media_setup, monkeypatch):
    """
    Resumption on MAX:
    When receipt status is 'cert_sent', only Message 2 (invitation) is sent.
    Certificate is NOT uploaded or sent again.
    """
    source, config = await seed_max_template(session_factory)

    from media_templates import current_issue_date
    today = current_issue_date()
    key = make_issuance_key(40, 100000000007, VALUES['recipient'], today)

    async with session_factory() as session:
        scen_cfg = await get_or_create_followup_config(session, 40, topic_id=1)
        scen_cfg.is_active = True
        receipt, _ = await get_or_create_receipt(session, 100000000007, 1, 1, key, config_id=scen_cfg.id)
        receipt.status = 'cert_sent'
        receipt.config_id = scen_cfg.id
        receipt.certificate_message_id = 'max-prev-msg-999'
        await session.commit()

    client = MaxTransport()
    monkeypatch.setattr(client, 'download_attachment_url', AsyncMock(return_value=source))

    ai_response = f'[RENDER_IMG_MEDIA: panda_certificate | {json.dumps(VALUES, ensure_ascii=False)}]'
    monkeypatch.setattr(max_ai, '_dispatch_provider', AsyncMock(return_value=ai_response))

    application = max_app.MaxBotApplication(client)
    await application.handle_update({
        'update_type': 'message_created',
        'update_id': 'max_resume_1',
        'message': {
            'sender': {'user_id': 7, 'name': 'MaxUser'},
            'recipient': {'chat_id': 100},
            'body': {'mid': 'm-resume-1', 'text': 'Продолжить'}
        }
    })
    if application.background_tasks:
        await asyncio.gather(*list(application.background_tasks))

    # Verify no file upload occurred
    assert len(client.uploads) == 0

    # Verify Message 2 was sent
    message_calls = [
        kwargs for method, path, kwargs in client.wire
        if path == '/messages' and method == 'POST' and kwargs.get('json_data', {}).get('text') != '🤖 Думаю...'
    ]
    assert len(message_calls) >= 1
    msg2 = message_calls[-1].get('json_data', {})
    attachments = msg2.get('attachments', [])
    assert any(a.get('type') == 'inline_keyboard' for a in attachments)

    # Verify receipt status updated to completed
    async with session_factory() as session:
        updated_receipt = await session.scalar(
            select(CertificateIssuanceReceipt).where(CertificateIssuanceReceipt.user_id == 100000000007)
        )
        assert updated_receipt.status == 'completed'
        assert updated_receipt.certificate_message_id == "max-prev-msg-999"
        assert updated_receipt.invitation_message_id is not None


@pytest.mark.asyncio
async def test_max_runtime_uncertain_status_skips_blind_retry(session_factory, media_setup, monkeypatch):
    """
    Uncertain delivery status on MAX:
    If receipt is 'uncertain', skips blind retry without re-uploading or sending duplicate.
    """
    source, config = await seed_max_template(session_factory)
    from media_templates import current_issue_date
    today = current_issue_date()
    key = make_issuance_key(40, 100000000007, VALUES['recipient'], today)

    async with session_factory() as session:
        scen_cfg = await get_or_create_followup_config(session, 40, topic_id=1)
        scen_cfg.is_active = True
        receipt, _ = await get_or_create_receipt(session, 100000000007, 1, 1, key)
        receipt.status = 'uncertain'
        receipt.error_details = 'Timeout during gateway transmit'
        await session.commit()

    client = MaxTransport()
    monkeypatch.setattr(client, 'download_attachment_url', AsyncMock(return_value=source))

    ai_response = f'[RENDER_IMG_MEDIA: panda_certificate | {json.dumps(VALUES, ensure_ascii=False)}]'
    monkeypatch.setattr(max_ai, '_dispatch_provider', AsyncMock(return_value=ai_response))

    application = max_app.MaxBotApplication(client)
    await application.handle_update({
        'update_type': 'message_created',
        'update_id': 'max_uncertain_1',
        'message': {
            'sender': {'user_id': 7, 'name': 'MaxUser'},
            'recipient': {'chat_id': 100},
            'body': {'mid': 'm-unc-1', 'text': 'Повторите'}
        }
    })
    if application.background_tasks:
        await asyncio.gather(*list(application.background_tasks))

    # Verify no file upload occurred
    assert len(client.uploads) == 0
    # No content message sent to client
    content_messages = [
        kwargs for method, path, kwargs in client.wire
        if path == '/messages' and method == 'POST' and kwargs.get('json_data', {}).get('text') != '🤖 Думаю...'
    ]
    assert len(content_messages) == 0


@pytest.mark.asyncio
async def test_max_runtime_disabled_preserves_legacy_behavior(session_factory, media_setup, monkeypatch):
    """
    When scenario is disabled (is_active=False), MAX delivers single message (file attachment)
    without Message 2, preserving full backward compatibility.
    """
    source, config = await seed_max_template(session_factory)

    async with session_factory() as session:
        scen_cfg = await get_or_create_followup_config(session, 40, topic_id=1)
        scen_cfg.is_active = False
        await session.commit()

    client = MaxTransport()
    monkeypatch.setattr(client, 'download_attachment_url', AsyncMock(return_value=source))

    ai_response = f'[RENDER_IMG_MEDIA: panda_certificate | {json.dumps(VALUES, ensure_ascii=False)}]'
    monkeypatch.setattr(max_ai, '_dispatch_provider', AsyncMock(return_value=ai_response))

    application = max_app.MaxBotApplication(client)
    await application.handle_update({
        'update_type': 'message_created',
        'update_id': 'max_legacy_1',
        'message': {
            'sender': {'user_id': 7, 'name': 'MaxUser'},
            'recipient': {'chat_id': 100},
            'body': {'mid': 'm-leg-1', 'text': 'Сертификат'}
        }
    })
    if application.background_tasks:
        await asyncio.gather(*list(application.background_tasks))

    # Single message with file attachment sent
    message_calls = [
        kwargs for method, path, kwargs in client.wire
        if path == '/messages' and method == 'POST' and kwargs.get('json_data', {}).get('text') != '🤖 Думаю...'
    ]
    assert len(message_calls) == 1
    attachments = message_calls[0].get('json_data', {}).get('attachments', [])
    assert len(attachments) == 1
    assert attachments[0].get('type') == 'file'

    # No inline_keyboard sent
    assert not any(a.get('type') == 'inline_keyboard' for a in attachments)

    # No receipt created
    async with session_factory() as session:
        receipt = await session.scalar(
            select(CertificateIssuanceReceipt).where(CertificateIssuanceReceipt.user_id == 100000000007)
        )
        assert receipt is None


@pytest.mark.asyncio
async def test_max_resume_different_user_id_and_chat_id(session_factory, media_setup, monkeypatch):
    """
    Defect 1 Verification on MAX:
    Internal MAX DB user ID (e.g. 100000000007) and real chat_id (e.g. 888777666) differ.
    resume_pending_certificate_followup must pass the actual recipient chat_id from the incoming message,
    and outgoing MaxApiClient call to /messages must target chat_id=888777666, NOT user_id=100000000007.
    """
    source, config = await seed_max_template(session_factory)
    from media_templates import current_issue_date
    today = current_issue_date()
    key = make_issuance_key(40, 100000000007, VALUES['recipient'], today)

    different_chat_id = 888777666

    async with session_factory() as session:
        scen_cfg = await get_or_create_followup_config(session, 40, topic_id=1)
        scen_cfg.is_active = True
        scen_cfg.invitation_text = "Приглашаем на персональную сессию."
        receipt, _ = await get_or_create_receipt(session, 100000000007, 1, 1, key, config_id=scen_cfg.id)
        receipt.status = 'cert_sent'
        receipt.config_id = scen_cfg.id
        receipt.certificate_message_id = 'max-cert-msg-1'
        await session.commit()
        receipt_id = receipt.id

    client = MaxTransport()

    # Part A: Direct call to resume_pending_certificate_followup
    success = await resume_pending_certificate_followup(
        client,
        user_id=100000000007,
        topic_id=1,
        dialogue_id=1,
        session_maker=session_factory,
        platform="max",
        chat_id=different_chat_id,
    )
    assert success is True

    # Verify outgoing /messages call targeted different_chat_id, NOT user_id
    post_messages = [
        kwargs for method, path, kwargs in client.wire
        if path == '/messages' and method == 'POST'
    ]
    assert len(post_messages) >= 1
    last_call = post_messages[-1]
    params = last_call.get('params', {})
    assert params.get('chat_id') == different_chat_id
    assert params.get('chat_id') != 100000000007
    assert params.get('user_id') is None

    # Part B: End-to-end update flow where incoming message recipient has chat_id
    async with session_factory() as session:
        r = await session.get(CertificateIssuanceReceipt, receipt_id)
        r.status = 'cert_sent'
        r.invitation_message_id = None
        await session.commit()

    wire_len_before = len(client.wire)
    monkeypatch.setattr(client, 'download_attachment_url', AsyncMock(return_value=source))
    ai_response = f'[RENDER_IMG_MEDIA: panda_certificate | {json.dumps(VALUES, ensure_ascii=False)}]'
    monkeypatch.setattr(max_ai, '_dispatch_provider', AsyncMock(return_value=ai_response))

    application = max_app.MaxBotApplication(client)
    await application.handle_update({
        'update_type': 'message_created',
        'update_id': 'max_resume_diff_chat_update',
        'message': {
            'sender': {'user_id': 7, 'name': 'MaxUser'},
            'recipient': {'chat_id': 999111222},
            'body': {'mid': 'm-diff-2', 'text': 'Продолжить'}
        }
    })
    if application.background_tasks:
        await asyncio.gather(*list(application.background_tasks))

    e2e_messages = [
        kwargs for method, path, kwargs in client.wire[wire_len_before:]
        if path == '/messages' and method == 'POST' and kwargs.get('json_data', {}).get('text') != '🤖 Думаю...'
    ]
    assert len(e2e_messages) >= 1
    invite_call = e2e_messages[-1]
    e2e_params = invite_call.get('params', {})
    assert e2e_params.get('chat_id') == 999111222
    assert e2e_params.get('chat_id') != 100000000007


@pytest.mark.asyncio
async def test_max_failure_point2_sending_cert_prevents_duplicate_attachment(session_factory, media_setup, monkeypatch):
    """
    Failure Point 2 on MAX:
    Process crash between sending certificate attachment and saving cert_sent.
    In-flight 'sending_cert' is detected on next attempt, marked 'uncertain',
    and does NOT upload or send a duplicate file.
    """
    source, config = await seed_max_template(session_factory)
    from media_templates import current_issue_date
    today = current_issue_date()
    key = make_issuance_key(40, 100000000007, VALUES['recipient'], today)

    async with session_factory() as session:
        scen_cfg = await get_or_create_followup_config(session, 40, topic_id=1)
        scen_cfg.is_active = True
        r, _ = await get_or_create_receipt(session, 100000000007, 1, 1, key, config_id=scen_cfg.id)
        # Simulate in-flight crash right after attachment was accepted by MAX
        r.status = "sending_cert"
        await session.commit()
        receipt_id = r.id

    client = MaxTransport()
    monkeypatch.setattr(client, 'download_attachment_url', AsyncMock(return_value=source))

    ai_response = f'[RENDER_IMG_MEDIA: panda_certificate | {json.dumps(VALUES, ensure_ascii=False)}]'
    monkeypatch.setattr(max_ai, '_dispatch_provider', AsyncMock(return_value=ai_response))

    application = max_app.MaxBotApplication(client)
    await application.handle_update({
        'update_type': 'message_created',
        'update_id': 'max_crash_sending_cert',
        'message': {
            'sender': {'user_id': 7, 'name': 'MaxUser'},
            'recipient': {'chat_id': 100},
            'body': {'mid': 'm-crash-1', 'text': 'Повторите сертификат'}
        }
    })
    if application.background_tasks:
        await asyncio.gather(*list(application.background_tasks))

    # Verify NO file was uploaded again
    assert len(client.uploads) == 0

    # Verify NO duplicate certificate attachment was sent
    message_calls = [
        kwargs for method, path, kwargs in client.wire
        if path == '/messages' and method == 'POST' and kwargs.get('json_data', {}).get('text') != '🤖 Думаю...'
    ]
    assert len(message_calls) == 0

    # Verify receipt was marked 'uncertain'
    async with session_factory() as session:
        r = await session.get(CertificateIssuanceReceipt, receipt_id)
        assert r.status == "uncertain"
        assert "Previous process interrupted" in (r.error_details or "")


@pytest.mark.asyncio
async def test_max_failure_point3_invite_network_error_marks_uncertain(session_factory, media_setup):
    """
    Failure Point 3 on MAX:
    Network error during resumed invitation sending does NOT blind-reset to cert_sent;
    it is marked 'uncertain' to prevent duplicate messages.
    """
    from media_templates import current_issue_date
    today = current_issue_date()
    key = make_issuance_key(40, 100000000007, VALUES['recipient'], today)

    async with session_factory() as session:
        scen_cfg = await get_or_create_followup_config(session, 40, topic_id=1)
        scen_cfg.is_active = True
        r, _ = await get_or_create_receipt(session, 100000000007, 1, 1, key, config_id=scen_cfg.id)
        r.status = "cert_sent"
        await session.commit()
        receipt_id = r.id

    client = MaxTransport()
    # Mock send_message to fail with network error
    async def failing_send(*args, **kwargs):
        raise RuntimeError("MAX gateway connection dropped")
    client.send_message = failing_send

    success = await resume_pending_certificate_followup(
        client,
        user_id=100000000007,
        topic_id=1,
        dialogue_id=1,
        session_maker=session_factory,
        platform="max",
        chat_id=100,
    )
    assert success is False

    # Verify status is uncertain, NOT reset to cert_sent
    async with session_factory() as session:
        r = await session.get(CertificateIssuanceReceipt, receipt_id)
        assert r.status == "uncertain"
        assert "Resumption delivery failed/uncertain" in (r.error_details or "")


@pytest.mark.asyncio
async def test_defect2_concurrent_certificate_issuance_max_single_wire_send(session_factory, media_setup, monkeypatch):
    """
    Defect 2: Concurrent certificate issuance in MAX.
    Two simultaneous requests to issue the same certificate.
    Atomic CAS ensures strictly ONE call to upload_file and send_media_attachment.
    """
    user_id = 100000000099
    source, config = await seed_max_template(session_factory)
    async with session_factory() as session:
        user = await session.get(User, user_id)
        if not user:
            user = User(
                id=user_id,
                name="ConcurrentMaxUser",
                first_name="ConcurrentMaxUser",
                gender="male",
                age="30",
                accepted_disclaimer=True,
                current_dialogue_id=1,
                current_topic_id=1,
            )
            session.add(user)
        else:
            user.current_dialogue_id = 1
            user.current_topic_id = 1
        scen_cfg = await get_or_create_followup_config(session, 40, topic_id=1)
        scen_cfg.is_active = True
        scen_cfg.caption_template = '🎉 Поздравляем, <b>{name}</b>! Ваш сертификат готов.'
        scen_cfg.invitation_text = 'Хотите записаться на консультацию?'
        scen_cfg.target_stage = 'OFFER_MEETING'
        await session.commit()

    client = MaxTransport()
    monkeypatch.setattr(client, 'download_attachment_url', AsyncMock(return_value=source))

    values = dict(VALUES, recipient="ConcurrentMaxIsolated")
    ai_response = f'[RENDER_IMG_MEDIA: panda_certificate | {json.dumps(values, ensure_ascii=False)}]'

    # Run two delivery requests concurrently
    res1, res2 = await asyncio.gather(
        common._deliver_media_template(client, chat_id=100, user_id=user_id, response=ai_response, dialogue_id=1, topic_id=1),
        common._deliver_media_template(client, chat_id=100, user_id=user_id, response=ai_response, dialogue_id=1, topic_id=1),
    )
    assert res1 is True and res2 is True

    # Strictly ONE upload_file call must be made
    assert len(client.uploads) == 1, f"Expected strictly 1 upload, got {len(client.uploads)}"

    # Check that receipt is completed
    from media_templates import current_issue_date
    expected_key = make_issuance_key(40, user_id, "ConcurrentMaxIsolated", current_issue_date())
    async with session_factory() as session:
        receipt = await session.scalar(
            select(CertificateIssuanceReceipt).where(
                CertificateIssuanceReceipt.issuance_key == expected_key,
            )
        )
        assert receipt is not None
        assert receipt.status == 'completed'


@pytest.mark.asyncio
async def test_defect3_canonical_max_message_ids_extraction_and_postgres_saving(session_factory, media_setup, monkeypatch):
    """
    Defect 3: Canonical MAX message IDs.
    Replaces stringified dict with canonical `mid` string extracted via `extract_sent_message_id`.
    Tests with a realistic full MAX API response (significantly longer than 64 characters),
    verifying clean persistence into String(64) columns and subsequent retrieval.
    """
    from max_messenger_bot.models import extract_sent_message_id

    source, config = await seed_max_template(session_factory)
    async with session_factory() as session:
        scen_cfg = await get_or_create_followup_config(session, 40, topic_id=1)
        scen_cfg.is_active = True
        scen_cfg.caption_template = '🎉 Поздравляем, <b>{name}</b>! Ваш сертификат готов.'
        scen_cfg.invitation_text = 'Хотите записаться на консультацию?'
        scen_cfg.target_stage = 'OFFER_MEETING'
        await session.commit()

    CERT_MID = "01931a54-73d2-7c82-99ab-6db52857e0f2"
    INVITE_MID = "01931a54-73d2-7c82-99ab-8fc93821a113"

    realistic_cert_response = {
        "success": True,
        "message": {
            "body": {
                "mid": CERT_MID,
                "text": "🎉 Поздравляем! Ваш сертификат готов.",
                "attachments": [
                    {
                        "type": "file",
                        "payload": {
                            "token": "token_abc1234567890_very_long_attachment_payload_exceeding_standard_limits_definitely_longer_than_64_characters",
                            "url": "https://storage.max.chat/files/cert.png"
                        }
                    }
                ]
            },
            "recipient": {"chat_id": 100},
            "sender": {"user_id": 999999, "name": "Psychobot"}
        }
    }
    realistic_invite_response = {
        "success": True,
        "message": {
            "body": {
                "mid": INVITE_MID,
                "text": "Хотите записаться на консультацию?",
                "attachments": [
                    {
                        "type": "inline_keyboard",
                        "payload": {
                            "buttons": [
                                [{"text": "🤝 Хочу встретиться", "callback_data": "cert_meet:accept:1"}],
                                [{"text": "ℹ️ Узнать подробнее", "callback_data": "cert_meet:details:1"}]
                            ]
                        }
                    }
                ]
            }
        }
    }

    # Verify that raw stringified dicts are much longer than 64 characters (which would break String(64))
    assert len(str(realistic_cert_response)) > 150
    assert len(str(realistic_invite_response)) > 150
    assert extract_sent_message_id(realistic_cert_response) == CERT_MID
    assert extract_sent_message_id(realistic_invite_response) == INVITE_MID

    client = MaxTransport()
    monkeypatch.setattr(client, 'download_attachment_url', AsyncMock(return_value=source))

    # Mock client methods to return realistic full API response dictionaries
    client.send_media_attachment = AsyncMock(return_value=realistic_cert_response)
    client.send_message = AsyncMock(return_value=realistic_invite_response)

    ai_response = f'[RENDER_IMG_MEDIA: panda_certificate | {json.dumps(VALUES, ensure_ascii=False)}]'

    # Part A: Two-phase delivery saves canonical mid
    success = await common._deliver_media_template(
        client,
        chat_id=100,
        user_id=100000000007,
        response=ai_response,
        dialogue_id=1,
        topic_id=1,
    )
    assert success is True

    async with session_factory() as session:
        receipt = await session.scalar(
            select(CertificateIssuanceReceipt).where(CertificateIssuanceReceipt.user_id == 100000000007)
        )
        assert receipt is not None
        assert receipt.status == 'completed'
        # Strict validation of canonical ID: exactly CERT_MID and INVITE_MID, no JSON / dict formatting
        assert receipt.certificate_message_id == CERT_MID
        assert len(receipt.certificate_message_id) == len(CERT_MID) <= 64
        assert not receipt.certificate_message_id.startswith("{")

        assert receipt.invitation_message_id == INVITE_MID
        assert len(receipt.invitation_message_id) == len(INVITE_MID) <= 64
        assert not receipt.invitation_message_id.startswith("{")

    # Part B: Resumption saves canonical mid
    RESUME_INVITE_MID = "01931a54-73d2-7c82-99ab-9aa44112c334"
    realistic_resume_invite_response = {
        "success": True,
        "message": {
            "body": {
                "mid": RESUME_INVITE_MID,
                "text": "Хотите продолжить?",
                "attachments": [{"type": "inline_keyboard", "payload": {"buttons": []}}]
            }
        }
    }
    client.send_message = AsyncMock(return_value=realistic_resume_invite_response)

    key2 = make_issuance_key(40, 100000000007, "Resumed Recipient", '09.11.2026')
    async with session_factory() as session:
        r, _ = await get_or_create_receipt(session, 100000000007, 1, 1, key2, config_id=scen_cfg.id)
        r.status = "cert_sent"
        r.certificate_message_id = CERT_MID
        await session.commit()
        receipt_id = r.id

    resume_success = await resume_pending_certificate_followup(
        client,
        user_id=100000000007,
        topic_id=1,
        dialogue_id=1,
        session_maker=session_factory,
        platform="max",
        chat_id=101,
    )
    assert resume_success is True

    async with session_factory() as session:
        resumed_receipt = await session.get(CertificateIssuanceReceipt, receipt_id)
        assert resumed_receipt.status == "completed"
        assert resumed_receipt.invitation_message_id == RESUME_INVITE_MID
        assert len(resumed_receipt.invitation_message_id) == len(RESUME_INVITE_MID) <= 64
        assert not resumed_receipt.invitation_message_id.startswith("{")

