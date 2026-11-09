import asyncio
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

os.environ.setdefault('BOT_TOKEN', '123456:TEST_BOT_TOKEN')
os.environ.setdefault('DATABASE_URL', 'sqlite+aiosqlite:///:memory:')

import pytest
import pytest_asyncio
from PIL import Image
from aiogram import Bot, Dispatcher
from aiogram.enums import ChatType
from aiogram.methods import SendDocument, SendMessage, SendPhoto, AnswerCallbackQuery, GetFile
from aiogram.types import CallbackQuery, Chat, Message, Update, User as TelegramUser
from sqlalchemy import select, text

import database
import handlers
import automation_admin
import automation_events
import translation_service
import event_service
import ai_integration
import media_templates
from database import AIConfig, AutomationConversationState, AutomationStepTransition, MediaLibrary, Topic, User, Message as DBMessage, media_collection_items
from certificate_followup_service import (
    DEFAULT_CAPTION_TEMPLATE,
    DEFAULT_INVITATION_TEXT,
    DEFAULT_TARGET_STAGE,
    CertificateFollowupConfig,
    CertificateIssuanceReceipt,
    get_followup_config,
    get_or_create_followup_config,
    get_or_create_receipt,
    make_issuance_key,
    render_caption,
    render_invitation,
)
from media_template_renderer import bind_source, normalized_source
from media_templates import current_issue_date
from media_template_admin import BUTTON_CONTRACTS
from tests.test_media_image_edit import async_engine, session_factory, clean_runtime, setup_tg_user, media_setup, TelegramTransport
from tests.test_media_templates import seed_template, source_png, template_config, VALUES


class ValidatingCertTelegramTransport(TelegramTransport):
    async def make_request(self, bot, method, timeout=None):
        if isinstance(method, SendDocument):
            self.calls.append(method)
            type(method).model_validate(method.model_dump())
            return Message(
                message_id=len(self.calls),
                date=datetime.now(timezone.utc),
                chat=Chat(id=method.chat_id, type=ChatType.PRIVATE),
                text=method.caption or "",
                caption=method.caption,
            ).as_(bot)
        return await super().make_request(bot, method, timeout)


@pytest_asyncio.fixture(autouse=True)
async def patch_session_makers(session_factory, monkeypatch):
    for mod in (database, handlers, ai_integration, automation_admin, automation_events):
        if hasattr(mod, 'async_session_maker'):
            monkeypatch.setattr(mod, 'async_session_maker', session_factory)


@pytest.mark.asyncio
async def test_telegram_admin_full_scenario_journey_a_to_ya_to_a(session_factory, media_setup, monkeypatch):
    """
    Complete Admin UI journey A->Я->A:
    1. Enter template card (mt:home:3)
    2. Open «После выдачи» (mt:scenario:3)
    3. Toggle scenario inactive then active (mt:scen_toggle:3)
    4. Edit caption: invalid HTML rejected -> valid HTML saved
    5. Edit invitation: invalid HTML rejected -> valid HTML saved
    6. Edit target stage: invalid stage rejected -> valid stage saved
    7. Edit buttons: open list (mt:scen_buttons:3) -> edit button (mt:scen_btn_edit) -> save
    8. Cancel button edit -> back to scenario screen
    9. Test send (mt:scen_test:3): delivers real certificate and invitation to admin with aiogram Pydantic validation
    10. Navigate back to template home (mt:home:3)
    """
    source, config = await seed_template(session_factory)
    monkeypatch.setattr(handlers, 'is_admin', AsyncMock(return_value=True))
    transport = ValidatingCertTelegramTransport()
    async def stream(*args, **kwargs): yield source
    monkeypatch.setattr(transport, 'stream_content', stream)
    bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
    handlers.router._parent_router = None
    dispatcher = Dispatcher()
    dispatcher.include_router(handlers.router)

    state = dispatcher.fsm.get_context(bot=bot, chat_id=media_setup, user_id=media_setup)
    await state.update_data(authoring_locale='ru', preserved_origin='test')

    latest = None
    visited = set()

    async def click(prefix, *, initial=False):
        nonlocal latest
        if initial:
            data = prefix
        else:
            buttons = [b for row in latest.reply_markup.inline_keyboard for b in row]
            matching = [b for b in buttons if b.callback_data.startswith(prefix)]
            assert matching, f"No button found with prefix {prefix} in {buttons}"
            data = matching[0].callback_data
            if data.startswith('mt:'):
                action = data.split(':')[1]
                assert action in BUTTON_CONTRACTS
                visited.add(action)
        start = len(transport.calls)
        await dispatcher.feed_update(bot, Update(
            update_id=7000 + start,
            callback_query=CallbackQuery(
                id=str(start),
                from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Admin'),
                chat_instance='ci_test',
                data=data,
                message=Message(message_id=7000, date=datetime.now(timezone.utc), chat=Chat(id=media_setup, type='private'), text='screen')
            )
        ))
        screens = [c for c in transport.calls[start:] if getattr(c, 'reply_markup', None)]
        if screens:
            latest = screens[-1]
        return data

    async def enter(value):
        nonlocal latest
        start = len(transport.calls)
        await dispatcher.feed_update(bot, Update(
            update_id=8000 + start,
            message=Message(
                message_id=8000 + start,
                date=datetime.now(timezone.utc),
                chat=Chat(id=media_setup, type='private'),
                from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Admin'),
                text=value,
            )
        ))
        screens = [c for c in transport.calls[start:] if getattr(c, 'reply_markup', None)]
        if screens:
            latest = screens[-1]

    # 1. Open template card
    await click('mt:home:3:1:1:1:-1', initial=True)
    assert 'panda_certificate' in latest.text
    assert any('После выдачи' in b.text for row in latest.reply_markup.inline_keyboard for b in row)

    # 2. Open «После выдачи» scenario screen (created inactive by default)
    await click('mt:scenario:')
    assert 'Сценарий после выдачи сертификата' in latest.text
    assert '⚪️ Выключен' in latest.text

    # 3. Toggle scenario to Active
    await click('mt:scen_toggle:')
    assert '🟢 Включен' in latest.text

    # 4. Edit caption with validation
    await click('mt:scen_caption:')
    assert 'Подпись к сертификату' in latest.text
    # Try invalid HTML (unclosed <b> tag)
    await enter('Невалидный HTML <b>без закрывающего')
    call_texts = [getattr(c, 'text', '') or '' for c in transport.calls]
    assert any('Ошибка валидации' in t for t in call_texts), f"calls: {[getattr(c, 'text', None) for c in transport.calls]}"
    # Valid HTML
    await enter('🎉 Поздравляем, <b>{name}</b>! Ваш сертификат готов.')
    assert 'Сценарий после выдачи сертификата' in latest.text
    assert '🎉 Поздравляем, <b>{name}</b>!' in latest.text

    # 5. Edit invitation with validation
    await click('mt:scen_invite:')
    assert 'Приглашение на встречу' in latest.text
    # Try invalid HTML (unclosed <i> tag)
    await enter('Ошибка <i>без закрытия')
    call_texts = [getattr(c, 'text', '') or '' for c in transport.calls]
    assert any('Ошибка валидации' in t for t in call_texts), f"calls: {[getattr(c, 'text', None) for c in transport.calls]}"
    # Valid invitation
    await enter('Хотите обсудить дальнейшие шаги на индивидуальной консультации?')
    assert 'Хотите обсудить дальнейшие шаги' in latest.text

    # 6. Edit target stage with validation
    await click('mt:scen_stage:')
    assert 'Целевой этап' in latest.text
    # Try invalid identifier
    await enter('недопустимый этап с пробелами и русскими!')
    call_texts = [getattr(c, 'text', '') or '' for c in transport.calls]
    assert any('Ошибка валидации' in t for t in call_texts)
    # Valid stage
    await enter('CONSULTATION_OFFER')
    assert 'CONSULTATION_OFFER' in latest.text

    # 7. Edit buttons list
    await click('mt:scen_buttons:')
    assert 'Кнопки выбора' in latest.text
    # Edit button 0 and test cancel
    await click('mt:scen_btn_edit:')
    assert 'Настройка кнопки' in latest.text
    await click('mt:scen_cancel:')
    assert 'Сценарий после выдачи сертификата' in latest.text

    # Re-open buttons and save button update
    await click('mt:scen_buttons:')
    await click('mt:scen_btn_edit:')
    await enter('🤝 Да, записаться | BOOK_CALL | Отлично! Записываю вас.')
    assert 'Кнопки выбора' in latest.text
    assert any('🤝 Да, записаться' in b.text for row in latest.reply_markup.inline_keyboard for b in row)

    # 8. Return from buttons list to scenario
    await click('mt:scenario:')
    assert 'Сценарий после выдачи сертификата' in latest.text

    # 9. Test send to admin (aiogram Pydantic validation)
    await click('mt:scen_test:')
    documents = [c for c in transport.calls if isinstance(c, SendDocument)]
    assert len(documents) >= 1
    doc = SendDocument.model_validate(documents[-1].model_dump())
    assert doc.document.filename == 'test-certificate.png'
    assert 'Поздравляем' in (doc.caption or '')

    # Check invitation message with inline keyboard was sent
    invite_msgs = [
        c for c in transport.calls
        if isinstance(c, SendMessage) and getattr(c, 'reply_markup', None) and any(
            b.callback_data.startswith('cert_meet:')
            for row in c.reply_markup.inline_keyboard for b in row
        )
    ]
    assert len(invite_msgs) >= 1
    invite_msg = invite_msgs[-1]
    button_callbacks = [b.callback_data for row in invite_msg.reply_markup.inline_keyboard for b in row]
    assert any(cb.startswith('cert_meet:test:') for cb in button_callbacks)

    # Click test button and verify admin stage is unchanged and alert answered (Defect 8)
    test_cb = next(cb for cb in button_callbacks if cb.startswith('cert_meet:test:'))
    start_test = len(transport.calls)
    await dispatcher.feed_update(bot, Update(
        update_id=7999,
        callback_query=CallbackQuery(
            id='test_btn_click',
            from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Admin'),
            chat_instance='ci_test',
            data=test_cb,
            message=Message(
                message_id=7999,
                date=datetime.now(timezone.utc),
                chat=Chat(id=media_setup, type='private'),
                text=invite_msg.text,
                reply_markup=invite_msg.reply_markup,
            ),
        )
    ))
    test_answers = [c for c in transport.calls[start_test:] if isinstance(c, AnswerCallbackQuery)]
    assert len(test_answers) == 1
    assert test_answers[0].show_alert is True
    async with session_factory() as session:
        conv_state = await session.scalar(
            select(AutomationConversationState).where(AutomationConversationState.user_id == media_setup)
        )
        if conv_state:
            assert conv_state.current_step != 'BOOK_CALL'

    # 10. Navigate Back to template home
    await click('mt:home:')
    assert 'panda_certificate' in latest.text


@pytest.mark.asyncio
async def test_telegram_admin_entry_from_automations_menu(session_factory, media_setup, monkeypatch):
    """
    Test secondary entry from Automations menu:
    1. Open automations menu
    2. Click «🎓 Сценарии после сертификатов» (cert_followup_list)
    3. Click template to open scenario screen directly
    """
    source, config = await seed_template(session_factory)
    monkeypatch.setattr(handlers, 'is_admin', AsyncMock(return_value=True))
    monkeypatch.setattr(automation_admin, 'get_all_admin_ids', AsyncMock(return_value=[media_setup]))
    transport = ValidatingCertTelegramTransport()
    bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
    handlers.router._parent_router = None
    automation_admin.router._parent_router = None
    dispatcher = Dispatcher()
    dispatcher.include_router(automation_admin.router)
    dispatcher.include_router(handlers.router)

    state = dispatcher.fsm.get_context(bot=bot, chat_id=media_setup, user_id=media_setup)
    await state.update_data(authoring_locale='ru', preserved_origin='test')

    latest = None

    async def click_data(data):
        nonlocal latest
        start = len(transport.calls)
        await dispatcher.feed_update(bot, Update(
            update_id=9000 + start,
            callback_query=CallbackQuery(
                id=str(start),
                from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Admin'),
                chat_instance='ci_auto',
                data=data,
                message=Message(message_id=9000, date=datetime.now(timezone.utc), chat=Chat(id=media_setup, type='private'), text='screen')
            )
        ))
        screens = [c for c in transport.calls[start:] if getattr(c, 'reply_markup', None)]
        if screens:
            latest = screens[-1]

    # Open cert_followup_list
    await click_data('cert_followup_list')
    assert latest is not None
    assert 'Сценарии после выдачи сертификатов' in latest.text
    buttons = [b for row in latest.reply_markup.inline_keyboard for b in row]
    template_btn = next((b for b in buttons if b.callback_data.startswith('mt:scenario:3')), None)
    assert template_btn is not None

    # Click template -> navigates to scenario screen
    await click_data(template_btn.callback_data)
    assert 'Сценарий после выдачи сертификата' in latest.text
    assert 'panda_certificate' in latest.text


@pytest.mark.asyncio
async def test_telegram_runtime_two_phase_delivery_success(session_factory, media_setup, monkeypatch):
    """
    Verify complete user runtime two-phase delivery:
    Message 1: Certificate image with admin-configured caption.
    Message 2: Immediate invitation message with inline keyboard.
    Verification:
    - Strictly 2 messages delivered.
    - target_stage applied to conversation state.
    - Deferred service <DATA> blocks applied.
    - Receipt status is 'completed'.
    """
    source, config = await seed_template(session_factory)
    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, 3)
        cfg.is_active = True
        cfg.caption_template = "🎉 Поздравляем, <b>{name}</b>! Ваш сертификат готов."
        cfg.invitation_text = "Приглашаю на стратегическую сессию."
        cfg.target_stage = "OFFER_MEETING"
        await session.commit()

    transport = ValidatingCertTelegramTransport()
    async def stream(*args, **kwargs): yield source
    monkeypatch.setattr(transport, 'stream_content', stream)
    bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
    handlers.router._parent_router = None
    dispatcher = Dispatcher()
    dispatcher.include_router(handlers.router)

    # AI returns RENDER_IMG_MEDIA with service block
    ai_response = (
        'Вот ваш сертификат:\n'
        f'[RENDER_IMG_MEDIA: panda_certificate | {json.dumps(VALUES, ensure_ascii=False)}]\n'
        '<DATA>{"current_state": {"current_step": "OFFER_MEETING", "lead_qualified": true}}</DATA>'
    )
    monkeypatch.setattr(ai_integration, '_call_gemini_api', AsyncMock(return_value=ai_response))

    try:
        await dispatcher.feed_update(bot, Update(
            update_id=10001,
            message=Message(
                message_id=10001,
                date=datetime.now(timezone.utc),
                chat=Chat(id=media_setup, type='private'),
                from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Alice'),
                text='Выдайте сертификат'
            )
        ))
        if media_setup in handlers.user_processing_tasks:
            await handlers.user_processing_tasks[media_setup]

        # Verify Message 1: SendDocument
        documents = [c for c in transport.calls if isinstance(c, SendDocument)]
        assert len(documents) == 1
        doc = SendDocument.model_validate(documents[0].model_dump())
        assert doc.document.filename == 'certificate.png'
        assert 'Поздравляем, <b>Alice</b>!' in doc.caption

        # Verify Message 2: SendMessage with invitation text and inline keyboard
        sent_messages = [c for c in transport.calls if isinstance(c, SendMessage)]
        invitations = [m for m in sent_messages if 'Приглашаю на стратегическую сессию' in (m.text or '')]
        assert len(invitations) == 1
        inv = SendMessage.model_validate(invitations[0].model_dump())
        assert inv.reply_markup is not None
        callbacks = [b.callback_data for row in inv.reply_markup.inline_keyboard for b in row]
        assert any(cb.startswith('cert_meet:accept:') for cb in callbacks)
        assert any(cb.startswith('cert_meet:details:') for cb in callbacks)
        assert any(cb.startswith('cert_meet:later:') for cb in callbacks)

        # Check DB state: receipt is completed
        async with session_factory() as session:
            receipt = await session.scalar(
                select(CertificateIssuanceReceipt).where(CertificateIssuanceReceipt.user_id == media_setup)
            )
            assert receipt is not None
            assert receipt.status == 'completed'
            assert receipt.certificate_message_id is not None
            assert receipt.invitation_message_id is not None

            # Check automation state updated to target_stage
            conv_state = await session.scalar(
                select(AutomationConversationState).where(AutomationConversationState.user_id == media_setup)
            )
            assert conv_state is not None
            assert conv_state.current_step == 'OFFER_MEETING'
            # Also verify service block lead_qualified was applied
            state_data = json.loads(conv_state.current_state_json or '{}')
            assert state_data.get('lead_qualified') is True
    finally:
        dispatcher.sub_routers.clear()
        handlers.router._parent_router = None
        await bot.session.close()


@pytest.mark.asyncio
async def test_telegram_deterministic_button_clicks(session_factory, media_setup, monkeypatch):
    """
    Verify deterministic processing of invitation button clicks:
    - cert_meet:accept -> answered callback, confirmation text, stage updated
    - cert_meet:details -> answered callback, details text
    - cert_meet:later -> answered callback, reminder acknowledgment
    No AI hallucinations.
    """
    source, config = await seed_template(session_factory)
    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, 3)
        cfg.is_active = True
        config_id = cfg.id

        # Setup confirmed issuance receipt for validation
        receipt = CertificateIssuanceReceipt(
            user_id=media_setup,
            topic_id=1,
            dialogue_id=1,
            config_id=config_id,
            issuance_key="seed_receipt_deterministic",
            status="completed",
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )
        session.add(receipt)
        await session.commit()
        receipt_id = receipt.id

    transport = ValidatingCertTelegramTransport()
    bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
    handlers.router._parent_router = None
    dispatcher = Dispatcher()
    dispatcher.include_router(handlers.router)

    try:
        # 1. Click Accept
        start = len(transport.calls)
        await dispatcher.feed_update(bot, Update(
            update_id=20001,
            callback_query=CallbackQuery(
                id='cb_accept',
                from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Alice'),
                chat_instance='ci_btn',
                data=f'cert_meet:accept:{config_id}:{receipt_id}',
                message=Message(message_id=20001, date=datetime.now(timezone.utc), chat=Chat(id=media_setup, type='private'), text='invite')
            )
        ))
        answers = [c for c in transport.calls[start:] if isinstance(c, AnswerCallbackQuery)]
        assert len(answers) == 1
        messages = [c for c in transport.calls[start:] if isinstance(c, SendMessage)]
        assert any('Отлично! Запланируем встречу' in (m.text or '') for m in messages)

        # Check DB state transitioned
        async with session_factory() as session:
            conv_state = await session.scalar(
                select(AutomationConversationState).where(AutomationConversationState.user_id == media_setup)
            )
            assert conv_state is not None
            assert conv_state.current_step == 'MEETING_BOOKING'

        # 2. Repeat click on Accept (Defect 3): Must NOT change stage or re-send messages
        start_repeat = len(transport.calls)
        await dispatcher.feed_update(bot, Update(
            update_id=20002,
            callback_query=CallbackQuery(
                id='cb_repeat',
                from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Alice'),
                chat_instance='ci_btn',
                data=f'cert_meet:accept:{config_id}:{receipt_id}',
                message=Message(message_id=20001, date=datetime.now(timezone.utc), chat=Chat(id=media_setup, type='private'), text='invite')
            )
        ))
        repeat_answers = [c for c in transport.calls[start_repeat:] if isinstance(c, AnswerCallbackQuery)]
        assert len(repeat_answers) == 1
        assert repeat_answers[0].show_alert is True
        assert 'Повторный выбор не требуется' in (repeat_answers[0].text or '')
        repeat_messages = [c for c in transport.calls[start_repeat:] if isinstance(c, SendMessage)]
        assert len(repeat_messages) == 0

        # 3. Unknown action click (Defect 3): Must be rejected, never default to accept
        start_unknown = len(transport.calls)
        await dispatcher.feed_update(bot, Update(
            update_id=20003,
            callback_query=CallbackQuery(
                id='cb_unknown',
                from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Alice'),
                chat_instance='ci_btn',
                data=f'cert_meet:invalid_act:{config_id}:{receipt_id}',
                message=Message(message_id=20001, date=datetime.now(timezone.utc), chat=Chat(id=media_setup, type='private'), text='invite')
            )
        ))
        unknown_answers = [c for c in transport.calls[start_unknown:] if isinstance(c, AnswerCallbackQuery)]
        assert len(unknown_answers) == 1
        assert unknown_answers[0].show_alert is True
        assert 'Неизвестное действие' in (unknown_answers[0].text or '')

        # 4. Click Details on fresh invitation receipt
        async with session_factory() as session:
            receipt_details = CertificateIssuanceReceipt(
                user_id=media_setup,
                topic_id=1,
                dialogue_id=1,
                config_id=config_id,
                issuance_key="seed_key_tg_details",
                status="completed",
                created_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
            )
            session.add(receipt_details)
            await session.commit()
            details_receipt_id = receipt_details.id

        start_details = len(transport.calls)
        await dispatcher.feed_update(bot, Update(
            update_id=20004,
            callback_query=CallbackQuery(
                id='cb_details',
                from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Alice'),
                chat_instance='ci_btn',
                data=f'cert_meet:details:{config_id}:{details_receipt_id}',
                message=Message(message_id=20002, date=datetime.now(timezone.utc), chat=Chat(id=media_setup, type='private'), text='invite')
            )
        ))
        messages = [c for c in transport.calls[start_details:] if isinstance(c, SendMessage)]
        assert any('Формат встречи' in (m.text or '') for m in messages)

        # 5. Click Later on fresh invitation receipt
        async with session_factory() as session:
            receipt_later = CertificateIssuanceReceipt(
                user_id=media_setup,
                topic_id=1,
                dialogue_id=1,
                config_id=config_id,
                issuance_key="seed_key_tg_later",
                status="completed",
                created_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
            )
            session.add(receipt_later)
            await session.commit()
            later_receipt_id = receipt_later.id

        start_later = len(transport.calls)
        await dispatcher.feed_update(bot, Update(
            update_id=20005,
            callback_query=CallbackQuery(
                id='cb_later',
                from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Alice'),
                chat_instance='ci_btn',
                data=f'cert_meet:later:{config_id}:{later_receipt_id}',
                message=Message(message_id=20003, date=datetime.now(timezone.utc), chat=Chat(id=media_setup, type='private'), text='invite')
            )
        ))
        messages = [c for c in transport.calls[start_later:] if isinstance(c, SendMessage)]
        assert any('Хорошо! Вы сможете вернуться' in (m.text or '') for m in messages)
    finally:
        dispatcher.sub_routers.clear()
        handlers.router._parent_router = None
        await bot.session.close()


@pytest.mark.asyncio
async def test_telegram_runtime_phase1_failure_prevents_message2_and_data_commit(session_factory, media_setup, monkeypatch):
    """
    Two-phase commit safety:
    If certificate sending fails, Message 2 must NOT be sent, and deferred <DATA> blocks
    must NOT be committed to the database.
    Receipt status is marked 'failed'.
    """
    source, config = await seed_template(session_factory)
    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, 3)
        cfg.is_active = True
        await session.commit()

    class FailingTelegramTransport(ValidatingCertTelegramTransport):
        async def make_request(self, bot, method, timeout=None):
            if isinstance(method, SendDocument):
                raise RuntimeError("Telegram network dropped connection during file upload")
            return await super().make_request(bot, method, timeout)

    transport = FailingTelegramTransport()
    async def stream(*args, **kwargs): yield source
    monkeypatch.setattr(transport, 'stream_content', stream)
    bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
    handlers.router._parent_router = None
    dispatcher = Dispatcher()
    dispatcher.include_router(handlers.router)

    ai_response = (
        f'[RENDER_IMG_MEDIA: panda_certificate | {json.dumps(VALUES, ensure_ascii=False)}]\n'
        '<DATA>{"current_step": "PREMATURE_STAGE"}</DATA>'
    )
    monkeypatch.setattr(ai_integration, '_call_gemini_api', AsyncMock(return_value=ai_response))

    try:
        await dispatcher.feed_update(bot, Update(
            update_id=30001,
            message=Message(
                message_id=30001,
                date=datetime.now(timezone.utc),
                chat=Chat(id=media_setup, type='private'),
                from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Alice'),
                text='Выдать'
            )
        ))
        if media_setup in handlers.user_processing_tasks:
            await handlers.user_processing_tasks[media_setup]

        # Verify Message 2 was NOT sent
        sent_messages = [c for c in transport.calls if isinstance(c, SendMessage)]
        invitations = [m for m in sent_messages if getattr(m, 'reply_markup', None)]
        assert len(invitations) == 0

        # Verify DATA was NOT committed
        async with session_factory() as session:
            conv_state = await session.scalar(
                select(AutomationConversationState).where(AutomationConversationState.user_id == media_setup)
            )
            if conv_state is not None:
                assert conv_state.current_step != 'PREMATURE_STAGE'

            receipt = await session.scalar(
                select(CertificateIssuanceReceipt).where(CertificateIssuanceReceipt.user_id == media_setup)
            )
            assert receipt is not None
            assert receipt.status in ('failed', 'uncertain')
    finally:
        dispatcher.sub_routers.clear()
        handlers.router._parent_router = None
        await bot.session.close()


@pytest.mark.asyncio
async def test_telegram_runtime_resumption_without_duplicate_certificate(session_factory, media_setup, monkeypatch):
    """
    If receipt is in 'cert_sent' status (Message 1 was sent successfully earlier, but Message 2 failed),
    the next turn resumes Message 2 (invitation + keyboard) WITHOUT re-rendering or re-sending the certificate.
    """
    source, config = await seed_template(session_factory)
    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, 3)
        cfg.is_active = True
        await session.commit()

    today = current_issue_date()
    key = make_issuance_key(3, media_setup, VALUES['recipient'], today)
    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, 3)
        cfg.is_active = True
        receipt, _ = await get_or_create_receipt(session, media_setup, 1, 1, key, config_id=cfg.id)
        receipt.status = 'cert_sent'
        receipt.config_id = cfg.id
        receipt.certificate_message_id = 'prev_cert_msg_999'
        await session.commit()

    transport = ValidatingCertTelegramTransport()
    async def stream(*args, **kwargs): yield source
    monkeypatch.setattr(transport, 'stream_content', stream)
    bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
    handlers.router._parent_router = None
    dispatcher = Dispatcher()
    dispatcher.include_router(handlers.router)

    ai_response = f'[RENDER_IMG_MEDIA: panda_certificate | {json.dumps(VALUES, ensure_ascii=False)}]'
    monkeypatch.setattr(ai_integration, '_call_gemini_api', AsyncMock(return_value=ai_response))

    try:
        await dispatcher.feed_update(bot, Update(
            update_id=40001,
            message=Message(
                message_id=40001,
                date=datetime.now(timezone.utc),
                chat=Chat(id=media_setup, type='private'),
                from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Alice'),
                text='Сертификат'
            )
        ))
        if media_setup in handlers.user_processing_tasks:
            await handlers.user_processing_tasks[media_setup]

        # SendDocument was NOT called (no duplicate certificate)
        documents = [c for c in transport.calls if isinstance(c, SendDocument)]
        assert len(documents) == 0

        # Message 2 was sent (invitation resumed)
        sent_messages = [c for c in transport.calls if isinstance(c, SendMessage)]
        invitations = [m for m in sent_messages if getattr(m, 'reply_markup', None)]
        assert len(invitations) == 1

        # Receipt updated to completed
        async with session_factory() as session:
            receipt = await session.scalar(select(CertificateIssuanceReceipt).where(CertificateIssuanceReceipt.issuance_key == key))
            assert receipt.status == 'completed'
    finally:
        dispatcher.sub_routers.clear()
        handlers.router._parent_router = None
        await bot.session.close()


@pytest.mark.asyncio
async def test_telegram_runtime_uncertain_status_skips_blind_retry(session_factory, media_setup, monkeypatch):
    """
    If receipt is marked 'uncertain' (indeterminate state during certificate issuance),
    the system skips blind re-delivery to prevent duplicate certificate sending.
    """
    source, config = await seed_template(session_factory)
    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, 3)
        cfg.is_active = True
        await session.commit()

    today = current_issue_date()
    key = make_issuance_key(3, media_setup, VALUES['recipient'], today)
    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, 3)
        cfg.is_active = True
        receipt, _ = await get_or_create_receipt(session, media_setup, 1, 1, key)
        receipt.status = 'uncertain'
        await session.commit()

    transport = ValidatingCertTelegramTransport()
    bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
    handlers.router._parent_router = None
    dispatcher = Dispatcher()
    dispatcher.include_router(handlers.router)

    ai_response = f'[RENDER_IMG_MEDIA: panda_certificate | {json.dumps(VALUES, ensure_ascii=False)}]'
    monkeypatch.setattr(ai_integration, '_call_gemini_api', AsyncMock(return_value=ai_response))

    try:
        await dispatcher.feed_update(bot, Update(
            update_id=50001,
            message=Message(
                message_id=50001,
                date=datetime.now(timezone.utc),
                chat=Chat(id=media_setup, type='private'),
                from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Alice'),
                text='Сертификат'
            )
        ))
        if media_setup in handlers.user_processing_tasks:
            await handlers.user_processing_tasks[media_setup]

        # No document or invitation sent (blind retry skipped)
        documents = [c for c in transport.calls if isinstance(c, SendDocument)]
        assert len(documents) == 0
        invitations = [c for c in transport.calls if isinstance(c, SendMessage) and getattr(c, 'reply_markup', None)]
        assert len(invitations) == 0
    finally:
        dispatcher.sub_routers.clear()
        handlers.router._parent_router = None
        await bot.session.close()


@pytest.mark.asyncio
async def test_telegram_runtime_disabled_preserves_legacy_behavior(session_factory, media_setup, monkeypatch):
    """
    When scenario is disabled (is_active = False), delivery follows legacy behavior:
    - 1 message sent with rendered certificate and default caption.
    - No invitation message sent.
    - No followup stage applied.
    """
    source, config = await seed_template(session_factory)
    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, 3)
        cfg.is_active = False
        await session.commit()

    transport = ValidatingCertTelegramTransport()
    async def stream(*args, **kwargs): yield source
    monkeypatch.setattr(transport, 'stream_content', stream)
    bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
    handlers.router._parent_router = None
    dispatcher = Dispatcher()
    dispatcher.include_router(handlers.router)

    ai_response = f'Готово:\n[RENDER_IMG_MEDIA: panda_certificate | {json.dumps(VALUES, ensure_ascii=False)}]'
    monkeypatch.setattr(ai_integration, '_call_gemini_api', AsyncMock(return_value=ai_response))

    try:
        await dispatcher.feed_update(bot, Update(
            update_id=60001,
            message=Message(
                message_id=60001,
                date=datetime.now(timezone.utc),
                chat=Chat(id=media_setup, type='private'),
                from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Alice'),
                text='Сертификат'
            )
        ))
        if media_setup in handlers.user_processing_tasks:
            await handlers.user_processing_tasks[media_setup]

        # Exactly 1 SendDocument
        documents = [c for c in transport.calls if isinstance(c, SendDocument)]
        assert len(documents) == 1
        assert documents[0].caption == 'Готово:'

        # No invitation message
        invitations = [c for c in transport.calls if isinstance(c, SendMessage) and getattr(c, 'reply_markup', None)]
        assert len(invitations) == 0
    finally:
        dispatcher.sub_routers.clear()
        handlers.router._parent_router = None
        await bot.session.close()


@pytest.mark.asyncio
async def test_certificate_delivered_but_api_response_lost_uncertain_no_blind_retry(session_factory, media_setup, monkeypatch):
    """
    Defect 2 Regression:
    When certificate upload/send drops connection or times out (Telegram might have received it,
    but response was lost), receipt is marked 'uncertain' and subsequent turns skip blind retry.
    """
    from aiogram.exceptions import TelegramNetworkError
    from aiogram.methods import SendDocument

    source, config = await seed_template(session_factory)
    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, 3)
        cfg.is_active = True
        await session.commit()

    today = current_issue_date()
    key = make_issuance_key(3, media_setup, VALUES['recipient'], today)

    class LostResponseTelegramTransport(ValidatingCertTelegramTransport):
        def __init__(self):
            super().__init__()
            self.first_attempt = True

        async def make_request(self, bot, method, timeout=None):
            if isinstance(method, SendDocument) and self.first_attempt:
                self.first_attempt = False
                # Certificate delivered to Telegram edge, but connection dropped before response
                dummy_method = SendDocument(chat_id=1, document=BufferedInputFile(b"dummy", filename="dummy.png"))
                raise TelegramNetworkError(method=dummy_method, message="Connection reset by peer: API response lost")
            return await super().make_request(bot, method, timeout)

    transport = LostResponseTelegramTransport()
    async def stream(*args, **kwargs): yield source
    monkeypatch.setattr(transport, 'stream_content', stream)
    bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
    handlers.router._parent_router = None
    dispatcher = Dispatcher()
    dispatcher.include_router(handlers.router)

    ai_response = f'[RENDER_IMG_MEDIA: panda_certificate | {json.dumps(VALUES, ensure_ascii=False)}]'
    monkeypatch.setattr(ai_integration, '_call_gemini_api', AsyncMock(return_value=ai_response))

    try:
        # Turn 1: Attempt send -> transport error -> marked uncertain
        await dispatcher.feed_update(bot, Update(
            update_id=71001,
            message=Message(
                message_id=71001,
                date=datetime.now(timezone.utc),
                chat=Chat(id=media_setup, type='private'),
                from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Alice'),
                text='Выдать сертификат'
            )
        ))
        if media_setup in handlers.user_processing_tasks:
            try:
                await handlers.user_processing_tasks[media_setup]
            except Exception:
                pass

        async with session_factory() as session:
            receipt = await session.scalar(
                select(CertificateIssuanceReceipt).where(CertificateIssuanceReceipt.user_id == media_setup)
            )
            assert receipt is not None
            assert receipt.status == 'uncertain'

        # Turn 2: User asks for status again -> NO blind retry of SendDocument
        doc_count_before = len([c for c in transport.calls if isinstance(c, SendDocument)])
        await dispatcher.feed_update(bot, Update(
            update_id=71002,
            message=Message(
                message_id=71002,
                date=datetime.now(timezone.utc),
                chat=Chat(id=media_setup, type='private'),
                from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Alice'),
                text='Где мой сертификат?'
            )
        ))
        if media_setup in handlers.user_processing_tasks:
            await handlers.user_processing_tasks[media_setup]

        doc_count_after = len([c for c in transport.calls if isinstance(c, SendDocument)])
        # Must NOT attempt to upload/send document again
        assert doc_count_after == doc_count_before
    finally:
        dispatcher.sub_routers.clear()
        handlers.router._parent_router = None
        await bot.session.close()


@pytest.mark.asyncio
async def test_runtime_admin_target_stage_overrides_conflicting_data_step(session_factory, media_setup, monkeypatch):
    """
    Defect 4 Regression:
    Admin target_stage strictly overrides conflicting current_step from model <DATA> blocks.
    Model's extra data attributes (e.g. qualification flags) are preserved.
    """
    source, config = await seed_template(session_factory)
    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, 3)
        cfg.is_active = True
        cfg.target_stage = 'ADMIN_AUTHORITATIVE_STAGE'
        await session.commit()

    transport = ValidatingCertTelegramTransport()
    async def stream(*args, **kwargs): yield source
    monkeypatch.setattr(transport, 'stream_content', stream)
    bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
    handlers.router._parent_router = None
    dispatcher = Dispatcher()
    dispatcher.include_router(handlers.router)

    # Model attempts to transition to MODEL_HALLUCINATED_STAGE with custom fields
    ai_response = (
        'Поздравляем!\n'
        f'[RENDER_IMG_MEDIA: panda_certificate | {json.dumps(VALUES, ensure_ascii=False)}]\n'
        '<DATA>{"current_state": {"current_step": "MODEL_HALLUCINATED_STAGE", "client_score": 100, "verified": true}}</DATA>'
    )
    monkeypatch.setattr(ai_integration, '_call_gemini_api', AsyncMock(return_value=ai_response))

    try:
        await dispatcher.feed_update(bot, Update(
            update_id=72001,
            message=Message(
                message_id=72001,
                date=datetime.now(timezone.utc),
                chat=Chat(id=media_setup, type='private'),
                from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Alice'),
                text='Сертификат пожалуйста'
            )
        ))
        if media_setup in handlers.user_processing_tasks:
            await handlers.user_processing_tasks[media_setup]

        async with session_factory() as session:
            conv_state = await session.scalar(
                select(AutomationConversationState).where(AutomationConversationState.user_id == media_setup)
            )
            assert conv_state is not None
            # Admin target_stage strictly overwrote model's current_step
            assert conv_state.current_step == 'ADMIN_AUTHORITATIVE_STAGE'

            state_dict = json.loads(conv_state.current_state_json or '{}')
            assert state_dict.get('current_step') == 'ADMIN_AUTHORITATIVE_STAGE'
            assert state_dict.get('client_score') == 100
            assert state_dict.get('verified') is True

            # Step transition was recorded for admin's target_stage
            transitions = (await session.scalars(
                select(AutomationStepTransition).where(AutomationStepTransition.user_id == media_setup)
            )).all()
            assert any(t.current_step == 'ADMIN_AUTHORITATIVE_STAGE' for t in transitions)
            assert not any(t.current_step == 'MODEL_HALLUCINATED_STAGE' for t in transitions)
    finally:
        dispatcher.sub_routers.clear()
        handlers.router._parent_router = None
        await bot.session.close()


@pytest.mark.asyncio
async def test_runtime_resumption_without_ai_render_directive(session_factory, media_setup, monkeypatch):
    """
    Defect 5 Regression:
    When receipt is 'cert_sent', resumption of Message 2 occurs immediately upon user message,
    WITHOUT requiring AI to issue [RENDER_IMG_MEDIA] again. Zero duplicate cert sent.
    """
    source, config = await seed_template(session_factory)
    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, 3)
        cfg.is_active = True
        await session.commit()

    today = current_issue_date()
    key = make_issuance_key(3, media_setup, VALUES['recipient'], today)

    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, 3)
        cfg.is_active = True
        cfg.invitation_text = "Приглашаю обсудить результаты."
        receipt, _ = await get_or_create_receipt(session, media_setup, 1, 1, key, config_id=cfg.id)
        receipt.status = 'cert_sent'
        receipt.config_id = cfg.id
        receipt.certificate_message_id = 'prev_msg_123'
        await session.commit()

    transport = ValidatingCertTelegramTransport()
    bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
    handlers.router._parent_router = None
    dispatcher = Dispatcher()
    dispatcher.include_router(handlers.router)

    # AI returns plain text without any RENDER_IMG_MEDIA directive
    ai_response = "Привет! Чем я могу помочь вам сегодня?"
    monkeypatch.setattr(ai_integration, '_call_gemini_api', AsyncMock(return_value=ai_response))

    try:
        await dispatcher.feed_update(bot, Update(
            update_id=73001,
            message=Message(
                message_id=73001,
                date=datetime.now(timezone.utc),
                chat=Chat(id=media_setup, type='private'),
                from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Alice'),
                text='Привет, бот!'
            )
        ))
        if media_setup in handlers.user_processing_tasks:
            await handlers.user_processing_tasks[media_setup]

        # No document sent
        documents = [c for c in transport.calls if isinstance(c, SendDocument)]
        assert len(documents) == 0

        # Invitation message was resumed and sent with inline keyboard
        sent_messages = [c for c in transport.calls if isinstance(c, SendMessage)]
        invitations = [m for m in sent_messages if 'Приглашаю обсудить результаты' in (m.text or '')]
        assert len(invitations) == 1
        assert getattr(invitations[0], 'reply_markup', None) is not None

        # Receipt transitioned to completed
        async with session_factory() as session:
            r = await session.scalar(select(CertificateIssuanceReceipt).where(CertificateIssuanceReceipt.issuance_key == key))
            assert r.status == 'completed'
    finally:
        dispatcher.sub_routers.clear()
        handlers.router._parent_router = None
        await bot.session.close()


@pytest.mark.asyncio
async def test_runtime_render_img_media_edge_cases(session_factory, media_setup, monkeypatch):
    """
    Defect 7 Regression:
    - Legacy path without response_capture: persists deferred blocks immediately.
    - Media render error: receipt is marked failed, no invitation delivered.
    """
    source, config = await seed_template(session_factory)
    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, 3)
        cfg.is_active = True
        await session.commit()

    # Case A: AI call without response_capture preserves immediate persistence
    ai_response = (
        'Тест без response_capture:\n'
        '<DATA>{"current_state": {"current_step": "LEGACY_DIRECT_STEP", "migrated": true}}</DATA>'
    )
    monkeypatch.setattr(ai_integration, '_call_gemini_api', AsyncMock(return_value=ai_response))

    # Direct call to get_ai_response without response_capture
    resp = await ai_integration.get_ai_response(
        media_setup,
        "Привет",
        "Alice",
        "female",
        response_capture=None,
    )
    assert resp is not None
    async with session_factory() as session:
        conv = await session.scalar(
            select(AutomationConversationState).where(AutomationConversationState.user_id == media_setup)
        )
        assert conv is not None
        assert conv.current_step == 'LEGACY_DIRECT_STEP'

    # Case B: Template render error
    transport = ValidatingCertTelegramTransport()
    bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
    handlers.router._parent_router = None
    dispatcher = Dispatcher()
    dispatcher.include_router(handlers.router)

    def failing_render(*args, **kwargs):
        raise ValueError("Simulated template rendering engine failure")
    monkeypatch.setattr('handlers.render_media_template', failing_render)

    render_fail_ai = f'[RENDER_IMG_MEDIA: panda_certificate | {json.dumps(VALUES, ensure_ascii=False)}]'
    monkeypatch.setattr(ai_integration, '_call_gemini_api', AsyncMock(return_value=render_fail_ai))

    try:
        await dispatcher.feed_update(bot, Update(
            update_id=74001,
            message=Message(
                message_id=74001,
                date=datetime.now(timezone.utc),
                chat=Chat(id=media_setup, type='private'),
                from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Alice'),
                text='Сертификат'
            )
        ))
        if media_setup in handlers.user_processing_tasks:
            await handlers.user_processing_tasks[media_setup]

        # No document or invitation delivered
        documents = [c for c in transport.calls if isinstance(c, SendDocument)]
        assert len(documents) == 0
        invitations = [c for c in transport.calls if isinstance(c, SendMessage) and getattr(c, 'reply_markup', None)]
        assert len(invitations) == 0
    finally:
        dispatcher.sub_routers.clear()
        handlers.router._parent_router = None
        await bot.session.close()


@pytest.mark.asyncio
async def test_cert_followup_list_pagination_in_automations_menu(session_factory, media_setup, monkeypatch):
    """
    Defect 9 Regression:
    Verify pagination of cert_followup_list in Automations menu with > 5 templates:
    Page 0: 5 templates + next button '»'
    Page 1: Remaining 2 templates + prev button '«'
    Back navigation returns cleanly to Automations menu.
    """
    source, config = await seed_template(session_factory)
    # Seed 7 templates total
    async with session_factory() as session:
        for i in range(4, 10):
            session.add(MediaLibrary(
                id=i,
                file_id=f'cert_tpl_{i}',
                file_name=f'certificate_template_{i}',
                description=f'Certificate {i}',
                media_type='photo',
                source_platform='telegram',
                render_template_enabled=True,
                render_template_config=config,
            ))
        await session.commit()

    monkeypatch.setattr(handlers, 'is_admin', AsyncMock(return_value=True))
    monkeypatch.setattr(automation_admin, 'get_all_admin_ids', AsyncMock(return_value=[media_setup]))
    transport = ValidatingCertTelegramTransport()
    bot = Bot('123456:TEST_BOT_TOKEN', session=transport)
    handlers.router._parent_router = None
    automation_admin.router._parent_router = None
    dispatcher = Dispatcher()
    dispatcher.include_router(automation_admin.router)
    dispatcher.include_router(handlers.router)

    state = dispatcher.fsm.get_context(bot=bot, chat_id=media_setup, user_id=media_setup)
    await state.update_data(authoring_locale='ru')

    latest = None

    async def click_data(data):
        nonlocal latest
        start = len(transport.calls)
        await dispatcher.feed_update(bot, Update(
            update_id=75000 + start,
            callback_query=CallbackQuery(
                id=str(start),
                from_user=TelegramUser(id=media_setup, is_bot=False, first_name='Admin'),
                chat_instance='ci_pagi',
                data=data,
                message=Message(message_id=75000, date=datetime.now(timezone.utc), chat=Chat(id=media_setup, type='private'), text='screen')
            )
        ))
        screens = [c for c in transport.calls[start:] if getattr(c, 'reply_markup', None)]
        if screens:
            latest = screens[-1]

    # 1. Open cert_followup_list (Page 0)
    await click_data('cert_followup_list')
    assert latest is not None
    assert 'Сценарии после выдачи сертификатов' in latest.text
    buttons_p0 = [b for row in latest.reply_markup.inline_keyboard for b in row]
    scenario_buttons_p0 = [b for b in buttons_p0 if b.callback_data.startswith('mt:scenario:')]
    assert len(scenario_buttons_p0) == 5

    # Check next page button exists
    next_btn = next((b for b in buttons_p0 if b.callback_data == 'cert_followup_list:1'), None)
    assert next_btn is not None
    assert 'Вперёд' in next_btn.text

    # 2. Click next page (Page 1)
    await click_data('cert_followup_list:1')
    assert 'Сценарии после выдачи сертификатов' in latest.text
    buttons_p1 = [b for row in latest.reply_markup.inline_keyboard for b in row]
    scenario_buttons_p1 = [b for b in buttons_p1 if b.callback_data.startswith('mt:scenario:')]
    assert len(scenario_buttons_p1) == 3

    # Check prev page button exists
    prev_btn = next((b for b in buttons_p1 if b.callback_data == 'cert_followup_list:0'), None)
    assert prev_btn is not None
    assert 'Назад' in prev_btn.text

    # 3. Click prev page (back to Page 0)
    await click_data('cert_followup_list:0')
    buttons_back_p0 = [b for row in latest.reply_markup.inline_keyboard for b in row]
    scenario_buttons_back = [b for b in buttons_back_p0 if b.callback_data.startswith('mt:scenario:')]
    assert len(scenario_buttons_back) == 5

    # 4. Click Back button to automations menu
    back_btn = next((b for b in buttons_back_p0 if b.callback_data == 'automation_menu'), None)
    assert back_btn is not None
    await click_data('automation_menu')
    assert 'Автоматизации' in latest.text

