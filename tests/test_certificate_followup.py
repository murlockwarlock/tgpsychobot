import json
import pytest
from datetime import datetime, timezone
from sqlalchemy import select

from database import (
    AutomationConversationState,
    AutomationStepTransition,
    CertificateFollowupConfig,
    CertificateIssuanceReceipt,
    MediaLibrary,
    User,
    async_session_maker,
)
from certificate_followup_service import (
    DEFAULT_BUTTONS_CONFIG,
    DEFAULT_CAPTION_TEMPLATE,
    DEFAULT_INVITATION_TEXT,
    DEFAULT_TARGET_STAGE,
    build_followup_telegram_keyboard,
    build_followup_max_keyboard,
    get_followup_config,
    get_or_create_followup_config,
    get_or_create_receipt,
    handle_cert_meet_click,
    make_issuance_key,
    mark_receipt_cert_sent,
    mark_receipt_completed,
    mark_receipt_failed,
    mark_receipt_uncertain,
    parse_buttons_config,
    render_caption,
    render_invitation,
    save_followup_caption,
    save_followup_invitation,
    save_followup_stage,
    toggle_followup_config,
    transition_user_to_stage,
    update_followup_button,
    validate_caption_template,
    validate_invitation_text,
    validate_stage_identifier,
)
from tests.test_media_image_edit import async_engine, clean_runtime, session_factory, setup_tg_user


def test_validate_caption_template_valid_and_invalid():
    # Valid templates
    validate_caption_template("Поздравляем, <b>{name}</b>! Дата: <code>{date}</code>. ФИО: {fio}")
    validate_caption_template("{recipient}, ваш сертификат готов!")

    # Empty
    with pytest.raises(ValueError, match="не может быть пустой"):
        validate_caption_template("")
    with pytest.raises(ValueError, match="не может быть пустой"):
        validate_caption_template("   ")

    # Unknown variable
    with pytest.raises(ValueError, match="Неизвестная переменная {promo}"):
        validate_caption_template("Поздравляем, {name}! Ваш промокод: {promo}")

    # Invalid / unclosed HTML
    with pytest.raises(ValueError):
        validate_caption_template("<b>Поздравляем, {name}!")

    # Unsupported tag
    with pytest.raises(ValueError):
        validate_caption_template("<script>alert(1)</script> {name}")


def test_validate_invitation_text_valid_and_invalid():
    # Valid
    validate_invitation_text("Здравствуйте, <b>{name}</b>! Предлагаем встретиться.")
    validate_invitation_text("Приглашаем на встречу для обсуждения шагов.")

    # Empty
    with pytest.raises(ValueError, match="не может быть пустым"):
        validate_invitation_text("")

    # Unknown variable in invitation
    with pytest.raises(ValueError, match="Неизвестная переменная {date}"):
        validate_invitation_text("Ждем вас {date}!")

    # Unclosed tag
    with pytest.raises(ValueError):
        validate_invitation_text("<i>Встреча {name}")


def test_validate_stage_identifier():
    assert validate_stage_identifier("OFFER_MEETING") == "OFFER_MEETING"
    assert validate_stage_identifier("step_123") == "step_123"
    assert validate_stage_identifier("STAGE-A") == "STAGE-A"

    with pytest.raises(ValueError, match="не может быть пустым"):
        validate_stage_identifier("")

    with pytest.raises(ValueError, match="не должно превышать 64"):
        validate_stage_identifier("A" * 65)

    with pytest.raises(ValueError, match="только латинские буквы"):
        validate_stage_identifier("СТАДИЯ_1")

    with pytest.raises(ValueError, match="только латинские буквы"):
        validate_stage_identifier("stage with spaces")


def test_render_caption_and_invitation():
    ctx = {
        "name": "Анна",
        "fio": "Анна Иванова",
        "date": "09.11.2026",
    }
    cap = render_caption("Выдан {fio} ({name}) от {date}", ctx)
    assert cap == "Выдан Анна Иванова (Анна) от 09.11.2026"

    inv = render_invitation("Привет, {name}! Встретимся?", ctx)
    assert inv == "Привет, Анна! Встретимся?"


def test_parse_buttons_config_resilience():
    # Defaults on None or empty
    assert parse_buttons_config(None) == DEFAULT_BUTTONS_CONFIG
    assert parse_buttons_config("") == DEFAULT_BUTTONS_CONFIG
    assert parse_buttons_config("broken json") == DEFAULT_BUTTONS_CONFIG

    # Valid custom json
    custom = [
        {"id": "accept", "text": "Да", "target_step": "STEP_YES", "response_text": "Отлично"},
        {"id": "later", "text": "Нет", "target_step": "STEP_NO", "response_text": "Жаль"},
    ]
    parsed = parse_buttons_config(json.dumps(custom))
    assert len(parsed) == 2
    assert parsed[0]["id"] == "accept"
    assert parsed[1]["target_step"] == "STEP_NO"


def test_make_issuance_key_deterministic():
    k1 = make_issuance_key(10, 1001, "Иван Иванов", "09.11.2026")
    k2 = make_issuance_key(10, 1001, "  иван иванов  ", "09.11.2026")
    k3 = make_issuance_key(10, 1002, "Иван Иванов", "09.11.2026")
    assert k1 == k2
    assert k1 != k3
    assert len(k1) == 32


@pytest.mark.asyncio
async def test_followup_config_crud_and_toggle(session_factory, setup_tg_user):
    user_id = setup_tg_user
    async with session_factory() as session:
        media = MediaLibrary(
            file_id="test_cert_file",
            media_type="photo",
            file_name="diploma.png",
            render_template_enabled=True,
        )
        session.add(media)
        await session.commit()
        media_id = media.id

    # 1. get_or_create
    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, media_id)
        await session.commit()
        assert cfg.media_id == media_id
        assert cfg.is_active is True
        assert cfg.caption_template == DEFAULT_CAPTION_TEMPLATE
        assert cfg.target_stage == DEFAULT_TARGET_STAGE

    # 2. toggle
    async with session_factory() as session:
        active = await toggle_followup_config(session, media_id)
        assert active is False
        cfg = await get_followup_config(session, media_id)
        assert cfg.is_active is False

        active = await toggle_followup_config(session, media_id)
        assert active is True

    # 3. save caption
    async with session_factory() as session:
        await save_followup_caption(session, media_id, "Новая подпись для {name}")
        cfg = await get_followup_config(session, media_id)
        assert cfg.caption_template == "Новая подпись для {name}"

    # 4. save invitation
    async with session_factory() as session:
        await save_followup_invitation(session, media_id, "Приглашаем {name} на встречу")
        cfg = await get_followup_config(session, media_id)
        assert cfg.invitation_text == "Приглашаем {name} на встречу"

    # 5. save stage
    async with session_factory() as session:
        await save_followup_stage(session, media_id, "STAGE_FOLLOWUP")
        cfg = await get_followup_config(session, media_id)
        assert cfg.target_stage == "STAGE_FOLLOWUP"

    # 6. update button
    async with session_factory() as session:
        await update_followup_button(session, media_id, 0, "Записаться", "BOOK_STAGE", "Записываем!")
        cfg = await get_followup_config(session, media_id)
        btns = parse_buttons_config(cfg.buttons_config_json)
        assert btns[0]["text"] == "Записаться"
        assert btns[0]["target_step"] == "BOOK_STAGE"
        assert btns[0]["response_text"] == "Записываем!"


@pytest.mark.asyncio
async def test_receipt_lifecycle_and_idempotency(session_factory, setup_tg_user):
    user_id = setup_tg_user
    key = make_issuance_key(5, user_id, "Петр", "10.11.2026")

    async with session_factory() as session:
        receipt, is_new = await get_or_create_receipt(session, user_id, 0, 1, key)
        await session.commit()
        assert is_new is True
        assert receipt.status == "started"
        receipt_id = receipt.id

    # Second check returns existing
    async with session_factory() as session:
        receipt, is_new = await get_or_create_receipt(session, user_id, 0, 1, key)
        assert is_new is False
        assert receipt.id == receipt_id

    # Transitions
    async with session_factory() as session:
        await mark_receipt_cert_sent(session, receipt_id, message_id="msg_101")
        r = await session.get(CertificateIssuanceReceipt, receipt_id)
        assert r.status == "cert_sent"
        assert r.certificate_message_id == "msg_101"

    async with session_factory() as session:
        await mark_receipt_completed(session, receipt_id, invitation_message_id="msg_102")
        r = await session.get(CertificateIssuanceReceipt, receipt_id)
        assert r.status == "completed"
        assert r.invitation_message_id == "msg_102"
        assert r.completed_at is not None

    # Error transitions
    async with session_factory() as session:
        r2, _ = await get_or_create_receipt(session, user_id, 0, 1, "another_key")
        await session.commit()
        await mark_receipt_failed(session, r2.id, "Network timeout")
        loaded = await session.get(CertificateIssuanceReceipt, r2.id)
        assert loaded.status == "failed"
        assert "Network timeout" in loaded.error_details

        await mark_receipt_uncertain(session, r2.id, "Uncertain status")
        loaded = await session.get(CertificateIssuanceReceipt, r2.id)
        assert loaded.status == "uncertain"


@pytest.mark.asyncio
async def test_transition_user_to_stage_and_history(session_factory, setup_tg_user):
    user_id = setup_tg_user

    async with session_factory() as session:
        # Initial transition
        await transition_user_to_stage(session, user_id, topic_id=0, dialogue_id=1, target_step="STAGE_1")
        await session.commit()

        conv = await session.scalar(
            select(AutomationConversationState).where(
                AutomationConversationState.user_id == user_id,
                AutomationConversationState.topic_id == 0,
                AutomationConversationState.dialogue_id == 1,
            )
        )
        assert conv.current_step == "STAGE_1"

        trans = (await session.scalars(select(AutomationStepTransition).where(AutomationStepTransition.user_id == user_id))).all()
        assert len(trans) == 1
        assert trans[0].previous_step is None
        assert trans[0].current_step == "STAGE_1"

        # Subsequent transition
        await transition_user_to_stage(session, user_id, topic_id=0, dialogue_id=1, target_step="STAGE_2")
        await session.commit()

        conv = await session.scalar(
            select(AutomationConversationState).where(
                AutomationConversationState.user_id == user_id,
                AutomationConversationState.topic_id == 0,
                AutomationConversationState.dialogue_id == 1,
            )
        )
        assert conv.current_step == "STAGE_2"

        trans = (await session.scalars(select(AutomationStepTransition).where(AutomationStepTransition.user_id == user_id).order_by(AutomationStepTransition.id.asc()))).all()
        assert len(trans) == 2
        assert trans[1].previous_step == "STAGE_1"
        assert trans[1].current_step == "STAGE_2"


@pytest.mark.asyncio
async def test_handle_cert_meet_click_deterministic(session_factory, setup_tg_user):
    user_id = setup_tg_user

    async with session_factory() as session:
        media = MediaLibrary(file_id="f1", media_type="photo", file_name="cert.png")
        session.add(media)
        await session.flush()
        cfg = await get_or_create_followup_config(session, media.id)
        await session.commit()
        config_id = cfg.id

    # Click accept
    resp, target = await handle_cert_meet_click(
        session_factory,
        user_id=user_id,
        topic_id=0,
        dialogue_id=1,
        action_id="accept",
        config_id=config_id,
    )
    assert target == "MEETING_BOOKING"
    assert "Запланируем" in resp

    async with session_factory() as session:
        conv = await session.scalar(
            select(AutomationConversationState).where(AutomationConversationState.user_id == user_id)
        )
        assert conv.current_step == "MEETING_BOOKING"

    # Click later
    resp, target = await handle_cert_meet_click(
        session_factory,
        user_id=user_id,
        topic_id=0,
        dialogue_id=1,
        action_id="later",
        config_id=config_id,
    )
    assert target == "MEETING_POSTPONED"
    assert "в любой момент" in resp

    async with session_factory() as session:
        conv = await session.scalar(
            select(AutomationConversationState).where(AutomationConversationState.user_id == user_id)
        )
        assert conv.current_step == "MEETING_POSTPONED"


def test_build_keyboards():
    cfg = CertificateFollowupConfig(
        id=42,
        media_id=1,
        buttons_config_json=json.dumps(DEFAULT_BUTTONS_CONFIG),
    )
    tg_kb = build_followup_telegram_keyboard(cfg)
    assert len(tg_kb.inline_keyboard) == 3
    assert tg_kb.inline_keyboard[0][0].callback_data == "cert_meet:accept:42"
    assert tg_kb.inline_keyboard[1][0].callback_data == "cert_meet:details:42"
    assert tg_kb.inline_keyboard[2][0].callback_data == "cert_meet:later:42"

    max_kb = build_followup_max_keyboard(cfg)
    assert isinstance(max_kb, list)
    assert max_kb[0]["type"] == "inline_keyboard"
    btns = max_kb[0]["payload"]["buttons"]
    assert len(btns) == 3
    assert btns[0][0]["payload"] == "cert_meet:accept:42"

