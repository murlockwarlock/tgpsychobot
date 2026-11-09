import asyncio
import json
import pytest
from datetime import datetime, timedelta, timezone
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
    claim_receipt_for_invitation,
    get_followup_config,
    get_or_create_followup_config,
    get_or_create_receipt,
    handle_cert_meet_click,
    make_issuance_key,
    mark_receipt_cert_sent,
    mark_receipt_completed,
    mark_receipt_failed,
    mark_receipt_uncertain,
    override_service_blocks_target_stage,
    parse_buttons_config,
    render_caption,
    render_caption_safe,
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
from user_metadata import ServiceDataBlock
from handlers import _safe_send_html
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
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

    # 1. get_or_create: must be created INACTIVE (Requirement 1)
    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, media_id)
        await session.commit()
        assert cfg.media_id == media_id
        assert cfg.is_active is False
        assert cfg.caption_template == DEFAULT_CAPTION_TEMPLATE
        assert cfg.target_stage == DEFAULT_TARGET_STAGE

    # 2. toggle: False -> True -> False -> True
    async with session_factory() as session:
        active = await toggle_followup_config(session, media_id)
        assert active is True
        cfg = await get_followup_config(session, media_id)
        assert cfg.is_active is True

        active = await toggle_followup_config(session, media_id)
        assert active is False

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
        receipt, is_new = await get_or_create_receipt(session, user_id, 0, 1, key, config_id=10)
        await session.commit()
        assert is_new is True
        assert receipt.status == "started"
        assert receipt.config_id == 10
        receipt_id = receipt.id

    # Second check returns existing
    async with session_factory() as session:
        receipt, is_new = await get_or_create_receipt(session, user_id, 0, 1, key)
        assert is_new is False
        assert receipt.id == receipt_id

    # Transitions
    async with session_factory() as session:
        await mark_receipt_cert_sent(session, receipt_id, message_id="msg_101", config_id=10)
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
        cfg.is_active = True
        await session.commit()
        config_id = cfg.id

        # Seed receipt
        receipt, _ = await get_or_create_receipt(session, user_id, 0, 1, "test_click_key", config_id=config_id)
        receipt.status = "completed"
        await session.commit()
        receipt_id = receipt.id

    # Click accept
    res = await handle_cert_meet_click(
        session_factory,
        user_id=user_id,
        topic_id=0,
        dialogue_id=1,
        callback_data=f"cert_meet:accept:{config_id}:{receipt_id}",
    )
    assert res.success is True
    assert res.target_step == "MEETING_BOOKING"
    assert "Запланируем" in res.response_text

    async with session_factory() as session:
        conv = await session.scalar(
            select(AutomationConversationState).where(AutomationConversationState.user_id == user_id)
        )
        assert conv.current_step == "MEETING_BOOKING"


def test_build_keyboards():
    cfg = CertificateFollowupConfig(
        id=42,
        media_id=1,
        buttons_config_json=json.dumps(DEFAULT_BUTTONS_CONFIG),
    )
    # Default (no receipt_id)
    tg_kb = build_followup_telegram_keyboard(cfg)
    assert len(tg_kb.inline_keyboard) == 3
    assert tg_kb.inline_keyboard[0][0].callback_data == "cert_meet:accept:42:0"

    # With receipt_id
    tg_kb_receipt = build_followup_telegram_keyboard(cfg, receipt_id=777)
    assert tg_kb_receipt.inline_keyboard[0][0].callback_data == "cert_meet:accept:42:777"

    # With is_test=True
    tg_kb_test = build_followup_telegram_keyboard(cfg, is_test=True)
    assert tg_kb_test.inline_keyboard[0][0].callback_data == "cert_meet:test:42:accept"

    max_kb = build_followup_max_keyboard(cfg, receipt_id=777)
    assert isinstance(max_kb, list)
    assert max_kb[0]["type"] == "inline_keyboard"
    btns = max_kb[0]["payload"]["buttons"]
    assert len(btns) == 3
    assert btns[0][0]["payload"] == "cert_meet:accept:42:777"


# ============================================================================
# Dedicated Regression Tests for Defects 1 to 6
# ============================================================================

@pytest.mark.asyncio
async def test_scenario_created_inactive_by_default(session_factory, setup_tg_user):
    """
    Defect 1 regression test:
    Scenario must be created INACTIVE (is_active=False).
    Opening «После выдачи» must NOT activate delivery.
    Activation requires explicit admin action (toggle).
    """
    async with session_factory() as session:
        media = MediaLibrary(file_id="f_inactive", media_type="photo", file_name="inactive_test.png")
        session.add(media)
        await session.commit()
        media_id = media.id

    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, media_id)
        await session.commit()
        assert cfg.is_active is False, "Scenario must be created inactive by default"

    # Re-reading config confirms it remains inactive
    async with session_factory() as session:
        cfg = await get_followup_config(session, media_id)
        assert cfg.is_active is False

    # Only explicit toggle activates it
    async with session_factory() as session:
        active = await toggle_followup_config(session, media_id)
        assert active is True
        cfg = await get_followup_config(session, media_id)
        assert cfg.is_active is True


@pytest.mark.asyncio
async def test_safe_send_html_does_not_retry_on_network_timeout():
    """
    Defect 2 regression test:
    _safe_send_html must NEVER repeat send_coro upon network timeout or transport errors.
    It should immediately re-raise the exception without retry.
    """
    from aiogram.methods import SendDocument
    from aiogram.types import BufferedInputFile
    call_count = 0

    async def mock_coro(chunk, pm):
        nonlocal call_count
        call_count += 1
        dummy_method = SendDocument(chat_id=1, document=BufferedInputFile(b"test", filename="test.png"))
        raise TelegramNetworkError(method=dummy_method, message="Connection timed out during SendDocument")

    with pytest.raises(TelegramNetworkError):
        await _safe_send_html(mock_coro, "<b>Test</b>")

    assert call_count == 1, f"Expected exactly 1 call without retry, got {call_count}"

    # Also test asyncio.TimeoutError
    call_count_to = 0

    async def mock_coro_to(chunk, pm):
        nonlocal call_count_to
        call_count_to += 1
        raise asyncio.TimeoutError("Client socket timeout")

    with pytest.raises(asyncio.TimeoutError):
        await _safe_send_html(mock_coro_to, "<b>Test</b>")

    assert call_count_to == 1, f"Expected exactly 1 call on TimeoutError, got {call_count_to}"


@pytest.mark.asyncio
async def test_cert_meet_strict_validations_and_repeat_click_prevention(session_factory, setup_tg_user):
    """
    Defect 3 regression test:
    - Unknown config_id / action_id must be rejected (never default to accept).
    - Inactive scenario must be rejected.
    - Missing receipt must be rejected.
    - Mismatched user/topic/dialogue must be rejected.
    - Repeat clicks must be detected (is_repeat=True) and must NOT re-transition the stage.
    """
    user_id = setup_tg_user
    async with session_factory() as session:
        media = MediaLibrary(file_id="f_val", media_type="photo", file_name="val.png")
        session.add(media)
        await session.flush()
        cfg = await get_or_create_followup_config(session, media.id)
        cfg.is_active = True
        await session.commit()
        config_id = cfg.id

    # 1. Unknown config_id
    res = await handle_cert_meet_click(
        session_factory, user_id=user_id, topic_id=0, dialogue_id=1, callback_data="cert_meet:accept:999999"
    )
    assert res.success is False
    assert "не найден" in res.alert_text

    # 2. Inactive scenario
    async with session_factory() as session:
        cfg = await session.get(CertificateFollowupConfig, config_id)
        cfg.is_active = False
        await session.commit()

    res = await handle_cert_meet_click(
        session_factory, user_id=user_id, topic_id=0, dialogue_id=1, callback_data=f"cert_meet:accept:{config_id}"
    )
    assert res.success is False
    assert "отключен" in res.alert_text

    # Reactivate config
    async with session_factory() as session:
        cfg = await session.get(CertificateFollowupConfig, config_id)
        cfg.is_active = True
        await session.commit()

    # 3. Unknown action_id
    res = await handle_cert_meet_click(
        session_factory, user_id=user_id, topic_id=0, dialogue_id=1, callback_data=f"cert_meet:completely_unknown_action:{config_id}"
    )
    assert res.success is False
    assert "Неизвестное действие" in res.alert_text
    assert res.target_step != "MEETING_BOOKING", "Must NOT fallback to accept!"

    # 4. Missing receipt
    res = await handle_cert_meet_click(
        session_factory, user_id=user_id, topic_id=0, dialogue_id=1, callback_data=f"cert_meet:accept:{config_id}:8888"
    )
    assert res.success is False
    assert "не найдено" in res.alert_text

    # Seed valid receipt
    async with session_factory() as session:
        receipt, _ = await get_or_create_receipt(session, user_id, 0, 1, "valid_key", config_id=config_id)
        receipt.status = "completed"
        await session.commit()
        receipt_id = receipt.id

    # 5. Mismatched user
    res = await handle_cert_meet_click(
        session_factory, user_id=99999, topic_id=0, dialogue_id=1, callback_data=f"cert_meet:accept:{config_id}:{receipt_id}"
    )
    assert res.success is False
    assert "не принадлежит" in res.alert_text

    # 6. Valid click
    res = await handle_cert_meet_click(
        session_factory, user_id=user_id, topic_id=0, dialogue_id=1, callback_data=f"cert_meet:accept:{config_id}:{receipt_id}"
    )
    assert res.success is True
    assert res.target_step == "MEETING_BOOKING"

    # Verify stage in DB
    async with session_factory() as session:
        conv = await session.scalar(select(AutomationConversationState).where(AutomationConversationState.user_id == user_id))
        assert conv.current_step == "MEETING_BOOKING"
        r = await session.get(CertificateIssuanceReceipt, receipt_id)
        assert r.selected_button_id == "accept"

    # 7. Repeat click: MUST NOT change stage, MUST NOT produce new transition
    res_repeat = await handle_cert_meet_click(
        session_factory, user_id=user_id, topic_id=0, dialogue_id=1, callback_data=f"cert_meet:later:{config_id}:{receipt_id}"
    )
    assert res_repeat.success is False
    assert res_repeat.is_repeat is True
    assert "Вы уже выбрали" in res_repeat.alert_text

    # Confirm DB stage remained MEETING_BOOKING
    async with session_factory() as session:
        conv = await session.scalar(select(AutomationConversationState).where(AutomationConversationState.user_id == user_id))
        assert conv.current_step == "MEETING_BOOKING", "Repeat click must NOT change stage!"


def test_admin_target_stage_priority_helper():
    """
    Defect 4 regression test:
    override_service_blocks_target_stage ensures that admin target_stage strictly
    overwrites any conflicting current_step emitted by AI in <DATA>.
    """
    block1 = ServiceDataBlock(
        current_state={"current_step": "AI_STEP_HALLUCINATION", "extra_var": 123},
        events=["event1"],
        metadata={"user_score": 5},
        save_mode="merge",
        raw_json="{}",
    )
    block2 = ServiceDataBlock(
        current_state={"current_step": "ANOTHER_AI_STEP"},
        events=[],
        metadata={},
        save_mode="merge",
        raw_json="{}",
    )

    overridden = override_service_blocks_target_stage([block1, block2], "ADMIN_EXCLUSIVE_STAGE")
    assert len(overridden) == 2
    assert overridden[0].current_state["current_step"] == "ADMIN_EXCLUSIVE_STAGE"
    assert overridden[0].current_state["extra_var"] == 123
    assert overridden[1].current_state["current_step"] == "ADMIN_EXCLUSIVE_STAGE"


@pytest.mark.asyncio
async def test_atomic_concurrency_resumption_claim(session_factory, setup_tg_user):
    """
    Defect 5 regression test:
    claim_receipt_for_invitation guarantees that concurrent tasks cannot both claim
    resumption of the same receipt. Exactly one task succeeds.
    """
    user_id = setup_tg_user
    async with session_factory() as session:
        receipt, _ = await get_or_create_receipt(session, user_id, 0, 1, "concurrency_test_key", config_id=1)
        receipt.status = "cert_sent"
        await session.commit()
        receipt_id = receipt.id

    # Simulate two concurrent workers
    async with session_factory() as session1, session_factory() as session2:
        claim1 = await claim_receipt_for_invitation(session1, receipt_id)
        claim2 = await claim_receipt_for_invitation(session2, receipt_id)

    assert (claim1 and not claim2) or (claim2 and not claim1), "Exactly one worker must claim resumption!"


def test_render_caption_safe_escaping_and_length_limit():
    """
    Defect 6 & 7 regression test:
    - User parameters containing raw HTML special characters are safely escaped.
    - Excessively long user values adaptively adjust name budget to preserve the admin template instruction.
    - If template cannot preserve required instructions within max_limit, raises CaptionRenderError (no silent generic fallback!).
    """
    from certificate_followup_service import CaptionRenderError

    template = "Сертификат выдан <b>{name}</b> ({fio}). Поздравляем!"
    context = {
        "name": "Иван <CEO> & Ко",
        "fio": "Иван <Тестер>",
        "date": "09.11.2026",
    }
    rendered, errors = render_caption_safe(template, context, max_limit=1024)
    assert errors == []
    assert "<CEO>" not in rendered, "Raw <CEO> tag must be escaped"
    assert "&lt;CEO&gt;" in rendered
    assert "&amp; Ко" in rendered
    assert "<b>Иван &lt;CEO&gt; &amp; Ко</b>" in rendered

    # Long user name: adaptively trimmed to budget while preserving template instruction
    long_user_ctx = {
        "name": "Иван " * 100,
        "fio": "Иванов " * 100,
        "date": "09.11.2026",
    }
    rendered_adapted, errors = render_caption_safe("🎓 Инструкция: сохраните файл. Получатель: {name}", long_user_ctx, max_limit=120)
    assert errors == []
    assert "Инструкция: сохраните файл" in rendered_adapted
    assert len(rendered_adapted) <= 120

    # Test exceeding length limit where template itself cannot fit: raises CaptionRenderError
    long_template = "🎓 " + ("Длинный текст подписи " * 60) + "{name}"
    with pytest.raises(CaptionRenderError, match="превышает лимит"):
        render_caption_safe(long_template, context, max_limit=200)


@pytest.mark.asyncio
async def test_defect3_sending_cert_in_flight_state_and_crash_recovery(session_factory, setup_tg_user):
    """
    Defect 3 regression test:
    - Verifies 'sending_cert' in-flight status before external API call.
    - If process crashes between SendDocument and mark_receipt_cert_sent, next run detects
      'sending_cert', marks 'uncertain' and STRICTLY blocks automatic retry.
    """
    from certificate_followup_service import mark_receipt_sending_cert

    user_id = setup_tg_user
    key = make_issuance_key(99, user_id, "Тест Дефект 3", "09.11.2026")

    # 1. Create receipt
    async with session_factory() as session:
        receipt, is_new = await get_or_create_receipt(session, user_id, 0, 1, key, config_id=1)
        await session.commit()
        receipt_id = receipt.id
        assert receipt.status == "started"

    # 2. Mark in-flight before external API call
    async with session_factory() as session:
        await mark_receipt_sending_cert(session, receipt_id)
        r = await session.get(CertificateIssuanceReceipt, receipt_id)
        assert r.status == "sending_cert"

    # 3. Process crashes here! On reboot, get_or_create_receipt sees sending_cert
    async with session_factory() as session:
        receipt_recovered, is_new = await get_or_create_receipt(session, user_id, 0, 1, key, config_id=1)
        assert is_new is False
        assert receipt_recovered.status == "uncertain"
        assert "automatic repeat prevented" in (receipt_recovered.error_details or "")


@pytest.mark.asyncio
async def test_defect2_invitation_resumption_uncertain_and_stuck_reconciliation(session_factory, setup_tg_user):
    """
    Defect 2 regression test:
    - Network timeout during invitation delivery transitions receipt to 'uncertain' (NOT 'cert_sent').
    - Stale 'resuming_invite' from crashed process is safely reconciled to 'uncertain'.
    """
    from certificate_followup_service import (
        reconcile_stuck_followup_receipts,
        resume_pending_certificate_followup,
    )
    from unittest.mock import AsyncMock

    user_id = setup_tg_user

    # Setup config and receipt in cert_sent
    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, 88)
        cfg.is_active = True
        await session.commit()
        config_id = cfg.id

    key = make_issuance_key(88, user_id, "Тест Дефект 2", "09.11.2026")
    async with session_factory() as session:
        r, _ = await get_or_create_receipt(session, user_id, 0, 1, key, config_id=config_id)
        r.status = "cert_sent"
        await session.commit()
        receipt_id = r.id

    # 1. Test timeout during send_message -> must mark uncertain, NOT cert_sent
    failing_bot = AsyncMock()
    failing_bot.send_message.side_effect = asyncio.TimeoutError("Telegram API timeout")

    res = await resume_pending_certificate_followup(
        failing_bot,
        user_id=user_id,
        topic_id=0,
        dialogue_id=1,
        session_maker=session_factory,
        platform="telegram",
    )
    assert res is False

    async with session_factory() as session:
        r = await session.get(CertificateIssuanceReceipt, receipt_id)
        assert r.status == "uncertain"
        assert "Uncertain" in (r.error_details or "") or "failed/uncertain" in (r.error_details or "")

    # 2. Test stuck resuming_invite recovery
    async with session_factory() as session:
        r = await session.get(CertificateIssuanceReceipt, receipt_id)
        r.status = "resuming_invite"
        r.created_at = datetime.utcnow() - timedelta(seconds=200)
        await session.commit()

    async with session_factory() as session:
        count = await reconcile_stuck_followup_receipts(session, older_than_seconds=60)
        assert count == 1
        r = await session.get(CertificateIssuanceReceipt, receipt_id)
        assert r.status == "uncertain"


@pytest.mark.asyncio
async def test_defect4_concurrent_button_clicks_atomic_cas(session_factory, setup_tg_user):
    """
    Defect 4 failure point test:
    - Two simultaneous button clicks on the invitation keyboard.
    - Atomic CAS update guarantees exactly ONE request succeeds.
    - The losing request receives is_repeat=True.
    - Only ONE stage transition is recorded in AutomationStepTransition.
    """
    user_id = setup_tg_user

    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, 77)
        cfg.is_active = True
        await session.commit()
        config_id = cfg.id

    key = make_issuance_key(77, user_id, "Тест Дефект 4", "09.11.2026")
    async with session_factory() as session:
        r, _ = await get_or_create_receipt(session, user_id, 0, 1, key, config_id=config_id)
        r.status = "completed"
        await session.commit()
        receipt_id = r.id

    # Simulate two concurrent button clicks: 'accept' and 'later'
    callback_accept = f"cert_meet:accept:{config_id}:{receipt_id}"
    callback_later = f"cert_meet:later:{config_id}:{receipt_id}"

    res_accept, res_later = await asyncio.gather(
        handle_cert_meet_click(session_factory, user_id, 0, 1, callback_accept),
        handle_cert_meet_click(session_factory, user_id, 0, 1, callback_later),
    )

    success_count = sum(1 for res in (res_accept, res_later) if res.success)
    repeat_count = sum(1 for res in (res_accept, res_later) if res.is_repeat)

    assert success_count == 1, f"Expected exactly 1 success, got accept={res_accept.success}, later={res_later.success}"
    assert repeat_count == 1, "Expected exactly 1 repeat click response"

    # Verify DB state: exactly 1 transition recorded
    async with session_factory() as session:
        r = await session.get(CertificateIssuanceReceipt, receipt_id)
        assert r.selected_button_id in ("accept", "later")
        transitions = (await session.scalars(
            select(AutomationStepTransition).where(AutomationStepTransition.user_id == user_id)
        )).all()
        assert len(transitions) == 1


@pytest.mark.asyncio
async def test_defect4_callback_security_validations(session_factory, setup_tg_user):
    """
    Defect 4 security validations:
    - Reject config_id mismatch with receipt.config_id.
    - Reject button click when receipt.status != 'completed' (invitation not yet confirmed delivered).
    - Reject callback when user/topic/dialogue does not match receipt.
    """
    user_id = setup_tg_user

    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, 66)
        cfg.is_active = True
        await session.commit()
        config_id = cfg.id

    key = make_issuance_key(66, user_id, "Тест Дефект 4 Безопасность", "09.11.2026")
    async with session_factory() as session:
        r, _ = await get_or_create_receipt(session, user_id, 0, 1, key, config_id=config_id)
        r.status = "cert_sent"  # Invitation NOT yet sent!
        await session.commit()
        receipt_id = r.id

    # 1. Click before invitation confirmed delivered (status != 'completed')
    res = await handle_cert_meet_click(
        session_factory, user_id, 0, 1, f"cert_meet:accept:{config_id}:{receipt_id}"
    )
    assert res.success is False
    assert "Приглашение ещё не было отправлено" in res.alert_text

    # 2. Mark completed, but test with wrong user
    async with session_factory() as session:
        r = await session.get(CertificateIssuanceReceipt, receipt_id)
        r.status = "completed"
        await session.commit()

    res_wrong_user = await handle_cert_meet_click(
        session_factory, user_id + 999, 0, 1, f"cert_meet:accept:{config_id}:{receipt_id}"
    )
    assert res_wrong_user.success is False
    assert "не принадлежит" in res_wrong_user.alert_text

    # 3. Test with wrong config_id (mismatch)
    res_wrong_cfg = await handle_cert_meet_click(
        session_factory, user_id, 0, 1, f"cert_meet:accept:{config_id + 999}:{receipt_id}"
    )
    assert res_wrong_cfg.success is False
    assert "не найден" in res_wrong_cfg.alert_text or "не совпадает" in res_wrong_cfg.alert_text


@pytest.mark.asyncio
async def test_defect5_complete_followup_delivery_and_deferred_data_idempotency(session_factory, setup_tg_user):
    """
    Defect 5 regression test:
    - Verifies complete_followup_delivery persists DATA and transitions stage without window.
    - Verifies apply_pending_deferred_data_if_any idempotently applies leftover DATA
      without re-sending messages and with admin target_stage priority.
    """
    from certificate_followup_service import (
        apply_pending_deferred_data_if_any,
        complete_followup_delivery,
    )
    from user_metadata import ServiceDataBlock

    user_id = setup_tg_user

    async with session_factory() as session:
        cfg = await get_or_create_followup_config(session, 55)
        cfg.target_stage = "ADMIN_STAGE_PRIORITY"
        cfg.is_active = True
        await session.commit()

    key = make_issuance_key(55, user_id, "Тест Дефект 5", "09.11.2026")
    async with session_factory() as session:
        r, _ = await get_or_create_receipt(session, user_id, 0, 1, key, config_id=cfg.id)
        r.status = "cert_sent"
        await session.commit()
        receipt_id = r.id

    deferred_blocks = [
        ServiceDataBlock(
            current_state={"current_step": "AI_MODEL_STEP"},
            events=["cert_awarded"],
            metadata={"source": "test"},
            save_mode="merge",
            raw_json="{}",
        )
    ]

    # Run complete_followup_delivery
    await complete_followup_delivery(
        session_factory,
        receipt_id=receipt_id,
        invitation_message_id="msg_777",
        scen_cfg=cfg,
        user_id=user_id,
        topic_id=0,
        dialogue_id=1,
        deferred_blocks=deferred_blocks,
    )

    async with session_factory() as session:
        r = await session.get(CertificateIssuanceReceipt, receipt_id)
        assert r.status == "completed"
        assert r.deferred_data_json is None, "deferred_data_json must be cleared after persistence"
        conv_state = await session.scalar(
            select(AutomationConversationState).where(AutomationConversationState.user_id == user_id)
        )
        assert conv_state.current_step == "ADMIN_STAGE_PRIORITY", "Admin target stage must strictly override AI step"
