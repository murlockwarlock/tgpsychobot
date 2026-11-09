import hashlib
import html
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database import (
    AutomationConversationState,
    AutomationStepTransition,
    CertificateFollowupConfig,
    CertificateIssuanceReceipt,
    MediaLibrary,
    User,
)
from event_content_validator import TelegramHtmlValidator, utf16_code_units_len

log = logging.getLogger(__name__)


class CaptionRenderError(ValueError):
    """Raised when certificate caption cannot be safely rendered preserving required instructions."""
    pass

DEFAULT_CAPTION_TEMPLATE = (
    "Поздравляем с успешным завершением! Ваш сертификат готов 🎓\n\n"
    "Вы можете сохранить его или распечатать на память."
)

DEFAULT_INVITATION_TEXT = (
    "Предлагаем встретиться лично или онлайн, чтобы обсудить дальнейшие шаги "
    "и ответить на все ваши вопросы. Выберите удобный вариант:"
)

DEFAULT_BUTTONS_CONFIG = [
    {
        "id": "accept",
        "text": "🤝 Хочу встретиться",
        "target_step": "MEETING_BOOKING",
        "response_text": "Отлично! Запланируем встречу в удобное для вас время.",
    },
    {
        "id": "details",
        "text": "ℹ️ Узнать подробнее",
        "target_step": "MEETING_INFO",
        "response_text": "Формат встречи: 30 минут онлайн или очно. Мы ответим на все вопросы и поможем определить цели.",
    },
    {
        "id": "later",
        "text": "⏳ Позже",
        "target_step": "MEETING_POSTPONED",
        "response_text": "Хорошо! Вы сможете вернуться к этому предложению в любой момент.",
    },
]

DEFAULT_TARGET_STAGE = "OFFER_MEETING"

ALLOWED_CAPTION_VARS = {"name", "fio", "recipient", "date"}
ALLOWED_INVITE_VARS = {"name"}


def render_caption_safe(
    template: str,
    context: dict[str, Any],
    max_limit: int = 1024,
) -> tuple[str, list[str]]:
    """
    Renders certificate caption safely:
    - Escapes all user inputs (name, fio, recipient, date) so raw characters do not break HTML.
    - If user dynamic fields cause overflow, adaptively bounds user name budget to preserve full admin template instructions.
    - Validates resulting HTML structure with TelegramHtmlValidator.
    - Strictly checks UTF-16 code units length against max_limit (no silent truncation of instructions!).
    - If template cannot preserve required instructions or HTML is invalid, raises CaptionRenderError.
    """
    raw_template = (template or DEFAULT_CAPTION_TEMPLATE).strip()
    raw_name = str(context.get("name") or "").strip()
    raw_fio = str(context.get("fio") or context.get("recipient") or raw_name).strip()
    raw_date = str(context.get("date") or "").strip()

    def _render_with(name_val: str, fio_val: str, date_val: str) -> str:
        esc = {
            "name": html.escape(name_val, quote=False),
            "fio": html.escape(fio_val, quote=False),
            "recipient": html.escape(fio_val, quote=False),
            "date": html.escape(date_val, quote=False),
        }
        res = raw_template
        for k, v in esc.items():
            res = res.replace(f"{{{k}}}", v)
        return res

    result = _render_with(raw_name or raw_fio, raw_fio or raw_name, raw_date)
    actual_length = utf16_code_units_len(result)

    if actual_length > max_limit:
        # Check if the template structure itself fits without user values
        base_result = _render_with("", "", raw_date)
        base_len = utf16_code_units_len(base_result)
        if base_len >= max_limit:
            raise CaptionRenderError(
                f"Шаблон подписи к сертификату ({base_len} UTF-16) превышает лимит {max_limit} и не позволяет сохранить инструкцию."
            )
        # Adaptively allocate available budget to dynamic user values to preserve full template instructions
        budget = max(0, max_limit - base_len - 1)
        safe_name = (raw_name or raw_fio)[:budget]
        safe_fio = (raw_fio or raw_name)[:budget]
        result = _render_with(safe_name, safe_fio, raw_date)
        actual_length = utf16_code_units_len(result)
        if actual_length > max_limit:
            raise CaptionRenderError(
                f"Подпись к сертификату ({actual_length} UTF-16) превышает лимит {max_limit} и не позволяет сохранить обязательную инструкцию."
            )

    try:
        TelegramHtmlValidator.validate(result)
    except Exception as exc:
        raise CaptionRenderError(f"Ошибка валидации HTML в подписи: {exc}") from exc

    return result, []


def validate_caption_template(template: str, max_limit: int = 1024) -> None:
    clean = (template or "").strip()
    if not clean:
        raise ValueError("Подпись к сертификату не может быть пустой.")
    for var in re.findall(r"\{([^{}]+)\}", clean):
        var_name = var.strip()
        if var_name not in ALLOWED_CAPTION_VARS:
            allowed_str = ", ".join(f"{{{v}}}" for v in sorted(ALLOWED_CAPTION_VARS))
            raise ValueError(f"Неизвестная переменная {{{var_name}}}. Доступны: {allowed_str}")
    TelegramHtmlValidator.validate(clean)

    # Check length using standard sample values to verify bounds
    sample_context = {
        "name": "Иван Иванович Иванов",
        "fio": "Иванов Иван Иванович",
        "recipient": "Иванов Иван Иванович",
        "date": "09.11.2026",
    }
    try:
        render_caption_safe(clean, sample_context, max_limit=max_limit)
    except CaptionRenderError as exc:
        raise ValueError(str(exc)) from exc


def validate_invitation_text(text: str, max_limit: int = 4096) -> None:
    clean = (text or "").strip()
    if not clean:
        raise ValueError("Текст приглашения не может быть пустым.")
    for var in re.findall(r"\{([^{}]+)\}", clean):
        var_name = var.strip()
        if var_name not in ALLOWED_INVITE_VARS:
            raise ValueError(f"Неизвестная переменная {{{var_name}}}. Доступна только {{name}}")
    TelegramHtmlValidator.validate(clean)
    if utf16_code_units_len(clean) > max_limit:
        raise ValueError(f"Длина текста приглашения превышает лимит {max_limit} символов.")


def validate_stage_identifier(stage: str) -> str:
    clean = (stage or "").strip()
    if not clean:
        raise ValueError("Название этапа не может быть пустым.")
    if len(clean) > 64:
        raise ValueError("Название этапа не должно превышать 64 символа.")
    if not re.match(r"^[A-Za-z0-9_-]+$", clean):
        raise ValueError("Название этапа должно содержать только латинские буквы, цифры, дефис или подчёркивание.")
    return clean


def parse_buttons_config(raw: Any) -> list[dict[str, str]]:
    if not raw:
        return [dict(b) for b in DEFAULT_BUTTONS_CONFIG]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return [dict(b) for b in DEFAULT_BUTTONS_CONFIG]
    elif isinstance(raw, list):
        parsed = raw
    else:
        return [dict(b) for b in DEFAULT_BUTTONS_CONFIG]

    valid = []
    for item in parsed:
        if isinstance(item, dict) and "id" in item and "text" in item and "target_step" in item:
            valid.append({
                "id": str(item["id"]).strip(),
                "text": str(item["text"]).strip(),
                "target_step": str(item["target_step"]).strip(),
                "response_text": str(item.get("response_text", "")).strip(),
            })
    return valid if valid else [dict(b) for b in DEFAULT_BUTTONS_CONFIG]


def render_caption(template: str, context: dict[str, Any], max_limit: int = 1024) -> str:
    rendered, _ = render_caption_safe(template, context, max_limit=max_limit)
    return rendered


def render_invitation(text: str, context: dict[str, Any]) -> str:
    if not text:
        text = DEFAULT_INVITATION_TEXT
    name = str(context.get("name") or "").strip()
    safe_name = html.escape(name, quote=False)
    return text.replace("{name}", safe_name)


def make_issuance_key(media_id: int, user_id: int, recipient: str, date: str | None = None) -> str:
    raw = f"{media_id}:{user_id}:{recipient.strip().lower()}:{date or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


async def get_followup_config(session: AsyncSession, media_id: int) -> CertificateFollowupConfig | None:
    return await session.scalar(
        select(CertificateFollowupConfig).where(CertificateFollowupConfig.media_id == media_id)
    )


async def get_or_create_followup_config(
    session: AsyncSession,
    media_id: int,
    topic_id: int | None = None,
) -> CertificateFollowupConfig:
    config = await get_followup_config(session, media_id)
    if config is not None:
        return config

    # Requirement 1: Scenario must be created INACTIVE.
    # Opening «После выдачи» does not activate scenario; publication requires explicit admin action.
    config = CertificateFollowupConfig(
        media_id=media_id,
        topic_id=topic_id,
        is_active=False,
        caption_template=DEFAULT_CAPTION_TEMPLATE,
        invitation_text=DEFAULT_INVITATION_TEXT,
        buttons_config_json=json.dumps(DEFAULT_BUTTONS_CONFIG, ensure_ascii=False),
        target_stage=DEFAULT_TARGET_STAGE,
    )
    session.add(config)
    await session.flush()
    return config


async def toggle_followup_config(session: AsyncSession, media_id: int) -> bool:
    config = await get_or_create_followup_config(session, media_id)
    config.is_active = not config.is_active
    config.updated_at = datetime.utcnow()
    await session.commit()
    return config.is_active


async def save_followup_caption(session: AsyncSession, media_id: int, caption: str) -> CertificateFollowupConfig:
    validate_caption_template(caption)
    config = await get_or_create_followup_config(session, media_id)
    config.caption_template = caption.strip()
    config.updated_at = datetime.utcnow()
    await session.commit()
    return config


async def save_followup_invitation(session: AsyncSession, media_id: int, invitation: str) -> CertificateFollowupConfig:
    validate_invitation_text(invitation)
    config = await get_or_create_followup_config(session, media_id)
    config.invitation_text = invitation.strip()
    config.updated_at = datetime.utcnow()
    await session.commit()
    return config


async def save_followup_stage(session: AsyncSession, media_id: int, target_stage: str) -> CertificateFollowupConfig:
    validated_stage = validate_stage_identifier(target_stage)
    config = await get_or_create_followup_config(session, media_id)
    config.target_stage = validated_stage
    config.updated_at = datetime.utcnow()
    await session.commit()
    return config


async def update_followup_button(
    session: AsyncSession,
    media_id: int,
    button_index: int,
    text: str,
    target_step: str,
    response_text: str = "",
) -> CertificateFollowupConfig:
    text_clean = text.strip()
    if not text_clean:
        raise ValueError("Текст кнопки не может быть пустым.")
    stage_clean = validate_stage_identifier(target_step)

    config = await get_or_create_followup_config(session, media_id)
    buttons = parse_buttons_config(config.buttons_config_json)
    if not 0 <= button_index < len(buttons):
        raise ValueError("Кнопка с таким индексом не найдена.")

    buttons[button_index]["text"] = text_clean
    buttons[button_index]["target_step"] = stage_clean
    if response_text.strip():
        buttons[button_index]["response_text"] = response_text.strip()

    config.buttons_config_json = json.dumps(buttons, ensure_ascii=False)
    config.updated_at = datetime.utcnow()
    await session.commit()
    return config


def build_followup_telegram_keyboard(
    config: CertificateFollowupConfig,
    receipt_id: int | None = None,
    *,
    is_test: bool = False,
) -> InlineKeyboardMarkup:
    buttons = parse_buttons_config(config.buttons_config_json)
    rows = []
    for btn in buttons:
        if is_test:
            callback_data = f"cert_meet:test:{config.id}:{btn['id']}"
        elif receipt_id is not None:
            callback_data = f"cert_meet:{btn['id']}:{config.id}:{receipt_id}"
        else:
            callback_data = f"cert_meet:{btn['id']}:{config.id}:0"
        rows.append([InlineKeyboardButton(text=btn["text"], callback_data=callback_data)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_followup_max_keyboard(
    config: CertificateFollowupConfig,
    receipt_id: int | None = None,
    *,
    is_test: bool = False,
) -> list[dict[str, Any]]:
    from max_messenger_bot.keyboards import callback_button, inline_keyboard

    buttons = parse_buttons_config(config.buttons_config_json)
    rows = []
    for btn in buttons:
        if is_test:
            callback_data = f"cert_meet:test:{config.id}:{btn['id']}"
        elif receipt_id is not None:
            callback_data = f"cert_meet:{btn['id']}:{config.id}:{receipt_id}"
        else:
            callback_data = f"cert_meet:{btn['id']}:{config.id}:0"
        rows.append([callback_button(btn["text"], callback_data)])
    return inline_keyboard(rows)


async def get_or_create_receipt(
    session: AsyncSession,
    user_id: int,
    topic_id: int,
    dialogue_id: int,
    issuance_key: str,
    config_id: int | None = None,
    deferred_data_json: str | None = None,
) -> tuple[CertificateIssuanceReceipt, bool]:
    existing = await session.scalar(
        select(CertificateIssuanceReceipt).where(
            CertificateIssuanceReceipt.user_id == user_id,
            CertificateIssuanceReceipt.topic_id == topic_id,
            CertificateIssuanceReceipt.dialogue_id == dialogue_id,
            CertificateIssuanceReceipt.issuance_key == issuance_key,
        )
    )
    if existing is not None:
        if config_id and not existing.config_id:
            existing.config_id = config_id
        if deferred_data_json and not existing.deferred_data_json:
            existing.deferred_data_json = deferred_data_json
        # Requirement 3: If previous process crashed while in 'sending_cert',
        # we cannot assume non-delivery without API proof; mark uncertain to prevent duplicate send.
        if existing.status == "sending_cert":
            existing.status = "uncertain"
            existing.error_details = "Previous process interrupted during sending_cert; automatic repeat prevented."
            await session.commit()
        return existing, False

    receipt = CertificateIssuanceReceipt(
        user_id=user_id,
        topic_id=topic_id,
        dialogue_id=dialogue_id,
        issuance_key=issuance_key,
        status="started",
        config_id=config_id,
        deferred_data_json=deferred_data_json,
    )
    session.add(receipt)
    await session.flush()
    return receipt, True


async def mark_receipt_sending_cert(session: AsyncSession, receipt_id: int) -> None:
    """
    Requirement 3: Persists in-flight status before the external API call to eliminate
    duplicate certificate sending after process crash.
    """
    receipt = await session.get(CertificateIssuanceReceipt, receipt_id)
    if receipt:
        receipt.status = "sending_cert"
        await session.commit()


async def mark_receipt_cert_sent(
    session: AsyncSession,
    receipt_id: int,
    message_id: str | None = None,
    config_id: int | None = None,
    deferred_data_json: str | None = None,
) -> None:
    receipt = await session.get(CertificateIssuanceReceipt, receipt_id)
    if receipt:
        receipt.status = "cert_sent"
        if message_id:
            receipt.certificate_message_id = str(message_id)
        if config_id:
            receipt.config_id = config_id
        if deferred_data_json:
            receipt.deferred_data_json = deferred_data_json
        await session.commit()


async def mark_receipt_completed(
    session: AsyncSession, receipt_id: int, invitation_message_id: str | None = None
) -> None:
    receipt = await session.get(CertificateIssuanceReceipt, receipt_id)
    if receipt:
        receipt.status = "completed"
        receipt.completed_at = datetime.utcnow()
        if invitation_message_id:
            receipt.invitation_message_id = str(invitation_message_id)
        await session.commit()


async def mark_receipt_failed(session: AsyncSession, receipt_id: int, error: str) -> None:
    receipt = await session.get(CertificateIssuanceReceipt, receipt_id)
    if receipt:
        receipt.status = "failed"
        receipt.error_details = str(error)[:1000]
        await session.commit()


async def mark_receipt_uncertain(session: AsyncSession, receipt_id: int, error: str) -> None:
    receipt = await session.get(CertificateIssuanceReceipt, receipt_id)
    if receipt:
        receipt.status = "uncertain"
        receipt.error_details = str(error)[:1000]
        await session.commit()


async def _safe_scalar(session: Any, stmt: Any) -> Any:
    try:
        if hasattr(session, "scalar"):
            res = await session.scalar(stmt)
            return res
        if hasattr(session, "execute"):
            res = await session.execute(stmt)
            if hasattr(res, "scalar_one_or_none"):
                return res.scalar_one_or_none()
            if hasattr(res, "scalar"):
                return res.scalar()
    except Exception:
        return None
    return None


async def claim_receipt_for_invitation(
    session: AsyncSession,
    receipt_id: int,
) -> bool:
    """
    Atomic compare-and-swap claiming the receipt to deliver the invitation.
    Guarantees that concurrent handlers cannot deliver duplicate invitations.
    """
    if not isinstance(receipt_id, int):
        return False
    try:
        stmt = (
            update(CertificateIssuanceReceipt)
            .where(
                CertificateIssuanceReceipt.id == receipt_id,
                CertificateIssuanceReceipt.status == "cert_sent",
            )
            .values(status="resuming_invite")
        )
        res = await session.execute(stmt)
        await session.commit()
        rowcount = getattr(res, "rowcount", None)
        if isinstance(rowcount, int):
            return rowcount > 0
        return False
    except Exception:
        return False


async def find_pending_followup_receipt(
    session: AsyncSession,
    user_id: int,
    topic_id: int,
    dialogue_id: int,
) -> CertificateIssuanceReceipt | None:
    try:
        # Requirement 2: Reconcile any stuck resuming_invite from an interrupted process
        stuck_stmt = (
            select(CertificateIssuanceReceipt).where(
                CertificateIssuanceReceipt.user_id == user_id,
                CertificateIssuanceReceipt.topic_id == topic_id,
                CertificateIssuanceReceipt.dialogue_id == dialogue_id,
                CertificateIssuanceReceipt.status == "resuming_invite",
            )
        )
        stuck = await _safe_scalar(session, stuck_stmt)
        if isinstance(stuck, CertificateIssuanceReceipt):
            stuck.status = "uncertain"
            stuck.error_details = "Process crashed or timed out during resuming_invite; marked uncertain."
            await session.commit()

        stmt = (
            select(CertificateIssuanceReceipt).where(
                CertificateIssuanceReceipt.user_id == user_id,
                CertificateIssuanceReceipt.topic_id == topic_id,
                CertificateIssuanceReceipt.dialogue_id == dialogue_id,
                CertificateIssuanceReceipt.status == "cert_sent",
            ).order_by(CertificateIssuanceReceipt.id.desc())
        )
        res = await _safe_scalar(session, stmt)
        if isinstance(res, CertificateIssuanceReceipt):
            return res
        return None
    except Exception:
        return None


async def reconcile_stuck_followup_receipts(
    session: AsyncSession,
    older_than_seconds: int = 120,
) -> int:
    """
    Reconciles receipts stuck in 'resuming_invite' or 'sending_cert' due to process crashes.
    Marks them 'uncertain' so duplicate deliveries never occur.
    """
    try:
        cutoff = datetime.utcnow() - timedelta(seconds=older_than_seconds)
        stmt = (
            update(CertificateIssuanceReceipt)
            .where(
                CertificateIssuanceReceipt.status.in_(("resuming_invite", "sending_cert")),
                CertificateIssuanceReceipt.created_at <= cutoff,
            )
            .values(
                status="uncertain",
                error_details="Reconciled from stuck in-flight state after process crash or timeout.",
            )
        )
        res = await session.execute(stmt)
        await session.commit()
        rowcount = getattr(res, "rowcount", None)
        return rowcount if isinstance(rowcount, int) else 0
    except Exception as exc:
        log.warning("reconcile_stuck_followup_receipts error: %s", exc)
        return 0


async def complete_followup_delivery(
    session_maker: Any,
    receipt_id: int,
    invitation_message_id: str | None,
    scen_cfg: CertificateFollowupConfig,
    user_id: int,
    topic_id: int,
    dialogue_id: int,
    deferred_blocks: list[Any] | None = None,
) -> None:
    """
    Requirement 5: Safely completes followup delivery:
    1. Persists deferred service blocks (with admin target_stage priority).
    2. Transitions user stage to admin target_stage.
    3. Atomically marks receipt completed and clears deferred_data_json.
    Eliminates window where receipt is completed but DATA is not saved.
    """
    raw_blocks_to_apply = list(deferred_blocks) if deferred_blocks else None
    if not raw_blocks_to_apply:
        async with session_maker() as session:
            r = await session.get(CertificateIssuanceReceipt, receipt_id)
            if r and r.deferred_data_json:
                raw_blocks_to_apply = deserialize_service_blocks(r.deferred_data_json)

    if raw_blocks_to_apply:
        from automation_engine import apply_service_data_blocks, persist_service_data_blocks
        if scen_cfg.target_stage:
            raw_blocks_to_apply = override_service_blocks_target_stage(raw_blocks_to_apply, scen_cfg.target_stage)
        await persist_service_data_blocks(
            session_maker,
            user_id=user_id,
            dialogue_id=dialogue_id,
            topic_id=topic_id,
            blocks=raw_blocks_to_apply,
            apply_fn=apply_service_data_blocks,
        )

    async with session_maker() as session:
        if scen_cfg.target_stage:
            await transition_user_to_stage(session, user_id, topic_id, dialogue_id, scen_cfg.target_stage)
        r = await session.get(CertificateIssuanceReceipt, receipt_id)
        if r:
            r.status = "completed"
            r.completed_at = datetime.utcnow()
            r.deferred_data_json = None  # Cleared upon persistence
            if invitation_message_id:
                r.invitation_message_id = str(invitation_message_id)
        await session.commit()


async def apply_pending_deferred_data_if_any(
    session_maker: Any,
    receipt_id: int,
    scen_cfg: CertificateFollowupConfig,
    user_id: int,
    topic_id: int,
    dialogue_id: int,
) -> None:
    """
    Requirement 5: Idempotently persists leftover deferred data from an interrupted
    completed receipt without repeating messages and without overwriting admin's target stage.
    """
    async with session_maker() as session:
        r = await session.get(CertificateIssuanceReceipt, receipt_id)
        if not r or not r.deferred_data_json:
            return
        blocks = deserialize_service_blocks(r.deferred_data_json)

    if blocks:
        from automation_engine import apply_service_data_blocks, persist_service_data_blocks
        if scen_cfg.target_stage:
            blocks = override_service_blocks_target_stage(blocks, scen_cfg.target_stage)
        await persist_service_data_blocks(
            session_maker,
            user_id=user_id,
            dialogue_id=dialogue_id,
            topic_id=topic_id,
            blocks=blocks,
            apply_fn=apply_service_data_blocks,
        )

    async with session_maker() as session:
        r = await session.get(CertificateIssuanceReceipt, receipt_id)
        if r:
            r.deferred_data_json = None
            await session.commit()


def override_service_blocks_target_stage(
    blocks: list[Any], target_stage: str
) -> list[Any]:
    """
    Enforces admin priority: ensures that admin's target_stage strictly overrides
    any conflicting current_step emitted by AI inside <DATA> blocks.
    """
    if not target_stage or not blocks:
        return blocks
    from user_metadata import ServiceDataBlock
    overridden = []
    for b in blocks:
        new_state = dict(getattr(b, "current_state", None) or {})
        new_state["current_step"] = target_stage
        overridden.append(
            ServiceDataBlock(
                current_state=new_state,
                events=list(getattr(b, "events", []) or []),
                metadata=dict(getattr(b, "metadata", None) or {}),
                save_mode=getattr(b, "save_mode", "merge"),
                raw_json=getattr(b, "raw_json", "{}"),
                legacy=getattr(b, "legacy", False),
            )
        )
    return overridden


def serialize_service_blocks(blocks: list[Any] | None) -> str | None:
    if not blocks:
        return None
    data = []
    for b in blocks:
        data.append({
            "current_state": getattr(b, "current_state", {}),
            "events": getattr(b, "events", []),
            "metadata": getattr(b, "metadata", {}),
            "save_mode": getattr(b, "save_mode", "merge"),
            "raw_json": getattr(b, "raw_json", "{}"),
            "legacy": getattr(b, "legacy", False),
        })
    return json.dumps(data, ensure_ascii=False)


def deserialize_service_blocks(raw: str | None) -> list[Any]:
    if not raw:
        return []
    from user_metadata import ServiceDataBlock
    try:
        items = json.loads(raw)
        blocks = []
        for item in items:
            blocks.append(
                ServiceDataBlock(
                    current_state=dict(item.get("current_state") or {}),
                    events=list(item.get("events") or []),
                    metadata=dict(item.get("metadata") or {}),
                    save_mode=str(item.get("save_mode") or "merge"),
                    raw_json=str(item.get("raw_json") or "{}"),
                    legacy=bool(item.get("legacy", False)),
                )
            )
        return blocks
    except Exception as exc:
        log.warning("Failed to deserialize service blocks: %s", exc)
        return []


async def transition_user_to_stage(
    session: AsyncSession,
    user_id: int,
    topic_id: int,
    dialogue_id: int,
    target_step: str,
) -> None:
    if not target_step:
        return

    conv_state = await session.scalar(
        select(AutomationConversationState).where(
            AutomationConversationState.user_id == user_id,
            AutomationConversationState.topic_id == topic_id,
            AutomationConversationState.dialogue_id == dialogue_id,
        )
    )
    previous_step = conv_state.current_step if conv_state else None

    if conv_state is None:
        conv_state = AutomationConversationState(
            user_id=user_id,
            topic_id=topic_id,
            dialogue_id=dialogue_id,
            current_step=target_step,
            current_state_json=json.dumps({"current_step": target_step}, ensure_ascii=False),
        )
        session.add(conv_state)
    else:
        conv_state.current_step = target_step
        try:
            state_data = json.loads(conv_state.current_state_json or "{}")
        except Exception:
            state_data = {}
        state_data["current_step"] = target_step
        conv_state.current_state_json = json.dumps(state_data, ensure_ascii=False)
        conv_state.updated_at = datetime.utcnow()

    user = await session.get(User, user_id)
    if user and hasattr(user, "current_step"):
        user.current_step = target_step

    if previous_step != target_step:
        session.add(
            AutomationStepTransition(
                user_id=user_id,
                dialogue_id=dialogue_id,
                topic_id=topic_id,
                previous_step=previous_step,
                current_step=target_step,
                state_json=conv_state.current_state_json,
            )
        )


async def resume_pending_certificate_followup(
    bot_or_client: Any,
    user_id: int,
    topic_id: int,
    dialogue_id: int,
    session_maker: Any,
    *,
    platform: str = "telegram",
    chat_id: int | None = None,
) -> bool:
    """
    Resumes delivery of Message 2 (invitation) if Message 1 (certificate) was already delivered
    (status == 'cert_sent').
    - Does NOT re-send certificate (zero duplication).
    - Does NOT require AI to issue RENDER_IMG_MEDIA again.
    - Protected against concurrent execution via atomic status claim ('resuming_invite').
    - Applies deferred data blocks with admin target_stage priority without window.
    - Sends to actual recipient (chat_id if provided, else user_id).
    """
    receipt_id = None
    config_id = None
    deferred_raw = None

    async with session_maker() as session:
        receipt = await find_pending_followup_receipt(session, user_id, topic_id, dialogue_id)
        if receipt is None:
            return False
        receipt_id = receipt.id
        config_id = receipt.config_id
        deferred_raw = receipt.deferred_data_json

    async with session_maker() as session:
        claimed = await claim_receipt_for_invitation(session, receipt_id)
        if not claimed:
            # Another concurrent handler already claimed resumption
            return False

    async with session_maker() as session:
        scen_cfg = await session.get(CertificateFollowupConfig, config_id) if config_id else None
        if scen_cfg is None or not scen_cfg.is_active:
            # Config missing or disabled; close receipt without duplicate action
            await mark_receipt_completed(session, receipt_id)
            return False

        user = await session.get(User, user_id)
        user_context = {
            "name": (user.name if user else "") or "",
            "fio": (user.name if user else "") or "",
            "date": "",
        }

    invite_text = render_invitation(scen_cfg.invitation_text, user_context)

    effective_chat_id = chat_id if chat_id is not None else user_id
    sent_msg_id = None
    try:
        if platform == "telegram":
            keyboard = build_followup_telegram_keyboard(scen_cfg, receipt_id=receipt_id)
            sent = await bot_or_client.send_message(
                chat_id=effective_chat_id,
                text=invite_text,
                parse_mode="HTML",
                reply_markup=keyboard,
            )
            sent_msg_id = getattr(sent, "message_id", None)
        else:
            keyboard = build_followup_max_keyboard(scen_cfg, receipt_id=receipt_id)
            sent = await bot_or_client.send_message(
                chat_id=effective_chat_id,
                text=invite_text,
                attachments=keyboard,
            )
            sent_msg_id = str(sent) if sent else None
    except Exception as exc:
        log.error("Failed to deliver resumed invitation user_id=%s chat_id=%s: %s", user_id, effective_chat_id, exc)
        # Requirement 2: Do NOT blind-reset to cert_sent! Mark uncertain so no duplicate invite is sent
        async with session_maker() as session:
            await mark_receipt_uncertain(session, receipt_id, f"Resumption delivery failed/uncertain: {exc}")
        return False

    # Safely complete delivery and persist DATA without window (Requirement 5)
    deferred_blocks = deserialize_service_blocks(deferred_raw) if deferred_raw else None
    await complete_followup_delivery(
        session_maker,
        receipt_id=receipt_id,
        invitation_message_id=str(sent_msg_id) if sent_msg_id else None,
        scen_cfg=scen_cfg,
        user_id=user_id,
        topic_id=topic_id,
        dialogue_id=dialogue_id,
        deferred_blocks=deferred_blocks,
    )
    return True


class CertCallbackResult:
    def __init__(
        self,
        success: bool,
        response_text: str = "",
        alert_text: str = "",
        target_step: str | None = None,
        is_repeat: bool = False,
        is_test: bool = False,
    ):
        self.success = success
        self.response_text = response_text
        self.alert_text = alert_text
        self.target_step = target_step
        self.is_repeat = is_repeat
        self.is_test = is_test


async def handle_cert_meet_click(
    session_maker: Any,
    user_id: int,
    topic_id: int,
    dialogue_id: int,
    callback_data: str,
    *,
    is_admin_user: bool = False,
) -> CertCallbackResult:
    """
    Handles cert_meet callback query with strict validations:
    - Rejects unknown config_id / action_id (never defaults to accept).
    - Requires active scenario config.
    - Requires confirmed issuance receipt for matching user, topic, dialogue.
    - Detects repeat clicks (idempotency): does not alter stage or re-send messages.
    - Handles test send clicks without modifying admin's real dialogue stage.
    """
    parts = (callback_data or "").split(":")
    if len(parts) < 3 or parts[0] != "cert_meet":
        return CertCallbackResult(False, alert_text="Некорректный формат данных кнопки.")

    # 1. Test click path: cert_meet:test:<config_id>:<action_id>
    if parts[1] == "test":
        if not is_admin_user:
            return CertCallbackResult(False, alert_text="⛔️ Тестовые кнопки доступны только администраторам.")
        if len(parts) < 4:
            return CertCallbackResult(False, alert_text="Некорректный формат тестовой кнопки.")
        try:
            config_id = int(parts[2])
        except ValueError:
            return CertCallbackResult(False, alert_text="Некорректный ID конфигурации.")
        action_id = parts[3]

        async with session_maker() as session:
            config = await session.get(CertificateFollowupConfig, config_id)
            if not config:
                return CertCallbackResult(False, alert_text="Сценарий не найден.")
            buttons = parse_buttons_config(config.buttons_config_json)
            btn = next((b for b in buttons if b["id"] == action_id), None)
            if not btn:
                return CertCallbackResult(False, alert_text="Кнопка не найдена в сценарии.")

            # Test mode: do NOT update user.current_step, do NOT record transition
            return CertCallbackResult(
                True,
                response_text=f"🧪 [Тест] Ответ бота:\n{btn.get('response_text', '')}\n\n(Ваш реальный этап диалога не изменён)",
                alert_text=f"🧪 Тестовый выбор: «{btn['text']}». Реальный этап не изменён.",
                target_step=btn["target_step"],
                is_test=True,
            )

    # 2. Production click path: cert_meet:<action_id>:<config_id>[:<receipt_id>]
    action_id = parts[1]
    try:
        config_id = int(parts[2])
    except ValueError:
        return CertCallbackResult(False, alert_text="Некорректный ID сценария.")

    receipt_id = None
    if len(parts) >= 4 and parts[3].isdigit():
        receipt_id = int(parts[3])

    async with session_maker() as session:
        # Check active config
        config = await session.get(CertificateFollowupConfig, config_id)
        if config is None:
            return CertCallbackResult(False, alert_text="❌ Сценарий не найден или был удалён.")
        if not config.is_active:
            return CertCallbackResult(False, alert_text="⚠️ Этот сценарий сейчас отключен администратором.")

        # Match button
        buttons = parse_buttons_config(config.buttons_config_json)
        btn = next((b for b in buttons if b["id"] == action_id), None)
        if not btn:
            return CertCallbackResult(False, alert_text="❌ Неизвестное действие кнопки.")

        # Find confirmed receipt
        receipt = None
        if receipt_id and receipt_id > 0:
            receipt = await session.get(CertificateIssuanceReceipt, receipt_id)
        if receipt is None:
            receipt = await session.scalar(
                select(CertificateIssuanceReceipt).where(
                    CertificateIssuanceReceipt.user_id == user_id,
                    CertificateIssuanceReceipt.topic_id == topic_id,
                    CertificateIssuanceReceipt.dialogue_id == dialogue_id,
                    CertificateIssuanceReceipt.config_id == config_id,
                ).order_by(CertificateIssuanceReceipt.id.desc())
            )

        if receipt is None:
            return CertCallbackResult(False, alert_text="❌ Подтверждение выдачи сертификата не найдено.")

        # Requirement 4: Verify receipt belongs to user, topic and dialogue
        if receipt.user_id != user_id or receipt.topic_id != topic_id or receipt.dialogue_id != dialogue_id:
            return CertCallbackResult(False, alert_text="⛔️ Приглашение не принадлежит вашему текущему диалогу.")

        # Requirement 4: Verify receipt config_id matches callback config_id
        if receipt.config_id is not None and receipt.config_id != config_id:
            return CertCallbackResult(False, alert_text="⛔️ Конфигурация сценария не совпадает с выданным сертификатом.")

        # Requirement 4: Do not accept buttons before invitation itself is confirmed sent (completed)
        if receipt.status != "completed":
            return CertCallbackResult(
                False,
                alert_text="⚠️ Приглашение ещё не было отправлено или ожидает подтверждения.",
            )

        # Idempotency / Repeat click check
        if receipt.selected_button_id is not None:
            prev_btn = next((b for b in buttons if b["id"] == receipt.selected_button_id), None)
            prev_label = prev_btn["text"] if prev_btn else receipt.selected_button_id
            return CertCallbackResult(
                False,
                alert_text=f"ℹ️ Вы уже выбрали: «{prev_label}». Повторный выбор не требуется.",
                is_repeat=True,
            )

        # Requirement 4: Atomic CAS for button selection (never separate read/write)
        target_receipt_id = receipt.id
        now = datetime.utcnow()
        stmt = (
            update(CertificateIssuanceReceipt)
            .where(
                CertificateIssuanceReceipt.id == target_receipt_id,
                CertificateIssuanceReceipt.selected_button_id.is_(None),
            )
            .values(
                selected_button_id=action_id,
                selected_at=now,
            )
        )
        res = await session.execute(stmt)
        rowcount = getattr(res, "rowcount", None)
        if not (isinstance(rowcount, int) and rowcount > 0):
            # CAS lost: another concurrent request already selected a button
            await session.commit()
            receipt = await session.get(CertificateIssuanceReceipt, target_receipt_id)
            prev_id = (receipt.selected_button_id if receipt else None) or action_id
            prev_btn = next((b for b in buttons if b["id"] == prev_id), None)
            prev_label = prev_btn["text"] if prev_btn else prev_id
            return CertCallbackResult(
                False,
                alert_text=f"ℹ️ Вы уже выбрали: «{prev_label}». Повторный выбор не требуется.",
                is_repeat=True,
            )

        # CAS won: persist reservation
        await session.commit()

        target_step = btn["target_step"]
        response_text = btn.get("response_text") or "Спасибо за ответ!"
        receipt.selected_button_id = action_id
        receipt.selected_at = now

        await transition_user_to_stage(
            session,
            user_id=user_id,
            topic_id=topic_id,
            dialogue_id=dialogue_id,
            target_step=target_step,
        )
        await session.commit()

        return CertCallbackResult(
            True,
            response_text=response_text,
            alert_text="✅ Выбор сохранён.",
            target_step=target_step,
        )


async def handle_cert_callback_max(client: Any, callback: Any) -> None:
    data = str(callback.payload or "")
    from database import async_session_maker
    user_id = callback.sender.user_id
    chat_id = callback.chat_id

    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        topic_id = user.current_topic_id if user and user.current_topic_id else 0
        dialogue_id = user.current_dialogue_id if user and user.current_dialogue_id else 1
        is_admin = bool(user and getattr(user, "is_admin", False))

    res = await handle_cert_meet_click(
        async_session_maker,
        user_id=user_id,
        topic_id=topic_id,
        dialogue_id=dialogue_id,
        callback_data=data,
        is_admin_user=is_admin,
    )

    await client.answer_callback(callback.callback_id)
    if not res.success and res.alert_text:
        await client.send_message(chat_id=chat_id, text=res.alert_text)
    elif res.success and res.response_text:
        await client.send_message(chat_id=chat_id, text=res.response_text)
