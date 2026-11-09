import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Any

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import (
    AutomationConversationState,
    AutomationStepTransition,
    CertificateFollowupConfig,
    CertificateIssuanceReceipt,
    MediaLibrary,
    User,
)
from event_content_validator import TelegramHtmlValidator

log = logging.getLogger(__name__)

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


def validate_caption_template(template: str) -> None:
    clean = (template or "").strip()
    if not clean:
        raise ValueError("Подпись к сертификату не может быть пустой.")
    for var in re.findall(r"\{([^{}]+)\}", clean):
        var_name = var.strip()
        if var_name not in ALLOWED_CAPTION_VARS:
            allowed_str = ", ".join(f"{{{v}}}" for v in sorted(ALLOWED_CAPTION_VARS))
            raise ValueError(f"Неизвестная переменная {{{var_name}}}. Доступны: {allowed_str}")
    TelegramHtmlValidator.validate(clean)


def validate_invitation_text(text: str) -> None:
    clean = (text or "").strip()
    if not clean:
        raise ValueError("Текст приглашения не может быть пустым.")
    for var in re.findall(r"\{([^{}]+)\}", clean):
        var_name = var.strip()
        if var_name not in ALLOWED_INVITE_VARS:
            raise ValueError(f"Неизвестная переменная {{{var_name}}}. Доступна только {{name}}")
    TelegramHtmlValidator.validate(clean)


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


def render_caption(template: str, context: dict[str, Any]) -> str:
    if not template:
        template = DEFAULT_CAPTION_TEMPLATE
    name = str(context.get("name") or "").strip()
    fio = str(context.get("fio") or context.get("recipient") or name).strip()
    date_str = str(context.get("date") or "").strip()
    values = {
        "name": name or fio,
        "fio": fio or name,
        "recipient": fio or name,
        "date": date_str,
    }
    result = template
    for key, val in values.items():
        result = result.replace(f"{{{key}}}", val)
    return result


def render_invitation(text: str, context: dict[str, Any]) -> str:
    if not text:
        text = DEFAULT_INVITATION_TEXT
    name = str(context.get("name") or "").strip()
    return text.replace("{name}", name)


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

    config = CertificateFollowupConfig(
        media_id=media_id,
        topic_id=topic_id,
        is_active=True,
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


def build_followup_telegram_keyboard(config: CertificateFollowupConfig) -> InlineKeyboardMarkup:
    buttons = parse_buttons_config(config.buttons_config_json)
    rows = []
    for btn in buttons:
        callback_data = f"cert_meet:{btn['id']}:{config.id}"
        rows.append([InlineKeyboardButton(text=btn["text"], callback_data=callback_data)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_followup_max_keyboard(config: CertificateFollowupConfig) -> list[dict[str, Any]]:
    from max_messenger_bot.keyboards import callback_button, inline_keyboard

    buttons = parse_buttons_config(config.buttons_config_json)
    rows = []
    for btn in buttons:
        callback_data = f"cert_meet:{btn['id']}:{config.id}"
        rows.append([callback_button(btn["text"], callback_data)])
    return inline_keyboard(rows)


async def get_or_create_receipt(
    session: AsyncSession,
    user_id: int,
    topic_id: int,
    dialogue_id: int,
    issuance_key: str,
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
        return existing, False

    receipt = CertificateIssuanceReceipt(
        user_id=user_id,
        topic_id=topic_id,
        dialogue_id=dialogue_id,
        issuance_key=issuance_key,
        status="started",
    )
    session.add(receipt)
    await session.flush()
    return receipt, True


async def mark_receipt_cert_sent(session: AsyncSession, receipt_id: int, message_id: str | None = None) -> None:
    receipt = await session.get(CertificateIssuanceReceipt, receipt_id)
    if receipt:
        receipt.status = "cert_sent"
        if message_id:
            receipt.certificate_message_id = str(message_id)
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


async def handle_cert_meet_click(
    session_maker: Any,
    user_id: int,
    topic_id: int,
    dialogue_id: int,
    action_id: str,
    config_id: int,
) -> tuple[str, str]:
    async with session_maker() as session:
        config = await session.get(CertificateFollowupConfig, config_id)
        if config is None:
            buttons = DEFAULT_BUTTONS_CONFIG
        else:
            buttons = parse_buttons_config(config.buttons_config_json)

        button_match = next((b for b in buttons if b["id"] == action_id), None)
        if not button_match:
            button_match = DEFAULT_BUTTONS_CONFIG[0]

        target_step = button_match["target_step"]
        response_text = button_match.get("response_text") or "Спасибо за ответ!"

        await transition_user_to_stage(
            session,
            user_id=user_id,
            topic_id=topic_id,
            dialogue_id=dialogue_id,
            target_step=target_step,
        )
        await session.commit()
        return response_text, target_step


async def handle_cert_callback_max(client: Any, callback: Any) -> None:
    data = str(callback.payload or "")
    parts = data.split(":")
    if len(parts) >= 3 and parts[0] == "cert_meet":
        action_id = parts[1]
        try:
            config_id = int(parts[2])
        except ValueError:
            config_id = 0
    else:
        return

    from database import async_session_maker
    user_id = callback.sender.user_id
    chat_id = callback.chat_id

    async with async_session_maker() as session:
        user = await session.get(User, user_id)
        topic_id = user.current_topic_id if user and user.current_topic_id else 0
        dialogue_id = user.current_dialogue_id if user and user.current_dialogue_id else 1

    await client.answer_callback(callback.callback_id)
    response_text, _ = await handle_cert_meet_click(
        async_session_maker,
        user_id=user_id,
        topic_id=topic_id,
        dialogue_id=dialogue_id,
        action_id=action_id,
        config_id=config_id,
    )
    await client.send_message(chat_id=chat_id, text=response_text)

