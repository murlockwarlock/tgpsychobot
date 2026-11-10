from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from aiogram import Bot
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from max_messenger_bot.api import MaxApiError
from translation_service import translate

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NotificationDeliveryResult:
    delivered: bool
    retryable: bool = True
    reason: str | None = None
    status: int | None = None
    error_code: str | None = None


def is_bot_blocked_reason(reason: str | None) -> bool:
    """
    Checks whether the delivery failure reason indicates that the user blocked the bot
    or the recipient chat was suspended/denied.
    Specific to recipient block/suspend - other Forbidden errors remain unsuppressed.
    """
    if not reason:
        return False
    r = str(reason).lower()
    if "bot was blocked" in r or "bot_blocked" in r or "blocked by the user" in r:
        return True
    if "max_chat_suspended" in r or "chat.denied" in r or "error.dialog.suspended" in r:
        return True
    if "user.blocked" in r or "suspended recipient" in r or "dialog.suspended" in r:
        return True
    return False


def clean_html_for_max(text: str) -> str:
    """Strips HTML tags like <b>, </b>, <i>, etc. for MAX messenger text."""
    return re.sub(r"<[^>]+>", "", text)


def build_subscribe_keyboard(locale: str = "ru") -> InlineKeyboardMarkup:
    """Standard tariff purchase/renewal keyboard."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text=translate("ui.subscription.subscribe", locale, fallback="💳 Оформить подписку"),
            callback_data="show_subscription_info_from_chat",
        )]
    ])


def build_robokassa_pending_keyboard(locale: str = "ru") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text=translate("ui.subscription.check_status", locale, fallback="🔄 Проверить статус"),
            callback_data="sub_retry_now",
        )],
        [InlineKeyboardButton(
            text=translate("ui.subscription.cancel_retry", locale, fallback="🔄 Отменить и оформить заново"),
            callback_data="sub_cancel_retry",
        )],
        [InlineKeyboardButton(
            text=translate("ui.subscription.enter_promo", locale, fallback="🎁 Ввести промокод"),
            callback_data="sub_enter_promo",
        )],
    ])


async def send_notification_transport_result(
    bot: Bot,
    recipient_id: int,
    text: str,
    keyboard_type: str | None = None,
    parse_mode: str | None = None,
    reply_markup: Any | None = None,
) -> NotificationDeliveryResult:
    """
    Unified TG / MAX notification delivery adapter.
    - If recipient_id < 100_000_000_000: sends via Telegram bot.send_message.
    - If recipient_id >= 100_000_000_000: sends via MaxApiClient.
    Returns a delivery result. Terminal recipient failures are marked non-retryable.
    Never throws exceptions.
    """
    if reply_markup is None:
        if keyboard_type == "subscribe":
            reply_markup = build_subscribe_keyboard()
        elif keyboard_type == "robokassa_pending":
            reply_markup = build_robokassa_pending_keyboard()

    if recipient_id < 100_000_000_000:
        try:
            await bot.send_message(
                recipient_id,
                text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
            return NotificationDeliveryResult(delivered=True)
        except Exception as e:
            err_text = f"{str(e)} {getattr(e, 'message', '')}".lower()
            if "bot was blocked" in err_text or "blocked by the user" in err_text:
                log.warning("Telegram notification delivery blocked by user recipient_id=%s: %s", recipient_id, e)
                return NotificationDeliveryResult(
                    delivered=False,
                    retryable=False,
                    reason="telegram_bot_blocked",
                    status=403,
                    error_code="bot_blocked",
                )
            log.error("Failed to send Telegram notification to %s: %s", recipient_id, e)
            return NotificationDeliveryResult(
                delivered=False,
                retryable=True,
                reason=f"telegram:{type(e).__name__}",
            )
    else:
        token = os.environ.get("MAX_BOT_TOKEN")
        if not token:
            log.warning("Cannot send MAX message to %s: MAX_BOT_TOKEN not configured in env", recipient_id)
            return NotificationDeliveryResult(
                delivered=False,
                retryable=True,
                reason="max_token_missing",
            )
        base_url = os.environ.get("MAX_API_BASE", "https://platform-api2.max.ru")

        attachments = None
        if reply_markup and hasattr(reply_markup, "inline_keyboard"):
            max_rows = []
            for row in reply_markup.inline_keyboard:
                max_row = []
                for btn in row:
                    if getattr(btn, "callback_data", None):
                        max_row.append({"type": "callback", "text": btn.text, "payload": btn.callback_data})
                    elif getattr(btn, "url", None):
                        max_row.append({"type": "link", "text": btn.text, "url": btn.url})
                if max_row:
                    max_rows.append(max_row)
            if max_rows:
                attachments = [{"type": "inline_keyboard", "payload": {"buttons": max_rows}}]

        try:
            from max_messenger_bot.api import MaxApiClient
            from max_messenger_bot.models import MAX_ID_OFFSET

            async with MaxApiClient(token=token, base_url=base_url) as client:
                max_api_user_id = recipient_id - MAX_ID_OFFSET
                clean_text = clean_html_for_max(text)
                await client.send_message(user_id=max_api_user_id, text=clean_text, attachments=attachments)
                return NotificationDeliveryResult(delivered=True)
        except MaxApiError as e:
            err_msg = f"{str(e)} {getattr(e, 'error_code', '')}".lower()
            is_blocked = (
                getattr(e, "is_suspended_recipient", False)
                or getattr(e, "error_code", None) in ("chat.denied", "user.blocked", "blocked")
                or "error.dialog.suspended" in err_msg
                or "chat.denied" in err_msg
                or (getattr(e, "status", None) == 403 and "blocked" in err_msg)
            )
            if is_blocked:
                log.warning(
                    "MAX notification delivery terminal for suspended recipient recipient_id=%s status=%s code=%s",
                    recipient_id,
                    e.status,
                    e.error_code,
                )
                return NotificationDeliveryResult(
                    delivered=False,
                    retryable=False,
                    reason="max_chat_suspended",
                    status=e.status,
                    error_code=e.error_code,
                )
            else:
                log.error("Failed to send MAX notification to %s: %s", recipient_id, e, exc_info=e)
            return NotificationDeliveryResult(
                delivered=False,
                retryable=True,
                reason=f"max:{e.error_code or type(e).__name__}",
                status=e.status,
                error_code=e.error_code,
            )
        except Exception as e:
            log.error("Failed to send MAX notification to %s: %s", recipient_id, e, exc_info=e)
            return NotificationDeliveryResult(
                delivered=False,
                retryable=True,
                reason=f"max:{type(e).__name__}",
            )


async def send_notification_transport(
    bot: Bot,
    recipient_id: int,
    text: str,
    keyboard_type: str | None = None,
    parse_mode: str | None = None,
    reply_markup: Any | None = None,
) -> bool:
    result = await send_notification_transport_result(
        bot,
        recipient_id,
        text,
        keyboard_type=keyboard_type,
        parse_mode=parse_mode,
        reply_markup=reply_markup,
    )
    return result.delivered
