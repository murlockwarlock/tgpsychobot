from __future__ import annotations

import logging
import os
import re
from typing import Any

from aiogram import Bot
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

log = logging.getLogger(__name__)


def clean_html_for_max(text: str) -> str:
    """Strips HTML tags like <b>, </b>, <i>, etc. for MAX messenger text."""
    return re.sub(r"<[^>]+>", "", text)


def build_subscribe_keyboard() -> InlineKeyboardMarkup:
    """Standard tariff purchase/renewal keyboard."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="💳 Оформить подписку", callback_data="show_subscription_info_from_chat")]
    ])


async def send_notification_transport(
    bot: Bot,
    recipient_id: int,
    text: str,
    keyboard_type: str | None = None,
    parse_mode: str | None = None,
    reply_markup: Any | None = None,
) -> bool:
    """
    Unified TG / MAX notification delivery adapter.
    - If recipient_id < 100_000_000_000: sends via Telegram bot.send_message.
    - If recipient_id >= 100_000_000_000: sends via MaxApiClient.
    Returns True if successfully delivered, False otherwise.
    Never throws exceptions.
    """
    if reply_markup is None and keyboard_type == "subscribe":
        reply_markup = build_subscribe_keyboard()

    if recipient_id < 100_000_000_000:
        try:
            await bot.send_message(
                recipient_id,
                text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
            return True
        except Exception as e:
            log.error("Failed to send Telegram notification to %s: %s", recipient_id, e)
            return False
    else:
        token = os.environ.get("MAX_BOT_TOKEN")
        if not token:
            log.warning("Cannot send MAX message to %s: MAX_BOT_TOKEN not configured in env", recipient_id)
            return False
        base_url = os.environ.get("MAX_API_BASE", "https://platform-api.max.ru")

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
                return True
        except Exception as e:
            log.error("Failed to send MAX notification to %s: %s", recipient_id, e, exc_info=e)
            return False
