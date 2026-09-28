import html
import re
from typing import Optional

from aiogram import Bot

from max_messenger_bot.formatting import markdown_to_html
from response_button_renderers import telegram_response_buttons_markup
from response_buttons import extract_response_buttons
from telegram_birthdate import format_birthdate


MAILING_AUDIENCE_LABELS = {
    "all": "Всем пользователям",
    "no_dialogue": "Кто не начал диалог",
    "no_subscription": "Кто ни разу не платил",
    "active_subscription": "Активным подписчикам",
    "inactive_subscription": "Кто отменил подписку",
    "self": "👤 Только себе (тест)",
    "birthday_today": "🎂 У кого сегодня день рождения",
}

BIRTHDAY_MAILING_TYPE = "birthday"
BIRTHDAY_PLACEHOLDER_HINT = "{name}, {first_name}, {username}, {birthdate}"
DEFAULT_BIRTHDAY_TEMPLATE = (
    "С днем рождения, {name}! 🎉\n\n"
    "Пусть этот день принесет вам радость, тепло и внутреннюю опору.\n\n"
    "Для вас сегодня есть особое предложение: добавьте сюда промокод или ссылку."
)


def get_mailing_audience_label(audience: str) -> str:
    return MAILING_AUDIENCE_LABELS.get(audience, audience)


def is_birthday_mailing(mailing) -> bool:
    return getattr(mailing, "recurring_type", None) == BIRTHDAY_MAILING_TYPE


def get_mailing_status_label(mailing) -> str:
    if is_birthday_mailing(mailing):
        return "🟢 Активен" if getattr(mailing, "is_enabled", True) else "⏸ Отключен"

    status_map = {
        "pending": "⏳ Ожидает",
        "sending": "🚀 Отправляется",
        "completed": "✅ Завершена",
        "failed": "❌ Ошибка",
    }
    return status_map.get(getattr(mailing, "status", ""), "❓")


def render_mailing_text(text: Optional[str], user) -> Optional[str]:
    if not text:
        return text

    first_name = (getattr(user, "name", None) or getattr(user, "first_name", None) or "друг").strip()
    username_raw = getattr(user, "username", None) or ""
    username = f"@{username_raw}" if username_raw else ""
    birthdate = format_birthdate(
        getattr(user, "birth_day", None),
        getattr(user, "birth_month", None),
        getattr(user, "birth_year", None),
    ) or ""

    replacements = {
        "{name}": html.escape(first_name),
        "{first_name}": html.escape(first_name),
        "{username}": html.escape(username),
        "{birthdate}": html.escape(birthdate),
    }

    rendered = text
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def parse_mailing_text(text: Optional[str]) -> tuple[str, list]:
    return extract_response_buttons(text or "")


def mailing_text_to_html(text: str) -> str:
    if not text:
        return ""
    if re.search(r"</?(?:b|strong|i|em|u|s|code|pre|blockquote|a)(?:\s|>)", text, re.IGNORECASE):
        return text
    return markdown_to_html(text)


async def send_mailing_content(bot: Bot, user_id: int, mailing, *, rendered_text: Optional[str] = None):
    text = rendered_text if rendered_text is not None else getattr(mailing, "text", None)
    media = getattr(mailing, "media_file_id", None)
    media_type = getattr(mailing, "media_file_type", None)
    position = getattr(mailing, "media_position", None) or "media_top"
    clean_text, response_button_rows = parse_mailing_text(text)
    html_text = mailing_text_to_html(clean_text)
    if not html_text and response_button_rows:
        html_text = "Выберите действие:"
    response_markup = telegram_response_buttons_markup(response_button_rows)

    if media and html_text and len(html_text) <= 1024 and position == "media_top":
        if media_type == "photo":
            await bot.send_photo(
                user_id,
                media,
                caption=html_text,
                parse_mode="HTML",
                reply_markup=response_markup,
            )
            return
        if media_type == "video":
            await bot.send_video(
                user_id,
                media,
                caption=html_text,
                parse_mode="HTML",
                reply_markup=response_markup,
            )
            return

    async def send_media_only():
        if not media:
            return
        if media_type == "photo":
            await bot.send_photo(user_id, media)
        elif media_type == "video":
            await bot.send_video(user_id, media)

    async def send_text_only():
        if html_text:
            await bot.send_message(
                user_id,
                html_text,
                parse_mode="HTML",
                reply_markup=response_markup,
            )

    if position == "media_top":
        await send_media_only()
        await send_text_only()
    else:
        await send_text_only()
        await send_media_only()
