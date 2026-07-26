import os

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession


def get_telegram_proxy() -> str | None:
    proxy = os.getenv("TELEGRAM_PROXY", "").strip().strip('"').strip("'")
    return proxy or None


def create_telegram_bot(
    token: str,
    *,
    default: DefaultBotProperties | None = None,
) -> Bot:
    session = AiohttpSession(proxy=get_telegram_proxy())
    return Bot(token=token, session=session, default=default)
