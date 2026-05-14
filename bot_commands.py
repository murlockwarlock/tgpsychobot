import logging

from aiogram import Bot
from aiogram.types import BotCommand, BotCommandScopeAllPrivateChats, BotCommandScopeChat
from sqlalchemy import select

from config import OWNER_IDS
from database import SubscriptionConfig, User, async_session_maker

MAX_PRIVATE_CHAT_ID = 100_000_000_000

log = logging.getLogger(__name__)


async def load_command_flags() -> tuple[bool, bool, bool]:
    referral_enabled = False
    topics_enabled = True
    subscriptions_enabled = True

    async with async_session_maker() as session:
        sub_config = await session.get(SubscriptionConfig, 1)
        if sub_config:
            referral_enabled = bool(sub_config.referral_enabled)
            topics_enabled = bool(sub_config.topics_enabled)
            subscriptions_enabled = bool(sub_config.subscriptions_enabled)

    return referral_enabled, topics_enabled, subscriptions_enabled


def build_user_commands(
    referral_enabled: bool,
    topics_enabled: bool,
    subscriptions_enabled: bool,
) -> list[BotCommand]:
    commands = [
        BotCommand(command="start", description="Запустить / Перезапустить бота"),
        BotCommand(command="help", description="Помощь"),
    ]
    if topics_enabled:
        commands.append(BotCommand(command="topics", description="Выбрать тему"))
    commands.append(BotCommand(command="new_dialogue", description="Новый диалог"))
    commands.append(BotCommand(command="settings", description="Настройки"))
    if subscriptions_enabled:
        commands.append(BotCommand(command="subscription", description="Подписка"))
    if referral_enabled:
        commands.insert(1, BotCommand(command="ref", description="🤝 Пригласить друзей"))

    return commands


def build_admin_commands(
    referral_enabled: bool,
    topics_enabled: bool,
    subscriptions_enabled: bool,
) -> list[BotCommand]:
    commands = [
        BotCommand(command="start", description="Запустить / Перезапустить бота"),
        BotCommand(command="admin", description="Админ-панель"),
        BotCommand(command="help", description="Помощь (для админов)"),
    ]
    if topics_enabled:
        commands.insert(-1, BotCommand(command="topics", description="Выбрать тему"))
    commands.insert(-1, BotCommand(command="new_dialogue", description="Новый диалог"))
    commands.insert(-1, BotCommand(command="settings", description="Настройки"))
    if subscriptions_enabled:
        commands.insert(-1, BotCommand(command="subscription", description="Подписка"))
    if referral_enabled:
        commands.insert(1, BotCommand(command="ref", description="🤝 Пригласить друзей"))

    return commands


async def build_command_sets() -> tuple[list[BotCommand], list[BotCommand], tuple[bool, bool, bool]]:
    flags = await load_command_flags()
    user_commands = build_user_commands(*flags)
    admin_commands = build_admin_commands(*flags)
    return user_commands, admin_commands, flags


async def get_admin_ids_for_command_scope() -> set[int]:
    admin_ids = {admin_id for admin_id in OWNER_IDS if admin_id < MAX_PRIVATE_CHAT_ID}

    async with async_session_maker() as session:
        result = await session.execute(
            select(User.id).where(User.is_admin == True, User.id < MAX_PRIVATE_CHAT_ID)
        )
        admin_ids.update(row[0] for row in result)

    return admin_ids


async def refresh_default_commands(bot: Bot, user_commands: list[BotCommand]) -> None:
    await bot.delete_my_commands()
    await bot.set_my_commands(user_commands)
    await bot.delete_my_commands(scope=BotCommandScopeAllPrivateChats())
    await bot.set_my_commands(user_commands, scope=BotCommandScopeAllPrivateChats())


async def refresh_chat_commands(bot: Bot, chat_id: int, commands: list[BotCommand]) -> None:
    scope = BotCommandScopeChat(chat_id=chat_id)
    await bot.delete_my_commands(scope=scope)
    await bot.set_my_commands(commands, scope=scope)


async def refresh_commands_for_user(bot: Bot, user_id: int, is_admin_user: bool) -> None:
    try:
        user_commands, admin_commands, _ = await build_command_sets()
        commands = admin_commands if is_admin_user else user_commands
        await refresh_chat_commands(bot, user_id, commands)
        log.info(
            "Refreshed command scope for user_id=%s is_admin=%s commands=%s",
            user_id,
            is_admin_user,
            [command.command for command in commands],
        )
    except Exception as exc:
        log.warning("Could not refresh command scope for user_id=%s: %s", user_id, exc)
