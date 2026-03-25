import asyncio
import logging
import os
from logging.handlers import RotatingFileHandler
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.webhook.aiohttp_server import SimpleRequestHandler
from aiohttp import web
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from aiogram.client.default import DefaultBotProperties
from aiogram.types import BotCommand, BotCommandScopeAllPrivateChats, BotCommandScopeChat
from sqlalchemy import select

from config import BOT_TOKEN, OWNER_IDS
from handlers import router
from database import init_db, async_session_maker, User
from background_worker import process_queue, process_mailings
from scheduler import check_subscriptions, check_kie_credit_balance
from webhooks import setup_webhooks

WEB_SERVER_HOST = '0.0.0.0'
APP_PORT = int(os.environ.get('APP_PORT', 8080))
WEBHOOK_PATH_PREFIX = os.environ.get('WEBHOOK_PATH_PREFIX', '')
TELEGRAM_WEBHOOK_PATH = f'{WEBHOOK_PATH_PREFIX}/webhook'
BASE_WEBHOOK_URL = os.environ.get("BASE_WEBHOOK_URL")
if not BASE_WEBHOOK_URL:
    logging.critical("BASE_WEBHOOK_URL не задан в env (например, https://bots.example.com)")
    raise ValueError("BASE_WEBHOOK_URL не задан в env")

WEBHOOK_SETUP_RETRIES = int(os.environ.get("WEBHOOK_SETUP_RETRIES", 10))
WEBHOOK_RETRY_DELAY_SEC = int(os.environ.get("WEBHOOK_RETRY_DELAY_SEC", 15))


async def configure_webhook(bot: Bot, webhook_url: str):
    last_error = None
    for attempt in range(1, WEBHOOK_SETUP_RETRIES + 1):
        try:
            logging.info(
                f"Setting webhook to: {webhook_url} "
                f"(attempt {attempt}/{WEBHOOK_SETUP_RETRIES})"
            )
            await bot.set_webhook(
                webhook_url,
                drop_pending_updates=True
            )
            logging.info("Webhook configured successfully.")
            return
        except Exception as exc:
            last_error = exc
            logging.warning(
                f"Failed to set webhook on attempt {attempt}/{WEBHOOK_SETUP_RETRIES}: {exc}"
            )
            if attempt < WEBHOOK_SETUP_RETRIES:
                await asyncio.sleep(WEBHOOK_RETRY_DELAY_SEC)

    raise last_error


async def on_startup(bot: Bot, dispatcher: Dispatcher):
    logging.info("Configuring startup...")
    await init_db()


    user_commands = [
        BotCommand(command="start", description="Запустить / Перезапустить бота"),
        BotCommand(command="help", description="Помощь")
    ]

    admin_commands = [
        BotCommand(command="start", description="Запустить / Перезапустить бота"),
        BotCommand(command="admin", description="Админ-панель"),
        BotCommand(command="help", description="Помощь (для админов)")
    ]

    await bot.set_my_commands(user_commands, scope=BotCommandScopeAllPrivateChats())

    all_admin_ids = set(OWNER_IDS)
    try:
        async with async_session_maker() as session:
            admin_result = await session.execute(
                select(User.id).where(User.is_admin == True)
            )
            db_admin_ids = {row[0] for row in admin_result}
            all_admin_ids.update(db_admin_ids)
    except Exception as e:
        logging.error(f"Failed to fetch admin IDs from DB for setting commands: {e}")

    for admin_id in all_admin_ids:
        try:
            await bot.set_my_commands(admin_commands, scope=BotCommandScopeChat(chat_id=admin_id))
        except Exception as e:
            logging.warning(f"Could not set admin commands for {admin_id}: {e}")

    logging.info(f"Set default commands for users and special commands for {len(all_admin_ids)} admins.")

    webhook_url = f"{BASE_WEBHOOK_URL}{TELEGRAM_WEBHOOK_PATH}"
    await configure_webhook(bot, webhook_url)

    logging.info("Running initial subscription check on startup...")
    try:
        await check_subscriptions(bot)
        logging.info("Initial subscription check complete.")
    except Exception as e:
        logging.error(f"Error during initial subscription check: {e}")

    try:
        await check_kie_credit_balance(bot)
    except Exception as e:
        logging.error(f"Error during initial KIE credit balance check: {e}")

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(check_subscriptions, 'interval', minutes=15, args=(bot,))
    scheduler.add_job(check_kie_credit_balance, 'interval', minutes=15, args=(bot,))
    scheduler.start()

    dispatcher.shutdown.register(scheduler.shutdown)

    bg_task_factories = {
        'process_queue': lambda: process_queue(bot),
        'process_mailings': lambda: process_mailings(bot),
    }
    bg_restart_counts: dict[str, int] = {}
    MAX_BG_RESTARTS = 5

    def _start_bg_task(name: str):
        task = asyncio.create_task(bg_task_factories[name](), name=name)
        task.add_done_callback(_on_bg_task_done)
        bg_tasks = dispatcher.get('_bg_tasks', {})
        bg_tasks[name] = task
        dispatcher['_bg_tasks'] = bg_tasks
        return task

    def _on_bg_task_done(task: asyncio.Task):
        name = task.get_name()
        if task.cancelled():
            logging.info(f"Background task {name} cancelled.")
            return
        exc = task.exception()
        if exc:
            count = bg_restart_counts.get(name, 0) + 1
            bg_restart_counts[name] = count
            logging.error(f"Background task {name} crashed ({count}/{MAX_BG_RESTARTS}): {exc}", exc_info=exc)
            if count >= MAX_BG_RESTARTS:
                logging.error(f"Background task {name} exceeded max restarts ({MAX_BG_RESTARTS}). Giving up.")
                return
            delay = min(5 * (2 ** (count - 1)), 300)
            logging.info(f"Restarting background task {name} in {delay}s...")
            asyncio.get_event_loop().call_later(delay, _start_bg_task, name)

    _start_bg_task('process_queue')
    _start_bg_task('process_mailings')
    logging.info("Startup complete.")


async def on_shutdown(bot: Bot, dispatcher: Dispatcher):
    logging.info("Shutting down...")
    bg_tasks = dispatcher.get('_bg_tasks', {})
    for task in bg_tasks.values():
        if not task.done():
            task.cancel()
    for task in bg_tasks.values():
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
    await bot.delete_webhook(drop_pending_updates=True)
    await bot.session.close()
    logging.info("Shutdown complete.")


async def aiogram_on_startup(app: web.Application):
    bot: Bot = app['bot']
    dp: Dispatcher = app['dp']
    await dp.emit_startup(bot=bot, dispatcher=dp)

async def aiogram_on_shutdown(app: web.Application):
    bot: Bot = app['bot']
    dp: Dispatcher = app['dp']
    await dp.emit_shutdown(bot=bot, dispatcher=dp)


class _MskFormatter(logging.Formatter):
    """Форматтер с временем МСК (UTC+3) для платёжного лога."""
    def formatTime(self, record, datefmt=None):
        from datetime import datetime, timezone, timedelta
        MSK = timezone(timedelta(hours=3))
        ct = datetime.fromtimestamp(record.created, tz=MSK)
        return ct.strftime(datefmt) if datefmt else ct.isoformat()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    payment_logger = logging.getLogger("payment_events")
    payment_logger.setLevel(logging.INFO)
    _payment_handler = RotatingFileHandler(
        os.path.join(log_dir, f"payment_events_{APP_PORT}.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    _payment_handler.setFormatter(_MskFormatter("%(asctime)s | %(message)s", datefmt="%d.%m.%Y %H:%M:%S"))
    payment_logger.addHandler(_payment_handler)

    if not BOT_TOKEN:
        logging.critical("BOT_TOKEN не найден. Остановка.")
        return
    if not os.environ.get('SERVER_IP'):
        logging.critical("SERVER_IP не задан. Остановка.")
        return

    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)

    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    dp.include_router(router)

    app = web.Application()

    app['bot'] = bot
    app['dp'] = dp

    app.on_startup.append(aiogram_on_startup)
    app.on_shutdown.append(aiogram_on_shutdown)

    setup_webhooks(app, bot, storage, prefix='')

    webhook_request_handler = SimpleRequestHandler(
        dispatcher=dp,
        bot=bot,
    )
    webhook_request_handler.register(app, path='/webhook')

    logging.info(f"Starting web server on {WEB_SERVER_HOST}:{APP_PORT}")
    try:
        web.run_app(app, host=WEB_SERVER_HOST, port=APP_PORT)
    finally:
        close_session = getattr(bot.session, "close", None)
        if callable(close_session):
            asyncio.run(close_session())


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Бот остановлен.")
