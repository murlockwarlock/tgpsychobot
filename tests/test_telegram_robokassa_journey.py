import importlib
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

os.environ.setdefault("BOT_TOKEN", "123456:test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.methods import EditMessageText, SendMessage
from aiogram.types import CallbackQuery, Chat, Message, Update, User as TelegramUser
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database import Base, SubscriptionConfig, SubscriptionPlan, User, UserSubscription


class ValidatingTelegramSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls = []
        self._next_message_ids = {}

    async def close(self):
        return None

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        type(method).model_validate(method.model_dump())
        if isinstance(method, SendMessage):
            chat_id = int(method.chat_id)
            message_id = self._next_message_ids.get(chat_id, 1)
            self._next_message_ids[chat_id] = message_id + 1
            return Message(
                message_id=message_id,
                date=datetime.now(timezone.utc),
                chat=Chat(id=chat_id, type="private"),
                from_user=TelegramUser(id=999, is_bot=True, first_name="Test bot"),
                text=method.text,
                reply_markup=method.reply_markup,
            ).as_(bot)
        if isinstance(method, EditMessageText):
            return Message(
                message_id=int(method.message_id),
                date=datetime.now(timezone.utc),
                chat=Chat(id=int(method.chat_id), type="private"),
                from_user=TelegramUser(id=999, is_bot=True, first_name="Test bot"),
                text=method.text,
                reply_markup=method.reply_markup,
            ).as_(bot)
        return True

    async def stream_content(self, *args, **kwargs):
        if False:
            yield b""


@pytest_asyncio.fixture
async def telegram_journey_db(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'telegram-robokassa-journey.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    import handlers
    import keyboards

    handler_module = handlers
    if handler_module.router.parent_router is not None:
        handler_module = importlib.reload(handler_module)
    monkeypatch.setattr(handler_module, "async_session_maker", sessions)
    monkeypatch.setattr(keyboards, "async_session_maker", sessions)
    try:
        yield sessions, handler_module
    finally:
        if handler_module.router.parent_router is not None:
            importlib.reload(handler_module)
        await engine.dispose()


def _telegram_message(bot, user_id, message_id, text):
    return Message(
        message_id=message_id,
        date=datetime.now(timezone.utc),
        chat=Chat(id=user_id, type="private"),
        from_user=TelegramUser(id=user_id, is_bot=False, first_name="Journey user"),
        text=text,
    ).as_(bot)


async def _feed_callback(dispatcher, bot, user_id, message_id, data, update_id):
    message = _telegram_message(bot, user_id, message_id, "Подписка")
    callback = CallbackQuery(
        id=f"journey-{update_id}",
        from_user=message.from_user,
        chat_instance="journey",
        message=message,
        data=data,
    ).as_(bot)
    await dispatcher.feed_update(
        bot,
        Update(update_id=update_id, callback_query=callback),
    )


def _keyboard_callbacks(method):
    return {
        button.callback_data
        for row in (method.reply_markup.inline_keyboard if method.reply_markup else [])
        for button in row
        if button.callback_data
    }


@pytest.mark.asyncio
async def test_telegram_robokassa_buttons_follow_real_dispatcher_journeys(telegram_journey_db, monkeypatch):
    sessions, handlers = telegram_journey_db
    plan_id = 9400
    user_ids = (9401, 9402, 9403)
    now = datetime.utcnow()
    async with sessions() as session:
        session.add(SubscriptionConfig(id=1, subscriptions_enabled=True, robokassa_password_2="password-2"))
        session.add(
            SubscriptionPlan(
                id=plan_id,
                name="Robokassa journey plan",
                price=500.0,
                duration_value=1,
                duration_unit="months",
                is_active=True,
            )
        )
        for index, user_id in enumerate(user_ids, start=1):
            session.add(User(id=user_id, first_name="Journey user"))
            session.add(
                UserSubscription(
                    user_id=user_id,
                    plan_id=plan_id,
                    start_date=now - timedelta(days=31),
                    end_date=now - timedelta(hours=2),
                    auto_renewal=True,
                    payment_provider="Robokassa",
                    payment_method_id=f"pm-{user_id}",
                    pending_robokassa_invoice_id=9500 + index,
                    payment_attempt_count=1,
                    last_payment_attempt=now - timedelta(minutes=30),
                )
            )
        await session.commit()

    dispatcher = Dispatcher(storage=MemoryStorage())
    dispatcher.include_router(handlers.router)
    session = ValidatingTelegramSession()
    bot = Bot("123456:test", session=session)

    cancel_user = user_ids[0]
    await dispatcher.feed_update(
        bot,
        Update(
            update_id=1,
            message=_telegram_message(bot, cancel_user, 101, "⭐️ Подписка"),
        ),
    )
    initial = next(method for method in reversed(session.calls) if isinstance(method, SendMessage))
    assert {"sub_retry_now", "sub_cancel_retry", "sub_enter_promo"}.issubset(_keyboard_callbacks(initial))
    await _feed_callback(dispatcher, bot, cancel_user, 1, "sub_cancel_retry", 2)
    async with sessions() as db:
        cancelled = await db.scalar(select(UserSubscription).where(UserSubscription.user_id == cancel_user))
    assert cancelled.auto_renewal is False
    assert any(isinstance(method, SendMessage) and "Подписка" in method.text for method in session.calls)

    promo_user = user_ids[1]
    await dispatcher.feed_update(
        bot,
        Update(update_id=3, message=_telegram_message(bot, promo_user, 201, "⭐️ Подписка")),
    )
    assert next(method for method in reversed(session.calls) if isinstance(method, SendMessage))
    await _feed_callback(dispatcher, bot, promo_user, 1, "sub_enter_promo", 4)
    promo_prompt = next(method for method in reversed(session.calls) if isinstance(method, EditMessageText))
    assert "промокод" in promo_prompt.text.lower()
    promo_state = dispatcher.fsm.get_context(bot=bot, chat_id=promo_user, user_id=promo_user)
    assert await promo_state.get_state() == "UserStates:awaiting_promo_code"
    await _feed_callback(dispatcher, bot, promo_user, 1, "back_to_sub_info", 5)
    assert await promo_state.get_state() is None
    assert isinstance(next(method for method in reversed(session.calls) if isinstance(method, SendMessage)), SendMessage)

    retry_user = user_ids[2]
    monkeypatch.setattr(handlers, "check_robokassa_op_state", AsyncMock(return_value="pending"))
    await dispatcher.feed_update(
        bot,
        Update(update_id=6, message=_telegram_message(bot, retry_user, 301, "⭐️ Подписка")),
    )
    assert next(method for method in reversed(session.calls) if isinstance(method, SendMessage))
    answers_before_retry = sum(method.__class__.__name__ == "AnswerCallbackQuery" for method in session.calls)
    await _feed_callback(dispatcher, bot, retry_user, 1, "sub_retry_now", 7)
    assert any(isinstance(method, SendMessage) and "обработке" in method.text for method in session.calls)
    assert sum(method.__class__.__name__ == "AnswerCallbackQuery" for method in session.calls) == answers_before_retry + 1
    await bot.session.close()
    await dispatcher.storage.close()
