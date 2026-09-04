import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import sqlalchemy.ext.asyncio as sqlalchemy_asyncio
from sqlalchemy import select
from sqlalchemy.pool import StaticPool

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

original_create_async_engine = sqlalchemy_asyncio.create_async_engine


def _sqlite_compatible_engine(*args, **kwargs):
    kwargs.pop("pool_recycle", None)
    kwargs.pop("pool_use_lifo", None)
    return original_create_async_engine(*args, **kwargs)


with patch.object(sqlalchemy_asyncio, "create_async_engine", _sqlite_compatible_engine):
    import database
    from database import (
        AIConfig,
        Base,
        Content,
        Message as DBMessage,
        SubscriptionConfig,
        Topic,
        User,
        UserTopicState,
        async_session_maker,
    )
    import handlers
    import keyboards as tg_kb
    import max_messenger_bot.legacy as max_legacy
    import max_messenger_bot.storage as max_storage
    from max_messenger_bot import app as max_app
    from max_messenger_bot.api import MaxApiClient
    from max_messenger_bot.services import common as max_common
    from max_messenger_bot.storage import StorageBase


class DialogueResetConfirmationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = sqlalchemy_asyncio.create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(StorageBase.metadata.create_all)

        self.session_factory = sqlalchemy_asyncio.async_sessionmaker(self.engine, expire_on_commit=False)

        self.patches = [
            patch.object(database, "async_session_maker", self.session_factory),
            patch.object(handlers, "async_session_maker", self.session_factory),
            patch.object(tg_kb, "async_session_maker", self.session_factory),
            patch.object(max_legacy, "async_session_maker", self.session_factory),
            patch.object(max_storage, "async_session_maker", self.session_factory),
            patch.object(max_common, "async_session_maker", self.session_factory),
            patch.object(max_app, "async_session_maker", self.session_factory),
        ]
        for p in self.patches:
            p.start()

        async with self.session_factory() as session:
            session.add(AIConfig(id=1, memory_mode="reset"))
            session.add(SubscriptionConfig(id=1, subscriptions_enabled=False))
            await session.commit()

        handlers.user_locks.clear()

    async def asyncTearDown(self):
        for p in reversed(self.patches):
            p.stop()
        await self.engine.dispose()

    async def test_telegram_ask_reset_generates_token_and_keyboard(self):
        """Telegram /new_dialogue stores reset token and snapshot in FSM state and sends tokenized confirmation keyboard."""
        async with self.session_factory() as session:
            user = User(id=301, current_dialogue_id=3, current_topic_id=None)
            session.add(user)
            await session.commit()

        state_data = {}

        async def fake_get_data():
            return state_data

        async def fake_update_data(**kwargs):
            state_data.update(kwargs)

        state = MagicMock()
        state.get_data = AsyncMock(side_effect=fake_get_data)
        state.update_data = AsyncMock(side_effect=fake_update_data)
        state.clear = AsyncMock(side_effect=state_data.clear)

        message = MagicMock()
        message.from_user = SimpleNamespace(id=301)
        message.answer = AsyncMock()

        await handlers.ask_delete_history(message, state)

        self.assertIn("reset_token", state_data)
        self.assertEqual(state_data["reset_dialogue_id"], 3)
        self.assertIsNone(state_data["reset_topic_id"])

        call_kwargs = message.answer.call_args.kwargs
        reply_markup = call_kwargs["reply_markup"]
        btn_callbacks = [b.callback_data for row in reply_markup.inline_keyboard for b in row]
        token = state_data["reset_token"]
        self.assertIn(f"delete_history_confirm:{token}", btn_callbacks)
        self.assertIn(f"delete_history_cancel:{token}", btn_callbacks)

    async def test_telegram_confirm_reset_executes_under_valid_token(self):
        """Clicking delete_history_confirm with matching token increments dialogue_id and clears test sessions."""
        async with self.session_factory() as session:
            user = User(id=302, current_dialogue_id=2, current_topic_id=None)
            session.add(user)
            await session.commit()

        state_data = {"reset_token": "abc12345", "reset_dialogue_id": 2, "reset_topic_id": None}

        state = MagicMock()
        state.get_data = AsyncMock(return_value=state_data)
        state.clear = AsyncMock()

        callback = MagicMock()
        callback.from_user = SimpleNamespace(id=302)
        callback.data = "delete_history_confirm:abc12345"
        callback.answer = AsyncMock()
        callback.message = MagicMock()
        callback.message.delete = AsyncMock()
        bot = MagicMock()
        bot.send_message = AsyncMock()

        with patch("handlers.render_static_content_telegram", new_callable=AsyncMock):
            await handlers.process_delete_history(callback, state, bot)

        async with self.session_factory() as session:
            db_user = await session.get(User, 302)
            self.assertEqual(db_user.current_dialogue_id, 3)

    async def test_telegram_stale_or_mismatched_token_rejected(self):
        """Stale token or missing state data causes immediate rejection without DB change."""
        async with self.session_factory() as session:
            user = User(id=303, current_dialogue_id=5, current_topic_id=None)
            session.add(user)
            await session.commit()

        state = MagicMock()
        state.get_data = AsyncMock(return_value={})  # state cleared / expired
        state.clear = AsyncMock()

        callback = MagicMock()
        callback.from_user = SimpleNamespace(id=303)
        callback.data = "delete_history_confirm:oldtoken"
        callback.answer = AsyncMock()
        callback.message = MagicMock()
        callback.message.delete = AsyncMock()
        bot = MagicMock()
        bot.send_message = AsyncMock()

        await handlers.process_delete_history(callback, state, bot)

        callback.answer.assert_called_with("Подтверждение устарело или уже использовано.", show_alert=True)
        async with self.session_factory() as session:
            db_user = await session.get(User, 303)
            self.assertEqual(db_user.current_dialogue_id, 5)  # Unchanged!

    async def test_telegram_cancel_reset_clears_state(self):
        """Cancel button clears token and leaves dialogue untouched."""
        state = MagicMock()
        state.clear = AsyncMock()

        callback = MagicMock()
        callback.from_user = SimpleNamespace(id=304)
        callback.data = "delete_history_cancel:tok"
        callback.answer = AsyncMock()
        callback.message = MagicMock()
        callback.message.delete = AsyncMock()
        callback.message.answer = AsyncMock()

        await handlers.cancel_delete_history(callback, state)
        state.clear.assert_called_once()
        callback.message.answer.assert_called_with("Ок. Продолжаем текущий диалог.")

    async def test_max_reset_confirmation_and_execution(self):
        """MAX request_reset_dialogue sends snapshot keyboard; execute_dialogue_reset validates snapshot and resets."""
        async with self.session_factory() as session:
            user = User(id=401, current_dialogue_id=4, current_topic_id=None)
            session.add(user)
            session.add(DBMessage(user_id=401, role="user", content="msg1", dialogue_id=4))
            await session.commit()

        client = MagicMock()
        client.send_message = AsyncMock()

        # Step 1: request reset
        await max_common.request_reset_dialogue(client, chat_id=401, user_id=401)
        send_call = client.send_message.call_args
        attachments = send_call.kwargs.get("attachments") or []
        buttons = attachments[0]["payload"]["buttons"]
        confirm_btn = buttons[0][0]
        self.assertEqual(confirm_btn["payload"], "confirm_reset_dialogue:4:0")

        # Step 2: execute reset with matching snapshot
        await max_common.execute_dialogue_reset(client, chat_id=401, user_id=401, expected_dialogue_id=4, expected_topic_id=0)
        async with self.session_factory() as session:
            db_user = await session.get(User, 401)
            self.assertEqual(db_user.current_dialogue_id, 5)

        # Step 3: try to execute again with stale snapshot -> rejected
        client.send_message.reset_mock()
        await max_common.execute_dialogue_reset(client, chat_id=401, user_id=401, expected_dialogue_id=4, expected_topic_id=0)
        self.assertTrue(any("изменилось" in str(c) for c in client.send_message.call_args_list))
