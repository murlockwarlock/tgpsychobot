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
    from max_messenger_bot.models import IncomingCallback, IncomingMessage, Sender
    from max_messenger_bot.services import common as max_common
    from max_messenger_bot.storage import StateStore, StorageBase


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
            return dict(state_data)

        async def fake_update_data(**kwargs):
            state_data.update(kwargs)

        state = MagicMock()
        state.get_data = AsyncMock(side_effect=fake_get_data)
        state.update_data = AsyncMock(side_effect=fake_update_data)

        message = MagicMock()
        message.from_user = SimpleNamespace(id=301)
        message.answer = AsyncMock()

        await handlers.ask_delete_history(message, state)

        self.assertIn("reset_token", state_data)
        token = state_data["reset_token"]
        self.assertEqual(state_data["reset_dialogue_id"], 3)
        self.assertIsNone(state_data["reset_topic_id"])

        message.answer.assert_called_once()
        _, kwargs = message.answer.call_args
        reply_markup = kwargs.get("reply_markup")
        self.assertIsNotNone(reply_markup)
        confirm_btn = reply_markup.inline_keyboard[0][0]
        cancel_btn = reply_markup.inline_keyboard[0][1]
        self.assertEqual(confirm_btn.callback_data, f"delete_history_confirm:{token}")
        self.assertEqual(cancel_btn.callback_data, f"delete_history_cancel:{token}")

    async def test_telegram_old_cancel_token_does_not_clear_new_confirmation(self):
        """Clicking an old Cancel button with Token A when Token B is active leaves Token B valid."""
        async with self.session_factory() as session:
            user = User(id=302, current_dialogue_id=1, current_topic_id=None)
            session.add(user)
            await session.commit()

        state_data = {
            "reset_token": "token_B",
            "reset_dialogue_id": 1,
            "reset_topic_id": None,
            "other_flow_key": "preserved_val",
        }

        async def fake_get_data():
            return dict(state_data)

        async def fake_update_data(**kwargs):
            state_data.update(kwargs)

        state = MagicMock()
        state.get_data = AsyncMock(side_effect=fake_get_data)
        state.update_data = AsyncMock(side_effect=fake_update_data)

        cb_old = MagicMock()
        cb_old.from_user = SimpleNamespace(id=302)
        cb_old.data = "delete_history_cancel:token_A"
        cb_old.answer = AsyncMock()
        cb_old.message = MagicMock()
        cb_old.message.delete = AsyncMock()
        cb_old.message.answer = AsyncMock()

        await handlers.cancel_delete_history(cb_old, state)

        # Token B must remain active in state and other keys must not be cleared
        self.assertEqual(state_data.get("reset_token"), "token_B")
        self.assertEqual(state_data.get("other_flow_key"), "preserved_val")

    async def test_telegram_cancel_token_invalidates_future_confirm(self):
        """Canceling Token B consumes the token; a subsequent Confirm with Token B is rejected."""
        async with self.session_factory() as session:
            user = User(id=303, current_dialogue_id=1, current_topic_id=None)
            session.add(user)
            await session.commit()

        state_data = {
            "reset_token": "token_B",
            "reset_dialogue_id": 1,
            "reset_topic_id": None,
        }

        async def fake_get_data():
            return dict(state_data)

        async def fake_update_data(**kwargs):
            state_data.update(kwargs)

        state = MagicMock()
        state.get_data = AsyncMock(side_effect=fake_get_data)
        state.update_data = AsyncMock(side_effect=fake_update_data)

        cb_cancel = MagicMock()
        cb_cancel.from_user = SimpleNamespace(id=303)
        cb_cancel.data = "delete_history_cancel:token_B"
        cb_cancel.answer = AsyncMock()
        cb_cancel.message = MagicMock()
        cb_cancel.message.delete = AsyncMock()
        cb_cancel.message.answer = AsyncMock()

        await handlers.cancel_delete_history(cb_cancel, state)
        self.assertIsNone(state_data.get("reset_token"))

        # Subsequent confirm with token_B
        bot = MagicMock()
        bot.send_message = AsyncMock()
        cb_confirm = MagicMock()
        cb_confirm.from_user = SimpleNamespace(id=303)
        cb_confirm.data = "delete_history_confirm:token_B"
        cb_confirm.answer = AsyncMock()
        cb_confirm.message = MagicMock()
        cb_confirm.message.delete = AsyncMock()

        await handlers.process_delete_history(cb_confirm, state, bot)
        cb_confirm.answer.assert_called_with("Подтверждение устарело или уже использовано.", show_alert=True)

        async with self.session_factory() as session:
            u = await session.get(User, 303)
            self.assertEqual(u.current_dialogue_id, 1)

    async def test_telegram_confirm_resets_dialogue_and_clears_reset_keys(self):
        """Valid confirm token advances dialogue ID and removes reset keys without state.clear() wiping everything."""
        async with self.session_factory() as session:
            user = User(id=304, current_dialogue_id=1, current_topic_id=None)
            session.add(user)
            await session.commit()

        state_data = {
            "reset_token": "valid_token",
            "reset_dialogue_id": 1,
            "reset_topic_id": None,
            "user_pref": "dark_mode",
        }

        async def fake_get_data():
            return dict(state_data)

        async def fake_update_data(**kwargs):
            state_data.update(kwargs)

        state = MagicMock()
        state.get_data = AsyncMock(side_effect=fake_get_data)
        state.update_data = AsyncMock(side_effect=fake_update_data)

        bot = MagicMock()
        bot.send_message = AsyncMock()
        cb_confirm = MagicMock()
        cb_confirm.from_user = SimpleNamespace(id=304)
        cb_confirm.data = "delete_history_confirm:valid_token"
        cb_confirm.answer = AsyncMock()
        cb_confirm.message = MagicMock()
        cb_confirm.message.delete = AsyncMock()

        await handlers.process_delete_history(cb_confirm, state, bot)

        self.assertIsNone(state_data.get("reset_token"))
        self.assertEqual(state_data.get("user_pref"), "dark_mode")

        async with self.session_factory() as session:
            u = await session.get(User, 304)
            self.assertEqual(u.current_dialogue_id, 2)

    async def test_max_cancel_invalidates_future_confirm(self):
        """MAX: Cancel clears confirm_reset state; subsequent confirm callback no-ops with zero DB mutation."""
        async with self.session_factory() as session:
            user = User(id=401, current_dialogue_id=1, current_topic_id=None)
            session.add(user)
            await session.commit()

        client = MagicMock()
        client.send_message = AsyncMock()
        client.answer_callback = AsyncMock()

        app = max_app.MaxBotApplication(client)

        # 1. Request reset
        await max_common.request_reset_dialogue(client, app.states, chat_id=401, user_id=401)
        snap = await app.states.get(401)
        self.assertEqual(snap.state, "confirm_reset")
        token = snap.data.get("reset_token")

        # 2. Cancel reset
        cb_cancel = IncomingCallback(
            raw={},
            callback_id="cb_c1",
            payload=f"cancel_reset_dialogue:{token}",
            chat_id=401,
            message_id="m1",
            sender=Sender(user_id=401, username="u401", first_name="User 401", last_name=None),
        )
        await app.handle_callback(cb_cancel)
        task_c = app.user_tasks.get(401)
        if task_c:
            await task_c

        snap_after = await app.states.get(401)
        self.assertIsNone(snap_after)

        # 3. Old confirm callback arrives
        cb_confirm = IncomingCallback(
            raw={},
            callback_id="cb_conf1",
            payload=f"confirm_reset_dialogue:{token}:1:0",
            chat_id=401,
            message_id="m1",
            sender=Sender(user_id=401, username="u401", first_name="User 401", last_name=None),
        )
        await app.handle_callback(cb_confirm)
        task_conf = app.user_tasks.get(401)
        if task_conf:
            await task_conf

        async with self.session_factory() as session:
            u = await session.get(User, 401)
            self.assertEqual(u.current_dialogue_id, 1)

    async def test_max_concurrent_double_confirm_resets_once(self):
        """MAX: Concurrent double confirm executes exactly one dialogue reset."""
        async with self.session_factory() as session:
            user = User(id=402, current_dialogue_id=1, current_topic_id=None)
            session.add(user)
            await session.commit()

        client = MagicMock()
        client.send_message = AsyncMock()
        client.answer_callback = AsyncMock()

        app = max_app.MaxBotApplication(client)

        await max_common.request_reset_dialogue(client, app.states, chat_id=402, user_id=402)
        snap = await app.states.get(402)
        token = snap.data.get("reset_token")

        cb1 = IncomingCallback(
            raw={},
            callback_id="cb_c1",
            payload=f"confirm_reset_dialogue:{token}:1:0",
            chat_id=402,
            message_id="m1",
            sender=Sender(user_id=402, username="u402", first_name="User 402", last_name=None),
        )
        cb2 = IncomingCallback(
            raw={},
            callback_id="cb_c2",
            payload=f"confirm_reset_dialogue:{token}:1:0",
            chat_id=402,
            message_id="m1",
            sender=Sender(user_id=402, username="u402", first_name="User 402", last_name=None),
        )

        await app.handle_callback(cb1)
        await app.handle_callback(cb2)

        task = app.user_tasks.get(402)
        if task:
            await task

        async with self.session_factory() as session:
            u = await session.get(User, 402)
            # Must advance exactly once from 1 -> 2
            self.assertEqual(u.current_dialogue_id, 2)

    async def test_max_old_confirmation_token_rejected_after_new_confirmation(self):
        """MAX: When a new confirmation R2 is requested, pressing old Confirm R1 is rejected."""
        async with self.session_factory() as session:
            user = User(id=403, current_dialogue_id=1, current_topic_id=None)
            session.add(user)
            await session.commit()

        client = MagicMock()
        client.send_message = AsyncMock()
        client.answer_callback = AsyncMock()

        app = max_app.MaxBotApplication(client)

        # First reset request (R1)
        await max_common.request_reset_dialogue(client, app.states, chat_id=403, user_id=403)
        snap1 = await app.states.get(403)
        token1 = snap1.data.get("reset_token")

        # Second reset request (R2)
        await max_common.request_reset_dialogue(client, app.states, chat_id=403, user_id=403)
        snap2 = await app.states.get(403)
        token2 = snap2.data.get("reset_token")
        self.assertNotEqual(token1, token2)

        # Press Confirm R1 (old)
        cb_r1 = IncomingCallback(
            raw={},
            callback_id="cb_r1",
            payload=f"confirm_reset_dialogue:{token1}:1:0",
            chat_id=403,
            message_id="m1",
            sender=Sender(user_id=403, username="u403", first_name="User 403", last_name=None),
        )
        await app.handle_callback(cb_r1)
        task1 = app.user_tasks.get(403)
        if task1:
            await task1

        async with self.session_factory() as session:
            u = await session.get(User, 403)
            self.assertEqual(u.current_dialogue_id, 1)

        # Token 2 remains active
        snap_current = await app.states.get(403)
        self.assertEqual(snap_current.data.get("reset_token"), token2)

    async def test_max_topic_changed_with_same_dialogue_id_rejects_confirm(self):
        """MAX: If expected_topic_id does not match user's current topic in DB, confirm is rejected."""
        async with self.session_factory() as session:
            user = User(id=404, current_dialogue_id=1, current_topic_id=5)
            topic = Topic(id=5, name="Карьера", is_active=True)
            session.add_all([user, topic])
            await session.commit()

        client = MagicMock()
        client.send_message = AsyncMock()
        client.answer_callback = AsyncMock()

        app = max_app.MaxBotApplication(client)

        await max_common.request_reset_dialogue(client, app.states, chat_id=404, user_id=404)
        snap = await app.states.get(404)
        token = snap.data.get("reset_token")

        # Switch user's topic to 6 behind the scenes
        async with self.session_factory() as session:
            u = await session.get(User, 404)
            u.current_topic_id = 6
            await session.commit()

        cb_confirm = IncomingCallback(
            raw={},
            callback_id="cb_conf",
            payload=f"confirm_reset_dialogue:{token}:1:5",
            chat_id=404,
            message_id="m1",
            sender=Sender(user_id=404, username="u404", first_name="User 404", last_name=None),
        )
        await app.handle_callback(cb_confirm)
        task = app.user_tasks.get(404)
        if task:
            await task

        async with self.session_factory() as session:
            u_check = await session.get(User, 404)
            self.assertEqual(u_check.current_dialogue_id, 1)

        client.send_message.assert_called_with(
            chat_id=404,
            text="Состояние диалога изменилось. Действие отменено.",
        )

    async def test_reset_confirm_and_cancel_next_user_message_dialogue_consistency(self):
        """Subsequent normal user messages correctly log to dialogue_id after reset confirm vs cancel."""
        async with self.session_factory() as session:
            user = User(id=405, name="Leo", current_dialogue_id=1, current_topic_id=None, accepted_disclaimer=True)
            session.add(user)
            await session.commit()

        client = MagicMock()
        client.send_message = AsyncMock(return_value={"message": {"mid": "m1"}})
        client.edit_message = AsyncMock()
        client.answer_callback = AsyncMock()

        app = max_app.MaxBotApplication(client)

        # 1. Reset dialogue
        await max_common.request_reset_dialogue(client, app.states, chat_id=405, user_id=405)
        snap = await app.states.get(405)
        token = snap.data.get("reset_token")

        cb_confirm = IncomingCallback(
            raw={},
            callback_id="cb_c",
            payload=f"confirm_reset_dialogue:{token}:1:0",
            chat_id=405,
            message_id="m1",
            sender=Sender(user_id=405, username="leo", first_name="Leo", last_name=None),
        )
        await app.handle_callback(cb_confirm)
        task = app.user_tasks.get(405)
        if task:
            await task

        async with self.session_factory() as session:
            u = await session.get(User, 405)
            self.assertEqual(u.current_dialogue_id, 2)

        # 2. Next normal message should be recorded under dialogue_id=2
        msg = IncomingMessage(
            raw={},
            message_id="m_next",
            chat_id=405,
            sender=Sender(user_id=405, username="leo", first_name="Leo", last_name=None),
            text="Новый вопрос в новом диалоге",
        )
        with patch("max_messenger_bot.services.common.get_ai_response", AsyncMock(return_value="Ответ")):
            await app.handle_message(msg)
            task_msg = app.user_tasks.get(405)
            if task_msg:
                await task_msg

        async with self.session_factory() as session:
            msgs = (await session.execute(select(DBMessage).where(DBMessage.user_id == 405).order_by(DBMessage.id.asc()))).scalars().all()
            self.assertGreaterEqual(len(msgs), 2)
            self.assertEqual(msgs[0].dialogue_id, 2)
            self.assertEqual(msgs[0].content, "Новый вопрос в новом диалоге")
            self.assertEqual(msgs[1].dialogue_id, 2)
