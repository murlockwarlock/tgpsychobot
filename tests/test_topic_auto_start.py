import asyncio
import os
import unittest
from datetime import datetime, timedelta
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
    from system_events import build_topic_auto_start_system_message, sanitize_synthetic_text_fragment
    import ai_integration
    import database
    from database import (
        AIConfig,
        Base,
        BotGeneralConfig,
        Content,
        Message as DBMessage,
        SubscriptionConfig,
        Topic,
        User,
        UserSubscription,
        async_session_maker,
        init_db,
    )
    import handlers
    import keyboards as tg_kb
    from max_messenger_bot import ai as max_ai
    import max_messenger_bot.legacy as max_legacy
    import max_messenger_bot.storage as max_storage
    from max_messenger_bot import app as max_app
    from max_messenger_bot.api import MaxApiClient
    from max_messenger_bot.services import (
        admin_topics as max_admin_topics,
        common as max_common,
        settings as max_settings,
        topics as max_topics,
    )
    from max_messenger_bot.storage import StateStore, StorageBase


class TopicAutoStartUnitAndIntegrationTests(unittest.IsolatedAsyncioTestCase):
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
            patch.object(ai_integration, "async_session_maker", self.session_factory),
            patch.object(max_ai, "async_session_maker", self.session_factory),
            patch.object(max_legacy, "async_session_maker", self.session_factory),
            patch.object(max_storage, "async_session_maker", self.session_factory),
            patch.object(max_common, "async_session_maker", self.session_factory),
            patch.object(max_topics, "async_session_maker", self.session_factory),
            patch.object(max_settings, "async_session_maker", self.session_factory),
            patch.object(max_app, "async_session_maker", self.session_factory),
            patch.object(max_admin_topics, "async_session_maker", self.session_factory),
        ]
        for p in self.patches:
            p.start()

        # Seed configs
        async with self.session_factory() as session:
            session.add(AIConfig(id=1, memory_mode="reset"))
            session.add(SubscriptionConfig(id=1, subscriptions_enabled=False))
            session.add(BotGeneralConfig(id=1, profile_collect_name=True, profile_collect_gender=True, profile_collect_age=False))
            await session.commit()

        handlers.user_message_buffers.clear()
        handlers.user_isolated_turn_queues.clear()
        handlers.user_processing_tasks.clear()
        handlers.user_scheduling_locks.clear()

    async def asyncTearDown(self):
        for p in reversed(self.patches):
            p.stop()
        await self.engine.dispose()

    def test_sanitize_and_build_synthetic_message(self):
        dirty_topic = 'Тревожность "и" [стресс] (бытовой) \\ тест\n\r  новая   строка  '
        clean = sanitize_synthetic_text_fragment(dirty_topic)
        self.assertIn('\\"', clean)
        self.assertIn('\\[', clean)
        self.assertIn('\\]', clean)
        self.assertIn('\\(', clean)
        self.assertIn('\\)', clean)
        self.assertIn('\\\\', clean)
        self.assertNotIn('\n', clean)
        self.assertNotIn('\r', clean)
        self.assertNotIn('\t', clean)

        msg = build_topic_auto_start_system_message(dirty_topic)
        self.assertIn("[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь выбрал тему", msg)
        self.assertIn("Тревожность", msg)

    async def test_telegram_topic_selection_auto_start_turn_isolation(self):
        """Verify that selecting an auto-start topic creates Turn 1 as synthetic prompt without visible echo, and subsequent user messages run as Turn 2 with debounce."""
        async with self.session_factory() as session:
            user = User(id=101, first_name="Alice", name="Alice", gender="female", age="25", accepted_disclaimer=True, current_dialogue_id=1)
            topic = Topic(id=1, name="Карьера", is_active=True, auto_start_dialogue=True)
            session.add_all([user, topic])
            await session.commit()

        bot = MagicMock()
        bot.send_message = AsyncMock()
        state = MagicMock()
        state.get_data = AsyncMock(return_value={})
        state.set_state = AsyncMock()
        state.clear = AsyncMock()

        callback = MagicMock()
        callback.from_user = SimpleNamespace(id=101, username="alice", full_name="Alice")
        callback.data = "select_topic_1"
        callback.answer = AsyncMock()
        callback.message = MagicMock()
        callback.message.chat = SimpleNamespace(id=101)
        callback.message.delete = AsyncMock()
        callback.message.answer = AsyncMock()

        ai_responses = []

        async def fake_generate_response(user_id, prompt, *args, **kwargs):
            ai_responses.append((prompt, kwargs.get("visible_user_text")))
            return "Здравствуйте! Давайте обсудим карьеру."

        with patch("handlers.ai_integration.generate_response", side_effect=fake_generate_response):
            await handlers.process_topic_selection(callback, state, bot)
            # Wait for drain runner
            task = handlers.user_processing_tasks.get(101)
            if task:
                await task

        self.assertEqual(len(ai_responses), 1)
        synthetic_prompt, visible_text = ai_responses[0]
        self.assertIn("[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь выбрал тему", synthetic_prompt)
        self.assertIn("Карьера", synthetic_prompt)
        self.assertIsNone(visible_text)

        # Verify DB state: topic is set and assistant response is saved
        async with self.session_factory() as session:
            db_user = await session.get(User, 101)
            self.assertEqual(db_user.current_topic_id, 1)
            msgs = (await session.execute(select(DBMessage).where(DBMessage.user_id == 101))).scalars().all()
            self.assertEqual(len(msgs), 2)  # 1 synthetic user message + 1 assistant message
            self.assertEqual(msgs[0].role, "user")
            self.assertIn("СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь выбрал тему", msgs[0].content)
            self.assertEqual(msgs[1].role, "assistant")
            self.assertIn("Здравствуйте! Давайте обсудим карьеру.", msgs[1].content)

    async def test_telegram_real_user_message_batches_separately_after_auto_start(self):
        """Verify Turn 1 (synthetic auto start) and Turn 2 (real user text) execute sequentially without merging."""
        async with self.session_factory() as session:
            user = User(id=102, first_name="Bob", name="Bob", gender="male", age="30", accepted_disclaimer=True, current_dialogue_id=1)
            topic = Topic(id=2, name="Отношения", is_active=True, auto_start_dialogue=True)
            session.add_all([user, topic])
            await session.commit()

        bot = MagicMock()
        bot.send_message = AsyncMock()
        state = MagicMock()
        state.get_data = AsyncMock(return_value={})
        state.set_state = AsyncMock()
        state.clear = AsyncMock()

        turns_executed = []

        async def fake_generate_response(user_id, prompt, *args, **kwargs):
            turns_executed.append(prompt)
            return "Ответ ИИ"

        with patch("handlers.ai_integration.generate_response", side_effect=fake_generate_response):
            # 1. Trigger topic auto start
            await handlers._start_telegram_topic_auto_start(102, bot, state, topic)

            # 2. Immediately enqueue real user text
            handlers.user_message_buffers.setdefault(102, []).append("Привет, хочу совет.")
            handlers._ensure_telegram_drain_runner(102, bot, state, initial_delay=0.0)

            # Wait for processing task to complete
            while handlers._has_user_turn_work(102) or (102 in handlers.user_processing_tasks and not handlers.user_processing_tasks[102].done()):
                task = handlers.user_processing_tasks.get(102)
                if task:
                    await task
                await asyncio.sleep(0.01)

        self.assertEqual(len(turns_executed), 2)
        self.assertIn("[СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь выбрал тему", turns_executed[0])
        self.assertEqual(turns_executed[1], "Привет, хочу совет.")

    async def test_telegram_disclaimer_gate_and_resume(self):
        """User without disclaimer acceptance gets disclaimer keyboard, and accepting resumes auto-start."""
        async with self.session_factory() as session:
            user = User(id=103, first_name="Charlie", name="Charlie", gender="male", age="22", accepted_disclaimer=False, current_dialogue_id=1)
            topic = Topic(id=3, name="Саморазвитие", is_active=True, auto_start_dialogue=True)
            session.add_all([user, topic])
            session.add(Content(key="disclaimer", text_content="Текст дисклеймера", is_visible=True))
            await session.commit()

        bot = MagicMock()
        bot.send_message = AsyncMock()

        state_data = {}

        async def fake_get_data():
            return state_data

        async def fake_update_data(**kwargs):
            state_data.update(kwargs)

        state = MagicMock()
        state.get_data = AsyncMock(side_effect=fake_get_data)
        state.update_data = AsyncMock(side_effect=fake_update_data)
        state.set_state = AsyncMock()
        state.clear = AsyncMock(side_effect=state_data.clear)

        callback = MagicMock()
        callback.from_user = SimpleNamespace(id=103, username="charlie", full_name="Charlie")
        callback.data = "select_topic_3"
        callback.answer = AsyncMock()
        callback.message = MagicMock()
        callback.message.chat = SimpleNamespace(id=103)
        callback.message.delete = AsyncMock()
        callback.message.answer = AsyncMock()

        # Topic selection triggers disclaimer prompt
        await handlers.process_topic_selection(callback, state, bot)
        self.assertEqual(state_data.get("pending_auto_start_topic_id"), 3)
        self.assertTrue(any("дисклеймер" in str(c).lower() for c in bot.send_message.call_args_list))

        # Accept disclaimer
        disc_callback = MagicMock()
        disc_callback.from_user = SimpleNamespace(id=103)
        disc_callback.data = "disclaimer_accepted"
        disc_callback.message = MagicMock()
        disc_callback.message.chat = SimpleNamespace(id=103)
        disc_callback.message.delete = AsyncMock()
        disc_callback.message.answer = AsyncMock()

        auto_started = []
        with patch("handlers._start_telegram_topic_auto_start", side_effect=lambda u, b, s, t: auto_started.append(t.id)):
            await handlers.disclaimer_accepted_handler(disc_callback, state, bot)

        self.assertEqual(auto_started, [3])

    async def test_telegram_admin_only_topic_security(self):
        """Non-admin user cannot access admin_only topic via deep link or callback."""
        async with self.session_factory() as session:
            user = User(id=104, first_name="Dan", is_admin=False, accepted_disclaimer=True)
            topic = Topic(id=4, name="Секретная тема", is_active=True, admin_only=True, auto_start_dialogue=True)
            session.add_all([user, topic])
            await session.commit()

        res = await handlers._perform_telegram_topic_switch(104, 4)
        self.assertEqual(res.status, "inaccessible")

        # Now test with admin user
        async with self.session_factory() as session:
            admin_user = User(id=105, first_name="Admin", is_admin=True, accepted_disclaimer=True)
            session.add(admin_user)
            await session.commit()

        res_admin = await handlers._perform_telegram_topic_switch(105, 4)
        self.assertEqual(res_admin.status, "switched")
        self.assertEqual(res_admin.topic.id, 4)

    async def test_max_topic_auto_start_journey(self):
        """In MAX, selecting an auto-start topic omits the manual start button and awaits run_ai_dialogue directly."""
        async with self.session_factory() as session:
            user = User(id=201, first_name="Eva", name="Eva", gender="female", age="28", accepted_disclaimer=True, current_dialogue_id=1)
            topic = Topic(id=10, name="Семья", is_active=True, auto_start_dialogue=True)
            session.add_all([user, topic])
            await session.commit()

        client = MagicMock()
        client.send_message = AsyncMock(return_value={"message": {"mid": "m1"}})
        client.edit_message = AsyncMock()
        states = StateStore()

        ai_called_prompts = []

        async def fake_get_ai_response(user_id, prompt, *args, **kwargs):
            ai_called_prompts.append(prompt)
            return "Привет! Готова поговорить о семье."

        with patch("max_messenger_bot.services.common.get_ai_response", side_effect=fake_get_ai_response):
            await max_topics.select_topic(client, chat_id=201, user_id=201, topic_id=10, states=states)

        self.assertEqual(len(ai_called_prompts), 1)
        self.assertIn("СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь выбрал тему", ai_called_prompts[0])
        self.assertIn("Семья", ai_called_prompts[0])

        # Verify buttons sent in intro message do not contain manual start button
        intro_call = client.send_message.call_args_list[0]
        attachments = intro_call.kwargs.get("attachments") or []
        buttons = attachments[0]["payload"]["buttons"]
        all_btn_payloads = [b["payload"] for row in buttons for b in row]
        self.assertNotIn("topic_start_dialogue", all_btn_payloads)

    async def test_max_same_topic_guard(self):
        """Clicking the currently active topic in MAX is a no-op."""
        async with self.session_factory() as session:
            user = User(id=202, first_name="Frank", name="Frank", current_topic_id=15, accepted_disclaimer=True)
            topic = Topic(id=15, name="Карьера", is_active=True, auto_start_dialogue=True)
            session.add_all([user, topic])
            await session.commit()

        client = MagicMock()
        client.send_message = AsyncMock()
        states = StateStore()

        await max_topics.select_topic(client, chat_id=202, user_id=202, topic_id=15, states=states)
        client.send_message.assert_not_called()

    async def test_max_onboarding_preserves_pending_auto_start_topic(self):
        """New MAX user selecting auto-start topic completes onboarding (name -> gender -> AI auto start)."""
        async with self.session_factory() as session:
            user = User(id=203, first_name="Grace", name=None, accepted_disclaimer=True)
            topic = Topic(id=20, name="Здоровье", is_active=True, auto_start_dialogue=True)
            session.add_all([user, topic])
            await session.commit()

        client = MagicMock()
        client.bot_name = "testbot"
        client.send_message = AsyncMock(return_value={"message": {"mid": "m1"}})
        client.edit_message = AsyncMock()
        states = StateStore()

        # Step 1: select topic triggers onboarding
        await max_topics.select_topic(client, chat_id=203, user_id=203, topic_id=20, states=states)
        snap1 = await states.get(203)
        self.assertEqual(snap1.state, "awaiting_name")
        self.assertEqual(snap1.data.get("pending_auto_start_topic_id"), 20)

        # Step 2: save name and transition to gender
        await max_settings.save_name_only(states, 203, "Grace")
        await max_settings.start_change_gender(
            client, states, 203, 203, is_settings=False,
            resume_data={k: v for k, v in snap1.data.items() if k != "initial_prompt"}
        )
        snap2 = await states.get(203)
        self.assertEqual(snap2.state, "awaiting_gender")
        self.assertEqual(snap2.data.get("pending_auto_start_topic_id"), 20)

        # Step 3: save gender and resume pending turn
        gender_data = await max_settings.save_gender(client, states, 203, 203, "female")
        self.assertEqual(gender_data.get("pending_auto_start_topic_id"), 20)

        ai_called = []
        with patch("max_messenger_bot.services.common.run_ai_dialogue", side_effect=lambda c, ch, u, p, **kw: ai_called.append(p)):
            await max_common.resume_pending_ai_turn(client, 203, 203, gender_data, states=states)

        self.assertEqual(len(ai_called), 1)
        self.assertIn("СИСТЕМНОЕ СООБЩЕНИЕ: Пользователь выбрал тему", ai_called[0])
        self.assertIn("Здоровье", ai_called[0])

    async def test_admin_toggle_auto_start_tg_and_max(self):
        """Admin can toggle auto_start_dialogue in TG and MAX."""
        async with self.session_factory() as session:
            topic = Topic(id=50, name="Учеба", is_active=True, auto_start_dialogue=False)
            session.add(topic)
            await session.commit()

        client = MagicMock()
        client.bot_name = "testbot"
        client.send_message = AsyncMock()

        # MAX toggle
        await max_admin_topics.toggle_auto_start(client, chat_id=999, topic_id=50)
        async with self.session_factory() as session:
            t = await session.get(Topic, 50)
            self.assertTrue(t.auto_start_dialogue)

        await max_admin_topics.toggle_auto_start(client, chat_id=999, topic_id=50)
        async with self.session_factory() as session:
            t = await session.get(Topic, 50)
            self.assertFalse(t.auto_start_dialogue)
