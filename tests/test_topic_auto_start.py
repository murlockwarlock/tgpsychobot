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
        KnowledgeBase,
        Message as DBMessage,
        SubscriptionConfig,
        Topic,
        User,
        UserSubscription,
        async_session_maker,
    )
    import handlers
    import keyboards as tg_kb
    from max_messenger_bot import ai as max_ai
    import max_messenger_bot.legacy as max_legacy
    import max_messenger_bot.storage as max_storage
    from max_messenger_bot import app as max_app
    from max_messenger_bot.api import MaxApiClient
    from max_messenger_bot.models import IncomingCallback, IncomingMessage, Sender
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
            session.add(AIConfig(id=1, memory_mode="reset", provider="gemini", gemini_model="gemini-2.5-flash", gemini_api_key="test-key"))
            session.add(SubscriptionConfig(id=1, subscriptions_enabled=False))
            session.add(BotGeneralConfig(id=1, profile_collect_name=True, profile_collect_gender=True, profile_collect_age=False))
            await session.commit()

        handlers.user_message_buffers.clear()
        handlers.user_isolated_turn_queues.clear()
        handlers.user_processing_tasks.clear()
        handlers.user_scheduling_locks.clear()
        handlers.user_locks.clear()

    async def asyncTearDown(self):
        for p in reversed(self.patches):
            p.stop()
        await self.engine.dispose()

    def _make_mock_bot(self):
        bot = MagicMock()
        bot.send_message = AsyncMock()
        bot.send_chat_action = AsyncMock()
        bot.delete_message = AsyncMock()
        return bot

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

    async def test_telegram_drain_race_window_a_during_provider_call(self):
        """Window A: User sends real message while provider call is actively in-flight -> processed sequentially."""
        async with self.session_factory() as session:
            user = User(id=110, first_name="Alice", name="Alice", gender="female", age="25", accepted_disclaimer=True, current_dialogue_id=1)
            topic = Topic(id=1, name="Карьера", is_active=True, auto_start_dialogue=True)
            session.add_all([user, topic])
            await session.commit()

        bot = self._make_mock_bot()
        state = MagicMock()
        state.get_data = AsyncMock(return_value={})
        state.set_state = AsyncMock()
        state.clear = AsyncMock()

        ai_prompts = []
        turn1_started = asyncio.Event()

        async def fake_generate_response(user_id, prompt, *args, **kwargs):
            ai_prompts.append(prompt)
            if len(ai_prompts) == 1:
                turn1_started.set()
                await asyncio.sleep(0.05)
            return "Ответ ИИ"

        with patch("handlers.ai_integration.generate_response", side_effect=fake_generate_response):
            await handlers._start_telegram_topic_auto_start(110, bot, state, topic)
            await turn1_started.wait()

            # Real user message arrives while Turn 1 is executing
            msg = MagicMock()
            msg.from_user = SimpleNamespace(id=110, username="alice", full_name="Alice")
            msg.chat = SimpleNamespace(id=110)
            msg.text = "Мой вопрос о карьере"
            await handlers.handle_ai_chat(msg, state, bot)

            # Wait for all tasks
            while handlers._has_user_turn_work(110) or (110 in handlers.user_processing_tasks and not handlers.user_processing_tasks[110].done()):
                task = handlers.user_processing_tasks.get(110)
                if task:
                    await task
                await asyncio.sleep(0.01)

        self.assertEqual(len(ai_prompts), 2)
        self.assertIn("СИСТЕМНОЕ СООБЩЕНИЕ", ai_prompts[0])
        self.assertEqual(ai_prompts[1], "Мой вопрос о карьере")

    async def test_telegram_drain_race_window_b_between_empty_and_unregister_event_barrier(self):
        """Window B: Message arrives after queue empty check but before runner unregisters -> atomic handoff schedules replacement."""
        async with self.session_factory() as session:
            user = User(id=111, first_name="Bob", name="Bob", gender="male", age="30", accepted_disclaimer=True, current_dialogue_id=1)
            topic = Topic(id=2, name="Отношения", is_active=True, auto_start_dialogue=True)
            session.add_all([user, topic])
            await session.commit()

        bot = self._make_mock_bot()
        state = MagicMock()
        state.get_data = AsyncMock(return_value={})
        state.set_state = AsyncMock()
        state.clear = AsyncMock()

        ai_prompts = []
        runner_empty_pause = asyncio.Event()
        runner_proceed = asyncio.Event()

        async def fake_generate_response(user_id, prompt, *args, **kwargs):
            ai_prompts.append(prompt)
            return "Ответ ИИ"

        async def cleanup_hook(uid):
            if uid == 111:
                runner_empty_pause.set()
                await runner_proceed.wait()

        with patch("handlers.ai_integration.generate_response", side_effect=fake_generate_response), \
             patch("handlers._drain_runner_before_cleanup_hook", side_effect=cleanup_hook):
            await handlers._start_telegram_topic_auto_start(111, bot, state, topic)
            initial_runner = handlers.user_processing_tasks[111]

            # Wait until runner finishes Turn 1, determines no work, and pauses before cleanup
            await runner_empty_pause.wait()
            self.assertIs(handlers.user_processing_tasks.get(111), initial_runner)
            self.assertFalse(initial_runner.done())

            # Enqueue user message exactly during this pause
            msg = MagicMock()
            msg.from_user = SimpleNamespace(id=111, username="bob", full_name="Bob")
            msg.chat = SimpleNamespace(id=111)
            msg.text = "Вопрос в окне B"
            await handlers.handle_ai_chat(msg, state, bot)

            # Allow initial runner to proceed with cleanup and handoff
            runner_proceed.set()
            await initial_runner

            while handlers._has_user_turn_work(111) or (111 in handlers.user_processing_tasks and not handlers.user_processing_tasks[111].done()):
                task = handlers.user_processing_tasks.get(111)
                if task:
                    await task
                await asyncio.sleep(0.01)

        self.assertEqual(len(ai_prompts), 2)
        self.assertIn("СИСТЕМНОЕ СООБЩЕНИЕ", ai_prompts[0])
        self.assertEqual(ai_prompts[1], "Вопрос в окне B")

    async def test_telegram_drain_race_window_c_immediately_after_unregister(self):
        """Window C: Message arrives immediately after runner task has completed and popped -> schedules fresh runner."""
        async with self.session_factory() as session:
            user = User(id=112, first_name="Charlie", name="Charlie", gender="male", age="22", accepted_disclaimer=True, current_dialogue_id=1)
            topic = Topic(id=3, name="Саморазвитие", is_active=True, auto_start_dialogue=True)
            session.add_all([user, topic])
            await session.commit()

        bot = self._make_mock_bot()
        state = MagicMock()
        state.get_data = AsyncMock(return_value={})
        state.set_state = AsyncMock()
        state.clear = AsyncMock()

        ai_prompts = []

        async def fake_generate_response(user_id, prompt, *args, **kwargs):
            ai_prompts.append(prompt)
            return "Ответ ИИ"

        with patch("handlers.ai_integration.generate_response", side_effect=fake_generate_response):
            await handlers._start_telegram_topic_auto_start(112, bot, state, topic)

            # Wait for runner to complete and unregister
            task = handlers.user_processing_tasks.get(112)
            if task:
                await task
            self.assertNotIn(112, handlers.user_processing_tasks)

            # Now send message immediately after unregister
            msg = MagicMock()
            msg.from_user = SimpleNamespace(id=112, username="charlie", full_name="Charlie")
            msg.chat = SimpleNamespace(id=112)
            msg.text = "Вопрос после завершения раннера"
            await handlers.handle_ai_chat(msg, state, bot)

            while handlers._has_user_turn_work(112) or (112 in handlers.user_processing_tasks and not handlers.user_processing_tasks[112].done()):
                task = handlers.user_processing_tasks.get(112)
                if task:
                    await task
                await asyncio.sleep(0.01)

        self.assertEqual(len(ai_prompts), 2)
        self.assertEqual(ai_prompts[1], "Вопрос после завершения раннера")

    async def test_telegram_stale_pending_topic_after_topic_switch(self):
        """TG: User selects Topic B -> disclaimer pending -> user switches to Topic C -> accept disclaimer does NOT start Topic B."""
        async with self.session_factory() as session:
            user = User(id=113, first_name="Dan", name="Dan", gender="male", age="29", accepted_disclaimer=False, current_dialogue_id=1, current_topic_id=1)
            topic_b = Topic(id=1, name="Topic B", is_active=True, auto_start_dialogue=True)
            topic_c = Topic(id=2, name="Topic C", is_active=True, auto_start_dialogue=False)
            session.add_all([user, topic_b, topic_c])
            session.add(Content(key="disclaimer", text_content="Disclaimer text", is_visible=True))
            await session.commit()

        bot = self._make_mock_bot()

        state_data = {"pending_auto_start_topic_id": 1}

        async def fake_get_data():
            return dict(state_data)

        async def fake_update_data(**kwargs):
            state_data.update(kwargs)

        state = MagicMock()
        state.get_data = AsyncMock(side_effect=fake_get_data)
        state.update_data = AsyncMock(side_effect=fake_update_data)
        state.clear = AsyncMock(side_effect=state_data.clear)

        # In DB, user current topic is switched to Topic C (id=2)
        async with self.session_factory() as session:
            user_db = await session.get(User, 113)
            user_db.current_topic_id = 2
            await session.commit()

        disc_callback = MagicMock()
        disc_callback.from_user = SimpleNamespace(id=113)
        disc_callback.data = "disclaimer_accepted"
        disc_callback.message = MagicMock()
        disc_callback.message.chat = SimpleNamespace(id=113)
        disc_callback.message.delete = AsyncMock()
        disc_callback.message.answer = AsyncMock()

        auto_started = []
        with patch("handlers._start_telegram_topic_auto_start", side_effect=lambda u, b, s, t: auto_started.append(t.id)):
            await handlers.disclaimer_accepted_handler(disc_callback, state, bot)

        # Topic B auto-start must NOT be triggered
        self.assertEqual(len(auto_started), 0)
        async with self.session_factory() as session:
            msgs = (await session.execute(select(DBMessage).where(DBMessage.user_id == 113))).scalars().all()
            self.assertEqual(len(msgs), 0)

    async def test_max_stale_pending_topic_after_topic_switch(self):
        """MAX: User selects Topic B -> onboarding pending -> user topic becomes C -> resume callback skips B auto-start."""
        async with self.session_factory() as session:
            user = User(id=210, first_name="Eve", name="Eve", accepted_disclaimer=True, current_dialogue_id=1, current_topic_id=2)
            topic_b = Topic(id=1, name="Topic B", is_active=True, auto_start_dialogue=True)
            topic_c = Topic(id=2, name="Topic C", is_active=True, auto_start_dialogue=False)
            session.add_all([user, topic_b, topic_c])
            await session.commit()

        client = MagicMock()
        client.send_message = AsyncMock()
        states = StateStore()

        ai_called = []
        with patch("max_messenger_bot.services.common.run_ai_dialogue", side_effect=lambda c, ch, u, p, **kw: ai_called.append(p)):
            await max_common.resume_pending_ai_turn(client, 210, 210, {"pending_auto_start_topic_id": 1}, states=states)

        self.assertEqual(len(ai_called), 0)
        async with self.session_factory() as session:
            msgs = (await session.execute(select(DBMessage).where(DBMessage.user_id == 210))).scalars().all()
            self.assertEqual(len(msgs), 0)

    async def test_max_real_app_onboarding_name_gender_auto_start(self):
        """Real MaxBotApplication flow: select auto-start topic -> awaiting_name -> handle_callback gender -> AI auto-start runs once."""
        async with self.session_factory() as session:
            user = User(id=211, first_name="Grace", name=None, gender=None, accepted_disclaimer=True, current_dialogue_id=1)
            topic = Topic(id=10, name="Тревожность", is_active=True, auto_start_dialogue=True)
            session.add_all([user, topic])
            await session.commit()

        client = MagicMock()
        client.send_message = AsyncMock(return_value={"message": {"mid": "m1"}})
        client.edit_message = AsyncMock()
        client.answer_callback = AsyncMock()

        app = max_app.MaxBotApplication(client)

        # Step 1: User clicks topic selection callback
        cb_topic = IncomingCallback(
            raw={},
            callback_id="cb_t1",
            payload="select_topic_10",
            chat_id=211,
            message_id="m1",
            sender=Sender(user_id=211, username="grace", first_name="Grace", last_name=None),
        )
        await app.handle_callback(cb_topic)
        task = app.user_tasks.get(211)
        if task:
            await task
        snap1 = await app.states.get(211)
        self.assertEqual(snap1.state, "awaiting_name")
        self.assertEqual(snap1.data.get("pending_auto_start_topic_id"), 10)
        self.assertTrue(snap1.data.get("is_onboarding"))

        # Step 2: User sends name message
        msg_name = IncomingMessage(
            raw={},
            message_id="m_name",
            chat_id=211,
            sender=Sender(user_id=211, username="grace", first_name="Grace", last_name=None),
            text="Грейс",
        )
        await app.handle_message(msg_name)
        snap2 = await app.states.get(211)
        self.assertEqual(snap2.state, "awaiting_gender")
        self.assertEqual(snap2.data.get("pending_auto_start_topic_id"), 10)
        self.assertTrue(snap2.data.get("is_onboarding"))

        ai_called = []

        async def fake_get_ai(u, p, **kw):
            ai_called.append(p)
            return "Здравствуйте, Грейс! Готов обсудить тревожность."

        # Step 3: User clicks gender callback
        cb_gender = IncomingCallback(
            raw={},
            callback_id="cb_g1",
            payload="gender_female",
            chat_id=211,
            message_id="m2",
            sender=Sender(user_id=211, username="grace", first_name="Grace", last_name=None),
        )
        with patch("max_messenger_bot.services.common.get_ai_response", side_effect=fake_get_ai):
            await app.handle_callback(cb_gender)
            client.answer_callback.assert_called_with("cb_g1", notification="Пол сохранён")

            # Wait for background task spawned by app
            task = app.user_tasks.get(211)
            if task:
                await task

        self.assertEqual(len(ai_called), 1)
        self.assertIn("СИСТЕМНОЕ СООБЩЕНИЕ", ai_called[0])
        self.assertIn("Тревожность", ai_called[0])

    async def test_max_topic_deeplink_serialized_under_user_lock(self):
        """MAX: Topic deep-link /start topic_<id> serializes behind existing user task under spawn_user_task."""
        async with self.session_factory() as session:
            user = User(id=212, first_name="Henry", name="Henry", gender="male", accepted_disclaimer=True, current_dialogue_id=1)
            topic = Topic(id=12, name="Финансы", is_active=True, auto_start_dialogue=True)
            session.add_all([user, topic])
            await session.commit()

        client = MagicMock()
        client.send_message = AsyncMock(return_value={"message": {"mid": "m1"}})
        client.edit_message = AsyncMock()

        app = max_app.MaxBotApplication(client)

        execution_order = []
        task1_started = asyncio.Event()
        task1_proceed = asyncio.Event()

        async def slow_ai_task():
            execution_order.append("task1_start")
            task1_started.set()
            await task1_proceed.wait()
            execution_order.append("task1_end")

        app.spawn_user_task(212, slow_ai_task())
        await task1_started.wait()

        # Send /start topic_12
        msg_start = IncomingMessage(
            raw={},
            message_id="m_start",
            chat_id=212,
            sender=Sender(user_id=212, username="henry", first_name="Henry", last_name=None),
            text="/start topic_12",
        )

        with patch("max_messenger_bot.services.common.get_ai_response", AsyncMock(return_value="Ответ")):
            await app.handle_message(msg_start)
            execution_order.append("deeplink_enqueued")
            task1_proceed.set()

            task = app.user_tasks.get(212)
            if task:
                await task

        self.assertEqual(execution_order, ["task1_start", "deeplink_enqueued", "task1_end"])
        async with self.session_factory() as session:
            u = await session.get(User, 212)
            self.assertEqual(u.current_topic_id, 12)

    async def test_max_topic_start_dialogue_ack_once(self):
        """MAX: topic_start_dialogue answers callback exactly once FIRST before sending prompt without AI mutation."""
        async with self.session_factory() as session:
            user = User(id=213, first_name="Ian", name="Ian", accepted_disclaimer=True, current_dialogue_id=1)
            topic = Topic(id=14, name="Спорт", is_active=True, auto_start_dialogue=False)
            session.add_all([user, topic])
            await session.commit()

        client = MagicMock()
        client.send_message = AsyncMock(return_value={"message": {"mid": "m1"}})
        client.answer_callback = AsyncMock()

        app = max_app.MaxBotApplication(client)

        cb = IncomingCallback(
            raw={},
            callback_id="cb_start_d",
            payload="topic_start_dialogue",
            chat_id=213,
            message_id="m1",
            sender=Sender(user_id=213, username="ian", first_name="Ian", last_name=None),
        )
        await app.handle_callback(cb)

        client.answer_callback.assert_called_once_with("cb_start_d")
        client.send_message.assert_called_once_with(
            chat_id=213,
            text="✍️ Напишите ваш первый вопрос, и я отвечу.",
        )
        async with self.session_factory() as session:
            msgs = (await session.execute(select(DBMessage).where(DBMessage.user_id == 213))).scalars().all()
            self.assertEqual(len(msgs), 0)

    async def test_telegram_access_helper_preloaded_user_eager_load(self):
        """Telegram access helper loads User.subscription without MissingGreenlet when User was pre-loaded into the session."""
        async with self.session_factory() as session:
            user = User(id=114, first_name="Jack", is_admin=False)
            sub = UserSubscription(user_id=114, start_date=datetime.utcnow() - timedelta(days=1), end_date=datetime.utcnow() + timedelta(days=10), auto_renewal=False)
            session.add_all([user, sub])
            await session.commit()

        async with self.session_factory() as session:
            # Preload user into identity map WITHOUT subscription loaded
            preloaded_user = await session.get(User, 114)
            self.assertIsNotNone(preloaded_user)

            bot = self._make_mock_bot()

            # Call helper with the same session
            has_access = await handlers._check_telegram_chat_access(session, 114, bot, 114)
            self.assertTrue(has_access)
            bot.send_message.assert_not_called()

    async def test_max_kb_auto_start_real_ai_pipeline(self):
        """Mandatory MAX KB auto-start through REAL max_messenger_bot.ai.get_ai_response without mocking get_ai_response."""
        async with self.session_factory() as session:
            user = User(id=215, first_name="Karen", name="Karen", gender="female", age="31", accepted_disclaimer=True, current_dialogue_id=1)
            kb_doc = KnowledgeBase(id=88, filename="cbt_anxiety.txt", indexed_content="Дыхание по квадрату снижает пульс.")
            topic = Topic(id=30, name="Тревожность", is_active=True, auto_start_dialogue=True, system_prompt="Вы — эксперт по КПТ.", knowledge_base_files=[kb_doc])
            session.add_all([user, topic, kb_doc])
            await session.commit()

        client = MagicMock()
        client.send_message = AsyncMock(return_value={"message": {"mid": "m1"}})
        client.edit_message = AsyncMock()
        states = StateStore()

        mock_chunks = ["Выдержка из КБ: Дыхание по квадрату 4x4."]
        search_chunks_mock = AsyncMock(return_value=mock_chunks)
        call_gemini_mock = AsyncMock(return_value="Здравствуйте, Карен! При тревожности помогает дыхание по квадрату.")

        with patch("max_messenger_bot.ai.search_relevant_chunks", search_chunks_mock), \
             patch("max_messenger_bot.ai._call_gemini", call_gemini_mock):

            await max_topics.select_topic(client, chat_id=215, user_id=215, topic_id=30, states=states)

        # Assert search_relevant_chunks was called with document_ids=[88]
        search_chunks_mock.assert_called_once()
        _, search_kwargs = search_chunks_mock.call_args
        self.assertEqual(search_kwargs.get("document_ids"), [88])

        # Assert provider call was made
        call_gemini_mock.assert_called_once()
        _, gemini_kwargs = call_gemini_mock.call_args
        request_layout = gemini_kwargs.get("request_layout")
        self.assertIsNotNone(request_layout)
        self.assertIn("Вы — эксперт по КПТ", request_layout.stable_system_prompt)
        self.assertIn("Выдержка из КБ: Дыхание по квадрату", "\n".join(request_layout.request_context))

        # Assert DBMessage records persisted
        async with self.session_factory() as session:
            msgs = (await session.execute(select(DBMessage).where(DBMessage.user_id == 215).order_by(DBMessage.id.asc()))).scalars().all()
            self.assertEqual(len(msgs), 2)
            self.assertEqual(msgs[0].role, "user")
            self.assertEqual(msgs[0].topic_id, 30)
            self.assertEqual(msgs[0].dialogue_id, 2)
            self.assertIn("СИСТЕМНОЕ СООБЩЕНИЕ", msgs[0].content)

            self.assertEqual(msgs[1].role, "assistant")
            self.assertEqual(msgs[1].topic_id, 30)
            self.assertEqual(msgs[1].dialogue_id, 2)
            self.assertIn("Здравствуйте, Карен!", msgs[1].content)

        # Next normal turn succeeds
        call_gemini_mock.reset_mock()
        call_gemini_mock.return_value = "Да, сделайте вдох на 4 счета."

        with patch("max_messenger_bot.ai.search_relevant_chunks", search_chunks_mock), \
             patch("max_messenger_bot.ai._call_gemini", call_gemini_mock):
            await max_common.run_ai_dialogue(client, chat_id=215, user_id=215, prompt_text="Как именно дышать?", states=states)

        call_gemini_mock.assert_called_once()
        async with self.session_factory() as session:
            msgs2 = (await session.execute(select(DBMessage).where(DBMessage.user_id == 215).order_by(DBMessage.id.asc()))).scalars().all()
            self.assertEqual(len(msgs2), 4)
            self.assertEqual(msgs2[2].role, "user")
            self.assertEqual(msgs2[2].content, "Как именно дышать?")
            self.assertEqual(msgs2[3].role, "assistant")
            self.assertEqual(msgs2[3].content, "Да, сделайте вдох на 4 счета.")
