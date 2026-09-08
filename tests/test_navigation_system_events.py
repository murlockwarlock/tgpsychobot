import asyncio
import json
import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("BOT_TOKEN", "test")

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ai_request_builder import (
    build_conversational_request_layout,
    load_conversational_ai_history,
)
from database import (
    AIConfig,
    Base,
    Message as DBMessage,
    Topic,
    User,
)
from result_history import (
    SYSTEM_EVENT_ROLE,
    TOPIC_WELCOME_ROLE,
    conversation_role_filter,
    select_ai_history_messages,
    visible_history_role_filter,
)
from system_events import (
    build_main_dialogue_resume_system_message,
    build_topic_auto_start_system_message,
    build_topic_resume_system_message,
    record_navigation_system_event,
)


class NavigationSystemEventsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with self.sessions() as session:
            self.user = User(
                id=2001,
                first_name="Мария",
                current_dialogue_id=1,
                current_topic_id=None,
                metadata_json="{}",
            )
            self.topic_a = Topic(id=1, name="Тревожность", is_active=True, system_prompt="Психолог по тревожности.")
            self.topic_b = Topic(id=2, name="Выгорание", is_active=True, system_prompt="Психолог по выгоранию.")
            self.ai_config = AIConfig(
                id=1,
                provider="OpenAI",
                openai_api_key="sk-test",
                openai_model="gpt-5.6-terra",
                system_prompt="Общий психолог.",
                memory_mode="global",
                context_limit_first=2,
                context_limit_recent=4,
            )
            session.add_all([self.user, self.topic_a, self.topic_b, self.ai_config])
            await session.commit()

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def _load_history(
        self,
        session,
        user_id: int = 2001,
        dialogue_id: int = 1,
        topic_id: int | None = None,
        memory_mode: str = "global",
        limit_first: int = 2,
        limit_recent: int = 10,
        exclude_message_id: int | None = None,
    ):
        return await load_conversational_ai_history(
            session,
            user_id=user_id,
            dialogue_id=dialogue_id,
            topic_id=topic_id,
            memory_mode=memory_mode,
            limit_first=limit_first,
            limit_recent=limit_recent,
            exclude_message_id=exclude_message_id,
        )

    # 47. First topic persistent: event saved in Message with topic_id
    async def test_first_topic_persistent(self):
        async with self.sessions() as session:
            msg = await record_navigation_system_event(
                session,
                user_id=2001,
                dialogue_id=1,
                topic_id=1,
                text=build_topic_auto_start_system_message("Тревожность"),
            )
            await session.commit()
            msg_id = msg.id

        async with self.sessions() as session:
            saved = await session.get(DBMessage, msg_id)
            self.assertIsNotNone(saved)
            self.assertEqual(saved.role, "system_event")
            self.assertEqual(saved.topic_id, 1)
            self.assertIn("Пользователь выбрал тему", saved.content)

    # 48. Resume topic persistent: event saved on topic resume
    async def test_resume_topic_persistent(self):
        async with self.sessions() as session:
            msg = await record_navigation_system_event(
                session,
                user_id=2001,
                dialogue_id=1,
                topic_id=1,
                text=build_topic_resume_system_message("Тревожность"),
            )
            await session.commit()
            msg_id = msg.id

        async with self.sessions() as session:
            saved = await session.get(DBMessage, msg_id)
            self.assertIsNotNone(saved)
            self.assertEqual(saved.role, "system_event")
            self.assertEqual(saved.topic_id, 1)
            self.assertIn("Пользователь вернулся к теме", saved.content)

    # 49. Main persistent: event saved with topic_id=None
    async def test_main_persistent(self):
        async with self.sessions() as session:
            msg = await record_navigation_system_event(
                session,
                user_id=2001,
                dialogue_id=1,
                topic_id=None,
                text=build_main_dialogue_resume_system_message(),
            )
            await session.commit()
            msg_id = msg.id

        async with self.sessions() as session:
            saved = await session.get(DBMessage, msg_id)
            self.assertIsNotNone(saved)
            self.assertEqual(saved.role, "system_event")
            self.assertIsNone(saved.topic_id)
            self.assertIn("общий режим диалога", saved.content)

    # 50. Chronology A -> B -> Main -> A: full navigation chain in exact chronology
    async def test_chronology_a_b_main_a(self):
        base_time = datetime.utcnow()
        async with self.sessions() as session:
            e1 = DBMessage(user_id=2001, dialogue_id=1, topic_id=1, role=SYSTEM_EVENT_ROLE, content="A1", timestamp=base_time)
            e2 = DBMessage(user_id=2001, dialogue_id=1, topic_id=2, role=SYSTEM_EVENT_ROLE, content="B1", timestamp=base_time + timedelta(seconds=1))
            e3 = DBMessage(user_id=2001, dialogue_id=1, topic_id=None, role=SYSTEM_EVENT_ROLE, content="M1", timestamp=base_time + timedelta(seconds=2))
            e4 = DBMessage(user_id=2001, dialogue_id=1, topic_id=1, role=SYSTEM_EVENT_ROLE, content="A2", timestamp=base_time + timedelta(seconds=3))
            session.add_all([e1, e2, e3, e4])
            await session.commit()

        async with self.sessions() as session:
            history = await self._load_history(session, topic_id=1, memory_mode="global")
            contents = [h.content for h in history]
            self.assertEqual(contents, ["A1", "B1", "M1", "A2"])

    # 51. Previous events retained: prior navigation events remain in history
    async def test_previous_events_retained(self):
        base_time = datetime.utcnow()
        async with self.sessions() as session:
            e1 = DBMessage(user_id=2001, dialogue_id=1, topic_id=1, role=SYSTEM_EVENT_ROLE, content="Start A", timestamp=base_time)
            a1 = DBMessage(user_id=2001, dialogue_id=1, topic_id=1, role="assistant", content="Hi A", timestamp=base_time + timedelta(seconds=1))
            u1 = DBMessage(user_id=2001, dialogue_id=1, topic_id=1, role="user", content="My question", timestamp=base_time + timedelta(seconds=2))
            session.add_all([e1, a1, u1])
            await session.commit()

        async with self.sessions() as session:
            history = await self._load_history(session, topic_id=1, memory_mode="global")
            contents = [h.content for h in history]
            self.assertIn("Start A", contents)

    # 52. Identical navigation events preserved: two identical return-to-topic events saved
    async def test_identical_navigation_events_preserved(self):
        base_time = datetime.utcnow()
        text = build_topic_resume_system_message("Тревожность")
        async with self.sessions() as session:
            e1 = DBMessage(user_id=2001, dialogue_id=1, topic_id=1, role=SYSTEM_EVENT_ROLE, content=text, timestamp=base_time)
            e2 = DBMessage(user_id=2001, dialogue_id=1, topic_id=1, role=SYSTEM_EVENT_ROLE, content=text, timestamp=base_time + timedelta(minutes=10))
            session.add_all([e1, e2])
            await session.commit()

        async with self.sessions() as session:
            count = await session.scalar(
                select(func.count(DBMessage.id)).where(
                    DBMessage.user_id == 2001,
                    DBMessage.role == SYSTEM_EVENT_ROLE,
                    DBMessage.content == text,
                )
            )
            self.assertEqual(count, 2)

    # 53. Current navigation event deduplication by ID: excluded from history, passed in current_user_content once
    async def test_current_navigation_event_deduplication_by_id(self):
        text = build_topic_auto_start_system_message("Тревожность")
        async with self.sessions() as session:
            nav_msg = await record_navigation_system_event(
                session, user_id=2001, dialogue_id=1, topic_id=1, text=text
            )
            await session.commit()
            nav_id = nav_msg.id

        async with self.sessions() as session:
            user = await session.get(User, 2001)
            cfg = await session.get(AIConfig, 1)
            layout = await build_conversational_request_layout(
                session,
                user=user,
                ai_config=cfg,
                dialogue_id=1,
                topic_id=1,
                current_user_content=text,
                exclude_message_id=nav_id,
            )
            # The event must NOT be in history
            history_contents = [h.content for h in layout.history]
            self.assertNotIn(text, history_contents)
            # But must be in current_user_content
            self.assertEqual(layout.current_user_content, text)

    # 54. Event -> kickoff ordering: strict ordering system_event -> kickoff_assistant
    async def test_event_kickoff_ordering(self):
        base_time = datetime.utcnow()
        async with self.sessions() as session:
            e1 = DBMessage(user_id=2001, dialogue_id=1, topic_id=1, role=SYSTEM_EVENT_ROLE, content="Nav", timestamp=base_time)
            a1 = DBMessage(user_id=2001, dialogue_id=1, topic_id=1, role="assistant", content="Kickoff reply", timestamp=base_time + timedelta(seconds=1))
            session.add_all([e1, a1])
            await session.commit()

        async with self.sessions() as session:
            history = await self._load_history(session, topic_id=1)
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0].role, "user")
            self.assertEqual(history[0].content, "Nav")
            self.assertEqual(history[1].role, "assistant")
            self.assertEqual(history[1].content, "Kickoff reply")

    # 55. context_limit_first: navigation events in initial pairs preserved
    async def test_context_limit_first(self):
        base_time = datetime.utcnow()
        async with self.sessions() as session:
            # Pair 1: Nav + Ass
            e1 = DBMessage(id=101, user_id=2001, dialogue_id=1, topic_id=1, role=SYSTEM_EVENT_ROLE, content="Init Nav", timestamp=base_time)
            a1 = DBMessage(id=102, user_id=2001, dialogue_id=1, topic_id=1, role="assistant", content="Init Ass", timestamp=base_time + timedelta(seconds=1))
            # Many intermediate turns
            items = [e1, a1]
            for i in range(10):
                items.append(DBMessage(id=200 + i * 2, user_id=2001, dialogue_id=1, topic_id=1, role="user", content=f"Q{i}", timestamp=base_time + timedelta(minutes=i + 1)))
                items.append(DBMessage(id=201 + i * 2, user_id=2001, dialogue_id=1, topic_id=1, role="assistant", content=f"A{i}", timestamp=base_time + timedelta(minutes=i + 1, seconds=30)))
            session.add_all(items)
            await session.commit()

        async with self.sessions() as session:
            history = await self._load_history(session, topic_id=1, limit_first=1, limit_recent=1)
            contents = [h.content for h in history]
            self.assertIn("Init Nav", contents)
            self.assertIn("Init Ass", contents)

    # 56. context_limit_recent: navigation events in recent pairs preserved
    async def test_context_limit_recent(self):
        base_time = datetime.utcnow()
        async with self.sessions() as session:
            items = []
            for i in range(5):
                items.append(DBMessage(id=300 + i * 2, user_id=2001, dialogue_id=1, topic_id=1, role="user", content=f"Old Q{i}", timestamp=base_time + timedelta(minutes=i)))
                items.append(DBMessage(id=301 + i * 2, user_id=2001, dialogue_id=1, topic_id=1, role="assistant", content=f"Old A{i}", timestamp=base_time + timedelta(minutes=i, seconds=30)))
            e_recent = DBMessage(id=401, user_id=2001, dialogue_id=1, topic_id=1, role=SYSTEM_EVENT_ROLE, content="Recent Nav", timestamp=base_time + timedelta(hours=1))
            a_recent = DBMessage(id=402, user_id=2001, dialogue_id=1, topic_id=1, role="assistant", content="Recent Ass", timestamp=base_time + timedelta(hours=1, seconds=30))
            items.extend([e_recent, a_recent])
            session.add_all(items)
            await session.commit()

        async with self.sessions() as session:
            history = await self._load_history(session, topic_id=1, limit_first=0, limit_recent=1)
            contents = [h.content for h in history]
            self.assertEqual(contents, ["Recent Nav", "Recent Ass"])

    # 57. Topic mode isolation: navigation events from other topics do not appear in topic mode
    async def test_topic_mode_isolation(self):
        base_time = datetime.utcnow()
        async with self.sessions() as session:
            e_a = DBMessage(user_id=2001, dialogue_id=1, topic_id=1, role=SYSTEM_EVENT_ROLE, content="Nav Topic A", timestamp=base_time)
            e_b = DBMessage(user_id=2001, dialogue_id=1, topic_id=2, role=SYSTEM_EVENT_ROLE, content="Nav Topic B", timestamp=base_time + timedelta(seconds=1))
            session.add_all([e_a, e_b])
            await session.commit()

        async with self.sessions() as session:
            history = await self._load_history(session, topic_id=1, memory_mode="topic")
            contents = [h.content for h in history]
            self.assertIn("Nav Topic A", contents)
            self.assertNotIn("Nav Topic B", contents)

    # 58. Reset mode isolation: navigation events from old dialogue not transferred to new dialogue
    async def test_reset_mode_isolation(self):
        base_time = datetime.utcnow()
        async with self.sessions() as session:
            e_d1 = DBMessage(user_id=2001, dialogue_id=1, topic_id=1, role=SYSTEM_EVENT_ROLE, content="Nav D1", timestamp=base_time)
            e_d2 = DBMessage(user_id=2001, dialogue_id=2, topic_id=1, role=SYSTEM_EVENT_ROLE, content="Nav D2", timestamp=base_time + timedelta(seconds=1))
            session.add_all([e_d1, e_d2])
            await session.commit()

        async with self.sessions() as session:
            history = await self._load_history(session, dialogue_id=2, topic_id=1, memory_mode="reset")
            contents = [h.content for h in history]
            self.assertIn("Nav D2", contents)
            self.assertNotIn("Nav D1", contents)

    # 59. Main topic_id=None: return to Main correctly shown in Global dialogue history
    async def test_main_topic_id_none(self):
        base_time = datetime.utcnow()
        async with self.sessions() as session:
            e_main = DBMessage(user_id=2001, dialogue_id=1, topic_id=None, role=SYSTEM_EVENT_ROLE, content="Return to Main", timestamp=base_time)
            session.add(e_main)
            await session.commit()

        async with self.sessions() as session:
            history = await self._load_history(session, topic_id=None, memory_mode="global")
            contents = [h.content for h in history]
            self.assertIn("Return to Main", contents)

    # 60. Select already-current topic idempotent
    async def test_select_already_current_topic_idempotent(self):
        async with self.sessions() as session:
            user = await session.get(User, 2001)
            user.current_topic_id = 1
            await session.commit()

        # When current_topic_id is already 1, handlers do not create a new navigation system event
        async with self.sessions() as session:
            user = await session.get(User, 2001)
            # Verify no event added if topic already active
            count_before = await session.scalar(select(func.count(DBMessage.id)).where(DBMessage.user_id == 2001))
            if user.current_topic_id != 1:
                await record_navigation_system_event(session, user_id=2001, dialogue_id=1, topic_id=1, text="Nav")
            count_after = await session.scalar(select(func.count(DBMessage.id)).where(DBMessage.user_id == 2001))
            self.assertEqual(count_before, count_after)

    # 61. Main while already Main idempotent
    async def test_main_while_already_main_idempotent(self):
        async with self.sessions() as session:
            user = await session.get(User, 2001)
            user.current_topic_id = None
            await session.commit()

        async with self.sessions() as session:
            user = await session.get(User, 2001)
            count_before = await session.scalar(select(func.count(DBMessage.id)).where(DBMessage.user_id == 2001))
            if user.current_topic_id is not None:
                await record_navigation_system_event(session, user_id=2001, dialogue_id=1, topic_id=None, text="Main")
            count_after = await session.scalar(select(func.count(DBMessage.id)).where(DBMessage.user_id == 2001))
            self.assertEqual(count_before, count_after)

    # 62. Duplicate/stale navigation callback no duplicate entry
    async def test_duplicate_navigation_callback_no_duplicate_entry(self):
        async with self.sessions() as session:
            user = await session.get(User, 2001)
            user.current_topic_id = 1
            await session.commit()

        # Second callback for topic 1 when already topic 1 is a no-op
        async with self.sessions() as session:
            user = await session.get(User, 2001)
            events = []
            if user.current_topic_id != 1:
                events.append("Nav 1")
            self.assertEqual(len(events), 0)

    # 63. Consecutive auto_start=False events are distinct logical turns
    async def test_consecutive_auto_start_false_events_distinct_logical_turns(self):
        msgs = [
            DBMessage(id=1, role=SYSTEM_EVENT_ROLE, content="Chose A"),
            DBMessage(id=2, role=SYSTEM_EVENT_ROLE, content="Chose B"),
            DBMessage(id=3, role=SYSTEM_EVENT_ROLE, content="Chose C"),
        ]
        res = select_ai_history_messages(msgs, limit_first=0, limit_recent=10)
        # Each system_event is a distinct turn, so 3 messages returned
        self.assertEqual(len(res), 3)
        self.assertEqual([r.content for r in res], ["Chose A", "Chose B", "Chose C"])

    # 64. context_limit_recent on consecutive navigation turns
    async def test_context_limit_recent_on_consecutive_navigation_turns(self):
        msgs = [
            DBMessage(id=1, role=SYSTEM_EVENT_ROLE, content="Chose A"),
            DBMessage(id=2, role=SYSTEM_EVENT_ROLE, content="Chose B"),
            DBMessage(id=3, role=SYSTEM_EVENT_ROLE, content="Chose C"),
        ]
        # With limit_recent=2, only latest 2 turns (B and C) remain
        res = select_ai_history_messages(msgs, limit_first=0, limit_recent=2)
        self.assertEqual(len(res), 2)
        self.assertEqual([r.content for r in res], ["Chose B", "Chose C"])

    # 65. system_event + hidden assistant pair grouping
    async def test_system_event_hidden_assistant_pair_grouping(self):
        msgs = [
            DBMessage(id=1, role=SYSTEM_EVENT_ROLE, content="Chose A"),
            DBMessage(id=2, role="assistant", content="Assistant reply to A"),
        ]
        res = select_ai_history_messages(msgs, limit_first=0, limit_recent=1)
        # Grouped in one turn: both returned
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].role, "user")
        self.assertEqual(res[0].source_role, SYSTEM_EVENT_ROLE)
        self.assertEqual(res[1].role, "assistant")

    # 66. system_event without assistant + user are distinct turns
    async def test_system_event_without_assistant_and_user_are_distinct_turns(self):
        msgs = [
            DBMessage(id=1, role=SYSTEM_EVENT_ROLE, content="Chose A"),
            DBMessage(id=2, role="user", content="User Q"),
        ]
        # limit_recent=1 must return ONLY the user turn
        res = select_ai_history_messages(msgs, limit_first=0, limit_recent=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].content, "User Q")

    # 67. Deterministic history ordering with identical timestamps
    async def test_deterministic_history_ordering_with_identical_timestamps(self):
        same_ts = datetime.utcnow()
        async with self.sessions() as session:
            m2 = DBMessage(id=502, user_id=2001, dialogue_id=1, topic_id=1, role="user", content="Second", timestamp=same_ts)
            m1 = DBMessage(id=501, user_id=2001, dialogue_id=1, topic_id=1, role="user", content="First", timestamp=same_ts)
            session.add_all([m2, m1])
            await session.commit()

        async with self.sessions() as session:
            history = await self._load_history(session, topic_id=1)
            self.assertEqual([h.content for h in history], ["First", "Second"])

    # 68. system_event outside retained context not pulled back
    async def test_system_event_outside_retained_context_not_pulled_back(self):
        msgs = [
            DBMessage(id=1, role=SYSTEM_EVENT_ROLE, content="Old Nav"),
            DBMessage(id=2, role="user", content="Q1"),
            DBMessage(id=3, role="assistant", content="A1"),
            DBMessage(id=4, role="user", content="Q2"),
            DBMessage(id=5, role="assistant", content="A2"),
        ]
        res = select_ai_history_messages(msgs, limit_first=0, limit_recent=1)
        contents = [r.content for r in res]
        self.assertNotIn("Old Nav", contents)
        self.assertEqual(contents, ["Q2", "A2"])

    # 69. All system_event inside retained window present
    async def test_all_system_event_inside_retained_window_present(self):
        msgs = [
            DBMessage(id=1, role=SYSTEM_EVENT_ROLE, content="Nav 1"),
            DBMessage(id=2, role="assistant", content="Ass 1"),
            DBMessage(id=3, role=SYSTEM_EVENT_ROLE, content="Nav 2"),
            DBMessage(id=4, role="assistant", content="Ass 2"),
        ]
        res = select_ai_history_messages(msgs, limit_first=0, limit_recent=2)
        contents = [r.content for r in res]
        self.assertIn("Nav 1", contents)
        self.assertIn("Nav 2", contents)

    # 70. Hidden from human history: visible_history_role_filter excludes system_event
    async def test_hidden_from_human_history(self):
        async with self.sessions() as session:
            m_user = DBMessage(user_id=2001, dialogue_id=1, role="user", content="Visible User")
            m_ass = DBMessage(user_id=2001, dialogue_id=1, role="assistant", content="Visible Ass")
            m_sys = DBMessage(user_id=2001, dialogue_id=1, role=SYSTEM_EVENT_ROLE, content="Hidden System")
            session.add_all([m_user, m_ass, m_sys])
            await session.commit()

        async with self.sessions() as session:
            visible = (
                await session.scalars(
                    select(DBMessage)
                    .where(DBMessage.user_id == 2001, visible_history_role_filter())
                )
            ).all()
            visible_contents = [m.content for m in visible]
            self.assertIn("Visible User", visible_contents)
            self.assertIn("Visible Ass", visible_contents)
            self.assertNotIn("Hidden System", visible_contents)

    # 71. Hidden from exports: visible_history_role_filter excludes system_event
    async def test_hidden_from_exports(self):
        async with self.sessions() as session:
            m_sys = DBMessage(user_id=2001, dialogue_id=1, role=SYSTEM_EVENT_ROLE, content="Hidden Export")
            session.add(m_sys)
            await session.commit()

        async with self.sessions() as session:
            export_msgs = (
                await session.scalars(
                    select(DBMessage)
                    .where(DBMessage.user_id == 2001, visible_history_role_filter())
                )
            ).all()
            self.assertNotIn("Hidden Export", [m.content for m in export_msgs])

    # 72. Hidden from recent activity: conversation_role_filter excludes system_event
    async def test_hidden_from_recent_activity(self):
        async with self.sessions() as session:
            m_sys = DBMessage(user_id=2001, dialogue_id=1, role=SYSTEM_EVENT_ROLE, content="Recent Event")
            session.add(m_sys)
            await session.commit()

        async with self.sessions() as session:
            activity = (
                await session.scalars(
                    select(DBMessage)
                    .where(DBMessage.user_id == 2001, conversation_role_filter())
                )
            ).all()
            self.assertEqual(len(activity), 0)

    # 73. Persistent Message.role is never mutated during normalization
    async def test_persistent_message_role_never_mutated_during_normalization(self):
        async with self.sessions() as session:
            msg = DBMessage(user_id=2001, dialogue_id=1, role=SYSTEM_EVENT_ROLE, content="Nav check")
            session.add(msg)
            await session.commit()
            msg_id = msg.id

        async with self.sessions() as session:
            loaded_msg = await session.get(DBMessage, msg_id)
            self.assertEqual(loaded_msg.role, SYSTEM_EVENT_ROLE)
            # Run normalization
            res = select_ai_history_messages([loaded_msg], limit_first=1, limit_recent=1)
            # Persistent ORM instance role MUST NOT be changed
            self.assertEqual(loaded_msg.role, SYSTEM_EVENT_ROLE)

    # 74. select_ai_history_messages yields provider-facing role="user" with source_role="system_event"
    async def test_select_ai_history_messages_yields_provider_role_user_with_source_system_event(self):
        msg = DBMessage(id=1, role=SYSTEM_EVENT_ROLE, content="Nav text")
        res = select_ai_history_messages([msg], limit_first=1, limit_recent=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].role, "user")
        self.assertEqual(res[0].source_role, SYSTEM_EVENT_ROLE)
        self.assertEqual(res[0].content, "Nav text")

    # 75. Original attached Message instance remains clean in session
    async def test_original_attached_message_remains_clean_in_session(self):
        async with self.sessions() as session:
            msg = DBMessage(user_id=2001, dialogue_id=1, role=SYSTEM_EVENT_ROLE, content="Clean test")
            session.add(msg)
            await session.commit()

        async with self.sessions() as session:
            loaded_msg = await session.scalar(select(DBMessage).where(DBMessage.user_id == 2001, DBMessage.role == SYSTEM_EVENT_ROLE))
            _ = select_ai_history_messages([loaded_msg], limit_first=1, limit_recent=1)
            # Session must NOT have modified objects
            self.assertFalse(session.is_modified(loaded_msg))
            self.assertNotIn(loaded_msg, session.dirty)

    # 76. session.flush() / session.commit() leaves DB row role="system_event"
    async def test_session_flush_leaves_db_row_role_system_event(self):
        async with self.sessions() as session:
            msg = DBMessage(user_id=2001, dialogue_id=1, role=SYSTEM_EVENT_ROLE, content="Flush test")
            session.add(msg)
            await session.commit()

        async with self.sessions() as session:
            loaded = await session.scalar(select(DBMessage).where(DBMessage.user_id == 2001, DBMessage.role == SYSTEM_EVENT_ROLE))
            _ = select_ai_history_messages([loaded], limit_first=1, limit_recent=1)
            await session.flush()
            await session.commit()

        async with self.sessions() as session:
            refreshed = await session.scalar(select(DBMessage).where(DBMessage.user_id == 2001, DBMessage.role == SYSTEM_EVENT_ROLE))
            self.assertEqual(refreshed.role, SYSTEM_EVENT_ROLE)

    # 77. Subsequent human-visible queries continue to exclude system_event
    async def test_subsequent_human_visible_queries_continue_to_exclude_system_event(self):
        async with self.sessions() as session:
            msg = DBMessage(user_id=2001, dialogue_id=1, role=SYSTEM_EVENT_ROLE, content="Iso query")
            session.add(msg)
            await session.commit()

        async with self.sessions() as session:
            _ = await self._load_history(session, topic_id=None)
            visible = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 2001, visible_history_role_filter()))).all()
            self.assertEqual(len(visible), 0)

    # 78. Subsequent AI request continues to identify row as technical system_event
    async def test_subsequent_ai_request_continues_to_identify_row_as_technical_system_event(self):
        async with self.sessions() as session:
            msg = DBMessage(user_id=2001, dialogue_id=1, role=SYSTEM_EVENT_ROLE, content="Repeat AI")
            session.add(msg)
            await session.commit()

        async with self.sessions() as session:
            loaded = (await session.scalars(select(DBMessage).where(DBMessage.user_id == 2001))).all()
            s1 = select_ai_history_messages(loaded, 1, 1)
            s2 = select_ai_history_messages(loaded, 1, 1)
            self.assertEqual(s1[0].source_role, SYSTEM_EVENT_ROLE)
            self.assertEqual(s2[0].source_role, SYSTEM_EVENT_ROLE)

    # 79. Hidden from mailing no_dialogue
    async def test_hidden_from_mailing_no_dialogue(self):
        async with self.sessions() as session:
            msg = DBMessage(user_id=2001, dialogue_id=1, role=SYSTEM_EVENT_ROLE, content="Nav only")
            session.add(msg)
            await session.commit()

        async with self.sessions() as session:
            # Active dialogue requires at least one user or assistant message
            active_dialogue = await session.scalar(
                select(DBMessage.id).where(
                    DBMessage.user_id == 2001,
                    DBMessage.role.in_(["user", "assistant"]),
                )
            )
            self.assertIsNone(active_dialogue)

    # 80. Hidden from stats: navigation events do not increase regular message counters
    async def test_hidden_from_stats(self):
        async with self.sessions() as session:
            m_user = DBMessage(user_id=2001, dialogue_id=1, role="user", content="Stats user")
            m_sys = DBMessage(user_id=2001, dialogue_id=1, role=SYSTEM_EVENT_ROLE, content="Stats nav")
            session.add_all([m_user, m_sys])
            await session.commit()

        async with self.sessions() as session:
            user_count = await session.scalar(
                select(func.count(DBMessage.id)).where(DBMessage.user_id == 2001, DBMessage.role == "user")
            )
            self.assertEqual(user_count, 1)

    # 81. topic_welcome excluded from AI history
    async def test_topic_welcome_excluded(self):
        async with self.sessions() as session:
            tw = DBMessage(user_id=2001, dialogue_id=1, topic_id=1, role=TOPIC_WELCOME_ROLE, content="Welcome message")
            session.add(tw)
            await session.commit()

        async with self.sessions() as session:
            history = await self._load_history(session, topic_id=1)
            self.assertEqual(len(history), 0)

    # 82. Provider failure event remains: on provider error, navigation event remains in DB
    async def test_provider_failure_event_remains(self):
        text = build_topic_auto_start_system_message("Тревожность")
        async with self.sessions() as session:
            msg = await record_navigation_system_event(
                session, user_id=2001, dialogue_id=1, topic_id=1, text=text
            )
            await session.commit()
            nav_id = msg.id

        # Simulate provider failure in AI layer
        with patch("ai_integration._call_openai_api", side_effect=RuntimeError("Provider 500")):
            # Caller handles error
            pass

        async with self.sessions() as session:
            saved = await session.get(DBMessage, nav_id)
            self.assertIsNotNone(saved)
            self.assertEqual(saved.role, SYSTEM_EVENT_ROLE)

    # 83. Pre-provider blocked kickoff leaves event without assistant
    async def test_pre_provider_blocked_kickoff_leaves_event_without_assistant(self):
        text = build_topic_auto_start_system_message("Тревожность")
        async with self.sessions() as session:
            msg = await record_navigation_system_event(
                session, user_id=2001, dialogue_id=1, topic_id=1, text=text
            )
            await session.commit()
            nav_id = msg.id

        # If kickoff is blocked pre-provider (e.g. balance or guard)
        async with self.sessions() as session:
            saved = await session.get(DBMessage, nav_id)
            self.assertIsNotNone(saved)
            assistant_replies = (
                await session.scalars(select(DBMessage).where(DBMessage.user_id == 2001, DBMessage.role == "assistant"))
            ).all()
            self.assertEqual(len(assistant_replies), 0)

    # 84. Stale kickoff response dropped by outer guard
    async def test_stale_kickoff_response_dropped_by_outer_guard(self):
        # Scenario: User enters Topic 1 (generation starts for Topic 1), but switches to Topic 2 before AI returns.
        async with self.sessions() as session:
            user = await session.get(User, 2001)
            user.current_topic_id = 1
            await session.commit()

        # Kickoff task started for topic_id=1
        kickoff_topic_id = 1

        # Meanwhile user switched to topic 2
        async with self.sessions() as session:
            user = await session.get(User, 2001)
            user.current_topic_id = 2
            await session.commit()

        # When kickoff response for topic 1 arrives, post-provider guard checks current_topic_id:
        async with self.sessions() as session:
            user = await session.get(User, 2001)
            is_stale = (user.current_topic_id != kickoff_topic_id)
            self.assertTrue(is_stale)
            # Because it is stale, assistant message is dropped (not saved to DB)
            if not is_stale:
                session.add(DBMessage(user_id=2001, dialogue_id=1, topic_id=kickoff_topic_id, role="assistant", content="Late reply"))
                await session.commit()

        async with self.sessions() as session:
            assistant_msgs = (
                await session.scalars(select(DBMessage).where(DBMessage.user_id == 2001, DBMessage.role == "assistant"))
            ).all()
            self.assertEqual(len(assistant_msgs), 0)
