import html
import json
import os
import unittest
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("BOT_TOKEN", "test")

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import ai_integration
from ai_request_builder import resolve_context_limit
from automation_engine import (
    ServiceDataBlock,
    apply_service_data_blocks,
    build_runtime_automation_context,
    get_dialogue_automation_state,
)
from database import (
    AIConfig,
    AILog,
    AutomationConversationState,
    AutomationDialogueState,
    AutomationEvent,
    Base,
    Message as DBMessage,
    Topic,
    User,
)
import handlers
from max_messenger_bot import ai as max_ai
from memory_mode import (
    MEMORY_MODE_GLOBAL,
    MEMORY_MODE_RESET,
    MEMORY_MODE_TOPIC,
)
from result_history import AIHistoryMessage, SYSTEM_EVENT_ROLE, select_ai_history_messages


class TestContextLimitsZeroAndGlobalMetadata(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        self._orig_tg_sessions = ai_integration.async_session_maker
        self._orig_max_sessions = max_ai.async_session_maker
        self._orig_handlers_sessions = handlers.async_session_maker
        ai_integration.async_session_maker = self.sessions
        max_ai.async_session_maker = self.sessions
        handlers.async_session_maker = self.sessions

    async def asyncTearDown(self):
        ai_integration.async_session_maker = self._orig_tg_sessions
        max_ai.async_session_maker = self._orig_max_sessions
        handlers.async_session_maker = self._orig_handlers_sessions
        await self.engine.dispose()

    # =========================================================================
    # 1. Resolver semantics: None, 0, positive, negative, invalid, bool
    # =========================================================================
    def test_resolver_semantics(self):
        # None -> supplied default
        self.assertEqual(resolve_context_limit(None, default=2), 2)
        self.assertEqual(resolve_context_limit(None, default=10), 10)

        # 0 -> 0 (honoring explicit zero!)
        self.assertEqual(resolve_context_limit(0, default=2), 0)
        self.assertEqual(resolve_context_limit(0, default=10), 0)
        self.assertEqual(resolve_context_limit("0", default=2), 0)

        # positive integer -> exact value
        self.assertEqual(resolve_context_limit(5, default=2), 5)
        self.assertEqual(resolve_context_limit("20", default=10), 20)

        # negative integer -> supplied default
        self.assertEqual(resolve_context_limit(-1, default=2), 2)
        self.assertEqual(resolve_context_limit(-100, default=10), 10)
        self.assertEqual(resolve_context_limit("-5", default=2), 2)

        # invalid / defensive input -> supplied default
        self.assertEqual(resolve_context_limit("invalid", default=2), 2)
        self.assertEqual(resolve_context_limit([], default=10), 10)
        self.assertEqual(resolve_context_limit({}, default=2), 2)

        # booleans -> supplied default (not coerced via int(False) == 0)
        self.assertEqual(resolve_context_limit(False, default=2), 2)
        self.assertEqual(resolve_context_limit(True, default=10), 10)

    # =========================================================================
    # 2. Selector zero semantics: 0/20, 2/0, 0/0, 2/10, overlap, system_event
    # =========================================================================
    def test_selector_zero_and_boundary_semantics(self):
        # Generate 25 turns (user + assistant)
        raw_msgs = []
        for i in range(1, 26):
            raw_msgs.append(AIHistoryMessage(role="user", content=f"old-turn-{i:02d}-user", topic_id=None, source_role="user"))
            raw_msgs.append(AIHistoryMessage(role="assistant", content=f"old-turn-{i:02d}-assistant", topic_id=None, source_role="assistant"))

        # first=0, recent=20 -> only last 20 logical turns (turns 06..25)
        res_0_20 = select_ai_history_messages(raw_msgs, limit_first=0, limit_recent=20)
        contents_0_20 = [m.content for m in res_0_20]
        # turns 01..05 must be absent
        for i in range(1, 6):
            self.assertNotIn(f"old-turn-{i:02d}-user", contents_0_20)
            self.assertNotIn(f"old-turn-{i:02d}-assistant", contents_0_20)
        # turns 06..25 must be present
        for i in range(6, 26):
            self.assertIn(f"old-turn-{i:02d}-user", contents_0_20)
            self.assertIn(f"old-turn-{i:02d}-assistant", contents_0_20)
        self.assertEqual(len(res_0_20), 40)  # 20 turns * 2 messages

        # first=2, recent=0 -> only first 2 logical turns (turns 01..02)
        res_2_0 = select_ai_history_messages(raw_msgs, limit_first=2, limit_recent=0)
        contents_2_0 = [m.content for m in res_2_0]
        self.assertIn("old-turn-01-user", contents_2_0)
        self.assertIn("old-turn-01-assistant", contents_2_0)
        self.assertIn("old-turn-02-user", contents_2_0)
        self.assertIn("old-turn-02-assistant", contents_2_0)
        for i in range(3, 26):
            self.assertNotIn(f"old-turn-{i:02d}-user", contents_2_0)
            self.assertNotIn(f"old-turn-{i:02d}-assistant", contents_2_0)
        self.assertEqual(len(res_2_0), 4)  # 2 turns * 2 messages

        # first=0, recent=0 -> no historical turns
        res_0_0 = select_ai_history_messages(raw_msgs, limit_first=0, limit_recent=0)
        self.assertEqual(res_0_0, [])

        # first=2, recent=10 on 25 turns -> first 2 + last 10 (turns 01, 02, 16..25)
        res_2_10 = select_ai_history_messages(raw_msgs, limit_first=2, limit_recent=10)
        contents_2_10 = [m.content for m in res_2_10]
        self.assertIn("old-turn-01-user", contents_2_10)
        self.assertIn("old-turn-02-user", contents_2_10)
        for i in range(3, 16):
            self.assertNotIn(f"old-turn-{i:02d}-user", contents_2_10)
        for i in range(16, 26):
            self.assertIn(f"old-turn-{i:02d}-user", contents_2_10)
        self.assertEqual(len(res_2_10), 24)  # 12 turns * 2 messages

        # first=2, recent=10 on <= 12 turns (e.g. 8 turns) -> all turns exactly once, no duplicates
        raw_8 = raw_msgs[:16]  # 8 turns
        res_8 = select_ai_history_messages(raw_8, limit_first=2, limit_recent=10)
        contents_8 = [m.content for m in res_8]
        self.assertEqual(len(res_8), 16)
        for i in range(1, 9):
            self.assertEqual(contents_8.count(f"old-turn-{i:02d}-user"), 1)
            self.assertEqual(contents_8.count(f"old-turn-{i:02d}-assistant"), 1)

        # system_event boundary behavior: system event starts a logical turn
        mixed_msgs = [
            AIHistoryMessage(role="user", content="event-1", topic_id=None, source_role=SYSTEM_EVENT_ROLE),
            AIHistoryMessage(role="assistant", content="kickoff-1", topic_id=None, source_role="assistant"),
            AIHistoryMessage(role="user", content="u-2", topic_id=None, source_role="user"),
            AIHistoryMessage(role="assistant", content="a-2", topic_id=None, source_role="assistant"),
        ]
        res_sys = select_ai_history_messages(mixed_msgs, limit_first=1, limit_recent=0)
        self.assertEqual(len(res_sys), 2)
        self.assertEqual(res_sys[0].content, "event-1")
        self.assertEqual(res_sys[1].content, "kickoff-1")

    # =========================================================================
    # Helper to setup user with 25 turns in DB
    # =========================================================================
    async def _setup_dialogue_with_25_turns(self, user_id: int = 1001, is_max: bool = False):
        async with self.sessions() as session:
            user = User(
                id=user_id,
                first_name="Тест",
                username="testuser",
                current_dialogue_id=1,
                current_topic_id=1,
            )
            topic = Topic(id=1, name="Основной топик", is_active=True, system_prompt="Промпт топика")
            ai_config = AIConfig(
                id=1,
                provider="OpenAI",
                openai_api_key="sk-test-key",
                openai_model="gpt-5.6-terra",
                system_prompt="Системный промпт бота",
                context_limit_first=2,
                context_limit_recent=10,
                memory_mode="global",
            )
            session.add_all([user, topic, ai_config])
            await session.commit()

            base_time = datetime(2026, 1, 1, 12, 0, 0)
            for i in range(1, 26):
                u_msg = DBMessage(
                    user_id=user_id,
                    role="user",
                    content=f"old-turn-{i:02d}-user",
                    dialogue_id=1,
                    topic_id=1,
                    timestamp=base_time + timedelta(seconds=i * 2),
                )
                a_msg = DBMessage(
                    user_id=user_id,
                    role="assistant",
                    content=f"old-turn-{i:02d}-assistant",
                    dialogue_id=1,
                    topic_id=1,
                    timestamp=base_time + timedelta(seconds=i * 2 + 1),
                )
                session.add_all([u_msg, a_msg])
            await session.commit()

    # =========================================================================
    # 3. TG: Existing dialogue picks up config (0/20 and 0/0) immediately
    # =========================================================================
    async def test_tg_existing_dialogue_picks_up_config_immediately(self):
        await self._setup_dialogue_with_25_turns(user_id=1001)

        # Update AIConfig to 0/20 on existing dialogue
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.context_limit_first = 0
            cfg.context_limit_recent = 20
            await session.commit()

        captured_requests = []

        async def fake_create(**kwargs):
            captured_requests.append(kwargs)
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "Ответ ассистента на 26"
            resp.choices = [choice]
            return resp

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create):
            response = await ai_integration.get_ai_response(
                user_id=1001,
                user_prompt="current-turn-26-user",
                user_name="Тест",
                user_gender="male",
                track_user_activity=False,
            )

        self.assertEqual(len(captured_requests), 1)
        wire_messages = captured_requests[0]["messages"]
        wire_contents = [m["content"] for m in wire_messages if isinstance(m.get("content"), str)]

        # Prove turns 01..05 absent
        for i in range(1, 6):
            self.assertFalse(any(f"old-turn-{i:02d}-user" in c for c in wire_contents))
            self.assertFalse(any(f"old-turn-{i:02d}-assistant" in c for c in wire_contents))

        # Prove turns 06..25 present exactly once
        for i in range(6, 26):
            u_matches = sum(1 for c in wire_contents if f"old-turn-{i:02d}-user" in c)
            a_matches = sum(1 for c in wire_contents if f"old-turn-{i:02d}-assistant" in c)
            self.assertEqual(u_matches, 1, f"turn {i} user should appear once")
            self.assertEqual(a_matches, 1, f"turn {i} assistant should appear once")

        # Prove chronological order retained
        indices = [wire_contents.index(f"old-turn-{i:02d}-user") for i in range(6, 26)]
        self.assertEqual(indices, sorted(indices))

        # Prove current-turn-26-user present as current request
        last_msg = wire_messages[-1]
        self.assertEqual(last_msg["role"], "user")
        self.assertIn("current-turn-26-user", last_msg["content"])

        # Prove dialogue ID unchanged
        async with self.sessions() as session:
            user = await session.get(User, 1001)
            self.assertEqual(user.current_dialogue_id, 1)

        # Now test 0/0 live setting change on the same dialogue
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.context_limit_first = 0
            cfg.context_limit_recent = 0
            await session.commit()

        captured_requests.clear()
        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create):
            await ai_integration.get_ai_response(
                user_id=1001,
                user_prompt="current-turn-27-user",
                user_name="Тест",
                user_gender="male",
                track_user_activity=False,
            )

        self.assertEqual(len(captured_requests), 1)
        wire_messages_0_0 = captured_requests[0]["messages"]
        wire_contents_0_0 = [m["content"] for m in wire_messages_0_0 if isinstance(m.get("content"), str)]

        # Prove zero historical turns
        for i in range(1, 26):
            self.assertFalse(any(f"old-turn-{i:02d}" in c for c in wire_contents_0_0))

        # Current user request still serialized normally
        last_msg_0_0 = wire_messages_0_0[-1]
        self.assertEqual(last_msg_0_0["role"], "user")
        self.assertIn("current-turn-27-user", last_msg_0_0["content"])

    # =========================================================================
    # 4. MAX: Existing dialogue picks up config (0/20 and 0/0) immediately
    # =========================================================================
    async def test_max_existing_dialogue_picks_up_config_immediately(self):
        # MAX user id >= 10_000_000_000
        max_user_id = 10_000_000_001
        await self._setup_dialogue_with_25_turns(user_id=max_user_id, is_max=True)

        # Update AIConfig to 0/20
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.context_limit_first = 0
            cfg.context_limit_recent = 20
            await session.commit()

        captured_requests = []

        async def fake_create(**kwargs):
            captured_requests.append(kwargs)
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "MAX ответ ассистента"
            resp.choices = [choice]
            return resp

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create):
            await max_ai.get_ai_response(
                user_id=max_user_id,
                user_prompt="current-turn-26-user",
                track_user_activity=False,
            )

        self.assertEqual(len(captured_requests), 1)
        wire_messages = captured_requests[0]["messages"]
        wire_contents = [m["content"] for m in wire_messages if isinstance(m.get("content"), str)]

        # Prove turns 01..05 absent
        for i in range(1, 6):
            self.assertFalse(any(f"old-turn-{i:02d}-user" in c for c in wire_contents))
            self.assertFalse(any(f"old-turn-{i:02d}-assistant" in c for c in wire_contents))

        # Prove turns 06..25 present exactly once
        for i in range(6, 26):
            u_matches = sum(1 for c in wire_contents if f"old-turn-{i:02d}-user" in c)
            a_matches = sum(1 for c in wire_contents if f"old-turn-{i:02d}-assistant" in c)
            self.assertEqual(u_matches, 1, f"MAX turn {i} user should appear once")
            self.assertEqual(a_matches, 1, f"MAX turn {i} assistant should appear once")

        # Prove chronological order retained
        indices = [wire_contents.index(f"old-turn-{i:02d}-user") for i in range(6, 26)]
        self.assertEqual(indices, sorted(indices))

        # Prove current-turn-26-user present as current request
        last_msg = wire_messages[-1]
        self.assertEqual(last_msg["role"], "user")
        self.assertIn("current-turn-26-user", last_msg["content"])

        # Prove dialogue ID unchanged
        async with self.sessions() as session:
            user = await session.get(User, max_user_id)
            self.assertEqual(user.current_dialogue_id, 1)

        # Test 0/0 live setting change for MAX
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.context_limit_first = 0
            cfg.context_limit_recent = 0
            await session.commit()

        captured_requests.clear()
        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create):
            await max_ai.get_ai_response(
                user_id=max_user_id,
                user_prompt="current-turn-27-user",
                track_user_activity=False,
            )

        self.assertEqual(len(captured_requests), 1)
        wire_messages_0_0 = captured_requests[0]["messages"]
        wire_contents_0_0 = [m["content"] for m in wire_messages_0_0 if isinstance(m.get("content"), str)]

        # Prove zero historical turns
        for i in range(1, 26):
            self.assertFalse(any(f"old-turn-{i:02d}" in c for c in wire_contents_0_0))

        # Current user request still serialized normally
        last_msg_0_0 = wire_messages_0_0[-1]
        self.assertEqual(last_msg_0_0["role"], "user")
        self.assertIn("current-turn-27-user", last_msg_0_0["content"])

    # =========================================================================
    # 5. GLOBAL Multi-Topic Merge: exact consistency across runtime, admin, export
    # =========================================================================
    async def test_global_multi_topic_merge_and_exact_equality(self):
        user_id = 42
        async with self.sessions() as session:
            user = User(
                id=user_id,
                first_name="Иван",
                username="ivan_user",
                current_dialogue_id=1,
                current_topic_id=1,
            )
            topic1 = Topic(id=1, name="Тревожность", is_active=True)
            topic2 = Topic(id=2, name="Сон", is_active=True)
            ai_config = AIConfig(
                id=1,
                provider="OpenAI",
                openai_api_key="sk-test",
                openai_model="gpt-5.6-terra",
                memory_mode=MEMORY_MODE_GLOBAL,
            )
            session.add_all([user, topic1, topic2, ai_config])
            await session.commit()

            # Realistic metadata fields
            meta_a = {
                "primary_theme": "Тревожность",
                "secondary_themes": ["Стресс на работе"],
                "situation_summary": "Клиент испытывает панические атаки на рабочем месте.",
                "emotional_state": "Подавленное, тревожное",
                "key_facts": ["Работает в IT", "Сменил команду месяц назад"],
                "depth_level": 2,
                "master_recommended": True,
                "master_response": "Рекомендована консультация психолога",
                "safety_flag": False,
            }
            from user_metadata import extract_service_data
            _, blocks_a, _ = extract_service_data(
                f"<DATA>{json.dumps({'current_state': {'current_step': 'step_intro'}, 'metadata': meta_a})}</DATA>"
            )
            # Topic A writes initial metadata
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=1,
                blocks=blocks_a,
                memory_mode=MEMORY_MODE_GLOBAL,
            )
            await session.commit()

            # Topic B writes additional/updated metadata
            meta_b = {
                "secondary_themes": ["Стресс на работе", "Бессонница"],
                "situation_summary": "Клиент испытывает панические атаки и плохо спит.",
                "emotional_state": "Истощенное",
                "key_facts": ["Работает в IT", "Сменил команду месяц назад", "Не спит по 4 часа"],
                "depth_level": 3,
                "safety_flag": False,
            }
            _, blocks_b, _ = extract_service_data(
                f"<DATA>{json.dumps({'current_state': {'current_step': 'step_sleep_assessment'}, 'metadata': meta_b})}</DATA>"
            )
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=2,
                blocks=blocks_b,
                memory_mode=MEMORY_MODE_GLOBAL,
            )
            await session.commit()

        async with self.sessions() as session:
            # 1. AutomationDialogueState.metadata_json
            diag_state = await get_dialogue_automation_state(session, user_id=user_id, dialogue_id=1)
            self.assertIsNotNone(diag_state)
            canonical_db_metadata = json.loads(diag_state.metadata_json)

            # 2. build_runtime_automation_context() metadata
            runtime_raw = await build_runtime_automation_context(
                session,
                user_id=user_id,
                dialogue_id=1,
                topic_id=2,
                memory_mode=MEMORY_MODE_GLOBAL,
            )
            # Parse runtime payload from second line
            runtime_payload_str = runtime_raw.split("\n", 1)[1]
            runtime_payload = json.loads(runtime_payload_str)
            runtime_metadata = runtime_payload["metadata"]

            # 3. Topic current_state / current_step remain topic-scoped
            conv1 = await session.scalar(
                select(AutomationConversationState).where(
                    AutomationConversationState.user_id == user_id,
                    AutomationConversationState.dialogue_id == 1,
                    AutomationConversationState.topic_id == 1,
                )
            )
            conv2 = await session.scalar(
                select(AutomationConversationState).where(
                    AutomationConversationState.user_id == user_id,
                    AutomationConversationState.dialogue_id == 1,
                    AutomationConversationState.topic_id == 2,
                )
            )
            self.assertEqual(conv1.current_step, "step_intro")
            self.assertEqual(json.loads(conv1.current_state_json), {"current_step": "step_intro"})
            self.assertEqual(conv2.current_step, "step_sleep_assessment")
            self.assertEqual(json.loads(conv2.current_state_json), {"current_step": "step_sleep_assessment"})

        # Prove exact equality: DB == runtime context
        self.assertEqual(canonical_db_metadata, runtime_metadata)
        self.assertEqual(canonical_db_metadata["primary_theme"], "Тревожность")
        self.assertEqual(canonical_db_metadata["secondary_themes"], ["Стресс на работе", "Бессонница"])
        self.assertEqual(canonical_db_metadata["depth_level"], 3)
        self.assertEqual(canonical_db_metadata["master_recommended"], True)

        # 4. Admin Merged Export canonical metadata exact equality
        export_callback = MagicMock()
        export_callback.from_user.id = 999999
        export_callback.data = f"run_metadata_export_merged_{user_id}"
        export_callback.message.answer_document = AsyncMock()
        export_callback.answer = AsyncMock()

        with patch("handlers.check_history_permission", AsyncMock(return_value=True)):
            await handlers.run_client_metadata_export(export_callback)

        export_callback.message.answer_document.assert_awaited_once()
        sent_doc = export_callback.message.answer_document.call_args[0][0]
        exported_json = json.loads(sent_doc.data.decode("utf-8"))

        self.assertEqual(exported_json["export_type"], "consolidated_merged_metadata")
        self.assertEqual(len(exported_json["dialogue_states"]), 1)
        exported_diag = exported_json["dialogue_states"][0]
        self.assertEqual(exported_diag["dialogue_id"], 1)
        self.assertEqual(exported_diag["memory_mode"], "global")
        # Assert canonical export metadata matches DB exactly
        self.assertEqual(exported_diag["metadata"], canonical_db_metadata)
        # Assert topic states retained independently
        self.assertEqual(len(exported_diag["topic_states"]), 2)
        topic_ids = [ts["topic_id"] for ts in exported_diag["topic_states"]]
        self.assertEqual(topic_ids, [1, 2])
        self.assertEqual(exported_diag["topic_states"][0]["current_step"], "step_intro")
        self.assertEqual(exported_diag["topic_states"][1]["current_step"], "step_sleep_assessment")

        # 5. Admin Merged Display canonical metadata exact equality
        view_callback = MagicMock()
        view_callback.from_user.id = 999999
        view_callback.data = f"client_merged_metadata_{user_id}_0"
        view_callback.message.edit_text = AsyncMock()
        view_callback.answer = AsyncMock()

        with patch("handlers.check_history_permission", AsyncMock(return_value=True)):
            await handlers.view_client_merged_metadata(view_callback)

        view_callback.message.edit_text.assert_awaited_once()
        view_text = view_callback.message.edit_text.call_args[0][0]
        # Assert the complete expected canonical metadata rendering is contained in admin view text
        expected_rendered = html.escape(
            json.dumps(canonical_db_metadata, ensure_ascii=False, indent=2)
        )
        self.assertIn(expected_rendered, view_text)
        self.assertIn("step_intro", view_text)
        self.assertIn("step_sleep_assessment", view_text)

    # =========================================================================
    # 6. Admin read must prove ZERO persistence mutation (both initialized & uninitialized)
    # =========================================================================
    async def test_admin_read_and_export_zero_persistence_mutation(self):
        user_id = 77
        dt_fixed = datetime(2026, 9, 13, 10, 0, 0)
        async with self.sessions() as session:
            user = User(id=user_id, first_name="Анна", username="anna_user", current_dialogue_id=2)
            ai_config = AIConfig(id=1, memory_mode=MEMORY_MODE_GLOBAL)
            # Dialogue 1: Uninitialized dialogue (topic state exists, but NO AutomationDialogueState)
            conv_uninit = AutomationConversationState(
                user_id=user_id,
                dialogue_id=1,
                topic_id=1,
                current_step="step_start",
                current_state_json='{"current_step": "step_start"}',
                metadata_json='{"local": "old"}',
                updated_at=dt_fixed,
            )
            # Dialogue 2: Fully initialized dialogue (AutomationDialogueState exists with metadata and updated_at)
            diag_init = AutomationDialogueState(
                user_id=user_id,
                dialogue_id=2,
                metadata_json='{"primary_theme": "Тревога", "depth_level": 2}',
                updated_at=dt_fixed,
            )
            conv_init = AutomationConversationState(
                user_id=user_id,
                dialogue_id=2,
                topic_id=2,
                current_step="step_progress",
                current_state_json='{"current_step": "step_progress"}',
                metadata_json='{"local": "ignore"}',
                updated_at=dt_fixed,
            )
            session.add_all([user, ai_config, conv_uninit, diag_init, conv_init])
            await session.commit()

        # Snapshot DB state BEFORE admin view and export
        async with self.sessions() as session:
            diag_count_before = await session.scalar(select(func.count(AutomationDialogueState.id)))
            conv_count_before = await session.scalar(select(func.count(AutomationConversationState.id)))
            diag_rows_before = [
                (d.id, d.user_id, d.dialogue_id, d.metadata_json, d.updated_at)
                for d in (await session.scalars(select(AutomationDialogueState).order_by(AutomationDialogueState.id))).all()
            ]
            conv_rows_before = [
                (c.id, c.user_id, c.dialogue_id, c.topic_id, c.current_step, c.current_state_json, c.metadata_json, c.updated_at)
                for c in (await session.scalars(select(AutomationConversationState).order_by(AutomationConversationState.id))).all()
            ]

        self.assertEqual(diag_count_before, 1)
        self.assertEqual(conv_count_before, 2)

        # 1. Invoke merged admin view on page 0 (dialogue 2 - initialized)
        view_callback_init = MagicMock()
        view_callback_init.from_user.id = 999999
        view_callback_init.data = f"client_merged_metadata_{user_id}_0"
        view_callback_init.message.edit_text = AsyncMock()
        view_callback_init.answer = AsyncMock()

        with patch("handlers.check_history_permission", AsyncMock(return_value=True)):
            await handlers.view_client_merged_metadata(view_callback_init)

        view_text_init = view_callback_init.message.edit_text.call_args[0][0]
        self.assertIn("Тревога", view_text_init)

        # 2. Invoke merged admin view on page 1 (dialogue 1 - uninitialized)
        view_callback_uninit = MagicMock()
        view_callback_uninit.from_user.id = 999999
        view_callback_uninit.data = f"client_merged_metadata_{user_id}_1"
        view_callback_uninit.message.edit_text = AsyncMock()
        view_callback_uninit.answer = AsyncMock()

        with patch("handlers.check_history_permission", AsyncMock(return_value=True)):
            await handlers.view_client_merged_metadata(view_callback_uninit)

        view_text_uninit = view_callback_uninit.message.edit_text.call_args[0][0]
        self.assertIn("не инициализированы", view_text_uninit)

        # 3. Invoke merged metadata export
        export_callback = MagicMock()
        export_callback.from_user.id = 999999
        export_callback.data = f"run_metadata_export_merged_{user_id}"
        export_callback.message.answer_document = AsyncMock()
        export_callback.answer = AsyncMock()

        with patch("handlers.check_history_permission", AsyncMock(return_value=True)):
            await handlers.run_client_metadata_export(export_callback)

        sent_doc = export_callback.message.answer_document.call_args[0][0]
        export_data = json.loads(sent_doc.data.decode("utf-8"))
        # Dialogue 2 has canonical metadata, dialogue 1 has null
        self.assertEqual(export_data["dialogue_states"][0]["dialogue_id"], 2)
        self.assertEqual(export_data["dialogue_states"][0]["metadata"], {"primary_theme": "Тревога", "depth_level": 2})
        self.assertEqual(export_data["dialogue_states"][1]["dialogue_id"], 1)
        self.assertIsNone(export_data["dialogue_states"][1]["metadata"])

        # Snapshot DB state AFTER admin operations
        async with self.sessions() as session:
            diag_count_after = await session.scalar(select(func.count(AutomationDialogueState.id)))
            conv_count_after = await session.scalar(select(func.count(AutomationConversationState.id)))
            diag_rows_after = [
                (d.id, d.user_id, d.dialogue_id, d.metadata_json, d.updated_at)
                for d in (await session.scalars(select(AutomationDialogueState).order_by(AutomationDialogueState.id))).all()
            ]
            conv_rows_after = [
                (c.id, c.user_id, c.dialogue_id, c.topic_id, c.current_step, c.current_state_json, c.metadata_json, c.updated_at)
                for c in (await session.scalars(select(AutomationConversationState).order_by(AutomationConversationState.id))).all()
            ]

        # Assert absolute zero mutation: row counts, fields, timestamps are 100% invariant
        self.assertEqual(diag_count_before, diag_count_after)
        self.assertEqual(conv_count_before, conv_count_after)
        self.assertEqual(diag_rows_before, diag_rows_after)
        self.assertEqual(conv_rows_before, conv_rows_after)

    # =========================================================================
    # 7. Real length regression test: safe budget <= 3900 chars under extreme load
    # =========================================================================
    async def test_global_admin_display_length_bounding_and_truncation(self):
        user_id = 888
        dt_fixed = datetime(2026, 9, 13, 10, 0, 0)
        async with self.sessions() as session:
            user = User(id=user_id, first_name="Большой Пользователь", username="big_user", current_dialogue_id=1)
            ai_config = AIConfig(id=1, memory_mode=MEMORY_MODE_GLOBAL)

            # 1. Canonical metadata large enough to exercise truncation (> 10,000 chars)
            large_meta = {
                "primary_theme": "Стресс и перегрузка",
                "secondary_themes": [f"Специфическая подтема номер {i}" for i in range(100)],
                "situation_summary": "Длинное описание ситуации " * 200,
                "key_facts": [f"Факт из биографии номер {i}: детальное описание жизненных обстоятельств" for i in range(50)],
                "depth_level": 5,
            }
            diag_state = AutomationDialogueState(
                user_id=user_id,
                dialogue_id=1,
                metadata_json=json.dumps(large_meta, ensure_ascii=False),
                updated_at=dt_fixed,
            )
            session.add_all([user, ai_config, diag_state])

            # 2. Many topic states (15 topics) with large current_state_json (> 3,000 chars each)
            for i in range(1, 16):
                large_state = {
                    "current_step": f"step_complex_topic_pipeline_{i}",
                    "diagnostic_markers": [f"marker_data_{j}_{'x' * 100}" for j in range(25)],
                    "session_notes": "Заметка по ходу сессии " * 50,
                }
                conv = AutomationConversationState(
                    user_id=user_id,
                    dialogue_id=1,
                    topic_id=i,
                    current_step=f"step_complex_topic_pipeline_{i}",
                    current_state_json=json.dumps(large_state, ensure_ascii=False),
                    updated_at=dt_fixed + timedelta(minutes=i),
                )
                session.add(conv)

            await session.commit()

        # Snapshot DB before
        async with self.sessions() as session:
            diag_before = [
                (d.id, d.metadata_json, d.updated_at)
                for d in (await session.scalars(select(AutomationDialogueState))).all()
            ]
            conv_before = [
                (c.id, c.current_step, c.current_state_json, c.updated_at)
                for c in (await session.scalars(select(AutomationConversationState))).all()
            ]

        # Invoke merged admin display
        view_callback = MagicMock()
        view_callback.from_user.id = 999999
        view_callback.data = f"client_merged_metadata_{user_id}_0"
        view_callback.message.edit_text = AsyncMock()
        view_callback.answer = AsyncMock()

        with patch("handlers.check_history_permission", AsyncMock(return_value=True)):
            await handlers.view_client_merged_metadata(view_callback)

        view_callback.message.edit_text.assert_awaited_once()
        final_text = view_callback.message.edit_text.call_args[0][0]

        # Assert final text length is safely within the safe message budget (<= 3900 chars)
        self.assertLessEqual(len(final_text), 3900)

        # Assert valid canonical metadata remains visibly represented
        self.assertIn("Стресс и перегрузка", final_text)
        self.assertIn("... (полный текст доступен при скачивании .json)", final_text)

        # Assert topic summary is bounded (only first 5 topics shown in UI)
        self.assertIn("Топик #1", final_text)
        self.assertIn("Топик #5", final_text)
        self.assertNotIn("Топик #6", final_text)

        # Assert omission marker appears for remaining 10 topic states
        self.assertIn("... ещё 10 состояний; полный список доступен в JSON", final_text)

        # Assert NO large current_state_json payloads were dumped into the Telegram text
        self.assertNotIn("diagnostic_markers", final_text)
        self.assertNotIn("marker_data", final_text)

        # Assert zero DB mutation after view
        async with self.sessions() as session:
            diag_after_view = [
                (d.id, d.metadata_json, d.updated_at)
                for d in (await session.scalars(select(AutomationDialogueState))).all()
            ]
            conv_after_view = [
                (c.id, c.current_step, c.current_state_json, c.updated_at)
                for c in (await session.scalars(select(AutomationConversationState))).all()
            ]
        self.assertEqual(diag_before, diag_after_view)
        self.assertEqual(conv_before, conv_after_view)

        # Invoke JSON export: full states must still be present in the JSON export
        export_callback = MagicMock()
        export_callback.from_user.id = 999999
        export_callback.data = f"run_metadata_export_merged_{user_id}"
        export_callback.message.answer_document = AsyncMock()
        export_callback.answer = AsyncMock()

        with patch("handlers.check_history_permission", AsyncMock(return_value=True)):
            await handlers.run_client_metadata_export(export_callback)

        sent_doc = export_callback.message.answer_document.call_args[0][0]
        export_data = json.loads(sent_doc.data.decode("utf-8"))

        # Assert full untruncated canonical metadata is present in export
        self.assertEqual(export_data["dialogue_states"][0]["metadata"]["primary_theme"], "Стресс и перегрузка")
        self.assertEqual(len(export_data["dialogue_states"][0]["metadata"]["secondary_themes"]), 100)

        # Assert ALL 15 topic states and full current_state payloads are present in export
        topic_states_export = export_data["dialogue_states"][0]["topic_states"]
        self.assertEqual(len(topic_states_export), 15)
        for i, ts in enumerate(topic_states_export, start=1):
            self.assertEqual(ts["topic_id"], i)
            self.assertIn("diagnostic_markers", ts["current_state"])
            self.assertEqual(len(ts["current_state"]["diagnostic_markers"]), 25)

        # Assert zero DB mutation after export as well
        async with self.sessions() as session:
            diag_after_export = [
                (d.id, d.metadata_json, d.updated_at)
                for d in (await session.scalars(select(AutomationDialogueState))).all()
            ]
            conv_after_export = [
                (c.id, c.current_step, c.current_state_json, c.updated_at)
                for c in (await session.scalars(select(AutomationConversationState))).all()
            ]
        self.assertEqual(diag_before, diag_after_export)
        self.assertEqual(conv_before, conv_after_export)

    # =========================================================================
    # 7. TOPIC and RESET Non-Regression
    # =========================================================================
    async def test_topic_and_reset_mode_non_regression(self):
        user_id = 99
        async with self.sessions() as session:
            user = User(id=user_id, first_name="Борис", current_dialogue_id=1)
            ai_config = AIConfig(id=1, memory_mode=MEMORY_MODE_TOPIC)
            conv = AutomationConversationState(
                user_id=user_id,
                dialogue_id=1,
                topic_id=5,
                current_step="step_topic_mode",
                current_state_json='{"current_step": "step_topic_mode"}',
                metadata_json='{"topic_specific_key": "topic_val"}',
            )
            session.add_all([user, ai_config, conv])
            await session.commit()

        # In TOPIC mode, export uses existing per-conversation state schema
        export_callback = MagicMock()
        export_callback.from_user.id = 999999
        export_callback.data = f"run_metadata_export_merged_{user_id}"
        export_callback.message.answer_document = AsyncMock()
        export_callback.answer = AsyncMock()

        with patch("handlers.check_history_permission", AsyncMock(return_value=True)):
            await handlers.run_client_metadata_export(export_callback)

        doc = export_callback.message.answer_document.call_args[0][0]
        data = json.loads(doc.data.decode("utf-8"))
        self.assertEqual(data["export_type"], "consolidated_merged_metadata")
        self.assertEqual(len(data["dialogue_states"]), 1)
        st = data["dialogue_states"][0]
        self.assertEqual(st["dialogue_id"], 1)
        self.assertEqual(st["topic_id"], 5)
        self.assertEqual(st["current_step"], "step_topic_mode")
        self.assertEqual(st["metadata"], {"topic_specific_key": "topic_val"})

        # In RESET mode
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.memory_mode = MEMORY_MODE_RESET
            await session.commit()

        export_callback.reset_mock()
        with patch("handlers.check_history_permission", AsyncMock(return_value=True)):
            await handlers.run_client_metadata_export(export_callback)

        doc2 = export_callback.message.answer_document.call_args[0][0]
        data2 = json.loads(doc2.data.decode("utf-8"))
        self.assertEqual(data2["export_type"], "consolidated_merged_metadata")
        self.assertEqual(len(data2["dialogue_states"]), 1)
        st2 = data2["dialogue_states"][0]
        self.assertEqual(st2["current_step"], "step_topic_mode")
