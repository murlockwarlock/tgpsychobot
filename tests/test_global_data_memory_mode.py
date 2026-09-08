import json
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("BOT_TOKEN", "test")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import ai_integration
from ai_integration import AIServiceError
from automation_engine import (
    apply_service_data_blocks,
    build_runtime_automation_context,
    get_conversation_automation_state,
    get_dialogue_automation_state,
    get_or_lazy_init_dialogue_automation_state,
)
from automation_events import _execute_action
from database import (
    AIConfig,
    AILog,
    AutomationAction,
    AutomationConversationState,
    AutomationDialogueState,
    AutomationEvent,
    AutomationMetadataRecord,
    AutomationStepTransition,
    Base,
    FollowupCampaign,
    Message as DBMessage,
    Topic,
    User,
)
from followups import check_campaign_eligibility
from max_messenger_bot import ai as max_ai
from user_metadata import extract_service_data


class GlobalDataMemoryModeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        self._orig_tg_sessions = ai_integration.async_session_maker
        self._orig_max_sessions = max_ai.async_session_maker
        ai_integration.async_session_maker = self.sessions
        max_ai.async_session_maker = self.sessions

        async with self.sessions() as session:
            self.user = User(
                id=42,
                first_name="Тест",
                current_dialogue_id=1,
                current_topic_id=None,
                metadata_json="{}",
            )
            self.ai_config = AIConfig(
                id=1,
                provider="OpenAI",
                openai_api_key="sk-test",
                openai_model="gpt-5.6-terra",
                memory_mode="global",
                system_prompt="Тест",
            )
            self.topic_a = Topic(id=10, name="Тема A", is_active=True)
            self.topic_b = Topic(id=20, name="Тема B", is_active=True)
            session.add_all([self.user, self.ai_config, self.topic_a, self.topic_b])
            await session.commit()

    async def asyncTearDown(self):
        ai_integration.async_session_maker = self._orig_tg_sessions
        max_ai.async_session_maker = self._orig_max_sessions
        await self.engine.dispose()

    @staticmethod
    def _blocks(payload: str):
        _, blocks, invalid = extract_service_data(f"<DATA>{payload}</DATA>")
        assert invalid == 0
        return blocks

    # 1. Global metadata A -> B: metadata set in topic A is visible in topic B
    async def test_global_metadata_a_to_b(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=10,
                blocks=self._blocks('{"metadata": {"name": "Alice", "score": 100}}'),
                memory_mode="global",
            )
            await session.commit()

        async with self.sessions() as session:
            diag_state = await get_dialogue_automation_state(session, user_id=42, dialogue_id=1)
            self.assertIsNotNone(diag_state)
            meta = json.loads(diag_state.metadata_json)
            self.assertEqual(meta.get("name"), "Alice")
            self.assertEqual(meta.get("score"), 100)

            ctx_b = await build_runtime_automation_context(
                session, user_id=42, dialogue_id=1, topic_id=20, memory_mode="global"
            )
            self.assertIn('"name":"Alice"', ctx_b)
            self.assertIn('"score":100', ctx_b)

    # 2. Global metadata B -> A: metadata updates in topic B are visible in topic A
    async def test_global_metadata_b_to_a(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=10,
                blocks=self._blocks('{"metadata": {"city": "Berlin"}}'),
                memory_mode="global",
            )
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=20,
                blocks=self._blocks('{"metadata": {"city": "Paris", "lang": "fr"}}'),
                memory_mode="global",
            )
            await session.commit()

        async with self.sessions() as session:
            diag_state = await get_dialogue_automation_state(session, user_id=42, dialogue_id=1)
            meta = json.loads(diag_state.metadata_json)
            self.assertEqual(meta.get("city"), "Paris")
            self.assertEqual(meta.get("lang"), "fr")

            ctx_a = await build_runtime_automation_context(
                session, user_id=42, dialogue_id=1, topic_id=10, memory_mode="global"
            )
            self.assertIn('"city":"Paris"', ctx_a)
            self.assertIn('"lang":"fr"', ctx_a)

    # 3. current_step isolation: current_step in topic A does not leak to topic B
    async def test_current_step_isolation(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=10,
                blocks=self._blocks('{"current_state": {"current_step": "STEP_A"}}'),
                memory_mode="global",
            )
            await session.commit()

        async with self.sessions() as session:
            state_b = await get_conversation_automation_state(
                session, user_id=42, dialogue_id=1, topic_id=20
            )
            self.assertIsNone(state_b)

            state_a = await get_conversation_automation_state(
                session, user_id=42, dialogue_id=1, topic_id=10
            )
            self.assertIsNotNone(state_a)
            self.assertEqual(state_a.current_step, "STEP_A")

    # 4. events isolation: events in topic A are not transferred to topic B
    async def test_events_isolation(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=10,
                blocks=self._blocks('{"events": ["EV_A"]}'),
                memory_mode="global",
            )
            await session.commit()

        async with self.sessions() as session:
            evs_b = (
                await session.scalars(
                    select(AutomationEvent).where(
                        AutomationEvent.user_id == 42,
                        AutomationEvent.dialogue_id == 1,
                        AutomationEvent.topic_id == 20,
                    )
                )
            ).all()
            self.assertEqual(len(evs_b), 0)

            evs_a = (
                await session.scalars(
                    select(AutomationEvent).where(
                        AutomationEvent.user_id == 42,
                        AutomationEvent.dialogue_id == 1,
                        AutomationEvent.topic_id == 10,
                    )
                )
            ).all()
            self.assertEqual(len(evs_a), 1)
            self.assertEqual(evs_a[0].name, "EV_A")

    # 5. Topic state restore: returning to topic A restores its topic-local current_state
    async def test_topic_state_restore(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=10,
                blocks=self._blocks('{"current_state": {"current_step": "STEP_A", "notes": "local_a"}}'),
                memory_mode="global",
            )
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=20,
                blocks=self._blocks('{"current_state": {"current_step": "STEP_B", "notes": "local_b"}}'),
                memory_mode="global",
            )
            await session.commit()

        async with self.sessions() as session:
            state_a = await get_conversation_automation_state(
                session, user_id=42, dialogue_id=1, topic_id=10
            )
            self.assertIsNotNone(state_a)
            self.assertEqual(state_a.current_step, "STEP_A")
            self.assertEqual(json.loads(state_a.current_state_json).get("notes"), "local_a")

    # 6. save_mode=merge: mutates AutomationDialogueState.metadata_json
    async def test_save_mode_merge(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=10,
                blocks=self._blocks('{"metadata": {"foo": "bar"}, "save_mode": "merge"}'),
                memory_mode="global",
            )
            await session.commit()

        async with self.sessions() as session:
            diag_state = await session.scalar(
                select(AutomationDialogueState).where(
                    AutomationDialogueState.user_id == 42,
                    AutomationDialogueState.dialogue_id == 1,
                )
            )
            self.assertIsNotNone(diag_state)
            self.assertEqual(json.loads(diag_state.metadata_json), {"foo": "bar"})

    # 7. save_mode=snapshot: writes AutomationMetadataRecord without mutating active metadata
    async def test_save_mode_snapshot(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=10,
                blocks=self._blocks('{"metadata": {"active": "data"}, "save_mode": "merge"}'),
                memory_mode="global",
            )
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=10,
                blocks=self._blocks('{"metadata": {"snapshot_key": "snap_val"}, "save_mode": "snapshot"}'),
                memory_mode="global",
            )
            await session.commit()

        async with self.sessions() as session:
            records = (
                await session.scalars(
                    select(AutomationMetadataRecord).where(
                        AutomationMetadataRecord.user_id == 42,
                        AutomationMetadataRecord.dialogue_id == 1,
                        AutomationMetadataRecord.save_mode == "snapshot",
                    )
                )
            ).all()
            self.assertEqual(len(records), 1)
            self.assertEqual(json.loads(records[0].data_json).get("snapshot_key"), "snap_val")

            # Active global state must not contain snapshot_key
            diag_state = await get_dialogue_automation_state(session, user_id=42, dialogue_id=1)
            meta = json.loads(diag_state.metadata_json)
            self.assertNotIn("snapshot_key", meta)
            self.assertEqual(meta.get("active"), "data")

    # 8. Dialogue isolation: new dialogue starts a clean global metadata scope
    async def test_dialogue_isolation(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=10,
                blocks=self._blocks('{"metadata": {"session": "dialogue_1"}}'),
                memory_mode="global",
            )
            user.current_dialogue_id = 2
            await session.commit()

        async with self.sessions() as session:
            diag_state_2 = await get_dialogue_automation_state(session, user_id=42, dialogue_id=2)
            self.assertIsNone(diag_state_2)

            ctx_2 = await build_runtime_automation_context(
                session, user_id=42, dialogue_id=2, topic_id=10, memory_mode="global"
            )
            self.assertNotIn("dialogue_1", ctx_2)

    # 9. Topic mode unchanged: metadata is strictly topic-local in memory_mode=topic
    async def test_topic_mode_unchanged(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=10,
                blocks=self._blocks('{"metadata": {"topic_local": "val_a"}}'),
                memory_mode="topic",
            )
            await session.commit()

        async with self.sessions() as session:
            state_b = await get_conversation_automation_state(
                session, user_id=42, dialogue_id=1, topic_id=20
            )
            self.assertIsNone(state_b)

            ctx_b = await build_runtime_automation_context(
                session, user_id=42, dialogue_id=1, topic_id=20, memory_mode="topic"
            )
            self.assertNotIn("topic_local", ctx_b)

    # 10. Reset mode unchanged: metadata resets on topic switch in memory_mode=reset
    async def test_reset_mode_unchanged(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=10,
                blocks=self._blocks('{"metadata": {"temp": "reset_val"}}'),
                memory_mode="reset",
            )
            await session.commit()

        async with self.sessions() as session:
            state_b = await get_conversation_automation_state(
                session, user_id=42, dialogue_id=1, topic_id=20
            )
            self.assertIsNone(state_b)

            ctx_b = await build_runtime_automation_context(
                session, user_id=42, dialogue_id=1, topic_id=20, memory_mode="reset"
            )
            self.assertNotIn("reset_val", ctx_b)

    # 11. Legacy DATA compatibility: [DATA]...[/DATA] works seamlessly
    async def test_legacy_data_compatibility(self):
        _, blocks, invalid = extract_service_data('[DATA]{"metadata": {"legacy": true}}[/DATA]')
        self.assertEqual(invalid, 0)
        self.assertEqual(len(blocks), 1)
        async with self.sessions() as session:
            user = await session.get(User, 42)
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=10,
                blocks=blocks,
                memory_mode="global",
            )
            await session.commit()

        async with self.sessions() as session:
            user = await session.get(User, 42)
            self.assertIn("legacy", user.metadata_json)

    # 12. Migration initialization (Zero-Conflict Lazy Init)
    async def test_migration_initialization_zero_conflict_lazy_init(self):
        # Case A: active topic has state -> seeded from active
        async with self.sessions() as session:
            state_10 = AutomationConversationState(
                user_id=101, dialogue_id=1, topic_id=10, metadata_json=json.dumps({"key": "topic_10"})
            )
            state_20 = AutomationConversationState(
                user_id=101, dialogue_id=1, topic_id=20, metadata_json=json.dumps({"key": "topic_20"})
            )
            session.add_all([state_10, state_20])
            await session.commit()

        async with self.sessions() as session:
            diag_state = await get_or_lazy_init_dialogue_automation_state(
                session, user_id=101, dialogue_id=1, active_topic_id=10
            )
            self.assertEqual(json.loads(diag_state.metadata_json), {"key": "topic_10"})

        # Case B: active has no state, exactly one other non-empty topic exists -> seeded from that topic
        async with self.sessions() as session:
            state_30 = AutomationConversationState(
                user_id=102, dialogue_id=1, topic_id=30, metadata_json=json.dumps({"single": "val"})
            )
            session.add(state_30)
            await session.commit()

        async with self.sessions() as session:
            diag_state = await get_or_lazy_init_dialogue_automation_state(
                session, user_id=102, dialogue_id=1, active_topic_id=999
            )
            self.assertEqual(json.loads(diag_state.metadata_json), {"single": "val"})

        # Case C: active has no state, multiple conflicting non-empty topics -> seeded with {}
        async with self.sessions() as session:
            s1 = AutomationConversationState(
                user_id=103, dialogue_id=1, topic_id=10, metadata_json=json.dumps({"diff": 1})
            )
            s2 = AutomationConversationState(
                user_id=103, dialogue_id=1, topic_id=20, metadata_json=json.dumps({"diff": 2})
            )
            session.add_all([s1, s2])
            await session.commit()

        async with self.sessions() as session:
            diag_state = await get_or_lazy_init_dialogue_automation_state(
                session, user_id=103, dialogue_id=1, active_topic_id=None
            )
            self.assertEqual(json.loads(diag_state.metadata_json), {})

    # 13. Followup metadata condition: in Global checks AutomationDialogueState
    async def test_followup_metadata_condition(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=10,
                blocks=self._blocks('{"metadata": {"status": "paid"}}'),
                memory_mode="global",
            )
            await session.commit()

        async with self.sessions() as session:
            campaign = FollowupCampaign(
                name="c1",
                all_topics=True,
                metadata_field_path="status",
                metadata_operator="equals",
                metadata_expected_value="paid",
            )
            elig = await check_campaign_eligibility(
                session, campaign, user_id=42, dialogue_id=1, topic_id=20
            )
            self.assertTrue(elig.eligible)

    # 14. Followup step condition: checking step condition remains strictly topic-local
    async def test_followup_step_condition(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=10,
                blocks=self._blocks('{"current_state": {"current_step": "STEP_ONBOARDING"}}'),
                memory_mode="global",
            )
            await session.commit()

        async with self.sessions() as session:
            campaign = FollowupCampaign(
                name="c2",
                all_topics=True,
                stage_mode="selected",
                stage_values="STEP_ONBOARDING",
                stage_include_unset=False,
            )
            # True in topic 10
            elig_10 = await check_campaign_eligibility(
                session, campaign, user_id=42, dialogue_id=1, topic_id=10
            )
            self.assertTrue(elig_10.eligible)
            # False in topic 20
            elig_20 = await check_campaign_eligibility(
                session, campaign, user_id=42, dialogue_id=1, topic_id=20
            )
            self.assertFalse(elig_20.eligible)

    # 15. Stop events contract: stop campaign events work topic-locally
    async def test_stop_events_contract(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=10,
                blocks=self._blocks('{"events": ["STOP_C1"]}'),
                memory_mode="global",
            )
            await session.commit()

        async with self.sessions() as session:
            campaign = FollowupCampaign(
                name="c3",
                all_topics=True,
                stop_events="STOP_C1",
            )
            # In topic 10, STOP_C1 occurred -> ineligible
            elig_10 = await check_campaign_eligibility(
                session, campaign, user_id=42, dialogue_id=1, topic_id=10
            )
            self.assertFalse(elig_10.eligible)
            # In topic 20, STOP_C1 did not occur -> eligible
            elig_20 = await check_campaign_eligibility(
                session, campaign, user_id=42, dialogue_id=1, topic_id=20
            )
            self.assertTrue(elig_20.eligible)

    # 16. Event metadata snapshot: metadata snapshot in AutomationEvent captures active global dialogue metadata
    async def test_event_metadata_snapshot(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=10,
                blocks=self._blocks('{"metadata": {"global_flag": 123}}'),
                memory_mode="global",
            )
            # Record an event in topic 20
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=20,
                blocks=self._blocks('{"events": ["TEST_EVENT"]}'),
                memory_mode="global",
            )
            await session.commit()

        async with self.sessions() as session:
            event = await session.scalar(
                select(AutomationEvent).where(
                    AutomationEvent.user_id == 42,
                    AutomationEvent.name == "TEST_EVENT",
                )
            )
            self.assertIsNotNone(event)
            meta = json.loads(event.metadata_json)
            self.assertEqual(meta.get("global_flag"), 123)

    # 17. save_metadata action: automation event action save_metadata in global mode updates AutomationDialogueState
    async def test_save_metadata_action(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            event = AutomationEvent(
                id=1,
                user_id=42,
                dialogue_id=1,
                topic_id=10,
                name="TRIGGER_EVENT",
                metadata_json=json.dumps({"global_key": "val1"}),
            )
            action = AutomationAction(
                id=1,
                handler_id=1,
                action_type="save_metadata",
                metadata_json=json.dumps({"action_updated": True}),
            )
            await get_or_lazy_init_dialogue_automation_state(
                session, user_id=42, dialogue_id=1, active_topic_id=10
            )
            await _execute_action(
                session,
                bot=None,
                event=event,
                handler=None,
                action=action,
                user=user,
            )
            await session.commit()

        async with self.sessions() as session:
            diag_state = await session.scalar(
                select(AutomationDialogueState).where(
                    AutomationDialogueState.user_id == 42,
                    AutomationDialogueState.dialogue_id == 1,
                )
            )
            self.assertIsNotNone(diag_state)
            self.assertEqual(json.loads(diag_state.metadata_json).get("action_updated"), True)

    # 18. Atomic rollback on step failure: failure in step transition rolls back AI transaction
    async def test_atomic_rollback_on_step_failure(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            orig_add = session.add
            def fail_on_transition(instance):
                if isinstance(instance, AutomationStepTransition):
                    raise RuntimeError("DB step transition error")
                orig_add(instance)

            with patch.object(session, "add", side_effect=fail_on_transition):
                with self.assertRaises(RuntimeError):
                    await apply_service_data_blocks(
                        session,
                        user=user,
                        dialogue_id=1,
                        topic_id=10,
                        blocks=self._blocks('{"current_state": {"current_step": "FAIL_STEP"}}'),
                        memory_mode="global",
                    )
                await session.rollback()

        async with self.sessions() as session:
            state = await get_conversation_automation_state(
                session, user_id=42, dialogue_id=1, topic_id=10
            )
            self.assertIsNone(state)

    # 19. TG valid DATA + AILog rollback together on apply failure
    async def test_tg_valid_data_and_ai_log_rollback_together_on_apply_failure(self):
        with patch("ai_integration.apply_service_data_blocks", side_effect=RuntimeError("Simulated data error")):
            with patch("ai_integration._call_openai_api", new_callable=AsyncMock) as mock_call:
                mock_call.return_value = 'Ответ<DATA>{"metadata":{"bad":1}}</DATA>'
                with self.assertRaises(AIServiceError):
                    await ai_integration.get_ai_response(
                        user_id=42,
                        user_prompt="Тестовый вопрос",
                        user_name="Тест",
                        user_gender="unknown",
                        dialogue_id_override=1,
                        topic_id_override=10,
                    )

        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog))).all()
            self.assertEqual(len(logs), 0)
            diag_state = await session.scalar(
                select(AutomationDialogueState).where(AutomationDialogueState.user_id == 42)
            )
            if diag_state is not None:
                self.assertNotIn("bad", json.loads(diag_state.metadata_json))

    # 20. MAX valid DATA + AILog rollback together on apply failure
    async def test_max_valid_data_and_ai_log_rollback_together_on_apply_failure(self):
        with patch("max_messenger_bot.ai.apply_service_data_blocks", side_effect=RuntimeError("Simulated MAX data error")):
            with patch("max_messenger_bot.ai._call_openai", new_callable=AsyncMock) as mock_call:
                mock_call.return_value = 'Ответ MAX<DATA>{"metadata":{"bad":2}}</DATA>'
                with self.assertRaises(max_ai.AIServiceError):
                    await max_ai.get_ai_response(
                        user_id=42,
                        user_prompt="Вопрос MAX",
                        dialogue_id_override=1,
                        topic_id_override=10,
                    )

        async with self.sessions() as session:
            logs = (await session.scalars(select(AILog))).all()
            self.assertEqual(len(logs), 0)
            diag_state = await session.scalar(
                select(AutomationDialogueState).where(AutomationDialogueState.user_id == 42)
            )
            if diag_state is not None:
                self.assertNotIn("bad", json.loads(diag_state.metadata_json))

    # 21. No assistant message on AI-layer DATA failure
    async def test_no_assistant_message_on_ai_layer_data_failure(self):
        with patch("ai_integration.apply_service_data_blocks", side_effect=RuntimeError("Data block fail")):
            with patch("ai_integration._call_openai_api", new_callable=AsyncMock) as mock_call:
                mock_call.return_value = 'Ответ<DATA>{"bad":1}</DATA>'
                with self.assertRaises(AIServiceError):
                    await ai_integration.get_ai_response(
                        user_id=42,
                        user_prompt="Привет",
                        user_name="Тест",
                        user_gender="unknown",
                        dialogue_id_override=1,
                        topic_id_override=10,
                    )

        async with self.sessions() as session:
            messages = (await session.scalars(select(DBMessage).where(DBMessage.role == "assistant"))).all()
            self.assertEqual(len(messages), 0)

    # 22. Admin Global view: renders separate sections for global metadata and topic state
    async def test_admin_global_view(self):
        async with self.sessions() as session:
            user = await session.get(User, 42)
            await apply_service_data_blocks(
                session,
                user=user,
                dialogue_id=1,
                topic_id=10,
                blocks=self._blocks('{"metadata": {"admin_global": "xyz"}, "current_state": {"step": "admin_step"}}'),
                memory_mode="global",
            )
            await session.commit()

        async with self.sessions() as session:
            diag_state = await get_dialogue_automation_state(session, user_id=42, dialogue_id=1)
            self.assertIsNotNone(diag_state)
            self.assertEqual(json.loads(diag_state.metadata_json).get("admin_global"), "xyz")

            state_10 = await get_conversation_automation_state(session, user_id=42, dialogue_id=1, topic_id=10)
            self.assertIsNotNone(state_10)
            self.assertEqual(json.loads(state_10.current_state_json).get("step"), "admin_step")

            # Runtime context contains both
            ctx = await build_runtime_automation_context(
                session, user_id=42, dialogue_id=1, topic_id=10, memory_mode="global"
            )
            self.assertIn("admin_global", ctx)
            self.assertIn("admin_step", ctx)
