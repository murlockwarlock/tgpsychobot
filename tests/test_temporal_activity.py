import asyncio
import json
import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

def _mock_openai_completion(text: str):
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = text
    mock_resp.choices = [mock_choice]
    return mock_resp


os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("BOT_TOKEN", "test")

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import ai_integration
from ai_integration import AIServiceError
from ai_request_builder import (
    ActivityTracker,
    build_conversational_request_layout,
    build_temporal_activity_context,
    get_user_ai_activity_gaps,
    upsert_user_ai_activity,
)
from database import (
    AIConfig,
    AILog,
    Base,
    Message as DBMessage,
    Topic,
    User,
    UserAIActivity,
)
from max_messenger_bot import ai as max_ai


class TemporalActivityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        self._orig_tg_sessions = ai_integration.async_session_maker
        self._orig_max_sessions = max_ai.async_session_maker
        ai_integration.async_session_maker = self.sessions
        max_ai.async_session_maker = self.sessions

        async with self.sessions() as session:
            self.user = User(
                id=3001,
                first_name="Ольга",
                current_dialogue_id=1,
                current_topic_id=10,
                metadata_json="{}",
            )
            self.topic_a = Topic(id=10, name="Тема 10", is_active=True, system_prompt="Тема 10")
            self.topic_b = Topic(id=20, name="Тема 20", is_active=True, system_prompt="Тема 20")
            self.ai_config = AIConfig(
                id=1,
                provider="OpenAI",
                openai_api_key="sk-test",
                openai_model="gpt-5.6-terra",
                system_prompt="Психолог",
                memory_mode="global",
            )
            session.add_all([self.user, self.topic_a, self.topic_b, self.ai_config])
            await session.commit()

    async def asyncTearDown(self):
        ai_integration.async_session_maker = self._orig_tg_sessions
        max_ai.async_session_maker = self._orig_max_sessions
        await self.engine.dispose()

    # 85. First AI request ever: minutes_since_last_message = 0 and minutes_since_last_visit = 0
    async def test_first_ai_request_ever(self):
        async with self.sessions() as session:
            visit_gap, msg_gap = await get_user_ai_activity_gaps(session, user_id=3001, topic_id=10)
            self.assertEqual(visit_gap, 0)
            self.assertEqual(msg_gap, 0)

    # 86. Same topic gap: Topic A at 10:00, Topic A at 11:00 -> message=60, visit=60
    async def test_same_topic_gap(self):
        t1 = datetime(2026, 1, 1, 10, 0, 0)
        t2 = datetime(2026, 1, 1, 11, 0, 0)
        async with self.sessions() as session:
            tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t1)
            await tracker.mark_outbound_attempt_once()

        async with self.sessions() as session:
            visit_gap, msg_gap = await get_user_ai_activity_gaps(session, user_id=3001, topic_id=10, now=t2)
            self.assertEqual(visit_gap, 60)
            self.assertEqual(msg_gap, 60)

    # 87. Cross-topic gap: Topic A at 10:00, Topic B at 10:40, Topic A at 11:00 -> message=20, visit=60
    async def test_cross_topic_gap(self):
        t_a = datetime(2026, 1, 1, 10, 0, 0)
        t_b = datetime(2026, 1, 1, 10, 40, 0)
        t_now = datetime(2026, 1, 1, 11, 0, 0)

        async with self.sessions() as session:
            tracker_a = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t_a)
            await tracker_a.mark_outbound_attempt_once()
            tracker_b = ActivityTracker(self.sessions, user_id=3001, topic_id=20, request_time=t_b)
            await tracker_b.mark_outbound_attempt_once()

        async with self.sessions() as session:
            visit_gap, msg_gap = await get_user_ai_activity_gaps(session, user_id=3001, topic_id=10, now=t_now)
            self.assertEqual(visit_gap, 60)
            self.assertEqual(msg_gap, 20)

    # 88. First entry auto_start=False leaves activity unchanged
    async def test_first_entry_auto_start_false_leaves_activity_unchanged(self):
        async with self.sessions() as session:
            # Switching topic without AI does not call activity tracker
            records = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(records), 0)

    # 89. Subsequent AI request after auto_start=False sees gap from prior AI activity
    async def test_subsequent_ai_request_after_auto_start_false_sees_gap_from_prior_ai_activity(self):
        t0 = datetime(2026, 1, 1, 9, 0, 0)
        t_entry = datetime(2026, 1, 1, 10, 0, 0)
        t_query = datetime(2026, 1, 1, 11, 0, 0)

        # Prior activity in Topic 10 at 9:00
        async with self.sessions() as session:
            tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t0)
            await tracker.mark_outbound_attempt_once()

        # At 10:00 user enters topic with auto_start=False (no AI call)
        # At 11:00 user asks a question
        async with self.sessions() as session:
            visit_gap, msg_gap = await get_user_ai_activity_gaps(session, user_id=3001, topic_id=10, now=t_query)
            # Both gaps must be 120 minutes (from 9:00, not from 10:00 entry)
            self.assertEqual(visit_gap, 120)
            self.assertEqual(msg_gap, 120)

    # 90. First entry auto_start=True updates activity on outbound
    async def test_first_entry_auto_start_true_updates_activity_on_outbound(self):
        t_start = datetime(2026, 1, 1, 10, 0, 0)
        async with self.sessions() as session:
            tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t_start)
            await tracker.mark_outbound_attempt_once()

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertTrue(len(acts) >= 2)
            scopes = {a.scope_key for a in acts}
            self.assertIn("global", scopes)
            self.assertIn("topic:10", scopes)

    # 91. Resume kickoff updates activity on outbound
    async def test_resume_kickoff_updates_activity_on_outbound(self):
        t_resume = datetime(2026, 1, 1, 12, 0, 0)
        async with self.sessions() as session:
            tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t_resume)
            await tracker.mark_outbound_attempt_once()

        async with self.sessions() as session:
            row = await session.scalar(
                select(UserAIActivity).where(UserAIActivity.user_id == 3001, UserAIActivity.scope_key == "topic:10")
            )
            self.assertEqual(row.last_request_at, t_resume)

    # 92. Pre-provider blocked kickoff does NOT update activity
    async def test_pre_provider_blocked_kickoff_does_not_update_activity(self):
        tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10)
        # Blocked before outbound network call, tracker.mark_outbound_attempt_once() is NOT called
        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 0)

    # 93. Idempotent navigation does NOT update activity
    async def test_idempotent_navigation_does_not_update_activity(self):
        # When clicking already-current topic, no AI kickoff runs and activity is not updated
        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 0)

    # 94. Navigation without AI does not change timers
    async def test_navigation_without_ai_does_not_change_timers(self):
        t0 = datetime(2026, 1, 1, 10, 0, 0)
        async with self.sessions() as session:
            tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t0)
            await tracker.mark_outbound_attempt_once()

        # Switch to Topic B without AI
        t_now = datetime(2026, 1, 1, 10, 30, 0)
        async with self.sessions() as session:
            visit_gap, msg_gap = await get_user_ai_activity_gaps(session, user_id=3001, topic_id=10, now=t_now)
            self.assertEqual(visit_gap, 30)
            self.assertEqual(msg_gap, 30)

    # 95. Button without AI does not change timers
    async def test_button_without_ai_does_not_change_timers(self):
        t0 = datetime(2026, 1, 1, 10, 0, 0)
        async with self.sessions() as session:
            tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t0)
            await tracker.mark_outbound_attempt_once()

        # Non-AI button clicked at 10:15
        t_now = datetime(2026, 1, 1, 10, 45, 0)
        async with self.sessions() as session:
            visit_gap, msg_gap = await get_user_ai_activity_gaps(session, user_id=3001, topic_id=10, now=t_now)
            self.assertEqual(visit_gap, 45)

    # 96. AI-triggering button updates timers
    async def test_ai_triggering_button_updates_timers(self):
        t_btn = datetime(2026, 1, 1, 10, 15, 0)
        async with self.sessions() as session:
            tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t_btn)
            await tracker.mark_outbound_attempt_once()

        async with self.sessions() as session:
            visit_gap, msg_gap = await get_user_ai_activity_gaps(
                session, user_id=3001, topic_id=10, now=datetime(2026, 1, 1, 10, 30, 0)
            )
            self.assertEqual(visit_gap, 15)

    # 97. Local provider validation failure no update: error BEFORE network does not record activity
    async def test_local_provider_validation_failure_no_update(self):
        with patch("ai_integration.ensure_model_available", side_effect=RuntimeError("Invalid model")):
            tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10)
            with self.assertRaises(AIServiceError):
                await ai_integration.get_ai_response(
                    user_id=3001,
                    user_prompt="Вопрос",
                    user_name="Ольга",
                    user_gender="female",
                    activity_tracker=tracker,
                )

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 0)

    # 98. Provider network timeout updates activity: network started -> activity marked
    async def test_provider_network_timeout_updates_activity(self):
        tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10)
        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=TimeoutError("Network timeout")):
            with self.assertRaises(AIServiceError):
                await ai_integration.get_ai_response(
                    user_id=3001,
                    user_prompt="Вопрос",
                    user_name="Ольга",
                    user_gender="female",
                    activity_tracker=tracker,
                )

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertTrue(len(acts) >= 2)

    # 99. Primary outbound + fallback marks activity once
    async def test_primary_outbound_fallback_marks_activity_once(self):
        t_req = datetime(2026, 1, 1, 10, 0, 0)
        tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t_req)
        # First outbound attempt (primary)
        await tracker.mark_outbound_attempt_once()
        # Second outbound attempt (fallback)
        await tracker.mark_outbound_attempt_once()

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            # Exactly 2 records (1 global + 1 topic), not 4
            self.assertEqual(len(acts), 2)

    # 100. Fallback marks activity when primary failed locally
    async def test_fallback_marks_activity_when_primary_failed_locally(self):
        tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10)
        # Primary failed before mark_outbound_attempt_once()
        # Fallback starts network call:
        await tracker.mark_outbound_attempt_once()

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 2)

    # 101. Internal provider retry marks activity once
    async def test_internal_provider_retry_marks_activity_once(self):
        tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10)
        for _ in range(3):
            await tracker.mark_outbound_attempt_once()

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 2)

    # 102. Universal test preliminary does not update activity
    async def test_universal_test_preliminary_does_not_update_activity(self):
        with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _mock_openai_completion("Preliminary test analysis")
            # In universal test, preliminary direct call passes track_user_activity=False
            await ai_integration.get_ai_response(
                user_id=3001,
                user_prompt="Preliminary direct prompt",
                user_name="Ольга",
                user_gender="female",
                track_user_activity=False,
            )

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 0)

    # 103. Universal test final handoff updates activity once
    async def test_universal_test_final_handoff_updates_activity_once(self):
        with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _mock_openai_completion("Final test summary")
            await ai_integration.get_ai_response(
                user_id=3001,
                user_prompt="Final handoff prompt",
                user_name="Ольга",
                user_gender="female",
                track_user_activity=True,
            )

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 2)

    # 104. Universal test preliminary + final updates activity exactly once
    async def test_universal_test_preliminary_final_updates_activity_exactly_once(self):
        with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _mock_openai_completion("AI result")
            # Preliminary: False
            await ai_integration.get_ai_response(
                user_id=3001, user_prompt="Prelim", user_name="Ольга", user_gender="female", track_user_activity=False
            )
            # Final: True
            await ai_integration.get_ai_response(
                user_id=3001, user_prompt="Final", user_name="Ольга", user_gender="female", track_user_activity=True
            )

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 2)

    # 105. Final request does not receive artificial zero gap (uses step snapshotted gaps)
    async def test_final_request_does_not_receive_artificial_zero_gap(self):
        # Snapshot gaps before chain
        t0 = datetime(2026, 1, 1, 10, 0, 0)
        async with self.sessions() as session:
            tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t0)
            await tracker.mark_outbound_attempt_once()

        t_step = datetime(2026, 1, 1, 11, 30, 0)
        async with self.sessions() as session:
            snap_visit, snap_msg = await get_user_ai_activity_gaps(session, user_id=3001, topic_id=10, now=t_step)
            self.assertEqual(snap_visit, 90)
            self.assertEqual(snap_msg, 90)

        # Passing snapshotted gaps to final handoff layout
        ctx = build_temporal_activity_context(snap_visit, snap_msg)
        self.assertIn("minutes_since_last_visit: 90", ctx)
        self.assertIn("minutes_since_last_message: 90", ctx)

    # 106. Direct prompt configured as final result updates activity once
    async def test_direct_prompt_configured_as_final_result_updates_activity_once(self):
        with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _mock_openai_completion("Direct final answer")
            await ai_integration.get_ai_response(
                user_id=3001, user_prompt="Direct prompt", user_name="Ольга", user_gender="female", track_user_activity=True
            )

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 2)

    # 107. Followup receives temporal values in payload
    async def test_followup_receives_temporal_values_in_payload(self):
        ctx = build_temporal_activity_context(minutes_since_last_visit=45, minutes_since_last_message=120)
        self.assertIn("minutes_since_last_visit: 45", ctx)
        self.assertIn("minutes_since_last_message: 120", ctx)

    # 108. Followup does NOT update activity timestamps
    async def test_followup_does_not_update_activity_timestamps(self):
        with patch("ai_integration._call_openai_api", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "Followup message"
            await ai_integration.get_ai_response(
                user_id=3001, user_prompt="Followup", user_name="Ольга", user_gender="female",
                track_user_activity=False, persist_service_data=False
            )

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 0)

    # 109. Repeated followups do not change user inactivity gap
    async def test_repeated_followups_do_not_change_user_inactivity_gap(self):
        t_user = datetime(2026, 1, 1, 10, 0, 0)
        async with self.sessions() as session:
            tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t_user)
            await tracker.mark_outbound_attempt_once()

        # Followup 1 at 12:00, Followup 2 at 14:00 (both track_user_activity=False)
        t_now = datetime(2026, 1, 1, 15, 0, 0)
        async with self.sessions() as session:
            visit_gap, msg_gap = await get_user_ai_activity_gaps(session, user_id=3001, topic_id=10, now=t_now)
            # Gap must be 5 hours (300 minutes), not reset by followups
            self.assertEqual(visit_gap, 300)
            self.assertEqual(msg_gap, 300)

    # 110. Followup keeps persist_service_data=False
    async def test_followup_keeps_persist_service_data_false(self):
        with patch("ai_integration._call_openai_api", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = 'Followup<DATA>{"metadata":{"bad":1}}</DATA>'
            await ai_integration.get_ai_response(
                user_id=3001, user_prompt="Followup", user_name="Ольга", user_gender="female",
                track_user_activity=False, persist_service_data=False
            )

        async with self.sessions() as session:
            user = await session.get(User, 3001)
            self.assertNotIn("bad", user.metadata_json)

    # 111. Next user request sees gap from previous USER activity
    async def test_next_user_request_sees_gap_from_previous_user_activity(self):
        t_user1 = datetime(2026, 1, 1, 10, 0, 0)
        async with self.sessions() as session:
            tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t_user1)
            await tracker.mark_outbound_attempt_once()

        t_user2 = datetime(2026, 1, 1, 12, 30, 0)
        async with self.sessions() as session:
            visit_gap, msg_gap = await get_user_ai_activity_gaps(session, user_id=3001, topic_id=10, now=t_user2)
            self.assertEqual(visit_gap, 150)
            self.assertEqual(msg_gap, 150)

    # 112. No self-zeroing: current request uses prior timestamps and doesn't count itself
    async def test_no_self_zeroing(self):
        t0 = datetime(2026, 1, 1, 10, 0, 0)
        async with self.sessions() as session:
            tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t0)
            await tracker.mark_outbound_attempt_once()

        # Next request at 11:00 reads gaps BEFORE updating activity
        t_req = datetime(2026, 1, 1, 11, 0, 0)
        async with self.sessions() as session:
            visit_gap, msg_gap = await get_user_ai_activity_gaps(session, user_id=3001, topic_id=10, now=t_req)
            self.assertEqual(visit_gap, 60)
            # Now tracker updates to t_req
            tracker2 = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t_req)
            await tracker2.mark_outbound_attempt_once()

    # 113. New dialogue retains global timer
    async def test_new_dialogue_retains_global_timer(self):
        t0 = datetime(2026, 1, 1, 10, 0, 0)
        async with self.sessions() as session:
            tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t0)
            await tracker.mark_outbound_attempt_once()
            user = await session.get(User, 3001)
            user.current_dialogue_id = 2
            await session.commit()

        t_now = datetime(2026, 1, 1, 11, 0, 0)
        async with self.sessions() as session:
            _, msg_gap = await get_user_ai_activity_gaps(session, user_id=3001, topic_id=10, now=t_now)
            self.assertEqual(msg_gap, 60)

    # 114. New dialogue retains topic activity
    async def test_new_dialogue_retains_topic_activity(self):
        t0 = datetime(2026, 1, 1, 10, 0, 0)
        async with self.sessions() as session:
            tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t0)
            await tracker.mark_outbound_attempt_once()
            user = await session.get(User, 3001)
            user.current_dialogue_id = 2
            await session.commit()

        t_now = datetime(2026, 1, 1, 11, 0, 0)
        async with self.sessions() as session:
            visit_gap, _ = await get_user_ai_activity_gaps(session, user_id=3001, topic_id=10, now=t_now)
            self.assertEqual(visit_gap, 60)

    # 115. Main independent scope
    async def test_main_independent_scope(self):
        t_main = datetime(2026, 1, 1, 10, 0, 0)
        async with self.sessions() as session:
            tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=None, request_time=t_main)
            await tracker.mark_outbound_attempt_once()

        async with self.sessions() as session:
            row = await session.scalar(
                select(UserAIActivity).where(UserAIActivity.user_id == 3001, UserAIActivity.scope_key == "main")
            )
            self.assertIsNotNone(row)
            self.assertEqual(row.last_request_at, t_main)

    # 116. Topic A vs Topic B independence
    async def test_topic_a_vs_topic_b_independence(self):
        t_a = datetime(2026, 1, 1, 10, 0, 0)
        async with self.sessions() as session:
            tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t_a)
            await tracker.mark_outbound_attempt_once()

        async with self.sessions() as session:
            row_b = await session.scalar(
                select(UserAIActivity).where(UserAIActivity.user_id == 3001, UserAIActivity.scope_key == "topic:20")
            )
            self.assertIsNone(row_b)

    # 117. Minutes floor: 59s -> 0m, 61s -> 1m
    async def test_minutes_floor(self):
        t0 = datetime(2026, 1, 1, 10, 0, 0)
        async with self.sessions() as session:
            tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t0)
            await tracker.mark_outbound_attempt_once()

        async with self.sessions() as session:
            # 59 seconds later -> 0 minutes
            g59, _ = await get_user_ai_activity_gaps(session, user_id=3001, topic_id=10, now=t0 + timedelta(seconds=59))
            self.assertEqual(g59, 0)
            # 61 seconds later -> 1 minute
            g61, _ = await get_user_ai_activity_gaps(session, user_id=3001, topic_id=10, now=t0 + timedelta(seconds=61))
            self.assertEqual(g61, 1)

    # 118. Clock anomaly protection: negative timedelta clamped to 0
    async def test_clock_anomaly_protection(self):
        t0 = datetime(2026, 1, 1, 10, 0, 0)
        async with self.sessions() as session:
            tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t0)
            await tracker.mark_outbound_attempt_once()

        async with self.sessions() as session:
            # If system clock shifts backward into the past
            g_past, _ = await get_user_ai_activity_gaps(session, user_id=3001, topic_id=10, now=t0 - timedelta(hours=1))
            self.assertEqual(g_past, 0)

    # 119. Vision request counts: conversational photo request updates activity
    async def test_vision_request_counts(self):
        with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _mock_openai_completion("Vision description")
            await ai_integration.get_ai_response(
                user_id=3001, user_prompt="Photo prompt", user_name="Ольга", user_gender="female",
                track_user_activity=True
            )

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 2)

    # 120. Transcription does NOT update activity
    async def test_transcription_does_not_update_activity(self):
        # Audio transcription endpoint (Deepgram/Whisper) operates without ActivityTracker
        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 0)

    # 121. Standalone image generation does NOT update activity
    async def test_standalone_image_generation_does_not_update_activity(self):
        # GEN_IMG operates independently of conversational UserAIActivity
        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 0)

    # 122. Embedding / Chroma operations do NOT update activity
    async def test_embedding_chroma_operations_do_not_update_activity(self):
        # Knowledge base vector searches do not track activity
        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 0)

    # 123. Admin AI utility does NOT update activity
    async def test_admin_ai_utility_does_not_update_activity(self):
        # Admin utility LLM invocation does not pass or track user activity
        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 0)

    # 124. Concurrent first activity-row creation is safe
    async def test_concurrent_first_activity_row_creation_is_safe(self):
        from pathlib import Path
        db_path = Path(__file__).parent / ".test_concurrent_activity.db"
        if db_path.exists():
            db_path.unlink()
        file_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        file_sessions = async_sessionmaker(file_engine, expire_on_commit=False)
        try:
            async with file_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            t0 = datetime(2026, 1, 1, 10, 0, 0)
            async def insert_activity():
                async with file_sessions() as session:
                    await upsert_user_ai_activity(session, user_id=3001, scope_key="topic:10", request_time=t0)
                    await session.commit()

            # Run 5 concurrent inserts across separate connections for the same new user and scope
            results = await asyncio.gather(
                insert_activity(),
                insert_activity(),
                insert_activity(),
                insert_activity(),
                insert_activity(),
                return_exceptions=True,
            )
            for r in results:
                self.assertFalse(isinstance(r, Exception), f"Concurrent insert raised: {r}")

            async with file_sessions() as session:
                rows = (
                    await session.scalars(
                        select(UserAIActivity).where(UserAIActivity.user_id == 3001, UserAIActivity.scope_key == "topic:10")
                    )
                ).all()
                self.assertEqual(len(rows), 1)
        finally:
            await file_engine.dispose()
            if db_path.exists():
                db_path.unlink()

    # 125. ActivityTracker persistence failure retry
    async def test_activity_tracker_persistence_failure_retries(self):
        import ai_request_builder

        tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10)
        self.assertFalse(tracker._marked)

        # Mock upsert_user_ai_activity to fail on first attempt
        call_count = 0
        orig_upsert = ai_request_builder.upsert_user_ai_activity

        async def failing_upsert(session, user_id, scope_key, request_time):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("DB connection lost")
            return await orig_upsert(session, user_id=user_id, scope_key=scope_key, request_time=request_time)

        with patch("ai_request_builder.upsert_user_ai_activity", side_effect=failing_upsert):
            # First attempt fails
            with self.assertRaises(RuntimeError):
                await tracker.mark_outbound_attempt_once()
            # _marked must remain False so retry is allowed
            self.assertFalse(tracker._marked)

            # Second attempt succeeds
            await tracker.mark_outbound_attempt_once()
            self.assertTrue(tracker._marked)

            # Third attempt is a no-op
            await tracker.mark_outbound_attempt_once()
            self.assertTrue(tracker._marked)

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 2)

    # 126. MAX activity: missing API key produces no network call and no activity
    async def test_max_activity_missing_key_no_activity(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.openai_api_key = ""
            cfg.allow_fallback = False
            await session.commit()

        with patch("openai.resources.chat.completions.AsyncCompletions.create") as mock_call:
            with self.assertRaises(Exception):
                await max_ai.get_ai_response(
                    3001,
                    "Привет",
                    track_user_activity=True,
                )
            mock_call.assert_not_called()

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 0)

    # 127. MAX activity: local invalid model produces no network call and no activity
    async def test_max_activity_invalid_model_no_activity(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.openai_api_key = "sk-valid"
            cfg.openai_model = "unsupported-model-xyz"
            cfg.allow_fallback = False
            await session.commit()

        with patch("openai.resources.chat.completions.AsyncCompletions.create") as mock_call:
            with self.assertRaises(Exception):
                await max_ai.get_ai_response(
                    3001,
                    "Привет",
                    track_user_activity=True,
                )
            mock_call.assert_not_called()

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 0)

    # 128. MAX activity: network timeout records activity
    async def test_max_activity_network_timeout_records_activity(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "OpenAI"
            cfg.openai_api_key = "sk-valid"
            cfg.openai_model = "gpt-5.6-terra"
            cfg.allow_fallback = False
            await session.commit()

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=TimeoutError("Network timeout")):
            with self.assertRaises(Exception):
                await max_ai.get_ai_response(
                    3001,
                    "Привет",
                    track_user_activity=True,
                )

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertTrue(len(acts) >= 2)

    # 129. MAX activity: fallback after local failure marks activity on fallback
    async def test_max_activity_fallback_after_local_failure_marks_activity(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "OpenAI"
            cfg.openai_api_key = ""  # Primary fails locally due to missing key
            cfg.allow_fallback = True
            cfg.fallback_provider = "Claude"
            cfg.claude_api_key = "sk-claude-test"
            cfg.fallback_model = "claude-sonnet-5"
            await session.commit()

        with patch("anthropic.resources.messages.AsyncMessages.create", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = MagicMock(content=[MagicMock(text="Ответ от Claude")])
            resp = await max_ai.get_ai_response(
                3001,
                "Привет",
                track_user_activity=True,
            )
            self.assertEqual(resp, "Ответ от Claude")
            mock_claude.assert_called_once()

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 2)

    # 130. MAX activity: fallback after network failure marks activity only once
    async def test_max_activity_fallback_after_network_failure_marks_only_once(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "OpenAI"
            cfg.openai_api_key = "sk-valid"
            cfg.openai_model = "gpt-5.6-terra"
            cfg.allow_fallback = True
            cfg.fallback_provider = "Claude"
            cfg.claude_api_key = "sk-claude-test"
            cfg.fallback_model = "claude-sonnet-5"
            await session.commit()

        # Primary fails with network timeout inside _call_openai
        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=TimeoutError("Primary network timeout")), \
             patch("max_messenger_bot.ai._call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = "Ответ от Claude"
            resp = await max_ai.get_ai_response(
                3001,
                "Привет",
                track_user_activity=True,
            )
            self.assertEqual(resp, "Ответ от Claude")
            mock_claude.assert_called_once()

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 2)

    # 131. Non-Integrity insert/flush DB failure propagates, tracker remains unmarked, retry persists same request_time
    async def test_non_integrity_db_failure_propagates_and_tracker_remains_unmarked(self):
        from sqlalchemy.exc import OperationalError
        fixed_time = datetime(2026, 1, 1, 12, 0, 0)
        tracker = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=fixed_time)
        self.assertFalse(tracker._marked)

        # Force a non-Integrity DB error during session.flush inside upsert
        with patch.object(
            AsyncSession, "flush", side_effect=OperationalError("disk I/O error", params=None, orig=Exception("disk error"))
        ):
            with self.assertRaises(OperationalError):
                await tracker.mark_outbound_attempt_once()
            self.assertFalse(tracker._marked)

        # Retry with healthy session succeeds and persists the identical request_time
        await tracker.mark_outbound_attempt_once()
        self.assertTrue(tracker._marked)

        async with self.sessions() as session:
            row = await session.scalar(
                select(UserAIActivity).where(UserAIActivity.user_id == 3001, UserAIActivity.scope_key == "topic:10")
            )
            self.assertIsNotNone(row)
            self.assertEqual(row.last_request_at, fixed_time)

    # 132. Primary local failure + fallback invalid model -> zero activity
    async def test_primary_local_failure_plus_fallback_invalid_model_records_zero_activity(self):
        async with self.sessions() as session:
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "OpenAI"
            cfg.openai_api_key = ""  # Primary fails locally (missing key)
            cfg.allow_fallback = True
            cfg.fallback_provider = "Claude"
            cfg.claude_api_key = "sk-claude-test"
            cfg.fallback_model = "unsupported-model-x"  # Fallback fails local validation
            await session.commit()

        with self.assertRaises(Exception):
            await max_ai.get_ai_response(
                3001,
                "Привет",
                track_user_activity=True,
            )

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 0)

    # 133. Minute-boundary single request_time
    async def test_minute_boundary_single_request_time(self):
        # Set prior activity at 10:00:00
        t_prev = datetime(2026, 1, 1, 10, 0, 0)
        async with self.sessions() as session:
            tracker_prev = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t_prev)
            await tracker_prev.mark_outbound_attempt_once()

        # New request at 10:00:59.999999
        t_req = datetime(2026, 1, 1, 10, 0, 59, 999999)
        tracker_now = ActivityTracker(self.sessions, user_id=3001, topic_id=10, request_time=t_req)

        async with self.sessions() as session:
            # Gaps calculated using the exact tracker request_time
            gap_visit, gap_msg = await get_user_ai_activity_gaps(session, user_id=3001, topic_id=10, now=tracker_now.request_time)
            # Both gaps must be 0 minutes
            self.assertEqual(gap_visit, 0)
            self.assertEqual(gap_msg, 0)

        # Mark activity with tracker_now
        await tracker_now.mark_outbound_attempt_once()

        async with self.sessions() as session:
            row = await session.scalar(
                select(UserAIActivity).where(UserAIActivity.user_id == 3001, UserAIActivity.scope_key == "topic:10")
            )
            # Persisted time must be exactly t_req, not a drifted time
            self.assertEqual(row.last_request_at, t_req)

    # 134. Upsert IntegrityError recovery timestamp verification
    async def test_upsert_integrity_recovery_older_timestamp_raises(self):
        from sqlalchemy.exc import IntegrityError
        t_req = datetime(2026, 1, 1, 12, 0, 0)
        t_older = datetime(2026, 1, 1, 11, 0, 0)

        mock_session = MagicMock()
        update_res_0 = MagicMock(rowcount=0)
        mock_session.execute = AsyncMock(return_value=update_res_0)
        mock_session.scalar = AsyncMock(side_effect=[None, t_older])
        mock_session.begin_nested = MagicMock()
        mock_nested = MagicMock()
        mock_nested.__aenter__ = AsyncMock(return_value=mock_nested)
        mock_nested.__aexit__ = AsyncMock(return_value=False)
        mock_session.begin_nested.return_value = mock_nested
        mock_session.flush = AsyncMock(side_effect=IntegrityError("duplicate key", params=None, orig=Exception("unique constraint")))

        with self.assertRaises(IntegrityError):
            await upsert_user_ai_activity(
                mock_session,
                user_id=3001,
                scope_key="topic:10",
                request_time=t_req,
            )

    async def test_upsert_integrity_recovery_same_or_newer_timestamp_succeeds(self):
        from sqlalchemy.exc import IntegrityError
        t_req = datetime(2026, 1, 1, 12, 0, 0)
        t_newer = datetime(2026, 1, 1, 12, 5, 0)

        mock_session = MagicMock()
        update_res_0 = MagicMock(rowcount=0)
        mock_session.execute = AsyncMock(return_value=update_res_0)
        mock_session.scalar = AsyncMock(side_effect=[None, t_newer])
        mock_session.begin_nested = MagicMock()
        mock_nested = MagicMock()
        mock_nested.__aenter__ = AsyncMock(return_value=mock_nested)
        mock_nested.__aexit__ = AsyncMock(return_value=False)
        mock_session.begin_nested.return_value = mock_nested
        mock_session.flush = AsyncMock(side_effect=IntegrityError("duplicate key", params=None, orig=Exception("unique constraint")))

        await upsert_user_ai_activity(
            mock_session,
            user_id=3001,
            scope_key="topic:10",
            request_time=t_req,
        )

    # 135. Tracked direct + invalid local model -> zero activity
    async def test_telegram_direct_invalid_local_model_zero_activity(self):
        import handlers
        handlers.async_session_maker = self.sessions
        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            for act in acts:
                await session.delete(act)
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "OpenAI"
            cfg.openai_api_key = "sk-test-valid"
            cfg.openai_model = "completely-invalid-model-name"
            await session.commit()

        with self.assertRaises(Exception):
            await handlers.get_ai_response_direct(
                user_id=3001,
                system_prompt="Системный",
                user_prompt="Тест direct",
                track_user_activity=True,
            )

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 0)

    # 136. Tracked direct + missing key/config -> zero activity
    async def test_telegram_direct_missing_key_zero_activity(self):
        import handlers
        handlers.async_session_maker = self.sessions
        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            for act in acts:
                await session.delete(act)
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "OpenAI"
            cfg.openai_api_key = ""
            cfg.openai_model = "gpt-5.6-terra"
            await session.commit()

        with self.assertRaises(Exception):
            await handlers.get_ai_response_direct(
                user_id=3001,
                system_prompt="Системный",
                user_prompt="Тест direct",
                track_user_activity=True,
            )

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 0)

    # 137. Tracked direct + real network timeout -> activity recorded
    async def test_telegram_direct_network_timeout_records_activity(self):
        import handlers
        handlers.async_session_maker = self.sessions
        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            for act in acts:
                await session.delete(act)
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "OpenAI"
            cfg.openai_api_key = "sk-test-valid"
            cfg.openai_model = "gpt-5.6-terra"
            await session.commit()

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=TimeoutError("Network timeout")):
            with self.assertRaises(AIServiceError):
                await handlers.get_ai_response_direct(
                    user_id=3001,
                    system_prompt="Системный",
                    user_prompt="Тест direct timeout",
                    track_user_activity=True,
                )

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 2)

    # 138. Tracked direct success -> activity once
    async def test_telegram_direct_success_records_activity_once(self):
        import handlers
        handlers.async_session_maker = self.sessions
        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            for act in acts:
                await session.delete(act)
            cfg = await session.get(AIConfig, 1)
            cfg.provider = "OpenAI"
            cfg.openai_api_key = "sk-test-valid"
            cfg.openai_model = "gpt-5.6-terra"
            await session.commit()

        async def fake_create(**kwargs):
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "Прямой ответ ИИ"
            resp.choices = [choice]
            return resp

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create):
            resp = await handlers.get_ai_response_direct(
                user_id=3001,
                system_prompt="Системный",
                user_prompt="Тест direct success",
                track_user_activity=True,
            )
            self.assertEqual(resp, "Прямой ответ ИИ")

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 2)

    # 139. Universal test integration: preliminary direct + final handoff records activity once at single request_time
    async def test_finish_test_generation_preliminary_plus_final_handoff_records_activity_once(self):
        import handlers
        from database import TestConfig, TestSession
        handlers.async_session_maker = self.sessions

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            for act in acts:
                await session.delete(act)

            test_cfg = TestConfig(
                id=1,
                separate_result_prompt_enabled=True,
                result_system_prompt="Предварительный системный промпт интерпретации.",
                result_prompt_is_final=False,  # Preliminary direct + final handoff
            )
            test_sess = TestSession(
                user_id=3001,
                invocation_dialogue_id=1,
                invocation_topic_id=10,
                is_finished=False,
            )
            session.add_all([test_cfg, test_sess])
            await session.commit()

        mock_msg = MagicMock()
        mock_msg.chat.id = 3001
        mock_msg.answer = AsyncMock()
        mock_msg.edit_text = AsyncMock()

        direct_calls = []
        handoff_calls = []

        async def fake_direct(*args, **kwargs):
            direct_calls.append(kwargs)
            return "Предварительная интерпретация теста"

        async def fake_get_ai_response(*args, **kwargs):
            handoff_calls.append(kwargs)
            return "Итоговая интерпретация теста"

        with patch("handlers.get_ai_response_direct", side_effect=fake_direct) as p_direct, \
             patch("handlers.get_ai_response", side_effect=fake_get_ai_response) as p_handoff:
            await handlers.finish_test_generation(
                message=mock_msg,
                user_id=3001,
                answers=[{"question": "Вопрос 1", "answer": "Да"}],
                questions=["Вопрос 1"],
            )

        # Preliminary direct had track_user_activity=False and activity_tracker=None
        self.assertEqual(len(direct_calls), 1)
        self.assertFalse(direct_calls[0].get("track_user_activity"))
        self.assertIsNone(direct_calls[0].get("activity_tracker"))

        # Final handoff had track_user_activity=True and received the activity_tracker
        self.assertEqual(len(handoff_calls), 1)
        self.assertTrue(handoff_calls[0].get("track_user_activity"))
        tracker = handoff_calls[0].get("activity_tracker")
        self.assertIsNotNone(tracker)

        # Mark outbound attempt as the handoff AI call would do
        await tracker.mark_outbound_attempt_once()

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 2)
            for act in acts:
                self.assertEqual(act.last_request_at, tracker.request_time)

    # 140. Universal test integration: direct is final records activity once at single request_time
    async def test_finish_test_generation_direct_final_records_activity_once(self):
        import handlers
        from database import TestConfig, TestSession
        handlers.async_session_maker = self.sessions

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            for act in acts:
                await session.delete(act)

            test_cfg = await session.get(TestConfig, 1)
            if not test_cfg:
                test_cfg = TestConfig(id=1)
                session.add(test_cfg)
            test_cfg.separate_result_prompt_enabled = True
            test_cfg.result_system_prompt = "Финальный прямой промпт интерпретации."
            test_cfg.result_prompt_is_final = True  # Direct is final

            test_sess = await session.get(TestSession, 3001)
            if not test_sess:
                test_sess = TestSession(user_id=3001)
                session.add(test_sess)
            test_sess.invocation_dialogue_id = 1
            test_sess.invocation_topic_id = 10
            test_sess.is_finished = False
            await session.commit()

        mock_msg = MagicMock()
        mock_msg.chat.id = 3001
        mock_msg.answer = AsyncMock()
        mock_msg.edit_text = AsyncMock()

        direct_calls = []

        async def fake_direct(*args, **kwargs):
            direct_calls.append(kwargs)
            tracker = kwargs.get("activity_tracker")
            if tracker:
                await tracker.mark_outbound_attempt_once()
            return "Финальная прямая интерпретация"

        with patch("handlers.get_ai_response_direct", side_effect=fake_direct), \
             patch("handlers.get_ai_response") as mock_handoff:
            await handlers.finish_test_generation(
                message=mock_msg,
                user_id=3001,
                answers=[{"question": "Вопрос 1", "answer": "Да"}],
                questions=["Вопрос 1"],
            )

        self.assertEqual(len(direct_calls), 1)
        self.assertTrue(direct_calls[0].get("track_user_activity"))
        tracker = direct_calls[0].get("activity_tracker")
        self.assertIsNotNone(tracker)
        mock_handoff.assert_not_called()

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 2)
            for act in acts:
                self.assertEqual(act.last_request_at, tracker.request_time)

    # 141. Telegram real photo handler: canonical builder, activity updated, model validation failure 0 activity, standalone gen/edit
    async def test_telegram_real_photo_handler_canonical_builder_and_activity(self):
        import handlers
        import io
        from database import SubscriptionConfig
        handlers.async_session_maker = self.sessions

        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            for act in acts:
                await session.delete(act)
            cfg = await session.get(AIConfig, 1)
            cfg.vision_provider = "OpenAI"
            cfg.vision_model = "gpt-5.6-terra"
            cfg.openai_api_key = "sk-test-valid"

            user = await session.get(User, 3001)
            user.accepted_disclaimer = True

            sub_cfg = await session.get(SubscriptionConfig, 1)
            if not sub_cfg:
                sub_cfg = SubscriptionConfig(id=1, subscriptions_enabled=False)
                session.add(sub_cfg)
            else:
                sub_cfg.subscriptions_enabled = False

            # Add previous historical message starting with "[Изображение]" to prove it is retained
            hist_msg = DBMessage(
                user_id=3001,
                dialogue_id=1,
                topic_id=10,
                role="user",
                content="[Изображение] Прошлое фото",
                timestamp=datetime(2026, 1, 1, 10, 0, 0),
            )
            session.add(hist_msg)
            await session.commit()

        mock_bot = MagicMock()
        mock_bot.send_chat_action = AsyncMock()
        mock_bot.get_file = AsyncMock(return_value=MagicMock(file_path="photos/sample.jpg"))
        mock_bot.download_file = AsyncMock(return_value=io.BytesIO(b"fake_image_bytes_tg"))
        mock_bot.delete_message = AsyncMock()
        mock_bot.send_message = AsyncMock()

        mock_msg = MagicMock()
        mock_msg.chat.id = 3001
        mock_msg.from_user.id = 3001
        mock_msg.from_user.username = "olga"
        mock_msg.from_user.full_name = "Ольга"
        mock_msg.caption = "Разбери рисунок дерева GEN_IMG: happy tree in forest"
        mock_msg.photo = [MagicMock(file_id="tg_photo_1")]
        mock_msg.answer = AsyncMock(return_value=MagicMock(message_id=555))
        mock_msg.answer_photo = AsyncMock()

        captured_calls = []
        async def fake_create(**kwargs):
            captured_calls.append(kwargs)
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "Анализ рисунка дерева завершен. GEN_IMG: happy tree in forest"
            resp.choices = [choice]
            return resp

        # 1. Successful photo request: records activity once, uses canonical layout, excludes exact ID, retains history
        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=fake_create), \
             patch("ai_integration.generate_image", new_callable=AsyncMock, return_value=b"fake_gen_png") as mock_gen, \
             patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        self.assertEqual(len(captured_calls), 1)
        wire_messages = captured_calls[0]["messages"]
        all_sys = "\n\n".join(m["content"] for m in wire_messages if m["role"] == "system")

        # Canonical builder components present
        self.assertIn("Тема 10", all_sys)
        self.assertIn("ВРЕМЕННОЙ КОНТЕКСТ:", all_sys)
        self.assertIn("minutes_since_last_visit:", all_sys)
        self.assertIn("ИНСТРУКЦИЯ ПО АНАЛИЗУ ФОТО:", all_sys)

        # Exact photo message ID excluded from history, but historical [Изображение] retained
        history_msgs = [m for m in wire_messages if m["role"] != "system"][:-1]
        self.assertTrue(any("[Изображение] Прошлое фото" in m["content"] for m in history_msgs))
        self.assertFalse(any("Разбери рисунок дерева" in m["content"] for m in history_msgs))

        # Current image multimodal part present once at end
        current_part = wire_messages[-1]
        self.assertEqual(current_part["role"], "user")
        self.assertIsInstance(current_part["content"], list)
        self.assertIn("You are a professional expert analyst", current_part["content"][0]["text"])
        self.assertEqual(current_part["content"][1]["type"], "image_url")

        # Activity recorded once
        async with self.sessions() as session:
            acts = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts), 2)
            first_timestamps = {a.scope_key: a.last_request_at for a in acts}

        # Standalone generation occurred without adding second activity
        mock_gen.assert_called_once()
        async with self.sessions() as session:
            acts2 = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts2), 2)
            for a in acts2:
                self.assertEqual(a.last_request_at, first_timestamps[a.scope_key])

        # 2. Local vision model validation failure: zero activity change
        async with self.sessions() as session:
            for act in acts2:
                await session.delete(act)
            cfg = await session.get(AIConfig, 1)
            cfg.vision_model = "unsupported-vision-model"
            await session.commit()

        captured_calls.clear()
        with patch("handlers.is_admin", new_callable=AsyncMock, return_value=True):
            await handlers.handle_photo_message(mock_msg, state=None, bot=mock_bot)

        # Ensure no network call was made and no activity was recorded
        self.assertEqual(len(captured_calls), 0)
        async with self.sessions() as session:
            acts_failed = (await session.scalars(select(UserAIActivity).where(UserAIActivity.user_id == 3001))).all()
            self.assertEqual(len(acts_failed), 0)


