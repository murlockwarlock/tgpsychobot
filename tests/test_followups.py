import os
import sys
import types
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import followups
from database import (
    AutomationConversationState,
    AutomationEvent,
    AutomationStepTransition,
    Base,
    FollowupCampaign,
    FollowupDelivery,
    FollowupRun,
    FollowupStep,
    Message,
    User,
)
from followups import (
    _outside_quiet_hours,
    check_campaign_eligibility,
    evaluate_followup_eligibility,
    process_due_followups,
    record_user_activity,
)


class FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text))
        return SimpleNamespace(message_id=len(self.sent))


class FollowupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with self.sessions() as session:
            session.add(User(id=42, first_name="Иван", current_dialogue_id=1, current_topic_id=None))
            campaign = FollowupCampaign(
                name="Возврат",
                is_active=True,
                include_main_dialogue=True,
                quiet_start_minute=0,
                quiet_end_minute=0,
                jitter_min_seconds=0,
                jitter_max_seconds=0,
            )
            campaign.steps.append(FollowupStep(
                sort_order=0,
                delay_minutes=1,
                message_type="static",
                message_text="Продолжим?",
            ))
            session.add(campaign)
            await session.commit()

    async def asyncTearDown(self):
        await self.engine.dispose()

    def test_quiet_hours_cross_midnight(self):
        campaign = SimpleNamespace(
            timezone="Europe/Moscow",
            quiet_start_minute=22 * 60,
            quiet_end_minute=9 * 60,
        )
        # 20:30 UTC is 23:30 Moscow and must move to 09:00 Moscow next day.
        shifted = _outside_quiet_hours(datetime(2026, 8, 4, 20, 30), campaign)
        self.assertEqual(shifted, datetime(2026, 8, 5, 6, 0))

    def test_default_and_stage_modes(self):
        default = SimpleNamespace()
        self.assertTrue(evaluate_followup_eligibility(default, current_step="completed", metadata={}).eligible)
        self.assertTrue(evaluate_followup_eligibility(default, current_step=None, metadata={}).eligible)

        selected = SimpleNamespace(stage_mode="selected", stage_values=" step_1, , step_2 ")
        self.assertTrue(evaluate_followup_eligibility(selected, current_step="step_1", metadata={}).eligible)
        self.assertFalse(evaluate_followup_eligibility(selected, current_step="STEP_1", metadata={}).eligible)
        self.assertFalse(evaluate_followup_eligibility(selected, current_step="other", metadata={}).eligible)
        self.assertFalse(evaluate_followup_eligibility(selected, current_step=None, metadata={}).eligible)

        excluded = SimpleNamespace(stage_mode="all_except", stage_values="completed, crisis")
        self.assertFalse(evaluate_followup_eligibility(excluded, current_step="completed", metadata={}).eligible)
        self.assertTrue(evaluate_followup_eligibility(excluded, current_step="active", metadata={}).eligible)
        self.assertTrue(evaluate_followup_eligibility(excluded, current_step=None, metadata={}).eligible)

        not_set = SimpleNamespace(stage_mode="not_set", stage_values="obsolete")
        self.assertTrue(evaluate_followup_eligibility(not_set, current_step=None, metadata={}).eligible)
        self.assertTrue(evaluate_followup_eligibility(not_set, current_step="", metadata={}).eligible)
        self.assertFalse(evaluate_followup_eligibility(not_set, current_step="active", metadata={}).eligible)

    def test_metadata_operators_nested_values_and_stop_veto(self):
        campaign = SimpleNamespace(
            stage_mode="all",
            stage_values="",
            metadata_field_path="profile.outcome",
            metadata_operator="equals",
            metadata_expected_value="signup",
            stop_events="",
        )
        self.assertTrue(
            evaluate_followup_eligibility(
                campaign,
                current_step=None,
                metadata={"profile": {"outcome": "signup"}},
            ).eligible
        )
        campaign.metadata_operator = "not_equals"
        self.assertTrue(
            evaluate_followup_eligibility(
                campaign,
                current_step=None,
                metadata={"profile": {"outcome": "other"}},
            ).eligible
        )
        campaign.metadata_field_path = "summary"
        campaign.metadata_operator = "contains"
        campaign.metadata_expected_value = "sign"
        self.assertTrue(
            evaluate_followup_eligibility(campaign, current_step=None, metadata={"summary": "signup"}).eligible
        )
        campaign.metadata_field_path = "tags"
        campaign.metadata_expected_value = "signup"
        self.assertTrue(
            evaluate_followup_eligibility(campaign, current_step=None, metadata={"tags": ["lead", "signup"]}).eligible
        )
        campaign.stage_mode = "selected"
        campaign.stage_values = "allowed"
        campaign.stop_events = "CRISIS_DETECTED, DIALOG_COMPLETED"
        result = evaluate_followup_eligibility(
            campaign,
            current_step="blocked",
            metadata={"tags": []},
            stop_event_names=["CRISIS_DETECTED"],
        )
        self.assertFalse(result.eligible)
        self.assertEqual(result.reason, "stop_event_found")

    async def test_metadata_mismatch_uses_scoped_merged_state_not_user_history(self):
        async with self.sessions() as session:
            campaign = await session.scalar(select(FollowupCampaign))
            campaign.metadata_field_path = "outcome"
            campaign.metadata_operator = "equals"
            campaign.metadata_expected_value = "signup"
            user = await session.get(User, 42)
            user.metadata_json = '{"outcome":"signup"}'
            session.add(AutomationConversationState(
                user_id=42,
                dialogue_id=1,
                topic_id=0,
                current_step="active",
                metadata_json='{"outcome":"other"}',
            ))
            await session.commit()

        with patch.object(followups, "async_session_maker", self.sessions):
            await record_user_activity(
                42,
                dialogue_id=1,
                topic_id=None,
                activity_at=datetime.utcnow() - timedelta(minutes=2),
            )

        async with self.sessions() as session:
            self.assertIsNone(await session.scalar(select(FollowupRun)))

    async def test_stop_event_added_after_scheduling_cancels_before_delivery(self):
        async with self.sessions() as session:
            campaign = await session.scalar(select(FollowupCampaign))
            campaign.stop_events = "CRISIS_DETECTED"
            await session.commit()

        with patch.object(followups, "async_session_maker", self.sessions):
            await record_user_activity(
                42,
                dialogue_id=1,
                topic_id=None,
                activity_at=datetime.utcnow() - timedelta(minutes=2),
            )

        async with self.sessions() as session:
            session.add(AutomationEvent(
                user_id=42,
                dialogue_id=1,
                topic_id=0,
                name="CRISIS_DETECTED",
            ))
            await session.commit()

        bot = FakeBot()
        with patch.object(followups, "async_session_maker", self.sessions):
            first = await process_due_followups(bot)
            second = await process_due_followups(bot)

        async with self.sessions() as session:
            run = await session.scalar(select(FollowupRun))
            deliveries = await session.scalar(select(func.count(FollowupDelivery.id)))
        self.assertEqual((first, second), (0, 0))
        self.assertEqual(run.status, "cancelled")
        self.assertEqual(deliveries, 0)
        self.assertEqual(bot.sent, [])

    async def test_state_change_after_scheduling_is_rechecked_and_not_retried(self):
        async with self.sessions() as session:
            campaign = await session.scalar(select(FollowupCampaign))
            campaign.stage_mode = "selected"
            campaign.stage_values = "allowed"
            session.add(AutomationConversationState(
                user_id=42,
                dialogue_id=1,
                topic_id=0,
                current_step="allowed",
                metadata_json="{}",
            ))
            await session.commit()

        with patch.object(followups, "async_session_maker", self.sessions):
            await record_user_activity(
                42,
                dialogue_id=1,
                topic_id=None,
                activity_at=datetime.utcnow() - timedelta(minutes=2),
            )

        async with self.sessions() as session:
            state = await session.scalar(select(AutomationConversationState))
            state.current_step = "blocked"
            await session.commit()

        bot = FakeBot()
        with patch.object(followups, "async_session_maker", self.sessions):
            first = await process_due_followups(bot)
            second = await process_due_followups(bot)
        async with self.sessions() as session:
            run = await session.scalar(select(FollowupRun))
        self.assertEqual((first, second), (0, 0))
        self.assertEqual(run.status, "cancelled")
        self.assertEqual(bot.sent, [])

    async def test_terminal_cancellation_survives_later_delivery_failure(self):
        now = datetime.utcnow()
        async with self.sessions() as session:
            blocked_campaign = FollowupCampaign(
                name="Заблокированная",
                is_active=True,
                include_main_dialogue=True,
                stop_events="CRISIS_DETECTED",
                quiet_start_minute=0,
                quiet_end_minute=0,
            )
            blocked_campaign.steps.append(FollowupStep(
                sort_order=0,
                delay_minutes=1,
                message_type="static",
                message_text="Заблокированное сообщение",
            ))
            failing_campaign = FollowupCampaign(
                name="С ошибкой",
                is_active=True,
                include_main_dialogue=True,
                quiet_start_minute=0,
                quiet_end_minute=0,
            )
            failing_campaign.steps.append(FollowupStep(
                sort_order=0,
                delay_minutes=1,
                message_type="static",
                message_text="Сообщение с ошибкой",
            ))
            session.add_all([blocked_campaign, failing_campaign])
            await session.flush()
            run_a = FollowupRun(
                campaign_id=blocked_campaign.id,
                user_id=42,
                dialogue_id=1,
                topic_id=0,
                next_step_index=0,
                generation=1,
                last_activity_at=now,
                due_at=now - timedelta(minutes=2),
                status="active",
            )
            run_b = FollowupRun(
                campaign_id=failing_campaign.id,
                user_id=42,
                dialogue_id=1,
                topic_id=0,
                next_step_index=0,
                generation=1,
                last_activity_at=now,
                due_at=now - timedelta(minutes=1),
                status="active",
            )
            session.add_all([
                run_a,
                run_b,
                AutomationEvent(
                    user_id=42,
                    dialogue_id=1,
                    topic_id=0,
                    name="CRISIS_DETECTED",
                ),
            ])
            await session.commit()
            run_a_id = run_a.id
            run_b_id = run_b.id

        class FailingBot(FakeBot):
            async def send_message(self, chat_id, text, **kwargs):
                if text == "Сообщение с ошибкой":
                    raise RuntimeError("delivery failure")
                return await super().send_message(chat_id, text, **kwargs)

        bot = FailingBot()
        with patch.object(followups, "async_session_maker", self.sessions):
            delivered = await process_due_followups(bot)
            next_delivered = await process_due_followups(bot)

        async with self.sessions() as session:
            stored_a = await session.get(FollowupRun, run_a_id)
            stored_b = await session.get(FollowupRun, run_b_id)
        self.assertEqual(0, delivered)
        self.assertEqual(0, next_delivered)
        self.assertEqual("cancelled", stored_a.status)
        self.assertEqual("active", stored_b.status)
        self.assertGreater(stored_b.due_at, now)
        self.assertEqual([], bot.sent)

    async def test_eligibility_checks_do_not_create_automation_history_rows(self):
        async with self.sessions() as session:
            campaign = await session.scalar(select(FollowupCampaign))
            session.add_all([
                AutomationConversationState(
                    user_id=42,
                    dialogue_id=1,
                    topic_id=0,
                    current_step="active",
                    metadata_json='{"outcome":"signup"}',
                ),
                AutomationStepTransition(
                    user_id=42,
                    dialogue_id=1,
                    topic_id=0,
                    current_step="active",
                    state_json='{"current_step":"active"}',
                ),
                AutomationEvent(
                    user_id=42,
                    dialogue_id=1,
                    topic_id=0,
                    name="OBSERVED",
                ),
            ])
            await session.commit()
            before_events = await session.scalar(select(func.count(AutomationEvent.id)))
            before_transitions = await session.scalar(select(func.count(AutomationStepTransition.id)))
            first = await check_campaign_eligibility(
                session, campaign, user_id=42, dialogue_id=1, topic_id=None
            )
            second = await check_campaign_eligibility(
                session, campaign, user_id=42, dialogue_id=1, topic_id=None
            )
            after_events = await session.scalar(select(func.count(AutomationEvent.id)))
            after_transitions = await session.scalar(select(func.count(AutomationStepTransition.id)))
        self.assertTrue(first.eligible)
        self.assertTrue(second.eligible)
        self.assertEqual(before_events, after_events)
        self.assertEqual(before_transitions, after_transitions)

    async def test_activity_restarts_generation_and_static_step_is_sent_once(self):
        with patch.object(followups, "async_session_maker", self.sessions):
            await record_user_activity(
                42,
                dialogue_id=1,
                topic_id=None,
                activity_at=datetime.utcnow() - timedelta(minutes=5),
            )
            await record_user_activity(
                42,
                dialogue_id=1,
                topic_id=None,
                activity_at=datetime.utcnow() - timedelta(minutes=2),
            )
            bot = FakeBot()
            first = await process_due_followups(bot)
            second = await process_due_followups(bot)

        async with self.sessions() as session:
            run = await session.scalar(select(FollowupRun))
            deliveries = await session.scalar(select(func.count(FollowupDelivery.id)))

        self.assertEqual(run.generation, 2)
        self.assertEqual(run.status, "completed")
        self.assertEqual(first, 1)
        self.assertEqual(second, 0)
        self.assertEqual(deliveries, 1)
        self.assertEqual(bot.sent, [(42, "Продолжим?")])

    async def test_old_dialogue_run_is_cancelled_before_delivery(self):
        with patch.object(followups, "async_session_maker", self.sessions):
            await record_user_activity(
                42,
                dialogue_id=1,
                topic_id=None,
                activity_at=datetime.utcnow() - timedelta(minutes=2),
            )
            async with self.sessions() as session:
                user = await session.get(User, 42)
                user.current_dialogue_id = 2
                await session.commit()
            bot = FakeBot()
            sent = await process_due_followups(bot)

        async with self.sessions() as session:
            run = await session.scalar(select(FollowupRun))
        self.assertEqual(sent, 0)
        self.assertEqual(run.status, "cancelled")
        self.assertEqual(bot.sent, [])

    async def test_ai_followup_uses_service_instruction_and_saves_sent_message(self):
        async with self.sessions() as session:
            step = await session.scalar(select(FollowupStep))
            step.message_type = "ai"
            step.ai_instruction = "Напомни, какой ответ мы ждём."
            await session.commit()

        with patch.object(followups, "async_session_maker", self.sessions), patch(
            "ai_integration.get_ai_response",
            new=AsyncMock(return_value="Вернись к нам\n[Дальше](btn:continue)"),
        ) as get_ai_response:
            await record_user_activity(
                42,
                dialogue_id=1,
                topic_id=None,
                activity_at=datetime.utcnow() - timedelta(minutes=2),
            )
            bot = FakeBot()
            fake_handlers = types.ModuleType("handlers")
            fake_send = AsyncMock()
            fake_handlers._send_generated_response = fake_send
            with patch.dict(sys.modules, {"handlers": fake_handlers}):
                sent = await process_due_followups(bot)

        self.assertEqual(sent, 1)
        get_ai_response.assert_awaited_once()
        call_kwargs = get_ai_response.await_args.kwargs
        self.assertEqual(call_kwargs["request_type"], "followup")
        self.assertFalse(call_kwargs["persist_service_data"])
        self.assertIn("[Служебная команда системы]: Пользователь замолчал.", get_ai_response.await_args.args[1])
        self.assertIn("Напомни, какой ответ мы ждём.", get_ai_response.await_args.args[1])
        fake_send.assert_awaited_once_with(bot, 42, "Вернись к нам\n[Дальше](btn:continue)")

        async with self.sessions() as session:
            message = await session.scalar(select(Message).where(Message.role == "assistant"))

        self.assertIsNotNone(message)
        self.assertEqual(message.content, "Вернись к нам")
