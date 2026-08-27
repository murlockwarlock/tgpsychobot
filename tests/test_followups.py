import asyncio
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import followups
import automation_admin
from database import (
    AutomationConversationState,
    AutomationEvent,
    AutomationStepTransition,
    Base,
    FollowupCampaign,
    FollowupDelivery,
    FollowupDeliveryAttempt,
    FollowupRun,
    FollowupStep,
    Message,
    User,
)
from followups import (
    _outside_quiet_hours,
    FollowupActivityMiddleware,
    check_campaign_eligibility,
    evaluate_followup_eligibility,
    begin_user_activity,
    finalize_user_activity,
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
        self.assertEqual("uncertain", stored_b.status)
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

    async def test_allowed_chain_advances_each_step_and_keeps_delivery_history(self):
        async with self.sessions() as session:
            campaign = await session.scalar(select(FollowupCampaign))
            second_step = FollowupStep(
                campaign_id=campaign.id,
                sort_order=1,
                delay_minutes=20,
                message_type="static",
                message_text="Ещё вопрос?",
            )
            session.add(second_step)
            await session.flush()
            first_step_id = await session.scalar(
                select(FollowupStep.id)
                .where(FollowupStep.campaign_id == campaign.id)
                .order_by(FollowupStep.sort_order.asc())
            )
            second_step_id = second_step.id
            await session.commit()

        bot = FakeBot()
        with patch.object(followups, "async_session_maker", self.sessions):
            await record_user_activity(
                42,
                dialogue_id=1,
                topic_id=None,
                activity_at=datetime.utcnow() - timedelta(minutes=5),
            )
            first = await process_due_followups(bot)
            async with self.sessions() as session:
                run = await session.scalar(select(FollowupRun))
                run.due_at = datetime.utcnow() - timedelta(minutes=1)
                await session.commit()
            second = await process_due_followups(bot)

        async with self.sessions() as session:
            run = await session.scalar(select(FollowupRun))
            deliveries = (
                await session.execute(
                    select(FollowupDelivery).order_by(FollowupDelivery.id.asc())
                )
            ).scalars().all()
            attempts = (
                await session.execute(
                    select(FollowupDeliveryAttempt).order_by(FollowupDeliveryAttempt.step_index.asc())
                )
            ).scalars().all()
        self.assertEqual((1, 1), (first, second))
        self.assertEqual((2, "completed"), (run.next_step_index, run.status))
        self.assertEqual([first_step_id, second_step_id], [item.step_id for item in deliveries])
        self.assertEqual([1, 1], [item.generation for item in deliveries])
        self.assertEqual(["delivered", "delivered"], [item.status for item in attempts])
        self.assertEqual([(42, "Продолжим?"), (42, "Ещё вопрос?")], bot.sent)

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


class FollowupRaceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.database_schema = os.environ.get("FOLLOWUP_TEST_SCHEMA")
        database_url = os.environ.get("FOLLOWUP_TEST_DATABASE_URL")
        if database_url and self.database_schema and database_url.startswith("postgresql+"):
            self.engine = create_async_engine(
                database_url,
                connect_args={
                    "server_settings": {
                        "search_path": f"{self.database_schema},public",
                    },
                },
            )
        else:
            self.engine = create_async_engine(database_url or (
                f"sqlite+aiosqlite:///{self.tempdir.name}/followups.db"
            ))
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as connection:
            if database_url and database_url.startswith("postgresql+"):
                await connection.run_sync(Base.metadata.drop_all)
            await connection.run_sync(Base.metadata.create_all)
        async with self.sessions() as session:
            session.add(User(id=42, first_name="Иван", current_dialogue_id=1, current_topic_id=None))
            campaign = FollowupCampaign(
                name="A",
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
                message_text="A",
            ))
            session.add(campaign)
            await session.commit()
            self.campaign_id = campaign.id
            self.step_id = campaign.steps[0].id

    async def asyncTearDown(self):
        if self.database_schema:
            async with self.engine.begin() as connection:
                await connection.execute(text(f'DROP SCHEMA "{self.database_schema}" CASCADE'))
        await self.engine.dispose()
        self.tempdir.cleanup()

    async def _add_due_run(self, campaign_id=None, due_at=None):
        async with self.sessions() as session:
            campaign_id = campaign_id or self.campaign_id
            step = await session.scalar(
                select(FollowupStep).where(FollowupStep.campaign_id == campaign_id)
            )
            run = FollowupRun(
                campaign_id=campaign_id,
                user_id=42,
                dialogue_id=1,
                topic_id=0,
                next_step_index=0,
                generation=1,
                last_activity_at=datetime.utcnow() - timedelta(minutes=3),
                due_at=due_at or datetime.utcnow() - timedelta(minutes=2),
                status="active",
            )
            session.add(run)
            await session.commit()
            return run.id, step.id

    async def _run_state(self, run_id):
        async with self.sessions() as session:
            return await session.get(FollowupRun, run_id)

    async def test_concurrent_schedulers_send_one_external_message(self):
        run_id, _ = await self._add_due_run()
        started = asyncio.Event()
        release = asyncio.Event()

        class BlockingBot(FakeBot):
            def __init__(self):
                super().__init__()
                self.calls = 0

            async def send_message(self, chat_id, text, **kwargs):
                self.calls += 1
                started.set()
                await release.wait()
                return await super().send_message(chat_id, text, **kwargs)

        bot = BlockingBot()
        with patch.object(followups, "async_session_maker", self.sessions):
            first_task = asyncio.create_task(process_due_followups(bot))
            await asyncio.wait_for(started.wait(), timeout=2)
            second_task = asyncio.create_task(process_due_followups(bot))
            second_result = await asyncio.wait_for(second_task, timeout=2)
            self.assertEqual(0, second_result)
            self.assertEqual(1, bot.calls)
            self.assertEqual([], bot.sent)
            release.set()
            first_result = await asyncio.wait_for(first_task, timeout=2)

        self.assertEqual(1, first_result)
        async with self.sessions() as session:
            self.assertEqual(1, await session.scalar(select(func.count(FollowupDelivery.id))))
            self.assertEqual(1, await session.scalar(select(func.count(FollowupDeliveryAttempt.id))))
            attempt = await session.scalar(select(FollowupDeliveryAttempt))
        self.assertEqual("delivered", attempt.status)
        self.assertEqual(run_id, attempt.run_id)

    async def test_claim_is_durable_before_worker_can_reach_external_send(self):
        await self._add_due_run()
        claim_started = asyncio.Event()
        release_claim = asyncio.Event()
        original_claim = followups._claim_due_followup

        async def hold_after_claim(run_id, now):
            claim = await original_claim(run_id, now)
            if claim is not None and not claim_started.is_set():
                claim_started.set()
                await release_claim.wait()
            return claim

        bot = FakeBot()
        with patch.object(followups, "async_session_maker", self.sessions), patch.object(
            followups, "_claim_due_followup", new=hold_after_claim
        ):
            first_task = asyncio.create_task(process_due_followups(bot))
            await asyncio.wait_for(claim_started.wait(), timeout=2)
            second_result = await asyncio.wait_for(process_due_followups(bot), timeout=2)
            self.assertEqual(0, second_result)
            self.assertEqual([], bot.sent)
            release_claim.set()
            first_result = await asyncio.wait_for(first_task, timeout=2)

        self.assertEqual(1, first_result)
        self.assertEqual([(42, "A")], bot.sent)

    async def test_activity_ingress_invalidates_claim_before_external_send(self):
        await self._add_due_run()
        claim_started = asyncio.Event()
        release_claim = asyncio.Event()
        original_claim = followups._claim_due_followup

        async def hold_after_claim(run_id, now):
            claim = await original_claim(run_id, now)
            if claim is not None and not claim_started.is_set():
                claim_started.set()
                await release_claim.wait()
            return claim

        bot = FakeBot()
        with patch.object(followups, "async_session_maker", self.sessions), patch.object(
            followups, "_claim_due_followup", new=hold_after_claim
        ):
            process_task = asyncio.create_task(process_due_followups(bot))
            await asyncio.wait_for(claim_started.wait(), timeout=2)
            ingress = await begin_user_activity(42, dialogue_id=1, topic_id=None)
            release_claim.set()
            self.assertEqual(0, await asyncio.wait_for(process_task, timeout=2))

        self.assertEqual({self.campaign_id: 2}, ingress.run_generations)
        self.assertEqual([], bot.sent)
        state = await self._run_state(ingress.run_ids[self.campaign_id])
        self.assertEqual((2, 0, "pending"), (state.generation, state.next_step_index, state.status))
        async with self.sessions() as session:
            attempt = await session.scalar(select(FollowupDeliveryAttempt))
        self.assertEqual("cancelled", attempt.status)

    async def test_activity_during_ai_preparation_skips_external_emission(self):
        run_id, _ = await self._add_due_run()
        async with self.sessions() as session:
            step = await session.get(FollowupStep, self.step_id)
            step.message_type = "ai"
            step.ai_instruction = "Подготовь продолжение диалога."
            await session.commit()

        ai_started = asyncio.Event()
        release_ai = asyncio.Event()

        async def blocked_ai(*args, **kwargs):
            ai_started.set()
            await release_ai.wait()
            return "Подготовленный ответ"

        fake_handlers = types.ModuleType("handlers")
        fake_send = AsyncMock()
        fake_handlers._send_generated_response = fake_send
        bot = FakeBot()

        async def handler(event, data):
            return "handled"

        event = SimpleNamespace(
            from_user=SimpleNamespace(id=42, is_bot=False),
            data="",
        )
        with patch.object(followups, "async_session_maker", self.sessions), patch(
            "ai_integration.get_ai_response",
            new=blocked_ai,
        ), patch.dict(sys.modules, {"handlers": fake_handlers}):
            process_task = asyncio.create_task(process_due_followups(bot))
            await asyncio.wait_for(ai_started.wait(), timeout=2)
            self.assertEqual("handled", await FollowupActivityMiddleware()(handler, event, {}))
            release_ai.set()
            self.assertEqual(0, await asyncio.wait_for(process_task, timeout=2))

        self.assertEqual([], bot.sent)
        fake_send.assert_not_awaited()
        async with self.sessions() as session:
            run = await session.get(FollowupRun, run_id)
            attempt = await session.scalar(select(FollowupDeliveryAttempt))
        self.assertEqual((2, 0, "active"), (run.generation, run.next_step_index, run.status))
        self.assertEqual("cancelled", attempt.status)

    async def test_campaign_disabled_during_ai_preparation_skips_external_emission(self):
        run_id, _ = await self._add_due_run()
        async with self.sessions() as session:
            step = await session.get(FollowupStep, self.step_id)
            step.message_type = "ai"
            await session.commit()

        ai_started = asyncio.Event()
        release_ai = asyncio.Event()

        async def blocked_ai(*args, **kwargs):
            ai_started.set()
            await release_ai.wait()
            return "Подготовленный ответ"

        fake_handlers = types.ModuleType("handlers")
        fake_send = AsyncMock()
        fake_handlers._send_generated_response = fake_send
        bot = FakeBot()

        with patch.object(followups, "async_session_maker", self.sessions), patch(
            "ai_integration.get_ai_response",
            new=blocked_ai,
        ), patch.dict(sys.modules, {"handlers": fake_handlers}):
            process_task = asyncio.create_task(process_due_followups(bot))
            await asyncio.wait_for(ai_started.wait(), timeout=2)
            async with self.sessions() as session:
                campaign = await session.get(FollowupCampaign, self.campaign_id)
                campaign.is_active = False
                await session.commit()
            release_ai.set()
            self.assertEqual(0, await asyncio.wait_for(process_task, timeout=2))

        self.assertEqual([], bot.sent)
        fake_send.assert_not_awaited()
        async with self.sessions() as session:
            run = await session.get(FollowupRun, run_id)
            attempt = await session.scalar(select(FollowupDeliveryAttempt))
        self.assertEqual("cancelled", run.status)
        self.assertEqual("cancelled", attempt.status)

    async def test_stop_event_during_ai_preparation_skips_external_emission(self):
        run_id, _ = await self._add_due_run()
        async with self.sessions() as session:
            campaign = await session.get(FollowupCampaign, self.campaign_id)
            campaign.stop_events = "CRISIS_DETECTED"
            step = await session.get(FollowupStep, self.step_id)
            step.message_type = "ai"
            await session.commit()

        ai_started = asyncio.Event()
        release_ai = asyncio.Event()

        async def blocked_ai(*args, **kwargs):
            ai_started.set()
            await release_ai.wait()
            return "Подготовленный ответ"

        fake_handlers = types.ModuleType("handlers")
        fake_send = AsyncMock()
        fake_handlers._send_generated_response = fake_send
        bot = FakeBot()

        with patch.object(followups, "async_session_maker", self.sessions), patch(
            "ai_integration.get_ai_response",
            new=blocked_ai,
        ), patch.dict(sys.modules, {"handlers": fake_handlers}):
            process_task = asyncio.create_task(process_due_followups(bot))
            await asyncio.wait_for(ai_started.wait(), timeout=2)
            async with self.sessions() as session:
                session.add(AutomationEvent(
                    user_id=42,
                    dialogue_id=1,
                    topic_id=0,
                    name="CRISIS_DETECTED",
                ))
                await session.commit()
            release_ai.set()
            self.assertEqual(0, await asyncio.wait_for(process_task, timeout=2))

        self.assertEqual([], bot.sent)
        fake_send.assert_not_awaited()
        async with self.sessions() as session:
            run = await session.get(FollowupRun, run_id)
            attempt = await session.scalar(select(FollowupDeliveryAttempt))
        self.assertEqual("cancelled", run.status)
        self.assertEqual("cancelled", attempt.status)

    async def test_ai_preparation_failure_is_cancelled_without_delivery_uncertainty(self):
        run_id, _ = await self._add_due_run()
        async with self.sessions() as session:
            step = await session.get(FollowupStep, self.step_id)
            step.message_type = "ai"
            await session.commit()

        async def failing_ai(*args, **kwargs):
            raise RuntimeError("AI unavailable")

        bot = FakeBot()
        with patch.object(followups, "async_session_maker", self.sessions), patch(
            "ai_integration.get_ai_response",
            new=failing_ai,
        ):
            self.assertEqual(0, await process_due_followups(bot))

        self.assertEqual([], bot.sent)
        async with self.sessions() as session:
            run = await session.get(FollowupRun, run_id)
            attempt = await session.scalar(select(FollowupDeliveryAttempt))
        self.assertEqual("cancelled", run.status)
        self.assertEqual("cancelled", attempt.status)

    async def test_old_completion_records_history_without_overwriting_new_generation(self):
        run_id, _ = await self._add_due_run()
        started = asyncio.Event()
        release = asyncio.Event()

        class BlockingBot(FakeBot):
            async def send_message(self, chat_id, text, **kwargs):
                started.set()
                await release.wait()
                return await super().send_message(chat_id, text, **kwargs)

        bot = BlockingBot()
        with patch.object(followups, "async_session_maker", self.sessions):
            process_task = asyncio.create_task(process_due_followups(bot))
            await asyncio.wait_for(started.wait(), timeout=2)
            activity_at = datetime.utcnow()
            ingress = await begin_user_activity(
                42,
                dialogue_id=1,
                topic_id=None,
                activity_at=activity_at,
            )
            release.set()
            self.assertEqual(1, await asyncio.wait_for(process_task, timeout=2))

        state = await self._run_state(run_id)
        self.assertEqual((2, 0, "pending", activity_at), (
            state.generation,
            state.next_step_index,
            state.status,
            state.due_at,
        ))
        async with self.sessions() as session:
            delivery = await session.scalar(select(FollowupDelivery))
            attempt = await session.scalar(select(FollowupDeliveryAttempt))
        self.assertEqual((1, self.step_id), (delivery.generation, delivery.step_id))
        self.assertEqual("delivered", attempt.status)
        self.assertEqual({self.campaign_id: 2}, ingress.run_generations)

    async def test_stale_batch_candidate_is_rechecked_after_another_run_restarts(self):
        run_a_id, _ = await self._add_due_run(due_at=datetime.utcnow() - timedelta(minutes=3))
        async with self.sessions() as session:
            campaign_b = FollowupCampaign(
                name="B",
                is_active=True,
                include_main_dialogue=True,
                quiet_start_minute=0,
                quiet_end_minute=0,
                jitter_min_seconds=0,
                jitter_max_seconds=0,
            )
            campaign_b.steps.append(FollowupStep(
                sort_order=0,
                delay_minutes=1,
                message_type="static",
                message_text="B",
            ))
            session.add(campaign_b)
            await session.flush()
            run_b = FollowupRun(
                campaign_id=campaign_b.id,
                user_id=42,
                dialogue_id=1,
                topic_id=0,
                next_step_index=0,
                generation=1,
                last_activity_at=datetime.utcnow() - timedelta(minutes=3),
                due_at=datetime.utcnow() - timedelta(minutes=2),
                status="active",
            )
            session.add(run_b)
            await session.commit()
            run_b_id = run_b.id

        started = asyncio.Event()
        release = asyncio.Event()

        class SlowFirstBot(FakeBot):
            async def send_message(self, chat_id, text, **kwargs):
                if text == "A":
                    started.set()
                    await release.wait()
                return await super().send_message(chat_id, text, **kwargs)

        bot = SlowFirstBot()
        with patch.object(followups, "async_session_maker", self.sessions):
            process_task = asyncio.create_task(process_due_followups(bot))
            await asyncio.wait_for(started.wait(), timeout=2)
            await record_user_activity(42, dialogue_id=1, topic_id=None)
            release.set()
            delivered = await asyncio.wait_for(process_task, timeout=2)

        self.assertEqual(1, delivered)
        self.assertEqual([(42, "A")], bot.sent)
        state_a = await self._run_state(run_a_id)
        state_b = await self._run_state(run_b_id)
        self.assertEqual((2, 0, "active"), (state_a.generation, state_a.next_step_index, state_a.status))
        self.assertEqual((2, 0, "active"), (state_b.generation, state_b.next_step_index, state_b.status))
        async with self.sessions() as session:
            deliveries = (await session.execute(select(FollowupDelivery))).scalars().all()
        self.assertEqual([(run_a_id, 1)], [(item.run_id, item.generation) for item in deliveries])

    async def test_finalizing_old_delivery_leaves_new_generation_scheduled(self):
        run_id, _ = await self._add_due_run()
        started = asyncio.Event()
        release = asyncio.Event()

        class BlockingBot(FakeBot):
            async def send_message(self, chat_id, text, **kwargs):
                started.set()
                await release.wait()
                return await super().send_message(chat_id, text, **kwargs)

        bot = BlockingBot()
        with patch.object(followups, "async_session_maker", self.sessions):
            process_task = asyncio.create_task(process_due_followups(bot))
            await asyncio.wait_for(started.wait(), timeout=2)
            activity_at = datetime.utcnow()
            ingress = await begin_user_activity(
                42,
                dialogue_id=1,
                topic_id=None,
                activity_at=activity_at,
            )
            release.set()
            await asyncio.wait_for(process_task, timeout=2)
            await finalize_user_activity(
                42,
                ingress,
                dialogue_id=1,
                topic_id=None,
            )

        state = await self._run_state(run_id)
        self.assertEqual(2, state.generation)
        self.assertEqual(0, state.next_step_index)
        self.assertEqual("active", state.status)
        self.assertEqual(_due_at_for_test(activity_at), state.due_at)

    async def test_stale_claim_becomes_uncertain_without_a_retry(self):
        run_id, step_id = await self._add_due_run()
        async with self.sessions() as session:
            session.add(FollowupDeliveryAttempt(
                run_id=run_id,
                step_id=step_id,
                step_index=0,
                generation=1,
                claim_token="crashed-worker",
                status="claimed",
                claimed_at=datetime.utcnow() - timedelta(hours=1),
            ))
            await session.commit()
        bot = FakeBot()
        with patch.object(followups, "async_session_maker", self.sessions):
            self.assertEqual(0, await process_due_followups(bot))
            self.assertEqual(0, await process_due_followups(bot))
        self.assertEqual([], bot.sent)
        async with self.sessions() as session:
            attempt = await session.scalar(select(FollowupDeliveryAttempt))
            run = await session.get(FollowupRun, run_id)
            self.assertEqual(0, await session.scalar(select(func.count(FollowupDelivery.id))))
        self.assertEqual("uncertain", attempt.status)
        self.assertEqual("uncertain", run.status)

    async def test_ambiguous_external_failure_is_uncertain_without_a_retry(self):
        run_id, _ = await self._add_due_run()

        class AmbiguousBot(FakeBot):
            def __init__(self):
                super().__init__()
                self.calls = 0

            async def send_message(self, chat_id, text, **kwargs):
                self.calls += 1
                self.sent.append((chat_id, text))
                raise RuntimeError("Telegram response was lost")

        bot = AmbiguousBot()
        with patch.object(followups, "async_session_maker", self.sessions):
            self.assertEqual(0, await process_due_followups(bot))
            self.assertEqual(0, await process_due_followups(bot))
        self.assertEqual(1, bot.calls)
        self.assertEqual([(42, "A")], bot.sent)
        async with self.sessions() as session:
            attempt = await session.scalar(select(FollowupDeliveryAttempt))
            run = await session.get(FollowupRun, run_id)
            self.assertEqual(0, await session.scalar(select(func.count(FollowupDelivery.id))))
        self.assertEqual("uncertain", attempt.status)
        self.assertEqual("uncertain", run.status)

    async def test_uncertain_runs_do_not_consume_due_limit_and_real_activity_recovers_one(self):
        now = datetime.utcnow()
        uncertain_user_ids = list(range(1000, 1101))
        async with self.sessions() as session:
            session.add_all([
                User(
                    id=user_id,
                    first_name=f"User {user_id}",
                    current_dialogue_id=1,
                    current_topic_id=None,
                )
                for user_id in uncertain_user_ids
            ])
            await session.flush()
            runs = [
                FollowupRun(
                    campaign_id=self.campaign_id,
                    user_id=user_id,
                    dialogue_id=1,
                    topic_id=0,
                    next_step_index=0,
                    generation=1,
                    last_activity_at=now - timedelta(hours=1),
                    due_at=now - timedelta(minutes=30),
                    status="uncertain",
                )
                for user_id in uncertain_user_ids
            ]
            session.add_all(runs)
            await session.flush()
            session.add_all([
                FollowupDeliveryAttempt(
                    run_id=run.id,
                    step_id=self.step_id,
                    step_index=0,
                    generation=1,
                    claim_token=f"uncertain-{run.id}",
                    status="uncertain",
                    claimed_at=now - timedelta(hours=1),
                    finished_at=now - timedelta(minutes=45),
                )
                for run in runs
            ])
            valid_run = FollowupRun(
                campaign_id=self.campaign_id,
                user_id=42,
                dialogue_id=1,
                topic_id=0,
                next_step_index=0,
                generation=1,
                last_activity_at=now - timedelta(hours=1),
                due_at=now - timedelta(minutes=1),
                status="active",
            )
            session.add(valid_run)
            await session.commit()
            uncertain_run_id = runs[0].id
            valid_run_id = valid_run.id

        bot = FakeBot()
        with patch.object(followups, "async_session_maker", self.sessions):
            self.assertEqual(
                [valid_run_id],
                await followups._due_followup_ids(now, limit=100),
            )
            self.assertEqual(1, await process_due_followups(bot))
            activity_at = datetime.utcnow()
            await record_user_activity(
                uncertain_user_ids[0],
                dialogue_id=1,
                topic_id=None,
                activity_at=activity_at,
            )

        self.assertEqual([(42, "A")], bot.sent)
        async with self.sessions() as session:
            recovered = await session.get(FollowupRun, uncertain_run_id)
            attempts = (
                await session.execute(
                    select(FollowupDeliveryAttempt)
                    .where(FollowupDeliveryAttempt.run_id == uncertain_run_id)
                )
            ).scalars().all()
        self.assertEqual((2, 0, "active", activity_at + timedelta(minutes=1)), (
            recovered.generation,
            recovered.next_step_index,
            recovered.status,
            recovered.due_at,
        ))
        self.assertEqual([(1, "uncertain")], [(item.generation, item.status) for item in attempts])

    async def test_middleware_finalization_does_not_increment_generation_twice(self):
        run_id, _ = await self._add_due_run()

        async def handler(event, data):
            return "handled"

        event = SimpleNamespace(
            from_user=SimpleNamespace(id=42, is_bot=False),
            data="",
        )
        with patch.object(followups, "async_session_maker", self.sessions):
            result = await FollowupActivityMiddleware()(handler, event, {})

        self.assertEqual("handled", result)
        state = await self._run_state(run_id)
        self.assertEqual((2, 0, "active"), (state.generation, state.next_step_index, state.status))

    async def test_middleware_invalidates_before_handler_and_uses_final_scope(self):
        run_id, _ = await self._add_due_run()
        observed = {}

        async def handler(event, data):
            async with self.sessions() as session:
                run = await session.get(FollowupRun, run_id)
                observed["run"] = (run.generation, run.status)
                user = await session.get(User, 42)
                user.current_dialogue_id = 2
                await session.commit()
            return "handled"

        event = SimpleNamespace(
            from_user=SimpleNamespace(id=42, is_bot=False),
            data="",
        )
        middleware = FollowupActivityMiddleware()
        with patch.object(followups, "async_session_maker", self.sessions):
            result = await middleware(handler, event, {})

        self.assertEqual("handled", result)
        self.assertEqual((2, "pending"), observed["run"])
        async with self.sessions() as session:
            old_run = await session.get(FollowupRun, run_id)
            new_run = await session.scalar(
                select(FollowupRun).where(
                    FollowupRun.campaign_id == self.campaign_id,
                    FollowupRun.user_id == 42,
                    FollowupRun.dialogue_id == 2,
                    FollowupRun.topic_id == 0,
                )
            )
        self.assertEqual((2, "cancelled"), (old_run.generation, old_run.status))
        self.assertEqual((1, "active"), (new_run.generation, new_run.status))

    async def test_admin_plain_dialogue_counts_but_admin_followup_callback_does_not(self):
        run_id, _ = await self._add_due_run()
        async with self.sessions() as session:
            user = await session.get(User, 42)
            user.is_admin = True
            await session.commit()

        middleware = FollowupActivityMiddleware()
        event = SimpleNamespace(
            from_user=SimpleNamespace(id=42, is_bot=False),
            data="followup_campaigns",
        )

        async def admin_handler(event, data):
            return "admin"

        with patch.object(followups, "async_session_maker", self.sessions):
            self.assertEqual("admin", await middleware(admin_handler, event, {}))

        state = await self._run_state(run_id)
        self.assertEqual((1, "active"), (state.generation, state.status))
        observed = {}

        async def dialogue_handler(event, data):
            async with self.sessions() as session:
                run = await session.get(FollowupRun, run_id)
                observed["state"] = (run.generation, run.status)
            return "handled"

        event.data = ""
        with patch.object(followups, "async_session_maker", self.sessions):
            result = await middleware(dialogue_handler, event, {})

        self.assertEqual("handled", result)
        self.assertEqual((2, "pending"), observed["state"])
        state = await self._run_state(run_id)
        self.assertEqual((2, "active"), (state.generation, state.status))

    async def test_middleware_finalizes_pending_activity_when_handler_raises(self):
        run_id, _ = await self._add_due_run()

        async def failing_handler(event, data):
            raise RuntimeError("handler failure")

        event = SimpleNamespace(
            from_user=SimpleNamespace(id=42, is_bot=False),
            data="",
        )
        with patch.object(followups, "async_session_maker", self.sessions):
            with self.assertRaisesRegex(RuntimeError, "handler failure"):
                await FollowupActivityMiddleware()(failing_handler, event, {})

        state = await self._run_state(run_id)
        self.assertEqual((2, "active"), (state.generation, state.status))

    async def test_postgres_claim_and_step_delete_preserve_claim_evidence(self):
        if not self.database_schema or not os.environ.get("FOLLOWUP_TEST_DATABASE_URL", "").startswith("postgresql+"):
            self.skipTest("PostgreSQL follow-up test requires FOLLOWUP_TEST_DATABASE_URL and FOLLOWUP_TEST_SCHEMA")
        run_id, _ = await self._add_due_run()
        claim_started = asyncio.Event()
        release_claim = asyncio.Event()
        original_claim = followups._claim_due_followup

        async def hold_after_claim(run_id, now):
            claim = await original_claim(run_id, now)
            if claim is not None:
                claim_started.set()
                await release_claim.wait()
            return claim

        bot = FakeBot()
        callback = SimpleNamespace(
            id="step-delete-race",
            data=f"followup_step_delete_{self.campaign_id}_{self.step_id}",
            message=SimpleNamespace(edit_text=AsyncMock(), edit_reply_markup=AsyncMock()),
            answer=AsyncMock(),
        )
        with patch.object(followups, "async_session_maker", self.sessions), patch.object(
            followups, "_claim_due_followup", new=hold_after_claim
        ), patch.object(automation_admin, "async_session_maker", self.sessions), patch.object(
            automation_admin, "followup_steps", new=AsyncMock()
        ):
            process_task = asyncio.create_task(process_due_followups(bot))
            await asyncio.wait_for(claim_started.wait(), timeout=2)
            await automation_admin.followup_step_delete(callback)
            release_claim.set()
            self.assertEqual(1, await asyncio.wait_for(process_task, timeout=2))

        self.assertEqual([(42, "A")], bot.sent)
        async with self.sessions() as session:
            self.assertIsNotNone(await session.get(FollowupStep, self.step_id))
            self.assertEqual(1, await session.scalar(select(func.count(FollowupDelivery.id))))
            self.assertEqual(1, await session.scalar(select(func.count(FollowupDeliveryAttempt.id))))


def _due_at_for_test(activity_at):
    return activity_at + timedelta(minutes=1)
