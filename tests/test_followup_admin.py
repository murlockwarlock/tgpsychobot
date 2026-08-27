import asyncio
import os
import sys
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiogram import Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, Chat, Message as TelegramMessage, Update, User as TelegramUser

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import automation_admin
from database import (
    AutomationConversationState,
    AutomationEvent,
    Base,
    FollowupCampaign,
    FollowupDelivery,
    FollowupDeliveryAttempt,
    FollowupRun,
    FollowupStep,
    Topic,
    User,
)


class MemoryState:
    def __init__(self, data=None):
        self.data = dict(data or {})
        self.current_state = None

    async def clear(self):
        self.data.clear()
        self.current_state = None

    async def update_data(self, **values):
        self.data.update(values)

    async def get_data(self):
        return dict(self.data)

    async def set_state(self, value):
        self.current_state = getattr(value, "state", value)

    async def get_state(self):
        return self.current_state


class FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text, kwargs))
        return SimpleNamespace(message_id=len(self.sent))


def make_callback(data, user_id=42, message=None):
    return SimpleNamespace(
        id=f"callback-{data}-{user_id}",
        data=data,
        from_user=SimpleNamespace(
            id=user_id,
            username="admin",
            first_name="Админ",
            is_bot=False,
        ),
        message=message or SimpleNamespace(edit_text=AsyncMock(), edit_reply_markup=AsyncMock()),
        answer=AsyncMock(),
    )


def callback_data(markup):
    return [
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if button.callback_data
    ]


def button_texts(markup):
    return [button.text for row in markup.inline_keyboard for button in row]


class FollowupAdminTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with self.sessions() as session:
            topic = Topic(id=7, name="Самооценка", is_active=True)
            user = User(
                id=42,
                username="admin",
                first_name="Админ",
                is_admin=True,
                current_dialogue_id=9,
                current_topic_id=7,
            )
            campaign = FollowupCampaign(
                name="Тестовая цепочка",
                is_active=True,
                include_main_dialogue=False,
                topics=[topic],
                quiet_start_minute=0,
                quiet_end_minute=0,
            )
            campaign.steps = [
                FollowupStep(
                    sort_order=0,
                    delay_minutes=10,
                    message_type="static",
                    message_text="Первое сообщение",
                ),
                FollowupStep(
                    sort_order=1,
                    delay_minutes=20,
                    message_type="static",
                    message_text="Второе сообщение",
                ),
            ]
            session.add_all([
                topic,
                user,
                campaign,
                AutomationConversationState(
                    user_id=42,
                    dialogue_id=9,
                    topic_id=7,
                    current_step="completed",
                    metadata_json='{"profile":{"outcome":"signup"}}',
                ),
            ])
            await session.commit()
            self.campaign_id = campaign.id
            self.first_step_id = campaign.steps[0].id
            self.second_step_id = campaign.steps[1].id
        automation_admin._manual_followup_tests_inflight.clear()

    async def asyncTearDown(self):
        automation_admin._manual_followup_tests_inflight.clear()
        await self.engine.dispose()

    async def _campaign(self):
        async with self.sessions() as session:
            return await session.scalar(select(FollowupCampaign).where(FollowupCampaign.id == self.campaign_id))

    async def _update_campaign(self, **values):
        async with self.sessions() as session:
            item = await session.get(FollowupCampaign, self.campaign_id)
            for key, value in values.items():
                setattr(item, key, value)
            await session.commit()

    async def _update_state(self, **values):
        async with self.sessions() as session:
            state = await session.scalar(
                select(AutomationConversationState).where(
                    AutomationConversationState.user_id == 42,
                    AutomationConversationState.dialogue_id == 9,
                    AutomationConversationState.topic_id == 7,
                )
            )
            for key, value in values.items():
                setattr(state, key, value)
            await session.commit()

    async def _show_self_test(self, state=None, campaign_id=None, user_id=42):
        campaign_id = campaign_id or self.campaign_id
        callback = make_callback(f"followup_self_test_{campaign_id}", user_id=user_id)
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_self_test(callback, state=state)
        return callback

    async def test_campaign_detail_exposes_self_test_and_back(self):
        message = SimpleNamespace(edit_text=AsyncMock())
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin._show_campaign(message, self.campaign_id, edit=True, state=MemoryState())
        markup = message.edit_text.await_args.kwargs["reply_markup"]
        values = callback_data(markup)
        self.assertIn(f"followup_self_test_{self.campaign_id}", values)
        self.assertIn(f"followup_campaign_rename_{self.campaign_id}", values)

    async def test_opening_self_test_uses_callback_user_and_loads_real_context(self):
        state = MemoryState({"followup_return_topic_id": 7})
        callback = await self._show_self_test(state=state, user_id=42)
        text = callback.message.edit_text.await_args.args[0]
        markup = callback.message.edit_text.await_args.kwargs["reply_markup"]
        self.assertIn("ID 42", text)
        self.assertNotIn("ID 999", text)
        self.assertIn("Диалог: 9", text)
        self.assertIn("Самооценка", text)
        self.assertIn("Этап: completed", text)
        self.assertIn(f"followup_self_test_send_{self.campaign_id}", callback_data(markup))
        self.assertIsNone(await state.get_state())
        self.assertEqual((await state.get_data())["followup_return_topic_id"], 7)

    async def test_open_rename_preserves_topic_origin_and_sets_dedicated_state(self):
        state = MemoryState({"followup_return_topic_id": 7})
        callback = make_callback(f"followup_campaign_rename_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_campaign_rename(callback, state)
        self.assertEqual(
            automation_admin.AutomationAdminStates.followup_campaign_rename.state,
            await state.get_state(),
        )
        data = await state.get_data()
        self.assertEqual((self.campaign_id, 7), (data["campaign_id"], data["followup_return_topic_id"]))
        self.assertIn(
            f"followup_campaign_{self.campaign_id}",
            callback_data(callback.message.edit_text.await_args.kwargs["reply_markup"]),
        )

    async def test_rename_rejects_invalid_name_and_successfully_returns_to_same_campaign(self):
        state = MemoryState({
            "campaign_id": self.campaign_id,
            "followup_return_topic_id": 7,
        })
        await state.set_state(automation_admin.AutomationAdminStates.followup_campaign_rename)
        invalid = SimpleNamespace(text=" x ", answer=AsyncMock())
        await automation_admin.followup_campaign_rename_received(invalid, state)
        invalid.answer.assert_awaited_once_with("Название должно содержать от 2 до 100 символов.")
        self.assertEqual(
            automation_admin.AutomationAdminStates.followup_campaign_rename.state,
            await state.get_state(),
        )

        message = SimpleNamespace(text="  Новое имя  ", answer=AsyncMock())
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_campaign_rename_received(message, state)
        async with self.sessions() as session:
            campaign = await session.get(FollowupCampaign, self.campaign_id)
        self.assertEqual("Новое имя", campaign.name)
        self.assertEqual(2, message.answer.await_count)
        self.assertIn("Новое имя", message.answer.await_args_list[-1].args[0])
        self.assertIsNone(await state.get_state())
        self.assertEqual(7, (await state.get_data())["followup_return_topic_id"])

    async def test_rename_back_keeps_name_and_clears_stale_fsm(self):
        state = MemoryState({"followup_return_topic_id": 7})
        rename = make_callback(f"followup_campaign_rename_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_campaign_rename(rename, state)
        back = make_callback(f"followup_campaign_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_campaign_view(back, state=state)
        async with self.sessions() as session:
            campaign = await session.get(FollowupCampaign, self.campaign_id)
        self.assertEqual("Тестовая цепочка", campaign.name)
        self.assertIsNone(await state.get_state())
        self.assertEqual(7, (await state.get_data())["followup_return_topic_id"])
        self.assertIn("Тестовая цепочка", back.message.edit_text.await_args.args[0])

    async def test_ineligible_stage_metadata_and_stop_hide_send(self):
        cases = [
            ({"stage_mode": "selected", "stage_values": "other"}, {}, "этап не подходит"),
            (
                {
                    "metadata_field_path": "profile.outcome",
                    "metadata_operator": "equals",
                    "metadata_expected_value": "other",
                },
                {},
                "метаданные не подходят",
            ),
            ({"stop_events": "CRISIS_DETECTED"}, {"event": "CRISIS_DETECTED"}, "найдено событие остановки"),
        ]
        for campaign_values, extra, expected_reason in cases:
            await self._update_campaign(**campaign_values)
            if extra:
                async with self.sessions() as session:
                    session.add(AutomationEvent(
                        user_id=42,
                        dialogue_id=9,
                        topic_id=7,
                        name=extra["event"],
                    ))
                    await session.commit()
            callback = await self._show_self_test()
            text = callback.message.edit_text.await_args.args[0]
            markup = callback.message.edit_text.await_args.kwargs["reply_markup"]
            self.assertIn(expected_reason, text)
            self.assertNotIn(f"followup_self_test_send_{self.campaign_id}", callback_data(markup))
            if extra:
                async with self.sessions() as session:
                    await session.execute(
                        AutomationEvent.__table__.delete().where(AutomationEvent.name == extra["event"])
                    )
                    await session.commit()
            await self._update_campaign(
                stage_mode="all",
                stage_values="",
                metadata_field_path=None,
                metadata_operator=None,
                metadata_expected_value=None,
                stop_events="",
            )

    async def test_refresh_rechecks_current_db_state(self):
        callback = await self._show_self_test()
        self.assertIn(
            f"followup_self_test_send_{self.campaign_id}",
            callback_data(callback.message.edit_text.await_args.kwargs["reply_markup"]),
        )
        await self._update_campaign(stage_mode="selected", stage_values="allowed")
        await self._update_state(current_step="blocked")
        refreshed = await self._show_self_test()
        self.assertNotIn(
            f"followup_self_test_send_{self.campaign_id}",
            callback_data(refreshed.message.edit_text.await_args.kwargs["reply_markup"]),
        )
        self.assertIn("этап не подходит", refreshed.message.edit_text.await_args.args[0])

    async def test_self_test_uses_selected_stage_with_unset_eligibility(self):
        await self._update_campaign(
            stage_mode="selected",
            stage_values="guide_choice",
            stage_include_unset=True,
        )
        await self._update_state(current_step="  \t")
        unset = await self._show_self_test()
        unset_text = unset.message.edit_text.await_args.args[0]
        unset_markup = unset.message.edit_text.await_args.kwargs["reply_markup"]
        self.assertIn("На выбранных этапах: guide_choice\n+ если этап не задан", unset_text)
        self.assertIn(f"followup_self_test_send_{self.campaign_id}", callback_data(unset_markup))

        await self._update_state(current_step="guide_choice")
        selected = await self._show_self_test()
        self.assertIn(
            f"followup_self_test_send_{self.campaign_id}",
            callback_data(selected.message.edit_text.await_args.kwargs["reply_markup"]),
        )

        await self._update_state(current_step="other")
        other = await self._show_self_test()
        self.assertNotIn(
            f"followup_self_test_send_{self.campaign_id}",
            callback_data(other.message.edit_text.await_args.kwargs["reply_markup"]),
        )
        self.assertIn("этап не подходит", other.message.edit_text.await_args.args[0])

    async def test_active_run_selects_current_step_and_without_run_selects_first(self):
        async with self.sessions() as session:
            run = FollowupRun(
                campaign_id=self.campaign_id,
                user_id=42,
                dialogue_id=9,
                topic_id=7,
                next_step_index=1,
                generation=4,
                last_activity_at=datetime.utcnow(),
                due_at=datetime.utcnow() + timedelta(days=2),
                status="active",
            )
            session.add(run)
            await session.commit()
        callback = await self._show_self_test()
        self.assertIn("#2 · через 20 мин", callback.message.edit_text.await_args.args[0])
        async with self.sessions() as session:
            await session.execute(FollowupRun.__table__.delete())
            await session.commit()
        callback = await self._show_self_test()
        self.assertIn("#1 · через 10 мин", callback.message.edit_text.await_args.args[0])

    async def test_static_manual_send_isolated_from_real_runs_and_deliveries(self):
        async with self.sessions() as session:
            selected_run = FollowupRun(
                campaign_id=self.campaign_id,
                user_id=42,
                dialogue_id=9,
                topic_id=7,
                next_step_index=0,
                generation=3,
                last_activity_at=datetime(2026, 8, 20, 10),
                due_at=datetime(2026, 8, 27, 10),
                status="active",
            )
            other_campaign = FollowupCampaign(
                name="Другая цепочка",
                is_active=True,
                include_main_dialogue=False,
                topics=[await session.get(Topic, 7)],
            )
            other_campaign.steps.append(FollowupStep(
                sort_order=0,
                delay_minutes=50,
                message_type="static",
                message_text="Другая",
            ))
            session.add_all([selected_run, other_campaign])
            await session.flush()
            other_run = FollowupRun(
                campaign_id=other_campaign.id,
                user_id=42,
                dialogue_id=9,
                topic_id=7,
                next_step_index=0,
                generation=2,
                last_activity_at=datetime(2026, 8, 20, 10),
                due_at=datetime(2026, 8, 28, 10),
                status="active",
            )
            session.add(other_run)
            await session.commit()
            selected_before = (selected_run.next_step_index, selected_run.generation, selected_run.due_at)
            other_campaign_id = other_campaign.id
        bot = FakeBot()
        callback = make_callback(f"followup_self_test_send_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_self_test_send(callback, bot, state=MemoryState())
        self.assertEqual([(42, "Первое сообщение")], [(item[0], item[1]) for item in bot.sent])
        self.assertEqual(1, callback.answer.await_count)
        self.assertEqual("Проверяю условия…", callback.answer.await_args.args[0])
        callback.message.edit_reply_markup.assert_awaited_once_with(reply_markup=None)
        async with self.sessions() as session:
            selected = await session.scalar(select(FollowupRun).where(FollowupRun.campaign_id == self.campaign_id))
            other = await session.scalar(select(FollowupRun).where(FollowupRun.campaign_id == other_campaign_id))
            deliveries = (await session.execute(select(FollowupDelivery))).scalars().all()
        self.assertEqual(selected_before, (selected.next_step_index, selected.generation, selected.due_at))
        self.assertEqual((0, 2), (other.next_step_index, other.generation))
        self.assertEqual([], deliveries)

    async def test_ai_manual_send_uses_followup_generation_without_persistence(self):
        async with self.sessions() as session:
            step = await session.get(FollowupStep, self.first_step_id)
            step.message_type = "ai"
            step.message_text = None
            step.ai_instruction = "Напомни мягко"
            await session.commit()
        ai_response = AsyncMock(return_value="Ответ AI")
        generated = AsyncMock()
        fake_handlers = SimpleNamespace(_send_generated_response=generated)
        bot = FakeBot()
        callback = make_callback(f"followup_self_test_send_{self.campaign_id}")
        with (
            patch.object(automation_admin, "async_session_maker", self.sessions),
            patch("ai_integration.get_ai_response", new=ai_response),
            patch.dict(sys.modules, {"handlers": fake_handlers}),
        ):
            await automation_admin.followup_self_test_send(callback, bot, state=MemoryState())
        self.assertEqual(1, ai_response.await_count)
        kwargs = ai_response.await_args.kwargs
        self.assertEqual("followup", kwargs["request_type"])
        self.assertFalse(kwargs["persist_service_data"])
        self.assertEqual(9, kwargs["dialogue_id_override"])
        generated.assert_awaited_once_with(bot, 42, "Ответ AI")
        async with self.sessions() as session:
            self.assertEqual([], (await session.execute(select(FollowupDelivery))).scalars().all())

    async def test_manual_send_double_click_produces_one_message(self):
        started = asyncio.Event()
        release = asyncio.Event()
        send_mock = AsyncMock()

        async def blocked_send(*args, **kwargs):
            started.set()
            await release.wait()

        send_mock.side_effect = blocked_send
        first = make_callback(f"followup_self_test_send_{self.campaign_id}")
        second = make_callback(f"followup_self_test_send_{self.campaign_id}")
        with (
            patch.object(automation_admin, "async_session_maker", self.sessions),
            patch.object(automation_admin, "send_followup_step", new=send_mock),
            patch.object(automation_admin, "_show_followup_self_test", new=AsyncMock()),
        ):
            task = asyncio.create_task(
                automation_admin.followup_self_test_send(first, FakeBot(), state=MemoryState())
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            await automation_admin.followup_self_test_send(second, FakeBot(), state=MemoryState())
            self.assertEqual("Проверка уже выполняется.", second.answer.await_args.args[0])
            release.set()
            await task
        send_mock.assert_awaited_once()

    async def test_back_from_self_test_returns_to_same_campaign_and_topic_context(self):
        state = MemoryState({"followup_return_topic_id": 7})
        await self._show_self_test(state=state)
        callback = make_callback(f"followup_campaign_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_campaign_view(callback, state=state)
        text = callback.message.edit_text.await_args.args[0]
        markup = callback.message.edit_text.await_args.kwargs["reply_markup"]
        self.assertIn("Тестовая цепочка", text)
        self.assertIn("topic_followup_campaigns_7", callback_data(markup))
        self.assertIsNone(await state.get_state())
        self.assertEqual(7, (await state.get_data())["followup_return_topic_id"])

    async def test_topic_scoped_navigation_keeps_intermediate_campaign_detail(self):
        state = MemoryState()
        topic_list = make_callback("topic_followup_campaigns_7")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.topic_followup_campaigns(topic_list, state=state)
        campaign = make_callback(f"followup_campaign_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_campaign_view(campaign, state=state)
        detail_markup = campaign.message.edit_text.await_args.kwargs["reply_markup"]
        self.assertIn("topic_followup_campaigns_7", callback_data(detail_markup))
        topics = make_callback(f"followup_topics_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_topics(topics, state=state)
        topics_markup = topics.message.edit_text.await_args.kwargs["reply_markup"]
        self.assertIn(f"followup_campaign_{self.campaign_id}", callback_data(topics_markup))
        self.assertNotIn("topic_followup_campaigns_7", callback_data(topics_markup))
        campaign_back = make_callback(f"followup_campaign_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_campaign_view(campaign_back, state=state)
        topic_back = make_callback("topic_followup_campaigns_7")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.topic_followup_campaigns(topic_back, state=state)
        self.assertIn(
            f"followup_campaign_{self.campaign_id}",
            callback_data(topic_back.message.edit_text.await_args.kwargs["reply_markup"]),
        )

    async def test_global_navigation_stays_global_through_topics(self):
        state = MemoryState()
        campaign_list = make_callback("followup_campaigns")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_campaigns(campaign_list, state=state)
        campaign = make_callback(f"followup_campaign_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_campaign_view(campaign, state=state)
        detail_markup = campaign.message.edit_text.await_args.kwargs["reply_markup"]
        self.assertIn("followup_campaigns", callback_data(detail_markup))
        topics = make_callback(f"followup_topics_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_topics(topics, state=state)
        self.assertIn(f"followup_campaign_{self.campaign_id}", callback_data(
            topics.message.edit_text.await_args.kwargs["reply_markup"]
        ))
        campaign_back = make_callback(f"followup_campaign_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_campaign_view(campaign_back, state=state)
        self.assertIn("followup_campaigns", callback_data(
            campaign_back.message.edit_text.await_args.kwargs["reply_markup"]
        ))

    async def test_stage_value_back_chain_invokes_previous_handlers(self):
        state = MemoryState({"followup_return_topic_id": 7})
        conditions = make_callback(f"followup_conditions_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_conditions(conditions, state=state)
        stages = make_callback(f"followup_stage_edit_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_stage_edit(stages, state=state)
        values = make_callback(f"followup_stage_mode_{self.campaign_id}_selected")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_stage_mode(values, state=state)
        self.assertIn(f"followup_stage_edit_{self.campaign_id}", callback_data(
            values.message.edit_text.await_args.kwargs["reply_markup"]
        ))
        stages_back = make_callback(f"followup_stage_edit_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_stage_edit(stages_back, state=state)
        conditions_back = make_callback(f"followup_conditions_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_conditions(conditions_back, state=state)
        campaign_back = make_callback(f"followup_campaign_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_campaign_view(campaign_back, state=state)
        self.assertIn("topic_followup_campaigns_7", callback_data(
            campaign_back.message.edit_text.await_args.kwargs["reply_markup"]
        ))
        self.assertIsNone(await state.get_state())

    async def test_all_except_back_preserves_selected_unset_configuration(self):
        await self._update_campaign(
            stage_mode="selected",
            stage_values="guide_choice",
            stage_include_unset=True,
        )
        state = MemoryState({"followup_return_topic_id": 7})
        await state.set_state(automation_admin.AutomationAdminStates.followup_stage_values)
        mode_callback = make_callback(f"followup_stage_mode_{self.campaign_id}_all_except")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_stage_mode(mode_callback, state)

        campaign = await self._campaign()
        self.assertEqual(("selected", "guide_choice", True), (
            campaign.stage_mode,
            campaign.stage_values,
            campaign.stage_include_unset,
        ))
        self.assertEqual(
            automation_admin.AutomationAdminStates.followup_stage_values.state,
            await state.get_state(),
        )

        back = make_callback(f"followup_stage_edit_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_stage_edit(back, state=state)

        campaign = await self._campaign()
        self.assertEqual(("selected", "guide_choice", True), (
            campaign.stage_mode,
            campaign.stage_values,
            campaign.stage_include_unset,
        ))
        self.assertIsNone(await state.get_state())
        self.assertEqual({"followup_return_topic_id": 7}, await state.get_data())

    async def test_metadata_back_chain_invokes_previous_handlers(self):
        state = MemoryState({"followup_return_topic_id": 7})
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_conditions(
                make_callback(f"followup_conditions_{self.campaign_id}"), state=state
            )
            field = make_callback(f"followup_metadata_edit_{self.campaign_id}")
            await automation_admin.followup_metadata_edit(field, state)
            field_message = SimpleNamespace(text="profile.outcome", answer=AsyncMock())
            await automation_admin.followup_metadata_field_received(field_message, state)
            self.assertIn(f"followup_metadata_edit_{self.campaign_id}", callback_data(
                field_message.answer.await_args.kwargs["reply_markup"]
            ))
            operator = make_callback(f"followup_metadata_operator_{self.campaign_id}_equals")
            await automation_admin.followup_metadata_operator(operator, state)
            self.assertIn(f"followup_metadata_operator_edit_{self.campaign_id}", callback_data(
                operator.message.edit_text.await_args.kwargs["reply_markup"]
            ))
            operator_back = make_callback(f"followup_metadata_operator_edit_{self.campaign_id}")
            await automation_admin.followup_metadata_operator_edit(operator_back, state)
            self.assertIn(f"followup_metadata_edit_{self.campaign_id}", callback_data(
                operator_back.message.edit_text.await_args.kwargs["reply_markup"]
            ))
            field_back = make_callback(f"followup_metadata_edit_{self.campaign_id}")
            await automation_admin.followup_metadata_edit(field_back, state)
            conditions_back = make_callback(f"followup_conditions_{self.campaign_id}")
            await automation_admin.followup_conditions(conditions_back, state=state)
            campaign_back = make_callback(f"followup_campaign_{self.campaign_id}")
            await automation_admin.followup_campaign_view(campaign_back, state=state)
        self.assertIn("topic_followup_campaigns_7", callback_data(
            campaign_back.message.edit_text.await_args.kwargs["reply_markup"]
        ))
        self.assertIsNone(await state.get_state())

    async def test_metadata_back_restores_fsm_states_and_clears_operator(self):
        state = MemoryState({"followup_return_topic_id": 7})
        await automation_admin.followup_metadata_edit(
            make_callback(f"followup_metadata_edit_{self.campaign_id}"), state
        )
        await automation_admin.followup_metadata_field_received(
            SimpleNamespace(text="profile.outcome", answer=AsyncMock()), state
        )
        self.assertEqual(
            automation_admin.AutomationAdminStates.followup_metadata_operator.state,
            await state.get_state(),
        )
        await automation_admin.followup_metadata_operator(
            make_callback(f"followup_metadata_operator_{self.campaign_id}_equals"), state
        )
        self.assertEqual(
            automation_admin.AutomationAdminStates.followup_metadata_value.state,
            await state.get_state(),
        )
        await automation_admin.followup_metadata_operator_edit(
            make_callback(f"followup_metadata_operator_edit_{self.campaign_id}"), state
        )
        data = await state.get_data()
        self.assertEqual(
            automation_admin.AutomationAdminStates.followup_metadata_operator.state,
            await state.get_state(),
        )
        self.assertEqual(self.campaign_id, data["campaign_id"])
        self.assertEqual("profile.outcome", data["metadata_field_path"])
        self.assertEqual(7, data["followup_return_topic_id"])
        self.assertIsNone(data.get("metadata_operator"))
        await automation_admin.followup_metadata_edit(
            make_callback(f"followup_metadata_edit_{self.campaign_id}"), state
        )
        data = await state.get_data()
        self.assertEqual(
            automation_admin.AutomationAdminStates.followup_metadata_field.state,
            await state.get_state(),
        )
        self.assertEqual(self.campaign_id, data["campaign_id"])
        self.assertEqual(7, data["followup_return_topic_id"])

    async def test_metadata_back_navigation_is_routed_by_dispatcher(self):
        class DispatcherBot:
            id = 999

            def __init__(self):
                self.methods = []

            async def __call__(self, method):
                self.methods.append(method)
                return True

        def callback_update(update_id, data):
            telegram_user = TelegramUser(
                id=42,
                is_bot=False,
                first_name="Админ",
                username="admin",
            )
            message = TelegramMessage(
                message_id=update_id,
                date=datetime.utcnow(),
                chat=Chat(id=42, type="private"),
                from_user=telegram_user,
            )
            return Update(
                update_id=update_id,
                callback_query=CallbackQuery(
                    id=f"dispatcher-{update_id}",
                    from_user=telegram_user,
                    chat_instance="dispatcher-test",
                    message=message,
                    data=data,
                ),
            )

        def message_update(update_id, text):
            telegram_user = TelegramUser(
                id=42,
                is_bot=False,
                first_name="Админ",
                username="admin",
            )
            message = TelegramMessage(
                message_id=update_id,
                date=datetime.utcnow(),
                chat=Chat(id=42, type="private"),
                from_user=telegram_user,
                text=text,
            )
            return Update(update_id=update_id, message=message)

        dispatcher = Dispatcher(storage=MemoryStorage())
        dispatcher.include_router(automation_admin.router)
        bot = DispatcherBot()
        with (
            patch.object(automation_admin, "async_session_maker", self.sessions),
            patch.object(automation_admin, "get_all_admin_ids", new=AsyncMock(return_value={42})),
        ):
            await dispatcher.feed_update(bot, callback_update(1, f"followup_conditions_{self.campaign_id}"))
            await dispatcher.feed_update(bot, callback_update(2, f"followup_metadata_edit_{self.campaign_id}"))
            context = dispatcher.fsm.get_context(bot, chat_id=42, user_id=42)
            self.assertEqual(
                automation_admin.AutomationAdminStates.followup_metadata_field.state,
                await context.get_state(),
            )
            await dispatcher.feed_update(bot, message_update(3, "profile.outcome"))
            context = dispatcher.fsm.get_context(bot, chat_id=42, user_id=42)
            self.assertEqual(
                automation_admin.AutomationAdminStates.followup_metadata_operator.state,
                await context.get_state(),
            )
            await dispatcher.feed_update(
                bot,
                callback_update(4, f"followup_metadata_operator_{self.campaign_id}_equals"),
            )
            context = dispatcher.fsm.get_context(bot, chat_id=42, user_id=42)
            self.assertEqual(
                automation_admin.AutomationAdminStates.followup_metadata_value.state,
                await context.get_state(),
            )
            await dispatcher.feed_update(
                bot,
                callback_update(5, f"followup_metadata_operator_edit_{self.campaign_id}"),
            )
            context = dispatcher.fsm.get_context(bot, chat_id=42, user_id=42)
            self.assertEqual(
                automation_admin.AutomationAdminStates.followup_metadata_operator.state,
                await context.get_state(),
            )
            await dispatcher.feed_update(
                bot,
                callback_update(6, f"followup_metadata_edit_{self.campaign_id}"),
            )
            context = dispatcher.fsm.get_context(bot, chat_id=42, user_id=42)
            self.assertEqual(
                automation_admin.AutomationAdminStates.followup_metadata_field.state,
                await context.get_state(),
            )
        rendered_texts = [getattr(method, "text", None) for method in bot.methods]
        self.assertTrue(any(text and "Введите путь поля" in text for text in rendered_texts))
        self.assertTrue(any(text and "Выберите оператор" in text for text in rendered_texts))

    async def test_explicit_delete_removes_unsent_step_without_touching_other_steps(self):
        callback = make_callback(f"followup_step_delete_{self.campaign_id}_{self.second_step_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_step_delete(callback, state=MemoryState())
        async with self.sessions() as session:
            remaining = (
                await session.execute(
                    select(FollowupStep)
                    .where(FollowupStep.campaign_id == self.campaign_id)
                    .order_by(FollowupStep.sort_order)
                )
            ).scalars().all()
        self.assertEqual([self.first_step_id], [step.id for step in remaining])
        self.assertEqual([0], [step.sort_order for step in remaining])

    async def test_claimed_attempt_blocks_step_delete(self):
        async with self.sessions() as session:
            run = FollowupRun(
                campaign_id=self.campaign_id,
                user_id=42,
                dialogue_id=9,
                topic_id=7,
                due_at=datetime.utcnow(),
            )
            session.add(run)
            await session.flush()
            session.add(FollowupDeliveryAttempt(
                run_id=run.id,
                step_id=self.first_step_id,
                step_index=0,
                generation=1,
                claim_token="claimed-step-delete",
                status="claimed",
            ))
            await session.commit()

        callback = make_callback(f"followup_step_delete_{self.campaign_id}_{self.first_step_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_step_delete(callback, state=MemoryState())

        async with self.sessions() as session:
            self.assertIsNotNone(await session.get(FollowupStep, self.first_step_id))
            attempt = await session.scalar(select(FollowupDeliveryAttempt))
        self.assertEqual("claimed", attempt.status)

    async def test_uncertain_attempt_blocks_step_delete(self):
        async with self.sessions() as session:
            run = FollowupRun(
                campaign_id=self.campaign_id,
                user_id=42,
                dialogue_id=9,
                topic_id=7,
                due_at=datetime.utcnow(),
            )
            session.add(run)
            await session.flush()
            session.add(FollowupDeliveryAttempt(
                run_id=run.id,
                step_id=self.first_step_id,
                step_index=0,
                generation=1,
                claim_token="uncertain-step-delete",
                status="uncertain",
            ))
            await session.commit()

        callback = make_callback(f"followup_step_delete_{self.campaign_id}_{self.first_step_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_step_delete(callback, state=MemoryState())

        async with self.sessions() as session:
            self.assertIsNotNone(await session.get(FollowupStep, self.first_step_id))
            attempt = await session.scalar(select(FollowupDeliveryAttempt))
        self.assertEqual("uncertain", attempt.status)

    async def test_delivered_step_blocks_step_delete(self):
        async with self.sessions() as session:
            run = FollowupRun(
                campaign_id=self.campaign_id,
                user_id=42,
                dialogue_id=9,
                topic_id=7,
                due_at=datetime.utcnow(),
            )
            session.add(run)
            await session.flush()
            session.add(FollowupDelivery(
                run_id=run.id,
                step_id=self.first_step_id,
                generation=1,
            ))
            await session.commit()

        callback = make_callback(f"followup_step_delete_{self.campaign_id}_{self.first_step_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_step_delete(callback, state=MemoryState())

        async with self.sessions() as session:
            self.assertIsNotNone(await session.get(FollowupStep, self.first_step_id))
            self.assertEqual(1, await session.scalar(select(FollowupDelivery.id).where(
                FollowupDelivery.step_id == self.first_step_id
            )))

    async def test_existing_step_opens_detail_and_static_edit_preserves_type_order_and_delivery(self):
        list_callback = make_callback(f"followup_steps_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_steps(list_callback, state=MemoryState())
        values = callback_data(list_callback.message.edit_text.await_args.kwargs["reply_markup"])
        self.assertIn(f"followup_step_{self.campaign_id}_{self.first_step_id}", values)
        self.assertNotIn(f"followup_step_delete_{self.campaign_id}_{self.first_step_id}", values)

        detail_callback = make_callback(f"followup_step_{self.campaign_id}_{self.first_step_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_step_detail(detail_callback, state=MemoryState())
        detail_text = detail_callback.message.edit_text.await_args.args[0]
        detail_values = callback_data(detail_callback.message.edit_text.await_args.kwargs["reply_markup"])
        self.assertIn("Тип: <b>static</b>", detail_text)
        self.assertIn(f"followup_step_edit_{self.campaign_id}_{self.first_step_id}", detail_values)
        self.assertIn(f"followup_step_delete_{self.campaign_id}_{self.first_step_id}", detail_values)

        state = MemoryState()
        edit_callback = make_callback(f"followup_step_edit_{self.campaign_id}_{self.first_step_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_step_edit(edit_callback, state)
            await automation_admin.followup_step_edit_received(
                SimpleNamespace(text="15\nОбновлённый текст", answer=AsyncMock()),
                state,
            )
        async with self.sessions() as session:
            step = await session.get(FollowupStep, self.first_step_id)
        self.assertEqual((15, "Обновлённый текст", "static", 0), (
            step.delay_minutes,
            step.message_text,
            step.message_type,
            step.sort_order,
        ))

        detail_back = make_callback(f"followup_step_{self.campaign_id}_{self.first_step_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_step_detail(detail_back, state=state)
        steps_back = make_callback(f"followup_steps_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_steps(steps_back, state=state)
        self.assertIn(
            f"followup_step_{self.campaign_id}_{self.first_step_id}",
            callback_data(steps_back.message.edit_text.await_args.kwargs["reply_markup"]),
        )

        async with self.sessions() as session:
            run = FollowupRun(
                campaign_id=self.campaign_id,
                user_id=42,
                dialogue_id=9,
                topic_id=7,
                due_at=datetime.utcnow(),
            )
            session.add(run)
            await session.flush()
            session.add(FollowupDelivery(run_id=run.id, step_id=self.first_step_id, generation=1))
            await session.commit()
        delete_callback = make_callback(f"followup_step_delete_{self.campaign_id}_{self.first_step_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_step_delete(delete_callback, state=MemoryState())
        async with self.sessions() as session:
            self.assertIsNotNone(await session.get(FollowupStep, self.first_step_id))
            self.assertEqual(1, len((await session.execute(select(FollowupDelivery))).scalars().all()))

    async def test_ai_step_edit_preserves_type_and_sort_order(self):
        async with self.sessions() as session:
            step = await session.get(FollowupStep, self.second_step_id)
            step.message_type = "ai"
            step.message_text = None
            step.ai_instruction = "Старая инструкция"
            await session.commit()
        state = MemoryState()
        callback = make_callback(f"followup_step_edit_{self.campaign_id}_{self.second_step_id}")
        message = SimpleNamespace(text="25\nНовая инструкция", answer=AsyncMock())
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_step_edit(callback, state)
            await automation_admin.followup_step_edit_received(message, state)
        async with self.sessions() as session:
            step = await session.get(FollowupStep, self.second_step_id)
        self.assertEqual((25, "Новая инструкция", "ai", 1), (
            step.delay_minutes,
            step.ai_instruction,
            step.message_type,
            step.sort_order,
        ))

    async def test_conditions_editor_navigation_saves_and_clears_values(self):
        state = MemoryState({"followup_return_topic_id": 7})
        callback = make_callback(f"followup_conditions_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_conditions(callback, state=state)
        text = callback.message.edit_text.await_args.args[0]
        markup = callback.message.edit_text.await_args.kwargs["reply_markup"]
        self.assertIn("На всех этапах", text)
        self.assertIn("не заданы", text)
        self.assertIn(f"followup_stage_edit_{self.campaign_id}", callback_data(markup))

        stage_callback = make_callback(f"followup_stage_edit_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_stage_edit(stage_callback, state)
        texts = button_texts(stage_callback.message.edit_text.await_args.kwargs["reply_markup"])
        self.assertTrue(all(any(label in text for text in texts)
                            for label in automation_admin.FOLLOWUP_STAGE_MODE_LABELS.values()))

        mode_callback = make_callback(f"followup_stage_mode_{self.campaign_id}_selected")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_stage_mode(mode_callback, state)
            await automation_admin.followup_stage_values_received(
                SimpleNamespace(text=" completed, , active ", answer=AsyncMock()),
                state,
            )
        campaign = await self._campaign()
        self.assertEqual(("selected", "completed, active"), (campaign.stage_mode, campaign.stage_values))

        selected_conditions = make_callback(f"followup_conditions_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_conditions(selected_conditions, state=state)
        selected_text = selected_conditions.message.edit_text.await_args.args[0]
        selected_markup = selected_conditions.message.edit_text.await_args.kwargs["reply_markup"]
        self.assertIn("На выбранных этапах: completed, active", selected_text)
        self.assertIn(
            f"followup_stage_include_unset_{self.campaign_id}",
            callback_data(selected_markup),
        )
        self.assertIn("☐ Также если этап не задан", button_texts(selected_markup))

        toggle = make_callback(f"followup_stage_include_unset_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_stage_include_unset_toggle(toggle, state=state)
        campaign = await self._campaign()
        self.assertTrue(campaign.stage_include_unset)
        toggle_text = toggle.message.edit_text.await_args.args[0]
        self.assertIn("+ если этап не задан", toggle_text)
        self.assertIn("✅ Также если этап не задан", button_texts(
            toggle.message.edit_text.await_args.kwargs["reply_markup"]
        ))

        toggle_off = make_callback(f"followup_stage_include_unset_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_stage_include_unset_toggle(toggle_off, state=state)
        self.assertFalse((await self._campaign()).stage_include_unset)

        await self._update_campaign(stage_include_unset=True)
        switch_all = make_callback(f"followup_stage_mode_{self.campaign_id}_all")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_stage_mode(switch_all, state)
        campaign = await self._campaign()
        self.assertEqual(("all", "", False), (
            campaign.stage_mode,
            campaign.stage_values,
            campaign.stage_include_unset,
        ))

        field_message = SimpleNamespace(text="profile.outcome", answer=AsyncMock())
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_metadata_edit(
                make_callback(f"followup_metadata_edit_{self.campaign_id}"), state
            )
            await automation_admin.followup_metadata_field_received(field_message, state)
            operator_callback = make_callback(f"followup_metadata_operator_{self.campaign_id}_not_equals")
            await automation_admin.followup_metadata_operator(operator_callback, state)
            await automation_admin.followup_metadata_value_received(
                SimpleNamespace(text="signup", answer=AsyncMock()),
                state,
            )
        campaign = await self._campaign()
        self.assertEqual(
            ("profile.outcome", "not_equals", "signup"),
            (campaign.metadata_field_path, campaign.metadata_operator, campaign.metadata_expected_value),
        )

        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_stop_events_edit(
                make_callback(f"followup_stop_events_edit_{self.campaign_id}"), state
            )
            await automation_admin.followup_stop_events_received(
                SimpleNamespace(text="CRISIS_DETECTED, DIALOG_COMPLETED", answer=AsyncMock()),
                state,
            )
        campaign = await self._campaign()
        self.assertEqual("CRISIS_DETECTED, DIALOG_COMPLETED", campaign.stop_events)

        clear_callback = make_callback(f"followup_metadata_clear_{self.campaign_id}")
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.followup_metadata_clear(clear_callback, state)
            await automation_admin.followup_stop_events_clear(
                make_callback(f"followup_stop_events_clear_{self.campaign_id}"), state
            )
        campaign = await self._campaign()
        self.assertIsNone(campaign.metadata_field_path)
        self.assertEqual("", campaign.stop_events)
        self.assertIsNone(await state.get_state())


if __name__ == "__main__":
    unittest.main()
