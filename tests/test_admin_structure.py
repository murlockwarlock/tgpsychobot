import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

import automation_admin
import automation_engine
import keyboards
from database import (
    AutomationCondition,
    AutomationConversationState,
    AutomationHandler,
    AutomationStepTransition,
    Base,
    Content,
    FollowupCampaign,
    Topic,
    User,
)


def callback_values(markup):
    return {
        button.callback_data
        for row in markup.inline_keyboard
        for button in row
        if button.callback_data
    }


class MemoryState:
    def __init__(self):
        self.data = {}

    async def clear(self):
        self.data.clear()

    async def update_data(self, **values):
        self.data.update(values)

    async def get_data(self):
        return dict(self.data)


class AdminStructureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        await self.engine.dispose()

    def test_main_admin_has_general_settings_and_automations(self):
        callbacks = callback_values(keyboards.admin_panel_keyboard())
        self.assertIn("admin_general_settings", callbacks)
        self.assertIn("automation_menu", callbacks)

    async def test_compact_algorithm_summary_counts_unique_users(self):
        async with self.sessions() as session:
            session.add(User(id=42, first_name="Иван"))
            session.add_all([
                AutomationStepTransition(
                    user_id=42, dialogue_id=1, topic_id=0, current_step="START", state_json="{}"
                ),
                AutomationStepTransition(
                    user_id=42, dialogue_id=2, topic_id=0, current_step="RESULT", state_json="{}"
                ),
                AutomationConversationState(
                    user_id=42,
                    dialogue_id=1,
                    topic_id=0,
                    current_step="START",
                    current_state_json="{}",
                ),
                AutomationConversationState(
                    user_id=42,
                    dialogue_id=2,
                    topic_id=0,
                    current_step="RESULT",
                    current_state_json="{}",
                ),
            ])
            await session.commit()

        async with self.sessions() as session:
            summary = await automation_engine.get_automation_summary(session)

        self.assertEqual(summary, {"users": 1, "current_users": 1, "transitions": 2})

    def test_topic_editor_has_all_three_automation_entries(self):
        callbacks = callback_values(keyboards.edit_topic_keyboard(17, True))

        self.assertIn("topic_automation_handlers_17", callbacks)
        self.assertIn("topic_followup_campaigns_17", callbacks)
        self.assertIn("topic_automation_stats_17", callbacks)

    def test_topic_filters_include_direct_and_all_topic_objects(self):
        topic_1 = SimpleNamespace(id=1)
        topic_2 = SimpleNamespace(id=2)
        direct = SimpleNamespace(all_topics=False, topics=[topic_1])
        global_item = SimpleNamespace(all_topics=True, topics=[])
        foreign = SimpleNamespace(all_topics=False, topics=[topic_2])

        self.assertTrue(automation_admin._handler_applies_to_topic(direct, 1))
        self.assertTrue(automation_admin._handler_applies_to_topic(global_item, 1))
        self.assertFalse(automation_admin._handler_applies_to_topic(foreign, 1))
        self.assertTrue(automation_admin._campaign_applies_to_topic(direct, 1))
        self.assertTrue(automation_admin._campaign_applies_to_topic(global_item, 1))
        self.assertFalse(automation_admin._campaign_applies_to_topic(foreign, 1))

    def test_profile_settings_are_not_in_test_menu(self):
        config = SimpleNamespace(
            is_enabled=True,
            show_progress=True,
            secret_test_enabled=True,
            formulas_enabled=False,
            interpretation_input_mode="all",
            separate_result_prompt_enabled=False,
            result_prompt_is_final=False,
        )
        callbacks = callback_values(keyboards.admin_test_menu_keyboard(config))

        self.assertFalse(any(value.startswith("admin_test_toggle_profile_") for value in callbacks))
        self.assertIn("edit_content_test_intro", callbacks)
        self.assertIn("edit_content_test_results", callbacks)
        self.assertIn("edit_content_secret_test_outro", callbacks)

    async def test_test_content_is_hidden_from_general_content_section(self):
        async with self.sessions() as session:
            session.add_all([
                Content(key="start_message", text_content="start", is_visible=True),
                Content(key="disclaimer", text_content="rules", is_visible=True),
                Content(key="about", button_title="О проекте", text_content="about", is_visible=True),
                Content(key="test_button", button_title="Пройти тест", text_content="test", is_visible=True),
                Content(key="test_intro", button_title="Приветствие теста", text_content="intro", is_visible=True),
                Content(key="test_results", button_title="Результаты теста", text_content="result", is_visible=True),
                Content(key="secret_test_outro", button_title="Финал теста", text_content="final", is_visible=True),
            ])
            await session.commit()

        with patch.object(keyboards, "async_session_maker", self.sessions):
            markup = await keyboards.content_management_keyboard()
        callbacks = callback_values(markup)

        self.assertIn("edit_content_about", callbacks)
        self.assertNotIn("edit_content_test_button", callbacks)
        self.assertNotIn("edit_content_test_intro", callbacks)
        self.assertNotIn("edit_content_test_results", callbacks)
        self.assertNotIn("edit_content_secret_test_outro", callbacks)

    async def test_handler_created_inside_topic_is_bound_only_to_that_topic(self):
        async with self.sessions() as session:
            session.add(Topic(id=7, name="Самооценка", is_active=True))
            await session.commit()
        state = SimpleNamespace(
            get_data=AsyncMock(return_value={"preset_topic_id": 7}),
            clear=AsyncMock(),
            update_data=AsyncMock(),
        )
        message = SimpleNamespace(text="Лид готов", answer=AsyncMock())

        with (
            patch.object(automation_admin, "async_session_maker", self.sessions),
            patch.object(automation_admin, "_show_handler", AsyncMock()),
        ):
            await automation_admin.automation_handler_name_received(message, state)

        async with self.sessions() as session:
            item = await session.scalar(
                select(AutomationHandler).options(selectinload(AutomationHandler.topics))
            )
        self.assertFalse(item.include_main_dialogue)
        self.assertEqual([topic.id for topic in item.topics], [7])

    async def test_topic_handler_list_is_filtered_and_returns_to_topic(self):
        async with self.sessions() as session:
            topic_1 = Topic(id=1, name="Самооценка", is_active=True)
            topic_2 = Topic(id=2, name="Тревожность", is_active=True)
            direct = AutomationHandler(name="Только тема", topics=[topic_1])
            global_item = AutomationHandler(name="Все темы", all_topics=True)
            foreign = AutomationHandler(name="Другая тема", topics=[topic_2])
            session.add_all([direct, global_item, foreign])
            await session.commit()
            direct_id, global_id, foreign_id = direct.id, global_item.id, foreign.id
        message = SimpleNamespace(edit_text=AsyncMock())
        callback = SimpleNamespace(message=message)

        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin._show_automation_handlers(callback, topic_id=1)

        callbacks = callback_values(message.edit_text.await_args.kwargs["reply_markup"])
        self.assertIn(f"automation_handler_{direct_id}", callbacks)
        self.assertIn(f"automation_handler_{global_id}", callbacks)
        self.assertNotIn(f"automation_handler_{foreign_id}", callbacks)
        self.assertIn(f"topic_automation_handler_unlink_1_{direct_id}", callbacks)
        self.assertNotIn(f"topic_automation_handler_unlink_1_{global_id}", callbacks)
        self.assertIn("topic_automation_handler_add_1", callbacks)
        self.assertIn("edit_topic_1", callbacks)

    async def test_unlink_handler_keeps_it_globally_and_disables_orphan(self):
        async with self.sessions() as session:
            topic = Topic(id=1, name="Самооценка", is_active=True)
            item = AutomationHandler(
                name="Только тема",
                topics=[topic],
                is_active=True,
                include_main_dialogue=False,
            )
            session.add(item)
            await session.commit()
            handler_id = item.id
        callback = SimpleNamespace(
            data=f"topic_automation_handler_unlink_1_{handler_id}",
            answer=AsyncMock(),
            message=SimpleNamespace(edit_text=AsyncMock()),
        )

        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.topic_automation_handler_unlink(callback)

        async with self.sessions() as session:
            item = await session.scalar(
                select(AutomationHandler).options(selectinload(AutomationHandler.topics))
            )
        self.assertIsNotNone(item)
        self.assertEqual(item.topics, [])
        self.assertFalse(item.is_active)

    async def test_followup_created_inside_topic_is_bound_only_to_that_topic(self):
        async with self.sessions() as session:
            session.add(Topic(id=8, name="Тревожность", is_active=True))
            await session.commit()
        state = SimpleNamespace(
            get_data=AsyncMock(return_value={"preset_topic_id": 8}),
            clear=AsyncMock(),
            update_data=AsyncMock(),
        )
        message = SimpleNamespace(text="Вернуть к упражнению", answer=AsyncMock())

        with (
            patch.object(automation_admin, "async_session_maker", self.sessions),
            patch.object(automation_admin, "_show_campaign", AsyncMock()),
        ):
            await automation_admin.followup_campaign_name_received(message, state)

        async with self.sessions() as session:
            item = await session.scalar(
                select(FollowupCampaign).options(selectinload(FollowupCampaign.topics))
            )
        self.assertFalse(item.include_main_dialogue)
        self.assertEqual([topic.id for topic in item.topics], [8])

    async def test_topic_followup_list_is_filtered_and_returns_to_topic(self):
        async with self.sessions() as session:
            topic_1 = Topic(id=1, name="Самооценка", is_active=True)
            topic_2 = Topic(id=2, name="Тревожность", is_active=True)
            direct = FollowupCampaign(name="Только тема", topics=[topic_1])
            global_item = FollowupCampaign(name="Все темы", all_topics=True)
            foreign = FollowupCampaign(name="Другая тема", topics=[topic_2])
            session.add_all([direct, global_item, foreign])
            await session.commit()
            direct_id, global_id, foreign_id = direct.id, global_item.id, foreign.id
        message = SimpleNamespace(edit_text=AsyncMock())
        callback = SimpleNamespace(message=message)

        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin._show_followup_campaigns(callback, topic_id=1)

        callbacks = callback_values(message.edit_text.await_args.kwargs["reply_markup"])
        self.assertIn(f"followup_campaign_{direct_id}", callbacks)
        self.assertIn(f"followup_campaign_{global_id}", callbacks)
        self.assertNotIn(f"followup_campaign_{foreign_id}", callbacks)
        self.assertIn(f"topic_followup_campaign_unlink_1_{direct_id}", callbacks)
        self.assertNotIn(f"topic_followup_campaign_unlink_1_{global_id}", callbacks)
        self.assertIn("topic_followup_campaign_add_1", callbacks)
        self.assertIn("edit_topic_1", callbacks)

    async def test_unlink_followup_keeps_it_globally_and_disables_orphan(self):
        async with self.sessions() as session:
            topic = Topic(id=1, name="Самооценка", is_active=True)
            item = FollowupCampaign(
                name="Только тема",
                topics=[topic],
                is_active=True,
                include_main_dialogue=False,
            )
            session.add(item)
            await session.commit()
            campaign_id = item.id
        callback = SimpleNamespace(
            data=f"topic_followup_campaign_unlink_1_{campaign_id}",
            answer=AsyncMock(),
            message=SimpleNamespace(edit_text=AsyncMock()),
        )

        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.topic_followup_campaign_unlink(callback)

        async with self.sessions() as session:
            item = await session.scalar(
                select(FollowupCampaign).options(selectinload(FollowupCampaign.topics))
            )
        self.assertIsNotNone(item)
        self.assertEqual(item.topics, [])
        self.assertFalse(item.is_active)

    async def test_topic_statistics_do_not_include_other_topics(self):
        async with self.sessions() as session:
            session.add_all([
                User(id=42, first_name="Иван"),
                Topic(id=1, name="Самооценка", is_active=True),
                Topic(id=2, name="Тревожность", is_active=True),
                AutomationStepTransition(
                    user_id=42, dialogue_id=1, topic_id=1, current_step="SELF_STEP", state_json="{}"
                ),
                AutomationStepTransition(
                    user_id=42, dialogue_id=1, topic_id=2, current_step="ANXIETY_STEP", state_json="{}"
                ),
                AutomationConversationState(
                    user_id=42, dialogue_id=1, topic_id=1, current_step="SELF_STEP", current_state_json="{}"
                ),
            ])
            await session.commit()
        message = SimpleNamespace(edit_text=AsyncMock())
        callback = SimpleNamespace(message=message)

        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin._show_automation_stage_stats(callback, topic_id=1)

        text = message.edit_text.await_args.args[0]
        markup = message.edit_text.await_args.kwargs["reply_markup"]
        self.assertIn("SELF_STEP", text)
        self.assertNotIn("ANXIETY_STEP", text)
        self.assertIn("edit_topic_1", callback_values(markup))

    async def test_global_statistics_return_to_automations_after_rendering_topic_rows(self):
        async with self.sessions() as session:
            session.add_all([
                User(id=42, first_name="Иван"),
                Topic(id=1, name="Самооценка", is_active=True),
                AutomationStepTransition(
                    user_id=42,
                    dialogue_id=1,
                    topic_id=1,
                    current_step="SELF_STEP",
                    state_json="{}",
                ),
                AutomationConversationState(
                    user_id=42,
                    dialogue_id=1,
                    topic_id=1,
                    current_step="SELF_STEP",
                    current_state_json="{}",
                ),
            ])
            await session.commit()
        message = SimpleNamespace(edit_text=AsyncMock())
        callback = SimpleNamespace(message=message)

        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin._show_automation_stage_stats(callback)

        text = message.edit_text.await_args.args[0]
        callbacks = callback_values(message.edit_text.await_args.kwargs["reply_markup"])
        self.assertIn("👥 Вошли:", text)
        self.assertIn("🔁 Переходы:", text)
        self.assertIn("📍 Сейчас:", text)
        self.assertIn("automation_menu", callbacks)
        self.assertNotIn("edit_topic_1", callbacks)

    async def test_admin_statistics_origin_returns_to_general_statistics(self):
        async with self.sessions() as session:
            session.add_all([
                User(id=42, first_name="Иван"),
                AutomationStepTransition(
                    user_id=42, dialogue_id=1, topic_id=0, current_step="START", state_json="{}"
                ),
            ])
            await session.commit()
        message = SimpleNamespace(edit_text=AsyncMock())

        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin._show_automation_stage_stats(
                SimpleNamespace(message=message), origin="a"
            )

        callbacks = callback_values(message.edit_text.await_args.kwargs["reply_markup"])
        self.assertIn("admin_stats", callbacks)
        self.assertTrue(any(value.endswith("_a") for value in callbacks))

    async def test_stage_and_user_lists_are_paginated_and_current_users_are_unique(self):
        async with self.sessions() as session:
            topic = Topic(id=1, name="Самооценка", is_active=True)
            users = [
                User(id=user_id, first_name=f"Пользователь {user_id}", username=f"user{user_id}")
                for user_id in range(1, 12)
            ]
            session.add_all([topic, *users])
            await session.flush()
            for user in users:
                session.add(AutomationStepTransition(
                    user_id=user.id,
                    dialogue_id=1,
                    topic_id=1,
                    current_step="COMMON_STEP",
                    state_json="{}",
                ))
            for index in range(1, 10):
                session.add(AutomationStepTransition(
                    user_id=1,
                    dialogue_id=index + 1,
                    topic_id=1,
                    current_step=f"STEP_{index}",
                    state_json="{}",
                ))
            session.add_all([
                AutomationConversationState(
                    user_id=1,
                    dialogue_id=1,
                    topic_id=1,
                    current_step="COMMON_STEP",
                    current_state_json="{}",
                ),
                AutomationConversationState(
                    user_id=1,
                    dialogue_id=2,
                    topic_id=1,
                    current_step="COMMON_STEP",
                    current_state_json="{}",
                ),
            ])
            await session.commit()

        stats_message = SimpleNamespace(edit_text=AsyncMock())
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin._show_automation_stage_stats(
                SimpleNamespace(message=stats_message), topic_id=1
            )

        stats_text = stats_message.edit_text.await_args.args[0]
        stats_callbacks = callback_values(stats_message.edit_text.await_args.kwargs["reply_markup"])
        self.assertIn("Страница <b>1/2</b>", stats_text)
        self.assertIn("📍 Сейчас: <b>1</b>", stats_text)
        self.assertIn("topic_automation_stats_page_1_1", stats_callbacks)
        async with self.sessions() as session:
            anchor_id = await session.scalar(
                select(func.min(AutomationStepTransition.id)).where(
                    AutomationStepTransition.current_step == "COMMON_STEP"
                )
            )

        users_message = SimpleNamespace(edit_text=AsyncMock())
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin._show_automation_stage_users(
                SimpleNamespace(message=users_message),
                anchor_id=anchor_id,
                page=0,
                origin="t",
            )

        users_text = users_message.edit_text.await_args.args[0]
        users_callbacks = callback_values(users_message.edit_text.await_args.kwargs["reply_markup"])
        self.assertIn("Уникальных пользователей: <b>11</b>", users_text)
        self.assertIn("Страница: <b>1/2</b>", users_text)
        self.assertIn("topic_automation_stats_1", users_callbacks)
        self.assertTrue(any(value.endswith("_1_t") for value in users_callbacks))

    async def test_callback_middleware_answers_once_and_deletion_refreshes_menu(self):
        async with self.sessions() as session:
            item = AutomationHandler(name="Проверка")
            session.add(item)
            await session.flush()
            condition = AutomationCondition(
                handler_id=item.id,
                condition_type="event",
                operator="equals",
                expected_value="READY",
            )
            session.add(condition)
            await session.commit()
            handler_id = item.id
            condition_id = condition.id

        callback = SimpleNamespace(
            id="delete-condition",
            data=f"automation_condition_delete_{handler_id}_{condition_id}",
            answer=AsyncMock(),
            message=SimpleNamespace(edit_text=AsyncMock()),
        )

        async def run_handler(event, _data):
            await automation_admin.automation_condition_delete(event)

        automation_admin._answered_callback_ids.clear()
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.EnsureCallbackAnsweredMiddleware()(run_handler, callback, {})

        callback.answer.assert_awaited_once()
        callback.message.edit_text.assert_awaited_once()
        async with self.sessions() as session:
            self.assertIsNone(await session.get(AutomationCondition, condition_id))

    async def test_callback_middleware_does_not_duplicate_explicit_answer(self):
        callback = SimpleNamespace(id="explicit-answer", answer=AsyncMock())

        async def run_handler(event, _data):
            await automation_admin._answer_callback(event, "Готово")

        automation_admin._answered_callback_ids.clear()
        await automation_admin.EnsureCallbackAnsweredMiddleware()(run_handler, callback, {})

        callback.answer.assert_awaited_once_with("Готово")

    async def test_topic_selection_refreshes_and_closes_callback(self):
        async with self.sessions() as session:
            topic = Topic(id=1, name="Самооценка", is_active=True)
            item = AutomationHandler(name="Выбор тем", include_main_dialogue=False)
            session.add_all([topic, item])
            await session.commit()
            handler_id = item.id

        callback = SimpleNamespace(
            id="select-handler-topic",
            data=f"automation_htopic_{handler_id}_1",
            answer=AsyncMock(),
            message=SimpleNamespace(edit_text=AsyncMock()),
        )

        async def run_handler(event, _data):
            await automation_admin.automation_handler_topic_toggle(event)

        automation_admin._answered_callback_ids.clear()
        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin.EnsureCallbackAnsweredMiddleware()(run_handler, callback, {})

        callback.answer.assert_awaited_once()
        callback.message.edit_text.assert_awaited_once()
        async with self.sessions() as session:
            item = await session.scalar(
                select(AutomationHandler).options(selectinload(AutomationHandler.topics))
            )
            self.assertEqual([topic.id for topic in item.topics], [1])

    async def test_handler_card_returns_to_topic_handler_list(self):
        async with self.sessions() as session:
            topic = Topic(id=1, name="Самооценка", is_active=True)
            item = AutomationHandler(name="Только тема", topics=[topic])
            session.add(item)
            await session.commit()
            handler_id = item.id
        state = MemoryState()
        list_message = SimpleNamespace(edit_text=AsyncMock())
        card_message = SimpleNamespace(edit_text=AsyncMock())

        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin._show_automation_handlers(
                SimpleNamespace(message=list_message), state=state, topic_id=1
            )
            await automation_admin._show_handler(card_message, handler_id, state=state)

        callbacks = callback_values(card_message.edit_text.await_args.kwargs["reply_markup"])
        self.assertIn("topic_automation_handlers_1", callbacks)
        self.assertNotIn("automation_handlers", callbacks)

    async def test_followup_card_returns_to_topic_campaign_list(self):
        async with self.sessions() as session:
            topic = Topic(id=1, name="Самооценка", is_active=True)
            item = FollowupCampaign(name="Только тема", topics=[topic])
            session.add(item)
            await session.commit()
            campaign_id = item.id
        state = MemoryState()
        list_message = SimpleNamespace(edit_text=AsyncMock())
        card_message = SimpleNamespace(edit_text=AsyncMock())

        with patch.object(automation_admin, "async_session_maker", self.sessions):
            await automation_admin._show_followup_campaigns(
                SimpleNamespace(message=list_message), state=state, topic_id=1
            )
            await automation_admin._show_campaign(card_message, campaign_id, state=state)

        callbacks = callback_values(card_message.edit_text.await_args.kwargs["reply_markup"])
        self.assertIn("topic_followup_campaigns_1", callbacks)
        self.assertNotIn("followup_campaigns", callbacks)
