import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import automation_events
from automation_events import condition_matches, handler_matches, process_pending_events, render_message_template
from database import (
    AutomationAction,
    AutomationActionExecution,
    AutomationMessageDelivery,
    AutomationCondition,
    AutomationEvent,
    AutomationHandler,
    Base,
    Topic,
    User,
)


class FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text):
        self.sent.append((chat_id, text))
        return SimpleNamespace(message_id=len(self.sent))


class PartiallyFailingBot(FakeBot):
    def __init__(self, failing_recipient):
        super().__init__()
        self.failing_recipient = failing_recipient
        self.fail = True

    async def send_message(self, chat_id, text):
        if self.fail and chat_id == self.failing_recipient:
            raise TimeoutError("temporary timeout")
        return await super().send_message(chat_id, text)


class AutomationEventTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        await self.engine.dispose()

    def test_conditions_are_combined_and_templates_read_nested_metadata(self):
        event = SimpleNamespace(
            name="LEAD_READY",
            topic_id=0,
            dialogue_id=3,
            state_json='{"current_step":"CONSULTATION"}',
            metadata_json='{"profile":{"city":"Москва"}}',
        )
        conditions = [
            SimpleNamespace(condition_type="event", operator="equals", expected_value="LEAD_READY", field_path=None),
            SimpleNamespace(condition_type="current_step", operator="equals", expected_value="CONSULTATION", field_path=None),
            SimpleNamespace(condition_type="metadata", operator="equals", expected_value="Москва", field_path="profile.city"),
        ]
        handler = SimpleNamespace(
            is_active=True,
            conditions=conditions,
            actions=[object()],
            all_topics=False,
            include_main_dialogue=True,
            topics=[],
        )
        user = SimpleNamespace(id=42, username="ivan", name="Иван", first_name="Ivan")

        self.assertTrue(all(condition_matches(item, event) for item in conditions))
        self.assertTrue(handler_matches(handler, event))
        self.assertEqual(
            render_message_template(
                "{name} {user} {event} {current_step} {metadata.profile.city}",
                event=event,
                user=user,
            ),
            "Иван Иван LEAD_READY CONSULTATION Москва",
        )

    async def test_successful_action_is_idempotent(self):
        async with self.sessions() as session:
            user = User(id=42, first_name="Иван", username="ivan", current_dialogue_id=1)
            handler = AutomationHandler(
                name="Новый лид",
                is_active=True,
                include_main_dialogue=True,
            )
            handler.conditions.append(AutomationCondition(
                condition_type="event",
                expected_value="LEAD_READY",
                operator="equals",
            ))
            handler.actions.append(AutomationAction(
                action_type="send_message",
                recipient_type="all_admins",
                message_template="Лид {user_id}: {event}",
            ))
            event = AutomationEvent(
                user_id=42,
                dialogue_id=1,
                topic_id=0,
                name="LEAD_READY",
                state_json='{"current_step":"DONE"}',
                metadata_json="{}",
            )
            session.add_all([user, handler, event])
            await session.commit()

        bot = FakeBot()
        with (
            patch.object(automation_events, "async_session_maker", self.sessions),
            patch.object(automation_events, "get_all_admin_ids", AsyncMock(return_value={9001})),
        ):
            first = await process_pending_events(bot)
            second = await process_pending_events(bot)

        async with self.sessions() as session:
            executions = await session.scalar(select(func.count(AutomationActionExecution.id)))
            stored_event = await session.scalar(select(AutomationEvent))

        self.assertEqual(first, 1)
        self.assertEqual(second, 0)
        self.assertEqual(bot.sent, [(9001, "Лид 42: LEAD_READY")])
        self.assertEqual(executions, 1)
        self.assertIsNotNone(stored_event.processed_at)

    async def test_partial_admin_delivery_retries_only_failed_recipient(self):
        async with self.sessions() as session:
            user = User(id=43, first_name="Иван", current_dialogue_id=1)
            handler = AutomationHandler(name="Новый лид", is_active=True, include_main_dialogue=True)
            handler.conditions.append(AutomationCondition(
                condition_type="event", expected_value="LEAD_READY", operator="equals",
            ))
            handler.actions.append(AutomationAction(
                action_type="send_message",
                recipient_type="all_admins",
                message_template="Лид {user_id}",
            ))
            session.add_all([
                user,
                handler,
                AutomationEvent(
                    user_id=43,
                    dialogue_id=1,
                    topic_id=0,
                    name="LEAD_READY",
                    state_json="{}",
                    metadata_json="{}",
                ),
            ])
            await session.commit()

        bot = PartiallyFailingBot(failing_recipient=9002)
        with (
            patch.object(automation_events, "async_session_maker", self.sessions),
            patch.object(automation_events, "get_all_admin_ids", AsyncMock(return_value={9001, 9002})),
        ):
            self.assertEqual(await process_pending_events(bot), 0)
            bot.fail = False
            self.assertEqual(await process_pending_events(bot), 1)

        async with self.sessions() as session:
            deliveries = await session.scalar(select(func.count(AutomationMessageDelivery.id)))
            stored_event = await session.scalar(select(AutomationEvent))

        self.assertEqual(bot.sent, [(9001, "Лид 43"), (9002, "Лид 43")])
        self.assertEqual(deliveries, 2)
        self.assertIsNotNone(stored_event.processed_at)
