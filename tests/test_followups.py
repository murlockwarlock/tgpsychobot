import os
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import followups
from database import Base, FollowupCampaign, FollowupDelivery, FollowupRun, FollowupStep, User
from followups import _outside_quiet_hours, process_due_followups, record_user_activity


class FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text):
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
