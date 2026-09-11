import asyncio
import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import sqlalchemy.ext.asyncio as sqlalchemy_asyncio

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

original_create_async_engine = sqlalchemy_asyncio.create_async_engine


def _sqlite_compatible_engine(*args, **kwargs):
    kwargs.pop("pool_recycle", None)
    kwargs.pop("pool_use_lifo", None)
    return original_create_async_engine(*args, **kwargs)


with patch.object(sqlalchemy_asyncio, "create_async_engine", _sqlite_compatible_engine):
    from database import (
        PaymentNotificationOutbox,
        User,
        YookassaPayment,
        async_session_maker,
        init_db,
    )
    from notification_outbox import (
        dispatch_outbox_by_key,
        enqueue_outbox_event,
        process_payment_notification_outbox,
    )


class NotificationOutboxConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await init_db()
        self.bot = MagicMock()
        self.bot.send_message = AsyncMock(return_value=True)

    async def test_sqlite_pk_and_rollback_isolation(self):
        """Transaction rollback discards outbox entry; commit persists with integer PK."""
        # 1. Rollback test with business transaction
        async with async_session_maker() as session:
            session.add(User(id=99991, first_name="Rollback User"))
            await session.flush()
            await enqueue_outbox_event(
                session,
                "test:rollback:key",
                "Yookassa",
                99991,
                "purchase_success",
                {"plan_name": "Rollback Plan"},
            )
            await session.rollback()

        async with async_session_maker() as session:
            from sqlalchemy import select
            user = await session.get(User, 99991)
            row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == "test:rollback:key")
            )
            self.assertIsNone(user)
            self.assertIsNone(row)

        # 2. Commit test
        async with async_session_maker() as session:
            enq = await enqueue_outbox_event(
                session,
                "test:commit:key",
                "Yookassa",
                12345,
                "purchase_success",
                {"plan_name": "Commit Plan"},
            )
            self.assertIsNotNone(enq)
            await session.commit()

        async with async_session_maker() as session:
            row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == "test:commit:key")
            )
            self.assertIsNotNone(row)
            self.assertIsInstance(row.id, int)
            self.assertEqual(row.status, "pending")

    async def test_duplicate_deduplication(self):
        """Second enqueue with same unique_key is safely ignored within the same transaction."""
        async with async_session_maker() as session:
            row1 = await enqueue_outbox_event(
                session,
                "test:dedup:key",
                "Yookassa",
                12345,
                "purchase_success",
                {"plan_name": "Plan A"},
            )
            self.assertIsNotNone(row1)

            row2 = await enqueue_outbox_event(
                session,
                "test:dedup:key",
                "Yookassa",
                12345,
                "purchase_success",
                {"plan_name": "Plan A Duplicate"},
            )
            self.assertIsNone(row2)
            await session.commit()

        from sqlalchemy import func, select
        async with async_session_maker() as session:
            count = await session.scalar(
                select(func.count(PaymentNotificationOutbox.id)).where(
                    PaymentNotificationOutbox.unique_key == "test:dedup:key"
                )
            )
            self.assertEqual(count, 1)

    async def test_two_worker_cas_claim_race(self):
        """Two concurrent workers trying to claim the same outbox row: exactly 1 delivers, 0 duplicates."""
        key = "test:race:key"
        async with async_session_maker() as session:
            await enqueue_outbox_event(
                session,
                key,
                "Yookassa",
                555123,
                "purchase_success",
                {"plan_name": "Race Plan"},
            )
            await session.commit()

        delivery_mock = AsyncMock(return_value=True)

        # Run two dispatches concurrently
        res1, res2 = await asyncio.gather(
            dispatch_outbox_by_key(self.bot, key, deliver_func=delivery_mock),
            dispatch_outbox_by_key(self.bot, key, deliver_func=delivery_mock),
        )

        # Exactly one should return True, the other False
        results = [res1, res2]
        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count(False), 1)
        # Deliver func called exactly once
        self.assertEqual(delivery_mock.await_count, 1)

        from sqlalchemy import select
        async with async_session_maker() as session:
            row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == key)
            )
            self.assertEqual(row.status, "delivered")
            self.assertIsNotNone(row.delivered_at)

    async def test_lease_recovery(self):
        """Expired lease is reclaimed and delivered by periodic worker."""
        key = "test:lease:expired"
        now = datetime.utcnow()
        async with async_session_maker() as session:
            row = PaymentNotificationOutbox(
                unique_key=key,
                provider="Yookassa",
                recipient_id=777,
                event_type="purchase_success",
                event_payload_json='{"plan_name": "Lease Plan"}',
                status="processing",
                claim_token="old-expired-token",
                lease_until=now - timedelta(minutes=5),  # expired 5 min ago
                attempts=1,
                max_attempts=7,
                next_retry_at=now - timedelta(minutes=5),
            )
            session.add(row)
            await session.commit()

        delivery_mock = AsyncMock(return_value=True)
        delivered_count = await process_payment_notification_outbox(self.bot, deliver_func=delivery_mock)
        self.assertGreaterEqual(delivered_count, 1)
        delivery_mock.assert_awaited()

        from sqlalchemy import select
        async with async_session_maker() as session:
            updated_row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == key)
            )
            self.assertEqual(updated_row.status, "delivered")

    async def test_at_least_once_retry_and_terminal_escalation(self):
        """Failed delivery retries with backoff, then escalates to failed_terminal when attempts >= max_attempts."""
        key = "test:terminal:key"
        now = datetime.utcnow()
        async with async_session_maker() as session:
            row = PaymentNotificationOutbox(
                unique_key=key,
                provider="Yookassa",
                recipient_id=888,
                event_type="purchase_success",
                event_payload_json='{"plan_name": "Fail Plan"}',
                status="pending",
                attempts=1,
                max_attempts=2,  # 2nd attempt will be the final attempt
                next_retry_at=now,
            )
            session.add(row)
            await session.commit()

        fail_mock = AsyncMock(return_value=False)

        with patch("notification_outbox._send_admin_terminal_failure_alert", new_callable=AsyncMock) as mock_alert:
            # 2nd attempt fails -> max_attempts reached -> failed_terminal
            res = await dispatch_outbox_by_key(self.bot, key, deliver_func=fail_mock)
            self.assertFalse(res)
            mock_alert.assert_awaited_once()

        from sqlalchemy import select
        async with async_session_maker() as session:
            final_row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == key)
            )
            self.assertEqual(final_row.status, "failed_terminal")
            self.assertEqual(final_row.attempts, 2)
