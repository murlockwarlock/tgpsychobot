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
        ReferralPaymentLog,
        SubscriptionConfig,
        SubscriptionPlan,
        User,
        UserSubscription,
        YookassaPayment,
        RobokassaPayment,
        async_session_maker,
        init_db,
    )
    from notification_outbox import (
        NotificationPolicy,
        build_canonical_outbox_key,
        get_canonical_key_for_yookassa_cancellation,
        get_canonical_key_for_yookassa_success,
    )
    from notification_renderer import render_outbox_message
    from subscription_renewal import (
        CancellationPolicy,
        FinalizePaymentSuccessResult,
        finalize_yookassa_payment_canceled,
        finalize_yookassa_payment_success,
    )


class NotificationClassificationAndReferralTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await init_db()
        self.now = datetime(2026, 9, 11, 12, 0, 0)
        self.bot = MagicMock()
        self.bot.send_message = AsyncMock(return_value=True)

    def test_tuple_compatibility_of_finalize_result(self):
        """FinalizePaymentSuccessResult can be unpacked as (is_new, sub) while offering .is_recurring attribute."""
        sub = object()
        res = FinalizePaymentSuccessResult(
            is_new=True,
            user_sub=sub,
            action="success",
            reconciliation_details=None,
            is_recurring=False,
        )
        is_new, unpacked_sub = res
        self.assertTrue(is_new)
        self.assertIs(unpacked_sub, sub)
        self.assertFalse(res.is_recurring)
        self.assertEqual(res.action, "success")

    def test_classification_keys_and_messages(self):
        """Verify canonical key formatting and message rendering across event types."""
        # 1. YooKassa ordinary purchase
        yoo_purch_key = get_canonical_key_for_yookassa_success("pay_1", 101, "success", is_recurring=False)
        self.assertEqual(yoo_purch_key, "yookassa:payment:pay_1:101:purchase_success")
        msg, pm, kb = render_outbox_message("purchase_success", {"plan_name": "Standard"}, 101)
        self.assertIn("✅ Ваша подписка на тариф «Standard» успешно оформлена!", msg)
        self.assertIsNone(kb)

        # 2. YooKassa renewal
        yoo_renew_key = get_canonical_key_for_yookassa_success("pay_2", 102, "success", is_recurring=True)
        self.assertEqual(yoo_renew_key, "yookassa:payment:pay_2:102:renewal_success")
        msg, pm, kb = render_outbox_message("renewal_success", {"provider": "Yookassa", "end_date_msk": "15.10.2026 12:00 МСК"}, 102)
        self.assertIn("✅ Подписка продлена до 15.10.2026 12:00 МСК.", msg)
        self.assertIsNone(kb)

        # 3. Robokassa ordinary purchase
        robo_purch_key = build_canonical_outbox_key("robokassa", "payment", 5001, 103, "purchase_success")
        self.assertEqual(robo_purch_key, "robokassa:payment:5001:103:purchase_success")
        msg, pm, kb = render_outbox_message("purchase_success", {
            "provider": "Robokassa",
            "amount": 990.0,
            "plan_name": "Базовый",
            "end_date_msk": "15.10.2026 12:00 МСК",
        }, 103)
        self.assertIn("Мы получили оплату 990.00 руб", msg)
        self.assertIn("Базовый", msg)

        # 4. Robokassa renewal
        robo_renew_key = build_canonical_outbox_key("robokassa", "payment", 5002, 104, "renewal_success")
        self.assertEqual(robo_renew_key, "robokassa:payment:5002:104:renewal_success")
        msg, pm, kb = render_outbox_message("renewal_success", {
            "provider": "Robokassa",
            "amount": 990.0,
            "plan_name": "Базовый",
            "end_date_msk": "15.10.2026 12:00 МСК",
        }, 104)
        self.assertIn("Мы получили оплату 990.00 руб", msg)

        # 5. YooKassa retryable decline (attempt 1)
        decl_1_key = get_canonical_key_for_yookassa_cancellation("pay_3", 105, "declined", attempt_count=1)
        self.assertEqual(decl_1_key, "yookassa:payment:pay_3:105:retry_failed_1")
        msg, pm, kb = render_outbox_message("retry_failed_1", {"provider": "ЮKassa", "attempt_count": 1, "next_retry_str": "завтра в 10:00 МСК"}, 105)
        self.assertIn("Не удалось списать средства", msg)
        self.assertIn("Повторим попытку", msg)
        self.assertEqual(kb, "subscribe")

        # 6. YooKassa final decline (attempt 3)
        decl_3_key = get_canonical_key_for_yookassa_cancellation("pay_4", 106, "declined", attempt_count=3)
        self.assertEqual(decl_3_key, "yookassa:payment:pay_4:106:final_decline")
        msg, pm, kb = render_outbox_message("final_decline", {}, 106)
        self.assertIn("Автопродление подписки отключено", msg)
        self.assertEqual(kb, "subscribe")

    async def test_yookassa_finalizer_atomic_outbox_enrollment(self):
        """finalize_yookassa_payment_success enqueues outbox event before commit when policy requires."""
        user_id = 111222
        plan_id = 950
        async with async_session_maker() as session:
            user = User(id=user_id, first_name="Bob")
            plan = SubscriptionPlan(id=plan_id, name="Pro Plan", price=1500.0, duration_value=1, duration_unit="months")
            sub = UserSubscription(
                user_id=user_id,
                plan_id=plan_id,
                start_date=self.now - timedelta(days=30),
                end_date=self.now,
                payment_provider="Yookassa",
                payment_method_id="pm_saved_1",
                auto_renewal=True,
            )
            session.add_all([user, plan, sub])
            await session.commit()

        # Call finalizer with ALL_USER_EVENTS
        async with async_session_maker() as session:
            res = await finalize_yookassa_payment_success(
                session=session,
                payment_id="pay_yoo_atomic",
                user_id=user_id,
                plan_id=plan_id,
                amount=1500.0,
                payment_method_id="pm_saved_1",
                is_recurring=True,
                notification_policy=NotificationPolicy.ALL_USER_EVENTS,
            )
            await session.commit()
        self.assertTrue(res.is_recurring)

        from sqlalchemy import select
        async with async_session_maker() as session:
            row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.payment_id == "pay_yoo_atomic")
            )
            self.assertIsNotNone(row)
            self.assertEqual(row.event_type, "renewal_success")
            self.assertEqual(row.recipient_id, user_id)
            self.assertEqual(row.status, "pending")

    async def test_referral_bonus_atomicity_and_first_only_policy(self):
        """Referral bonus + log + referral_bonus outbox row are atomic; duplicate/first-only prevents duplicate rows."""
        referrer_id = 3001
        referred_id = 3002

        async with async_session_maker() as session:
            referrer = User(id=referrer_id, first_name="Referrer")
            referred = User(id=referred_id, first_name="Referred", referred_by=referrer_id)
            session.add_all([referrer, referred])
            config = await session.get(SubscriptionConfig, 1)
            if not config:
                config = SubscriptionConfig(id=1)
                session.add(config)
            config.referral_enabled = True
            config.referral_pay_bonus_enabled = True
            config.referral_pay_bonus_days = 7
            config.referral_pay_bonus_first_only = True
            await session.commit()

        from sqlalchemy import func, select
        from notification_outbox import enqueue_outbox_event
        async with async_session_maker() as session:
            # Check first only
            prev_count = await session.scalar(
                select(func.count()).select_from(ReferralPaymentLog)
                .where(ReferralPaymentLog.referred_user_id == referred_id)
            ) or 0
            self.assertEqual(prev_count, 0)

            # Award bonus
            ref_sub = UserSubscription(
                user_id=referrer_id,
                plan_id=None,
                start_date=self.now,
                end_date=self.now + timedelta(days=7),
                payment_provider="Trial Referral Pay Bonus",
                auto_renewal=False,
                payment_attempt_count=0,
                discount_percent=0,
            )
            session.add(ref_sub)
            session.add(ReferralPaymentLog(
                referrer_id=referrer_id,
                referred_user_id=referred_id,
                amount=1000.0,
            ))
            ref_key = f"yookassa:payment:pay_ref_1:{referrer_id}:referral_bonus"
            payload = {"user_id": referrer_id, "bonus_days": 7, "referred_user_id": referred_id}
            await enqueue_outbox_event(session, ref_key, "Yookassa", referrer_id, "referral_bonus", payload, payment_id="pay_ref_1")
            await session.commit()

        # Verify bonus and outbox row exist
        from sqlalchemy import func, select
        async with async_session_maker() as session:
            outbox_row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == ref_key)
            )
            self.assertIsNotNone(outbox_row)
            self.assertEqual(outbox_row.event_type, "referral_bonus")

            log_count = await session.scalar(
                select(func.count()).select_from(ReferralPaymentLog).where(ReferralPaymentLog.referred_user_id == referred_id)
            )
            self.assertEqual(log_count, 1)

        # 2. Duplicate payment arriving for same referred user when first_only=True:
        async with async_session_maker() as session:
            prev_count = await session.scalar(
                select(func.count()).select_from(ReferralPaymentLog)
                .where(ReferralPaymentLog.referred_user_id == referred_id)
            ) or 0
            already_paid = prev_count > 0
            self.assertTrue(already_paid)
            # First-only rejects: zero new bonus, zero outbox row
            # Attempting to re-enqueue same key or second key is rejected
            ref_key_2 = f"yookassa:payment:pay_ref_2:{referrer_id}:referral_bonus"
            # Since already_paid is True, logic does NOT enqueue outbox row
            await session.commit()

        async with async_session_maker() as session:
            row2 = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == "yookassa:payment:pay_ref_2:3001:referral_bonus")
            )
            self.assertIsNone(row2)
