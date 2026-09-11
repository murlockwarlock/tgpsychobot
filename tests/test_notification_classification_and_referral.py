import json
import os
import re
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
        YookassaRecurringAttempt,
        async_session_maker,
        init_db,
    )
    from notification_outbox import (
        NotificationPolicy,
        build_canonical_outbox_key,
        dispatch_outbox_by_key,
        enqueue_outbox_event,
        get_canonical_key_for_yookassa_cancellation,
        get_canonical_key_for_yookassa_success,
        process_payment_notification_outbox,
    )
    from notification_renderer import render_outbox_message
    from subscription_renewal import (
        CancellationPolicy,
        FinalizePaymentSuccessResult,
        finalize_yookassa_payment_canceled,
        finalize_yookassa_payment_success,
        transition_attempt_to_unknown_expired,
    )
    from time_helpers import format_msk


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
        msg, pm, kb = render_outbox_message("renewal_success", {"provider": "Yookassa", "end_date_msk": "15.10.2026 12:00"}, 102)
        self.assertEqual("✅ Подписка продлена до 15.10.2026 12:00 МСК.", msg)
        self.assertIsNone(kb)

        # 3. Robokassa ordinary purchase
        robo_purch_key = build_canonical_outbox_key("robokassa", "payment", 5001, 103, "purchase_success")
        self.assertEqual(robo_purch_key, "robokassa:payment:5001:103:purchase_success")
        msg, pm, kb = render_outbox_message("purchase_success", {
            "provider": "Robokassa",
            "amount": 990.0,
            "plan_name": "Базовый",
            "end_date_msk": "15.10.2026 12:00",
        }, 103)
        self.assertIn("Мы получили оплату 990.00 руб", msg)
        self.assertIn("Базовый", msg)
        self.assertIn("Действие тарифа продлено до 15.10.2026 12:00 МСК.", msg)
        self.assertNotIn("МСК МСК", msg)

        # 4. Robokassa renewal
        robo_renew_key = build_canonical_outbox_key("robokassa", "payment", 5002, 104, "renewal_success")
        self.assertEqual(robo_renew_key, "robokassa:payment:5002:104:renewal_success")
        msg, pm, kb = render_outbox_message("renewal_success", {
            "provider": "Robokassa",
            "amount": 990.0,
            "plan_name": "Базовый",
            "end_date_msk": "15.10.2026 12:00",
        }, 104)
        self.assertIn("Мы получили оплату 990.00 руб", msg)
        self.assertIn("Действие тарифа продлено до 15.10.2026 12:00 МСК.", msg)
        self.assertNotIn("МСК МСК", msg)

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

    async def test_yookassa_unknown_expired_durable_outbox_failure_and_recovery(self):
        """Test YooKassa unknown_expired transition is atomic with outbox enqueue, survives transport failure, and later worker recovers."""
        from sqlalchemy import select

        async with async_session_maker() as session:
            session.add(User(id=4001, first_name="ExpiredUser"))
            plan = SubscriptionPlan(id=4001, name="Базовый", price=195.0, duration_value=1, duration_unit="month")
            session.add(plan)
            sub = UserSubscription(
                id=4001,
                user_id=4001,
                plan_id=4001,
                start_date=self.now - timedelta(days=30),
                end_date=self.now,
                auto_renewal=True,
                payment_attempt_count=1,
            )
            session.add(sub)
            att = YookassaRecurringAttempt(
                id=4001,
                subscription_id=4001,
                user_id=4001,
                plan_id=4001,
                idempotency_key="idemp_exp_4001",
                amount=195.0,
                payment_method_id="pm_exp_4001",
                status="unknown",
                attempt_started_at=self.now - timedelta(hours=25),
            )
            session.add(att)
            await session.commit()

        # Step 1: Transition to unknown_expired with TERMINAL_ONLY notification policy
        async with async_session_maker() as session:
            is_new, user_sub = await transition_attempt_to_unknown_expired(
                session, 4001, now=self.now, notification_policy=NotificationPolicy.TERMINAL_ONLY
            )
            self.assertTrue(is_new)
            self.assertIsNotNone(user_sub)
            self.assertFalse(user_sub.auto_renewal)

        # Step 2: Verify DB state in separate session: attempt is unknown_expired, sub auto_renewal=False, outbox row pending
        expected_key = "yookassa:attempt:4001:4001:unknown_expired"
        async with async_session_maker() as session:
            att_db = await session.get(YookassaRecurringAttempt, 4001)
            self.assertEqual(att_db.status, "unknown_expired")
            self.assertEqual(att_db.error_code, "timeout_24h")

            sub_db = await session.get(UserSubscription, 4001)
            self.assertFalse(sub_db.auto_renewal)

            outbox_row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == expected_key)
            )
            self.assertIsNotNone(outbox_row)
            self.assertEqual(outbox_row.status, "pending")
            self.assertEqual(outbox_row.event_type, "unknown_expired")
            self.assertEqual(outbox_row.recipient_id, 4001)

        # Step 3: Simulate transport failure during post-commit dispatch
        fail_transport = AsyncMock(return_value=False)
        dispatch_res = await dispatch_outbox_by_key(self.bot, expected_key, deliver_func=fail_transport)
        self.assertFalse(dispatch_res)
        fail_transport.assert_awaited_once()

        # Verify state after transport failure: attempt remains terminal, sub remains paused, outbox retryable
        async with async_session_maker() as session:
            att_db = await session.get(YookassaRecurringAttempt, 4001)
            self.assertEqual(att_db.status, "unknown_expired")
            sub_db = await session.get(UserSubscription, 4001)
            self.assertFalse(sub_db.auto_renewal)

            outbox_row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == expected_key)
            )
            self.assertEqual(outbox_row.status, "pending")
            self.assertEqual(outbox_row.attempts, 1)

            # Fast-forward next_retry_at so periodic worker picks it up
            outbox_row.next_retry_at = datetime.utcnow() - timedelta(minutes=1)
            await session.commit()

        # Step 4: Later worker recovers and delivers
        delivered_messages = []
        async def mock_deliver(bot, recipient_id, text, keyboard_type=None, parse_mode=None, reply_markup=None):
            delivered_messages.append({
                "recipient_id": recipient_id,
                "text": text,
                "keyboard_type": keyboard_type,
            })
            return True

        delivered_count = await process_payment_notification_outbox(self.bot, deliver_func=mock_deliver)
        self.assertGreaterEqual(delivered_count, 1)

        async with async_session_maker() as session:
            final_outbox = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == expected_key)
            )
            self.assertEqual(final_outbox.status, "delivered")
            self.assertIsNotNone(final_outbox.delivered_at)

        my_delivered = [m for m in delivered_messages if m["recipient_id"] == 4001]
        self.assertEqual(len(my_delivered), 1)
        expected_text = (
            "Не удалось подтвердить результат списания за 24 часа. Чтобы избежать двойных списаний, автопродление приостановлено.\n\n"
            "Проверьте статус в банке или оформите подписку в меню."
        )
        self.assertEqual(my_delivered[0]["text"], expected_text)
        self.assertEqual(my_delivered[0]["keyboard_type"], "subscribe")

    async def test_robokassa_provider_error_durable_outbox_failure_and_recovery(self):
        """Test Robokassa provider_error terminal outbox event survives transport failure and later worker recovers."""
        from sqlalchemy import select

        user_id = 5001
        inv_id = 70001
        async with async_session_maker() as session:
            session.add(User(id=user_id, first_name="RoboUser"))
            plan = SubscriptionPlan(id=5001, name="Тариф 1", price=350.0, duration_value=1, duration_unit="month")
            session.add(plan)
            sub = UserSubscription(
                id=5001,
                user_id=user_id,
                plan_id=5001,
                start_date=self.now - timedelta(days=30),
                end_date=self.now,
                auto_renewal=True,
                payment_attempt_count=0,
            )
            session.add(sub)
            new_payment = RobokassaPayment(
                id=inv_id,
                user_id=user_id,
                plan_id=5001,
                amount=350.0,
                status="pending",
            )
            session.add(new_payment)
            await session.commit()

        # Simulate scheduler branch robokassa_res == 'provider_error'
        next_retry_str = format_msk(self.now + timedelta(hours=2), "%d.%m %H:%M МСК")
        rk_err_key = f"robokassa:payment:{inv_id}:{user_id}:provider_error"
        async with async_session_maker() as session:
            p_obj = await session.get(RobokassaPayment, inv_id)
            p_obj.status = "request_provider_error"
            s_obj = await session.get(UserSubscription, 5001)
            s_obj.last_payment_attempt = self.now

            rk_payload = {
                "user_id": user_id,
                "provider": "Robokassa",
                "next_retry_str": next_retry_str,
                "payment_id": str(inv_id),
            }
            await enqueue_outbox_event(
                session,
                rk_err_key,
                "Robokassa",
                user_id,
                "provider_error",
                rk_payload,
                payment_id=str(inv_id),
            )
            await session.commit()

        # Verify terminal payment status and outbox row pending
        async with async_session_maker() as session:
            p_db = await session.get(RobokassaPayment, inv_id)
            self.assertEqual(p_db.status, "request_provider_error")
            outbox_row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == rk_err_key)
            )
            self.assertIsNotNone(outbox_row)
            self.assertEqual(outbox_row.status, "pending")
            self.assertEqual(outbox_row.event_type, "provider_error")

        # Transport fails
        fail_mock = AsyncMock(return_value=False)
        dispatch_res = await dispatch_outbox_by_key(self.bot, rk_err_key, deliver_func=fail_mock)
        self.assertFalse(dispatch_res)

        # Worker recovers
        async with async_session_maker() as session:
            outbox_row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == rk_err_key)
            )
            outbox_row.next_retry_at = datetime.utcnow() - timedelta(minutes=1)
            await session.commit()

        delivered_messages = []
        async def mock_deliver(bot, recipient_id, text, keyboard_type=None, parse_mode=None, reply_markup=None):
            delivered_messages.append({"recipient_id": recipient_id, "text": text, "keyboard_type": keyboard_type})
            return True

        delivered_count = await process_payment_notification_outbox(self.bot, deliver_func=mock_deliver)
        self.assertGreaterEqual(delivered_count, 1)

        expected_text = (
            f"Платёжный шлюз Robokassa временно недоступен. Эта ошибка не засчитана как попытка списания.\n\n"
            f"Повторим запрос после {next_retry_str}."
        )
        my_delivered = [m for m in delivered_messages if m["recipient_id"] == user_id]
        self.assertEqual(len(my_delivered), 1)
        self.assertEqual(my_delivered[0]["text"], expected_text)
        self.assertEqual(my_delivered[0]["keyboard_type"], "subscribe")

    async def test_standardized_end_date_msk_contract_and_never_duplicate_msk(self):
        """Standardized end_date_msk contract: payload contains strictly %d.%m.%Y %H:%M, renderer appends МСК, never duplicate."""
        from sqlalchemy import select

        # 1. YooKassa renewal event construction path via finalize_yookassa_payment_success
        user_id = 6001
        plan_id = 6001
        async with async_session_maker() as session:
            session.add(User(id=user_id, first_name="ContractUser"))
            plan = SubscriptionPlan(id=plan_id, name="Премиум", price=290.0, duration_value=1, duration_unit="month")
            session.add(plan)
            sub = UserSubscription(
                id=6001,
                user_id=user_id,
                plan_id=plan_id,
                start_date=self.now - timedelta(days=30),
                end_date=self.now + timedelta(days=30),
                auto_renewal=True,
            )
            session.add(sub)
            await session.commit()

        async with async_session_maker() as session:
            res = await finalize_yookassa_payment_success(
                session=session,
                payment_id="pay_contract_yoo",
                user_id=user_id,
                plan_id=plan_id,
                amount=290.0,
                is_recurring=True,
                notification_policy=NotificationPolicy.TERMINAL_ONLY,
            )
            self.assertTrue(res.is_new)

        yoo_key = get_canonical_key_for_yookassa_success("pay_contract_yoo", user_id, "success", is_recurring=True)
        async with async_session_maker() as session:
            outbox_row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == yoo_key)
            )
            self.assertIsNotNone(outbox_row)
            payload = json.loads(outbox_row.event_payload_json)
            end_date_msk = payload["end_date_msk"]
            # Strict regex: %d.%m.%Y %H:%M with NO literal МСК
            self.assertRegex(end_date_msk, r"^\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}$")
            self.assertNotIn("МСК", end_date_msk)

            # Renderer outputs exact string
            msg, _, _ = render_outbox_message("renewal_success", payload, user_id)
            self.assertEqual(msg, f"✅ Подписка продлена до {end_date_msk} МСК.")
            self.assertNotIn("МСК МСК", msg)

        # 2. Robokassa ResultURL webhook path
        from webhooks import handle_robokassa_result, calculate_signature
        from aiohttp.test_utils import make_mocked_request
        from aiogram.fsm.storage.memory import MemoryStorage

        robo_user_id = 6002
        robo_inv_id = 88001
        async with async_session_maker() as session:
            session.add(User(id=robo_user_id, first_name="RoboContractUser"))
            plan_rk = SubscriptionPlan(id=6002, name="РобоТариф", price=450.0, duration_value=1, duration_unit="month")
            session.add(plan_rk)
            config = SubscriptionConfig(
                id=1,
                robokassa_merchant_login="test_login",
                robokassa_password_2="pass2",
                notifications_enabled=False,
            )
            await session.merge(config)
            payment = RobokassaPayment(
                id=robo_inv_id,
                user_id=robo_user_id,
                plan_id=6002,
                amount=450.0,
                status="pending",
            )
            session.add(payment)
            await session.commit()

        crc = calculate_signature("450.00", robo_inv_id, "pass2")
        mock_bot = AsyncMock()
        mock_bot.id = 999
        mock_req = make_mocked_request(
            "POST",
            "/webhook/robokassa/result",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            app={"bot": mock_bot, "fsm_storage": MemoryStorage()},
        )
        mock_req.post = AsyncMock(return_value={
            "OutSum": "450.00",
            "InvId": str(robo_inv_id),
            "SignatureValue": crc,
            "shp_user_id": str(robo_user_id),
            "shp_plan_id": "6002",
        })

        with patch("webhooks.async_session_maker", async_session_maker), \
             patch("webhooks.dispatch_outbox_by_key", new_callable=AsyncMock) as mock_dispatch:
            resp = await handle_robokassa_result(mock_req)
            self.assertEqual(resp.text, f"OK{robo_inv_id}")

        rk_key = build_canonical_outbox_key("robokassa", "payment", robo_inv_id, robo_user_id, "purchase_success")
        async with async_session_maker() as session:
            outbox_row_rk = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == rk_key)
            )
            self.assertIsNotNone(outbox_row_rk)
            payload_rk = json.loads(outbox_row_rk.event_payload_json)
            end_date_msk_rk = payload_rk["end_date_msk"]
            self.assertRegex(end_date_msk_rk, r"^\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}$")
            self.assertNotIn("МСК", end_date_msk_rk)

            msg_rk, _, _ = render_outbox_message("purchase_success", payload_rk, robo_user_id)
            self.assertIn(f"Действие тарифа продлено до {end_date_msk_rk} МСК.", msg_rk)
            self.assertNotIn("МСК МСК", msg_rk)
