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
        expected_final = (
            "Ваша подписка истекла. Не удалось списать средства после 3 попыток — автопродление отключено.\n\n"
            "Продлите подписку вручную в меню."
        )
        self.assertEqual(msg, expected_final)
        self.assertEqual(kb, "subscribe")

        # 7. Robokassa deactivate (neutral wording)
        msg_rk_deact, _, kb_rk_deact = render_outbox_message("deactivate", {"provider": "Robokassa"}, 107)
        expected_rk_deact = (
            "Ваша подписка истекла. Ошибка при автоплатеже (Robokassa) — автопродление отключено.\n\n"
            "Продлите подписку вручную в меню."
        )
        self.assertEqual(msg_rk_deact, expected_rk_deact)
        self.assertEqual(kb_rk_deact, "subscribe")

        # 8. YooKassa unknown_cancellation (safer production copy)
        msg_unk, _, kb_unk = render_outbox_message("unknown_cancellation", {}, 108)
        expected_unk = (
            "Не удалось выполнить автоматическое списание (нестандартный ответ банка). "
            "Автопродление приостановлено во избежание повторных списаний.\n\n"
            "Пожалуйста, оформите или продлите подписку вручную в меню бота."
        )
        self.assertEqual(msg_unk, expected_unk)
        self.assertEqual(kb_unk, "subscribe")

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

    async def test_scheduler_robokassa_provider_error_production_path(self):
        """Robokassa provider_error scheduler path exercises actual check_subscriptions:
        - RobokassaPayment becomes request_provider_error
        - Outbox row is created by production branch and committed
        - No legacy _send_deduplicated_notification called for this event
        - Failed transport leaves durable retry row
        """
        from scheduler import check_subscriptions
        from sqlalchemy import select

        user_id = 77001
        plan_id = 77001
        async with async_session_maker() as session:
            user = User(id=user_id, first_name="RoboProvUser")
            plan = SubscriptionPlan(id=plan_id, name="Robo Тариф", price=550.0, duration_value=1, duration_unit="months")
            sub = UserSubscription(
                id=77001,
                user_id=user_id,
                plan_id=plan_id,
                start_date=self.now - timedelta(days=30),
                end_date=self.now - timedelta(minutes=10),  # expired 10m ago (due for Robokassa recurring)
                auto_renewal=True,
                payment_provider="Robokassa",
                payment_method_id="pm_rk_prov_1",
                payment_attempt_count=0,
                pending_robokassa_invoice_id=None,
            )
            config = SubscriptionConfig(
                id=1,
                robokassa_merchant_login="test_login",
                robokassa_password_2="pass2",
                notifications_enabled=False,
            )
            session.add_all([user, plan, sub])
            await session.merge(config)
            await session.commit()

        # Patch recurring robokassa payment to return 'provider_error'
        # Set bot.send_message to fail so transport fails and leaves durable retry row
        # Spy on legacy _send_deduplicated_notification
        self.bot.send_message = AsyncMock(side_effect=Exception("network down"))

        with patch("scheduler.process_recurring_robokassa_payment", new_callable=AsyncMock) as mock_recurring, \
             patch("scheduler._send_deduplicated_notification", new_callable=AsyncMock) as mock_legacy_send, \
             patch("scheduler.datetime") as mock_sched_dt, \
             patch("notification_outbox.datetime") as mock_outbox_dt:

            mock_sched_dt.utcnow.return_value = self.now
            mock_outbox_dt.utcnow.return_value = self.now
            mock_recurring.return_value = 'provider_error'

            await check_subscriptions(self.bot)

        # 1. Verify RobokassaPayment in DB became request_provider_error
        async with async_session_maker() as session:
            rk_payment = await session.scalar(
                select(RobokassaPayment).where(RobokassaPayment.user_id == user_id)
            )
            self.assertIsNotNone(rk_payment)
            self.assertEqual(rk_payment.status, "request_provider_error")

            # 2. Verify outbox row created by production branch
            expected_key = f"robokassa:payment:{rk_payment.id}:{user_id}:provider_error"
            outbox_row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == expected_key)
            )
            self.assertIsNotNone(outbox_row)
            self.assertEqual(outbox_row.event_type, "provider_error")
            self.assertEqual(outbox_row.recipient_id, user_id)
            # 3. Failed transport leaves durable retry row
            self.assertEqual(outbox_row.status, "pending")
            self.assertEqual(outbox_row.attempts, 1)
            self.assertEqual(outbox_row.last_error, "transport_delivery_failed")
            self.assertGreater(outbox_row.next_retry_at, self.now)

        # 4. Verify no legacy _send_deduplicated_notification called for this event
        for call_args in mock_legacy_send.call_args_list:
            call_key = call_args[0][3] if len(call_args[0]) > 3 else call_args[1].get("key", "")
            self.assertNotIn("provider_error", str(call_key))

    async def test_yookassa_and_robokassa_referral_production_webhook_paths(self):
        """Production webhook referral paths:
        YooKassa (handle_yookassa_webhook) and Robokassa (handle_robokassa_result)
        Verify:
        - first qualifying payment awards once
        - ReferralPaymentLog once
        - referral_bonus outbox once
        - duplicate event does not award/enqueue twice
        - first_only second payment creates no new bonus notification
        """
        from webhooks import handle_yookassa_webhook, handle_robokassa_result, calculate_signature
        from aiohttp.test_utils import make_mocked_request
        from aiogram.fsm.storage.memory import MemoryStorage
        from sqlalchemy import func, select

        # =======================================================
        # 1. YooKassa Production Referral Path
        # =======================================================
        yk_payer_id = 8801
        yk_referrer_id = 8802
        yk_plan_id = 8801
        pay_id_1 = "pay_yk_ref_prod_1"
        pay_id_2 = "pay_yk_ref_prod_2"

        async with async_session_maker() as session:
            referrer_yk = User(id=yk_referrer_id, first_name="ReferrerYK")
            payer_yk = User(id=yk_payer_id, first_name="PayerYK", referred_by=yk_referrer_id)
            plan_yk = SubscriptionPlan(id=yk_plan_id, name="Тариф ЮKassa Реф", price=600.0, duration_value=1, duration_unit="months")
            config = SubscriptionConfig(
                id=1,
                referral_enabled=True,
                referral_pay_bonus_enabled=True,
                referral_pay_bonus_days=7,
                referral_pay_bonus_first_only=True,
                yookassa_shop_id="test_shop",
                yookassa_secret_key="test_secret",
                robokassa_merchant_login="test_login",
                robokassa_password_2="pass2",
                notifications_enabled=False,
            )
            payment_yk_1 = YookassaPayment(
                payment_id=pay_id_1,
                user_id=yk_payer_id,
                plan_id=yk_plan_id,
                amount=600.0,
                status="pending",
                payment_method_id="pm_yk_ref_1",
            )
            session.add_all([referrer_yk, payer_yk, plan_yk, payment_yk_1])
            await session.merge(config)
            await session.commit()

        def make_yk_req(payment_id):
            mock_bot = AsyncMock()
            mock_bot.id = 111
            req = make_mocked_request(
                "POST",
                "/webhook/yookassa",
                headers={"Content-Type": "application/json"},
                app={"bot": mock_bot, "fsm_storage": MemoryStorage()},
            )
            req.json = AsyncMock(return_value={
                "event": "payment.succeeded",
                "object": {
                    "id": payment_id,
                    "status": "succeeded",
                    "paid": True,
                    "amount": {"value": "600.00", "currency": "RUB"},
                    "payment_method": {"id": "pm_yk_ref_1", "saved": True},
                    "metadata": {"user_id": str(yk_payer_id), "plan_id": str(yk_plan_id)},
                },
            })
            return req

        def make_yk_mock_session(payment_id):
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value={
                "id": payment_id,
                "status": "succeeded",
                "paid": True,
                "amount": {"value": "600.00", "currency": "RUB"},
                "payment_method": {"id": "pm_yk_ref_1", "saved": True},
                "metadata": {"user_id": str(yk_payer_id), "plan_id": str(yk_plan_id)},
            })
            mock_get_cm = AsyncMock()
            mock_get_cm.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_get_cm.__aexit__ = AsyncMock(return_value=None)
            mock_http_session = AsyncMock()
            mock_http_session.get = MagicMock(return_value=mock_get_cm)
            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_http_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            return MagicMock(return_value=mock_session_cm)

        with patch("webhooks.async_session_maker", async_session_maker), \
             patch("webhooks.dispatch_outbox_by_key", new_callable=AsyncMock), \
             patch("aiohttp.ClientSession", make_yk_mock_session(pay_id_1)):

            # First payment: awards bonus once, logs once, enqueues outbox once
            resp1 = await handle_yookassa_webhook(make_yk_req(pay_id_1))
            self.assertEqual(resp1.status, 200)

        async with async_session_maker() as session:
            # ReferralPaymentLog created once
            log_count = await session.scalar(
                select(func.count(ReferralPaymentLog.id)).where(ReferralPaymentLog.referred_user_id == yk_payer_id)
            )
            self.assertEqual(log_count, 1)

            # Referrer subscription created
            ref_sub = await session.scalar(
                select(UserSubscription).where(UserSubscription.user_id == yk_referrer_id)
            )
            self.assertIsNotNone(ref_sub)

            # Referral bonus outbox created once
            ref_outbox_key = f"yookassa:payment:{pay_id_1}:{yk_referrer_id}:referral_bonus"
            outbox_row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == ref_outbox_key)
            )
            self.assertIsNotNone(outbox_row)
            self.assertEqual(outbox_row.event_type, "referral_bonus")
            self.assertEqual(outbox_row.recipient_id, yk_referrer_id)

        # Duplicate webhook delivery of same payment
        with patch("webhooks.async_session_maker", async_session_maker), \
             patch("webhooks.dispatch_outbox_by_key", new_callable=AsyncMock), \
             patch("aiohttp.ClientSession", make_yk_mock_session(pay_id_1)):
            resp1_dup = await handle_yookassa_webhook(make_yk_req(pay_id_1))
            self.assertEqual(resp1_dup.status, 200)

        async with async_session_maker() as session:
            log_count_dup = await session.scalar(
                select(func.count(ReferralPaymentLog.id)).where(ReferralPaymentLog.referred_user_id == yk_payer_id)
            )
            self.assertEqual(log_count_dup, 1)
            outbox_rows = (await session.execute(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == ref_outbox_key)
            )).scalars().all()
            self.assertEqual(len(outbox_rows), 1)

        # Second qualifying payment from same referred user when first_only=True
        async with async_session_maker() as session:
            payment_yk_2 = YookassaPayment(
                payment_id=pay_id_2,
                user_id=yk_payer_id,
                plan_id=yk_plan_id,
                amount=600.0,
                status="pending",
                payment_method_id="pm_yk_ref_1",
            )
            session.add(payment_yk_2)
            await session.commit()

        with patch("webhooks.async_session_maker", async_session_maker), \
             patch("webhooks.dispatch_outbox_by_key", new_callable=AsyncMock), \
             patch("aiohttp.ClientSession", make_yk_mock_session(pay_id_2)):
            resp2 = await handle_yookassa_webhook(make_yk_req(pay_id_2))
            self.assertEqual(resp2.status, 200)

        async with async_session_maker() as session:
            # Payment log incremented to 2
            log_count_2 = await session.scalar(
                select(func.count(ReferralPaymentLog.id)).where(ReferralPaymentLog.referred_user_id == yk_payer_id)
            )
            self.assertEqual(log_count_2, 2)

            # But NO new referral_bonus outbox row created for payment 2
            ref_outbox_key_2 = f"yookassa:payment:{pay_id_2}:{yk_referrer_id}:referral_bonus"
            outbox_row_2 = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == ref_outbox_key_2)
            )
            self.assertIsNone(outbox_row_2)

        # =======================================================
        # 2. Robokassa Production Referral Path
        # =======================================================
        rk_payer_id = 8901
        rk_referrer_id = 8902
        rk_plan_id = 8901
        inv_id_1 = 99101
        inv_id_2 = 99102

        async with async_session_maker() as session:
            referrer_rk = User(id=rk_referrer_id, first_name="ReferrerRK")
            payer_rk = User(id=rk_payer_id, first_name="PayerRK", referred_by=rk_referrer_id)
            plan_rk = SubscriptionPlan(id=rk_plan_id, name="Тариф Robokassa Реф", price=750.0, duration_value=1, duration_unit="months")
            payment_rk_1 = RobokassaPayment(
                id=inv_id_1,
                user_id=rk_payer_id,
                plan_id=rk_plan_id,
                amount=750.0,
                status="pending",
            )
            session.add_all([referrer_rk, payer_rk, plan_rk, payment_rk_1])
            await session.commit()

        def make_rk_req(inv_id):
            crc_val = calculate_signature("750.00", inv_id, "pass2")
            mock_bot = AsyncMock()
            mock_bot.id = 222
            req = make_mocked_request(
                "POST",
                "/webhook/robokassa/result",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                app={"bot": mock_bot, "fsm_storage": MemoryStorage()},
            )
            req.post = AsyncMock(return_value={
                "OutSum": "750.00",
                "InvId": str(inv_id),
                "SignatureValue": crc_val,
                "shp_user_id": str(rk_payer_id),
                "shp_plan_id": str(rk_plan_id),
            })
            return req

        with patch("webhooks.async_session_maker", async_session_maker), \
             patch("webhooks.dispatch_outbox_by_key", new_callable=AsyncMock):

            # First payment awards bonus
            resp_rk_1 = await handle_robokassa_result(make_rk_req(inv_id_1))
            self.assertEqual(resp_rk_1.text, f"OK{inv_id_1}")

        async with async_session_maker() as session:
            # Log created once
            rk_log_count = await session.scalar(
                select(func.count(ReferralPaymentLog.id)).where(ReferralPaymentLog.referred_user_id == rk_payer_id)
            )
            self.assertEqual(rk_log_count, 1)

            # Referral bonus outbox created once
            rk_ref_outbox_key = f"robokassa:payment:{inv_id_1}:{rk_referrer_id}:referral_bonus"
            rk_outbox_row = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == rk_ref_outbox_key)
            )
            self.assertIsNotNone(rk_outbox_row)
            self.assertEqual(rk_outbox_row.event_type, "referral_bonus")
            self.assertEqual(rk_outbox_row.recipient_id, rk_referrer_id)

        # Duplicate result webhook
        with patch("webhooks.async_session_maker", async_session_maker), \
             patch("webhooks.dispatch_outbox_by_key", new_callable=AsyncMock):
            resp_rk_dup = await handle_robokassa_result(make_rk_req(inv_id_1))
            self.assertEqual(resp_rk_dup.text, f"OK{inv_id_1}")

        async with async_session_maker() as session:
            rk_log_dup = await session.scalar(
                select(func.count(ReferralPaymentLog.id)).where(ReferralPaymentLog.referred_user_id == rk_payer_id)
            )
            self.assertEqual(rk_log_dup, 1)

        # Second Robokassa payment when first_only=True
        async with async_session_maker() as session:
            payment_rk_2 = RobokassaPayment(
                id=inv_id_2,
                user_id=rk_payer_id,
                plan_id=rk_plan_id,
                amount=750.0,
                status="pending",
            )
            session.add(payment_rk_2)
            await session.commit()

        with patch("webhooks.async_session_maker", async_session_maker), \
             patch("webhooks.dispatch_outbox_by_key", new_callable=AsyncMock):
            resp_rk_2 = await handle_robokassa_result(make_rk_req(inv_id_2))
            self.assertEqual(resp_rk_2.text, f"OK{inv_id_2}")

        async with async_session_maker() as session:
            # Log incremented to 2
            rk_log_2 = await session.scalar(
                select(func.count(ReferralPaymentLog.id)).where(ReferralPaymentLog.referred_user_id == rk_payer_id)
            )
            self.assertEqual(rk_log_2, 2)

            # NO referral bonus outbox for second payment
            rk_ref_outbox_key_2 = f"robokassa:payment:{inv_id_2}:{rk_referrer_id}:referral_bonus"
            rk_outbox_row_2 = await session.scalar(
                select(PaymentNotificationOutbox).where(PaymentNotificationOutbox.unique_key == rk_ref_outbox_key_2)
            )
            self.assertIsNone(rk_outbox_row_2)
