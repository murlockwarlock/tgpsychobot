import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from max_messenger_bot.api import MaxApiError
from notification_transport import (
    build_subscribe_keyboard,
    clean_html_for_max,
    send_notification_transport,
    send_notification_transport_result,
)


class NotificationTransportReliabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_telegram_delivery_success(self):
        bot = MagicMock()
        bot.send_message = AsyncMock(return_value=True)

        res = await send_notification_transport(bot, 123456789, "Hello Telegram")
        self.assertTrue(res)
        bot.send_message.assert_awaited_once()

    async def test_telegram_delivery_failure_returns_false(self):
        bot = MagicMock()
        bot.send_message = AsyncMock(side_effect=Exception("Telegram Network error"))

        res = await send_notification_transport(bot, 123456789, "Hello Telegram")
        self.assertFalse(res)

    async def test_max_delivery_missing_token_fail_closed(self):
        bot = MagicMock()
        max_user_id = 100_000_000_555

        with patch.dict(os.environ, {"MAX_BOT_TOKEN": ""}, clear=False):
            res = await send_notification_transport(bot, max_user_id, "Hello MAX")
            self.assertFalse(res)

    async def test_max_delivery_success(self):
        bot = MagicMock()
        max_user_id = 100_000_000_555

        mock_client_instance = MagicMock()
        mock_client_instance.send_message = AsyncMock(return_value=True)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        with patch.dict(os.environ, {"MAX_BOT_TOKEN": "valid_max_token"}, clear=False), \
             patch("max_messenger_bot.api.MaxApiClient", return_value=mock_client_instance):
            res = await send_notification_transport(bot, max_user_id, "<b>Hello</b> MAX")
            self.assertTrue(res)
            mock_client_instance.send_message.assert_awaited_once()
            # Clean text verification
            call_kwargs = mock_client_instance.send_message.call_args[1]
            self.assertEqual(call_kwargs["text"], "Hello MAX")
            self.assertEqual(call_kwargs["user_id"], 555)

    async def test_max_robokassa_pending_delivery_preserves_all_actions(self):
        bot = MagicMock()
        max_user_id = 100_000_000_556

        mock_client_instance = MagicMock()
        mock_client_instance.send_message = AsyncMock(return_value=True)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        with patch.dict(os.environ, {"MAX_BOT_TOKEN": "valid_max_token"}, clear=False), \
             patch("max_messenger_bot.api.MaxApiClient", return_value=mock_client_instance):
            result = await send_notification_transport_result(
                bot,
                max_user_id,
                "Ожидаем подтверждение Robokassa",
                keyboard_type="robokassa_pending",
            )

        self.assertTrue(result.delivered)
        attachments = mock_client_instance.send_message.await_args.kwargs["attachments"]
        buttons = attachments[0]["payload"]["buttons"]
        self.assertEqual(
            [button["payload"] for row in buttons for button in row],
            ["sub_retry_now", "sub_cancel_retry", "sub_enter_promo"],
        )

    async def test_max_delivery_failure_returns_false(self):
        bot = MagicMock()
        max_user_id = 100_000_000_555

        mock_client_instance = MagicMock()
        mock_client_instance.send_message = AsyncMock(side_effect=Exception("MAX HTTP 500"))
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        with patch.dict(os.environ, {"MAX_BOT_TOKEN": "valid_max_token"}, clear=False), \
             patch("max_messenger_bot.api.MaxApiClient", return_value=mock_client_instance):
            res = await send_notification_transport(bot, max_user_id, "Hello MAX")
            self.assertFalse(res)

    async def test_suspended_max_recipient_is_terminal_delivery_failure(self):
        error = MaxApiError(
            "POST /messages failed: HTTP 403",
            status=403,
            error_code="chat.denied",
            payload={"code": "chat.denied", "message": "Key: error.dialog.suspended"},
        )
        self.assertTrue(error.is_suspended_recipient)

    async def test_suspended_max_delivery_is_not_retried_by_scheduler_outbox(self):
        from datetime import datetime

        import database
        from scheduler import _send_deduplicated_notification

        await database.init_db()
        max_user_id = 100_000_000_777
        error = MaxApiError(
            "POST /messages failed: HTTP 403",
            status=403,
            error_code="chat.denied",
            payload={"code": "chat.denied", "message": "Key: error.dialog.suspended"},
        )
        mock_client_instance = MagicMock()
        mock_client_instance.send_message = AsyncMock(side_effect=error)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        bot = MagicMock()
        bot.send_message = AsyncMock(return_value=True)

        with patch.dict(os.environ, {"MAX_BOT_TOKEN": "valid_max_token"}, clear=False), \
             patch("max_messenger_bot.api.MaxApiClient", return_value=mock_client_instance):
            first = await _send_deduplicated_notification(
                bot,
                max_user_id,
                "Reminder",
                "terminal-max-reminder",
                datetime.utcnow(),
            )
            second = await _send_deduplicated_notification(
                bot,
                max_user_id,
                "Reminder",
                "terminal-max-reminder",
                datetime.utcnow(),
            )

        self.assertFalse(first)
        self.assertFalse(second)
        mock_client_instance.send_message.assert_awaited_once()
        async with database.async_session_maker() as session:
            from sqlalchemy import select
            from database import PaymentNotificationOutbox

            row = await session.scalar(
                select(PaymentNotificationOutbox).where(
                    PaymentNotificationOutbox.unique_key == (
                        f"scheduler:notification:{max_user_id}:terminal-max-reminder"
                    )
                )
            )
        self.assertIsNotNone(row)
        self.assertEqual(row.status, "failed_terminal")
        self.assertEqual(row.last_error, "max_chat_suspended")

    async def test_transient_notification_failure_is_backed_off(self):
        from datetime import datetime

        import database
        from scheduler import _send_deduplicated_notification

        await database.init_db()
        bot = MagicMock()
        bot.send_message = AsyncMock(side_effect=Exception("temporary network failure"))
        user_id = 778

        first = await _send_deduplicated_notification(
            bot,
            user_id,
            "Reminder",
            "transient-reminder",
            datetime.utcnow(),
        )
        second = await _send_deduplicated_notification(
            bot,
            user_id,
            "Reminder",
            "transient-reminder",
            datetime.utcnow(),
        )

        self.assertFalse(first)
        self.assertFalse(second)
        bot.send_message.assert_awaited_once()

    async def test_stale_subscription_reminder_is_cancelled_before_retry(self):
        from datetime import datetime, timedelta

        import database
        from database import User, UserSubscription
        from scheduler import _send_deduplicated_notification, _subscription_notification_snapshot
        from sqlalchemy import select
        from database import PaymentNotificationOutbox

        await database.init_db()
        user_id = 779
        original_end = datetime.utcnow() + timedelta(days=2, minutes=30)
        async with database.async_session_maker() as session:
            session.add(User(id=user_id, first_name="Stale reminder user"))
            subscription = UserSubscription(
                id=779,
                user_id=user_id,
                plan_id=None,
                start_date=datetime.utcnow(),
                end_date=original_end,
                payment_provider="Trial Referral",
                auto_renewal=False,
            )
            session.add(subscription)
            await session.commit()

        bot = MagicMock()
        bot.send_message = AsyncMock(side_effect=Exception("temporary network failure"))
        key = "stale-reminder-after-renewal-779"
        snapshot = {
            **_subscription_notification_snapshot(subscription, "2d"),
            "end_date": original_end.isoformat(),
        }
        first = await _send_deduplicated_notification(
            bot,
            user_id,
            "Старое предупреждение",
            key,
            datetime.utcnow(),
            subscription_snapshot=snapshot,
        )
        self.assertFalse(first)

        async with database.async_session_maker() as session:
            current_sub = await session.scalar(
                select(UserSubscription).where(UserSubscription.user_id == user_id)
            )
            current_sub.end_date = datetime.utcnow() + timedelta(days=30)
            row = await session.scalar(
                select(PaymentNotificationOutbox).where(
                    PaymentNotificationOutbox.unique_key == f"scheduler:notification:{user_id}:{key}"
                )
            )
            row.next_retry_at = datetime.utcnow() - timedelta(minutes=1)
            await session.commit()

        second = await _send_deduplicated_notification(
            bot,
            user_id,
            "Старое предупреждение",
            key,
            datetime.utcnow(),
            subscription_snapshot=snapshot,
        )
        self.assertFalse(second)
        bot.send_message.assert_awaited_once()

        async with database.async_session_maker() as session:
            row = await session.scalar(
                select(PaymentNotificationOutbox).where(
                    PaymentNotificationOutbox.unique_key == f"scheduler:notification:{user_id}:{key}"
                )
            )
        self.assertEqual(row.status, "cancelled")
        self.assertEqual(row.last_error, "subscription_reminder_stale:end_date_changed")

    async def test_subscription_renewal_uses_new_reminder_outbox_key(self):
        from datetime import datetime, timedelta

        import database
        from database import User, UserSubscription
        from scheduler import (
            _send_deduplicated_notification,
            _subscription_notification_snapshot,
            _subscription_reminder_key,
        )
        from sqlalchemy import select

        await database.init_db()
        user_id = 781
        first_end = datetime.utcnow() + timedelta(days=2, minutes=30)
        async with database.async_session_maker() as session:
            session.add(
                UserSubscription(
                    id=user_id,
                    user_id=user_id,
                    plan_id=None,
                    start_date=datetime.utcnow(),
                    end_date=first_end,
                    payment_provider="Yookassa",
                    auto_renewal=True,
                )
            )
            session.add(User(id=user_id, first_name="Renewed reminder user"))
            await session.commit()

        async with database.async_session_maker() as session:
            subscription = await session.get(UserSubscription, user_id)
        first_key = _subscription_reminder_key(subscription, "renewal_reminder", "trial", "2d")
        first_result = await _send_deduplicated_notification(
            MagicMock(send_message=AsyncMock(return_value=True)),
            user_id,
            "До окончания осталось 2 дня",
            first_key,
            datetime.utcnow(),
            subscription_snapshot=_subscription_notification_snapshot(subscription, "2d"),
        )

        async with database.async_session_maker() as session:
            subscription = await session.get(UserSubscription, user_id)
            subscription.end_date = datetime.utcnow() + timedelta(days=2, minutes=30)
            await session.commit()
        second_key = _subscription_reminder_key(subscription, "renewal_reminder", "trial", "2d")
        second_result = await _send_deduplicated_notification(
            MagicMock(send_message=AsyncMock(return_value=True)),
            user_id,
            "До окончания осталось 2 дня",
            second_key,
            datetime.utcnow(),
            subscription_snapshot=_subscription_notification_snapshot(subscription, "2d"),
        )

        self.assertNotEqual(first_key, second_key)
        self.assertTrue(first_result)
        self.assertTrue(second_result)
        async with database.async_session_maker() as session:
            rows = (
                await session.execute(
                    select(database.PaymentNotificationOutbox).where(
                        database.PaymentNotificationOutbox.recipient_id == user_id
                    )
                )
            ).scalars().all()
        self.assertEqual(len(rows), 2)
        self.assertEqual({row.status for row in rows}, {"delivered"})

    async def test_yookassa_attempt_notification_is_cancelled_after_attempt_state_changes(self):
        from datetime import datetime, timedelta

        import database
        from database import User, UserSubscription, YookassaRecurringAttempt
        from scheduler import (
            _send_deduplicated_notification,
            _subscription_notification_snapshot,
            _yookassa_attempt_notification_snapshot,
        )
        from sqlalchemy import select

        await database.init_db()
        user_id = 782
        now = datetime.utcnow()
        async with database.async_session_maker() as session:
            subscription = UserSubscription(
                id=user_id,
                user_id=user_id,
                plan_id=1,
                start_date=now - timedelta(days=30),
                end_date=now + timedelta(days=30),
                payment_provider="Yookassa",
                payment_method_id="pm-yookassa-782",
                auto_renewal=True,
            )
            attempt = YookassaRecurringAttempt(
                id=user_id,
                subscription_id=user_id,
                user_id=user_id,
                plan_id=1,
                idempotency_key="yk-rec-782-state",
                amount=500.0,
                payment_method_id="pm-yookassa-782",
                status="pending",
                payment_id="yk-payment-782",
                attempt_started_at=now,
            )
            session.add_all([User(id=user_id, first_name="Yookassa user"), subscription, attempt])
            await session.commit()

        bot = MagicMock()
        bot.send_message = AsyncMock(side_effect=Exception("temporary network failure"))
        key = "yk_pending:782:yk-payment-782"
        async with database.async_session_maker() as session:
            subscription = await session.get(UserSubscription, user_id)
            attempt = await session.get(YookassaRecurringAttempt, user_id)
            snapshot = _subscription_notification_snapshot(subscription, "yookassa_pending")
            attempt_snapshot = _yookassa_attempt_notification_snapshot(attempt, "pending")

        first = await _send_deduplicated_notification(
            bot,
            user_id,
            "Запрос YooKassa ожидает подтверждения",
            key,
            datetime.utcnow(),
            subscription_snapshot=snapshot,
            payment_attempt_snapshot=attempt_snapshot,
        )
        self.assertFalse(first)

        async with database.async_session_maker() as session:
            attempt = await session.get(YookassaRecurringAttempt, user_id)
            attempt.status = "succeeded"
            row = await session.scalar(
                select(database.PaymentNotificationOutbox).where(
                    database.PaymentNotificationOutbox.unique_key == f"scheduler:notification:{user_id}:{key}"
                )
            )
            row.next_retry_at = datetime.utcnow() - timedelta(minutes=1)
            await session.commit()

        second = await _send_deduplicated_notification(
            bot,
            user_id,
            "Запрос YooKassa ожидает подтверждения",
            key,
            datetime.utcnow(),
            subscription_snapshot=snapshot,
            payment_attempt_snapshot=attempt_snapshot,
        )
        self.assertFalse(second)
        bot.send_message.assert_awaited_once()

        async with database.async_session_maker() as session:
            row = await session.scalar(
                select(database.PaymentNotificationOutbox).where(
                    database.PaymentNotificationOutbox.unique_key == f"scheduler:notification:{user_id}:{key}"
                )
            )
        self.assertEqual(row.status, "cancelled")
        self.assertEqual(row.last_error, "subscription_reminder_stale:status_changed")

    async def test_robokassa_pending_keyboard_survives_scheduler_outbox(self):
        from datetime import datetime, timedelta

        import database
        import keyboards
        from database import User, UserSubscription
        from scheduler import _send_deduplicated_notification, _subscription_notification_snapshot

        await database.init_db()
        user_id = 780
        async with database.async_session_maker() as session:
            subscription = UserSubscription(
                id=780,
                user_id=user_id,
                plan_id=1,
                start_date=datetime.utcnow() - timedelta(days=30),
                end_date=datetime.utcnow() + timedelta(hours=2),
                payment_provider="Robokassa",
                payment_method_id="pm-780",
                pending_robokassa_invoice_id=987,
                auto_renewal=True,
            )
            session.add_all([User(id=user_id, first_name="Robokassa user"), subscription])
            await session.commit()

        bot = MagicMock()
        bot.send_message = AsyncMock(return_value=True)
        result = await _send_deduplicated_notification(
            bot,
            user_id,
            "Ожидаем подтверждение Robokassa",
            "robokassa-pending-keyboard-780",
            datetime.utcnow(),
            reply_markup=keyboards.subscription_pending_keyboard("ru"),
            keyboard_type="robokassa_pending",
            keyboard_locale="ru",
            subscription_snapshot=_subscription_notification_snapshot(
                subscription,
                "robokassa_pending",
                pending_robokassa_invoice_id=987,
            ),
        )

        self.assertTrue(result)
        markup = bot.send_message.await_args.kwargs["reply_markup"]
        buttons = [button for row in markup.inline_keyboard for button in row]
        self.assertEqual(
            [button.callback_data for button in buttons],
            ["sub_retry_now", "sub_cancel_retry", "sub_enter_promo"],
        )
        self.assertEqual(
            [button.text for button in buttons],
            ["🔄 Проверить статус", "🔄 Отменить и оформить заново", "🎁 Ввести промокод"],
        )

    async def test_other_max_forbidden_response_is_not_suspended_recipient(self):
        error = MaxApiError(
            "POST /messages failed: HTTP 403",
            status=403,
            error_code="auth.forbidden",
            payload={"code": "auth.forbidden", "message": "forbidden"},
        )
        self.assertFalse(error.is_suspended_recipient)

    async def test_false_notify_sent_regression_in_scheduler(self):
        """On transport failure, _send_deduplicated_notification does not log NOTIFY_SENT and returns False."""
        from scheduler import _send_deduplicated_notification
        from types import SimpleNamespace

        bot = MagicMock()
        bot.send_message = AsyncMock(side_effect=Exception("Network failure"))
        sub = SimpleNamespace(id=42, user_id=123456)

        with patch("scheduler.plog") as mock_log:
            from datetime import datetime
            res = await _send_deduplicated_notification(bot, sub.user_id, "Test message", "test_key", datetime.utcnow(), logger=mock_log)
            self.assertFalse(res)
            # Ensure NOTIFY_SENT is never logged on failure
            for call in mock_log.info.call_args_list:
                msg = str(call)
                self.assertNotIn("NOTIFY_SENT", msg)
