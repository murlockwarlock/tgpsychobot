import os
import time
import unittest
from datetime import datetime, timezone, timedelta

from time_helpers import format_msk, to_msk, utc_now
from notification_renderer import render_outbox_message


class TimezoneNormalizationTests(unittest.TestCase):
    def test_exact_naive_utc_regression(self):
        """2026-09-13 16:47:05 naive UTC must render: 13.09.2026 19:47 МСК."""
        sample_naive_utc = datetime(2026, 9, 13, 16, 47, 5)
        formatted = format_msk(sample_naive_utc)
        self.assertEqual(formatted, "13.09.2026 19:47 МСК")

    def test_host_timezone_invariance(self):
        """format_msk must produce identical results regardless of host TZ (UTC, US/Pacific, Asia/Tokyo)."""
        sample_naive_utc = datetime(2026, 9, 13, 16, 47, 5)
        expected = "13.09.2026 19:47 МСК"

        original_tz = os.environ.get("TZ")
        try:
            for tz_name in ["UTC", "America/New_York", "Asia/Tokyo", "Europe/London", "US/Pacific"]:
                os.environ["TZ"] = tz_name
                if hasattr(time, "tzset"):
                    time.tzset()
                self.assertEqual(
                    format_msk(sample_naive_utc),
                    expected,
                    f"Failed under host TZ={tz_name}",
                )
        finally:
            if original_tz is not None:
                os.environ["TZ"] = original_tz
            else:
                os.environ.pop("TZ", None)
            if hasattr(time, "tzset"):
                time.tzset()

    def test_renderer_with_msk_date(self):
        """Rendered outbox messages use MSK formatted end dates according to contract."""
        sample_naive_utc = datetime(2026, 9, 13, 16, 47, 5)
        msk_str = format_msk(sample_naive_utc, "%d.%m.%Y %H:%M")

        # YooKassa renewal
        yoo_payload = {
            "user_id": 12345,
            "provider": "Yookassa",
            "end_date_msk": msk_str,
        }
        text, _, _ = render_outbox_message("renewal_success", yoo_payload, 12345)
        self.assertEqual(text, "✅ Подписка продлена до 13.09.2026 19:47 МСК.")

        # Robokassa renewal
        robo_payload = {
            "user_id": 12345,
            "provider": "Robokassa",
            "amount": 490.0,
            "plan_name": "Базовый",
            "end_date_msk": msk_str,
        }
        text, _, _ = render_outbox_message("renewal_success", robo_payload, 12345)
        self.assertIn("Действие тарифа продлено до 13.09.2026 19:47 МСК.", text)
        self.assertNotIn("МСК МСК", text)
        self.assertIn("490.00 руб", text)
        self.assertIn("Базовый", text)


class MaxSubscriptionTimezoneTests(unittest.IsolatedAsyncioTestCase):
    async def test_max_subscriptions_set_renewal_tz_invariance(self):
        """Test the actual affected MAX subscriptions set_renewal path under UTC, Europe/Moscow, America/New_York."""
        from unittest.mock import AsyncMock, patch, MagicMock
        from sqlalchemy import select
        from database import init_db, async_session_maker, User, SubscriptionPlan, UserSubscription
        from max_messenger_bot.services.subscriptions import set_renewal

        await init_db()

        # Seed plan and user with naive UTC end_date: 2026-09-13 16:47:05
        sample_naive_utc = datetime(2026, 9, 13, 16, 47, 5)
        max_user_id = 100_000_000_777

        original_tz = os.environ.get("TZ")
        try:
            for tz_name in ["UTC", "Europe/Moscow", "America/New_York"]:
                os.environ["TZ"] = tz_name
                if hasattr(time, "tzset"):
                    time.tzset()

                async with async_session_maker() as session:
                    plan = await session.get(SubscriptionPlan, 888)
                    if not plan:
                        plan = SubscriptionPlan(
                            id=888,
                            name="MAX VIP",
                            duration_value=1,
                            duration_unit="month",
                            price=500.0,
                        )
                        session.add(plan)
                    user = await session.get(User, max_user_id)
                    if not user:
                        user = User(id=max_user_id, first_name="MaxUser")
                        session.add(user)
                    sub = await session.scalar(
                        select(UserSubscription).where(UserSubscription.user_id == max_user_id)
                    )
                    if not sub:
                        sub = UserSubscription(
                            user_id=max_user_id,
                            plan_id=888,
                            start_date=datetime(2026, 8, 13, 16, 47, 5),
                            end_date=sample_naive_utc,
                            auto_renewal=True,
                        )
                        session.add(sub)
                    else:
                        sub.end_date = sample_naive_utc
                        sub.auto_renewal = True
                    await session.commit()

                mock_client = MagicMock()
                mock_client.send_message = AsyncMock(return_value=True)

                with patch("max_messenger_bot.services.common.notify_telegram_admins", new_callable=AsyncMock) as mock_notify, \
                     patch("max_messenger_bot.services.subscriptions.show_subscription_info", new_callable=AsyncMock):
                    await set_renewal(mock_client, 12345, max_user_id, False)
                    mock_notify.assert_awaited_once()
                    admin_msg = mock_notify.call_args[0][0]
                    self.assertIn(
                        "Подписка до: 13.09.2026 19:47 МСК",
                        admin_msg,
                        f"Failed in host TZ={tz_name}: got {admin_msg}",
                    )
                    self.assertNotIn("МСК МСК", admin_msg)
        finally:
            if original_tz is not None:
                os.environ["TZ"] = original_tz
            else:
                os.environ.pop("TZ", None)
            if hasattr(time, "tzset"):
                time.tzset()
