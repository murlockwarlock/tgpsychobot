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
        """Rendered outbox messages use MSK formatted end dates."""
        sample_naive_utc = datetime(2026, 9, 13, 16, 47, 5)
        msk_str = format_msk(sample_naive_utc)

        # YooKassa renewal
        yoo_payload = {
            "user_id": 12345,
            "provider": "Yookassa",
            "end_date_msk": msk_str,
        }
        text, _, _ = render_outbox_message("renewal_success", yoo_payload, 12345)
        self.assertIn("13.09.2026 19:47 МСК", text)

        # Robokassa renewal
        robo_payload = {
            "user_id": 12345,
            "provider": "Robokassa",
            "amount": 490.0,
            "plan_name": "Базовый",
            "end_date_msk": msk_str,
        }
        text, _, _ = render_outbox_message("renewal_success", robo_payload, 12345)
        self.assertIn("13.09.2026 19:47 МСК", text)
        self.assertIn("490.00 руб", text)
        self.assertIn("Базовый", text)
