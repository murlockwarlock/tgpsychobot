import unittest
from datetime import datetime, timedelta

from subscription_retry_policy import can_retry_manually, can_retry_now, get_next_retry_at


class SubscriptionRetryPolicyTests(unittest.TestCase):
    def test_first_attempt_can_run_immediately(self):
        now = datetime(2026, 3, 14, 12, 0, 0)
        self.assertTrue(can_retry_now(0, None, now))

    def test_provider_error_on_first_attempt_waits_two_hours_without_incrementing_counter(self):
        last_attempt = datetime(2026, 3, 14, 11, 14, 57)
        self.assertFalse(can_retry_now(0, last_attempt, datetime(2026, 3, 14, 13, 0, 0)))
        self.assertTrue(can_retry_now(0, last_attempt, datetime(2026, 3, 14, 13, 14, 57)))

    def test_second_attempt_waits_two_hours(self):
        last_attempt = datetime(2026, 3, 14, 11, 14, 57)
        self.assertEqual(
            get_next_retry_at(1, last_attempt),
            last_attempt + timedelta(hours=2),
        )
        self.assertFalse(can_retry_now(1, last_attempt, datetime(2026, 3, 14, 13, 0, 0)))
        self.assertTrue(can_retry_now(1, last_attempt, datetime(2026, 3, 14, 13, 14, 57)))

    def test_third_attempt_waits_twenty_four_hours(self):
        last_attempt = datetime(2026, 3, 14, 13, 14, 57)
        self.assertEqual(
            get_next_retry_at(2, last_attempt),
            last_attempt + timedelta(hours=24),
        )
        self.assertFalse(can_retry_now(2, last_attempt, datetime(2026, 3, 14, 18, 53, 23)))
        self.assertFalse(can_retry_now(2, last_attempt, datetime(2026, 3, 15, 9, 14, 57)))
        self.assertTrue(can_retry_now(2, last_attempt, datetime(2026, 3, 15, 13, 14, 57)))

    def test_no_retry_window_after_three_attempts(self):
        last_attempt = datetime(2026, 3, 14, 13, 14, 57)
        self.assertIsNone(get_next_retry_at(3, last_attempt))
        self.assertFalse(can_retry_now(3, last_attempt, datetime(2026, 3, 15, 12, 0, 0)))

    def test_manual_retry_uses_only_total_attempt_limit(self):
        self.assertTrue(can_retry_manually(0))
        self.assertTrue(can_retry_manually(1))
        self.assertTrue(can_retry_manually(2))
        self.assertFalse(can_retry_manually(3))


if __name__ == "__main__":
    unittest.main()
