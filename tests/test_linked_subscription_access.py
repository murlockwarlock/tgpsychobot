import os
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import sqlalchemy.ext.asyncio as sqlalchemy_asyncio


os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

original_create_async_engine = sqlalchemy_asyncio.create_async_engine


def _sqlite_compatible_engine(*args, **kwargs):
    kwargs.pop("pool_recycle", None)
    kwargs.pop("pool_use_lifo", None)
    return original_create_async_engine(*args, **kwargs)


with patch.object(sqlalchemy_asyncio, "create_async_engine", _sqlite_compatible_engine):
    from client_search import normalize_client_search_query
    from max_messenger_bot.services.subscription_access import choose_active_subscription


class LinkedSubscriptionPriorityTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 8, 5, 7, 0, 0)

    @staticmethod
    def subscription(end_date):
        return SimpleNamespace(end_date=end_date)

    def test_active_telegram_subscription_has_priority(self):
        max_subscription = self.subscription(self.now + timedelta(days=60))
        telegram_subscription = self.subscription(self.now + timedelta(days=10))

        result = choose_active_subscription(max_subscription, telegram_subscription, self.now)

        self.assertIs(result.subscription, telegram_subscription)
        self.assertEqual(result.source, "telegram")

    def test_active_max_subscription_is_fallback(self):
        max_subscription = self.subscription(self.now + timedelta(days=5))
        telegram_subscription = self.subscription(self.now - timedelta(seconds=1))

        result = choose_active_subscription(max_subscription, telegram_subscription, self.now)

        self.assertIs(result.subscription, max_subscription)
        self.assertEqual(result.source, "max")

    def test_no_active_subscription_returns_none(self):
        expired = self.subscription(self.now - timedelta(seconds=1))
        self.assertIsNone(choose_active_subscription(expired, expired, self.now))


class ClientSearchNormalizationTests(unittest.TestCase):
    def test_leading_at_sign_is_accepted(self):
        self.assertEqual(normalize_client_search_query("  @dgsoldatov  "), "dgsoldatov")

    def test_plain_username_is_unchanged(self):
        self.assertEqual(normalize_client_search_query("dgsoldatov"), "dgsoldatov")
