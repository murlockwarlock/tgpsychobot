from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from subscription_context import (
    ACTIVE_SUBSCRIPTION_FLAG,
    active_subscription_flag,
    has_active_subscription,
    should_include_subscription_status,
)


def test_subscription_status_is_hidden_when_payments_are_disabled():
    config = SimpleNamespace(subscriptions_enabled=False)

    assert should_include_subscription_status(config) is False


def test_subscription_status_is_included_when_payments_are_enabled():
    config = SimpleNamespace(subscriptions_enabled=True)

    assert should_include_subscription_status(config) is True


def test_missing_config_preserves_existing_enabled_default():
    assert should_include_subscription_status(None) is True


def test_active_subscription_produces_positive_flag():
    now = datetime(2026, 7, 27, 12, 0, 0)
    config = SimpleNamespace(subscriptions_enabled=True)
    subscription = SimpleNamespace(end_date=now + timedelta(days=1))

    assert active_subscription_flag(config, subscription, now=now) == ACTIVE_SUBSCRIPTION_FLAG


def test_inactive_subscription_produces_no_flag():
    now = datetime(2026, 7, 27, 12, 0, 0)
    config = SimpleNamespace(subscriptions_enabled=True)
    subscription = SimpleNamespace(end_date=now - timedelta(seconds=1))

    assert active_subscription_flag(config, subscription, now=now) is None
    assert active_subscription_flag(config, None, now=now) is None


def test_disabled_payments_hide_flag_even_for_active_subscription():
    now = datetime(2026, 7, 27, 12, 0, 0)
    config = SimpleNamespace(subscriptions_enabled=False)
    subscription = SimpleNamespace(end_date=now + timedelta(days=1))

    assert active_subscription_flag(config, subscription, now=now) is None


def test_timezone_aware_subscription_is_supported():
    now = datetime(2026, 7, 27, 12, 0, 0, tzinfo=timezone.utc)
    subscription = SimpleNamespace(end_date=now + timedelta(hours=1))

    assert has_active_subscription(subscription, now=now) is True
