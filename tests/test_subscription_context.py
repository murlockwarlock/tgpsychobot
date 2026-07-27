from types import SimpleNamespace

from subscription_context import should_include_subscription_status


def test_subscription_status_is_hidden_when_payments_are_disabled():
    config = SimpleNamespace(subscriptions_enabled=False)

    assert should_include_subscription_status(config) is False


def test_subscription_status_is_included_when_payments_are_enabled():
    config = SimpleNamespace(subscriptions_enabled=True)

    assert should_include_subscription_status(config) is True


def test_missing_config_preserves_existing_enabled_default():
    assert should_include_subscription_status(None) is True
