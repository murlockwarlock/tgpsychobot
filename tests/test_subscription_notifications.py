from types import SimpleNamespace

from subscription_notifications import is_trial_bonus_subscription, is_trial_expiring_subscription


def test_referral_trial_expiration_does_not_require_admin_notification():
    for provider in ("Trial Referral", "Trial Referral Bonus", "Trial Referral Pay Bonus"):
        subscription = SimpleNamespace(plan_id=None, payment_provider=provider)
        assert is_trial_bonus_subscription(subscription) is True


def test_regular_subscription_expiration_keeps_admin_notification():
    assert is_trial_bonus_subscription(SimpleNamespace(plan_id=5, payment_provider="YooKassa")) is False
    assert is_trial_bonus_subscription(SimpleNamespace(plan_id=None, payment_provider="Trial Promo")) is False


def test_all_trial_bonus_providers_receive_expiration_reminders():
    for provider in (
        "Trial Promo",
        "Trial Welcome",
        "Trial Referral",
        "Trial Referral Bonus",
        "Trial Referral Pay Bonus",
    ):
        assert is_trial_expiring_subscription(
            SimpleNamespace(plan_id=None, payment_provider=provider)
        ) is True

    assert is_trial_expiring_subscription(
        SimpleNamespace(plan_id=5, payment_provider="Trial Referral")
    ) is False
