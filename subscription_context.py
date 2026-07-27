from datetime import datetime, timezone


ACTIVE_SUBSCRIPTION_FLAG = "SUBSCRIPTION_ACTIVE=true"


def should_include_subscription_status(subscription_config) -> bool:
    if subscription_config is None:
        return True
    return bool(getattr(subscription_config, "subscriptions_enabled", True))


def has_active_subscription(user_subscription, now: datetime | None = None) -> bool:
    end_date = getattr(user_subscription, "end_date", None)
    if end_date is None:
        return False

    current = now or datetime.now(timezone.utc)
    if end_date.tzinfo is not None and current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    elif end_date.tzinfo is None and current.tzinfo is not None:
        current = current.replace(tzinfo=None)
    return end_date > current


def active_subscription_flag(
    subscription_config,
    user_subscription,
    now: datetime | None = None,
) -> str | None:
    if not should_include_subscription_status(subscription_config):
        return None
    if not has_active_subscription(user_subscription, now=now):
        return None
    return ACTIVE_SUBSCRIPTION_FLAG
