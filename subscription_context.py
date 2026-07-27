def should_include_subscription_status(subscription_config) -> bool:
    if subscription_config is None:
        return True
    return bool(getattr(subscription_config, "subscriptions_enabled", True))
