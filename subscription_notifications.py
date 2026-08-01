TRIAL_BONUS_PAYMENT_PROVIDERS = frozenset({
    "Trial Referral",
    "Trial Referral Bonus",
    "Trial Referral Pay Bonus",
})


def is_trial_bonus_subscription(subscription) -> bool:
    return bool(
        subscription
        and getattr(subscription, "plan_id", None) is None
        and getattr(subscription, "payment_provider", None) in TRIAL_BONUS_PAYMENT_PROVIDERS
    )


def get_target_plan(plan):
    if not plan:
        return None
    if getattr(plan, "is_trial", False) and getattr(plan, "upgrades_to_plan", None):
        return plan.upgrades_to_plan
    return plan


def should_send_upcoming_charge_notification(auto_renewal_enabled, plan):
    target_plan = get_target_plan(plan)
    if not auto_renewal_enabled or not target_plan:
        return False
    return bool(getattr(target_plan, "allow_auto_renewal", True))
