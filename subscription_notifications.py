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
