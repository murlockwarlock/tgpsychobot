from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import exists, not_, or_, select
from sqlalchemy.orm import selectinload

from ..legacy import User, UserSubscription
from ..time_utils import utc_now


@dataclass(frozen=True, slots=True)
class ActiveSubscription:
    subscription: UserSubscription
    source: str


@dataclass(frozen=True, slots=True)
class EffectiveSubscriptionFilters:
    active: object
    paid: object
    bonus: object


def effective_subscription_filters(max_user, now: datetime) -> EffectiveSubscriptionFilters:
    linked_active = exists(
        select(UserSubscription.id).where(
            UserSubscription.user_id == max_user.tg_user_id,
            UserSubscription.end_date > now,
        )
    )
    linked_paid = exists(
        select(UserSubscription.id).where(
            UserSubscription.user_id == max_user.tg_user_id,
            UserSubscription.end_date > now,
            UserSubscription.plan_id.is_not(None),
        )
    )
    linked_bonus = exists(
        select(UserSubscription.id).where(
            UserSubscription.user_id == max_user.tg_user_id,
            UserSubscription.end_date > now,
            UserSubscription.plan_id.is_(None),
        )
    )
    own_paid = exists(
        select(UserSubscription.id).where(
            UserSubscription.user_id == max_user.id,
            UserSubscription.end_date > now,
            UserSubscription.plan_id.is_not(None),
        )
    )
    own_bonus = exists(
        select(UserSubscription.id).where(
            UserSubscription.user_id == max_user.id,
            UserSubscription.end_date > now,
            UserSubscription.plan_id.is_(None),
        )
    )
    return EffectiveSubscriptionFilters(
        active=or_(linked_active, own_paid, own_bonus),
        paid=or_(linked_paid, not_(linked_active) & own_paid),
        bonus=or_(linked_bonus, not_(linked_active) & own_bonus),
    )


def is_active_subscription(subscription: UserSubscription | None, now: datetime | None = None) -> bool:
    current_time = now or utc_now()
    return bool(subscription and subscription.end_date and subscription.end_date > current_time)


def choose_active_subscription(
    max_subscription: UserSubscription | None,
    telegram_subscription: UserSubscription | None,
    now: datetime | None = None,
) -> ActiveSubscription | None:
    """Telegram is the source of truth while its linked subscription is active."""
    current_time = now or utc_now()
    if is_active_subscription(telegram_subscription, current_time):
        return ActiveSubscription(telegram_subscription, "telegram")
    if is_active_subscription(max_subscription, current_time):
        return ActiveSubscription(max_subscription, "max")
    return None


async def load_active_subscription(
    session,
    max_user_id: int,
    now: datetime | None = None,
) -> ActiveSubscription | None:
    options = [selectinload(User.subscription).selectinload(UserSubscription.plan)]
    max_user = await session.get(User, max_user_id, options=options)
    if not max_user:
        return None

    telegram_subscription = None
    if max_user.tg_user_id is not None:
        telegram_user = await session.get(User, max_user.tg_user_id, options=options)
        if telegram_user:
            telegram_subscription = telegram_user.subscription

    return choose_active_subscription(max_user.subscription, telegram_subscription, now)
