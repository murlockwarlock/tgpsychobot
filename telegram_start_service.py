from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import and_, or_, select, update

from database import SubscriptionBenefitGrant, TelegramStartIntent, User, UserSubscription
from translation_service import normalize_enabled_languages, normalize_locale


LANGUAGE_INTENT_STATUS = "awaiting_language"
PROFILE_INTENT_STATUS = "awaiting_profile"
COMPLETED_INTENT_STATUS = "completed"
LANGUAGE_RESUME_LEASE_SECONDS = 120


@dataclass(frozen=True)
class StartIntentLease:
    user_id: int
    token: str


def parse_referral_payload(args: str | None, user_id: int) -> int | None:
    if not isinstance(args, str) or not args.startswith("ref_"):
        return None
    raw = args[4:]
    try:
        referrer_id = int(raw)
    except (TypeError, ValueError):
        return None
    if referrer_id <= 0 or referrer_id == user_id:
        return None
    return referrer_id


def language_selection_enabled_for_user(
    *,
    selector_enabled: bool,
    enabled_languages,
    user_language,
    intent_new_user_eligible: bool,
) -> bool:
    if not selector_enabled or not intent_new_user_eligible:
        return False
    enabled = normalize_enabled_languages(enabled_languages)
    if len(enabled) <= 1:
        return False
    return normalize_locale(user_language) not in enabled


async def record_start_intent(
    session,
    *,
    user_id: int,
    args: str | None,
    new_user_eligible: bool,
) -> TelegramStartIntent:
    intent = await session.get(TelegramStartIntent, user_id)
    if intent is None:
        intent = TelegramStartIntent(
            user_id=user_id,
            new_user_eligible=bool(new_user_eligible),
            status=LANGUAGE_INTENT_STATUS,
        )
        session.add(intent)
        await session.flush()
    elif new_user_eligible:
        intent.new_user_eligible = True

    intent.navigation_payload = args

    candidate_referrer = parse_referral_payload(args, user_id)
    if intent.acquisition_referrer_id is None and candidate_referrer:
        referrer_exists = await session.scalar(
            select(User.id).where(User.id == candidate_referrer)
        )
        if referrer_exists is not None:
            intent.acquisition_referrer_id = candidate_referrer
            intent.acquisition_payload = f"ref_{candidate_referrer}"

    if args == "test" and not intent.deferred_test_key:
        intent.deferred_test_key = f"telegram-test:{user_id}:{secrets.token_hex(12)}"
    intent.updated_at = datetime.utcnow()
    return intent


async def grant_subscription_days(
    session,
    *,
    grant_key: str,
    grant_type: str,
    beneficiary_user_id: int,
    days: int,
    payment_provider: str,
    now: datetime | None = None,
    source_user_id: int | None = None,
) -> bool:
    if not isinstance(days, int) or days <= 0:
        return False
    now = now or datetime.utcnow()
    user = await session.scalar(
        select(User)
        .where(User.id == beneficiary_user_id)
        .with_for_update()
    )
    if user is None:
        return False

    bind = session.get_bind()
    dialect_name = getattr(getattr(bind, "dialect", None), "name", "")
    if dialect_name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as dialect_insert
    else:
        from sqlalchemy.dialects.sqlite import insert as dialect_insert

    result = await session.execute(
        dialect_insert(SubscriptionBenefitGrant)
        .values(
            grant_key=grant_key,
            grant_type=grant_type,
            beneficiary_user_id=beneficiary_user_id,
            source_user_id=source_user_id,
            days=days,
            created_at=now,
        )
        .on_conflict_do_nothing(index_elements=["grant_key"])
    )
    if result.rowcount != 1:
        return False

    subscription = await session.scalar(
        select(UserSubscription)
        .where(UserSubscription.user_id == beneficiary_user_id)
        .with_for_update()
    )
    if subscription and subscription.end_date and subscription.end_date > now:
        subscription.end_date += timedelta(days=days)
    elif subscription:
        subscription.plan_id = None
        subscription.start_date = now
        subscription.end_date = now + timedelta(days=days)
        subscription.payment_provider = payment_provider
        subscription.auto_renewal = False
        subscription.payment_attempt_count = 0
        subscription.last_payment_attempt = None
        subscription.retry_not_before = None
    else:
        session.add(
            UserSubscription(
                user_id=beneficiary_user_id,
                plan_id=None,
                start_date=now,
                end_date=now + timedelta(days=days),
                auto_renewal=False,
                payment_provider=payment_provider,
                payment_attempt_count=0,
                discount_percent=0,
            )
        )
    await session.flush()
    return True


async def claim_language_resume_lease(
    session,
    *,
    user_id: int,
    now: datetime | None = None,
) -> StartIntentLease | None:
    now = now or datetime.utcnow()
    token = secrets.token_hex(24)
    lease_until = now + timedelta(seconds=LANGUAGE_RESUME_LEASE_SECONDS)
    result = await session.execute(
        update(TelegramStartIntent)
        .where(
            TelegramStartIntent.user_id == user_id,
            TelegramStartIntent.status == LANGUAGE_INTENT_STATUS,
            or_(
                TelegramStartIntent.lease_until.is_(None),
                TelegramStartIntent.lease_until <= now,
            ),
        )
        .values(lease_token=token, lease_until=lease_until, updated_at=now)
    )
    if result.rowcount != 1:
        return None
    return StartIntentLease(user_id=user_id, token=token)


async def complete_language_selection(
    session,
    *,
    user_id: int,
    locale: str,
    lease_token: str | None = None,
) -> bool:
    normalized = normalize_locale(locale)
    if normalized is None:
        return False
    conditions = [
        TelegramStartIntent.user_id == user_id,
        TelegramStartIntent.status == LANGUAGE_INTENT_STATUS,
    ]
    if lease_token is not None:
        conditions.append(TelegramStartIntent.lease_token == lease_token)
    result = await session.execute(
        update(TelegramStartIntent)
        .where(and_(*conditions))
        .values(
            status=PROFILE_INTENT_STATUS,
            lease_token=None,
            lease_until=None,
            updated_at=datetime.utcnow(),
        )
    )
    if result.rowcount != 1:
        return False
    user = await session.get(User, user_id)
    if user is None:
        return False
    user.telegram_language_code = normalized
    return True


async def mark_start_intent_completed(session, user_id: int) -> None:
    intent = await session.get(TelegramStartIntent, user_id)
    if intent is None:
        return
    intent.status = COMPLETED_INTENT_STATUS
    intent.lease_token = None
    intent.lease_until = None
    intent.updated_at = datetime.utcnow()
