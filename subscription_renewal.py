from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import logging
from typing import Any
import uuid

try:
    from yookassa import Configuration, Payment
    from yookassa.domain.exceptions import (
        BadRequestError,
        ForbiddenError,
        InternalServerError,
        TooManyRequestsError,
        UnauthorizedError,
    )
except ModuleNotFoundError:
    Configuration = None
    Payment = None
    BadRequestError = type("BadRequestError", (Exception,), {})
    ForbiddenError = type("ForbiddenError", (Exception,), {})
    InternalServerError = type("InternalServerError", (Exception,), {})
    TooManyRequestsError = type("TooManyRequestsError", (Exception,), {})
    UnauthorizedError = type("UnauthorizedError", (Exception,), {})

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from database import User, UserSubscription, SubscriptionPlan, YookassaPayment, YookassaRecurringAttempt
from payment_failure_reasons import (
    YOOKASSA_CANCELLATION_REASON_LABELS,
    format_yookassa_cancellation_reason,
    get_yookassa_cancellation_reason,
)
from subscription_dates import extend_subscription_end_date

# Default to payment_events logger to ensure records reach logs/payment_events_<port>.log
plog = logging.getLogger("payment_events")
log = logging.getLogger("subscription_renewal")

# Circuit breaker for global shop credential alerts
_last_auth_alert_time: datetime | None = None


class CancellationPolicy(str, Enum):
    TERMINAL_DEACTIVATE = "deactivate"
    TEMPORARY_PROVIDER = "provider_error"
    LIMIT_EXCEEDED = "limit_exceeded"
    RETRYABLE_DECLINE = "declined"
    UNKNOWN = "unknown"


# Canonical cancellation taxonomy
CANCELLATION_TAXONOMY: dict[str, CancellationPolicy] = {
    # Terminal invalid payment method: clear payment_method_id, auto_renewal=False
    "card_expired": CancellationPolicy.TERMINAL_DEACTIVATE,
    "payment_method_restricted": CancellationPolicy.TERMINAL_DEACTIVATE,
    "permission_revoked": CancellationPolicy.TERMINAL_DEACTIVATE,
    "country_forbidden": CancellationPolicy.TERMINAL_DEACTIVATE,
    "fraud_suspected": CancellationPolicy.TERMINAL_DEACTIVATE,
    "payment_method_not_found": CancellationPolicy.TERMINAL_DEACTIVATE,
    "payment_method_id": CancellationPolicy.TERMINAL_DEACTIVATE,
    "invalid_request": CancellationPolicy.TERMINAL_DEACTIVATE,
    "deactivate": CancellationPolicy.TERMINAL_DEACTIVATE,
    # Temporary / provider: retry later without burning card decline attempts
    "issuer_unavailable": CancellationPolicy.TEMPORARY_PROVIDER,
    "internal_timeout": CancellationPolicy.TEMPORARY_PROVIDER,
    # Limit exceeded: retry with longer cooldown / recommend alternative
    "payment_method_limit_exceeded": CancellationPolicy.LIMIT_EXCEEDED,
    # Retryable card declines: regular policy up to 3 attempts
    "insufficient_funds": CancellationPolicy.RETRYABLE_DECLINE,
    "general_decline": CancellationPolicy.RETRYABLE_DECLINE,
    "call_issuer": CancellationPolicy.RETRYABLE_DECLINE,
    "3d_secure_failed": CancellationPolicy.RETRYABLE_DECLINE,
    "expired_on_capture": CancellationPolicy.RETRYABLE_DECLINE,
    "expired_on_confirmation": CancellationPolicy.RETRYABLE_DECLINE,
    "invalid_card_number": CancellationPolicy.TERMINAL_DEACTIVATE,
    "invalid_csc": CancellationPolicy.RETRYABLE_DECLINE,
}


def classify_yookassa_cancellation_reason(reason: str | None) -> tuple[CancellationPolicy, str]:
    """
    Classify YooKassa cancellation reason into policy and human-readable text.
    Unknown reasons fail-safe to UNKNOWN without permanent deactivation or infinite retry.
    """
    if not reason:
        return CancellationPolicy.UNKNOWN, "Причина не указана провайдером"

    clean_reason = str(reason).strip().lower()
    policy = CANCELLATION_TAXONOMY.get(clean_reason, CancellationPolicy.UNKNOWN)
    label = format_yookassa_cancellation_reason(clean_reason) or f"код ошибки: {clean_reason}"
    return policy, label


@dataclass(frozen=True)
class YooKassaRecurringResult:
    outcome: str  # 'success' | 'pending' | 'declined' | 'deactivate' | 'provider_error' | 'integration_error' | 'rate_limit' | 'unknown'
    payment_id: str | None
    payment_status: str | None
    failure_reason: str | None
    attempt_started_at: datetime
    error: Exception | None = None
    is_permanent_deactivate: bool = False


@dataclass(frozen=True)
class RenewalDetails:
    plan_to_charge: Any
    plan_name: str
    duration_text: str
    current_discount: float
    final_price: float
    attempt_count: int


@dataclass(frozen=True)
class ClaimAttemptResult:
    claimed: bool
    attempt: YookassaRecurringAttempt | None
    reason: str | None  # 'unresolved_exists' | 'ineligible' | 'conflict'


def mask_payment_method_id(pm_id: str | None) -> str:
    """Mask payment method ID to prevent raw token exposure in logs."""
    if not pm_id:
        return "none"
    clean_id = str(pm_id).strip()
    if len(clean_id) <= 8:
        return "***"
    return f"{clean_id[:4]}...{clean_id[-4:]}"


def _plog_yookassa_tech(event: str, logger: logging.Logger | None = None, **fields):
    """Log tech event to payment_events logger without exposing credentials or tokens."""
    target_logger = logger or plog
    parts = [event, "Yookassa"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    target_logger.info(" | ".join(parts))


def classify_yookassa_bad_request(e: BadRequestError) -> tuple[bool, str | None, str | None]:
    """
    Classify YooKassa BadRequestError.
    Returns (is_permanent_deactivate, error_code, parameter).
    """
    error_content = getattr(e, "content", None)
    error_code = (error_content or {}).get("code") if isinstance(error_content, dict) else None
    parameter = (error_content or {}).get("parameter") if isinstance(error_content, dict) else None
    description = (error_content or {}).get("description", "") if isinstance(error_content, dict) else str(e)

    is_permanent = (
        error_code == "payment_method_not_found"
        or parameter == "payment_method_id"
        or (
            isinstance(description, str)
            and "payment_method_id" in description.lower()
            and "exist" in description.lower()
        )
    )
    return is_permanent, error_code, parameter


def should_send_auth_alert(now: datetime) -> bool:
    """Bounded circuit breaker for global shop credentials / permissions error."""
    global _last_auth_alert_time
    if _last_auth_alert_time is None or (now - _last_auth_alert_time) > timedelta(hours=1):
        _last_auth_alert_time = now
        return True
    return False


async def claim_yookassa_recurring_attempt(
    session: AsyncSession,
    sub: UserSubscription,
    plan: SubscriptionPlan,
    price_to_charge: float,
    client_context: str = "scheduler",
    attempt_started_at: datetime | None = None,
) -> ClaimAttemptResult:
    """
    Atomic claim of one logical recurring attempt BEFORE outbound network call.
    Runs inside a DB transaction and commits before any network I/O.
    """
    # 1. Check if an unresolved attempt already exists for this subscription
    existing_unresolved = await session.scalar(
        select(YookassaRecurringAttempt)
        .where(
            YookassaRecurringAttempt.subscription_id == sub.id,
            YookassaRecurringAttempt.status.in_(["claimed", "pending", "unknown"]),
        )
        .order_by(YookassaRecurringAttempt.id.desc())
    )
    if existing_unresolved:
        return ClaimAttemptResult(claimed=False, attempt=existing_unresolved, reason="unresolved_exists")

    # 2. Check subscription eligibility
    if (
        not sub.auto_renewal
        or not sub.payment_method_id
        or sub.payment_provider != "Yookassa"
        or sub.payment_attempt_count >= 3
    ):
        return ClaimAttemptResult(claimed=False, attempt=None, reason="ineligible")

    # 3. Create stable attempt record
    now = attempt_started_at or datetime.utcnow()
    stable_key = f"yk-rec-{sub.id}-{sub.user_id}-{sub.payment_attempt_count + 1}-{uuid.uuid4().hex[:12]}"

    attempt = YookassaRecurringAttempt(
        subscription_id=sub.id,
        user_id=sub.user_id,
        plan_id=plan.id,
        idempotency_key=stable_key,
        amount=price_to_charge,
        payment_method_id=sub.payment_method_id,
        status="claimed",
        attempt_started_at=now,
        client_context=client_context,
        created_at=now,
        updated_at=now,
    )

    try:
        session.add(attempt)
        await session.commit()
        return ClaimAttemptResult(claimed=True, attempt=attempt, reason=None)
    except IntegrityError:
        await session.rollback()
        # Race: concurrent caller inserted an unresolved attempt
        existing_conflict = await session.scalar(
            select(YookassaRecurringAttempt)
            .where(
                YookassaRecurringAttempt.subscription_id == sub.id,
                YookassaRecurringAttempt.status.in_(["claimed", "pending", "unknown"]),
            )
            .order_by(YookassaRecurringAttempt.id.desc())
        )
        return ClaimAttemptResult(claimed=False, attempt=existing_conflict, reason="unresolved_exists")


async def execute_or_replay_yookassa_recurring_attempt(
    attempt: YookassaRecurringAttempt,
    plan_name: str,
    config: Any,
    logger: logging.Logger | None = None,
) -> YooKassaRecurringResult:
    """
    Execute or replay recurring attempt outside of any DB transaction.
    If attempt has a known payment_id, reconciles via GET.
    If payment_id is None, dispatches/replays POST with original Idempotence-Key.
    """
    target_logger = logger or plog

    if not config or not getattr(config, "yookassa_shop_id", None) or not getattr(config, "yookassa_secret_key", None):
        cfg_err = ValueError("YooKassa shop_id or secret_key is not configured")
        return YooKassaRecurringResult(
            outcome="provider_error",
            payment_id=attempt.payment_id,
            payment_status=None,
            failure_reason="configuration_error",
            attempt_started_at=attempt.attempt_started_at,
            error=cfg_err,
        )

    Configuration.account_id = config.yookassa_shop_id
    Configuration.secret_key = config.yookassa_secret_key

    # Reconcile if payment_id already known
    if attempt.payment_id:
        _plog_yookassa_tech(
            "TECH_RECURRING_RECONCILE_GET",
            target_logger,
            PaymentId=attempt.payment_id,
            SubscriptionId=attempt.subscription_id,
        )
        try:
            payment = await asyncio.to_thread(Payment.find_one, attempt.payment_id)
            status = getattr(payment, "status", None)
            _plog_yookassa_tech(
                "TECH_RECURRING_RECONCILE_RESPONSE",
                target_logger,
                PaymentId=attempt.payment_id,
                Status=status,
            )
            if status == "succeeded":
                return YooKassaRecurringResult(
                    outcome="success",
                    payment_id=payment.id,
                    payment_status=status,
                    failure_reason=None,
                    attempt_started_at=attempt.attempt_started_at,
                )
            elif status in ("pending", "waiting_for_capture"):
                return YooKassaRecurringResult(
                    outcome="pending",
                    payment_id=payment.id,
                    payment_status=status,
                    failure_reason=None,
                    attempt_started_at=attempt.attempt_started_at,
                )
            else:
                reason = get_yookassa_cancellation_reason(payment)
                return YooKassaRecurringResult(
                    outcome="declined",
                    payment_id=payment.id,
                    payment_status=status,
                    failure_reason=reason,
                    attempt_started_at=attempt.attempt_started_at,
                )
        except Exception as e:
            target_logger.error("Error reconciling payment_id %s: %s", attempt.payment_id, e)
            return YooKassaRecurringResult(
                outcome="unknown",
                payment_id=attempt.payment_id,
                payment_status=None,
                failure_reason=type(e).__name__,
                attempt_started_at=attempt.attempt_started_at,
                error=e,
            )

    # Dispatch / Replay POST with exact immutable payload & stable key
    payload = {
        "amount": {
            "value": f"{attempt.amount:.2f}",
            "currency": "RUB",
        },
        "capture": True,
        "payment_method_id": attempt.payment_method_id,
        "description": f"Автопродление подписки: {plan_name}",
        "metadata": {
            "user_id": str(attempt.user_id),
            "plan_id": str(attempt.plan_id),
            "recurring": "true",
        },
    }

    payload_for_log = dict(payload)
    payload_for_log["payment_method_id"] = mask_payment_method_id(attempt.payment_method_id)

    _plog_yookassa_tech(
        "TECH_RECURRING_REQUEST",
        target_logger,
        Method="POST",
        Endpoint="/v3/payments",
        UserId=attempt.user_id,
        SubscriptionId=attempt.subscription_id,
        IdempotenceKey=attempt.idempotency_key,
        Payload=json.dumps(payload_for_log, ensure_ascii=False, separators=(",", ":")),
    )

    try:
        payment = await asyncio.to_thread(Payment.create, payload, attempt.idempotency_key)
        pay_id = getattr(payment, "id", None)
        status = getattr(payment, "status", None)

        _plog_yookassa_tech(
            "TECH_RECURRING_RESPONSE",
            target_logger,
            PaymentId=pay_id,
            Status=status,
        )

        if status == "succeeded":
            return YooKassaRecurringResult(
                outcome="success",
                payment_id=pay_id,
                payment_status=status,
                failure_reason=None,
                attempt_started_at=attempt.attempt_started_at,
            )
        elif status in ("pending", "waiting_for_capture"):
            return YooKassaRecurringResult(
                outcome="pending",
                payment_id=pay_id,
                payment_status=status,
                failure_reason=None,
                attempt_started_at=attempt.attempt_started_at,
            )
        else:
            reason = get_yookassa_cancellation_reason(payment)
            return YooKassaRecurringResult(
                outcome="declined",
                payment_id=pay_id,
                payment_status=status,
                failure_reason=reason,
                attempt_started_at=attempt.attempt_started_at,
            )

    except BadRequestError as e:
        is_permanent, error_code, parameter = classify_yookassa_bad_request(e)
        _plog_yookassa_tech(
            "TECH_RECURRING_ERROR",
            target_logger,
            ErrorClass="BadRequestError",
            ErrorCode=error_code,
            Parameter=parameter,
            IsPermanent=is_permanent,
        )
        if is_permanent:
            return YooKassaRecurringResult(
                outcome="deactivate",
                payment_id=None,
                payment_status=None,
                failure_reason=error_code or "payment_method_id",
                attempt_started_at=attempt.attempt_started_at,
                error=e,
                is_permanent_deactivate=True,
            )
        return YooKassaRecurringResult(
            outcome="integration_error",
            payment_id=None,
            payment_status=None,
            failure_reason=error_code or "bad_request",
            attempt_started_at=attempt.attempt_started_at,
            error=e,
        )

    except (UnauthorizedError, ForbiddenError) as e:
        _plog_yookassa_tech(
            "TECH_RECURRING_ERROR",
            target_logger,
            ErrorClass=type(e).__name__,
        )
        return YooKassaRecurringResult(
            outcome="auth_error",
            payment_id=None,
            payment_status=None,
            failure_reason=type(e).__name__,
            attempt_started_at=attempt.attempt_started_at,
            error=e,
        )

    except TooManyRequestsError as e:
        _plog_yookassa_tech(
            "TECH_RECURRING_ERROR",
            target_logger,
            ErrorClass="TooManyRequestsError",
        )
        return YooKassaRecurringResult(
            outcome="rate_limit",
            payment_id=None,
            payment_status=None,
            failure_reason="too_many_requests",
            attempt_started_at=attempt.attempt_started_at,
            error=e,
        )

    except InternalServerError as e:
        # HTTP 500 = UNKNOWN OUTCOME
        _plog_yookassa_tech(
            "TECH_RECURRING_UNKNOWN_OUTCOME",
            target_logger,
            ErrorClass="InternalServerError",
            IdempotenceKey=attempt.idempotency_key,
        )
        return YooKassaRecurringResult(
            outcome="unknown",
            payment_id=None,
            payment_status=None,
            failure_reason="internal_server_error",
            attempt_started_at=attempt.attempt_started_at,
            error=e,
        )

    except Exception as e:
        # Ambiguous network drop = UNKNOWN OUTCOME
        _plog_yookassa_tech(
            "TECH_RECURRING_UNKNOWN_OUTCOME",
            target_logger,
            ErrorClass=type(e).__name__,
            IdempotenceKey=attempt.idempotency_key,
        )
        return YooKassaRecurringResult(
            outcome="unknown",
            payment_id=None,
            payment_status=None,
            failure_reason=type(e).__name__,
            attempt_started_at=attempt.attempt_started_at,
            error=e,
        )


async def execute_yookassa_recurring_attempt(
    sub: Any,
    plan: Any,
    price_to_charge: float,
    config: Any,
    attempt_started_at: datetime | None = None,
    idempotence_key: str | None = None,
    logger: logging.Logger | None = None,
) -> YooKassaRecurringResult:
    """
    Backward-compatible wrapper for tests and direct callers.
    Wraps parameters into a YookassaRecurringAttempt and delegates to execute_or_replay_yookassa_recurring_attempt.
    """
    now = attempt_started_at or datetime.utcnow()
    stable_key = idempotence_key or f"yk-rec-{getattr(sub, 'id', 0)}-{getattr(sub, 'user_id', 0)}-{uuid.uuid4().hex[:12]}"
    attempt = YookassaRecurringAttempt(
        subscription_id=getattr(sub, "id", 0),
        user_id=getattr(sub, "user_id", 0),
        plan_id=getattr(plan, "id", 0),
        idempotency_key=stable_key,
        amount=price_to_charge,
        payment_method_id=getattr(sub, "payment_method_id", None),
        status="claimed",
        attempt_started_at=now,
        client_context="compat",
        created_at=now,
        updated_at=now,
    )
    plan_name = getattr(plan, "name", "Подписка")
    return await execute_or_replay_yookassa_recurring_attempt(attempt, plan_name, config, logger=logger)


async def finalize_yookassa_payment_success(
    session: AsyncSession,
    payment_id: str,
    user_id: int | None = None,
    plan_id: int | None = None,
    amount: float = 0.0,
    payment_method_id: str | None = None,
    is_recurring: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[bool, UserSubscription | None]:
    """
    Canonical exact-once finalization for a successful YooKassa payment.
    Shared by webhook, scheduler, TG manual, and MAX manual callers.
    Returns (is_newly_finalized, user_sub).
    """
    now = datetime.utcnow()
    yk_payment = await session.scalar(
        select(YookassaPayment).where(YookassaPayment.payment_id == payment_id).with_for_update()
    )

    if yk_payment and yk_payment.status == "completed" and yk_payment.processed_at:
        # Already finalized: idempotent no-op
        sub = None
        target_uid = yk_payment.user_id or user_id
        if target_uid:
            sub = await session.scalar(
                select(UserSubscription).where(UserSubscription.user_id == target_uid)
            )
        return False, sub

    effective_uid = (yk_payment.user_id if yk_payment else None) or user_id
    effective_plan_id = (yk_payment.plan_id if yk_payment else None) or plan_id
    effective_amount = (yk_payment.amount if yk_payment else None) or amount

    if not yk_payment:
        yk_payment = YookassaPayment(
            payment_id=payment_id,
            user_id=effective_uid,
            plan_id=effective_plan_id,
            amount=effective_amount,
            status="completed",
            payment_method_id=payment_method_id,
            is_recurring=is_recurring,
            processed_at=now,
        )
        session.add(yk_payment)
    else:
        yk_payment.status = "completed"
        yk_payment.processed_at = now
        if payment_method_id:
            yk_payment.payment_method_id = payment_method_id

    user_sub = None
    if effective_uid:
        user_sub = await session.scalar(
            select(UserSubscription).where(UserSubscription.user_id == effective_uid).with_for_update()
        )
        if user_sub:
            plan = None
            if effective_plan_id:
                plan = await session.get(SubscriptionPlan, effective_plan_id)
            plan_to_apply = plan or user_sub.plan
            if plan_to_apply:
                ptc = plan_to_apply.upgrades_to_plan if (getattr(plan_to_apply, "is_trial", False) and getattr(plan_to_apply, "upgrades_to_plan", None)) else plan_to_apply
                user_sub.end_date = extend_subscription_end_date(
                    user_sub.end_date,
                    now,
                    ptc.duration_value,
                    ptc.duration_unit,
                )
                user_sub.plan_id = ptc.id
            user_sub.payment_provider = "Yookassa"
            user_sub.payment_attempt_count = 0
            user_sub.last_payment_attempt = None
            if payment_method_id:
                user_sub.payment_method_id = payment_method_id

            # Close any open YookassaRecurringAttempt for this subscription
            attempt = await session.scalar(
                select(YookassaRecurringAttempt)
                .where(
                    YookassaRecurringAttempt.subscription_id == user_sub.id,
                    YookassaRecurringAttempt.status.in_(["claimed", "pending", "unknown"]),
                )
                .order_by(YookassaRecurringAttempt.id.desc())
            )
            if attempt:
                attempt.status = "succeeded"
                attempt.payment_id = payment_id
                attempt.updated_at = now

    await session.commit()
    return True, user_sub


async def finalize_yookassa_payment_canceled(
    session: AsyncSession,
    payment_id: str,
    cancellation_reason: str | None = None,
    user_id: int | None = None,
    plan_id: int | None = None,
    amount: float = 0.0,
    payment_method_id: str | None = None,
    is_recurring: bool = True,
    attempt_started_at: datetime | None = None,
    force_deactivate: bool = False,
    logger: logging.Logger | None = None,
) -> tuple[bool, str, UserSubscription | None]:
    """
    Canonical exact-once finalization for a canceled YooKassa payment.
    Shared by webhook, scheduler, TG manual, and MAX manual callers.
    Returns (is_newly_finalized, action, user_sub).
    action is one of: 'deactivate', 'provider_error', 'limit_exceeded', 'declined', 'already_processed'.
    """
    now = datetime.utcnow()
    attempt_ts = attempt_started_at or now

    yk_payment = await session.scalar(
        select(YookassaPayment).where(YookassaPayment.payment_id == payment_id).with_for_update()
    )

    if yk_payment and yk_payment.status == "canceled" and yk_payment.processed_at:
        # Already finalized: do not increment attempts or deactivate twice!
        sub = None
        target_uid = yk_payment.user_id or user_id
        if target_uid:
            sub = await session.scalar(
                select(UserSubscription).where(UserSubscription.user_id == target_uid)
            )
        return False, "already_processed", sub

    effective_uid = (yk_payment.user_id if yk_payment else None) or user_id
    effective_plan_id = (yk_payment.plan_id if yk_payment else None) or plan_id
    effective_amount = (yk_payment.amount if yk_payment else None) or amount

    if not yk_payment:
        yk_payment = YookassaPayment(
            payment_id=payment_id,
            user_id=effective_uid,
            plan_id=effective_plan_id,
            amount=effective_amount,
            status="canceled",
            payment_method_id=payment_method_id,
            is_recurring=is_recurring,
            processed_at=now,
        )
        session.add(yk_payment)
    else:
        yk_payment.status = "canceled"
        yk_payment.processed_at = now

    if force_deactivate:
        policy = CancellationPolicy.TERMINAL_DEACTIVATE
    else:
        policy, _ = classify_yookassa_cancellation_reason(cancellation_reason)

    user_sub = None
    action_taken = policy.value

    if effective_uid:
        user_sub = await session.scalar(
            select(UserSubscription).where(UserSubscription.user_id == effective_uid).with_for_update()
        )
        if user_sub and is_recurring:
            if policy == CancellationPolicy.TERMINAL_DEACTIVATE:
                user_sub.auto_renewal = False
                user_sub.payment_method_id = None
                user_sub.last_payment_attempt = attempt_ts
                action_taken = "deactivate"
            elif policy == CancellationPolicy.TEMPORARY_PROVIDER:
                # Do not increment attempts
                user_sub.last_payment_attempt = attempt_ts
                action_taken = "provider_error"
            else:
                # Declined or limit exceeded
                user_sub.payment_attempt_count += 1
                user_sub.last_payment_attempt = attempt_ts
                if user_sub.payment_attempt_count >= 3:
                    user_sub.auto_renewal = False
                action_taken = "declined" if policy == CancellationPolicy.RETRYABLE_DECLINE else policy.value

            # Update open attempt
            attempt = await session.scalar(
                select(YookassaRecurringAttempt)
                .where(
                    YookassaRecurringAttempt.subscription_id == user_sub.id,
                    YookassaRecurringAttempt.status.in_(["claimed", "pending", "unknown"]),
                )
                .order_by(YookassaRecurringAttempt.id.desc())
            )
            if attempt:
                attempt.status = "deactivated" if action_taken == "deactivate" else "canceled"
                attempt.payment_id = payment_id
                attempt.cancellation_reason = cancellation_reason
                attempt.updated_at = now

    await session.commit()
    return True, action_taken, user_sub


async def update_yookassa_attempt_pending(
    session: AsyncSession,
    attempt_id: int,
    payment_id: str,
) -> None:
    """Update attempt state to pending and persist pending YookassaPayment."""
    now = datetime.utcnow()
    attempt = await session.get(YookassaRecurringAttempt, attempt_id)
    if attempt:
        attempt.status = "pending"
        attempt.payment_id = payment_id
        attempt.last_reconciled_at = now
        attempt.updated_at = now

        yk_payment = await session.scalar(
            select(YookassaPayment).where(YookassaPayment.payment_id == payment_id)
        )
        if not yk_payment:
            session.add(
                YookassaPayment(
                    payment_id=payment_id,
                    user_id=attempt.user_id,
                    plan_id=attempt.plan_id,
                    amount=attempt.amount,
                    status="pending",
                    payment_method_id=attempt.payment_method_id,
                    is_recurring=True,
                    created_at=now,
                )
            )
    await session.commit()


async def update_yookassa_attempt_unknown(
    session: AsyncSession,
    attempt_id: int,
    error_code: str | None = None,
    error_message: str | None = None,
) -> None:
    """Update attempt state to unknown (remains unresolved for reconciliation)."""
    now = datetime.utcnow()
    attempt = await session.get(YookassaRecurringAttempt, attempt_id)
    if attempt:
        attempt.status = "unknown"
        attempt.error_code = error_code
        attempt.error_message = error_message
        attempt.last_reconciled_at = now
        attempt.updated_at = now
    await session.commit()


def should_reconcile_attempt(attempt: YookassaRecurringAttempt, now: datetime) -> bool:
    """
    Bounded reconciliation backoff to avoid hammering YooKassa every 15 minutes.
    """
    if attempt.status not in ("claimed", "pending", "unknown"):
        return False

    time_since_attempt = now - attempt.attempt_started_at
    if time_since_attempt >= timedelta(hours=24):
        return False

    if attempt.last_reconciled_at is None:
        if attempt.status == "claimed":
            # If claimed and not updated after 2 minutes, caller crashed or timed out
            return (now - attempt.attempt_started_at) > timedelta(minutes=2)
        # Pending or unknown: wait at least 15 minutes from attempt start
        return (now - attempt.attempt_started_at) >= timedelta(minutes=15)

    time_since_last_check = now - attempt.last_reconciled_at

    if time_since_attempt < timedelta(hours=1):
        return time_since_last_check >= timedelta(minutes=30)
    elif time_since_attempt < timedelta(hours=6):
        return time_since_last_check >= timedelta(hours=1)
    else:
        return time_since_last_check >= timedelta(hours=2)


def calculate_renewal_details(user: Any, sub: Any) -> RenewalDetails | None:
    """
    Calculate target plan, promo discount, final renewal price, duration text,
    and payment attempt count. Shared identically between Telegram and MAX.
    """
    if not sub or not getattr(sub, "plan", None):
        return None
    plan = sub.plan
    plan_to_charge = plan.upgrades_to_plan if (getattr(plan, "is_trial", False) and getattr(plan, "upgrades_to_plan", None)) else plan
    if not plan_to_charge:
        return None

    current_discount = float(getattr(sub, "discount_percent", 0.0) or 0.0)
    user_promos = getattr(user, "promo_codes", None) if user else None
    if user_promos:
        best_promo = next(
            (
                p for p in user_promos
                if not getattr(p, "applies_to_all_plans", False)
                and any(ap.id == plan_to_charge.id for ap in getattr(p, "applicable_plans", []))
            ),
            None,
        )
        if best_promo and best_promo.discount_percent > current_discount:
            current_discount = float(best_promo.discount_percent)
        elif not best_promo:
            global_promo = next(
                (p for p in user_promos if getattr(p, "applies_to_all_plans", False)),
                None,
            )
            if global_promo and global_promo.discount_percent > current_discount:
                current_discount = float(global_promo.discount_percent)

    final_price = float(plan_to_charge.price) * (1.0 - current_discount / 100.0)
    unit = "дн." if getattr(plan_to_charge, "duration_unit", "") == "days" else "мес."
    duration_text = f"{getattr(plan_to_charge, 'duration_value', '')} {unit}"
    plan_name = getattr(plan_to_charge, "name", None) or getattr(plan, "name", "Подписка")

    return RenewalDetails(
        plan_to_charge=plan_to_charge,
        plan_name=plan_name,
        duration_text=duration_text,
        current_discount=current_discount,
        final_price=final_price,
        attempt_count=getattr(sub, "payment_attempt_count", 0) or 0,
    )
