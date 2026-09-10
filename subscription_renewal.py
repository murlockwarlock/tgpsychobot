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

from sqlalchemy import func, or_, select, update
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
    "invalid_card_number": CancellationPolicy.TERMINAL_DEACTIVATE,
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
        if hasattr(session, "in_transaction") and session.in_transaction():
            await session.commit()
        return ClaimAttemptResult(claimed=False, attempt=existing_unresolved, reason="unresolved_exists")

    # 2. Check subscription eligibility
    if (
        not sub.auto_renewal
        or not sub.payment_method_id
        or sub.payment_provider != "Yookassa"
        or sub.payment_attempt_count >= 3
    ):
        if hasattr(session, "in_transaction") and session.in_transaction():
            await session.commit()
        return ClaimAttemptResult(claimed=False, attempt=None, reason="ineligible")

    # 3. Create stable attempt record with immutable request snapshot
    now = attempt_started_at or datetime.utcnow()
    stable_key = f"yk-rec-{sub.id}-{sub.user_id}-{sub.payment_attempt_count + 1}-{uuid.uuid4().hex[:12]}"

    exact_payload = {
        "amount": {
            "value": f"{price_to_charge:.2f}",
            "currency": "RUB",
        },
        "capture": True,
        "payment_method_id": sub.payment_method_id,
        "description": f"Автопродление подписки: {plan.name}",
        "metadata": {
            "user_id": str(sub.user_id),
            "plan_id": str(plan.id),
            "recurring": "true",
            "recurring_attempt_key": stable_key,
        },
    }
    request_payload_json = json.dumps(exact_payload, ensure_ascii=False)

    attempt = YookassaRecurringAttempt(
        subscription_id=sub.id,
        user_id=sub.user_id,
        plan_id=plan.id,
        idempotency_key=stable_key,
        amount=price_to_charge,
        payment_method_id=sub.payment_method_id,
        request_payload=request_payload_json,
        status="claimed",
        attempt_started_at=now,
        client_context=client_context,
        created_at=now,
        updated_at=now,
    )

    sub_id = sub.id
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
                YookassaRecurringAttempt.subscription_id == sub_id,
                YookassaRecurringAttempt.status.in_(["claimed", "pending", "unknown"]),
            )
            .order_by(YookassaRecurringAttempt.id.desc())
        )
        if hasattr(session, "in_transaction") and session.in_transaction():
            await session.commit()
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
    raw_payload = getattr(attempt, "request_payload", None)
    if not raw_payload:
        target_logger.error(
            "TECH_RECURRING_CORRUPT_PAYLOAD | Attempt %s has NULL request_payload; failing closed",
            attempt.id,
        )
        return YooKassaRecurringResult(
            outcome="manual_review",
            payment_id=None,
            payment_status=None,
            failure_reason="missing_request_payload",
            attempt_started_at=attempt.attempt_started_at,
            error=ValueError("Missing request_payload for immutable replay"),
        )

    try:
        payload = json.loads(raw_payload)
        if not isinstance(payload, dict) or "amount" not in payload:
            raise ValueError("Invalid payload structure: missing amount dictionary")
    except Exception as e:
        target_logger.error(
            "TECH_RECURRING_CORRUPT_PAYLOAD | Attempt %s has corrupt JSON in request_payload: %s",
            attempt.id,
            e,
        )
        return YooKassaRecurringResult(
            outcome="manual_review",
            payment_id=None,
            payment_status=None,
            failure_reason="corrupt_request_payload",
            attempt_started_at=attempt.attempt_started_at,
            error=e,
        )

    payload_for_log = dict(payload)
    if "payment_method_id" in payload_for_log:
        payload_for_log["payment_method_id"] = mask_payment_method_id(payload_for_log["payment_method_id"])

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
    stable_key = (
        idempotence_key
        or f"yk-rec-sub{getattr(sub, 'id', 0)}-u{getattr(sub, 'user_id', 0)}-{now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )
    plan_name = getattr(plan, "name", "")
    payload = {
        "amount": {"value": f"{price_to_charge:.2f}", "currency": "RUB"},
        "capture": True,
        "payment_method_id": getattr(sub, "payment_method_id", None),
        "description": f"Автопродление подписки: {plan_name}",
        "metadata": {
            "user_id": str(getattr(sub, "user_id", 0)),
            "plan_id": str(getattr(plan, "id", 0)),
            "recurring": "true",
            "recurring_attempt_key": stable_key,
        },
    }
    attempt = YookassaRecurringAttempt(
        subscription_id=getattr(sub, "id", 0),
        user_id=getattr(sub, "user_id", 0),
        plan_id=getattr(plan, "id", 0),
        idempotency_key=stable_key,
        amount=price_to_charge,
        payment_method_id=getattr(sub, "payment_method_id", None),
        request_payload=json.dumps(payload, ensure_ascii=False),
        status="claimed",
        attempt_started_at=now,
        client_context="compat",
        created_at=now,
        updated_at=now,
    )
    return await execute_or_replay_yookassa_recurring_attempt(attempt, plan_name, config, logger=logger)


class FinalizePaymentSuccessResult(tuple):
    """
    Backwards-compatible 2-tuple (is_new, user_sub) that also exposes
    .is_new, .user_sub, .action, and .reconciliation_details.
    """
    def __new__(
        cls,
        is_new: bool,
        user_sub: UserSubscription | None,
        action: str = "success",
        reconciliation_details: dict[str, Any] | None = None,
    ):
        obj = super().__new__(cls, (is_new, user_sub))
        obj.is_new = is_new
        obj.user_sub = user_sub
        obj.action = action
        obj.reconciliation_details = reconciliation_details or {}
        return obj

    @property
    def subscription(self) -> UserSubscription | None:
        return self.user_sub


async def mark_unresolved_attempts_superseded(
    session: AsyncSession,
    subscription_id: int,
    now: datetime | None = None,
) -> int:
    """
    Transitions all open recurring attempts ('claimed', 'pending', 'unknown')
    for the given subscription to 'superseded'.
    Must be called inside the canonical transaction applying a confirmed explicit purchase.
    """
    current_time = now or datetime.utcnow()
    stmt = (
        update(YookassaRecurringAttempt)
        .where(
            YookassaRecurringAttempt.subscription_id == subscription_id,
            YookassaRecurringAttempt.status.in_(["claimed", "pending", "unknown"]),
        )
        .values(status="superseded", updated_at=current_time)
    )
    res = await session.execute(stmt)
    return getattr(res, "rowcount", 0)


async def finalize_yookassa_payment_success(
    session: AsyncSession,
    payment_id: str,
    user_id: int | None = None,
    plan_id: int | None = None,
    amount: float = 0.0,
    payment_method_id: str | None = None,
    is_recurring: bool = True,
    recurring_attempt_key: str | None = None,
    logger: logging.Logger | None = None,
) -> FinalizePaymentSuccessResult:
    """
    Canonical exact-once finalization for a successful YooKassa payment.
    Shared by webhook, scheduler, TG manual, and MAX manual callers.
    Correlates exact attempt via payment_id or recurring_attempt_key.
    Returns FinalizePaymentSuccessResult(is_new, user_sub, action, reconciliation_details).
    """
    if not payment_id or payment_id.startswith("yk-rec-"):
        raise ValueError(
            "A real payment_id is required for finalize_yookassa_payment_success; "
            "for no-payment outcomes use finalize_yookassa_attempt_no_payment"
        )

    now = datetime.utcnow()

    # Step 1: Correlate exact attempt if recurring (reloading fresh state from DB)
    attempt = None
    if is_recurring:
        if payment_id:
            attempt = await session.scalar(
                select(YookassaRecurringAttempt)
                .where(YookassaRecurringAttempt.payment_id == payment_id)
                .execution_options(populate_existing=True)
                .order_by(YookassaRecurringAttempt.id.desc())
            )
        if not attempt and recurring_attempt_key:
            attempt = await session.scalar(
                select(YookassaRecurringAttempt)
                .where(YookassaRecurringAttempt.idempotency_key == recurring_attempt_key)
                .execution_options(populate_existing=True)
                .order_by(YookassaRecurringAttempt.id.desc())
            )

    # Step 2: Establish authoritative parameters (exact attempt is authoritative)
    if attempt:
        effective_uid = attempt.user_id
        effective_plan_id = attempt.plan_id
        effective_amount = attempt.amount
        effective_pm = attempt.payment_method_id
        target_sub_id = attempt.subscription_id
        if plan_id and plan_id != attempt.plan_id and logger:
            logger.warning(
                f"RECURRING_ATTEMPT_PLAN_CONFLICT | payment_id={payment_id} | "
                f"attempt_plan_id={attempt.plan_id} | caller_plan_id={plan_id}"
            )
        if amount and abs(amount - attempt.amount) > 0.01 and logger:
            logger.warning(
                f"RECURRING_ATTEMPT_AMOUNT_CONFLICT | payment_id={payment_id} | "
                f"attempt_amount={attempt.amount} | caller_amount={amount}"
            )
    else:
        effective_uid = user_id
        effective_plan_id = plan_id
        effective_amount = amount
        effective_pm = payment_method_id
        target_sub_id = None

    # Step 3: Exact-once atomic CAS boundary
    prior_attempt_status = getattr(attempt, "status", None) if attempt else None

    yk_payment = await session.scalar(
        select(YookassaPayment)
        .where(YookassaPayment.payment_id == payment_id)
        .execution_options(populate_existing=True)
        .with_for_update()
    )
    if yk_payment and yk_payment.processed_at:
        if yk_payment.status in ("canceled", "deactivated"):
            # Terminal local conflict: canceled payment cannot be transitioned to completed!
            if logger:
                logger.warning(
                    f"TECH_PAYMENT_TERMINAL_CONFLICT | payment_id={payment_id} | "
                    f"local_status={yk_payment.status} | incoming_action=succeeded"
                )
            user_sub = None
            target_uid = yk_payment.user_id or effective_uid
            if target_sub_id:
                user_sub = await session.get(
                    UserSubscription, target_sub_id, execution_options={"populate_existing": True}
                )
            elif not attempt and target_uid:
                user_sub = await session.scalar(
                    select(UserSubscription).where(UserSubscription.user_id == target_uid)
                )
            return FinalizePaymentSuccessResult(False, user_sub, action="already_processed")
        elif yk_payment.status == "completed":
            # Already finalized: do not extend subscription twice!
            if attempt and attempt.status != "succeeded":
                attempt.status = "succeeded"
                attempt.payment_id = payment_id
                attempt.updated_at = now
                await session.commit()
            user_sub = None
            target_uid = yk_payment.user_id or effective_uid
            if target_sub_id:
                user_sub = await session.get(
                    UserSubscription, target_sub_id, execution_options={"populate_existing": True}
                )
            elif not attempt and target_uid:
                user_sub = await session.scalar(
                    select(UserSubscription).where(UserSubscription.user_id == target_uid)
                )
            return FinalizePaymentSuccessResult(False, user_sub, action="already_processed")

    if attempt:
        # Atomic CAS update on the attempt
        stmt = (
            update(YookassaRecurringAttempt)
            .where(
                YookassaRecurringAttempt.id == attempt.id,
                YookassaRecurringAttempt.status.in_([
                    "claimed", "pending", "unknown", "superseded", "unknown_expired", "manual_review"
                ]),
            )
            .values(
                status="succeeded",
                payment_id=payment_id,
                updated_at=now,
            )
        )
        res_cas = await session.execute(stmt)
        is_winner = (getattr(res_cas, "rowcount", 0) == 1)
        if not is_winner:
            user_sub = None
            if target_sub_id:
                user_sub = await session.get(
                    UserSubscription, target_sub_id, execution_options={"populate_existing": True}
                )
            return FinalizePaymentSuccessResult(False, user_sub, action="already_processed")
        attempt.status = "succeeded"
        attempt.payment_id = payment_id
        attempt.updated_at = now
    else:
        # Legacy recurring or direct checkout: atomic boundary on YookassaPayment
        if yk_payment:
            stmt = (
                update(YookassaPayment)
                .where(
                    YookassaPayment.payment_id == payment_id,
                    YookassaPayment.status.notin_(["completed", "canceled", "deactivated"]),
                    YookassaPayment.processed_at.is_(None),
                )
                .values(
                    status="completed",
                    processed_at=now,
                    payment_method_id=effective_pm or YookassaPayment.payment_method_id,
                )
            )
            res_cas = await session.execute(stmt)
            if getattr(res_cas, "rowcount", 0) == 0:
                sub = None
                if effective_uid:
                    sub = await session.scalar(
                        select(UserSubscription)
                        .where(UserSubscription.user_id == effective_uid)
                        .execution_options(populate_existing=True)
                    )
                return FinalizePaymentSuccessResult(False, sub, action="already_processed")
        else:
            try:
                async with session.begin_nested():
                    yk_payment = YookassaPayment(
                        payment_id=payment_id,
                        user_id=effective_uid,
                        plan_id=effective_plan_id,
                        amount=effective_amount,
                        status="completed",
                        payment_method_id=effective_pm,
                        is_recurring=is_recurring,
                        processed_at=now,
                    )
                    session.add(yk_payment)
                    await session.flush()
            except IntegrityError:
                sub = None
                if effective_uid:
                    sub = await session.scalar(
                        select(UserSubscription)
                        .where(UserSubscription.user_id == effective_uid)
                        .execution_options(populate_existing=True)
                    )
                return FinalizePaymentSuccessResult(False, sub, action="already_processed")

    # If attempt was present, ensure YookassaPayment is also persisted/updated
    if attempt:
        yk_payment = await session.scalar(
            select(YookassaPayment)
            .where(YookassaPayment.payment_id == payment_id)
            .execution_options(populate_existing=True)
        )
        if not yk_payment:
            try:
                async with session.begin_nested():
                    yk_payment = YookassaPayment(
                        payment_id=payment_id,
                        user_id=effective_uid,
                        plan_id=effective_plan_id,
                        amount=effective_amount,
                        status="completed",
                        payment_method_id=effective_pm,
                        is_recurring=is_recurring,
                        processed_at=now,
                    )
                    session.add(yk_payment)
                    await session.flush()
            except IntegrityError:
                pass
        else:
            yk_payment.status = "completed"
            yk_payment.processed_at = now
            if effective_pm:
                yk_payment.payment_method_id = effective_pm

    # Step 4: Reload fresh DB state for UserSubscription (Issue #1)
    user_sub = None
    if target_sub_id:
        user_sub = await session.get(
            UserSubscription,
            target_sub_id,
            execution_options={"populate_existing": True},
            with_for_update=True,
        )
    elif not attempt and effective_uid:
        user_sub = await session.scalar(
            select(UserSubscription)
            .where(UserSubscription.user_id == effective_uid)
            .execution_options(populate_existing=True)
            .with_for_update()
        )

    if not user_sub:
        rec_details = {
            "reason": "subscription_unresolved",
            "payment_id": payment_id,
            "attempt_id": getattr(attempt, "id", None) if attempt else None,
            "subscription_id": target_sub_id,
            "user_id": effective_uid,
            "plan_id": effective_plan_id,
            "paid_plan_id": effective_plan_id,
            "amount": effective_amount,
        }
        if logger:
            logger.error(
                f"TECH_RECURRING_SUBSCRIPTION_UNRESOLVED | payment_id={payment_id} | "
                f"attempt_id={rec_details.get('attempt_id')} | "
                f"subscription_id={target_sub_id} | user_id={effective_uid}"
            )
        await session.commit()
        return FinalizePaymentSuccessResult(
            True,
            None,
            action="manual_reconciliation_required",
            reconciliation_details=rec_details,
        )

    # Step 5: Check ownership on fresh DB state
    newer_attempt_exists = False
    if attempt:
        newer_count = await session.scalar(
            select(func.count(YookassaRecurringAttempt.id)).where(
                YookassaRecurringAttempt.subscription_id == user_sub.id,
                YookassaRecurringAttempt.id > attempt.id,
            )
        )
        newer_attempt_exists = bool(newer_count and newer_count > 0)

    attempt_was_active = prior_attempt_status in ("claimed", "pending", "unknown")

    attempt_owns_payment_binding = False
    if not is_recurring:
        attempt_owns_payment_binding = True
    elif attempt:
        if attempt_was_active and not newer_attempt_exists:
            provider_matches = (user_sub.payment_provider == "Yookassa")
            method_matches = (
                user_sub.payment_method_id is not None
                and user_sub.payment_method_id == attempt.payment_method_id
            )
            if provider_matches and method_matches:
                attempt_owns_payment_binding = True
    else:
        provider_matches = (user_sub.payment_provider in ("Yookassa", None))
        method_matches = (
            not effective_pm
            or not user_sub.payment_method_id
            or user_sub.payment_method_id == effective_pm
        )
        active_att = await session.scalar(
            select(YookassaRecurringAttempt.id).where(
                YookassaRecurringAttempt.subscription_id == user_sub.id,
                YookassaRecurringAttempt.status.in_(["claimed", "pending", "unknown"]),
            ).limit(1)
        )
        if provider_matches and method_matches and not active_att:
            attempt_owns_payment_binding = True

    # Step 6: Resolve plans fresh from DB (never rely on stale user_sub.plan relationship)
    paid_plan = None
    if effective_plan_id:
        paid_plan = await session.get(
            SubscriptionPlan, effective_plan_id, execution_options={"populate_existing": True}
        )
    current_sub_plan = None
    if user_sub.plan_id:
        current_sub_plan = await session.get(
            SubscriptionPlan, user_sub.plan_id, execution_options={"populate_existing": True}
        )

    # Blocker 1: If exact attempt plan (or effective_plan_id) cannot be resolved:
    # NEVER substitute current_sub_plan!
    if not paid_plan and (attempt or effective_plan_id):
        if logger:
            logger.error(
                f"TECH_RECURRING_PLAN_UNRESOLVED | payment_id={payment_id} | "
                f"user_id={effective_uid} | plan_id={effective_plan_id}"
            )
        rec_details = {
            "payment_id": payment_id,
            "paid_plan_id": effective_plan_id,
            "paid_plan_name": f"Unknown (ID {effective_plan_id})",
            "current_plan_id": user_sub.plan_id,
            "current_plan_name": getattr(current_sub_plan, "name", "Unknown") if current_sub_plan else "Unknown",
            "user_id": effective_uid,
            "subscription_id": user_sub.id,
            "amount": effective_amount,
            "reason": "paid_plan_unresolved",
        }
        await session.commit()
        return FinalizePaymentSuccessResult(
            True,
            user_sub,
            action="manual_reconciliation_required",
            reconciliation_details=rec_details,
        )

    plan_to_apply = paid_plan or current_sub_plan
    if not plan_to_apply:
        rec_details = {
            "payment_id": payment_id,
            "paid_plan_id": effective_plan_id,
            "paid_plan_name": f"Unknown (ID {effective_plan_id})",
            "current_plan_id": user_sub.plan_id,
            "current_plan_name": getattr(current_sub_plan, "name", "Unknown") if current_sub_plan else "Unknown",
            "user_id": effective_uid,
            "subscription_id": user_sub.id,
            "amount": effective_amount,
            "reason": "paid_plan_unresolved",
        }
        await session.commit()
        return FinalizePaymentSuccessResult(
            True,
            user_sub,
            action="manual_reconciliation_required",
            reconciliation_details=rec_details,
        )

    paid_ptc = (
        plan_to_apply.upgrades_to_plan
        if (getattr(plan_to_apply, "is_trial", False) and getattr(plan_to_apply, "upgrades_to_plan", None))
        else plan_to_apply
    )

    current_sub_ptc = (
        current_sub_plan.upgrades_to_plan
        if (current_sub_plan and getattr(current_sub_plan, "is_trial", False) and getattr(current_sub_plan, "upgrades_to_plan", None))
        else current_sub_plan
    )

    is_same_effective_plan = (
        (current_sub_ptc is not None and current_sub_ptc.id == paid_ptc.id)
        or (user_sub.plan_id == paid_ptc.id)
    )

    attempt_owns_plan_state = (
        attempt_owns_payment_binding
        and prior_attempt_status != "superseded"
        and is_same_effective_plan
    )

    if is_same_effective_plan or not is_recurring:
        # Same effective plan (or fresh explicit purchase):
        user_sub.end_date = extend_subscription_end_date(
            user_sub.end_date,
            now,
            paid_ptc.duration_value,
            paid_ptc.duration_unit,
        )
        if attempt_owns_plan_state or not is_recurring:
            user_sub.plan_id = paid_ptc.id

        if attempt_owns_payment_binding or not is_recurring:
            user_sub.payment_provider = "Yookassa"
            user_sub.payment_attempt_count = 0
            user_sub.last_payment_attempt = None
            user_sub.retry_not_before = None
            user_sub.pending_robokassa_invoice_id = None
            if not getattr(paid_ptc, 'allow_auto_renewal', True):
                user_sub.auto_renewal = False
            if effective_pm:
                user_sub.payment_method_id = effective_pm

        await session.commit()
        return FinalizePaymentSuccessResult(True, user_sub, action="success")
    else:
        # Cross-plan late success:
        rec_details = {
            "payment_id": payment_id,
            "paid_plan_id": paid_ptc.id if paid_ptc else effective_plan_id,
            "paid_plan_name": getattr(paid_ptc, "name", "Unknown"),
            "current_plan_id": user_sub.plan_id,
            "current_plan_name": getattr(current_sub_plan, "name", "Unknown") if current_sub_plan else "Unknown",
            "user_id": effective_uid,
            "subscription_id": user_sub.id,
            "amount": effective_amount,
        }
        await session.commit()
        return FinalizePaymentSuccessResult(
            True,
            user_sub,
            action="manual_reconciliation_required",
            reconciliation_details=rec_details,
        )


async def finalize_yookassa_payment_canceled(
    session: AsyncSession,
    payment_id: str,
    cancellation_reason: str | None = None,
    user_id: int | None = None,
    plan_id: int | None = None,
    amount: float = 0.0,
    payment_method_id: str | None = None,
    attempt_started_at: datetime | None = None,
    is_recurring: bool = True,
    force_deactivate: bool = False,
    recurring_attempt_key: str | None = None,
    logger: logging.Logger | None = None,
) -> tuple[bool, str, UserSubscription | None]:
    """
    Canonical exact-once finalization for a canceled/failed YooKassa payment.
    Shared by webhook, scheduler, TG manual, and MAX manual callers.
    """
    if not payment_id or payment_id.startswith("yk-rec-"):
        raise ValueError(
            "A real payment_id is required for finalize_yookassa_payment_canceled; "
            "for no-payment outcomes use finalize_yookassa_attempt_no_payment"
        )

    now = datetime.utcnow()
    attempt_ts = attempt_started_at or now

    if force_deactivate:
        policy = CancellationPolicy.TERMINAL_DEACTIVATE
    else:
        policy, _ = classify_yookassa_cancellation_reason(cancellation_reason)

    target_terminal_status = "deactivated" if policy == CancellationPolicy.TERMINAL_DEACTIVATE else "canceled"

    # Step 1: Correlate exact attempt if recurring (reloading fresh state from DB)
    attempt = None
    if is_recurring:
        if payment_id:
            attempt = await session.scalar(
                select(YookassaRecurringAttempt)
                .where(YookassaRecurringAttempt.payment_id == payment_id)
                .execution_options(populate_existing=True)
                .order_by(YookassaRecurringAttempt.id.desc())
            )
        if not attempt and recurring_attempt_key:
            attempt = await session.scalar(
                select(YookassaRecurringAttempt)
                .where(YookassaRecurringAttempt.idempotency_key == recurring_attempt_key)
                .execution_options(populate_existing=True)
                .order_by(YookassaRecurringAttempt.id.desc())
            )

    # Step 2: Establish authoritative parameters
    if attempt:
        effective_uid = attempt.user_id
        effective_plan_id = attempt.plan_id
        effective_amount = attempt.amount
        effective_pm = attempt.payment_method_id
        target_sub_id = attempt.subscription_id
    else:
        effective_uid = user_id
        effective_plan_id = plan_id
        effective_amount = amount
        effective_pm = payment_method_id
        target_sub_id = None

    # Step 3: Exact-once atomic CAS boundary
    prior_attempt_status = getattr(attempt, "status", None) if attempt else None

    yk_payment = await session.scalar(
        select(YookassaPayment)
        .where(YookassaPayment.payment_id == payment_id)
        .execution_options(populate_existing=True)
        .with_for_update()
    )
    if yk_payment and yk_payment.processed_at:
        if yk_payment.status == "completed":
            # Terminal local conflict: completed payment cannot be transitioned to canceled!
            if logger:
                logger.warning(
                    f"TECH_PAYMENT_TERMINAL_CONFLICT | payment_id={payment_id} | "
                    f"local_status=completed | incoming_action=canceled"
                )
            user_sub = None
            target_uid = yk_payment.user_id or effective_uid
            if target_sub_id:
                user_sub = await session.get(
                    UserSubscription, target_sub_id, execution_options={"populate_existing": True}
                )
            elif not attempt and target_uid:
                user_sub = await session.scalar(
                    select(UserSubscription).where(UserSubscription.user_id == target_uid)
                )
            return False, "already_processed", user_sub
        elif yk_payment.status in ("canceled", "deactivated"):
            # Already finalized: do not increment attempts or deactivate twice!
            if attempt and attempt.status not in ("canceled", "deactivated"):
                attempt.status = target_terminal_status
                attempt.payment_id = payment_id
                attempt.cancellation_reason = cancellation_reason
                attempt.updated_at = now
                await session.commit()
            user_sub = None
            target_uid = yk_payment.user_id or effective_uid
            if target_sub_id:
                user_sub = await session.get(
                    UserSubscription, target_sub_id, execution_options={"populate_existing": True}
                )
            elif not attempt and target_uid:
                user_sub = await session.scalar(
                    select(UserSubscription).where(UserSubscription.user_id == target_uid)
                )
            return False, "already_processed", user_sub

    if attempt:
        stmt = (
            update(YookassaRecurringAttempt)
            .where(
                YookassaRecurringAttempt.id == attempt.id,
                YookassaRecurringAttempt.status.in_([
                    "claimed", "pending", "unknown", "superseded", "unknown_expired", "manual_review"
                ]),
            )
            .values(
                status=target_terminal_status,
                payment_id=payment_id,
                cancellation_reason=cancellation_reason,
                updated_at=now,
            )
        )
        res_cas = await session.execute(stmt)
        is_winner = (getattr(res_cas, "rowcount", 0) == 1)
        if not is_winner:
            user_sub = None
            if target_sub_id:
                user_sub = await session.get(
                    UserSubscription, target_sub_id, execution_options={"populate_existing": True}
                )
            return False, "already_processed", user_sub
        attempt.status = target_terminal_status
        attempt.payment_id = payment_id
        attempt.cancellation_reason = cancellation_reason
        attempt.updated_at = now
    else:
        if yk_payment:
            stmt = (
                update(YookassaPayment)
                .where(
                    YookassaPayment.payment_id == payment_id,
                    YookassaPayment.status.notin_(["completed", "canceled", "deactivated"]),
                    YookassaPayment.processed_at.is_(None),
                )
                .values(
                    status="canceled",
                    processed_at=now,
                )
            )
            res_cas = await session.execute(stmt)
            if getattr(res_cas, "rowcount", 0) == 0:
                sub = None
                if effective_uid:
                    sub = await session.scalar(
                        select(UserSubscription)
                        .where(UserSubscription.user_id == effective_uid)
                        .execution_options(populate_existing=True)
                    )
                return False, "already_processed", sub
        else:
            try:
                async with session.begin_nested():
                    yk_payment = YookassaPayment(
                        payment_id=payment_id,
                        user_id=effective_uid,
                        plan_id=effective_plan_id,
                        amount=effective_amount,
                        status="canceled",
                        payment_method_id=effective_pm,
                        is_recurring=is_recurring,
                        processed_at=now,
                    )
                    session.add(yk_payment)
                    await session.flush()
            except IntegrityError:
                sub = None
                if effective_uid:
                    sub = await session.scalar(
                        select(UserSubscription)
                        .where(UserSubscription.user_id == effective_uid)
                        .execution_options(populate_existing=True)
                    )
                return False, "already_processed", sub

    if not is_recurring:
        # Ordinary checkout cancellation: exact-once payment recorded, NEVER apply recurring failure policy
        user_sub = None
        if effective_uid:
            user_sub = await session.scalar(
                select(UserSubscription)
                .where(UserSubscription.user_id == effective_uid)
                .execution_options(populate_existing=True)
            )
        await session.commit()
        return True, "ordinary_canceled", user_sub

    # Persist/update YookassaPayment record if attempt was correlated
    if attempt:
        yk_payment = await session.scalar(
            select(YookassaPayment)
            .where(YookassaPayment.payment_id == payment_id)
            .execution_options(populate_existing=True)
        )
        if not yk_payment:
            try:
                async with session.begin_nested():
                    yk_payment = YookassaPayment(
                        payment_id=payment_id,
                        user_id=effective_uid,
                        plan_id=effective_plan_id,
                        amount=effective_amount,
                        status="canceled",
                        payment_method_id=effective_pm,
                        is_recurring=is_recurring,
                        processed_at=now,
                    )
                    session.add(yk_payment)
                    await session.flush()
            except IntegrityError:
                pass
        else:
            if yk_payment.status != "completed":
                yk_payment.status = "canceled"
                yk_payment.processed_at = now

    # Step 4: Reload fresh DB state for UserSubscription (Issue #1)
    user_sub = None
    if target_sub_id:
        user_sub = await session.get(
            UserSubscription,
            target_sub_id,
            execution_options={"populate_existing": True},
            with_for_update=True,
        )
    elif not attempt and effective_uid:
        user_sub = await session.scalar(
            select(UserSubscription)
            .where(UserSubscription.user_id == effective_uid)
            .execution_options(populate_existing=True)
            .with_for_update()
        )

    if not user_sub:
        if logger:
            logger.warning(
                f"TECH_RECURRING_CANCELED_ORPHAN | payment_id={payment_id} | "
                f"attempt_id={getattr(attempt, 'id', None)} | "
                f"subscription_id={target_sub_id} | user_id={effective_uid}"
            )
        await session.commit()
        return True, "orphan_canceled", None

    # Step 5: Check binding ownership on fresh DB state
    newer_attempt_exists = False
    if attempt:
        newer_count = await session.scalar(
            select(func.count(YookassaRecurringAttempt.id)).where(
                YookassaRecurringAttempt.subscription_id == user_sub.id,
                YookassaRecurringAttempt.id > attempt.id,
            )
        )
        newer_attempt_exists = bool(newer_count and newer_count > 0)

    attempt_was_active = prior_attempt_status in ("claimed", "pending", "unknown")

    attempt_owns_payment_binding = False
    if attempt:
        if attempt_was_active and not newer_attempt_exists:
            provider_matches = (user_sub.payment_provider == "Yookassa")
            method_matches = (
                user_sub.payment_method_id is not None
                and user_sub.payment_method_id == attempt.payment_method_id
            )
            if provider_matches and method_matches:
                attempt_owns_payment_binding = True
    else:
        provider_matches = (user_sub.payment_provider in ("Yookassa", None))
        method_matches = (
            not effective_pm
            or not user_sub.payment_method_id
            or user_sub.payment_method_id == effective_pm
        )
        active_att = await session.scalar(
            select(YookassaRecurringAttempt.id).where(
                YookassaRecurringAttempt.subscription_id == user_sub.id,
                YookassaRecurringAttempt.status.in_(["claimed", "pending", "unknown"]),
            ).limit(1)
        )
        if provider_matches and method_matches and not active_att:
            attempt_owns_payment_binding = True

    action_taken = policy.value
    if attempt_owns_payment_binding:
        if policy == CancellationPolicy.TERMINAL_DEACTIVATE:
            user_sub.auto_renewal = False
            user_sub.payment_method_id = None
            user_sub.last_payment_attempt = attempt_ts
            action_taken = "deactivate"
        elif policy == CancellationPolicy.TEMPORARY_PROVIDER:
            user_sub.last_payment_attempt = attempt_ts
            action_taken = "provider_error"
        elif policy == CancellationPolicy.LIMIT_EXCEEDED:
            user_sub.payment_attempt_count += 1
            user_sub.last_payment_attempt = attempt_ts
            user_sub.retry_not_before = attempt_ts + timedelta(hours=24)
            if user_sub.payment_attempt_count >= 3:
                user_sub.auto_renewal = False
            action_taken = "limit_exceeded"
        elif policy == CancellationPolicy.UNKNOWN:
            user_sub.auto_renewal = False
            user_sub.last_payment_attempt = attempt_ts
            action_taken = "unknown_cancellation"
        else:
            user_sub.payment_attempt_count += 1
            user_sub.last_payment_attempt = attempt_ts
            if user_sub.payment_attempt_count >= 3:
                user_sub.auto_renewal = False
            action_taken = "declined"
    else:
        # Historical or superseded attempt: zero mutations to current subscription binding/retry state
        action_taken = "historical_canceled"

    await session.commit()
    return True, action_taken, user_sub


async def finalize_yookassa_attempt_no_payment(
    session: AsyncSession,
    attempt_id: int | None = None,
    outcome: str = "canceled",  # 'deactivate' | 'integration_error' | 'canceled' | 'declined'
    error_code: str | None = None,
    error_message: str | None = None,
    attempt_started_at: datetime | None = None,
    sub: UserSubscription | None = None,
    attempt: YookassaRecurringAttempt | None = None,
    logger: logging.Logger | None = None,
) -> tuple[bool, UserSubscription | None]:
    """
    Dedicated finalizer for outcomes where YooKassa rejected the request
    BEFORE creating a payment object (payment_id is None).
    NEVER creates a YookassaPayment row!
    Validates exact subscription existence BEFORE executing terminal CAS.
    If subscription is missing, fails closed without executing or committing terminal CAS.
    Uses cross-dialect atomic CAS on status IN ('claimed', 'pending', 'unknown') -> terminal_status.
    Only the CAS winner mutates the exact subscription matching attempt.subscription_id.
    Returns (is_newly_finalized, user_sub).
    """
    now = datetime.utcnow()
    target_attempt_id = attempt_id or getattr(attempt, "id", None)

    if outcome == "deactivate":
        terminal_status = "deactivated"
    elif outcome == "integration_error":
        terminal_status = "integration_error"
    else:
        terminal_status = "canceled"

    # In-memory attempt reference - always refresh from DB with populate_existing=True
    att = attempt
    if target_attempt_id is not None:
        att = await session.get(YookassaRecurringAttempt, target_attempt_id, execution_options={"populate_existing": True})
    elif att is not None and getattr(att, "id", None) is not None:
        att = await session.get(YookassaRecurringAttempt, att.id, execution_options={"populate_existing": True})

    # If attempt was already terminal in memory or DB
    if att is not None and getattr(att, "status", None) in (
        "succeeded", "deactivated", "integration_error", "canceled", "unknown_expired", "manual_review", "superseded"
    ):
        user_sub = sub if (sub and getattr(att, "subscription_id", None) == getattr(sub, "id", None)) else None
        return False, user_sub

    # Resolve and validate exact target subscription BEFORE terminal CAS (fresh from DB)
    sub_id = getattr(att, "subscription_id", None) if att else None
    user_sub = None
    if sub_id:
        db_sub = await session.get(UserSubscription, sub_id, execution_options={"populate_existing": True})
        if db_sub is not None and (isinstance(db_sub, UserSubscription) or hasattr(db_sub, "auto_renewal")):
            user_sub = db_sub
    if user_sub is None and sub and getattr(sub, "id", None) == sub_id:
        user_sub = sub

    # Missing subscription: Fail closed!
    # Do NOT execute terminal CAS! Do NOT commit terminal attempt!
    if user_sub is None:
        if logger:
            logger.error(
                f"TECH_RECURRING_ORPHAN_ATTEMPT | attempt_id={target_attempt_id} | "
                f"subscription_id={sub_id} not found in user_subscriptions"
            )
        return False, None

    # Only with exact valid subscription: perform conditional active -> terminal CAS
    is_new = True
    if target_attempt_id is not None and hasattr(session, "execute"):
        stmt = (
            update(YookassaRecurringAttempt)
            .where(
                YookassaRecurringAttempt.id == target_attempt_id,
                YookassaRecurringAttempt.status.in_(["claimed", "pending", "unknown"]),
            )
            .values(
                status=terminal_status,
                error_code=error_code,
                error_message=error_message,
                updated_at=now,
            )
        )
        res = await session.execute(stmt)
        is_new = (getattr(res, "rowcount", 0) == 1)
    elif target_attempt_id is None and att is None:
        return False, None

    if not is_new:
        return False, user_sub

    # Update in-memory attempt attributes
    if att is not None:
        att.status = terminal_status
        att.error_code = error_code
        att.error_message = error_message
        att.updated_at = now

    # Only CAS winner mutates exact subscription state
    attempt_ts = attempt_started_at or getattr(att, "attempt_started_at", None) or now
    if outcome == "deactivate":
        user_sub.auto_renewal = False
        user_sub.payment_method_id = None
        user_sub.last_payment_attempt = attempt_ts
    elif outcome == "integration_error":
        user_sub.last_payment_attempt = attempt_ts
        user_sub.retry_not_before = attempt_ts + timedelta(hours=24)
    else:
        user_sub.last_payment_attempt = attempt_ts
        if error_code == "payment_method_limit_exceeded":
            user_sub.retry_not_before = attempt_ts + timedelta(hours=24)

    if hasattr(session, "commit"):
        await session.commit()
    return True, user_sub


async def transition_attempt_to_manual_review(
    session: AsyncSession,
    attempt_id: int,
    reason: str,
    now: datetime | None = None,
    logger: logging.Logger | None = None,
) -> tuple[bool, UserSubscription | None]:
    """
    Canonical transport-neutral DB transition for attempts requiring manual review
    (e.g., corrupt or missing immutable request payloads).
    Transitions attempt status to 'manual_review', pauses unattended auto-renewal on the
    associated subscription, preserves payment_method_id, and removes attempt from active unresolved set.
    Returns (is_new, user_sub).
    """
    current_time = now or datetime.utcnow()
    stmt = (
        update(YookassaRecurringAttempt)
        .where(
            YookassaRecurringAttempt.id == attempt_id,
            YookassaRecurringAttempt.status.in_(["claimed", "pending", "unknown"]),
        )
        .values(
            status="manual_review",
            error_code="corrupt_or_missing_payload",
            error_message=reason,
            updated_at=current_time,
        )
    )
    res = await session.execute(stmt)
    is_new = (res.rowcount == 1)

    att = await session.get(YookassaRecurringAttempt, attempt_id)
    user_sub = None
    if att and att.subscription_id:
        user_sub = await session.get(UserSubscription, att.subscription_id)

    if not is_new:
        return False, user_sub

    if user_sub:
        user_sub.auto_renewal = False
        user_sub.last_payment_attempt = att.attempt_started_at if att else current_time

    await session.commit()
    return True, user_sub


async def transition_attempt_to_unknown_expired(
    session: AsyncSession,
    attempt_id: int,
    now: datetime | None = None,
) -> tuple[bool, UserSubscription | None]:
    """
    Explicit >24h terminal safety transition for unresolved attempts.
    Transitions attempt to 'unknown_expired', pauses subscription auto_renewal safely,
    and removes attempt from partial unique unresolved index.
    Returns (is_new, user_sub).
    """
    current_time = now or datetime.utcnow()
    stmt = (
        update(YookassaRecurringAttempt)
        .where(
            YookassaRecurringAttempt.id == attempt_id,
            YookassaRecurringAttempt.status.in_(["claimed", "pending", "unknown"]),
        )
        .values(
            status="unknown_expired",
            error_code="timeout_24h",
            error_message="Unresolved attempt exceeded 24 hours reconciliation window",
            last_reconciled_at=current_time,
            updated_at=current_time,
        )
    )
    res = await session.execute(stmt)
    is_new = (res.rowcount == 1)

    att = await session.get(YookassaRecurringAttempt, attempt_id)
    user_sub = None
    if att and att.subscription_id:
        user_sub = await session.get(UserSubscription, att.subscription_id)

    if not is_new:
        return False, user_sub

    if user_sub:
        user_sub.auto_renewal = False
        user_sub.last_payment_attempt = att.attempt_started_at if att else current_time

    await session.commit()
    return True, user_sub


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
