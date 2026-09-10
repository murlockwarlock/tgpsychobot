from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import logging
from typing import Any

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

from payment_failure_reasons import get_yookassa_cancellation_reason

log = logging.getLogger("subscription_renewal")
plog = logging.getLogger("payments")


@dataclass(frozen=True)
class YooKassaRecurringResult:
    outcome: str  # 'success' | 'pending' | 'declined' | 'deactivate' | 'provider_error' | 'integration_error'
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


def mask_payment_method_id(pm_id: str | None) -> str:
    """Mask payment method ID to prevent raw token exposure in technical logs."""
    if not pm_id:
        return "none"
    clean_id = str(pm_id).strip()
    if len(clean_id) <= 8:
        return "***"
    return f"{clean_id[:4]}...{clean_id[-4:]}"


def _plog_yookassa_tech(event: str, **fields):
    """Log tech event without exposing raw payment_method_id or secrets."""
    parts = [event, "Yookassa"]
    for key, value in fields.items():
        if value is None:
            continue
        val_str = str(value)
        parts.append(f"{key}={val_str}")
    plog.info(" | ".join(parts))


def classify_yookassa_bad_request(e: BadRequestError) -> tuple[bool, str | None, str | None]:
    """
    Classify YooKassa BadRequestError.
    Returns (is_permanent_deactivate, error_code, parameter).
    Permanent invalid payment method if error specifically refers to payment_method_id.
    """
    error_content = getattr(e, "content", None)
    error_code = (error_content or {}).get("code") if isinstance(error_content, dict) else None
    parameter = (error_content or {}).get("parameter") if isinstance(error_content, dict) else None
    description = (error_content or {}).get("description", "") if isinstance(error_content, dict) else str(e)

    # Observed production shape: code='invalid_request', parameter='payment_method_id'
    # Legacy shape: code='payment_method_not_found'
    # Defensive fallback only: description contains 'payment_method_id' and 'exist'
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


async def execute_yookassa_recurring_attempt(
    sub: Any,
    plan: Any,
    price_to_charge: float,
    config: Any,
    attempt_started_at: datetime,
) -> YooKassaRecurringResult:
    """
    Pure transport-neutral YooKassa recurring payment execution.
    Has NO aiogram or MAX UI dependencies.
    Does NOT send user or admin notifications.
    """
    log.info(
        "Attempting recurring payment for user %s, sub %s for plan %s (%.2f RUB)",
        getattr(sub, "user_id", "unknown"),
        getattr(sub, "id", "unknown"),
        getattr(plan, "name", "unknown"),
        price_to_charge,
    )

    if not config or not getattr(config, "yookassa_shop_id", None) or not getattr(config, "yookassa_secret_key", None):
        cfg_err = ValueError("YooKassa shop_id or secret_key is not configured")
        return YooKassaRecurringResult(
            outcome="provider_error",
            payment_id=None,
            payment_status=None,
            failure_reason="configuration_error",
            attempt_started_at=attempt_started_at,
            error=cfg_err,
            is_permanent_deactivate=False,
        )

    Configuration.account_id = config.yookassa_shop_id
    Configuration.secret_key = config.yookassa_secret_key

    sub_user_id = getattr(sub, "user_id", "")
    sub_id = getattr(sub, "id", "")
    plan_id = getattr(plan, "id", "")
    sub_attempts = getattr(sub, "payment_attempt_count", 0)
    payment_method_id = getattr(sub, "payment_method_id", None)

    idempotence_key = hashlib.md5(
        f"yk-recurring:{sub_user_id}:{sub_id}:{plan_id}:{price_to_charge:.2f}:{sub_attempts}:{attempt_started_at.isoformat()}".encode()
    ).hexdigest()

    payload = {
        "amount": {
            "value": f"{price_to_charge:.2f}",
            "currency": "RUB",
        },
        "capture": True,
        "payment_method_id": payment_method_id,
        "description": f"Автопродление подписки на тариф «{getattr(plan, 'name', '')}»",
        "metadata": {
            "user_id": str(sub_user_id),
            "plan_id": str(plan_id),
            "recurring": "true",
        },
    }

    # Redact sensitive payment_method_id from technical logs
    payload_for_log = dict(payload)
    payload_for_log["payment_method_id"] = mask_payment_method_id(payment_method_id)

    _plog_yookassa_tech(
        "TECH_RECURRING_REQUEST",
        Method="POST",
        Endpoint="/v3/payments",
        UserId=sub_user_id,
        SubscriptionId=sub_id,
        IdempotenceKey=idempotence_key,
        Payload=json.dumps(payload_for_log, ensure_ascii=False, separators=(",", ":")),
    )

    try:
        payment = await asyncio.to_thread(Payment.create, payload, idempotence_key)

        _plog_yookassa_tech(
            "TECH_RECURRING_RESPONSE",
            PaymentId=getattr(payment, "id", None),
            Status=getattr(payment, "status", None),
        )

        if payment.status == "succeeded":
            log.info("Successfully charged user %s for plan %s", sub_user_id, getattr(plan, "name", ""))
            return YooKassaRecurringResult(
                outcome="success",
                payment_id=payment.id,
                payment_status=payment.status,
                failure_reason=None,
                attempt_started_at=attempt_started_at,
                error=None,
                is_permanent_deactivate=False,
            )
        if payment.status in ("pending", "waiting_for_capture"):
            log.info("Recurring payment for user %s is pending: %s", sub_user_id, payment.id)
            return YooKassaRecurringResult(
                outcome="pending",
                payment_id=payment.id,
                payment_status=payment.status,
                failure_reason=None,
                attempt_started_at=attempt_started_at,
                error=None,
                is_permanent_deactivate=False,
            )
        else:
            cancellation_reason = get_yookassa_cancellation_reason(payment)
            log.warning("Payment for user %s was created but status is %s", sub_user_id, payment.status)
            return YooKassaRecurringResult(
                outcome="declined",
                payment_id=payment.id,
                payment_status=payment.status,
                failure_reason=cancellation_reason,
                attempt_started_at=attempt_started_at,
                error=None,
                is_permanent_deactivate=False,
            )

    except BadRequestError as e:
        is_permanent, error_code, parameter = classify_yookassa_bad_request(e)
        _plog_yookassa_tech(
            "TECH_RECURRING_ERROR",
            ErrorClass=type(e).__name__,
            ErrorCode=error_code,
            Parameter=parameter,
            IsPermanent=is_permanent,
        )
        if is_permanent:
            log.warning("Permanent invalid payment_method for user %s: code=%s, parameter=%s", sub_user_id, error_code, parameter)
            return YooKassaRecurringResult(
                outcome="deactivate",
                payment_id=None,
                payment_status=None,
                failure_reason=error_code or "payment_method_id",
                attempt_started_at=attempt_started_at,
                error=e,
                is_permanent_deactivate=True,
            )
        log.error("Integration BadRequestError for user %s: %s", sub_user_id, e)
        return YooKassaRecurringResult(
            outcome="integration_error",
            payment_id=None,
            payment_status=None,
            failure_reason=error_code,
            attempt_started_at=attempt_started_at,
            error=e,
            is_permanent_deactivate=False,
        )

    except (ForbiddenError, InternalServerError, TooManyRequestsError, UnauthorizedError) as e:
        log.error("API error during recurring charge for user %s: %s", sub_user_id, e)
        _plog_yookassa_tech(
            "TECH_RECURRING_ERROR",
            ErrorClass=type(e).__name__,
        )
        return YooKassaRecurringResult(
            outcome="provider_error",
            payment_id=None,
            payment_status=None,
            failure_reason=type(e).__name__,
            attempt_started_at=attempt_started_at,
            error=e,
            is_permanent_deactivate=False,
        )

    except Exception as e:
        log.error("Unexpected error during recurring charge for user %s: %s", sub_user_id, e)
        _plog_yookassa_tech(
            "TECH_RECURRING_ERROR",
            ErrorClass=type(e).__name__,
        )
        return YooKassaRecurringResult(
            outcome="provider_error",
            payment_id=None,
            payment_status=None,
            failure_reason=type(e).__name__,
            attempt_started_at=attempt_started_at,
            error=e,
            is_permanent_deactivate=False,
        )


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
