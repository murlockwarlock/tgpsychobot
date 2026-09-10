import asyncio
import json
import logging
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from yookassa.domain.exceptions import (
    BadRequestError,
    ForbiddenError,
    InternalServerError,
    TooManyRequestsError,
    UnauthorizedError,
)

from subscription_renewal import (
    RenewalDetails,
    YooKassaRecurringResult,
    calculate_renewal_details,
    classify_yookassa_bad_request,
    execute_yookassa_recurring_attempt,
    mask_payment_method_id,
)


def _make_bad_request(content: dict) -> BadRequestError:
    return BadRequestError(content)


def test_classify_yookassa_bad_request_production_observed():
    # Observed production shape: code='invalid_request', parameter='payment_method_id'
    err = _make_bad_request({
        "type": "error",
        "id": "abc-123",
        "code": "invalid_request",
        "description": "Invalid parameter",
        "parameter": "payment_method_id",
    })
    is_perm, code, param = classify_yookassa_bad_request(err)
    assert is_perm is True
    assert code == "invalid_request"
    assert param == "payment_method_id"


def test_classify_yookassa_bad_request_legacy_payment_method_not_found():
    # Legacy shape: code='payment_method_not_found'
    err = _make_bad_request({
        "type": "error",
        "id": "abc-456",
        "code": "payment_method_not_found",
        "description": "Payment method not found",
    })
    is_perm, code, param = classify_yookassa_bad_request(err)
    assert is_perm is True
    assert code == "payment_method_not_found"


def test_classify_yookassa_bad_request_description_fallback():
    # Defensive description fallback
    err = _make_bad_request({
        "type": "error",
        "code": "invalid_request",
        "description": "payment_method_id does not exist",
    })
    is_perm, code, param = classify_yookassa_bad_request(err)
    assert is_perm is True


def test_classify_yookassa_bad_request_unrelated_is_not_permanent():
    # Unrelated invalid_request, e.g. amount or metadata
    err = _make_bad_request({
        "type": "error",
        "code": "invalid_request",
        "description": "Invalid amount",
        "parameter": "amount",
    })
    is_perm, code, param = classify_yookassa_bad_request(err)
    assert is_perm is False
    assert code == "invalid_request"
    assert param == "amount"


def test_mask_payment_method_id():
    assert mask_payment_method_id(None) == "none"
    assert mask_payment_method_id("") == "none"
    assert mask_payment_method_id("short") == "***"
    assert mask_payment_method_id("31d55000-000f-5000-9000-111111111111") == "31d5...1111"


@pytest.mark.asyncio
async def test_execute_yookassa_recurring_attempt_redacts_payment_method_id_in_logs(caplog):
    sub = SimpleNamespace(
        id=10,
        user_id=12345678,
        payment_attempt_count=0,
        payment_method_id="31d55000-000f-5000-9000-1e54ca65d0cf",
    )
    plan = SimpleNamespace(id=1, name="Standard", price=195.0)
    config = SimpleNamespace(yookassa_shop_id="test-shop", yookassa_secret_key="test-secret")
    attempt_time = datetime(2026, 9, 10, 12, 0, 0)

    fake_payment = SimpleNamespace(id="yk-pay-1", status="succeeded")

    with caplog.at_level(logging.INFO, logger="payment_events"):
        with patch("subscription_renewal.Payment.create", return_value=fake_payment):
            res = await execute_yookassa_recurring_attempt(sub, plan, 195.0, config, attempt_time)

    assert res.outcome == "success"
    assert res.payment_id == "yk-pay-1"

    # Verify technical log output
    log_text = caplog.text
    # Raw payment method id MUST NOT be present
    assert "31d55000-000f-5000-9000-1e54ca65d0cf" not in log_text
    # Masked token SHOULD be present
    assert "31d5...d0cf" in log_text


@pytest.mark.asyncio
async def test_execute_yookassa_recurring_attempt_deactivate_on_invalid_saved_method():
    sub = SimpleNamespace(
        id=10,
        user_id=12345678,
        payment_attempt_count=0,
        payment_method_id="31d55000-000f-5000-9000-1e54ca65d0cf",
    )
    plan = SimpleNamespace(id=1, name="Standard", price=195.0)
    config = SimpleNamespace(yookassa_shop_id="test-shop", yookassa_secret_key="test-secret")
    attempt_time = datetime(2026, 9, 10, 12, 0, 0)

    bad_req = _make_bad_request({
        "type": "error",
        "code": "invalid_request",
        "parameter": "payment_method_id",
    })

    with patch("subscription_renewal.Payment.create", side_effect=bad_req):
        res = await execute_yookassa_recurring_attempt(sub, plan, 195.0, config, attempt_time)

    assert res.outcome == "deactivate"
    assert res.is_permanent_deactivate is True
    assert res.payment_id is None


@pytest.mark.asyncio
async def test_execute_yookassa_recurring_attempt_unrelated_bad_request_is_integration_error():
    sub = SimpleNamespace(
        id=10,
        user_id=12345678,
        payment_attempt_count=0,
        payment_method_id="31d55000-000f-5000-9000-1e54ca65d0cf",
    )
    plan = SimpleNamespace(id=1, name="Standard", price=195.0)
    config = SimpleNamespace(yookassa_shop_id="test-shop", yookassa_secret_key="test-secret")
    attempt_time = datetime(2026, 9, 10, 12, 0, 0)

    bad_req = _make_bad_request({
        "type": "error",
        "code": "invalid_request",
        "parameter": "amount",
    })

    with patch("subscription_renewal.Payment.create", side_effect=bad_req):
        res = await execute_yookassa_recurring_attempt(sub, plan, 195.0, config, attempt_time)

    assert res.outcome == "integration_error"
    assert res.is_permanent_deactivate is False


@pytest.mark.asyncio
async def test_execute_yookassa_recurring_attempt_unauthorized_is_auth_error():
    sub = SimpleNamespace(
        id=10,
        user_id=12345678,
        payment_attempt_count=0,
        payment_method_id="31d55000-000f-5000-9000-1e54ca65d0cf",
    )
    plan = SimpleNamespace(id=1, name="Standard", price=195.0)
    config = SimpleNamespace(yookassa_shop_id="test-shop", yookassa_secret_key="test-secret")
    attempt_time = datetime(2026, 9, 10, 12, 0, 0)

    with patch("subscription_renewal.Payment.create", side_effect=UnauthorizedError({"code": "unauthorized"})):
        res = await execute_yookassa_recurring_attempt(sub, plan, 195.0, config, attempt_time)

    assert res.outcome == "auth_error"
    assert res.is_permanent_deactivate is False


@pytest.mark.asyncio
async def test_execute_yookassa_recurring_attempt_network_error_is_unknown_outcome():
    sub = SimpleNamespace(
        id=10,
        user_id=12345678,
        payment_attempt_count=0,
        payment_method_id="31d55000-000f-5000-9000-1e54ca65d0cf",
    )
    plan = SimpleNamespace(id=1, name="Standard", price=195.0)
    config = SimpleNamespace(yookassa_shop_id="test-shop", yookassa_secret_key="test-secret")
    attempt_time = datetime(2026, 9, 10, 12, 0, 0)

    with patch("subscription_renewal.Payment.create", side_effect=ConnectionResetError("Connection reset")):
        res = await execute_yookassa_recurring_attempt(sub, plan, 195.0, config, attempt_time)

    assert res.outcome == "unknown"
    assert res.is_permanent_deactivate is False


def test_calculate_renewal_details_trial_upgrade_and_promos():
    regular_plan = SimpleNamespace(
        id=2,
        name="Месячный",
        price=1000.0,
        duration_value=1,
        duration_unit="months",
        is_trial=False,
    )
    trial_plan = SimpleNamespace(
        id=1,
        name="Пробный",
        price=1.0,
        duration_value=3,
        duration_unit="days",
        is_trial=True,
        upgrades_to_plan=regular_plan,
    )
    promo = SimpleNamespace(
        applies_to_all_plans=False,
        applicable_plans=[SimpleNamespace(id=2)],
        discount_percent=20.0,
    )
    user = SimpleNamespace(promo_codes=[promo])
    sub = SimpleNamespace(
        plan=trial_plan,
        discount_percent=0.0,
        payment_attempt_count=1,
    )

    details = calculate_renewal_details(user, sub)
    assert details is not None
    assert details.plan_name == "Месячный"
    assert details.duration_text == "1 мес."
    assert details.current_discount == 20.0
    assert details.final_price == 800.0
    assert details.attempt_count == 1
