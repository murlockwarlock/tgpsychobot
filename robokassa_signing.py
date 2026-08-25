"""Robokassa request signing and safe payment-build diagnostics.

The provider signs the exact, unencoded field values.  URL encoding is only
performed after those values and the signature input have been established.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Mapping
from urllib.parse import urlencode


ROBOKASSA_PAYMENT_URL = "https://auth.robokassa.ru/Merchant/Index.aspx"
ROBOKASSA_RECURRING_URL = "https://auth.robokassa.ru/Merchant/Recurring"
ROBOKASSA_DEFAULT_HASH_ALGORITHM = "md5"
ROBOKASSA_DEFAULT_IS_TEST = 0
ROBOKASSA_MSK = timezone(timedelta(hours=3))


def format_robokassa_amount(value: Decimal | float | int | str) -> str:
    """Return the exact two-decimal OutSum string used by this integration."""
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value))
        if not amount.is_finite():
            raise InvalidOperation
        return format(amount.quantize(Decimal("0.01")), "f")
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("Robokassa OutSum must be a finite numeric value") from exc


def format_robokassa_expiration(value: datetime) -> str:
    """Format an expiration instant for Robokassa in Moscow wall-clock time.

    Database timestamps in this project are naive UTC values.  Treating a
    naive value as the host's local timezone can shift the deadline after a
    deploy to a host with a different timezone and create an already-expired
    provider invoice.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(ROBOKASSA_MSK).strftime("%Y-%m-%dT%H:%M")


def normalize_robokassa_hash_algorithm(algorithm: str | None) -> str:
    normalized = (algorithm or ROBOKASSA_DEFAULT_HASH_ALGORITHM).strip().lower().replace("-", "")
    aliases = {
        "md5": "md5",
        "sha1": "sha1",
        "sha256": "sha256",
        "sha512": "sha512",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported Robokassa hash algorithm: {algorithm}") from exc


def _sorted_shp_params(shp_params: Mapping[str, object] | None) -> list[tuple[str, str]]:
    if not shp_params:
        return []
    normalized: list[tuple[str, str]] = []
    for name, value in shp_params.items():
        name = str(name)
        if not name.startswith("Shp_"):
            raise ValueError(f"Robokassa custom parameter must start with Shp_: {name}")
        normalized.append((name, str(value)))
    return sorted(normalized, key=lambda item: item[0])


def build_robokassa_signature_input(
    merchant_login: str,
    out_sum: str,
    invoice_id: int | str,
    merchant_password_1: str,
    *,
    shp_params: Mapping[str, object] | None = None,
) -> str:
    """Build the unencoded initial-payment SignatureValue input string."""
    fields = [str(merchant_login), str(out_sum), str(invoice_id), str(merchant_password_1)]
    fields.extend(f"{name}={value}" for name, value in _sorted_shp_params(shp_params))
    return ":".join(fields)


def calculate_robokassa_signature(
    merchant_login: str,
    out_sum: str,
    invoice_id: int | str,
    merchant_password_1: str,
    *,
    shp_params: Mapping[str, object] | None = None,
    hash_algorithm: str = ROBOKASSA_DEFAULT_HASH_ALGORITHM,
) -> str:
    signature_input = build_robokassa_signature_input(
        merchant_login,
        out_sum,
        invoice_id,
        merchant_password_1,
        shp_params=shp_params,
    )
    digest = hashlib.new(normalize_robokassa_hash_algorithm(hash_algorithm))
    digest.update(signature_input.encode("utf-8"))
    return digest.hexdigest()


def build_robokassa_payment_params(
    merchant_login: str,
    merchant_password_1: str,
    cost: Decimal | float | int | str,
    invoice_id: int | str,
    description: str,
    *,
    expiration_date: datetime | None = None,
    recurring: bool = False,
    is_test: int | bool = ROBOKASSA_DEFAULT_IS_TEST,
    hash_algorithm: str = ROBOKASSA_DEFAULT_HASH_ALGORITHM,
    shp_params: Mapping[str, object] | None = None,
) -> dict[str, str | int]:
    """Build initial-payment parameters with one canonical value per field."""
    out_sum = format_robokassa_amount(cost)
    invoice_text = str(invoice_id)
    params: dict[str, str | int] = {
        "MerchantLogin": str(merchant_login),
        "OutSum": out_sum,
        "InvId": invoice_text,
        "Description": str(description),
        "SignatureValue": calculate_robokassa_signature(
            str(merchant_login),
            out_sum,
            invoice_text,
            merchant_password_1,
            shp_params=shp_params,
            hash_algorithm=hash_algorithm,
        ),
        "IsTest": 1 if is_test else 0,
    }
    for name, value in _sorted_shp_params(shp_params):
        params[name] = value
    if recurring:
        params["Recurring"] = "true"
    if expiration_date:
        params["ExpirationDate"] = format_robokassa_expiration(expiration_date)
    return params


def generate_robokassa_payment_url(
    merchant_login: str,
    merchant_password_1: str,
    cost: Decimal | float | int | str,
    invoice_id: int | str,
    description: str,
    *,
    expiration_date: datetime | None = None,
    recurring: bool = False,
    is_test: int | bool = ROBOKASSA_DEFAULT_IS_TEST,
    hash_algorithm: str = ROBOKASSA_DEFAULT_HASH_ALGORITHM,
    shp_params: Mapping[str, object] | None = None,
    payment_url: str = ROBOKASSA_PAYMENT_URL,
) -> str:
    params = build_robokassa_payment_params(
        merchant_login,
        merchant_password_1,
        cost,
        invoice_id,
        description,
        expiration_date=expiration_date,
        recurring=recurring,
        is_test=is_test,
        hash_algorithm=hash_algorithm,
        shp_params=shp_params,
    )
    return f"{payment_url}?{urlencode(params)}"


def build_robokassa_recurring_params(
    merchant_login: str,
    merchant_password_1: str,
    cost: Decimal | float | int | str,
    parent_invoice_id: int | str,
    invoice_id: int | str,
    description: str,
    *,
    hash_algorithm: str = ROBOKASSA_DEFAULT_HASH_ALGORITHM,
    shp_params: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Build a child recurring request.

    Robokassa requires ``InvoiceID`` and ``PreviousInvoiceID`` for this
    endpoint, and explicitly excludes ``PreviousInvoiceID`` from the
    SignatureValue input.
    """
    out_sum = format_robokassa_amount(cost)
    invoice_text = str(invoice_id)
    params = {
        "MerchantLogin": str(merchant_login),
        "OutSum": out_sum,
        "InvoiceID": invoice_text,
        "PreviousInvoiceID": str(parent_invoice_id),
        "Description": str(description),
        "SignatureValue": calculate_robokassa_signature(
            str(merchant_login),
            out_sum,
            invoice_text,
            merchant_password_1,
            shp_params=shp_params,
            hash_algorithm=hash_algorithm,
        ),
    }
    for name, value in _sorted_shp_params(shp_params):
        params[name] = value
    return params


def robokassa_payment_diagnostics(
    *,
    merchant_login: str | None,
    merchant_password_1: str | None,
    merchant_password_2: str | None,
    cost: Decimal | float | int | str,
    invoice_id: int | str,
    description: str = "",
    expiration_date: datetime | None = None,
    recurring: bool = False,
    is_test: int | bool = ROBOKASSA_DEFAULT_IS_TEST,
    hash_algorithm: str = ROBOKASSA_DEFAULT_HASH_ALGORITHM,
    shp_params: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return safe, non-secret build facts for logs/tests/admin diagnostics."""
    merchant_login_text = str(merchant_login or "")
    out_sum = format_robokassa_amount(cost)
    shp_items = _sorted_shp_params(shp_params)
    algorithm = normalize_robokassa_hash_algorithm(hash_algorithm)
    params = build_robokassa_payment_params(
        merchant_login_text,
        merchant_password_1 or "",
        out_sum,
        invoice_id,
        description,
        expiration_date=expiration_date,
        recurring=recurring,
        is_test=is_test,
        shp_params=shp_params,
        hash_algorithm=algorithm,
    )
    canonical_fields = ["MerchantLogin", "OutSum", "InvId", "MerchantPass1"] + [
        name for name, _ in shp_items
    ]
    canonical_values_match_url = (
        params["MerchantLogin"] == merchant_login_text
        and params["OutSum"] == out_sum
        and params["InvId"] == str(invoice_id)
        and all(params[name] == value for name, value in shp_items)
    )
    safe_url_parameters = {
        key: value for key, value in params.items() if key != "SignatureValue"
    }
    return {
        "runtime_config_source": "database:SubscriptionConfig",
        "MerchantLogin": merchant_login_text,
        "IsTest": 1 if is_test else 0,
        "hash_algorithm": algorithm,
        "pass1_configured": bool(merchant_password_1),
        "pass2_configured": bool(merchant_password_2),
        "OutSum": out_sum,
        "InvId": str(invoice_id),
        "Shp_names": [name for name, _ in shp_items],
        "Shp_values": {name: value for name, value in shp_items},
        "canonical_field_order": canonical_fields,
        "canonical_values_match_url": canonical_values_match_url,
        "signature_present": bool(params.get("SignatureValue")),
        "signature_length": len(str(params.get("SignatureValue") or "")),
        "url_parameters": safe_url_parameters,
    }
