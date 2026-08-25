from datetime import datetime
from decimal import Decimal
from urllib.parse import parse_qs, urlparse

import handlers
try:
    import webhooks
except ModuleNotFoundError:
    webhooks = None
from robokassa_signing import (
    build_robokassa_recurring_params,
    build_robokassa_signature_input,
    calculate_robokassa_signature,
    format_robokassa_expiration,
    generate_robokassa_payment_url,
    robokassa_payment_diagnostics,
)


def test_initial_signature_vector_and_url_use_the_same_canonical_values():
    shp = {"Shp_user": 42, "Shp_order": 25}
    canonical = build_robokassa_signature_input(
        "demo",
        "10.00",
        42,
        "pass1",
        shp_params=shp,
    )
    assert canonical == "demo:10.00:42:pass1:Shp_order=25:Shp_user=42"
    assert calculate_robokassa_signature(
        "demo", "10.00", 42, "pass1", shp_params=shp
    ) == "04e752cd6cf8f7f963c52beecfdf05a2"

    url = generate_robokassa_payment_url(
        "demo",
        "pass1",
        Decimal("10.00"),
        42,
        "Тестовый заказ",
        shp_params=shp,
    )
    params = parse_qs(urlparse(url).query)
    assert params["MerchantLogin"] == ["demo"]
    assert params["OutSum"] == ["10.00"]
    assert params["InvId"] == ["42"]
    assert params["Shp_order"] == ["25"]
    assert params["Shp_user"] == ["42"]
    assert params["SignatureValue"] == ["04e752cd6cf8f7f963c52beecfdf05a2"]


def test_initial_diagnostics_are_safe_and_include_request_specific_fields():
    diagnostics = robokassa_payment_diagnostics(
        merchant_login="demo",
        merchant_password_1="pass1",
        merchant_password_2="pass2",
        cost=Decimal("10.00"),
        invoice_id=42,
        shp_params={"Shp_user": 42, "Shp_order": 25},
    )

    assert diagnostics["runtime_config_source"] == "database:SubscriptionConfig"
    assert diagnostics["MerchantLogin"] == "demo"
    assert diagnostics["IsTest"] == 0
    assert diagnostics["hash_algorithm"] == "md5"
    assert diagnostics["pass1_configured"] is True
    assert diagnostics["pass2_configured"] is True
    assert diagnostics["OutSum"] == "10.00"
    assert diagnostics["InvId"] == "42"
    assert diagnostics["Shp_names"] == ["Shp_order", "Shp_user"]
    assert diagnostics["canonical_values_match_url"] is True
    assert diagnostics["url_parameters"]["OutSum"] == "10.00"
    assert diagnostics["url_parameters"]["Shp_order"] == "25"
    assert "SignatureValue" not in diagnostics["url_parameters"]
    diagnostics_text = str(diagnostics)
    assert "SignatureValue" not in diagnostics_text
    assert "'pass1'" not in diagnostics_text
    assert "'pass2'" not in diagnostics_text


def test_actual_initial_payment_wrappers_preserve_signature_and_url_values():
    expected_signature = "70aa371c7594b731aeda96ded889a048"
    expiration = datetime(2026, 8, 26, 12, 0)
    urls = [
        handlers.generate_payment_link(
            "demo",
            "pass1",
            Decimal("10.00"),
            42,
            "Тестовый заказ",
            expiration_date=expiration,
        ),
    ]
    if webhooks is not None:
        urls.append(
            webhooks.generate_robokassa_payment_url(
                "demo",
                "pass1",
                Decimal("10.00"),
                42,
                "Тестовый заказ",
                expiration_date=expiration,
            )
        )

    for url in urls:
        params = parse_qs(urlparse(url).query)
        assert params["MerchantLogin"] == ["demo"]
        assert params["OutSum"] == ["10.00"]
        assert params["InvId"] == ["42"]
        assert params["IsTest"] == ["0"]
        assert params["ExpirationDate"] == ["2026-08-26T15:00"]
        assert params["SignatureValue"] == [expected_signature]


def test_naive_database_utc_expiration_is_converted_to_moscow_without_host_tz_shift():
    assert format_robokassa_expiration(datetime(2026, 8, 26, 12, 0)) == "2026-08-26T15:00"


def test_recurring_request_uses_invoice_id_and_does_not_sign_parent_invoice_id():
    params = build_robokassa_recurring_params(
        "demo",
        "pass1",
        Decimal("10.00"),
        parent_invoice_id=41,
        invoice_id=42,
        description="Renewal",
    )

    assert params["MerchantLogin"] == "demo"
    assert params["OutSum"] == "10.00"
    assert params["InvoiceID"] == "42"
    assert params["PreviousInvoiceID"] == "41"
    assert params["SignatureValue"] == "70aa371c7594b731aeda96ded889a048"
    changed_parent = build_robokassa_recurring_params(
        "demo", "pass1", Decimal("10.00"), 99, 42, "Renewal"
    )
    assert changed_parent["SignatureValue"] == params["SignatureValue"]
