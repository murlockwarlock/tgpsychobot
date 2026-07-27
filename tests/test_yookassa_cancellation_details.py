from types import SimpleNamespace

from payment_failure_reasons import (
    format_yookassa_admin_reason_line,
    format_yookassa_cancellation_reason,
    get_yookassa_cancellation_reason,
)


def test_cancellation_reason_is_extracted_and_translated():
    payment = SimpleNamespace(
        cancellation_details=SimpleNamespace(
            party="payment_network",
            reason="insufficient_funds",
        )
    )

    reason = get_yookassa_cancellation_reason(payment)

    assert reason == "insufficient_funds"
    assert format_yookassa_cancellation_reason(reason) == "недостаточно средств"
    assert format_yookassa_admin_reason_line(reason) == (
        "\nПричина: недостаточно средств (insufficient_funds)"
    )


def test_unknown_cancellation_reason_keeps_provider_code():
    assert format_yookassa_admin_reason_line("new_provider_reason") == (
        "\nПричина: код ЮKassa: new_provider_reason"
    )


def test_missing_cancellation_reason_is_explicit():
    assert format_yookassa_admin_reason_line(None) == "\nПричина: ЮKassa не указала"
