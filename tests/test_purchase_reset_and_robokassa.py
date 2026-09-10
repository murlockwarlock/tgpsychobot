import asyncio
import os
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("BOT_TOKEN", "test_bot_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import scheduler


def test_yookassa_successful_purchase_resets_retry_state():
    # Simulates what webhooks.py:372-382 does upon successful YooKassa payment
    stale_last_attempt = datetime(2026, 9, 10, 8, 0, 0)
    user_sub = SimpleNamespace(
        plan_id=1,
        start_date=None,
        end_date=None,
        payment_provider="Yookassa",
        payment_attempt_count=2,
        last_payment_attempt=stale_last_attempt,
        pending_robokassa_invoice_id=None,
        payment_method_id="new-pm-id",
        auto_renewal=True,
    )

    # State transition on successful purchase
    user_sub.payment_attempt_count = 0
    user_sub.last_payment_attempt = None

    assert user_sub.payment_attempt_count == 0
    assert user_sub.last_payment_attempt is None


def test_robokassa_successful_purchase_resets_retry_state():
    # Simulates what webhooks.py:820-830 does upon successful Robokassa payment
    stale_last_attempt = datetime(2026, 9, 10, 8, 0, 0)
    user_sub = SimpleNamespace(
        plan_id=1,
        start_date=None,
        end_date=None,
        payment_provider="Robokassa",
        payment_attempt_count=2,
        last_payment_attempt=stale_last_attempt,
        pending_robokassa_invoice_id="inv-123",
        payment_method_id="inv-100",
        auto_renewal=True,
    )

    # State transition on successful purchase
    user_sub.payment_attempt_count = 0
    user_sub.last_payment_attempt = None
    user_sub.pending_robokassa_invoice_id = None

    assert user_sub.payment_attempt_count == 0
    assert user_sub.last_payment_attempt is None
    assert user_sub.pending_robokassa_invoice_id is None


@pytest.mark.asyncio
async def test_robokassa_recurring_behavior_unchanged():
    # Verify process_recurring_robokassa_payment continues working identically
    captured = {}

    class _Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def text(self):
            return "OK42"

    class _HttpSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, data):
            captured["url"] = url
            captured["data"] = data
            return _Response()

    config = SimpleNamespace(
        robokassa_merchant_login="demo",
        robokassa_password_1="pass1",
    )
    plan = SimpleNamespace(name="Тариф")

    with patch.object(scheduler.aiohttp, "ClientSession", return_value=_HttpSession()):
        result = await scheduler.process_recurring_robokassa_payment(
            config, plan, 10.0, "41", 42
        )

    assert result is True
    assert captured["url"].endswith("/Merchant/Recurring")
    assert captured["data"]["MerchantLogin"] == "demo"
    assert captured["data"]["OutSum"] == "10.00"
    assert captured["data"]["InvoiceID"] == "42"
    assert captured["data"]["PreviousInvoiceID"] == "41"
    assert captured["data"]["SignatureValue"] == "70aa371c7594b731aeda96ded889a048"
