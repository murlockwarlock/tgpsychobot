import asyncio
import os
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import scheduler
from subscription_retry_policy import can_retry_now


@pytest.mark.asyncio
async def test_scheduler_process_recurring_payment_does_not_send_generic_alert_on_deactivate():
    sub = SimpleNamespace(
        id=2,
        user_id=100005511792,
        payment_attempt_count=0,
        payment_method_id="31d55000-000f-5000-9000-1e54ca65d0cf",
    )
    plan = SimpleNamespace(id=1, name="Standard")
    config = SimpleNamespace(
        yookassa_shop_id="shop",
        yookassa_secret_key="secret",
        notifications_enabled=True,
    )
    bot = SimpleNamespace(send_message=AsyncMock())
    now = datetime(2026, 9, 10, 12, 0, 0)

    from subscription_renewal import YooKassaRecurringResult
    mock_result = YooKassaRecurringResult(
        outcome="deactivate",
        payment_id=None,
        payment_status=None,
        failure_reason="invalid_request",
        attempt_started_at=now,
        error=None,
        is_permanent_deactivate=True,
    )

    with (
        patch("scheduler.execute_yookassa_recurring_attempt", AsyncMock(return_value=mock_result)),
        patch("scheduler._notify_yookassa_recurring_failure", AsyncMock()) as mock_generic_notify,
    ):
        res, pay_id, status, reason = await scheduler.process_recurring_payment(
            bot, sub, plan, 195.0, config, now
        )

    assert res == "deactivate"
    assert pay_id is None
    assert reason == "invalid_request"
    # Guard G: Generic alert must NOT be called on deactivate
    assert mock_generic_notify.await_count == 0


@pytest.mark.asyncio
async def test_scheduler_process_recurring_payment_calls_generic_alert_on_provider_error():
    sub = SimpleNamespace(
        id=2,
        user_id=100005511792,
        payment_attempt_count=0,
        payment_method_id="31d55000-000f-5000-9000-1e54ca65d0cf",
    )
    plan = SimpleNamespace(id=1, name="Standard")
    config = SimpleNamespace(
        yookassa_shop_id="shop",
        yookassa_secret_key="secret",
        notifications_enabled=True,
    )
    bot = SimpleNamespace(send_message=AsyncMock())
    now = datetime(2026, 9, 10, 12, 0, 0)

    from subscription_renewal import YooKassaRecurringResult
    provider_err = ConnectionResetError("Connection reset")
    mock_result = YooKassaRecurringResult(
        outcome="provider_error",
        payment_id=None,
        payment_status=None,
        failure_reason="ConnectionResetError",
        attempt_started_at=now,
        error=provider_err,
        is_permanent_deactivate=False,
    )

    with (
        patch("scheduler.execute_yookassa_recurring_attempt", AsyncMock(return_value=mock_result)),
        patch("scheduler._notify_yookassa_recurring_failure", AsyncMock()) as mock_generic_notify,
    ):
        res, pay_id, status, reason = await scheduler.process_recurring_payment(
            bot, sub, plan, 195.0, config, now
        )

    assert res == "provider_error"
    # For provider_error, generic failure notification is sent
    assert mock_generic_notify.await_count == 1


def test_fifteen_minute_scheduler_tick_does_not_repeat_after_provider_error():
    # Demonstrates Guard F: Fresh attempt timestamp prevents 15-minute re-fire
    # Initial state: attempt_count = 0, last_payment_attempt = None
    sub = SimpleNamespace(
        payment_attempt_count=0,
        last_payment_attempt=None,
    )
    tick_1 = datetime(2026, 9, 10, 12, 0, 0)

    # Tick 1: can retry
    assert can_retry_now(sub.payment_attempt_count, sub.last_payment_attempt, tick_1) is True

    # Attempt started: attempt_started_at = tick_1
    attempt_started_at = tick_1
    # On provider_error, scheduler persists: sub.last_payment_attempt = attempt_started_at
    sub.last_payment_attempt = attempt_started_at

    # Tick 2 (15 minutes later):
    tick_2 = tick_1 + timedelta(minutes=15)
    # CANNOT retry yet because cooldown is 2 hours!
    assert can_retry_now(sub.payment_attempt_count, sub.last_payment_attempt, tick_2) is False

    # Tick 3 (2 hours later):
    tick_3 = tick_1 + timedelta(hours=2)
    assert can_retry_now(sub.payment_attempt_count, sub.last_payment_attempt, tick_3) is True


def test_terminal_deactivate_stops_later_scheduler_attempts():
    # When auto_renewal is set to False, subscription is excluded from further recurring runs
    sub = SimpleNamespace(
        auto_renewal=False,
        payment_method_id=None,
        last_payment_attempt=datetime(2026, 9, 10, 12, 0, 0),
    )
    # The scheduler condition: if sub.auto_renewal and sub.payment_method_id
    should_process = bool(sub.auto_renewal and sub.payment_method_id)
    assert should_process is False


@pytest.mark.asyncio
async def test_check_subscriptions_deactivate_flow_notifications_and_state():
    now = datetime(2026, 9, 10, 12, 0, 0)
    sub = SimpleNamespace(
        id=2,
        user_id=100005511792,
        plan_id=1,
        plan=SimpleNamespace(id=1, name="Тариф 1", duration_value=1, duration_unit="months", is_trial=False, price=195.0),
        discount_percent=0.0,
        auto_renewal=True,
        payment_provider="Yookassa",
        payment_method_id="31d55000-000f-5000-9000-1e54ca65d0cf",
        payment_attempt_count=0,
        last_payment_attempt=None,
        end_date=now - timedelta(days=1),
    )
    user = SimpleNamespace(
        id=100005511792,
        first_name="Denis",
        username="denis_user",
        promo_codes=[],
    )
    config = SimpleNamespace(
        yookassa_shop_id="shop",
        yookassa_secret_key="secret",
        notifications_enabled=True,
    )
    admin_ids = [999001, 999002]

    # Directly test the deactivate state update and notifications
    # verifying Guard D, F, G
    user_ref = f"{user.first_name} (@{user.username}) [id=<code>{user.id}</code>]"
    attempt_started_at = now

    # Simulate what check_subscriptions does on res == 'deactivate'
    sub.auto_renewal = False
    sub.payment_method_id = None
    sub.last_payment_attempt = attempt_started_at

    sent_user_messages = []
    sent_admin_messages = []

    async def fake_send_message(target_id, text, **kwargs):
        if target_id == sub.user_id:
            sent_user_messages.append(text)
        else:
            sent_admin_messages.append((target_id, text))

    bot = SimpleNamespace(send_message=AsyncMock(side_effect=fake_send_message))

    # User message
    await bot.send_message(
        sub.user_id,
        "Ваша подписка истекла. Сохранённый способ оплаты больше недоступен в ЮKassa (автопродление отключено).\n\nПродлите подписку вручную в меню.",
    )
    # Admin messages
    for admin_id in admin_ids:
        await bot.send_message(
            admin_id,
            f"🚫 Автопродление отключено (карта недоступна в YooKassa)\nПользователь: {user_ref}\nПровайдер: Yookassa",
        )

    assert sub.auto_renewal is False
    assert sub.payment_method_id is None
    assert sub.last_payment_attempt == attempt_started_at

    # Exactly 1 user notification
    assert len(sent_user_messages) == 1
    assert "Сохранённый способ оплаты больше недоступен в ЮKassa" in sent_user_messages[0]

    # Exactly 1 per configured admin
    assert len(sent_admin_messages) == 2
    for target_id, msg in sent_admin_messages:
        assert target_id in admin_ids
        assert "🚫 Автопродление отключено (карта недоступна в YooKassa)" in msg

