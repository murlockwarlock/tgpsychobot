import asyncio
import os
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("BOT_TOKEN", "test_bot_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
from max_messenger_bot.services import subscriptions as max_subs
from max_messenger_bot.app import MaxBotApplication
from subscription_renewal import YooKassaRecurringResult


@pytest.mark.asyncio
async def test_max_app_callback_ack_before_work_for_cancel_retry():
    dummy_client = SimpleNamespace(answer_callback=AsyncMock())
    app = MaxBotApplication(client=dummy_client)
    sender = SimpleNamespace(user_id=100005511792, name="Denis", username="denis", full_name="Denis", public_name="Denis")
    callback = SimpleNamespace(
        callback_id="cb-cancel-1",
        payload="sub_cancel_retry",
        chat_id=100005511792,
        user_id=100005511792,
        sender=sender,
    )
    events = []

    async def mock_answer_callback(cb_id):
        events.append("ack")

    async def mock_cancel_retry(client, chat_id, user_id):
        events.append("work")

    app.client.answer_callback = AsyncMock(side_effect=mock_answer_callback)

    with (
        patch("max_messenger_bot.app.common.ensure_user", AsyncMock()),
        patch.object(max_subs, "cancel_retry", side_effect=mock_cancel_retry),
    ):
        await app.handle_callback(callback)

    assert app.client.answer_callback.await_count == 1
    assert events == ["ack", "work"]


@pytest.mark.asyncio
async def test_max_app_callback_ack_before_work_for_retry_now():
    dummy_client = SimpleNamespace(answer_callback=AsyncMock())
    app = MaxBotApplication(client=dummy_client)
    sender = SimpleNamespace(user_id=100005511792, name="Denis", username="denis", full_name="Denis", public_name="Denis")
    callback = SimpleNamespace(
        callback_id="cb-retry-1",
        payload="sub_retry_now",
        chat_id=100005511792,
        user_id=100005511792,
        sender=sender,
    )
    events = []

    async def mock_answer_callback(cb_id):
        events.append("ack")

    async def mock_handle_retry(client, chat_id, user_id):
        events.append("work")

    app.client.answer_callback = AsyncMock(side_effect=mock_answer_callback)

    with (
        patch("max_messenger_bot.app.common.ensure_user", AsyncMock()),
        patch.object(max_subs, "handle_max_manual_retry", side_effect=mock_handle_retry),
    ):
        await app.handle_callback(callback)

    assert app.client.answer_callback.await_count == 1
    assert events == ["ack", "work"]


@pytest.mark.asyncio
async def test_max_cancel_retry_preserves_retry_fields_and_shows_plans():
    # Contract: auto_renewal=False; do NOT alter payment_attempt_count; do NOT alter last_payment_attempt; do NOT clear payment_method_id
    last_attempt = datetime(2026, 9, 10, 10, 0, 0)
    sub = SimpleNamespace(
        id=1,
        user_id=100005511792,
        auto_renewal=True,
        payment_attempt_count=2,
        last_payment_attempt=last_attempt,
        payment_method_id="31d55000-000f-5000-9000-1e54ca65d0cf",
    )

    class MockSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def scalar(self, stmt):
            return sub

        async def commit(self):
            pass

    client = SimpleNamespace(send_message=AsyncMock())

    with (
        patch("max_messenger_bot.services.subscriptions.async_session_maker", return_value=MockSession()),
        patch.object(max_subs, "show_plans", AsyncMock()) as mock_show_plans,
    ):
        await max_subs.cancel_retry(client, chat_id=100005511792, user_id=100005511792)

    # Verification of Guard H:
    assert sub.auto_renewal is False
    assert sub.payment_attempt_count == 2  # preserved!
    assert sub.last_payment_attempt == last_attempt  # preserved!
    assert sub.payment_method_id == "31d55000-000f-5000-9000-1e54ca65d0cf"  # preserved!

    # Verification of user confirmation + show_plans
    assert client.send_message.await_count == 1
    assert "Автопродление отключено" in client.send_message.await_args.kwargs["text"]
    assert mock_show_plans.await_count == 1


@pytest.mark.asyncio
async def test_max_manual_retry_respects_cooldown():
    now = datetime(2026, 9, 10, 12, 0, 0)
    sub = SimpleNamespace(
        id=1,
        user_id=100005511792,
        auto_renewal=True,
        payment_attempt_count=1,
        last_payment_attempt=now - timedelta(minutes=30),  # Attempted 30m ago, 2h cooldown
        payment_method_id="31d55000-000f-5000-9000-1e54ca65d0cf",
    )
    user = SimpleNamespace(
        id=100005511792,
        subscription=sub,
        promo_codes=[],
    )

    class MockSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, model, ident, *args, **kwargs):
            if model.__name__ == "User":
                return user
            return SimpleNamespace()

    client = SimpleNamespace(send_message=AsyncMock())

    with (
        patch("max_messenger_bot.services.subscriptions.async_session_maker", return_value=MockSession()),
        patch.object(max_subs, "utc_now", return_value=now),
        patch.object(max_subs, "execute_yookassa_recurring_attempt", AsyncMock()) as mock_attempt,
    ):
        await max_subs.handle_max_manual_retry(client, chat_id=100005511792, user_id=100005511792)

    # Provider should NOT be called due to cooldown
    assert mock_attempt.await_count == 0
    assert client.send_message.await_count == 1
    assert "Повторное списание пока недоступно" in client.send_message.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_max_manual_retry_permanent_deactivate_handles_state_and_shows_plans():
    now = datetime(2026, 9, 10, 12, 0, 0)
    plan = SimpleNamespace(
        id=1,
        name="Тариф 1",
        price=195.0,
        duration_value=1,
        duration_unit="months",
        is_trial=False,
    )
    sub = SimpleNamespace(
        id=1,
        user_id=100005511792,
        auto_renewal=True,
        payment_attempt_count=0,
        last_payment_attempt=None,
        payment_method_id="31d55000-000f-5000-9000-1e54ca65d0cf",
        plan=plan,
        discount_percent=0.0,
    )
    user = SimpleNamespace(
        id=100005511792,
        subscription=sub,
        promo_codes=[],
        first_name="Denis",
    )

    class MockSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, model, ident, *args, **kwargs):
            if model.__name__ == "User":
                return user
            return SimpleNamespace(yookassa_shop_id="shop", yookassa_secret_key="secret")

        async def merge(self, obj):
            pass

        async def commit(self):
            pass

    client = SimpleNamespace(send_message=AsyncMock())
    deactivate_res = YooKassaRecurringResult(
        outcome="deactivate",
        payment_id=None,
        payment_status=None,
        failure_reason="invalid_request",
        attempt_started_at=now,
        error=None,
        is_permanent_deactivate=True,
    )

    with (
        patch("max_messenger_bot.services.subscriptions.async_session_maker", return_value=MockSession()),
        patch("max_messenger_bot.services.common.notify_telegram_admins", AsyncMock()),
        patch.object(max_subs, "utc_now", return_value=now),
        patch.object(max_subs, "execute_yookassa_recurring_attempt", AsyncMock(return_value=deactivate_res)),
        patch.object(max_subs, "show_plans", AsyncMock()) as mock_show_plans,
    ):
        await max_subs.handle_max_manual_retry(client, chat_id=100005511792, user_id=100005511792)

    # Verification:
    assert sub.auto_renewal is False
    assert sub.payment_method_id is None
    assert sub.last_payment_attempt == now

    # Informs user that saved method is invalid and redirects to plans
    messages = [call.kwargs["text"] for call in client.send_message.await_args_list]
    assert any("Сохранённый способ оплаты больше недоступен в ЮKassa" in msg for msg in messages)
    assert mock_show_plans.await_count == 1
