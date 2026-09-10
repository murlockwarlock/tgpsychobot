import asyncio
import os
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("BOT_TOKEN", "test_bot_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import handlers
from subscription_renewal import YooKassaRecurringResult


@pytest.mark.asyncio
async def test_telegram_handle_sub_retry_now_permanent_deactivate_flow(caplog):
    now = datetime(2026, 9, 10, 12, 0, 0)
    user_id = 42

    plan = SimpleNamespace(
        id=1,
        name="Тариф 1",
        price=195.0,
        duration_value=1,
        duration_unit="months",
        is_trial=False,
        upgrades_to_plan=None,
    )
    user_sub = SimpleNamespace(
        id=10,
        user_id=user_id,
        auto_renewal=True,
        payment_attempt_count=0,
        last_payment_attempt=None,
        payment_provider="Yookassa",
        payment_method_id="31d55000-000f-5000-9000-1e54ca65d0cf",
        plan=plan,
        discount_percent=0.0,
        pending_robokassa_invoice_id=None,
    )
    user = SimpleNamespace(
        id=user_id,
        first_name="User",
        username="user42",
        promo_codes=[],
    )
    config = SimpleNamespace(
        yookassa_shop_id="shop",
        yookassa_secret_key="secret",
        notifications_enabled=True,
    )

    class MockSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, model, ident, *args, **kwargs):
            if model.__name__ == "SubscriptionConfig":
                return config
            if model.__name__ == "User":
                return user
            return None

        async def scalar(self, stmt):
            return user_sub

        async def commit(self):
            pass

    callback = SimpleNamespace(
        answer=AsyncMock(),
        from_user=SimpleNamespace(id=user_id, first_name="User", username="user42"),
        message=SimpleNamespace(
            chat=SimpleNamespace(id=user_id),
            delete=AsyncMock(),
        ),
    )
    bot = SimpleNamespace(send_message=AsyncMock())
    state = AsyncMock()

    deactivate_res = ('deactivate', None, None, 'invalid_request')

    with (
        patch.object(handlers, "async_session_maker", return_value=MockSession()),
        patch.object(handlers, "get_all_admin_ids", AsyncMock(return_value=[999001])),
        patch.object(handlers, "process_recurring_payment", AsyncMock(return_value=deactivate_res)),
        patch.object(handlers, "_send_subscription_info", AsyncMock()),
        patch.object(handlers.plog, "info") as mock_plog_info,
    ):
        await handlers.handle_sub_retry_now(callback, state, bot)

    # Verification of Guard F & Guard D:
    assert user_sub.auto_renewal is False
    assert user_sub.payment_method_id is None
    assert user_sub.last_payment_attempt is not None

    # Verification of Guard E: payment_method_id was masked in log
    # Find call to RUCHNOY_RETRAY_OTPRAVKA
    retry_log_call = next(
        (call for call in mock_plog_info.call_args_list if "РУЧНОЙ_РЕТРАЙ_ОТПРАВКА" in call.args[0]),
        None,
    )
    assert retry_log_call is not None
    # 31d55000-000f-5000-9000-1e54ca65d0cf must NOT appear as argument
    assert "31d55000-000f-5000-9000-1e54ca65d0cf" not in retry_log_call.args
    assert "31d5...d0cf" in retry_log_call.args

    # Verification of user message
    user_msgs = [call.args[1] for call in bot.send_message.await_args_list if call.args[0] == user_id]
    assert any("Сохранённый способ оплаты больше недоступен в ЮKassa" in msg for msg in user_msgs)

    # Verification of admin message
    admin_msgs = [call.args[1] for call in bot.send_message.await_args_list if call.args[0] == 999001]
    assert any("🚫 Автопродление отключено (карта недоступна в YooKassa)" in msg for msg in admin_msgs)
