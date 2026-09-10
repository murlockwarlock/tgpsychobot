import asyncio
import os
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("BOT_TOKEN", "test_bot_token")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
from max_messenger_bot.services import subscriptions as max_subs


@pytest.mark.asyncio
async def test_max_show_subscription_info_expired_awaiting_renewal_shows_details():
    now = datetime(2026, 9, 10, 12, 0, 0)
    plan = SimpleNamespace(
        id=1,
        name="Премиум",
        price=300.0,
        duration_value=1,
        duration_unit="months",
        is_trial=False,
        upgrades_to_plan=None,
        allow_auto_renewal=True,
    )
    sub = SimpleNamespace(
        id=5,
        plan_id=1,
        plan=plan,
        discount_percent=10.0,
        end_date=now - timedelta(hours=1),
        auto_renewal=True,
        payment_method_id="31d55000-000f-5000-9000-1e54ca65d0cf",
        payment_attempt_count=1,
    )
    user = SimpleNamespace(
        id=100005511792,
        tg_user_id=None,
        subscription=sub,
        promo_codes=[],
    )
    config = SimpleNamespace(referral_enabled=False)

    client = SimpleNamespace(send_message=AsyncMock())

    with (
        patch.object(max_subs, "_get_user_and_subscription", AsyncMock(return_value=(user, config))),
        patch.object(max_subs, "load_active_subscription", AsyncMock(return_value=None)),
        patch.object(max_subs, "utc_now", return_value=now),
    ):
        await max_subs.show_subscription_info(client, chat_id=100005511792, user_id=100005511792)

    assert client.send_message.await_count == 1
    call_kwargs = client.send_message.await_args.kwargs
    sent_text = call_kwargs["text"]

    # Must contain warning header
    assert "<b>⚠️ Подписка истекла, ожидается оплата по автопродлению</b>" in sent_text
    # Must contain plan
    assert "<b>Тариф:</b> Премиум" in sent_text
    # Must contain duration
    assert "<b>Период:</b> 1 мес." in sent_text
    # Must contain calculated final price: 300 * (1 - 0.10) = 270.00 руб.
    assert "<b>Сумма к списанию:</b> 270.00 руб." in sent_text
    # Must contain attempt count: 1 из 3
    assert "<b>Попыток списания:</b> 1 из 3" in sent_text
    # Must have retry keyboard attached
    assert call_kwargs["attachments"] is not None


@pytest.mark.asyncio
async def test_max_show_subscription_info_after_permanent_deactivation_shows_normal_plans():
    now = datetime(2026, 9, 10, 12, 0, 0)
    plan = SimpleNamespace(
        id=1,
        name="Премиум",
        price=300.0,
        duration_value=1,
        duration_unit="months",
        is_trial=False,
    )
    # Permanent deactivation sets auto_renewal=False, payment_method_id=None
    sub = SimpleNamespace(
        id=5,
        plan_id=1,
        plan=plan,
        discount_percent=0.0,
        end_date=now - timedelta(hours=1),
        auto_renewal=False,
        payment_method_id=None,
        payment_attempt_count=0,
    )
    user = SimpleNamespace(
        id=100005511792,
        tg_user_id=None,
        subscription=sub,
        promo_codes=[],
    )
    config = SimpleNamespace(referral_enabled=False)

    client = SimpleNamespace(send_message=AsyncMock())

    with (
        patch.object(max_subs, "_get_user_and_subscription", AsyncMock(return_value=(user, config))),
        patch.object(max_subs, "load_active_subscription", AsyncMock(return_value=None)),
        patch.object(max_subs, "utc_now", return_value=now),
    ):
        await max_subs.show_subscription_info(client, chat_id=100005511792, user_id=100005511792)

    assert client.send_message.await_count == 1
    call_kwargs = client.send_message.await_args.kwargs
    sent_text = call_kwargs["text"]

    # Must NOT show awaiting renewal message
    assert "ожидается оплата по автопродлению" not in sent_text
    # Must offer normal subscription message
    assert "У вас нет активной подписки" in sent_text
