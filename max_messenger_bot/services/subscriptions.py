from __future__ import annotations

import asyncio
import hashlib
from datetime import timedelta
from urllib.parse import urlencode

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload
from yookassa import Configuration, Payment

from ..api import MaxApiClient
from ..keyboards import callback_button, link_button, payment_providers_keyboard, plans_keyboard, retry_subscription_keyboard, subscription_keyboard
from ..logging_utils import get_payments_logger
from ..legacy import (
    PromoCode,
    SubscriptionConfig,
    SubscriptionPlan,
    TrialUsageHistory,
    User,
    UserSubscription,
    YookassaPayment,
    async_session_maker,
)
from ..storage import StateStore
from ..time_utils import format_msk, utc_now


log = get_payments_logger("subscriptions")


def _calculate_signature(*args) -> str:
    return hashlib.md5(":".join(str(arg) for arg in args).encode()).hexdigest()


def _generate_robokassa_payment_url(
    merchant_login: str,
    merchant_password_1: str,
    cost: float,
    invoice_id: int,
    description: str,
) -> str:
    signature = _calculate_signature(merchant_login, f"{cost:.2f}", invoice_id, merchant_password_1)
    return "https://auth.robokassa.ru/Merchant/Index.aspx?" + urlencode(
        {
            "MerchantLogin": merchant_login,
            "OutSum": f"{cost:.2f}",
            "InvId": invoice_id,
            "Description": description,
            "SignatureValue": signature,
            "IsTest": 0,
        }
    )


async def _get_user_and_subscription(user_id: int):
    async with async_session_maker() as session:
        user = await session.get(
            User,
            user_id,
            options=[
                selectinload(User.subscription).selectinload(UserSubscription.plan).selectinload(SubscriptionPlan.upgrades_to_plan),
                selectinload(User.promo_codes).selectinload(PromoCode.applicable_plans),
            ],
        )
        config = await session.get(SubscriptionConfig, 1)
    return user, config


async def show_subscription_info(client: MaxApiClient, chat_id: int, user_id: int) -> None:
    user, config = await _get_user_and_subscription(user_id)
    if not user:
        return

    now = utc_now()
    referral_enabled = bool(config and config.referral_enabled)
    referral_btn_name = config.referral_sub_btn_name if config else "🤝 Реферальная программа"

    text = "У вас нет активной подписки.\n\nОформите её, чтобы получить доступ ко всем возможностям бота."
    sub_info = None
    if user.subscription and user.subscription.end_date > now:
        sub = user.subscription
        if sub.plan_id and sub.plan:
            plan = sub.plan
            unit = "дн." if plan.duration_unit == "days" else "мес."
            renewal_line = ""
            if getattr(plan, "allow_auto_renewal", True):
                renewal_line = f"<br><b>Автопродление:</b> {'✅ Включено' if sub.auto_renewal else '❌ Выключено'}"
            text = (
                "<b>⭐️ Ваша подписка активна</b><br><br>"
                f"<b>Тариф:</b> {plan.name} ({plan.duration_value} {unit})<br>"
                f"<b>Действует до:</b> {format_msk(sub.end_date)} МСК"
                f"{renewal_line}"
            )
            sub_info = {"auto_renewal": sub.auto_renewal, "allow_auto_renewal": getattr(plan, "allow_auto_renewal", True)}
        else:
            text = (
                "<b>🎁 У вас активен бонусный доступ</b><br><br>"
                f"Действует до: {format_msk(sub.end_date)} МСК"
            )
    elif user.subscription and user.subscription.end_date <= now and user.subscription.auto_renewal and user.subscription.payment_method_id:
        text = (
            "<b>⚠️ Подписка истекла, ожидается оплата по автопродлению</b><br><br>"
            "Вы можете повторить списание вручную или оформить подписку заново."
        )
        await client.send_message(chat_id=chat_id, text=text, attachments=retry_subscription_keyboard())
        return

    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=subscription_keyboard(sub_info, referral_enabled, referral_btn_name),
    )


async def show_plans(client: MaxApiClient, chat_id: int, user_id: int) -> None:
    user, _ = await _get_user_and_subscription(user_id)
    if not user:
        return
    now = utc_now()
    async with async_session_maker() as session:
        trial_history = (
            await session.execute(select(TrialUsageHistory).where(TrialUsageHistory.user_id == user_id))
        ).scalars().all()
        all_plans = (
            await session.execute(
                select(SubscriptionPlan)
                .where(SubscriptionPlan.is_active == True)
                .options(selectinload(SubscriptionPlan.upgrades_to_plan))
                .order_by(SubscriptionPlan.price.asc())
            )
        ).scalars().all()

    eligible = []
    for plan in all_plans:
        if not plan.is_trial:
            eligible.append(plan)
            continue
        usage = next((item for item in trial_history if item.plan_id == plan.id or item.plan_id is None), None)
        if not usage:
            eligible.append(plan)
            continue
        if plan.trial_cooldown_days > 0 and now > usage.used_at + timedelta(days=plan.trial_cooldown_days):
            eligible.append(plan)

    global_discount = user.subscription.discount_percent if user.subscription else 0
    text = "Выберите подходящий тарифный план:"
    await client.send_message(chat_id=chat_id, text=text, attachments=plans_keyboard(eligible, global_discount, user.promo_codes))


async def choose_payment_provider(client: MaxApiClient, chat_id: int, user_id: int, plan_id: int) -> None:
    user, config = await _get_user_and_subscription(user_id)
    if not user or not config:
        return
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id, options=[selectinload(SubscriptionPlan.upgrades_to_plan)])
    if not plan:
        await client.send_message(chat_id=chat_id, text="Тариф не найден.")
        return

    discount_percent = user.subscription.discount_percent if user.subscription else 0
    specific = next((promo for promo in user.promo_codes if not promo.applies_to_all_plans and any(item.id == plan_id for item in promo.applicable_plans)), None)
    if specific:
        discount_percent = specific.discount_percent
    elif discount_percent == 0:
        all_plans = next((promo for promo in user.promo_codes if promo.applies_to_all_plans), None)
        if all_plans:
            discount_percent = all_plans.discount_percent

    final_price = plan.price
    if discount_percent > 0 and not plan.is_trial:
        final_price = plan.price * (1 - discount_percent / 100)

    text = (
        f"<b>Тариф:</b> {plan.name}<br>"
        f"<b>Стоимость:</b> {final_price:.2f} руб.<br><br>"
        "Выберите способ оплаты:"
    )
    providers = []
    if config.yookassa_shop_id and config.yookassa_secret_key:
        providers.append(callback_button("ЮKassa", f"pay_yookassa_{plan_id}"))
    if config.robokassa_merchant_login and config.robokassa_password_1:
        providers.append(callback_button("Robokassa", f"pay_robokassa_{plan_id}"))
    await client.send_message(chat_id=chat_id, text=text, attachments=payment_providers_keyboard(providers))
    log.info("Payment providers shown user_id=%s plan_id=%s providers=%s", user_id, plan_id, [item["text"] for item in providers])


async def create_yookassa_link(client: MaxApiClient, chat_id: int, user_id: int, plan_id: int) -> None:
    user, config = await _get_user_and_subscription(user_id)
    if not user or not config:
        return
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
    if not plan or not config.yookassa_shop_id or not config.yookassa_secret_key:
        await client.send_message(chat_id=chat_id, text="ЮKassa не настроена.")
        return

    Configuration.account_id = config.yookassa_shop_id
    Configuration.secret_key = config.yookassa_secret_key

    discount_percent = user.subscription.discount_percent if user.subscription else 0
    price = plan.price * (1 - discount_percent / 100) if discount_percent and not plan.is_trial else plan.price

    try:
        me = await client.get_me()
        username = me.get("username") or me.get("name") or "bot"
        payment = await asyncio.to_thread(
            Payment.create,
            {
                "amount": {"value": f"{price:.2f}", "currency": "RUB"},
                "confirmation": {"type": "redirect", "return_url": f"https://max.ru/{username}"},
                "capture": True,
                "description": f"Оплата подписки «{plan.name}»",
                "metadata": {"user_id": str(user_id), "plan_id": str(plan_id)},
                "merchant_customer_id": str(user_id),
                "save_payment_method": True,
            },
            f"max-{user_id}-{plan_id}-{int(utc_now().timestamp())}",
        )
        async with async_session_maker() as session:
            session.add(
                YookassaPayment(
                    payment_id=payment.id,
                    user_id=user_id,
                    plan_id=plan_id,
                    amount=price,
                    status=payment.status,
                    payment_method_id=payment.payment_method.id if payment.payment_method else None,
                    is_recurring=False,
                )
            )
            await session.commit()
        log.info("Yookassa payment created user_id=%s plan_id=%s payment_id=%s amount=%.2f", user_id, plan_id, payment.id, price)
        text = f"Ссылка на оплату готова.<br><br><b>Сумма:</b> {price:.2f} руб."
        await client.send_message(chat_id=chat_id, text=text, attachments=[{"type": "inline_keyboard", "payload": {"buttons": [[link_button("💳 Оплатить через ЮKassa", payment.confirmation.confirmation_url)], [callback_button("⬅️ Назад", f"sub_pay_{plan_id}")]]}}])
    except Exception:
        log.exception("Yookassa payment creation failed user_id=%s plan_id=%s amount=%.2f", user_id, plan_id, price)
        await client.send_message(chat_id=chat_id, text="Не удалось сформировать ссылку ЮKassa. Попробуйте позже.")


async def create_robokassa_link(client: MaxApiClient, chat_id: int, user_id: int, plan_id: int) -> None:
    user, config = await _get_user_and_subscription(user_id)
    if not user or not config:
        return
    async with async_session_maker() as session:
        plan = await session.get(SubscriptionPlan, plan_id)
    if not plan or not config.robokassa_merchant_login or not config.robokassa_password_1:
        await client.send_message(chat_id=chat_id, text="Robokassa не настроена.")
        return
    discount_percent = user.subscription.discount_percent if user.subscription else 0
    price = plan.price * (1 - discount_percent / 100) if discount_percent and not plan.is_trial else plan.price
    try:
        invoice_id = int(utc_now().timestamp())
        url = _generate_robokassa_payment_url(
            config.robokassa_merchant_login,
            config.robokassa_password_1,
            price,
            invoice_id,
            f"Подписка {plan.name}",
        )
        log.info("Robokassa payment link created user_id=%s plan_id=%s invoice_id=%s amount=%.2f", user_id, plan_id, invoice_id, price)
        await client.send_message(
            chat_id=chat_id,
            text=f"Ссылка на оплату готова.<br><br><b>Сумма:</b> {price:.2f} руб.",
            attachments=[{"type": "inline_keyboard", "payload": {"buttons": [[link_button("💳 Оплатить через Robokassa", url)], [callback_button("⬅️ Назад", f"sub_pay_{plan_id}")]]}}],
        )
    except Exception:
        log.exception("Robokassa payment link creation failed user_id=%s plan_id=%s amount=%.2f", user_id, plan_id, price)
        await client.send_message(chat_id=chat_id, text="Не удалось сформировать ссылку Robokassa. Попробуйте позже.")


async def start_promo_entry(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "awaiting_promo_code", {})
    await client.send_message(chat_id=chat_id, text="Введите промокод сообщением.")


async def apply_promo_code(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, code_text: str) -> None:
    code_text = code_text.strip()
    async with async_session_maker() as session:
        user = await session.get(
            User,
            user_id,
            options=[
                selectinload(User.subscription),
                selectinload(User.promo_codes).selectinload(PromoCode.applicable_plans),
            ],
        )
        promo = await session.scalar(select(PromoCode).where(PromoCode.code == code_text))
        if not user or not promo or not promo.is_active or promo.times_used >= promo.max_uses:
            await client.send_message(chat_id=chat_id, text="❌ Промокод не найден, истёк или недействителен.")
            return
        if any(item.id == promo.id for item in user.promo_codes):
            await client.send_message(chat_id=chat_id, text="❌ Вы уже активировали этот промокод.")
            return

        now = utc_now()
        sub = user.subscription
        if not sub:
            sub = UserSubscription(
                user_id=user_id,
                plan_id=None,
                start_date=now,
                end_date=now,
                auto_renewal=False,
                payment_provider="Promo",
                payment_attempt_count=0,
                discount_percent=0,
            )
            session.add(sub)

        is_active_sub = bool(sub and sub.end_date > now and sub.plan_id is not None)
        if promo.free_days > 0 and not is_active_sub:
            base_date = sub.end_date if sub.end_date and sub.end_date > now else now
            sub.plan_id = None
            sub.start_date = now
            sub.end_date = base_date + timedelta(days=promo.free_days)
            sub.auto_renewal = False
            sub.payment_provider = "Trial Promo"
            session.add(TrialUsageHistory(user_id=user_id, plan_id=None, used_at=now))
        if promo.discount_percent > 0:
            sub.discount_percent = max(sub.discount_percent, promo.discount_percent)

        promo.times_used += 1
        user.promo_codes.append(promo)
        await session.commit()
        log.info(
            "Promo applied user_id=%s promo_id=%s code=%s free_days=%s discount=%s",
            user_id,
            promo.id,
            promo.code,
            promo.free_days,
            promo.discount_percent,
        )

    await states.clear(user_id)
    if promo.free_days > 0 and (not is_active_sub):
        await client.send_message(chat_id=chat_id, text=f"✅ Пробный период активирован: {promo.free_days} дн.")
    elif promo.discount_percent > 0:
        await client.send_message(chat_id=chat_id, text=f"✅ Скидка {promo.discount_percent}% сохранена.")
    await show_subscription_info(client, chat_id, user_id)


async def set_renewal(client: MaxApiClient, chat_id: int, user_id: int, enabled: bool) -> None:
    async with async_session_maker() as session:
        sub = await session.scalar(select(UserSubscription).where(UserSubscription.user_id == user_id).options(selectinload(UserSubscription.plan)))
        if not sub:
            await client.send_message(chat_id=chat_id, text="Подписка не найдена.")
            return
        sub.auto_renewal = enabled
        if not enabled:
            sub.payment_attempt_count = 0
            sub.last_payment_attempt = None
            sub.pending_robokassa_invoice_id = None
        await session.commit()
    await show_subscription_info(client, chat_id, user_id)


async def show_referral_info(client: MaxApiClient, chat_id: int, user_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        if not config or not config.referral_enabled:
            await client.send_message(chat_id=chat_id, text="Реферальная программа недоступна.")
            return
        referral_count = await session.scalar(select(func.count()).select_from(User).where(User.referred_by == user_id)) or 0
    me = await client.get_me()
    username = me.get("username") or me.get("name") or ""
    link = f"https://max.ru/{username}?start=ref_{user_id}" if username else f"ref_{user_id}"
    text = (
        "<b>🔗 Реферальная программа</b><br><br>"
        f"<b>Ваша ссылка:</b><br>{link}<br><br>"
        f"👥 <b>Приглашено:</b> {referral_count}<br><br>"
        f"За каждого приглашённого вы и ваш друг получите по <b>{config.referral_bonus_days_referrer} дн.</b>"
    )
    await client.send_message(chat_id=chat_id, text=text)
