from __future__ import annotations

import asyncio
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload
try:
    from yookassa import Configuration, Payment
except ModuleNotFoundError:
    Configuration = None
    Payment = None

from ..api import MaxApiClient
from ..identity import is_max_user_id, max_communication_name, max_username, raw_max_user_id
from ..keyboards import callback_button, inline_keyboard, link_button, main_menu_row, payment_providers_keyboard, plans_keyboard, retry_subscription_keyboard, subscription_keyboard
from ..logging_utils import get_payments_logger
from ..legacy import (
    PromoCode,
    ReferralTemplate,
    RobokassaPayment,
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
from .subscription_access import load_active_subscription
from robokassa_signing import generate_robokassa_payment_url
from error_reporting import sanitize_secret_values
from subscription_renewal import (
    calculate_renewal_details,
    claim_yookassa_recurring_attempt,
    execute_or_replay_yookassa_recurring_attempt,
    execute_yookassa_recurring_attempt,
    finalize_yookassa_payment_success,
    finalize_yookassa_payment_canceled,
    finalize_yookassa_attempt_no_payment,
    transition_attempt_to_manual_review,
    update_yookassa_attempt_pending,
    update_yookassa_attempt_unknown,
    mask_payment_method_id,
)
from subscription_retry_policy import can_retry_now, get_next_retry_at
from subscription_dates import extend_subscription_end_date


log = get_payments_logger("subscriptions")


def _generate_robokassa_payment_url(
    merchant_login: str,
    merchant_password_1: str,
    cost: float,
    invoice_id: int,
    description: str,
) -> str:
    return generate_robokassa_payment_url(
        merchant_login=merchant_login,
        merchant_password_1=merchant_password_1,
        cost=cost,
        invoice_id=invoice_id,
        description=description,
        is_test=0,
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
    async with async_session_maker() as session:
        active_subscription = await load_active_subscription(session, user_id, now)
    referral_enabled = bool(config and config.referral_enabled)
    referral_btn_name = getattr(config, "referral_sub_btn_name", "🤝 Реферальная программа") if config else "🤝 Реферальная программа"
    tg_link_line = (
        f"\n\n<b>Привязанный TG ID:</b> <code>{user.tg_user_id}</code>"
        if user.tg_user_id is not None
        else ""
    )

    text = "У вас нет активной подписки.\n\nОформите её, чтобы получить доступ ко всем возможностям бота."
    sub_info = None
    if active_subscription:
        sub = active_subscription.subscription
        source_line = "\n<b>Источник:</b> привязанный Telegram" if active_subscription.source == "telegram" else ""
        if sub.plan_id and sub.plan:
            plan = sub.plan
            unit = "дн." if plan.duration_unit == "days" else "мес."
            renewal_line = ""
            if getattr(plan, "allow_auto_renewal", True):
                renewal_line = f"\n<b>Автопродление:</b> {'✅ Включено' if sub.auto_renewal else '❌ Выключено'}"
            text = (
                "<b>⭐️ Ваша подписка активна</b>\n\n"
                f"<b>Тариф:</b> {plan.name} ({plan.duration_value} {unit})\n"
                f"<b>Действует до:</b> {format_msk(sub.end_date)} МСК"
                f"{source_line}"
                f"{renewal_line}"
            )
            sub_info = None if active_subscription.source == "telegram" else {
                "auto_renewal": sub.auto_renewal,
                "allow_auto_renewal": getattr(plan, "allow_auto_renewal", True),
            }
        else:
            text = (
                "<b>🎁 У вас активен бонусный доступ</b>\n\n"
                f"Действует до: {format_msk(sub.end_date)} МСК"
                f"{source_line}"
            )
    elif (
        user.subscription
        and user.subscription.end_date <= now
        and user.subscription.auto_renewal
        and user.subscription.payment_method_id
    ):
        details = calculate_renewal_details(user, user.subscription)
        plan_line = f"\n<b>Тариф:</b> {details.plan_name}" if details else ""
        period_line = f"\n<b>Период:</b> {details.duration_text}" if details else ""
        price_line = f"\n<b>Сумма к списанию:</b> {details.final_price:.2f} руб." if details else ""
        attempts_count = details.attempt_count if details else user.subscription.payment_attempt_count
        attempts_line = f"\n<b>Попыток списания:</b> {attempts_count} из 3"
        text = (
            "<b>⚠️ Подписка истекла, ожидается оплата по автопродлению</b>\n"
            f"{plan_line}"
            f"{period_line}"
            f"{price_line}"
            f"{attempts_line}\n\n"
            "Вы можете повторить списание вручную или оформить подписку заново."
        )
        await client.send_message(chat_id=chat_id, text=f"{text}{tg_link_line}", attachments=retry_subscription_keyboard())
        return

    text = f"{text}{tg_link_line}"
    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=subscription_keyboard(sub_info, referral_enabled, referral_btn_name, user.tg_user_id),
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
        stmt = select(SubscriptionPlan).where(SubscriptionPlan.is_active == True)
        if not (user and user.is_admin):
            stmt = stmt.where(SubscriptionPlan.admin_only == False)
        all_plans = (
            await session.execute(
                stmt.options(selectinload(SubscriptionPlan.upgrades_to_plan))
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

    import html
    duration_unit_text = "дн." if plan.duration_unit == 'days' else "мес."
    text = (
        f"<b>Тариф:</b> {plan.name} ({plan.duration_value} {duration_unit_text})\n"
        f"<b>Стоимость:</b> {final_price:.2f} руб.\n"
    )
    if plan.description:
        text += f"{html.escape(plan.description)}\n"
    text += "\n"

    if plan.is_trial and plan.upgrades_to_plan:
        upgrade_plan = plan.upgrades_to_plan
        upgrade_price = upgrade_plan.price
        upgrade_plan_allows_renewal = getattr(upgrade_plan, 'allow_auto_renewal', True)

        if discount_percent > 0:
            upgrade_price = upgrade_price * (1 - discount_percent / 100)

        upgrade_duration_unit_text = "дн." if upgrade_plan.duration_unit == 'days' else "мес."
        if upgrade_plan_allows_renewal:
            text += (
                f"<b>Далее:</b> {upgrade_price:.2f} руб. / "
                f"{upgrade_plan.duration_value} {upgrade_duration_unit_text}\n"
                f"(автопереход на «{upgrade_plan.name}»)\n\n"
            )
        else:
            text += (
                f"<b>После пробного периода:</b> {upgrade_price:.2f} руб. / "
                f"{upgrade_plan.duration_value} {upgrade_duration_unit_text}\n"
                f"(тариф «{upgrade_plan.name}», оформление вручную)\n\n"
            )

    text += "Выберите способ оплаты:"
    providers = []
    if config.yookassa_shop_id and config.yookassa_secret_key:
        providers.append(callback_button("ЮKassa", f"pay_yookassa_{plan_id}"))
    if config.robokassa_merchant_login and config.robokassa_password_1:
        providers.append(callback_button("Robokassa", f"pay_robokassa_{plan_id}"))
    await client.send_message(chat_id=chat_id, text=text, attachments=payment_providers_keyboard(providers))
    log.info("Payment providers shown user_id=%s plan_id=%s providers=%s", user_id, plan_id, [item["text"] for item in providers])


async def create_yookassa_link(client: MaxApiClient, chat_id: int, user_id: int, plan_id: int) -> None:
    if Configuration is None or Payment is None:
        await client.send_message(chat_id=chat_id, text="ЮKassa недоступна: модуль оплаты не установлен.")
        return
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
        privacy_url = config.privacy_policy_url or "#"
        offer_url = config.offer_agreement_url or "#"
        plan_allows_renewal = getattr(plan, 'allow_auto_renewal', True)
        payment_type_line = (
            "Регулярная оплата, можно отключить в любой момент"
            if (plan_allows_renewal or plan.is_trial)
            else "Разовая оплата"
        )
        text = (
            "Ваша ссылка на оплату готова.\n\n"
            f"Нажимая «Оплатить», я даю согласие на <a href='{privacy_url}'>обработку персональных данных</a> и принимаю <a href='{offer_url}'>договор оферты</a>.\n\n"
            f"<b>Сумма:</b> {price:.2f} руб.\n"
            f"{payment_type_line}"
        )
        await client.send_message(chat_id=chat_id, text=text, attachments=[{"type": "inline_keyboard", "payload": {"buttons": [[link_button("💳 Оплатить через ЮKassa", payment.confirmation.confirmation_url)], [callback_button("⬅️ Назад", f"sub_pay_{plan_id}")], main_menu_row()]}}])
    except Exception as exc:
        log.error(
            "Yookassa payment creation failed user_id=%s plan_id=%s amount=%.2f error=%s",
            user_id,
            plan_id,
            price,
            sanitize_secret_values(str(exc)),
        )
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
        # Create DB record first so webhook can look it up by auto-increment id
        async with async_session_maker() as session:
            new_payment = RobokassaPayment(
                user_id=user_id,
                plan_id=plan_id,
                amount=price,
                expires_at=utc_now() + timedelta(hours=24),
            )
            session.add(new_payment)
            await session.commit()
            await session.refresh(new_payment)
            invoice_id = new_payment.id

        url = _generate_robokassa_payment_url(
            config.robokassa_merchant_login,
            config.robokassa_password_1,
            price,
            invoice_id,
            f"Подписка {plan.name}",
        )
        log.info("Robokassa payment link created user_id=%s plan_id=%s invoice_id=%s amount=%.2f", user_id, plan_id, invoice_id, price)
        privacy_url = config.privacy_policy_url or "#"
        offer_url = config.offer_agreement_url or "#"
        plan_allows_renewal = getattr(plan, 'allow_auto_renewal', True)
        payment_type_line = (
            "Регулярная оплата, можно отключить в любой момент"
            if (plan_allows_renewal or plan.is_trial)
            else "Разовая оплата"
        )
        text = (
            "Ваша ссылка на оплату готова.\n\n"
            f"Нажимая «Оплатить», я даю согласие на <a href='{privacy_url}'>обработку персональных данных</a> и принимаю <a href='{offer_url}'>договор оферты</a>.\n\n"
            f"<b>Сумма:</b> {price:.2f} руб.\n"
            f"{payment_type_line}"
        )
        await client.send_message(
            chat_id=chat_id,
            text=text,
            attachments=[{"type": "inline_keyboard", "payload": {"buttons": [[link_button("💳 Оплатить через Robokassa", url)], [callback_button("⬅️ Назад", f"sub_pay_{plan_id}")], main_menu_row()]}}],
        )
    except Exception as exc:
        log.error(
            "Robokassa payment link creation failed user_id=%s plan_id=%s amount=%.2f error=%s",
            user_id,
            plan_id,
            price,
            sanitize_secret_values(str(exc)),
        )
        await client.send_message(chat_id=chat_id, text="Не удалось сформировать ссылку Robokassa. Попробуйте позже.")


async def start_promo_entry(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "awaiting_promo_code", {})
    await client.send_message(chat_id=chat_id, text="Введите промокод сообщением.", attachments=inline_keyboard([main_menu_row()]))


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

        user_ref_pc = max_communication_name(user) if is_max_user_id(user.id) else f"{user.first_name or ''}"
        username = max_username(user) if is_max_user_id(user.id) else (f"@{user.username}" if user.username else "")
        if username and username != "не указан":
            user_ref_pc += f" ({username})"
        display_id = raw_max_user_id(user_id) if is_max_user_id(user_id) else user_id
        user_ref_pc += f" [id=<code>{display_id}</code>]"

        from .common import notify_telegram_admins
        await notify_telegram_admins(
            f"🎁 Активирован промокод «{promo.code}»\n"
            f"Пользователь: {user_ref_pc} (MAX)\n"
            f"Скидка: {promo.discount_percent}%\n"
            f"Дней: {promo.free_days}"
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
            
            user = await session.get(User, user_id)
            user_ref_cr = (
                max_communication_name(user)
                if user and is_max_user_id(user.id)
                else f"{user.first_name or ''}" if user else ""
            )
            username = max_username(user) if user and is_max_user_id(user.id) else (f"@{user.username}" if user and user.username else "")
            if username and username != "не указан":
                user_ref_cr += f" ({username})"
            display_id = raw_max_user_id(user_id) if is_max_user_id(user_id) else user_id
            user_ref_cr += f" [id=<code>{display_id}</code>]"
            
            plan_name_cr = sub.plan.name if sub.plan else "Unknown"
            from datetime import timezone, timedelta
            MSK = timezone(timedelta(hours=3))
            end_date_msk_cr = sub.end_date.astimezone(MSK).strftime('%d.%m.%Y %H:%M') if sub.end_date else "Неизвестно"
            
            from .common import notify_telegram_admins
            await notify_telegram_admins(
                f"🔕 Пользователь отключил автопродление\n"
                f"Пользователь: {user_ref_cr} (MAX)\n"
                f"Тариф: {plan_name_cr}\n"
                f"Подписка до: {end_date_msk_cr} МСК"
            )
        await session.commit()
    await show_subscription_info(client, chat_id, user_id)


async def _send_referral_templates(client: MaxApiClient, chat_id: int, ref_link: str) -> None:
    """Sends enabled referral invite templates as individual messages."""
    async with async_session_maker() as session:
        result = await session.execute(
            select(ReferralTemplate)
            .where(ReferralTemplate.is_enabled == True)
            .order_by(ReferralTemplate.order_num.asc(), ReferralTemplate.id.asc())
        )
        templates = result.scalars().all()
    if not templates:
        return
    await client.send_message(
        chat_id=chat_id,
        text="📩 <b>Шаблоны приглашений</b>\n\nНиже — готовые сообщения. Выберите любое и отправьте друзьям.",
        attachments=inline_keyboard([main_menu_row()]),
    )
    for tpl in templates:
        tpl_text = tpl.text.replace("{ref_link}", ref_link)
        await client.send_message(chat_id=chat_id, text=tpl_text)


async def show_referral_info(client: MaxApiClient, chat_id: int, user_id: int) -> None:
    from ..models import MAX_ID_OFFSET
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        if not config or not config.referral_enabled:
            await client.send_message(chat_id=chat_id, text="Реферальная программа недоступна.")
            return
        referral_count = await session.scalar(select(func.count()).select_from(User).where(User.referred_by == user_id)) or 0
    me = await client.get_me()
    username = me.get("username") or me.get("name") or ""
    original_id = user_id - MAX_ID_OFFSET
    link = f"https://max.ru/{username}?start=ref_{original_id}" if username else f"ref_{original_id}"
    text = (
        "<b>🔗 Реферальная программа</b>\n\n"
        f"<b>Ваша ссылка:</b>\n{link}\n\n"
        f"👥 <b>Приглашено:</b> {referral_count}\n\n"
        f"За каждого приглашённого вы и ваш друг получите по <b>{config.referral_bonus_days_referrer} дн.</b>"
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=inline_keyboard([main_menu_row()]))
    await _send_referral_templates(client, chat_id, link)


async def cancel_retry(client: MaxApiClient, chat_id: int, user_id: int) -> None:
    async with async_session_maker() as session:
        sub = await session.scalar(
            select(UserSubscription)
            .where(UserSubscription.user_id == user_id)
            .options(selectinload(UserSubscription.plan))
        )
        if not sub:
            await client.send_message(chat_id=chat_id, text="Подписка не найдена.")
            return
        sub.auto_renewal = False
        await session.commit()

    await client.send_message(
        chat_id=chat_id,
        text="Автопродление отключено. Вы можете оформить подписку заново, выбрав подходящий тариф.",
    )
    await show_plans(client, chat_id, user_id)


async def handle_max_manual_retry(client: MaxApiClient, chat_id: int, user_id: int) -> None:
    now = utc_now()
    async with async_session_maker() as session:
        user = await session.get(
            User,
            user_id,
            options=[
                selectinload(User.subscription)
                .selectinload(UserSubscription.plan)
                .selectinload(SubscriptionPlan.upgrades_to_plan),
                selectinload(User.promo_codes).selectinload(PromoCode.applicable_plans),
            ],
        )
        config = await session.get(SubscriptionConfig, 1)
        sub = user.subscription if user else None

        if (
            not sub
            or not sub.auto_renewal
            or not sub.payment_method_id
            or sub.payment_attempt_count >= 3
        ):
            await client.send_message(chat_id=chat_id, text="Невозможно выполнить списание.")
            return

        if not can_retry_now(sub.payment_attempt_count, sub.last_payment_attempt, now, retry_not_before=getattr(sub, 'retry_not_before', None)):
            next_retry_at = get_next_retry_at(sub.payment_attempt_count, sub.last_payment_attempt, retry_not_before=getattr(sub, 'retry_not_before', None))
            next_retry_str = format_msk(next_retry_at) if next_retry_at else "позже"
            await client.send_message(
                chat_id=chat_id,
                text=f"Повторное списание пока недоступно. Следующая попытка после {next_retry_str} МСК.",
            )
            return

        renewal_details = calculate_renewal_details(user, sub)
        if not renewal_details:
            await client.send_message(chat_id=chat_id, text="Тариф не найден.")
            return

        plan_to_charge = renewal_details.plan_to_charge
        final_price = renewal_details.final_price

        # Provider dispatch
        if sub.payment_provider == 'Robokassa':
            new_payment = RobokassaPayment(user_id=user_id, plan_id=plan_to_charge.id, amount=final_price)
            session.add(new_payment)
            await session.commit()
            log.info(
                "РУЧНОЙ_РЕТРАЙ_ОТПРАВКА | %s | Robokassa | attempts=%s | amount=%.2f | parent_inv=%s | new_inv=%s",
                user_id,
                sub.payment_attempt_count,
                final_price,
                sub.payment_method_id or "none",
                new_payment.id,
            )

            from scheduler import process_recurring_robokassa_payment
            robokassa_res = await process_recurring_robokassa_payment(
                config, plan_to_charge, final_price, sub.payment_method_id, new_payment.id
            )
            log.info(
                "РУЧНОЙ_РЕТРАЙ_РЕЗУЛЬТАТ | %s | Robokassa | result=%s | new_inv=%s",
                user_id,
                robokassa_res,
                new_payment.id,
            )

            if robokassa_res is True:
                sub.pending_robokassa_invoice_id = new_payment.id
                sub.payment_attempt_count += 1
                sub.last_payment_attempt = now
                await session.commit()
                await client.send_message(
                    chat_id=chat_id,
                    text="⏳ Запрос на списание отправлен в Robokassa. Ожидайте подтверждения оплаты.",
                )
                return
            elif robokassa_res == 'deactivate':
                sub.auto_renewal = False
                sub.payment_attempt_count = 0
                new_payment.status = 'request_deactivated'
                await session.commit()
                from .common import notify_telegram_admins
                user_ref = max_communication_name(user) if is_max_user_id(user_id) else (user.first_name or "")
                await notify_telegram_admins(
                    f"🚫 Автопродление отключено (Robokassa, MAX)\n"
                    f"Пользователь: {user_ref}\n"
                    f"Провайдер: Robokassa"
                )
                await client.send_message(
                    chat_id=chat_id,
                    text="Не удалось списать средства через Robokassa. Автопродление отключено.\n\nОформите подписку заново.",
                )
                await show_plans(client, chat_id, user_id)
                return
            elif robokassa_res == 'provider_error':
                new_payment.status = 'request_provider_error'
                sub.last_payment_attempt = now
                await session.commit()
                await client.send_message(
                    chat_id=chat_id,
                    text="Сервис Robokassa временно недоступен. Эта ошибка не засчитана как попытка списания.\n\nПопробуйте повторить запрос позже.",
                )
                return
            else:
                attempt_num = sub.payment_attempt_count + 1
                sub.payment_attempt_count = attempt_num
                sub.last_payment_attempt = now
                new_payment.status = 'request_failed'
                if attempt_num >= 3:
                    sub.auto_renewal = False
                    await session.commit()
                    from .common import notify_telegram_admins
                    user_ref = max_communication_name(user) if is_max_user_id(user_id) else (user.first_name or "")
                    await notify_telegram_admins(
                        f"🚫 Автопродление отключено (3 попытки Robokassa, MAX)\n"
                        f"Пользователь: {user_ref}\n"
                        f"Провайдер: Robokassa"
                    )
                    await client.send_message(
                        chat_id=chat_id,
                        text="Не удалось списать средства после 3 попыток через Robokassa. Автопродление отключено.\n\nОформите подписку заново.",
                    )
                    await show_plans(client, chat_id, user_id)
                else:
                    await session.commit()
                    next_retry_at = get_next_retry_at(sub.payment_attempt_count, sub.last_payment_attempt)
                    next_retry_str = format_msk(next_retry_at) if next_retry_at else "позже"
                    await client.send_message(
                        chat_id=chat_id,
                        text=f"Банк отклонил платёж (Robokassa). Попытка {attempt_num} из 3.\n\nСледующая попытка после {next_retry_str} МСК.",
                    )
                return

        elif sub.payment_provider == 'Yookassa':
            claim_res = await claim_yookassa_recurring_attempt(
                session=session,
                sub=sub,
                plan=plan_to_charge,
                price_to_charge=final_price,
                client_context="max_manual_retry",
                attempt_started_at=now,
            )
            if not claim_res.claimed:
                if claim_res.attempt:
                    if claim_res.attempt.status == "pending":
                        await client.send_message(
                            chat_id=chat_id,
                            text="Запрос в ЮKassa принят и ожидает подтверждения оплаты. Мы проверяем статус операции.",
                        )
                    else:
                        await client.send_message(
                            chat_id=chat_id,
                            text="Предыдущий платёж ещё обрабатывается шлюзом. Пожалуйста, подождите завершения операции.",
                        )
                else:
                    await client.send_message(chat_id=chat_id, text="Повторное списание недоступно.")
                return

            attempt = claim_res.attempt
            attempt_started_at = attempt.attempt_started_at

            await client.send_message(chat_id=chat_id, text="Отправляем запрос на списание...")

            if hasattr(session, "in_transaction") and session.in_transaction():
                await session.commit()

            res = await execute_or_replay_yookassa_recurring_attempt(
                attempt, plan_to_charge.name, config, logger=log
            )

            if res.outcome == "success" and res.payment_id:
                res_fin = await finalize_yookassa_payment_success(
                    session=session,
                    payment_id=res.payment_id,
                    user_id=user_id,
                    plan_id=plan_to_charge.id,
                    amount=final_price,
                    payment_method_id=sub.payment_method_id,
                    is_recurring=True,
                    recurring_attempt_key=attempt.idempotency_key,
                    logger=log,
                )
                is_new, updated_sub = res_fin[0], res_fin[1]
                action = getattr(res_fin, "action", "success")
                rec_details = getattr(res_fin, "reconciliation_details", {})
                if not is_new:
                    await client.send_message(
                        chat_id=chat_id,
                        text="Платёж уже обработан. Подписка активна.",
                    )
                    await show_subscription_info(client, chat_id, user_id)
                    return

                if action == "manual_reconciliation_required":
                    reason = rec_details.get("reason")
                    charge_amount = rec_details.get("amount", final_price)
                    from .common import notify_telegram_admins
                    user_ref = max_communication_name(user) if is_max_user_id(user_id) else (user.first_name or "")
                    if reason == "subscription_unresolved":
                        await notify_telegram_admins(
                            f"⚠️ ТРЕБУЕТСЯ РУЧНАЯ СВЕРКА: ПОДПИСКА НЕ НАЙДЕНА (YooKassa, MAX manual)\n\n"
                            f"Пользователь: {user_ref}\n"
                            f"Сумма: {charge_amount:.2f} руб\n"
                            f"PayId: {res.payment_id}\n"
                            f"Попытка (Attempt ID): {rec_details.get('attempt_id')}\n"
                            f"ID подписки: {rec_details.get('subscription_id')}\n"
                            f"Тариф (Plan ID): {rec_details.get('paid_plan_id') or rec_details.get('plan_id')}\n"
                            f"Действие: подписка НЕ продлена автоматически. Требуется ручное решение администратора."
                        )
                        await client.send_message(
                            chat_id=chat_id,
                            text=(
                                f"⚠️ Мы получили оплату ({charge_amount:.2f} руб), но не удалось найти вашу подписку для автоматического продления. "
                                f"Платёж отправлен на проверку администратору."
                            ),
                        )
                        await show_subscription_info(client, chat_id, user_id)
                        return
                    else:
                        paid_name = rec_details.get("paid_plan_name", plan_to_charge.name)
                        curr_name = rec_details.get("current_plan_name", "текущий тариф")
                        await notify_telegram_admins(
                            f"⚠️ ТРЕБУЕТСЯ РУЧНАЯ СВЕРКА ТАРИФА (YooKassa, MAX manual)\n\n"
                            f"Пользователь: {user_ref}\n"
                            f"Оплачен старый тариф: {paid_name} (ID {rec_details.get('paid_plan_id')})\n"
                            f"Текущий тариф: {curr_name} (ID {rec_details.get('current_plan_id')})\n"
                            f"Сумма: {charge_amount:.2f} руб\n"
                            f"PayId: {res.payment_id}\n"
                            f"Действие: подписка НЕ продлена автоматически. Требуется ручное решение администратора."
                        )
                        await client.send_message(
                            chat_id=chat_id,
                            text=(
                                f"⚠️ Мы получили оплату ({charge_amount:.2f} руб) по вашему предыдущему тарифу «{paid_name}». "
                                f"Поскольку сейчас у вас активен тариф «{curr_name}», платёж отправлен на проверку администратору. "
                                f"Срок действия текущей подписки не был изменён автоматически."
                            ),
                        )
                        await show_subscription_info(client, chat_id, user_id)
                        return

                end_date_ref = (updated_sub or sub).end_date
                await client.send_message(
                    chat_id=chat_id,
                    text=f"✅ Подписка успешно продлена до {format_msk(end_date_ref)} МСК.",
                )
                from .common import notify_telegram_admins
                user_ref = max_communication_name(user) if is_max_user_id(user_id) else (user.first_name or "")
                await notify_telegram_admins(
                    f"🔔 Автопродление (YooKassa, MAX)!\n\n"
                    f"Пользователь: {user_ref}\n"
                    f"Тариф: {plan_to_charge.name}\n"
                    f"Сумма: {final_price:.2f} руб\n"
                    f"До: {format_msk(end_date_ref)} МСК\n"
                    f"PayId: {res.payment_id}"
                )
                await show_subscription_info(client, chat_id, user_id)
                return

            elif res.outcome == "deactivate":
                if res.payment_id:
                    is_new, action, _ = await finalize_yookassa_payment_canceled(
                        session=session,
                        payment_id=res.payment_id,
                        cancellation_reason=res.failure_reason,
                        user_id=user_id,
                        plan_id=plan_to_charge.id,
                        amount=final_price,
                        payment_method_id=sub.payment_method_id,
                        attempt_started_at=attempt_started_at,
                        force_deactivate=True,
                        recurring_attempt_key=attempt.idempotency_key,
                        logger=log,
                    )
                else:
                    is_new, updated_sub = await finalize_yookassa_attempt_no_payment(
                        session=session,
                        attempt_id=attempt.id,
                        outcome="deactivate",
                        error_code=res.failure_reason,
                        error_message=str(res.error) if res.error else None,
                        attempt_started_at=attempt_started_at,
                        sub=sub,
                        attempt=attempt,
                        logger=log,
                    )
                    action = "deactivate"
                if not is_new:
                    await client.send_message(
                        chat_id=chat_id,
                        text="Платёж уже обработан.",
                    )
                    await show_subscription_info(client, chat_id, user_id)
                    return

                if action in ("historical_canceled", "orphan_canceled"):
                    await client.send_message(
                        chat_id=chat_id,
                        text="Предыдущая попытка списания завершена. Текущие настройки вашей подписки сохранены.",
                    )
                    await show_subscription_info(client, chat_id, user_id)
                    return

                from .common import notify_telegram_admins
                user_ref = max_communication_name(user) if is_max_user_id(user_id) else (user.first_name or "")
                await notify_telegram_admins(
                    f"🚫 Автопродление отключено (карта недоступна в YooKassa, MAX)\n"
                    f"Пользователь: {user_ref}\n"
                    f"Провайдер: Yookassa"
                )
                await client.send_message(
                    chat_id=chat_id,
                    text="Сохранённый способ оплаты больше недоступен в ЮKassa. Автопродление отключено.\n\nПожалуйста, выберите тариф и оформите подписку заново.",
                )
                await show_plans(client, chat_id, user_id)
                return

            elif res.outcome == "manual_review":
                is_new_mr, _ = await transition_attempt_to_manual_review(
                    session=session,
                    attempt_id=attempt.id,
                    reason=res.failure_reason or "missing_or_corrupt_payload",
                    logger=log,
                )
                if not is_new_mr:
                    await client.send_message(
                        chat_id=chat_id,
                        text="Платёж уже обработан.",
                    )
                    await show_subscription_info(client, chat_id, user_id)
                    return

                from .common import notify_telegram_admins
                user_ref = max_communication_name(user) if is_max_user_id(user_id) else (user.first_name or "")
                await notify_telegram_admins(
                    f"⚠️ Платёж YooKassa требует ручной проверки (manual_review, MAX)\n"
                    f"Пользователь: {user_ref}\n"
                    f"Тариф: {plan_to_charge.name}\n"
                    f"AttemptId: {attempt.id}\n"
                    f"Причина: {res.failure_reason or 'missing_or_corrupt_payload'}\n"
                    f"Автопродление приостановлено."
                )
                await client.send_message(
                    chat_id=chat_id,
                    text="Автопродление приостановлено для ручной проверки. Пожалуйста, оформите подписку заново в меню.",
                )
                await show_plans(client, chat_id, user_id)
                return

            elif res.outcome == "pending":
                if res.payment_id:
                    await update_yookassa_attempt_pending(session, attempt.id, res.payment_id)
                await client.send_message(
                    chat_id=chat_id,
                    text="Запрос в ЮKassa принят и ожидает подтверждения. Мы проверяем статус операции. Пока не отправляйте повторный запрос.",
                )
                return

            elif res.outcome == "unknown":
                await update_yookassa_attempt_unknown(
                    session, attempt.id, error_code="unknown", error_message=res.failure_reason
                )
                await client.send_message(
                    chat_id=chat_id,
                    text="Платёжный шлюз ЮKassa обрабатывает запрос. Мы проверяем статус операции. Попробуйте снова позже.",
                )
                return

            elif res.outcome == "integration_error":
                is_new_ie, _ = await finalize_yookassa_attempt_no_payment(
                    session=session,
                    attempt_id=attempt.id,
                    outcome="integration_error",
                    error_code=res.failure_reason,
                    error_message=str(res.error) if res.error else None,
                    attempt_started_at=attempt_started_at,
                    sub=sub,
                    attempt=attempt,
                    logger=log,
                )
                if not is_new_ie:
                    await client.send_message(
                        chat_id=chat_id,
                        text="Платёж уже обработан.",
                    )
                    await show_subscription_info(client, chat_id, user_id)
                    return

                await client.send_message(
                    chat_id=chat_id,
                    text="Ошибка интеграции с платёжным сервисом. Списание временно приостановлено. Мы уже разбираемся с проблемой.",
                )
                return

            elif res.outcome in ("provider_error", "rate_limit", "auth_error"):
                await update_yookassa_attempt_unknown(
                    session, attempt.id, error_code=res.outcome, error_message=res.failure_reason
                )
                await client.send_message(
                    chat_id=chat_id,
                    text="Платёжный сервис ЮKassa временно недоступен или вернул ошибку. Эта ошибка не засчитана как попытка списания.\n\nПопробуйте повторить запрос позже.",
                )
                return

            else:  # declined
                if res.payment_id:
                    is_new, action, updated_sub = await finalize_yookassa_payment_canceled(
                        session=session,
                        payment_id=res.payment_id,
                        cancellation_reason=res.failure_reason,
                        user_id=user_id,
                        plan_id=plan_to_charge.id,
                        amount=final_price,
                        payment_method_id=sub.payment_method_id,
                        attempt_started_at=attempt_started_at,
                        recurring_attempt_key=attempt.idempotency_key,
                        logger=log,
                    )
                else:
                    is_new, updated_sub = await finalize_yookassa_attempt_no_payment(
                        session=session,
                        attempt_id=attempt.id,
                        outcome="canceled",
                        error_code=res.failure_reason,
                        error_message=str(res.error) if res.error else None,
                        attempt_started_at=attempt_started_at,
                        sub=sub,
                        attempt=attempt,
                        logger=log,
                    )
                    action = "canceled"

                if not is_new:
                    await client.send_message(
                        chat_id=chat_id,
                        text="Платёж уже обработан.",
                    )
                    await show_subscription_info(client, chat_id, user_id)
                    return

                if action in ("historical_canceled", "orphan_canceled"):
                    await client.send_message(
                        chat_id=chat_id,
                        text="Предыдущая попытка списания завершена. Текущие настройки вашей подписки сохранены.",
                    )
                    await show_subscription_info(client, chat_id, user_id)
                    return

                if action == "unknown_cancellation":
                    from .common import notify_telegram_admins
                    user_ref = max_communication_name(user) if is_max_user_id(user_id) else (user.first_name or "")
                    await notify_telegram_admins(
                        f"⚠️ Автопродление приостановлено (неизвестная причина отмены YooKassa, MAX)\n"
                        f"Пользователь: {user_ref}\nТариф: {plan_to_charge.name}\n"
                        f"PayId: {res.payment_id}\nПричина: {res.failure_reason or 'не указана'}"
                    )
                    await client.send_message(
                        chat_id=chat_id,
                        text="Не удалось выполнить списание (нестандартный ответ банка). Автопродление приостановлено во избежание повторных списаний. Пожалуйста, оформите подписку вручную.",
                    )
                    await show_plans(client, chat_id, user_id)
                    return

                attempt_num = (updated_sub or sub).payment_attempt_count
                if attempt_num >= 3 or action == "deactivate":
                    from .common import notify_telegram_admins
                    user_ref = max_communication_name(user) if is_max_user_id(user_id) else (user.first_name or "")
                    await notify_telegram_admins(
                        f"🚫 Автопродление отключено (3 попытки YooKassa, MAX)\n"
                        f"Пользователь: {user_ref}\n"
                        f"Провайдер: Yookassa"
                    )
                    await client.send_message(
                        chat_id=chat_id,
                        text="Не удалось списать средства после 3 попыток. Автопродление отключено.\n\nОформите подписку заново.",
                    )
                    await show_plans(client, chat_id, user_id)
                else:
                    retry_nb = getattr(updated_sub or sub, 'retry_not_before', None)
                    next_retry_at = get_next_retry_at(attempt_num, attempt_started_at, retry_not_before=retry_nb)
                    next_retry_str = format_msk(next_retry_at) if next_retry_at else "позже"
                    reason_txt = f" ({res.failure_reason})" if res.failure_reason else ""
                    await client.send_message(
                        chat_id=chat_id,
                        text=f"Банк отклонил платёж{reason_txt}. Попытка {attempt_num} из 3.\n\nСледующая попытка после {next_retry_str} МСК.",
                    )
                return

        else:
            await client.send_message(
                chat_id=chat_id,
                text="Платёжный провайдер подписки не поддерживает автоматическое списание.\n\nОформите подписку заново.",
            )
            return
