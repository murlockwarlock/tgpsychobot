from aiohttp import web
from aiogram import Bot
import json
from datetime import datetime, timedelta, timezone
from sqlalchemy import select, update
import aiohttp
import decimal
import hashlib
from urllib import parse
from urllib.parse import urlparse
from dateutil.relativedelta import relativedelta
from database import RobokassaPayment, PromoCode, YookassaPayment
import logging

from database import (async_session_maker, UserSubscription, SubscriptionPlan, SubscriptionConfig, User,
                      TrialUsageHistory, get_all_admin_ids, ReferralPaymentLog)
from sqlalchemy import func
from aiogram.fsm.context import FSMContext
from subscription_retry_policy import get_next_retry_at
from error_reporting import notify_admins_about_error

import os
import re

log = logging.getLogger(__name__)
plog = logging.getLogger("payment_events")
ROBOKASSA_INVOICE_LIFETIME = timedelta(hours=2)


def clean_html_for_max(text: str) -> str:
    # Remove HTML tags like <b>, </b>, <i>, </i>, <code>, </code>, etc.
    return re.sub(r'<[^>]+>', '', text)


async def send_msg_universal(bot: Bot, user_id: int, text: str, parse_mode: str | None = None, reply_markup=None) -> bool:
    if user_id < 100_000_000_000:
        # Telegram notification
        try:
            await bot.send_message(user_id, text, parse_mode=parse_mode, reply_markup=reply_markup)
            return True
        except Exception as e:
            log.error(f"Failed to send Telegram message to {user_id}: {e}")
            return False
    else:
        # MAX notification
        token = os.environ.get("MAX_BOT_TOKEN")
        if not token:
            log.warning(f"Cannot send MAX message to {user_id}: MAX_BOT_TOKEN not configured in env")
            return False
        base_url = os.environ.get("MAX_API_BASE", "https://platform-api.max.ru")
        
        attachments = None
        if reply_markup and hasattr(reply_markup, "inline_keyboard"):
            max_rows = []
            for row in reply_markup.inline_keyboard:
                max_row = []
                for btn in row:
                    if getattr(btn, "callback_data", None):
                        max_row.append({"type": "callback", "text": btn.text, "payload": btn.callback_data})
                    elif getattr(btn, "url", None):
                        max_row.append({"type": "link", "text": btn.text, "url": btn.url})
                if max_row:
                    max_rows.append(max_row)
            if max_rows:
                attachments = [{"type": "inline_keyboard", "payload": {"buttons": max_rows}}]

        try:
            from max_messenger_bot.api import MaxApiClient
            from max_messenger_bot.models import MAX_ID_OFFSET
            async with MaxApiClient(token=token, base_url=base_url) as client:
                max_api_user_id = user_id - MAX_ID_OFFSET
                clean_text = clean_html_for_max(text)
                await client.send_message(user_id=max_api_user_id, text=clean_text, attachments=attachments)
                return True
        except Exception as e:
            log.error(f"Failed to send MAX message to {user_id}: {e}", exc_info=e)
            return False




def calculate_signature(*args) -> str:
    return hashlib.md5(':'.join(str(arg) for arg in args).encode()).hexdigest()


def build_robokassa_invoice_access_token(invoice_id: int, user_id: int, password: str) -> str:
    return hashlib.md5(f"{invoice_id}:{user_id}:{password}".encode()).hexdigest()


def generate_robokassa_payment_url(
    merchant_login: str,
    merchant_password_1: str,
    cost: decimal.Decimal,
    number: int,
    description: str,
    expiration_date: datetime | None = None,
    recurring: bool = False
) -> str:
    signature = calculate_signature(
        merchant_login,
        cost,
        number,
        merchant_password_1
    )

    data = {
        'MerchantLogin': merchant_login,
        'OutSum': cost,
        'InvId': number,
        'Description': description,
        'SignatureValue': signature,
        'IsTest': 0
    }
    if recurring:
        data['Recurring'] = 'true'
    if expiration_date:
        msk_expiration = expiration_date.astimezone(timezone(timedelta(hours=3)))
        data['ExpirationDate'] = msk_expiration.strftime('%Y-%m-%dT%H:%M')

    return f"https://auth.robokassa.ru/Merchant/Index.aspx?{parse.urlencode(data)}"


def is_robokassa_invoice_expired(payment: RobokassaPayment, now: datetime) -> bool:
    expires_at = payment.expires_at or (payment.created_at + ROBOKASSA_INVOICE_LIFETIME)
    return expires_at <= now


async def get_payment_context(session, payment: RobokassaPayment) -> tuple[str, str]:
    user = await session.get(User, payment.user_id)
    plan = await session.get(SubscriptionPlan, payment.plan_id)

    user_ref = f"[id={payment.user_id}]"
    if user:
        user_ref = user.first_name or str(payment.user_id)
        if user.username:
            user_ref += f" (@{user.username})"
        else:
            user_ref += f" (ID: {payment.user_id})"

    plan_name = plan.name if plan else f"plan_id={payment.plan_id}"
    return user_ref, plan_name

async def parse_robokassa_data(request: web.Request) -> dict:
    if request.method == 'POST':
        data = await request.post()
        return dict(data)
    else:
        return dict(request.query)

def check_signature_result(
    order_number: int,
    received_sum: str,
    received_signature: str,
    password: str
) -> bool:
    signature = calculate_signature(received_sum, order_number, password)
    if signature.lower() == received_signature.lower():
        return True
    return False


async def handle_yookassa_webhook(request: web.Request):
    bot = request.app['bot']
    data = await request.json()
    fsm_storage = request.app['fsm_storage']

    try:
        event = data.get('event')
        payment_object = data.get('object') or {}
        payment_id = payment_object.get('id')
        if not payment_id:
            return web.Response(status=400)

        async with async_session_maker() as session:
            config = await session.get(SubscriptionConfig, 1)
            if not config or not config.yookassa_shop_id or not config.yookassa_secret_key:
                return web.Response(status=503)

        auth = aiohttp.BasicAuth(config.yookassa_shop_id, config.yookassa_secret_key)
        api_timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(auth=auth, timeout=api_timeout) as http_session:
            async with http_session.get(f"https://api.yookassa.ru/v3/payments/{payment_id}") as response:
                if response.status != 200:
                    log.error(f"YooKassa verify failed for payment {payment_id}: HTTP {response.status}")
                    return web.Response(status=500)
                payment_object = await response.json()

        payment_status = payment_object.get('status')
        metadata = payment_object.get('metadata', {})
        user_id_str = metadata.get('user_id')
        plan_id_raw = metadata.get('plan_id')
        is_recurring_payment = str(metadata.get('recurring', '')).lower() == 'true'

        if event == 'payment.canceled' or payment_status == 'canceled':
            user_ref_log = None
            plan_name_log = None
            now = datetime.utcnow()
            attempt_num = None
            next_retry_at = None
            auto_renewal_disabled = False
            user_end_date = None
            async with async_session_maker() as session:
                existing = await session.scalar(
                    select(YookassaPayment).where(YookassaPayment.payment_id == payment_id).with_for_update()
                )
                previous_status = existing.status if existing else None
                if existing and existing.status == 'canceled' and existing.processed_at:
                    return web.Response(status=200)
                if existing:
                    existing.status = 'canceled'
                    existing.processed_at = now
                else:
                    session.add(YookassaPayment(
                        payment_id=payment_id,
                        user_id=int(user_id_str) if user_id_str else None,
                        plan_id=int(plan_id_raw) if plan_id_raw else None,
                        amount=float(payment_object.get('amount', {}).get('value', 0)),
                        status='canceled',
                        payment_method_id=(payment_object.get('payment_method') or {}).get('id'),
                        is_recurring=is_recurring_payment,
                        processed_at=now
                    ))
                    previous_status = None
                if is_recurring_payment and user_id_str:
                    user_sub = await session.scalar(
                        select(UserSubscription).where(UserSubscription.user_id == int(user_id_str)).with_for_update()
                    )
                    if user_sub and user_sub.payment_provider == 'Yookassa' and previous_status in (None, 'pending'):
                        user_sub.payment_attempt_count += 1
                        if not user_sub.last_payment_attempt:
                            user_sub.last_payment_attempt = now
                        attempt_num = user_sub.payment_attempt_count
                        user_end_date = user_sub.end_date
                        next_retry_at = get_next_retry_at(user_sub.payment_attempt_count, user_sub.last_payment_attempt)
                        if user_sub.payment_attempt_count >= 3:
                            user_sub.auto_renewal = False
                            auto_renewal_disabled = True
                if user_id_str:
                    user = await session.get(User, int(user_id_str))
                    if user:
                        user_ref_log = user.first_name or str(user.id)
                        if user.username:
                            user_ref_log += f" (@{user.username})"
                        else:
                            user_ref_log += f" (ID: {user.id})"
                if plan_id_raw:
                    plan = await session.get(SubscriptionPlan, int(plan_id_raw))
                    if plan:
                        plan_name_log = plan.name
                config = await session.get(SubscriptionConfig, 1)
                await session.commit()
            if user_id_str and plan_id_raw:
                user_part = user_ref_log or f"[id={user_id_str}]"
                plan_part = plan_name_log or f"plan_id={plan_id_raw}"
                plog.info(f"ОПЛАТА_ПРЕРВАНА | Yookassa | {user_part} | {plan_part} | PayId={payment_id}")
                uid_int = int(user_id_str)
                if is_recurring_payment and attempt_num:
                    try:
                        if auto_renewal_disabled:
                            end_date_text = user_end_date.astimezone(timezone(timedelta(hours=3))).strftime('%d.%m.%Y %H:%M МСК') if user_end_date else "неизвестной даты"
                            await send_msg_universal(
                                bot,
                                uid_int,
                                f"Не удалось списать средства (ЮKassa) после 3 попыток. Автопродление отключено.\n\n"
                                f"Текущая подписка активна до {end_date_text}."
                            )
                        elif next_retry_at:
                            next_retry_text = next_retry_at.astimezone(timezone(timedelta(hours=3))).strftime('%d.%m %H:%M МСК')
                            retry_text = (
                                f"Не удалось списать средства (ЮKassa). Повторим попытку {next_retry_text}."
                                if attempt_num == 1
                                else f"Не удалось списать средства (ЮKassa). Последняя попытка — {next_retry_text}."
                            )
                            await send_msg_universal(bot, uid_int, retry_text)
                        else:
                            await send_msg_universal(bot, uid_int, "Не удалось списать средства (ЮKassa).")
                    except Exception:
                        pass
                if config and config.notifications_enabled:
                    for admin_id in await get_all_admin_ids():
                        try:
                            admin_text = f"⏹ Платёж прерван (YooKassa)\nПользователь: {user_part}\nТариф: {plan_part}\nPayId: {payment_id}"
                            if is_recurring_payment and attempt_num:
                                admin_text += f"\nПопытка: {attempt_num}/3"
                                if auto_renewal_disabled:
                                    admin_text += "\nАвтопродление отключено."
                            await bot.send_message(admin_id, admin_text)
                        except Exception:
                            pass
            else:
                plog.info(f"ОПЛАТА_ПРЕРВАНА | Yookassa | PayId={payment_id}")
            return web.Response(status=200)

        if payment_status != 'succeeded':
            log.info(f"YooKassa webhook ignored: payment_id={payment_id}, status={payment_status}, event={event}")
            return web.Response(status=200)

        if not user_id_str or not plan_id_raw:
            log.info(f"YooKassa webhook succeeded without metadata: payment_id={payment_id}")
            return web.Response(status=200)

        user_id = int(user_id_str)
        plan_id = int(plan_id_raw)
        payment_method = payment_object.get('payment_method') or {}
        payment_method_id = payment_method.get('id')
        payment_method_saved = bool(payment_method.get('saved'))
        user_fsm_key = f"{bot.id}:{user_id}:{user_id}"
        user_state = FSMContext(storage=fsm_storage, key=user_fsm_key)
        await user_state.clear()

        plan_name_for_notif = "Неизвестный тариф"
        plan_price_for_notif = float(payment_object.get('amount', {}).get('value', 0))
        user_display = f"ID: {user_id}"
        now = datetime.utcnow()

        async with async_session_maker() as session:
            referrer_bonus_user_id = None
            referrer_bonus_days = 0
            yk_payment = await session.scalar(
                select(YookassaPayment).where(YookassaPayment.payment_id == payment_id).with_for_update()
            )
            if yk_payment and yk_payment.status == 'completed' and yk_payment.processed_at:
                plog.info(f"WEBHOOK_ДУБЛЬ | Yookassa | payment_id={payment_id} | status=completed")
                return web.Response(status=200)

            if not yk_payment:
                yk_payment = YookassaPayment(
                    payment_id=payment_id,
                    user_id=user_id,
                    plan_id=plan_id,
                    amount=plan_price_for_notif,
                    status='pending',
                    payment_method_id=payment_method_id,
                    is_recurring=is_recurring_payment
                )
                session.add(yk_payment)

            plan = await session.get(SubscriptionPlan, plan_id)
            if not plan:
                return web.Response(status=400)

            plan_name_for_notif = plan.name
            if plan.is_trial:
                new_trial_record = TrialUsageHistory(
                    user_id=user_id,
                    plan_id=plan_id,
                    used_at=now
                )
                session.add(new_trial_record)

            duration_value = plan.duration_value
            duration_unit = plan.duration_unit
            user_sub = await session.scalar(
                select(UserSubscription).where(UserSubscription.user_id == user_id).with_for_update()
            )

            if user_sub and user_sub.end_date > now:
                start_date = user_sub.start_date
                base_end_date = user_sub.end_date
            else:
                start_date = now
                base_end_date = now

            if duration_unit == 'months':
                end_date = base_end_date + relativedelta(months=duration_value)
            else:
                end_date = base_end_date + timedelta(days=duration_value)

            plan_allows_renewal = getattr(plan, 'allow_auto_renewal', True)
            effective_payment_method_id = payment_method_id if (plan_allows_renewal and payment_method_saved) else None

            if user_sub:
                user_sub.plan_id = plan_id
                user_sub.start_date = start_date
                user_sub.end_date = end_date
                user_sub.payment_provider = 'Yookassa'
                user_sub.payment_attempt_count = 0
                user_sub.last_payment_attempt = None
                user_sub.pending_robokassa_invoice_id = None
                if effective_payment_method_id:
                    user_sub.payment_method_id = effective_payment_method_id
                user_sub.auto_renewal = bool(user_sub.payment_method_id) and plan_allows_renewal
            else:
                session.add(UserSubscription(
                    user_id=user_id,
                    plan_id=plan_id,
                    start_date=start_date,
                    end_date=end_date,
                    auto_renewal=bool(effective_payment_method_id) and plan_allows_renewal,
                    payment_provider='Yookassa',
                    payment_method_id=effective_payment_method_id,
                    payment_attempt_count=0,
                    last_payment_attempt=None,
                    discount_percent=0
                ))

            yk_payment.status = 'completed'
            yk_payment.user_id = user_id
            yk_payment.plan_id = plan_id
            yk_payment.amount = plan_price_for_notif
            yk_payment.payment_method_id = payment_method_id
            yk_payment.processed_at = now

            paying_user = await session.get(User, user_id)
            if paying_user and paying_user.referred_by:
                ref_config = await session.get(SubscriptionConfig, 1)
                if ref_config and ref_config.referral_enabled:
                    if ref_config.referral_pay_bonus_enabled and ref_config.referral_pay_bonus_days > 0:
                        already_paid = False
                        if ref_config.referral_pay_bonus_first_only:
                            prev_count = await session.scalar(
                                select(func.count()).select_from(ReferralPaymentLog)
                                .where(ReferralPaymentLog.referred_user_id == user_id)
                            ) or 0
                            already_paid = prev_count > 0
                        if not already_paid:
                            bonus_days = ref_config.referral_pay_bonus_days
                            referrer_sub = await session.scalar(
                                select(UserSubscription).where(
                                    UserSubscription.user_id == paying_user.referred_by
                                )
                            )
                            now_b = datetime.utcnow()
                            if referrer_sub and referrer_sub.end_date > now_b:
                                referrer_sub.end_date += timedelta(days=bonus_days)
                            elif referrer_sub:
                                referrer_sub.plan_id = None
                                referrer_sub.start_date = now_b
                                referrer_sub.end_date = now_b + timedelta(days=bonus_days)
                                referrer_sub.payment_provider = 'Trial Referral Pay Bonus'
                                referrer_sub.auto_renewal = False
                                referrer_sub.payment_attempt_count = 0
                            else:
                                session.add(UserSubscription(
                                    user_id=paying_user.referred_by,
                                    plan_id=None,
                                    start_date=now_b,
                                    end_date=now_b + timedelta(days=bonus_days),
                                    auto_renewal=False,
                                    payment_provider='Trial Referral Pay Bonus',
                                    payment_attempt_count=0,
                                    discount_percent=0
                                ))
                            referrer_bonus_user_id = paying_user.referred_by
                            referrer_bonus_days = bonus_days
                    session.add(ReferralPaymentLog(
                        referrer_id=paying_user.referred_by,
                        referred_user_id=user_id,
                        amount=plan_price_for_notif,
                    ))

            await session.commit()

            user = paying_user
            if user:
                user_display = f"{user.first_name}"
                if user.username:
                    user_display += f" (@{user.username})"
                else:
                    user_display += f" (ID: {user_id})"
                user_ref_log = user.first_name
                if user.username:
                    user_ref_log += f" (@{user.username})"
                user_ref_log += f" [id={user_id}]"
            else:
                user_ref_log = f"[id={user_id}]"

            plog.info(f"ОПЛАТА | Yookassa | {user_ref_log} | {plan_name_for_notif} | {plan_price_for_notif:.2f} руб | PayId={payment_id}")
            await send_msg_universal(bot, user_id, f"✅ Ваша подписка на тариф «{plan_name_for_notif}» успешно оформлена!")
            if referrer_bonus_user_id and referrer_bonus_days > 0:
                await send_msg_universal(
                    bot,
                    referrer_bonus_user_id,
                    f"💰 Ваш реферал оформил подписку! Вам начислено <b>{referrer_bonus_days} бонусных дн.</b>",
                    parse_mode="HTML"
                )

            config = await session.get(SubscriptionConfig, 1)
            if config and config.notifications_enabled:
                for admin_id in await get_all_admin_ids():
                    await send_msg_universal(
                        bot,
                        admin_id,
                        f"🔔 Новый платеж (YooKassa)!\n\nПользователь: {user_display}\nТариф: {plan_name_for_notif}\nСумма: {plan_price_for_notif:.2f} руб.\nPayId: {payment_id}"
                    )

        return web.Response(status=200)

    except Exception as e:
        log.error("Ошибка в обработке вебхука YooKassa: %s", e, exc_info=e)
        await notify_admins_about_error(
            bot,
            title="Сбой webhook YooKassa",
            provider="YooKassa",
            stage="handle_yookassa_webhook",
            details=str(e),
            exception=e,
            logger=log,
        )
        return web.Response(status=500)


def setup_webhooks(app: web.Application, bot: Bot, fsm_storage, prefix: str = ''):
    app['bot'] = bot
    app['fsm_storage'] = fsm_storage

    app.router.add_post(f'{prefix}/webhooks/yookassa', handle_yookassa_webhook)
    app.router.add_get(f'{prefix}/pay/robokassa/{{payment_id}}', handle_robokassa_invoice_redirect)
    app.router.add_get(f'{prefix}/webhooks/robokassa/result', handle_robokassa_result)
    app.router.add_post(f'{prefix}/webhooks/robokassa/result', handle_robokassa_result)
    app.router.add_get(f'{prefix}/webhooks/robokassa/success', handle_robokassa_success)
    app.router.add_post(f'{prefix}/webhooks/robokassa/success', handle_robokassa_success)
    app.router.add_get(f'{prefix}/webhooks/robokassa/fail', handle_robokassa_fail)
    app.router.add_post(f'{prefix}/webhooks/robokassa/fail', handle_robokassa_fail)


async def handle_robokassa_invoice_redirect(request: web.Request):
    try:
        payment_id = int(request.match_info['payment_id'])
    except (KeyError, ValueError):
        return web.Response(text="Некорректный идентификатор счёта.", status=400)

    token = request.query.get('token', '')
    now = datetime.utcnow()

    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        if not config or not config.robokassa_merchant_login or not config.robokassa_password_1:
            return web.Response(text="Платежная система временно недоступна.", status=503)

        payment = await session.get(RobokassaPayment, payment_id)
        if not payment:
            return web.Response(text="Счёт не найден.", status=404)

        expected_token = build_robokassa_invoice_access_token(
            payment.id, payment.user_id, config.robokassa_password_1
        )
        if token != expected_token:
            return web.Response(text="Недействительная ссылка на оплату.", status=403)

        if payment.status == 'completed':
            user_ref, plan_name = await get_payment_context(session, payment)
            plog.info(f"СЧЕТ_ОТКРЫТИЕ_ОТКЛОНЕНО | Robokassa | {user_ref} | {plan_name} | InvId={payment.id} | причина=completed")
            return web.Response(text="Этот счёт уже оплачен. Вернитесь в бот, если нужна новая оплата.", status=200)

        current_payment = payment
        while current_payment.replaced_by_payment_id:
            replacement = await session.get(RobokassaPayment, current_payment.replaced_by_payment_id)
            if not replacement:
                break
            current_payment = replacement
            if current_payment.status == 'completed':
                user_ref, plan_name = await get_payment_context(session, current_payment)
                plog.info(f"СЧЕТ_ОТКРЫТИЕ_ОТКЛОНЕНО | Robokassa | {user_ref} | {plan_name} | InvId={current_payment.id} | причина=completed")
                return web.Response(text="Этот счёт уже оплачен. Вернитесь в бот, если нужна новая оплата.", status=200)
            if current_payment.status == 'pending' and not is_robokassa_invoice_expired(current_payment, now):
                break

        if current_payment.status == 'pending' and is_robokassa_invoice_expired(current_payment, now):
            current_payment.status = 'expired'

        if current_payment.status == 'expired':
            old_payment_id = current_payment.id
            replacement = RobokassaPayment(
                user_id=current_payment.user_id,
                plan_id=current_payment.plan_id,
                promo_code=current_payment.promo_code,
                amount=current_payment.amount,
                expires_at=now + ROBOKASSA_INVOICE_LIFETIME
            )
            session.add(replacement)
            await session.flush()
            current_payment.replaced_by_payment_id = replacement.id
            current_payment.status = 'replaced'
            current_payment = replacement
            await session.commit()
            user_ref, plan_name = await get_payment_context(session, current_payment)
            expires_at_msk = current_payment.expires_at.astimezone(timezone(timedelta(hours=3))).strftime('%d.%m.%Y %H:%M МСК')
            plog.info(
                f"СЧЕТ_ОБНОВЛЕН | Robokassa | {user_ref} | {plan_name} | "
                f"InvId={old_payment_id} -> InvId={current_payment.id} | до {expires_at_msk}"
            )
        elif current_payment.status != 'pending':
            user_ref, plan_name = await get_payment_context(session, current_payment)
            plog.info(
                f"СЧЕТ_ОТКРЫТИЕ_ОТКЛОНЕНО | Robokassa | {user_ref} | {plan_name} | "
                f"InvId={current_payment.id} | причина={current_payment.status}"
            )
            return web.Response(text="Счёт недоступен. Вернитесь в бот и запросите новый.", status=409)
        else:
            if current_payment.expires_at is None:
                current_payment.expires_at = current_payment.created_at + ROBOKASSA_INVOICE_LIFETIME
                await session.commit()

        plan = await session.get(SubscriptionPlan, current_payment.plan_id)
        if not plan:
            return web.Response(text="Тариф не найден. Вернитесь в бот и оформите новый счёт.", status=404)

        description = ''.join(c for c in f"Оплата подписки на тариф «{plan.name}»" if ord(c) <= 0xFFFF)
        payment_url = generate_robokassa_payment_url(
            merchant_login=config.robokassa_merchant_login,
            merchant_password_1=config.robokassa_password_1,
            cost=decimal.Decimal(f"{current_payment.amount:.2f}"),
            number=current_payment.id,
            description=description,
            expiration_date=current_payment.expires_at,
            recurring=getattr(plan, 'allow_auto_renewal', True)
        )
        user_ref, plan_name = await get_payment_context(session, current_payment)
        expires_at_msk = current_payment.expires_at.astimezone(timezone(timedelta(hours=3))).strftime('%d.%m.%Y %H:%M МСК')
        plog.info(f"СЧЕТ_ОТКРЫТ | Robokassa | {user_ref} | {plan_name} | InvId={current_payment.id} | до {expires_at_msk}")
        return web.HTTPFound(payment_url)


async def handle_robokassa_result(request: web.Request):
    bot = request.app['bot']
    fsm_storage = request.app['fsm_storage']
    data = await parse_robokassa_data(request)
    MSK = timezone(timedelta(hours=3))

    normalized_data = {k.lower(): v for k, v in data.items()}

    try:
        cost = normalized_data['outsum']
        inv_id = int(normalized_data['invid'])
        signature = normalized_data['signaturevalue']
    except KeyError as e:
        print(f"Robokassa ResultURL Error: Missing parameter {e} in {data}")
        return web.Response(text="bad sign", status=400)

    try:
        config = None
        plan_name_for_notif = "Неизвестный тариф"
        payment_amount_for_notif = 0.0
        payment_user_id = 0
        user_display = ""
        now = datetime.utcnow()

        async with async_session_maker() as session:
            config = await session.get(SubscriptionConfig, 1)
            if not config or not config.robokassa_password_2:
                print("Robokassa ResultURL Error: Password 2 not set.")
                return web.Response(text="bad sign", status=400)

            if not check_signature_result(inv_id, cost, signature, config.robokassa_password_2):
                print("Robokassa ResultURL Error: Invalid signature.")
                return web.Response(text="bad sign", status=400)

            payment = await session.scalar(
                select(RobokassaPayment).where(RobokassaPayment.id == inv_id).with_for_update()
            )
            if not payment:
                print(f"Robokassa ResultURL Error: Payment with InvId {inv_id} not found.")
                return web.Response(text="bad sign", status=404)

            if payment.status == 'completed':
                print(f"Robokassa ResultURL Info: Payment {inv_id} already processed.")
                return web.Response(text=f"OK{inv_id}")

            payment_amount_for_notif = payment.amount
            payment_user_id = payment.user_id
            payment.status = 'completed'

            user_fsm_key = f"{bot.id}:{payment_user_id}:{payment_user_id}"
            user_state = FSMContext(storage=fsm_storage, key=user_fsm_key)
            await user_state.clear()

            user_sub = await session.scalar(
                select(UserSubscription).where(UserSubscription.user_id == payment.user_id).with_for_update()
            )

            plan = await session.get(SubscriptionPlan, payment.plan_id)
            if not plan:
                print(f"Robokassa ResultURL Error: Plan {payment.plan_id} not found.")
                return web.Response(text="bad sign", status=400)

            if plan.is_trial:
                new_trial_record = TrialUsageHistory(
                    user_id=payment.user_id,
                    plan_id=payment.plan_id,
                    used_at=now
                )
                session.add(new_trial_record)

            plan_name_for_notif = plan.name

            user = await session.get(User, payment.user_id)
            user_display = f"ID: {payment.user_id}"
            if user:
                user_display = f"{user.first_name}"
                if user.username:
                    user_display += f" (@{user.username})"
                else:
                    user_display += f" (ID: {payment.user_id})"

            duration_value = plan.duration_value
            duration_unit = plan.duration_unit

            is_renewal = user_sub is not None and user_sub.pending_robokassa_invoice_id == inv_id

            if user_sub and user_sub.end_date > now:
                start_date = user_sub.start_date
                base_end_date = user_sub.end_date
            else:
                start_date = now
                base_end_date = now

            if duration_unit == 'months':
                end_date = base_end_date + relativedelta(months=duration_value)
            else:
                end_date = base_end_date + timedelta(days=duration_value)

            plan_allows_renewal = getattr(plan, 'allow_auto_renewal', True)
            effective_payment_method_id = str(inv_id) if plan_allows_renewal else None

            if user_sub:
                user_sub.plan_id = payment.plan_id
                user_sub.start_date = start_date
                user_sub.end_date = end_date
                user_sub.auto_renewal = plan_allows_renewal
                user_sub.payment_provider = 'Robokassa'
                # Для Robokassa в payment_method_id должен оставаться корневой InvId
                # цепочки рекуррентов. На renewal не перепривязываем его к дочернему
                # invoice, иначе следующий PreviousInvoiceID станет неверным.
                if effective_payment_method_id and (not is_renewal or not user_sub.payment_method_id):
                    user_sub.payment_method_id = effective_payment_method_id
                user_sub.pending_robokassa_invoice_id = None
                user_sub.payment_attempt_count = 0
                user_sub.last_payment_attempt = None
            else:
                new_sub = UserSubscription(
                    user_id=payment.user_id,
                    plan_id=payment.plan_id,
                    start_date=start_date,
                    end_date=end_date,
                    auto_renewal=plan_allows_renewal,
                    payment_provider='Robokassa',
                    payment_method_id=effective_payment_method_id,
                    pending_robokassa_invoice_id=None,
                    payment_attempt_count=0,
                    last_payment_attempt=None,
                    discount_percent=0
                )
                session.add(new_sub)

            await session.commit()

            # Referral: log payment and optionally give bonus days to referrer
            paying_user_rk = await session.get(User, payment_user_id)
            if paying_user_rk and paying_user_rk.referred_by:
                ref_config_rk = await session.get(SubscriptionConfig, 1)
                if ref_config_rk and ref_config_rk.referral_enabled:
                    if ref_config_rk.referral_pay_bonus_enabled and ref_config_rk.referral_pay_bonus_days > 0:
                        already_paid_rk = False
                        if ref_config_rk.referral_pay_bonus_first_only:
                            prev_count_rk = await session.scalar(
                                select(func.count()).select_from(ReferralPaymentLog)
                                .where(ReferralPaymentLog.referred_user_id == payment_user_id)
                            ) or 0
                            already_paid_rk = prev_count_rk > 0
                        if not already_paid_rk:
                            bonus_days_rk = ref_config_rk.referral_pay_bonus_days
                            referrer_sub_rk = await session.scalar(
                                select(UserSubscription).where(
                                    UserSubscription.user_id == paying_user_rk.referred_by
                                )
                            )
                            now_rk = datetime.utcnow()
                            if referrer_sub_rk and referrer_sub_rk.end_date > now_rk:
                                referrer_sub_rk.end_date += timedelta(days=bonus_days_rk)
                            elif referrer_sub_rk:
                                referrer_sub_rk.plan_id = None
                                referrer_sub_rk.start_date = now_rk
                                referrer_sub_rk.end_date = now_rk + timedelta(days=bonus_days_rk)
                                referrer_sub_rk.payment_provider = 'Trial Referral Pay Bonus'
                                referrer_sub_rk.auto_renewal = False
                                referrer_sub_rk.payment_attempt_count = 0
                            else:
                                session.add(UserSubscription(
                                    user_id=paying_user_rk.referred_by,
                                    plan_id=None,
                                    start_date=now_rk,
                                    end_date=now_rk + timedelta(days=bonus_days_rk),
                                    auto_renewal=False,
                                    payment_provider='Trial Referral Pay Bonus',
                                    payment_attempt_count=0,
                                    discount_percent=0
                                ))
                            referrer_bonus_user_id_rk = paying_user_rk.referred_by
                            referrer_bonus_days_rk = bonus_days_rk
                    session.add(ReferralPaymentLog(
                        referrer_id=paying_user_rk.referred_by,
                        referred_user_id=payment_user_id,
                        amount=payment_amount_for_notif,
                    ))
                    await session.commit()
                    if ref_config_rk.referral_pay_bonus_enabled and ref_config_rk.referral_pay_bonus_days > 0:
                        if not already_paid_rk:
                            await send_msg_universal(
                                bot,
                                referrer_bonus_user_id_rk,
                                f"💰 Ваш реферал оформил подписку! Вам начислено <b>{referrer_bonus_days_rk} бонусных дн.</b>",
                                parse_mode="HTML"
                            )

        end_date_msk = end_date.astimezone(MSK).strftime('%d.%m.%Y %H:%M')

        await send_msg_universal(
            bot,
            payment_user_id,
            f"Мы получили оплату {payment_amount_for_notif:.2f} руб по вашему тарифу «{plan_name_for_notif}».\n"
            f"Действие тарифа продлено до {end_date_msk} МСК.\n\n"
            f"Благодарим, что продолжаете пользоваться ботом!\n"
            f"Вы всегда можете направить нам свои пожелания, предложения по его работе."
        )

        if is_renewal:
            plog.info(f"ПРОДЛЕНИЕ | Robokassa | {user_display} | {plan_name_for_notif} | {payment_amount_for_notif:.2f} руб | InvId={inv_id}")
        else:
            plog.info(f"ОПЛАТА | Robokassa | {user_display} | {plan_name_for_notif} | {payment_amount_for_notif:.2f} руб | InvId={inv_id}")

        if config and config.notifications_enabled:
            for admin_id in await get_all_admin_ids():
                await send_msg_universal(
                    bot,
                    admin_id,
                    f"🔔 Новый платеж (Robokassa)!\n\nПользователь: {user_display}\nТариф: {plan_name_for_notif}\nСумма: {payment_amount_for_notif} руб."
                )

        return web.Response(text=f"OK{inv_id}")

    except Exception as e:
        log.error("Ошибка в обработке вебхука Robokassa Result: %s", e, exc_info=e)
        await notify_admins_about_error(
            bot,
            title="Сбой webhook Robokassa",
            provider="Robokassa",
            stage="handle_robokassa_result",
            details=str(e),
            exception=e,
            logger=log,
        )
        return web.Response(status=500)


async def handle_robokassa_success(request: web.Request):
    bot = request.app['bot']
    data = await parse_robokassa_data(request)

    normalized_data = {k.lower(): v for k, v in data.items()}

    try:
        cost = normalized_data['outsum']
        inv_id = int(normalized_data['invid'])
        signature = normalized_data['signaturevalue']

        async with async_session_maker() as session:
            config = await session.get(SubscriptionConfig, 1)
            if not config or not config.robokassa_password_1:
                return web.Response(text="bad sign", status=400)

            if not check_signature_result(inv_id, cost, signature, config.robokassa_password_1):
                return web.Response(text="bad sign", status=400)

        bot_info = await bot.get_me()
        return web.HTTPFound(f"https://t.me/{bot_info.username}")

    except KeyError as e:
        log.error("Robokassa SuccessURL missing parameter %s in %s", e, data)
        return web.Response(text="bad sign", status=400)
    except Exception as e:
        log.error("Ошибка в обработке вебхука Robokassa Success: %s", e, exc_info=e)
        await notify_admins_about_error(
            bot,
            title="Сбой webhook Robokassa",
            provider="Robokassa",
            stage="handle_robokassa_success",
            details=str(e),
            exception=e,
            logger=log,
        )
        return web.Response(status=500)


async def handle_robokassa_fail(request: web.Request):
    bot = request.app['bot']
    data = await parse_robokassa_data(request)
    normalized_data = {k.lower(): v for k, v in data.items()}
    try:
        inv_id_raw = normalized_data.get('invid')
        if inv_id_raw:
            inv_id = int(inv_id_raw)
            async with async_session_maker() as session:
                payment = await session.get(RobokassaPayment, inv_id)
                if payment:
                    user_ref, plan_name = await get_payment_context(session, payment)
                    plog.info(f"ОПЛАТА_ПРЕРВАНА | Robokassa | {user_ref} | {plan_name} | InvId={inv_id}")
    except Exception:
        pass
    bot_info = await bot.get_me()
    return web.HTTPFound(f"https://t.me/{bot_info.username}")
