import asyncio
import logging
from datetime import datetime, timedelta, timezone
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from aiogram import Bot
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from database import (async_session_maker, UserSubscription, SubscriptionPlan, SubscriptionConfig, User,
                      RobokassaPayment, PromoCode, YookassaPayment, AIConfig, get_all_admin_ids)
from ai_integration import get_kie_remaining_credits, AIServiceError
from dateutil.relativedelta import relativedelta
from yookassa import Configuration, Payment
from yookassa.domain.exceptions import (BadRequestError, ForbiddenError, InternalServerError,
                                        TooManyRequestsError, UnauthorizedError)
from uuid import uuid4
import html
import os
import json
from config import OWNER_IDS
import keyboards as kb
import aiohttp
import hashlib
import decimal
import math
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
from birthday_mailings import process_birthday_mailings
from subscription_notifications import should_send_upcoming_charge_notification
from subscription_retry_policy import can_retry_now, get_next_retry_at
from error_reporting import notify_admins_about_error

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
plog = logging.getLogger("payment_events")
ROBOKASSA_PENDING_TIMEOUT = timedelta(hours=3)
EXPIRATION_DEDUP_WINDOW = timedelta(hours=2)
UTC = timezone.utc
MSK = timezone(timedelta(hours=3))


def _sanitize_log_value(value, limit: int = 2000) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if len(text) > limit:
        return f"{text[:limit - 3]}..."
    return text


def _plog_robokassa_tech(event: str, **fields):
    parts = [event, "Robokassa"]
    parts.extend(
        f"{key}={_sanitize_log_value(value)}"
        for key, value in fields.items()
        if value is not None
    )
    plog.info(" | ".join(parts))


def _plog_yookassa_tech(event: str, **fields):
    parts = [event, "Yookassa"]
    parts.extend(
        f"{key}={_sanitize_log_value(value)}"
        for key, value in fields.items()
        if value is not None
    )
    plog.info(" | ".join(parts))


def _encode_log_params(fields: dict[str, object]) -> str:
    return urlencode([(key, str(value)) for key, value in fields.items() if value is not None])


def _normalize_log_json_value(value):
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_log_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_log_json_value(item) for item in value]
    return value


def _encode_log_json(payload: dict[str, object]) -> str:
    return json.dumps(
        _normalize_log_json_value(payload),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _serialize_yookassa_payment(payment) -> dict[str, object]:
    amount = getattr(payment, "amount", None)
    confirmation = getattr(payment, "confirmation", None)
    payment_method = getattr(payment, "payment_method", None)
    metadata = getattr(payment, "metadata", None)
    return {
        "id": getattr(payment, "id", None),
        "status": getattr(payment, "status", None),
        "paid": getattr(payment, "paid", None),
        "amount": {
            "value": getattr(amount, "value", None),
            "currency": getattr(amount, "currency", None),
        } if amount else None,
        "description": getattr(payment, "description", None),
        "confirmation": {
            "type": getattr(confirmation, "type", None),
            "confirmation_url": getattr(confirmation, "confirmation_url", None),
        } if confirmation else None,
        "payment_method": {
            "id": getattr(payment_method, "id", None),
            "type": getattr(payment_method, "type", None),
            "saved": getattr(payment_method, "saved", None),
        } if payment_method else None,
        "metadata": metadata if isinstance(metadata, dict) else None,
    }


def _payment_log_path() -> str:
    app_port = os.environ.get("APP_PORT", "8080")
    return os.path.join(os.path.dirname(__file__), "logs", f"payment_events_{app_port}.log")


def _to_msk(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(MSK)


def _format_msk(dt: datetime, fmt: str = "%d.%m.%Y %H:%M МСК") -> str:
    return _to_msk(dt).strftime(fmt)


def _was_recent_notification_logged(recipient_id: int, key: str, now: datetime, window: timedelta) -> bool:
    log_path = _payment_log_path()
    if not os.path.exists(log_path):
        return False

    event_suffix = f"NOTIFY_SENT | recipient_id={recipient_id} | key={key}"

    try:
        with open(log_path, "r", encoding="utf-8") as log_file:
            recent_lines = log_file.readlines()[-500:]
    except OSError:
        return False

    now_local = _to_msk(now).replace(tzinfo=None)

    for line in reversed(recent_lines):
        line = line.strip()
        if event_suffix not in line:
            continue

        timestamp_raw = line.split(" | ", 1)[0]
        try:
            logged_at = datetime.strptime(timestamp_raw, "%d.%m.%Y %H:%M:%S")
        except ValueError:
            continue

        if now_local - logged_at <= window:
            return True

    return False


async def _send_deduplicated_notification(
        bot: Bot,
        recipient_id: int,
        text: str,
        key: str,
        now: datetime,
        *,
        reply_markup=None,
        window: timedelta = EXPIRATION_DEDUP_WINDOW,
) -> bool:
    if _was_recent_notification_logged(recipient_id, key, now, window):
        return False
    await bot.send_message(recipient_id, text, reply_markup=reply_markup)
    plog.info(f"NOTIFY_SENT | recipient_id={recipient_id} | key={key}")
    return True


def patch_bot_send_message(bot: Bot):
    if hasattr(bot, "_original_send_message"):
        return

    bot._original_send_message = bot.send_message

    async def patched_send_message(chat_id, text, *args, **kwargs):
        # Determine if it's a MAX user
        try:
            chat_id_int = int(chat_id)
        except (ValueError, TypeError):
            chat_id_int = 0

        if chat_id_int >= 100_000_000_000:
            # MAX notification
            import os
            import re
            token = os.environ.get("MAX_BOT_TOKEN")
            if not token:
                logging.getLogger("scheduler").warning(f"Cannot send MAX message to {chat_id_int}: MAX_BOT_TOKEN not configured in env")
                return None
            base_url = os.environ.get("MAX_API_BASE", "https://platform-api.max.ru")
            
            # Map reply_markup to attachments if present
            attachments = None
            reply_markup = kwargs.get("reply_markup")
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
                    max_api_user_id = chat_id_int - MAX_ID_OFFSET
                    # Clean HTML tags
                    clean_text = re.sub(r'<[^>]+>', '', text)
                    await client.send_message(user_id=max_api_user_id, text=clean_text, attachments=attachments)
                    return None
            except Exception as e:
                logging.getLogger("scheduler").error(f"Failed to send MAX message to {chat_id_int}: {e}", exc_info=e)
                return None
        else:
            return await bot._original_send_message(chat_id, text, *args, **kwargs)

    bot.send_message = patched_send_message


async def check_kie_credit_balance(bot: Bot):
    patch_bot_send_message(bot)
    try:
        async with async_session_maker() as session:
            config = await session.get(AIConfig, 1)
            if not config:
                return

            api_key = getattr(config, "kie_api_key", None)
            threshold = float(getattr(config, "kie_credit_alert_threshold", 0) or 0)
            base_url = (getattr(config, "kie_base_url", None) or "https://api.kie.ai").rstrip("/")

            if not api_key or threshold <= 0:
                if config.kie_credit_alert_sent:
                    config.kie_credit_alert_sent = False
                    await session.commit()
                return

            remaining_credits = await get_kie_remaining_credits(api_key, base_url)

            if remaining_credits < threshold:
                if not config.kie_credit_alert_sent:
                    config.kie_credit_alert_sent = True
                    await session.commit()
                    text = (
                        "⚠️ <b>Низкий баланс KIE</b>\n\n"
                        f"Остаток кредитов: <b>{remaining_credits}</b>\n"
                        f"Порог уведомления: <b>{threshold}</b>\n\n"
                        "Пополните баланс KIE, чтобы генерации и мультимодальные запросы не остановились."
                    )
                    for admin_id in await get_all_admin_ids():
                        try:
                            await bot.send_message(admin_id, text, parse_mode="HTML")
                        except Exception as exc:
                            log.error("Failed to send KIE balance alert admin_id=%s error=%s", admin_id, exc)
            else:
                if config.kie_credit_alert_sent:
                    config.kie_credit_alert_sent = False
                    await session.commit()
    except AIServiceError as exc:
        log.error("KIE credit balance check failed: %s", exc, exc_info=exc)
        await notify_admins_about_error(
            bot,
            title="Сбой проверки баланса KIE",
            provider="KIE",
            stage="check_credit_balance",
            details=str(exc),
            exception=exc,
            logger=log,
        )
    except Exception as exc:
        log.error("Unexpected KIE credit balance check error: %s", exc, exc_info=exc)
        await notify_admins_about_error(
            bot,
            title="Сбой проверки баланса KIE",
            provider="KIE",
            stage="check_credit_balance",
            details=str(exc),
            exception=exc,
            logger=log,
        )


async def disable_auto_renewal_after_failed_attempts(
        session,
        bot: Bot,
        sub: UserSubscription,
        user_ref: str,
        plan_name: str,
        subscribe_kb: InlineKeyboardMarkup,
        config: SubscriptionConfig,
        all_admin_ids: set[int]
):
    plog.warning(f"АВТОПРОДЛ_ОТКЛ | {user_ref} | причина: 3 попытки | {plan_name}")
    sub.auto_renewal = False
    sub.pending_robokassa_invoice_id = None
    sub.last_payment_attempt = None
    await session.commit()

    try:
        await bot.send_message(
            sub.user_id,
            "Ваша подписка истекла. Не удалось списать средства после 3 попыток — автопродление отключено.\n\nПродлите подписку вручную в меню.",
            reply_markup=subscribe_kb
        )
    except Exception:
        pass

    if config and config.notifications_enabled:
        for admin_id in all_admin_ids:
            try:
                await bot.send_message(
                    admin_id,
                    f"🚫 Автопродление отключено системой\nПользователь: {user_ref}\nТариф: {plan_name}\n3 неудачных попытки списания"
                )
            except Exception:
                pass


def has_robokassa_pending_timed_out(sub: UserSubscription, now: datetime) -> bool:
    return bool(
        sub.last_payment_attempt
        and (now - sub.last_payment_attempt) > ROBOKASSA_PENDING_TIMEOUT
    )


async def process_recurring_payment(bot: Bot, sub: UserSubscription, plan: SubscriptionPlan,
                                    price_to_charge: float, config: SubscriptionConfig,
                                    attempt_started_at: datetime):
    log.info(f"Attempting recurring payment for user {sub.user_id}, sub {sub.id} for plan {plan.name} ({price_to_charge} RUB)")
    try:
        Configuration.account_id = config.yookassa_shop_id
        Configuration.secret_key = config.yookassa_secret_key

        idempotence_key = hashlib.md5(
            f"yk-recurring:{sub.user_id}:{sub.id}:{plan.id}:{price_to_charge:.2f}:{sub.payment_attempt_count}:{attempt_started_at.isoformat()}".encode()
        ).hexdigest()

        payload = {
            "amount": {
                "value": f"{price_to_charge:.2f}",
                "currency": "RUB"
            },
            "capture": True,
            "payment_method_id": sub.payment_method_id,
            "description": f"Автопродление подписки на тариф «{plan.name}»",
            "metadata": {
                "user_id": str(sub.user_id),
                "plan_id": str(plan.id),
                "recurring": "true",
            }
        }
        _plog_yookassa_tech(
            "TECH_RECURRING_REQUEST",
            Method="POST",
            Endpoint="/v3/payments",
            UserId=sub.user_id,
            SubscriptionId=sub.id,
            IdempotenceKey=idempotence_key,
            Payload=_encode_log_json(payload),
        )

        payment = await asyncio.to_thread(Payment.create, payload, idempotence_key)
        _plog_yookassa_tech(
            "TECH_RECURRING_RESPONSE",
            PaymentId=payment.id,
            Status=payment.status,
            Body=_encode_log_json(_serialize_yookassa_payment(payment)),
        )

        if payment.status == 'succeeded':
            log.info(f"Successfully charged user {sub.user_id} for plan {plan.name}")
            return True, payment.id, payment.status
        if payment.status in ('pending', 'waiting_for_capture'):
            log.info(f"Recurring payment for user {sub.user_id} is pending: {payment.id}")
            return 'pending', payment.id, payment.status
        else:
            log.warning(f"Payment for user {sub.user_id} was created but status is {payment.status}")
            return False, payment.id, payment.status

    except BadRequestError as e:
        log.error(f"Failed to charge user {sub.user_id}. API BadRequestError: {e}")
        error_code = (e.content or {}).get('code') if isinstance(e.content, dict) else None
        _plog_yookassa_tech(
            "TECH_RECURRING_ERROR",
            ErrorClass=type(e).__name__,
            ErrorCode=error_code,
            Body=_encode_log_json(e.content) if isinstance(e.content, dict) else str(e),
        )
        if error_code == 'payment_method_not_found':
            return 'deactivate', None, None
        return 'integration_error', None, None
    except (ForbiddenError, InternalServerError, TooManyRequestsError, UnauthorizedError) as e:
        log.error(f"Failed to charge user {sub.user_id}. API Error: {e}")
        _plog_yookassa_tech(
            "TECH_RECURRING_ERROR",
            ErrorClass=type(e).__name__,
            Body=_encode_log_json(e.content) if isinstance(getattr(e, "content", None), dict) else str(e),
        )
        return 'provider_error', None, None
    except Exception as e:
        log.error(f"Unknown error during payment processing for user {sub.user_id}: {e}")
        _plog_yookassa_tech(
            "TECH_RECURRING_ERROR",
            ErrorClass=type(e).__name__,
            Body=str(e),
        )
        return 'provider_error', None, None


async def check_subscriptions(bot: Bot):
    patch_bot_send_message(bot)
    log.info("Running subscription check...")
    now = datetime.utcnow()

    try:
        await process_birthday_mailings(bot, now=now)
    except Exception as exc:
        log.error("Birthday mailing processing failed: %s", exc)
        await notify_admins_about_error(
            bot,
            title="Сбой обработки ДР-рассылок",
            stage="process_birthday_mailings",
            details=str(exc),
            exception=exc,
            logger=log,
        )

    two_hours_later = now + timedelta(hours=2)
    lookback_cutoff = now - timedelta(days=30)
    notification_threshold = now - timedelta(hours=1, minutes=15)

    subscribe_kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="💳 Оформить подписку", callback_data="show_subscription_info_from_chat")]
    ])

    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        if not config:
            return

        all_admin_ids = set(OWNER_IDS)
        try:
            admin_result = await session.execute(select(User.id).where(User.is_admin == True))
            db_admin_ids = {row[0] for row in admin_result}
            all_admin_ids.update(db_admin_ids)
        except Exception:
            pass

        stmt = select(UserSubscription).where(
            UserSubscription.end_date > lookback_cutoff
        ).options(
            selectinload(UserSubscription.plan).selectinload(SubscriptionPlan.upgrades_to_plan)
        )

        result = await session.execute(stmt)
        subscriptions = result.scalars().all()

        for sub in subscriptions:
            try:
                user = await session.get(User, sub.user_id, options=[
                    selectinload(User.promo_codes).selectinload(PromoCode.applicable_plans)])
                if not user:
                    continue

                user_ref = user.first_name or ""
                if user.username:
                    user_ref += f" (@{user.username})"
                user_ref += f" [id=<code>{sub.user_id}</code>]"

                message_text = None
                is_trial_promo = sub.plan_id is None and sub.payment_provider in ['Trial Promo', 'Trial Welcome']
                time_left = sub.end_date - now

                two_day_window_start = timedelta(days=2)
                two_day_window_end = timedelta(days=2, hours=1)
                one_day_window_start = timedelta(days=1)
                one_day_window_end = timedelta(days=1, hours=1)
                two_hour_window_start = timedelta(hours=2)
                two_hour_window_end = timedelta(hours=3)

                should_send = False
                reminder_bucket = None
                if two_day_window_start <= time_left < two_day_window_end:
                    should_send = True
                    reminder_bucket = "2d"
                elif one_day_window_start <= time_left < one_day_window_end:
                    should_send = True
                    reminder_bucket = "1d"
                elif two_hour_window_start <= time_left < two_hour_window_end:
                    should_send = True
                    reminder_bucket = "2h"

                date_str = _format_msk(sub.end_date)

                if should_send:
                    if is_trial_promo:
                        discount = sub.discount_percent
                        remaining_seconds = time_left.total_seconds()
                        if remaining_seconds > 86400:
                            d_val = int(remaining_seconds // 86400)
                            time_display = f"{d_val} дн."
                        elif remaining_seconds > 3600:
                            h_val = int(remaining_seconds // 3600)
                            time_display = f"{h_val} ч."
                        else:
                            time_display = "менее часа"

                        message_text = f"Ваш пробный период истекает {date_str} (через {time_display})."
                        if discount > 0:
                            message_text += f"\n\nОформите подписку, чтобы сохранить скидку {discount}%!"
                        else:
                            message_text += "\n\nОформите подписку для продолжения работы."

                    elif sub.auto_renewal and sub.plan:
                        plan_to_charge = sub.plan.upgrades_to_plan if (
                                sub.plan.is_trial and sub.plan.upgrades_to_plan) else sub.plan
                        if should_send_upcoming_charge_notification(sub.auto_renewal, sub.plan):
                            user_promos = user.promo_codes if user else []
                            current_discount = sub.discount_percent

                            best_promo = next((p for p in user_promos if not p.applies_to_all_plans and any(
                                ap.id == plan_to_charge.id for ap in p.applicable_plans)), None)
                            if best_promo and best_promo.discount_percent > current_discount:
                                current_discount = best_promo.discount_percent
                            elif not best_promo:
                                global_promo = next((p for p in user_promos if p.applies_to_all_plans), None)
                                if global_promo and global_promo.discount_percent > current_discount:
                                    current_discount = global_promo.discount_percent

                            price = plan_to_charge.price * (1 - current_discount / 100)
                            message_text = f"Напоминаем: {date_str} продление тарифа «{plan_to_charge.name}» на сумму {price:.2f} руб."

                if message_text:
                    try:
                        reply_markup = subscribe_kb if is_trial_promo else None
                        plan_marker = sub.plan.id if sub.plan else "trial"
                        reminder_kind = "trial_reminder" if is_trial_promo else "renewal_reminder"
                        reminder_key = f"{reminder_kind}:{sub.id}:{plan_marker}:{reminder_bucket}"
                        await _send_deduplicated_notification(
                            bot,
                            sub.user_id,
                            message_text,
                            reminder_key,
                            now,
                            reply_markup=reply_markup,
                            window=timedelta(hours=3),
                        )
                    except Exception:
                        pass

                if sub.end_date <= two_hours_later:
                    if is_trial_promo:
                        if sub.discount_percent > 0:
                            sub.discount_percent = 0
                            await session.commit()
                            if sub.end_date > notification_threshold:
                                try:
                                    await _send_deduplicated_notification(
                                        bot,
                                        sub.user_id,
                                        "Пробный период завершен. Выберите тариф для продолжения.",
                                        f"trial_finished:{sub.id}",
                                        now,
                                        reply_markup=subscribe_kb,
                                        window=timedelta(hours=6),
                                    )
                                except Exception:
                                    pass
                        continue

                    if not sub.auto_renewal:
                        if sub.end_date <= now and sub.end_date > notification_threshold:
                            plan_name_exp = sub.plan.name if sub.plan else (sub.payment_provider or "Trial")
                            if not _was_recent_notification_logged(
                                    sub.user_id,
                                    f"expired:{sub.id}:{plan_name_exp}",
                                    now,
                                    EXPIRATION_DEDUP_WINDOW,
                            ):
                                plog.info(f"ИСТЕЧЕНИЕ | {user_ref} | {plan_name_exp}")
                                try:
                                    await _send_deduplicated_notification(
                                        bot,
                                        sub.user_id,
                                        "Подписка истекла. Продлите её в меню.",
                                        f"expired:{sub.id}:{plan_name_exp}",
                                        now,
                                        reply_markup=subscribe_kb,
                                    )
                                except Exception:
                                    pass
                                if config and config.notifications_enabled:
                                    for admin_id in all_admin_ids:
                                        try:
                                            await bot.send_message(admin_id,
                                                                   f"⏰ Подписка истекла\nПользователь: {user_ref}\nТариф: {plan_name_exp}")
                                        except Exception:
                                            pass
                        continue

                    # Для Robokassa не инициируем новое автосписание до фактического истечения подписки.
                    # Иначе банк может показать пользователю ранний отказ, хотя доступ ещё активен.
                    if (
                        sub.payment_provider == 'Robokassa'
                        and sub.end_date > now
                        and not sub.pending_robokassa_invoice_id
                    ):
                        continue

                    # OpStateExt: проверяем статус pending-платежа Robokassa не дожидаясь 3ч таймаута
                    if (sub.payment_provider == 'Robokassa'
                            and sub.pending_robokassa_invoice_id
                            and config
                            and config.robokassa_password_2):
                        pending_inv_id = sub.pending_robokassa_invoice_id
                        op_state = await check_robokassa_op_state(config, pending_inv_id)

                        if op_state == 'success':
                            pending_payment_ok = await session.scalar(
                                select(RobokassaPayment).where(RobokassaPayment.id == pending_inv_id).with_for_update()
                            )
                            if pending_payment_ok and pending_payment_ok.status == 'completed':
                                sub.pending_robokassa_invoice_id = None
                                sub.payment_attempt_count = 0
                                sub.last_payment_attempt = None
                                await session.commit()
                                continue
                            ok_amount = float(pending_payment_ok.amount) if pending_payment_ok else 0.0
                            plan_ok = sub.plan
                            ok_plan_name = plan_ok.name if plan_ok else "Unknown"
                            if plan_ok:
                                ptc_ok = plan_ok.upgrades_to_plan if (plan_ok.is_trial and plan_ok.upgrades_to_plan) else plan_ok
                                add_days_ok = ptc_ok.duration_value if ptc_ok.duration_unit == 'days' else 0
                                add_months_ok = ptc_ok.duration_value if ptc_ok.duration_unit == 'months' else 0
                                sub.end_date = sub.end_date + relativedelta(months=add_months_ok) + timedelta(days=add_days_ok)
                                sub.plan_id = ptc_ok.id
                                ok_plan_name = ptc_ok.name
                            sub.payment_attempt_count = 0
                            sub.last_payment_attempt = None
                            sub.pending_robokassa_invoice_id = None
                            if pending_payment_ok:
                                pending_payment_ok.status = 'completed'
                            await session.commit()
                            plog.info(f"ПРОДЛЕНИЕ | Robokassa | {user_ref} | {ok_plan_name} | {ok_amount:.2f} руб | InvId={pending_inv_id} (OpState)")
                            try:
                                await _send_deduplicated_notification(
                                    bot,
                                    sub.user_id,
                                    f"✅ Подписка продлена до {_format_msk(sub.end_date, '%d.%m.%Y %H:%M')}.",
                                    f"rk_success:{sub.id}:{pending_inv_id}",
                                    now,
                                    window=timedelta(days=2),
                                )
                            except Exception:
                                pass
                            if config.notifications_enabled:
                                for admin_id in all_admin_ids:
                                    try:
                                        await bot.send_message(admin_id,
                                                               f"🔔 Автопродление (Robokassa)!\n\nПользователь: {user_ref}\nТариф: {ok_plan_name}\nСумма: {ok_amount:.2f} руб\nДо: {_format_msk(sub.end_date, '%d.%m.%Y %H:%M')}")
                                    except Exception:
                                        pass
                            continue

                        elif op_state == 'failed':
                            pending_payment_fail = await session.scalar(
                                select(RobokassaPayment).where(RobokassaPayment.id == pending_inv_id).with_for_update()
                            )
                            if pending_payment_fail and pending_payment_fail.status == 'completed':
                                sub.pending_robokassa_invoice_id = None
                                sub.payment_attempt_count = 0
                                sub.last_payment_attempt = None
                                await session.commit()
                                continue
                            attempt_num_op = sub.payment_attempt_count
                            plan_name_op = sub.plan.name if sub.plan else "Unknown"
                            plog.warning(f"ОШИБКА_СПИСАНИЯ | Robokassa | {user_ref} | попытка {attempt_num_op} | {plan_name_op}")
                            sub.pending_robokassa_invoice_id = None
                            if pending_payment_fail:
                                pending_payment_fail.status = 'failed'
                            if attempt_num_op >= 3:
                                await disable_auto_renewal_after_failed_attempts(
                                    session, bot, sub, user_ref, plan_name_op, subscribe_kb, config, all_admin_ids
                                )
                                continue
                            await session.commit()
                            next_retry_at = get_next_retry_at(sub.payment_attempt_count, sub.last_payment_attempt)
                            user_msg_op = "Не удалось провести автосписание (Robokassa)."
                            if attempt_num_op == 1 and next_retry_at:
                                next_retry_str = _format_msk(next_retry_at, '%d.%m %H:%M МСК')
                                user_msg_op = f"Не удалось провести автосписание (Robokassa). Повторим попытку {next_retry_str}."
                            elif attempt_num_op == 2 and next_retry_at:
                                next_retry_str = _format_msk(next_retry_at, '%d.%m %H:%M МСК')
                                user_msg_op = f"Не удалось провести автосписание (Robokassa). Последняя попытка — {next_retry_str}."
                            try:
                                await _send_deduplicated_notification(
                                    bot,
                                    sub.user_id,
                                    user_msg_op,
                                    f"rk_failed_op:{sub.id}:{attempt_num_op}",
                                    now,
                                    reply_markup=subscribe_kb,
                                    window=timedelta(days=1),
                                )
                            except Exception:
                                pass
                            if config.notifications_enabled:
                                for admin_id in all_admin_ids:
                                    try:
                                        await bot.send_message(admin_id,
                                                               f"⚠️ Ошибка автосписания [{attempt_num_op}/3]\nПользователь: {user_ref}\nТариф: {plan_name_op}\nПровайдер: Robokassa")
                                    except Exception:
                                        pass
                            continue

                        else:  # 'pending' или 'unknown' — 3ч hard fallback
                            if has_robokassa_pending_timed_out(sub, now):
                                pending_payment_timeout = await session.scalar(
                                    select(RobokassaPayment).where(RobokassaPayment.id == pending_inv_id).with_for_update()
                                )
                                if pending_payment_timeout and pending_payment_timeout.status == 'completed':
                                    sub.pending_robokassa_invoice_id = None
                                    sub.payment_attempt_count = 0
                                    sub.last_payment_attempt = None
                                    await session.commit()
                                    continue
                                attempt_num_to = sub.payment_attempt_count
                                plan_name_to = sub.plan.name if sub.plan else "Unknown"
                                sub.pending_robokassa_invoice_id = None
                                if pending_payment_timeout and pending_payment_timeout.status == 'pending':
                                    pending_payment_timeout.status = 'timeout'
                                plog.warning(f"ОШИБКА_СПИСАНИЯ | Robokassa | {user_ref} | попытка {attempt_num_to} | {plan_name_to}")
                                if attempt_num_to >= 3:
                                    await disable_auto_renewal_after_failed_attempts(
                                        session, bot, sub, user_ref, plan_name_to, subscribe_kb, config, all_admin_ids
                                    )
                                    continue
                                await session.commit()
                                next_retry_at = get_next_retry_at(sub.payment_attempt_count, sub.last_payment_attempt)
                                user_msg_to = "Не удалось провести автосписание (Robokassa)."
                                if attempt_num_to == 1 and next_retry_at:
                                    next_retry_str = _format_msk(next_retry_at, '%d.%m %H:%M МСК')
                                    user_msg_to = f"Не удалось провести автосписание (Robokassa). Повторим попытку {next_retry_str}."
                                elif attempt_num_to == 2 and next_retry_at:
                                    next_retry_str = _format_msk(next_retry_at, '%d.%m %H:%M МСК')
                                    user_msg_to = f"Не удалось провести автосписание (Robokassa). Последняя попытка — {next_retry_str}."
                                try:
                                    await _send_deduplicated_notification(
                                        bot,
                                        sub.user_id,
                                        user_msg_to,
                                        f"rk_failed_timeout:{sub.id}:{attempt_num_to}",
                                        now,
                                        reply_markup=subscribe_kb,
                                        window=timedelta(days=1),
                                    )
                                except Exception:
                                    pass
                                if config.notifications_enabled:
                                    for admin_id in all_admin_ids:
                                        try:
                                            await bot.send_message(admin_id,
                                                                   f"⚠️ Ошибка автосписания [{attempt_num_to}/3]\nПользователь: {user_ref}\nТариф: {plan_name_to}\nПровайдер: Robokassa")
                                        except Exception:
                                            pass
                            continue  # ждём подтверждения или прошёл таймаут — пропускаем цикл

                    can_attempt = can_retry_now(
                        sub.payment_attempt_count,
                        sub.last_payment_attempt,
                        now,
                    )

                    if not can_attempt:
                        if sub.payment_attempt_count >= 3:
                            plan_name_3att = sub.plan.name if sub.plan else (sub.payment_provider or "Trial")
                            await disable_auto_renewal_after_failed_attempts(
                                session, bot, sub, user_ref, plan_name_3att, subscribe_kb, config, all_admin_ids
                            )
                        continue

                    if not sub.plan:
                        sub.auto_renewal = False
                        await session.commit()
                        continue

                    if not getattr(sub.plan, 'allow_auto_renewal', True):
                        sub.auto_renewal = False
                        await session.commit()
                        continue

                    plan_to_charge = sub.plan.upgrades_to_plan if (
                            sub.plan.is_trial and sub.plan.upgrades_to_plan) else sub.plan

                    user_promos = user.promo_codes if user else []
                    current_discount = sub.discount_percent

                    best_promo = next((p for p in user_promos if not p.applies_to_all_plans and any(
                        ap.id == plan_to_charge.id for ap in p.applicable_plans)), None)
                    if best_promo and best_promo.discount_percent > current_discount:
                        current_discount = best_promo.discount_percent
                    elif not best_promo:
                        global_promo = next((p for p in user_promos if p.applies_to_all_plans), None)
                        if global_promo and global_promo.discount_percent > current_discount:
                            current_discount = global_promo.discount_percent

                    if current_discount > sub.discount_percent:
                        sub.discount_percent = current_discount

                    final_price = plan_to_charge.price * (1 - current_discount / 100)

                    if sub.payment_provider == 'Yookassa' and sub.payment_method_id:
                        attempt_started_at = sub.last_payment_attempt or now
                        res, yk_payment_id, yk_payment_status = await process_recurring_payment(
                            bot, sub, plan_to_charge, final_price, config, attempt_started_at
                        )
                        existing_yk_payment = (
                            await session.scalar(
                                select(YookassaPayment).where(YookassaPayment.payment_id == yk_payment_id).with_for_update()
                            ) if yk_payment_id else None
                        )
                        yk_payment_already_processed = bool(
                            existing_yk_payment
                            and existing_yk_payment.status == 'completed'
                            and existing_yk_payment.processed_at
                        )
                        if yk_payment_id and not yk_payment_already_processed:
                            await session.merge(YookassaPayment(
                                payment_id=yk_payment_id,
                                user_id=sub.user_id,
                                plan_id=plan_to_charge.id,
                                amount=final_price,
                                status='completed' if res is True else (yk_payment_status or 'failed'),
                                payment_method_id=sub.payment_method_id,
                                is_recurring=True,
                                processed_at=now if res is True else None
                            ))
                        if res is True and yk_payment_already_processed:
                            sub.payment_attempt_count = 0
                            sub.last_payment_attempt = None
                            await session.commit()
                            continue
                        if res is True:
                            add_days = plan_to_charge.duration_value if plan_to_charge.duration_unit == 'days' else 0
                            add_months = plan_to_charge.duration_value if plan_to_charge.duration_unit == 'months' else 0
                            sub.end_date = sub.end_date + relativedelta(months=add_months) + timedelta(days=add_days)
                            sub.plan_id = plan_to_charge.id
                            sub.payment_attempt_count = 0
                            sub.last_payment_attempt = None
                            await session.commit()
                            pay_id_suffix = f" | PayId={yk_payment_id}" if yk_payment_id else ""
                            plog.info(f"ПРОДЛЕНИЕ | Yookassa | {user_ref} | {plan_to_charge.name} | {final_price:.2f} руб{pay_id_suffix}")
                            await _send_deduplicated_notification(
                                bot,
                                sub.user_id,
                                f"✅ Подписка продлена до {_format_msk(sub.end_date, '%d.%m.%Y %H:%M')}.",
                                f"yk_success:{sub.id}:{yk_payment_id or 'noid'}",
                                now,
                                window=timedelta(days=2),
                            )
                            if config and config.notifications_enabled:
                                for admin_id in all_admin_ids:
                                    try:
                                        await bot.send_message(admin_id,
                                                               f"🔔 Автопродление (YooKassa)!\n\nПользователь: {user_ref}\nТариф: {plan_to_charge.name}\nСумма: {final_price:.2f} руб\nДо: {_format_msk(sub.end_date, '%d.%m.%Y %H:%M')}" + (f"\nPayId: {yk_payment_id}" if yk_payment_id else ""))
                                    except Exception:
                                        pass
                        elif res == 'deactivate':
                            plog.warning(f"АВТОПРОДЛ_ОТКЛ | {user_ref} | причина: deactivate | {plan_to_charge.name}")
                            sub.auto_renewal = False
                            await session.commit()
                            await bot.send_message(sub.user_id,
                                                   "Ваша подписка истекла. Ошибка при автоплатеже (ЮKassa) — автопродление отключено.\n\nПродлите подписку вручную в меню.",
                                                   reply_markup=subscribe_kb)
                            if config and config.notifications_enabled:
                                for admin_id in all_admin_ids:
                                    try:
                                        await bot.send_message(admin_id,
                                                                f"🚫 Автопродление отключено (отказ провайдера)\nПользователь: {user_ref}\nПровайдер: Yookassa")
                                    except Exception:
                                        pass
                        elif res == 'provider_error':
                            sub.last_payment_attempt = attempt_started_at
                            await session.commit()
                            next_retry_str = _format_msk(
                                get_next_retry_at(sub.payment_attempt_count, sub.last_payment_attempt)
                                or (sub.last_payment_attempt + timedelta(hours=2)),
                                '%d.%m %H:%M МСК'
                            )
                            await _send_deduplicated_notification(
                                bot,
                                sub.user_id,
                                f"Платёжный шлюз ЮKassa временно недоступен. Эта ошибка не засчитана как попытка списания.\n\n"
                                f"Повторим запрос после {next_retry_str}.",
                                f"yk_provider_error:{sub.id}:{sub.last_payment_attempt.isoformat() if sub.last_payment_attempt else 'none'}",
                                now,
                                reply_markup=subscribe_kb,
                                window=timedelta(days=1),
                            )
                            if config and config.notifications_enabled:
                                for admin_id in all_admin_ids:
                                    try:
                                        await bot.send_message(
                                            admin_id,
                                            f"⚠️ Сбой провайдера YooKassa\nПользователь: {user_ref}\n"
                                            f"Тариф: {plan_to_charge.name}\nСумма: {final_price:.2f} руб\n"
                                            f"Попытки списания не увеличены."
                                        )
                                    except Exception:
                                        pass
                        elif res == 'integration_error':
                            sub.last_payment_attempt = attempt_started_at
                            await session.commit()
                            await _send_deduplicated_notification(
                                bot,
                                sub.user_id,
                                "Во время автосписания ЮKassa вернула ошибку интеграции. Эта ошибка не засчитана как попытка списания.\n\n"
                                "Мы повторим запрос после исправления проблемы.",
                                f"yk_integration_error:{sub.id}:{sub.last_payment_attempt.isoformat() if sub.last_payment_attempt else 'none'}",
                                now,
                                reply_markup=subscribe_kb,
                                window=timedelta(days=1),
                            )
                            if config and config.notifications_enabled:
                                for admin_id in all_admin_ids:
                                    try:
                                        await bot.send_message(
                                            admin_id,
                                            f"⚠️ Ошибка интеграции YooKassa\nПользователь: {user_ref}\n"
                                            f"Тариф: {plan_to_charge.name}\nСумма: {final_price:.2f} руб\n"
                                            f"Попытки списания не увеличены."
                                        )
                                    except Exception:
                                        pass
                        elif res == 'pending':
                            sub.last_payment_attempt = attempt_started_at
                            await session.commit()
                            next_retry_str = _format_msk(
                                get_next_retry_at(sub.payment_attempt_count, sub.last_payment_attempt)
                                or (sub.last_payment_attempt + timedelta(hours=2)),
                                '%d.%m %H:%M МСК'
                            )
                            pay_id_suffix = f" | PayId={yk_payment_id}" if yk_payment_id else ""
                            plog.info(
                                f"ЗАПРОС_ПРОДЛЕНИЯ | Yookassa | {user_ref} | {plan_to_charge.name} | "
                                f"{final_price:.2f} руб{pay_id_suffix}"
                            )
                            await _send_deduplicated_notification(
                                bot,
                                sub.user_id,
                                f"⏳ Запрос автопродления ЮKassa принят и ожидает подтверждения.\n\n"
                                f"Если платёж не завершится, повторим проверку после {next_retry_str}.",
                                f"yk_pending:{sub.id}:{yk_payment_id or (sub.last_payment_attempt.isoformat() if sub.last_payment_attempt else 'none')}",
                                now,
                                reply_markup=subscribe_kb,
                                window=timedelta(days=1),
                            )
                            if config and config.notifications_enabled:
                                for admin_id in all_admin_ids:
                                    try:
                                        await bot.send_message(
                                            admin_id,
                                            f"⏳ Запрос автопродления (YooKassa)\n\nПользователь: {user_ref}\n"
                                            f"Тариф: {plan_to_charge.name}\nСумма: {final_price:.2f} руб"
                                            + (f"\nPayId: {yk_payment_id}" if yk_payment_id else "")
                                        )
                                    except Exception:
                                        pass
                        else:
                            attempt_num_yk = sub.payment_attempt_count + 1
                            pay_id_suffix = f" | PayId={yk_payment_id}" if yk_payment_id else ""
                            plog.warning(f"ОШИБКА_СПИСАНИЯ | Yookassa | {user_ref} | попытка {attempt_num_yk} | {plan_to_charge.name}{pay_id_suffix}")
                            sub.payment_attempt_count += 1
                            sub.last_payment_attempt = attempt_started_at
                            if attempt_num_yk >= 3:
                                await disable_auto_renewal_after_failed_attempts(
                                    session, bot, sub, user_ref, plan_to_charge.name, subscribe_kb, config, all_admin_ids
                                )
                                continue
                            await session.commit()
                            next_retry_at = get_next_retry_at(sub.payment_attempt_count, sub.last_payment_attempt)
                            if attempt_num_yk == 1 and next_retry_at:
                                next_retry_str = _format_msk(
                                    next_retry_at,
                                    '%d.%m %H:%M МСК'
                                )
                                user_msg_yk = f"Не удалось списать средства (ЮKassa). Повторим попытку {next_retry_str}."
                            elif attempt_num_yk == 2 and next_retry_at:
                                next_retry_str = _format_msk(
                                    next_retry_at,
                                    '%d.%m %H:%M МСК'
                                )
                                user_msg_yk = f"Не удалось списать средства (ЮKassa). Последняя попытка — {next_retry_str}."
                            else:
                                user_msg_yk = "Не удалось списать средства (ЮKassa). Повторим позже."
                            await _send_deduplicated_notification(
                                bot,
                                sub.user_id,
                                user_msg_yk,
                                f"yk_failed:{sub.id}:{attempt_num_yk}",
                                now,
                                reply_markup=subscribe_kb,
                                window=timedelta(days=1),
                            )
                            if config and config.notifications_enabled:
                                for admin_id in all_admin_ids:
                                    try:
                                        await bot.send_message(admin_id,
                                                               f"⚠️ Ошибка автосписания [{attempt_num_yk}/3]\nПользователь: {user_ref}\nТариф: {plan_to_charge.name}\nСумма: {final_price:.2f} руб\nПровайдер: Yookassa")
                                    except Exception:
                                        pass

                    elif sub.payment_provider == 'Robokassa' and sub.payment_method_id:
                        if sub.pending_robokassa_invoice_id:
                            continue  # обработано выше через OpStateExt

                        new_payment = RobokassaPayment(user_id=sub.user_id, plan_id=plan_to_charge.id,
                                                       amount=final_price)
                        session.add(new_payment)
                        await session.commit()

                        robokassa_res = await process_recurring_robokassa_payment(
                            config, plan_to_charge, final_price, sub.payment_method_id, new_payment.id
                        )

                        if robokassa_res is True:
                            plog.info(f"ЗАПРОС_ПРОДЛЕНИЯ | Robokassa | {user_ref} | {plan_to_charge.name} | {final_price:.2f} руб | InvId={new_payment.id}")
                            sub.pending_robokassa_invoice_id = new_payment.id
                            sub.payment_attempt_count += 1
                            sub.last_payment_attempt = now
                            await session.commit()
                            try:
                                await _send_deduplicated_notification(
                                    bot,
                                    sub.user_id,
                                    f"⏳ Попытка автопродления подписки «{plan_to_charge.name}» на сумму {final_price:.2f} руб.\n\n"
                                    f"Если деньги не спишутся в течение нескольких часов — проверьте, что карта активна и разрешены интернет-платежи.",
                                    f"rk_request_pending:{sub.id}:{new_payment.id}",
                                    now,
                                    reply_markup=kb.subscription_pending_keyboard(),
                                    window=timedelta(days=1),
                                )
                            except Exception:
                                pass
                            if config and config.notifications_enabled:
                                for admin_id in all_admin_ids:
                                    try:
                                        await bot.send_message(admin_id,
                                                               f"⏳ Запрос автопродления (Robokassa)\n\nПользователь: {user_ref}\nТариф: {plan_to_charge.name}\nСумма: {final_price:.2f} руб\n\nЗапрос принят, ожидаем подтверждения. Уведомление об оплате придёт отдельно.")
                                    except Exception:
                                        pass
                        elif robokassa_res == 'deactivate':
                            plog.warning(f"АВТОПРОДЛ_ОТКЛ | {user_ref} | причина: deactivate | {plan_to_charge.name}")
                            sub.auto_renewal = False
                            sub.payment_attempt_count = 0
                            new_payment.status = 'request_deactivated'
                            await session.commit()
                            await bot.send_message(sub.user_id,
                                                   "Ваша подписка истекла. Ошибка при автоплатеже (Robokassa) — автопродление отключено.\n\nПродлите подписку вручную в меню.",
                                                   reply_markup=subscribe_kb)
                            if config and config.notifications_enabled:
                                for admin_id in all_admin_ids:
                                    try:
                                        await bot.send_message(admin_id,
                                                               f"🚫 Автопродление отключено (отказ провайдера)\nПользователь: {user_ref}\nПровайдер: Robokassa")
                                    except Exception:
                                        pass
                        elif robokassa_res == 'provider_error':
                            new_payment.status = 'request_provider_error'
                            sub.last_payment_attempt = now
                            await session.commit()
                            next_retry_at = (
                                now + timedelta(hours=2)
                                if sub.payment_attempt_count <= 0
                                else get_next_retry_at(sub.payment_attempt_count, sub.last_payment_attempt)
                            )
                            next_retry_str = (
                                _format_msk(next_retry_at, '%d.%m %H:%M МСК')
                                if next_retry_at else "позже"
                            )
                            try:
                                await _send_deduplicated_notification(
                                    bot,
                                    sub.user_id,
                                    f"Платёжный шлюз Robokassa временно недоступен. Эта ошибка не засчитана как попытка списания.\n\n"
                                    f"Повторим запрос после {next_retry_str}.",
                                    f"rk_provider_error:{sub.id}:{sub.last_payment_attempt.isoformat() if sub.last_payment_attempt else 'none'}",
                                    now,
                                    reply_markup=subscribe_kb,
                                    window=timedelta(days=1),
                                )
                            except Exception:
                                pass
                            if config and config.notifications_enabled:
                                for admin_id in all_admin_ids:
                                    try:
                                        await bot.send_message(
                                            admin_id,
                                            f"⚠️ Сбой провайдера Robokassa\nПользователь: {user_ref}\n"
                                            f"Тариф: {plan_to_charge.name}\nСумма: {final_price:.2f} руб\n"
                                            f"Попытки списания не увеличены."
                                        )
                                    except Exception:
                                        pass
                        else:
                            attempt_num_rk = sub.payment_attempt_count + 1
                            plog.warning(f"ОШИБКА_СПИСАНИЯ | Robokassa | {user_ref} | попытка {attempt_num_rk} | {plan_to_charge.name}")
                            sub.payment_attempt_count += 1
                            sub.last_payment_attempt = now
                            new_payment.status = 'request_failed'
                            if attempt_num_rk >= 3:
                                await disable_auto_renewal_after_failed_attempts(
                                    session, bot, sub, user_ref, plan_to_charge.name, subscribe_kb, config, all_admin_ids
                                )
                                continue
                            await session.commit()
                            next_retry_at = get_next_retry_at(sub.payment_attempt_count, sub.last_payment_attempt)
                            if attempt_num_rk == 1 and next_retry_at:
                                next_retry_str = _format_msk(next_retry_at, '%d.%m %H:%M МСК')
                                user_msg_rk = f"Не удалось провести автосписание (Robokassa). Повторим попытку {next_retry_str}."
                            elif attempt_num_rk == 2 and next_retry_at:
                                next_retry_str = _format_msk(next_retry_at, '%d.%m %H:%M МСК')
                                user_msg_rk = f"Не удалось провести автосписание (Robokassa). Последняя попытка — {next_retry_str}."
                            await _send_deduplicated_notification(
                                bot,
                                sub.user_id,
                                user_msg_rk,
                                f"rk_failed:{sub.id}:{attempt_num_rk}",
                                now,
                                reply_markup=subscribe_kb,
                                window=timedelta(days=1),
                            )
                            if config and config.notifications_enabled:
                                for admin_id in all_admin_ids:
                                    try:
                                        await bot.send_message(admin_id,
                                                               f"⚠️ Ошибка автосписания [{attempt_num_rk}/3]\nПользователь: {user_ref}\nТариф: {plan_to_charge.name}\nСумма: {final_price:.2f} руб\nПровайдер: Robokassa")
                                    except Exception:
                                        pass

            except Exception as e:
                log.error(f"Error checking sub {sub.id}: {e}", exc_info=e)
                await notify_admins_about_error(
                    bot,
                    title="Сбой обработки подписки",
                    stage="check_subscriptions",
                    details=str(e),
                    extra={"subscription_id": sub.id, "user_id": sub.user_id},
                    exception=e,
                    logger=log,
                )


def calculate_signature(*args) -> str:
    return hashlib.md5(':'.join(str(arg) for arg in args).encode()).hexdigest()


async def process_recurring_robokassa_payment(
        config: SubscriptionConfig,
        plan_to_charge: SubscriptionPlan,
        price_to_charge: float,
        parent_invoice_id: str,
        new_invoice_id: int
) -> bool | str:
    log.info(f"Robokassa Recurring: NewInv={new_invoice_id}, ParentInv={parent_invoice_id}, Sum={price_to_charge}")

    merchant_login = config.robokassa_merchant_login
    password = config.robokassa_password_1
    cost = decimal.Decimal(f"{price_to_charge:.2f}")
    description = ''.join(c for c in f"Автопродление подписки: {plan_to_charge.name}" if ord(c) <= 0xFFFF)[:100]

    signature = calculate_signature(
        merchant_login,
        cost,
        new_invoice_id,
        password
    )

    data = {
        'MerchantLogin': merchant_login,
        'OutSum': str(cost),
        'InvId': str(new_invoice_id),
        'PreviousInvoiceID': str(parent_invoice_id),
        'Description': description,
        'SignatureValue': signature
    }

    url = 'https://auth.robokassa.ru/Merchant/Recurring'
    _plog_robokassa_tech(
        "TECH_RECURRING_REQUEST",
        Method="POST",
        URL=url,
        ParentInv=parent_invoice_id,
        NewInv=new_invoice_id,
        Sum=f"{cost:.2f}",
        Plan=plan_to_charge.name,
        Body=_encode_log_params(data)
    )

    try:
        api_timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=api_timeout) as session:
            async with session.post(url, data=data) as response:
                response_text = await response.text()
                log.info(f"Robokassa response ({response.status}): {response_text}")
                _plog_robokassa_tech(
                    "TECH_RECURRING_RESPONSE",
                    NewInv=new_invoice_id,
                    HTTP=response.status,
                    Body=response_text
                )

                if response.status == 200:
                    if response_text.startswith("OK"):
                        return True
                    else:
                        log.error(f"Robokassa logical error: {response_text}")
                        return False
                elif 400 <= response.status < 500:
                    log.error(f"Robokassa client error: {response.status} - {response_text}")
                    return 'deactivate'
                else:
                    log.error(f"Robokassa server error: {response.status}")
                    return 'provider_error'

    except Exception as e:
        log.error(f"Robokassa connection error: {e}")
        _plog_robokassa_tech(
            "TECH_RECURRING_EXCEPTION",
            NewInv=new_invoice_id,
            Error=e
        )
        return 'provider_error'


async def check_robokassa_op_state(config: SubscriptionConfig, invoice_id: int) -> str:
    """Проверяет статус платежа через OpStateExt.
    Возвращает: 'success', 'failed', 'pending', 'unknown'
    """
    if not config.robokassa_merchant_login or not config.robokassa_password_2:
        return 'unknown'

    merchant_login = config.robokassa_merchant_login
    password2 = config.robokassa_password_2
    signature = hashlib.md5(f"{merchant_login}:{invoice_id}:{password2}".encode()).hexdigest()

    url = "https://auth.robokassa.ru/Merchant/WebService/Service.asmx/OpStateExt"
    params = {
        'MerchantLogin': merchant_login,
        'InvoiceID': str(invoice_id),
        'Signature': signature
    }

    _plog_robokassa_tech(
        "TECH_OPSTATE_REQUEST",
        Method="GET",
        URL=url,
        InvId=invoice_id,
        Query=_encode_log_params(params)
    )

    try:
        api_timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=api_timeout) as http_session:
            async with http_session.get(url, params=params) as response:
                if response.status != 200:
                    log.warning(f"OpStateExt HTTP {response.status} for InvId={invoice_id}")
                    _plog_robokassa_tech(
                        "TECH_OPSTATE_HTTP",
                        InvId=invoice_id,
                        HTTP=response.status
                    )
                    return 'unknown'
                text = await response.text()
                ns = {'rb': 'http://merchant.roboxchange.com/WebService/'}
                root = ET.fromstring(text)

                result_code_el = root.find('rb:Result/rb:Code', ns)
                result_desc_el = root.find('rb:Result/rb:Description', ns)
                if result_code_el is None or result_code_el.text != '0':
                    log.warning(f"OpStateExt Result/Code={result_code_el.text if result_code_el else 'None'} for InvId={invoice_id}")
                    _plog_robokassa_tech(
                        "TECH_OPSTATE_RESULT",
                        InvId=invoice_id,
                        ResultCode=result_code_el.text if result_code_el is not None else None,
                        ResultDescription=result_desc_el.text if result_desc_el is not None else None
                    )
                    return 'unknown'

                state_code_el = root.find('rb:State/rb:Code', ns)
                state_desc_el = root.find('rb:State/rb:Description', ns)
                if state_code_el is None:
                    _plog_robokassa_tech(
                        "TECH_OPSTATE_STATE",
                        InvId=invoice_id,
                        ResultCode=result_code_el.text,
                        ResultDescription=result_desc_el.text if result_desc_el is not None else None,
                        StateCode=None
                    )
                    return 'unknown'

                state_code = int(state_code_el.text)
                log.info(f"OpStateExt InvId={invoice_id} → state={state_code}")
                _plog_robokassa_tech(
                    "TECH_OPSTATE_STATE",
                    InvId=invoice_id,
                    ResultCode=result_code_el.text,
                    ResultDescription=result_desc_el.text if result_desc_el is not None else None,
                    StateCode=state_code,
                    StateDescription=state_desc_el.text if state_desc_el is not None else None
                )

                if state_code in (50, 100):
                    return 'success'
                elif state_code in (10, 60):
                    return 'failed'
                elif state_code == 5:
                    return 'pending'
                else:
                    return 'unknown'

    except Exception as e:
        log.error(f"OpStateExt error for InvId={invoice_id}: {e}")
        _plog_robokassa_tech(
            "TECH_OPSTATE_EXCEPTION",
            InvId=invoice_id,
            Error=e
        )
        return 'unknown'
