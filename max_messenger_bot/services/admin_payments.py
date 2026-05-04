from __future__ import annotations

import html
import math
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from ..api import MaxApiClient
from ..keyboards import admin_payment_keys_keyboard, admin_payment_settings_keyboard, callback_button, inline_keyboard
from ..logging_utils import get_payments_logger
from ..legacy import RobokassaPayment, SubscriptionConfig, SubscriptionPlan, User, UserSubscription, YookassaPayment, async_session_maker
from ..storage import StateStore


log = get_payments_logger("admin")


KEY_NAMES = {
    "yookassa_shop_id": "ЮKassa Shop ID",
    "yookassa_secret_key": "ЮKassa Secret Key",
    "robokassa_merchant_login": "Robokassa Merchant Login",
    "robokassa_password_1": "Robokassa Password 1",
    "robokassa_password_2": "Robokassa Password 2",
    "telegram_pay_token": "Telegram Pay Token",
    "offer_agreement_url": "URL оферты",
    "privacy_policy_url": "URL политики конфиденциальности",
}


def _mask(value: str | None) -> str:
    if not value:
        return "Не задан"
    if len(value) <= 8:
        return value
    return f"{value[:4]}...{value[-4:]}"


async def show_settings(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        if not config:
            config = SubscriptionConfig(id=1)
            session.add(config)
            await session.commit()
    await client.send_message(
        chat_id=chat_id,
        text="⚙️ <b>Настройки платежей</b>\n\nУправление уведомлениями, подписками и ключами.",
        attachments=admin_payment_settings_keyboard(config),
    )


async def toggle_notifications(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        config.notifications_enabled = not config.notifications_enabled
        await session.commit()
        log.info("Payment notifications toggled enabled=%s", config.notifications_enabled)
    await show_settings(client, chat_id)


async def toggle_subscriptions(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        config.subscriptions_enabled = not config.subscriptions_enabled
        await session.commit()
        log.info("Subscriptions toggled enabled=%s", config.subscriptions_enabled)
    await show_settings(client, chat_id)


async def start_set_bonus(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_payment_set_bonus", {})
    await client.send_message(chat_id=chat_id, text="Введите количество дней приветственного бонуса.")


async def save_bonus(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    try:
        days = int(text.strip())
        if days < 0:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Введите целое неотрицательное число.")
        return
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        config.welcome_bonus_days = days
        await session.commit()
        log.info("Welcome bonus updated days=%s", days)
    await states.clear(user_id)
    await show_settings(client, chat_id)


async def show_keys(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
    text = (
        "<b>Текущие платёжные ключи и ссылки</b>\n\n"
        f"<b>ЮKassa Shop ID:</b> <code>{config.yookassa_shop_id or 'Не задан'}</code>\n"
        f"<b>ЮKassa Secret Key:</b> <code>{_mask(config.yookassa_secret_key)}</code>\n"
        f"<b>Robokassa Merchant:</b> <code>{config.robokassa_merchant_login or 'Не задан'}</code>\n"
        f"<b>Robokassa Pass 1:</b> <code>{_mask(config.robokassa_password_1)}</code>\n"
        f"<b>Robokassa Pass 2:</b> <code>{_mask(config.robokassa_password_2)}</code>\n"
        f"<b>Telegram Pay Token:</b> <code>{_mask(config.telegram_pay_token)}</code>\n"
        f"<b>Оферта:</b> <code>{config.offer_agreement_url or 'Не задан'}</code>\n"
        f"<b>Политика:</b> <code>{config.privacy_policy_url or 'Не задан'}</code>"
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_payment_keys_keyboard())


async def start_set_key(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, key_name: str) -> None:
    await states.set(user_id, chat_id, "admin_payment_set_key", {"key_name": key_name})
    await client.send_message(chat_id=chat_id, text=f"Введите новое значение для {KEY_NAMES.get(key_name, key_name)}.")


async def save_key(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    snapshot = await states.get(user_id)
    key_name = snapshot.data.get("key_name") if snapshot else None
    if not key_name:
        await client.send_message(chat_id=chat_id, text="Ключ не определён.")
        return
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        setattr(config, key_name, text.strip())
        await session.commit()
        log.info("Payment config key updated key=%s", key_name)
    await states.clear(user_id)
    await show_keys(client, chat_id)


PAGE_LOG_SIZE = 15


async def show_payment_stats(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        now = datetime.utcnow()
        total_users = (await session.execute(select(func.count(User.id)))).scalar() or 0

        active_paid_subs = (await session.execute(
            select(UserSubscription)
            .where(UserSubscription.plan_id.is_not(None), UserSubscription.end_date > now)
            .options(selectinload(UserSubscription.plan))
        )).scalars().all()
        active_paid_count = len(active_paid_subs)
        current_mrr = sum(sub.plan.price for sub in active_paid_subs if sub.plan)

        active_trials_count = (await session.execute(
            select(func.count(UserSubscription.id)).where(
                UserSubscription.plan_id.is_(None), UserSubscription.end_date > now
            )
        )).scalar() or 0

        expired_count = (await session.execute(
            select(func.count(UserSubscription.id)).where(UserSubscription.end_date <= now)
        )).scalar() or 0

        total_robo_revenue = (await session.execute(select(func.sum(RobokassaPayment.amount)))).scalar() or 0.0
        total_yoo_revenue = (await session.execute(select(func.sum(YookassaPayment.amount)))).scalar() or 0.0

        plan_breakdown = (await session.execute(
            select(SubscriptionPlan.name, func.count(UserSubscription.id))
            .join(UserSubscription)
            .where(UserSubscription.end_date > now)
            .group_by(SubscriptionPlan.name)
        )).all()

    text = (
        "<b>📊 Расширенная статистика</b>\n\n"
        "👥 <b>Пользователи:</b>\n"
        f"• Всего в базе: {total_users}\n"
        f"• Активные платные: {active_paid_count}\n"
        f"• На пробном периоде: {active_trials_count}\n"
        f"• Истекшие подписки: {expired_count}\n\n"
        "💰 <b>Финансы:</b>\n"
        f"• Текущий MRR (активные): {current_mrr:,.2f} руб.\n"
        f"• Доход Robokassa: {total_robo_revenue:,.2f} руб.\n"
        f"• Доход YooKassa: {total_yoo_revenue:,.2f} руб.\n\n"
        "📉 <b>Популярность тарифов (Активные):</b>\n"
    )
    if plan_breakdown:
        for name, count in plan_breakdown:
            text += f"• {html.escape(name)}: {count} шт.\n"
    else:
        text += "• Нет активных тарифов\n"

    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=inline_keyboard([[callback_button("◀️ Назад", "admin_payment_settings")]])
    )


async def show_payment_log(client: MaxApiClient, chat_id: int, page: int = 0, filter_key: str = "all") -> None:
    async with async_session_maker() as session:
        entries = []
        if filter_key in ("all", "robokassa"):
            robo = (await session.execute(
                select(RobokassaPayment).order_by(RobokassaPayment.created_at.desc())
            )).scalars().all()
            for r in robo:
                entries.append({
                    "source": "Robokassa",
                    "user_id": r.user_id,
                    "amount": r.amount,
                    "status": r.status,
                    "created_at": r.created_at,
                })
        if filter_key in ("all", "yookassa"):
            yoo = (await session.execute(
                select(YookassaPayment).order_by(YookassaPayment.created_at.desc())
            )).scalars().all()
            for y in yoo:
                entries.append({
                    "source": "YooKassa",
                    "user_id": y.user_id,
                    "amount": y.amount,
                    "status": y.status,
                    "created_at": y.created_at,
                })

    entries.sort(key=lambda x: x["created_at"] or datetime.min, reverse=True)
    total = len(entries)
    total_pages = max(1, math.ceil(total / PAGE_LOG_SIZE))
    page = max(0, min(page, total_pages - 1))
    page_entries = entries[page * PAGE_LOG_SIZE:(page + 1) * PAGE_LOG_SIZE]

    lines = [f"<b>💳 Журнал платежей · стр. {page + 1}/{total_pages}</b> ({total} записей)\n"]
    for e in page_entries:
        dt = e["created_at"].strftime('%d.%m %H:%M') if e["created_at"] else "?"
        lines.append(f"<code>{dt}</code> {html.escape(e['source'])} user:{e['user_id']} {e['amount']:.2f}₽ {html.escape(e['status'] or '?')}")

    text = "\n".join(lines)

    nav_row = []
    if page > 0:
        nav_row.append(callback_button("◀️", f"admin_plog_{page - 1}_{filter_key}"))
    nav_row.append(callback_button(f"{page + 1}/{total_pages}", "noop"))
    if page < total_pages - 1:
        nav_row.append(callback_button("▶️", f"admin_plog_{page + 1}_{filter_key}"))

    filter_row = [
        callback_button("· Все ·" if filter_key == "all" else "Все", "admin_plog_0_all"),
        callback_button("· Robokassa ·" if filter_key == "robokassa" else "Robokassa", "admin_plog_0_robokassa"),
        callback_button("· YooKassa ·" if filter_key == "yookassa" else "YooKassa", "admin_plog_0_yookassa"),
    ]
    rows = [filter_row]
    if nav_row:
        rows.append(nav_row)
    rows.append([callback_button("◀️ Назад", "admin_payment_settings")])
    await client.send_message(chat_id=chat_id, text=text, attachments=inline_keyboard(rows))
