from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import func, select

from ..api import MaxApiClient
from ..keyboards import admin_panel_keyboard, admin_subscriptions_keyboard, callback_button, inline_keyboard
from ..legacy import Message as DBMessage, PromoCode, ReferralPaymentLog, SubscriptionConfig, SubscriptionPlan, Topic, User, UserSubscription, async_session_maker
from ..models import MAX_ID_OFFSET


async def show_admin_panel(client: MaxApiClient, chat_id: int) -> None:
    await client.send_message(chat_id=chat_id, text="Добро пожаловать в админ-панель MAX.", attachments=admin_panel_keyboard())


def _build_stats_text(
    *,
    total_users: int,
    users_today: int,
    users_week: int,
    users_month: int,
    total_messages: int,
    active_subs: int,
    active_paid_subs: int,
    active_bonus_subs: int,
) -> str:
    return (
        "📊 <b>Статистика MAX</b>\n\n"
        "👥 <b>Пользователи MAX:</b>\n"
        f"• Всего зарегистрировано: <b>{total_users}</b>\n\n"
        "🆕 <b>Новые регистрации:</b>\n"
        f"• Сегодня: {users_today}\n"
        f"• За текущую неделю: {users_week}\n"
        f"• За текущий месяц: {users_month}\n\n"
        "⭐️ <b>Активный доступ сейчас:</b>\n"
        f"• Всего: <b>{active_subs}</b>\n"
        f"• По тарифу: {active_paid_subs}\n"
        f"• Бонусный или пробный: {active_bonus_subs}\n\n"
        f"💬 <b>Сообщений всего:</b> {total_messages}"
    )


async def show_stats(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=today_start.weekday())
        month_start = today_start.replace(day=1)

        total_users = await session.scalar(
            select(func.count()).select_from(User).where(User.id >= MAX_ID_OFFSET)
        ) or 0
        users_today = await session.scalar(
            select(func.count()).select_from(User).where(User.id >= MAX_ID_OFFSET, User.created_at >= today_start)
        ) or 0
        users_week = await session.scalar(
            select(func.count()).select_from(User).where(User.id >= MAX_ID_OFFSET, User.created_at >= week_start)
        ) or 0
        users_month = await session.scalar(
            select(func.count()).select_from(User).where(User.id >= MAX_ID_OFFSET, User.created_at >= month_start)
        ) or 0
        total_messages = await session.scalar(
            select(func.count()).select_from(DBMessage).where(DBMessage.user_id >= MAX_ID_OFFSET)
        ) or 0
        active_subs = await session.scalar(
            select(func.count()).select_from(UserSubscription).where(
                UserSubscription.user_id >= MAX_ID_OFFSET,
                UserSubscription.end_date > now,
            )
        ) or 0
        active_paid_subs = await session.scalar(
            select(func.count()).select_from(UserSubscription).where(
                UserSubscription.user_id >= MAX_ID_OFFSET,
                UserSubscription.end_date > now,
                UserSubscription.plan_id.is_not(None),
            )
        ) or 0
        active_bonus_subs = await session.scalar(
            select(func.count()).select_from(UserSubscription).where(
                UserSubscription.user_id >= MAX_ID_OFFSET,
                UserSubscription.end_date > now,
                UserSubscription.plan_id.is_(None),
            )
        ) or 0

    text = _build_stats_text(
        total_users=total_users,
        users_today=users_today,
        users_week=users_week,
        users_month=users_month,
        total_messages=total_messages,
        active_subs=active_subs,
        active_paid_subs=active_paid_subs,
        active_bonus_subs=active_bonus_subs,
    )
    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=inline_keyboard([[callback_button("◀️ Назад", "admin_panel")]]),
    )


async def show_subscriptions_summary(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        active_subs = await session.scalar(
            select(func.count()).select_from(UserSubscription).where(
                UserSubscription.end_date > func.now(), UserSubscription.user_id >= MAX_ID_OFFSET
            )
        ) or 0
        total_subs = await session.scalar(
            select(func.count()).select_from(UserSubscription).where(UserSubscription.user_id >= MAX_ID_OFFSET)
        ) or 0
        plans_count = await session.scalar(select(func.count()).select_from(SubscriptionPlan)) or 0
        promo_count = await session.scalar(select(func.count()).select_from(PromoCode)) or 0
    status = "✅ Включены" if (config and config.subscriptions_enabled) else "❌ Выключены"
    text = (
        "<b>⭐️ Подписки</b>\n\n"
        f"Статус системы: {status}\n"
        f"Активных подписок: {active_subs}\n"
        f"Всего записей подписок: {total_subs}\n"
        f"Тарифов в базе: {plans_count}\n"
        f"Промокодов в базе: {promo_count}\n\n"
        "Управляйте тарифами, промокодами и платёжными настройками из этого раздела."
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_subscriptions_keyboard())


async def show_topics_summary(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        topics = (await session.execute(select(Topic).order_by(Topic.sort_order.asc(), Topic.id.asc()))).scalars().all()
    if not topics:
        await client.send_message(chat_id=chat_id, text="Темы пока не созданы.", attachments=admin_panel_keyboard())
        return
    lines = ["<b>💬 Темы диалогов</b>\n\n"]
    for topic in topics[:25]:
        status = "🟢" if topic.is_active else "⚪️"
        admin_only = " 🔒" if topic.admin_only else ""
        lines.append(f"{status} {topic.name}{admin_only}\n")
    await client.send_message(chat_id=chat_id, text="".join(lines), attachments=admin_panel_keyboard())


async def show_referral_summary(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        referrers = await session.scalar(select(func.count(func.distinct(User.referred_by))).where(User.referred_by != None)) or 0
        referrals = await session.scalar(select(func.count()).select_from(User).where(User.referred_by != None)) or 0
        turnover = await session.scalar(select(func.coalesce(func.sum(ReferralPaymentLog.amount), 0.0))) or 0.0
    status = "✅ Включена" if (config and config.referral_enabled) else "❌ Выключена"
    text = (
        "<b>👫 Реферальная программа</b>\n\n"
        f"Статус: {status}\n"
        f"Рефереров: {referrers}\n"
        f"Рефералов: {referrals}\n"
        f"Оборот: {turnover:.2f} руб."
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_panel_keyboard())


async def show_test_summary(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
    text = (
        "<b>🧩 Тест</b>\n\n"
        "Пользовательский сценарий теста уже вынесен в отдельный модуль MAX.\n"
        "Админское редактирование вопросов и кейсов в этом переносе будет следующим блоком."
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_panel_keyboard())
