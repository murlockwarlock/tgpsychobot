from __future__ import annotations

import html
import math

from sqlalchemy import desc, func, select

from ..api import MaxApiClient
from ..keyboards import callback_button, inline_keyboard
from ..legacy import ReferralPaymentLog, SubscriptionConfig, User, async_session_maker
from ..storage import StateStore

REFERRAL_REFERRERS_PAGE_SIZE = 10


async def show_menu(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        referrers_count = (
            await session.execute(
                select(func.count(func.distinct(User.referred_by))).where(User.referred_by.isnot(None))
            )
        ).scalar() or 0
        referrals_count = (
            await session.execute(
                select(func.count()).where(User.referred_by.isnot(None))
            )
        ).scalar() or 0
        total_turnover = (
            await session.execute(select(func.sum(ReferralPaymentLog.amount)))
        ).scalar() or 0.0

    status = "✅ Включена" if (config and config.referral_enabled) else "❌ Выключена"
    text = (
        f"👫 <b>Реферальная программа</b>\n\n"
        f"Статус: {status}\n"
        f"Рефереров: {referrers_count}\n"
        f"Рефералов: {referrals_count}\n"
        f"Общий оборот: {total_turnover:.2f} руб."
    )
    rows = [
        [callback_button("⚙️ Настройки", "admin_referral_settings")],
        [callback_button("👥 Рефереры", "admin_referral_referrers_0")],
        [callback_button("◀️ Назад", "admin_panel")],
    ]
    await client.send_message(chat_id=chat_id, text=text, attachments=inline_keyboard(rows))


async def show_settings(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
    config = config or SubscriptionConfig()
    text = "⚙️ <b>Настройки реферальной программы</b>"
    await client.send_message(chat_id=chat_id, text=text, attachments=_settings_keyboard(config))


def _settings_keyboard(config: SubscriptionConfig):
    rows = [
        [callback_button(
            "🔘 Программа: ✅ Вкл" if config.referral_enabled else "🔘 Программа: ❌ Выкл",
            "admin_referral_toggle_enabled",
        )],
        [callback_button(
            "💰 Бонус рефереру за оплату: ✅" if config.referral_pay_bonus_enabled else "💰 Бонус рефереру за оплату: ❌",
            "admin_referral_toggle_pay_bonus",
        )],
        [callback_button(
            "1️⃣ Только первая оплата: ✅" if config.referral_pay_bonus_first_only else "1️⃣ Только первая оплата: ❌",
            "admin_referral_toggle_pay_first_only",
        )],
        [callback_button(f"👤 Бонус рефереру: {config.referral_bonus_days_referrer} дн.", "admin_referral_set_bonus_referrer")],
        [callback_button(f"🆕 Бонус новому (реферал): {config.referral_bonus_days_referral} дн.", "admin_referral_set_bonus_referral")],
        [callback_button(f"💳 Дней за оплату реферала: {config.referral_pay_bonus_days}", "admin_referral_set_pay_days")],
        [callback_button(f"🔤 Кнопка меню: «{config.referral_btn_name}»", "admin_referral_set_btn_name")],
        [callback_button(f"🔤 Кнопка подписки: «{config.referral_sub_btn_name}»", "admin_referral_set_sub_btn_name")],
        [callback_button("◀️ Назад к реферальной программе", "admin_referral_menu")],
    ]
    return inline_keyboard(rows)


async def toggle_enabled(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        if config:
            config.referral_enabled = not config.referral_enabled
            await session.commit()
    await show_settings(client, chat_id)


async def toggle_pay_bonus(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        if config:
            config.referral_pay_bonus_enabled = not config.referral_pay_bonus_enabled
            await session.commit()
    await show_settings(client, chat_id)


async def toggle_pay_first_only(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        if config:
            config.referral_pay_bonus_first_only = not config.referral_pay_bonus_first_only
            await session.commit()
    await show_settings(client, chat_id)


async def start_set_bonus_referrer(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_referral_set_bonus_referrer", {})
    await client.send_message(
        chat_id=chat_id,
        text="Введите количество бонусных дней рефереру (за каждого приведённого):",
    )


async def save_bonus_referrer(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    try:
        value = int(text.strip())
        if value < 0:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Введите целое неотрицательное число.")
        return
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        if config:
            config.referral_bonus_days_referrer = value
            await session.commit()
    await states.clear(user_id)
    await show_settings(client, chat_id)


async def start_set_bonus_referral(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_referral_set_bonus_referral", {})
    await client.send_message(
        chat_id=chat_id,
        text="Введите количество бонусных дней рефералу (новому пользователю):",
    )


async def save_bonus_referral(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    try:
        value = int(text.strip())
        if value < 0:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Введите целое неотрицательное число.")
        return
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        if config:
            config.referral_bonus_days_referral = value
            await session.commit()
    await states.clear(user_id)
    await show_settings(client, chat_id)


async def start_set_pay_days(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_referral_set_pay_days", {})
    await client.send_message(
        chat_id=chat_id,
        text="Введите количество бонусных дней рефереру за оплату реферала:",
    )


async def save_pay_days(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    try:
        value = int(text.strip())
        if value < 0:
            raise ValueError
    except ValueError:
        await client.send_message(chat_id=chat_id, text="Введите целое неотрицательное число.")
        return
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        if config:
            config.referral_pay_bonus_days = value
            await session.commit()
    await states.clear(user_id)
    await show_settings(client, chat_id)


def _cancel_keyboard():
    return inline_keyboard([[callback_button("❌ Отмена", "admin_referral_cancel_input")]])


async def start_set_btn_name(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_referral_set_btn_name", {})
    await client.send_message(
        chat_id=chat_id,
        text="Введите название кнопки реферальной программы в главном меню:",
        attachments=_cancel_keyboard(),
    )


async def save_btn_name(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    value = text.strip()
    if not value:
        await client.send_message(chat_id=chat_id, text="Название не может быть пустым.")
        return
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        if config:
            config.referral_btn_name = html.escape(value)
            await session.commit()
    await states.clear(user_id)
    await show_settings(client, chat_id)


async def start_set_sub_btn_name(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_referral_set_sub_btn_name", {})
    await client.send_message(
        chat_id=chat_id,
        text="Введите название кнопки реферальной программы в меню подписки:",
        attachments=_cancel_keyboard(),
    )


async def save_sub_btn_name(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    value = text.strip()
    if not value:
        await client.send_message(chat_id=chat_id, text="Название не может быть пустым.")
        return
    async with async_session_maker() as session:
        config = await session.get(SubscriptionConfig, 1)
        if config:
            config.referral_sub_btn_name = html.escape(value)
            await session.commit()
    await states.clear(user_id)
    await show_settings(client, chat_id)


async def cancel_input(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.clear(user_id)
    await show_settings(client, chat_id)


async def show_referrers_page(client: MaxApiClient, chat_id: int, page: int) -> None:
    async with async_session_maker() as session:
        # Count per referrer (referrer_id = referred_by value)
        subq = (
            select(User.referred_by.label("ref_id"), func.count().label("cnt"))
            .where(User.referred_by.isnot(None))
            .group_by(User.referred_by)
            .subquery()
        )
        total_result = await session.execute(select(func.count()).select_from(subq))
        total = total_result.scalar() or 0
        total_pages = max(1, math.ceil(total / REFERRAL_REFERRERS_PAGE_SIZE))
        page = max(0, min(page, total_pages - 1))

        rows_result = await session.execute(
            select(subq.c.ref_id, subq.c.cnt)
            .order_by(desc(subq.c.cnt))
            .offset(page * REFERRAL_REFERRERS_PAGE_SIZE)
            .limit(REFERRAL_REFERRERS_PAGE_SIZE)
        )
        referrer_rows = rows_result.fetchall()

        keyboard_rows = []
        for ref_id, count in referrer_rows:
            user = await session.get(User, ref_id)
            if user:
                name = html.escape(user.first_name or "")
                if user.username:
                    name += f" @{html.escape(user.username)}"
            else:
                name = str(ref_id)

            turnover_result = await session.execute(
                select(func.sum(ReferralPaymentLog.amount)).where(ReferralPaymentLog.referrer_id == ref_id)
            )
            turnover = turnover_result.scalar() or 0.0
            keyboard_rows.append([
                callback_button(f"👤 {name} — {count} реф. | {turnover:.0f}₽", f"admin_referral_referrer_{ref_id}_{page}")
            ])

    # Navigation row
    nav = []
    if page > 0:
        nav.append(callback_button("◀️", f"admin_referral_referrers_{page - 1}"))
    if page < total_pages - 1:
        nav.append(callback_button("▶️", f"admin_referral_referrers_{page + 1}"))
    if nav:
        keyboard_rows.append(nav)
    keyboard_rows.append([callback_button("◀️ Назад", "admin_referral_menu")])

    text = f"<b>👫 Рефереры · стр. {page + 1}/{total_pages}</b>\n\nВсего: {total}"
    await client.send_message(chat_id=chat_id, text=text, attachments=inline_keyboard(keyboard_rows))


async def show_referrer_detail(client: MaxApiClient, chat_id: int, referrer_id: int, page: int) -> None:
    async with async_session_maker() as session:
        user = await session.get(User, referrer_id)
        if user:
            name = html.escape(user.first_name or "")
            if user.username:
                name += f" @{html.escape(user.username)}"
        else:
            name = str(referrer_id)

        referrals_result = await session.execute(
            select(User).where(User.referred_by == referrer_id)
        )
        referrals = referrals_result.scalars().all()

        turnover_result = await session.execute(
            select(func.sum(ReferralPaymentLog.amount)).where(ReferralPaymentLog.referrer_id == referrer_id)
        )
        total_turnover = turnover_result.scalar() or 0.0

    referral_lines = []
    for r in referrals:
        r_name = html.escape(r.first_name or "")
        if r.username:
            r_name += f" @{html.escape(r.username)}"
        referral_lines.append(r_name or str(r.user_id))

    referrals_text = "\n".join(referral_lines) if referral_lines else "—"
    text = (
        f"<b>👤 Реферер: {name}</b>\n\n"
        f"ID: <code>{referrer_id}</code>\n"
        f"Привлечённых: {len(referrals)}\n"
        f"Оборот: {total_turnover:.2f} руб.\n\n"
        f"<b>Рефералы:</b>\n"
        f"{referrals_text}"
    )
    rows = [[callback_button("◀️ Назад к реферерам", f"admin_referral_referrers_{page}")]]
    await client.send_message(chat_id=chat_id, text=text, attachments=inline_keyboard(rows))
