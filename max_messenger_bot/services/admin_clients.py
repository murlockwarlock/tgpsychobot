from __future__ import annotations

import html
import math
from datetime import datetime

from sqlalchemy import func, or_, select
from sqlalchemy.orm import selectinload

from ..api import MaxApiClient
from ..formatting import markdown_to_html, split_text
from ..keyboards import admin_client_profile_keyboard, admin_client_search_keyboard, admin_clients_keyboard, admin_history_keyboard, callback_button, inline_keyboard
from ..legacy import Message as DBMessage, RobokassaPayment, Topic, User, UserSubscription, YookassaPayment, async_session_maker
from ..models import MAX_ID_OFFSET
from ..storage import StateStore


PAGE_SIZE = 10
HISTORY_SAFE_LIMIT = 3500


async def list_clients(client: MaxApiClient, chat_id: int, page: int = 0) -> None:
    async with async_session_maker() as session:
        total_users = await session.scalar(
            select(func.count()).select_from(User).where(User.id >= MAX_ID_OFFSET)
        ) or 0
        total_pages = max(1, math.ceil(total_users / PAGE_SIZE))
        page = max(0, min(page, total_pages - 1))
        clients = (
            await session.execute(
                select(User)
                .where(User.id >= MAX_ID_OFFSET)
                .outerjoin(DBMessage, User.id == DBMessage.user_id)
                .group_by(User.id)
                .order_by(func.max(DBMessage.timestamp).desc().nulls_last(), User.created_at.desc())
                .offset(page * PAGE_SIZE)
                .limit(PAGE_SIZE)
            )
        ).scalars().all()

    text = f"<b>👥 Список клиентов</b>\n\nСтраница {page + 1}/{total_pages}"
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_clients_keyboard(page, total_pages, clients))


async def show_client_profile(client: MaxApiClient, chat_id: int, target_user_id: int) -> None:
    async with async_session_maker() as session:
        user = await session.get(User, target_user_id, options=[selectinload(User.subscription).selectinload(UserSubscription.plan)])
    if user and user.id < MAX_ID_OFFSET:
        user = None
    if not user:
        await client.send_message(chat_id=chat_id, text="Клиент не найден.")
        return

    subscription_line = "нет"
    if user.subscription:
        if user.subscription.plan:
            subscription_line = user.subscription.plan.name
        elif user.subscription.end_date:
            subscription_line = f"бонус до {user.subscription.end_date.strftime('%d.%m.%Y %H:%M')}"

    text = (
        "<b>Профиль клиента</b>\n\n"
        f"<b>ID:</b> <code>{user.id}</code>\n"
        f"<b>Имя:</b> {html.escape(user.name or user.first_name or 'Не указано')}\n"
        f"<b>Username:</b> {html.escape(user.username or 'Не указан')}\n"
        f"<b>Пол:</b> {html.escape(user.gender or 'Не указан')}\n"
        f"<b>Возраст:</b> {html.escape(user.age or 'Не указан')}\n"
        f"<b>Подписка:</b> {html.escape(subscription_line)}\n"
        + (f"TG ID: <code>{user.tg_user_id}</code>\n" if user.tg_user_id else "TG аккаунт: не привязан\n")
        + f"<b>Админ:</b> {'да' if user.is_admin else 'нет'}\n"
        + f"<b>Дата регистрации:</b> {user.created_at.strftime('%d.%m.%Y %H:%M')}"
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_client_profile_keyboard(target_user_id))


async def show_client_history(client: MaxApiClient, chat_id: int, target_user_id: int, page: int = 0) -> None:
    async with async_session_maker() as session:
        target_user = await session.get(User, target_user_id)
        if not target_user:
            await client.send_message(chat_id=chat_id, text="Пользователь не найден.")
            return
        topics = (await session.execute(select(Topic))).scalars().all()
        topic_map = {topic.id: topic.name for topic in topics}
        messages = (
            await session.execute(
                select(DBMessage)
                .where(DBMessage.user_id == target_user_id)
                .order_by(DBMessage.timestamp.asc())
            )
        ).scalars().all()

    if not messages:
        await client.send_message(chat_id=chat_id, text="История сообщений пользователя пуста.", attachments=admin_client_profile_keyboard(target_user_id))
        return

    rendered_parts: list[str] = []
    last_dialogue_id = None
    for msg in messages:
        if msg.dialogue_id != last_dialogue_id:
            rendered_parts.append(f"\n--- <b>Диалог №{msg.dialogue_id}</b> ---\n")
            last_dialogue_id = msg.dialogue_id
        role = "👤 Клиент" if msg.role == "user" else "🤖 Бот"
        topic_name = topic_map.get(msg.topic_id, "Общий")
        content = html.escape(msg.content or "")
        if msg.role == "assistant":
            content = markdown_to_html(msg.content or "")
        rendered_parts.append(
            f"<b>{role}</b> [<i>{msg.timestamp.strftime('%d.%m.%Y %H:%M')}</i>] [<i>{html.escape(topic_name)}</i>]:\n{content}\n"
        )

    full_text = f"📜 <b>История клиента:</b> {html.escape(target_user.first_name or str(target_user.id))} (<a href='https://t.me/@id{target_user.id}'><code>{target_user.id}</code></a>)\n\n{''.join(rendered_parts)}"
    pages = split_text(full_text, HISTORY_SAFE_LIMIT)
    total_pages = max(1, len(pages))
    page = max(0, min(page, total_pages - 1))
    await client.send_message(chat_id=chat_id, text=pages[page], attachments=admin_history_keyboard(target_user_id, page, total_pages))



async def start_search(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_client_search", {})
    await client.send_message(
        chat_id=chat_id,
        text="🔍 <b>Поиск клиента</b>\n\nВведите имя, username или ID пользователя:",
        attachments=admin_client_search_keyboard(),
    )


async def search_clients(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, query: str) -> None:
    await states.clear(user_id)
    query = query.strip()
    if not query:
        await list_clients(client, chat_id, 0)
        return

    query_lower = query.lower()
    async with async_session_maker() as session:
        is_id_query = query.lstrip("-").isdigit()
        if is_id_query:
            stmt = select(User).where(User.id >= MAX_ID_OFFSET, User.id == int(query)).limit(20)
        else:
            pattern = f"%{query_lower}%"
            stmt = (
                select(User)
                .where(
                    User.id >= MAX_ID_OFFSET,
                    or_(
                        func.lower(User.first_name).like(pattern),
                        func.lower(User.name).like(pattern),
                        func.lower(User.username).like(pattern),
                    )
                )
                .order_by(User.created_at.desc())
                .limit(20)
            )
        clients = (await session.execute(stmt)).scalars().all()

    if not clients:
        await client.send_message(
            chat_id=chat_id,
            text=f"❌ По запросу <b>{html.escape(query)}</b> ничего не найдено.",
            attachments=admin_client_search_keyboard(),
        )
        return

    rows = []
    for c in clients:
        name = c.name or c.first_name or str(c.id)
        username = f"@{c.username}" if c.username else "без username"
        rows.append([callback_button(f"{name} ({username})", f"view_client_{c.id}")])
    rows.append([callback_button("⬅️ К клиентам", "admin_clients")])
    text = f"🔍 Результаты поиска <b>{html.escape(query)}</b>: найдено {len(clients)} клиент(ов)"
    await client.send_message(chat_id=chat_id, text=text, attachments=inline_keyboard(rows))


async def download_history_txt(client: MaxApiClient, chat_id: int, target_user_id: int) -> None:
    async with async_session_maker() as session:
        user = await session.get(User, target_user_id)
        topics = (await session.execute(select(Topic))).scalars().all()
        topic_map = {t.id: t.name for t in topics}
        messages = (await session.execute(
            select(DBMessage).where(DBMessage.user_id == target_user_id).order_by(DBMessage.timestamp.asc())
        )).scalars().all()

    if not messages:
        await client.send_message(chat_id=chat_id, text="История пуста.", attachments=admin_client_profile_keyboard(target_user_id))
        return

    lines = [f"История: {user.name or user.first_name or str(target_user_id)} (ID: {target_user_id})", "=" * 40]
    for m in messages:
        dt = m.timestamp.strftime('%Y-%m-%d %H:%M')
        topic = topic_map.get(m.topic_id, "Общий")
        role = "Клиент" if m.role == "user" else "Бот"
        lines.append(f"[{dt}] [{topic}] {role}:\n{m.content or ''}\n")
    full_text = "\n".join(lines)

    chunks = split_text(f"<pre>{html.escape(full_text[:50000])}</pre>", 4000)
    for chunk in chunks[:3]:
        await client.send_message(chat_id=chat_id, text=chunk)
    await client.send_message(
        chat_id=chat_id,
        text=f"📋 История пользователя <code>{target_user_id}</code> выведена выше ({len(messages)} сообщений).",
        attachments=admin_client_profile_keyboard(target_user_id)
    )


async def confirm_delete_history(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, target_user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_delete_history_confirm", {"target_user_id": target_user_id})
    await client.send_message(
        chat_id=chat_id,
        text=f"⚠️ Удалить всю историю пользователя <code>{target_user_id}</code>? Действие необратимо.",
        attachments=inline_keyboard([
            [callback_button("✅ Да, удалить", f"admin_delete_history_confirmed_{target_user_id}")],
            [callback_button("❌ Отмена", f"view_client_{target_user_id}")],
        ])
    )


async def delete_history_confirmed(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, target_user_id: int) -> None:
    from sqlalchemy import delete as sql_delete
    async with async_session_maker() as session:
        await session.execute(sql_delete(DBMessage).where(DBMessage.user_id == target_user_id))
        await session.commit()
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text=f"✅ История пользователя <code>{target_user_id}</code> удалена.")
    await show_client_profile(client, chat_id, target_user_id)


async def show_client_payment_info(client: MaxApiClient, chat_id: int, target_user_id: int) -> None:
    async with async_session_maker() as session:
        user = await session.get(User, target_user_id)
        robo_payments = (await session.execute(
            select(RobokassaPayment).where(RobokassaPayment.user_id == target_user_id).order_by(RobokassaPayment.created_at.desc()).limit(10)
        )).scalars().all()
        yoo_payments = (await session.execute(
            select(YookassaPayment).where(YookassaPayment.user_id == target_user_id).order_by(YookassaPayment.created_at.desc()).limit(10)
        )).scalars().all()
        total_robo = (await session.execute(select(func.sum(RobokassaPayment.amount)).where(RobokassaPayment.user_id == target_user_id))).scalar() or 0.0
        total_yoo = (await session.execute(select(func.sum(YookassaPayment.amount)).where(YookassaPayment.user_id == target_user_id))).scalar() or 0.0

    name = user.name or user.first_name or str(target_user_id) if user else str(target_user_id)
    text = (
        f"<b>💳 Платежи клиента {html.escape(name)}</b>\n\n"
        f"Итого Robokassa: {total_robo:.2f} руб.\n"
        f"Итого YooKassa: {total_yoo:.2f} руб.\n\n"
    )
    if robo_payments or yoo_payments:
        text += "<b>Последние платежи:</b>\n"
        all_payments = []
        for r in robo_payments:
            all_payments.append((r.created_at, "Robo", r.amount, r.status))
        for y in yoo_payments:
            all_payments.append((y.created_at, "Yoo", y.amount, y.status))
        all_payments.sort(key=lambda x: x[0] or datetime.min, reverse=True)
        for dt, src, amount, status in all_payments[:10]:
            dt_str = dt.strftime('%d.%m.%Y %H:%M') if dt else "?"
            text += f"<code>{dt_str}</code> {src}: {amount:.2f}₽ ({html.escape(status or '?')})\n"
    else:
        text += "Платежи не найдены.\n"

    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=inline_keyboard([[callback_button("◀️ Назад", f"view_client_{target_user_id}")]])
    )


async def reset_subscription(client: MaxApiClient, chat_id: int, target_user_id: int) -> None:
    await client.send_message(
        chat_id=chat_id,
        text=f"⚠️ Сбросить подписку пользователя <code>{target_user_id}</code>?",
        attachments=inline_keyboard([
            [callback_button("✅ Да, сбросить", f"admin_reset_sub_confirmed_{target_user_id}")],
            [callback_button("❌ Отмена", f"view_client_{target_user_id}")],
        ])
    )


async def reset_subscription_confirmed(client: MaxApiClient, chat_id: int, target_user_id: int) -> None:
    from sqlalchemy import delete as sql_delete
    async with async_session_maker() as session:
        await session.execute(sql_delete(UserSubscription).where(UserSubscription.user_id == target_user_id))
        await session.commit()
    await client.send_message(chat_id=chat_id, text=f"✅ Подписка пользователя <code>{target_user_id}</code> сброшена.")
    await show_client_profile(client, chat_id, target_user_id)
