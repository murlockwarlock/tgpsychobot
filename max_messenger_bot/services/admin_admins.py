from __future__ import annotations

import html
import os

from sqlalchemy import select

from ..api import MaxApiClient
from ..keyboards import admin_admins_keyboard, admin_profile_keyboard
from ..legacy import User, async_session_maker
from ..storage import StateStore


def _owner_ids() -> set[int]:
    raw = os.getenv("OWNER_IDS", "")
    values: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.add(int(part))
        except ValueError:
            continue
    return values


def is_owner(user_id: int) -> bool:
    return user_id in _owner_ids()


async def show_admins(client: MaxApiClient, chat_id: int) -> None:
    owners = sorted(_owner_ids())
    async with async_session_maker() as session:
        admins = (await session.execute(select(User).where(User.is_admin == True).order_by(User.first_name.asc(), User.id.asc()))).scalars().all()
    db_admins = [admin for admin in admins if admin.id not in owners]
    lines = ["👮 <b>Управление администраторами</b><br/><br/>", "<b>Владельцы:</b><br/>"]
    if owners:
        for owner_id in owners:
            lines.append(f"• <code>{owner_id}</code><br/>")
    else:
        lines.append("не заданы<br/>")
    lines.append("<br/><b>Администраторы из БД:</b><br/>")
    if not db_admins:
        lines.append("список пуст")
    text = "".join(lines)
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_admins_keyboard(db_admins))


async def start_add_admin(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    if not is_owner(user_id):
        await client.send_message(chat_id=chat_id, text="Только владелец может назначать администраторов.")
        return
    await states.set(user_id, chat_id, "admin_add_admin_id", {})
    await client.send_message(chat_id=chat_id, text="Введите ID пользователя, которого нужно назначить администратором.")


async def add_admin(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    if not is_owner(user_id):
        await client.send_message(chat_id=chat_id, text="Только владелец может назначать администраторов.")
        return
    try:
        target_user_id = int(text.strip())
    except ValueError:
        await client.send_message(chat_id=chat_id, text="ID должен быть числом.")
        return
    async with async_session_maker() as session:
        target = await session.get(User, target_user_id)
        if not target:
            await client.send_message(chat_id=chat_id, text="Пользователь с таким ID не найден в базе.")
            return
        target.is_admin = True
        await session.commit()
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text=f"✅ Пользователь <code>{target_user_id}</code> назначен администратором.")
    await show_admins(client, chat_id)


async def show_admin_profile(client: MaxApiClient, chat_id: int, viewer_id: int, admin_id: int) -> None:
    async with async_session_maker() as session:
        admin = await session.get(User, admin_id)
    if not admin:
        await client.send_message(chat_id=chat_id, text="Администратор не найден.")
        return
    owner = is_owner(viewer_id)
    text = (
        "<b>Профиль администратора</b><br/><br/>"
        f"<b>ID:</b> <code>{admin.id}</code><br/>"
        f"<b>Имя:</b> {html.escape(admin.first_name or admin.name or 'Не указано')}<br/>"
        f"<b>Username:</b> {html.escape(admin.username or 'Не указан')}<br/>"
        f"<b>Доступ к истории:</b> {'да' if admin.can_view_history else 'нет'}<br/>"
        f"<b>Владелец:</b> {'да' if admin.id in _owner_ids() else 'нет'}"
    )
    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=admin_profile_keyboard(admin.id, bool(admin.can_view_history), owner and admin.id not in _owner_ids() and admin.id != viewer_id),
    )


async def toggle_history_access(client: MaxApiClient, chat_id: int, viewer_id: int, admin_id: int) -> None:
    if not is_owner(viewer_id):
        await client.send_message(chat_id=chat_id, text="Только владелец может менять доступ к истории.")
        return
    async with async_session_maker() as session:
        admin = await session.get(User, admin_id)
        if not admin:
            await client.send_message(chat_id=chat_id, text="Администратор не найден.")
            return
        admin.can_view_history = not bool(admin.can_view_history)
        await session.commit()
    await show_admin_profile(client, chat_id, viewer_id, admin_id)


async def revoke_admin(client: MaxApiClient, chat_id: int, viewer_id: int, admin_id: int) -> None:
    if not is_owner(viewer_id):
        await client.send_message(chat_id=chat_id, text="Только владелец может отзывать права.")
        return
    if admin_id == viewer_id:
        await client.send_message(chat_id=chat_id, text="Нельзя разжаловать самого себя.")
        return
    if admin_id in _owner_ids():
        await client.send_message(chat_id=chat_id, text="Нельзя разжаловать владельца из конфига.")
        return
    async with async_session_maker() as session:
        admin = await session.get(User, admin_id)
        if not admin:
            await client.send_message(chat_id=chat_id, text="Администратор не найден.")
            return
        admin.is_admin = False
        admin.can_view_history = False
        await session.commit()
    await client.send_message(chat_id=chat_id, text=f"✅ Права администратора для <code>{admin_id}</code> отозваны.")
    await show_admins(client, chat_id)
