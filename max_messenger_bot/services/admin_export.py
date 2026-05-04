from __future__ import annotations

import csv
import io
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from ..api import MaxApiClient
from ..keyboards import admin_date_filter_keyboard, admin_export_keyboard
from ..legacy import Message as DBMessage, User, async_session_maker
from ..storage import StateStore


async def show_export_menu(client: MaxApiClient, chat_id: int) -> None:
    await client.send_message(
        chat_id=chat_id,
        text="📤 <b>Экспорт данных</b><br><br>Выберите, что экспортировать:",
        attachments=admin_export_keyboard(),
    )


async def show_date_filter_menu(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
) -> None:
    await states.set(user_id, chat_id, "admin_export_date_filter", {})
    await client.send_message(
        chat_id=chat_id,
        text="🗓 <b>Фильтр по датам</b><br><br>Выберите диапазон для экспорта сообщений:",
        attachments=admin_date_filter_keyboard(),
    )


async def set_date_preset(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    days: int,
) -> None:
    if days == 0:
        date_from_str = None
        label = "все даты"
    else:
        date_from_str = (datetime.now() - timedelta(days=days)).strftime("%d-%m-%Y")
        label = f"последние {days} дней (с {date_from_str})"
    await client.send_message(
        chat_id=chat_id,
        text=f"⏳ Экспортирую сообщения: {label}...",
    )
    await run_export(client, chat_id, date_from_str=date_from_str)


async def start_date_manual_from(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
) -> None:
    await states.set(user_id, chat_id, "admin_export_date_from", {})
    await client.send_message(
        chat_id=chat_id,
        text=(
            "✏️ Введите дату начала в формате <b>ДД-ММ-ГГГГ</b><br>"
            "Пример: <b>01-09-2025</b><br><br>"
            "Или отправьте <b>0</b>, чтобы не ограничивать начало."
        ),
    )


async def save_date_from(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str,
) -> None:
    text = text.strip()
    if text == "0":
        date_from_str = None
    else:
        try:
            datetime.strptime(text, "%d-%m-%Y")
            date_from_str = text
        except ValueError:
            await client.send_message(
                chat_id=chat_id,
                text="❌ Неверный формат. Введите дату как <b>ДД-ММ-ГГГГ</b>, например <b>01-09-2025</b>",
            )
            return
    await states.set(user_id, chat_id, "admin_export_date_to", {"date_from": date_from_str})
    await client.send_message(
        chat_id=chat_id,
        text=(
            "✏️ Теперь введите дату окончания в формате <b>ДД-ММ-ГГГГ</b><br>"
            "Пример: <b>31-12-2025</b><br><br>"
            "Или отправьте <b>0</b>, чтобы не ограничивать конец (до сегодня)."
        ),
    )


async def save_date_to(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str,
) -> None:
    snapshot = await states.get(user_id)
    data = snapshot.data if snapshot else {}
    date_from_str = data.get("date_from")

    text = text.strip()
    if text == "0":
        date_to_str = None
    else:
        try:
            datetime.strptime(text, "%d-%m-%Y")
            date_to_str = text
        except ValueError:
            await client.send_message(
                chat_id=chat_id,
                text="❌ Неверный формат. Введите дату как <b>ДД-ММ-ГГГГ</b>, например <b>31-12-2025</b>",
            )
            return

    await client.send_message(chat_id=chat_id, text="⏳ Экспортирую сообщения...")
    await run_export(client, chat_id, date_from_str=date_from_str, date_to_str=date_to_str)


async def run_export(
    client: MaxApiClient,
    chat_id: int,
    date_from_str: str | None = None,
    date_to_str: str | None = None,
    anonymize: bool = False,
) -> None:
    date_from: datetime | None = None
    date_to: datetime | None = None
    if date_from_str:
        date_from = datetime.strptime(date_from_str, "%d-%m-%Y")
    if date_to_str:
        date_to = datetime.strptime(date_to_str, "%d-%m-%Y").replace(hour=23, minute=59, second=59)

    async with async_session_maker() as session:
        stmt = (
            select(DBMessage, User)
            .join(User, DBMessage.user_id == User.id)
            .order_by(DBMessage.timestamp.asc())
        )
        if date_from:
            stmt = stmt.where(DBMessage.timestamp >= date_from)
        if date_to:
            stmt = stmt.where(DBMessage.timestamp <= date_to)
        rows = (await session.execute(stmt)).all()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["user_id", "user_name", "role", "content", "created_at"])
    for msg, user in rows:
        user_id_val = "***" if anonymize else msg.user_id
        user_name = "***" if anonymize else (user.name or user.first_name or user.username or "")
        writer.writerow([
            user_id_val,
            user_name,
            msg.role,
            (msg.content or "").replace("\n", " "),
            msg.timestamp.strftime("%Y-%m-%d %H:%M:%S") if msg.timestamp else "",
        ])

    csv_bytes = buf.getvalue().encode("utf-8-sig")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    date_label = ""
    if date_from_str or date_to_str:
        date_label = f" ({date_from_str or 'начало'} — {date_to_str or 'сегодня'})"

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(csv_bytes)
        tmp_path = tmp.name

    try:
        result = await client.upload_file("file", tmp_path)
        token = result.get("token") or result.get("fileId")
        if token:
            await client.send_media_attachment(
                chat_id=chat_id,
                media_type="file",
                token=token,
                caption=f"💬 Сообщения{date_label}: {len(rows)} записей",
            )
        else:
            await client.send_message(
                chat_id=chat_id,
                text=f"Экспорт выполнен, но токен файла не получен. Строк: {len(rows)}",
            )
    except Exception as exc:
        await client.send_message(
            chat_id=chat_id,
            text=f"Ошибка при отправке файла: {exc}<br><br>Записей: {len(rows)}",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def export_users_csv(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        users = (await session.execute(select(User).order_by(User.created_at.asc()))).scalars().all()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "first_name", "name", "username", "created_at", "has_subscription"])
    for u in users:
        writer.writerow([
            u.id,
            u.first_name or "",
            u.name or "",
            u.username or "",
            u.created_at.strftime("%Y-%m-%d %H:%M:%S") if u.created_at else "",
            bool(u.subscription),
        ])

    csv_bytes = buf.getvalue().encode("utf-8-sig")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(csv_bytes)
        tmp_path = tmp.name

    try:
        result = await client.upload_file("file", tmp_path)
        token = result.get("token") or result.get("fileId")
        if token:
            await client.send_media_attachment(
                chat_id=chat_id,
                media_type="file",
                token=token,
                caption=f"👥 Пользователи: {len(users)} записей",
            )
        else:
            await client.send_message(
                chat_id=chat_id,
                text=f"Экспорт выполнен, но токен файла не получен. Строк: {len(users)}",
            )
    except Exception as exc:
        await client.send_message(
            chat_id=chat_id,
            text=f"Ошибка при отправке файла: {exc}<br><br>Записей в файле: {len(users)}",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def export_messages_csv(
    client: MaxApiClient,
    chat_id: int,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> None:
    date_from_str = date_from.strftime("%d-%m-%Y") if date_from else None
    date_to_str = date_to.strftime("%d-%m-%Y") if date_to else None
    await run_export(client, chat_id, date_from_str=date_from_str, date_to_str=date_to_str)

