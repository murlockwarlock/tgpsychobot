from __future__ import annotations

import csv
import io
import json
import math
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import func, select

from ..api import MaxApiClient
from ..keyboards import admin_clients_keyboard, admin_date_filter_keyboard, admin_export_keyboard, mass_export_options_keyboard
from ..legacy import Message as DBMessage, User, async_session_maker
from ..models import MAX_ID_OFFSET
from ..storage import StateStore
from ..time_utils import format_msk


PAGE_SIZE = 10
MAX_DIRECT_FILE_SIZE = 50 * 1024 * 1024


async def show_export_menu(client: MaxApiClient, chat_id: int) -> None:
    await client.send_message(
        chat_id=chat_id,
        text="📤 <b>Экспорт данных</b>\n\nВыберите, что экспортировать:",
        attachments=admin_export_keyboard(),
    )


async def start_export_mode(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
) -> None:
    await states.set(user_id, chat_id, "admin_selecting_for_export", {"selected_export_ids": []})
    await show_export_clients(client, states, chat_id, user_id, 0)


async def show_export_clients(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    page: int = 0,
) -> None:
    snapshot = await states.get(user_id)
    data = snapshot.data if snapshot else {}
    selected_ids = [int(x) for x in data.get("selected_export_ids", [])]

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

    text = f"📦 Выберите клиентов для экспорта ({len(selected_ids)} выбрано)\n\nСтраница {page + 1}/{total_pages}"
    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=admin_clients_keyboard(page, total_pages, clients, export_mode=True, selected_ids=selected_ids),
    )


async def toggle_export_selection(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    target_user_id: int,
    page: int,
) -> None:
    snapshot = await states.get(user_id)
    data = snapshot.data if snapshot else {}
    selected = {int(x) for x in data.get("selected_export_ids", [])}
    if target_user_id in selected:
        selected.remove(target_user_id)
    else:
        selected.add(target_user_id)
    await states.update(user_id, selected_export_ids=sorted(selected))
    await show_export_clients(client, states, chat_id, user_id, page)


async def select_all_no_admins(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
) -> None:
    async with async_session_maker() as session:
        ids = (
            await session.execute(
                select(User.id)
                .where(User.id >= MAX_ID_OFFSET, User.is_admin == False)
                .order_by(User.id.asc())
            )
        ).scalars().all()
    await states.update(user_id, selected_export_ids=[int(x) for x in ids])
    await show_export_clients(client, states, chat_id, user_id, 0)


async def confirm_all_export(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
) -> None:
    await states.set(user_id, chat_id, "admin_export_date_filter", {"export_all": True, "selected_export_ids": []})
    await show_date_filter_menu(client, states, chat_id, user_id)


async def confirm_selected_export(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
) -> None:
    snapshot = await states.get(user_id)
    data = snapshot.data if snapshot else {}
    selected_ids = [int(x) for x in data.get("selected_export_ids", [])]
    if not selected_ids:
        await client.send_message(chat_id=chat_id, text="❌ Сначала выберите хотя бы одного клиента.")
        return
    await states.set(
        user_id,
        chat_id,
        "admin_export_date_filter",
        {"export_all": False, "selected_export_ids": selected_ids},
    )
    await show_date_filter_menu(client, states, chat_id, user_id)


async def show_date_filter_menu(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
) -> None:
    await client.send_message(
        chat_id=chat_id,
        text="🗓 <b>Фильтр по датам</b>\n\nВыберите диапазон для экспорта сообщений:",
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
    await states.update(user_id, date_from=date_from_str, date_to=None)
    await show_export_format_options(client, states, chat_id, user_id, f"✅ Фильтр дат: {label}")


async def start_date_manual_from(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
) -> None:
    snapshot = await states.get(user_id)
    data = snapshot.data if snapshot else {}
    await states.set(user_id, chat_id, "admin_export_date_from", data)
    await client.send_message(
        chat_id=chat_id,
        text=(
            "✏️ Введите дату начала в формате <b>ДД-ММ-ГГГГ</b>\n"
            "Пример: <b>01-09-2025</b>\n\n"
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
    snapshot = await states.get(user_id)
    data = snapshot.data if snapshot else {}
    data["date_from"] = date_from_str
    await states.set(user_id, chat_id, "admin_export_date_to", data)
    await client.send_message(
        chat_id=chat_id,
        text=(
            "✏️ Теперь введите дату окончания в формате <b>ДД-ММ-ГГГГ</b>\n"
            "Пример: <b>31-12-2025</b>\n\n"
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

    await states.update(user_id, date_to=date_to_str)
    label = f"✅ Фильтр дат: {date_from_str or 'начало'} — {date_to_str or 'сегодня'}"
    await show_export_format_options(client, states, chat_id, user_id, label)


async def show_export_format_options(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    prefix: str = "✅ Фильтр дат выбран",
) -> None:
    snapshot = await states.get(user_id)
    data = snapshot.data if snapshot else {}
    count_label = "всех клиентов" if data.get("export_all") else f"{len(data.get('selected_export_ids', []))} выбранных"
    await client.send_message(
        chat_id=chat_id,
        text=f"{prefix}\n\nВыберите формат экспорта для {count_label}:",
        attachments=mass_export_options_keyboard(),
    )


def _parse_date(date_str: str | None, end_of_day: bool = False) -> datetime | None:
    if not date_str:
        return None
    parsed = datetime.strptime(date_str, "%d-%m-%Y")
    if end_of_day:
        return parsed.replace(hour=23, minute=59, second=59)
    return parsed


def _user_label(user: User, anonymize: bool, index: int) -> str:
    if anonymize:
        return f"user_{index}"
    name = user.name or user.first_name or "Без имени"
    username = f", @{user.username}" if user.username else ""
    return f"{name} (ID: {user.id}{username})"


def _date_label(date_from_str: str | None, date_to_str: str | None) -> str:
    if not date_from_str and not date_to_str:
        return ""
    return f" | фильтр: {date_from_str or 'начало'} → {date_to_str or 'сегодня'}"


async def run_mass_export(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    fmt: str,
    anonymize: bool,
) -> None:
    snapshot = await states.get(user_id)
    data = snapshot.data if snapshot else {}
    export_all = bool(data.get("export_all"))
    selected_ids = [int(x) for x in data.get("selected_export_ids", [])]
    date_from_str = data.get("date_from")
    date_to_str = data.get("date_to")
    date_from = _parse_date(date_from_str)
    date_to = _parse_date(date_to_str, end_of_day=True)

    if fmt not in {"txt", "json"}:
        await client.send_message(chat_id=chat_id, text="❌ Неизвестный формат экспорта.")
        return
    if not export_all and not selected_ids:
        await client.send_message(chat_id=chat_id, text="❌ Нет выбранных клиентов для экспорта.")
        return

    await client.send_message(chat_id=chat_id, text="⏳ Экспортирую историю...")

    async with async_session_maker() as session:
        users_stmt = select(User).where(User.id >= MAX_ID_OFFSET)
        if not export_all:
            users_stmt = users_stmt.where(User.id.in_(selected_ids))
        users = (await session.execute(users_stmt.order_by(User.id.asc()))).scalars().all()
        user_ids = [user.id for user in users]

        messages_by_user: dict[int, list[DBMessage]] = {uid: [] for uid in user_ids}
        if user_ids:
            msg_stmt = select(DBMessage).where(DBMessage.user_id.in_(user_ids))
            if date_from:
                msg_stmt = msg_stmt.where(DBMessage.timestamp >= date_from)
            if date_to:
                msg_stmt = msg_stmt.where(DBMessage.timestamp <= date_to)
            messages = (await session.execute(msg_stmt.order_by(DBMessage.user_id.asc(), DBMessage.timestamp.asc()))).scalars().all()
            for message in messages:
                messages_by_user.setdefault(message.user_id, []).append(message)

        topic_rows = (await session.execute(select(DBMessage.topic_id).distinct())).scalars().all()
        topic_ids = [tid for tid in topic_rows if tid is not None]
        topic_map: dict[int, str] = {}
        if topic_ids:
            from ..legacy import Topic

            topics = (await session.execute(select(Topic).where(Topic.id.in_(topic_ids)))).scalars().all()
            topic_map = {topic.id: topic.name for topic in topics}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    date_suffix = _date_label(date_from_str, date_to_str)
    if fmt == "txt":
        content = _build_txt_export(users, messages_by_user, topic_map, anonymize, timestamp, date_suffix)
        file_bytes = content.encode("utf-8")
        suffix = ".txt"
    else:
        content_obj = _build_json_export(users, messages_by_user, topic_map, anonymize)
        file_bytes = json.dumps(content_obj, ensure_ascii=False, indent=2).encode("utf-8")
        suffix = ".json"

    base_name = f"mass_export_{timestamp}{suffix}"
    await _send_export_file(
        client,
        chat_id,
        file_bytes,
        base_name,
        f"📥 Экспорт готов: {len(users)} клиент(ов), {sum(len(v) for v in messages_by_user.values())} сообщений",
    )


def _build_txt_export(
    users: list[User],
    messages_by_user: dict[int, list[DBMessage]],
    topic_map: dict[int, str],
    anonymize: bool,
    timestamp: str,
    date_suffix: str,
) -> str:
    lines = [f"MASS EXPORT - {timestamp}{date_suffix}", "=" * 60, ""]
    for index, user in enumerate(users, 1):
        messages = messages_by_user.get(user.id, [])
        if not messages:
            continue
        lines.extend([f"ДАННЫЕ КЛИЕНТА: {_user_label(user, anonymize, index)}", "-" * 40])
        for message in messages:
            topic = topic_map.get(message.topic_id, "General")
            role = "Client" if message.role == "user" else "Bot"
            timestamp = format_msk(message.timestamp, "%Y-%m-%d %H:%M МСК") if message.timestamp else ""
            lines.append(f"[{timestamp}] [{topic}] {role}: {message.content or ''}")
        lines.extend(["", "=" * 60, ""])
    if len(lines) == 3:
        lines.append("Нет сообщений по выбранным условиям.")
    return "\n".join(lines)


def _build_json_export(
    users: list[User],
    messages_by_user: dict[int, list[DBMessage]],
    topic_map: dict[int, str],
    anonymize: bool,
) -> list[dict]:
    export_data: list[dict] = []
    for index, user in enumerate(users, 1):
        messages = messages_by_user.get(user.id, [])
        if not messages:
            continue
        user_label = f"user_{index}" if anonymize else str(user.id)
        export_data.append(
            {
                "user_info": {
                    "label": user_label,
                    "id": None if anonymize else user.id,
                    "name": None if anonymize else (user.name or user.first_name),
                    "username": None if anonymize else user.username,
                },
                "history": [
                    {
                        "timestamp": format_msk(message.timestamp, "%Y-%m-%dT%H:%M:%S+03:00") if message.timestamp else None,
                        "topic": topic_map.get(message.topic_id, "General"),
                        "role": message.role,
                        "content": message.content or "",
                    }
                    for message in messages
                ],
            }
        )
    return export_data


async def _send_export_file(
    client: MaxApiClient,
    chat_id: int,
    file_bytes: bytes,
    filename: str,
    caption: str,
) -> None:
    tmp_path = ""
    try:
        data = file_bytes
        upload_name = filename
        suffix = Path(filename).suffix
        if len(file_bytes) > MAX_DIRECT_FILE_SIZE:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(filename, file_bytes)
            data = zip_buffer.getvalue()
            upload_name = f"{Path(filename).stem}.zip"
            suffix = ".zip"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        result = await client.upload_file("file", tmp_path)
        token = result.get("token") or result.get("fileId")
        if token:
            await client.send_media_attachment(
                chat_id=chat_id,
                media_type="file",
                token=token,
                caption=caption if upload_name == filename else f"{caption}\nФайл сжат в ZIP.",
            )
        else:
            await client.send_message(chat_id=chat_id, text=f"Экспорт выполнен, но токен файла не получен.")
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"Ошибка при отправке файла: {exc}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)



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
            text=f"Ошибка при отправке файла: {exc}\n\nЗаписей: {len(rows)}",
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
            text=f"Ошибка при отправке файла: {exc}\n\nЗаписей в файле: {len(users)}",
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
