from __future__ import annotations

import html
import math

from sqlalchemy import func, select

from ..api import MaxApiClient
from ..keyboards import callback_button, inline_keyboard
from ..legacy import (
    MediaCollection,
    MediaLibrary,
    Topic,
    async_session_maker,
    media_collection_items,
    topic_collection_association,
)
from ..storage import StateStore

COLL_PAGE_SIZE = 8
COLL_FILES_PAGE_SIZE = 10


# ---------------------------------------------------------------------------
# Keyboards
# ---------------------------------------------------------------------------

def _list_keyboard(colls: list[dict], page: int, total_pages: int) -> list[dict]:
    rows = [
        [callback_button(f"📁 {c['name']} ({c['count']})", f"admin_coll_view_{c['id']}")]
        for c in colls
    ]
    nav: list[dict] = []
    if page > 0:
        nav.append(callback_button("◀️", f"admin_collections_page_{page - 1}"))
    nav.append(callback_button(f"{page + 1}/{total_pages}", "noop"))
    if page < total_pages - 1:
        nav.append(callback_button("▶️", f"admin_collections_page_{page + 1}"))
    if nav:
        rows.append(nav)
    rows.append([callback_button("➕ Создать коллекцию", "admin_coll_create")])
    rows.append([callback_button("◀️ Назад", "admin_panel")])
    return inline_keyboard(rows)


def _detail_keyboard(coll_id: int) -> list[dict]:
    return inline_keyboard([
        [callback_button("📎 Файлы коллекции", f"admin_coll_files_{coll_id}_0")],
        [callback_button("✏️ Переименовать", f"admin_coll_rename_{coll_id}")],
        [callback_button("🗑 Удалить", f"admin_coll_delete_{coll_id}")],
        [callback_button("◀️ Назад", "admin_collections_page_0")],
    ])


def _files_keyboard(
    coll_id: int,
    page_media: list,
    assigned_ids: set[int],
    page: int,
    total_pages: int,
) -> list[dict]:
    rows = []
    for m in page_media:
        check = "✅" if m.id in assigned_ids else "☐"
        action = "remove" if m.id in assigned_ids else "add"
        rows.append([
            callback_button(
                f"{check} {m.file_name}",
                f"coll_file_{action}_{coll_id}_{m.id}_{page}",
            )
        ])
    nav: list[dict] = []
    if page > 0:
        nav.append(callback_button("◀️", f"admin_coll_files_{coll_id}_{page - 1}"))
    nav.append(callback_button(f"{page + 1}/{total_pages}", "noop"))
    if page < total_pages - 1:
        nav.append(callback_button("▶️", f"admin_coll_files_{coll_id}_{page + 1}"))
    if nav:
        rows.append(nav)
    rows.append([callback_button("◀️ Назад", f"admin_coll_view_{coll_id}")])
    return inline_keyboard(rows)


# ---------------------------------------------------------------------------
# Service functions
# ---------------------------------------------------------------------------

async def show_list(client: MaxApiClient, chat_id: int, page: int = 0) -> None:
    async with async_session_maker() as session:
        stmt = (
            select(MediaCollection.id, MediaCollection.name, func.count(media_collection_items.c.media_id))
            .outerjoin(media_collection_items, media_collection_items.c.collection_id == MediaCollection.id)
            .group_by(MediaCollection.id)
            .order_by(MediaCollection.name)
        )
        res = await session.execute(stmt)
        all_colls = [{"id": r[0], "name": r[1], "count": r[2]} for r in res.all()]

    total = len(all_colls)
    total_pages = max(1, math.ceil(total / COLL_PAGE_SIZE))
    page = max(0, min(page, total_pages - 1))
    page_colls = all_colls[page * COLL_PAGE_SIZE: (page + 1) * COLL_PAGE_SIZE]

    text = f"🎨 <b>Медиа-коллекции</b> ({total})<br>Нажмите на коллекцию для управления."
    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=_list_keyboard(page_colls, page, total_pages),
    )


async def show_detail(client: MaxApiClient, chat_id: int, coll_id: int) -> None:
    async with async_session_maker() as session:
        coll = await session.get(MediaCollection, coll_id)
        if not coll:
            await client.send_message(chat_id=chat_id, text="Коллекция не найдена.")
            return
        count_res = await session.execute(
            select(func.count()).select_from(media_collection_items).where(
                media_collection_items.c.collection_id == coll_id
            )
        )
        file_count = count_res.scalar() or 0
        topics_res = await session.execute(
            select(Topic.name)
            .join(topic_collection_association, topic_collection_association.c.topic_id == Topic.id)
            .where(topic_collection_association.c.collection_id == coll_id)
        )
        topic_names = [r[0] for r in topics_res.all()]

    topics_text = ", ".join(html.escape(n) for n in topic_names) if topic_names else "нет"
    text = (
        f"📂 <b>{html.escape(coll.name)}</b><br><br>"
        f"Файлов: {file_count}<br>"
        f"Привязана к темам: {topics_text}"
    )
    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=_detail_keyboard(coll_id),
    )


async def start_create(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
) -> None:
    await states.set(user_id, chat_id, "admin_coll_creating", {})
    await client.send_message(
        chat_id=chat_id,
        text="Введите название для новой коллекции:",
        attachments=inline_keyboard([
            [callback_button("❌ Отмена", "admin_collections_page_0")]
        ]),
    )


async def save_create(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str,
) -> None:
    name = text.strip()
    if not name:
        await client.send_message(chat_id=chat_id, text="Название не может быть пустым. Попробуйте ещё раз:")
        return

    async with async_session_maker() as session:
        existing = await session.execute(
            select(MediaCollection).where(MediaCollection.name == name)
        )
        if existing.scalar_one_or_none():
            await client.send_message(
                chat_id=chat_id,
                text=f"Коллекция «{html.escape(name)}» уже существует. Введите другое название:",
            )
            return
        coll = MediaCollection(name=name)
        session.add(coll)
        await session.commit()
        coll_id = coll.id

    await states.clear(user_id)
    await client.send_message(
        chat_id=chat_id,
        text=f"✅ Коллекция «{html.escape(name)}» создана.",
        attachments=_detail_keyboard(coll_id),
    )


async def start_rename(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    coll_id: int,
) -> None:
    await states.set(user_id, chat_id, "admin_coll_renaming", {"coll_id": coll_id})
    await client.send_message(
        chat_id=chat_id,
        text="Введите новое название коллекции:",
        attachments=inline_keyboard([
            [callback_button("❌ Отмена", f"admin_coll_view_{coll_id}")]
        ]),
    )


async def save_rename(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str,
) -> None:
    snapshot = await states.get(user_id)
    coll_id = snapshot.data.get("coll_id") if snapshot and snapshot.data else None
    if not coll_id:
        await client.send_message(chat_id=chat_id, text="Ошибка: не найден ID коллекции.")
        return

    name = text.strip()
    if not name:
        await client.send_message(chat_id=chat_id, text="Название не может быть пустым.")
        return

    async with async_session_maker() as session:
        existing = await session.execute(
            select(MediaCollection).where(
                MediaCollection.name == name,
                MediaCollection.id != coll_id,
            )
        )
        if existing.scalar_one_or_none():
            await client.send_message(
                chat_id=chat_id,
                text=f"Коллекция «{html.escape(name)}» уже существует.",
            )
            return
        coll = await session.get(MediaCollection, coll_id)
        if coll:
            coll.name = name
            await session.commit()

    await states.clear(user_id)
    await client.send_message(
        chat_id=chat_id,
        text=f"✅ Коллекция переименована в «{html.escape(name)}».",
        attachments=_detail_keyboard(coll_id),
    )


async def delete(client: MaxApiClient, chat_id: int, coll_id: int) -> None:
    async with async_session_maker() as session:
        coll = await session.get(MediaCollection, coll_id)
        if coll:
            await session.delete(coll)
            await session.commit()
    await show_list(client, chat_id, page=0)


async def show_files(
    client: MaxApiClient,
    chat_id: int,
    coll_id: int,
    page: int = 0,
) -> None:
    async with async_session_maker() as session:
        coll = await session.get(MediaCollection, coll_id)
        if not coll:
            await client.send_message(chat_id=chat_id, text="Коллекция не найдена.")
            return
        assigned_res = await session.execute(
            select(media_collection_items.c.media_id).where(
                media_collection_items.c.collection_id == coll_id
            )
        )
        assigned_ids: set[int] = {r[0] for r in assigned_res.all()}
        all_media_res = await session.execute(
            select(MediaLibrary).order_by(MediaLibrary.file_name, MediaLibrary.id)
        )
        all_media = list(all_media_res.scalars().all())

    total = len(all_media)
    total_pages = max(1, math.ceil(total / COLL_FILES_PAGE_SIZE))
    page = max(0, min(page, total_pages - 1))
    page_media = all_media[page * COLL_FILES_PAGE_SIZE: (page + 1) * COLL_FILES_PAGE_SIZE]

    text = (
        f"📎 Файлы коллекции «{html.escape(coll.name)}» "
        f"(отмечено {len(assigned_ids)} из {total})<br>"
        f"Нажмите, чтобы добавить/убрать файл."
    )
    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=_files_keyboard(coll_id, page_media, assigned_ids, page, total_pages),
    )


async def toggle_file(
    client: MaxApiClient,
    chat_id: int,
    action: str,
    coll_id: int,
    media_id: int,
    page: int,
) -> None:
    async with async_session_maker() as session:
        if action == "add":
            try:
                await session.execute(
                    media_collection_items.insert().values(
                        collection_id=coll_id, media_id=media_id
                    )
                )
                await session.commit()
            except Exception:
                await session.rollback()
        elif action == "remove":
            await session.execute(
                media_collection_items.delete().where(
                    media_collection_items.c.collection_id == coll_id,
                    media_collection_items.c.media_id == media_id,
                )
            )
            await session.commit()

    await show_files(client, chat_id, coll_id, page)
