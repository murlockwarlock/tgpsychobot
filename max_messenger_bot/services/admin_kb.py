from __future__ import annotations

import html
import io
import logging
import math

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from ..api import MaxApiClient
from ..keyboards import (
    admin_kb_editor_keyboard,
    admin_kb_list_keyboard,
    admin_topic_kb_keyboard,
    admin_kb_finish_upload_keyboard,
)
from ..legacy import KnowledgeBase, Topic, async_session_maker
from ..storage import StateStore
from ..models import IncomingMessage
from vector_store import update_vector_index, delete_document_vectors

log = logging.getLogger("max_messenger_bot.services.admin_kb")

PAGE_SIZE = 8


def _content_preview(value: str | None, limit: int = 2200) -> str:
    text = (value or "").strip()
    if not text:
        return "Содержимое не задано."
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


async def list_entries(client: MaxApiClient, chat_id: int, page: int = 0) -> None:
    async with async_session_maker() as session:
        total = await session.scalar(select(func.count()).select_from(KnowledgeBase)) or 0
        total_pages = max(1, math.ceil(total / PAGE_SIZE))
        page = max(0, min(page, total_pages - 1))
        entries = (
            await session.execute(
                select(KnowledgeBase)
                .options(selectinload(KnowledgeBase.topics))
                .order_by(KnowledgeBase.uploaded_at.desc(), KnowledgeBase.id.desc())
                .offset(page * PAGE_SIZE)
                .limit(PAGE_SIZE)
            )
        ).scalars().all()

    text = (
        "📚 <b>База знаний</b>\n\n"
        "Здесь управляются записи `knowledge_base` для общего режима и для тем.\n"
        "✅ = участвует в общем диалоге без темы. 🎯 = привязана хотя бы к одной теме."
    )
    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=admin_kb_list_keyboard(entries, page, total_pages),
    )


async def show_entry_editor(client: MaxApiClient, chat_id: int, kb_id: int) -> None:
    async with async_session_maker() as session:
        entry = await session.get(KnowledgeBase, kb_id, options=[selectinload(KnowledgeBase.topics)])
    if not entry:
        await client.send_message(chat_id=chat_id, text="Запись базы знаний не найдена.")
        return

    topics = ", ".join(html.escape(topic.name) for topic in sorted(entry.topics, key=lambda item: item.name.lower()))
    if not topics:
        topics = "Не привязана ни к одной теме"
    text = (
        f"📚 <b>{html.escape(entry.filename or f'KB #{entry.id}')}</b>\n\n"
        f"<b>ID:</b> {entry.id}\n"
        f"<b>Общий режим:</b> {'да' if entry.use_in_general_mode else 'нет'}\n"
        f"<b>Тем:</b> {len(entry.topics)}\n"
        f"<b>Привязки:</b> {topics}\n\n"
        f"<b>Содержимое:</b>\n<pre><code>{html.escape(_content_preview(entry.indexed_content))}</code></pre>"
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_kb_editor_keyboard(kb_id, bool(entry.use_in_general_mode)))


async def start_create_entry(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_kb_create_filename", {})
    await client.send_message(chat_id=chat_id, text="Введите название новой записи базы знаний.")


async def save_new_filename(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    filename = text.strip()
    if not filename:
        await client.send_message(chat_id=chat_id, text="Название не может быть пустым.")
        return
    await states.set(user_id, chat_id, "admin_kb_create_content", {"filename": filename})
    await client.send_message(chat_id=chat_id, text="Отправьте содержимое записи базы знаний одним сообщением.")


async def save_new_content(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    snapshot = await states.get(user_id)
    filename = snapshot.data.get("filename") if snapshot else None
    if not filename:
        await client.send_message(chat_id=chat_id, text="Состояние создания записи потеряно.")
        return
    content = text.strip()
    if not content:
        await client.send_message(chat_id=chat_id, text="Содержимое не может быть пустым.")
        return
    async with async_session_maker() as session:
        entry = KnowledgeBase(filename=filename, indexed_content=content, use_in_general_mode=True)
        session.add(entry)
        await session.commit()
        kb_id = entry.id

    await update_vector_index(kb_id, content)

    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text=f"✅ Запись «{html.escape(filename)}» создана.")
    await show_entry_editor(client, chat_id, kb_id)


async def start_upload_file(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_kb_upload_file", {})
    await client.send_message(
        chat_id=chat_id,
        text="Вы вошли в режим добавления файлов в Базу Знаний.\n\n"
             "Отправляйте файлы (txt, md, pdf, docx, xlsx) по одному. "
             "Когда закончите, нажмите кнопку ниже.",
        attachments=admin_kb_finish_upload_keyboard(),
    )


async def receive_upload_file(client: MaxApiClient, states: StateStore, message: IncomingMessage) -> None:
    chat_id = message.chat_id
    user_id = message.sender.user_id

    filename = "uploaded_file"
    if message.attachments:
        for att in message.attachments:
            payload = att.get("payload") or {}
            for field in ["name", "filename", "file_name", "title"]:
                if payload.get(field):
                    filename = str(payload[field])
                    break
            if filename != "uploaded_file":
                break
            for field in ["name", "filename", "file_name", "title"]:
                if att.get(field):
                    filename = str(att[field])
                    break
            if filename != "uploaded_file":
                break

    progress_msg = await client.send_message(
        chat_id=chat_id,
        text=f"⏳ Начинаю обработку файла: `{filename}`...",
    )
    progress_msg_id = progress_msg.get("id") or progress_msg.get("message_id")

    try:
        if progress_msg_id:
            await client.edit_message(
                message_id=progress_msg_id,
                text=f"⏳ Загружаю файл: `{filename}`...",
            )
        file_bytes = await client.download_attachment(message.media_token, message.media_url)

        if progress_msg_id:
            await client.edit_message(
                message_id=progress_msg_id,
                text=f"⏳ Анализирую и извлекаю текст из `{filename}`...",
            )
        from file_parser import parse_file
        indexed_content = await parse_file(io.BytesIO(file_bytes), filename)

        if not indexed_content:
            raise ValueError("Не удалось извлечь текст из файла или файл пустой.")

        if progress_msg_id:
            await client.edit_message(
                message_id=progress_msg_id,
                text=f"⏳ Создаю векторы и индексирую `{filename}`...",
            )

        async with async_session_maker() as session:
            entry = KnowledgeBase(filename=filename, indexed_content=indexed_content, use_in_general_mode=True)
            session.add(entry)
            await session.commit()
            kb_id = entry.id

        await update_vector_index(kb_id, indexed_content)

        if progress_msg_id:
            await client.delete_message(message_id=progress_msg_id)

        await client.send_message(
            chat_id=chat_id,
            text=f"✅ Файл `{filename}` полностью обработан и добавлен в Базу Знаний.",
            attachments=admin_kb_finish_upload_keyboard(),
        )
    except Exception as e:
        log.exception("Error processing KB file upload via MAX bot")
        if progress_msg_id:
            await client.edit_message(
                message_id=progress_msg_id,
                text=f"❌ Ошибка при обработке `{filename}`: {e}",
            )
        else:
            await client.send_message(
                chat_id=chat_id,
                text=f"❌ Ошибка при обработке `{filename}`: {e}",
            )


async def start_edit_filename(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, kb_id: int) -> None:
    await states.set(user_id, chat_id, "admin_kb_edit_filename", {"kb_id": kb_id})
    await client.send_message(chat_id=chat_id, text="Введите новое название записи базы знаний.")


async def save_filename(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    snapshot = await states.get(user_id)
    kb_id = snapshot.data.get("kb_id") if snapshot else None
    if not kb_id:
        await client.send_message(chat_id=chat_id, text="Состояние редактирования потеряно.")
        return
    filename = text.strip()
    if not filename:
        await client.send_message(chat_id=chat_id, text="Название не может быть пустым.")
        return
    async with async_session_maker() as session:
        entry = await session.get(KnowledgeBase, kb_id)
        if not entry:
            await client.send_message(chat_id=chat_id, text="Запись базы знаний не найдена.")
            return
        entry.filename = filename
        await session.commit()
    await states.clear(user_id)
    await show_entry_editor(client, chat_id, kb_id)


async def start_edit_content(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, kb_id: int) -> None:
    await states.set(user_id, chat_id, "admin_kb_edit_content", {"kb_id": kb_id})
    await client.send_message(chat_id=chat_id, text="Отправьте новое содержимое записи базы знаний.")


async def save_content(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    snapshot = await states.get(user_id)
    kb_id = snapshot.data.get("kb_id") if snapshot else None
    if not kb_id:
        await client.send_message(chat_id=chat_id, text="Состояние редактирования потеряно.")
        return
    content = text.strip()
    if not content:
        await client.send_message(chat_id=chat_id, text="Содержимое не может быть пустым.")
        return
    async with async_session_maker() as session:
        entry = await session.get(KnowledgeBase, kb_id)
        if not entry:
            await client.send_message(chat_id=chat_id, text="Запись базы знаний не найдена.")
            return
        entry.indexed_content = content
        await session.commit()

    delete_document_vectors(kb_id)
    await update_vector_index(kb_id, content)

    await states.clear(user_id)
    await show_entry_editor(client, chat_id, kb_id)


async def toggle_general_mode(client: MaxApiClient, chat_id: int, kb_id: int) -> None:
    async with async_session_maker() as session:
        entry = await session.get(KnowledgeBase, kb_id)
        if not entry:
            await client.send_message(chat_id=chat_id, text="Запись базы знаний не найдена.")
            return
        entry.use_in_general_mode = not bool(entry.use_in_general_mode)
        await session.commit()
    await show_entry_editor(client, chat_id, kb_id)


async def delete_entry(client: MaxApiClient, chat_id: int, kb_id: int) -> None:
    async with async_session_maker() as session:
        entry = await session.get(KnowledgeBase, kb_id)
        if not entry:
            await client.send_message(chat_id=chat_id, text="Запись базы знаний не найдена.")
            return
        await session.delete(entry)
        await session.commit()

    delete_document_vectors(kb_id)

    await client.send_message(chat_id=chat_id, text="✅ Запись базы знаний удалена.")
    await list_entries(client, chat_id, 0)


async def show_topic_assignments(client: MaxApiClient, chat_id: int, topic_id: int, page: int = 0) -> None:
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id, options=[selectinload(Topic.knowledge_base_files)])
        if not topic:
            await client.send_message(chat_id=chat_id, text="Тема не найдена.")
            return
        total = await session.scalar(select(func.count()).select_from(KnowledgeBase)) or 0
        total_pages = max(1, math.ceil(total / PAGE_SIZE))
        page = max(0, min(page, total_pages - 1))
        entries = (
            await session.execute(
                select(KnowledgeBase)
                .order_by(KnowledgeBase.filename.asc(), KnowledgeBase.id.asc())
                .offset(page * PAGE_SIZE)
                .limit(PAGE_SIZE)
            )
        ).scalars().all()
        assigned_ids = {item.id for item in topic.knowledge_base_files}

    text = (
        f"📚 <b>База знаний темы</b>\n\n"
        f"Тема: <b>{html.escape(topic.name)}</b>\n"
        "Нажмите на запись, чтобы добавить или убрать её из темы."
    )
    await client.send_message(
        chat_id=chat_id,
        text=text,
        attachments=admin_topic_kb_keyboard(topic_id, entries, assigned_ids, page, total_pages),
    )


async def toggle_topic_assignment(client: MaxApiClient, chat_id: int, topic_id: int, kb_id: int, page: int) -> None:
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id, options=[selectinload(Topic.knowledge_base_files)])
        entry = await session.get(KnowledgeBase, kb_id)
        if not topic or not entry:
            await client.send_message(chat_id=chat_id, text="Тема или запись базы знаний не найдена.")
            return
        if any(item.id == kb_id for item in topic.knowledge_base_files):
            topic.knowledge_base_files.remove(next(item for item in topic.knowledge_base_files if item.id == kb_id))
        else:
            topic.knowledge_base_files.append(entry)
        await session.commit()
    await show_topic_assignments(client, chat_id, topic_id, page)
