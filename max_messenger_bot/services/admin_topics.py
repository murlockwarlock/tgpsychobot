from __future__ import annotations

import html
from pathlib import Path
from urllib.parse import quote

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ..api import MaxApiClient
from ..keyboards import admin_topic_editor_keyboard, admin_topic_prompt_input_keyboard, admin_topic_prompt_keyboard, admin_topics_list_keyboard
from ..legacy import Topic, async_session_maker
from ..storage import StateStore


def _topic_url(client: MaxApiClient, topic_id: int) -> str | None:
    if not client.bot_name:
        return None
    return f"https://max.ru/{quote(client.bot_name)}?start=topic_{topic_id}"


async def list_topics(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        topics = (
            await session.execute(select(Topic).order_by(Topic.sort_order.asc(), Topic.id.asc()))
        ).scalars().all()
    if not topics:
        await client.send_message(
            chat_id=chat_id,
            text="💬 <b>Темы диалогов</b>\n\nТем пока нет.",
            attachments=admin_topics_list_keyboard([]),
        )
        return
    await client.send_message(
        chat_id=chat_id,
        text="💬 <b>Темы диалогов</b>\n\nВыберите тему для редактирования.",
        attachments=admin_topics_list_keyboard(topics),
    )


async def show_topic_editor(client: MaxApiClient, chat_id: int, topic_id: int) -> None:
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id, options=[selectinload(Topic.knowledge_base_files)])
    if not topic:
        await client.send_message(chat_id=chat_id, text="Тема не найдена.")
        return
    _INTRO_LIMIT = 500

    def _preview(s: str, limit: int) -> str:
        return s[:limit] + "…" if len(s) > limit else s

    prompt_raw = topic.system_prompt or ""
    intro_raw = topic.start_message or ""
    topic_payload = f"topic_{topic.id}"
    topic_url = _topic_url(client, topic.id)
    topic_link_line = f"<b>Ссылка:</b> {html.escape(topic_url)}\n" if topic_url else ""
    text = (
        f"<b>{html.escape(topic.name)}</b>\n\n"
        f"<b>ID:</b> {topic.id}\n"
        f"<b>Старт-параметр:</b> <code>{topic_payload}</code>\n"
        f"{topic_link_line}"
        f"<b>Активна:</b> {'да' if topic.is_active else 'нет'}\n"
        f"<b>Только для админов:</b> {'да' if topic.admin_only else 'нет'}\n"
        f"<b>В главном меню:</b> {'да' if topic.show_in_main_menu else 'нет'}\n"
        f"<b>В списке тем:</b> {'да' if topic.show_in_list else 'нет'}\n"
        f"<b>Файлов базы знаний:</b> {len(topic.knowledge_base_files)}\n"
        f"<b>Системный промпт:</b> {len(prompt_raw)} симв.\n\n"
        f"<b>Приветствие ({len(intro_raw)} симв.):</b>\n<pre><code>{html.escape(_preview(intro_raw or 'Не задано', _INTRO_LIMIT))}</code></pre>"
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_topic_editor_keyboard(topic, topic_url))


async def show_topic_prompt(client: MaxApiClient, chat_id: int, topic_id: int) -> None:
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id)
    if not topic:
        await client.send_message(chat_id=chat_id, text="Тема не найдена.")
        return
    prompt_raw = topic.system_prompt or ""
    preview = prompt_raw[:3000] + "…" if len(prompt_raw) > 3000 else prompt_raw
    text = (
        f"<b>Системный промпт темы «{html.escape(topic.name)}»</b>\n"
        f"<b>Длина:</b> {len(prompt_raw)} симв.\n\n"
        f"<pre><code>{html.escape(preview or 'Не задан')}</code></pre>"
    )
    await client.send_message(chat_id=chat_id, text=text, attachments=admin_topic_prompt_keyboard(topic_id))


async def start_create_topic(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int) -> None:
    await states.set(user_id, chat_id, "admin_create_topic_name", {})
    await client.send_message(chat_id=chat_id, text="Введите название новой темы.")


async def create_topic(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    name = text.strip()
    if not name:
        await client.send_message(chat_id=chat_id, text="Название не может быть пустым.")
        return
    async with async_session_maker() as session:
        topic = Topic(name=name, is_active=True, show_in_list=True, show_in_main_menu=False)
        session.add(topic)
        await session.commit()
        topic_id = topic.id
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text=f"✅ Тема «{html.escape(name)}» создана.")
    await show_topic_editor(client, chat_id, topic_id)


async def start_edit_name(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, topic_id: int) -> None:
    await states.set(user_id, chat_id, "admin_edit_topic_name", {"topic_id": topic_id})
    await client.send_message(chat_id=chat_id, text="Введите новое название темы.")


async def save_name(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    snapshot = await states.get(user_id)
    topic_id = snapshot.data.get("topic_id") if snapshot else None
    if not topic_id:
        await client.send_message(chat_id=chat_id, text="Состояние темы потеряно.")
        return
    name = text.strip()
    if not name:
        await client.send_message(chat_id=chat_id, text="Название не может быть пустым.")
        return
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id)
        if not topic:
            await client.send_message(chat_id=chat_id, text="Тема не найдена.")
            return
        topic.name = name
        await session.commit()
    await states.clear(user_id)
    await show_topic_editor(client, chat_id, topic_id)


async def start_edit_prompt(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, topic_id: int) -> None:
    await states.set(user_id, chat_id, "admin_edit_topic_prompt", {"topic_id": topic_id})
    await client.send_message(
        chat_id=chat_id,
        text="Отправьте новый системный промпт темы сообщением или загрузите .txt/.md файл.",
        attachments=admin_topic_prompt_input_keyboard(topic_id),
    )


async def save_prompt(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    snapshot = await states.get(user_id)
    topic_id = snapshot.data.get("topic_id") if snapshot else None
    if not topic_id:
        await client.send_message(chat_id=chat_id, text="Состояние темы потеряно.")
        return
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id)
        if not topic:
            await client.send_message(chat_id=chat_id, text="Тема не найдена.")
            return
        topic.system_prompt = text
        await session.commit()
    await states.clear(user_id)
    await show_topic_editor(client, chat_id, topic_id)


async def download_prompt(client: MaxApiClient, chat_id: int, topic_id: int) -> None:
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id)
    if not topic:
        await client.send_message(chat_id=chat_id, text="Тема не найдена.")
        return
    filename = f"topic_{topic.id}_system_prompt.txt"
    safe_name = Path(filename).name
    try:
        await client.send_text_file(
            chat_id=chat_id,
            filename=safe_name,
            content=topic.system_prompt or "",
            caption=f"📥 {safe_name}",
        )
    except Exception as exc:
        await client.send_message(chat_id=chat_id, text=f"Не удалось отправить файл: {html.escape(str(exc))}")


async def cancel_prompt_input(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, topic_id: int) -> None:
    await states.clear(user_id)
    await show_topic_prompt(client, chat_id, topic_id)


async def start_edit_intro(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, topic_id: int) -> None:
    await states.set(user_id, chat_id, "admin_edit_topic_intro", {"topic_id": topic_id})
    await client.send_message(chat_id=chat_id, text="Отправьте новое приветственное сообщение темы.")


async def save_intro(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    snapshot = await states.get(user_id)
    topic_id = snapshot.data.get("topic_id") if snapshot else None
    if not topic_id:
        await client.send_message(chat_id=chat_id, text="Состояние темы потеряно.")
        return
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id)
        if not topic:
            await client.send_message(chat_id=chat_id, text="Тема не найдена.")
            return
        topic.start_message = text
        await session.commit()
    await states.clear(user_id)
    await show_topic_editor(client, chat_id, topic_id)


async def toggle_active(client: MaxApiClient, chat_id: int, topic_id: int) -> None:
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id)
        if topic:
            topic.is_active = not topic.is_active
            await session.commit()
    await show_topic_editor(client, chat_id, topic_id)


async def toggle_admin_only(client: MaxApiClient, chat_id: int, topic_id: int) -> None:
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id)
        if topic:
            topic.admin_only = not topic.admin_only
            await session.commit()
    await show_topic_editor(client, chat_id, topic_id)


async def toggle_menu(client: MaxApiClient, chat_id: int, topic_id: int) -> None:
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id)
        if topic:
            topic.show_in_main_menu = not topic.show_in_main_menu
            await session.commit()
    await show_topic_editor(client, chat_id, topic_id)


async def toggle_list(client: MaxApiClient, chat_id: int, topic_id: int) -> None:
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id)
        if topic:
            topic.show_in_list = not topic.show_in_list
            await session.commit()
    await show_topic_editor(client, chat_id, topic_id)


async def delete_topic(client: MaxApiClient, chat_id: int, topic_id: int) -> None:
    async with async_session_maker() as session:
        topic = await session.get(Topic, topic_id)
        if topic:
            await session.delete(topic)
            await session.commit()
    await client.send_message(chat_id=chat_id, text="✅ Тема удалена.")
    await list_topics(client, chat_id)
