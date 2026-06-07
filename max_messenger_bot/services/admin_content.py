from __future__ import annotations

import html

from sqlalchemy import func, select, update

from ..api import MaxApiClient
from ..keyboards import admin_content_editor_keyboard, admin_content_list_keyboard
from ..legacy import Content, async_session_maker
from ..storage import MaxContentMedia, StateStore


SYSTEM_TITLES = {
    "start_message": "Приветствие (/start)",
    "disclaimer": "Дисклеймер",
    "test_intro": "Вступление теста",
    "test_results": "Результаты теста",
    "secret_test_outro": "Финал секретного теста",
}


def _display_title(item: Content) -> str:
    return SYSTEM_TITLES.get(item.key, item.button_title or item.key)


async def show_content_list(client: MaxApiClient, chat_id: int) -> None:
    async with async_session_maker() as session:
        items = (
            await session.execute(select(Content).order_by(Content.sort_order.asc(), Content.key.asc()))
        ).scalars().all()
    rows = [(item.key, _display_title(item), bool(item.is_visible)) for item in items]
    await client.send_message(
        chat_id=chat_id,
        text="✏️ <b>Управление контентом</b>\n\nВыберите раздел для редактирования.",
        attachments=admin_content_list_keyboard(rows),
    )


async def show_content_editor(client: MaxApiClient, chat_id: int, content_key: str) -> None:
    async with async_session_maker() as session:
        item = await session.get(Content, content_key)
        media_count = await session.scalar(select(func.count()).select_from(MaxContentMedia).where(MaxContentMedia.content_key == content_key))
    if not item:
        await client.send_message(chat_id=chat_id, text="Раздел контента не найден.")
        return
    source_display = html.escape(item.text_content or "Текст не задан.")
    rendered_display = item.text_content or "Текст не задан."
    visible = "✅ Виден пользователям" if item.is_visible else "❌ Скрыт от пользователей"
    message = (
        f"📝 <b>{html.escape(_display_title(item))}</b>\n\n"
        f"<b>Ключ:</b> <code>{item.key}</code>\n"
        f"<b>Статус:</b> {visible}\n"
        f"<b>Порядок:</b> {html.escape(item.content_order or 'media_top')}\n\n"
        f"<b>MAX-медиа:</b> {media_count or 0}\n\n"
        f"<b>Предпросмотр:</b>\n{rendered_display}\n\n"
        f"<b>Исходник:</b>\n<pre><code>{source_display}</code></pre>\n"
        "Поддерживается HTML-разметка: <b>жирный</b>, <i>курсив</i>, <u>подчёркнутый</u>."
    )
    await client.send_message(chat_id=chat_id, text=message, attachments=admin_content_editor_keyboard(content_key, bool(item.is_visible)))


async def start_text_edit(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, content_key: str) -> None:
    await states.set(user_id, chat_id, "admin_edit_content_text", {"content_key": content_key})
    await client.send_message(
        chat_id=chat_id,
        text=(
            f"Отправьте новый HTML-текст для <code>{content_key}</code> одним сообщением.\n\n"
            "Пример: <code>&lt;b&gt;жирный&lt;/b&gt;</code>"
        ),
    )


async def save_text_edit(client: MaxApiClient, states: StateStore, chat_id: int, user_id: int, text: str) -> None:
    snapshot = await states.get(user_id)
    if not snapshot:
        await client.send_message(chat_id=chat_id, text="Состояние редактирования потеряно.")
        return
    content_key = snapshot.data.get("content_key")
    if not content_key:
        await client.send_message(chat_id=chat_id, text="Не найден ключ контента.")
        return
    async with async_session_maker() as session:
        item = await session.get(Content, content_key)
        if not item:
            item = Content(key=content_key)
            session.add(item)
        item.text_content = text
        await session.commit()
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text="✅ Текст обновлён.")
    await show_content_editor(client, chat_id, content_key)


async def toggle_visibility(client: MaxApiClient, chat_id: int, content_key: str) -> None:
    async with async_session_maker() as session:
        item = await session.get(Content, content_key)
        if not item:
            await client.send_message(chat_id=chat_id, text="Раздел не найден.")
            return
        item.is_visible = not item.is_visible
        await session.commit()
    await show_content_editor(client, chat_id, content_key)
