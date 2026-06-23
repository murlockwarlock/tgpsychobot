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


async def show_content_editor(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    content_key: str
) -> None:
    snapshot = await states.get(user_id)
    if snapshot and snapshot.state == "admin_edit_content" and snapshot.data.get("content_key") == content_key:
        data = snapshot.data
    else:
        async with async_session_maker() as session:
            item = await session.get(Content, content_key)
            if not item:
                await client.send_message(chat_id=chat_id, text="Раздел контента не найден.")
                return
            media_rows = (
                await session.execute(
                    select(MaxContentMedia)
                    .where(MaxContentMedia.content_key == content_key)
                    .order_by(MaxContentMedia.id.asc())
                )
            ).scalars().all()

            data = {
                "content_key": content_key,
                "text_content": item.text_content or "",
                "media_files": [{"type": row.media_type, "token": row.token} for row in media_rows],
                "content_order": item.content_order or "media_top",
            }
            await states.set(user_id, chat_id, "admin_edit_content", data)

    text_content = data.get("text_content") or ""
    media_files = data.get("media_files") or []
    content_order = data.get("content_order") or "media_top"

    async with async_session_maker() as session:
        item = await session.get(Content, content_key)
        is_visible = item.is_visible if item else True

    title = _display_title(item) if item else content_key
    visible_str = "✅ Виден пользователям" if is_visible else "❌ Скрыт от пользователей"
    order_desc = "Сначала медиа, потом текст" if content_order == "media_top" else "Сначала текст, потом медиа"

    media_lines = []
    if media_files:
        for i, mf in enumerate(media_files):
            emoji = "🖼️ Фото" if mf["type"] == "photo" else "📹 Видео"
            media_lines.append(f"  {i + 1}. {emoji}")
        media_display = "\n".join(media_lines)
    else:
        media_display = "<i>Медиафайлы не добавлены.</i>"

    preview_limit = 700
    if text_content and len(text_content) > preview_limit:
        truncated = text_content[:preview_limit]
        from ..formatting import _open_tags
        unclosed = _open_tags(truncated)
        rendered_display = truncated + "".join(f"</{t}>" for t in reversed(unclosed)) + "\n... (текст обрезан для предпросмотра)"
        source_display = html.escape(text_content[:preview_limit]) + "\n... (исходный код обрезан)"
    else:
        rendered_display = text_content or "Текст не задан."
        source_display = html.escape(text_content or "Текст не задан.")

    message = (
        f"📝 <b>{html.escape(title)}</b>\n\n"
        f"<b>Ключ:</b> <code>{content_key}</code>\n"
        f"<b>Статус:</b> {visible_str}\n"
        f"<b>Порядок:</b> {order_desc}\n\n"
        f"<b>Текущие медиафайлы:</b>\n{media_display}\n\n"
        f"<b>Предпросмотр:</b>\n{rendered_display}\n\n"
        f"<b>Исходник:</b>\n<pre><code>{source_display}</code></pre>\n"
        f"Поддерживается HTML-разметка: <b>жирный</b>, <i>курсив</i>, <u>подчёркнутый</u>.\n\n"
        f"Отправьте новый текст (сохранится форматирование) или прикрепите фото/видео, чтобы добавить медиа. "
        f"Используйте кнопки ниже для настроек."
    )

    keyboard = admin_content_editor_keyboard(content_key, media_files, content_order, is_visible)
    await client.send_message(chat_id=chat_id, text=message, attachments=keyboard)


async def receive_message(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    text: str | None,
    media_token: str | None,
    media_type: str | None,
) -> None:
    snapshot = await states.get(user_id)
    if not snapshot or snapshot.state != "admin_edit_content":
        await client.send_message(chat_id=chat_id, text="Состояние редактирования потеряно.")
        return

    data = snapshot.data
    content_key = data.get("content_key")
    if not content_key:
        await client.send_message(chat_id=chat_id, text="Не найден ключ контента.")
        return

    updated = False
    if media_token and media_type:
        mtype = "photo" if media_type == "image" else "video" if media_type == "video" else media_type
        media_files = data.setdefault("media_files", [])
        media_files.append({"type": mtype, "token": media_token})
        updated = True
        await client.send_message(chat_id=chat_id, text=f"✅ Медиафайл добавлен (всего: {len(media_files)}).")

    if text:
        data["text_content"] = text
        updated = True
        await client.send_message(chat_id=chat_id, text="✅ Текст обновлен.")

    if not updated:
        await client.send_message(chat_id=chat_id, text="Пожалуйста, отправьте текст или медиафайл.")
        return

    await states.set(user_id, chat_id, "admin_edit_content", data)
    await show_content_editor(client, states, chat_id, user_id, content_key)


async def handle_order_toggle(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    content_key: str
) -> None:
    snapshot = await states.get(user_id)
    if not snapshot or snapshot.state != "admin_edit_content":
        await client.send_message(chat_id=chat_id, text="Состояние редактирования потеряно.")
        return

    data = snapshot.data
    current_order = data.get("content_order", "media_top")
    new_order = "text_top" if current_order == "media_top" else "media_top"
    data["content_order"] = new_order

    await states.set(user_id, chat_id, "admin_edit_content", data)
    await show_content_editor(client, states, chat_id, user_id, content_key)


async def handle_media_delete(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    content_key: str,
    index: int
) -> None:
    snapshot = await states.get(user_id)
    if not snapshot or snapshot.state != "admin_edit_content":
        await client.send_message(chat_id=chat_id, text="Состояние редактирования потеряно.")
        return

    data = snapshot.data
    media_files = data.get("media_files") or []
    if 0 <= index < len(media_files):
        media_files.pop(index)
        data["media_files"] = media_files
        await states.set(user_id, chat_id, "admin_edit_content", data)
        await client.send_message(chat_id=chat_id, text=f"🗑️ Медиафайл #{index + 1} удален.")
    else:
        await client.send_message(chat_id=chat_id, text="Неверный индекс медиафайла.")

    await show_content_editor(client, states, chat_id, user_id, content_key)


async def handle_save_content(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    content_key: str
) -> None:
    snapshot = await states.get(user_id)
    if not snapshot or snapshot.state != "admin_edit_content":
        await client.send_message(chat_id=chat_id, text="Состояние редактирования потеряно.")
        return

    data = snapshot.data
    text_content = data.get("text_content")
    media_files = data.get("media_files") or []
    content_order = data.get("content_order", "media_top")

    from sqlalchemy import delete
    async with async_session_maker() as session:
        item = await session.get(Content, content_key)
        if not item:
            item = Content(key=content_key)
            session.add(item)
        item.text_content = text_content
        item.content_order = content_order

        # Delete old MaxContentMedia rows
        await session.execute(delete(MaxContentMedia).where(MaxContentMedia.content_key == content_key))

        # Insert new MaxContentMedia rows
        for mf in media_files:
            session.add(
                MaxContentMedia(
                    content_key=content_key,
                    media_type=mf["type"],
                    token=mf["token"],
                )
            )
        await session.commit()

    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text="✅ Изменения успешно сохранены.")
    await show_content_list(client, chat_id)


async def handle_cancel_edit(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    content_key: str
) -> None:
    await states.clear(user_id)
    await client.send_message(chat_id=chat_id, text="❌ Редактирование отменено.")
    await show_content_list(client, chat_id)


async def toggle_visibility(
    client: MaxApiClient,
    states: StateStore,
    chat_id: int,
    user_id: int,
    content_key: str
) -> None:
    async with async_session_maker() as session:
        item = await session.get(Content, content_key)
        if not item:
            await client.send_message(chat_id=chat_id, text="Раздел не найден.")
            return
        item.is_visible = not item.is_visible
        is_visible = item.is_visible
        await session.commit()

    status_str = "показан" if is_visible else "скрыт"
    await client.send_message(chat_id=chat_id, text=f"Раздел теперь {status_str}.")
    await show_content_editor(client, states, chat_id, user_id, content_key)
